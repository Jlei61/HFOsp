"""Step-wise state modulation for a frozen LBSS contact-sequence decoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.topic5_group_event_state.v034_spatial_state.we_decoder import per_event_scores
from src.topic5_lbss_rnn_v0_2 import LBSSModel


@dataclass(frozen=True)
class StepwiseAdapterConfig:
    context_dim: int
    rank: int = 8
    stop_weight: float = 1.0


class StaticStepAdapter(nn.Module):
    """State-free per-step recalibration; frozen before dynamic state fitting."""

    def __init__(self, hidden_width: int, n_contacts: int) -> None:
        super().__init__()
        # Constant, linear and quadratic step basis.  No future event length is
        # used; t is divided by the known contact count.
        self.hidden_gamma = nn.Parameter(torch.zeros(3, hidden_width))
        self.hidden_beta = nn.Parameter(torch.zeros(3, hidden_width))
        self.contact_shift = nn.Parameter(torch.zeros(3, n_contacts))
        self.stop_shift = nn.Parameter(torch.zeros(3))

    @staticmethod
    def basis(t_norm: Tensor) -> Tensor:
        return torch.stack((torch.ones_like(t_norm), t_norm, t_norm.square()), dim=-1)

    def forward(self, t_norm: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        b = self.basis(t_norm)
        return b @ self.hidden_gamma, b @ self.hidden_beta, b @ self.contact_shift, b @ self.stop_shift


class DynamicStepAdapter(nn.Module):
    """Low-rank, contact-specific modulation applied after every frozen step."""

    def __init__(self, config: StepwiseAdapterConfig, hidden_width: int, n_contacts: int) -> None:
        super().__init__()
        self.config = config
        width = 2 * int(config.context_dim) + 2  # state, state*t, t, t^2
        rank = min(int(config.rank), width, hidden_width)
        self.down = nn.Linear(width, rank, bias=False)
        self.gamma = nn.Linear(rank, hidden_width, bias=False)
        self.beta = nn.Linear(rank, hidden_width, bias=False)
        self.contact = nn.Linear(rank, n_contacts, bias=False)
        self.stop = nn.Linear(rank, 1, bias=False)
        nn.init.xavier_uniform_(self.down.weight)
        # Exact decoder parity at construction while retaining a trainable
        # path.  Once output maps move, gradients reach context/state producer.
        for module in (self.gamma, self.beta, self.contact, self.stop):
            nn.init.zeros_(module.weight)

    def forward(self, context: Tensor, t_norm: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x = torch.cat((context, context * t_norm[:, None], t_norm[:, None], t_norm.square()[:, None]), dim=-1)
        h = torch.nn.functional.gelu(self.down(x))
        return self.gamma(h), self.beta(h), self.contact(h), self.stop(h).squeeze(-1)


class StepwiseConditionedDecoder(nn.Module):
    """Frozen tissue decoder plus nested static and dynamic residual adapters."""

    def __init__(self, decoder: LBSSModel, config: StepwiseAdapterConfig) -> None:
        super().__init__()
        self.decoder = decoder
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(False)
        width = decoder.n_nodes * decoder.state_dim
        self.static = StaticStepAdapter(width, decoder.n_contacts)
        self.dynamic = DynamicStepAdapter(config, width, decoder.n_contacts)
        self.stop_weight = float(config.stop_weight)

    def train(self, mode: bool = True):
        super().train(mode)
        self.decoder.eval()
        return self

    def hidden_sequence(
        self, x: Tensor, recruited: Tensor, valid: Tensor, context: Tensor | None,
        *, use_static: bool, use_dynamic: bool,
        extra_context: Tensor | None = None,
        extra_adapter: DynamicStepAdapter | None = None,
    ) -> Tensor:
        b, steps, _ = x.shape
        h = torch.zeros(b, self.decoder.n_nodes * self.decoder.state_dim, device=x.device)
        hidden = []
        denom = max(1, self.decoder.n_contacts - 1)
        if use_dynamic and context is None:
            raise ValueError("dynamic step modulation requires a pre-event context")
        for step in range(steps):
            h = self.decoder._step(h, x[:, step])
            t_norm = torch.full((b,), step / denom, dtype=h.dtype, device=h.device)
            gamma = beta = contact = stop = 0.0
            if use_static:
                sg, sb, sc, ss = self.static(t_norm)
                gamma, beta, contact, stop = sg, sb, sc, ss
            if use_dynamic:
                dg, db, dc, ds = self.dynamic(context.to(h.dtype), t_norm)
                gamma, beta, contact, stop = gamma + dg, beta + db, contact + dc, stop + ds
            if extra_context is not None:
                if extra_adapter is None:
                    raise ValueError("extra_context requires its registered extra_adapter")
                eg, eb, ec, es = extra_adapter(extra_context.to(h.dtype), t_norm)
                gamma, beta, contact, stop = gamma + eg, beta + eb, contact + ec, stop + es
            conditioned = h * (1.0 + gamma) + beta
            hidden.append(conditioned)
        return torch.stack(hidden, 1)

    def forward(
        self, x: Tensor, recruited: Tensor, valid: Tensor, context: Tensor | None,
        *, use_static: bool, use_dynamic: bool,
        extra_context: Tensor | None = None,
        extra_adapter: DynamicStepAdapter | None = None,
    ) -> tuple[Tensor, Tensor]:
        conditioned = self.hidden_sequence(
            x, recruited, valid, context, use_static=use_static, use_dynamic=use_dynamic,
            extra_context=extra_context, extra_adapter=extra_adapter,
        )
        b, steps, _ = conditioned.shape
        logits, stops = [], []
        denom = max(1, self.decoder.n_contacts - 1)
        for step in range(steps):
            h = conditioned[:, step]
            t_norm = torch.full((b,), step / denom, dtype=h.dtype, device=h.device)
            # Recompute only the logit-space residuals.  The conditioned hidden
            # already contains FiLM; contact/STOP shifts are evaluated at the
            # same step so forward remains bit-identical to the old path.
            contact = stop = 0.0
            if use_static:
                _sg, _sb, contact, stop = self.static(t_norm)
            if use_dynamic:
                _dg, _db, dc, ds = self.dynamic(context.to(h.dtype), t_norm)
                contact, stop = contact + dc, stop + ds
            if extra_context is not None:
                _eg, _eb, ec, es = extra_adapter(extra_context.to(h.dtype), t_norm)
                contact, stop = contact + ec, stop + es
            logits.append(self.decoder._readout(h) + contact)
            stops.append(self.decoder._stop(h, t_norm, recruited[:, step].mean(-1)) + stop)
        return torch.stack(logits, 1), torch.stack(stops, 1)

    def scores(self, batch: dict[str, Tensor], context: Tensor | None, *, use_static: bool, use_dynamic: bool,
               extra_context: Tensor | None = None,
               extra_adapter: DynamicStepAdapter | None = None) -> dict[str, Tensor]:
        logits, stops = self.forward(batch["x"], batch["recruited"], batch["valid"], context,
                                     use_static=use_static, use_dynamic=use_dynamic,
                                     extra_context=extra_context, extra_adapter=extra_adapter)
        return per_event_scores(logits, stops, batch, self.stop_weight)
