"""Configurable residual state model for the training laboratory (design §4).

The model is v0.3.2's residual marked-history state with every T2/T3 knob
exposed as configuration: encoder depth / width / activation / dropout /
hidden LayerNorm / init / write scale, write width, time bank, gate init, and
an explicitly *exploratory* gated event state with TBPTT.  Two things are not
configurable on purpose: the state never receives a per-time LayerNorm, and
the residual gate ``alpha`` is trainable from the first step (no freeze API).

Contract clauses (plan Task 3):
  [M1] LayerNorm only in the encoder; [M2] alpha trainable from construction;
  [M3] every requires_grad parameter in exactly one named group, no decay on
       bias / gate / dispersion; [M4] write_scale rescales only the output-layer
       init, orthogonal init is orthogonal; [M5] state_dim = n_taus * write_width;
  [M6] gated TBPTT cuts the graph at chunk edges but carries the value;
  [M7] no free intercept; [M8] dropout off in eval.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from src.topic5_group_event_state.v032_model.state import (  # re-use, do not re-invent
    MarkedLeakyBank,
    _taus_full,
    _validate_stream,
    anchor_states,
)

from .paths import payload_hash

STATE_FAMILIES = ("fixed_leaky", "gated_exploratory")
ACTIVATIONS: dict[str, type[nn.Module]] = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
INITS = ("xavier", "orthogonal")
HIDDEN_NORMS = ("none", "layernorm")
TIME_BANKS: dict[str, tuple[float, float, float]] = {
    "5_30_120": (300.0, 1800.0, 7200.0),
    "10_60_180": (600.0, 3600.0, 10800.0),
}
GROUP_NAMES = ("encoder_weights", "encoder_bias", "state_weights", "state_bias",
               "adapter_w", "adapter_gate_alpha", "adapter_dispersion")


@dataclass(frozen=True)
class ArchConfig:
    state_family: str = "fixed_leaky"
    taus_seconds: tuple[float, ...] = (300.0, 1800.0, 7200.0)
    write_width: int = 4
    depth: int = 1
    width: int = 32
    activation: str = "gelu"
    dropout: float = 0.0
    hidden_norm: str = "none"
    init: str = "xavier"
    write_scale: float = 1.0
    alpha_init: float = 0.03
    gate_bias_init: float = 0.0
    tbptt_seconds: float = 1800.0
    rnn_hidden: int = 32
    chunk_seconds: float = 3600.0

    @property
    def state_dim(self) -> int:
        return len(self.taus_seconds) * int(self.write_width)

    @property
    def event_dim(self) -> int:
        """Event embedding width fed to the gated update net (4x write width; 16 at the v0.3.2 default)."""

        return 4 * int(self.write_width)

    def validate(self) -> "ArchConfig":
        if self.state_family not in STATE_FAMILIES:
            raise ValueError(f"state_family {self.state_family!r} not in {STATE_FAMILIES}")
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation {self.activation!r} not in {tuple(ACTIVATIONS)}")
        if self.init not in INITS:
            raise ValueError(f"init {self.init!r} not in {INITS}")
        if self.hidden_norm not in HIDDEN_NORMS:
            raise ValueError(f"hidden_norm {self.hidden_norm!r} not in {HIDDEN_NORMS}")
        if self.write_width < 1 or self.depth < 1 or self.width < 1 or self.rnn_hidden < 1:
            raise ValueError("write_width, depth, width and rnn_hidden must be >= 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.alpha_init <= 0 or self.write_scale <= 0:
            raise ValueError("alpha_init and write_scale must be strictly positive")
        if any(t <= 0 for t in self.taus_seconds) or self.tbptt_seconds <= 0 or self.chunk_seconds <= 0:
            raise ValueError("taus, tbptt_seconds and chunk_seconds must be positive")
        return self

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def arch_hash(self) -> str:
        return payload_hash(self.as_dict())


# ------------------------------------------------------------------ encoder
class EncoderMLP(nn.Module):
    """``depth`` hidden blocks ``Linear -> [LayerNorm] -> act -> dropout`` then ``Linear(width, out)``."""

    def __init__(self, in_dim: int, width: int, depth: int, activation: str, dropout: float,
                 hidden_norm: str, init: str, write_scale: float, out_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        d = int(in_dim)
        for _ in range(int(depth)):
            self.layers.append(nn.Linear(d, int(width)))
            self.norms.append(nn.LayerNorm(int(width)) if hidden_norm == "layernorm" else nn.Identity())
            d = int(width)
        self.act = ACTIVATIONS[activation]()
        self.drop = nn.Dropout(float(dropout))
        self.output = nn.Linear(d, int(out_dim))
        self.reset_parameters(init, write_scale)

    def reset_parameters(self, init: str, write_scale: float) -> None:
        for linear in [*self.layers, self.output]:
            if init == "xavier":
                nn.init.xavier_uniform_(linear.weight)
            else:
                nn.init.orthogonal_(linear.weight)
            nn.init.zeros_(linear.bias)
        with torch.no_grad():                                                   # [M4]
            self.output.weight.mul_(float(write_scale))
            self.output.bias.mul_(float(write_scale))

    def forward(self, x: Tensor) -> Tensor:
        h = x.to(torch.float32)
        for linear, norm in zip(self.layers, self.norms):
            h = self.drop(self.act(norm(linear(h))))
        return self.output(h)


# -------------------------------------------------------------- gated state
class GatedEventState(nn.Module):
    """Exploratory gated event state: v0.3.2 repaired update rule + TBPTT chunk detach + gate bias init.

    ``S_post = S_pre + sigmoid(g) * frac(tau) * (tanh(c) - S_pre)`` with fixed
    autonomous decay between events.  At a TBPTT chunk edge the carried state
    is *detached* (graph cut) but never reset (value carried) -- [M6].
    """

    def __init__(self, taus_seconds: Sequence[float], channels_per_tau: int, *, event_dim: int, hidden: int,
                 tbptt_seconds: float, gate_bias_init: float = 0.0,
                 update_fraction_numerator: float = 2.0, update_fraction_cap: float = 0.2) -> None:
        super().__init__()
        taus_full = _taus_full(taus_seconds, channels_per_tau)
        self.register_buffer("taus", torch.tensor(list(taus_seconds), dtype=torch.float32))
        self.register_buffer("taus_full", taus_full)
        self.register_buffer("update_fraction",
                             (float(update_fraction_numerator) / torch.sqrt(taus_full)).clamp(max=float(update_fraction_cap)))
        d = int(taus_full.numel())
        self.update_net = nn.Sequential(nn.Linear(int(event_dim) + d, int(hidden)), nn.GELU(),
                                        nn.Linear(int(hidden), 2 * d))
        with torch.no_grad():
            self.update_net[-1].bias[:d].fill_(float(gate_bias_init))
        self.tbptt_seconds = float(tbptt_seconds)
        self.channels_per_tau = int(channels_per_tau)

    @property
    def state_dim(self) -> int:
        return int(self.taus_full.numel())

    def initial_gate_mean(self) -> float:
        d = self.state_dim
        return float(torch.sigmoid(self.update_net[-1].bias[:d].detach()).mean())

    def _update(self, state_pre: Tensor, event: Tensor) -> Tensor:
        gate, candidate = self.update_net(torch.cat([state_pre, event], dim=-1)).chunk(2, dim=-1)
        fraction = torch.sigmoid(gate) * self.update_fraction
        return state_pre + fraction * (torch.tanh(candidate) - state_pre)

    def forward(self, e: Tensor, times: Tensor, segment_ids: Tensor) -> tuple[Tensor, Tensor]:
        _validate_stream(e, times, segment_ids)
        n, d, device = e.shape[0], self.state_dim, e.device
        seg_np = segment_ids.detach().cpu().numpy()
        times_np = times.detach().cpu().numpy().astype(np.float64)
        starts = np.flatnonzero(np.r_[True, seg_np[1:] != seg_np[:-1]])
        stops = np.r_[starts[1:], n]
        n_seg = starts.size
        max_len = int((stops - starts).max()) if n_seg else 0
        padded = np.full((n_seg, max_len), -1, dtype=np.int64)
        chunk_np = np.zeros(n, dtype=np.int64)
        for s, (a, b) in enumerate(zip(starts, stops)):
            padded[s, : b - a] = np.arange(a, b)
            chunk_np[a:b] = np.floor((times_np[a:b] - times_np[a]) / self.tbptt_seconds).astype(np.int64)
        padded_t = torch.from_numpy(padded).to(device)
        chunk_t = torch.from_numpy(chunk_np).to(device)
        times64 = times.to(device=device, dtype=torch.float64)
        taus64 = self.taus_full.to(torch.float64)
        state = e.new_zeros((n_seg, d))
        prev_time = torch.zeros(n_seg, dtype=torch.float64, device=device)
        prev_chunk = torch.zeros(n_seg, dtype=torch.long, device=device)
        rows_all: list[Tensor] = []
        pre_all: list[Tensor] = []
        post_all: list[Tensor] = []
        for step in range(max_len):
            rows = padded_t[:, step]
            active = rows >= 0
            if not bool(active.any()):
                break
            rows = rows[active]
            t_now = times64[rows]
            carried = state[active]
            if step > 0:
                dt = t_now - prev_time[active]
                new_chunk = chunk_t[rows] != prev_chunk[active]
                carried = torch.where(new_chunk[:, None], carried.detach(), carried)   # [M6] cut, not reset
            else:
                dt = torch.zeros_like(t_now)
            decay = torch.exp(-dt[:, None] / taus64[None, :]).to(e.dtype)
            state_pre = carried * decay
            state_post = self._update(state_pre, e[rows])
            new_state = state.clone()
            new_state[active] = state_post
            state = new_state
            prev_time = prev_time.clone()
            prev_time[active] = t_now
            prev_chunk = prev_chunk.clone()
            prev_chunk[active] = chunk_t[rows]
            rows_all.append(rows)
            pre_all.append(state_pre)
            post_all.append(state_post)
        order = torch.cat(rows_all)
        pre = e.new_zeros((n, d)).index_put((order,), torch.cat(pre_all))
        post = e.new_zeros((n, d)).index_put((order,), torch.cat(post_all))
        return pre, post

    def anchor(self, state_post: Tensor, event_times: Tensor, t_anchor: Tensor, last_event_pos: Tensor) -> Tensor:
        return anchor_states(state_post, event_times, t_anchor, last_event_pos, self.taus_full)


# ----------------------------------------------------------------- adapter
class CountProfileAdapter(nn.Module):
    """``log mu_{a,b} = log mu_H,{a,b} + alpha * (W S~_a)_b`` -- no bias, per-bin NB dispersion."""

    def __init__(self, state_dim: int, n_bins: int, alpha_init: float, log_r_init: Sequence[float] | np.ndarray) -> None:
        super().__init__()
        if float(alpha_init) <= 0:
            raise ValueError("alpha_init must be positive (a zero gate is a dead zone)")
        self.W = nn.Linear(int(state_dim), int(n_bins), bias=False)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))               # [M2] trainable, no freeze
        log_r = torch.as_tensor(np.asarray(log_r_init, dtype=np.float32)).reshape(int(n_bins)).clone()
        self.log_r = nn.Parameter(log_r)

    @property
    def n_bins(self) -> int:
        return int(self.W.weight.shape[0])

    def modulation(self, state_std: Tensor) -> Tensor:
        return self.alpha * self.W(state_std.to(torch.float32))

    def forward(self, log_mu_h: Tensor, state_std: Tensor) -> Tensor:
        return log_mu_h.to(torch.float32) + self.modulation(state_std)               # [M7]


# ------------------------------------------------------------------- model
class FlexibleResidualStateModel(nn.Module):
    def __init__(self, arch: ArchConfig, in_dim: int, n_bins: int, log_r_init: Sequence[float] | np.ndarray) -> None:
        super().__init__()
        arch.validate()
        self.arch = arch
        if arch.state_family == "fixed_leaky":
            write_dim = arch.write_width
            self.state: nn.Module = MarkedLeakyBank(arch.taus_seconds, arch.write_width, chunk_seconds=arch.chunk_seconds)
        else:
            write_dim = arch.event_dim
            self.state = GatedEventState(arch.taus_seconds, arch.write_width, event_dim=write_dim, hidden=arch.rnn_hidden,
                                         tbptt_seconds=arch.tbptt_seconds, gate_bias_init=arch.gate_bias_init)
        self.encoder = EncoderMLP(in_dim, arch.width, arch.depth, arch.activation, arch.dropout, arch.hidden_norm,
                                  arch.init, arch.write_scale, write_dim)
        self.adapter = CountProfileAdapter(self.state.state_dim, n_bins, arch.alpha_init, log_r_init)
        self.register_buffer("phi_mean", torch.zeros(write_dim))
        self.register_buffer("train_mean_state", torch.zeros(self.state.state_dim))
        self.register_buffer("train_state_scale", torch.ones(self.state.state_dim))
        self.amp_encoder = False

    @property
    def state_dim(self) -> int:
        return int(self.state.state_dim)

    @property
    def n_bins(self) -> int:
        return self.adapter.n_bins

    # ---------------------------------------------------------------- forward
    def project(self, x_scaled: Tensor) -> Tensor:
        """Encoder forward; bf16 autocast only here when ``amp_encoder`` is set on a CUDA tensor."""

        if getattr(self, "amp_encoder", False) and x_scaled.is_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.encoder(x_scaled)
            return out.float()
        return self.encoder(x_scaled)

    def writes(self, x_scaled: Tensor, train_event_mask: Tensor | None = None) -> Tensor:
        phi = self.project(x_scaled)
        if train_event_mask is not None:
            mask = train_event_mask.to(torch.bool)
            mean = phi[mask].mean(dim=0)
            self.phi_mean.copy_(mean.detach())
        else:
            mean = self.phi_mean
        if self.arch.state_family == "fixed_leaky":
            return torch.tanh(phi - mean)
        return phi - mean

    @torch.no_grad()
    def refresh_train_mean(self, x_scaled: Tensor, train_event_mask: Tensor) -> None:
        mask = train_event_mask.to(torch.bool)
        if int(mask.sum()) == 0:
            raise ValueError("train mean needs at least one TRAIN event")
        self.phi_mean.copy_(self.project(x_scaled[mask]).mean(dim=0).detach())

    def trajectory(self, x_scaled: Tensor, times: Tensor, segment_ids: Tensor,
                   train_event_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        return self.state(self.writes(x_scaled, train_event_mask), times, segment_ids)

    def anchor_states(self, state_post: Tensor, event_times: Tensor, t_anchor: Tensor, last_event_pos: Tensor) -> Tensor:
        return self.state.anchor(state_post, event_times, t_anchor, last_event_pos)

    def standardize_state(self, anchor_state: Tensor, train_state: Tensor | None = None) -> Tensor:
        """TRAIN-fixed mean/scale (differentiable on the training forward; frozen buffers otherwise).
        Not a per-time LayerNorm: no per-sample normalisation, time constants untouched."""

        s = anchor_state.to(torch.float32)
        if train_state is not None:
            ref = train_state.to(torch.float32)
            mean = ref.mean(dim=0)
            scale = ref.std(dim=0, unbiased=False)
            scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
            self.train_mean_state.copy_(mean.detach())
            self.train_state_scale.copy_(scale.detach())
            return (s - mean) / scale
        return (s - self.train_mean_state) / self.train_state_scale

    def log_mu(self, log_mu_h: Tensor, anchor_state: Tensor, train_state: Tensor | None = None) -> Tensor:
        return self.adapter(log_mu_h, self.standardize_state(anchor_state, train_state))

    def modulation_jacobian(self) -> Tensor:
        """d(log mu)/d(raw anchor state): alpha * W / scale, shape (n_bins, state_dim)."""

        return (self.adapter.alpha * self.adapter.W.weight / self.train_state_scale[None, :]).detach()

    @torch.no_grad()
    def refresh_train_statistics(self, x_scaled: Tensor, train_event_mask: Tensor, times: Tensor, segment_ids: Tensor,
                                 train_anchor_time: Tensor, train_last_event_pos: Tensor) -> None:
        self.refresh_train_mean(x_scaled, train_event_mask)
        _pre, post = self.trajectory(x_scaled, times, segment_ids)
        state = self.anchor_states(post, times, train_anchor_time, train_last_event_pos).to(torch.float32)
        if state.shape[0] == 0:
            raise ValueError("train statistics need at least one TRAIN anchor")
        scale = state.std(dim=0, unbiased=False)
        self.train_mean_state.copy_(state.mean(dim=0))
        self.train_state_scale.copy_(torch.where(scale > 1e-6, scale, torch.ones_like(scale)))

    # -------------------------------------------------------------- optimiser
    def param_groups(self, lrs: Mapping[str, float], weight_decay: float) -> list[dict[str, Any]]:
        """[M3] Seven named groups; biases / LayerNorm affine / gate / dispersion never decay."""

        missing = [g for g in GROUP_NAMES if g not in lrs]
        if missing:
            raise ValueError(f"learning rates missing for groups {missing}")
        enc_w = [p for _n, p in self.encoder.named_parameters() if p.ndim > 1]
        enc_b = [p for _n, p in self.encoder.named_parameters() if p.ndim <= 1]
        st_w = [p for _n, p in self.state.named_parameters() if p.ndim > 1]
        st_b = [p for _n, p in self.state.named_parameters() if p.ndim <= 1]
        wd = float(weight_decay)
        return [
            {"name": "encoder_weights", "params": enc_w, "lr": float(lrs["encoder_weights"]), "weight_decay": wd},
            {"name": "encoder_bias", "params": enc_b, "lr": float(lrs["encoder_bias"]), "weight_decay": 0.0},
            {"name": "state_weights", "params": st_w, "lr": float(lrs["state_weights"]), "weight_decay": wd},
            {"name": "state_bias", "params": st_b, "lr": float(lrs["state_bias"]), "weight_decay": 0.0},
            {"name": "adapter_w", "params": [self.adapter.W.weight], "lr": float(lrs["adapter_w"]), "weight_decay": wd},
            {"name": "adapter_gate_alpha", "params": [self.adapter.alpha], "lr": float(lrs["adapter_gate_alpha"]),
             "weight_decay": 0.0},
            {"name": "adapter_dispersion", "params": [self.adapter.log_r], "lr": float(lrs["adapter_dispersion"]),
             "weight_decay": 0.0},
        ]


def build_flexible_model(arch: ArchConfig, *, in_dim: int, n_bins: int, log_r_init: Sequence[float] | np.ndarray,
                         seed: int) -> FlexibleResidualStateModel:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2 ** 32))
    return FlexibleResidualStateModel(arch, in_dim, n_bins, log_r_init)
