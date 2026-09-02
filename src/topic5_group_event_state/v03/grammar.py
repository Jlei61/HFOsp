"""Patient-specific contact grammar with an exact tied-group likelihood.

The legacy decoder supplies a useful contact scaffold and within-event GRU, but
its old categorical action likelihood is not the v0.3 scientific instrument.
This wrapper calibrates group size/STOP and contact logits on outer TRAIN under
the exact unordered-without-replacement likelihood.  After calibration every
grammar parameter is frozen; small state adapters remain trainable and gradient
flows through the frozen operations to the cross-event state producer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import Tensor, nn

from src.topic5_continuous_marked_state_r1.mark_likelihood import (
    TiedMarkTerms,
    tied_group_mark_log_prob,
)
from src.topic5_rank_distribution import FullHistorySequenceGRU


@dataclass(frozen=True)
class GrammarInputs:
    contact_features: Tensor
    contact_mask: Tensor
    local_offset: Tensor


class FrozenContactGrammar(nn.Module):
    """Exact tied-group decoder around a legacy patient-specific GRU scaffold."""

    def __init__(
        self,
        base: FullHistorySequenceGRU,
        inputs: GrammarInputs,
        *,
        state_dim: int,
        max_group_size: int | None = None,
        adapter_rank: int = 8,
    ) -> None:
        super().__init__()
        self.base = base
        self.register_buffer("contact_features", inputs.contact_features.float())
        self.register_buffer("contact_mask", inputs.contact_mask.bool())
        # Patient-local contact offsets are grammar parameters.  In the primary
        # arm they start at zero and are fitted on outer TRAIN only; importing a
        # legacy held-out offset would silently import the legacy time split.
        self.local_offset = nn.Parameter(inputs.local_offset.float())
        n_contacts = int(self.contact_features.shape[0])
        self.n_contacts = n_contacts
        self.max_group_size = int(max_group_size or n_contacts)
        hidden = int(base.hidden_size)
        embed = int(base.contact_embedding_dim)

        # Grammar-v0.3 calibration.  These parameters are trained on outer TRAIN
        # before the state model exists, then frozen together with ``base``.
        self.positive_size_head = nn.Linear(hidden, n_contacts)
        self.contact_log_temperature = nn.Parameter(torch.tensor(0.0))
        self.contact_bias = nn.Parameter(torch.zeros(n_contacts))
        self.stop_scale_raw = nn.Parameter(torch.tensor(0.5413))  # softplus ~= 1
        self.stop_bias = nn.Parameter(torch.tensor(0.0))

        # Low-capacity state adapters.  Near-zero gates make state=0 reproduce
        # the calibrated grammar while allowing gradients to reach the state.
        rank = int(min(adapter_rank, state_dim, hidden, embed))
        self.state_norm = nn.LayerNorm(state_dim)
        self.state_to_initial = nn.Sequential(
            nn.Linear(state_dim, rank, bias=False), nn.Linear(rank, hidden, bias=False)
        )
        self.state_to_query = nn.Sequential(
            nn.Linear(state_dim, rank, bias=False), nn.Linear(rank, embed, bias=False)
        )
        self.state_to_size = nn.Sequential(
            nn.Linear(state_dim, rank, bias=False), nn.Linear(rank, n_contacts + 1, bias=False)
        )
        self.initial_gate = nn.Parameter(torch.tensor(-4.0))
        self.query_gate = nn.Parameter(torch.tensor(-4.0))
        self.size_gate = nn.Parameter(torch.tensor(-4.0))
        self._init_adapters()

    def _init_adapters(self) -> None:
        for module in (self.state_to_initial, self.state_to_query, self.state_to_size):
            first, last = module[0], module[1]
            nn.init.normal_(first.weight, std=0.02)
            # A mathematically exact zero output also gives the state producer an
            # exact zero mark gradient on step 1.  That reproduced the old
            # "raw branch never actually trained" failure mode.  The gate keeps
            # the perturbation tiny; the final projection must remain non-zero.
            nn.init.normal_(last.weight, std=1e-3)

    @property
    def calibration_parameters(self) -> list[nn.Parameter]:
        return list(self.base.parameters()) + list(self.positive_size_head.parameters()) + [
            self.local_offset,
            self.contact_log_temperature,
            self.contact_bias,
            self.stop_scale_raw,
            self.stop_bias,
        ]

    @property
    def adapter_parameters(self) -> list[nn.Parameter]:
        out: list[nn.Parameter] = []
        for module in (self.state_norm, self.state_to_initial, self.state_to_query, self.state_to_size):
            out.extend(module.parameters())
        out.extend([self.initial_gate, self.query_gate, self.size_gate])
        return out

    def set_phase(self, phase: str) -> None:
        """Choose exactly one trainable parameter family."""

        if phase not in {"calibration", "adapter", "frozen"}:
            raise ValueError(f"unknown grammar phase {phase!r}")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        selected = (
            self.calibration_parameters if phase == "calibration"
            else self.adapter_parameters if phase == "adapter"
            else []
        )
        for parameter in selected:
            parameter.requires_grad_(True)

    def _expand_inputs(self, batch: int) -> tuple[Tensor, Tensor, Tensor]:
        features = self.contact_features.unsqueeze(0).expand(batch, -1, -1)
        mask = self.contact_mask.unsqueeze(0).expand(batch, -1)
        offset = self.local_offset.unsqueeze(0).expand(batch, -1, -1)
        return features, mask, offset

    def forward(
        self,
        group_ids: Tensor,
        group_count: Tensor,
        state: Tensor,
    ) -> tuple[TiedMarkTerms, Mapping[str, Tensor]]:
        if group_ids.ndim != 2 or group_ids.shape[1] != self.n_contacts:
            raise ValueError("group_ids has wrong shape")
        batch = int(group_ids.shape[0])
        if state.shape != (batch, self.state_norm.normalized_shape[0]):
            raise ValueError("state has wrong shape")
        features, mask, offset = self._expand_inputs(batch)
        embedding, encoder_input = self.base._encode(features, offset)
        h = self.base._initial_hidden(embedding, mask)
        s = self.state_norm(state)
        h = h + torch.sigmoid(self.initial_gate) * self.state_to_initial(s)
        state_query = torch.sigmoid(self.query_gate) * self.state_to_query(s)
        state_size = torch.sigmoid(self.size_gate) * self.state_to_size(s)

        recruited = torch.zeros_like(mask)
        n_steps = int(group_count.max().detach().cpu()) + 1
        contact_steps: list[Tensor] = []
        size_steps: list[Tensor] = []
        hidden_steps: list[Tensor] = []
        temperature = torch.exp(self.contact_log_temperature.clamp(-3.0, 3.0))
        stop_scale = nn.functional.softplus(self.stop_scale_raw) + 1e-4
        scale = math.sqrt(float(self.base.contact_embedding_dim))
        for step in range(n_steps):
            candidate = mask & ~recruited
            query = self.base.action_query(h) + state_query
            contact = torch.einsum("bce,be->bc", embedding, query) / scale
            contact = contact + self.base.action_bias(encoder_input).squeeze(-1)
            contact = contact / temperature + self.contact_bias
            contact = contact.masked_fill(~candidate, -1e9)
            stop = stop_scale * self.base.stop_head(h).squeeze(-1) + self.stop_bias
            positive = self.positive_size_head(h)
            size = torch.cat([stop[:, None], positive], dim=-1) + state_size
            # Impossible positive group sizes are removed by the exact scorer;
            # sizes above the explicit pilot cap are additionally forbidden.
            if self.max_group_size < self.n_contacts:
                size[:, self.max_group_size + 1 :] = -1e9
            contact_steps.append(contact)
            size_steps.append(size)
            hidden_steps.append(h)
            if step + 1 == n_steps:
                break
            current = (group_ids == step) & mask
            active = (group_count > step).unsqueeze(1)
            updated_recruited = recruited | current
            updated_h = self.base._advance(embedding, current, updated_recruited, h, mask)
            h = torch.where(active, updated_h, h)
            recruited = torch.where(active, updated_recruited, recruited)

        contact_logits = torch.stack(contact_steps, dim=1)
        size_logits = torch.stack(size_steps, dim=1)
        terms = tied_group_mark_log_prob(
            group_ids.long(), group_count.long(), size_logits, contact_logits, mask
        )
        return terms, {
            "contact_logits": contact_logits,
            "size_logits": size_logits,
            "hidden": torch.stack(hidden_steps, dim=1),
        }


def load_legacy_grammar(
    checkpoint: Path,
    dataset_npz: Path,
    *,
    state_dim: int,
    device: torch.device | str = "cpu",
) -> FrozenContactGrammar:
    """Load and validate one patient's legacy decoder as v0.3 initialisation."""

    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    with np.load(Path(dataset_npz), allow_pickle=True) as data:
        contact_features = np.asarray(data["contact_features"], dtype=np.float32)
        if "contact_mask" in data:
            contact_mask = np.asarray(data["contact_mask"], dtype=bool)
            if contact_mask.ndim > 1:
                contact_mask = contact_mask.any(axis=0)
        else:
            contact_mask = np.ones(contact_features.shape[0], dtype=bool)
    kwargs = dict(payload["model_kwargs"])
    base = FullHistorySequenceGRU(contact_features.shape[1], **kwargs)
    base.load_state_dict(payload["model_state"], strict=True)
    local_offset = torch.as_tensor(payload["heldout_local_offset"], dtype=torch.float32)
    if tuple(local_offset.shape[:1]) != (contact_features.shape[0],):
        raise ValueError("legacy local offset and contact universe disagree")
    grammar = FrozenContactGrammar(
        base,
        GrammarInputs(
            contact_features=torch.from_numpy(contact_features),
            contact_mask=torch.from_numpy(contact_mask),
            local_offset=local_offset,
        ),
        state_dim=state_dim,
    )
    grammar.set_phase("calibration")
    return grammar.to(device)


def build_train_only_grammar(
    architecture_checkpoint: Path,
    contact_features: np.ndarray,
    *,
    state_dim: int,
    device: torch.device | str = "cpu",
) -> FrozenContactGrammar:
    """Build the primary grammar without importing legacy learned weights.

    The legacy file is read only for architecture hyperparameters.  All network
    weights and patient-local offsets are newly initialised and therefore can be
    fitted solely on the new outer TRAIN partition.
    """

    payload = torch.load(
        Path(architecture_checkpoint), map_location="cpu", weights_only=False
    )
    features = np.asarray(contact_features, dtype=np.float32)
    kwargs = dict(payload["model_kwargs"])
    base = FullHistorySequenceGRU(features.shape[1], **kwargs)
    local_dim = int(kwargs.get("local_offset_dim", 8))
    grammar = FrozenContactGrammar(
        base,
        GrammarInputs(
            contact_features=torch.from_numpy(features),
            contact_mask=torch.ones(features.shape[0], dtype=torch.bool),
            local_offset=torch.zeros(features.shape[0], local_dim),
        ),
        state_dim=state_dim,
    )
    grammar.set_phase("calibration")
    return grammar.to(device)
