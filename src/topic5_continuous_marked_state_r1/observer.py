"""Explicit-feature observer with a zero-initialised raw Transformer residual."""
from __future__ import annotations

import math

import torch
from torch import nn


class RawTemporalTransformer(nn.Module):
    """Shared per-contact 30 s waveform encoder."""

    def __init__(self, *, d_model: int = 64, patch_samples: int = 128,
                 n_heads: int = 4, n_layers: int = 2,
                 max_patches: int = 128):
        super().__init__()
        self.d_model = int(d_model)
        self.patch_samples = int(patch_samples)
        self.tokenizer = nn.Conv1d(
            1, d_model, kernel_size=patch_samples, stride=patch_samples,
            bias=True,
        )
        self.valid_projection = nn.Linear(1, d_model, bias=False)
        self.position = nn.Parameter(torch.zeros(1, max_patches, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.position, std=0.01)

    def forward(self, waveform: torch.Tensor,
                sample_valid: torch.Tensor) -> torch.Tensor:
        if waveform.shape != sample_valid.shape or waveform.ndim != 3:
            raise ValueError("waveform/sample_valid must have shape (B,C,T)")
        batch, contacts, samples = waveform.shape
        patches = samples // self.patch_samples
        if patches < 1 or patches > self.position.shape[1]:
            raise ValueError("unsupported raw window/patch count")
        used = patches * self.patch_samples
        valid = sample_valid[..., :used].to(waveform.dtype)
        clean = torch.where(
            sample_valid[..., :used].to(torch.bool),
            waveform[..., :used], torch.zeros_like(waveform[..., :used]),
        )
        flat = clean.reshape(batch * contacts, 1, used)
        token = self.tokenizer(flat).transpose(1, 2)
        patch_valid = valid.reshape(
            batch * contacts, patches, self.patch_samples
        ).mean(-1)
        token = (
            token + self.valid_projection(patch_valid.unsqueeze(-1))
            + self.position[:, :patches]
        )
        padding = patch_valid < 0.50
        # Transformer softmax is undefined if every token is masked.  Keep one
        # all-zero sentinel token; the contact mask removes this contact later.
        all_bad = padding.all(-1)
        if bool(all_bad.any()):
            padding = padding.clone()
            padding[all_bad, 0] = False
            token = token.clone()
            token[all_bad, 0] = 0.0
        encoded = self.transformer(token, src_key_padding_mask=padding)
        keep = (~padding).to(encoded.dtype)
        pooled = (encoded * keep.unsqueeze(-1)).sum(1) / keep.sum(1, keepdim=True).clamp(min=1)
        return self.norm(pooled).reshape(batch, contacts, self.d_model)


class ObservationTransformer(nn.Module):
    """Per-contact explicit+raw fusion followed by masked spatial attention."""

    def __init__(self, explicit_dim: int, *, d_model: int = 64,
                 patch_samples: int = 128, n_heads: int = 4,
                 temporal_layers: int = 2, spatial_layers: int = 1,
                 max_shafts: int = 64, raw_enabled: bool = True):
        super().__init__()
        self.raw_enabled = bool(raw_enabled)
        self.d_model = int(d_model)
        self.explicit = nn.Sequential(
            nn.Linear(explicit_dim, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
        )
        self.raw = RawTemporalTransformer(
            d_model=d_model, patch_samples=patch_samples, n_heads=n_heads,
            n_layers=temporal_layers,
        ) if raw_enabled else None
        self.raw_gain = nn.Parameter(torch.zeros(()), requires_grad=raw_enabled)
        self.coordinate = nn.Sequential(
            nn.Linear(4, d_model), nn.Tanh(), nn.Linear(d_model, d_model),
        )
        self.shaft = nn.Embedding(max_shafts, d_model)
        self.pool_token = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.spatial = nn.TransformerEncoder(layer, num_layers=spatial_layers)
        self.output_norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.pool_token, std=0.02)

    def forward(self, explicit: torch.Tensor, waveform: torch.Tensor,
                sample_valid: torch.Tensor, contact_mask: torch.Tensor,
                coordinates: torch.Tensor, coordinate_valid: torch.Tensor,
                shaft_index: torch.Tensor,
                use_raw: bool | None = None) -> torch.Tensor:
        if explicit.ndim != 3:
            raise ValueError("explicit features must have shape (B,C,F)")
        batch, contacts, _ = explicit.shape
        if contact_mask.shape != (batch, contacts):
            raise ValueError("contact mask shape disagrees")
        if coordinates.shape != (batch, contacts, 3):
            raise ValueError("coordinates shape disagrees")
        if coordinate_valid.shape != (batch, contacts):
            raise ValueError("coordinate-valid shape disagrees")
        if shaft_index.shape != (batch, contacts):
            raise ValueError("shaft index shape disagrees")
        node = self.explicit(explicit)
        raw_active = self.raw is not None if use_raw is None else bool(use_raw)
        if raw_active and self.raw is None:
            raise ValueError("use_raw=True but the observer has no raw encoder")
        if raw_active:
            raw = self.raw(waveform, sample_valid)
            node = node + self.raw_gain * raw
        coord_input = torch.cat([
            torch.where(
                coordinate_valid.unsqueeze(-1), coordinates,
                torch.zeros_like(coordinates),
            ),
            coordinate_valid.to(coordinates.dtype).unsqueeze(-1),
        ], dim=-1)
        node = node + self.coordinate(coord_input) + self.shaft(shaft_index)
        node = torch.where(contact_mask.unsqueeze(-1), node, torch.zeros_like(node))
        pool = self.pool_token.expand(batch, -1, -1)
        sequence = torch.cat([pool, node], dim=1)
        padding = torch.cat([
            torch.zeros((batch, 1), dtype=torch.bool, device=node.device),
            ~contact_mask.to(torch.bool),
        ], dim=1)
        encoded = self.spatial(sequence, src_key_padding_mask=padding)
        return self.output_norm(encoded[:, 0])


def copy_common_observer_state(source: ObservationTransformer,
                               target: ObservationTransformer) -> None:
    """Copy only parameters shared by explicit-only and explicit+raw arms."""
    source_state = source.state_dict()
    target_state = target.state_dict()
    common = {
        key: value for key, value in source_state.items()
        if key in target_state and target_state[key].shape == value.shape
        and not key.startswith("raw.")
    }
    target.load_state_dict(common, strict=False)
