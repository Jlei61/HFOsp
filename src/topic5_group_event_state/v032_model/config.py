"""Frozen model configuration for v0.3.2 (design §3-§5)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any

from .paths import payload_hash

ARCHITECTURES = ("leaky_bank", "repaired_rnn")


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "leaky_bank"
    # 12-dimensional constrained bank: three physical scales x four channels.
    taus_seconds: tuple[float, ...] = (300.0, 1800.0, 7200.0)
    channels_per_tau: int = 4
    phi_dim: int = 4
    encoder_hidden: int = 32
    rnn_event_dim: int = 16
    rnn_hidden: int = 32
    # Safe residual adapter: non-zero alpha that is held fixed for the first steps.
    alpha_init: float = 0.03
    alpha_freeze_steps: int = 50
    lr_encoder: float = 1e-3
    lr_state: float = 1e-3
    lr_adapter: float = 3e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    # Full-batch optimisation over every state-train anchor per step.
    max_steps: int = 600
    min_steps: int = 100
    validate_every: int = 10
    patience: int = 10
    horizon_seconds: float = 1800.0
    diagnostic_horizons_seconds: tuple[float, ...] = (300.0,)
    secondary_horizon_seconds: float = 7200.0
    amp_encoder: bool = False
    # Exact closed-form trajectory is evaluated in rescaled chunks; the chunk
    # edge is a numerical device, not a gradient truncation, unless detach_chunks.
    chunk_seconds: float = 3600.0
    detach_chunks: bool = False
    shift_fractions: tuple[float, ...] = (0.5, 0.25, 0.75)
    bootstrap_resamples: int = 1000
    bootstrap_block_anchors: int = 6

    @property
    def state_dim(self) -> int:
        return len(self.taus_seconds) * self.channels_per_tau

    def validate(self) -> "ModelConfig":
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"unknown architecture {self.architecture!r}; allowed {ARCHITECTURES}")
        if self.channels_per_tau != self.phi_dim and self.architecture == "leaky_bank":
            raise ValueError("leaky bank writes phi into every tau block: phi_dim must equal channels_per_tau")
        if not 0.0 < self.alpha_init:
            raise ValueError("alpha_init must be strictly positive (a zero gate is a dead zone)")
        if self.alpha_freeze_steps < 0 or self.min_steps > self.max_steps:
            raise ValueError("invalid step schedule")
        if any(t <= 0 for t in self.taus_seconds):
            raise ValueError("taus must be positive seconds")
        return self

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        return payload_hash(self.as_dict())


def _coerce(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    return value


def load_config(path: Path | str | None = None, **overrides: Any) -> ModelConfig:
    """Load a YAML/JSON config (optional) and apply keyword overrides."""

    payload: dict[str, Any] = {}
    if path is not None:
        text = Path(path).read_text()
        try:
            import yaml  # type: ignore

            payload = dict(yaml.safe_load(text) or {})
        except ModuleNotFoundError:  # pragma: no cover - yaml is available in cuda_env
            payload = dict(json.loads(text))
    payload.update(overrides)
    known = {k: _coerce(v) for k, v in payload.items() if k in ModelConfig.__dataclass_fields__}
    unknown = sorted(set(payload) - set(known))
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    return replace(ModelConfig(), **known).validate()
