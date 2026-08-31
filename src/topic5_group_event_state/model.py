"""Event-driven latent state model for Group-Event State v0.1.

One complete interictal group event is one causal step.  At every step:

    predict timing of the next event from z(t_prev^+)
    evolve  z(t^-) = b + (z(t_prev^+) - b) * exp(-dt / tau)      [real seconds]
    predict this event's content from z(t^-)                     [no leakage]
    optionally correct z(t^-) with background SEEG observed before t
    encode the observed event and update z(t^+)

The event's own waveform, participation, delays and band content reach the state
only *after* its likelihood has been computed, so nothing about the current event
can leak into its own prediction.

State runs at two timescales.  Both use ``tau = exp(clamp(log_tau, ...))``:
``softplus`` would cap the achievable timescale far below the hours this project
needs, which is a failure mode this codebase has already paid for once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn


# ----------------------------------------------------------------------------- config


@dataclass
class EncoderConfig:
    """Which observation channels this arm is allowed to see."""

    use_participation: bool = True
    use_exact_delay: bool = True
    use_tied_groups: bool = True
    use_legacy_rank: bool = False
    use_waveform: bool = True
    use_multiband: bool = True
    use_geometry: bool = True
    d_contact: int = 64
    d_event: int = 128
    n_attention_heads: int = 4
    n_attention_layers: int = 1
    waveform_channels: int = 32
    dropout: float = 0.1


@dataclass
class StateConfig:
    d_fast: int = 64
    d_slow: int = 32
    tau_fast_range_s: tuple[float, float] = (1.0, 3600.0)
    tau_slow_range_s: tuple[float, float] = (60.0, 172800.0)
    tau_fast_init_s: tuple[float, float] = (10.0, 600.0)
    tau_slow_init_s: tuple[float, float] = (1800.0, 86400.0)
    slow_step_scale: float = 0.05
    persistent: bool = True
    use_real_dt: bool = True
    surrogate_dt_s: float = 1.0
    use_background: bool = False


@dataclass
class DataShape:
    n_contacts: int
    n_bands: int
    n_band_features: int
    n_cross_band_pairs: int
    n_views: int
    n_waveform_samples: int
    n_envelope_bins: int
    n_background_features: int
    band_available: tuple[bool, ...]


@dataclass
class InputStats:
    """Robust per-modality scales, estimated on the TRAIN split only.

    Raw inputs are physical: waveforms are hundreds of microvolts and band
    energies are logs with an arbitrary offset.  Feeding those straight into a
    CNN under bfloat16 autocast diverges to NaN in the first epoch (observed on
    epilepsiae_384).  Normalisation is therefore part of the model, and its
    statistics are frozen into the checkpoint.
    """

    waveform_scale: float
    envelope_scale: float
    cross_band_scale: float
    band_feature_mean: Any
    band_feature_std: Any
    background_mean: Any
    background_std: Any


@dataclass
class TargetStats:
    """Train-split target locations, used only to initialise head biases."""

    delay_mean: float
    delay_log_sigma: float
    band_energy_mean: Any
    band_energy_log_sigma: Any
    band_peak_mean: Any
    band_peak_log_sigma: Any
    cross_band_mean: Any
    cross_band_log_sigma: Any
    timing_log_mean: float
    timing_log_sigma: float
    participation_logit: Any


@dataclass
class ArmSpec:
    name: str
    encoder: EncoderConfig
    state: StateConfig
    notes: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------- encoder


class WaveformBranch(nn.Module):
    """Shared strided 1-D CNN over the native-rate waveform of every contact.

    A learned per-view embedding is added so the encoder always knows which
    reference montage a trace came from; concatenating views without that tag
    would let the model average two different physical signals.
    """

    def __init__(self, cfg: EncoderConfig, shape: DataShape, d_out: int, scale: float = 1.0):
        super().__init__()
        c = cfg.waveform_channels
        self.register_buffer("scale", torch.tensor(float(max(scale, 1e-6))))
        self.view_embed = nn.Parameter(torch.zeros(shape.n_views, 1))
        self.net = nn.Sequential(
            nn.Conv1d(shape.n_views, c, kernel_size=15, stride=4, padding=7),
            nn.GELU(),
            nn.Conv1d(c, c * 2, kernel_size=9, stride=4, padding=4),
            nn.GELU(),
            nn.Conv1d(c * 2, c * 2, kernel_size=9, stride=4, padding=4),
            nn.GELU(),
        )
        self.proj = nn.Linear(c * 4, d_out)

    def forward(self, waveform: Tensor) -> Tensor:
        # waveform: (B, C, V, T)
        b, c, v, t = waveform.shape
        x = torch.nan_to_num(waveform).reshape(b * c, v, t) / self.scale + self.view_embed
        h = self.net(x)
        pooled = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=-1)
        return self.proj(pooled).reshape(b, c, -1)


class TimeFrequencyBranch(nn.Module):
    """Per-band energy trajectory, band summaries and cross-band lags.

    Unsupported bands are masked to zero *and* flagged, never imputed: a band the
    sampling rate cannot represent is missing, not silent.
    """

    def __init__(
        self,
        cfg: EncoderConfig,
        shape: DataShape,
        d_out: int,
        stats: "InputStats | None" = None,
    ):
        super().__init__()
        self.register_buffer(
            "band_mask",
            torch.tensor(shape.band_available, dtype=torch.float32).view(1, 1, -1, 1),
        )
        env_scale = float(stats.envelope_scale) if stats else 1.0
        self.register_buffer("env_scale", torch.tensor(max(env_scale, 1e-6)))
        self.register_buffer("lag_scale", torch.tensor(
            max(float(stats.cross_band_scale) if stats else 1.0, 1e-6)))
        feat_mean = (
            torch.as_tensor(stats.band_feature_mean, dtype=torch.float32)
            if stats else torch.zeros(shape.n_bands, shape.n_band_features)
        )
        feat_std = (
            torch.as_tensor(stats.band_feature_std, dtype=torch.float32)
            if stats else torch.ones(shape.n_bands, shape.n_band_features)
        )
        self.register_buffer("feat_mean", feat_mean.view(1, 1, shape.n_bands, -1))
        self.register_buffer("feat_std", feat_std.clamp_min(1e-3).view(1, 1, shape.n_bands, -1))
        c = cfg.waveform_channels
        self.env = nn.Sequential(
            nn.Conv1d(shape.n_bands, c, kernel_size=9, stride=3, padding=4),
            nn.GELU(),
            nn.Conv1d(c, c, kernel_size=7, stride=3, padding=3),
            nn.GELU(),
        )
        n_summary = shape.n_bands * (shape.n_band_features + 1) + shape.n_cross_band_pairs
        self.proj = nn.Linear(c * 2 + n_summary, d_out)

    def forward(
        self, envelope: Tensor, band_features: Tensor, cross_band_lag: Tensor
    ) -> Tensor:
        # envelope (B,C,Bd,E); band_features (B,C,Bd,F); cross_band_lag (B,C,P)
        b, c, nb, e = envelope.shape
        env = torch.log1p(torch.nan_to_num(envelope).clamp_min(0.0) / self.env_scale)
        env = (env * self.band_mask).reshape(b * c, nb, e)
        h = self.env(env)
        pooled = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=-1).reshape(b, c, -1)
        feat = ((torch.nan_to_num(band_features) - self.feat_mean) / self.feat_std) * self.band_mask
        mask_flag = self.band_mask.expand(b, c, nb, 1)
        summary = torch.cat([feat, mask_flag], dim=-1).reshape(b, c, -1)
        lag = torch.nan_to_num(cross_band_lag) / self.lag_scale
        return self.proj(torch.cat([pooled, summary, lag], dim=-1))


class EventEncoder(nn.Module):
    """One complete group event -> contact tokens + a pooled event embedding."""

    def __init__(
        self,
        cfg: EncoderConfig,
        shape: DataShape,
        geometry: Tensor | None,
        stats: "InputStats | None" = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.shape = shape
        d = cfg.d_contact
        self.contact_embed = nn.Embedding(shape.n_contacts, d)

        n_struct = 0
        if cfg.use_participation:
            n_struct += 1
        if cfg.use_exact_delay:
            n_struct += 3  # delay, delay-rank fraction, finite flag
        if cfg.use_tied_groups:
            n_struct += 2  # group index fraction, group size fraction
        if cfg.use_legacy_rank:
            n_struct += 1
        self.struct = nn.Linear(max(n_struct, 1), d) if n_struct else None

        if cfg.use_geometry and geometry is not None:
            self.register_buffer("geometry", geometry)
            self.geom = nn.Linear(geometry.shape[-1], d)
        else:
            self.geometry = None
            self.geom = None

        self.waveform = (
            WaveformBranch(cfg, shape, d, stats.waveform_scale if stats else 1.0)
            if cfg.use_waveform
            else None
        )
        self.tf = TimeFrequencyBranch(cfg, shape, d, stats) if cfg.use_multiband else None

        self.norm = nn.LayerNorm(d)
        self.attn = nn.ModuleList(
            [
                nn.MultiheadAttention(d, cfg.n_attention_heads, batch_first=True, dropout=cfg.dropout)
                for _ in range(cfg.n_attention_layers)
            ]
        )
        self.attn_norm = nn.ModuleList(
            [nn.LayerNorm(d) for _ in range(cfg.n_attention_layers)]
        )
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 2), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(d * 2, d)
        )
        self.pool_proj = nn.Linear(2 * d + 2, cfg.d_event)

    def forward(self, batch: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
        part = batch["participation"].float()  # (B, C)
        b, c = part.shape
        tokens = self.contact_embed.weight.unsqueeze(0).expand(b, c, -1).clone()

        struct_parts: list[Tensor] = []
        if self.cfg.use_participation:
            struct_parts.append(part.unsqueeze(-1))
        if self.cfg.use_exact_delay:
            delay = batch["rel_delay"]
            finite = torch.isfinite(delay).float() * part
            filled = torch.nan_to_num(delay) * finite
            denom = filled.amax(dim=1, keepdim=True).clamp_min(1e-3)
            order = torch.argsort(torch.argsort(torch.where(finite > 0, filled, torch.full_like(filled, 1e6)), dim=1), dim=1).float()
            struct_parts += [
                (filled / denom).unsqueeze(-1),
                (order / max(c - 1, 1) * finite).unsqueeze(-1),
                finite.unsqueeze(-1),
            ]
        if self.cfg.use_tied_groups:
            tie = batch["tied_group_id"].float()
            valid = (tie >= 0).float()
            n_groups = tie.amax(dim=1, keepdim=True).clamp_min(0) + 1
            size = (tie.unsqueeze(-1) == tie.unsqueeze(1)).float().sum(-1) * valid
            struct_parts += [
                (tie.clamp_min(0) / n_groups * valid).unsqueeze(-1),
                (size / max(c, 1)).unsqueeze(-1),
            ]
        if self.cfg.use_legacy_rank:
            rank = batch["legacy_rank"].float() * part
            struct_parts.append((rank / max(c - 1, 1)).unsqueeze(-1))
        if self.struct is not None and struct_parts:
            tokens = tokens + self.struct(torch.cat(struct_parts, dim=-1))

        if self.geom is not None:
            tokens = tokens + self.geom(self.geometry).unsqueeze(0)
        if self.waveform is not None:
            tokens = tokens + self.waveform(batch["waveform"])
        if self.tf is not None:
            tokens = tokens + self.tf(
                batch["band_envelope"], batch["band_features"], batch["cross_band_lag"]
            )

        tokens = self.norm(tokens)
        key_padding = ~batch["contact_ok"]
        # A row with every key masked makes softmax return NaN; let such a row
        # attend to everything instead of silently poisoning the state.
        all_masked = key_padding.all(dim=1, keepdim=True)
        key_padding = key_padding & ~all_masked
        for attn, norm in zip(self.attn, self.attn_norm):
            attended, _ = attn(tokens, tokens, tokens, key_padding_mask=key_padding)
            tokens = norm(tokens + attended)
        tokens = tokens + self.ffn(tokens)

        weight = (~key_padding).float().unsqueeze(-1)
        mean = (tokens * weight).sum(1) / weight.sum(1).clamp_min(1.0)
        peak = tokens.masked_fill(key_padding.unsqueeze(-1), -1e4).amax(dim=1)
        size = part.sum(1, keepdim=True) / max(c, 1)
        span = torch.nan_to_num(batch["rel_delay"]).amax(dim=1, keepdim=True)
        event = self.pool_proj(torch.cat([mean, peak, size, span], dim=-1))
        return event, tokens


# ----------------------------------------------------------------------------- state


def _log_uniform_init(n: int, lo: float, hi: float, generator: torch.Generator | None) -> Tensor:
    u = torch.rand(n, generator=generator)
    return math.log(lo) + u * (math.log(hi) - math.log(lo))


class ContinuousState(nn.Module):
    """Two-timescale latent state that evolves in real seconds between events."""

    def __init__(self, cfg: StateConfig, d_event: int, generator: torch.Generator | None = None):
        super().__init__()
        self.cfg = cfg
        self.log_tau_fast = nn.Parameter(
            _log_uniform_init(cfg.d_fast, *cfg.tau_fast_init_s, generator)
        )
        self.log_tau_slow = nn.Parameter(
            _log_uniform_init(cfg.d_slow, *cfg.tau_slow_init_s, generator)
        )
        self.bias_fast = nn.Parameter(torch.zeros(cfg.d_fast))
        self.bias_slow = nn.Parameter(torch.zeros(cfg.d_slow))
        self.init_fast = nn.Parameter(torch.zeros(cfg.d_fast))
        self.init_slow = nn.Parameter(torch.zeros(cfg.d_slow))
        self.fast_cell = nn.GRUCell(d_event + cfg.d_slow, cfg.d_fast)
        self.slow_gate = nn.Linear(d_event + cfg.d_slow + cfg.d_fast, cfg.d_slow)
        self.slow_delta = nn.Linear(d_event + cfg.d_slow + cfg.d_fast, cfg.d_slow)

    def taus(self) -> tuple[Tensor, Tensor]:
        lo_f, hi_f = self.cfg.tau_fast_range_s
        lo_s, hi_s = self.cfg.tau_slow_range_s
        tau_f = torch.exp(self.log_tau_fast.clamp(math.log(lo_f), math.log(hi_f)))
        tau_s = torch.exp(self.log_tau_slow.clamp(math.log(lo_s), math.log(hi_s)))
        return tau_f, tau_s

    def initial(self, batch: int, device: torch.device) -> tuple[Tensor, Tensor]:
        return (
            self.init_fast.unsqueeze(0).expand(batch, -1).contiguous().to(device),
            self.init_slow.unsqueeze(0).expand(batch, -1).contiguous().to(device),
        )

    def evolve(
        self,
        z_fast: Tensor,
        z_slow: Tensor,
        dt: Tensor,
        taus: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Relax both states toward their bias over the real elapsed interval.

        ``taus`` may be precomputed once per chunk; recomputing the clamped
        exponential inside a 128-step Python loop is pure kernel-launch overhead.
        """

        if not self.cfg.use_real_dt:
            dt = torch.full_like(dt, self.cfg.surrogate_dt_s)
        dt = dt.clamp_min(0.0).unsqueeze(-1).to(torch.float32)
        tau_f, tau_s = taus if taus is not None else self.taus()
        decay_f = torch.exp(-dt / tau_f)
        decay_s = torch.exp(-dt / tau_s)
        z_fast = self.bias_fast + (z_fast - self.bias_fast) * decay_f
        z_slow = self.bias_slow + (z_slow - self.bias_slow) * decay_s
        return z_fast, z_slow

    def update(
        self, z_fast: Tensor, z_slow: Tensor, event: Tensor
    ) -> tuple[Tensor, Tensor]:
        if not self.cfg.persistent:
            return self.initial(event.shape[0], event.device)
        new_fast = self.fast_cell(torch.cat([event, z_slow], dim=-1), z_fast)
        joint = torch.cat([event, z_slow, new_fast], dim=-1)
        gate = torch.sigmoid(self.slow_gate(joint))
        delta = torch.tanh(self.slow_delta(joint))
        new_slow = z_slow + self.cfg.slow_step_scale * gate * delta
        return new_fast, new_slow


class BackgroundCorrector(nn.Module):
    """Additive correction from the last background window observed before t."""

    def __init__(
        self,
        n_contacts: int,
        n_features: int,
        d_fast: int,
        d_slow: int,
        stats: "InputStats | None" = None,
    ):
        super().__init__()
        mean = (
            torch.as_tensor(stats.background_mean, dtype=torch.float32)
            if stats else torch.zeros(n_features)
        )
        std = (
            torch.as_tensor(stats.background_std, dtype=torch.float32)
            if stats else torch.ones(n_features)
        )
        self.register_buffer("bg_mean", mean.view(1, 1, -1))
        self.register_buffer("bg_std", std.clamp_min(1e-3).view(1, 1, -1))
        self.summary = nn.Sequential(
            nn.Linear(2 * n_features + 1, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU()
        )
        self.to_fast = nn.Linear(64, d_fast)
        self.to_slow = nn.Linear(64, d_slow)

    def encode(self, background: Tensor, age_s: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        """Background depends only on the observation, so it is encoded for the
        whole chunk at once and merely added inside the sequential recurrence."""

        bg = (torch.nan_to_num(background) - self.bg_mean) / self.bg_std
        pooled = torch.cat(
            [bg.mean(1), bg.amax(1), torch.log1p(age_s.clamp_min(0.0)).unsqueeze(-1)], dim=-1
        )
        h = self.summary(pooled) * valid.float().unsqueeze(-1)
        return self.to_fast(h), 0.1 * self.to_slow(h)

    def forward(
        self, z_fast: Tensor, z_slow: Tensor, background: Tensor, age_s: Tensor, valid: Tensor
    ) -> tuple[Tensor, Tensor]:
        d_fast, d_slow = self.encode(background, age_s, valid)
        return z_fast + d_fast, z_slow + d_slow


# ----------------------------------------------------------------------------- heads


class PredictionHeads(nn.Module):
    """Every endpoint the plan asks to be reported separately gets its own head."""

    def __init__(self, d_state: int, shape: DataShape, hidden: int = 128):
        super().__init__()
        c, nb, npair = shape.n_contacts, shape.n_bands, shape.n_cross_band_pairs
        self.trunk = nn.Sequential(
            nn.Linear(d_state, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU()
        )
        self.timing = nn.Linear(hidden, 2)
        self.participation = nn.Linear(hidden, c)
        self.delay = nn.Linear(hidden, 2 * c)
        self.band_energy = nn.Linear(hidden, 2 * c * nb)
        self.band_peak = nn.Linear(hidden, 2 * c * nb)
        self.cross_band = nn.Linear(hidden, 2 * c * npair)
        self.shape = shape

    @torch.no_grad()
    def initialise_from_targets(self, stats: "TargetStats") -> None:
        """Start every head at the train-split location of its own target.

        Without this the band-energy head starts at 0 while its target sits near
        log-energy of order -10, and the first gradients are large enough to blow
        the shared trunk up.
        """

        c, nb, npair = self.shape.n_contacts, self.shape.n_bands, self.shape.n_cross_band_pairs
        self.timing.bias.copy_(torch.tensor([stats.timing_log_mean, stats.timing_log_sigma]))
        self.timing.weight.mul_(0.01)
        self.participation.bias.copy_(torch.as_tensor(stats.participation_logit, dtype=torch.float32))
        self.participation.weight.mul_(0.01)
        for layer, mean, log_sigma, shape_out in (
            (self.delay, stats.delay_mean, stats.delay_log_sigma, (c,)),
            (self.band_energy, stats.band_energy_mean, stats.band_energy_log_sigma, (c, nb)),
            (self.band_peak, stats.band_peak_mean, stats.band_peak_log_sigma, (c, nb)),
            (self.cross_band, stats.cross_band_mean, stats.cross_band_log_sigma, (c, npair)),
        ):
            mu = torch.as_tensor(mean, dtype=torch.float32).expand(*shape_out).clone()
            ls = torch.as_tensor(log_sigma, dtype=torch.float32).expand(*shape_out).clone()
            layer.bias.copy_(torch.stack([mu, ls], dim=-1).reshape(-1))
            layer.weight.mul_(0.01)

    def forward(self, state: Tensor) -> dict[str, Tensor]:
        h = self.trunk(state)
        c, nb, npair = self.shape.n_contacts, self.shape.n_bands, self.shape.n_cross_band_pairs
        timing = self.timing(h)
        delay = self.delay(h).reshape(-1, c, 2)
        energy = self.band_energy(h).reshape(-1, c, nb, 2)
        peak = self.band_peak(h).reshape(-1, c, nb, 2)
        cross = self.cross_band(h).reshape(-1, c, npair, 2)
        return {
            "timing_mu": timing[:, 0],
            "timing_log_sigma": timing[:, 1].clamp(-4.0, 3.0),
            "participation_logit": self.participation(h),
            "delay_mu": delay[..., 0],
            "delay_log_sigma": delay[..., 1].clamp(-7.0, 2.0),
            "band_energy_mu": energy[..., 0],
            "band_energy_log_sigma": energy[..., 1].clamp(-4.0, 4.0),
            "band_peak_mu": peak[..., 0],
            "band_peak_log_sigma": peak[..., 1].clamp(-7.0, 2.0),
            "cross_band_mu": cross[..., 0],
            "cross_band_log_sigma": cross[..., 1].clamp(-7.0, 2.0),
        }


class RecentHistoryFeatures(nn.Module):
    """Memoryless baseline input: fixed summaries of the last k events.

    This is the arm that must be beaten before any 'the model learned a state'
    sentence is allowed: it already knows the recent past, just not a state.
    """

    def __init__(self, n_features: int, d_out: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.GELU(), nn.Linear(hidden, d_out), nn.GELU()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(torch.nan_to_num(x))


# ----------------------------------------------------------------------------- losses


def gaussian_nll(value: Tensor, mu: Tensor, log_sigma: Tensor, mask: Tensor) -> Tensor:
    """Masked Gaussian NLL, accumulated in float32 regardless of autocast."""

    value = value.float()
    mu = mu.float()
    log_sigma = log_sigma.float()
    inv_var = torch.exp(-2.0 * log_sigma)
    nll = 0.5 * (math.log(2 * math.pi) + 2.0 * log_sigma + (value - mu) ** 2 * inv_var)
    nll = torch.where(mask, nll, torch.zeros_like(nll))
    return nll.sum(), mask.float().sum()


def lognormal_nll(dt: Tensor, mu: Tensor, log_sigma: Tensor) -> tuple[Tensor, Tensor]:
    """NLL of a strictly positive interval under a log-normal density."""

    x = dt.float().clamp_min(1e-3)
    log_x = torch.log(x)
    log_sigma = log_sigma.float()
    inv_var = torch.exp(-2.0 * log_sigma)
    nll = 0.5 * (math.log(2 * math.pi) + 2.0 * log_sigma + (log_x - mu.float()) ** 2 * inv_var) + log_x
    return nll.sum(), torch.tensor(float(x.numel()), device=x.device)
