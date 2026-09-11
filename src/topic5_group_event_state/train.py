"""Training and evaluation for one (patient, arm, seed) of Group-Event State v0.1.

Causal order inside every step, enforced by construction:

    1. predict the interval to this event from the state left by the previous one
    2. relax the state over the real elapsed seconds
    3. optionally correct it with background SEEG observed strictly before now
    4. predict this event's content
    5. only now encode the observed event and update the state

Endpoints are reported separately, never pooled into one score: an arm that only
improves group size is an extent model, not a repertoire model, and the numbers
have to be able to say so.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .dataset import SubjectSequence
from .model import (
    ArmSpec,
    InputStats,
    TargetStats,
    BackgroundCorrector,
    ContinuousState,
    DataShape,
    EncoderConfig,
    EventEncoder,
    PredictionHeads,
    RecentHistoryFeatures,
    StateConfig,
    gaussian_nll,
    lognormal_nll,
)

ENDPOINTS = (
    "timing",
    "participation",
    "group_size",
    "delay",
    "band_energy",
    "band_peak",
    "cross_band_lag",
)

# History-truncation probes for H1: resetting the state every K events turns the
# same trained model into a K-event-memory model at evaluation time.
TRUNCATION_PROBES = (1, 20, 100, 0)  # 0 = full session history

# How many of the event's actual first contacts are excluded from scoring by the
# same-prefix continuation endpoint.
PREFIX_GIVEN = 2


def build_arms() -> dict[str, ArmSpec]:
    """The five core arms plus the ablations the plan names."""

    def enc(**kw: Any) -> EncoderConfig:
        base = dict(
            use_participation=True,
            use_exact_delay=False,
            use_tied_groups=False,
            use_legacy_rank=False,
            use_waveform=False,
            use_multiband=False,
            use_geometry=True,
        )
        base.update(kw)
        return EncoderConfig(**base)

    arms: dict[str, ArmSpec] = {}
    arms["a1_static_recent_history"] = ArmSpec(
        "a1_static_recent_history",
        enc(),
        StateConfig(persistent=False),
        "no latent state; fixed summaries of the last 1/5/20 events",
        {"baseline_only": True},
    )
    arms["a2_rank_group_state"] = ArmSpec(
        "a2_rank_group_state",
        enc(use_legacy_rank=True),
        StateConfig(),
        "participation + legacy integer rank only; the low-information ablation",
    )
    arms["a3_delay_group_state"] = ArmSpec(
        "a3_delay_group_state",
        enc(use_exact_delay=True, use_tied_groups=True),
        StateConfig(),
        "participation + tied recruitment groups + exact continuous delay",
    )
    arms["a4_full_multimodal_state"] = ArmSpec(
        "a4_full_multimodal_state",
        enc(use_exact_delay=True, use_tied_groups=True, use_waveform=True, use_multiband=True),
        StateConfig(),
        "adds native waveform views and multiband energy/lag structure",
    )
    arms["a5_full_plus_background"] = ArmSpec(
        "a5_full_plus_background",
        enc(use_exact_delay=True, use_tied_groups=True, use_waveform=True, use_multiband=True),
        StateConfig(use_background=True),
        "a4 plus background-SEEG observation correction",
    )

    full = arms["a4_full_multimodal_state"]
    arms["b1_no_real_dt"] = ArmSpec(
        "b1_no_real_dt", full.encoder, replace(full.state, use_real_dt=False),
        "a4 with the event-count clock instead of real elapsed seconds",
    )
    arms["b2_no_waveform"] = ArmSpec(
        "b2_no_waveform", replace(full.encoder, use_waveform=False), full.state,
        "a4 minus the waveform branch",
    )
    arms["b3_no_multiband"] = ArmSpec(
        "b3_no_multiband", replace(full.encoder, use_multiband=False), full.state,
        "a4 minus the multiband branch",
    )
    arms["b4_memoryless"] = ArmSpec(
        "b4_memoryless", full.encoder, replace(full.state, persistent=False),
        "a4 encoder, state reset at every event",
    )
    arms["b5_no_geometry"] = ArmSpec(
        "b5_no_geometry", replace(full.encoder, use_geometry=False), full.state,
        "a4 minus static contact geometry",
    )
    arms["b6_slow_only"] = ArmSpec(
        "b6_slow_only", full.encoder, replace(full.state, d_fast=8),
        "a4 with the fast state shrunk to 8 dimensions",
    )
    return arms


@dataclass
class TrainConfig:
    chunk_events: int = 128
    max_epochs: int = 24
    patience: int = 5
    lr_encoder: float = 3e-4
    lr_state: float = 1e-3
    lr_heads: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    n_streams: int = 8
    amp: bool = True
    max_train_seconds: float = 3600.0
    min_epochs: int = 3
    # Compatibility fields retained in the result schema.  v0.2 requires an
    # exact causal replay from the start of the currently observed session;
    # positive caps are rejected in ``train_one`` because an event-count cap is
    # neither a common physical-time warm-up nor a valid "full session" state.
    warm_events: int = 0
    eval_warm_events: int = 0


def estimate_stats(
    seq: SubjectSequence, train_lo: int, train_hi: int, *, max_events: int = 2048, seed: int = 0
) -> tuple[InputStats, TargetStats]:
    """Robust input scales and target locations from the TRAIN split only.

    Estimating these on the whole stream would leak test-time distribution into
    the model's normalisation constants, which is a quiet but real form of the
    hard-stop leakage the plan forbids.
    """

    rng = np.random.default_rng(seed)
    n = train_hi - train_lo
    take = min(max_events, n)
    pos = np.sort(rng.choice(np.arange(train_lo, train_hi), size=take, replace=False))
    batch = seq.gather_positions(pos)

    wave = np.abs(np.nan_to_num(batch["waveform"].astype(np.float32)))
    waveform_scale = float(np.percentile(wave[wave > 0], 90)) if np.any(wave > 0) else 1.0
    env = np.nan_to_num(batch["band_envelope"].astype(np.float32))
    envelope_scale = float(np.percentile(env[env > 0], 50)) if np.any(env > 0) else 1.0
    lag = np.nan_to_num(batch["cross_band_lag"].astype(np.float32))
    cross_band_scale = float(np.percentile(np.abs(lag), 90)) or 1.0

    feats = np.nan_to_num(batch["band_features"].astype(np.float32))  # (N, C, B, F)
    feat_mean = feats.mean(axis=(0, 1))
    feat_std = feats.std(axis=(0, 1))
    bg = np.nan_to_num(batch["background"].astype(np.float32))
    bg_mean = bg.mean(axis=(0, 1))
    bg_std = bg.std(axis=(0, 1))
    inputs = InputStats(
        waveform_scale=waveform_scale,
        envelope_scale=envelope_scale,
        cross_band_scale=cross_band_scale,
        band_feature_mean=feat_mean,
        band_feature_std=feat_std,
        background_mean=bg_mean,
        background_std=bg_std,
    )

    part = batch["participation"].astype(bool)
    delay = batch["rel_delay"].astype(np.float32)
    d_valid = part & np.isfinite(delay)
    delay_mean = float(delay[d_valid].mean()) if d_valid.any() else 0.0
    delay_sigma = float(delay[d_valid].std()) if d_valid.any() else 1.0
    energy = feats[:, :, :, 2]
    peak = feats[:, :, :, 0]
    mask = part[:, :, None]
    def _loc(x):
        vals = np.where(mask, x, np.nan)
        m = np.nanmean(vals, axis=(0, 1))
        sd = np.nanstd(vals, axis=(0, 1))
        return np.nan_to_num(m), np.log(np.clip(np.nan_to_num(sd, nan=1.0), 1e-4, None))
    e_mu, e_ls = _loc(energy)
    p_mu, p_ls = _loc(peak)
    lag_vals = np.where(part[:, :, None], lag, np.nan)
    l_mu = np.nan_to_num(np.nanmean(lag_vals, axis=(0, 1)))
    l_ls = np.log(np.clip(np.nan_to_num(np.nanstd(lag_vals, axis=(0, 1)), nan=1.0), 1e-4, None))
    dt = batch["dt_prev"].astype(np.float64)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    log_dt = np.log(dt) if dt.size else np.zeros(1)
    rate = np.clip(part.mean(axis=0), 1e-3, 1 - 1e-3)
    targets = TargetStats(
        delay_mean=delay_mean,
        delay_log_sigma=float(np.log(max(delay_sigma, 1e-4))),
        band_energy_mean=e_mu,
        band_energy_log_sigma=e_ls,
        band_peak_mean=p_mu,
        band_peak_log_sigma=p_ls,
        cross_band_mean=l_mu,
        cross_band_log_sigma=l_ls,
        timing_log_mean=float(np.mean(log_dt)),
        timing_log_sigma=float(np.log(max(np.std(log_dt), 1e-3))),
        participation_logit=np.log(rate / (1 - rate)).astype(np.float32),
    )
    return inputs, targets


class GroupEventStateModel(nn.Module):
    def __init__(self, arm: ArmSpec, shape: DataShape, geometry: Tensor | None,
                 n_history_features: int, generator: torch.Generator | None = None,
                 stats: InputStats | None = None, targets: TargetStats | None = None):
        super().__init__()
        self.arm = arm
        self.shape = shape
        self.baseline_only = bool(arm.extra.get("baseline_only", False))
        d_state = arm.state.d_fast + arm.state.d_slow
        if self.baseline_only:
            self.history = RecentHistoryFeatures(n_history_features, d_state)
            # Content may condition on the observed interval (a marked point
            # process factorises as p(t | past) * p(mark | t, past)); timing may not.
            self.content_mix = nn.Parameter(torch.eye(d_state + 1, d_state))
            self.encoder = None
            self.state = None
            self.background = None
        else:
            self.history = None
            self.encoder = EventEncoder(arm.encoder, shape, geometry, stats)
            self.state = ContinuousState(arm.state, arm.encoder.d_event, generator)
            self.background = (
                BackgroundCorrector(
                    shape.n_contacts, shape.n_background_features,
                    arm.state.d_fast, arm.state.d_slow, stats,
                )
                if arm.state.use_background
                else None
            )
        self.heads = PredictionHeads(d_state, shape)
        if targets is not None:
            self.heads.initialise_from_targets(targets)


def _to_device(batch: Mapping[str, np.ndarray], device: torch.device) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for key, value in batch.items():
        if value.dtype == np.bool_:
            out[key] = torch.from_numpy(np.ascontiguousarray(value)).to(device)
        elif value.dtype == np.float16:
            out[key] = torch.from_numpy(np.ascontiguousarray(value)).to(device).float()
        else:
            out[key] = torch.from_numpy(np.ascontiguousarray(value.astype(np.float32))).to(device)
    return out


def _endpoint_losses(
    pred: Mapping[str, Tensor],
    timing_pred: Mapping[str, Tensor],
    truth: Mapping[str, Tensor],
    dt: Tensor,
    dt_valid: Tensor,
    slot_valid: Tensor | None = None,
) -> dict[str, tuple[Tensor, Tensor]]:
    """All endpoints for a whole chunk at once.

    Vectorised on purpose: calling the heads and rebuilding these masks once per
    event turned the run into a kernel-launch benchmark (19% GPU utilisation),
    not a training run.  Only the state recurrence is inherently sequential.
    """

    part = truth["participation"].bool()
    ok = truth["contact_ok"].bool()
    if slot_valid is not None:
        # padded stream slots must not contribute a single observation
        part = part & slot_valid.unsqueeze(-1)
        ok = ok & slot_valid.unsqueeze(-1)
        dt_valid = dt_valid & slot_valid
    n_bands = truth["band_features"].shape[2]
    band_mask = part.unsqueeze(-1).expand(-1, -1, n_bands)
    pair_mask = part.unsqueeze(-1).expand(-1, -1, truth["cross_band_lag"].shape[2])

    out: dict[str, tuple[Tensor, Tensor]] = {}
    if bool(dt_valid.any()):
        keep = dt_valid
        total, count = lognormal_nll(
            dt[keep], timing_pred["timing_mu"][keep], timing_pred["timing_log_sigma"][keep]
        )
        out["timing"] = (total, count)

    logits = pred["participation_logit"]
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits.float(), part.float(), reduction="none"
    )
    bce = torch.where(ok, bce, torch.zeros_like(bce))
    out["participation"] = (bce.sum(), ok.float().sum())

    prob = torch.sigmoid(logits.float())
    size_pred = (prob * ok.float()).sum(-1)
    size_true = part.float().sum(-1)
    live = ok.any(-1).float()
    out["group_size"] = (((size_pred - size_true).abs() * live).sum(), live.sum())

    delay = truth["rel_delay"]
    out["delay"] = gaussian_nll(
        torch.nan_to_num(delay), pred["delay_mu"], pred["delay_log_sigma"],
        part & torch.isfinite(delay),
    )
    energy = truth["band_features"][:, :, :, 2]
    peak = truth["band_features"][:, :, :, 0]
    out["band_energy"] = gaussian_nll(
        torch.nan_to_num(energy), pred["band_energy_mu"], pred["band_energy_log_sigma"],
        band_mask & torch.isfinite(energy),
    )
    out["band_peak"] = gaussian_nll(
        torch.nan_to_num(peak), pred["band_peak_mu"], pred["band_peak_log_sigma"],
        band_mask & torch.isfinite(peak),
    )
    lag = truth["cross_band_lag"]
    out["cross_band_lag"] = gaussian_nll(
        torch.nan_to_num(lag), pred["cross_band_mu"], pred["cross_band_log_sigma"],
        pair_mask & torch.isfinite(lag),
    )
    return out


def endpoint_predictions(
    pred: Mapping[str, Tensor], truth: Mapping[str, Tensor]
) -> dict[str, np.ndarray]:
    """Derived, human-readable endpoints for H2a beyond the likelihoods."""

    part = truth["participation"].bool()
    delay = truth["rel_delay"]
    valid = part & torch.isfinite(delay)
    order_rho: list[float] = []
    tie_agree: list[float] = []
    prefix_rho: list[float] = []
    prefix_next_hit: list[float] = []
    for i in range(part.shape[0]):
        idx = torch.nonzero(valid[i], as_tuple=False).flatten()
        if idx.numel() < 3:
            continue
        truth_d = delay[i, idx].float().cpu().numpy()
        pred_d = pred["delay_mu"][i, idx].float().detach().cpu().numpy()
        if np.std(truth_d) == 0 or np.std(pred_d) == 0:
            continue
        tr = np.argsort(np.argsort(truth_d)).astype(float)
        pr = np.argsort(np.argsort(pred_d)).astype(float)
        order_rho.append(float(np.corrcoef(tr, pr)[0, 1]))
        tol = 0.010
        same_true = (np.abs(truth_d[:, None] - truth_d[None, :]) <= tol)
        same_pred = (np.abs(pred_d[:, None] - pred_d[None, :]) <= tol)
        iu = np.triu_indices(idx.numel(), k=1)
        tie_agree.append(float((same_true[iu] == same_pred[iu]).mean()))

        # Same-prefix continuation: score only the part of the sequence after the
        # two contacts that actually went first.
        #
        # This is NOT a prefix-conditional prediction. The delay head is
        # deliberately unconditional -- it emits every contact's delay without
        # seeing the participation mask, which is what keeps it leak-free -- so
        # excluding the observed prefix from scoring does not make the remaining
        # prediction depend on it. On synthetic data a model that knows only a
        # fixed per-contact habit still scores 0.771 here (next-contact hit 0.545
        # against a 0.167 chance rate). Read it only as an arm contrast: the
        # static-history baseline IS the fixed habit, so the difference against it
        # is what the state adds beyond habit. An absolute value means little.
        if idx.numel() >= 4:
            order_true = np.argsort(truth_d)
            rest = order_true[PREFIX_GIVEN:]
            rest_truth, rest_pred = truth_d[rest], pred_d[rest]
            if np.std(rest_truth) > 0 and np.std(rest_pred) > 0:
                prefix_rho.append(
                    float(np.corrcoef(
                        np.argsort(np.argsort(rest_truth)).astype(float),
                        np.argsort(np.argsort(rest_pred)).astype(float),
                    )[0, 1])
                )
            prefix_next_hit.append(float(int(np.argmin(rest_pred)) == int(np.argmin(rest_truth))))
    prob = torch.sigmoid(pred["participation_logit"].float()).detach().cpu().numpy()
    ok = truth["contact_ok"].float()
    return {
        "order_spearman": np.asarray(order_rho, dtype=np.float64),
        "tied_group_agreement": np.asarray(tie_agree, dtype=np.float64),
        "prefix_continuation_spearman": np.asarray(prefix_rho, dtype=np.float64),
        "prefix_next_contact_hit": np.asarray(prefix_next_hit, dtype=np.float64),
        "participation_prob": prob,
        "participation_true": part.cpu().numpy(),
        # Per-event series H3 needs: the part of each observed event the model did
        # not expect.  Residuals, not raw counts, are what "exposure beyond
        # expectation" means.
        "size_pred": (torch.sigmoid(pred["participation_logit"].float()) * ok).sum(-1).detach().cpu().numpy(),
        "size_true": part.float().sum(-1).cpu().numpy(),
        "timing_mu": pred["timing_mu"].float().detach().cpu().numpy(),
        "delay_span_true": torch.nan_to_num(delay).amax(-1).float().cpu().numpy(),
    }


class _Accum:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {k: 0.0 for k in ENDPOINTS}
        self.counts: dict[str, float] = {k: 0.0 for k in ENDPOINTS}

    def add(self, losses: Mapping[str, tuple[Tensor, Tensor]]) -> None:
        for key, (total, count) in losses.items():
            self.sums[key] += float(total.detach())
            self.counts[key] += float(count.detach())

    def totals(self) -> dict[str, tuple[float, float]]:
        return {k: (self.sums[k], self.counts[k]) for k in ENDPOINTS}

    def means(self) -> dict[str, float]:
        return {
            k: (self.sums[k] / self.counts[k]) if self.counts[k] > 0 else float("nan")
            for k in ENDPOINTS
        }


def run_streams(
    model: GroupEventStateModel,
    seq: SubjectSequence,
    lo: int,
    hi: int,
    device: torch.device,
    cfg: TrainConfig,
    optimizer: torch.optim.Optimizer,
    *,
    loss_weights: Mapping[str, float] | None = None,
    grad_norms: list[float] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Training pass that advances several contiguous segments at once.

    The recurrence and its backward are 92% of a single-stream step and their cost
    is per *step*, not per event, so running B segments side by side is close to a
    free B-fold speed-up.  Each segment keeps its own state chain, so truncated
    BPTT inside a segment is exactly what it was; only evaluation, which must
    honour the single true chronological chain, still runs one stream.
    """

    ranges = seq.streams(lo, hi, cfg.n_streams)
    # The baseline arm has no encoder and no state chain, so there is nothing to
    # parallelise and nothing that would survive being split.
    if model.baseline_only or len(ranges) == 1:
        return run_sequence(
            model, seq, lo, hi, device, cfg, train=True, optimizer=optimizer,
            loss_weights=loss_weights, grad_norms=grad_norms,
        )

    accum = _Accum()
    weights = dict(loss_weights or {k: 1.0 for k in ENDPOINTS})
    extra: dict[str, Any] = {"n_events": 0, "state_norm": [], "slow_delta": []}
    b = len(ranges)
    lengths = np.array([r[1] - r[0] for r in ranges])
    starts = np.array([r[0] for r in ranges])
    max_len = int(lengths.max())
    fast = slow = None

    for offset in range(0, max_len, cfg.chunk_events):
        length = min(cfg.chunk_events, max_len - offset)
        pos = starts[:, None] + offset + np.arange(length)[None, :]
        valid = pos < (starts + lengths)[:, None]
        pos = np.minimum(pos, (starts + lengths - 1)[:, None])
        raw = seq.gather_positions(pos.reshape(-1))
        batch = _to_device(raw, device)
        slot_valid = torch.from_numpy(valid.reshape(-1)).to(device)

        dt_all = batch["dt_prev"]
        dt_valid = torch.isfinite(dt_all)
        dt_safe = torch.where(dt_valid, dt_all, torch.zeros_like(dt_all))
        new_session = batch["new_session"].bool().reshape(b, length)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.amp and device.type == "cuda"):
            event_emb, _tokens = model.encoder(batch)
        event_emb = event_emb.float().reshape(b, length, -1)
        if fast is None or offset == 0:
            fast, slow = model.state.initial(b, device)
        taus = model.state.taus()
        if model.background is not None:
            bg_age = batch["background_age"]
            bg_valid = torch.isfinite(bg_age)
            bg_fast, bg_slow = model.background.encode(
                batch["background"],
                torch.where(bg_valid, bg_age, torch.zeros_like(bg_age)),
                bg_valid,
            )
            bg_fast = bg_fast.reshape(b, length, -1)
            bg_slow = bg_slow.reshape(b, length, -1)

        dt_step = dt_safe.reshape(b, length)
        timing_list, content_list = [], []
        init_f, init_s = model.state.initial(b, device)
        for step in range(length):
            reset = new_session[:, step].unsqueeze(-1)
            fast = torch.where(reset, init_f, fast)
            slow = torch.where(reset, init_s, slow)
            timing_list.append(torch.cat([fast, slow], dim=-1))
            fast_e, slow_e = model.state.evolve(fast, slow, dt_step[:, step], taus)
            if model.background is not None:
                fast_e = fast_e + bg_fast[:, step]
                slow_e = slow_e + bg_slow[:, step]
            content_list.append(torch.cat([fast_e, slow_e], dim=-1))
            prev_slow = slow_e
            fast, slow = model.state.update(fast_e, slow_e, event_emb[:, step])
            extra["slow_delta"].append(float((slow - prev_slow).detach().abs().mean()))
        timing_states = torch.stack(timing_list, dim=1).reshape(b * length, -1)
        content_states = torch.stack(content_list, dim=1).reshape(b * length, -1)
        extra["state_norm"].append(float(content_states.detach().norm(dim=-1).mean()))

        pred = model.heads(content_states)
        timing_pred = model.heads(timing_states)
        losses = _endpoint_losses(pred, timing_pred, batch, dt_safe, dt_valid, slot_valid)
        accum.add(losses)

        chunk_loss = torch.zeros((), device=device)
        for key, (total, count) in losses.items():
            if float(count) > 0:
                chunk_loss = chunk_loss + weights.get(key, 1.0) * total / count
        if torch.isfinite(chunk_loss):
            optimizer.zero_grad(set_to_none=True)
            chunk_loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            if torch.isfinite(norm):
                optimizer.step()
            if grad_norms is not None:
                grad_norms.append(float(norm))
        else:
            extra["n_nonfinite_steps"] = extra.get("n_nonfinite_steps", 0) + 1
        fast = fast.detach()
        slow = slow.detach()
        extra["n_events"] += int(valid.sum())

    extra["state_norm"] = float(np.mean(extra["state_norm"])) if extra["state_norm"] else float("nan")
    extra["slow_delta"] = float(np.mean(extra["slow_delta"])) if extra["slow_delta"] else float("nan")
    extra["loss_totals"] = accum.totals()
    return accum.means(), extra


def run_sequence(
    model: GroupEventStateModel,
    seq: SubjectSequence,
    lo: int,
    hi: int,
    device: torch.device,
    cfg: TrainConfig,
    *,
    train: bool,
    optimizer: torch.optim.Optimizer | None = None,
    truncate_every: int = 0,
    collect_states: bool = False,
    state_override: np.ndarray | None = None,
    timing_override: np.ndarray | None = None,
    loss_weights: Mapping[str, float] | None = None,
    grad_norms: list[float] | None = None,
    collect_endpoints: bool = False,
    initial_state: tuple[Tensor, Tensor] | None = None,
    initial_since_reset: int = 0,
) -> tuple[dict[str, float], dict[str, Any]]:
    """One causal pass over ``[lo, hi)`` of the patient's interictal stream."""

    accum = _Accum()
    weights = dict(loss_weights or {k: 1.0 for k in ENDPOINTS})
    states: list[np.ndarray] = []
    timing_collect: list[np.ndarray] = []
    derived: list[dict[str, np.ndarray]] = []
    if initial_state is None:
        fast = slow = None
        since_reset = 0
    else:
        fast = initial_state[0].detach().to(device)
        slow = initial_state[1].detach().to(device)
        since_reset = int(initial_since_reset)
    extra: dict[str, Any] = {"n_events": 0, "state_norm": [], "slow_delta": []}

    for chunk_lo, chunk_hi, starts_session in seq.chunks(lo, hi, cfg.chunk_events):
        raw = seq.gather(chunk_lo, chunk_hi)
        batch = _to_device(raw, device)
        n = chunk_hi - chunk_lo
        dt_all = batch["dt_prev"]
        dt_valid = torch.isfinite(dt_all)
        dt_safe = torch.where(dt_valid, dt_all, torch.zeros_like(dt_all))

        if model.baseline_only:
            timing_states = model.history(batch["history"])
            content_states = torch.cat(
                [timing_states, torch.log1p(dt_safe).unsqueeze(-1)], dim=-1
            ) @ model.content_mix
        else:
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=cfg.amp and device.type == "cuda"
            ):
                event_emb, _tokens = model.encoder(batch)
            event_emb = event_emb.float()
            if starts_session or fast is None:
                fast, slow = model.state.initial(1, device)
                since_reset = 0
            taus = model.state.taus()
            if model.background is not None:
                bg_age = batch["background_age"]
                bg_valid = torch.isfinite(bg_age)
                bg_fast, bg_slow = model.background.encode(
                    batch["background"],
                    torch.where(bg_valid, bg_age, torch.zeros_like(bg_age)),
                    bg_valid,
                )
            timing_list: list[Tensor] = []
            content_list: list[Tensor] = []
            for step in range(n):
                if truncate_every and since_reset >= truncate_every:
                    fast, slow = model.state.initial(1, device)
                    since_reset = 0
                # Timing is read off the state BEFORE it relaxes over dt: evolving
                # first would leak the interval into its own prediction through the
                # amount of decay applied.
                timing_list.append(torch.cat([fast, slow], dim=-1))
                fast_e, slow_e = model.state.evolve(
                    fast, slow, dt_safe[step : step + 1], taus
                )
                if model.background is not None:
                    fast_e = fast_e + bg_fast[step : step + 1]
                    slow_e = slow_e + bg_slow[step : step + 1]
                content_list.append(torch.cat([fast_e, slow_e], dim=-1))
                prev_slow = slow_e
                fast, slow = model.state.update(fast_e, slow_e, event_emb[step : step + 1])
                since_reset += 1
                extra["slow_delta"].append(float((slow - prev_slow).detach().abs().mean()))
            timing_states = torch.cat(timing_list, dim=0)
            content_states = torch.cat(content_list, dim=0)
            extra["state_norm"].append(float(content_states.detach().norm(dim=-1).mean()))

        if state_override is not None and not train:
            # Both the content state AND the pre-evolution timing state must be
            # scrambled.  Overriding only the content state left the timing head
            # reading its true state, so the timing endpoint came out numerically
            # identical and the control silently said nothing about it.
            sl = slice(chunk_lo - lo, chunk_hi - lo)
            content_states = torch.from_numpy(state_override[sl]).to(device).float()
            if timing_override is not None:
                timing_states = torch.from_numpy(timing_override[sl]).to(device).float()
        if collect_states:
            states.append(content_states.detach().float().cpu().numpy())
            timing_collect.append(timing_states.detach().float().cpu().numpy())

        pred = model.heads(content_states)
        timing_pred = model.heads(timing_states)
        losses = _endpoint_losses(pred, timing_pred, batch, dt_safe, dt_valid)
        accum.add(losses)
        if collect_endpoints:
            derived.append(endpoint_predictions(pred, batch))

        if train and optimizer is not None:
            chunk_loss = torch.zeros((), device=device)
            for key, (total, count) in losses.items():
                if float(count) > 0:
                    chunk_loss = chunk_loss + weights.get(key, 1.0) * total / count
            if torch.isfinite(chunk_loss):
                optimizer.zero_grad(set_to_none=True)
                chunk_loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                if torch.isfinite(norm):
                    optimizer.step()
                if grad_norms is not None:
                    grad_norms.append(float(norm))
            else:
                extra["n_nonfinite_steps"] = extra.get("n_nonfinite_steps", 0) + 1
        if not model.baseline_only and fast is not None:
            fast = fast.detach()
            slow = slow.detach()
        extra["n_events"] += n

    if collect_states:
        extra["states"] = (
            np.concatenate(states, axis=0) if states else np.zeros((0, 1), np.float32)
        )
        extra["timing_states"] = (
            np.concatenate(timing_collect, axis=0)
            if timing_collect
            else np.zeros((0, 1), np.float32)
        )
    if collect_endpoints and derived:
        extra["order_spearman"] = np.concatenate([d["order_spearman"] for d in derived])
        extra["tied_group_agreement"] = np.concatenate(
            [d["tied_group_agreement"] for d in derived]
        )
        for key in ("prefix_continuation_spearman", "prefix_next_contact_hit"):
            extra[key] = np.concatenate([d[key] for d in derived])
        extra["participation_prob"] = np.concatenate([d["participation_prob"] for d in derived])
        extra["participation_true"] = np.concatenate([d["participation_true"] for d in derived])
        for key in ("size_pred", "size_true", "timing_mu", "delay_span_true"):
            extra[key] = np.concatenate([d[key] for d in derived])
    extra["state_norm"] = float(np.mean(extra["state_norm"])) if extra["state_norm"] else float("nan")
    extra["slow_delta"] = float(np.mean(extra["slow_delta"])) if extra["slow_delta"] else float("nan")
    extra["loss_totals"] = accum.totals()
    if not model.baseline_only and fast is not None and slow is not None:
        extra["final_state"] = (fast.detach(), slow.detach())
        extra["final_since_reset"] = int(since_reset)
    return accum.means(), extra


def _session_start_before(seq: SubjectSequence, position: int) -> int:
    """Start of the recorded session containing ``position``.

    A split may cut through a session.  Replaying from a fixed number of events
    before that split silently changes the physical warm-up across patients and
    cannot support a full-session state claim.
    """

    position = int(position)
    if position <= 0:
        return 0
    starts = np.flatnonzero(seq.new_session[: position + 1])
    return int(starts[-1]) if starts.size else 0


def _causal_state_before(
    model: GroupEventStateModel,
    seq: SubjectSequence,
    position: int,
    device: torch.device,
    cfg: TrainConfig,
    *,
    truncate_every: int = 0,
) -> tuple[tuple[Tensor, Tensor] | None, int, int]:
    """Replay the observed session up to ``position`` and return its carry.

    The old implementation called a warm pass and then discarded its terminal
    state because ``run_sequence`` initialised local state on every invocation.
    This helper makes the hand-off explicit and reports the actual replay count.
    """

    if model.baseline_only:
        return None, 0, 0
    warm_lo = _session_start_before(seq, position)
    if warm_lo >= position:
        return None, 0, 0
    _means, extra = run_sequence(
        model,
        seq,
        warm_lo,
        position,
        device,
        cfg,
        train=False,
        truncate_every=truncate_every,
    )
    return (
        extra.get("final_state"),
        int(extra.get("final_since_reset", 0)),
        int(position - warm_lo),
    )


def _load_geometry(seq: SubjectSequence) -> Tensor | None:
    """Static contact geometry: coordinates when available, shaft layout always."""

    contacts = seq.index["contacts"]
    coords = np.zeros((len(contacts), 3), dtype=np.float32)
    have_coords = False
    coord_file = Path(seq.root) / "coords.npy"
    if coord_file.exists():
        loaded = np.load(coord_file)
        if loaded.shape == coords.shape and np.isfinite(loaded).all():
            coords = loaded.astype(np.float32)
            have_coords = True
    if have_coords:
        coords = coords - coords.mean(0, keepdims=True)
        scale = float(np.linalg.norm(coords, axis=1).max()) or 1.0
        coords = coords / scale
    shafts = sorted({c["shaft"] for c in contacts})
    shaft_idx = np.array([shafts.index(c["shaft"]) for c in contacts], dtype=np.float32)
    number = np.array([float(c["number"]) for c in contacts], dtype=np.float32)
    extra = np.stack(
        [
            shaft_idx / max(len(shafts) - 1, 1),
            number / max(number.max(), 1.0),
            np.full(len(contacts), 1.0 if have_coords else 0.0, dtype=np.float32),
        ],
        axis=1,
    )
    return torch.from_numpy(np.concatenate([coords, extra], axis=1))


def _data_shape(seq: SubjectSequence) -> DataShape:
    idx = seq.index
    return DataShape(
        n_contacts=int(idx["n_contacts"]),
        n_bands=len(idx["bands"]),
        n_band_features=len(idx["band_feature_names"]),
        n_cross_band_pairs=len(idx["cross_band_pairs"]),
        n_views=len(idx["views"]),
        n_waveform_samples=int(idx["n_context_samples"]),
        n_envelope_bins=int(idx["envelope_bins"]),
        n_background_features=len(idx["background_feature_names"]),
        band_available=tuple(bool(b) for b in idx["band_available"]),
    )


def _auto_chunk(seq: SubjectSequence, requested: int) -> int:
    """Keep contacts x samples x chunk bounded so wide patients still fit."""

    load = int(seq.index["n_contacts"]) * int(seq.index["n_context_samples"]) * max(
        len(seq.index["views"]), 1
    )
    budget = 24_000_000
    return int(max(16, min(requested, budget // max(load, 1))))


def _save_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float))
    os.replace(tmp, path)


def _param_snapshot(model: nn.Module) -> dict[str, Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items() if v.is_floating_point()}


def _update_magnitude(before: Mapping[str, Tensor], model: nn.Module) -> dict[str, float]:
    after = model.state_dict()
    out: dict[str, float] = {}
    for group in ("encoder", "state", "heads", "background", "history"):
        num = den = 0.0
        for key, value in before.items():
            if not key.startswith(group):
                continue
            delta = (after[key].detach().float() - value.float()).norm().item()
            num += delta**2
            den += value.float().norm().item() ** 2
        if den > 0:
            out[group] = math.sqrt(num) / math.sqrt(den)
    return out


def train_one(
    seq: SubjectSequence,
    arm: ArmSpec,
    seed: int,
    cfg: TrainConfig,
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    if cfg.warm_events or cfg.eval_warm_events:
        raise ValueError(
            "group-event state v0.2 requires exact replay from the observed "
            "session start; warm_events/eval_warm_events caps must both be 0"
        )
    torch.manual_seed(seed)
    np.random.seed(seed)
    generator = torch.Generator().manual_seed(seed)
    cfg = replace(cfg, chunk_events=_auto_chunk(seq, cfg.chunk_events))

    shape = _data_shape(seq)
    geometry = _load_geometry(seq) if arm.encoder.use_geometry else None
    train_lo, train_hi = seq.split_slice("train")
    input_stats, target_stats = estimate_stats(seq, train_lo, train_hi, seed=seed)
    model = GroupEventStateModel(
        arm, shape, geometry.to(device) if geometry is not None else None,
        seq.history.shape[1], generator, input_stats, target_stats,
    ).to(device)
    init_state = _param_snapshot(model)

    groups = []
    if model.encoder is not None:
        groups.append({"params": model.encoder.parameters(), "lr": cfg.lr_encoder})
    if model.state is not None:
        groups.append({"params": model.state.parameters(), "lr": cfg.lr_state})
    if model.background is not None:
        groups.append({"params": model.background.parameters(), "lr": cfg.lr_state})
    if model.history is not None:
        groups.append({"params": model.history.parameters(), "lr": cfg.lr_heads})
    groups.append({"params": model.heads.parameters(), "lr": cfg.lr_heads})
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)

    val_lo, val_hi = seq.split_slice("val")
    test_lo, test_hi = seq.split_slice("test")

    history: list[dict[str, Any]] = []
    best = {"val_total": float("inf"), "epoch": -1}
    best_state: dict[str, Tensor] | None = None
    grad_norms: list[float] = []
    started = time.time()
    stop_reason = "max_epochs"

    for epoch in range(cfg.max_epochs):
        model.train()
        epoch_grads: list[float] = []
        train_means, train_extra = run_streams(
            model, seq, train_lo, train_hi, device, cfg, optimizer,
            grad_norms=epoch_grads,
        )
        model.eval()
        with torch.no_grad():
            val_state, val_since_reset, _n_val_warm = _causal_state_before(
                model, seq, val_lo, device, cfg
            )
            val_means, val_extra = run_sequence(
                model,
                seq,
                val_lo,
                val_hi,
                device,
                cfg,
                train=False,
                initial_state=val_state,
                initial_since_reset=val_since_reset,
            )
        val_total = float(np.nansum([val_means[k] for k in ENDPOINTS if k != "group_size"]))
        grad_norms.extend(epoch_grads)
        history.append(
            {
                "epoch": epoch,
                "train": train_means,
                "val": val_means,
                "val_total": val_total,
                "grad_norm_mean": float(np.mean(epoch_grads)) if epoch_grads else float("nan"),
                "state_norm": train_extra["state_norm"],
            "n_nonfinite_steps": train_extra.get("n_nonfinite_steps", 0),
                "slow_delta": train_extra["slow_delta"],
                "seconds": round(time.time() - started, 1),
            }
        )
        if val_total < best["val_total"] - 1e-6:
            best = {"val_total": val_total, "epoch": epoch}
            best_state = _param_snapshot(model)
        elif epoch - best["epoch"] >= cfg.patience and epoch + 1 >= cfg.min_epochs:
            stop_reason = "early_stopping"
            break
        if time.time() - started > cfg.max_train_seconds and epoch + 1 >= cfg.min_epochs:
            stop_reason = "time_budget"
            break

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    model.eval()

    # The frozen artefact H2b consumes.  Saved for every run so that "the
    # checkpoint actually moved away from initialisation" is checkable after the
    # fact rather than asserted.
    ckpt_tmp = out_dir / "checkpoint.pt.tmp"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "arm": arm.name,
            "seed": seed,
            "subject": seq.index["subject"],
            "selected_epoch": best["epoch"],
            "encoder_config": asdict(arm.encoder),
            "state_config": asdict(arm.state),
            "chunk_events": cfg.chunk_events,
        },
        ckpt_tmp,
    )
    os.replace(ckpt_tmp, out_dir / "checkpoint.pt")

    results: dict[str, Any] = {
        "subject": seq.index["subject"],
        "dataset": seq.index["dataset"],
        "arm": arm.name,
        "arm_notes": arm.notes,
        "seed": seed,
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "n_events_total": len(seq),
        "n_events_train": train_hi - train_lo,
        "n_events_val": val_hi - val_lo,
        "n_events_test": test_hi - test_lo,
        "chunk_events": cfg.chunk_events,
        "selected_epoch": best["epoch"],
        "n_epochs_run": len(history),
        "stop_reason": stop_reason,
        "train_seconds": round(time.time() - started, 1),
        "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else float("nan"),
        "param_update_magnitude": _update_magnitude(init_state, model),
        "history": history,
        "config": {"train": asdict(cfg), "encoder": asdict(arm.encoder), "state": asdict(arm.state)},
    }
    if model.state is not None:
        tau_f, tau_s = model.state.taus()
        results["tau_fast_seconds"] = [float(tau_f.min()), float(tau_f.median()), float(tau_f.max())]
        results["tau_slow_seconds"] = [float(tau_s.min()), float(tau_s.median()), float(tau_s.max())]

    with torch.no_grad():
        test_state, test_since_reset, n_eval_warm = _causal_state_before(
            model, seq, test_lo, device, cfg
        )
        test_means, test_extra = run_sequence(
            model, seq, test_lo, test_hi, device, cfg, train=False,
            collect_states=True, collect_endpoints=True,
            initial_state=test_state, initial_since_reset=test_since_reset,
        )
    results["eval_warm_events"] = int(n_eval_warm)
    results["eval_warm_source"] = "recorded_session_start"
    results["test"] = test_means
    results["test_state_norm"] = test_extra["state_norm"]

    for key in ("prefix_continuation_spearman", "prefix_next_contact_hit"):
        values = test_extra.get(key)
        if values is not None and values.size:
            results[key] = {
                "median": float(np.median(values)),
                "mean": float(np.mean(values)),
                "n_events": int(values.size),
            }
    rho = test_extra.get("order_spearman")
    tie = test_extra.get("tied_group_agreement")
    if rho is not None and rho.size:
        results["recruitment_order_spearman"] = {
            "median": float(np.median(rho)), "mean": float(np.mean(rho)), "n_events": int(rho.size),
        }
    if tie is not None and tie.size:
        results["tied_group_agreement"] = {
            "median": float(np.median(tie)), "mean": float(np.mean(tie)), "n_events": int(tie.size),
        }
    prob = test_extra.get("participation_prob")
    truth = test_extra.get("participation_true")
    if prob is not None and truth is not None and truth.size:
        pos = prob[truth]
        neg = prob[~truth]
        if pos.size and neg.size:
            sample = min(pos.size, neg.size, 20000)
            rng = np.random.default_rng(seed)
            auc = float(
                (rng.choice(pos, sample) > rng.choice(neg, sample)).mean()
                + 0.5 * (rng.choice(pos, sample) == rng.choice(neg, sample)).mean()
            )
            results["participation_auc_sampled"] = auc
    if "size_pred" in test_extra:
        raw = seq.gather(test_lo, test_hi)
        with (out_dir / "test_series.npz.tmp").open("wb") as handle:
            np.savez(
                handle,
                t_abs=raw["t_abs"],
                dt_prev=raw["dt_prev"],
                size_pred=test_extra["size_pred"],
                size_true=test_extra["size_true"],
                timing_mu=test_extra["timing_mu"],
                delay_span_true=test_extra["delay_span_true"],
            )
        os.replace(out_dir / "test_series.npz.tmp", out_dir / "test_series.npz")

    # --- H1 probe 1: how much history does the trained model actually use? ---
    truncation: dict[str, dict[str, float]] = {}
    for k in TRUNCATION_PROBES:
        if model.baseline_only:
            break
        if k == 0:
            # truncate_every=0 disables truncation, which is bit-for-bit the main
            # test pass that has already run.  Recomputing it costs a full warm-up
            # plus a full test pass for a number we already hold.
            truncation["full_session"] = test_means
            continue
        with torch.no_grad():
            trunc_state, trunc_since_reset, _n_trunc_warm = _causal_state_before(
                model, seq, test_lo, device, cfg, truncate_every=k
            )
            means, _ = run_sequence(
                model,
                seq,
                test_lo,
                test_hi,
                device,
                cfg,
                train=False,
                truncate_every=k,
                initial_state=trunc_state,
                initial_since_reset=trunc_since_reset,
            )
        truncation[f"reset_every_{k}"] = means
    results["history_truncation"] = truncation

    # --- H1 probe 2: correct-time state vs a matched wrong-time state ---
    states = test_extra.get("states")
    if states is not None and states.shape[0] == (test_hi - test_lo) and states.shape[0] > 8:
        rng = np.random.default_rng(seed + 977)
        order = rng.permutation(states.shape[0])
        permuted = states[order]
        timing_states = test_extra.get("timing_states")
        permuted_timing = (
            timing_states[order]
            if timing_states is not None and timing_states.shape[0] == states.shape[0]
            else None
        )
        with torch.no_grad():
            if permuted_timing is None:
                # Only the timing state would still come from the live recurrence,
                # so only then does warming it up change anything.
                run_sequence(model, seq, eval_warm_lo, val_hi, device, cfg, train=False)
            wrong, _ = run_sequence(
                model, seq, test_lo, test_hi, device, cfg, train=False,
                state_override=permuted, timing_override=permuted_timing,
            )
        results["wrong_time_state"] = wrong
        np.save(out_dir / "test_states.npy", states.astype(np.float32))
    return results
