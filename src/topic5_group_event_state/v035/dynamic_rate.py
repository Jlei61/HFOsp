"""Causal dynamic event-load baseline on fixed physical-time anchors.

The static intercept, deterministic multiscale history and learned residual
are strictly nested.  All count targets are built inside real coverage pieces;
unrecorded gaps are never interpreted as silence.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from .contracts import (
    DATASET_ROOT, FORMAT_PREFIX, INPUT_ROOT, RATE_TAUS_SECONDS, RateTrainConfig,
    atomic_json, seed_all,
)
from .long_windows import (
    exposure_and_gap_count,
    matched_wrong_time_donors,
    merge_artificial_cuts,
    phase_for_times,
    plan_horizon_specific_split,
)


PHASES = ("CALIBRATION", "FIT", "INNER", "SELECTION")

# Fixed, pre-declared scale for the causal session-position feature: the longest
# time constant already registered in the q(t) bank.  Never the segment length.
SESSION_POSITION_SCALE_SECONDS = float(RATE_TAUS_SECONDS[-1])


@dataclass(frozen=True)
class RateData:
    subject: str
    anchor_time: np.ndarray
    segment: np.ndarray
    phase: np.ndarray
    q_raw: np.ndarray
    q_names: tuple[str, ...]
    target_count: np.ndarray
    target_valid: np.ndarray
    target_exposure_seconds: np.ndarray
    target_gap_count: np.ndarray
    target_seizure_count: np.ndarray
    horizons_seconds: tuple[float, ...]
    segment_bounds: np.ndarray
    observed_support_bounds: np.ndarray
    phase_boundaries: Mapping[str, float]
    provenance: Mapping[str, Any]


def _phase(t: float, bounds: Mapping[str, float]) -> str | None:
    if t < bounds["20pct"]:
        return "CALIBRATION"
    if t < bounds["60pct"]:
        return "FIT"
    if t < bounds["70pct"]:
        return "INNER"
    if t < bounds["80pct"]:
        return "SELECTION"
    return None


def _event_mark_matrix(feature: np.ndarray, names: tuple[str, ...]) -> tuple[np.ndarray, tuple[str, ...]]:
    wanted = (
        "n_participating", "extent_fraction", "n_tied_groups",
        "first_tied_group_fraction", "delay_span_seconds",
        "ied_low_energy_mean", "gamma_energy_mean", "low_ripple_energy_mean",
        "ripple_energy_mean", "fast_ripple_energy_mean",
    )
    columns, used = [], []
    lookup = {name: i for i, name in enumerate(names)}
    for name in wanted:
        if name in lookup:
            columns.append(feature[:, lookup[name]])
            used.append(name)
    if not columns:
        raise ValueError("no registered burden/mark summary exists in event_features_r0")
    x = np.stack(columns, axis=1).astype(np.float64)
    # Missing band summaries do not create future information.  They fall back
    # to the training-prefix median later, before causal accumulation.
    return x, tuple(used)


def _causal_features(
    anchor_time: np.ndarray,
    anchor_segment: np.ndarray,
    event_time: np.ndarray,
    event_segment: np.ndarray,
    event_mark: np.ndarray,
    segment_bounds: np.ndarray,
    taus: tuple[float, ...],
    train_event: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    train_values = np.where(train_event[:, None], event_mark, np.nan)
    fill = np.nanmedian(train_values, axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    mark = np.where(np.isfinite(event_mark), event_mark, fill[None])
    out = np.zeros((anchor_time.size, len(taus) * (1 + mark.shape[1]) + 4), dtype=np.float64)
    for seg in np.unique(anchor_segment):
        ai = np.flatnonzero(anchor_segment == seg)
        ei = np.flatnonzero(event_segment == seg)
        ai = ai[np.argsort(anchor_time[ai], kind="stable")]
        ei = ei[np.argsort(event_time[ei], kind="stable")]
        acc_n = np.zeros(len(taus), dtype=np.float64)
        acc_m = np.zeros((len(taus), mark.shape[1]), dtype=np.float64)
        cursor = 0
        last_t = float(segment_bounds[int(seg), 0])
        last_event = -np.inf
        for row in ai:
            target_t = float(anchor_time[row])
            while cursor < ei.size and float(event_time[ei[cursor]]) < target_t:
                te = float(event_time[ei[cursor]])
                decay = np.exp(-(te - last_t) / np.asarray(taus))
                acc_n *= decay
                acc_m *= decay[:, None]
                acc_n += 1.0
                acc_m += mark[ei[cursor]][None]
                last_t = te
                last_event = te
                cursor += 1
            decay = np.exp(-(target_t - last_t) / np.asarray(taus))
            n = acc_n * decay
            m = acc_m * decay[:, None]
            rate_per_minute = 60.0 * n / np.asarray(taus)
            means = m / np.maximum(n[:, None], 1e-8)
            body = np.concatenate([np.log1p(rate_per_minute)[:, None], means], axis=1).reshape(-1)
            lo = float(segment_bounds[int(seg), 0])
            since = target_t - last_event if np.isfinite(last_event) else 7 * 86400.0
            clock = 2 * np.pi * (target_t % 86400.0) / 86400.0
            # Session position must be causal, and it must be expressed on a
            # scale that is known in advance.  The former (t - lo) / (hi - lo)
            # divided by the segment END: target segments end at seizure onsets
            # for most patients, so that fraction was a countdown to the next
            # seizure (review 2026-09-04).  Elapsed time is therefore put on the
            # longest already-registered bank constant (8 h) and capped, which
            # keeps the feature bounded like the original without importing any
            # future quantity.  The scale is fixed a priori, not tuned.
            tail = np.asarray([
                np.log1p(max(0.0, since)), np.sin(clock), np.cos(clock),
                min(max(0.0, target_t - lo), SESSION_POSITION_SCALE_SECONDS) / SESSION_POSITION_SCALE_SECONDS,
            ])
            out[row] = np.concatenate([body, tail])
    names = []
    # mark feature names are supplied by provenance; generic indices keep this
    # helper independent of the selected feature subset.
    for tau in taus:
        names.append(f"log_rate_per_min_tau{int(tau)}")
        names.extend(f"mark_{j}_tau{int(tau)}" for j in range(event_mark.shape[1]))
    names.extend(("log_time_since_last_event", "clock_sin", "clock_cos", "segment_elapsed_over_8h"))
    if not np.isfinite(out).all():
        raise ValueError("causal q features contain non-finite values")
    return out, tuple(names)


def load_rate_data(subject: str, config: RateTrainConfig) -> RateData:
    config.validate()
    manifest_path = INPUT_ROOT / subject / "manifest_v3.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sealed") is not False or manifest.get("development_evaluation_used_for_fitting") is not False:
        raise PermissionError("v0.3.5 rate input is not a safe development-prefix manifest")
    input_path = Path(manifest["input_path"])
    with np.load(input_path, allow_pickle=False) as z:
        meta = json.loads(str(np.asarray(z["metadata_json"]).item()))
        event_time_all = np.asarray(z["event_time"], dtype=np.float64)
        # ``event_carry`` is intentionally coarser than the target coverage
        # pieces and may bridge a seizure exclusion or ordinary recording gap.
        # Targets and silence evidence therefore use a segment re-derived from
        # ``target_segment_bounds`` below.
        event_carry_all = np.asarray(z["event_carry"], dtype=np.int64)
        feature_all = np.asarray(z["event_features_r0"], dtype=np.float64)
        feature_valid_all = np.asarray(z["event_feature_valid"], dtype=bool)
        train_event_all = np.asarray(z["train_event_mask"], dtype=bool)
        observed_support_bounds = np.asarray(z["target_segment_bounds"], dtype=np.float64)
    legacy_bounds = {k: float(v) for k, v in manifest["report"]["phase_boundaries_epoch"].items()}
    index = json.loads((DATASET_ROOT / subject / "index.json").read_text(encoding="utf-8"))
    segment_bounds = observed_support_bounds.copy()
    merge_audit = None
    if config.window_contract == "observed_support":
        segment_bounds, merge_audit = merge_artificial_cuts(
            observed_support_bounds, index.get("seizures", ()),
            max_gap_seconds=float(config.merge_artificial_cuts_seconds),
        )
    bounds = dict(legacy_bounds)
    split_plan = None
    if config.split_contract in {
        "horizon_specific_observed_time",
        "shared_multi_horizon_observed_time",
    }:
        # One producer must have one causal training boundary.  In the shared
        # contract the largest horizon sizes the common FIT/INNER/SELECTION
        # blocks.  Using the first/smallest horizon here would allow its later
        # FIT rows to overlap the final holdout used by a longer head.
        split_horizon = (
            max(config.horizons_seconds)
            if config.split_contract == "shared_multi_horizon_observed_time"
            else float(config.horizons_seconds[0])
        )
        split_plan = plan_horizon_specific_split(
            observed_support_bounds, legacy_bounds, float(split_horizon),
            inner_horizons=float(config.inner_holdout_horizons),
            selection_horizons=float(config.selection_holdout_horizons),
            minimum_fit_horizons=float(config.minimum_fit_horizons),
        )
        if split_plan.status != "ESTIMABLE" or split_plan.boundaries is None:
            raise ValueError(f"{subject}: long split is not estimable: {split_plan.reason}")
        bounds = dict(split_plan.boundaries)
    keep = event_time_all < bounds["80pct"]
    event_time = event_time_all[keep]
    event_carry = event_carry_all[keep]
    features = feature_all[keep]
    valid = feature_valid_all[keep]
    train_event = train_event_all[keep]
    event_segment = np.full(event_time.size, -1, dtype=np.int64)
    anchor_support = (
        observed_support_bounds
        if config.window_contract == "observed_support" else segment_bounds
    )
    for lo, hi in anchor_support:
        carry_match = np.flatnonzero(
            (segment_bounds[:, 0] <= float(lo) + 1e-9)
            & (segment_bounds[:, 1] >= float(hi) - 1e-9)
        )
        if carry_match.size != 1:
            raise ValueError("observed support does not map to exactly one state-carry segment")
        seg = int(carry_match[0])
        inside = (event_time >= float(lo)) & (event_time < float(hi))
        if np.any(event_segment[inside] >= 0):
            raise ValueError("target coverage segments overlap")
        event_segment[inside] = int(seg)
    if np.any(event_segment < 0):
        # Events outside eligible target coverage may exist in the source
        # prefix.  They cannot provide evidence across a gap/seizure boundary.
        keep_covered = event_segment >= 0
        event_time = event_time[keep_covered]
        event_carry = event_carry[keep_covered]
        event_segment = event_segment[keep_covered]
        features = features[keep_covered]
        valid = valid[keep_covered]
        train_event = train_event[keep_covered]
    features = np.where(valid, features, np.nan)
    names = tuple(str(v) for v in meta["event_feature_names_r0"])
    mark, mark_names = _event_mark_matrix(features, names)

    anchors: list[float] = []
    segments: list[int] = []
    phases: list[str] = []
    grid = float(config.grid_seconds)
    # Anchors require a real observation at t.  Iterate the original coverage
    # pieces, then map each piece to the coarser state-carry segment.  Iterating
    # ``segment_bounds`` here would silently create anchors inside a bridged
    # recording gap.
    for lo, hi in anchor_support:
        carry_match = np.flatnonzero(
            (segment_bounds[:, 0] <= float(lo) + 1e-9)
            & (segment_bounds[:, 1] >= float(hi) - 1e-9)
        )
        if carry_match.size != 1:
            raise ValueError("anchor support does not map to exactly one state-carry segment")
        seg = int(carry_match[0])
        # The trajectory exists from the first legal grid point.  Burn-in
        # controls *loss eligibility* below; deleting those states broke the
        # frozen contact decoder's otherwise legal early anchors.
        first = math.ceil(float(lo) / grid) * grid
        for t in np.arange(first, min(float(hi), bounds["80pct"]), grid):
            ph = _phase(float(t), bounds)
            if ph is not None:
                anchors.append(float(t)); segments.append(seg); phases.append(ph)
    anchor_time = np.asarray(anchors, dtype=np.float64)
    anchor_segment = np.asarray(segments, dtype=np.int64)
    phase = np.asarray(phases)
    q_raw, q_names_generic = _causal_features(
        anchor_time, anchor_segment, event_time, event_segment, mark,
        segment_bounds, tuple(config.taus_seconds), train_event,
    )
    q_names = []
    width = 1 + len(mark_names)
    for tau in config.taus_seconds:
        q_names.append(f"log_rate_per_min_tau{int(tau)}")
        q_names.extend(f"{name}_tau{int(tau)}" for name in mark_names)
    q_names.extend(q_names_generic[-4:])
    target = np.zeros((anchor_time.size, len(config.horizons_seconds)), dtype=np.float32)
    eligible = np.zeros_like(target, dtype=bool)
    exposure = np.zeros_like(target, dtype=np.float64)
    gap_count = np.zeros_like(target, dtype=np.int16)
    seizure_count = np.zeros_like(target, dtype=np.int16)
    seizure_onsets = np.sort(np.asarray([
        float(row["onset_epoch"])
        for row in index.get("seizures", ())
        if float(row["onset_epoch"]) < float(bounds["80pct"])
    ], dtype=np.float64))
    phase_hi_lookup = {
        "CALIBRATION": bounds["20pct"], "FIT": bounds["60pct"],
        "INNER": bounds["70pct"], "SELECTION": bounds["80pct"],
    }
    if config.window_contract == "observed_support":
        sorted_events = np.sort(event_time)
        for j, horizon in enumerate(config.horizons_seconds):
            stops = anchor_time + float(horizon)
            exposure[:, j], gap_count[:, j] = exposure_and_gap_count(
                observed_support_bounds, anchor_time, stops,
            )
            if seizure_onsets.size:
                seizure_count[:, j] = (
                    np.searchsorted(seizure_onsets, stops, side="left")
                    - np.searchsorted(seizure_onsets, anchor_time, side="left")
                ).astype(np.int16)
            for row, (t, stop) in enumerate(zip(anchor_time, stops)):
                seg = int(anchor_segment[row])
                phase_hi = phase_hi_lookup[str(phase[row])]
                enough = exposure[row, j] >= float(config.minimum_exposure_fraction) * float(horizon)
                if (t >= float(segment_bounds[seg, 0]) + config.burn_in_seconds
                        and stop <= phase_hi + 1e-9 and enough):
                    left = np.searchsorted(sorted_events, t, side="left")
                    right = np.searchsorted(sorted_events, stop, side="left")
                    target[row, j] = float(right - left)
                    eligible[row, j] = True
    else:
        for seg in np.unique(anchor_segment):
            ai = np.flatnonzero(anchor_segment == seg)
            et = np.sort(event_time[event_segment == seg])
            lo, hi = segment_bounds[int(seg)]
            for row in ai:
                t = float(anchor_time[row])
                ph = str(phase[row])
                phase_hi = phase_hi_lookup[ph]
                for j, horizon in enumerate(config.horizons_seconds):
                    stop = t + float(horizon)
                    if t >= float(lo) + config.burn_in_seconds and stop <= float(hi) and stop <= phase_hi:
                        left = np.searchsorted(et, t, side="left")
                        right = np.searchsorted(et, stop, side="left")
                        target[row, j] = float(right - left)
                        eligible[row, j] = True
                        exposure[row, j] = float(horizon)
    if not np.any((phase == "FIT")[:, None] & eligible):
        raise ValueError(f"{subject}: no fitting targets")
    return RateData(
        subject=subject, anchor_time=anchor_time, segment=anchor_segment, phase=phase,
        q_raw=q_raw.astype(np.float32), q_names=tuple(q_names), target_count=target,
        target_valid=eligible, target_exposure_seconds=exposure.astype(np.float32),
        target_gap_count=gap_count, target_seizure_count=seizure_count,
        horizons_seconds=tuple(config.horizons_seconds),
        segment_bounds=segment_bounds, observed_support_bounds=observed_support_bounds,
        phase_boundaries=bounds,
        provenance={
            "manifest": str(manifest_path), "input": str(input_path),
            "event_input": "complete group-event stream before registered 80pct boundary",
            "anchor_weighting": "one fixed 5min physical-time anchor, not one event row",
            "mark_summaries": list(mark_names), "q_taus_seconds": list(config.taus_seconds),
            "phase_rule": (
                (
                    "one shared observed-exposure split sized by the largest horizon"
                    if config.split_contract == "shared_multi_horizon_observed_time"
                    else "horizon-specific observed-exposure INNER and holdout inside the immutable <80pct prefix"
                )
                if split_plan is not None else
                "CAL<20, FIT=20-60, INNER=60-70, SELECTION=70-80 percent recorded-time boundaries"
            ),
            "phase_boundaries_epoch": bounds,
            "long_split_plan": None if split_plan is None else split_plan.as_dict(),
            "artificial_cut_merge": None if merge_audit is None else merge_audit.__dict__,
            "state_carry_gap_seconds": float(config.merge_artificial_cuts_seconds),
            "exposure_support": "original target coverage intervals before state-carry merging",
            "target_containment": (
                "wall-clock window may cross excluded/unobserved intervals; count likelihood is offset by effective observed seconds"
                if config.window_contract == "observed_support" else
                "same real coverage segment and same chronological phase"
            ),
            "development_targets_read": False, "sealed_partition_opened": False,
            "seizure_outcomes_read": False,
        },
    )


def negative_binomial_nll(count: Tensor, log_mu: Tensor, log_dispersion: Tensor) -> Tensor:
    r = torch.nn.functional.softplus(log_dispersion).clamp_min(1e-4)
    mu = torch.exp(log_mu).clamp(1e-6, 1e8)
    return -(
        torch.lgamma(count + r) - torch.lgamma(r) - torch.lgamma(count + 1.0)
        + r * (torch.log(r) - torch.log(r + mu))
        + count * (torch.log(mu) - torch.log(r + mu))
    )


class DynamicRateModel(nn.Module):
    """Static rate is an exact special case of both dynamic arms."""

    def __init__(self, n_features: int, config: RateTrainConfig) -> None:
        super().__init__()
        h = len(config.horizons_seconds)
        self.register_buffer("log_horizon_minutes", torch.log(torch.tensor(config.horizons_seconds) / 60.0))
        self.static_log_rate = nn.Parameter(torch.zeros(h))
        self.log_dispersion = nn.Parameter(torch.zeros(h))
        self.dynamic = nn.Linear(n_features, h, bias=False)
        layers: list[nn.Module] = []
        width_in = n_features
        for _ in range(config.residual_depth):
            layers += [nn.Linear(width_in, config.residual_width), nn.GELU(), nn.LayerNorm(config.residual_width)]
            width_in = config.residual_width
        layers.append(nn.Linear(width_in, h, bias=False))
        self.residual = nn.Sequential(*layers)
        self.residual_gate_logit = nn.Parameter(torch.tensor(float(config.residual_gate_logit)))
        nn.init.zeros_(self.dynamic.weight)
        nn.init.zeros_(self.residual[-1].weight)

    def forward(
        self, q: Tensor, *, dynamic: bool, residual: bool,
        exposure_seconds: Tensor | None = None,
    ) -> Tensor:
        if exposure_seconds is None:
            log_exposure = self.log_horizon_minutes.expand(q.shape[0], -1)
        else:
            log_exposure = torch.log((exposure_seconds / 60.0).clamp_min(1e-6))
        out = self.static_log_rate + log_exposure
        if dynamic:
            out = out + self.dynamic(q)
        if residual:
            out = out + torch.sigmoid(self.residual_gate_logit) * self.residual(q)
        return out


def _score(model: DynamicRateModel, q: Tensor, y: Tensor, valid: Tensor,
           exposure: Tensor, rows: Tensor, *, dynamic: bool, residual: bool) -> Tensor:
    log_mu = model(q[rows], dynamic=dynamic, residual=residual,
                   exposure_seconds=exposure[rows])
    mask = valid[rows]
    loss = negative_binomial_nll(y[rows], log_mu, model.log_dispersion)
    return (loss * mask).sum() / mask.sum().clamp_min(1)


def _fit_stage(
    model: DynamicRateModel, q: Tensor, y: Tensor, valid: Tensor, exposure: Tensor,
    fit_rows: Tensor, inner_rows: Tensor, *, stage: str, params: list[nn.Parameter],
    lr: float, max_steps: int, config: RateTrainConfig,
) -> dict[str, Any]:
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=config.weight_decay)
    dynamic = stage in {"dynamic", "residual"}
    residual = stage == "residual"
    best = float(_score(model, q, y, valid, exposure, inner_rows, dynamic=dynamic, residual=residual).detach().cpu())
    best_step, stale = 0, 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history = [{"step": 0, "inner_nll": best}]
    edge = False
    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = _score(model, q, y, valid, exposure, fit_rows, dynamic=dynamic, residual=residual)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, config.gradient_clip)
        optimizer.step()
        if step % config.validate_every == 0 or step == max_steps:
            value = float(_score(model, q, y, valid, exposure, inner_rows, dynamic=dynamic, residual=residual).detach().cpu())
            history.append({"step": step, "inner_nll": value, "fit_nll": float(loss.detach().cpu())})
            if np.isfinite(value) and value < best - 1e-6:
                best, best_step, stale = value, step, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if stale >= config.patience_checks:
                break
    model.load_state_dict({k: v.to(q.device) for k, v in best_state.items()})
    edge = best_step == max_steps
    return {"stage": stage, "selected_step": best_step, "steps_run": history[-1]["step"],
            "best_inner_nll": best, "selected_at_init": best_step == 0,
            "selected_at_budget_edge": edge, "history": history}


def _shift_selection(q: np.ndarray, time_: np.ndarray, segment: np.ndarray, rows: np.ndarray, min_shift: float) -> tuple[np.ndarray, np.ndarray]:
    shifted = q.copy()
    valid = np.zeros(time_.size, dtype=bool)
    for seg in np.unique(segment[rows]):
        rr = rows[segment[rows] == seg]
        if rr.size < 3:
            continue
        span = float(time_[rr[-1]] - time_[rr[0]])
        shift = max(int(math.ceil(min_shift / 300.0)), rr.size // 2)
        if span >= 2 * min_shift:
            donor = np.roll(rr, shift % rr.size)
            ok = np.abs(time_[donor] - time_[rr]) >= min_shift
            shifted[rr[ok]] = q[donor[ok]]
            valid[rr[ok]] = True
    return shifted, valid


def run_rate_subject(
    data: RateData,
    config: RateTrainConfig,
    *,
    device: torch.device,
    out_dir: Path,
    overwrite: bool = False,
    report_selection: bool = True,
) -> dict[str, Any]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    card_path = out_dir / "card.json"
    if card_path.exists() and not overwrite:
        return json.loads(card_path.read_text(encoding="utf-8"))
    started = time.time(); seed_all(config.seed)
    fit_np = np.flatnonzero(data.phase == "FIT")
    inner_np = np.flatnonzero(data.phase == "INNER")
    sel_np = np.flatnonzero(data.phase == "SELECTION")
    if min(fit_np.size, inner_np.size, sel_np.size) == 0:
        raise ValueError(f"{data.subject}: missing a chronological phase")
    centre = np.nanmedian(data.q_raw[fit_np], axis=0)
    scale = 1.4826 * np.nanmedian(np.abs(data.q_raw[fit_np] - centre), axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-5), scale, 1.0)
    q_np = np.clip((data.q_raw - centre) / scale, -12, 12).astype(np.float32)
    q = torch.as_tensor(q_np, device=device)
    y = torch.as_tensor(data.target_count, dtype=torch.float32, device=device)
    valid = torch.as_tensor(data.target_valid, dtype=torch.bool, device=device)
    exposure = torch.as_tensor(data.target_exposure_seconds, dtype=torch.float32, device=device)
    fit = torch.as_tensor(fit_np, dtype=torch.long, device=device)
    inner = torch.as_tensor(inner_np, dtype=torch.long, device=device)
    sel = torch.as_tensor(sel_np, dtype=torch.long, device=device)
    model = DynamicRateModel(q.shape[1], config).to(device)
    stages = {}
    stages["static"] = _fit_stage(
        model, q, y, valid, exposure, fit, inner, stage="static",
        params=[model.static_log_rate, model.log_dispersion], lr=config.lr_static,
        max_steps=config.max_steps_static, config=config,
    )
    stages["dynamic"] = _fit_stage(
        model, q, y, valid, exposure, fit, inner, stage="dynamic",
        params=list(model.dynamic.parameters()), lr=config.lr_dynamic,
        max_steps=config.max_steps_dynamic, config=config,
    )
    stages["residual"] = _fit_stage(
        model, q, y, valid, exposure, fit, inner, stage="residual",
        params=[*model.residual.parameters(), model.residual_gate_logit], lr=config.lr_residual,
        max_steps=config.max_steps_residual, config=config,
    )
    shifted_np = q_np.copy()
    shift_valid_np = np.zeros(data.anchor_time.size, dtype=bool)
    matched_donors = np.full((data.anchor_time.size, 5), -1, dtype=np.int64)
    if report_selection:
        if config.window_contract == "observed_support":
            targets = sel_np[data.target_valid[sel_np, 0]]
            donor_pool = np.flatnonzero(np.isin(data.phase, ("FIT", "INNER", "SELECTION")))
            donor = matched_wrong_time_donors(
                data.anchor_time, targets, donor_pool,
                minimum_time_separation=float(data.horizons_seconds[0]),
                recent_rate=q_np[:, 0],
                exposure_fraction=(
                    data.target_exposure_seconds[:, 0] / float(data.horizons_seconds[0])
                ),
                n_donors=5,
            )
            ok = np.all(donor >= 0, axis=1)
            if np.any(ok):
                shifted_np[targets[ok]] = np.median(q_np[donor[ok]], axis=1)
                shift_valid_np[targets[ok]] = True
                matched_donors[targets[ok]] = donor[ok]
        else:
            shifted_np, shift_valid_np = _shift_selection(
                q_np, data.anchor_time, data.segment, sel_np, max(data.horizons_seconds)
            )
    shifted = torch.as_tensor(shifted_np, device=device)
    with torch.no_grad():
        arms = {}
        selection_strata = {}
        if report_selection:
            sel_exposure = exposure[sel]
            log_static = model(q[sel], dynamic=False, residual=False, exposure_seconds=sel_exposure)
            log_dynamic = model(q[sel], dynamic=True, residual=False, exposure_seconds=sel_exposure)
            log_residual = model(q[sel], dynamic=True, residual=True, exposure_seconds=sel_exposure)
            log_shifted = model(shifted[sel], dynamic=True, residual=True, exposure_seconds=sel_exposure)
            yy, vv = y[sel], valid[sel]
            for name, lm in (("static", log_static), ("dynamic", log_dynamic), ("residual", log_residual), ("block_shift", log_shifted)):
                loss = negative_binomial_nll(yy, lm, model.log_dispersion)
                arm_valid = vv if name != "block_shift" else vv & torch.as_tensor(
                    shift_valid_np[sel_np, None], dtype=torch.bool, device=device
                )
                arms[name] = {
                    "n": [int(arm_valid[:, j].sum().cpu()) for j in range(vv.shape[1])],
                    "nll": [float((loss[:, j] * arm_valid[:, j]).sum().cpu() / arm_valid[:, j].sum().clamp_min(1).cpu()) if arm_valid[:, j].any() else None for j in range(vv.shape[1])],
                    "predicted_mean": [float((torch.exp(lm[:, j]) * arm_valid[:, j]).sum().cpu() / arm_valid[:, j].sum().clamp_min(1).cpu()) if arm_valid[:, j].any() else None for j in range(vv.shape[1])],
                    "observed_mean": [float((yy[:, j] * arm_valid[:, j]).sum().cpu() / arm_valid[:, j].sum().clamp_min(1).cpu()) if arm_valid[:, j].any() else None for j in range(vv.shape[1])],
                }
            # The correct-time state must be rescored on exactly the anchors
            # for which a matched wrong-time donor set exists.  Comparing the
            # all-anchor residual NLL with a donor-eligible subset silently
            # changes the estimand when matching is incomplete.
            paired_valid = vv & torch.as_tensor(
                shift_valid_np[sel_np, None], dtype=torch.bool, device=device
            )
            paired_loss = negative_binomial_nll(yy, log_residual, model.log_dispersion)
            arms["residual_paired"] = {
                "n": [int(paired_valid[:, j].sum().cpu()) for j in range(vv.shape[1])],
                "nll": [float((paired_loss[:, j] * paired_valid[:, j]).sum().cpu() / paired_valid[:, j].sum().clamp_min(1).cpu()) if paired_valid[:, j].any() else None for j in range(vv.shape[1])],
                "predicted_mean": [float((torch.exp(log_residual[:, j]) * paired_valid[:, j]).sum().cpu() / paired_valid[:, j].sum().clamp_min(1).cpu()) if paired_valid[:, j].any() else None for j in range(vv.shape[1])],
                "observed_mean": [float((yy[:, j] * paired_valid[:, j]).sum().cpu() / paired_valid[:, j].sum().clamp_min(1).cpu()) if paired_valid[:, j].any() else None for j in range(vv.shape[1])],
            }
            selection_strata["all"] = arms
            if config.window_contract == "observed_support":
                no_seizure = torch.as_tensor(
                    data.target_seizure_count[sel_np] == 0,
                    dtype=torch.bool, device=device,
                )
                no_seizure_arms = {}
                for name, lm in (("static", log_static), ("dynamic", log_dynamic), ("residual", log_residual), ("block_shift", log_shifted)):
                    loss = negative_binomial_nll(yy, lm, model.log_dispersion)
                    arm_valid = vv & no_seizure
                    if name == "block_shift":
                        arm_valid = arm_valid & torch.as_tensor(
                            shift_valid_np[sel_np, None], dtype=torch.bool, device=device
                        )
                    no_seizure_arms[name] = {
                        "n": [int(arm_valid[:, j].sum().cpu()) for j in range(vv.shape[1])],
                        "nll": [float((loss[:, j] * arm_valid[:, j]).sum().cpu() / arm_valid[:, j].sum().clamp_min(1).cpu()) if arm_valid[:, j].any() else None for j in range(vv.shape[1])],
                        "predicted_mean": [float((torch.exp(lm[:, j]) * arm_valid[:, j]).sum().cpu() / arm_valid[:, j].sum().clamp_min(1).cpu()) if arm_valid[:, j].any() else None for j in range(vv.shape[1])],
                        "observed_mean": [float((yy[:, j] * arm_valid[:, j]).sum().cpu() / arm_valid[:, j].sum().clamp_min(1).cpu()) if arm_valid[:, j].any() else None for j in range(vv.shape[1])],
                    }
                paired_no_seizure = vv & no_seizure & torch.as_tensor(
                    shift_valid_np[sel_np, None], dtype=torch.bool, device=device
                )
                paired_loss = negative_binomial_nll(yy, log_residual, model.log_dispersion)
                no_seizure_arms["residual_paired"] = {
                    "n": [int(paired_no_seizure[:, j].sum().cpu()) for j in range(vv.shape[1])],
                    "nll": [float((paired_loss[:, j] * paired_no_seizure[:, j]).sum().cpu() / paired_no_seizure[:, j].sum().clamp_min(1).cpu()) if paired_no_seizure[:, j].any() else None for j in range(vv.shape[1])],
                    "predicted_mean": [float((torch.exp(log_residual[:, j]) * paired_no_seizure[:, j]).sum().cpu() / paired_no_seizure[:, j].sum().clamp_min(1).cpu()) if paired_no_seizure[:, j].any() else None for j in range(vv.shape[1])],
                    "observed_mean": [float((yy[:, j] * paired_no_seizure[:, j]).sum().cpu() / paired_no_seizure[:, j].sum().clamp_min(1).cpu()) if paired_no_seizure[:, j].any() else None for j in range(vv.shape[1])],
                }
                selection_strata["no_seizure_crossing"] = no_seizure_arms
        per_anchor = {
            "anchor_time": data.anchor_time, "segment": data.segment, "phase": data.phase,
            "horizons_seconds": np.asarray(data.horizons_seconds, dtype=np.float64),
            "q_standardized": q_np, "target_count": data.target_count,
            "target_valid": data.target_valid.astype(np.uint8),
            "target_exposure_seconds": data.target_exposure_seconds,
            "target_gap_count": data.target_gap_count,
            "target_seizure_count": data.target_seizure_count,
            "block_shift_valid": shift_valid_np.astype(np.uint8),
            "wrong_time_donor_rows": matched_donors,
            "pred_static": torch.exp(model(q, dynamic=False, residual=False, exposure_seconds=exposure)).cpu().numpy(),
            "pred_dynamic": torch.exp(model(q, dynamic=True, residual=False, exposure_seconds=exposure)).cpu().numpy(),
            "pred_residual": torch.exp(model(q, dynamic=True, residual=True, exposure_seconds=exposure)).cpu().numpy(),
            "segment_bounds": data.segment_bounds,
            "observed_support_bounds": data.observed_support_bounds,
            "phase_boundaries_json": np.asarray(json.dumps(dict(data.phase_boundaries), sort_keys=True)),
            "window_contract": np.asarray(config.window_contract),
        }
    np.savez_compressed(out_dir / "trajectory_and_scores.npz", **per_anchor)
    torch.save({"model": model.state_dict(), "q_centre": centre, "q_scale": scale,
                "q_names": data.q_names, "config": config.as_dict()}, out_dir / "checkpoint.pt")
    n_eligible = {
        ph: [int(((data.phase == ph)[:, None] & data.target_valid)[:, j].sum()) for j in range(len(data.horizons_seconds))]
        for ph in PHASES
    }
    independent = {
        ph: [int(sum(math.floor(max(0.0, min(hi, {"CALIBRATION": -np.inf, "FIT": np.inf, "INNER": np.inf, "SELECTION": np.inf}.get(ph, np.inf)) - lo) / h)
                     for lo, hi in data.segment_bounds)) for h in data.horizons_seconds]
        for ph in ()
    }
    card = {
        "format": f"{FORMAT_PREFIX}_dynamic_rate_card_v1", "subject": data.subject,
        "seed": config.seed, "config": config.as_dict(), "q_names": list(data.q_names),
        "n_eligible_anchors": n_eligible, "selection_arms": arms,
        "selection_strata": selection_strata, "stages": stages,
        "selection": {
            "status": "SCORED_AFTER_RECIPE_LOCK" if report_selection else "HELD_UNREAD_DURING_HYPERPARAMETER_SEARCH",
            "targets_read": bool(report_selection),
        },
        "wrong_time_contract": (
            "median of five same-patient donors >= horizon away, within 2 h clock time, nearest in recent rate and observed-exposure fraction"
            if config.window_contract == "observed_support" else
            "within-coverage-piece circular shift"
        ),
        "trajectory_path": str(out_dir / "trajectory_and_scores.npz"),
        "checkpoint_path": str(out_dir / "checkpoint.pt"), "provenance": dict(data.provenance),
        "elapsed_seconds": time.time() - started, "development_targets_read": False,
        "sealed_partition_opened": False, "seizure_outcomes_read": False,
    }
    atomic_json(card_path, card)
    return card
