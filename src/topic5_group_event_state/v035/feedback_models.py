"""Long-scale common-drive versus event-feedback models for v0.3.5 H3.

Each example is one non-overlapping exposure+future block inside a real
coverage segment and one chronological phase.  M0, M1 and M2 share the same
intercept and feature slots.  M1 fills only a burden slot; M2 additionally
fills a mark/waveform slot.  This avoids both the old free-intercept artefact
and the old sliding-window pseudo sample size.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import DATASET_ROOT, FORMAT_PREFIX, INPUT_ROOT, atomic_json
from .dynamic_rate import SESSION_POSITION_SCALE_SECONDS
from .functional_readouts import Endpoint, build_endpoints


PHYSICAL_EXPOSURES = (1800.0, 7200.0, 21600.0)
EVENT_EXPOSURES = (1000, 5000, 10000)
FUTURE_SECONDS = 1800.0
SOURCE_RANK = 4
RIDGES = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
NESTED_MSE_RATIO_LIMIT = 4.0


@dataclass(frozen=True)
class Block:
    phase: str
    segment: int
    exposure_start: float
    boundary: float
    future_stop: float
    exposure_rows: np.ndarray
    future_rows: np.ndarray


def _phase_pieces(segment_bounds: np.ndarray, bounds: dict[str, float]):
    phases = (("FIT", bounds["20pct"], bounds["60pct"]),
              ("INNER", bounds["60pct"], bounds["70pct"]),
              ("SELECTION", bounds["70pct"], bounds["80pct"]))
    for seg, (slo, shi) in enumerate(segment_bounds):
        for phase, plo, phi in phases:
            lo, hi = max(float(slo), plo), min(float(shi), phi)
            if hi > lo:
                yield phase, seg, lo, hi


def _greedy(blocks: list[Block]) -> list[Block]:
    chosen, stop = [], -np.inf
    for block in sorted(blocks, key=lambda b: (b.future_stop, b.exposure_start)):
        if block.exposure_start >= stop:
            chosen.append(block); stop = block.future_stop
    return chosen


def build_blocks(event_time: np.ndarray, event_segment: np.ndarray,
                 segment_bounds: np.ndarray, bounds: dict[str, float],
                 *, kind: str, value: float) -> tuple[list[Block], dict[str, Any]]:
    candidates: list[Block] = []
    for phase, seg, lo, hi in _phase_pieces(segment_bounds, bounds):
        er = np.flatnonzero((event_segment == seg) & (event_time >= lo) & (event_time < hi))
        if er.size == 0: continue
        # One candidate boundary per fixed five minutes, independent of event rate.
        first = np.ceil(lo / 300.0) * 300.0
        for boundary in np.arange(first, hi - FUTURE_SECONDS + 1e-9, 300.0):
            future = er[(event_time[er] >= boundary) & (event_time[er] < boundary + FUTURE_SECONDS)]
            if future.size == 0: continue
            before = er[event_time[er] < boundary]
            if kind == "physical":
                start = boundary - float(value)
                if start < lo: continue
                exposure = before[event_time[before] >= start]
            else:
                n = int(value)
                if before.size < n: continue
                exposure = before[-n:]
                start = float(event_time[exposure[0]])
                if start < lo: continue
            if exposure.size == 0: continue
            candidates.append(Block(phase, seg, float(start), float(boundary),
                                    float(boundary + FUTURE_SECONDS), exposure, future))
    chosen = []
    by_phase = {}
    for phase in ("FIT", "INNER", "SELECTION"):
        current = _greedy([b for b in candidates if b.phase == phase])
        chosen.extend(current); by_phase[phase] = len(current)
    return chosen, {"n_sliding_candidates": len(candidates), "n_nonoverlap_by_phase": by_phase,
                    "nonoverlap_rule": "greedy earliest-finish on full exposure+future support"}


def _masked_mean(endpoint: Endpoint, rows: np.ndarray) -> np.ndarray:
    mask = endpoint.valid[rows]
    count = mask.sum(0)
    return np.where(count > 0,
                    np.where(mask, endpoint.values[rows], 0.0).sum(0) / np.maximum(count, 1),
                    0.0).astype(np.float64)


def _burden_features(endpoint: dict[str, Endpoint], blocks: list[Block]) -> np.ndarray:
    extent = endpoint["extent_fraction"].values[:, 0]
    rows = []
    for block in blocks:
        er = block.exposure_rows
        duration = max(block.boundary - block.exposure_start, 1.0)
        rows.append([np.log1p(er.size), np.log1p(60.0 * er.size / duration),
                     float(np.mean(extent[er])), float(np.std(extent[er])), float(np.max(extent[er]))])
    return np.asarray(rows, dtype=np.float64)


def _source_features(endpoints: dict[str, Endpoint], blocks: list[Block], event_time: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    burden = _burden_features(endpoints, blocks)
    quartiles = np.zeros((len(blocks), 4), dtype=np.float64)
    marks = []
    names = ("participation_field", "continuous_lag_field", "multiband_log_energy_field",
             "multiband_peak_time_field", "cross_band_lag_field", "waveform_morphology")
    for i, block in enumerate(blocks):
        edges = np.linspace(block.exposure_start, block.boundary, 5)
        quartiles[i] = [np.sum((event_time[block.exposure_rows] >= a) &
                               (event_time[block.exposure_rows] < b)) for a, b in zip(edges[:-1], edges[1:])]
        quartiles[i] /= max(block.exposure_rows.size, 1)
        marks.append(np.concatenate([_masked_mean(endpoints[name], block.exposure_rows) for name in names]))
    return np.concatenate((burden, quartiles), axis=1), np.asarray(marks, dtype=np.float64)


def _common_time_features(blocks: list[Block], segment_bounds: np.ndarray) -> np.ndarray:
    """Known elapsed time/clock terms shared by all H3 model arms.

    Event-count windows have patient- and state-dependent physical duration.
    If duration is available only to the burden arm, M1 can appear to win by
    learning elapsed time rather than an IED feedback-like dependence.  These
    terms therefore belong to the common-drive arm.

    Segment position is expressed as elapsed time since the coverage segment
    started, on a fixed 8 h scale.  The segment end is never used: target
    segments end exactly at seizure onsets for most patients, so a fraction of
    the total segment length would hand every arm a countdown to the next
    seizure or gap (review 2026-09-04).
    """

    rows = []
    for block in blocks:
        duration = max(block.boundary - block.exposure_start, 1.0)
        start_clock = 2.0 * np.pi * (block.exposure_start % 86400.0) / 86400.0
        end_clock = 2.0 * np.pi * (block.boundary % 86400.0) / 86400.0
        lo = float(segment_bounds[int(block.segment), 0])
        rows.append([
            np.log1p(duration),
            np.sin(start_clock), np.cos(start_clock),
            np.sin(end_clock), np.cos(end_clock),
            min(max(block.exposure_start - lo, 0.0), SESSION_POSITION_SCALE_SECONDS) / SESSION_POSITION_SCALE_SECONDS,
            min(max(block.boundary - lo, 0.0), SESSION_POSITION_SCALE_SECONDS) / SESSION_POSITION_SCALE_SECONDS,
        ])
    return np.asarray(rows, dtype=np.float64)


def _outcomes(endpoints: dict[str, Endpoint], blocks: list[Block]) -> dict[str, np.ndarray]:
    return {
        "future_event_count": np.asarray([[b.future_rows.size] for b in blocks], dtype=np.float64),
        "future_extent": np.asarray([_masked_mean(endpoints["extent_fraction"], b.future_rows) for b in blocks]),
        "future_participation_field": np.asarray([_masked_mean(endpoints["participation_field"], b.future_rows) for b in blocks]),
        "future_continuous_lag_field": np.asarray([_masked_mean(endpoints["continuous_lag_field"], b.future_rows) for b in blocks]),
        "future_multiband_energy_field": np.asarray([_masked_mean(endpoints["multiband_log_energy_field"], b.future_rows) for b in blocks]),
        "future_waveform_morphology": np.asarray([_masked_mean(endpoints["waveform_morphology"], b.future_rows) for b in blocks]),
    }


def _feature_at_times(time_: np.ndarray, segment: np.ndarray, trajectory: dict[str, np.ndarray],
                      rate: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    et, es, post = trajectory["event_time"], trajectory["event_segment"], trajectory["state_post"]
    mean, taus = trajectory["state_mean"], trajectory["fixed_taus_seconds"]
    state = np.broadcast_to(mean, (time_.size, mean.size)).copy()
    q = np.zeros((time_.size, rate["q_standardized"].shape[1]), dtype=np.float64)
    for seg in np.unique(segment):
        rows = np.flatnonzero(segment == seg); er = np.flatnonzero(es == seg); ar = np.flatnonzero(rate["segment"] == seg)
        ep = np.searchsorted(et[er], time_[rows], side="left") - 1 if er.size else np.full(rows.size, -1)
        ok = ep >= 0
        if er.size:
            donor = er[np.maximum(ep, 0)]; dt = time_[rows] - et[donor]
            current = mean[None] + (post[donor] - mean[None]) * np.exp(-dt[:, None] / taus[None])
            state[rows[ok]] = current[ok]
        if ar.size:
            qp = np.searchsorted(rate["anchor_time"][ar], time_[rows], side="right") - 1
            qok = qp >= 0; q[rows[qok]] = rate["q_standardized"][ar[qp[qok]]]
    return state, q


def _rank_projection(values: np.ndarray, fit: np.ndarray, rank: int = SOURCE_RANK) -> tuple[np.ndarray, dict[str, Any]]:
    centre = np.mean(values[fit], axis=0, keepdims=True)
    scale = np.std(values[fit], axis=0, keepdims=True)
    scale = np.where(scale > 1e-6, scale, 1.0)
    z = (values - centre) / scale
    _u, _s, vt = np.linalg.svd(z[fit], full_matrices=False)
    r = min(rank, vt.shape[0], vt.shape[1])
    projected = np.zeros((values.shape[0], rank), dtype=np.float64)
    if r: projected[:, :r] = z @ vt[:r].T
    return projected, {"requested_rank": rank, "actual_rank": r,
                       "train_variance": float(np.var(projected[fit, :r])) if r else 0.0}


def _ridge_fit(x: np.ndarray, y: np.ndarray, fit: np.ndarray, alpha: float) -> np.ndarray:
    xx = np.column_stack((np.ones(x.shape[0]), x))
    penalty = np.eye(xx.shape[1]) * alpha; penalty[0, 0] = 0.0
    return np.linalg.solve(xx[fit].T @ xx[fit] + penalty, xx[fit].T @ y[fit])


def _mse(x: np.ndarray, coef: np.ndarray, y: np.ndarray, rows: np.ndarray) -> float:
    pred = np.column_stack((np.ones(x.shape[0]), x)) @ coef
    return float(np.mean((pred[rows] - y[rows]) ** 2))


def _nested_arm_admissibility(parent: dict[str, float], child: dict[str, float],
                              *, limit: float = NESTED_MSE_RATIO_LIMIT,
                              null: dict[str, float] | None = None) -> dict[str, Any]:
    """Classify numerical stability of an added H3 feedback slot.

    The comparison is deliberately blind to the sign of the reported gain.
    An added slot is inadmissible when, on either the INNER or SELECTION
    split, the child and parent mean-squared errors differ by more than a
    factor of ``limit`` in EITHER direction.  A child that explodes is a
    poorly identified extrapolating ridge fit; a parent that explodes while
    the child stays bounded is the same divergence wearing a favourable sign
    (the 6 h arm produced "gains" of +20 to +90 standardized units this way).
    Raw scores are retained in the card; only the scientific contrast is
    withheld.  Symmetric clause added at review 2026-09-04.

    ``null`` (optional) carries the INNER/SELECTION MSE of the FIT-mean
    predictor for the same standardized outcome.  When both parent and child
    diverge together (6 h arm: MSE of 90 to 400 on a unit-variance outcome),
    their ratio can look innocent, so either arm exceeding ``limit`` times the
    null predictor's MSE also withholds the contrast.
    """

    ratios: dict[str, float | None] = {}
    reasons: list[str] = []
    eps = np.finfo(np.float64).eps
    if null is not None:
        for split in ("inner", "selection"):
            z = float(null.get(f"{split}_mse", np.nan))
            for label, arm in (("parent", parent), ("child", child)):
                v = float(arm.get(f"{split}_mse", np.nan))
                if np.isfinite(z) and np.isfinite(v) and v > limit * max(z, eps):
                    reasons.append(f"{split}_{label}_mse_{v:.6g}_exceeds_{limit:g}x_fit_mean_null_{z:.6g}")
    for split in ("inner", "selection"):
        p = float(parent.get(f"{split}_mse", np.nan))
        c = float(child.get(f"{split}_mse", np.nan))
        if not np.isfinite(p) or not np.isfinite(c):
            ratios[split] = None
            reasons.append(f"nonfinite_{split}_mse")
            continue
        ratio = c / max(p, np.finfo(np.float64).eps)
        ratios[split] = float(ratio)
        if ratio > limit:
            reasons.append(f"{split}_mse_ratio_{ratio:.6g}_exceeds_{limit:g}")
        elif ratio < 1.0 / limit:
            reasons.append(f"{split}_parent_mse_ratio_{1.0 / ratio:.6g}_exceeds_{limit:g}")
    return {
        "admissible": not reasons,
        "parent_relative_mse_ratio": ratios,
        "fit_mean_null_mse": None if null is None else {k: float(v) for k, v in null.items()},
        "limit": float(limit),
        "reasons": reasons,
    }


def _compare_models(common: np.ndarray, burden: np.ndarray, mark: np.ndarray,
                    outcomes: dict[str, np.ndarray], phase: np.ndarray) -> dict[str, Any]:
    fit, inner, selection = (np.flatnonzero(phase == p) for p in ("FIT", "INNER", "SELECTION"))
    if fit.size < 4 or inner.size < 2 or selection.size < 2:
        return {"status": "NOT_ESTIMABLE", "support": {"FIT": int(fit.size), "INNER": int(inner.size),
                "SELECTION": int(selection.size)}, "reason": "need >=4/2/2 independent full blocks"}
    cp, ca = _rank_projection(common, fit)
    bp, ba = _rank_projection(burden, fit)
    # Mark content is residualised against burden using FIT only, so M2 does
    # not relabel a count/rate effect as content-specific feedback.
    design = np.column_stack((np.ones(burden.shape[0]), burden))
    ridge = np.eye(design.shape[1]) * 1e-2; ridge[0, 0] = 0.0
    coef_mark = np.linalg.solve(design[fit].T @ design[fit] + ridge,
                                design[fit].T @ mark[fit])
    mp, ma = _rank_projection(mark - design @ coef_mark, fit)
    zero = np.zeros_like(bp)
    xarm = {
        "M0_common_drive": np.concatenate((cp, zero, zero), axis=1),
        "M1_burden_feedback": np.concatenate((cp, bp, zero), axis=1),
        "M2_mark_feedback": np.concatenate((cp, bp, mp), axis=1),
    }
    results = {"status": "ESTIMATED", "support": {"FIT": int(fit.size), "INNER": int(inner.size),
               "SELECTION": int(selection.size)}, "projection": {"common": ca, "burden": ba, "mark": ma},
               "parameter_template": "intercept + common_rank4 + burden_slot4 + mark_slot4 in all arms"}
    endpoint_results = {}
    for name, yraw in outcomes.items():
        centre = np.mean(yraw[fit], axis=0, keepdims=True)
        scale = np.std(yraw[fit], axis=0, keepdims=True); scale = np.where(scale > 1e-6, scale, 1.0)
        y = (yraw - centre) / scale
        # FIT-mean predictor (zero in standardized units): the absolute
        # reference for the admissibility clause below.
        null_mse = {"inner_mse": float(np.mean(y[inner] ** 2)),
                    "selection_mse": float(np.mean(y[selection] ** 2))}
        arm = {}
        fitted = {}
        for model_name, x in xarm.items():
            best = None
            for alpha in RIDGES:
                coef = _ridge_fit(x, y, fit, alpha); value = _mse(x, coef, y, inner)
                if best is None or value < best[0]: best = (value, alpha, coef)
            arm[model_name] = {"selection_mse": _mse(x, best[2], y, selection),
                               "inner_mse": best[0], "alpha": best[1]}
            fitted[model_name] = best[2]
        burden_stability = _nested_arm_admissibility(
            arm["M0_common_drive"], arm["M1_burden_feedback"], null=null_mse)
        mark_stability = _nested_arm_admissibility(
            arm["M1_burden_feedback"], arm["M2_mark_feedback"], null=null_mse)
        arm["null_fit_mean"] = null_mse
        # A content-specific comparison also depends on the burden parent
        # being a stable fitted model.  Keep its own diagnostic separately so
        # the exact failure can be audited.
        mark_admissible = bool(burden_stability["admissible"] and mark_stability["admissible"])
        raw_contrasts = {
            "burden_gain_over_common": arm["M0_common_drive"]["selection_mse"] - arm["M1_burden_feedback"]["selection_mse"],
            "mark_gain_over_burden": arm["M1_burden_feedback"]["selection_mse"] - arm["M2_mark_feedback"]["selection_mse"],
        }
        arm["admissibility"] = {
            "burden_feedback": burden_stability,
            "mark_feedback": {
                **mark_stability,
                "admissible": mark_admissible,
                "parent_burden_admissible": bool(burden_stability["admissible"]),
            },
            "rule": (
                "post-execution numerical admissibility clarification: an added nested arm is withheld "
                "when INNER or SELECTION MSE differs from its nested parent by more than 4x in either "
                "direction, or when parent or child exceeds 4x the FIT-mean null predictor's MSE; "
                "raw scores remain reported"
            ),
        }
        arm["raw_contrasts"] = raw_contrasts
        arm["contrasts"] = {
            "burden_gain_over_common": raw_contrasts["burden_gain_over_common"]
            if burden_stability["admissible"] else None,
            "mark_gain_over_burden": raw_contrasts["mark_gain_over_burden"]
            if mark_admissible else None,
        }
        for model_name, slot in (("M1_burden_feedback", 1), ("M2_mark_feedback", 2)):
            x = xarm[model_name].copy(); counter = x.copy()
            counter[:, SOURCE_RANK * slot:SOURCE_RANK * (slot + 1)] = 0.0
            full = np.column_stack((np.ones(x.shape[0]), x)) @ fitted[model_name]
            no = np.column_stack((np.ones(x.shape[0]), counter)) @ fitted[model_name]
            arm[model_name]["signed_impulse_mean_selection"] = float(np.mean(full[selection] - no[selection]))
        endpoint_results[name] = arm
    results["endpoints"] = endpoint_results
    return results


def _functional_innovation(data, endpoints: dict[str, Endpoint], trajectory: dict[str, np.ndarray]) -> dict[str, Any]:
    """Map event-triggered state updates into held-out future-event predictions.

    The functional map is fitted on FIT and tuned on INNER.  Selection then
    reports how the current event's post-minus-pre update changes the predicted
    future extent and participation field at 1/5/20 events.  This is an observer
    innovation diagnostic; it is deliberately not labelled a causal effect.
    """
    pre = np.asarray(trajectory["state_pre"], dtype=np.float64)
    post = np.asarray(trajectory["state_post"], dtype=np.float64)
    q = np.asarray(data.q_context, dtype=np.float64)
    finite = np.isfinite(pre).all(1) & np.isfinite(post).all(1)
    fit = np.flatnonzero((data.phase == "FIT") & finite)
    inner = np.flatnonzero((data.phase == "INNER") & finite)
    selection = np.flatnonzero((data.phase == "SELECTION") & finite)
    if min(fit.size, inner.size, selection.size) == 0:
        return {"status": "NOT_ESTIMABLE", "support": {"FIT": int(fit.size), "INNER": int(inner.size),
                                                           "SELECTION": int(selection.size)}}
    current_extent = endpoints["extent_fraction"].values[:, 0].astype(np.float64)
    energy_endpoint = endpoints["multiband_log_energy_field"]
    energy_count = energy_endpoint.valid.sum(axis=1)
    energy_sum = np.where(energy_endpoint.valid, energy_endpoint.values, 0.0).sum(axis=1)
    current_energy = np.full(energy_count.shape, np.nan, dtype=np.float64)
    np.divide(energy_sum, energy_count, out=current_energy, where=energy_count > 0)
    output = {"status": "ESTIMATED", "semantics": "functional readout(post state) minus functional readout(pre state)"}
    for offset_j, offset in enumerate((1, 5, 20)):
        target = data.next_index[:, offset_j]
        good = (target >= 0) & finite
        rows_by_phase = {
            "FIT": fit[good[fit]], "INNER": inner[good[inner]], "SELECTION": selection[good[selection]],
        }
        if min(v.size for v in rows_by_phase.values()) == 0:
            output[f"next_{offset}_events"] = {"status": "NOT_ESTIMABLE"}; continue
        target_extent = current_extent[target[good]]
        part = endpoints["participation_field"]
        target_part = part.values[target[good]].astype(np.float64)
        all_rows = np.flatnonzero(good)
        # A TRAIN-only participation projection keeps the functional target compact.
        fit_local = np.searchsorted(all_rows, rows_by_phase["FIT"])
        part_pc, pc_audit = _rank_projection(target_part, fit_local, rank=SOURCE_RANK)
        y = np.column_stack((target_extent, part_pc))
        x = np.column_stack((q[good], pre[good]))
        local = {k: np.searchsorted(all_rows, v) for k, v in rows_by_phase.items()}
        centre = np.mean(y[local["FIT"]], axis=0, keepdims=True)
        scale = np.std(y[local["FIT"]], axis=0, keepdims=True); scale = np.where(scale > 1e-6, scale, 1.0)
        yz = (y - centre) / scale
        best = None
        for alpha in RIDGES:
            coef = _ridge_fit(x, yz, local["FIT"], alpha)
            value = _mse(x, coef, yz, local["INNER"])
            if best is None or value < best[0]: best = (value, alpha, coef)
        # _ridge_fit adds intercept; state coefficients follow q coefficients.
        q_width = q.shape[1]
        state_coef = best[2][1 + q_width:]
        innovation = (post[all_rows] - pre[all_rows]) @ state_coef
        sel_local = local["SELECTION"]
        scalar = np.mean(innovation[sel_local], axis=1)
        source_rows = rows_by_phase["SELECTION"]
        def corr(a, b):
            if a.size < 4 or np.std(a) < 1e-9 or np.std(b) < 1e-9: return None
            value = np.corrcoef(a, b)[0, 1]
            return float(value) if np.isfinite(value) else None
        output[f"next_{offset}_events"] = {
            "status": "ESTIMATED", "alpha": best[1], "inner_mse": best[0],
            "n_selection_events": int(sel_local.size),
            "signed_functional_innovation_mean": float(np.mean(scalar)),
            "association_with_current_extent": corr(scalar, current_extent[source_rows]),
            "association_with_current_multiband_energy": corr(scalar, np.nan_to_num(current_energy[source_rows])),
            "participation_projection": pc_audit,
        }
    return output


def run_feedback_models(subject: str, trajectory_path: Path, rate_path: Path,
                        *, out_dir: Path, overwrite: bool = False) -> dict[str, Any]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    card_path = out_dir / "card.json"
    if card_path.exists() and not overwrite: return json.loads(card_path.read_text(encoding="utf-8"))
    manifest = json.loads((INPUT_ROOT / subject / "manifest_v3.json").read_text(encoding="utf-8"))
    bounds = {k: float(v) for k, v in manifest["report"]["phase_boundaries_epoch"].items()}
    with np.load(manifest["input_path"], allow_pickle=False) as z:
        segment_bounds = np.asarray(z["target_segment_bounds"], dtype=np.float64)
    with np.load(trajectory_path, allow_pickle=False) as z: trajectory = {k: np.asarray(z[k]) for k in z.files}
    with np.load(rate_path, allow_pickle=False) as z: rate = {k: np.asarray(z[k]) for k in z.files}
    # Reuse the exact W3 event universe and full event endpoints.
    from src.topic5_group_event_state.v034_spatial_state.we_decoder import load_frozen_decoder
    from .contracts import DECODER_ROOT, V035_DECODER_FITS
    from .full_mark_state import load_full_mark_data
    fits = V035_DECODER_FITS
    # Decoder seed is encoded in the output unit name and passed through an environment-free sidecar.
    decoder_seed = int(Path(trajectory_path).parent.name.split("decoder_seed")[1].split("_")[0])
    import torch
    bundle = load_frozen_decoder(DECODER_ROOT / "formal_units" / fits[subject] /
                                 "L3_LOCAL_PLUS_LEARNED_LR" / f"seed{decoder_seed}",
                                 DECODER_ROOT / "cache" / fits[subject], device=torch.device("cpu"))
    data = load_full_mark_data(subject, bundle, rate_path)
    endpoints = build_endpoints(data)
    innovation = _functional_innovation(data, endpoints, trajectory)
    results = {}
    for kind, values in (("physical", PHYSICAL_EXPOSURES), ("event_count", EVENT_EXPOSURES)):
        for value in values:
            blocks, audit = build_blocks(data.event_time, data.event_segment, segment_bounds, bounds,
                                         kind=kind, value=value)
            key = f"{kind}_{int(value)}"
            if not blocks:
                results[key] = {"status": "NOT_ESTIMABLE", "audit": audit}; continue
            start = np.asarray([b.exposure_start for b in blocks]); seg = np.asarray([b.segment for b in blocks])
            state, q = _feature_at_times(start, seg, trajectory, rate)
            burden, mark = _source_features(endpoints, blocks, data.event_time)
            time_common = _common_time_features(blocks, segment_bounds)
            common = np.concatenate((state, q, time_common), axis=1)
            outcomes = _outcomes(endpoints, blocks)
            phase = np.asarray([b.phase for b in blocks])
            current = _compare_models(common, burden, mark, outcomes, phase)
            current["audit"] = audit
            current["common_drive_controls"] = (
                "pre-exposure frozen state/q plus exposure duration, start/end clock, "
                "and start/end position within the real coverage segment"
            )
            current["exposure_kind"] = kind; current["exposure_value"] = value
            current["future_seconds"] = FUTURE_SECONDS
            results[key] = current
    card = {"format": f"{FORMAT_PREFIX}_long_feedback_models_v3", "subject": subject,
            "trajectory": str(trajectory_path), "rate_trajectory": str(rate_path), "designs": results,
            "functional_innovation": innovation,
            "interpretation_ceiling": "event-feedback-like predictive dependence; not human causal proof",
            "numerical_admissibility": (
                "added feedback arm must have finite INNER/SELECTION MSE and neither split may differ "
                "from the corresponding nested-parent MSE by more than 4x in either direction; raw scores are retained"
            ),
            "development_targets_read": False, "sealed_partition_opened": False,
            "seizure_outcomes_read": False}
    atomic_json(card_path, card); return card
