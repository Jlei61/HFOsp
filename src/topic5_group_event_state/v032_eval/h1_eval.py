"""H1 paired evaluation: identical NB readouts on identical anchors for every arm.

Arms (all on top of the same explicit history ``H``)::

    H | H+S_correct | H+S_shifted:j (j=1..5) | H+S_mean | H+random_reservoir
      | H+times_only | H+linear_marked_ema | intercept_only (diagnostic)

Nothing is fitted on scoring rows.  Ridge is selected on ``inner_val`` after a
``base_fit`` fit, the selected configuration is refitted on ``base_refit``, and
dev_val / dev_test are scored once with every fitted quantity frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .blocks import block_ids_for_times, paired_gain_summary
from .contract import atomic_json, atomic_npz, finite_or_none, now_iso
from .controls import linear_marked_ema, random_reservoir_state, times_only_state
from .history import HistoryFeatureBuilder, history_inputs_from_timeline
from .nb_glm import NegativeBinomialRidge, select_and_refit
from .partition import EVAL_PHASES, REFIT_PHASE
from .shift import apply_donor, predefined_session_shifts
from .state_registry import StateBundle
from .timeline import EvalTimeline

SCORE_PHASES = ("dev_val", "dev_test")
CONTROL_ARMS = ("H+random_reservoir", "H+times_only", "H+linear_marked_ema")
STATE_ARMS = ("H+S_correct", "H+S_mean")


@dataclass
class H1Design:
    anchor_time: np.ndarray
    anchor_segment: np.ndarray
    anchor_session: np.ndarray
    history: dict[str, tuple[np.ndarray, list[str]]]
    controls: dict[str, tuple[np.ndarray, list[str]]]


def build_h1_design(tl: EvalTimeline, cfg: Mapping[str, Any]) -> H1Design:
    inputs = history_inputs_from_timeline(tl)
    hist_cfg = cfg["history"]
    builder = HistoryFeatureBuilder(
        inputs,
        lookback_seconds=hist_cfg["lookback_seconds"],
        ewma_tau_seconds=hist_cfg["ewma_tau_seconds"],
        field_tau_seconds=hist_cfg["field_tau_seconds"],
    )
    t, seg = tl.grid.t_anchor, tl.grid.segment_index
    history = {v: builder.features(t, seg, variant=v) for v in cfg["history_variants"]}
    ctrl = cfg["controls"]
    controls = {
        "H+random_reservoir": random_reservoir_state(
            inputs, t, seg, dim=int(ctrl["reservoir_dim"]), taus=ctrl["reservoir_taus_seconds"],
            seed=int(ctrl["reservoir_seed"])),
        "H+times_only": times_only_state(inputs, t, seg),
        "H+linear_marked_ema": linear_marked_ema(inputs, t, seg, taus=ctrl["linear_ema_taus_seconds"]),
    }
    return H1Design(anchor_time=t, anchor_segment=seg, anchor_session=tl.grid.session_id,
                    history=history, controls=controls)


def state_mean_over_base_fit(tl: EvalTimeline, anchor_state: np.ndarray) -> np.ndarray:
    """TRAIN-mean state: average over base_fit anchors with a finite state."""

    mask = tl.partition.mask_for_phase(tl.grid.t_anchor, "base_fit") & np.isfinite(anchor_state).all(axis=1)
    if not mask.any():
        return np.full(anchor_state.shape[1], np.nan)
    return anchor_state[mask].mean(axis=0)


def shifted_states(design: H1Design, anchor_state: np.ndarray, cfg: Mapping[str, Any],
                   horizon: float) -> list[dict[str, Any]]:
    sh = cfg["shift"]
    specs = predefined_session_shifts(
        design.anchor_time, design.anchor_session,
        n_shifts=int(sh["n_shifts"]), denominator=int(sh["fraction_denominator"]),
        min_distance_seconds=float(horizon) + float(sh["min_gap_over_horizon_seconds"]),
    )
    out = []
    for spec in specs:
        out.append({**spec, "state": apply_donor(anchor_state, spec["donor_index"])})
    return out


def _finite_rows(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x).all(axis=1)


def score_arms(
    y: np.ndarray,
    rows: Mapping[str, np.ndarray],
    designs: Mapping[str, np.ndarray],
    cfg: Mapping[str, Any],
    *,
    dispersion_rule: str = "per_arm",
    reference_arm: str = "H",
) -> dict[str, Any]:
    """Fit every arm with the same recipe and score the same rows.

    ``rows`` holds anchor indices per phase (``base_fit``, ``inner_val``,
    ``base_refit``, ``dev_val``, ``dev_test``).  ``designs`` maps an arm to its
    full (A, F) design; rows with a non-finite design value (e.g. a shifted
    anchor without a donor) are excluded for that arm and reported.
    """

    nb = cfg["nb_glm"]
    out: dict[str, Any] = {"arms": {}, "dispersion_rule": dispersion_rule}
    fixed_alpha = None
    order = [reference_arm] + [a for a in designs if a != reference_arm]
    for arm in order:
        x = np.asarray(designs[arm], dtype=np.float64)
        finite = _finite_rows(x)
        fit_rows = rows["base_fit"][finite[rows["base_fit"]]]
        select_rows = rows["inner_val"][finite[rows["inner_val"]]]
        refit_rows = rows[REFIT_PHASE][finite[rows[REFIT_PHASE]]]
        entry: dict[str, Any] = {
            "n_features": int(x.shape[1]),
            "n_rows_dropped_nonfinite": {p: int((~finite[rows[p]]).sum()) for p in rows},
        }
        if fit_rows.size < 3 or select_rows.size < 1 or refit_rows.size < 3:
            entry["status"] = "not_estimable"
            entry["reason"] = f"fit={fit_rows.size} select={select_rows.size} refit={refit_rows.size}"
            out["arms"][arm] = entry
            continue
        try:
            fit = select_and_refit(
                x, y, fit_rows=fit_rows, select_rows=select_rows, refit_rows=refit_rows,
                ridge_grid=nb["ridge_grid"], alpha_log_bounds=tuple(nb["alpha_log_bounds"]),
                fixed_alpha=(fixed_alpha if (dispersion_rule == "shared_H_alpha" and arm != reference_arm) else None),
                max_iter=int(nb["max_irls_iter"]),
            )
        except (RuntimeError, np.linalg.LinAlgError, ValueError) as exc:
            entry["status"] = "solver_failure"
            entry["reason"] = f"{type(exc).__name__}: {exc}"
            out["arms"][arm] = entry
            continue
        model: NegativeBinomialRidge = fit["model"]
        if arm == reference_arm:
            fixed_alpha = float(model.alpha_)
        scores = {}
        for phase in SCORE_PHASES:
            idx = rows[phase]
            nll = np.full(idx.size, np.nan)
            mu = np.full(idx.size, np.nan)
            ok = finite[idx]
            if ok.any():
                nll[ok] = model.nll(x[idx[ok]], y[idx[ok]])
                mu[ok] = model.predict_mu(x[idx[ok]])
            scores[phase] = {"nll": nll, "mu": mu}
        entry.update({
            "status": "ok",
            "selected_ridge": fit["selected_ridge"],
            "ridge_at_edge": fit["ridge_at_edge"],
            "selection_nll": fit["selection_nll"],
            "ridge_path": fit["path"],
            "solver_failures": fit["solver_failures"],
            "alpha": float(model.alpha_),
            "intercept": float(model.intercept_),
            "converged": bool(model.converged_),
            "n_fit_rows": fit["n_fit_rows"], "n_select_rows": fit["n_select_rows"], "n_refit_rows": fit["n_refit_rows"],
            "scores": scores,
            "calibration": {
                phase: {
                    "mean_observed": finite_or_none(np.nanmean(y[rows[phase]])) if rows[phase].size else None,
                    "mean_predicted": finite_or_none(np.nanmean(scores[phase]["mu"])) if rows[phase].size else None,
                } for phase in SCORE_PHASES
            },
        })
        out["arms"][arm] = entry
    return out


def _mean_over_shifts(arms: Mapping[str, Any], phase: str, n_rows: int) -> tuple[np.ndarray, list[str]]:
    used = []
    stack = []
    for name, entry in arms.items():
        if name.startswith("H+S_shifted:") and entry.get("status") == "ok":
            stack.append(entry["scores"][phase]["nll"])
            used.append(name)
    if not stack:
        return np.full(n_rows, np.nan), used
    arr = np.stack(stack, axis=0)
    with np.errstate(all="ignore"):
        mean = np.nanmean(arr, axis=0)
    return mean, used


def paired_summaries(arms: Mapping[str, Any], rows: Mapping[str, np.ndarray],
                     blocks: Mapping[str, np.ndarray], cfg: Mapping[str, Any]) -> dict[str, Any]:
    inf = cfg["inference"]
    n_boot, seed = int(inf["bootstrap_replicates"]), int(inf["bootstrap_seed"])
    out: dict[str, Any] = {}
    for phase in SCORE_PHASES:
        n = rows[phase].size
        entry: dict[str, Any] = {}
        ref = arms.get("H")
        if not ref or ref.get("status") != "ok":
            out[phase] = {"status": "reference_H_not_estimable"}
            continue
        h_nll = ref["scores"][phase]["nll"]
        for name, arm in arms.items():
            if name == "H" or arm.get("status") != "ok":
                continue
            entry[f"{name}_vs_H"] = paired_gain_summary(h_nll, arm["scores"][phase]["nll"], blocks[phase], n_boot=n_boot, seed=seed)
        correct = arms.get("H+S_correct")
        if correct and correct.get("status") == "ok":
            c_nll = correct["scores"][phase]["nll"]
            shift_mean, used = _mean_over_shifts(arms, phase, n)
            entry["H+S_correct_vs_H+S_shifted_mean"] = paired_gain_summary(shift_mean, c_nll, blocks[phase], n_boot=n_boot, seed=seed)
            entry["H+S_correct_vs_H+S_shifted_mean"]["shifts_used"] = used
            per_shift = {}
            for name in used:
                per_shift[name] = paired_gain_summary(arms[name]["scores"][phase]["nll"], c_nll, blocks[phase], n_boot=n_boot, seed=seed)
            entry["H+S_correct_vs_each_shift"] = per_shift
            mean_arm = arms.get("H+S_mean")
            if mean_arm and mean_arm.get("status") == "ok":
                entry["H+S_correct_vs_H+S_mean"] = paired_gain_summary(mean_arm["scores"][phase]["nll"], c_nll, blocks[phase], n_boot=n_boot, seed=seed)
            for ctrl in CONTROL_ARMS:
                if arms.get(ctrl, {}).get("status") == "ok":
                    entry[f"H+S_correct_vs_{ctrl}"] = paired_gain_summary(arms[ctrl]["scores"][phase]["nll"], c_nll, blocks[phase], n_boot=n_boot, seed=seed)
        out[phase] = {"status": "ok", "n_anchors": int(n), "n_blocks": int(np.unique(blocks[phase]).size), "pairs": entry}
    return out


def evaluate_h1_patient(
    tl: EvalTimeline,
    cfg: Mapping[str, Any],
    design: H1Design,
    *,
    state: StateBundle | None,
    out_dir: Path,
    label: str,
) -> dict[str, Any]:
    """Score every arm at every horizon for one patient (and one state seed)."""

    out_dir = Path(out_dir)
    horizons = tl.horizons_seconds
    seg_start = tl.segment_start_map()
    block_min = float(cfg["inference"]["block_seconds_min"])
    arrays: dict[str, np.ndarray] = {"anchor_time": design.anchor_time}
    report: dict[str, Any] = {
        "format": "group_event_state_v0_3_2_h1_patient_result",
        "subject": tl.subject, "label": label, "generated": now_iso(),
        "state": None if state is None else {
            "seed": state.seed, "state_dim": state.state_dim, "event_state_mode": state.event_state_mode,
            "n_anchor_matched": state.n_anchor_matched, "n_anchor_missing": state.n_anchor_missing,
            "provenance": state.provenance,
        },
        "horizons": {},
        "test_time_fit": False,
        "sealed_partition_opened": False,
    }
    for h_i, horizon in enumerate(horizons):
        key = f"{int(horizon)}s"
        rows = {p: tl.anchor_indices(p, h_i) for p in EVAL_PHASES + (REFIT_PHASE,)}
        y = np.zeros(tl.grid.n_anchors, dtype=np.int64)
        all_idx = np.flatnonzero(tl.grid.eligible[:, h_i])
        y[all_idx] = tl.window_counts(all_idx, h_i)
        blocks = {p: block_ids_for_times(design.anchor_time[rows[p]], design.anchor_segment[rows[p]], seg_start,
                                         max(float(horizon), block_min)) if rows[p].size else np.zeros(0, np.int64)
                  for p in SCORE_PHASES}
        entry: dict[str, Any] = {
            "horizon_seconds": float(horizon),
            "n_anchors_by_phase": {p: int(v.size) for p, v in rows.items()},
            "n_blocks_by_phase": {p: int(np.unique(b).size) for p, b in blocks.items()},
            "block_seconds": max(float(horizon), block_min),
        }
        if rows["inner_val"].size == 0 or rows["base_fit"].size < 3 or rows["dev_test"].size == 0:
            entry["status"] = "not_estimable_insufficient_phase_anchors"
            report["horizons"][key] = entry
            continue
        for p in SCORE_PHASES:
            arrays[f"h{int(horizon)}_{p}_anchor_index"] = rows[p]
            arrays[f"h{int(horizon)}_{p}_count"] = y[rows[p]]
            arrays[f"h{int(horizon)}_{p}_block"] = blocks[p]
        # state-derived designs
        s_designs: dict[str, np.ndarray] = {}
        shift_meta = []
        if state is not None and np.isfinite(state.anchor_state).any():
            s_designs["S_correct"] = state.anchor_state
            mean_vec = state_mean_over_base_fit(tl, state.anchor_state)
            s_designs["S_mean"] = np.broadcast_to(mean_vec, state.anchor_state.shape).copy()
            for spec in shifted_states(design, state.anchor_state, cfg, float(horizon)):
                s_designs[f"S_shifted:{spec['shift_id']}"] = spec["state"]
                shift_meta.append({"shift_id": spec["shift_id"], "fraction": spec["fraction"],
                                   "n_valid_anchors": spec["n_valid"], "min_distance_seconds": spec["min_distance_seconds"],
                                   "shift_anchors_by_session": spec["shift_anchors_by_session"]})
        entry["shifts"] = shift_meta
        entry["variants"] = {}
        for variant, (h_x, h_names) in design.history.items():
            designs: dict[str, np.ndarray] = {"H": h_x, "intercept_only": np.zeros((h_x.shape[0], 1))}
            for ctrl, (c_x, _c_names) in design.controls.items():
                designs[ctrl] = np.concatenate([h_x, c_x], axis=1)
            for name, s_x in s_designs.items():
                designs[f"H+{name}"] = np.concatenate([h_x, s_x], axis=1)
            variant_entry = {"history_features": len(h_names)}
            for rule in ("per_arm", "shared_H_alpha"):
                scored = score_arms(y, rows, designs, cfg, dispersion_rule=rule)
                # intercept-only must not receive the shared alpha treatment as a "state" arm; it is diagnostic
                for arm, arm_entry in scored["arms"].items():
                    if arm_entry.get("status") != "ok":
                        continue
                    for p in SCORE_PHASES:
                        arrays[f"h{int(horizon)}_{variant}_{arm}_{rule}_{p}_nll"] = arm_entry["scores"][p]["nll"]
                        arrays[f"h{int(horizon)}_{variant}_{arm}_{rule}_{p}_mu"] = arm_entry["scores"][p]["mu"]
                summaries = paired_summaries(scored["arms"], rows, blocks, cfg)
                variant_entry[rule] = {
                    "arms": {
                        arm: {k: v for k, v in arm_entry.items() if k != "scores"}
                        for arm, arm_entry in scored["arms"].items()
                    },
                    "paired": summaries,
                }
            entry["variants"][variant] = variant_entry
        entry["status"] = "ok"
        report["horizons"][key] = entry
    out_dir.mkdir(parents=True, exist_ok=True)
    array_path = atomic_npz(out_dir / f"h1_arrays_{label}.npz", arrays)
    report["arrays"] = str(array_path)
    atomic_json(out_dir / f"h1_result_{label}.json", report)
    return report
