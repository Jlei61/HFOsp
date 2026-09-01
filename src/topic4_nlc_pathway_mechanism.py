"""Frozen mode-rate and pathway-dynamics readouts for rev11-NLC."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from src.topic4_shaft_aware_direction import assign_direction_modes


ARM_IDS = (
    "node_baseline",
    "joint_04_ee_only",
    "joint_04_etoi_only",
    "joint_04_control",
)
MODE_NAMES = {0: "TA-like", 1: "TB-like"}


def formal_mode_assignments(
    onsets,
    event_returned,
    *,
    groups,
    embedding,
    classifier: Mapping,
    minimum_recruited_contacts=3,
):
    """Apply the frozen patient classifier and the preregistered event filter."""
    onsets = np.asarray(onsets, float)
    returned = np.asarray(event_returned, bool)
    if onsets.ndim != 2 or returned.shape != (len(onsets),):
        raise ValueError("onsets and event_returned must align")
    assigned = assign_direction_modes(
        onsets, groups=groups, embedding=embedding, classifier=classifier,
    )
    icl = np.isfinite(onsets[:, np.asarray(groups["ICL"], int)]).any(axis=1)
    scl = np.isfinite(onsets[:, np.asarray(groups["SCL"], int)]).any(axis=1)
    readable = np.isfinite(onsets).sum(axis=1) >= int(minimum_recruited_contacts)
    clean = returned & icl & scl & readable & ~np.asarray(assigned["ood"], bool)
    return {
        **assigned,
        "clean": clean,
        "joint_shaft": icl & scl,
        "readable": readable,
    }


def network_mode_endpoints(assignments, duration_ms):
    labels = np.asarray(assignments["labels"], int)
    clean = np.asarray(assignments["clean"], bool)
    returned = np.asarray(assignments.get("returned"), bool)
    if returned.shape != labels.shape:
        returned = np.ones_like(clean)
    counts = {name: int(np.sum(clean & (labels == mode)))
              for mode, name in MODE_NAMES.items()}
    duration_s = float(duration_ms) / 1000.0
    total = counts["TA-like"] + counts["TB-like"]
    returned_count = int(np.sum(returned))
    return {
        "TA_like_count": counts["TA-like"],
        "TB_like_count": counts["TB-like"],
        "TA_like_rate_hz": counts["TA-like"] / duration_s,
        "TB_like_rate_hz": counts["TB-like"] / duration_s,
        "clean_rate_hz": total / duration_s,
        "TB_like_fraction": counts["TB-like"] / max(total, 1),
        "n_clean_events": total,
        "n_returned_events": returned_count,
        "ood_fraction_returned": (
            float(np.mean(np.asarray(assignments["ood"], bool)[returned]))
            if returned_count else 0.0
        ),
    }


def bootstrap_mean(values, *, draws, seed):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"status": "NOT_EVALUABLE", "n_networks": 0}
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(values), size=(int(draws), len(values)))
    sampled = np.mean(values[indices], axis=1)
    return {
        "status": "OK",
        "n_networks": int(len(values)),
        "mean": float(np.mean(values)),
        "q05": float(np.quantile(sampled, 0.05)),
        "q50": float(np.quantile(sampled, 0.50)),
        "q95": float(np.quantile(sampled, 0.95)),
        "draws": int(draws),
    }


def paired_bootstrap(left, right, *, draws, seed):
    left, right = np.asarray(left, float), np.asarray(right, float)
    if left.shape != right.shape:
        raise ValueError("paired endpoints must align")
    finite = np.isfinite(left) & np.isfinite(right)
    delta = left[finite] - right[finite]
    if not len(delta):
        return {"status": "NOT_EVALUABLE", "n_paired_networks": 0}
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(delta), size=(int(draws), len(delta)))
    sampled = np.mean(delta[indices], axis=1)
    return {
        "status": "OK",
        "n_paired_networks": int(len(delta)),
        "mean_delta": float(np.mean(delta)),
        "q05": float(np.quantile(sampled, 0.05)),
        "q50": float(np.quantile(sampled, 0.50)),
        "q95": float(np.quantile(sampled, 0.95)),
        "probability_positive": float(np.mean(sampled > 0.0)),
        "probability_negative": float(np.mean(sampled < 0.0)),
        "draws": int(draws),
    }


def factorial_bootstrap(node, ee, etoi, joint, *, draws, seed):
    arrays = [np.asarray(values, float) for values in (node, ee, etoi, joint)]
    if len({values.shape for values in arrays}) != 1:
        raise ValueError("factorial endpoints must align")
    finite = np.logical_and.reduce([np.isfinite(values) for values in arrays])
    interaction = arrays[3][finite] - arrays[1][finite] - arrays[2][finite] + arrays[0][finite]
    return bootstrap_mean(interaction, draws=draws, seed=seed)


def event_aligned_pathway_readout(
    time_ms,
    traces,
    event_onsets_ms,
    labels,
    clean,
    *,
    event_window_ms,
    baseline_window_ms,
    summary_windows_ms,
    trace_dt_ms,
):
    """Return within-network, mode-conditioned baseline-corrected traces."""
    time_ms = np.asarray(time_ms, float)
    event_onsets_ms = np.asarray(event_onsets_ms, float)
    labels, clean = np.asarray(labels, int), np.asarray(clean, bool)
    if labels.shape != clean.shape or labels.shape != event_onsets_ms.shape:
        raise ValueError("event labels, mask and onset times must align")
    relative = np.arange(
        float(event_window_ms[0]),
        float(event_window_ms[1]) + 0.5 * float(trace_dt_ms),
        float(trace_dt_ms),
    )
    baseline = (
        (relative >= float(baseline_window_ms[0]))
        & (relative < float(baseline_window_ms[1]))
    )
    output = {"relative_time_ms": relative.tolist(), "modes": {}}
    for mode, mode_name in MODE_NAMES.items():
        selected = np.flatnonzero(clean & (labels == mode))
        mode_output = {"n_events": int(len(selected)), "traces": {}}
        for trace_name, trace_values in traces.items():
            trace_values = np.asarray(trace_values, float)
            rows = []
            for event_index in selected:
                query = event_onsets_ms[event_index] + relative
                row = np.interp(query, time_ms, trace_values, left=np.nan, right=np.nan)
                base = np.nanmean(row[baseline])
                rows.append(row - base)
            curve = (
                np.nanmean(np.asarray(rows, float), axis=0)
                if rows else np.full(len(relative), np.nan)
            )
            windows = {}
            for window_name, bounds in summary_windows_ms.items():
                mask = (relative >= float(bounds[0])) & (relative < float(bounds[1]))
                windows[window_name] = (
                    float(np.nanmean(curve[mask])) if np.isfinite(curve[mask]).any()
                    else None
                )
            mode_output["traces"][trace_name] = {
                "mean": curve.tolist(),
                "windows": windows,
            }
        output["modes"][mode_name] = mode_output
    return output
