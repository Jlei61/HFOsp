"""Frozen, horizon-specific evaluators for one shared S_N or S_G trajectory.

The producer is never updated here.  Every horizon may select its own small
ridge readout on INNER, but all read the same causal state trajectory.  The
reported comparisons use identical SELECTION anchors and include a FIT-period
constant-state arm plus a distant circular time shift.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import gammaln, softmax

from .contracts import FORMAT_PREFIX, atomic_json
from .full_mark_state import FullMarkData
from .grammar_targets import GrammarBlockTargets


RIDGES = (1e-4, 1e-2, 1.0, 100.0)
COUNT_DISPERSIONS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def grid_state_from_trajectory(data: FullMarkData, trajectory: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(trajectory, allow_pickle=False) as z:
        event_time = np.asarray(z["event_time"], dtype=np.float64)
        post = np.asarray(z["state_post"], dtype=np.float64)
        taus = np.asarray(z["fixed_taus_seconds"], dtype=np.float64)
        mean = np.asarray(z["state_mean"], dtype=np.float64)
    if not np.array_equal(event_time, data.event_time):
        raise ValueError("frozen trajectory is not aligned to the evaluator event stream")
    out = np.broadcast_to(mean, (data.grid_time.size, mean.size)).copy()
    source = np.asarray(data.grid_source_event, dtype=np.int64)
    safe = np.maximum(source, 0)
    valid = (source >= 0) & np.isfinite(post[safe]).all(axis=1)
    if np.any(valid):
        decay = np.exp(-data.grid_source_dt[valid, None] / taus[None])
        out[valid] = mean + (post[source[valid]] - mean) * decay
    return out.astype(np.float64), valid


def _standardise(fit: np.ndarray, value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(fit, axis=0)
    scale = np.nanstd(fit, axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return (value - mean) / scale, mean, scale


def _ridge_fit(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    design = np.column_stack((np.ones(x.shape[0]), x))
    penalty = np.eye(design.shape[1]); penalty[0, 0] = 0.0
    # Define ridge on the *mean* squared-error scale.  Without division by n,
    # duplicating identical anchors weakens regularisation and the same nominal
    # grid means something different for 2 h and 8 h heads.  That is the same
    # silent failure mode that made the earlier long-exposure ridge diverge.
    n = max(int(design.shape[0]), 1)
    gram = (design.T @ design) / float(n)
    rhs = (design.T @ y) / float(n)
    return np.linalg.solve(gram + float(ridge) * penalty, rhs)


def _predict(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones(x.shape[0]), x)) @ beta


def _poisson_nll(log_rate: np.ndarray, count: np.ndarray, exposure_seconds: np.ndarray) -> np.ndarray:
    mu = np.exp(np.clip(log_rate, -15.0, 15.0)) * np.maximum(exposure_seconds, 1.0) / 3600.0
    mu = np.clip(mu, 1e-8, 1e9)
    return mu - count * np.log(mu) + gammaln(count + 1.0)


def _negative_binomial_nll(
    log_rate: np.ndarray,
    count: np.ndarray,
    exposure_seconds: np.ndarray,
    dispersion: float,
) -> np.ndarray:
    """NB2 count score under the same exposure convention as training."""

    mu = np.exp(np.clip(log_rate, -15.0, 15.0)) * np.maximum(exposure_seconds, 1.0) / 3600.0
    mu = np.clip(mu, 1e-8, 1e9)
    r = max(float(dispersion), 1e-6)
    return -(
        gammaln(count + r) - gammaln(r) - gammaln(count + 1.0)
        + r * (np.log(r) - np.log(r + mu))
        + count * (np.log(mu) - np.log(r + mu))
    )


def _distribution_score(logit: np.ndarray, target: np.ndarray) -> np.ndarray:
    return -(target * np.log(np.clip(softmax(logit, axis=1), 1e-9, 1.0))).sum(axis=1)


def _shift_donors(times: np.ndarray, rows: np.ndarray, minimum_separation: float) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size < 4:
        return rows[:0], rows[:0]
    candidates = np.roll(rows, rows.size // 2)
    keep = np.abs(times[candidates] - times[rows]) >= float(minimum_separation)
    return rows[keep], candidates[keep]


def _non_overlapping_window_count(times: np.ndarray, horizon_seconds: float) -> int:
    """Greedy count of disjoint wall-clock prediction windows.

    Grid anchors overlap heavily.  Reporting their row count as the amount of
    independent long-horizon evidence would be misleading, so every card also
    carries this conservative physical-window count.
    """

    values = np.sort(np.unique(np.asarray(times, dtype=np.float64)))
    if values.size == 0:
        return 0
    count = 0
    available = -np.inf
    for value in values:
        if value >= available:
            count += 1
            available = float(value) + float(horizon_seconds)
    return count


def _select_beta(
    x_fit: np.ndarray, y_fit: np.ndarray, x_inner: np.ndarray, y_inner: np.ndarray,
    scorer,
) -> tuple[np.ndarray, float, float]:
    best: tuple[float, float, np.ndarray] | None = None
    for ridge in RIDGES:
        beta = _ridge_fit(x_fit, y_fit, ridge)
        value = float(np.mean(scorer(_predict(x_inner, beta), y_inner)))
        if best is None or value < best[0]:
            best = (value, ridge, beta)
    assert best is not None
    return best[2], best[1], best[0]


def _select_reference_dispersion(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_inner: np.ndarray,
    count_inner: np.ndarray,
    exposure_inner: np.ndarray,
) -> float:
    """Choose one q-only NB dispersion and freeze it across all state arms."""

    best: tuple[float, float] | None = None
    for ridge in RIDGES:
        beta = _ridge_fit(x_fit, y_fit, ridge)
        prediction = _predict(x_inner, beta)[:, 0]
        for dispersion in COUNT_DISPERSIONS:
            value = float(np.mean(_negative_binomial_nll(
                prediction, count_inner, exposure_inner, dispersion,
            )))
            if best is None or value < best[0]:
                best = (value, float(dispersion))
    assert best is not None
    return best[1]


def evaluate_frozen_shared(
    data: FullMarkData,
    trajectory: Path,
    *,
    family: str,
    out_dir: Path,
    grammar_targets: GrammarBlockTargets | None = None,
    target_family: str | None = None,
    extra_trajectory: Path | None = None,
) -> dict[str, Any]:
    """Fit tiny horizon heads, with no gradient or selection path to producer."""

    if family not in {"S_N", "S_G"}:
        raise ValueError("family must be S_N or S_G")
    target_family = family if target_family is None else target_family
    if target_family not in {"S_N", "S_G"}:
        raise ValueError("target_family must be S_N or S_G")
    if target_family == "S_G" and grammar_targets is None:
        raise ValueError("S_G target evaluator requires frozen grammar targets")
    state, state_valid = grid_state_from_trajectory(data, trajectory)
    trajectories = [Path(trajectory)]
    if extra_trajectory is not None:
        extra, extra_valid = grid_state_from_trajectory(data, extra_trajectory)
        state = np.column_stack((state, extra))
        state_valid &= extra_valid
        trajectories.append(Path(extra_trajectory))
    fit_all = np.flatnonzero((data.grid_phase == "FIT") & state_valid)
    if fit_all.size == 0:
        raise ValueError("no FIT state anchors")
    state_z, state_mean, state_scale = _standardise(state[fit_all], state)
    out: dict[str, Any] = {
        "format": f"{FORMAT_PREFIX}_frozen_shared_evaluator_v1",
        "family": family,
        "source_family": family if extra_trajectory is None else "S_N+S_G",
        "target_family": target_family,
        "subject": data.subject,
        "producer_trajectory": str(trajectory),
        "producer_sha256": _sha256(trajectory),
        "producer_trajectories": [str(path) for path in trajectories],
        "producer_sha256_all": [_sha256(path) for path in trajectories],
        "producer_frozen": True,
        "horizons": {},
        "selection_targets_read": True,
        "development_targets_read": False,
        "sealed_partition_opened": False,
        "seizure_outcomes_read": False,
        "contract": "horizon-specific ridge evaluators read one shared frozen trajectory",
        "ridge_contract": "penalty defined on mean-loss scale and invariant to duplicate anchor rows",
        "future_window_seizure_policy": "primary evaluator excludes blocks crossing a known seizure",
    }
    saved: dict[str, np.ndarray] = {"state_fit_mean": state_mean, "state_fit_scale": state_scale}
    for j, horizon in enumerate(data.physical_horizons_seconds):
        valid = state_valid & data.future_valid[:, j] & (data.future_seizure_count[:, j] == 0)
        fit = np.flatnonzero((data.grid_phase == "FIT") & valid)
        inner = np.flatnonzero((data.grid_phase == "INNER") & valid)
        selection = np.flatnonzero((data.grid_phase == "SELECTION") & valid)
        record: dict[str, Any] = {
            "horizon_seconds": float(horizon), "n_fit": int(fit.size),
            "n_inner": int(inner.size), "n_selection": int(selection.size), "endpoints": {},
            "n_independent_fit_windows": _non_overlapping_window_count(data.grid_time[fit], horizon),
            "n_independent_inner_windows": _non_overlapping_window_count(data.grid_time[inner], horizon),
            "n_independent_selection_windows": _non_overlapping_window_count(
                data.grid_time[selection], horizon
            ),
            "independence_contract": "greedy non-overlapping wall-clock target windows",
        }
        if min(fit.size, inner.size, selection.size) == 0:
            record["status"] = "NOT_ESTIMABLE"
            out["horizons"][str(int(horizon))] = record
            continue
        record["status"] = "ESTIMATED"
        q = np.asarray(data.grid_q, dtype=np.float64)
        x_const = np.zeros((q.shape[0], 0), dtype=np.float64)
        x_q = q
        x_qs = np.column_stack((q, state_z))
        shift_rows, donor_rows = _shift_donors(data.grid_time, selection, float(horizon))

        endpoint_specs = []
        if target_family == "S_N":
            exposure = np.exp(data.future_count_log_offset[:, j]) * float(horizon)
            target = np.log((data.future_count[:, j] + 0.5) / np.maximum(exposure, 1.0) * 3600.0)
            count = np.asarray(data.future_count[:, j], dtype=np.float64)
            endpoint_specs.append(("burden", target[:, None], "count", count, exposure))
        else:
            assert grammar_targets is not None
            endpoint_specs.extend([
                ("community_occupancy", grammar_targets.community_occupancy[:, j], "simplex", None, None),
                ("cross_community_coupling", grammar_targets.cross_community_coupling[:, j], "simplex", None, None),
                ("repertoire_mixture", grammar_targets.repertoire_mixture[:, j], "simplex", None, None),
                ("repertoire_embedding", grammar_targets.repertoire_embedding_mean[:, j], "embedding", None, None),
            ])
        for name, target, kind, count, exposure in endpoint_specs:
            endpoint_valid = np.isfinite(target).all(axis=1)
            if target_family == "S_G":
                endpoint_valid &= {
                    "community_occupancy": grammar_targets.community_valid[:, j],
                    "cross_community_coupling": grammar_targets.coupling_valid[:, j],
                    "repertoire_mixture": grammar_targets.repertoire_valid[:, j],
                    "repertoire_embedding": grammar_targets.repertoire_embedding_valid[:, j],
                }[name]
            fr, ir, sr = (rows[endpoint_valid[rows]] for rows in (fit, inner, selection))
            if min(fr.size, ir.size, sr.size) == 0:
                record["endpoints"][name] = {"status": "NOT_ESTIMABLE"}
                continue
            if kind == "count":
                dispersion = _select_reference_dispersion(
                    x_q[fr], target[fr], x_q[ir], count[ir], exposure[ir],
                )
                def score(pred, _target, rows=None):
                    assert rows is not None
                    return _negative_binomial_nll(
                        pred[:, 0], count[rows], exposure[rows], dispersion,
                    )
                scorer_fit = lambda pred, _y: score(pred, _y, ir)
                score_rows = lambda pred, rows: _negative_binomial_nll(
                    pred[:, 0], count[rows], exposure[rows], dispersion,
                )
            elif kind == "simplex":
                scorer_fit = lambda pred, y: _distribution_score(pred, y)
                score_rows = lambda pred, rows: _distribution_score(pred, target[rows])
            else:
                scorer_fit = lambda pred, y: np.mean((pred - y) ** 2, axis=1)
                score_rows = lambda pred, rows: np.mean((pred - target[rows]) ** 2, axis=1)
            betas = {}
            meta = {}
            for arm, x in (("constant", x_const), ("q_only", x_q), ("q_plus_state", x_qs)):
                beta, ridge, inner_score = _select_beta(x[fr], target[fr], x[ir], target[ir], scorer_fit)
                betas[arm] = beta
                meta[arm] = {"ridge": ridge, "inner_score": inner_score,
                             "selection_score": float(np.mean(score_rows(_predict(x[sr], beta), sr)))}
                saved[f"h{int(horizon)}_{name}_{arm}_beta"] = beta
            matched = np.intersect1d(sr, shift_rows, assume_unique=False)
            if matched.size:
                donor_map = {int(a): int(b) for a, b in zip(shift_rows, donor_rows)}
                donors = np.asarray([donor_map[int(row)] for row in matched], dtype=np.int64)
                x_correct = x_qs[matched]
                x_shift = np.column_stack((q[matched], state_z[donors]))
                beta = betas["q_plus_state"]
                correct = float(np.mean(score_rows(_predict(x_correct, beta), matched)))
                shifted = float(np.mean(score_rows(_predict(x_shift, beta), matched)))
            else:
                correct = shifted = None
            meta["contrasts"] = {
                "state_gain_over_q": meta["q_only"]["selection_score"] - meta["q_plus_state"]["selection_score"],
                "state_gain_over_constant": meta["constant"]["selection_score"] - meta["q_plus_state"]["selection_score"],
                "correct_time_gain_over_block_shift": None if correct is None else shifted - correct,
                "n_block_shift_support": int(matched.size),
                "n_independent_block_shift_windows": _non_overlapping_window_count(
                    data.grid_time[matched], horizon
                ),
            }
            if kind == "count":
                meta["count_likelihood"] = {
                    "family": "negative_binomial",
                    "dispersion": float(dispersion),
                    "selection": "q-only INNER; frozen across constant/q/state/shift arms",
                }
            meta["status"] = "ESTIMATED"
            record["endpoints"][name] = meta
        out["horizons"][str(int(horizon))] = record
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "frozen_evaluator_heads.npz", **saved)
    atomic_json(out_dir / "card.json", out)
    return out
