"""Deterministic CPU canaries for the v0.3.4 H3 measuring instrument."""

from __future__ import annotations

import numpy as np

from .controls import (
    audit_replacement_event_overlap,
    event_window_overlap_fraction,
    rolling_prefix_slow_level,
    selection_period_mean_oracle,
)
from .model_contract import build_feedback_arm_contracts
from .ridge import fit_scale_stable_ridge


def run_synthetic_canary(seed: int = 20260903) -> dict:
    rng = np.random.default_rng(seed)
    n = 600
    t = np.arange(n, dtype=np.float64)
    slow = 0.7 + 0.002 * t + 0.15 * np.sin(t / 80.0)
    burden = slow + rng.normal(0.0, 0.35, n)
    mark = rng.normal(0.0, 1.0, n)
    noise = rng.normal(0.0, 0.08, n)
    tr = slice(0, 400)
    va = slice(400, None)

    def mse(pred, y):
        return float(np.mean((np.asarray(pred) - np.asarray(y)) ** 2))

    # Canary 1: a free constant wins spectacularly over a zero/no-edge arm,
    # yet this is not exposure evidence.
    y_const = 3.0 + noise
    zero_loss = mse(np.zeros(n)[va], y_const[va])
    intercept_loss = mse(np.repeat(y_const[tr].mean(), n)[va], y_const[va])
    c1 = zero_loss > 100.0 * intercept_loss

    # Canary 2: common slow drive should absorb a zero-feedback target.
    y0 = 1.4 * slow + noise
    p0, f0 = fit_scale_stable_ridge(slow[tr, None], y0[tr], slow[va, None], y0[va])
    p1z, f1z = fit_scale_stable_ridge(
        np.c_[slow[tr], burden[tr]], y0[tr], np.c_[slow[va], burden[va]], y0[va]
    )
    c2 = mse(p1z, y0[va]) >= 0.95 * mse(p0, y0[va])

    # Canary 3/4: true burden and true mark edges are recovered on unseen rows.
    y1 = 1.4 * slow + 1.8 * (burden - slow) + noise
    p0b, _ = fit_scale_stable_ridge(slow[tr, None], y1[tr], slow[va, None], y1[va])
    p1, f1 = fit_scale_stable_ridge(
        np.c_[slow[tr], burden[tr] - slow[tr]], y1[tr],
        np.c_[slow[va], burden[va] - slow[va]], y1[va]
    )
    c3 = mse(p1, y1[va]) < 0.2 * mse(p0b, y1[va]) and f1.estimable
    y2 = 1.4 * slow + 1.5 * mark + noise
    pm0, _ = fit_scale_stable_ridge(slow[tr, None], y2[tr], slow[va, None], y2[va])
    pm2, f2 = fit_scale_stable_ridge(
        np.c_[slow[tr], mark[tr]], y2[tr], np.c_[slow[va], mark[va]], y2[va]
    )
    c4 = mse(pm2, y2[va]) < 0.2 * mse(pm0, y2[va]) and f2.estimable

    # Canary 5: ridge prediction and lambda are invariant to column units.
    scaled = np.c_[slow * 1e6, mark * 1e-5]
    base = np.c_[slow, mark]
    pb, fb = fit_scale_stable_ridge(base[tr], y2[tr], base[va], y2[va])
    ps, fs = fit_scale_stable_ridge(scaled[tr], y2[tr], scaled[va], y2[va])
    c5 = np.allclose(pb, ps, atol=1e-9, rtol=1e-9) and fb.selected_lambda == fs.selected_lambda

    # Canary 6: overlap reports the August-26 (N-delay)/N failure exactly.
    c6 = event_window_overlap_fraction(0, 10_000, 1_000, 11_000) == 0.9

    # Canary 7/8: causal control excludes same-time events; period mean is
    # explicitly an oracle rather than a primary comparator.
    levels, causal_audit = rolling_prefix_slow_level(
        np.array([1.0, 2.0, 3.0]), np.zeros(3, dtype=int),
        np.array([1.0, 2.5, 4.0]), np.zeros(3, dtype=int), half_life_seconds=10.0,
    )
    _oracle, definition = selection_period_mean_oracle(levels)
    c7 = causal_audit["causal_at_anchor"] and levels[0, 0] == 0.0 and levels[1, 0] > 0.0
    c8 = (not definition.causal_at_anchor) and (not definition.allowed_primary_comparator)

    replacement = audit_replacement_event_overlap(
        np.array([0, 100]), np.array([100, 200]), np.array([0, 0]),
        np.array([100, 0]), np.array([200, 100]), np.array([0, 1]),
    )
    c9 = replacement["passed"] and replacement["n_pairs_with_overlap"] == 0

    build_feedback_arm_contracts()  # raises unless exact template matching holds
    checks = {
        "free_intercept_failure_detected": c1,
        "zero_feedback_absorbed_by_common_drive": c2,
        "burden_feedback_recovered": c3,
        "mark_feedback_recovered": c4,
        "ridge_unit_scale_invariant": c5,
        "delayed_overlap_detected": c6,
        "rolling_prefix_is_strictly_causal": c7,
        "selection_period_mean_is_noncausal_oracle": c8,
        "state_matched_replacement_is_nonoverlapping": c9,
    }
    return {
        "seed": int(seed),
        "checks": checks,
        "n_passed": int(sum(bool(v) for v in checks.values())),
        "n_total": len(checks),
        "passed": bool(all(checks.values())),
        "diagnostics": {
            "free_intercept_zero_loss": zero_loss,
            "free_intercept_matched_loss": intercept_loss,
            "zero_feedback_m0_loss": mse(p0, y0[va]),
            "zero_feedback_m1_loss": mse(p1z, y0[va]),
            "burden_m0_loss": mse(p0b, y1[va]),
            "burden_m1_loss": mse(p1, y1[va]),
            "mark_m0_loss": mse(pm0, y2[va]),
            "mark_m2_loss": mse(pm2, y2[va]),
        },
    }
