"""Pure adjudication contracts for conditional LC3 X calibration."""
from __future__ import annotations

import numpy as np


SCHEMA_VERSION = "fcxr-lc3-xcal-1.0"
HIGH_LABELS = {"FINITE_HIGH_FIXED", "FINITE_HIGH_ORBIT"}
RETURN_LABEL = "INTERICTAL_WORKPOINT"


def return_brackets(geometry_rows, d_means):
    """Extract empirical high-start return/survival brackets at each D field."""

    out = []
    labels = sorted({row["d_label"] for row in geometry_rows
                     if row.get("point_id") == "H1_ts1.25_r025"
                     and row.get("state_kind") == "high"})
    for label in labels:
        rows = [row for row in geometry_rows
                if row.get("point_id") == "H1_ts1.25_r025"
                and row.get("state_kind") == "high" and row.get("d_label") == label]
        returns = sorted(float(row["a_x"]) for row in rows
                         if row.get("resolved_label") == RETURN_LABEL)
        survives = sorted(float(row["a_x"]) for row in rows
                          if row.get("resolved_label") in HIGH_LABELS)
        if not returns or not survives:
            continue
        a_return = max(returns)
        above = [value for value in survives if value > a_return]
        if not above:
            continue
        a_survive = min(above)
        out.append(dict(
            d_label=label, mean_D=float(d_means[label]),
            a_return_max=a_return, a_survive_min=a_survive,
            a_off_midpoint=0.5 * (a_return + a_survive),
            bracket_width=a_survive - a_return,
        ))
    return sorted(out, key=lambda row: row["mean_D"])


def relay_x_inf(y, *, y_gate: float, K_y: float, hill_n: int, x_min: float):
    y = np.asarray(y, float)
    u = np.maximum(y - float(y_gate), 0.0)
    un = u ** int(hill_n)
    hill = un / (float(K_y) ** int(hill_n) + un)
    return 1.0 - (1.0 - float(x_min)) * hill


def choose_calibration_family(*, observed_x, inferred_x_inf,
                              a_return_max: float, a_survive_min: float):
    """Route exactly one two-knob family from measured reach versus speed."""

    observed_x = float(observed_x)
    inferred_x_inf = float(inferred_x_inf)
    boundary = float(a_survive_min)
    if not all(np.isfinite(v) for v in (observed_x, inferred_x_inf, boundary, a_return_max)):
        raise ValueError("X adjudication inputs must be finite")
    if inferred_x_inf > boundary:
        return "SENSOR_GATE_AND_HILL_MIDPOINT"
    if observed_x > boundary:
        return "HILL_MIDPOINT_AND_RISE_TIME"
    return "BOUNDARY_ALREADY_REACHED_NO_RECALIBRATION_NEEDED"


def select_x_candidates(rows, *, max_candidates: int = 2):
    """Select at most two candidates using only pre-registered low/high gates."""

    admissible = [row for row in rows
                  if row.get("numerical_safe")
                  and row.get("low_label") == RETURN_LABEL
                  and int(row.get("n_low_returning_events", 0)) >= 3
                  and row.get("ied_mean_a_x", -np.inf) > 0.9
                  and row.get("crossing_time_ms") is not None
                  and 1000.0 <= float(row["crossing_time_ms"]) <= 3000.0
                  and row.get("high_returned_to_low")]
    admissible.sort(key=lambda row: (
        abs(float(row["crossing_time_ms"]) - 2000.0),
        -float(row["ied_mean_a_x"]), str(row["candidate_id"])))
    return admissible[:int(max_candidates)]


def multivariate_statistical_return(pre, post):
    """Require the recovered events to re-enter the full pre-onset neighbourhood."""

    if pre is None or post is None or pre.get("n_events", 0) < 3 or post.get("n_events", 0) < 3:
        return dict(pass_=False, reason="fewer_than_3_returning_events_in_pre_or_post")
    ratios = {}
    for key in ("event_rate_hz", "median_iei_ms", "median_duration_ms",
                "median_participation", "median_compactness_mm"):
        if pre.get(key) is None or post.get(key) is None or float(pre[key]) <= 0:
            return dict(pass_=False, reason=f"undefined_{key}")
        ratios[key] = float(post[key] / pre[key])
    polarity_diff = abs(float(post["fraction_A"]) - float(pre["fraction_A"]))
    passed = bool(
        0.5 <= ratios["event_rate_hz"] <= 2.0
        and 0.5 <= ratios["median_iei_ms"] <= 2.0
        and 0.5 <= ratios["median_duration_ms"] <= 2.0
        and 0.5 <= ratios["median_participation"] <= 2.0
        and (2.0 / 3.0) <= ratios["median_compactness_mm"] <= 1.5
        and polarity_diff <= 0.34)
    return dict(pass_=passed, ratios=ratios, forward_fraction_abs_diff=polarity_diff)


def lifecycle_candidate_gate(*, lifecycle_label: str, onset_ms, high_duration_ms,
                             x_activates_after_onset: bool,
                             postictal_suppression: bool, statistical_return: dict,
                             numerical_unsafe: bool,
                             refractory_ceiling_fraction: float):
    clauses = dict(
        recovered_label=lifecycle_label == "RECOVERED_INTERICTAL",
        pre_onset_at_least_8s=onset_ms is not None and float(onset_ms) >= 8000.0,
        bounded_high_1_to_5s=(high_duration_ms is not None
                             and 1000.0 <= float(high_duration_ms) <= 5000.0),
        x_after_onset=bool(x_activates_after_onset),
        postictal_suppression=bool(postictal_suppression),
        multivariate_statistical_return=bool(statistical_return.get("pass_", False)),
        numerical_safe=not bool(numerical_unsafe),
        non_tonic_ceiling=float(refractory_ceiling_fraction) < 0.05,
    )
    return dict(pass_=bool(all(clauses.values())), clauses=clauses)
