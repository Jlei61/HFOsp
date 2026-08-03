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
                  and row.get("ied_mean_a_x", -np.inf) > 0.9
                  and row.get("crossing_time_ms") is not None
                  and 1000.0 <= float(row["crossing_time_ms"]) <= 3000.0
                  and row.get("high_returned_to_low")]
    admissible.sort(key=lambda row: (
        abs(float(row["crossing_time_ms"]) - 2000.0),
        -float(row["ied_mean_a_x"]), str(row["candidate_id"])))
    return admissible[:int(max_candidates)]
