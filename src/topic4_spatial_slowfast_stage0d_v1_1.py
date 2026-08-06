"""Stage 0D v1.1 engineering-repair helpers.

No model or scientific-screen term is introduced here.  This module enforces
the written Stage 0D confirm-vs-dt/2 frequency tolerance that v1 implemented too
loosely and provides deterministic v1-v1.1 comparison helpers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from src.topic4_spatial_slowfast_stage0c_transfer import temporal_refinement_status


FREQUENCY_ABS_HZ: float = 0.25
FREQUENCY_RELATIVE: float = 0.10
RATE_ABS_HZ: float = 1.0
RATE_RELATIVE: float = 0.10
AMPLITUDE_ABS_HZ: float = 5.0
AMPLITUDE_RELATIVE: float = 0.10
FIGURE_B_EMPTY_TEXT: str = "none passed locked gate"


def strict_temporal_amplitude_status(
    confirm_row: Mapping[str, Any],
    refined_row: Mapping[str, Any],
    *,
    exact_error_pass: bool,
) -> tuple[str, dict[str, float | bool]]:
    """Apply the Stage 0D written rate/frequency/amplitude dt/2 contract.

    The older helper is retained as a fail-closed first screen for state/support,
    class, rate, and its looser frequency check.  A candidate then *always*
    passes through the explicit 0.25-Hz/10-percent frequency gate below.
    """

    legacy_status = temporal_refinement_status(
        confirm_row,
        refined_row,
        exact_error_pass=exact_error_pass,
    )
    rates = np.asarray([confirm_row.get("tail_mean_hz"), refined_row.get("tail_mean_hz")], dtype=float)
    frequencies = np.asarray(
        [confirm_row.get("dominant_frequency_hz"), refined_row.get("dominant_frequency_hz")], dtype=float
    )
    amplitudes = np.asarray(
        [
            float(confirm_row.get("tail_peak_hz", np.nan)) - float(confirm_row.get("tail_trough_hz", np.nan)),
            float(refined_row.get("tail_peak_hz", np.nan)) - float(refined_row.get("tail_trough_hz", np.nan)),
        ],
        dtype=float,
    )
    rate_difference = float(np.ptp(rates)) if np.all(np.isfinite(rates)) else np.inf
    frequency_difference = float(np.ptp(frequencies)) if np.all(np.isfinite(frequencies)) else np.inf
    amplitude_difference = float(np.ptp(amplitudes)) if np.all(np.isfinite(amplitudes)) else np.inf
    rate_limit = max(RATE_ABS_HZ, RATE_RELATIVE * float(np.mean(rates))) if np.all(np.isfinite(rates)) else np.nan
    frequency_limit = (
        max(FREQUENCY_ABS_HZ, FREQUENCY_RELATIVE * float(np.mean(frequencies)))
        if np.all(np.isfinite(frequencies))
        else np.nan
    )
    amplitude_limit = (
        max(AMPLITUDE_ABS_HZ, AMPLITUDE_RELATIVE * float(np.mean(amplitudes)))
        if np.all(np.isfinite(amplitudes))
        else np.nan
    )
    rate_pass = bool(np.isfinite(rate_limit) and rate_difference <= rate_limit)
    frequency_pass = bool(np.isfinite(frequency_limit) and frequency_difference <= frequency_limit)
    amplitude_pass = bool(np.isfinite(amplitude_limit) and amplitude_difference <= amplitude_limit)
    status = (
        "candidate_survives"
        if legacy_status == "candidate_survives" and rate_pass and frequency_pass and amplitude_pass
        else legacy_status
    )
    if legacy_status == "candidate_survives" and not (rate_pass and frequency_pass and amplitude_pass):
        status = "numerical_unresolved"
    audit: dict[str, float | bool] = {
        "legacy_temporal_gate_candidate": legacy_status == "candidate_survives",
        "rate_difference_hz": rate_difference,
        "rate_limit_hz": float(rate_limit),
        "rate_pass": rate_pass,
        "frequency_difference_hz": frequency_difference,
        "frequency_limit_hz": float(frequency_limit),
        "frequency_abs_floor_hz": FREQUENCY_ABS_HZ,
        "frequency_relative_fraction": FREQUENCY_RELATIVE,
        "frequency_pass": frequency_pass,
        "amplitude_difference_hz": amplitude_difference,
        "amplitude_limit_hz": float(amplitude_limit),
        "amplitude_pass": amplitude_pass,
        "strict_gate_pass": status == "candidate_survives",
    }
    return status, audit


def centre_final_survivor_indices(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    """Return only final survivors at the frozen centre point."""

    return [
        index
        for index, row in enumerate(rows)
        if np.isclose(float(row["z"]), 0.85)
        and np.isclose(float(row["alpha_G"]), 16.0)
        and row.get("final_status") == "candidate_survives"
        and row.get("dt_half_classification") is not None
    ]


def compare_fork_outcomes(
    v1_rows: Sequence[Mapping[str, Any]],
    repaired_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare aligned histories and report every repaired status change."""

    def key(row: Mapping[str, Any]) -> tuple[float, float, str, str]:
        return (
            round(float(row["z"]), 2),
            float(row["alpha_G"]),
            str(row["phase_id"]),
            str(row["history"]),
        )

    old = {key(row): row for row in v1_rows}
    new = {key(row): row for row in repaired_rows}
    if len(old) != 180 or len(new) != 180 or set(old) != set(new):
        raise ValueError("v1 and v1.1 must contain the same 180-history battery")
    changes = []
    for history in sorted(old):
        old_status = str(old[history]["final_status"])
        new_status = str(new[history]["final_status"])
        if old_status != new_status:
            changes.append(
                {
                    "z": history[0],
                    "alpha_G": history[1],
                    "phase_id": history[2],
                    "history": history[3],
                    "v1_final_status": old_status,
                    "v1_1_final_status": new_status,
                }
            )
    return {
        "n_aligned_histories": len(old),
        "n_fork_status_changes": len(changes),
        "fork_status_changes": changes,
        "any_fork_status_changed": bool(changes),
    }
