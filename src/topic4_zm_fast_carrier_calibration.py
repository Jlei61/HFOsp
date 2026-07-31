"""Pure baseline-only calibration contract for Phase-D conductance arms."""
from __future__ import annotations

from dataclasses import asdict
import itertools
from typing import Any, Iterable, Mapping

import numpy as np

from src.snn_engine.zm_conductance import distribution_magnitude_anchor


CALIBRATION_SCHEMA = "zm_fast_carrier_calibration_v1_2026-07-31"
SCALE_VALUES = (0.8, 1.0, 1.2)


class CalibrationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationError(message)


def scale_lattice() -> tuple[tuple[float, float, float], ...]:
    """The preregistered 3^3 lattice in deterministic lexical order."""
    return tuple(itertools.product(SCALE_VALUES, repeat=3))


def build_reference_anchor(
    state: Mapping[str, Any],
    *,
    n_e: int,
    v_th_median: float,
    v_reset: float,
    eta_m: float,
) -> dict:
    """Measure the locked free-E voltage distribution and anchor scales."""
    voltage = np.asarray(state.get("V"))
    refractory = np.asarray(state.get("ref"))
    _require(voltage.ndim == 1, "state V must be one-dimensional")
    _require(refractory.shape == voltage.shape, "state ref/V shape mismatch")
    _require(0 < int(n_e) <= voltage.size, "invalid E population size")
    free = refractory[:n_e] == 0
    _require(np.count_nonzero(free) >= 2, "fewer than two free E cells")
    cfg, diagnostics = distribution_magnitude_anchor(
        V_free=voltage[:n_e][free],
        V_th_median=v_th_median,
        V_reset=v_reset,
        eta_m=eta_m,
    )
    return {
        "schema": CALIBRATION_SCHEMA,
        "source": "locked_pre_entry_free_E_snapshot",
        "n_e": int(n_e),
        "n_free_e": int(np.count_nonzero(free)),
        "base_config": asdict(cfg),
        "diagnostics": diagnostics,
        "scale_lattice": [list(row) for row in scale_lattice()],
        "candidate_outcomes_accessed": False,
    }


def candidate_config(reference: Mapping[str, Any], scales: Iterable[float]) -> dict:
    """Apply one literal scale triplet to a locked reference anchor."""
    scales = tuple(float(value) for value in scales)
    _require(scales in scale_lattice(), f"scale triplet is outside lock: {scales}")
    base = dict(reference["base_config"])
    base["kappa_E"] *= scales[0]
    base["kappa_I"] *= scales[1]
    base["g_M"] *= scales[2]
    return base


def adjudicate_row(row: Mapping[str, Any]) -> dict:
    """Apply the six ordered baseline-preservation constraints to one row."""
    required = {
        "scale_E",
        "scale_I",
        "scale_M",
        "data_scope",
        "baseline_reference_sha256",
        "median_e_rate_ratio",
        "returning_event_count_ratio",
        "returning_event_count",
        "event_order_preserved",
        "two_source_geometry_readable",
        "vinf_error_mv",
        "charge_ratio_relative_error",
        "tau_eff_ratio",
        "prevention",
        "whole_sheet_plateau",
    }
    missing = required - set(row)
    _require(not missing, f"calibration row missing fields: {sorted(missing)}")
    scales = tuple(float(row[key]) for key in ("scale_E", "scale_I", "scale_M"))
    _require(scales in scale_lattice(), f"unregistered scale triplet: {scales}")
    _require(row["data_scope"] == "pre_entry_only", "candidate-state data leaked into calibration")
    rate_error = abs(float(row["median_e_rate_ratio"]) - 1.0)
    count_error = abs(float(row["returning_event_count_ratio"]) - 1.0)
    vinf_error = abs(float(row["vinf_error_mv"]))
    charge_error = abs(float(row["charge_ratio_relative_error"]))
    tau_ratio = float(row["tau_eff_ratio"])
    numeric = (rate_error, count_error, vinf_error, charge_error, tau_ratio)
    _require(all(np.isfinite(value) for value in numeric), "non-finite calibration metric")
    reasons = []
    if rate_error > 0.15 or count_error > 0.15:
        reasons.append("rate_or_returning_count_outside_15pct")
    if int(row["returning_event_count"]) <= 0 or bool(row["prevention"]):
        reasons.append("returning_events_prevented")
    if not bool(row["event_order_preserved"]):
        reasons.append("event_order_lost")
    if not bool(row["two_source_geometry_readable"]):
        reasons.append("two_source_geometry_lost")
    if vinf_error > 0.5:
        reasons.append("vinf_error_over_0p5mV")
    if charge_error > 0.15:
        reasons.append("charge_ratio_outside_15pct")
    if not 0.25 <= tau_ratio <= 1.0:
        reasons.append("tau_eff_ratio_outside_0p25_1p0")
    if bool(row["whole_sheet_plateau"]):
        reasons.append("baseline_whole_sheet_plateau")
    distance = float(np.linalg.norm(np.asarray(scales) - 1.0))
    objective = (
        max(rate_error, count_error),
        0.0 if bool(row["event_order_preserved"]) and bool(row["two_source_geometry_readable"]) else 1.0,
        vinf_error,
        charge_error,
        abs(tau_ratio - 1.0),
        0.0 if not bool(row["prevention"]) and not bool(row["whole_sheet_plateau"]) else 1.0,
        distance,
        *scales,
    )
    return {
        "scales": list(scales),
        "valid": not reasons,
        "reasons": reasons,
        "objective": list(objective),
    }


def select_calibration(rows: Iterable[Mapping[str, Any]]) -> dict:
    """Fail closed on incomplete evidence and select one valid row."""
    rows = list(rows)
    identities = [
        tuple(float(row[key]) for key in ("scale_E", "scale_I", "scale_M"))
        for row in rows
    ]
    _require(len(rows) == len(scale_lattice()), "calibration lattice is incomplete")
    _require(len(set(identities)) == len(identities), "duplicate calibration scale triplet")
    _require(set(identities) == set(scale_lattice()), "calibration lattice identity drift")
    reference_hashes = {row.get("baseline_reference_sha256") for row in rows}
    _require(len(reference_hashes) == 1 and None not in reference_hashes, "baseline reference drift")
    adjudicated = [adjudicate_row(row) for row in rows]
    valid = [item for item in adjudicated if item["valid"]]
    if not valid:
        return {
            "schema": CALIBRATION_SCHEMA,
            "verdict": "NO_GO_baseline_calibration_failed",
            "selected_scales": None,
            "n_valid": 0,
            "rows": adjudicated,
        }
    selected = min(valid, key=lambda item: tuple(item["objective"]))
    return {
        "schema": CALIBRATION_SCHEMA,
        "verdict": "baseline_calibration_passed",
        "selected_scales": selected["scales"],
        "selected_objective": selected["objective"],
        "n_valid": len(valid),
        "rows": adjudicated,
    }


__all__ = [
    "CALIBRATION_SCHEMA",
    "CalibrationError",
    "adjudicate_row",
    "build_reference_anchor",
    "candidate_config",
    "scale_lattice",
    "select_calibration",
]
