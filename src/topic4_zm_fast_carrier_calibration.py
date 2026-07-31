"""Pure baseline-only calibration contract for Phase-D conductance arms."""
from __future__ import annotations

from dataclasses import asdict
import itertools
from typing import Any, Iterable, Mapping

import numpy as np

from src.snn_engine.zm_conductance import ZMConductanceConfig


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
    alpha_G_reference: float = 16.0,
    gamma: float = 1.0 / 6.0,
) -> dict:
    """Match Arm-A effective E/I/M charge on the locked free-E snapshot."""
    voltage = np.asarray(state.get("V"))
    refractory = np.asarray(state.get("ref"))
    _require(voltage.ndim == 1, "state V must be one-dimensional")
    _require(refractory.shape == voltage.shape, "state ref/V shape mismatch")
    _require(0 < int(n_e) <= voltage.size, "invalid E population size")
    free = refractory[:n_e] == 0
    _require(np.count_nonzero(free) >= 2, "fewer than two free E cells")
    required = ("I_E", "I_E_rec", "I_I", "slow.z", "slow.m", "slow.S_G")
    _require(all(key in state for key in required), "state lacks effective-charge anchor fields")
    arrays = {
        key: np.asarray(state[key])[:n_e][free]
        for key in ("I_E", "I_E_rec", "I_I", "slow.z", "slow.m")
    }
    _require(
        all(value.shape == (int(np.count_nonzero(free)),) for value in arrays.values()),
        "effective-charge anchor array shape mismatch",
    )
    V = voltage[:n_e][free].astype(float, copy=False)
    I_E = arrays["I_E"].astype(float, copy=False)
    I_E_rec = arrays["I_E_rec"].astype(float, copy=False)
    I_I = arrays["I_I"].astype(float, copy=False)
    z = arrays["slow.z"].astype(float, copy=False)
    m = arrays["slow.m"].astype(float, copy=False)
    S_G = float(np.asarray(state["slow.S_G"]))
    _require(np.isfinite(S_G) and S_G >= 0.0, "invalid locked S_G")
    _require(0.0 <= gamma <= 1.0, "gamma must lie in [0,1]")
    E_E = 2.0 * float(v_th_median) - float(v_reset)
    E_I = float(v_reset)
    E_K = 0.0
    old_fraction = alpha_G_reference * S_G / (1.0 + alpha_G_reference * S_G)
    effective_E = I_E - old_fraction * I_E_rec
    mixed_I = (1.0 - gamma) * I_I + gamma * float(np.mean(I_I))
    target_E = float(np.mean(np.abs(effective_E)))
    target_I = float(np.mean(np.abs(z * I_I)))
    target_M = float(np.mean(np.abs(float(eta_m) * m)))
    denom_E = float(np.mean(np.abs(I_E * (E_E - V))))
    denom_I = float(np.mean(np.abs(z * mixed_I * (E_I - V))))
    denom_M = float(np.mean(np.abs(m * (E_K - V))))
    _require(min(denom_E, denom_I, denom_M) > 0.0, "zero effective-charge denominator")
    cfg = ZMConductanceConfig(
        kappa_E=target_E / denom_E,
        kappa_I=target_I / denom_I,
        g_M=target_M / denom_M,
        gamma=float(gamma),
        z_spares_global=False,
        g_L=1.0,
        E_L=0.0,
        E_E=E_E,
        E_I=E_I,
        E_K=E_K,
        tau_m_E=20.0,
    ).validate()
    diagnostics = {
        "definition": "locked_free_E_effective_charge_weighted_anchor",
        "n_free_e": int(np.count_nonzero(free)),
        "V_free_percentiles_mv": {
            str(q): float(np.percentile(V, q)) for q in (5, 25, 50, 75, 95)
        },
        "fraction_V_above_EI": float(np.mean(V > E_I)),
        "fraction_V_above_EK": float(np.mean(V > E_K)),
        "signed_point_tangent_feasible_at_median": bool(float(np.median(V)) > E_I),
        "pointwise_sign_equivalence_claimed": False,
        "old_SG": S_G,
        "old_SG_divisive_fraction": float(old_fraction),
        "gamma_primary": float(gamma),
        "target_mean_abs_charge": {"E": target_E, "I": target_I, "M": target_M},
        "anchor_mean_abs_charge": {
            "E": float(cfg.kappa_E * denom_E),
            "I": float(cfg.kappa_I * denom_I),
            "M": float(cfg.g_M * denom_M),
        },
        "central_anchor_exact_at_source_snapshot": bool(
            np.allclose(
                [cfg.kappa_E * denom_E, cfg.kappa_I * denom_I, cfg.g_M * denom_M],
                [target_E, target_I, target_M],
                rtol=1e-12,
                atol=1e-12,
            )
        ),
    }
    return {
        "schema": CALIBRATION_SCHEMA,
        "source": "locked_pre_entry_free_E_effective_charge_snapshot",
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
    _require(
        row["data_scope"] == "dynamic_preentry_t0_to_8500ms_only",
        "candidate-state data leaked into calibration",
    )
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
