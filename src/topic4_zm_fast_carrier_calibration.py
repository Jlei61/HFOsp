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


def event_windows(r_core, *, bin_ms: float = 25.0) -> list[tuple[int, int]]:
    """Return the same half-amplitude returning-event windows as the source lock."""
    trace = np.asarray(r_core, float)
    width = max(1, int(round(100.0 / float(bin_ms))))
    smooth = np.convolve(trace, np.ones(width) / width, mode="same")
    if smooth.size < 4:
        return []
    baseline = float(np.percentile(smooth, 20))
    amplitude = float(np.max(smooth) - baseline)
    if amplitude <= 1e-12:
        return []
    on = smooth >= baseline + 0.5 * amplitude
    windows, start = [], None
    for index, active in enumerate(on):
        if active and start is None:
            start = index
        elif not active and start is not None:
            windows.append((start, index))
            start = None
    if start is not None:
        windows.append((start, on.size))
    return windows


def trajectory_signature(arrays: Mapping[str, Any], *, bin_ms: float = 25.0) -> dict:
    """Event order and two-source readability from one saved baseline trace."""
    required = ("r_core", "r_source", "r_sink", "lfp_abs_binned")
    _require(all(key in arrays for key in required), "trajectory signature arrays missing")
    core = np.asarray(arrays["r_core"], float)
    source = np.asarray(arrays["r_source"], float)
    sink = np.asarray(arrays["r_sink"], float)
    lfp = np.asarray(arrays["lfp_abs_binned"], float)
    _require(source.shape == sink.shape == core.shape, "source/sink/core shape mismatch")
    _require(lfp.ndim == 2 and lfp.shape[0] == core.size, "LFP/bin shape mismatch")
    windows = event_windows(core, bin_ms=bin_ms)
    baseline = np.percentile(lfp, 20, axis=0)
    baseline = np.maximum(baseline, np.finfo(float).eps)
    events = []
    for lo, hi in windows:
        src_peak = int(lo + np.argmax(source[lo:hi]))
        sink_peak = int(lo + np.argmax(sink[lo:hi]))
        contact_peaks = lo + np.argmax(lfp[lo:hi], axis=0)
        order = np.argsort(np.argsort(contact_peaks, kind="stable"), kind="stable")
        gain = np.mean(lfp[lo:hi], axis=0) / baseline
        events.append(
            {
                "lo_bin": int(lo),
                "hi_bin": int(hi),
                "direction": int(np.sign(sink_peak - src_peak)),
                "contact_order": order.astype(int).tolist(),
                "source_peak_hz": float(np.max(source[lo:hi])),
                "sink_peak_hz": float(np.max(sink[lo:hi])),
                "median_contact_gain": float(np.median(gain)),
            }
        )
    return {"n_events": len(events), "events": events, "bin_ms": float(bin_ms)}


def compare_trajectory_signatures(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict:
    """Paired event-order and two-source gates without amplitude-unit matching."""
    ref_events = list(reference.get("events", []))
    can_events = list(candidate.get("events", []))
    n = min(len(ref_events), len(can_events))
    if n == 0:
        return {
            "event_order_preserved": False,
            "two_source_geometry_readable": False,
            "median_contact_order_correlation": None,
            "direction_agreement": None,
            "n_paired_events": 0,
        }
    correlations, directions = [], []
    for ref, can in zip(ref_events[:n], can_events[:n], strict=True):
        left = np.asarray(ref["contact_order"], float)
        right = np.asarray(can["contact_order"], float)
        _require(left.shape == right.shape and left.size >= 2, "contact-order shape drift")
        corr = float(np.corrcoef(left, right)[0, 1])
        correlations.append(corr if np.isfinite(corr) else 0.0)
        directions.append(int(ref["direction"]) == int(can["direction"]))
    ref_src = np.median([event["source_peak_hz"] for event in ref_events[:n]])
    ref_sink = np.median([event["sink_peak_hz"] for event in ref_events[:n]])
    can_src = np.median([event["source_peak_hz"] for event in can_events[:n]])
    can_sink = np.median([event["sink_peak_hz"] for event in can_events[:n]])
    can_gain = np.median([event["median_contact_gain"] for event in can_events[:n]])
    median_corr = float(np.median(correlations))
    direction_agreement = float(np.mean(directions))
    return {
        "event_order_preserved": bool(median_corr >= 0.5 and direction_agreement >= 0.5),
        "two_source_geometry_readable": bool(
            can_gain >= 1.5
            and can_src >= 0.5 * max(ref_src, 1e-12)
            and can_sink >= 0.5 * max(ref_sink, 1e-12)
        ),
        "median_contact_order_correlation": median_corr,
        "direction_agreement": direction_agreement,
        "candidate_median_contact_gain": float(can_gain),
        "source_peak_ratio": float(can_src / max(ref_src, 1e-12)),
        "sink_peak_ratio": float(can_sink / max(ref_sink, 1e-12)),
        "n_paired_events": int(n),
    }


def whole_sheet_plateau(arrays: Mapping[str, Any], *, bin_ms: float = 25.0) -> bool:
    """Baseline failure: broad high-rate activity sustained for 500 ms."""
    rate = np.asarray(arrays["r_all"], float)
    active = np.asarray(arrays["active_fraction"], float)
    _require(rate.shape == active.shape, "rate/active-fraction shape mismatch")
    above = (active >= 0.5) | ((rate >= 100.0) & (active >= 0.2))
    need = max(1, int(round(500.0 / float(bin_ms))))
    run = 0
    for flag in above:
        run = run + 1 if flag else 0
        if run >= need:
            return True
    return False


def build_calibration_row(
    reference_receipt: Mapping[str, Any],
    reference_arrays: Mapping[str, Any],
    candidate_receipt: Mapping[str, Any],
    candidate_arrays: Mapping[str, Any],
) -> dict:
    """Convert one paired dynamic-pre-entry run into the selector schema."""
    _require(reference_receipt["mode"] == "reference", "reference receipt mode drift")
    _require(candidate_receipt["mode"] == "cell", "candidate receipt mode drift")
    _require(
        reference_receipt["external_drive_sha256"]
        == candidate_receipt["external_drive_sha256"],
        "paired external drive mismatch",
    )
    ref_events = reference_receipt["returning_events"]
    can_events = candidate_receipt["returning_events"]
    _require(
        int(ref_events["n_events"]) >= 10
        and float(ref_events["median_peak_hz"]) >= 20.0,
        "reference lacks source-scale returning IEDs",
    )
    signatures = compare_trajectory_signatures(
        trajectory_signature(reference_arrays), trajectory_signature(candidate_arrays)
    )
    ref_diag = reference_receipt["diagnostics"]
    can_diag = candidate_receipt["diagnostics"]
    ref_rate = float(reference_receipt["median_e_rate_hz"])
    ref_count = int(ref_events["n_events"])
    ref_charge = float(ref_diag["effective_inhibitory_to_excitatory_charge_ratio"])
    scales = candidate_receipt["scales"]
    return {
        "scale_E": float(scales[0]),
        "scale_I": float(scales[1]),
        "scale_M": float(scales[2]),
        "data_scope": "dynamic_preentry_t0_to_8500ms_only",
        "baseline_reference_sha256": reference_receipt["manifest_sha256"],
        "median_e_rate_ratio": float(candidate_receipt["median_e_rate_hz"]) / max(ref_rate, 1e-12),
        "returning_event_count_ratio": int(can_events["n_events"]) / max(ref_count, 1),
        "returning_event_count": int(can_events["n_events"]),
        "event_order_preserved": signatures["event_order_preserved"],
        "two_source_geometry_readable": signatures["two_source_geometry_readable"],
        "vinf_error_mv": float(can_diag["median_vinf_mv"] - ref_diag["median_vinf_mv"]),
        "charge_ratio_relative_error": float(
            can_diag["effective_inhibitory_to_excitatory_charge_ratio"] / max(ref_charge, 1e-12) - 1.0
        ),
        "tau_eff_ratio": float(can_diag["median_tau_eff_ms"] / 20.0),
        "prevention": int(can_events["n_events"]) == 0,
        "whole_sheet_plateau": whole_sheet_plateau(candidate_arrays),
        "trajectory_comparison": signatures,
        "runaway_early_stop_ms": candidate_receipt["runaway_early_stop_ms"],
    }


def zero_spike_dominance_stop(rows: Iterable[Mapping[str, Any]]) -> dict:
    """Cheap proof that no lattice cell can emit its first E spike.

    Before the first E spike, ``m=0`` and the I-population/raw-z trajectory is
    identical for fixed ``scale_I``.  The maximum registered excitatory scale
    therefore upper-bounds first-spike reachability; all three inhibitory
    scales are enumerated and the M scale is irrelevant while m is zero.
    """
    rows = list(rows)
    expected = {(1.2, scale_I, 1.0) for scale_I in SCALE_VALUES}
    identities = {
        (float(row["scale_E"]), float(row["scale_I"]), float(row["scale_M"]))
        for row in rows
    }
    _require(identities == expected and len(rows) == 3, "dominance panel identity drift")
    _require(
        len({row["external_drive_sha256"] for row in rows}) == 1,
        "dominance panel future-noise mismatch",
    )
    zero = all(
        int(row["total_e_spikes"]) == 0
        and int(row["returning_event_count"]) == 0
        and float(row["peak_active_fraction"]) == 0.0
        and row["runaway_early_stop_ms"] is None
        for row in rows
    )
    return {
        "stop": bool(zero),
        "verdict": (
            "NO_GO_baseline_calibration_failed_zero_spike_dominance"
            if zero
            else "continue_full_lattice"
        ),
        "proof_scope": "first_E_spike_reachability_only",
        "enumerated_scale_I": list(SCALE_VALUES),
        "max_scale_E": 1.2,
        "scale_M_irrelevant_before_first_E_spike": True,
        "n_rows": len(rows),
    }


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
        "runaway_early_stop_ms",
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
    if row["runaway_early_stop_ms"] is not None:
        reasons.append("baseline_runaway")
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
    "build_calibration_row",
    "compare_trajectory_signatures",
    "event_windows",
    "scale_lattice",
    "select_calibration",
    "trajectory_signature",
    "whole_sheet_plateau",
    "zero_spike_dominance_stop",
]
