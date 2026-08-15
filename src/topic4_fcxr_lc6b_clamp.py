"""FCXR-LC6B: hold the two slow fields of a live state fixed, and read what the fast subsystem does.

The natural trajectories never separate two explanations of the same escalation.  Either the fast
recurrent loop has no bounded branch at all, or a bounded branch exists and the slow fields --
synaptic wear ``D = 1 - z`` and the local recurrent-drive memory ``H`` -- walk the tissue straight
through it.  In a natural run both are moving, so the two look identical.  Pinning them is the only
way to ask the fast subsystem the question on its own.

``freeze_dynamic_state`` in :mod:`src.topic4_fcxr_lc3_dxprobe` looks close enough to reuse and is
not: it pins ``D`` *and* the presynaptic relay ``X`` at chosen grid values, because its question is a
D/X phase map.  Here ``X`` is already frozen at 1.0 by the LC6A configuration and must not be touched
at all, the pinned value is whatever the source snapshot itself carries, and ``H`` -- which that
helper has no path for -- is half the experiment.  Different question, different helper.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from src.topic4_fcxr_lc3 import clone_loop_state, state_hash

LC6B_CLAMP_SCHEMA = "fcxr-lc6b-slow-clamp-1.0"

#: Registered arm names -> (clamp D, clamp H).  ``NAT`` is the paired control for its own snapshot,
#: not a fresh-t0 natural trajectory.
ARMS = {
    "NAT": (False, False),
    "H_CLAMP": (False, True),
    "D_CLAMP": (True, False),
    "DH_CLAMP": (True, True),
}


def _validated_field(values, ne, name, *, lo=None, hi=None):
    field = np.asarray(values, dtype=float)
    if field.shape != (int(ne),):
        raise ValueError(f"{name} must have shape ({ne},), got {field.shape}")
    if not np.all(np.isfinite(field)):
        raise ValueError(f"{name} must be finite")
    if lo is not None and float(field.min()) < lo:
        raise ValueError(f"{name} must be >= {lo}")
    if hi is not None and float(field.max()) > hi:
        raise ValueError(f"{name} must be <= {hi}")
    return field


def _field_digest(field):
    array = np.ascontiguousarray(np.asarray(field, dtype=float))
    return hashlib.sha256(array.tobytes()).hexdigest()


def apply_slow_clamp(state, *, clamp_d: bool, clamp_h: bool):
    """Clone ``state`` and hold its D and/or H field at exactly the value it already carries.

    Returns ``(child, record)``.  ``record`` carries a digest of the *configuration* change alone, so
    a caller can tell a config difference from a state difference instead of reading one as the other.

    Freezing D uses the engine's existing Stage-D path: ``membrane_terms`` modulates received GABA by
    ``self.z`` whenever ``use_z`` **or** ``z_frozen_E`` is live, while ``step`` writes ``z`` only when
    ``use_z``.  Freezing H uses the LC6B hook, which leaves the whole membrane path -- including the
    ``gH`` the frozen field produces -- byte-identical and skips only the state update.
    """

    child = clone_loop_state(state)
    slow = child.slow
    cfg = slow.cfg
    ne = int(slow.NE)
    frozen = {}

    if clamp_d:
        z = _validated_field(slow.z[:ne], ne, "clamped z field", lo=0.0, hi=1.0).copy()
        cfg.z_frozen_E = z
        cfg.use_z = False           # the engine rejects a frozen field that still evolves
        frozen["z"] = z
    if clamp_h:
        if not getattr(cfg, "use_h_lc2", False):
            raise ValueError("clamp_h requires the LC2 H path to be active")
        h = _validated_field(slow.h_lc2_E, ne, "clamped H field", lo=0.0).copy()
        cfg.h_lc2_frozen_E = h
        frozen["h_lc2_E"] = h

    record = {
        "schema": LC6B_CLAMP_SCHEMA,
        "clamp_d": bool(clamp_d),
        "clamp_h": bool(clamp_h),
        "use_z": bool(cfg.use_z),
        "use_h_lc2": bool(getattr(cfg, "use_h_lc2", False)),
        "frozen_field_sha256": {name: _field_digest(value) for name, value in frozen.items()},
        "frozen_field_stats": {
            name: {
                "mean": float(value.mean()), "min": float(value.min()),
                "max": float(value.max()), "median": float(np.median(value)),
            }
            for name, value in frozen.items()
        },
        "state_hash_after_clamp": state_hash(child),
    }
    record["clamp_config_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "schema": record["schema"], "clamp_d": record["clamp_d"],
                "clamp_h": record["clamp_h"], "use_z": record["use_z"],
                "use_h_lc2": record["use_h_lc2"],
                "frozen_field_sha256": record["frozen_field_sha256"],
            },
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return child, record


def slow_field_constancy(states) -> dict:
    """Bitwise constancy of the two slow fields across an ordered list of captured states."""

    if len(states) < 2:
        raise ValueError("constancy needs at least two states")
    ne = int(states[0].slow.NE)
    out = {}
    for name, getter in (
        ("z", lambda s: np.asarray(s.slow.z[:ne])),
        ("h_lc2_E", lambda s: np.asarray(s.slow.h_lc2_E)),
    ):
        digests = [_field_digest(getter(s)) for s in states]
        out[name] = {
            "bitwise_constant": bool(all(d == digests[0] for d in digests)),
            "n_states": len(digests),
            "sha256_first": digests[0], "sha256_last": digests[-1],
        }
    return out


# ---------------------------------------------------------------------- classifier

#: Registered thresholds, inherited unchanged from LC6A so the two rounds read against the same lines.
GLOBAL_SATURATION_HZ = 250.0
REFRACTORY_CEILING_HZ = 500.0
NEAR_REFRACTORY_FRACTION_GATE = 0.05
INTERICTAL_ROLL_HI_HZ = 9.7382291667
DRIFT_CI_GATE_PER_S = 0.05
SILENCE_BIN_FRACTION_GATE = 0.25
CELL_RATE_BANDS_HZ = (250.0, 300.0, 350.0, 400.0, 450.0)

LABELS = (
    "ESCALATING_SATURATION", "BOUNDED_STATIONARY", "BOUNDED_OSCILLATORY", "LOW_STATE",
    "SILENCE", "AFTER_DISCHARGE", "NUMERICAL_FAIL", "RIGHT_CENSORED",
)


def _longest_run_ms(mask, bin_ms):
    best = run = 0
    for value in np.asarray(mask, bool):
        run = run + 1 if value else 0
        best = max(best, run)
    return float(best) * float(bin_ms)


def classify_clamp_window(
    *, rate_bins_hz, cell_rates_hz, completed_ms, registered_ms,
    numerical_fail: bool = False, bin_ms: float = 20.0,
    global_saturation_hz: float = GLOBAL_SATURATION_HZ,
    refractory_ceiling_hz: float = REFRACTORY_CEILING_HZ,
    near_refractory_fraction_gate: float = NEAR_REFRACTORY_FRACTION_GATE,
    interictal_roll_hi_hz: float = INTERICTAL_ROLL_HI_HZ,
    drift_ci_gate_per_s: float = DRIFT_CI_GATE_PER_S,
    silence_bin_fraction_gate: float = SILENCE_BIN_FRACTION_GATE,
    tail_s: float = 2.0,
) -> dict:
    """Label one clamped continuation window.

    Deliberately takes **no** D or H trace.  Under a clamp those two fields are bitwise constant, so
    a slope criterion on them is satisfied by the intervention itself rather than by the dynamics --
    LC6A's ``classify_high_state`` requires exactly that and must not be reused here.  Every criterion
    below is a property of the spiking readout alone.

    ``rate_bins_hz`` is the population E rate in ``bin_ms`` bins; ``cell_rates_hz`` is
    ``(n_seconds, n_cells)`` per-cell rates.
    """

    from src.topic4_fcxr_lc6_phenotype import normalized_theil_sen

    rate = np.asarray(rate_bins_hz, float)
    cells = np.asarray(cell_rates_hz, float)
    if rate.ndim != 1 or cells.ndim != 2:
        raise ValueError("rate_bins_hz must be 1-D and cell_rates_hz must be 2-D")
    bins_per_s = int(round(1000.0 / float(bin_ms)))
    tail_bins = min(rate.size, int(round(float(tail_s) * bins_per_s)))
    tail = rate[-tail_bins:] if tail_bins else rate

    per_second_mean = cells.mean(axis=1) if cells.size else np.zeros(0)
    near_ceiling = 0.9 * float(refractory_ceiling_hz)
    per_second_near_refractory = (
        (cells >= near_ceiling).mean(axis=1) if cells.size else np.zeros(0)
    )
    global_saturated = bool(
        per_second_mean.size and float(per_second_mean.max()) >= float(global_saturation_hz)
    )
    local_saturated = bool(
        per_second_near_refractory.size
        and float(per_second_near_refractory.max()) >= float(near_refractory_fraction_gate)
    )

    above = rate > float(interictal_roll_hi_hz)
    silent_bins_tail = ~above[-tail_bins:] if tail_bins else ~above
    silence_fraction_tail = float(silent_bins_tail.mean()) if silent_bins_tail.size else 1.0
    half_second_bins = min(rate.size, int(round(0.5 * bins_per_s)))
    fully_silent_tail = bool(half_second_bins and np.all(rate[-half_second_bins:] <= 0.0))

    # Drift is read on 100 ms bins so the gate is the same statistic LC6A applied to its rate trace.
    group = max(1, int(round(100.0 / float(bin_ms))))
    usable = (rate.size // group) * group
    rate_100ms = rate[:usable].reshape(-1, group).mean(axis=1) if usable else rate
    drift = normalized_theil_sen(rate_100ms, dt_s=0.1, tail_s=float(tail_s))
    drift_ok = bool(
        np.isfinite(drift["normalized_ci_high_per_s"])
        and drift["normalized_ci_high_per_s"] <= float(drift_ci_gate_per_s)
    )

    incomplete = float(completed_ms) + 1e-9 < float(registered_ms)
    tail_mean = float(tail.mean()) if tail.size else 0.0
    time_above_ms = float(np.count_nonzero(above)) * float(bin_ms)

    # Evaluation order: numerical > incomplete > saturation > resolved-low > still-escalating >
    # bounded.  Saturation and a fall back to the interictal band are both RESOLVED outcomes, so they
    # are read before the still-escalating censor; otherwise a slowly recovering low state would be
    # censored rather than reported.
    if numerical_fail:
        label, reason = "NUMERICAL_FAIL", "NON_FINITE_OR_ENGINE_NUMERICAL_FAILURE"
    elif incomplete:
        label, reason = "RIGHT_CENSORED", "INCOMPLETE_REGISTERED_WINDOW"
    elif global_saturated or local_saturated:
        label = "ESCALATING_SATURATION"
        reason = "+".join(
            ([] if not global_saturated else ["GLOBAL_1S_MEAN_AT_OR_ABOVE_REGISTERED_LINE"])
            + ([] if not local_saturated else ["LOCAL_NEAR_REFRACTORY_FRACTION_AT_OR_ABOVE_GATE"])
        )
    elif fully_silent_tail:
        label, reason = "SILENCE", "NO_E_SPIKE_IN_FINAL_500_MS"
    elif tail_mean <= float(interictal_roll_hi_hz):
        if time_above_ms < 2000.0:
            label, reason = "AFTER_DISCHARGE", "ELEVATED_FOR_UNDER_2000_MS_THEN_BACK_IN_BAND"
        else:
            label, reason = "LOW_STATE", "FELL_BACK_INTO_THE_INTERICTAL_BAND"
    elif not drift_ok:
        label, reason = "RIGHT_CENSORED", "STILL_ESCALATING_AT_WINDOW_END"
    elif silence_fraction_tail >= float(silence_bin_fraction_gate):
        label, reason = "BOUNDED_OSCILLATORY", "STATIONARY_ENVELOPE_WITH_SUB_BAND_GAPS"
    else:
        label, reason = "BOUNDED_STATIONARY", "STATIONARY_AND_CONTINUOUSLY_ELEVATED"

    return {
        "label": label, "reason": reason,
        "bounded_candidate": label in ("BOUNDED_STATIONARY", "BOUNDED_OSCILLATORY"),
        # A first-round label answers "does this branch exist across this window", never "does a weak
        # perturbation return to the same envelope".  Recorded unconditionally so the distinction
        # cannot be lost downstream.
        "perturbation_return_tested": False,
        "completed_ms": float(completed_ms), "registered_ms": float(registered_ms),
        "bin_ms": float(bin_ms), "tail_s": float(tail_s),
        "global_saturated": global_saturated, "local_saturated": local_saturated,
        "max_global_1s_mean_hz": float(per_second_mean.max()) if per_second_mean.size else None,
        "max_near_refractory_fraction": (
            float(per_second_near_refractory.max()) if per_second_near_refractory.size else None
        ),
        "per_second_mean_hz": per_second_mean.tolist(),
        "per_second_near_refractory_fraction": per_second_near_refractory.tolist(),
        "tail_mean_rate_hz": tail_mean,
        "time_above_interictal_band_ms": time_above_ms,
        "silence_bin_fraction_tail": silence_fraction_tail,
        "longest_sub_band_run_ms_tail": _longest_run_ms(silent_bins_tail, bin_ms),
        "rate_drift": drift, "rate_drift_ok": drift_ok,
        "thresholds": {
            "global_saturation_hz": float(global_saturation_hz),
            "refractory_ceiling_hz": float(refractory_ceiling_hz),
            "near_refractory_rate_hz": near_ceiling,
            "near_refractory_fraction_gate": float(near_refractory_fraction_gate),
            "interictal_roll_hi_hz": float(interictal_roll_hi_hz),
            "drift_ci_gate_per_s": float(drift_ci_gate_per_s),
            "silence_bin_fraction_gate": float(silence_bin_fraction_gate),
        },
    }


def cell_rate_distribution(cell_rates_hz, *, bands_hz=CELL_RATE_BANDS_HZ,
                           refractory_ceiling_hz: float = REFRACTORY_CEILING_HZ) -> dict:
    """Per-second quantiles and supra-band fractions of the per-cell rate distribution.

    A global mean cannot tell a whole sheet held at a moderate rate from a small patch pinned at the
    refractory ceiling while the rest is quiet; the quantiles and band fractions can.
    """

    cells = np.asarray(cell_rates_hz, float)
    if cells.ndim != 2:
        raise ValueError("cell_rates_hz must be (n_seconds, n_cells)")
    quantiles = {
        f"q{int(q * 100)}": np.quantile(cells, q, axis=1).tolist()
        for q in (0.50, 0.75, 0.90, 0.95, 0.99)
    }
    return {
        "quantiles_hz": quantiles,
        "fraction_above_hz": {
            f"{band:g}": (cells >= float(band)).mean(axis=1).tolist() for band in bands_hz
        },
        "near_refractory_fraction": (
            cells >= 0.9 * float(refractory_ceiling_hz)
        ).mean(axis=1).tolist(),
        "refractory_ceiling_hz": float(refractory_ceiling_hz),
    }
