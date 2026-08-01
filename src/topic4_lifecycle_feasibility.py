"""Pre-flight quantitative diagnostics for an ictal-lifecycle design.

Motivation
----------
Four Topic-4 SNN mechanism rounds exposed useful scale mismatches that can be
computed before another production run:

* Z/M adaptation ``m`` has a refractory-capped steady-current scale of
  ``eta_m * tau_adp / tau_ref`` mV, 3.85% of the E-cell reset-to-threshold gap.
* Z/M inhibitory efficacy ``z`` -- the entry variable -- has ``z_inf = 0`` for every
  state with elevated inhibitory current, i.e. its target inside the ictal state lies
  *further* from the interictal basin than the entry point.  Entry is a latch.
* Z/M divisive pool ``S_G`` has ample authority (94% recurrent removal at ceiling)
  but ``tau_S = 80 ms``, so it provides little post-activity hold by itself.
* FCXR-HYB1 potassium ``tau_K = 650 ms`` against an interictal event interval of
  400-600 ms accumulates event-to-event by 1.7-2.2x, so the baseline floor ratchets.

These are diagnostics, not mathematical non-existence tests.  In particular:

* a sub-millivolt current can still cross a *network* bifurcation without spanning
  the single-cell reset-to-threshold gap;
* an 80 ms inhibitory loop can participate in a multi-second attractor while it is
  continually driven;
* a high mean firing rate limits upward ISI headroom but does not forbid downward
  population-rate modulation; and
* an entry variable may be a latch when a different variable supplies offset.

The module therefore reports design-risk flags and intermediate quantities.  It
must never emit ``infeasible`` or authorize/forbid a simulation by itself.

Scope
-----
Clearing a diagnostic does not establish entry, carrier, offset or recovery, and a
risk flag does not prove that a mechanism is impossible.  Hard GO/NO-GO decisions
must come from the registered state-fork, baseline-preservation and observation
gates in the mechanism spec.

Units
-----
Times are milliseconds, voltages millivolts, rates Hz.  Per-neuron rates used
internally are spikes/ms (Hz * 1e-3).  Accumulator gains are mV per unit of the
accumulator, where the accumulator increments by 1.0 per postsynaptic spike.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

LIFECYCLE_FEASIBILITY_VERSION = "topic4_lifecycle_design_diagnostics_v2_2026-08-01"

# Reference bars used to flag scale risks.  They are heuristics, not necessary
# conditions.  A mechanism spec may use different registered values, but must keep
# the underlying dimensional quantities visible.
DEFAULT_MIN_AUTHORITY_RATIO = 1.0        # brake must at least span reset -> threshold
DEFAULT_MAX_ACCUMULATION = 1.5           # inter-event floor elevation of an accumulator
DEFAULT_MIN_HOLD_RATIO = 1.0             # brake decay time / target ictal duration
DEFAULT_MAX_REFRACTORY_OCCUPANCY = 0.5   # target ictal rate / (1 / tau_ref)


@dataclass(frozen=True)
class CheckResult:
    """One quantitative diagnostic.  ``passed=False`` means a risk flag, not NO-GO."""

    name: str
    passed: bool
    reason: str
    detail: dict

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "reason": self.reason,
            "detail": dict(self.detail),
        }


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value}")
    return value


def brake_authority(
    *,
    name: str,
    gain_mv_per_unit: float,
    tau_accum_ms: float,
    tau_ref_ms: float,
    v_th_mv: float,
    v_reset_mv: float,
    min_authority_ratio: float = DEFAULT_MIN_AUTHORITY_RATIO,
) -> CheckResult:
    """Compare a spike-driven brake scale with the reset-to-threshold gap.

    The brake is the standard adaptation form ``du/dt = -u/tau_accum + sum_k
    delta(t - t_k)`` with a subtracted current ``gain * u``.  At a sustained
    per-neuron rate ``r`` its steady state is ``u_ss = r * tau_accum``, so the
    largest current it can ever produce is set by the largest sustainable rate,
    which the absolute refractory period caps at ``1 / tau_ref``:

        ceiling_mv = gain * tau_accum / tau_ref .

    Spanning the reset-to-threshold gap is a conservative single-cell reference,
    not a necessary condition for network offset: a much smaller current can move
    a recurrent network across a bifurcation.  A failed comparison is therefore a
    scale-risk flag that must be resolved by frozen-state continuation/ablation.
    """
    gain = float(gain_mv_per_unit)
    if not math.isfinite(gain) or gain < 0.0:
        raise ValueError(f"gain_mv_per_unit must be finite and >= 0, got {gain}")
    tau_accum = _positive("tau_accum_ms", tau_accum_ms)
    tau_ref = _positive("tau_ref_ms", tau_ref_ms)
    min_ratio = _positive("min_authority_ratio", min_authority_ratio)
    gap = float(v_th_mv) - float(v_reset_mv)
    if not math.isfinite(gap) or gap <= 0.0:
        raise ValueError(f"v_th_mv must exceed v_reset_mv, got gap {gap}")

    r_max_per_ms = 1.0 / tau_ref
    u_ceiling = r_max_per_ms * tau_accum
    ceiling_mv = gain * u_ceiling
    ratio = ceiling_mv / gap
    passed = ratio >= min_ratio
    return CheckResult(
        name=f"brake_authority[{name}]",
        passed=passed,
        reason=(
            f"brake ceiling {ceiling_mv:.4g} mV is {ratio:.3g}x the "
            f"{gap:.4g} mV reset-to-threshold gap "
            f"({'>=' if passed else '<'} reference {min_ratio:.3g}x; diagnostic only)"
        ),
        detail={
            "gain_mv_per_unit": gain,
            "tau_accum_ms": tau_accum,
            "tau_ref_ms": tau_ref,
            "max_sustained_rate_hz": r_max_per_ms * 1e3,
            "accumulator_ceiling": u_ceiling,
            "brake_ceiling_mv": ceiling_mv,
            "reset_to_threshold_gap_mv": gap,
            "authority_ratio": ratio,
            "min_authority_ratio": min_ratio,
        },
    )


def slow_variable_reversal(
    *,
    name: str,
    u_interictal: float,
    u_entry: float,
    u_inf_ictal: float,
) -> CheckResult:
    """Can this variable, *if assigned the offset role*, turn around in ictus?

    A permissive variable ``u`` with ``tau du/dt = u_inf(state) - u`` drifts from
    ``u_interictal`` across the entry point ``u_entry``.  For the ictal state to end
    under the mechanism's own dynamics, ``u`` must drift back across ``u_entry``
    while the ictal state is on, which requires the ictal target ``u_inf_ictal`` to
    lie strictly on the interictal side of ``u_entry``.

    If ``u_inf_ictal`` sits at or beyond ``u_entry`` in the entry direction, the
    variable is a **latch** and cannot be the sole offset coordinate.  That is
    compatible with a design in which ``z`` controls entry and ``m`` (or another
    independently registered feedback) controls offset.
    """
    u0 = float(u_interictal)
    ue = float(u_entry)
    ui = float(u_inf_ictal)
    for label, value in (("u_interictal", u0), ("u_entry", ue), ("u_inf_ictal", ui)):
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite, got {value}")
    if ue == u0:
        raise ValueError("u_entry must differ from u_interictal (no entry direction)")

    direction = 1.0 if ue > u0 else -1.0        # sign of the drift that causes entry
    margin = direction * (ui - ue)              # < 0 means the ictal target is back inside
    passed = margin < 0.0
    return CheckResult(
        name=f"slow_variable_reversal[{name}]",
        passed=passed,
        reason=(
            f"ictal target {ui:.4g} is "
            f"{'on the interictal side of' if passed else 'at or beyond'} "
            f"the entry point {ue:.4g} (entry direction "
            f"{'increasing' if direction > 0 else 'decreasing'}); "
            f"{'can reverse across the entry coordinate' if passed else 'latch if used alone; another offset coordinate is required'}"
        ),
        detail={
            "u_interictal": u0,
            "u_entry": ue,
            "u_inf_ictal": ui,
            "entry_direction": direction,
            "reversal_margin": margin,
        },
    )


def timescale_separation(
    *,
    name: str,
    tau_recover_ms: float,
    interictal_event_interval_ms: float,
    target_ictal_duration_ms: float,
    max_accumulation: float = DEFAULT_MAX_ACCUMULATION,
    min_hold_ratio: float = DEFAULT_MIN_HOLD_RATIO,
) -> CheckResult:
    """Report baseline-ratchet and post-activity-hold scale diagnostics.

    **Ratchet (against the interictal train).**  A leaky accumulator driven once per
    ``interictal_event_interval_ms`` reaches a steady-state floor of
    ``1 / (1 - exp(-interval / tau_recover))`` times its single-event response.  With
    ``tau_recover`` comparable to the interictal interval the baseline creeps up
    between events instead of returning, which destroys baseline preservation before
    the ictal question is ever reached (FCXR-HYB1).

    **Post-activity hold.**  ``tau_recover / target_duration`` measures whether the
    variable can remain as a post-activity memory after its drive disappears.  It is
    not a bound on the duration of a driven attractor and is not, by itself, a
    necessary condition for termination.
    """
    tau_rec = _positive("tau_recover_ms", tau_recover_ms)
    interval = _positive("interictal_event_interval_ms", interictal_event_interval_ms)
    duration = _positive("target_ictal_duration_ms", target_ictal_duration_ms)
    max_acc = _positive("max_accumulation", max_accumulation)
    min_hold = _positive("min_hold_ratio", min_hold_ratio)

    accumulation = 1.0 / (1.0 - math.exp(-interval / tau_rec))
    hold_ratio = tau_rec / duration
    ratchet_ok = accumulation <= max_acc
    hold_ok = hold_ratio >= min_hold
    passed = ratchet_ok and hold_ok

    failures = []
    if not ratchet_ok:
        failures.append(
            f"ratchets to {accumulation:.3g}x the single-event floor across a "
            f"{interval:.4g} ms interictal train (max {max_acc:.3g}x)"
        )
    if not hold_ok:
        failures.append(
            f"post-activity decay is {tau_rec:.4g} ms against a {duration:.4g} ms "
            f"reference episode (hold ratio {hold_ratio:.3g}, reference {min_hold:.3g}); "
            "this flags missing refractory memory, not inability to affect a driven state"
        )
    reason = (
        f"accumulation {accumulation:.3g}x, hold ratio {hold_ratio:.3g}: both bars cleared"
        if passed
        else "; ".join(failures)
    )
    return CheckResult(
        name=f"timescale_separation[{name}]",
        passed=passed,
        reason=reason,
        detail={
            "tau_recover_ms": tau_rec,
            "interictal_event_interval_ms": interval,
            "target_ictal_duration_ms": duration,
            "accumulation_factor": accumulation,
            "max_accumulation": max_acc,
            "ratchet_ok": ratchet_ok,
            "hold_ratio": hold_ratio,
            "min_hold_ratio": min_hold,
            "hold_ok": hold_ok,
        },
    )


def operating_point_headroom(
    *,
    name: str,
    target_rate_hz: float,
    tau_ref_ms: float,
    max_occupancy: float = DEFAULT_MAX_REFRACTORY_OCCUPANCY,
) -> CheckResult:
    """Quantify refractory-ceiling occupancy at a proposed operating point.

    The absolute refractory period caps the per-neuron rate at ``1 / tau_ref``.  A
    population sitting at occupancy ``o`` of that ceiling has little *upward* ISI
    headroom.  This is a saturation-risk diagnostic; it does not forbid downward
    modulation, population bursting, phase staggering or a lower-rate carrier.
    """
    rate = _positive("target_rate_hz", target_rate_hz)
    tau_ref = _positive("tau_ref_ms", tau_ref_ms)
    max_occ = _positive("max_occupancy", max_occupancy)
    if max_occ >= 1.0:
        raise ValueError(f"max_occupancy must be < 1, got {max_occ}")

    ceiling_hz = 1e3 / tau_ref
    occupancy = rate / ceiling_hz
    mean_isi_ms = 1e3 / rate
    headroom_ms = mean_isi_ms - tau_ref
    passed = occupancy <= max_occ
    return CheckResult(
        name=f"operating_point_headroom[{name}]",
        passed=passed,
        reason=(
            f"{rate:.4g} Hz is {occupancy:.3g} of the {ceiling_hz:.4g} Hz refractory "
            f"ceiling ({headroom_ms:.3g} ms of interval headroom); "
            f"{'below the reference occupancy' if passed else 'high-occupancy risk; carrier is not ruled out'}"
        ),
        detail={
            "target_rate_hz": rate,
            "tau_ref_ms": tau_ref,
            "refractory_ceiling_hz": ceiling_hz,
            "refractory_occupancy": occupancy,
            "max_occupancy": max_occ,
            "mean_isi_ms": mean_isi_ms,
            "isi_headroom_ms": headroom_ms,
        },
    )


def screen_mechanism(mechanism: str, checks) -> dict:
    """Aggregate diagnostics without producing a scientific GO/NO-GO."""
    checks = list(checks)
    if not checks:
        raise ValueError("screen_mechanism requires at least one check")
    for check in checks:
        if not isinstance(check, CheckResult):
            raise TypeError(f"expected CheckResult, got {type(check).__name__}")
    failed = [c for c in checks if not c.passed]
    return {
        "version": LIFECYCLE_FEASIBILITY_VERSION,
        "mechanism": str(mechanism),
        "verdict": "diagnostic_risks_present" if failed else "no_diagnostic_flags",
        "n_checks": len(checks),
        "n_failed": len(failed),
        "failed_checks": [c.name for c in failed],
        "checks": [c.as_dict() for c in checks],
        "interpretation": (
            "one or more scale-risk flags require explicit mechanism-role assignment "
            "and empirical state-fork tests; they are not an infeasibility proof"
            if failed
            else "no selected heuristic is flagged; this is not evidence for any "
            "lifecycle leg and does not authorize a scientific claim"
        ),
        "claim_boundary": (
            "diagnostic only; never establishes or excludes carrier, entry, offset, "
            "recovery, control, or lifecycle"
        ),
    }


__all__ = [
    "LIFECYCLE_FEASIBILITY_VERSION",
    "DEFAULT_MIN_AUTHORITY_RATIO",
    "DEFAULT_MAX_ACCUMULATION",
    "DEFAULT_MIN_HOLD_RATIO",
    "DEFAULT_MAX_REFRACTORY_OCCUPANCY",
    "CheckResult",
    "brake_authority",
    "slow_variable_reversal",
    "timescale_separation",
    "operating_point_headroom",
    "screen_mechanism",
]
