"""Five regimes for one trajectory, separating a carrier from a burst train that re-ignites.

The accepted workpoint classifier already tells a high branch from an interictal one and a fixed
high state from an oscillating one.  What it does not do is separate the two ways of oscillating,
and they are different objects: a discharge whose troughs stay recruited is one sustained state
being modulated, while a train whose troughs fall back to the interictal level is a sequence of
separate events that keep re-igniting from rest.  Only the first is a carrier.

The accepted classifier is reused for the one question it is the right shape for -- is this a
sustained high branch at all, measured against the accepted interictal band -- and for nothing
else.  Its fixed-versus-orbit split cannot be borrowed here: it runs on a 300 ms rolling mean,
and the discharge bursts every 86 ms, so three and a half bursts are averaged into each window
before the modulation is measured.  That split answers "is the envelope modulated", which is a
different question from "where do the troughs between bursts sit".  The trough test therefore
runs on the unsmoothed recruited fraction:

    R1  runaway or refractory-ceiling saturation
    R2  bounded continuous high state
    R3  bounded oscillatory high state whose troughs stay above interictal  (carrier)
    R4  burst train whose troughs return to interictal                      (re-ignition)
    R5  high state that ends on its own and is followed by interictal again (closed loop)

R5 outranks the rest: a trajectory that terminated and recovered is described by what it did, not
by the shape it had while it was up.

The interictal reference is taken from the trajectory's own pre-onset window rather than from a
frozen constant, so an arm that changes the interictal statistics is compared against what it
actually produced instead of against a run it no longer resembles.
"""
from __future__ import annotations

import numpy as np

from src.topic4_mz_fcxr_dynamics import classify_run_workpoint, workpoint_metrics

REGIMES = ("R1_runaway", "R2_bounded_high", "R3_carrier", "R4_burst_train",
           "R5_closed_loop", "R0_interictal_only")
TROUGH_Q = 25.0          # per cent; the low quartile of the epoch stands for its troughs
INTERICTAL_Q = 95.0      # per cent; the top of the pre-onset spread is what a trough must clear
# (p90-p10)/p90 on the UNSMOOTHED recruited fraction.  Below this the high state is continuous
# rather than oscillating, so it has no troughs to place.
MOD_MIN = 0.30


def epoch_modulation(af, bin_ms, t0_ms, t1_ms):
    """How much the recruited fraction swings inside the epoch, measured without smoothing.

    Same form as the accepted classifier's modulation, on the raw trace instead of a 300 ms rolling
    mean -- at 86 ms between bursts that window is the difference between seeing the swing and not.
    """
    af = np.asarray(af, float)
    t = np.arange(af.size) * float(bin_ms)
    seg = af[(t >= t0_ms) & (t < t1_ms)]
    if seg.size < 4:
        return float("nan")
    p90, p10 = float(np.percentile(seg, 90)), float(np.percentile(seg, 10))
    return float((p90 - p10) / max(p90, 1e-9))


def trough_level(af, bin_ms, t0_ms, t1_ms, q=TROUGH_Q):
    """How low the recruited fraction gets between bursts inside one epoch."""
    af = np.asarray(af, float)
    t = np.arange(af.size) * float(bin_ms)
    seg = af[(t >= t0_ms) & (t < t1_ms)]
    return float(np.percentile(seg, q)) if seg.size else float("nan")


def interictal_ceiling(af, bin_ms, t_end_ms, q=INTERICTAL_Q):
    """Top of the pre-onset spread: what a trough must stay above to count as still recruited."""
    af = np.asarray(af, float)
    t = np.arange(af.size) * float(bin_ms)
    seg = af[t < t_end_ms]
    return float(np.percentile(seg, q)) if seg.size else float("nan")


def classify_regime(*, af, af_bin_ms, rate_hz, dt_ms, baseline_roll_hi_hz,
                    onset_ms, offset_ms, run_ms, terminated, recovered,
                    refractory_ceiling_fraction=0.0, ceiling_max=0.5,
                    numerical_unsafe=False):
    """One regime label for one trajectory, with the numbers the label rests on.

    ``terminated`` must already encode that a bout ending at the end of the record has not ended;
    this function does not re-derive it, because the same mistake made twice in two places is
    harder to find than the same mistake made once.
    """
    if numerical_unsafe:
        return dict(regime="R1_runaway", reason="numerical failure", carrier=False)
    if onset_ms is None:
        return dict(regime="R0_interictal_only", reason="no high-state epoch in the record",
                    carrier=False)
    if refractory_ceiling_fraction >= ceiling_max:
        return dict(regime="R1_runaway", carrier=False,
                    reason=(f"{refractory_ceiling_fraction:.0%} of cells sit at the refractory "
                            f"ceiling: a saturated tonic branch, not a bounded state"),
                    refractory_ceiling_fraction=float(refractory_ceiling_fraction))

    epoch_end = offset_ms if (terminated and offset_ms is not None) else run_ms
    wp = workpoint_metrics(np.asarray(rate_hz, float), float(dt_ms),
                           float(baseline_roll_hi_hz), float(onset_ms))
    wp["numerical_unsafe"] = False
    wp_label = classify_run_workpoint(wp)

    trough = trough_level(af, af_bin_ms, onset_ms, epoch_end)
    ceiling = interictal_ceiling(af, af_bin_ms, onset_ms)
    mod = epoch_modulation(af, af_bin_ms, onset_ms, epoch_end)
    stays_recruited = bool(np.isfinite(trough) and np.isfinite(ceiling) and trough > ceiling)
    oscillates = bool(np.isfinite(mod) and mod >= MOD_MIN)

    out = dict(workpoint_label=wp_label, trough_af=trough, interictal_ceiling_af=ceiling,
               epoch_modulation=mod, stays_recruited=stays_recruited, oscillates=oscillates,
               carrier=False)
    if terminated and recovered:
        out.update(regime="R5_closed_loop",
                   reason=(f"the high state ended at {offset_ms / 1000:.1f} s of "
                           f"{run_ms / 1000:.0f} s and interictal events came back"))
        return out
    if wp_label not in ("FINITE_HIGH_FIXED", "FINITE_HIGH_ORBIT") and not terminated:
        out.update(regime="R0_interictal_only",
                   reason=f"entered but the epoch does not read as a high branch ({wp_label})")
        return out
    if not oscillates:
        out.update(regime="R2_bounded_high",
                   reason=(f"sustained above the band and continuous: the recruited fraction "
                           f"swings by only {mod:.2f} inside the epoch, so it has no troughs"))
        return out
    out.update(regime="R3_carrier" if stays_recruited else "R4_burst_train",
               carrier=stays_recruited,
               reason=(f"sustained and oscillating (swing {mod:.2f}); troughs "
                       f"{'stayed above' if stays_recruited else 'fell back to'} the pre-onset "
                       f"spread ({trough:.4f} vs {ceiling:.4f})"
                       + (f", and it ended at {offset_ms / 1000:.1f} s without interictal "
                          f"statistics returning" if terminated else "")))
    return out
