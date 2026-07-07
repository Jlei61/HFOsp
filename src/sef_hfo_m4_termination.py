"""M4-2 termination classifier (spec 2026-07-07 rev2, §7.1).

Two-field schema, kept SEPARATE (a quiet tail is NOT the same as a re-triggerable interictal state):

  termination_class = classify_termination(af, ...)  # SHAPE of the event
      persist         -- rises to a plateau and stays high through the end (M4 pass-1 state)
      terminate_clean -- high plateau -> relatively sharp offset -> quiet tail
      fade            -- monotone decline to baseline, no sustained plateau (NOT a termination)
      fragment        -- intermittent short bursts, never a sustained plateau
      suppress        -- never rises meaningfully above baseline (brake too strong)
      rebound         -- one clean event, quiet gap, then a re-ignition

  retrigger_probe   = retrigger_verdict(termination_class, post_af, ...)  # is the RECOVERED state excitable?
      pass / fail / not_run  (not_run unless termination_class == terminate_clean)

Thresholds are calibrated on SYNTHETIC fixtures (tests/test_m4_2_termination.py), independent of simulation
data, to avoid the "tune thresholds on the same real traces you then classify" circularity. Real Arm-0 /
runaway instances are a sanity check only, not a threshold source.

`af` is a per-bin activity trace (active fraction in [0,1], or any normalized rate); `bin_ms` is the bin width.
The clean-vs-fade discriminator is NEAR-PEAK duration (time within ~10% of peak): a flat plateau spends a long
time near peak; a monotone decline barely touches it.
"""
import numpy as np

CLASSES = ("persist", "terminate_clean", "fade", "fragment", "suppress", "rebound")


def _episodes(mask, gap_bins):
    """Contiguous True runs of `mask`, merging quiet gaps shorter than `gap_bins`. Returns [(i0, i1), ...]."""
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    eps = []
    start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev > gap_bins:
            eps.append((start, prev + 1))
            start = i
        prev = i
    eps.append((start, prev + 1))
    return eps


def classify_termination(af, bin_ms, baseline=None, *,
                         a_min=0.05, on_frac=0.3, plateau_frac=0.9, tail_frac=0.25,
                         min_plateau_ms=250.0, gap_ms=50.0, tail_ms=None):
    """Classify an activity trace into a termination_class. Returns (cls, info) where info carries
    'peak', 'baseline', 'amp', 'offset_ms' (offset time for terminate_clean / fade / rebound, else None)."""
    af = np.asarray(af, float)
    n = af.size
    if baseline is None:                                   # default: median of the leading ~5% of the trace
        baseline = float(np.median(af[:max(1, n // 20)]))
    peak = float(af.max())
    amp = peak - baseline
    info = dict(peak=peak, baseline=baseline, amp=amp, offset_ms=None)
    if amp < a_min:                                        # never rose above baseline -> brake killed it
        return "suppress", info

    on = baseline + on_frac * amp                          # "event on" level
    plat = baseline + plateau_frac * amp                   # "near-peak / plateau" level (flatness proxy)
    gap_bins = int(round(gap_ms / bin_ms))
    min_plateau = int(round(min_plateau_ms / bin_ms))
    tail_bins = int(round(tail_ms / bin_ms)) if tail_ms else max(1, n // 10)

    eps = _episodes(af >= on, gap_bins)

    def plateau_len(i0, i1):
        return int((af[i0:i1] >= plat).sum())

    if len(eps) >= 2:                                      # multiple events: rebound vs fragment
        if len(eps) == 2 and plateau_len(*eps[0]) >= min_plateau:
            info["offset_ms"] = eps[0][1] * bin_ms         # offset of the first (clean) event
            return "rebound", info
        return "fragment", info
    if len(eps) == 0:                                      # above a_min but never crosses `on` -> suppressed
        return "suppress", info

    i0, i1 = eps[0]
    tail = float(af[-tail_bins:].mean())
    if tail >= baseline + tail_frac * amp:                 # still active at the end -> never terminated
        return "persist", info
    info["offset_ms"] = i1 * bin_ms
    if plateau_len(i0, i1) >= min_plateau:                 # sustained plateau then quiet -> clean termination
        return "terminate_clean", info
    return "fade", info                                    # terminated but only a gradual decline


def retrigger_verdict(termination_class, post_af=None, baseline=None, ref_peak=None, *,
                      reig_frac=0.5, runaway_tail_frac=0.8, tail_ms=None, bin_ms=5.0):
    """Verdict for the post-offset re-trigger kick window. `not_run` unless the event terminated cleanly.
    pass  = the recovered state re-ignites a BOUNDED event (rises to a substantial fraction of the original
            event peak, then comes back down); fail = fizzle (no re-ignition) or runaway (stays high)."""
    if termination_class != "terminate_clean" or post_af is None:
        return "not_run"
    post_af = np.asarray(post_af, float)
    amp = ref_peak - baseline
    if float(post_af.max()) < baseline + reig_frac * amp:  # fizzle: kick did not re-ignite an event
        return "fail"
    tail_bins = int(round(tail_ms / bin_ms)) if tail_ms else max(1, post_af.size // 5)
    if float(post_af[-tail_bins:].mean()) >= baseline + runaway_tail_frac * amp:  # stayed high -> runaway
        return "fail"
    return "pass"                                          # re-ignited AND came back down = bounded re-event
