"""Task 0 (M2 data-side audit) pure metrics — event axial/lateral footprint + matched null.

Used by `scripts/run_topic4_event_extent_audit.py` to decide whether real interictal HFO
group events axially self-limit (cover only a SEGMENT of the propagable axis) or merely run
laterally narrow along it. Inputs are along-/off-axis coordinates (mm) of an event's
participating channels, projected onto the ACCEPTED reproducible axis (source->sink).

No I/O here — see the runner for the pinned data contract (broad pool, loaders, axis source).
"""

import numpy as np

from src.propagation_skeleton_geometry import parse_shaft


def event_shaft_counts(names):
    """{shaft_prefix: count} for an event's participating channel names (drops unparseable).

    Feeds `matched_null_extent(shaft=..., shaft_counts=...)` so the shaft-matched null draws
    the SAME per-shaft channel counts as the real event (implant-sampling control)."""
    counts = {}
    for nm in names:
        shaft, _ = parse_shaft(nm)
        if shaft is None:
            continue
        counts[shaft] = counts.get(shaft, 0) + 1
    return counts


def event_extent(along, off, axis_length):
    """Axial / lateral footprint of one event's participating channels.

    axial_span    = p95(along) - p5(along)        (mm along the source->sink axis)
    lateral_span  = p95(off)   - p5(off)          (mm transverse to the axis)
    axial_fraction = axial_span / axis_length     (how much of the axis the event fills)
    lateral_ratio  = lateral_span / axial_span    (how narrow sideways relative to axial run)
    """
    along = np.asarray(along, float)
    off = np.asarray(off, float)
    axial = float(np.percentile(along, 95) - np.percentile(along, 5))
    lateral = float(np.percentile(off, 95) - np.percentile(off, 5))
    return dict(axial_span=axial, lateral_span=lateral,
                axial_fraction=axial / max(axis_length, 1e-9),
                lateral_ratio=lateral / max(axial, 1e-9))


def matched_null_extent(along_all, off_all, n_part, axis_length, n_draw, rng,
                        *, shaft=None, shaft_counts=None, rate=None):
    """Null distribution of event_extent over n_draw random size-n_part subsets of a
    subject's eligible (coord-mapped) channels, under up to three matching modes — to test
    whether an event's observed confinement is below random same-n_part sampling:
      uniform      : same-subject, same n_part, uniform draw (always returned)
      rate         : participation-rate-weighted draw (returned iff `rate` given)
      shaft_matched: draw shaft_counts[s] channels from each shaft s (returned iff
                     `shaft` + `shaft_counts` given; controls the implant-sampling confound).
    Returns {mode: {axial_fraction_med, lateral_ratio_med, axial_fraction[], lateral_ratio[]}}.
    shaft_matched SKIPS (records, never borrows cross-shaft) a draw where a shaft lacks
    enough eligible channels, so the shaft null can't silently relax the matching.
    """
    along_all = np.asarray(along_all, float)
    off_all = np.asarray(off_all, float)
    idx_all = np.arange(len(along_all))
    k = min(n_part, len(idx_all))
    out = {}

    def _draw_metrics(draw_fn):
        af, lr = [], []
        for _ in range(n_draw):
            pick = draw_fn()
            if pick is None or len(pick) < 2:
                continue
            e = event_extent(along_all[pick], off_all[pick], axis_length)
            af.append(e["axial_fraction"])
            lr.append(e["lateral_ratio"])
        return dict(axial_fraction_med=float(np.median(af)) if af else float("nan"),
                    lateral_ratio_med=float(np.median(lr)) if lr else float("nan"),
                    axial_fraction=af, lateral_ratio=lr)

    out["uniform"] = _draw_metrics(lambda: rng.choice(idx_all, size=k, replace=False))

    if rate is not None:
        p = np.asarray(rate, float)
        p = p / p.sum()
        out["rate"] = _draw_metrics(lambda: rng.choice(idx_all, size=k, replace=False, p=p))

    if shaft is not None and shaft_counts is not None:
        shaft = np.asarray(shaft, object)
        by_shaft = {s: idx_all[shaft == s] for s in shaft_counts}

        def _shaft_draw():
            picks = []
            for s, c in shaft_counts.items():
                pool = by_shaft.get(s, np.array([], int))
                if len(pool) < c:
                    return None  # not enough eligible on this shaft -> skip (don't borrow)
                picks.append(rng.choice(pool, size=c, replace=False))
            return np.concatenate(picks) if picks else None

        out["shaft_matched"] = _draw_metrics(_shaft_draw)

    return out


def _boot_mean_ci(deltas, rng, n_boot=2000, alpha=0.05):
    """Bootstrap (lo, hi) percentile CI of the MEAN of a per-subject delta array."""
    d = np.asarray(deltas, float)
    means = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(n_boot)])
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def cohort_verdict(per_subject, rng, n_boot=2000, min_subjects=10):
    """PRE-REGISTERED Task-0 Step-9 gate. Each per-subject record carries the subject's median
    observed vs shaft-matched-null metric: keys axial_obs / axial_null / lateral_obs / lateral_null.

    Δ = obs − null (CONFINEMENT ⇒ Δ < 0). "below null" ⇒ the Δ-mean bootstrap-CI upper bound < 0.
    Branches (deterministic order):
      INCONCLUSIVE   if < min_subjects, OR 0.5 < AF < 0.75 (gray band)
      AXIAL_EXTENDED_LATERAL_NARROW  if AF ≥ 0.75 AND LR ≤ 0.5 AND lateral Δ below null  → reframe
      AXIAL_SEGMENT  if AF ≤ 0.5 AND axial Δ below null                                  → model it
      SAMPLING_ARTIFACT  if AF ≤ 0.5 AND axial Δ CI includes 0 (obs ≈ null)              → don't over-model
      INCONCLUSIVE   otherwise
    AF / LR are cohort medians of the per-subject observed medians.
    """
    axial_obs = np.array([r["axial_obs"] for r in per_subject], float)
    axial_null = np.array([r["axial_null"] for r in per_subject], float)
    lateral_obs = np.array([r["lateral_obs"] for r in per_subject], float)
    lateral_null = np.array([r["lateral_null"] for r in per_subject], float)
    n = len(per_subject)

    AF = float(np.median(axial_obs)) if n else float("nan")
    LR = float(np.median(lateral_obs)) if n else float("nan")
    axial_delta = axial_obs - axial_null
    lateral_delta = lateral_obs - lateral_null

    axial_ci = _boot_mean_ci(axial_delta, rng, n_boot) if n else (float("nan"), float("nan"))
    lateral_ci = _boot_mean_ci(lateral_delta, rng, n_boot) if n else (float("nan"), float("nan"))

    def _wilcoxon_p(d):
        if n < 1 or np.allclose(d, 0.0):
            return float("nan")
        try:
            from scipy.stats import wilcoxon
            return float(wilcoxon(d).pvalue)
        except Exception:
            return float("nan")

    axial_below_null = axial_ci[1] < 0.0
    lateral_below_null = lateral_ci[1] < 0.0
    axial_ci_includes_0 = axial_ci[0] <= 0.0 <= axial_ci[1]

    if n < min_subjects:
        verdict = "INCONCLUSIVE"
    elif 0.5 < AF < 0.75:
        verdict = "INCONCLUSIVE"
    elif AF >= 0.75 and LR <= 0.5 and lateral_below_null:
        verdict = "AXIAL_EXTENDED_LATERAL_NARROW"
    elif AF <= 0.5 and axial_below_null:
        verdict = "AXIAL_SEGMENT"
    elif AF <= 0.5 and axial_ci_includes_0:
        verdict = "SAMPLING_ARTIFACT"
    else:
        verdict = "INCONCLUSIVE"

    return dict(
        verdict=verdict, n_subjects=n, AF=AF, LR=LR,
        axial_delta_mean=float(np.mean(axial_delta)) if n else float("nan"),
        lateral_delta_mean=float(np.mean(lateral_delta)) if n else float("nan"),
        axial_ci=axial_ci, lateral_ci=lateral_ci,
        axial_wilcoxon_p=_wilcoxon_p(axial_delta),
        lateral_wilcoxon_p=_wilcoxon_p(lateral_delta),
    )
