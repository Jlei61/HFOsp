"""Find a real baseline instead of asserting one.

The 2 s checkpoint cannot be called a baseline: measured against forty
non-overlapping 500 ms windows from the SAME-SEED Z/M-off run, the Joint arm's
median 20 ms-EMA rate over [1500, 2000] ms is 37 / 64 / 50 Hz against a q95 of
30 / 30 / 31 Hz. It is an early-transition checkpoint.

A baseline here is defined by the Z/M-off distribution, not by a clock reading:
the first stable window after burn-in in which the population rate, the
h-weighted z and the h-weighted m all sit inside the Z/M-off q95 support. If no
such window exists, that is the answer -- the figure is then labelled
`early transition vs pre-ictal`, and the word baseline is not used.
"""
from __future__ import annotations

import numpy as np


def ema_rate_hz(active_fraction, bin_ms, tau_ms=20.0):
    """Active fraction -> 20 ms-EMA per-neuron rate, the engine's own units."""
    active = np.asarray(active_fraction, float)
    alpha = 1.0 - np.exp(-float(bin_ms) / float(tau_ms))
    out = np.empty_like(active)
    value = 0.0
    for index, sample in enumerate(active):
        value += alpha * (sample / (float(bin_ms) * 1e-3) - value)
        out[index] = value
    return out


def window_medians(series, bin_ms, window_ms):
    """Non-overlapping window medians -- a distribution, not a single number.

    The first attempt took one window per reference run and compared against the
    q95 of three numbers spanning 11x. Forty windows per run is what makes the
    support meaningful.
    """
    width = int(round(float(window_ms) / float(bin_ms)))
    if width < 1:
        raise ValueError("window is shorter than one bin")
    n = len(series) // width
    if n < 1:
        return np.empty(0)
    return np.array([np.median(series[i * width:(i + 1) * width]) for i in range(n)])


def zm_off_support(reference_series, bin_ms, *, window_ms=500.0, quantile=95.0):
    """Upper support of a Z/M-off quantity, pooled over reference runs."""
    pooled = np.concatenate([window_medians(ema_rate_hz(series, bin_ms), bin_ms,
                                            window_ms)
                             for series in reference_series])
    return {"q95": float(np.percentile(pooled, quantile)),
            "median": float(np.median(pooled)),
            "n_windows": int(len(pooled))}


def slow_support(reference_values, *, quantile=95.0):
    """Upper support of an h-weighted slow quantity under Z/M-off.

    Both slow clauses are stated on the quantity that GROWS as the network
    approaches the transition -- disinhibition (1 - z) and adaptation m -- so a
    single upper bound is the right shape for both. Stating the z clause on z
    itself inverts it: z falls from 1, so `z <= q95(z)` demands that the
    candidate window be at least as DISINHIBITED as the reference's 95th
    percentile, which rejects exactly the windows that still look like the
    non-transitioning network.
    """
    pooled = np.concatenate([np.asarray(v, float) for v in reference_values])
    return {"q95": float(np.percentile(pooled, quantile)),
            "median": float(np.median(pooled)), "n": int(len(pooled))}


def find_baseline_window(rate_series, bin_ms, *, rate_q95, z_trace, m_trace,
                         zm_time_ms, disinhibition_q95, m_q95, burn_in_ms=500.0,
                         window_ms=500.0, search_end_ms=None):
    """First post-burn-in window where ALL THREE quantities sit inside support.

    All three clauses are upper bounds on quantities that GROW toward the
    transition: population rate, disinhibition (1 - z), and adaptation m. The
    disinhibition clause is deliberately NOT stated on z. z falls from 1, so an
    upper bound on z asks the window to be at least as disinhibited as the
    reference -- the opposite of the question, and it rejected the quietest
    windows on all three canary seeds before this was corrected.

    Returns the window and a per-clause verdict. `found=False` is a legitimate
    outcome and must be reported, not worked around by relaxing a clause.
    """
    ema = ema_rate_hz(rate_series, bin_ms)
    width = int(round(float(window_ms) / float(bin_ms)))
    start = int(round(float(burn_in_ms) / float(bin_ms)))
    stop = len(ema) if search_end_ms is None else int(round(float(search_end_ms) / bin_ms))
    zm_time = np.asarray(zm_time_ms, float)
    z_trace = np.asarray(z_trace, float)
    m_trace = np.asarray(m_trace, float)
    attempts = []
    for lo in range(start, max(start, stop - width) + 1, width):
        hi = lo + width
        t0, t1 = lo * bin_ms, hi * bin_ms
        rate_median = float(np.median(ema[lo:hi]))
        inside = (zm_time >= t0) & (zm_time < t1)
        z_median = float(np.median(z_trace[inside])) if inside.any() else np.nan
        m_median = float(np.median(m_trace[inside])) if inside.any() else np.nan
        disinhibition = 1.0 - z_median
        clauses = {"rate_within_zm_off_support": rate_median <= rate_q95,
                   "disinhibition_within_support": bool(
                       np.isfinite(disinhibition) and disinhibition <= disinhibition_q95),
                   "m_within_support": bool(np.isfinite(m_median) and m_median <= m_q95)}
        attempts.append({"window_ms": [t0, t1], "rate_median_hz": rate_median,
                         "z_median": z_median, "disinhibition_median": disinhibition,
                         "m_median": m_median,
                         "clauses": clauses, "pass": all(clauses.values())})
        if all(clauses.values()):
            return {"found": True, "window_ms": [t0, t1],
                    "checkpoint_ms": float(t1), "attempts": attempts,
                    "supports": {"rate_q95": rate_q95,
                                 "disinhibition_q95": disinhibition_q95,
                                 "m_q95": m_q95}}
    return {"found": False, "attempts": attempts,
            "supports": {"rate_q95": rate_q95,
                         "disinhibition_q95": disinhibition_q95, "m_q95": m_q95},
            "consequence": ("no window after burn-in has all three quantities inside "
                            "the Z/M-off support; the two states must be labelled "
                            "'early transition vs pre-ictal', not 'baseline vs "
                            "pre-ictal'")}
