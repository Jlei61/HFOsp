"""Outcome distance between two runs of the same pinned slow field — the versioned adjudicator.

Round 2 answered "do the low and high initialisations reach different end states?" with a tolerance
rule that lived in the plotting script.  That is the wrong home for a scientific verdict: it is not
versioned with the result, it borrowed a tolerance registered for something else, and it compared
only a scalar rate and a scalar area.  This module is that comparison done properly -- declared
thresholds, four independent readouts, and tests -- and it is the only place the verdict is produced.

Why a phase-aligned comparison is required.  Every non-saturated outcome on this substrate is a
burst train at roughly ten per second.  Two runs of the same field settle onto the same rhythm at
different phase, so their zero-lag population correlation is strongly NEGATIVE (-0.42 to -0.47 was
measured across the atlas) while the trains are in fact the same object.  A verdict built on
zero-lag correlation would therefore report a difference that is only a phase offset.

Why the alignment window is declared and short.  A single best lag cannot track a slowly drifting
phase: at the weakest field the 5 s aligned correlation is 0.855 while the 1 s aligned correlation is
0.933.  The registered window is stated below so the number is reproducible rather than tuned.
"""
from __future__ import annotations

import numpy as np

SCHEMA = "fcxr-lc6b-outcome-distance-1.0"

#: Registered here, for this comparison, rather than borrowed from another experiment's contract.
TAIL_MS = 5000.0                    #: window every readout is computed over
RATE_BIN_MS = 20.0                  #: population-rate resolution; the burst period is ~95-100 ms
PHASE_ALIGN_WINDOW_MS = 1000.0      #: window for the phase-aligned correlation (see module docstring)
MAX_LAG_MS = 200.0                  #: lag search range, about two burst periods
RATE_RELATIVE_TOLERANCE = 0.15
AREA_RELATIVE_TOLERANCE = 0.15
PER_CELL_CORRELATION_FLOOR = 0.95
PHASE_ALIGNED_CORRELATION_FLOOR = 0.80
SPATIAL_MAP_CORRELATION_FLOOR = 0.90


def per_cell_rate_vector(steps, cells, *, n_steps, n_cells, tail_ms=TAIL_MS, dt_ms=0.05):
    """Spike count per cell over the final ``tail_ms``.

    This is the readout a scalar population rate cannot give: two runs can share a mean while
    recruiting different cells, and only a per-cell vector separates those.
    """
    steps = np.asarray(steps, np.int64)
    start = int(n_steps) - int(round(float(tail_ms) / float(dt_ms)))
    keep = steps >= start
    return np.bincount(np.asarray(cells, np.int64)[keep], minlength=int(n_cells)).astype(float)


def population_rate(steps, *, n_steps, n_cells, tail_ms=TAIL_MS, bin_ms=RATE_BIN_MS, dt_ms=0.05):
    steps = np.asarray(steps, np.int64)
    start = int(n_steps) - int(round(float(tail_ms) / float(dt_ms)))
    width = int(round(float(bin_ms) / float(dt_ms)))
    keep = steps >= start
    counts = np.bincount((steps[keep] - start) // width,
                         minlength=int(round(float(tail_ms) / float(bin_ms))))
    return counts.astype(float) / int(n_cells) / (float(bin_ms) / 1000.0)


def phase_aligned_correlation(a, b, *, bin_ms=RATE_BIN_MS, window_ms=PHASE_ALIGN_WINDOW_MS,
                              max_lag_ms=MAX_LAG_MS):
    """Best correlation of two rate trains over a lag search, on the final ``window_ms``.

    Returns ``(r, lag_ms)``.  A limit cycle perturbed or re-seeded returns at a different phase, so a
    zero-lag comparison answers a question nobody asked.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = int(round(float(window_ms) / float(bin_ms)))
    a, b = a[-n:], b[-n:]
    max_lag = int(round(float(max_lag_ms) / float(bin_ms)))
    if a.size <= 2 * max_lag + 2:
        raise ValueError("window too short for the requested lag search")
    best_r, best_lag = -2.0, 0
    reference = b[max_lag:b.size - max_lag]
    for lag in range(-max_lag, max_lag + 1):
        shifted = a[max_lag + lag:a.size - max_lag + lag]
        if np.std(shifted) == 0.0 or np.std(reference) == 0.0:
            continue
        r = float(np.corrcoef(shifted, reference)[0, 1])
        if r > best_r:
            best_r, best_lag = r, lag
    return best_r, best_lag * float(bin_ms)


def coarse_spatial_map(steps, cells, cell_bins, occupancy, *, n_steps, tail_ms=TAIL_MS, dt_ms=0.05):
    """Mean per-bin firing over the final window, on the registered 32x32 grid."""
    steps = np.asarray(steps, np.int64)
    start = int(n_steps) - int(round(float(tail_ms) / float(dt_ms)))
    keep = steps >= start
    bins = np.asarray(cell_bins, np.int64)[np.asarray(cells, np.int64)[keep]]
    counts = np.bincount(bins, minlength=int(np.asarray(occupancy).size)).astype(float)
    occupancy = np.asarray(occupancy, float)
    return np.divide(counts, occupancy, out=np.zeros_like(counts), where=occupancy > 0)


def _relative(a, b):
    scale = max(abs(float(a)), abs(float(b)), 1e-9)
    return abs(float(a) - float(b)) / scale


def _correlation(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def outcome_distance(low, high) -> dict:
    """Compare two runs of one pinned field across four independent readouts.

    Each argument is a mapping with ``final_second_rate_hz``, ``median_active_area_mm2``,
    ``per_cell_rate_vector``, ``population_rate`` and ``coarse_spatial_map``.

    The verdict is deliberately conservative in its wording: passing every readout supports
    ``NO_MACROSCOPIC_INITIALISATION_SPLIT_DETECTED``, which is a statement about what this comparison
    can see, not a proof that no split exists.  Failing any readout yields a candidate, never a
    demonstrated split -- confirming one needs a perturbation-return test and a second input stream.
    """
    rate_rel = _relative(low["final_second_rate_hz"], high["final_second_rate_hz"])
    area_rel = _relative(low["median_active_area_mm2"], high["median_active_area_mm2"])
    per_cell = _correlation(low["per_cell_rate_vector"], high["per_cell_rate_vector"])
    zero_lag = _correlation(low["population_rate"], high["population_rate"])
    aligned, lag_ms = phase_aligned_correlation(low["population_rate"], high["population_rate"])
    spatial = _correlation(low["coarse_spatial_map"], high["coarse_spatial_map"])

    checks = {
        "final_second_rate": rate_rel <= RATE_RELATIVE_TOLERANCE,
        "median_active_area": area_rel <= AREA_RELATIVE_TOLERANCE,
        "per_cell_rate_vector": per_cell >= PER_CELL_CORRELATION_FLOOR,
        "phase_aligned_population_rate": aligned >= PHASE_ALIGNED_CORRELATION_FLOOR,
        "coarse_spatial_map": spatial >= SPATIAL_MAP_CORRELATION_FLOOR,
    }
    same = all(checks.values())
    return {
        "schema": SCHEMA,
        "final_second_rate_relative_difference": rate_rel,
        "median_active_area_relative_difference": area_rel,
        "per_cell_rate_vector_correlation": per_cell,
        "population_rate_zero_lag_correlation": zero_lag,
        "phase_aligned_population_rate_correlation": aligned,
        "phase_alignment_lag_ms": lag_ms,
        "coarse_spatial_map_correlation": spatial,
        "checks": checks,
        "failed_checks": sorted(name for name, ok in checks.items() if not ok),
        "same_outcome_regime": bool(same),
        "verdict": ("NO_MACROSCOPIC_INITIALISATION_SPLIT_DETECTED" if same
                    else "INITIALISATION_SPLIT_CANDIDATE_PENDING_PERTURBATION_AND_SECOND_STREAM"),
        "thresholds": {
            "tail_ms": TAIL_MS, "rate_bin_ms": RATE_BIN_MS,
            "phase_align_window_ms": PHASE_ALIGN_WINDOW_MS, "max_lag_ms": MAX_LAG_MS,
            "rate_relative_tolerance": RATE_RELATIVE_TOLERANCE,
            "area_relative_tolerance": AREA_RELATIVE_TOLERANCE,
            "per_cell_correlation_floor": PER_CELL_CORRELATION_FLOOR,
            "phase_aligned_correlation_floor": PHASE_ALIGNED_CORRELATION_FLOOR,
            "spatial_map_correlation_floor": SPATIAL_MAP_CORRELATION_FLOOR,
        },
        "claim_boundary": (
            "This compares two runs under ONE shared input stream.  Identical or near-identical "
            "outcomes are consistent with a single attractor and also with common-noise "
            "synchronisation of two attractors, which this design cannot separate; the safe reading "
            "is that no macroscopic split was detected under this input realisation."),
    }
