"""Topic 4 M3 Layer-2 model-vs-data acceptance (pre-registered tolerance band).

A spatial-extent model emits two per-event summary metrics:

- ``AF`` = axial_fraction   — how far along the propagation axis an event
  spreads (~0..1).
- ``LR`` = lateral_ratio    — sideways spread divided by axial spread.

We do NOT accept the model on a "not-rejected" p-value (which false-passes
under low power). Instead we pre-register a tolerance band from the REAL
per-subject medians and accept only when the MODEL's per-subject medians fall
inside that band — a TOST-style equivalence decision. A separate ``min_af``
gate additionally rejects a "short axial footprint" model even if its AF
median happens to sneak into the band.

Pure numpy; no I/O, no plotting.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def subject_tolerance_band(
    ref_per_subject_af: Sequence[float],
    ref_per_subject_lr: Sequence[float],
    q: Sequence[float] = (10, 90),
) -> Dict[str, float]:
    """Pre-registered tolerance band from real per-subject medians.

    Parameters
    ----------
    ref_per_subject_af, ref_per_subject_lr
        Arrays of REAL per-subject median AF / LR (one value per real subject).
    q
        Lower/upper percentile pair defining the band edges (default 10th/90th).

    Returns
    -------
    dict
        ``af_lo``/``af_hi``/``lr_lo``/``lr_hi`` = the ``q[0]``/``q[1]``
        percentiles (``np.percentile``) of the real per-subject medians.
    """
    af = np.asarray(ref_per_subject_af, dtype=float)
    lr = np.asarray(ref_per_subject_lr, dtype=float)
    q_lo, q_hi = float(q[0]), float(q[1])
    return {
        "af_lo": float(np.percentile(af, q_lo)),
        "af_hi": float(np.percentile(af, q_hi)),
        "lr_lo": float(np.percentile(lr, q_lo)),
        "lr_hi": float(np.percentile(lr, q_hi)),
    }


def layer2_equivalence(
    model_subject_af: Sequence[float],
    model_subject_lr: Sequence[float],
    band: Dict[str, float],
    min_af: float = 0.75,
) -> Dict[str, object]:
    """Layer-2 subject-level equivalence decision against a pre-registered band.

    Parameters
    ----------
    model_subject_af, model_subject_lr
        Arrays of the MODEL's per-subject (per network-realization / seed)
        median AF / LR.
    band
        Output of :func:`subject_tolerance_band`.
    min_af
        Minimum acceptable AF median. The gate REJECTS a "short axial
        footprint" model even if its AF median lands inside the band.

    Returns
    -------
    dict
        ``pass_`` is True only when both medians fall inside the band AND the
        AF median clears ``min_af``. ``af_margin``/``lr_margin`` are signed
        distances to the nearest band edge (positive = inside); descriptive
        only, they do NOT gate the decision.
    """
    af_median = float(np.median(np.asarray(model_subject_af, dtype=float)))
    lr_median = float(np.median(np.asarray(model_subject_lr, dtype=float)))

    af_in_band = bool(band["af_lo"] <= af_median <= band["af_hi"])
    lr_in_band = bool(band["lr_lo"] <= lr_median <= band["lr_hi"])

    pass_ = bool(af_in_band and lr_in_band and (af_median >= min_af))

    # Signed distance to the nearest band edge: positive inside, negative
    # outside. Descriptive only — does not gate.
    af_margin = float(min(af_median - band["af_lo"], band["af_hi"] - af_median))
    lr_margin = float(min(lr_median - band["lr_lo"], band["lr_hi"] - lr_median))

    note = (
        "PASS means the model per-subject medians fall within the "
        "pre-registered real-subject tolerance band AND AF is not short "
        "(>= min_af); it is NOT a non-significant p-value."
    )

    return {
        "pass_": pass_,
        "af_in_band": af_in_band,
        "lr_in_band": lr_in_band,
        "af_median": af_median,
        "lr_median": lr_median,
        "af_margin": af_margin,
        "lr_margin": lr_margin,
        "min_af": float(min_af),
        "note": note,
    }
