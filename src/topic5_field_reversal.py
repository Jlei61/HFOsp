"""Topic 5 — TA/TB interictal propagation FIELD reversal gate (broad substrate).

Signed, no-mirror test of whether the two interictal templates' smoothed fields are
anti-correlated beyond a within-shaft permutation null, on a subject-fixed shared frame
(the t_a readout plane). See docs/superpowers/specs/2026-07-06-topic5-tatb-field-reversal-design.md
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from src.propagation_contact_plane_readout import _support_corr, S_THRESH, OVERLAP_MIN


def signed_reversal_corr(field0: dict, field1: dict,
                         s_thresh: float = S_THRESH,
                         overlap_min: int = OVERLAP_MIN) -> dict:
    """Signed (identity-orientation, NO y-mirror) support-gated Pearson between two fields.

    field{0,1} = {"T","S"} on the SAME grid/frame. Negative => reversed pair. Returns
    signed_corr (None if unusable), n_overlap, insufficient_overlap (overlap<overlap_min).
    """
    corr, n = _support_corr(field0["T"], field1["T"], field0["S"], field1["S"], s_thresh)
    insufficient = (n < overlap_min) or (not np.isfinite(corr))
    return {"signed_corr": (float(corr) if np.isfinite(corr) else None),
            "n_overlap": int(n), "insufficient_overlap": bool(insufficient)}
