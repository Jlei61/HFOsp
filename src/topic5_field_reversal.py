"""Topic 5 — TA/TB interictal propagation FIELD reversal gate (broad substrate).

Signed, no-mirror test of whether the two interictal templates' smoothed fields are
anti-correlated beyond a within-shaft permutation null, on a subject-fixed shared frame
(the t_a readout plane). See docs/superpowers/specs/2026-07-06-topic5-tatb-field-reversal-design.md
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from src.propagation_contact_plane_readout import _support_corr, S_THRESH, OVERLAP_MIN
from src.topic5_event_resolved_alignment import (
    field_from_contact_values, class_template_sigma, build_plane_xy)


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


def _finite_on_plane(cav: dict, plane_xy: dict) -> list:
    """Return names that are finite in cav AND present with finite coords on plane_xy."""
    return [n for n, d in cav.items()
            if n in plane_xy and d.get("value") is not None and np.isfinite(d["value"])]


def build_reversal_fields(plane_ref: dict, cav0: dict, cav1: dict, *,
                          X, Y, sigma: Optional[float] = None,
                          s_thresh: float = S_THRESH) -> dict:
    """Build TA (cav0) and TB (cav1) fields on the SAME reference plane (P0). Raw class-mean
    values (P1) with per-class participation support; single median-nn sigma for both."""
    if sigma is None:
        sigma = class_template_sigma(plane_ref, X=X, Y=Y, s_thresh=s_thresh)
    plane_xy = build_plane_xy(plane_ref)
    v0 = {n: d["value"] for n, d in cav0.items()}
    s0 = {n: d["support"] for n, d in cav0.items()}
    v1 = {n: d["value"] for n, d in cav1.items()}
    s1 = {n: d["support"] for n, d in cav1.items()}
    field0 = field_from_contact_values(plane_ref, v0, support_by_name=s0,
                                       sigma=sigma, X=X, Y=Y, s_thresh=s_thresh)
    field1 = field_from_contact_values(plane_ref, v1, support_by_name=s1,
                                       sigma=sigma, X=X, Y=Y, s_thresh=s_thresh)
    return {"field0": field0, "field1": field1, "sigma": float(sigma),
            "names_used": _finite_on_plane(cav1, plane_xy)}
