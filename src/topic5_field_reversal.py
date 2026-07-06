"""Topic 5 — TA/TB interictal propagation FIELD reversal gate (broad substrate).

Signed, no-mirror test of whether the two interictal templates' smoothed fields are
anti-correlated beyond a within-shaft permutation null, on a subject-fixed shared frame
(the t_a readout plane). See docs/superpowers/specs/2026-07-06-topic5-tatb-field-reversal-design.md
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from src.propagation_contact_plane_readout import (
    _support_corr, S_THRESH, OVERLAP_MIN, make_plane_grid, placement_in_distribution)
from src.topic5_event_resolved_alignment import (
    field_from_contact_values, class_template_sigma, build_plane_xy)
from src.topic5_axis_alignment import within_shaft_shuffle, channel_shuffle, effective_shuffle_n


def signed_reversal_corr(field0: dict, field1: dict,
                         s_thresh: float = S_THRESH,
                         overlap_min: int = OVERLAP_MIN) -> dict:
    """Signed (identity-orientation, NO y-mirror) support-gated Pearson between two fields.

    field{0,1} = {"T","S"} on the SAME grid/frame. Negative => reversed pair. Returns
    signed_corr (None if unusable), n_overlap, insufficient_overlap (overlap<overlap_min).
    """
    if field0 is None or field1 is None:
        return {"signed_corr": None, "n_overlap": 0, "insufficient_overlap": True}
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


def _perm_cav(cav: dict, names_used: Sequence[str], perm_values: np.ndarray) -> dict:
    """cav with values on names_used replaced by perm_values (support/others untouched)."""
    out = dict(cav)
    for n, v in zip(names_used, perm_values):
        out[n] = {"value": float(v), "support": cav[n]["support"]}
    return out


def within_shaft_reversal_gate(plane_ref, cav0, cav1, *, X, Y, sigma=None,
                               n_perm=1000, rng, min_eff=6,
                               s_thresh=S_THRESH, overlap_min=OVERLAP_MIN) -> dict:
    """Primary null: is the observed TA/TB signed reversal corr beyond a within-shaft
    permutation of TB's (cav1) VALUES (support/coords/names fixed)? degenerate_null if too
    few channels are actually permutable (effective_shuffle_n on names_used < min_eff).
    passed = not degenerate AND left-tail percentile<5 AND signed_corr<0 (spec §4/§12)."""
    if X is None or Y is None:
        X, Y = make_plane_grid()
    built = build_reversal_fields(plane_ref, cav0, cav1, X=X, Y=Y, sigma=sigma, s_thresh=s_thresh)
    names_used = built["names_used"]
    obs = signed_reversal_corr(built["field0"], built["field1"], s_thresh, overlap_min)
    eff = int(effective_shuffle_n(names_used, None, "within_shaft"))
    degenerate = eff < min_eff
    base = {"signed_corr": obs["signed_corr"], "n_overlap": obs["n_overlap"],
            "insufficient_overlap": obs["insufficient_overlap"],
            "effective_n": eff, "degenerate_null": bool(degenerate),
            "sigma": built["sigma"], "null_corrs": [],
            "null_p05": float("nan"), "null_p50": float("nan"), "null_p95": float("nan"),
            "percentile": float("nan"), "passed": False}
    if degenerate or obs["insufficient_overlap"] or obs["signed_corr"] is None:
        return base
    vals1 = np.array([cav1[n]["value"] for n in names_used], float)
    null = []
    for _ in range(n_perm):
        perm = within_shaft_shuffle(vals1, names_used, rng)
        cav1p = _perm_cav(cav1, names_used, perm)
        fp = build_reversal_fields(plane_ref, cav0, cav1p, X=X, Y=Y,
                                   sigma=built["sigma"], s_thresh=s_thresh)
        r = signed_reversal_corr(fp["field0"], fp["field1"], s_thresh, overlap_min)
        if r["signed_corr"] is not None:
            null.append(r["signed_corr"])
    null = np.asarray(null, float)
    place = placement_in_distribution(obs["signed_corr"], null)   # percentile = %(null < obs)
    pcts = np.nanpercentile(null, [5, 50, 95]) if null.size else [np.nan, np.nan, np.nan]
    base.update({"null_corrs": null.tolist(),
                 "null_p05": float(pcts[0]), "null_p50": float(pcts[1]), "null_p95": float(pcts[2]),
                 "percentile": place["percentile"],
                 "passed": bool(place["percentile"] < 5.0 and obs["signed_corr"] < 0.0)})
    return base


def _aggregate_over_events(masked: np.ndarray, names: Sequence[str], cols: np.ndarray) -> dict:
    """Per-contact masked-rank mean over the given event columns -> {name:{value,support}}."""
    sub = masked[:, cols]
    with np.errstate(invalid="ignore"):
        val = np.where(np.all(np.isnan(sub), axis=1), np.nan, np.nanmean(sub, axis=1))
    sup = np.isfinite(sub).mean(axis=1)
    return {n: {"value": float(val[i]), "support": float(sup[i])} for i, n in enumerate(names)}


def channel_floor(plane_ref, cav0, cav1, *, X, Y, sigma, n_perm, rng,
                  s_thresh=S_THRESH, overlap_min=OVERLAP_MIN) -> dict:
    """Coarse-floor null: within_shaft_reversal_gate's null shape, but TB's values are
    fully permuted across ALL contacts (channel_shuffle) rather than within-shaft only.
    Answers "is there any coarse shared structure at all", not a same-shaft-controlled test."""
    built = build_reversal_fields(plane_ref, cav0, cav1, X=X, Y=Y, sigma=sigma, s_thresh=s_thresh)
    names = built["names_used"]
    obs = signed_reversal_corr(built["field0"], built["field1"], s_thresh, overlap_min)
    if obs["insufficient_overlap"] or obs["signed_corr"] is None:
        return {"null_corrs": [], "percentile": float("nan"),
                "null_p05": float("nan"), "null_p50": float("nan"), "null_p95": float("nan")}
    vals1 = np.array([cav1[n]["value"] for n in names], float)
    null = []
    for _ in range(n_perm):
        cav1p = _perm_cav(cav1, names, channel_shuffle(vals1, rng))
        fp = build_reversal_fields(plane_ref, cav0, cav1p, X=X, Y=Y, sigma=built["sigma"], s_thresh=s_thresh)
        r = signed_reversal_corr(fp["field0"], fp["field1"], s_thresh, overlap_min)
        if r["signed_corr"] is not None:
            null.append(r["signed_corr"])
    null = np.asarray(null, float)
    place = placement_in_distribution(obs["signed_corr"], null) if null.size else {"percentile": float("nan")}
    pcts = np.nanpercentile(null, [5, 50, 95]) if null.size else [np.nan, np.nan, np.nan]
    return {"null_corrs": null.tolist(), "percentile": place["percentile"],
            "null_p05": float(pcts[0]), "null_p50": float(pcts[1]), "null_p95": float(pcts[2])}


def random_split_contrast(bundle, plane_ref, *, X, Y, sigma, n_split=200, rng,
                          s_thresh=S_THRESH, overlap_min=OVERLAP_MIN) -> dict:
    """Descriptive, NON-INFERENTIAL contrast: split events into 2 random balanced halves,
    ignoring the true A/B labels, and correlate the two halves' fields. Shows that the
    observed (labelled) A/B reversal is not merely an artifact of "splitting events in two" —
    random splits should center positive while the true A/B corr is negative."""
    masked = bundle["masked"]; names = list(bundle["channel_names"]); n_ev = masked.shape[1]
    if sigma is None:
        sigma = class_template_sigma(plane_ref, X=X, Y=Y, s_thresh=s_thresh)
    # observed A/B (true labels)
    labels = np.asarray(bundle["labels"])
    cav0 = _aggregate_over_events(masked, names, np.where(labels == 0)[0])
    cav1 = _aggregate_over_events(masked, names, np.where(labels == 1)[0])
    b = build_reversal_fields(plane_ref, cav0, cav1, X=X, Y=Y, sigma=sigma, s_thresh=s_thresh)
    obs = signed_reversal_corr(b["field0"], b["field1"], s_thresh, overlap_min)["signed_corr"]
    splits = []
    for _ in range(n_split):
        perm = rng.permutation(n_ev); half = n_ev // 2
        ch = _aggregate_over_events(masked, names, perm[:half])
        cl = _aggregate_over_events(masked, names, perm[half:])
        fb = build_reversal_fields(plane_ref, ch, cl, X=X, Y=Y, sigma=sigma, s_thresh=s_thresh)
        r = signed_reversal_corr(fb["field0"], fb["field1"], s_thresh, overlap_min)["signed_corr"]
        if r is not None:
            splits.append(r)
    return {"split_corrs": splits, "split_median": float(np.median(splits)) if splits else float("nan"),
            "observed_ab_corr": (float(obs) if obs is not None else float("nan")),
            "note": "non_inferential"}
