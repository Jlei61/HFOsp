"""Topic 5 — TA/TB interictal propagation FIELD reversal gate (broad substrate).

Signed, no-mirror test of whether the two interictal templates' smoothed fields are
anti-correlated beyond a within-shaft permutation null, on a subject-fixed shared frame
(the t_a readout plane). See docs/superpowers/specs/2026-07-06-topic5-tatb-field-reversal-design.md
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from scipy.stats import spearmanr, binomtest

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


def contact_reversal_gate(cav0: dict, cav1: dict, *, n_perm: int = 1000, rng, min_eff: int = 6) -> dict:
    """No-geometry head-to-head signed Spearman between two per-contact value vectors (over
    contacts finite in both), within-shaft null on cav1 values. passed = not degenerate AND
    left-tail percentile<5 AND signed_spearman<0 (spec §4)."""
    common = [n for n in cav0 if n in cav1
              and np.isfinite(cav0[n]["value"]) and np.isfinite(cav1[n]["value"])]
    v0 = np.array([cav0[n]["value"] for n in common], float)
    v1 = np.array([cav1[n]["value"] for n in common], float)
    eff = int(effective_shuffle_n(common, None, "within_shaft"))
    degenerate = eff < min_eff or len(common) < 3
    obs = float(spearmanr(v0, v1).correlation) if len(common) >= 3 else float("nan")
    base = {"signed_spearman": obs, "effective_n": eff, "degenerate_null": bool(degenerate),
            "percentile": float("nan"), "null_p05": float("nan"), "null_p50": float("nan"),
            "null_p95": float("nan"), "passed": False}
    if degenerate or not np.isfinite(obs):
        return base
    null = np.array([spearmanr(v0, within_shaft_shuffle(v1, common, rng)).correlation
                     for _ in range(n_perm)], float)
    place = placement_in_distribution(obs, null)
    pcts = np.nanpercentile(null, [5, 50, 95])
    base.update({"percentile": place["percentile"],
                 "null_p05": float(pcts[0]), "null_p50": float(pcts[1]), "null_p95": float(pcts[2]),
                 "passed": bool(place["percentile"] < 5.0 and obs < 0.0)})
    return base


def _loo_field_predict(names, plane_xy, values, support, sigma):
    """LOO kernel regression at each contact from OTHER contacts. Returns (pred, den) where
    den = support-weighted kernel mass at the contact (the field support at that location,
    directly comparable to S_THRESH — the caller NaNs contacts with den < s_thresh)."""
    pts = np.array([plane_xy[n] for n in names], float)
    v = np.array([values[n] for n in names], float)
    sup = np.array([support[n] for n in names], float)
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    W = sup[None, :] * np.exp(-d2 / (2.0 * sigma ** 2))
    np.fill_diagonal(W, 0.0)                      # LOO: exclude self
    den = W.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        pred = np.where(den > 1e-12, (W @ v) / den, np.nan)
    return pred, den


def _class_split_rhos(masked, bools, names, plane_xy, cols, sigma, rng, s_thresh):
    perm = rng.permutation(cols); half = perm.size // 2
    if half < 1:
        return None
    a, b = perm[:half], perm[half:]
    with np.errstate(invalid="ignore"):
        train = np.array([np.nanmean(masked[c, a]) if np.any(np.isfinite(masked[c, a])) else np.nan
                          for c in range(len(names))])
        held = np.array([np.nanmean(masked[c, b]) if np.any(np.isfinite(masked[c, b])) else np.nan
                         for c in range(len(names))])
    train_part = np.asarray(bools)[:, a].mean(axis=1)         # per-contact participation on train half
    order = [names[i] for i in range(len(names))
             if names[i] in plane_xy and np.isfinite(train[i]) and train_part[i] > 0]
    if len(order) < 3:
        return None
    tv = {n: float(train[names.index(n)]) for n in order}
    sup = {n: float(train_part[names.index(n)]) for n in order}       # participation weight, NOT 1.0
    pred, den = _loo_field_predict(order, plane_xy, tv, sup, sigma)
    # S_THRESH gate: spatially-isolated contacts (low kernel mass) -> NaN, so field & contact
    # are scored on the SAME definable-contact intersection (spec §6 common-support fairness).
    loo_by = {n: (float(pred[j]) if den[j] >= s_thresh else np.nan) for j, n in enumerate(order)}
    common = [n for n in order if np.isfinite(held[names.index(n)]) and np.isfinite(loo_by[n])]
    if len(common) < 3:
        return None
    hv = np.array([held[names.index(n)] for n in common])
    cv = np.array([tv[n] for n in common])                            # contact scored on SAME common set
    fv = np.array([loo_by[n] for n in common])
    return (float(spearmanr(cv, hv).correlation), float(spearmanr(fv, hv).correlation), len(common))


def loo_reproducibility(bundle, plane_ref, *, n_split=50, rng, sigma, s_thresh=S_THRESH) -> dict:
    X, Y = make_plane_grid()
    if sigma is None:
        sigma = class_template_sigma(plane_ref, X=X, Y=Y)
    masked = bundle["masked"]; bools = np.asarray(bundle["bools"]); names = list(bundle["channel_names"])
    labels = np.asarray(bundle["labels"]); plane_xy = build_plane_xy(plane_ref)
    c_rhos, f_rhos, ncs = [], [], []
    for g in (0, 1):
        cols = np.where(labels == g)[0]
        for _ in range(n_split):
            r = _class_split_rhos(masked, bools, names, plane_xy, cols, sigma, rng, s_thresh)
            if r is not None:
                c_rhos.append(r[0]); f_rhos.append(r[1]); ncs.append(r[2])
    return {"contact_rho": float(np.nanmean(c_rhos)) if c_rhos else float("nan"),
            "field_rho": float(np.nanmean(f_rhos)) if f_rhos else float("nan"),
            "n_contacts_common": int(np.median(ncs)) if ncs else 0}


def cohort_binomial(pass_flags: Sequence[bool]) -> dict:
    """One-sided binomial test of pass count vs 0.05 expected rate over non-degenerate subjects.

    pass_flags: sequence of boolean pass/fail flags.
    Returns: {"n": n_subjects, "k": n_passes, "p_binom": float} where n=0 → p_binom=NaN.
    Test is one-sided `greater` against the null p=0.05.
    """
    flags = [bool(x) for x in pass_flags]
    n = len(flags)
    k = int(sum(flags))
    if n == 0:
        return {"n": 0, "k": 0, "p_binom": float("nan")}
    return {"n": n, "k": k, "p_binom": float(binomtest(k, n, 0.05, alternative="greater").pvalue)}
