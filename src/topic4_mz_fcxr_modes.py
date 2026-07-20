"""FCXR-RC1 mode-resolved clip audit (reviewer 2026-07-20 §6 Stage A).

Question: is the arm-C recurrent-conductance clip a DOMINANT LOCALIZED spatial mode (a stable set of
high-recurrent-gain / low-Vth / eigenvector-hotspot cells) or a random long tail / discretization artifact?

Pure analysis (no simulation here): reconstructs the E->E weight matrix W_EE from the network, its
recurrent in/out-strength and leading (right/left/singular) modes + IPR, and overlaps those with the
per-cell clip identity recorded by MZSlowVars(record_clip_identity=True).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr


def build_W_EE(net, NE):
    """E->E weight matrix summed over delay bins, oriented (target row, source col), CSR."""
    M = net["max_delay_steps"] + 1
    ampa = net["ampa_by_delay"]
    W = sp.csr_matrix((NE, NE), dtype=np.float64)
    for d in range(M):
        A = ampa[d]
        if A.nnz == 0:
            continue
        W = W + A.tocsr()[:NE, :NE]          # rows=target, cols=source; E->E block
    return W.tocsr()


def strengths(W_EE):
    """in_strength[target] = total incoming E->E weight (row sum); out_strength[source] = col sum."""
    in_s = np.asarray(W_EE.sum(axis=1)).ravel()
    out_s = np.asarray(W_EE.sum(axis=0)).ravel()
    return in_s, out_s


def ipr(v):
    """Inverse participation ratio: 1/N (global, spread) .. 1 (single site, localized)."""
    v = np.abs(np.asarray(v, float)); s2 = float((v * v).sum())
    if s2 <= 0:
        return float("nan")
    return float((v ** 4).sum() / (s2 * s2))


def leading_modes(W_EE, k=6):
    """Leading right/left eigenvectors (largest |eig|) and singular vectors of W_EE + their IPR.
    Non-normal network -> report all three (eigvecs miss transient amplification that singular vecs catch)."""
    from scipy.sparse.linalg import eigs, svds
    out = {}
    try:
        w_r, V_r = eigs(W_EE, k=k, which="LM", maxiter=2000)
        order = np.argsort(-np.abs(w_r))
        w_r, V_r = w_r[order], V_r[:, order]
        out["eig_vals"] = w_r
        out["right_ipr"] = [ipr(V_r[:, i].real) for i in range(V_r.shape[1])]
        out["right_vecs"] = np.abs(V_r.real)
    except Exception as e:  # pragma: no cover
        out["eig_error"] = str(e)
    try:
        w_l, V_l = eigs(W_EE.T.tocsr(), k=k, which="LM", maxiter=2000)
        order = np.argsort(-np.abs(w_l))
        out["left_ipr"] = [ipr(V_l[:, order][:, i].real) for i in range(min(k, V_l.shape[1]))]
        out["left_vecs"] = np.abs(V_l[:, order].real)
    except Exception as e:  # pragma: no cover
        out["left_error"] = str(e)
    try:
        U, s, Vt = svds(W_EE, k=k)
        order = np.argsort(-s)
        out["sing_vals"] = s[order]
        out["sing_ipr_left"] = [ipr(U[:, order][:, i]) for i in range(k)]     # output/receiving side
        out["sing_ipr_right"] = [ipr(Vt[order][i, :]) for i in range(k)]      # input/sending side
        out["sing_left_vecs"] = np.abs(U[:, order])
    except Exception as e:  # pragma: no cover
        out["sing_error"] = str(e)
    return out


def effective_jacobian_modes(W_EE, g_raw, g_sat, k=6):
    """Leading modes of the saturation-weighted recurrent operator diag(sech^2(g_raw/g_sat)) @ W_EE
    (reviewer P1-2): recurrent smooth saturation changes the effective ROW gain even though the connectivity
    W_EE is fixed, so the effective Jacobian's leading eigenvector/IPR can differ from raw W_EE. Compare the
    two IPR to check whether saturation localized/preserved the dominant spatial mode."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigs
    w = 1.0 / np.cosh(np.asarray(g_raw, float) / float(g_sat)) ** 2      # sech^2, in [0,1]; ~1 quiet, <1 active-core
    Jw = sp.diags(w).tocsr() @ W_EE
    out = dict(sech2_min=float(w.min()), sech2_mean=float(w.mean()), sech2_p05=float(np.percentile(w, 5)))
    try:
        wr, Vr = eigs(Jw, k=k, which="LM", maxiter=2000)
        order = np.argsort(-np.abs(wr))
        out["eff_eig_vals_abs"] = np.abs(wr[order])
        out["eff_leading_ipr"] = ipr(Vr[:, order][:, 0].real)
    except Exception as e:  # pragma: no cover
        out["eff_error"] = str(e)
    return out


def _top_overlap(clip_cells, vec, frac=0.02):
    """Jaccard-ish: fraction of clip cells that fall in the top-`frac` loading nodes of `vec`."""
    if clip_cells.sum() == 0:
        return float("nan")
    n_top = max(1, int(frac * vec.size))
    top = np.zeros(vec.size, bool); top[np.argsort(-vec)[:n_top]] = True
    return float(top[clip_cells].mean())          # P(clip cell is a top-loading node)


def clip_mode_audit(*, clip_count, max_raw_gErec, in_strength, out_strength, vth_E, core_mask_E,
                    posE, modes, dt_ms, clip_frames):
    """Overlaps + correlations + spatial spread + verdict for the clip-identity data."""
    clip_count = np.asarray(clip_count); NE = clip_count.size
    clipped = clip_count > 0
    n_clip = int(clipped.sum())
    repeat = clip_count >= max(2, int(np.percentile(clip_count[clipped], 75)) if n_clip else 2)  # persistent clippers

    def _corr(x):
        if n_clip < 3:
            return float("nan")
        r = spearmanr(clip_count.astype(float), np.asarray(x, float))
        return float(r.statistic)

    # spatial spread of the clipping cells (are they a compact cluster or spread over the sheet?)
    spread = {}
    if n_clip:
        p = np.asarray(posE, float)[clipped]
        c = p.mean(axis=0)
        r = np.linalg.norm(p - c, axis=1)
        spread = dict(centroid=[float(c[0]), float(c[1])], rms_radius_mm=float(np.sqrt((r ** 2).mean())),
                      p90_radius_mm=float(np.percentile(r, 90)), n=n_clip)

    ov = {}
    for name in ("right_vecs", "left_vecs", "sing_left_vecs"):
        if name in modes:
            V = modes[name]
            ov[name + "_top2pct_overlap_lead"] = _top_overlap(clipped, V[:, 0])   # leading mode only

    core_base = float(np.asarray(core_mask_E, bool).mean())      # fraction of ALL E cells that are low-Vth core
    core_frac = float(core_mask_E[clipped].mean()) if n_clip else float("nan")
    core_enrich = float(core_frac / core_base) if (n_clip and core_base > 0) else float("nan")
    in_pct = float((in_strength[clipped] > np.percentile(in_strength, 90)).mean()) if n_clip else float("nan")
    # persistence: does a stable subset carry most of the clipping, or is it spread thinly over many one-off cells?
    persistent_share = float(clip_count[repeat].sum() / max(1, clip_count.sum())) if n_clip else float("nan")

    # Verdict (descriptive). In-degree is ~fixed (constant in_strength, global leading eigenvector), so the
    # clip localization is judged by SPATIAL compactness + low-Vth-core enrichment + persistence, not by a
    # localized W_EE eigenvector; the eigenvector overlap is reported as a bonus.
    lead_ipr = modes.get("right_ipr", [float("nan")])[0]
    corr_in = _corr(in_strength)
    compact = bool(spread and spread["p90_radius_mm"] < 4.0)       # < shaft-scale cluster
    core_localized = bool(np.isfinite(core_enrich) and core_enrich >= 3.0)
    persistent = bool(np.isfinite(persistent_share) and persistent_share >= 0.5)
    n_frames = int(len(clip_frames)) if clip_frames is not None else None
    # Localization is judged by core-enrichment + spatial compactness (the strong signals); persistence
    # (same cells vs rotating within the core) is a descriptor, not a gate.
    if n_clip == 0:
        verdict = "no_clip"
    elif core_localized and compact:
        verdict = ("localized_recurrent_mode (low-Vth core, compact cluster) -> real modal problem, "
                   "treat with recurrent smooth saturation")
    elif core_localized and not compact:
        verdict = "core-localized but spatially extended (both cores) -> modal, still saturation-treatable"
    elif compact and not core_localized:
        verdict = "compact non-core cluster -> localized elsewhere, inspect before saturation"
    else:
        verdict = "diffuse / not core-enriched -> more numerical/random-tail than a single dominant mode"
    return dict(
        n_clip_cells=n_clip, n_persistent_clippers=int(repeat.sum()), n_clip_frames=n_frames,
        clip_count_max=int(clip_count.max()), clip_count_total=int(clip_count.sum()),
        max_raw_gErec_over_clip=float(np.max(max_raw_gErec[clipped])) if n_clip else float("nan"),
        spatial=spread,
        corr_clipcount_in_strength=corr_in, corr_clipcount_out_strength=_corr(out_strength),
        corr_clipcount_neg_vth=_corr(-np.asarray(vth_E, float)),
        clip_in_top10pct_in_strength_frac=in_pct, clip_in_low_vth_core_frac=core_frac,
        core_base_frac=core_base, core_enrichment=core_enrich, persistent_share=persistent_share,
        leading_right_ipr=float(lead_ipr) if np.isfinite(lead_ipr) else None,
        leading_mode_overlaps=ov, verdict=verdict)
