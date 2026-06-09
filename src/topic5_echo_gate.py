"""Topic 5 Stage-1 ictal-template-echo gate (proxy triage).

PURE math, no file I/O. Spec:
docs/superpowers/specs/2026-06-08-topic5-ictal-template-echo-gate-design.md (v4)

Phantom-safety (§3.6): templates arrive masked — non-participating channels are
NaN. Every Spearman here drops NaN on either side, so a phantom can never enter
the correlation. "full participating-channel set" = both-finite intersection.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import rankdata


def _spearman_fast(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho = Pearson of average-ranks. Value-identical to scipy.spearmanr
    (verified incl. ties) but ~6x faster — the null loop calls this millions of times."""
    ra = rankdata(a)
    rb = rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra @ rb) / denom) if denom > 0 else float("nan")


def spearman_common(rank_a, rank_b, *, min_ch: int) -> float:
    """Spearman rho on channels finite in BOTH vectors; NaN if fewer than min_ch."""
    a = np.asarray(rank_a, dtype=float)
    b = np.asarray(rank_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"rank vectors must align by channel index: {a.shape} vs {b.shape}")
    common = np.isfinite(a) & np.isfinite(b)
    if int(common.sum()) < min_ch:
        return float("nan")
    rho = _spearman_fast(a[common], b[common])
    return float(rho) if np.isfinite(rho) else float("nan")


def echo_r_obs(seizure_rank, template_ranks: Sequence, *, min_ch: int) -> float:
    """max_m Spearman(seizure, template_m) over templates with enough overlap; NaN if none.

    k handling (§4.1): k=1 -> the single rho; k=2 -> max(rho_a, rho_b); k>2 -> max over k.
    """
    rhos = []
    for t in template_ranks:
        r = spearman_common(seizure_rank, t, min_ch=min_ch)
        if np.isfinite(r):
            rhos.append(r)
    if not rhos:
        return float("nan")
    return float(max(rhos))


from collections import defaultdict

_NULL_MODE_KIND = {
    "channel": "within",
    "within_shaft": "within",
    "anchor_matched": "within",
    "shaft_block": "between",
}


def _unique_blocks(blocks: np.ndarray):
    """Unique block labels, dropping None/NaN. Uses a set (not np.unique) so a mix of
    int bins and None (anchor_bins) or None shaft ids (unparseable names) never trips
    np.unique's int-vs-None sort."""
    seen = []
    for b in blocks.tolist():
        if b is None:
            continue
        if isinstance(b, float) and b != b:   # NaN
            continue
        if b not in seen:
            seen.append(b)
    return seen


def _block_permute(values: np.ndarray, blocks: np.ndarray, kind: str, rng) -> np.ndarray:
    """within: permute values inside each block. between: swap whole equal-size blocks.
    Channels whose block label is None/NaN stay put (never permuted)."""
    values = np.asarray(values, dtype=float)
    blocks = np.asarray(blocks, dtype=object)
    out = values.copy()
    uniq = _unique_blocks(blocks)
    if kind == "within":
        for b in uniq:
            idx = np.where(blocks == b)[0]
            out[idx] = values[idx][rng.permutation(len(idx))]
    elif kind == "between":
        by_size = defaultdict(list)
        for b in uniq:
            idx = np.where(blocks == b)[0]
            by_size[len(idx)].append(idx)
        for size, idx_list in by_size.items():
            if len(idx_list) < 2:
                continue
            src_vals = [values[ix].copy() for ix in idx_list]
            perm = rng.permutation(len(idx_list))
            for tgt_pos, src_pos in enumerate(perm):
                out[idx_list[tgt_pos]] = src_vals[src_pos]
    else:
        raise ValueError(f"unknown kind {kind}")
    return out


def shuffle_null(seizure_rank, template_ranks, *, B, rng, null_mode, min_ch, blocks=None):
    """Null distribution of echo_r_obs under the requested channel-label shuffle (§4.6)."""
    kind = _NULL_MODE_KIND[null_mode]
    n = len(np.asarray(seizure_rank))
    if null_mode == "channel":
        blk = np.zeros(n, dtype=int)
    else:
        if blocks is None:
            raise ValueError(f"null_mode={null_mode} requires blocks")
        blk = np.asarray(blocks)
    out = np.empty(B, dtype=float)
    for i in range(B):
        shuf = _block_permute(np.asarray(seizure_rank, float), blk, kind, rng)
        out[i] = echo_r_obs(shuf, template_ranks, min_ch=min_ch)
    return out


def shaft_block_capacity(blocks) -> Dict:
    """How many channels CAN be block-exchanged (shafts sharing a length with >=1
    other shaft). Unequal shafts are NOT exchangeable and stay put (§4.6 P1-B).
    insufficient_block_exchange=True -> shaft_block null is degenerate; fail closed."""
    blocks = np.asarray(blocks, dtype=object)
    uniq = _unique_blocks(blocks)
    sizes = defaultdict(list)
    for b in uniq:
        sizes[int(np.sum(blocks == b))].append(b)
    n_exch = sum(size * len(grp) for size, grp in sizes.items() if len(grp) >= 2)
    total = sum(int(np.sum(blocks == b)) for b in uniq)
    return {"n_exchangeable_channels": int(n_exch), "n_total_channels": int(total),
            "insufficient_block_exchange": bool(n_exch < 2)}


def compute_echo_strength(seizure_rank, template_ranks, *, B, rng, min_ch,
                          null_mode="channel", blocks=None) -> Dict:
    """Standardized echo strength e_k = (r_obs - null_mean)/null_sd, plus one-sided
    percentile p_k, null quantiles, and e_k_baddata (P0-C: a REAL held-out null draw
    used as a fake observation — pools NS by construction, see bad_data_regression)."""
    r_obs = echo_r_obs(seizure_rank, template_ranks, min_ch=min_ch)
    if not np.isfinite(r_obs):
        return {"e_k": float("nan"), "p_k": float("nan"), "r_obs": float("nan"),
                "e_k_baddata": float("nan"), "null_mean": float("nan"),
                "null_sd": float("nan"), "null_q": [float("nan")] * 3,
                "n_null": 0, "null_mode": null_mode}
    null = shuffle_null(seizure_rank, template_ranks, B=B, rng=rng,
                        null_mode=null_mode, min_ch=min_ch, blocks=blocks)
    null = null[np.isfinite(null)]
    if null.size < 2:
        return {"e_k": float("nan"), "p_k": float("nan"), "r_obs": float(r_obs),
                "e_k_baddata": float("nan"), "null_mean": float("nan"),
                "null_sd": float("nan"), "null_q": [float("nan")] * 3,
                "n_null": int(null.size), "null_mode": null_mode}
    sd = float(null.std(ddof=1))
    e_k = float((r_obs - null.mean()) / sd) if sd > 0 else float("nan")
    p_k = float((np.sum(null >= r_obs) + 1) / (null.size + 1))
    # P0-C: one REAL null draw standardized against the REST of the null. E[.]~0.
    if null.size >= 3:
        rest = null[1:]
        rsd = float(rest.std(ddof=1))
        e_k_baddata = float((null[0] - rest.mean()) / rsd) if rsd > 0 else float("nan")
    else:
        e_k_baddata = float("nan")
    return {"e_k": e_k, "p_k": p_k, "r_obs": float(r_obs),
            "e_k_baddata": e_k_baddata,
            "null_mean": float(null.mean()), "null_sd": sd,
            "null_q": [float(q) for q in np.quantile(null, [0.05, 0.5, 0.95])],
            "n_null": int(null.size), "null_mode": null_mode}


def loo_anchor(per_seizure_ranks) -> np.ndarray:
    """r_bar_{c,-k}: per-channel mean rank over OTHER seizures (current k excluded).

    Leave-one-seizure-out — the current seizure never enters its own anchor (§4.1b
    leakage fix). Channels NaN across all other seizures stay NaN.
    """
    M = np.asarray(per_seizure_ranks, dtype=float)
    ns = M.shape[0]
    out = np.full_like(M, np.nan)
    for k in range(ns):
        others = np.delete(M, k, axis=0)
        with np.errstate(invalid="ignore"):
            col_mean = np.where(np.all(np.isnan(others), axis=0), np.nan,
                                np.nanmean(others, axis=0))
        out[k] = col_mean
    return out


def compute_deanchor_echo(per_seizure_ranks, template_ranks, *, B, rng, min_ch) -> List[Dict]:
    """Echo on de-anchored deltas, keeping max-over-templates (P1-A — same contract as
    §4.1, so k=2 subjects are not arbitrarily dominated by template 0):
    delta_seiz = seiz_k - r_bar_{-k}; delta_templ_m = template_m - r_bar_{-k}.
    """
    M = np.asarray(per_seizure_ranks, dtype=float)
    templs = [np.asarray(t, dtype=float) for t in template_ranks]
    anc = loo_anchor(M)
    records = []
    for k in range(M.shape[0]):
        d_seiz = M[k] - anc[k]
        d_templs = [t - anc[k] for t in templs]      # de-anchor EACH template
        records.append(compute_echo_strength(d_seiz, d_templs, B=B, rng=rng, min_ch=min_ch))
    return records


def anchor_reliability(per_seizure_ranks) -> float:
    """Kendall's W (coefficient of concordance) across seizures over channels."""
    M = np.asarray(per_seizure_ranks, dtype=float)
    valid = ~np.any(np.isnan(M), axis=0)
    if valid.sum() < 3 or M.shape[0] < 2:
        return float("nan")
    sub = M[:, valid]
    from scipy.stats import rankdata
    R = np.vstack([rankdata(row) for row in sub])         # n_seiz x n_ch ranks
    n, m = R.shape                                        # n raters, m items
    Rj = R.sum(axis=0)
    S = np.sum((Rj - Rj.mean()) ** 2)
    W = 12 * S / (n ** 2 * (m ** 3 - m))
    return float(W)


def pool_echo_subject_level(records, *, n_boot: int = 2000, seed: int = 0) -> Dict:
    """Subject-level primary pooling (§4.1.4). records: list of {subject, e_k}.
    E_s = mean over a subject's finite e_k. One-sided (>0) Wilcoxon signed-rank +
    sign test + bootstrap CI on median(E_s). Statistical unit is the SUBJECT."""
    from scipy.stats import wilcoxon, binomtest
    by_subj = defaultdict(list)
    for r in records:
        if r.get("e_k") is not None and np.isfinite(r["e_k"]):
            by_subj[r["subject"]].append(float(r["e_k"]))
    Es = np.array([np.mean(v) for v in by_subj.values() if v], dtype=float)
    n = int(Es.size)
    out = {"n_subjects": n, "E_s": Es.tolist(),
           "median_E_s": float(np.median(Es)) if n else float("nan"),
           "mean_E_s": float(np.mean(Es)) if n else float("nan")}
    if n < 2 or np.allclose(Es, 0):
        out["wilcoxon_p_onesided"] = float("nan")
        out["sign_p_onesided"] = float("nan")
        out["boot_ci95"] = [float("nan"), float("nan")]
        return out
    try:
        out["wilcoxon_p_onesided"] = float(wilcoxon(Es, alternative="greater").pvalue)
    except ValueError:
        out["wilcoxon_p_onesided"] = float("nan")
    n_pos = int(np.sum(Es > 0))
    out["sign_p_onesided"] = float(binomtest(n_pos, n, 0.5, alternative="greater").pvalue)
    rng = np.random.default_rng(seed)
    meds = [np.median(rng.choice(Es, size=n, replace=True)) for _ in range(n_boot)]
    out["boot_ci95"] = [float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))]
    return out


def bad_data_regression(echo_records) -> Dict:
    """P0-C: pool the e_k_baddata field (each a REAL null draw used as a fake
    observation) exactly like the primary pool. Must come out non-significant; a
    significant result means the pooling machinery manufactures signal -> stop & fix.
    echo_records carry 'subject' + 'e_k_baddata'."""
    recs = [{"subject": r["subject"], "e_k": r.get("e_k_baddata")} for r in echo_records]
    return pool_echo_subject_level(recs)


def compute_atlas_quality(ictal_rank, *, tie_max: float, min_channels: int) -> Dict:
    """Atlas-quality flags (§3.5): does the ER-derived ictal rank carry ORDER info?
    (tie fraction, dynamic range, channel count). This does NOT verify the order is
    propagation — that is the separate construct-validity sentinel (runner, P1-2)."""
    r = np.asarray(ictal_rank, dtype=float)
    finite = r[np.isfinite(r)]
    n = int(finite.size)
    if n == 0:
        return {"atlas_quality_flag": "fail", "rank_tie_fraction": float("nan"),
                "rank_dynamic_range": 0.0, "n_ranked_channels": 0}
    _, counts = np.unique(finite, return_counts=True)
    tie_frac = float(np.sum(counts[counts > 1]) / n)
    dyn = float(finite.max() - finite.min())
    ok = (n >= min_channels) and (tie_frac <= tie_max) and (dyn > 0)
    return {"atlas_quality_flag": "pass" if ok else "fail",
            "rank_tie_fraction": tie_frac, "rank_dynamic_range": dyn,
            "n_ranked_channels": n}


def between_subject_control(seizure_rank, this_channels, foreign_named_templates, *,
                            B, rng, min_ch, null_mode="channel", blocks=None) -> Dict:
    """Null D (§4.6): echo of this seizure against OTHER subjects' templates, NAME-
    ALIGNED onto this subject's channels (apples-to-apples with the name-aligned
    primary — NOT positional truncation). A foreign template contributes its rank only
    at channels whose NAME this subject also has; this subject's other channels are NaN.

    this_channels: list[str] aligned to seizure_rank.
    foreign_named_templates: list of (rank_seq, channel_names_seq) from OTHER subjects.

    Under generic-echo scope (spec v4 §3.2) the cohort pool of this is the FORMAL
    negative control: it should be neutral. If it pools significant, the echo is
    anatomy-general (not subject-specific) -> primary not specific."""
    idx = {c: i for i, c in enumerate(this_channels)}
    n = len(this_channels)
    remapped = []
    for f_rank, f_channels in foreign_named_templates:
        v = np.full(n, np.nan)
        for fr, fc in zip(f_rank, f_channels):
            fr = float(fr) if fr is not None else np.nan
            if fc in idx and np.isfinite(fr):
                v[idx[fc]] = fr
        if np.sum(np.isfinite(v)) >= min_ch:      # only foreign templates that actually overlap
            remapped.append(v)
    if not remapped:
        return {"e_k": float("nan"), "p_k": float("nan"), "r_obs": float("nan"),
                "e_k_baddata": float("nan"), "null_mean": float("nan"),
                "null_sd": float("nan"), "null_q": [float("nan")] * 3,
                "n_null": 0, "null_mode": null_mode, "n_foreign_overlapping": 0}
    res = compute_echo_strength(seizure_rank, remapped, B=B, rng=rng, min_ch=min_ch,
                                null_mode=null_mode, blocks=blocks)
    res["n_foreign_overlapping"] = len(remapped)
    return res


def masked_template_rank_1d(agg_rank, valid_mask):
    """P0-A contract #1: a 1-D ALREADY-AGGREGATED template rank + per-cluster valid_mask.
    Mask with np.where — do NOT call mask_phantom_ranks (that helper is a 2-D
    (n_ch, n_ev) per-event re-ranker; passing 1-D is wrong)."""
    r = np.asarray(agg_rank, dtype=float)
    m = np.asarray(valid_mask, dtype=bool)
    if r.shape != m.shape:
        raise ValueError(f"agg_rank {r.shape} != valid_mask {m.shape}")
    return np.where(m, r, np.nan)


def rebuild_template_from_events(raw_ranks_2d, bools_2d):
    """P0-A contract #2: event-level (n_ch, n_ev) raw ranks + bools. Run
    mask_phantom_ranks (per-event re-rank, phantom discarded -> NaN), then aggregate
    to a 1-D template by nanmean across events. Channel never participating -> NaN."""
    from src.lagpat_rank_audit import mask_phantom_ranks
    masked = np.asarray(mask_phantom_ranks(np.asarray(raw_ranks_2d, float),
                                           np.asarray(bools_2d, bool)), dtype=float)
    with np.errstate(invalid="ignore"):
        templ = np.where(np.all(np.isnan(masked), axis=1), np.nan,
                         np.nanmean(masked, axis=1))
    return templ
