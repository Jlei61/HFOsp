"""Propagation entry-dispersion analysis (real data).

Scientific question (SEF-ITP "propagation channel attractor" framing, Topic 4
on Topic 1 PR-2/PR-2.5 template clusters)
------------------------------------------------------------------------------
Re-frame the narrative from "two fixed endpoints repeatedly ignite" to "events
enter one stable propagation channel from a small entry region". The handoff's
naive success criterion was: *earliest channel dispersed + downstream order
stable -> supports an entry region*.

That criterion is NOT discriminating. A SINGLE noisy propagation template per
cluster produces BOTH effects automatically: the earliest channel is an
``argmin`` (an extreme order statistic, the most jitter-sensitive position), so
it is dispersed even when there is exactly one template; and the downstream
order is, by construction, stable. Beating a fixed-endpoint baseline (Neff~=1)
or a rank-shuffle null therefore proves nothing.

The LOAD-BEARING test implemented here (advisor 2026-06-08) is whether observed
entry dispersion EXCEEDS a single-template-plus-calibrated-noise null. Only an
excess supports an entry region beyond one noisy template. We also report the
entry *shape* (monotonic decay along template rank = one noisy template; a
secondary peak / spatial offset = a distinct entry mode) and whether the entry
identity predicts the downstream order (it should NOT, if events funnel into one
channel). The post-entry stability leg is reported raw AND centered, but is
explicitly NON-load-bearing: it passes near-trivially because ~92% of
within-cluster ordering is fixed channel identity (Topic 0 §3.1 / Topic 1 §3.2).

Contract clauses honored (CLAUDE.md §6; see also AGENTS.md cross-PR lookups):
  (1) PHANTOM-MASK   : earliest channel is derived from ``mask_phantom_ranks``
                       (re-rank among PARTICIPATING channels only); raw
                       ``lagPatRank`` is never argmin'd directly.
  (2) CHANNEL-ALIGN  : caller must pass channel_names matching the loader union
                       order; coords requested in that same order.
  (3) LABEL-ALIGN    : per-valid-event labels align to
                       ``_valid_event_indices(bools, min_participating=3)``.
  (4) PER-CLUSTER    : every metric is computed per cluster; forward/reverse
                       clusters are never pooled.
  (5) NULL           : single-template-noise = per-channel template mean
                       (masked normalized rank, participating only) + POOLED
                       residual resample on the SAME per-event participation
                       masks; earliest = argmin among participating; directional
                       p = P(null >= observed) [observed MORE dispersed => entry
                       region beyond one noisy template].
  (6) CENTERED-ALONG : post-entry stability reports raw AND centered tau.
  (7) COORD-MASK     : unmapped channels dropped; n_mapped reported; NaN when
                       too few; coord NaN is never coerced to 0.
  (8) DATASET-STRAT  : this module computes per-subject scalars only; the runner
                       must never pool Epilepsiae (MNI) and Yuquan (native RAS)
                       coordinates into one point cloud.
  (9) MIN-EVENTS     : null / spatial / downstream metrics are NaN-flagged for
                       clusters below the event/group minimums.

The earliest channel is an ARRIVAL LEADER (earliest *detected* participant), NOT
a physiological source. Do not relabel it "source".
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import spearmanr

from src.lagpat_rank_audit import mask_phantom_ranks
from src.interictal_propagation import (
    _valid_event_indices,
    _multi_seed_tau_summary,
    _center_rank_matrix,
)

MIN_PARTICIPATING = 3          # matches PR-2 min_shared_channels (LABEL-ALIGN)
MIN_CLUSTER_EVENTS = 50        # below this, null is unstable -> NaN (MIN-EVENTS)
MIN_MAPPED_FOR_RADIUS = 2      # need >=2 mapped earliest contacts for a spread
MIN_GROUP_EVENTS = 50          # min events for a downstream-invariance group


# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------
def earliest_channel_per_event(masked_norm_ranks: np.ndarray) -> np.ndarray:
    """Earliest participating channel index per event.

    Parameters
    ----------
    masked_norm_ranks : (n_ch, n_ev) float
        Output of ``mask_phantom_ranks(normalize=...)``: non-participating
        cells are NaN, participating cells carry within-event ranks. Smaller
        rank = earlier (rank 0 = earliest). (1) PHANTOM-MASK.

    Returns
    -------
    earliest : (n_ev,) int
        argmin over participating channels. Events with no participant get -1.
    """
    score = np.where(np.isnan(masked_norm_ranks), np.inf, masked_norm_ranks)
    earliest = score.argmin(axis=0)
    no_part = ~np.isfinite(score).any(axis=0)
    earliest[no_part] = -1
    return earliest


def effective_number(earliest: np.ndarray, n_ch: int) -> Dict[str, float]:
    """Effective number of distinct earliest channels = exp(Shannon entropy)."""
    e = earliest[earliest >= 0]
    if e.size == 0:
        return {"neff": float("nan"), "top_share": float("nan"), "n": 0}
    counts = np.bincount(e, minlength=n_ch).astype(float)
    counts = counts[counts > 0]
    p = counts / counts.sum()
    H = float(-(p * np.log(p)).sum())
    return {"neff": float(np.exp(H)), "top_share": float(p.max()), "n": int(e.size)}


def earliest_prob_vector(earliest: np.ndarray, n_ch: int) -> np.ndarray:
    """Per-channel probability of being the earliest channel in the cluster."""
    e = earliest[earliest >= 0]
    counts = np.bincount(e, minlength=n_ch).astype(float)
    s = counts.sum()
    return counts / s if s > 0 else counts


def spatial_radius(
    prob: np.ndarray, coords: np.ndarray, mapped: np.ndarray
) -> Dict[str, float]:
    """Probability-weighted spatial dispersion of the entry set (mm).

    (7) COORD-MASK: restrict to mapped channels, renormalize the entry
    probability over them, compute the weighted centroid and the
    probability-weighted mean distance to that centroid. NaN if too few mapped.
    """
    use = mapped & (prob > 0)
    if int(use.sum()) < MIN_MAPPED_FOR_RADIUS:
        return {"radius_mm": float("nan"), "n_mapped_entry": int(use.sum())}
    w = prob[use]
    w = w / w.sum()
    xyz = coords[use]
    centroid = (w[:, None] * xyz).sum(axis=0)
    d = np.linalg.norm(xyz - centroid, axis=1)
    return {"radius_mm": float((w * d).sum()), "n_mapped_entry": int(use.sum())}


# ---------------------------------------------------------------------------
# (5) Single-template-noise null  -- THE LOAD-BEARING TEST
# ---------------------------------------------------------------------------
def single_template_noise_null(
    masked_sub: np.ndarray,
    coords: Optional[np.ndarray],
    mapped: Optional[np.ndarray],
    *,
    n_reps: int = 500,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Null = one fixed template + calibrated noise on the same masks.

    template[ch]  = mean over cluster events of the masked normalized rank
                    (participating cells only).
    residual pool = (observed normalized rank - template) over participating
                    cells (POOLED across channels; homoscedastic "one template +
                    common jitter" null). A genuine *second entry mode* (a
                    channel that jumps to the front beyond its jitter) is a
                    DEVIATION the pooled null does not reproduce -> detectable.
    each rep      : score[ch] = template[ch] + resample(residual); earliest =
                    argmin among the SAME participating channels; recompute Neff
                    (and spatial radius if coords given).

    TWO nulls are computed as a liberal/conservative BRACKET (advisor 2026-06-08):
      * pooled    : residuals pooled across channels (homoscedastic "one template
                    + one common jitter"). Rejecting it can mean a genuine entry
                    region OR merely heteroscedastic per-channel noise -> LIBERAL.
      * per_channel_gauss : template[ch] + N(0, per-channel residual SD). Keeps
                    each channel's own noise SCALE (absorbs heteroscedasticity)
                    but forces UNIMODAL symmetric jitter, so surviving excess
                    means a departure from clean single-template (e.g. a second
                    entry mode / loose blend) -> CONSERVATIVE.
    NB these metrics REJECT clean-single-template better than they CONFIRM an
    entry region: a compact near-tied entry is reproduced by both nulls. Lead
    with the rejection of fixed single-point ignition, not a region claim.

    Returns observed-vs-null Neff (both nulls) + radius (pooled) with directional
    p: p_neff_excess_* = P(null >= obs) [obs MORE dispersed than that null];
       p_neff_concentrated_* = P(null <= obs) [obs MORE concentrated].
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n_ch, n_ev = masked_sub.shape
    part = ~np.isnan(masked_sub)
    with np.errstate(invalid="ignore"):
        template = np.nanmean(np.where(part, masked_sub, np.nan), axis=1)
    resid = (masked_sub - template[:, None])[part]
    resid = resid[np.isfinite(resid)]
    # per-channel residual SD (per_channel_gauss null preserves noise scale)
    chan_sd = np.zeros(n_ch)
    for c in range(n_ch):
        rc = (masked_sub[c] - template[c])[part[c]]
        rc = rc[np.isfinite(rc)]
        chan_sd[c] = float(rc.std()) if rc.size else 0.0

    obs_earl = earliest_channel_per_event(masked_sub)
    obs = effective_number(obs_earl, n_ch)
    obs_prob = earliest_prob_vector(obs_earl, n_ch)
    do_spatial = coords is not None and mapped is not None
    if do_spatial:
        obs_sr = spatial_radius(obs_prob, coords, mapped)
        obs_rad = obs_sr["radius_mm"]
        obs_n_mapped_entry = obs_sr["n_mapped_entry"]   # CLUSTER-level mapped entry contacts
    else:
        obs_rad = float("nan")
        obs_n_mapped_entry = 0

    null_pool = np.empty(n_reps)
    null_gauss = np.empty(n_reps)
    null_rad = np.full(n_reps, np.nan)
    null_prob_accum = np.zeros(n_ch)         # pooled null per-channel entry prob
    null_prob_gauss_accum = np.zeros(n_ch)   # per-channel-Gaussian null per-channel entry prob
    tcol = template[:, None]
    for b in range(n_reps):
        sc_p = np.where(part, tcol + rng.choice(resid, size=masked_sub.shape), np.inf)
        e_p = sc_p.argmin(axis=0); e_p[~np.isfinite(sc_p).any(axis=0)] = -1
        null_pool[b] = effective_number(e_p, n_ch)["neff"]
        null_prob_accum += earliest_prob_vector(e_p, n_ch)
        if do_spatial:
            null_rad[b] = spatial_radius(earliest_prob_vector(e_p, n_ch), coords, mapped)["radius_mm"]
        sc_g = np.where(part, tcol + rng.standard_normal(masked_sub.shape) * chan_sd[:, None], np.inf)
        e_g = sc_g.argmin(axis=0); e_g[~np.isfinite(sc_g).any(axis=0)] = -1
        null_gauss[b] = effective_number(e_g, n_ch)["neff"]
        null_prob_gauss_accum += earliest_prob_vector(e_g, n_ch)
    null_earliest_prob = (null_prob_accum / n_reps).tolist()
    null_earliest_prob_gauss = (null_prob_gauss_accum / n_reps).tolist()

    def _summ(arr):
        a = arr[np.isfinite(arr)]
        if a.size == 0:
            return dict(mean=float("nan"), sd=float("nan"), lo=float("nan"), hi=float("nan"))
        return dict(mean=float(a.mean()), sd=float(a.std()),
                    lo=float(np.percentile(a, 2.5)), hi=float(np.percentile(a, 97.5)))

    def _p(null, direction):
        if not np.isfinite(obs["neff"]):
            return float("nan")
        return float(np.mean(null >= obs["neff"]) if direction == "excess"
                     else np.mean(null <= obs["neff"]))

    return {
        "obs_neff": obs["neff"],
        "obs_top_share": obs["top_share"],
        "obs_radius_mm": obs_rad,
        "null_neff": _summ(null_pool),                 # pooled (kept key for back-compat)
        "null_neff_gauss": _summ(null_gauss),
        "null_radius_mm": _summ(null_rad),
        "p_neff_excess": _p(null_pool, "excess"),              # pooled (liberal)
        "p_neff_concentrated": _p(null_pool, "concentrated"),
        "p_neff_excess_gauss": _p(null_gauss, "excess"),       # per-channel (conservative)
        "p_neff_concentrated_gauss": _p(null_gauss, "concentrated"),
        "p_radius_excess": float(np.mean(null_rad[np.isfinite(null_rad)] >= obs_rad))
        if (do_spatial and np.isfinite(obs_rad) and np.isfinite(null_rad).any()) else float("nan"),
        "obs_radius_n_mapped_entry": int(obs_n_mapped_entry),  # CLUSTER-level (for compact-sub-region next step)
        "obs_earliest_prob": obs_prob.tolist(),
        "null_earliest_prob": null_earliest_prob,             # pooled (liberal) null
        "null_earliest_prob_gauss": null_earliest_prob_gauss,  # per-channel-Gaussian (conservative) null
        "chan_resid_sd_median": float(np.median(chan_sd[chan_sd > 0])) if (chan_sd > 0).any() else float("nan"),
        "n_reps": int(n_reps),
    }


# ---------------------------------------------------------------------------
# Entry shape (diagnostic): monotonic decay vs distinct second mode
# ---------------------------------------------------------------------------
def entry_shape(earliest: np.ndarray, template_rank: Sequence[int], n_ch: int) -> Dict[str, Any]:
    """Is the earliest distribution the noisy low-rank tail of one template, or
    does it have a peak at a channel that is NOT template-early?

    spearman_prob_vs_trank : Spearman between per-channel earliest-prob and
        template rank. Strongly negative => earliest-prob decays monotonically
        from the template's early end (== one noisy template). Near 0 / positive
        => the entry does not track the template order (a distinct entry mode).
    secondary_peak_trank   : template rank of the channel with the 2nd-highest
        earliest-prob; if this is large (late in template) with non-trivial
        share, that is a distinct entry mode.
    """
    prob = earliest_prob_vector(earliest, n_ch)
    tr = np.asarray(template_rank, dtype=float)
    if tr.size != n_ch or np.all(prob == 0):
        return {"spearman_prob_vs_trank": float("nan"),
                "modal_trank": -1, "secondary_peak_trank": -1,
                "secondary_peak_share": float("nan")}
    rho, _ = spearmanr(prob, tr)
    order = np.argsort(prob)[::-1]
    modal_trank = int(tr[order[0]])
    sec_trank = int(tr[order[1]]) if n_ch > 1 else -1
    sec_share = float(prob[order[1]]) if n_ch > 1 else float("nan")
    return {
        "spearman_prob_vs_trank": float(rho),
        "modal_trank": modal_trank,
        "secondary_peak_trank": sec_trank,
        "secondary_peak_share": sec_share,
    }


# ---------------------------------------------------------------------------
# Downstream invariance to entry (positive attractor signature)
# ---------------------------------------------------------------------------
def downstream_invariance_to_entry(
    masked_sub: np.ndarray, earliest: np.ndarray, n_ch: int
) -> Dict[str, Any]:
    """Does the identity of the entry channel predict the downstream order?

    Group cluster events by their earliest channel (groups with
    >= MIN_GROUP_EVENTS). For each group, build a downstream template = mean
    masked normalized rank of the *non-entry* channels. Pairwise Spearman
    between group downstream templates. High median => downstream order is
    INVARIANT to which channel entered first (== events funnel into one channel,
    the attractor-consistent reading). Low => different entries take different
    downstream paths (multiple templates rather than one channel).
    """
    groups: Dict[int, np.ndarray] = {}
    for ch in np.unique(earliest[earliest >= 0]):
        idx = np.where(earliest == ch)[0]
        if idx.size >= MIN_GROUP_EVENTS:
            groups[int(ch)] = idx
    if len(groups) < 2:
        return {"median_cross_entry_spearman": float("nan"),
                "n_groups": len(groups)}
    import warnings
    templ: Dict[int, np.ndarray] = {}
    for ch, idx in groups.items():
        block = masked_sub[:, idx].copy()
        block[ch, :] = np.nan  # exclude the entry channel from its own downstream
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            templ[ch] = np.nanmean(block, axis=1)  # NaN for never-participating ch (handled below)
    keys = sorted(templ)
    rs: List[float] = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = templ[keys[i]], templ[keys[j]]
            both = np.isfinite(a) & np.isfinite(b)
            if int(both.sum()) >= 3 and np.std(a[both]) > 0 and np.std(b[both]) > 0:
                r, _ = spearmanr(a[both], b[both])
                if np.isfinite(r):
                    rs.append(float(r))
    return {
        "median_cross_entry_spearman": float(np.median(rs)) if rs else float("nan"),
        "n_groups": len(groups),
        "n_pairs": len(rs),
    }


# ---------------------------------------------------------------------------
# Post-entry stability (NON-load-bearing; reported raw + centered)
# ---------------------------------------------------------------------------
def post_entry_stability(
    ranks: np.ndarray,
    bools: np.ndarray,
    cluster_events: np.ndarray,
    earliest_full: np.ndarray,
    centered_meta: Dict[str, Any],
    *,
    n_sample: int = 200,
) -> Dict[str, Any]:
    """Within-cluster Kendall tau (raw + centered), full vs earliest-removed.

    (6) CENTERED-ALONG. Reuses ``_multi_seed_tau_summary`` (phantom-safe: it
    only reads participating channels via bools). The earliest-removal delta
    tests whether the downstream order survives dropping the entry channel: a
    small delta means the entry channel is not the anchor of the stereotypy.
    """
    # earliest_full is indexed over the cluster's local event order; map back to
    # global event indices to zero the earliest channel per event (vectorized).
    bools_trunc = bools.copy()
    has = earliest_full >= 0
    bools_trunc[earliest_full[has], cluster_events[has]] = False

    def _tau(b, mask=None):
        return _multi_seed_tau_summary(
            ranks if mask is None else centered_meta["centered_ranks"],
            b, event_indices=cluster_events, n_sample=n_sample,
            min_shared_channels=MIN_PARTICIPATING, channel_mask=mask,
        )["mean_tau"]

    cmask = centered_meta["valid_center_mask"]
    raw_full = _tau(bools)
    raw_trunc = _tau(bools_trunc)
    cen_full = _tau(bools, mask=cmask)
    cen_trunc = _tau(bools_trunc, mask=cmask)
    return {
        "raw_tau_full": float(raw_full),
        "raw_tau_earliest_removed": float(raw_trunc),
        "raw_tau_delta": float(raw_full - raw_trunc)
        if np.isfinite(raw_full) and np.isfinite(raw_trunc) else float("nan"),
        "centered_tau_full": float(cen_full),
        "centered_tau_earliest_removed": float(cen_trunc),
        "centered_tau_delta": float(cen_full - cen_trunc)
        if np.isfinite(cen_full) and np.isfinite(cen_trunc) else float("nan"),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def analyze_cluster(
    ranks: np.ndarray,
    bools: np.ndarray,
    cluster_events: np.ndarray,
    template_rank: Sequence[int],
    centered_meta: Dict[str, Any],
    coords: Optional[np.ndarray],
    mapped: Optional[np.ndarray],
    *,
    n_reps: int = 500,
    seed: int = 0,
    with_stability: bool = True,
) -> Dict[str, Any]:
    """All entry-dispersion metrics for one cluster. (4) PER-CLUSTER."""
    n_ch = ranks.shape[0]
    n_ev = int(cluster_events.size)
    out: Dict[str, Any] = {"n_events": n_ev, "template_rank": list(template_rank)}
    if n_ev < MIN_CLUSTER_EVENTS:
        out["skipped"] = "too_few_events"
        return out

    masked_sub = mask_phantom_ranks(
        ranks[:, cluster_events], bools[:, cluster_events], normalize=True
    )  # (1) PHANTOM-MASK
    earliest = earliest_channel_per_event(masked_sub)

    out.update(effective_number(earliest, n_ch))  # neff, top_share, n
    rng = np.random.default_rng(seed)
    out["null"] = single_template_noise_null(
        masked_sub, coords, mapped, n_reps=n_reps, rng=rng
    )
    out["shape"] = entry_shape(earliest, template_rank, n_ch)
    out["downstream_invariance"] = downstream_invariance_to_entry(
        masked_sub, earliest, n_ch
    )
    if with_stability:
        out["stability"] = post_entry_stability(
            ranks, bools, cluster_events, earliest, centered_meta
        )
    return out
