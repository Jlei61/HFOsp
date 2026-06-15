"""Topic 5 A-line — pure helpers for ictal-vs-interictal axis alignment (no I/O).

The A-line primary statistic is | corr_pair_mirror_invariant(interictal_rank_field,
ictal_activation_field) |, both smoothed on the SAME normalized contact plane (the
interictal axis record's plane). This module owns the contract-critical pure pieces:
  - the join: which axis-record channels have an ictal activation value (drop the rest;
    the >=80% montage gate already guaranteed >=MIN_CH remain), keeping BOTH fields on the
    same channel support so the mirror-invariant field correlation is fair;
  - the null shuffle DOMAINS: channel-shuffle (coarse anchor), within-shaft (controls for
    same-shaft/anatomy). anchor-matched (controls for "source is just more active") needs a
    per-channel baseline-activity anchor that is NOT yet in the T0 cache -> raises (§6: a
    stub returns nothing plausible).

Field smoothing + the mirror-invariant correlation themselves are REUSED from
src.propagation_contact_plane_readout (R_smooth_rank / corr_pair_mirror_invariant); the
orchestration that loads the cache + axis record lives in scripts/run_topic5_axis_alignment.py.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def matched_channels(axis_record: dict, value_by_name: Dict[str, float]) -> List[dict]:
    """Axis-record channels that ALSO have an ictal activation value, in record order.
    Channels absent from value_by_name are dropped (never index-aligned)."""
    return [c for c in axis_record["channels"] if c["name"] in value_by_name]


def make_field_record(matched: Sequence[dict], values: Sequence[float]) -> dict:
    """A smooth_field-ready record over `matched` channels with typical_rank := values[i].
    Keeps each channel's x_norm/y_norm/support (the shared plane); only the scalar changes."""
    if len(matched) != len(values):
        raise ValueError(f"matched ({len(matched)}) and values ({len(values)}) length mismatch")
    chans = [dict(c, typical_rank=float(v)) for c, v in zip(matched, values)]
    return {"channels": chans}


def interictal_and_ictal_values(matched: Sequence[dict], value_by_name: Dict[str, float]):
    """(interictal_rank[], ictal_value[]) aligned to `matched`. Interictal rank = the axis
    record's own typical_rank; ictal value = the joined activation (e.g. broadband_auc)."""
    inter = np.array([float(c["typical_rank"]) for c in matched], float)
    ict = np.array([float(value_by_name[c["name"]]) for c in matched], float)
    return inter, ict


def channel_shuffle(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute the ictal values across ALL channels (coarse-anchor null)."""
    v = np.asarray(values, float)
    return v[rng.permutation(v.shape[0])]


def within_shaft_shuffle(values: np.ndarray, names: Sequence[str],
                         rng: np.random.Generator) -> np.ndarray:
    """Permute the ictal values WITHIN each electrode shaft (controls same-shaft/anatomy).
    The multiset of values on each shaft is preserved; values never cross shafts."""
    from src.propagation_skeleton_geometry import parse_shaft
    v = np.asarray(values, float)
    out = v.copy()
    groups: Dict[str, List[int]] = {}
    for i, n in enumerate(names):
        groups.setdefault(parse_shaft(n)[0], []).append(i)
    for idx in groups.values():
        ix = np.array(idx)
        out[ix] = v[ix][rng.permutation(ix.shape[0])]
    return out


def anchor_matched_shuffle(values, anchor, rng, *, n_bins=4):
    """Permute the ictal values WITHIN bins of similar `anchor` (per-channel baseline
    activity), controlling for 'the source is simply more active'. Channels are binned into
    `n_bins` quantile bins of the anchor; values are permuted only within a bin, so the
    activation's coupling to baseline activity is preserved while its coupling to the AXIS is
    destroyed. Channels with a non-finite anchor form their own pass-through group.
    `anchor` must be aligned to `values` (same per-channel order)."""
    v = np.asarray(values, float)
    a = np.asarray(anchor, float)
    if v.shape[0] != a.shape[0]:
        raise ValueError(f"values ({v.shape[0]}) and anchor ({a.shape[0]}) length mismatch")
    out = v.copy()
    finite = np.isfinite(a)
    idx_fin = np.where(finite)[0]
    if idx_fin.size >= 2:
        # rank -> quantile bin (ties handled by argsort of argsort); >=1 channel per bin
        order = np.argsort(np.argsort(a[idx_fin]))
        nb = int(max(1, min(n_bins, idx_fin.size)))
        bin_of = (order * nb // idx_fin.size)
        for b in range(nb):
            grp = idx_fin[bin_of == b]
            if grp.size >= 2:
                out[grp] = v[grp][rng.permutation(grp.size)]
    # non-finite-anchor channels: shuffle among themselves (their own group)
    idx_nan = np.where(~finite)[0]
    if idx_nan.size >= 2:
        out[idx_nan] = v[idx_nan][rng.permutation(idx_nan.size)]
    return out


def _anchor_bin_ids(anchor, n_bins):
    """Per-channel quantile bin of the finite anchor (-1 for non-finite). Helper for the
    anchor-matched and joint nulls + effective-shuffle accounting."""
    a = np.asarray(anchor, float)
    bin_id = np.full(a.shape[0], -1)
    idx = np.where(np.isfinite(a))[0]
    if idx.size:
        order = np.argsort(np.argsort(a[idx]))
        nb = int(max(1, min(n_bins, idx.size)))
        bin_id[idx] = order * nb // idx.size
    return bin_id


def _groups(keys):
    g = {}
    for i, k in enumerate(keys):
        g.setdefault(k, []).append(i)
    return g


def within_shaft_anchor_shuffle(values, names, anchor, rng, *, n_bins=4):
    """JOINT null: permute WITHIN (shaft x baseline-activity-bin) cells — simultaneously
    controls same-shaft anatomy AND baseline activity. Strictly nested inside both the
    within-shaft and the anchor-matched nulls, so this is the conservative 'both confounds
    at once' control (the separate within-shaft / anchor nulls are NOT nested in each other)."""
    from src.propagation_skeleton_geometry import parse_shaft
    v = np.asarray(values, float)
    out = v.copy()
    bin_id = _anchor_bin_ids(anchor, n_bins)
    keys = [(parse_shaft(n)[0], int(bin_id[i])) for i, n in enumerate(names)]
    for grp in _groups(keys).values():
        if len(grp) >= 2:
            ix = np.array(grp)
            out[ix] = v[ix][rng.permutation(ix.shape[0])]
    return out


def effective_shuffle_n(names, anchor, kind, *, n_bins=4):
    """How many channels actually move under a null = those in a group of size >= 2. Small/
    few-shaft subjects can leave most channels un-shuffled (singleton bins) -> weak null;
    report this so a 'beat' on a near-degenerate null is not over-read. kind in
    {channel, within_shaft, anchor, joint}."""
    from src.propagation_skeleton_geometry import parse_shaft
    n = len(names)
    if kind == "channel":
        keys = [0] * n
    elif kind == "within_shaft":
        keys = [parse_shaft(nm)[0] for nm in names]
    elif kind == "anchor":
        keys = [int(b) for b in _anchor_bin_ids(anchor, n_bins)]
    elif kind == "joint":
        bid = _anchor_bin_ids(anchor, n_bins)
        keys = [(parse_shaft(nm)[0], int(bid[i])) for i, nm in enumerate(names)]
    else:
        raise ValueError(kind)
    return int(sum(len(g) for g in _groups(keys).values() if len(g) >= 2))


def along_axis_sign(rank, activation):
    """1D along-axis sign: does ictal activation sit toward the SOURCE (early, low rank) or
    SINK (late, high rank) end of the interictal axis? This is a SIGNED scalar orthogonal to
    the field / mirror-invariant primary stat — it is purely 1D (rank vs activation), so it is
    mirror-irrelevant by construction (no y-coordinate enters). `rank` and `activation` must be
    aligned to the SAME matched channels. spearman(rank, activation): sign<0 = source/early end
    hotter (forward); sign>0 = sink/late end hotter (reverse). Channels where either value is
    non-finite are dropped; <3 usable channels (or an undefined rho on constant input) -> sign 0."""
    from scipy.stats import spearmanr
    r = np.asarray(rank, float)
    a = np.asarray(activation, float)
    ok = np.isfinite(r) & np.isfinite(a)
    k = int(ok.sum())
    if k < 3:
        return {"signed_corr": np.nan, "sign": 0, "n": k}
    rho = spearmanr(r[ok], a[ok]).statistic
    if not np.isfinite(rho):
        return {"signed_corr": np.nan, "sign": 0, "n": k}
    return {"signed_corr": float(rho), "sign": int(np.sign(rho)), "n": k}


def seizure_parity_subsets(eligible_idxs):
    """Split a subject's eligible-seizure index list into ODD / EVEN halves BY POSITION
    (not by the raw integer value), for a within-subject split-half robustness check of the
    A-line cohort verdict. Position 0,2,4,... -> 'even' half; 1,3,5,... -> 'odd' half.
    Returns (even_set, odd_set) as sets of the original idx values. A subject with a single
    eligible seizure yields one empty half (it then drops from that half's cohort, by design).
    """
    idxs = list(eligible_idxs)
    even = {idxs[i] for i in range(0, len(idxs), 2)}
    odd = {idxs[i] for i in range(1, len(idxs), 2)}
    return even, odd
