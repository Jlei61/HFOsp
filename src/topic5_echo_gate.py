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
from scipy.stats import spearmanr


def spearman_common(rank_a, rank_b, *, min_ch: int) -> float:
    """Spearman rho on channels finite in BOTH vectors; NaN if fewer than min_ch."""
    a = np.asarray(rank_a, dtype=float)
    b = np.asarray(rank_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"rank vectors must align by channel index: {a.shape} vs {b.shape}")
    common = np.isfinite(a) & np.isfinite(b)
    if int(common.sum()) < min_ch:
        return float("nan")
    rho = spearmanr(a[common], b[common]).statistic
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


def _block_permute(values: np.ndarray, blocks: np.ndarray, kind: str, rng) -> np.ndarray:
    """within: permute values inside each block. between: swap whole equal-size blocks."""
    values = np.asarray(values, dtype=float)
    blocks = np.asarray(blocks)
    out = values.copy()
    uniq = [b for b in np.unique(blocks) if b is not None and b == b]  # drop None/NaN
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
    blocks = np.asarray(blocks)
    uniq = [b for b in np.unique(blocks) if b is not None and b == b]
    sizes = defaultdict(list)
    for b in uniq:
        sizes[int(np.sum(blocks == b))].append(b)
    n_exch = sum(size * len(grp) for size, grp in sizes.items() if len(grp) >= 2)
    total = sum(int(np.sum(blocks == b)) for b in uniq)
    return {"n_exchangeable_channels": int(n_exch), "n_total_channels": int(total),
            "insufficient_block_exchange": bool(n_exch < 2)}
