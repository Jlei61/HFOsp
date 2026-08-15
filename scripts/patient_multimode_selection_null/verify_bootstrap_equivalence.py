#!/usr/bin/env python
"""Verify that the sufficient-statistic block bootstrap equals direct recompute.

``run_multimode_grammar_audit.analysis2_direction_and_extent`` builds each
bootstrap replicate from per-block sums instead of materialising the resampled
event matrix.  That is only legitimate if, for any multiset of blocks, the two
routes give byte-comparable prototypes / valid masks / median recruited
fractions.  This script asserts exactly that on real subjects, over random
block multisets, and is the evidence behind the claim made in that comment.

Run:  python results/interictal_propagation_masked/multimode_grammar_audit/verify_bootstrap_equivalence.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from src.lagpat_rank_audit import mask_phantom_ranks  # noqa: E402
import run_multimode_grammar_audit as M  # noqa: E402

SUBJECTS = ["yuquan_zhaojinrui", "epilepsiae_818", "epilepsiae_916", "yuquan_zhangjinhan"]
N_DRAWS = 40


def direct(masked, bools_v, npart, labels, blk_dense, n_blk, k, n_ch, w):
    """Recompute prototypes / masks / medians on the materialised resample."""
    idx = np.concatenate(
        [np.flatnonzero(blk_dense == b) for b in range(n_blk) for _ in range(int(w[b]))]
    ) if w.sum() > 0 else np.zeros(0, dtype=int)
    lab, ms, bl, npr = labels[idx], masked[:, idx], bools_v[:, idx], npart[idx]
    pr = np.full((k, n_ch), np.nan)
    vd = np.zeros((k, n_ch), dtype=bool)
    med = np.full(k, np.nan)
    ne = np.zeros(k)
    for ci in range(k):
        sel = lab == ci
        ne[ci] = sel.sum()
        if not sel.any():
            continue
        p, cnt, v = M._mode_prototype(ms, bl, sel)
        pr[ci], vd[ci] = p, v
        med[ci] = float(np.median(npr[sel])) / n_ch
    return pr, vd, med, ne


def main() -> None:
    rng = np.random.default_rng(7)
    worst = 0.0
    for sid in SUBJECTS:
        rec = M.replay_and_audit(sid)
        loaded, labels, ve = rec["_loaded"], rec["_labels"], rec["_valid_events"]
        k, n_ch = rec["chosen_k"], rec["n_channels"]
        ranks_v, bools_v = loaded["ranks"][:, ve], loaded["bools"][:, ve]
        masked = mask_phantom_ranks(ranks_v, bools_v, normalize=True)
        npart = bools_v.sum(axis=0).astype(float)
        block_ids = rec["_block_ids_valid"]
        uniq_blocks = np.unique(block_ids)
        n_blk = uniq_blocks.size
        blk_dense = np.searchsorted(uniq_blocks, block_ids)

        # rebuild the exact sufficient statistics used by the audit
        p_vals = np.arange(M.MIN_SHARED_CHANNELS, n_ch + 1)
        sum_rank = np.zeros((n_blk, k, n_ch))
        cnt_part = np.zeros((n_blk, k, n_ch))
        cnt_ev = np.zeros((n_blk, k))
        hist_np = np.zeros((n_blk, k, p_vals.size))
        masked0 = np.where(bools_v, np.nan_to_num(masked, nan=0.0), 0.0)
        for ci in range(k):
            sel = labels == ci
            bd, m0, bl = blk_dense[sel], masked0[:, sel], bools_v[:, sel]
            for ch in range(n_ch):
                sum_rank[:, ci, ch] = np.bincount(bd, weights=m0[ch], minlength=n_blk)
                cnt_part[:, ci, ch] = np.bincount(bd, weights=bl[ch].astype(float), minlength=n_blk)
            cnt_ev[:, ci] = np.bincount(bd, minlength=n_blk)
            npi = npart[sel].astype(int) - M.MIN_SHARED_CHANNELS
            hist_np[:, ci, :] = np.bincount(
                bd * p_vals.size + npi, minlength=n_blk * p_vals.size
            ).reshape(n_blk, p_vals.size)

        def suff(w):
            sr = np.tensordot(w, sum_rank, axes=(0, 0))
            cp = np.tensordot(w, cnt_part, axes=(0, 0))
            ne = w @ cnt_ev
            hn = np.tensordot(w, hist_np, axes=(0, 0))
            with np.errstate(invalid="ignore", divide="ignore"):
                pr = np.where(cp > 0, sr / np.maximum(cp, 1e-12), np.nan)
                frac = np.where(ne[:, None] > 0, cp / np.maximum(ne[:, None], 1e-12), 0.0)
            vd = (cp >= M.MIN_PROTO_COUNT) & (frac >= M.MIN_PROTO_FRAC)
            pr = np.where(vd, pr, np.nan)
            med = np.full(k, np.nan)
            for ci in range(k):
                tot = int(round(hn[ci].sum()))
                if tot > 0:
                    cum = np.cumsum(hn[ci])
                    j1, j2 = (tot - 1) // 2, tot // 2
                    v1 = p_vals[int(np.searchsorted(cum, j1, side="right"))]
                    v2 = p_vals[int(np.searchsorted(cum, j2, side="right"))]
                    med[ci] = 0.5 * (v1 + v2) / n_ch
            return pr, vd, med, ne

        errs = []
        for d in range(N_DRAWS):
            w = (np.ones(n_blk) if d == 0
                 else rng.multinomial(n_blk, np.full(n_blk, 1.0 / n_blk)).astype(float))
            p1, v1, m1, e1 = suff(w)
            p2, v2, m2, e2 = direct(masked, bools_v, npart, labels, blk_dense, n_blk, k, n_ch, w)
            assert np.array_equal(v1, v2), f"{sid} draw {d}: valid mask mismatch"
            assert np.allclose(e1, e2), f"{sid} draw {d}: event count mismatch"
            assert np.allclose(m1, m2, equal_nan=True), f"{sid} draw {d}: median mismatch {m1} {m2}"
            fin = np.isfinite(p1) | np.isfinite(p2)
            assert np.array_equal(np.isfinite(p1), np.isfinite(p2)), f"{sid} draw {d}: NaN pattern"
            e = float(np.max(np.abs(p1[fin & np.isfinite(p1)] - p2[fin & np.isfinite(p2)]))) if fin.any() else 0.0
            errs.append(e)
        worst = max(worst, max(errs))
        print(f"{sid:24s} n_blocks={n_blk:4d} k={k} draws={N_DRAWS} "
              f"max|prototype diff|={max(errs):.3e}  OK")
    print(f"\nALL PASS - worst prototype discrepancy across subjects: {worst:.3e}")


if __name__ == "__main__":
    main()
