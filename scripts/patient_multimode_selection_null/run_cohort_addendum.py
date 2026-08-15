#!/usr/bin/env python
"""Cohort addendum: endpoint enumeration, MI bias floor, and group contrasts.

Three gaps the main audit leaves open, each of which would let a wrong reading
survive:

1. **Endpoint enumeration.**  Reading the K>2 prototypes by eye suggests each
   mode is simply "a different contact fires first (or last)".  That has to be
   counted, not eyeballed: how many of the K modes carry a distinct
   earliest-contact, a distinct latest-contact, and a distinct
   (earliest, latest) pair -- and how large is the space of endpoint pairs the
   array can even express.

2. **Plug-in mutual-information bias.**  The main audit's split of the mode
   label into "explained by which contacts participated" vs "explained by their
   order" uses plug-in MI.  When the participation-set space approaches the
   event count (many contacts, few events) the plug-in estimate saturates at
   H(mode) for purely combinatorial reasons.  Without a label-shuffled floor,
   a 38-contact subject looks "100% participation-set driven" by construction.
   The floor is computed here and the corrected fraction reported alongside.

3. **Group contrast.**  Whether the K>2 subjects differ from the K=2 subjects
   in switching behaviour at all, tested rather than asserted.

Output: cohort_addendum.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from src.lagpat_rank_audit import build_masked_kmeans_features  # noqa: E402
import run_multimode_grammar_audit as M  # noqa: E402

N_SHUFFLE = 200
SEED = 20260815


def main() -> None:
    per_subject = sorted((HERE / "per_subject").glob("*.json"))
    rows = []
    for si, p in enumerate(per_subject):
        rng = np.random.default_rng(SEED + 9000 + si)
        d = json.load(open(p))
        sid = d["subject_id"]
        k = d["engineering_audit"]["chosen_k"]
        n_ch = d["engineering_audit"]["n_channels"]
        a2 = d["analysis2_direction_extent"]
        a1 = d["analysis1_occupancy_transitions"]

        # ---- 1. endpoint enumeration -----------------------------------
        srcs, sinks, pairs = [], [], []
        for m in a2["modes"]:
            pr = np.array([np.nan if v is None else v for v in m["prototype_masked_rank"]])
            if np.all(np.isnan(pr)):
                continue
            srcs.append(int(np.nanargmin(pr)))
            sinks.append(int(np.nanargmax(pr)))
            pairs.append((srcs[-1], sinks[-1]))
        n_src, n_sink, n_pair = len(set(srcs)), len(set(sinks)), len(set(pairs))
        n_endpoint_pairs_possible = n_ch * (n_ch - 1)

        # ---- 2. MI bias floor ------------------------------------------
        rec = M.replay_and_audit(sid)
        labels, ve = rec["_labels"], rec["_valid_events"]
        ranks_v = rec["_loaded"]["ranks"][:, ve]
        bools_v = rec["_loaded"]["bools"][:, ve]
        set_code = np.packbits(bools_v.T.astype(np.uint8), axis=1)
        set_id = np.asarray(np.unique(set_code, axis=0, return_inverse=True)[1]).ravel()
        X = build_masked_kmeans_features(ranks_v, bools_v, impute=M.IMPUTE)
        inv = np.asarray(np.unique(np.round(X, 9), axis=0, return_inverse=True)[1]).ravel()
        h_mode = M._entropy(np.bincount(labels, minlength=k))
        mi_set = M._mutual_information(labels, set_id)

        # ---- 1b. is a mode a template, or just a bin of the lattice? -----
        # A genuine propagation template concentrates its events on one
        # ordering; a lattice bin spreads them evenly over the distinct feature
        # vectors it happens to contain.  Reported as the share of the mode's
        # events sitting on its single most common ordering, against the
        # uniform-over-own-vectors reference 1 / n_vectors_in_mode.
        top1, nvec, unif = [], [], []
        for ci in range(k):
            sub = inv[labels == ci]
            if sub.size == 0:
                continue
            c = np.bincount(sub)
            c = c[c > 0]
            top1.append(float(c.max() / c.sum()))
            nvec.append(int(c.size))
            unif.append(1.0 / c.size)
        floor_set = float(np.mean([
            M._mutual_information(rng.permutation(labels), set_id) for _ in range(N_SHUFFLE)]))
        denom = h_mode - floor_set
        frac_set_corr = float((mi_set - floor_set) / denom) if denom > 1e-12 else None
        rows.append({
            "subject_id": sid,
            "dataset": d["dataset"],
            "chosen_k": k,
            "n_channels": n_ch,
            "n_shafts": a2["n_shafts"],
            "n_valid_events": d["engineering_audit"]["n_valid_events"],
            # 1
            "n_distinct_source_contacts": n_src,
            "n_distinct_sink_contacts": n_sink,
            "n_distinct_source_sink_pairs": n_pair,
            "frac_modes_with_distinct_source": n_src / k,
            "frac_modes_with_distinct_sink": n_sink / k,
            "frac_modes_with_distinct_endpoint_pair": n_pair / k,
            "n_endpoint_pairs_expressible_by_array": n_endpoint_pairs_possible,
            "modes_per_expressible_endpoint_pair": k / n_endpoint_pairs_possible,
            "n_unique_set_ids": int(np.unique(set_id).size),
            "n_unique_feature_vectors": int(np.unique(inv).size),
            "per_mode_top1_ordering_share": top1,
            "per_mode_n_distinct_orderings": nvec,
            "per_mode_uniform_reference_share": unif,
            "median_top1_ordering_share": float(np.median(top1)) if top1 else None,
            "median_uniform_reference_share": float(np.median(unif)) if unif else None,
            "median_top1_over_uniform": (
                float(np.median(np.asarray(top1) / np.asarray(unif))) if top1 else None),
            # 2
            "H_mode_nats": float(h_mode),
            "I_mode_set_raw": float(mi_set),
            "I_mode_set_shuffled_floor": floor_set,
            "frac_mode_by_set_raw": float(mi_set / h_mode) if h_mode > 0 else None,
            "frac_mode_by_set_bias_corrected": frac_set_corr,
            "frac_mode_by_order_bias_corrected": (None if frac_set_corr is None else 1.0 - frac_set_corr),
            "mi_floor_fraction_of_H": float(floor_set / h_mode) if h_mode > 0 else None,
            "mi_estimate_trustworthy": bool(h_mode > 0 and floor_set / h_mode < 0.10),
            # carried for the group contrast
            "excess_switch_rate": a1["excess_switch_rate"],
            "observed_switch_rate": a1["observed_switch_rate"],
            "normalized_entropy": a1["normalized_entropy"],
            "mode_block_cramers_v": a1["mode_block_cramers_v"],
        })
        print(f"{sid:24s} K={k} nch={n_ch:2d} | distinct src {n_src}/{k} sink {n_sink}/{k} "
              f"pair {n_pair}/{k} | MI floor {floor_set/h_mode*100:5.1f}% of H "
              f"| set-frac raw {mi_set/h_mode*100:5.1f}% -> corrected "
              f"{'n/a' if frac_set_corr is None else f'{frac_set_corr*100:5.1f}%'}"
              f" | top1-ordering {np.median(top1)*100:5.1f}% vs uniform "
              f"{np.median(unif)*100:5.1f}% ({np.median(np.asarray(top1)/np.asarray(unif)):.1f}x)",
              flush=True)

    # ---- 3. group contrast ---------------------------------------------
    hi = [r for r in rows if r["chosen_k"] > 2]
    lo = [r for r in rows if r["chosen_k"] == 2]
    def arr(g, key):
        return np.array([r[key] for r in g], dtype=float)
    contrast = {}
    for key in ["excess_switch_rate", "observed_switch_rate", "normalized_entropy",
                "mode_block_cramers_v", "n_channels"]:
        a, b = arr(hi, key), arr(lo, key)
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        contrast[key] = {
            "K_gt_2_n": int(a.size), "K_gt_2_median": float(np.median(a)),
            "K_gt_2_range": [float(a.min()), float(a.max())],
            "K_eq_2_n": int(b.size), "K_eq_2_median": float(np.median(b)),
            "K_eq_2_range": [float(b.min()), float(b.max())],
            "mannwhitney_u": float(u.statistic), "p_two_sided": float(u.pvalue),
        }
    a, b = np.abs(arr(hi, "excess_switch_rate")), np.abs(arr(lo, "excess_switch_rate"))
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    contrast["abs_excess_switch_rate"] = {
        "K_gt_2_n": int(a.size), "K_gt_2_median": float(np.median(a)),
        "K_eq_2_n": int(b.size), "K_eq_2_median": float(np.median(b)),
        "mannwhitney_u": float(u.statistic), "p_two_sided": float(u.pvalue),
    }
    K = arr(rows, "chosen_k")
    predictors = {}
    for key in ["n_channels", "n_shafts", "n_valid_events", "n_unique_feature_vectors",
                "modes_per_expressible_endpoint_pair"]:
        rho, pv = stats.spearmanr(arr(rows, key), K)
        predictors[key] = {"spearman_rho_vs_chosen_k": float(rho), "p": float(pv)}

    out = {
        "provenance": {"git_commit": M._git_commit(), "seed": SEED, "n_shuffle": N_SHUFFLE},
        "rows": rows,
        "group_contrast_K_gt_2_vs_K_eq_2": contrast,
        "chosen_k_predictors_spearman": predictors,
    }
    with open(HERE / "cohort_addendum.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== group contrast (K>2 n=%d vs K=2 n=%d) ===" % (len(hi), len(lo)))
    for key, v in contrast.items():
        print(f"  {key:28s} median {v['K_gt_2_median']:+.4f} vs {v['K_eq_2_median']:+.4f}  p={v['p_two_sided']:.4g}")
    print("\n=== chosen-K predictors ===")
    for key, v in predictors.items():
        print(f"  {key:38s} rho={v['spearman_rho_vs_chosen_k']:+.3f} p={v['p']:.3g}")
    print(f"\nWrote {HERE/'cohort_addendum.json'}")


if __name__ == "__main__":
    main()
