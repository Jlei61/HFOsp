#!/usr/bin/env python
"""Why the frozen selector lands on K>2 in exactly six subjects.

The producer picks ``chosen_k`` as the k with the highest median silhouette
among the k values that pass both an assignment-stability gate and a
minimum-cluster-fraction gate.  That rule is only as good as the silhouette's
behaviour across k.  This script reads the FROZEN scan out of each artifact --
no clustering is re-run and no k is re-selected -- and asks two questions:

  1. Does the median silhouette rise or fall with k, and does that sign track
     the size of the contact array?
  2. Is ``chosen_k`` the largest admissible k (i.e. the selector ran up against
     a gate) or an interior maximum (i.e. cluster quality genuinely peaked)?

Both are read-only summaries of ``adaptive_cluster.scan``.

Output: k_selection_mechanism.json
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

import run_multimode_grammar_audit as M  # noqa: E402

PER_SUBJECT_IN = REPO / "results/interictal_propagation_masked/per_subject"


def main() -> None:
    rows = []
    for p in sorted(PER_SUBJECT_IN.glob("*.json")):
        d = json.load(open(p))
        ac = d["adaptive_cluster"]
        scan = ac["scan"]
        ks = np.array([e["k"] for e in scan], float)
        sil = np.array([e["median_silhouette"] for e in scan], float)
        passing = [e["k"] for e in scan if e["passes_both"]]
        chosen = int(ac["chosen_k"])
        at_chosen = next(e for e in scan if e["k"] == chosen)
        # what stopped the selector: the first k above chosen that fails, and why
        above = [e for e in scan if e["k"] == chosen + 1]
        blocker = None
        if above:
            e = above[0]
            fails = []
            if not e.get("passes_fraction"):
                fails.append("min_cluster_fraction")
            if not e.get("passes_stability"):
                fails.append("assignment_stability")
            blocker = fails or ["none (k+1 admissible but lower silhouette)"]
        rows.append({
            "subject_id": f"{d['dataset']}_{d['subject']}",
            "chosen_k": chosen,
            "n_channels": int(d["n_channels"]),
            "silhouette_slope_vs_k": float(np.polyfit(ks, sil, 1)[0]),
            "median_silhouette_at_k2": float(sil[0]),
            "median_silhouette_at_k8": float(sil[-1]),
            "median_silhouette_at_chosen_k": float(at_chosen["median_silhouette"]),
            "min_cluster_fraction_at_chosen_k": float(at_chosen["worst_min_cluster_fraction"]),
            "k_values_passing_both_gates": passing,
            "chosen_k_is_largest_admissible": bool(passing and chosen == max(passing)),
            "gate_that_blocked_k_plus_1": blocker,
        })

    hi = [r for r in rows if r["chosen_k"] > 2]
    lo = [r for r in rows if r["chosen_k"] == 2]
    a = np.array([r["silhouette_slope_vs_k"] for r in hi])
    b = np.array([r["silhouette_slope_vs_k"] for r in lo])
    u = stats.mannwhitneyu(a, b, alternative="greater")
    rr = stats.spearmanr(np.array([r["n_channels"] for r in rows], float),
                         np.array([r["silhouette_slope_vs_k"] for r in rows]))
    out = {
        "provenance": {"git_commit": M._git_commit(),
                       "note": "read-only summary of the frozen adaptive_cluster.scan; "
                               "no clustering re-run, no k re-selected"},
        "rows": rows,
        "summary": {
            "slope_K_gt_2": {"n": len(hi), "median": float(np.median(a)),
                             "range": [float(a.min()), float(a.max())],
                             "n_positive": int((a > 0).sum())},
            "slope_K_eq_2": {"n": len(lo), "median": float(np.median(b)),
                             "range": [float(b.min()), float(b.max())],
                             "n_positive": int((b > 0).sum())},
            "mannwhitney_slope_K_gt_2_greater_p": float(u.pvalue),
            "spearman_slope_vs_n_channels": {"rho": float(rr.statistic), "p": float(rr.pvalue)},
            "n_chosen_k_is_largest_admissible_K_gt_2":
                int(sum(r["chosen_k_is_largest_admissible"] for r in hi)),
            "n_chosen_k_is_largest_admissible_K_eq_2":
                int(sum(r["chosen_k_is_largest_admissible"] for r in lo)),
            "min_cluster_fraction_at_chosen_k_K_gt_2":
                [float(r["min_cluster_fraction_at_chosen_k"]) for r in hi],
            "min_cluster_fraction_at_chosen_k_K_eq_2_range":
                [float(min(r["min_cluster_fraction_at_chosen_k"] for r in lo)),
                 float(max(r["min_cluster_fraction_at_chosen_k"] for r in lo))],
        },
    }
    with open(HERE / "k_selection_mechanism.json", "w") as f:
        json.dump(out, f, indent=2)
    s = out["summary"]
    print(f"silhouette-vs-k slope   K>2: median {s['slope_K_gt_2']['median']:+.4f}, "
          f"{s['slope_K_gt_2']['n_positive']}/{s['slope_K_gt_2']['n']} positive")
    print(f"                        K=2: median {s['slope_K_eq_2']['median']:+.4f}, "
          f"{s['slope_K_eq_2']['n_positive']}/{s['slope_K_eq_2']['n']} positive")
    print(f"  Mann-Whitney p={s['mannwhitney_slope_K_gt_2_greater_p']:.3g}; "
          f"slope vs contacts rho={s['spearman_slope_vs_n_channels']['rho']:+.3f} "
          f"(P={s['spearman_slope_vs_n_channels']['p']:.3g})")
    print(f"chosen_k == largest admissible k:  K>2 {s['n_chosen_k_is_largest_admissible_K_gt_2']}/{len(hi)}, "
          f"K=2 {s['n_chosen_k_is_largest_admissible_K_eq_2']}/{len(lo)}")
    print(f"min cluster fraction at chosen k:  K>2 {s['min_cluster_fraction_at_chosen_k_K_gt_2']}")
    print(f"                                   K=2 range {s['min_cluster_fraction_at_chosen_k_K_eq_2_range']}")
    for r in hi:
        print(f"  {r['subject_id']:24s} K={r['chosen_k']} blocked at k+1 by {r['gate_that_blocked_k_plus_1']}")


if __name__ == "__main__":
    main()
