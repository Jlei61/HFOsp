"""Would another seed have changed the answer?

Seeds are pooled inside a patient and never counted as samples, so extra seeds
tighten a per-patient estimate rather than adding statistical power.  That is the
intent; whether it holds is a separate question, and a cheap one to settle: redo
every headline comparison using the first seed alone, then using every seed a
patient has, and see whether any verdict moves.

A conclusion that flips between those two is not a conclusion about the models.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/topic5_spatial_latent_propagation_rnn_v0_1"
METRIC = "test_next_bce"

COMPARISONS = [
    ("recurrence over static", "ORDINARY_GRU", "STATIC_CONTACT"),
    ("contact graph over recurrence", "CONTACT_GRAPH_RNN", "ORDINARY_GRU"),
    ("tissue field over recurrence", "LATENT_LEARNED_SPATIAL_RNN", "ORDINARY_GRU"),
    ("tissue field over static", "LATENT_LEARNED_SPATIAL_RNN", "STATIC_CONTACT"),
    ("learned graph over fixed local", "LATENT_LEARNED_SPATIAL_RNN",
     "LATENT_FIXED_LOCAL_RNN"),
]


def build(rows: list, seeds: set | None) -> dict:
    table: dict = {}
    for row in rows:
        seed = int(row["seed"])
        if seeds is not None and seed not in seeds:
            continue
        table.setdefault(row["arm"], {}).setdefault(row["subject"], {})[seed] = float(
            row[METRIC]
        )
    return table


def paired(table: dict, better: str, worse: str) -> dict:
    delta = []
    for subject in sorted(set(table.get(better, {})) & set(table.get(worse, {}))):
        shared = sorted(set(table[better][subject]) & set(table[worse][subject]))
        if not shared:
            continue
        delta.append(float(np.median([table[worse][subject][s] for s in shared])
                           - np.median([table[better][subject][s] for s in shared])))
    if len(delta) < 3:
        return {"status": "INSUFFICIENT", "n": len(delta)}
    delta = np.array(delta)
    return {
        "status": "COMPLETE",
        "n": len(delta),
        "median": float(np.median(delta)),
        "n_positive": int((delta > 0).sum()),
        "wilcoxon_two_sided_p": float(stats.wilcoxon(delta).pvalue),
    }


def main() -> int:
    argparse.ArgumentParser().parse_args()
    rows = list(csv.DictReader((OUT / "patient_prediction_metrics.csv").open()))
    first_only = build(rows, {1})
    all_seeds = build(rows, None)

    results = []
    for label, better, worse in COMPARISONS:
        one = paired(first_only, better, worse)
        every = paired(all_seeds, better, worse)
        if one["status"] != "COMPLETE" or every["status"] != "COMPLETE":
            continue
        same_sign = np.sign(one["median"]) == np.sign(every["median"])
        same_verdict = (one["wilcoxon_two_sided_p"] < 0.05) == (
            every["wilcoxon_two_sided_p"] < 0.05
        )
        results.append({
            "comparison": label,
            "first_seed_only": one,
            "every_available_seed": every,
            "sign_agrees": bool(same_sign),
            "verdict_agrees": bool(same_verdict),
        })
        print(f"{label:32s} seed1 {one['median']:+.4f} {one['n_positive']:2d}/{one['n']:2d} "
              f"p={one['wilcoxon_two_sided_p']:.2g}   "
              f"all {every['median']:+.4f} {every['n_positive']:2d}/{every['n']:2d} "
              f"p={every['wilcoxon_two_sided_p']:.2g}   "
              f"{'stable' if same_sign and same_verdict else 'MOVED'}")

    seeds_present = sorted({int(r["seed"]) for r in rows})
    verdict = {
        "contract": "topic5_slp_seed_stability_v0_1",
        "metric": METRIC,
        "seeds_present": seeds_present,
        "n_comparisons": len(results),
        "all_signs_agree": all(r["sign_agrees"] for r in results),
        "all_verdicts_agree": all(r["verdict_agrees"] for r in results),
        "reading": (
            "every headline comparison keeps its direction and its verdict when the "
            "extra seeds are added, so the conclusions do not depend on which "
            "starting point a patient happened to get"
            if results and all(r["sign_agrees"] and r["verdict_agrees"] for r in results)
            else "at least one comparison changes when extra seeds are added, so it "
                 "describes the starting point as much as the model"
        ),
        "comparisons": results,
    }
    (OUT / "seed_stability.json").write_text(json.dumps(verdict, indent=1))
    print(f"\nseeds present: {seeds_present}\n{verdict['reading']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
