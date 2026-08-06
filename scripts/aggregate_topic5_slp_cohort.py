"""Aggregate the SLP-RNN cohort to patient-level statistics.

The patient is the unit.  Seeds are aggregated inside a patient and never counted
as independent samples.  Every primary comparison is reported on the full cohort
and on both pre-registered strata, whatever the direction -- an earlier Topic 5
comparison changed sign once low-support patients were removed, so the strata are
not optional.
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

METRIC = "test_next_bce"          # lower is better
SECONDARY = "test_contact_nll"    # comparable with the existing Topic 5 line

# Every comparison is written so that a POSITIVE delta means "a is better".
COMPARISONS = [
    ("H1_recurrence", "ORDINARY_GRU", "STATIC_CONTACT"),
    ("H1", "CONTACT_GRAPH_RNN", "STATIC_CONTACT"),
    ("H1b_contact_graph", "CONTACT_GRAPH_RNN", "ORDINARY_GRU"),
    ("H1b_latent_learned", "LATENT_LEARNED_SPATIAL_RNN", "ORDINARY_GRU"),
    ("H1_latent", "LATENT_LEARNED_SPATIAL_RNN", "STATIC_CONTACT"),
    ("H3", "LATENT_LEARNED_SPATIAL_RNN", "LATENT_FIXED_LOCAL_RNN"),
]


def collect() -> list:
    rows = []
    root = OUT / "per_subject"
    if not root.exists():
        return rows
    for subject_dir in sorted(root.iterdir()):
        for arm_dir in sorted(subject_dir.iterdir()):
            for seed_dir in sorted(arm_dir.iterdir()):
                done = seed_dir / "DONE.json"
                if not done.exists():
                    continue
                payload = json.loads(done.read_text())
                payload["subject"] = subject_dir.name
                payload["arm"] = arm_dir.name
                rows.append(payload)
    return rows


def per_patient(rows: list, metric: str) -> dict:
    """{arm: {subject: median across seeds}} plus a converged flag per cell."""
    table: dict = {}
    converged: dict = {}
    for row in rows:
        table.setdefault(row["arm"], {}).setdefault(row["subject"], []).append(row[metric])
        converged.setdefault(row["arm"], {}).setdefault(row["subject"], []).append(
            bool(row.get("converged", True))
        )
    return (
        {arm: {s: float(np.median(v)) for s, v in subs.items()}
         for arm, subs in table.items()},
        {arm: {s: all(v) for s, v in subs.items()} for arm, subs in converged.items()},
    )


def paired(values_a: dict, values_b: dict, subjects: list) -> dict:
    """Positive delta means arm a beats arm b (the metric is a loss)."""
    common = [s for s in subjects if s in values_a and s in values_b]
    if len(common) < 3:
        return {"status": "INSUFFICIENT", "n": len(common), "subjects": common}
    delta = np.array([values_b[s] - values_a[s] for s in common])
    rng = np.random.default_rng(20260806)
    boot = np.array([
        np.median(rng.choice(delta, len(delta), replace=True)) for _ in range(4000)
    ])
    try:
        p = float(stats.wilcoxon(delta).pvalue)
    except ValueError:
        p = float("nan")
    return {
        "status": "COMPLETE",
        "n": len(common),
        "subjects": common,
        "median_delta": float(np.median(delta)),
        "bootstrap_95ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "n_positive": int((delta > 0).sum()),
        "n_negative": int((delta < 0).sum()),
        "wilcoxon_two_sided_p": p,
        "per_patient_delta": {s: float(d) for s, d in zip(common, delta)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    manifest = json.loads((OUT / "INPUT_MANIFEST.json").read_text())
    cohort = manifest["frozen_cohort"]
    strata = {
        "all": cohort["primary"],
        "planar": cohort["strata"]["planar"]["subjects"],
        "well_sampled": cohort["strata"]["well_sampled"]["subjects"],
    }

    rows = collect()
    if not rows:
        raise SystemExit("no completed units under per_subject/")

    with (args.out / "patient_prediction_metrics.csv").open("w", newline="") as handle:
        fields = ["subject", "arm", "seed", "test_next_bce", "test_contact_nll",
                  "test_top1", "test_stop_bce", "converged", "epochs_run",
                  "n_parameters", "mean_degree", "n_edges", "hop_reachability",
                  "wiring_cost", "mean_edge_length"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary: dict = {
        "contract": "topic5_slp_cohort_aggregate_v0_1",
        "metric": METRIC,
        "metric_direction": "lower is better; a positive delta means arm A wins",
        "n_units": len(rows),
        "arms_present": sorted({r["arm"] for r in rows}),
        "comparisons": {},
    }

    for metric, label in ((METRIC, "primary"), (SECONDARY, "secondary")):
        table, converged = per_patient(rows, metric)
        for name, arm_a, arm_b in COMPARISONS:
            if arm_a not in table or arm_b not in table:
                continue
            entry = {}
            for stratum, subjects in strata.items():
                entry[stratum] = paired(table[arm_a], table[arm_b], subjects)
            # a cell that never converged cannot carry a negative verdict
            unconverged = sorted(
                s for s in strata["all"]
                if not converged.get(arm_a, {}).get(s, True)
                or not converged.get(arm_b, {}).get(s, True)
            )
            entry["patients_with_an_unconverged_arm"] = unconverged
            summary["comparisons"].setdefault(label, {})[name] = entry

    # How each metric scales with montage size.  The paired comparisons above are
    # within patient and so are safe either way, but the LEVELS are not
    # comparable across patients on the multi-label loss: with more contacts most
    # are absent at any step, so predicting absence well is enough to look good.
    cache = json.loads((OUT / "cache" / "CACHE_SUMMARY.json").read_text())
    n_contacts = {p["subject"]: p["n_contacts"] for p in cache["patients"]}
    scaling = {}
    for metric in (METRIC, SECONDARY):
        table, _ = per_patient(rows, metric)
        reference = table.get("STATIC_CONTACT", {})
        common = [s for s in reference if s in n_contacts]
        if len(common) >= 5:
            rho = stats.spearmanr(
                [n_contacts[s] for s in common], [reference[s] for s in common]
            ).statistic
            scaling[metric] = {
                "spearman_level_vs_contact_count": float(rho),
                "comparable_across_patients": bool(abs(rho) < 0.4),
            }
    summary["metric_scaling"] = scaling
    summary["metric_scaling_note"] = (
        "Levels of the multi-label loss fall as the montage grows, because most "
        "contacts are absent at any step and predicting absence is easy. The "
        "cardinality-conditioned rank loss rises instead, which is the honest "
        "direction: more contacts to choose between is a harder question. Read "
        "levels only on the rank loss; the paired differences are valid on both."
    )

    (args.out / "cohort_statistics.json").write_text(json.dumps(summary, indent=1))

    print(f"units aggregated: {len(rows)}   arms: {summary['arms_present']}")
    for label in summary["comparisons"]:
        print(f"\n--- {label} ({METRIC if label == 'primary' else SECONDARY}) ---")
        for name, entry in summary["comparisons"][label].items():
            allr = entry["all"]
            if allr["status"] != "COMPLETE":
                print(f"{name:22s} {allr['status']} n={allr['n']}")
                continue
            print(
                f"{name:22s} n={allr['n']:2d} median={allr['median_delta']:+.4f} "
                f"CI[{allr['bootstrap_95ci'][0]:+.4f},{allr['bootstrap_95ci'][1]:+.4f}] "
                f"pos={allr['n_positive']}/{allr['n']} p={allr['wilcoxon_two_sided_p']:.3g}"
            )
            for stratum in ("planar", "well_sampled"):
                s = entry[stratum]
                if s["status"] == "COMPLETE":
                    print(f"  {stratum:14s} n={s['n']:2d} median={s['median_delta']:+.4f} "
                          f"pos={s['n_positive']}/{s['n']} p={s['wilcoxon_two_sided_p']:.3g}")
            if entry["patients_with_an_unconverged_arm"]:
                print(f"  unconverged arms in: {entry['patients_with_an_unconverged_arm']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
