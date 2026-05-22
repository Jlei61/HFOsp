"""SEF-ITP Phase 1 cohort aggregator.

Reads per-subject JSONs produced by scripts/run_sef_itp_phase1.py and writes:

  - results/topic4_sef_itp/phase1_spatial_geometry/cohort_summary.json
  - results/topic4_sef_itp/phase1_spatial_geometry/cohort_subjects.csv

Reports the cohort funnel (40→34→30→23), verdict distributions per hypothesis,
coord-coverage stats, and cohort-level binomial-style aggregations.

Plan: docs/archive/topic4/sef_itp_phase1/plan_2026-05-21.md §5.4 cohort summary
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Verdict categories used for cohort-level aggregation.
H1_PASS_LIKE = {"PASS", "partial_PASS", "PASS_one_side_untestable"}
H1_NULL_LIKE = {"NULL", "NULL_one_side_untestable"}
H1_FAIL_LIKE = {"FAIL", "FAIL_DIFFUSE"}
H1_UNTESTABLE = {
    "UNTESTABLE_BOTH_SIDES",
    "INCOMPLETE_GATED_ON_COORDS",
    "INCONCLUSIVE_ENVELOPE_INDETERMINATE",
}

H2_PASS_LIKE = {"PASS", "partial_PASS"}
H2_NULL_LIKE = {"NULL"}
H2_FAIL_LIKE = {"FAIL"}
H2_UNTESTABLE = {"GATED_NO_COORDS", "EMPTY_SET", "INSUFFICIENT_NULL", "DEGENERATE"}

H6_PASS_LIKE = {"PASS", "PARTIAL"}
H6_NULL_LIKE = {"NULL"}
H6_UNTESTABLE = {
    "EXCLUDED_SINGLE_SHAFT",
    "INSUFFICIENT_SPLIT",
    "INSUFFICIENT_NULL",
    "GATED_NO_COORDS",
}


def _categorize(verdict: Optional[str], pass_set, null_set, fail_set, untestable_set) -> str:
    if verdict in pass_set:
        return "pass_like"
    if verdict in null_set:
        return "null_like"
    if verdict in fail_set:
        return "fail_like"
    if verdict in untestable_set:
        return "untestable"
    return "other"


def aggregate(per_subject_dir: Path) -> Dict[str, Any]:
    files = sorted(per_subject_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No per-subject JSONs under {per_subject_dir}")

    subjects: List[Dict[str, Any]] = []

    # Verdict counters
    h6_verdicts = Counter()
    h1_verdicts_per_cluster = Counter()
    h2_verdicts_per_pair = Counter()
    h2_set_verdicts = Counter()
    h2_spatial_verdicts = Counter()

    # Coord coverage
    n_mapped_list = []
    n_dropped_total = 0
    coord_spaces = Counter()
    norm_certs = Counter()

    # Cohort categorical buckets
    h1_categorical = Counter()  # per (subject, cluster)
    h6_categorical = Counter()  # per subject
    h2_categorical = Counter()  # per (subject, pair)

    # Per-subject row for CSV
    csv_rows: List[Dict[str, Any]] = []

    for f in files:
        d = json.load(open(f))
        sid = d["subject_id"]
        ds = d.get("dataset", "unknown")

        # Coord coverage
        n_mapped = d.get("n_coord_mapped")
        if n_mapped is not None:
            n_mapped_list.append(n_mapped)
        coord_space = d.get("coord_space")
        if coord_space:
            coord_spaces[coord_space] += 1
        prov = d.get("coord_provenance") or {}
        nc = prov.get("normalization_certainty")
        if nc:
            norm_certs[nc] += 1
        dropped = d.get("n_dropped_endpoints_no_coords_per_cluster", {}) or {}
        n_drop_subject = sum(int(v) for v in dropped.values())
        n_dropped_total += n_drop_subject

        # H6
        h6_v = (d.get("h6") or {}).get("verdict")
        h6_verdicts[h6_v] += 1
        h6_categorical[_categorize(h6_v, H6_PASS_LIKE, H6_NULL_LIKE, set(), H6_UNTESTABLE)] += 1

        # H1 per cluster
        h1_cluster_block = (d.get("h1") or {}).get("per_cluster") or {}
        h1_subject_overall = []
        for cid, body in h1_cluster_block.items():
            v = body.get("h1_overall_verdict")
            h1_verdicts_per_cluster[v] += 1
            h1_categorical[_categorize(v, H1_PASS_LIKE, H1_NULL_LIKE, H1_FAIL_LIKE, H1_UNTESTABLE)] += 1
            h1_subject_overall.append({"cluster_id": int(cid), "verdict": v})

        # H2 per pair
        h2_pairs = (d.get("h2") or {}).get("per_pair") or []
        h2_subject_pairs = []
        for pair in h2_pairs:
            v_integrated = pair.get("h2_integrated_verdict")
            v_set = (pair.get("h2_set_reversal") or {}).get("verdict")
            v_spatial = (pair.get("h2_spatial_reversal") or {}).get("verdict")
            h2_verdicts_per_pair[v_integrated] += 1
            h2_set_verdicts[v_set] += 1
            h2_spatial_verdicts[v_spatial] += 1
            h2_categorical[_categorize(v_integrated, H2_PASS_LIKE, H2_NULL_LIKE, H2_FAIL_LIKE, H2_UNTESTABLE)] += 1
            h2_subject_pairs.append({
                "cluster_A_id": pair.get("cluster_A_id"),
                "cluster_B_id": pair.get("cluster_B_id"),
                "h2_integrated_verdict": v_integrated,
                "h2_set_verdict": v_set,
                "h2_spatial_verdict": v_spatial,
            })

        subjects.append({
            "subject_id": sid,
            "dataset": ds,
            "n_channels": d.get("n_channels"),
            "n_coord_mapped": n_mapped,
            "n_dropped_endpoints_no_coords": n_drop_subject,
            "coord_space": coord_space,
            "h6_verdict": h6_v,
            "h1_clusters": h1_subject_overall,
            "h2_pairs": h2_subject_pairs,
        })

        csv_rows.append({
            "subject": f"{ds}_{sid}",
            "dataset": ds,
            "n_channels": d.get("n_channels"),
            "n_coord_mapped": n_mapped,
            "n_dropped": n_drop_subject,
            "coord_space": coord_space,
            "h6": h6_v,
            "h1_c0": h1_subject_overall[0]["verdict"] if len(h1_subject_overall) > 0 else None,
            "h1_c1": h1_subject_overall[1]["verdict"] if len(h1_subject_overall) > 1 else None,
            "h2_integrated_first_pair": h2_subject_pairs[0]["h2_integrated_verdict"] if h2_subject_pairs else None,
            "n_h2_pairs": len(h2_subject_pairs),
        })

    # Cohort funnel — derived from Phase 0a + PR-6 + coord availability
    # (we don't recompute here; we report what made it through to Phase 1)
    cohort_funnel = {
        "phase1_admitted": len(files),
        "by_dataset": {
            "yuquan": sum(1 for s in subjects if s["dataset"] == "yuquan"),
            "epilepsiae": sum(1 for s in subjects if s["dataset"] == "epilepsiae"),
        },
    }

    return {
        "schema_version": "sef_itp_phase1_cohort_v1_2026_05_22",
        "n_subjects": len(files),
        "cohort_funnel": cohort_funnel,
        "coord_coverage": {
            "n_coord_mapped_distribution": {
                "median": float(_median(n_mapped_list)) if n_mapped_list else None,
                "min": min(n_mapped_list) if n_mapped_list else None,
                "max": max(n_mapped_list) if n_mapped_list else None,
                "mean": sum(n_mapped_list) / len(n_mapped_list) if n_mapped_list else None,
            },
            "n_endpoints_dropped_total": n_dropped_total,
            "coord_space_distribution": dict(coord_spaces),
            "normalization_certainty_distribution": dict(norm_certs),
        },
        "h6": {
            "verdict_distribution": dict(h6_verdicts),
            "categorical_distribution": dict(h6_categorical),
            "n_subjects": sum(h6_verdicts.values()),
        },
        "h1": {
            "verdict_distribution_per_cluster": dict(h1_verdicts_per_cluster),
            "categorical_distribution": dict(h1_categorical),
            "n_clusters_total": sum(h1_verdicts_per_cluster.values()),
        },
        "h2": {
            "verdict_distribution_per_pair_integrated": dict(h2_verdicts_per_pair),
            "verdict_distribution_per_pair_set": dict(h2_set_verdicts),
            "verdict_distribution_per_pair_spatial": dict(h2_spatial_verdicts),
            "categorical_distribution": dict(h2_categorical),
            "n_pairs_total": sum(h2_verdicts_per_pair.values()),
        },
        "subjects": subjects,
    }, csv_rows


def _median(xs: List[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return float("nan")
    if n % 2 == 1:
        return xs[n // 2]
    return (xs[n // 2 - 1] + xs[n // 2]) / 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-subject-dir",
        type=Path,
        default=Path("results/topic4_sef_itp/phase1_spatial_geometry/per_subject"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/topic4_sef_itp/phase1_spatial_geometry"),
    )
    args = parser.parse_args(argv)

    summary, csv_rows = aggregate(args.per_subject_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    out_json = args.output_dir / "cohort_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    out_csv = args.output_dir / "cohort_subjects.csv"
    with open(out_csv, "w", newline="") as f:
        if csv_rows:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    # Print summary to stdout for quick inspection
    print(f"== SEF-ITP Phase 1 cohort summary (n={summary['n_subjects']}) ==")
    print(f"  by dataset: {summary['cohort_funnel']['by_dataset']}")
    print(f"  coord coverage median: {summary['coord_coverage']['n_coord_mapped_distribution']['median']}")
    print(f"  total endpoints dropped (no coord): {summary['coord_coverage']['n_endpoints_dropped_total']}")
    print(f"\nH6 verdicts:")
    for v, n in sorted(summary['h6']['verdict_distribution'].items(), key=lambda x: -x[1]):
        print(f"  {v}: {n}")
    print(f"  categorical: {summary['h6']['categorical_distribution']}")
    print(f"\nH1 per-cluster verdicts (total {summary['h1']['n_clusters_total']} clusters):")
    for v, n in sorted(summary['h1']['verdict_distribution_per_cluster'].items(), key=lambda x: -x[1]):
        print(f"  {v}: {n}")
    print(f"  categorical: {summary['h1']['categorical_distribution']}")
    print(f"\nH2 per-pair integrated verdicts (total {summary['h2']['n_pairs_total']} pairs):")
    for v, n in sorted(summary['h2']['verdict_distribution_per_pair_integrated'].items(), key=lambda x: -x[1]):
        print(f"  {v}: {n}")
    print(f"  set-based: {summary['h2']['verdict_distribution_per_pair_set']}")
    print(f"  spatial:   {summary['h2']['verdict_distribution_per_pair_spatial']}")
    print(f"  categorical: {summary['h2']['categorical_distribution']}")
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
