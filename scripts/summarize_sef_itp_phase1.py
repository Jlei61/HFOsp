"""SEF-ITP Phase 1 cohort aggregator.

Reads per-subject JSONs produced by scripts/run_sef_itp_phase1.py and writes:

  - results/topic4_sef_itp/phase1_spatial_geometry/cohort_summary.json
  - results/topic4_sef_itp/phase1_spatial_geometry/cohort_subjects.csv

Reports the cohort funnel (40→34→30→23), verdict distributions per hypothesis,
coord-coverage stats, and H2 rank-displacement swap-class counts.

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

    # H2 rank-displacement swap-k accumulators (mechanism sanity — no verdict)
    h2_swap_scores: List[float] = []
    h2_p_fws: List[float] = []
    h2_n_testable = 0
    h2_n_not_testable = 0  # rank-displacement missing or not stable_k=2 one-pair
    h2_swap_class_counts = Counter({"strict": 0, "candidate": 0, "none": 0, "unknown": 0})
    h2_spatial_verdicts = {
        "source_side": Counter(),
        "sink_side": Counter(),
        "combined_endpoint": Counter(),
    }

    # Coord coverage
    n_mapped_list = []
    n_dropped_total = 0
    coord_spaces = Counter()
    norm_certs = Counter()

    # Cohort categorical buckets
    h1_categorical = Counter()  # per (subject, cluster)
    h6_categorical = Counter()  # per subject

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

        # H2 (v1.0.10): rank-displacement swap_sweep channel labels.
        # NO cohort PASS/NULL/FAIL verdict (mechanism-sanity tier).
        h2_block = d.get("h2") or {}
        h2_subject_swap: Optional[Dict] = None
        if h2_block.get("available"):
            sc = h2_block.get("swap_score")
            p_fw = h2_block.get("p_fw", h2_block.get("null_p"))
            swap_class = h2_block.get("swap_class", "unknown")
            if swap_class not in h2_swap_class_counts:
                swap_class = "unknown"
            h2_subject_swap = {
                "swap_score": sc,
                "p_fw": p_fw,
                "null_p": p_fw,  # backward-compatible alias for old CSV readers
                "swap_class": swap_class,
                "decision_k": h2_block.get("decision_k"),
                "swap_source_channels": h2_block.get("swap_source_channels"),
                "swap_sink_channels": h2_block.get("swap_sink_channels"),
                "swap_endpoint_channels": h2_block.get("swap_endpoint_channels"),
                "n_valid": h2_block.get("n_valid"),
                "source_contract": h2_block.get("source_contract"),
                "spatial_compactness": h2_block.get("spatial_compactness"),
            }
            h2_n_testable += 1
            if sc is not None:
                h2_swap_scores.append(float(sc))
            if p_fw is not None:
                h2_p_fws.append(float(p_fw))
            h2_swap_class_counts[swap_class] += 1
            spatial = h2_block.get("spatial_compactness") or {}
            if spatial.get("available"):
                for side in ("source_side", "sink_side", "combined_endpoint"):
                    verdict = (spatial.get(side) or {}).get("verdict")
                    if verdict is not None:
                        h2_spatial_verdicts[side][verdict] += 1
        else:
            h2_n_not_testable += 1

        subjects.append({
            "subject_id": sid,
            "dataset": ds,
            "n_channels": d.get("n_channels"),
            "n_coord_mapped": n_mapped,
            "n_dropped_endpoints_no_coords": n_drop_subject,
            "coord_space": coord_space,
            "h6_verdict": h6_v,
            "h1_clusters": h1_subject_overall,
            "h2_swap_check": h2_subject_swap,
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
            "h2_swap_score": h2_subject_swap["swap_score"] if h2_subject_swap else None,
            "h2_p_fw": h2_subject_swap["p_fw"] if h2_subject_swap else None,
            "h2_null_p": h2_subject_swap["null_p"] if h2_subject_swap else None,
            "h2_swap_class": h2_subject_swap["swap_class"] if h2_subject_swap else None,
            "h2_decision_k": h2_subject_swap["decision_k"] if h2_subject_swap else None,
            "h2_source_spatial_verdict": (
                ((h2_subject_swap.get("spatial_compactness") or {}).get("source_side") or {}).get("verdict")
                if h2_subject_swap else None
            ),
            "h2_sink_spatial_verdict": (
                ((h2_subject_swap.get("spatial_compactness") or {}).get("sink_side") or {}).get("verdict")
                if h2_subject_swap else None
            ),
            "h2_combined_spatial_verdict": (
                ((h2_subject_swap.get("spatial_compactness") or {}).get("combined_endpoint") or {}).get("verdict")
                if h2_subject_swap else None
            ),
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
            "tier_lock": "directional_mechanism_sanity_not_cohort_claim",
            "source_contract": "rank_displacement_swap_sweep",
            "n_testable_total": h2_n_testable,
            "n_not_testable": h2_n_not_testable,
            "swap_class_distribution": dict(h2_swap_class_counts),
            "n_strict": h2_swap_class_counts["strict"],
            "n_candidate": h2_swap_class_counts["candidate"],
            "n_strict_or_candidate": (
                h2_swap_class_counts["strict"] + h2_swap_class_counts["candidate"]
            ),
            "spatial_compactness_verdict_distribution": {
                side: dict(counter) for side, counter in h2_spatial_verdicts.items()
            },
            "swap_score_median": float(_median(h2_swap_scores)) if h2_swap_scores else None,
            "p_fw_median": float(_median(h2_p_fws)) if h2_p_fws else None,
            "null_p_median": float(_median(h2_p_fws)) if h2_p_fws else None,
            "note": (
                "Topic4 H2: rank-displacement swap-k defines labels; spatial "
                "compactness tests source/sink swap-k sides against other valid nodes. "
                "No cohort PASS/NULL/FAIL verdict."
            ),
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
    h2 = summary["h2"]
    print(f"\nH2 (rank-displacement swap-k — mechanism sanity, NOT cohort claim):")
    print(f"  TOTAL n_testable: {h2['n_testable_total']}, n_not_testable: {h2['n_not_testable']}")
    print(f"  swap_class distribution: {h2['swap_class_distribution']}")
    print(f"  strict + candidate: {h2['n_strict_or_candidate']} / {h2['n_testable_total']}")
    print(f"  source-side spatial verdicts: {h2['spatial_compactness_verdict_distribution']['source_side']}")
    print(f"  sink-side spatial verdicts: {h2['spatial_compactness_verdict_distribution']['sink_side']}")
    print(f"  swap_score median: {h2['swap_score_median']}")
    print(f"  p_fw median: {h2['p_fw_median']}")
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
