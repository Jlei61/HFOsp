#!/usr/bin/env python3
"""PR-6 Template Endpoint Anatomical Anchoring runner.

Step 2 deliverable (audit + per-subject) and Step 3+ (cohort statistics).
Plan: docs/archive/topic1/pr6_template_endpoint_anchoring_plan_2026-04-25.md

Usage:
    # Cohort audit -> cohort_audit.csv
    python scripts/run_pr6_template_anchoring.py --audit

    # Per-subject endpoint/middle SOZ enrichment
    python scripts/run_pr6_template_anchoring.py --per-subject

    # Cohort H1/H2/H3 statistics + dataset-specific sensitivity
    python scripts/run_pr6_template_anchoring.py --cohort

    # Run all three in order
    python scripts/run_pr6_template_anchoring.py --all
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.template_anatomical_anchoring import (
    audit_subject_eligibility,
    cohort_sign_test,
    cohort_wilcoxon,
    compute_subject_delta,
    compute_template_anchoring,
    forward_reverse_swap_check,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PER_SUBJECT_DIR = ROOT / "results" / "interictal_propagation" / "per_subject"
YUQUAN_SOZ_PATH = ROOT / "results" / "yuquan_soz_core_channels.json"
EPILEPSIAE_SOZ_PATH = ROOT / "results" / "epilepsiae_soz_core_channels.json"
EPILEPSIAE_FOCUS_REL_PATH = ROOT / "results" / "epilepsiae_electrode_focus_rel.json"

OUT_DIR = ROOT / "results" / "interictal_propagation" / "template_anchoring"
PER_SUBJECT_OUT = OUT_DIR / "per_subject"
AUDIT_CSV = OUT_DIR / "cohort_audit.csv"
COHORT_SUMMARY = OUT_DIR / "cohort_summary.json"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        return json.load(f)


def _parse_per_subject_filename(stem: str) -> Tuple[str, str]:
    """`yuquan_chenziyang` -> ('yuquan', 'chenziyang'); `epilepsiae_1073` -> ('epilepsiae', '1073')."""
    if stem.startswith("yuquan_"):
        return "yuquan", stem[len("yuquan_") :]
    if stem.startswith("epilepsiae_"):
        return "epilepsiae", stem[len("epilepsiae_") :]
    raise ValueError(f"Unknown dataset prefix in filename: {stem}")


def _load_candidate(path: Path, soz_yuquan: Dict, soz_epi: Dict, focus_rel: Dict) -> Dict[str, Any]:
    data = _load_json(path)
    dataset, subject_id = _parse_per_subject_filename(path.stem)

    ac = data.get("adaptive_cluster") or {}
    stable_k = ac.get("stable_k")
    clusters = ac.get("clusters") or []
    template_ranks = [c.get("template_rank") for c in clusters]
    channel_names = data.get("channel_names") or []

    soz_dict = soz_yuquan if dataset == "yuquan" else soz_epi
    soz_channels = list(soz_dict.get(subject_id) or [])
    focus_rel_dict = focus_rel.get(subject_id) if dataset == "epilepsiae" else None

    return {
        "subject_id": subject_id,
        "dataset": dataset,
        "stable_k": stable_k,
        "soz_channels": soz_channels,
        "channel_names": channel_names,
        "template_ranks": template_ranks,
        "focus_rel": focus_rel_dict,
        "raw_path": str(path),
        "inter_cluster_corr_matrix": ac.get("inter_cluster_corr_matrix"),
        "time_split_reproducibility": data.get("time_split_reproducibility") or {},
    }


def load_all_candidates() -> List[Dict[str, Any]]:
    if not PER_SUBJECT_DIR.exists():
        raise FileNotFoundError(f"Missing {PER_SUBJECT_DIR}")
    soz_yuquan = _load_json(YUQUAN_SOZ_PATH)
    soz_epi = _load_json(EPILEPSIAE_SOZ_PATH)
    focus_rel = _load_json(EPILEPSIAE_FOCUS_REL_PATH)

    paths = sorted(PER_SUBJECT_DIR.glob("*.json"))
    return [_load_candidate(p, soz_yuquan, soz_epi, focus_rel) for p in paths]


# ---------------------------------------------------------------------------
# Step 2a: audit
# ---------------------------------------------------------------------------
AUDIT_CSV_FIELDS = [
    "subject_id",
    "dataset",
    "stable_k",
    "n_ch",
    "n_soz_listed",
    "n_soz_matched",
    "endpoint_defined",
    "h1_primary_eligible",
    "pass",
    "exit_reason",
]


def _decorate_audit_rows(rows: List[Dict[str, Any]], candidates: List[Dict[str, Any]]):
    by_id = {c["subject_id"]: c for c in candidates}
    for row in rows:
        cand = by_id.get(row["subject_id"], {})
        tsr = cand.get("time_split_reproducibility") or {}
        splits = tsr.get("splits") or {}
        fh = splits.get("first_half_second_half") or {}
        row["forward_reverse_full_data"] = int(tsr.get("full_data_forward_reverse_pairs") or 0) > 0
        row["forward_reverse_reproduced"] = bool(fh.get("forward_reverse_reproduced") or False)
        row["has_focus_rel"] = cand.get("focus_rel") is not None


def run_audit() -> List[Dict[str, Any]]:
    candidates = load_all_candidates()
    rows = audit_subject_eligibility(candidates)
    _decorate_audit_rows(rows, candidates)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full_fields = AUDIT_CSV_FIELDS + [
        "forward_reverse_full_data",
        "forward_reverse_reproduced",
        "has_focus_rel",
    ]
    with AUDIT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=full_fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in full_fields})

    # Console summary
    n_total = len(rows)
    n_pass = sum(1 for r in rows if r["pass"])
    n_endpoint_defined = sum(1 for r in rows if r["endpoint_defined"])
    n_h1 = sum(1 for r in rows if r["h1_primary_eligible"])
    n_fwd_rev = sum(
        1 for r in rows if r["pass"] and r["forward_reverse_reproduced"]
    )
    print(f"[audit] candidates={n_total}")
    print(f"[audit] endpoint_defined (n_ch>=6)={n_endpoint_defined}")
    print(f"[audit] h1_primary_eligible (n_ch>=7)={n_h1}")
    print(f"[audit] pass (=h1_primary_eligible)={n_pass}")
    print(f"[audit] forward_reverse reproduced within pass cohort={n_fwd_rev}")
    by_exit = {}
    for r in rows:
        by_exit.setdefault(r["exit_reason"], 0)
        by_exit[r["exit_reason"]] += 1
    print("[audit] exit_reason distribution:")
    for k, v in sorted(by_exit.items(), key=lambda x: -x[1]):
        print(f"        {str(k):<20} {v}")
    print(f"[audit] csv -> {AUDIT_CSV}")

    return rows


# ---------------------------------------------------------------------------
# Step 2b: per-subject endpoint/middle SOZ enrichment
# ---------------------------------------------------------------------------
def run_per_subject(audit_rows: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    candidates = load_all_candidates()
    by_id = {c["subject_id"]: c for c in candidates}

    if audit_rows is None:
        audit_rows = audit_subject_eligibility(candidates)
        _decorate_audit_rows(audit_rows, candidates)

    PER_SUBJECT_OUT.mkdir(parents=True, exist_ok=True)
    written: List[Dict[str, Any]] = []

    for row in audit_rows:
        # Process both pass (H1) and endpoint_defined-only (case-series); skip
        # subjects where endpoint cannot be extracted at all.
        if not row["endpoint_defined"]:
            continue

        cand = by_id[row["subject_id"]]
        per_template_records = []
        for k_idx, tr in enumerate(cand["template_ranks"]):
            if tr is None:
                continue
            rec = compute_template_anchoring(
                channel_names=cand["channel_names"],
                template_rank=tr,
                soz_channels=cand["soz_channels"],
                focus_rel_dict=cand["focus_rel"],
                n=3,
            )
            rec["cluster_id"] = k_idx
            per_template_records.append(rec)

        delta = compute_subject_delta(
            per_template_records, focus_rel=cand["focus_rel"] is not None
        )

        # H2 swap mechanism: only meaningful when the subject has a forward/reverse
        # pair AND we have at least 2 templates with valid endpoint extraction.
        swap_check: Optional[Dict[str, Any]] = None
        valid_templates = [
            r for r in per_template_records if r.get("exit_reason") is None
        ]
        if (
            row["forward_reverse_reproduced"]
            and len(valid_templates) >= 2
        ):
            t0 = valid_templates[0]
            t1 = valid_templates[1]
            swap_check = forward_reverse_swap_check(
                t0_source=t0["source"],
                t0_sink=t0["sink"],
                t1_source=t1["source"],
                t1_sink=t1["sink"],
                channel_names=cand["channel_names"],
                n_perm=1000,
                seed=0,
            )

        out_obj = {
            "subject_id": row["subject_id"],
            "dataset": row["dataset"],
            "audit": row,
            "per_template": per_template_records,
            "subject_delta": delta,
            "h2_swap_check": swap_check,
        }
        out_path = PER_SUBJECT_OUT / f"{row['dataset']}_{row['subject_id']}.json"
        with out_path.open("w") as f:
            json.dump(out_obj, f, indent=2, default=_json_default)
        written.append(out_obj)

    print(f"[per-subject] wrote {len(written)} subject json files -> {PER_SUBJECT_OUT}")
    return written


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"Object of type {type(o)} not serializable")


# ---------------------------------------------------------------------------
# Step 3: cohort statistics (H1/H1b/H2/H3)
# ---------------------------------------------------------------------------
def run_cohort(per_subject_records: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    if per_subject_records is None:
        if not PER_SUBJECT_OUT.exists():
            raise FileNotFoundError(
                f"{PER_SUBJECT_OUT} missing — run --per-subject first"
            )
        per_subject_records = []
        for p in sorted(PER_SUBJECT_OUT.glob("*.json")):
            with p.open() as f:
                per_subject_records.append(json.load(f))

    # H1 cohort: pooled (all h1_primary_eligible) subject deltas
    h1_subjects = [
        r for r in per_subject_records if r["audit"].get("h1_primary_eligible")
    ]
    h1_deltas_pooled = [
        r["subject_delta"]["delta_endpoint_vs_middle"] for r in h1_subjects
    ]
    h1_deltas_yuquan = [
        r["subject_delta"]["delta_endpoint_vs_middle"]
        for r in h1_subjects
        if r["dataset"] == "yuquan"
    ]
    h1_deltas_epi = [
        r["subject_delta"]["delta_endpoint_vs_middle"]
        for r in h1_subjects
        if r["dataset"] == "epilepsiae"
    ]

    h1_pooled = {
        "subject_ids": [r["subject_id"] for r in h1_subjects],
        "wilcoxon_greater": cohort_wilcoxon(h1_deltas_pooled, "greater"),
        "wilcoxon_two_sided": cohort_wilcoxon(h1_deltas_pooled, "two-sided"),
        "sign_test": cohort_sign_test(h1_deltas_pooled),
    }
    h1_yuquan = {
        "n": len(h1_deltas_yuquan),
        "wilcoxon_greater": cohort_wilcoxon(h1_deltas_yuquan, "greater"),
        "sign_test": cohort_sign_test(h1_deltas_yuquan),
    }
    h1_epi = {
        "n": len(h1_deltas_epi),
        "wilcoxon_greater": cohort_wilcoxon(h1_deltas_epi, "greater"),
        "sign_test": cohort_sign_test(h1_deltas_epi),
    }

    # H1b polarity (source vs sink): only on non-forward/reverse subset to avoid
    # cancellation
    h1b_subjects = [
        r
        for r in h1_subjects
        if not r["audit"].get("forward_reverse_reproduced")
    ]
    h1b_deltas = [
        r["subject_delta"]["delta_source_vs_sink"] for r in h1b_subjects
    ]
    h1b = {
        "n": len(h1b_deltas),
        "subject_ids": [r["subject_id"] for r in h1b_subjects],
        "wilcoxon_two_sided": cohort_wilcoxon(h1b_deltas, "two-sided"),
        "sign_test": cohort_sign_test(h1b_deltas),
    }

    # H2 forward/reverse swap mechanism
    h2_subjects = [
        r
        for r in per_subject_records
        if r["audit"].get("forward_reverse_reproduced")
        and r.get("h2_swap_check") is not None
    ]
    h2_records = []
    for r in h2_subjects:
        sc = r["h2_swap_check"]
        h2_records.append(
            {
                "subject_id": r["subject_id"],
                "dataset": r["dataset"],
                "swap_score": sc["swap_score"],
                "null_p": sc["null_p"],
                "null_95th": sc["null_95th"],
                "exceeds_null_95": sc["swap_score"] > sc["null_95th"],
            }
        )
    n_exceed = sum(1 for r in h2_records if r["exceeds_null_95"])
    h2 = {
        "n": len(h2_records),
        "n_exceeds_null_95": n_exceed,
        "per_subject": h2_records,
    }

    # H3 focus_rel (Epilepsiae only): i / l / e endpoint vs middle deltas
    h3_subjects = [
        r
        for r in h1_subjects
        if r["dataset"] == "epilepsiae"
        and r["audit"].get("has_focus_rel")
    ]
    h3 = {}
    for label in ("i", "l", "e"):
        key = f"delta_{label}_endpoint_vs_middle"
        deltas = []
        sids = []
        for r in h3_subjects:
            d = r["subject_delta"].get(key)
            if d is not None and np.isfinite(d):
                deltas.append(d)
                sids.append(r["subject_id"])
        h3[label] = {
            "n": len(deltas),
            "subject_ids": sids,
            "wilcoxon_greater": cohort_wilcoxon(deltas, "greater"),
            "wilcoxon_two_sided": cohort_wilcoxon(deltas, "two-sided"),
            "sign_test": cohort_sign_test(deltas),
        }

    summary = {
        "h1_pooled": h1_pooled,
        "h1_dataset_specific": {"yuquan": h1_yuquan, "epilepsiae": h1_epi},
        "h1b_polarity_non_fwdrev": h1b,
        "h2_forward_reverse_swap": h2,
        "h3_focus_rel_epilepsiae": h3,
        "n_per_subject_records": len(per_subject_records),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with COHORT_SUMMARY.open("w") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    print(f"[cohort] H1 pooled n={h1_pooled['wilcoxon_greater']['n']} "
          f"median={h1_pooled['wilcoxon_greater']['median']:.4f} "
          f"p_greater={h1_pooled['wilcoxon_greater']['p_value']:.4g}")
    print(f"[cohort] H1 yuquan   n={h1_yuquan['n']}  | epilepsiae n={h1_epi['n']}")
    print(f"[cohort] H1b polarity (non-fwdrev) n={h1b['n']}")
    print(f"[cohort] H2 swap n={h2['n']} exceeding null_95th={n_exceed}")
    for label, rec in h3.items():
        print(f"[cohort] H3 {label} n={rec['n']} "
              f"median={rec['wilcoxon_greater']['median']}")
    print(f"[cohort] summary -> {COHORT_SUMMARY}")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit", action="store_true", help="Step 2a: cohort audit -> cohort_audit.csv")
    ap.add_argument(
        "--per-subject",
        action="store_true",
        help="Step 2b: per-subject endpoint/middle JSON",
    )
    ap.add_argument(
        "--cohort",
        action="store_true",
        help="Step 3: H1/H1b/H2/H3 cohort statistics",
    )
    ap.add_argument("--all", action="store_true", help="Run audit + per-subject + cohort")
    args = ap.parse_args()

    if not (args.audit or args.per_subject or args.cohort or args.all):
        ap.print_help()
        sys.exit(2)

    audit_rows = None
    per_subj_recs = None
    if args.audit or args.all:
        audit_rows = run_audit()
    if args.per_subject or args.all:
        per_subj_recs = run_per_subject(audit_rows)
    if args.cohort or args.all:
        run_cohort(per_subj_recs)


if __name__ == "__main__":
    main()
