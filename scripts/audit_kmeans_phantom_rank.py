#!/usr/bin/env python3
"""Cohort audit driver for lagPatRank phantom pseudo-rank bug.

Per the plan in
``docs/superpowers/plans/2026-05-20-lagpatrank-phantom-rank-audit-and-fix.md``
Step 2. Loops over all 40 subjects, computes per-subject AMI noise floor on
original (phantom-contaminated) and masked feature matrices, plus the
cross-AMI(orig@seed=0, masked@seed=0) audit signal. Cohort summary CSV +
per-subject JSON.

The Step 3 gate rule is applied OUTSIDE this script — this script only
emits the raw numbers.

Usage::

    python scripts/audit_kmeans_phantom_rank.py --all
    python scripts/audit_kmeans_phantom_rank.py --subject yuquan_chengshuai
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.interictal_propagation import (  # noqa: E402
    _valid_event_indices,
    load_subject_propagation_events,
)
from src.lagpat_rank_audit import kmeans_label_ami_audit  # noqa: E402
from sklearn.metrics import adjusted_mutual_info_score  # noqa: E402


YUQUAN_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")

PR2_JSON_DIR = REPO_ROOT / "results" / "interictal_propagation" / "per_subject"
STEP0_AUDIT_CSV = REPO_ROOT / "results" / "topic4_attractor" / "step0_audit.csv"
OUT_DIR = REPO_ROOT / "results" / "lagpatrank_audit"


def _epilepsiae_subject_dir(subject: str) -> Path:
    legacy = EPILEPSIAE_ROOT / subject / "all_recs"
    return legacy if legacy.exists() else EPILEPSIAE_ROOT / subject


def _subject_dir(dataset: str, subject: str) -> Path:
    if dataset == "yuquan":
        return YUQUAN_ROOT / subject
    return _epilepsiae_subject_dir(subject)


def _load_subject_inventory() -> List[Tuple[str, str, str]]:
    """Return list of (sid, dataset, subject) for ALL 40 subjects (not just
    eligible_for_main). High-k subjects may have polluted clusters too — audit
    them regardless of Topic 4 eligibility gates."""
    out: List[Tuple[str, str, str]] = []
    with open(STEP0_AUDIT_CSV) as f:
        for row in csv.DictReader(f):
            out.append((row["sid"], row["dataset"], row["subject"]))
    return out


def _read_pr2(json_path: Path) -> Dict[str, Any]:
    """Return PR-2 metadata for one subject, or {} if file missing."""
    if not json_path.exists():
        return {}
    with open(json_path) as f:
        full = json.load(f)
    ac = full.get("adaptive_cluster", {}) or {}
    return {
        "chosen_k": ac.get("chosen_k"),
        "stable_k": ac.get("stable_k"),
        "chosen_reason": ac.get("chosen_reason"),
        "n_valid_events": ac.get("n_valid_events"),
        "labels": ac.get("labels"),
        "cluster_fractions": [c.get("fraction") for c in ac.get("clusters", [])],
    }


def _audit_one(
    sid: str, dataset: str, subject: str, n_seeds: int = 5,
) -> Dict[str, Any]:
    """Audit one subject. Returns flat dict (no arrays) for CSV + JSON."""
    sub_dir = _subject_dir(dataset, subject)
    if not sub_dir.exists():
        return {"sid": sid, "status": "subject_dir_missing", "subject_dir": str(sub_dir)}

    try:
        evt = load_subject_propagation_events(sub_dir)
    except FileNotFoundError as e:
        return {"sid": sid, "status": "no_lagpat_npz", "error": str(e)}
    except Exception as e:  # noqa: BLE001 — defensive for cohort driver
        return {"sid": sid, "status": "load_error", "error": repr(e)}

    ranks = evt["ranks"]
    bools = evt["bools"]
    chans = evt["channel_names"]
    n_ch, n_ev_total = ranks.shape
    valid = _valid_event_indices(bools, min_participating=3)

    pr2 = _read_pr2(PR2_JSON_DIR / f"{sid}.json")
    chosen_k = pr2.get("chosen_k")
    if chosen_k is None or int(chosen_k) < 2:
        return {
            "sid": sid, "dataset": dataset, "subject": subject,
            "status": "no_pr2_chosen_k", "n_ch": n_ch, "n_ev_total": n_ev_total,
            "n_valid": int(valid.size),
        }
    chosen_k = int(chosen_k)

    if valid.size < 2 * chosen_k:
        return {
            "sid": sid, "dataset": dataset, "subject": subject,
            "status": "too_few_valid_events", "n_valid": int(valid.size),
            "chosen_k": chosen_k, "n_ch": n_ch, "n_ev_total": n_ev_total,
        }

    t0 = time.perf_counter()
    audit = kmeans_label_ami_audit(
        ranks, bools, k=chosen_k, n_seeds=n_seeds, valid_event_indices=valid,
    )
    elapsed = time.perf_counter() - t0

    # Sanity: AMI between our seed=0 original labels and PR-2 stored labels.
    pr2_label_recovery_ami: Optional[float] = None
    labels_stored = pr2.get("labels")
    if labels_stored is not None:
        labels_stored_arr = np.asarray(labels_stored, dtype=int)
        if labels_stored_arr.size == audit["labels_original_seed0"].size:
            pr2_label_recovery_ami = float(
                adjusted_mutual_info_score(
                    labels_stored_arr, audit["labels_original_seed0"]
                )
            )

    # Cluster size delta
    lab_o = np.asarray(audit["labels_original_seed0"], dtype=int)
    lab_m = np.asarray(audit["labels_masked_seed0"], dtype=int)
    frac_o = float((lab_o == 0).sum()) / float(lab_o.size)
    frac_m = float((lab_m == 0).sum()) / float(lab_m.size)

    return {
        "sid": sid, "dataset": dataset, "subject": subject,
        "status": "ok",
        "n_ch": n_ch, "n_ev_total": n_ev_total, "n_valid": int(valid.size),
        "chosen_k": chosen_k,
        "stable_k": pr2.get("stable_k"),
        "phantom_fraction": audit["phantom_fraction"],
        "ami_seed_floor_original": audit["ami_seed_floor_original"],
        "ami_seed_floor_original_min": audit["ami_seed_floor_original_min"],
        "ami_seed_floor_masked": audit["ami_seed_floor_masked"],
        "ami_seed_floor_masked_min": audit["ami_seed_floor_masked_min"],
        "ami_audit": audit["ami_audit"],
        "ami_audit_minus_floor": audit["ami_audit_minus_floor"],
        "frac_cluster0_original": frac_o,
        "frac_cluster0_masked": frac_m,
        "pr2_label_recovery_ami": pr2_label_recovery_ami,
        "n_seeds": int(n_seeds),
        "elapsed_sec": float(elapsed),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="Audit all 40 subjects")
    ap.add_argument("--subject", type=str, default=None,
                    help='Single subject id like "yuquan_chengshuai" or "epilepsiae_548"')
    ap.add_argument("--n-seeds", type=int, default=5)
    args = ap.parse_args()

    if not args.all and not args.subject:
        ap.error("--all or --subject required")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    inv = _load_subject_inventory()
    if args.subject is not None:
        inv = [(sid, d, s) for (sid, d, s) in inv if sid == args.subject]
        if not inv:
            print(f"subject {args.subject} not in step0 audit csv", file=sys.stderr)
            return 1

    rows: List[Dict[str, Any]] = []
    for i, (sid, dataset, subject) in enumerate(inv, 1):
        print(f"[{i:2d}/{len(inv)}] {sid} ...", flush=True)
        out = _audit_one(sid, dataset, subject, n_seeds=args.n_seeds)
        rows.append(out)
        out_path = OUT_DIR / f"{sid}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2, default=lambda o: int(o) if isinstance(o, np.integer)
                       else float(o) if isinstance(o, np.floating) else str(o))
        if out.get("status") == "ok":
            print(
                f"    n_ch={out['n_ch']} n_valid={out['n_valid']} k={out['chosen_k']} "
                f"phantom={out['phantom_fraction']:.3f} "
                f"floor_orig={out['ami_seed_floor_original']:.3f} "
                f"floor_mask={out['ami_seed_floor_masked']:.3f} "
                f"audit={out['ami_audit']:.3f} "
                f"Δ={out['ami_audit_minus_floor']:+.3f} "
                f"({out['elapsed_sec']:.1f}s)"
            )
        else:
            print(f"    status={out.get('status')} {out.get('error', '')}")

    # Cohort summary CSV
    cohort_csv = OUT_DIR / "cohort_summary.csv"
    fieldnames = [
        "sid", "dataset", "subject", "status",
        "n_ch", "n_ev_total", "n_valid",
        "chosen_k", "stable_k", "phantom_fraction",
        "ami_seed_floor_original", "ami_seed_floor_original_min",
        "ami_seed_floor_masked", "ami_seed_floor_masked_min",
        "ami_audit", "ami_audit_minus_floor",
        "frac_cluster0_original", "frac_cluster0_masked",
        "pr2_label_recovery_ami", "n_seeds", "elapsed_sec",
    ]
    with open(cohort_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\ncohort summary -> {cohort_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
