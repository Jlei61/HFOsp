#!/usr/bin/env python3
"""Augment lagpatrank_audit per-subject JSONs with masked stable_k computation.

Companion to ``scripts/audit_kmeans_phantom_rank.py``. The base audit
computes AMI at PR-2's chosen_k. This script answers the orthogonal
question: **if we re-cluster on masked features, what k does the stable_k
selection rule pick?**

Per the plan
``docs/superpowers/plans/2026-05-20-lagpatrank-phantom-rank-audit-and-fix.md``
§2.1, this is needed before the archive doc can claim which downstream PRs
require re-derivation (a cohort-wide stable_k flip implies the entire
adaptive-cluster pipeline changes).

Loops the same 40 subjects, scans k ∈ {2..6} on masked features via
``_kmeans_stability_for_k``, picks ``stable_k_masked`` as the k with the
highest median silhouette among those passing both stability and
min-fraction gates. Falls back to "no_stable_k" if none pass.

Reads/writes ``results/lagpatrank_audit/<sid>.json`` in place, and updates
``cohort_summary.csv`` with new columns ``stable_k_masked`` /
``stable_k_changed``.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.interictal_propagation import (  # noqa: E402
    _kmeans_stability_for_k,
    _valid_event_indices,
    load_subject_propagation_events,
)
from src.lagpat_rank_audit import build_masked_kmeans_features  # noqa: E402

YUQUAN_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")
STEP0_AUDIT_CSV = REPO_ROOT / "results" / "topic4_attractor" / "step0_audit.csv"
OUT_DIR = REPO_ROOT / "results" / "lagpatrank_audit"

K_RANGE = list(range(2, 7))  # match PR-2 default (k_range = (2, 8) truncated for cost)


def _epi_dir(subject: str) -> Path:
    legacy = EPILEPSIAE_ROOT / subject / "all_recs"
    return legacy if legacy.exists() else EPILEPSIAE_ROOT / subject


def _subject_dir(dataset: str, subject: str) -> Path:
    return YUQUAN_ROOT / subject if dataset == "yuquan" else _epi_dir(subject)


def _pick_stable_k(scan: List[Dict[str, Any]]) -> Dict[str, Any]:
    passing = [r for r in scan if r.get("passes_both")]
    if not passing:
        return {"stable_k_masked": None, "chosen_reason": "no_passing_k"}
    best = max(passing, key=lambda r: r.get("median_silhouette", -np.inf))
    return {
        "stable_k_masked": int(best["k"]),
        "chosen_reason": "stable_k",
        "passing_ks": [int(r["k"]) for r in passing],
    }


def main() -> int:
    inv: List[tuple[str, str, str]] = []
    with open(STEP0_AUDIT_CSV) as f:
        for row in csv.DictReader(f):
            inv.append((row["sid"], row["dataset"], row["subject"]))

    rows_updated: List[Dict[str, Any]] = []
    for i, (sid, dataset, subject) in enumerate(inv, 1):
        per_subj_path = OUT_DIR / f"{sid}.json"
        if not per_subj_path.exists():
            print(f"[{i:2d}/{len(inv)}] {sid}: per-subject JSON missing, skip")
            continue
        with open(per_subj_path) as f:
            payload = json.load(f)
        if payload.get("status") != "ok":
            print(f"[{i:2d}/{len(inv)}] {sid}: base audit status={payload.get('status')}, skip")
            continue

        sub_dir = _subject_dir(dataset, subject)
        try:
            evt = load_subject_propagation_events(sub_dir)
        except Exception as e:  # noqa: BLE001
            print(f"[{i:2d}/{len(inv)}] {sid}: load failed {e!r}")
            continue
        valid = _valid_event_indices(evt["bools"], min_participating=3)
        ranks_v = evt["ranks"][:, valid]
        bools_v = evt["bools"][:, valid]
        masked = build_masked_kmeans_features(ranks_v, bools_v, impute="event_median")

        t0 = time.perf_counter()
        scan = [_kmeans_stability_for_k(masked, k) for k in K_RANGE]
        pick = _pick_stable_k(scan)
        elapsed = time.perf_counter() - t0

        original_stable_k = payload.get("stable_k")  # PR-2 stored
        masked_stable_k = pick["stable_k_masked"]
        # robust int comparison handling Nones
        if masked_stable_k is None or original_stable_k is None:
            stable_k_changed: Any = None
        else:
            stable_k_changed = bool(int(masked_stable_k) != int(original_stable_k))

        payload["masked_stable_k"] = {
            "k_range": K_RANGE,
            "scan": [
                {
                    "k": int(r["k"]),
                    "passes_both": bool(r.get("passes_both", False)),
                    "median_silhouette": float(r.get("median_silhouette", np.nan)),
                    "median_ami": float(r.get("median_ami", np.nan)),
                    "worst_min_cluster_fraction": float(r.get("worst_min_cluster_fraction", 0.0)),
                }
                for r in scan
            ],
            "stable_k_masked": masked_stable_k,
            "chosen_reason": pick["chosen_reason"],
            "passing_ks": pick.get("passing_ks", []),
            "elapsed_sec": float(elapsed),
        }
        payload["stable_k_changed"] = stable_k_changed
        with open(per_subj_path, "w") as f:
            json.dump(payload, f, indent=2,
                      default=lambda o: int(o) if isinstance(o, np.integer)
                       else float(o) if isinstance(o, np.floating) else str(o))

        rows_updated.append({
            "sid": sid,
            "stable_k_original": original_stable_k,
            "stable_k_masked": masked_stable_k,
            "stable_k_changed": stable_k_changed,
            "chosen_reason": pick["chosen_reason"],
            "passing_ks": "|".join(str(k) for k in pick.get("passing_ks", [])),
            "elapsed_sec": elapsed,
        })
        print(
            f"[{i:2d}/{len(inv)}] {sid}: stable_k_orig={original_stable_k} "
            f"-> stable_k_masked={masked_stable_k} "
            f"(reason={pick['chosen_reason']}, "
            f"passing={pick.get('passing_ks', [])}, {elapsed:.1f}s)"
        )

    # Merge into cohort summary
    cohort_csv = OUT_DIR / "cohort_summary.csv"
    if not cohort_csv.exists():
        print("cohort_summary.csv missing; aborting merge", file=sys.stderr)
        return 1
    with open(cohort_csv) as f:
        original_rows = list(csv.DictReader(f))
    updates_by_sid = {r["sid"]: r for r in rows_updated}
    for row in original_rows:
        upd = updates_by_sid.get(row["sid"])
        if upd:
            row["stable_k_masked"] = upd["stable_k_masked"]
            row["stable_k_changed"] = upd["stable_k_changed"]
            row["masked_scan_passing_ks"] = upd["passing_ks"]
        else:
            row["stable_k_masked"] = ""
            row["stable_k_changed"] = ""
            row["masked_scan_passing_ks"] = ""
    new_fields = list(original_rows[0].keys())
    for col in ("stable_k_masked", "stable_k_changed", "masked_scan_passing_ks"):
        if col not in new_fields:
            new_fields.append(col)
    with open(cohort_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=new_fields, extrasaction="ignore")
        w.writeheader()
        for r in original_rows:
            w.writerow(r)
    print(f"\nupdated cohort summary -> {cohort_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
