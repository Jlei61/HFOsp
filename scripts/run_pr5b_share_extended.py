"""Extend PR-5-B template recruitment shift to full stable_k=2 cohort.

Loads existing PR-1 per-subject JSONs (n40 cohort), runs
``compute_template_recruitment_shift`` for the main + auxiliary configs on
the subjects missing from the original PR-5-B retained set, and emits a
combined cohort table (CSV + JSON) covering all stable_k=2 subjects.

Output: results/interictal_propagation/pr5b_recruitment_shift_extended.json
        results/interictal_propagation/pr5b_recruitment_shift_extended.csv

This script BYPASSES the PR-5-A gate: it produces descriptive figure data,
not a cohort claim. The original PR-5-B retained-cohort statistics in
``pr5b_recruitment_shift.json`` remain the source of truth for §4.5.
"""
from __future__ import annotations
import json
import csv
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.interictal_propagation import (  # noqa: E402
    SEIZURE_PROXIMITY_CONFIGS,
    _build_seizure_proximity_windows,
    _intersect_seconds,
    _valid_event_indices,
    compute_template_recruitment_shift,
    load_subject_propagation_events,
)
from src.event_periodicity import load_seizure_times  # noqa: E402

# Reuse path helpers from main runner
from scripts.run_interictal_propagation import (  # noqa: E402
    YUQUAN_ROOT,
    EPILEPSIAE_ROOT,
    _subject_dir,
    _has_propagation_inputs,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _build_subject_share_record(
    dataset: str,
    subject: str,
    base: Dict[str, Any],
    *,
    config_name: str,
) -> Dict[str, Any] | None:
    """Run compute_template_recruitment_shift for one subject + config."""
    ac = base.get("adaptive_cluster", {})
    if "error" in ac or "labels" not in ac:
        return {"skip_reason": "adaptive_cluster_missing"}
    stable_k = int(ac.get("stable_k") or 0)
    if stable_k != 2:
        return {"skip_reason": f"stable_k={stable_k}"}

    root = YUQUAN_ROOT if dataset == "yuquan" else EPILEPSIAE_ROOT
    subject_dir = _subject_dir(dataset, root, subject)
    if not _has_propagation_inputs(dataset, subject_dir):
        return {"skip_reason": "lagPat_missing"}

    loaded = load_subject_propagation_events(subject_dir)
    valid_events = _valid_event_indices(loaded["bools"], min_participating=3)
    labels = np.asarray(ac["labels"], dtype=int)
    chosen_k = int(ac["chosen_k"])
    if valid_events.size != labels.size:
        return {"skip_reason": "label_valid_mismatch"}

    v_times = loaded["event_abs_times"][valid_events]
    v_bools = loaded["bools"][:, valid_events]
    n_part = np.sum(v_bools > 0, axis=0).astype(int)

    counts_full = np.bincount(labels, minlength=chosen_k).astype(int)
    if int(counts_full.sum()) == 0:
        return {"skip_reason": "zero_events"}
    dominant_global_id = int(np.argmax(counts_full))

    seizure_times = load_seizure_times(subject, dataset)
    if not seizure_times:
        return {"skip_reason": "no_seizures"}

    block_ranges = loaded.get("block_time_ranges")
    coverage_ranges = (
        [(float(lo), float(hi)) for lo, hi in block_ranges]
        if block_ranges is not None else None
    )

    cfg = SEIZURE_PROXIMITY_CONFIGS[config_name]
    proximity = _build_seizure_proximity_windows(
        v_times,
        seizure_times,
        baseline_hours=cfg["baseline_hours"],
        pre_ictal_hours=cfg["pre_ictal_hours"],
        post_ictal_hours=cfg["post_ictal_hours"],
    )
    state_ranges = {
        "baseline": cfg["baseline_hours"],
        "pre": cfg["pre_ictal_hours"],
        "post": cfg["post_ictal_hours"],
    }
    windows_with_hours: List[Dict[str, Any]] = []
    for w in proximity["usable_windows"]:
        sz_t = float(w["seizure_time"])
        covered = {}
        for state_name, (lo, hi) in state_ranges.items():
            t0 = sz_t + float(lo) * 3600.0
            t1 = sz_t + float(hi) * 3600.0
            if coverage_ranges is None:
                covered[state_name] = float(hi - lo)
            else:
                covered[state_name] = (
                    _intersect_seconds(t0, t1, coverage_ranges) / 3600.0
                )
        enriched = dict(w)
        enriched["state_covered_hours"] = covered
        windows_with_hours.append(enriched)

    shift = compute_template_recruitment_shift(
        cluster_labels=labels,
        n_part_per_event=n_part,
        n_clusters=chosen_k,
        dominant_global_id=dominant_global_id,
        proximity_windows=windows_with_hours,
        min_participating_l3=5,
    )

    return {
        "subject_id": subject,
        "dataset": dataset,
        "stable_k": stable_k,
        "config_name": config_name,
        "dom_global_id": int(shift["dom_global_id"]),
        "n_seizures_usable": int(len(windows_with_hours)),
        "n_windows_used": int(shift["n_windows_used"]),
        "dom_agreement": float(shift["dom_agreement"]) if np.isfinite(shift["dom_agreement"]) else float("nan"),
        "weighted_per_state": shift["weighted_per_state"],
        "deltas": shift["deltas"],
        "share_state_n_eligible_windows": shift["share_state_n_eligible_windows"],
        "share_pair_eligible": shift["share_pair_eligible"],
    }


def main() -> None:
    results_dir = ROOT / "results" / "interictal_propagation"
    pr1_path = results_dir / "pr1_subject_summary_n40.json"
    with open(pr1_path) as f:
        pr1 = json.load(f)

    target_subjects: List[tuple[str, str]] = []
    for key, val in pr1.items():
        sk = val.get("adaptive_cluster", {}).get("stable_k")
        if sk == 2:
            ds, sub = key.split("/", 1)
            target_subjects.append((ds, sub))
    target_subjects.sort()
    logger.info("Found %d stable_k=2 subjects in n40 cohort", len(target_subjects))

    records_by_config: Dict[str, List[Dict[str, Any]]] = {"main": [], "auxiliary": []}
    skipped: List[Dict[str, Any]] = []

    for ds, sub in target_subjects:
        base = pr1[f"{ds}/{sub}"]
        for cfg_name in ("main", "auxiliary"):
            rec = _build_subject_share_record(ds, sub, base, config_name=cfg_name)
            if rec is None:
                continue
            if "skip_reason" in rec:
                skipped.append({"subject": f"{ds}/{sub}", "config": cfg_name, **rec})
                logger.warning("SKIP %s/%s [%s]: %s", ds, sub, cfg_name, rec["skip_reason"])
                continue
            records_by_config[cfg_name].append(rec)
            logger.info(
                "OK %s/%s [%s]: n_sz=%d windows=%d dom_agree=%.2f"
                " base=%.3f pre=%.3f post=%.3f",
                ds, sub, cfg_name,
                rec["n_seizures_usable"], rec["n_windows_used"], rec["dom_agreement"],
                rec["weighted_per_state"]["dom_global_share"]["baseline"],
                rec["weighted_per_state"]["dom_global_share"]["pre"],
                rec["weighted_per_state"]["dom_global_share"]["post"],
            )

    out_json = results_dir / "pr5b_recruitment_shift_extended.json"
    payload = {
        "cohort_source": "stable_k=2 in pr1_subject_summary_n40.json",
        "n_targeted": len(target_subjects),
        "n_main_computed": len(records_by_config["main"]),
        "n_auxiliary_computed": len(records_by_config["auxiliary"]),
        "skipped": skipped,
        "per_subject": records_by_config,
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote %s", out_json)

    out_csv = results_dir / "pr5b_recruitment_shift_extended.csv"
    fields = [
        "config_name", "dataset", "subject_id", "stable_k",
        "dom_global_id", "n_seizures_usable", "n_windows_used", "dom_agreement",
        "share_baseline", "share_pre", "share_post",
        "delta_share_pre_minus_baseline", "delta_share_post_minus_baseline",
        "delta_share_post_minus_pre",
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for cfg_name in ("main", "auxiliary"):
            for r in records_by_config[cfg_name]:
                sh = r["weighted_per_state"]["dom_global_share"]
                dh = r["deltas"]["dom_global_share"]
                w.writerow({
                    "config_name": r["config_name"],
                    "dataset": r["dataset"],
                    "subject_id": r["subject_id"],
                    "stable_k": r["stable_k"],
                    "dom_global_id": r["dom_global_id"],
                    "n_seizures_usable": r["n_seizures_usable"],
                    "n_windows_used": r["n_windows_used"],
                    "dom_agreement": f"{r['dom_agreement']:.4f}",
                    "share_baseline": f"{sh['baseline']:.4f}",
                    "share_pre": f"{sh['pre']:.4f}",
                    "share_post": f"{sh['post']:.4f}",
                    "delta_share_pre_minus_baseline": f"{dh['pre_minus_baseline']:.4f}",
                    "delta_share_post_minus_baseline": f"{dh['post_minus_baseline']:.4f}",
                    "delta_share_post_minus_pre": f"{dh['post_minus_pre']:.4f}",
                })
    logger.info("Wrote %s", out_csv)


if __name__ == "__main__":
    main()
