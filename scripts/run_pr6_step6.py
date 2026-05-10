#!/usr/bin/env python3
"""PR-6 Step 6 — held-out time template stability runner.

Plan-of-record:
    docs/archive/topic1/pr6_template_anchoring/
        pr6_step6_held_out_template_plan_2026-05-10.md

CLI modes
---------
--pilot
    Run on three sentinel subjects (chengshuai_yuquan / epilepsiae_548 /
    epilepsiae_1146) and write ``pilot_3subjects.json``. Halt-condition:
    if endpoint_position_recall SE > 0.20 across the three, plan §7.3
    requires user decision before cohort run.

--cohort
    Run on the PR-6 main cohort (h1_pooled.subject_ids from PR-6
    cohort_summary.json) plus the n=35 sensitivity cohort (stable_k=2 from
    rank_displacement cohort_summary.json). Writes per_subject/<stem>.json
    and cohort_summary.json.

--subjects <stem ...>
    Explicit list (e.g., ``epilepsiae_548 yuquan_chengshuai``).

--output-dir
    Override default ``results/interictal_propagation/pr6_step6_held_out_template/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

WORKTREE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKTREE_ROOT))

from src.interictal_propagation import (  # noqa: E402
    compute_held_out_endpoint_validation,
    load_subject_propagation_events,
)


YUQUAN_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_ROOT = Path(
    "/mnt/epilepsia_data/interilca_inter_results/all_data_lns"
)

DATA_ROOT = WORKTREE_ROOT
PR2_DIR = DATA_ROOT / "results" / "interictal_propagation" / "per_subject"
PR6_DIR = (
    DATA_ROOT
    / "results"
    / "interictal_propagation"
    / "template_anchoring"
    / "per_subject"
)
PR6_COHORT_SUMMARY = (
    DATA_ROOT
    / "results"
    / "interictal_propagation"
    / "template_anchoring"
    / "cohort_summary.json"
)
RD_COHORT_SUMMARY = (
    DATA_ROOT
    / "results"
    / "interictal_propagation"
    / "rank_displacement"
    / "cohort_summary.json"
)
SOZ_FILES = {
    "epilepsiae": DATA_ROOT / "results" / "epilepsiae_soz_core_channels.json",
    "yuquan": DATA_ROOT / "results" / "yuquan_soz_core_channels.json",
}

DEFAULT_OUTPUT_DIR = (
    DATA_ROOT
    / "results"
    / "interictal_propagation"
    / "pr6_step6_held_out_template"
)

PILOT_SUBJECTS = [
    "yuquan_chengshuai",
    "epilepsiae_548",
    "epilepsiae_1146",
]


YUQUAN_LEGACY_VARIANT_SUBJECTS = frozenset(
    {
        "zhangkexuan",
        "pengzihang",
        "songzishuo",
        "zhangbichen",
        "zhaochenxi",
        "zhaojinrui",
        "zhourongxuan",
    }
)


def _subject_dir(dataset: str, subject: str) -> Path:
    if dataset == "yuquan":
        return YUQUAN_ROOT / subject
    legacy = EPILEPSIAE_ROOT / subject / "all_recs"
    if legacy.exists():
        return legacy
    return EPILEPSIAE_ROOT / subject


def _split_stem(stem: str) -> Tuple[str, str]:
    if stem.startswith("epilepsiae_"):
        return "epilepsiae", stem[len("epilepsiae_") :]
    if stem.startswith("yuquan_"):
        return "yuquan", stem[len("yuquan_") :]
    raise ValueError(f"unrecognized stem: {stem!r}")


def _load_soz_lookup() -> Dict[str, Dict[str, Set[str]]]:
    out: Dict[str, Dict[str, Set[str]]] = {}
    for ds, path in SOZ_FILES.items():
        if not path.exists():
            out[ds] = {}
            continue
        raw = json.loads(path.read_text())
        out[ds] = {sub: set(chs or []) for sub, chs in raw.items()}
    return out


def _derive_valid_mask(
    pr6_path: Path,
    pr2_clusters: List[dict],
    n_channels: int,
) -> Tuple[np.ndarray, str]:
    """PR-6 priority union; PR-2 template_rank sentinel fallback."""
    if pr6_path.exists():
        pr6 = json.loads(pr6_path.read_text())
        masks: List[np.ndarray] = []
        for entry in pr6.get("per_template", []):
            vm = entry.get("valid_mask")
            if vm is None or len(vm) != n_channels:
                continue
            masks.append(np.asarray(vm, dtype=bool))
        if masks:
            return np.any(np.stack(masks, axis=0), axis=0), "pr6_union"

    masks2: List[np.ndarray] = []
    for c in pr2_clusters:
        tr = c.get("template_rank")
        if tr is None or len(tr) != n_channels:
            continue
        masks2.append(np.asarray(np.asarray(tr) != -1, dtype=bool))
    if masks2:
        return np.any(np.stack(masks2, axis=0), axis=0), "pr2_sentinel_union"

    return np.zeros(n_channels, dtype=bool), "none"


def _process_one(stem: str, soz_lookup: Dict[str, Dict[str, Set[str]]]) -> Optional[dict]:
    dataset, subject = _split_stem(stem)
    pr2_path = PR2_DIR / f"{stem}.json"
    pr6_path = PR6_DIR / f"{stem}.json"

    if not pr2_path.exists():
        return {
            "subject": subject,
            "dataset": dataset,
            "exit_reason": f"no_pr2_json ({pr2_path.name})",
        }

    pr2 = json.loads(pr2_path.read_text())
    ac = pr2.get("adaptive_cluster", {})
    stable_k = ac.get("stable_k")
    clusters = ac.get("clusters", [])
    pr2_channels = pr2.get("channel_names", [])
    if stable_k is None or len(clusters) == 0 or not pr2_channels:
        return {
            "subject": subject,
            "dataset": dataset,
            "exit_reason": "incomplete_pr2_adaptive_cluster",
        }

    if int(stable_k) != 2:
        return {
            "subject": subject,
            "dataset": dataset,
            "stable_k": int(stable_k),
            "exit_reason": "stable_k_not_2",
        }

    valid_mask_pr2, vm_provenance = _derive_valid_mask(
        pr6_path, clusters, len(pr2_channels)
    )
    if int(valid_mask_pr2.sum()) < 6:
        return {
            "subject": subject,
            "dataset": dataset,
            "stable_k": int(stable_k),
            "n_valid_channels": int(valid_mask_pr2.sum()),
            "exit_reason": "n_valid_below_6",
        }

    subj_dir = _subject_dir(dataset, subject)
    if not subj_dir.exists():
        return {
            "subject": subject,
            "dataset": dataset,
            "exit_reason": f"subject_dir_missing ({subj_dir})",
        }

    try:
        loaded = load_subject_propagation_events(subj_dir)
    except Exception as exc:  # noqa: BLE001
        return {
            "subject": subject,
            "dataset": dataset,
            "exit_reason": f"load_subject_failed: {exc}",
        }

    ranks = loaded["ranks"]
    bools = loaded["bools"]
    event_abs_times = loaded["event_abs_times"]
    block_ids = loaded["block_ids"]
    block_time_ranges = loaded["block_time_ranges"]
    channel_names = loaded["channel_names"]

    # Align valid_mask from PR-2 channel ordering to lagPat channel ordering.
    pr2_idx = {ch: i for i, ch in enumerate(pr2_channels)}
    valid_mask = np.zeros(len(channel_names), dtype=bool)
    for i, ch in enumerate(channel_names):
        if ch in pr2_idx and valid_mask_pr2[pr2_idx[ch]]:
            valid_mask[i] = True

    n_valid_lagpat = int(valid_mask.sum())
    if n_valid_lagpat < 6:
        return {
            "subject": subject,
            "dataset": dataset,
            "stable_k": int(stable_k),
            "n_valid_channels": n_valid_lagpat,
            "valid_mask_provenance": vm_provenance,
            "exit_reason": "n_valid_below_6_after_alignment",
        }

    valid_event_indices = np.arange(ranks.shape[1], dtype=int)
    soz_channels = soz_lookup.get(dataset, {}).get(str(subject), set())

    try:
        result = compute_held_out_endpoint_validation(
            ranks=ranks,
            bools=bools,
            event_abs_times=event_abs_times,
            block_ids=block_ids,
            block_time_ranges=block_time_ranges,
            chosen_k=2,
            valid_event_indices=valid_event_indices,
            channel_names=channel_names,
            soz_channels=soz_channels,
            valid_mask=valid_mask,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "subject": subject,
            "dataset": dataset,
            "stable_k": int(stable_k),
            "n_valid_channels": n_valid_lagpat,
            "valid_mask_provenance": vm_provenance,
            "exit_reason": f"validation_failed: {exc}",
        }

    result.update(
        {
            "subject": subject,
            "dataset": dataset,
            "stem": stem,
            "stable_k": int(stable_k),
            "valid_mask_provenance": vm_provenance,
            "n_valid_channels_pr2": int(valid_mask_pr2.sum()),
            "n_valid_channels_lagpat_aligned": n_valid_lagpat,
            "soz_channels": sorted(soz_channels),
        }
    )
    return result


def _slim_summary(record: dict) -> dict:
    if record.get("exit_reason") and not record.get("validation"):
        return {
            "subject": record.get("subject"),
            "dataset": record.get("dataset"),
            "stem": record.get("stem"),
            "exit_reason": record["exit_reason"],
        }
    val = record.get("validation", {})
    return {
        "subject": record.get("subject"),
        "dataset": record.get("dataset"),
        "stem": record.get("stem"),
        "stable_k": record.get("stable_k"),
        "valid_mask_provenance": record.get("valid_mask_provenance"),
        "n_valid_channels": record.get("n_valid_channels_lagpat_aligned"),
        "n_first": record.get("first_half", {}).get("n_events"),
        "n_second": record.get("second_half", {}).get("n_events"),
        "swap_class_first": record.get("first_half", {}).get("swap_class"),
        "swap_class_second_projected": record.get("second_half", {}).get(
            "swap_class_projected"
        ),
        "template_spearman": val.get("template_spearman"),
        "endpoint_position_recall": val.get("endpoint_position_recall"),
        "cluster_assignment_purity": val.get("cluster_assignment_purity"),
        "swap_class_concordant": val.get("swap_class_concordant"),
        "tier": val.get("tier"),
    }


def _cohort_stats(slim_records: List[dict]) -> dict:
    valid = [
        r for r in slim_records
        if r.get("tier") in {"strong", "moderate", "weak", "fail"}
    ]
    n = len(valid)
    if n == 0:
        return {"n": 0, "tier_counts": {}}

    tier_counts: Dict[str, int] = {}
    for r in valid:
        tier_counts[r["tier"]] = tier_counts.get(r["tier"], 0) + 1

    def _vals(field: str) -> np.ndarray:
        out = []
        for r in valid:
            v = r.get(field)
            if v is not None and isinstance(v, (int, float)) and np.isfinite(v):
                out.append(float(v))
        return np.asarray(out, dtype=float)

    spear = _vals("template_spearman")
    recall = _vals("endpoint_position_recall")
    purity = _vals("cluster_assignment_purity")
    n_concord = sum(1 for r in valid if r.get("swap_class_concordant") is True)

    return {
        "n": n,
        "tier_counts": tier_counts,
        "template_spearman": {
            "median": float(np.median(spear)) if spear.size else None,
            "iqr": [float(np.percentile(spear, 25)), float(np.percentile(spear, 75))]
            if spear.size
            else None,
            "n": int(spear.size),
        },
        "endpoint_position_recall": {
            "median": float(np.median(recall)) if recall.size else None,
            "iqr": [float(np.percentile(recall, 25)), float(np.percentile(recall, 75))]
            if recall.size
            else None,
            "se": float(np.std(recall, ddof=1) / np.sqrt(recall.size))
            if recall.size > 1
            else None,
            "n": int(recall.size),
        },
        "cluster_assignment_purity": {
            "median": float(np.median(purity)) if purity.size else None,
            "iqr": [float(np.percentile(purity, 25)), float(np.percentile(purity, 75))]
            if purity.size
            else None,
            "n": int(purity.size),
        },
        "swap_class_concordant_count": n_concord,
        "swap_class_concordant_fraction": (n_concord / n) if n > 0 else None,
    }


def _resolve_cohort(args: argparse.Namespace) -> List[str]:
    if args.subjects:
        return list(args.subjects)
    if args.pilot:
        return list(PILOT_SUBJECTS)
    if args.cohort:
        stems: List[str] = []
        if PR6_COHORT_SUMMARY.exists():
            d = json.loads(PR6_COHORT_SUMMARY.read_text())
            for sub_id in d.get("h1_pooled", {}).get("subject_ids", []):
                ds = "epilepsiae" if str(sub_id).isdigit() else "yuquan"
                stems.append(f"{ds}_{sub_id}")
        if RD_COHORT_SUMMARY.exists():
            rd = json.loads(RD_COHORT_SUMMARY.read_text())
            for r in rd:
                if r.get("stable_k") == 2:
                    stems.append(f"{r['dataset']}_{r['subject']}")
        # Deduplicate while preserving first-seen order
        seen: Set[str] = set()
        unique: List[str] = []
        for s in stems:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        return unique
    raise SystemExit("must pass --pilot, --cohort, or --subjects")


def main() -> None:
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--pilot", action="store_true", help="Run 3 sentinel subjects")
    grp.add_argument(
        "--cohort", action="store_true",
        help="Run PR-6 main cohort + n=35 sensitivity",
    )
    ap.add_argument(
        "--subjects", nargs="*", default=None,
        help="Explicit stems, e.g. yuquan_chengshuai epilepsiae_548",
    )
    ap.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Override output (default {DEFAULT_OUTPUT_DIR})",
    )
    args = ap.parse_args()

    stems = _resolve_cohort(args)
    soz_lookup = _load_soz_lookup()

    out_dir: Path = args.output_dir
    per_subject_dir = out_dir / "per_subject"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_subject_dir.mkdir(parents=True, exist_ok=True)

    slim_records: List[dict] = []
    for stem in stems:
        record = _process_one(stem, soz_lookup)
        if record is None:
            print(f"[{stem}] skipped (None)")
            continue

        out_path = per_subject_dir / f"{stem}.json"
        out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
        slim = _slim_summary(record)
        slim_records.append(slim)

        if "exit_reason" in slim:
            print(f"[{stem}] exit: {slim['exit_reason']}")
        else:
            print(
                f"[{stem}] tier={slim['tier']} "
                f"spearman={slim['template_spearman']:.3f} "
                f"recall={slim['endpoint_position_recall']:.3f} "
                f"purity={slim['cluster_assignment_purity']:.3f} "
                f"swap_concord={slim['swap_class_concordant']}"
            )

    if args.pilot:
        pilot_path = out_dir / "pilot_3subjects.json"
        pilot_payload = {
            "subjects": slim_records,
            "stats": _cohort_stats(slim_records),
        }
        pilot_path.write_text(json.dumps(pilot_payload, indent=2, ensure_ascii=False))
        se = pilot_payload["stats"].get("endpoint_position_recall", {}).get("se")
        print(f"\nWrote {pilot_path}")
        print(f"Pilot endpoint_position_recall SE = {se}")
        if se is not None and se > 0.20:
            print("WARNING: SE > 0.20 — plan §7.3 requires user decision before cohort.")
    elif args.cohort or args.subjects:
        summary_path = out_dir / "cohort_summary.json"
        summary_payload = {
            "subjects": slim_records,
            "stats": _cohort_stats(slim_records),
        }
        summary_path.write_text(
            json.dumps(summary_payload, indent=2, ensure_ascii=False)
        )
        print(f"\nWrote {summary_path} with n={len(slim_records)} subjects")


if __name__ == "__main__":
    main()
