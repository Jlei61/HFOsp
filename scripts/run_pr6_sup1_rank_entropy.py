#!/usr/bin/env python3
"""PR-6-sup1 — first-rank entropy / symmetry-breaking diagnostic runner.

Plan-of-record (v3, 2026-05-10):
    docs/archive/topic1/pr6_template_anchoring/
        pr6_supplementary_rank_entropy_plan_2026-05-10.md

CLI modes
---------
--pilot
    Run on three sentinel subjects (chengshuai_yuquan = none /
    epilepsiae_1146 = strict / epilepsiae_1125 = candidate) and write
    pilot_3subjects.json. Hand back to user before cohort.

--cohort
    Run on the n=35 stable_k=2 cohort (PR-2 stable_k=2 ∩
    rank_displacement v14 universe).

--subjects <stem ...>
    Explicit list, e.g. epilepsiae_548 yuquan_chengshuai.
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
    _valid_event_indices,
    load_subject_propagation_events,
)
from src.rank_displacement import run_subject_rank_entropy  # noqa: E402


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
RD_COHORT_SUMMARY = (
    DATA_ROOT
    / "results"
    / "interictal_propagation"
    / "rank_displacement"
    / "cohort_summary.json"
)

DEFAULT_OUTPUT_DIR = (
    DATA_ROOT / "results" / "interictal_propagation" / "pr6_sup1_rank_entropy"
)

PILOT_SUBJECTS = [
    "yuquan_chengshuai",      # §8 swap_class = none
    "epilepsiae_1146",        # §8 strict (decision_k = 7)
    "epilepsiae_1125",        # §8 candidate
]


def _split_stem(stem: str) -> Tuple[str, str]:
    if stem.startswith("epilepsiae_"):
        return "epilepsiae", stem[len("epilepsiae_"):]
    if stem.startswith("yuquan_"):
        return "yuquan", stem[len("yuquan_"):]
    raise ValueError(f"unrecognized stem: {stem!r}")


def _subject_dir(dataset: str, subject: str) -> Path:
    if dataset == "yuquan":
        return YUQUAN_ROOT / subject
    legacy = EPILEPSIAE_ROOT / subject / "all_recs"
    if legacy.exists():
        return legacy
    return EPILEPSIAE_ROOT / subject


def _derive_valid_mask(
    pr6_path: Path,
    pr2_clusters: List[dict],
    n_channels: int,
) -> Tuple[np.ndarray, str]:
    """PR-6 priority union; PR-2 template_rank sentinel fallback (matches
    Step 6 + rank_displacement v14 provenance)."""
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


def _load_swap_class(stem: str) -> str:
    if not RD_COHORT_SUMMARY.exists():
        return "unknown"
    rd = json.loads(RD_COHORT_SUMMARY.read_text())
    target_dataset, target_subject = _split_stem(stem)
    for r in rd:
        if (
            r.get("dataset") == target_dataset
            and str(r.get("subject")) == str(target_subject)
        ):
            for p in r.get("pairs") or []:
                sw = p.get("swap_sweep") or {}
                cand = sw.get("swap_class")
                if cand == "strict":
                    return "strict"
                if cand == "candidate":
                    return "candidate"
            return "none"
    return "unknown"


def _process_one(stem: str, *, n_perm_N0: int = 1000) -> Optional[dict]:
    dataset, subject = _split_stem(stem)
    pr2_path = PR2_DIR / f"{stem}.json"
    pr6_path = PR6_DIR / f"{stem}.json"

    if not pr2_path.exists():
        return {
            "subject": subject,
            "dataset": dataset,
            "stem": stem,
            "exit_reason": f"no_pr2_json ({pr2_path.name})",
        }

    pr2 = json.loads(pr2_path.read_text())
    ac = pr2.get("adaptive_cluster", {})
    stable_k = ac.get("stable_k")
    clusters = ac.get("clusters", [])
    pr2_channels = pr2.get("channel_names", [])
    pr2_labels = ac.get("labels")

    if (stable_k is None or len(clusters) == 0 or not pr2_channels
            or pr2_labels is None):
        return {
            "subject": subject, "dataset": dataset, "stem": stem,
            "exit_reason": "incomplete_pr2_adaptive_cluster",
        }

    if int(stable_k) != 2:
        return {
            "subject": subject, "dataset": dataset, "stem": stem,
            "stable_k": int(stable_k),
            "exit_reason": "stable_k_not_2",
        }

    valid_mask_pr2, vm_provenance = _derive_valid_mask(
        pr6_path, clusters, len(pr2_channels)
    )
    if int(valid_mask_pr2.sum()) < 6:
        return {
            "subject": subject, "dataset": dataset, "stem": stem,
            "stable_k": int(stable_k),
            "n_valid_channels": int(valid_mask_pr2.sum()),
            "exit_reason": "n_valid_below_6",
        }

    subj_dir = _subject_dir(dataset, subject)
    if not subj_dir.exists():
        return {
            "subject": subject, "dataset": dataset, "stem": stem,
            "exit_reason": f"subject_dir_missing ({subj_dir})",
        }

    try:
        loaded = load_subject_propagation_events(subj_dir)
    except Exception as exc:  # noqa: BLE001
        return {
            "subject": subject, "dataset": dataset, "stem": stem,
            "exit_reason": f"load_subject_failed: {exc}",
        }

    ranks = loaded["ranks"]
    bools = loaded["bools"]
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
            "subject": subject, "dataset": dataset, "stem": stem,
            "stable_k": int(stable_k),
            "n_valid_channels": n_valid_lagpat,
            "valid_mask_provenance": vm_provenance,
            "exit_reason": "n_valid_below_6_after_alignment",
        }

    # Map PR-2 labels (length = n_valid_events under min_shared=3) back to
    # the global event axis. Events outside PR-2's valid set get label = -1
    # so run_subject_rank_entropy ignores them (cluster index >= 0 only).
    valid_event_indices = _valid_event_indices(bools, min_participating=3)
    global_labels = np.full(ranks.shape[1], -1, dtype=int)
    if len(pr2_labels) != int(valid_event_indices.size):
        return {
            "subject": subject, "dataset": dataset, "stem": stem,
            "exit_reason": (
                f"pr2_labels_len_mismatch ({len(pr2_labels)} vs valid_idx="
                f"{int(valid_event_indices.size)})"
            ),
        }
    global_labels[valid_event_indices] = np.asarray(pr2_labels, dtype=int)

    subject_data = {
        "ranks": ranks,
        "bools": bools,
        "labels": global_labels,
        "valid_mask": valid_mask,
        "channel_names": channel_names,
        "n_clusters": 2,
    }

    try:
        result = run_subject_rank_entropy(
            subject_data, n_perm_N0=n_perm_N0, base_seed_N0=0
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "subject": subject, "dataset": dataset, "stem": stem,
            "stable_k": int(stable_k),
            "n_valid_channels": n_valid_lagpat,
            "valid_mask_provenance": vm_provenance,
            "exit_reason": f"sup1_pipeline_failed: {exc}",
        }

    swap_class = _load_swap_class(stem)
    # ``run_subject_rank_entropy`` returns dict with key "subject" being the
    # subject-level payload; rename it before injecting metadata so the
    # bare-string subject_id doesn't clobber the analysis result.
    result["subject_level"] = result.pop("subject")
    result.update(
        {
            "subject_id": subject,
            "dataset": dataset,
            "stem": stem,
            "stable_k": int(stable_k),
            "valid_mask_provenance": vm_provenance,
            "n_valid_channels": n_valid_lagpat,
            "swap_class_full": swap_class,
        }
    )
    return result


def _slim_summary(record: dict) -> dict:
    if record.get("exit_reason") and not record.get("clusters"):
        return {
            "subject_id": record.get("subject") or record.get("subject_id"),
            "dataset": record.get("dataset"),
            "stem": record.get("stem"),
            "exit_reason": record["exit_reason"],
        }

    subject_payload = record.get("subject_level") or {}
    cluster_payload = record.get("clusters", {})
    cluster_summaries = {
        cid: {
            "n_events_total_k": c.get("n_events_total_k"),
            "n_events_kept_k": c.get("n_events_kept_k"),
            "drop_rate_k": c.get("drop_rate_k"),
            "eligibility_flag": c.get("eligibility_flag"),
            "delta": c.get("delta"),
            "asymmetry": c.get("asymmetry"),
            "p_N0": c.get("p_N0"),
            "endpoint_pair_percentile": (
                c.get("N1_cluster", {}).get("endpoint_pair_percentile")
                if c.get("N1_cluster")
                else None
            ),
            "is_endpoint_pair_max": (
                c.get("N1_cluster", {}).get("is_endpoint_pair_max")
                if c.get("N1_cluster")
                else None
            ),
            "p_N1": (
                c.get("N1_cluster", {}).get("p_N1")
                if c.get("N1_cluster")
                else None
            ),
            "min_attainable_p_N1": (
                c.get("N1_cluster", {}).get("min_attainable_p_N1")
                if c.get("N1_cluster")
                else None
            ),
        }
        for cid, c in cluster_payload.items()
    }

    return {
        "subject_id": record.get("subject_id"),
        "dataset": record.get("dataset"),
        "stem": record.get("stem"),
        "stable_k": record.get("stable_k"),
        "valid_mask_provenance": record.get("valid_mask_provenance"),
        "n_valid_channels": record.get("n_valid_channels"),
        "swap_class_full": record.get("swap_class_full"),
        "clusters": cluster_summaries,
        "subject_level": {
            "subject_eligibility_flag": subject_payload.get("subject_eligibility_flag"),
            "delta_obs_subject": subject_payload.get("delta_obs_subject"),
            "subject_combo_percentile": subject_payload.get("subject_combo_percentile"),
            "is_subject_combo_max": subject_payload.get("is_subject_combo_max"),
            "p_N1_subject": subject_payload.get("p_N1_subject"),
            "min_attainable_p_N1_subject": subject_payload.get("min_attainable_p_N1_subject"),
            "n_combos": subject_payload.get("n_combos"),
        },
    }


def _cohort_drop_rate_summary(slim: List[dict]) -> dict:
    drops = []
    n_warn = 0
    n_excluded = 0
    for r in slim:
        for c in (r.get("clusters") or {}).values():
            dr = c.get("drop_rate_k")
            if dr is not None and isinstance(dr, (int, float)) and np.isfinite(dr):
                drops.append(float(dr))
            if c.get("eligibility_flag") == "high_drop_rate_warning":
                n_warn += 1
            if c.get("eligibility_flag") == "excluded_low_kept_events":
                n_excluded += 1
    drops_arr = np.asarray(drops, dtype=float)
    return {
        "n_clusters_total": int(drops_arr.size),
        "drop_rate_median": float(np.median(drops_arr)) if drops_arr.size else None,
        "drop_rate_max": float(np.max(drops_arr)) if drops_arr.size else None,
        "n_clusters_high_drop_rate_warning": int(n_warn),
        "n_clusters_excluded_low_kept_events": int(n_excluded),
    }


def _cohort_stratified_summary(slim: List[dict]) -> dict:
    by_class: Dict[str, List[dict]] = {"strict": [], "candidate": [], "none": [], "unknown": []}
    for r in slim:
        cls = r.get("swap_class_full") or "unknown"
        by_class.setdefault(cls, []).append(r)

    out = {}
    for cls, group in by_class.items():
        deltas = []
        percentiles = []
        n_max = 0
        for r in group:
            sl = r.get("subject_level") or {}
            if sl.get("subject_eligibility_flag") != "ok":
                continue
            d = sl.get("delta_obs_subject")
            p = sl.get("subject_combo_percentile")
            if d is not None and isinstance(d, (int, float)) and np.isfinite(d):
                deltas.append(float(d))
            if p is not None and isinstance(p, (int, float)) and np.isfinite(p):
                percentiles.append(float(p))
            if sl.get("is_subject_combo_max") is True:
                n_max += 1
        deltas_arr = np.asarray(deltas, dtype=float)
        pct_arr = np.asarray(percentiles, dtype=float)
        out[cls] = {
            "n_subjects": len(group),
            "n_eligible": int(deltas_arr.size),
            "delta_obs_subject": {
                "median": float(np.median(deltas_arr)) if deltas_arr.size else None,
                "iqr": (
                    [float(np.percentile(deltas_arr, 25)),
                     float(np.percentile(deltas_arr, 75))]
                    if deltas_arr.size >= 4 else None
                ),
            },
            "subject_combo_percentile": {
                "median": float(np.median(pct_arr)) if pct_arr.size else None,
                "iqr": (
                    [float(np.percentile(pct_arr, 25)),
                     float(np.percentile(pct_arr, 75))]
                    if pct_arr.size >= 4 else None
                ),
            },
            "n_subject_combo_max": int(n_max),
        }
    return out


def _resolve_cohort(args: argparse.Namespace) -> List[str]:
    if args.subjects:
        return list(args.subjects)
    if args.pilot:
        return list(PILOT_SUBJECTS)
    if args.cohort:
        if not RD_COHORT_SUMMARY.exists():
            raise SystemExit("rank_displacement cohort_summary.json missing")
        rd = json.loads(RD_COHORT_SUMMARY.read_text())
        stems = [
            f"{r['dataset']}_{r['subject']}"
            for r in rd if r.get("stable_k") == 2
        ]
        return stems
    raise SystemExit("must pass --pilot, --cohort, or --subjects")


def _make_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--pilot", action="store_true")
    grp.add_argument("--cohort", action="store_true")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--n-perm-N0", type=int, default=1000)
    args = ap.parse_args()

    stems = _resolve_cohort(args)

    out_dir: Path = args.output_dir
    per_subject_dir = out_dir / "per_subject"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_subject_dir.mkdir(parents=True, exist_ok=True)

    slim_records: List[dict] = []
    for stem in stems:
        record = _process_one(stem, n_perm_N0=args.n_perm_N0)
        if record is None:
            print(f"[{stem}] skipped (None)")
            continue

        out_path = per_subject_dir / f"{stem}.json"
        out_path.write_text(
            json.dumps(_make_serializable(record), indent=2, ensure_ascii=False)
        )
        slim = _slim_summary(record)
        slim_records.append(slim)

        if "exit_reason" in slim and not slim.get("clusters"):
            print(f"[{stem}] exit: {slim['exit_reason']}")
        else:
            sl = slim.get("subject_level") or {}
            d = sl.get("delta_obs_subject")
            pct = sl.get("subject_combo_percentile")
            is_max = sl.get("is_subject_combo_max")
            flag = sl.get("subject_eligibility_flag")
            print(
                f"[{stem}] swap_class={slim.get('swap_class_full')} "
                f"flag={flag} Δ={d:.3f} pct={pct:.3f} is_max={is_max}"
                if d is not None and pct is not None
                else f"[{stem}] swap_class={slim.get('swap_class_full')} "
                     f"flag={flag} (Δ/pct unavailable)"
            )

    payload = {
        "subjects": slim_records,
        "cohort_drop_rate_summary": _cohort_drop_rate_summary(slim_records),
        "stratified_by_swap_class": _cohort_stratified_summary(slim_records),
    }

    if args.pilot:
        out_path = out_dir / "pilot_3subjects.json"
    else:
        out_path = out_dir / "cohort_summary.json"

    out_path.write_text(
        json.dumps(_make_serializable(payload), indent=2, ensure_ascii=False)
    )
    print(f"\nWrote {out_path} with n={len(slim_records)} subjects")


if __name__ == "__main__":
    main()
