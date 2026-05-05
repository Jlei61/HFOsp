#!/usr/bin/env python3
"""Topic 1 cluster geometry visualization — batch driver.

Loads per-subject ``pr1_subject_summary.json`` adaptive_cluster output, the
matching ``*_lagPat.npz`` arrays, and runs ``compute_subject_geometry`` to
produce per-subject geometry JSONs + a cohort summary.

Design doc: docs/archive/topic1/propagation/cluster_geometry_viz_plan_2026-05-06.md
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cluster_geometry import (  # noqa: E402
    DEFAULT_MIN_SHARED_CHANNELS,
    DEFAULT_N_MAX_FOR_MDS,
    DEFAULT_SUBSAMPLE_SEED,
    compute_subject_geometry,
    summarize_cohort_geometry,
)
from src.interictal_propagation import (  # noqa: E402
    _valid_event_indices,
    load_subject_propagation_events,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("cluster_geometry")


YUQUAN_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")
RESULTS_DIR = Path("results/interictal_propagation")
GEOMETRY_DIR = RESULTS_DIR / "cluster_geometry"
PER_SUBJECT_DIR = GEOMETRY_DIR / "per_subject"
EXISTING_PER_SUBJECT_DIR = RESULTS_DIR / "per_subject"

YUQUAN_SUBJECTS = [
    "zhangkexuan", "pengzihang", "chengshuai", "huangwanling",
    "liyouran", "songzishuo", "zhangbichen", "zhaochenxi",
    "zhaojinrui", "zhourongxuan", "zhangjiaqi",
    "chenziyang", "hanyuxuan", "huanghanwen", "litengsheng",
    "xuxinyi", "zhangjinhan", "sunyuanxin",
]

EPILEPSIAE_SUBJECTS = [
    "1096", "1084", "958", "922", "590", "1150", "442", "1073",
    "253", "1146", "916", "620", "583", "548", "384", "139",
    "1125", "1077", "818", "635",
]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return v
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def _epilepsiae_subject_dir(root: Path, subject: str) -> Path:
    legacy = root / subject / "all_recs"
    if legacy.exists():
        return legacy
    return root / subject


def _subject_dir(dataset: str, subject: str) -> Path:
    if dataset == "yuquan":
        return YUQUAN_ROOT / subject
    return _epilepsiae_subject_dir(EPILEPSIAE_ROOT, subject)


def _per_subject_json_path(dataset: str, subject: str) -> Path:
    return EXISTING_PER_SUBJECT_DIR / f"{dataset}_{subject}.json"


def _save(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, cls=NumpyEncoder, indent=2, allow_nan=True)
    logger.info("Saved %s", path)


def _resolve_dataset_subjects(
    datasets: List[str],
    subject_filter: Optional[List[str]],
) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for ds in datasets:
        names = YUQUAN_SUBJECTS if ds == "yuquan" else EPILEPSIAE_SUBJECTS
        for sub in names:
            if subject_filter and sub not in subject_filter:
                continue
            out.append((ds, sub))
    return out


def _load_adaptive_cluster(dataset: str, subject: str) -> Optional[Dict[str, Any]]:
    """Load adaptive_cluster from per-subject JSON; returns None if missing."""
    path = _per_subject_json_path(dataset, subject)
    if not path.exists():
        return None
    with path.open() as f:
        existing = json.load(f)
    ac = existing.get("adaptive_cluster", {})
    if "error" in ac or "labels" not in ac:
        return None
    return ac


def _run_one(
    dataset: str,
    subject: str,
    min_shared: int,
    max_events_for_mds: int,
    subsample_seed: int,
    dry_run: bool = False,
) -> Optional[Dict[str, Any]]:
    key = f"{dataset}/{subject}"
    subject_dir = _subject_dir(dataset, subject)
    if not subject_dir.exists() or not list(subject_dir.glob("*_lagPat.npz")):
        logger.warning("Skip %s: no *_lagPat.npz at %s", key, subject_dir)
        return None

    ac = _load_adaptive_cluster(dataset, subject)
    if ac is None:
        logger.warning("Skip %s: adaptive_cluster missing in per-subject JSON", key)
        return None

    if dry_run:
        logger.info("[dry-run] would run %s (chosen_k=%s n_valid=%s)",
                    key, ac.get("chosen_k"), ac.get("n_valid_events"))
        return None

    logger.info("Running %s ...", key)
    loaded = load_subject_propagation_events(subject_dir)
    valid_events = _valid_event_indices(loaded["bools"], min_participating=min_shared)
    labels = np.asarray(ac["labels"], dtype=int)
    chosen_k = int(ac["chosen_k"])

    if valid_events.size != labels.size:
        logger.error(
            "Skip %s: valid_events size %d != labels size %d",
            key, valid_events.size, labels.size,
        )
        return None

    out = compute_subject_geometry(
        ranks=loaded["ranks"],
        bools=loaded["bools"],
        channel_names=loaded["channel_names"],
        adaptive_labels=labels,
        chosen_k=chosen_k,
        valid_event_indices=valid_events,
        event_abs_times=loaded["event_abs_times"],
        block_ids=loaded["block_ids"],
        min_shared=min_shared,
        max_events_for_mds=max_events_for_mds,
        subsample_seed=subsample_seed,
    )
    out["dataset"] = dataset
    out["subject"] = subject

    json_path = PER_SUBJECT_DIR / f"{dataset}_{subject}.json"
    _save(out, json_path)

    if out["status"] == "ok":
        logger.info(
            "  %s: status=ok n_events=%d k=%d sil_med=%.3f agreement=%.3f stress=%.3f",
            key,
            out["n_events_total"],
            out["chosen_k"],
            out["silhouette_median"],
            out["agreement_overall"],
            out["stress"],
        )
    else:
        logger.warning("  %s: excluded (%s)", key, out.get("excluded_reason"))

    return out


def _aggregate_cohort() -> Dict[str, Any]:
    """Re-load all per-subject JSONs from PER_SUBJECT_DIR and aggregate."""
    per_subject: Dict[str, Dict[str, Any]] = {}
    if not PER_SUBJECT_DIR.exists():
        logger.warning("%s does not exist; cohort summary will be empty", PER_SUBJECT_DIR)
        return {"per_subject": {}, "excluded": {}, "n_subjects_included": 0, "n_subjects_excluded": 0}

    for path in sorted(PER_SUBJECT_DIR.glob("*.json")):
        with path.open() as f:
            data = json.load(f)
        ds = data.get("dataset", "unknown")
        subj = data.get("subject", path.stem)
        key = f"{ds}/{subj}"
        per_subject[key] = data

    summary = summarize_cohort_geometry(per_subject)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["yuquan", "epilepsiae", "both"],
        default="both",
        help="Which dataset(s) to process.",
    )
    parser.add_argument(
        "--subject",
        action="append",
        default=None,
        help="Restrict to specific subject(s) (repeatable). If omitted, runs all configured subjects in --dataset.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all configured subjects (default behavior; explicit flag for clarity).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List subjects that would be processed; do not run.",
    )
    parser.add_argument(
        "--min-shared",
        type=int,
        default=DEFAULT_MIN_SHARED_CHANNELS,
        help="Minimum jointly-participating channels for a valid pair distance.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=DEFAULT_N_MAX_FOR_MDS,
        help="Subsample threshold above which events are randomly downsampled before MDS.",
    )
    parser.add_argument(
        "--subsample-seed",
        type=int,
        default=DEFAULT_SUBSAMPLE_SEED,
        help="Seed for subsample reproducibility.",
    )
    parser.add_argument(
        "--cohort-only",
        action="store_true",
        help="Skip per-subject runs; only re-aggregate from existing per-subject JSONs.",
    )
    args = parser.parse_args()

    datasets = ["yuquan", "epilepsiae"] if args.dataset == "both" else [args.dataset]
    targets = _resolve_dataset_subjects(datasets, args.subject)
    logger.info("%d subject(s) targeted: %s", len(targets), targets if len(targets) <= 8 else f"{len(targets)} total")

    if not args.cohort_only:
        for dataset, subject in targets:
            _run_one(
                dataset=dataset,
                subject=subject,
                min_shared=args.min_shared,
                max_events_for_mds=args.max_events,
                subsample_seed=args.subsample_seed,
                dry_run=args.dry_run,
            )

    if args.dry_run:
        logger.info("dry-run complete")
        return

    cohort = _aggregate_cohort()
    _save(cohort, GEOMETRY_DIR / "cohort_summary.json")
    logger.info(
        "Cohort summary: included=%d excluded=%d high_stress=%d high_imputation=%d",
        cohort.get("n_subjects_included", 0),
        cohort.get("n_subjects_excluded", 0),
        len(cohort.get("subjects_high_stress", [])),
        len(cohort.get("subjects_high_imputation", [])),
    )


if __name__ == "__main__":
    main()
