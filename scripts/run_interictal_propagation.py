#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.interictal_propagation import (  # noqa: E402
    _valid_event_indices,
    compute_time_split_reproducibility,
    load_subject_propagation_events,
    run_subject_interictal_propagation_pr1,
    summarize_propagation_cohort,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("interictal_propagation")

YUQUAN_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")
RESULTS_DIR = Path("results/interictal_propagation")

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

SOZ_FILE_YQ = Path("results/yuquan_soz_core_channels.json")
SOZ_FILE_EPI = Path("results/epilepsiae_soz_core_channels.json")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        try:
            import numpy as np

            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
        except Exception:
            pass
        return super().default(obj)


def _load_soz(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _save(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, cls=NumpyEncoder, indent=2)
    logger.info("Saved %s", path)


def _run_pr25(
    datasets_list: List,
    per_subject_dir: Path,
) -> None:
    """PR-2.5: augment existing per-subject JSONs with cross-time reproducibility."""
    import numpy as np

    all_results: Dict[str, Dict[str, Any]] = {}

    for dataset, root, subjects, _soz_map in datasets_list:
        for subject in subjects:
            subject_dir = root / subject if dataset == "yuquan" else root / subject / "all_recs"
            json_path = per_subject_dir / f"{dataset}_{subject}.json"
            key = f"{dataset}/{subject}"

            if not json_path.exists():
                logger.warning("Skip %s: no existing JSON at %s", key, json_path)
                continue

            with open(json_path) as f:
                existing = json.load(f)

            ac = existing.get("adaptive_cluster", {})
            if "error" in ac or "labels" not in ac:
                logger.warning("Skip %s: adaptive_cluster missing or errored", key)
                all_results[key] = existing
                continue

            if not subject_dir.exists() or not list(subject_dir.glob("*_lagPat.npz")):
                logger.warning("Skip %s: raw data dir missing", key)
                all_results[key] = existing
                continue

            logger.info("PR-2.5 reproducibility: %s", key)
            try:
                loaded = load_subject_propagation_events(subject_dir)
                valid_events = _valid_event_indices(loaded["bools"], min_participating=3)
                labels = np.array(ac["labels"], dtype=int)
                chosen_k = int(ac["chosen_k"])

                if valid_events.size != len(labels):
                    logger.error(
                        "Skip %s: valid_events size %d != labels size %d",
                        key, valid_events.size, len(labels),
                    )
                    all_results[key] = existing
                    continue

                repro = compute_time_split_reproducibility(
                    ranks=loaded["ranks"],
                    bools=loaded["bools"],
                    event_abs_times=loaded["event_abs_times"],
                    block_ids=loaded["block_ids"],
                    chosen_k=chosen_k,
                    adaptive_labels=labels,
                    valid_event_indices=valid_events,
                )
                existing["time_split_reproducibility"] = repro
                logger.info(
                    "  %s: grade=%s  split-half corr=%.3f agree=%.3f",
                    key,
                    repro["reproducibility_grade"],
                    repro["splits"].get("first_half_second_half", {}).get("mean_match_corr", float("nan")),
                    repro["splits"].get("first_half_second_half", {}).get("assignment_agreement", float("nan")),
                )
            except Exception as exc:
                logger.exception("Failed PR-2.5 for %s", key)
                existing["time_split_reproducibility"] = {"error": str(exc)}

            _save(existing, json_path)
            all_results[key] = existing

    cohort = summarize_propagation_cohort(all_results)
    _save(all_results, RESULTS_DIR / "pr1_subject_summary.json")
    _save(cohort, RESULTS_DIR / "pr1_cohort_summary.json")
    logger.info("PR-2.5 done. Cohort reproducibility: %s", cohort.get("reproducibility_analysis", {}))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interictal group-event internal propagation PR-1 analysis"
    )
    parser.add_argument("--dataset", choices=["yuquan", "epilepsiae", "both"], default="both")
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--smoke", action="store_true", help="Run chengshuai + 548 only")
    parser.add_argument("--pr25", action="store_true", help="PR-2.5: cross-time template reproducibility only")
    parser.add_argument("--n-sample", type=int, default=200)
    parser.add_argument("--n-seeds", type=int, default=5)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    per_subject_dir = RESULTS_DIR / "per_subject"
    soz_yq = _load_soz(SOZ_FILE_YQ)
    soz_epi = _load_soz(SOZ_FILE_EPI)

    if args.smoke:
        yq_subjects = ["chengshuai"]
        epi_subjects = ["548"]
    elif args.subjects:
        yq_subjects = [s for s in args.subjects if s in YUQUAN_SUBJECTS]
        epi_subjects = [s for s in args.subjects if s in EPILEPSIAE_SUBJECTS]
    else:
        yq_subjects = YUQUAN_SUBJECTS
        epi_subjects = EPILEPSIAE_SUBJECTS

    datasets_list = []
    if args.dataset in ("yuquan", "both"):
        datasets_list.append(("yuquan", YUQUAN_ROOT, yq_subjects, soz_yq))
    if args.dataset in ("epilepsiae", "both"):
        datasets_list.append(("epilepsiae", EPILEPSIAE_ROOT, epi_subjects, soz_epi))

    if args.pr25:
        _run_pr25(datasets_list, per_subject_dir)
        return

    subject_results: Dict[str, Dict[str, Any]] = {}
    for dataset, root, subjects, soz_map in datasets_list:
        for subject in subjects:
            subject_dir = root / subject if dataset == "yuquan" else root / subject / "all_recs"
            if not subject_dir.exists():
                logger.warning("Skip %s/%s: subject dir missing", dataset, subject)
                continue
            if not list(subject_dir.glob("*_lagPat.npz")):
                logger.warning("Skip %s/%s: no lagPat", dataset, subject)
                continue

            key = f"{dataset}/{subject}"
            logger.info("Running %s", key)
            try:
                result = run_subject_interictal_propagation_pr1(
                    subject_dir=subject_dir,
                    dataset=dataset,
                    subject=subject,
                    soz_channels=soz_map.get(subject, []),
                    n_sample=args.n_sample,
                    n_seeds=args.n_seeds,
                )
            except Exception as exc:
                logger.exception("Failed %s", key)
                result = {"dataset": dataset, "subject": subject, "error": str(exc)}

            subject_results[key] = result
            _save(result, per_subject_dir / f"{dataset}_{subject}.json")

    cohort_summary = summarize_propagation_cohort(subject_results)
    _save(subject_results, RESULTS_DIR / "pr1_subject_summary.json")
    _save(cohort_summary, RESULTS_DIR / "pr1_cohort_summary.json")


if __name__ == "__main__":
    main()
