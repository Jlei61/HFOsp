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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interictal group-event internal propagation PR-1 analysis"
    )
    parser.add_argument("--dataset", choices=["yuquan", "epilepsiae", "both"], default="both")
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--smoke", action="store_true", help="Run chengshuai + 548 only")
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

    subject_results: Dict[str, Dict[str, Any]] = {}
    datasets = []
    if args.dataset in ("yuquan", "both"):
        datasets.append(("yuquan", YUQUAN_ROOT, yq_subjects, soz_yq))
    if args.dataset in ("epilepsiae", "both"):
        datasets.append(("epilepsiae", EPILEPSIAE_ROOT, epi_subjects, soz_epi))

    for dataset, root, subjects, soz_map in datasets:
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
