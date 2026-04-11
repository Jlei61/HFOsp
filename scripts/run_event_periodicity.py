#!/usr/bin/env python3
"""
Run event periodicity analysis on Yuquan and/or Epilepsiae datasets.

Usage:
    # Smoke test (2 subjects, no surrogates)
    python scripts/run_event_periodicity.py --smoke

    # Full Yuquan
    python scripts/run_event_periodicity.py --dataset yuquan

    # Full Epilepsiae
    python scripts/run_event_periodicity.py --dataset epilepsiae

    # Both datasets, with surrogates
    python scripts/run_event_periodicity.py --dataset both --surrogates 200

This script reproduces the legacy-style periodicity metrics and adds the null
model checks used in the current review. Its outputs are analysis artifacts,
not direct scientific conclusions by themselves: the key interpretation lives in
`docs/archive/topic2/event_periodicity_analysis.md` and the follow-up review docs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.event_periodicity import (
    run_subject_periodicity,
    save_subject_result,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_event_periodicity")

# ---- Dataset definitions ----

YUQUAN_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")

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

YUQUAN_YNUM = {s: f"Y{i+1}" for i, s in enumerate(YUQUAN_SUBJECTS)}
EPILEPSIAE_ENUM = {s: f"E{i+1}" for i, s in enumerate(EPILEPSIAE_SUBJECTS)}

RESULTS_DIR = Path("results/event_periodicity")


def run_dataset(
    dataset: str,
    subjects: list,
    root: Path,
    n_surrogates: int,
    run_surrogates: bool,
):
    out_dir = RESULTS_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for sub in subjects:
        if dataset == "epilepsiae":
            sub_dir = root / sub / "all_recs"
        else:
            sub_dir = root / sub

        if not sub_dir.exists():
            logger.warning(f"Skipping {sub}: {sub_dir} not found")
            continue

        logger.info(f"=== {dataset} / {sub} ===")
        t0 = time.time()

        try:
            result = run_subject_periodicity(
                subject_dir=sub_dir,
                dataset=dataset,
                subject_name=sub,
                n_surrogates=n_surrogates,
                run_surrogates=run_surrogates,
            )
            out_file = out_dir / f"{sub}_periodicity.json"
            save_subject_result(result, out_file)

            grp = result.group
            row = {
                "subject": sub,
                "n_channels": len(result.channels),
                "group_n_events": grp.n_events if grp else 0,
                "group_duration_h": grp.recording_duration_sec / 3600.0 if grp else 0,
            }
            if grp and grp.specparam and len(grp.specparam.peaks) > 0:
                peaks = grp.specparam.peaks
                primary = peaks[np.argmin(peaks[:, 0])] if len(peaks) > 0 else None
                if primary is not None and 0.5 < primary[0] < 4.0:
                    row["group_peak_freq"] = float(primary[0])
                else:
                    row["group_peak_freq"] = None
                row["group_psd_r2"] = float(grp.specparam.r_squared)
            else:
                row["group_peak_freq"] = None
                row["group_psd_r2"] = None

            if grp and grp.iei_fit:
                row["group_iei_alpha"] = float(grp.iei_fit.alpha)
                row["group_iei_pl_vs_ln_p"] = float(grp.iei_fit.pl_vs_ln_p)
            else:
                row["group_iei_alpha"] = None
                row["group_iei_pl_vs_ln_p"] = None

            if grp and grp.surrogate_gamma:
                row["group_gamma_null_p"] = float(grp.surrogate_gamma.p_value)
            else:
                row["group_gamma_null_p"] = None

            summary.append(row)

            elapsed = time.time() - t0
            logger.info(
                f"  {sub}: {grp.n_events if grp else 0} group events, "
                f"peak={row.get('group_peak_freq', 'N/A')}, "
                f"elapsed={elapsed:.1f}s"
            )

        except Exception as e:
            logger.error(f"  {sub}: FAILED — {e}", exc_info=True)
            summary.append({"subject": sub, "error": str(e)})

    summary_file = out_dir / "cohort_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Cohort summary: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Event periodicity analysis")
    parser.add_argument("--dataset", choices=["yuquan", "epilepsiae", "both"],
                        default="both")
    parser.add_argument("--surrogates", type=int, default=0,
                        help="Number of surrogate permutations (0=skip)")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test: chengshuai + 384, no surrogates")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Override subject list")
    args = parser.parse_args()

    if args.smoke:
        logger.info("=== SMOKE TEST ===")
        run_dataset("yuquan", ["chengshuai"], YUQUAN_ROOT,
                    n_surrogates=0, run_surrogates=False)
        run_dataset("epilepsiae", ["384"], EPILEPSIAE_ROOT,
                    n_surrogates=0, run_surrogates=False)
        return

    run_surrogates = args.surrogates > 0

    if args.dataset in ("yuquan", "both"):
        subs = args.subjects if args.subjects else YUQUAN_SUBJECTS
        run_dataset("yuquan", subs, YUQUAN_ROOT,
                    n_surrogates=args.surrogates, run_surrogates=run_surrogates)

    if args.dataset in ("epilepsiae", "both"):
        subs = args.subjects if args.subjects else EPILEPSIAE_SUBJECTS
        run_dataset("epilepsiae", subs, EPILEPSIAE_ROOT,
                    n_surrogates=args.surrogates, run_surrogates=run_surrogates)


if __name__ == "__main__":
    main()
