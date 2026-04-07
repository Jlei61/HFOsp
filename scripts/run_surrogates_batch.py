#!/usr/bin/env python3
"""Run group-level surrogate tests for all subjects, skipping finished outputs.

This is the batch helper for the two key null models behind the current
scientific conclusion:
  - ISI-shuffle: preserves the IEI distribution but destroys temporal order
  - Gamma renewal: matches rate + refractory structure without an oscillator
"""

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.event_periodicity import run_subject_periodicity, save_subject_result

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("surrogates_batch")

YUQUAN_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")
RESULTS_DIR = Path("results/event_periodicity")

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


def needs_surrogates(json_path: Path) -> bool:
    if not json_path.exists():
        return True
    with open(json_path) as f:
        d = json.load(f)
    g = d.get("group")
    if not g:
        return True
    return g.get("surrogate_gamma") is None


def run_all():
    summary = []

    for sub in YUQUAN_SUBJECTS:
        out = RESULTS_DIR / "yuquan" / f"{sub}_periodicity.json"
        if not needs_surrogates(out):
            logger.info(f"SKIP {sub} (already done)")
            continue
        sub_dir = YUQUAN_ROOT / sub
        if not sub_dir.exists():
            continue
        if not list(sub_dir.glob("*_lagPat.npz")):
            continue

        t0 = time.time()
        try:
            r = run_subject_periodicity(sub_dir, "yuquan", sub, n_surrogates=200)
            save_subject_result(r, out)
            g = r.group
            if g and g.surrogate_gamma:
                row = f"yuquan/{sub}: ISI_p={g.surrogate_isi.p_value:.3f}, Gamma_p={g.surrogate_gamma.p_value:.3f}, {time.time()-t0:.0f}s"
            else:
                row = f"yuquan/{sub}: no group surrogates"
            logger.info(row)
            summary.append(row)
        except Exception as e:
            logger.error(f"yuquan/{sub}: FAILED — {e}")
            summary.append(f"yuquan/{sub}: ERROR {e}")

    for sub in EPILEPSIAE_SUBJECTS:
        out = RESULTS_DIR / "epilepsiae" / f"{sub}_periodicity.json"
        if not needs_surrogates(out):
            logger.info(f"SKIP {sub} (already done)")
            continue
        sub_dir = EPILEPSIAE_ROOT / sub / "all_recs"
        if not sub_dir.exists():
            continue
        if not list(sub_dir.glob("*_lagPat.npz")):
            continue

        t0 = time.time()
        try:
            r = run_subject_periodicity(sub_dir, "epilepsiae", sub, n_surrogates=200)
            save_subject_result(r, out)
            g = r.group
            if g and g.surrogate_gamma:
                row = f"epilepsiae/{sub}: ISI_p={g.surrogate_isi.p_value:.3f}, Gamma_p={g.surrogate_gamma.p_value:.3f}, {time.time()-t0:.0f}s"
            else:
                row = f"epilepsiae/{sub}: no group surrogates"
            logger.info(row)
            summary.append(row)
        except Exception as e:
            logger.error(f"epilepsiae/{sub}: FAILED — {e}")
            summary.append(f"epilepsiae/{sub}: ERROR {e}")

    logger.info("=== SUMMARY ===")
    for row in summary:
        logger.info(row)


if __name__ == "__main__":
    run_all()
