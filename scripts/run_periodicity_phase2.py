#!/usr/bin/env python3
"""
Phase 2 event periodicity analysis: isolate artifact source and apply correct tools.

Experiments:
  1. PackWinLen sweep — f_peak vs window_sec (Yuquan only, needs _gpu.npz)
  2. Centroid bypass — window_start vs mean/ignition centroid PSD
  3. Hazard function — H(t) of IEI for packed and centroid events
  4. IEI return map — IEI[n] vs IEI[n+1] serial structure
  5. Propagation stereotypy — Kendall tau of channel activation order

Usage:
    # All experiments, smoke test
    python scripts/run_periodicity_phase2.py --smoke

    # Experiment 1 only, specific subjects
    python scripts/run_periodicity_phase2.py --exp 1 --subjects chengshuai huangwanling

    # All experiments, both datasets
    python scripts/run_periodicity_phase2.py --exp all

Interpretation notes:
  - Exp 2 compares alternative anchors within the existing legacy packed-window
    framework; it is not a fully independent absolute-timestamp rebuild.
  - Exp 3 hazard outputs are qualitative summaries of dead-time structure.
  - Exp 4 stores serial correlation on log(IEI[n]), log(IEI[n+1]); the Pearson
    p-value is descriptive only and should not be used as the final inferential
    statistic.
  - Exp 5 is an exploratory, sub-sampled stereotypy summary intended for
    follow-up with more rigorous resampling / mixed-model analysis.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.event_periodicity import (
    build_pulse_train,
    compute_event_psd,
    compute_hazard_function,
    compute_iei,
    compute_iei_return_map,
    compute_propagation_stereotypy,
    fit_psd_periodic,
    load_centroid_event_times,
    load_epilepsiae_subject_events,
    load_yuquan_subject_events,
    run_centroid_bypass,
    run_packing_sweep,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("phase2")

YUQUAN_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")
RESULTS_DIR = Path("results/event_periodicity/phase2")

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

SOZ_FILE = Path("results/yuquan_soz_core_channels.json")
SOZ_FILE_EPI = Path("results/epilepsiae_soz_core_channels.json")


def _load_soz(path: Path) -> Dict[str, List[str]]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        return super().default(obj)


def _save(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, cls=NumpyEncoder, indent=2)
    logger.info(f"Saved {path}")


# ==========================================================================
# Experiment 1: PackWinLen sweep (Yuquan only)
# ==========================================================================

def run_exp1(subjects: List[str], out_dir: Path):
    """Sweep window_sec, track f_peak for each subject."""
    logger.info("=== Experiment 1: PackWinLen Sweep ===")
    all_results = {}

    for sub in subjects:
        sub_dir = YUQUAN_ROOT / sub
        if not sub_dir.exists():
            logger.warning(f"  SKIP {sub}: dir not found")
            continue
        if not list(sub_dir.glob("*_gpu.npz")):
            logger.warning(f"  SKIP {sub}: no _gpu.npz")
            continue

        logger.info(f"  {sub}: starting sweep")
        t0 = time.time()
        try:
            results = run_packing_sweep(sub_dir)
            all_results[sub] = results
            freqs_str = ", ".join(
                f"W={r['window_sec']:.2f}→{r.get('peak_freq', 'N/A')}"
                for r in results
            )
            logger.info(f"  {sub}: {time.time()-t0:.1f}s — {freqs_str}")
        except Exception as e:
            logger.error(f"  {sub}: FAILED — {e}", exc_info=True)
            all_results[sub] = {"error": str(e)}

    _save(all_results, out_dir / "exp1_packing_sweep.json")
    return all_results


# ==========================================================================
# Experiment 2: Centroid bypass
# ==========================================================================

def run_exp2(subjects: List[str], roots: Dict[str, Path], out_dir: Path):
    """Compare window_start vs centroid event definitions."""
    logger.info("=== Experiment 2: Centroid Bypass ===")
    all_results = {}

    for dataset, root in roots.items():
        for sub in subjects:
            sub_dir = root / sub if dataset == "yuquan" else root / sub / "all_recs"
            if not sub_dir.exists():
                continue
            if not list(sub_dir.glob("*_lagPat.npz")):
                continue

            key = f"{dataset}/{sub}"
            logger.info(f"  {key}: centroid bypass")
            t0 = time.time()
            try:
                loader = load_yuquan_subject_events if dataset == "yuquan" else load_epilepsiae_subject_events
                _, _, _, block_ranges = loader(sub_dir)
                results = run_centroid_bypass(sub_dir, dataset, block_ranges)
                all_results[key] = results
                for method, r in results.items():
                    logger.info(f"    {method}: peak={r.get('peak_freq', 'N/A')}, n={r.get('n_events', 0)}")
                logger.info(f"  {key}: {time.time()-t0:.1f}s")
            except Exception as e:
                logger.error(f"  {key}: FAILED — {e}", exc_info=True)
                all_results[key] = {"error": str(e)}

    _save(all_results, out_dir / "exp2_centroid_bypass.json")
    return all_results


# ==========================================================================
# Experiment 3: Hazard function
# ==========================================================================

def run_exp3(subjects: List[str], roots: Dict[str, Path], out_dir: Path):
    """Compute hazard function for packed and centroid events."""
    logger.info("=== Experiment 3: Hazard Function ===")
    all_results = {}

    for dataset, root in roots.items():
        for sub in subjects:
            sub_dir = root / sub if dataset == "yuquan" else root / sub / "all_recs"
            if not sub_dir.exists():
                continue
            if not list(sub_dir.glob("*_lagPat.npz")):
                continue

            key = f"{dataset}/{sub}"
            logger.info(f"  {key}: hazard function")
            t0 = time.time()
            try:
                loader = load_yuquan_subject_events if dataset == "yuquan" else load_epilepsiae_subject_events
                _, packed, _, block_ranges = loader(sub_dir)

                sub_result = {}
                if len(packed) >= 10:
                    iei = compute_iei(packed, block_ranges=block_ranges)
                    t_grid, hazard, pdf = compute_hazard_function(iei)
                    sub_result["packed"] = {
                        "n_iei": len(iei),
                        "iei_min": float(np.min(iei)) if len(iei) else None,
                        "t": t_grid.tolist(),
                        "hazard": hazard.tolist(),
                        "pdf": pdf.tolist(),
                    }

                centroid_series = load_centroid_event_times(sub_dir, dataset)
                for method in ["mean_centroid", "ignition_centroid"]:
                    events = centroid_series.get(method, np.zeros((0, 2)))
                    if len(events) >= 10:
                        iei = compute_iei(events, block_ranges=block_ranges)
                        t_grid, hazard, pdf = compute_hazard_function(iei)
                        sub_result[method] = {
                            "n_iei": len(iei),
                            "iei_min": float(np.min(iei)) if len(iei) else None,
                            "t": t_grid.tolist(),
                            "hazard": hazard.tolist(),
                            "pdf": pdf.tolist(),
                        }

                all_results[key] = sub_result
                logger.info(f"  {key}: done ({time.time()-t0:.1f}s)")
            except Exception as e:
                logger.error(f"  {key}: FAILED — {e}", exc_info=True)
                all_results[key] = {"error": str(e)}

    _save(all_results, out_dir / "exp3_hazard.json")
    return all_results


# ==========================================================================
# Experiment 4: IEI Return map
# ==========================================================================

def run_exp4(subjects: List[str], roots: Dict[str, Path], out_dir: Path):
    """Compute IEI return map for packed events.

    The returned serial_corr_p is a naive Pearson p-value on consecutive
    log-IEI pairs. Keep it for descriptive output only; formal reporting should
    use subject-level direction consistency or block/bootstrap procedures.
    """
    logger.info("=== Experiment 4: IEI Return Map ===")
    all_results = {}

    for dataset, root in roots.items():
        for sub in subjects:
            sub_dir = root / sub if dataset == "yuquan" else root / sub / "all_recs"
            if not sub_dir.exists():
                continue
            if not list(sub_dir.glob("*_lagPat.npz")):
                continue

            key = f"{dataset}/{sub}"
            logger.info(f"  {key}: return map")
            try:
                loader = load_yuquan_subject_events if dataset == "yuquan" else load_epilepsiae_subject_events
                _, packed, _, block_ranges = loader(sub_dir)
                sub_result = {}

                if len(packed) >= 10:
                    iei = compute_iei(packed, block_ranges=block_ranges)
                    rm = compute_iei_return_map(iei)
                    sub_result["packed"] = {
                        "serial_corr": rm["serial_corr"],
                        "serial_corr_p": rm["serial_corr_p"],
                        "n_iei": len(iei),
                        "iei_n": rm["iei_n"].tolist(),
                        "iei_n1": rm["iei_n1"].tolist(),
                    }

                centroid_series = load_centroid_event_times(sub_dir, dataset)
                for method in ["mean_centroid", "ignition_centroid"]:
                    events = centroid_series.get(method, np.zeros((0, 2)))
                    if len(events) >= 10:
                        iei = compute_iei(events, block_ranges=block_ranges)
                        rm = compute_iei_return_map(iei)
                        sub_result[method] = {
                            "serial_corr": rm["serial_corr"],
                            "serial_corr_p": rm["serial_corr_p"],
                            "n_iei": len(iei),
                        }

                all_results[key] = sub_result
                logger.info(f"  {key}: serial_corr={sub_result.get('packed', {}).get('serial_corr', 'N/A')}")
            except Exception as e:
                logger.error(f"  {key}: FAILED — {e}", exc_info=True)
                all_results[key] = {"error": str(e)}

    _save(all_results, out_dir / "exp4_return_map.json")
    return all_results


# ==========================================================================
# Experiment 5: Propagation stereotypy
# ==========================================================================

def run_exp5(subjects: List[str], roots: Dict[str, Path], out_dir: Path):
    """Compute propagation stereotypy (Kendall tau) with SOZ stratification."""
    logger.info("=== Experiment 5: Propagation Stereotypy ===")
    soz_yq = _load_soz(SOZ_FILE)
    soz_epi = _load_soz(SOZ_FILE_EPI)
    all_results = {}

    for dataset, root in roots.items():
        soz_map = soz_yq if dataset == "yuquan" else soz_epi
        for sub in subjects:
            sub_dir = root / sub if dataset == "yuquan" else root / sub / "all_recs"
            if not sub_dir.exists():
                continue
            if not list(sub_dir.glob("*_lagPat.npz")):
                continue

            key = f"{dataset}/{sub}"
            soz_ch = soz_map.get(sub, [])
            logger.info(f"  {key}: stereotypy (SOZ={len(soz_ch)} channels)")
            try:
                result = compute_propagation_stereotypy(
                    sub_dir, dataset, soz_channels=soz_ch if soz_ch else None)
                all_results[key] = result
                logger.info(f"  {key}: mean_tau={result.get('mean_tau', 'N/A'):.3f}, "
                            f"n={result.get('n_events', 0)}")
            except Exception as e:
                logger.error(f"  {key}: FAILED — {e}", exc_info=True)
                all_results[key] = {"error": str(e)}

    _save(all_results, out_dir / "exp5_stereotypy.json")
    return all_results


# ==========================================================================
# Main
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 2 periodicity experiments")
    parser.add_argument("--exp", default="all",
                        help="Experiment number(s): 1,2,3,4,5 or 'all'")
    parser.add_argument("--dataset", choices=["yuquan", "epilepsiae", "both"],
                        default="both")
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="Quick test: chengshuai + 548")
    args = parser.parse_args()

    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.exp == "all":
        exps = {1, 2, 3, 4, 5}
    else:
        exps = {int(x) for x in args.exp.split(",")}

    if args.smoke:
        yq_subs = ["chengshuai"]
        epi_subs = ["548"]
    elif args.subjects:
        yq_subs = [s for s in args.subjects if s in YUQUAN_SUBJECTS]
        epi_subs = [s for s in args.subjects if s in EPILEPSIAE_SUBJECTS]
    else:
        yq_subs = YUQUAN_SUBJECTS
        epi_subs = EPILEPSIAE_SUBJECTS

    roots: Dict[str, Path] = {}
    all_subs: List[str] = []
    if args.dataset in ("yuquan", "both"):
        roots["yuquan"] = YUQUAN_ROOT
        all_subs.extend(yq_subs)
    if args.dataset in ("epilepsiae", "both"):
        roots["epilepsiae"] = EPILEPSIAE_ROOT
        all_subs.extend(epi_subs)

    t_total = time.time()

    if 1 in exps:
        run_exp1(yq_subs, out_dir)

    if 2 in exps:
        run_exp2(all_subs, roots, out_dir)

    if 3 in exps:
        run_exp3(all_subs, roots, out_dir)

    if 4 in exps:
        run_exp4(all_subs, roots, out_dir)

    if 5 in exps:
        run_exp5(all_subs, roots, out_dir)

    logger.info(f"=== Phase 2 complete: {time.time()-t_total:.0f}s total ===")


if __name__ == "__main__":
    main()
