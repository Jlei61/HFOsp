#!/usr/bin/env python3
"""
Phase 2 event periodicity analysis: isolate artifact source and apply correct tools.

Experiments:
  1. PackWinLen sweep — f_peak vs window_sec (Yuquan only, needs _gpu.npz)
  2. Centroid bypass — window_start vs mean/ignition centroid PSD
  3. Hazard function — H(t) of IEI for packed and centroid events
  4. IEI return map — IEI[n] vs IEI[n+1] serial structure
  5. Propagation stereotypy — Kendall tau of channel activation order
  6. Renewal analytic PSD + SOZ dead-time stratification
  7. Serial-correlation deep dive — lag-k decay, detrending, within-block, SOZ split

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
    _log_iei_pairs,
    _pearson_corr,
    _rolling_log_iei_residuals,
    _serial_corr_decay_from_sequences,
    _split_events_by_block,
    build_pulse_train,
    compute_contiguous_daynight_detrending,
    compute_daynight_stratified_detrending,
    compute_detrended_psd_backfill,
    compute_long_timescale_rate_summary,
    compute_multiscale_detrend_fraction,
    compute_nparticipating_autocorrelation,
    compute_renewal_psd_analytic,
    compute_event_psd,
    compute_hazard_function,
    compute_iei,
    compute_iei_return_map,
    compute_propagation_stereotypy,
    compute_serial_corr_soz_stratified,
    compute_soz_stratified_deadtime,
    compute_within_block_serial_corr,
    fit_psd_periodic,
    load_centroid_event_times,
    load_epilepsiae_subject_events,
    load_yuquan_subject_events,
    merge_contiguous_blocks,
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


def _load_phase1_group(dataset: str, subject: str) -> Dict[str, Any]:
    path = Path("results/event_periodicity") / dataset / f"{subject}_periodicity.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return data.get("group", {})


def _normalize_unit_area(freqs: np.ndarray, psd: np.ndarray,
                         f_lo: float = 0.5, f_hi: float = 10.0) -> np.ndarray:
    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)
    out = np.zeros_like(psd, dtype=float)
    band = (freqs >= f_lo) & (freqs <= f_hi) & np.isfinite(psd) & (psd >= 0)
    if np.any(band):
        area = np.trapz(psd[band], freqs[band])
        if area > 0:
            out = psd / area
    return out


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
# Experiment 6: Renewal analytic PSD + SOZ dead-time
# ==========================================================================

def run_exp6(subjects: List[str], roots: Dict[str, Path], out_dir: Path):
    """Overlay analytic renewal PSD prediction and compute SOZ dead-time split."""
    logger.info("=== Experiment 6: Renewal PSD + SOZ dead-time ===")
    soz_yq = _load_soz(SOZ_FILE)
    soz_epi = _load_soz(SOZ_FILE_EPI)
    psd_results: Dict[str, Any] = {}
    deadtime_results: Dict[str, Any] = {}

    for dataset, root in roots.items():
        soz_map = soz_yq if dataset == "yuquan" else soz_epi
        loader = load_yuquan_subject_events if dataset == "yuquan" else load_epilepsiae_subject_events

        for sub in subjects:
            sub_dir = root / sub if dataset == "yuquan" else root / sub / "all_recs"
            if not sub_dir.exists():
                continue
            if not list(sub_dir.glob("*_lagPat.npz")):
                continue

            key = f"{dataset}/{sub}"
            logger.info(f"  {key}: renewal overlay")
            try:
                group = _load_phase1_group(dataset, sub)
                if not group:
                    psd_results[key] = {"warning": "phase1_group_json_missing"}
                    deadtime_results[key] = {"warning": "phase1_group_json_missing"}
                    continue

                _, packed, _, block_ranges = loader(sub_dir)
                if len(packed) < 10:
                    psd_results[key] = {"warning": "insufficient_group_events"}
                    deadtime_results[key] = {"warning": "insufficient_group_events"}
                    continue

                iei = compute_iei(packed, block_ranges=block_ranges)
                if len(iei) < 10:
                    psd_results[key] = {"warning": "insufficient_iei"}
                    deadtime_results[key] = {"warning": "insufficient_iei"}
                    continue

                fs = float(group.get("psd", {}).get("fs_pulse", 100.0))
                nperseg_samples = int(group.get("psd", {}).get("nperseg", int(500 * fs)))
                nperseg_sec = nperseg_samples / fs if fs > 0 else 500.0

                pulse_delta, mask_delta, _ = build_pulse_train(
                    packed, fs=fs, mode="delta", block_ranges=block_ranges
                )
                psd_delta = compute_event_psd(
                    pulse_delta, mask_delta, fs=fs, nperseg_sec=nperseg_sec
                )

                freqs = psd_delta.freqs
                positive = freqs > 0
                mean_event_dur = float(np.mean(packed[:, 1] - packed[:, 0]))

                analytic_delta = np.zeros_like(freqs, dtype=float)
                analytic_rect = np.zeros_like(freqs, dtype=float)
                analytic_delta[positive] = compute_renewal_psd_analytic(
                    iei, freqs[positive], event_dur=0.0
                )
                analytic_rect[positive] = compute_renewal_psd_analytic(
                    iei, freqs[positive], event_dur=mean_event_dur
                )

                phase1_psd = np.asarray(group.get("psd", {}).get("power", []), dtype=float)
                phase1_freqs = np.asarray(group.get("psd", {}).get("freqs", []), dtype=float)
                phase1_ap = np.asarray(group.get("specparam", {}).get("ap_fit", []), dtype=float)
                phase1_sp_freqs = np.asarray(group.get("specparam", {}).get("freqs", []), dtype=float)
                has_peak = False
                peaks = np.asarray(group.get("specparam", {}).get("peaks", []), dtype=float)
                if peaks.size > 0 and peaks.ndim == 2 and peaks.shape[1] >= 2:
                    peak_mask = (peaks[:, 0] >= 0.5) & (peaks[:, 0] <= 10.0)
                    has_peak = bool(np.any(peak_mask))

                psd_results[key] = {
                    "dataset": dataset,
                    "subject": sub,
                    "has_peak_0p5_10hz": has_peak,
                    "iei_n": int(len(iei)),
                    "iei_min": float(np.min(iei)),
                    "iei_median": float(np.median(iei)),
                    "iei_mean": float(np.mean(iei)),
                    "event_dur_mean": mean_event_dur,
                    "phase1_freqs": phase1_freqs,
                    "phase1_psd": phase1_psd,
                    "phase1_sp_freqs": phase1_sp_freqs,
                    "phase1_ap_fit_log10": phase1_ap,
                    "delta_freqs": freqs,
                    "delta_psd": psd_delta.power,
                    "analytic_delta": analytic_delta,
                    "analytic_rect": analytic_rect,
                    "delta_psd_norm": _normalize_unit_area(freqs, psd_delta.power),
                    "analytic_delta_norm": _normalize_unit_area(freqs, analytic_delta),
                    "analytic_rect_norm": _normalize_unit_area(freqs, analytic_rect),
                }

                deadtime_results[key] = compute_soz_stratified_deadtime(
                    subject_dir=sub_dir,
                    dataset=dataset,
                    soz_channels=soz_map.get(sub, []),
                    block_ranges=block_ranges,
                )
            except Exception as e:
                logger.error(f"  {key}: FAILED — {e}", exc_info=True)
                psd_results[key] = {"error": str(e)}
                deadtime_results[key] = {"error": str(e)}

    _save(psd_results, out_dir / "exp6_renewal_psd.json")
    _save(deadtime_results, out_dir / "exp6_soz_deadtime.json")
    return psd_results, deadtime_results


# ==========================================================================
# Experiment 7: Serial-correlation deep dive
# ==========================================================================

def run_exp7(subjects: List[str], roots: Dict[str, Path], out_dir: Path):
    """Deep analysis of IEI serial correlation with block-aware pooling."""
    logger.info("=== Experiment 7: Serial-Correlation Deep Dive ===")
    soz_yq = _load_soz(SOZ_FILE)
    soz_epi = _load_soz(SOZ_FILE_EPI)
    all_results: Dict[str, Any] = {}

    max_lag = 100
    min_pairs = 50
    detrend_window_sec = 600.0

    for dataset, root in roots.items():
        soz_map = soz_yq if dataset == "yuquan" else soz_epi
        loader = load_yuquan_subject_events if dataset == "yuquan" else load_epilepsiae_subject_events

        for sub in subjects:
            sub_dir = root / sub if dataset == "yuquan" else root / sub / "all_recs"
            if not sub_dir.exists():
                continue
            if not list(sub_dir.glob("*_lagPat.npz")):
                continue

            key = f"{dataset}/{sub}"
            logger.info(f"  {key}: serial-correlation deep dive")
            try:
                _, packed, _, block_ranges = loader(sub_dir)
                if len(packed) < 10:
                    all_results[key] = {"warning": "insufficient_group_events"}
                    continue

                iei_all = compute_iei(packed, block_ranges=block_ranges)
                event_blocks = _split_events_by_block(packed, block_ranges)
                iei_blocks = []
                start_blocks = []
                for block in event_blocks:
                    iei_block = compute_iei(block)
                    if iei_block.size < 3:
                        continue
                    iei_blocks.append(iei_block)
                    start_blocks.append(block[:, 0])

                decay = _serial_corr_decay_from_sequences(
                    iei_blocks,
                    max_lag=max_lag,
                    min_pairs=min_pairs,
                )
                lag1_r = float(decay["rs"][0]) if len(decay["rs"]) else np.nan
                within_block_iei = np.concatenate(iei_blocks) if iei_blocks else np.array([])
                median_iei = float(np.median(within_block_iei)) if within_block_iei.size else np.nan
                half_life_lag = float(decay["half_life_lag"]) if np.isfinite(decay["half_life_lag"]) else np.nan
                half_life_sec = float(half_life_lag * median_iei) if np.isfinite(half_life_lag) and np.isfinite(median_iei) else np.nan

                raw_x = []
                raw_y = []
                det_x = []
                det_y = []
                n_valid_intervals = 0
                for starts, iei_block in zip(start_blocks, iei_blocks):
                    x_block, y_block = _log_iei_pairs(iei_block, lag=1)
                    if x_block.size:
                        raw_x.append(x_block)
                        raw_y.append(y_block)

                    rolled = _rolling_log_iei_residuals(
                        starts,
                        iei_block,
                        window_sec=detrend_window_sec,
                    )
                    resid = rolled["residual"]
                    valid_pair = rolled["valid_pair"]
                    if resid.size and np.any(valid_pair):
                        det_x.append(resid[:-1][valid_pair])
                        det_y.append(resid[1:][valid_pair])
                    n_valid_intervals += int(np.sum(rolled["valid_interval"]))

                raw_r, _ = _pearson_corr(
                    np.concatenate(raw_x) if raw_x else np.array([]),
                    np.concatenate(raw_y) if raw_y else np.array([]),
                )
                detrended_r, _ = _pearson_corr(
                    np.concatenate(det_x) if det_x else np.array([]),
                    np.concatenate(det_y) if det_y else np.array([]),
                )
                detrend_fraction = np.nan
                if np.isfinite(raw_r) and abs(raw_r) > 1e-12 and np.isfinite(detrended_r):
                    detrend_fraction = 1.0 - (detrended_r / raw_r)

                within_block = compute_within_block_serial_corr(packed, block_ranges)
                soz_summary = compute_serial_corr_soz_stratified(
                    subject_dir=sub_dir,
                    dataset=dataset,
                    soz_channels=soz_map.get(sub, []),
                    block_ranges=block_ranges,
                )

                all_results[key] = {
                    "dataset": dataset,
                    "subject": sub,
                    "n_events": int(packed.shape[0]),
                    "n_iei": int(iei_all.size),
                    "median_iei": median_iei,
                    "serial_decay": {
                        "max_lag": max_lag,
                        "min_pairs": min_pairs,
                        "lags": decay["lags"],
                        "rs": decay["rs"],
                        "n_pairs": decay["n_pairs"],
                        "lag1_r": lag1_r,
                        "half_life_lag": half_life_lag,
                        "half_life_sec": half_life_sec,
                    },
                    "detrended": {
                        "window_sec": detrend_window_sec,
                        "raw_r": raw_r,
                        "detrended_r": detrended_r,
                        "detrend_fraction": detrend_fraction,
                        "n_valid_intervals": n_valid_intervals,
                        "n_valid_pairs": int(sum(arr.size for arr in det_x)),
                    },
                    "within_block": within_block,
                    "soz_stratified": soz_summary,
                }
                logger.info(
                    f"  {key}: lag1={lag1_r:.3f}, detrended={detrended_r:.3f}, "
                    f"half-life={half_life_sec:.1f}s"
                )
            except Exception as e:
                logger.error(f"  {key}: FAILED — {e}", exc_info=True)
                all_results[key] = {"error": str(e)}

    _save(all_results, out_dir / "exp7_serial_corr_deep.json")
    return all_results


# ==========================================================================
# Experiment 7B: Multi-scale modulation anatomy (PR-2.5)
# ==========================================================================

ESCAPE_SUBJECTS = ["1084", "1096"]


def run_exp7b(subjects: List[str], roots: Dict[str, Path], out_dir: Path):
    """PR-2.5: multi-scale detrending, n_participating autocorr, day/night, backfill."""
    logger.info("=== Experiment 7B: Multi-Scale Modulation Anatomy (PR-2.5) ===")

    multiscale_results: Dict[str, Any] = {}
    npart_results: Dict[str, Any] = {}
    daynight_results: Dict[str, Any] = {}
    merge_results: Dict[str, Any] = {}
    backfill_results: Dict[str, Any] = {}

    windows = [60, 180, 600, 1800, 3600, 7200]

    for dataset, root in roots.items():
        loader = (load_yuquan_subject_events if dataset == "yuquan"
                  else load_epilepsiae_subject_events)
        for sub in subjects:
            sub_dir = root / sub if dataset == "yuquan" else root / sub / "all_recs"
            if not sub_dir.exists():
                continue
            if not list(sub_dir.glob("*_lagPat.npz")):
                continue

            key = f"{dataset}/{sub}"
            logger.info(f"  {key}: PR-2.5 multi-scale analysis")
            t0 = time.time()

            try:
                _, packed, _, block_ranges = loader(sub_dir)
                if len(packed) < 10:
                    logger.warning(f"  {key}: SKIP (insufficient events)")
                    continue

                iei = compute_iei(packed, block_ranges=block_ranges)
                if iei.size < 10:
                    logger.warning(f"  {key}: SKIP (insufficient IEI)")
                    continue

                starts = packed[:, 0]
                event_times = starts if starts.size == iei.size + 1 else starts[:iei.size]

                # --- 7B: multi-scale detrending ---
                ms = compute_multiscale_detrend_fraction(
                    event_times, iei, windows=windows)
                ms["dataset"] = dataset
                ms["subject"] = sub
                ms["n_iei"] = int(iei.size)
                ms["median_iei"] = float(np.median(iei))
                multiscale_results[key] = ms
                logger.info(
                    f"    7B multi-scale: fractions="
                    f"{[pw['detrend_fraction'] for pw in ms['per_window']]}"
                )

                # --- 7C: n_participating autocorrelation ---
                npart = compute_nparticipating_autocorrelation(
                    sub_dir, dataset, block_ranges, max_lag=100, min_pairs=50)
                npart["dataset"] = dataset
                npart["subject"] = sub
                npart_results[key] = npart
                logger.info(
                    f"    7C n_part: lag1_r={npart.get('lag1_r', 'N/A')}, "
                    f"half_life={npart.get('half_life_lag', 'N/A')}"
                )

                # --- 7D: day/night stratified detrending ---
                dn = compute_daynight_stratified_detrending(
                    event_times, iei, dataset=dataset, window_sec=600.0)
                dn["dataset"] = dataset
                dn["subject"] = sub
                daynight_results[key] = dn
                day_r = dn["day"].get("detrended_r", np.nan)
                night_r = dn["night"].get("detrended_r", np.nan)
                logger.info(
                    f"    7D day/night: day_det_r={day_r:.3f}, "
                    f"night_det_r={night_r:.3f}"
                    if np.isfinite(day_r) and np.isfinite(night_r)
                    else f"    7D day/night: day_n={dn['n_day']}, night_n={dn['n_night']}"
                )

                # --- 7E: block merge sensitivity ---
                merged_blocks = merge_contiguous_blocks(block_ranges, max_gap_sec=5.0)
                event_blocks_merged = []
                for block in _split_events_by_block(packed, merged_blocks):
                    iei_b = compute_iei(block)
                    if iei_b.size >= 3:
                        event_blocks_merged.append(iei_b)
                if event_blocks_merged:
                    from src.event_periodicity import _serial_corr_decay_from_sequences
                    decay_merged = _serial_corr_decay_from_sequences(
                        event_blocks_merged, max_lag=100, min_pairs=50)
                    iei_merged = np.concatenate(event_blocks_merged)
                    med_iei_m = float(np.median(iei_merged)) if iei_merged.size else np.nan
                    hl_lag_m = float(decay_merged["half_life_lag"])
                    hl_sec_m = hl_lag_m * med_iei_m if np.isfinite(hl_lag_m) else np.nan
                    merge_results[key] = {
                        "dataset": dataset, "subject": sub,
                        "n_blocks_original": len(block_ranges),
                        "n_blocks_merged": len(merged_blocks),
                        "half_life_lag_merged": hl_lag_m,
                        "half_life_sec_merged": hl_sec_m,
                        "lag1_r_merged": float(decay_merged["rs"][0])
                            if decay_merged["rs"].size else np.nan,
                    }
                    logger.info(
                        f"    7E merge: {len(block_ranges)} → {len(merged_blocks)} blocks, "
                        f"half_life={hl_sec_m:.1f}s"
                    )

                # --- 7F: backfill escape subjects ---
                if sub in ESCAPE_SUBJECTS:
                    n_surr = 50 if len(packed) > 50000 else 200
                    logger.info(f"    7F backfill: {sub} (escape, n_surr={n_surr})")
                    bf = compute_detrended_psd_backfill(
                        packed, block_ranges, window_sec=600.0,
                        fs=100.0, nperseg_sec=500.0, n_surrogates=n_surr)
                    bf["dataset"] = dataset
                    bf["subject"] = sub
                    backfill_results[key] = bf
                    logger.info(
                        f"    7F: raw_peak={bf['raw']['peak_freq']:.2f}Hz "
                        f"(gamma_p={bf['raw']['gamma_p']:.3f}), "
                        f"det_peak={bf['detrended']['peak_freq']:.2f}Hz "
                        f"(gamma_p={bf['detrended']['gamma_p']:.3f}), "
                        f"disappeared={bf['peak_disappeared']}"
                    )

                logger.info(f"  {key}: done ({time.time()-t0:.1f}s)")
            except Exception as e:
                logger.error(f"  {key}: FAILED — {e}", exc_info=True)
                multiscale_results[key] = {"error": str(e)}

    # Cross-correlation of IEI decay vs n_participating decay
    logger.info("  Computing IEI vs n_participating decay cross-correlations...")
    exp7_path = out_dir / "exp7_serial_corr_deep.json"
    exp7_data = {}
    if exp7_path.exists():
        with open(exp7_path) as f:
            exp7_data = json.load(f)

    for key in npart_results:
        npart_rec = npart_results[key]
        exp7_rec = exp7_data.get(key, {})
        iei_decay = exp7_rec.get("serial_decay", {})
        iei_rs = np.asarray(iei_decay.get("rs", []), dtype=float)
        npart_rs = np.asarray(npart_rec.get("rs", []), dtype=float)

        min_len = min(iei_rs.size, npart_rs.size)
        if min_len >= 5:
            from scipy.stats import pearsonr
            try:
                cross_r, cross_p = pearsonr(iei_rs[:min_len], npart_rs[:min_len])
            except Exception:
                cross_r, cross_p = np.nan, np.nan
            npart_rec["cross_decay_r"] = float(cross_r) if np.isfinite(cross_r) else np.nan
            npart_rec["cross_decay_p"] = float(cross_p) if np.isfinite(cross_p) else np.nan
            npart_rec["iei_half_life_lag"] = float(iei_decay.get("half_life_lag", np.nan))
            npart_rec["iei_half_life_sec"] = float(iei_decay.get("half_life_sec", np.nan))
            logger.info(
                f"    {key}: cross_decay_r={cross_r:.3f}"
            )
        else:
            npart_rec["cross_decay_r"] = np.nan
            npart_rec["cross_decay_p"] = np.nan

    _save(multiscale_results, out_dir / "exp7b_multiscale.json")
    _save(npart_results, out_dir / "exp7b_npart_autocorr.json")
    _save(daynight_results, out_dir / "exp7b_daynight.json")
    _save(merge_results, out_dir / "exp7b_merge_sensitivity.json")
    if backfill_results:
        _save(backfill_results, out_dir / "exp7b_backfill.json")

    return {
        "multiscale": multiscale_results,
        "npart": npart_results,
        "daynight": daynight_results,
        "merge": merge_results,
        "backfill": backfill_results,
    }


# ==========================================================================
# Experiment 7C: Continuous long-timescale modulation (PR-2.6)
# ==========================================================================

def run_exp7c(subjects: List[str], roots: Dict[str, Path], out_dir: Path):
    """PR-2.6: true continuous-time long-timescale analysis."""
    logger.info("=== Experiment 7C: Continuous Long-Timescale Modulation (PR-2.6) ===")
    all_results: Dict[str, Any] = {}

    for dataset, root in roots.items():
        loader = load_yuquan_subject_events if dataset == "yuquan" else load_epilepsiae_subject_events
        for sub in subjects:
            sub_dir = root / sub if dataset == "yuquan" else root / sub / "all_recs"
            if not sub_dir.exists():
                continue
            if not list(sub_dir.glob("*_lagPat.npz")):
                continue

            key = f"{dataset}/{sub}"
            logger.info(f"  {key}: PR-2.6 continuous long-timescale analysis")
            t0 = time.time()
            try:
                _, packed, _, block_ranges = loader(sub_dir)
                if len(packed) < 10:
                    all_results[key] = {"warning": "insufficient_group_events"}
                    continue

                rate_summary = compute_long_timescale_rate_summary(
                    events=packed,
                    block_ranges=block_ranges,
                    bin_sec=300.0,
                    smooth_windows_sec=(1800, 3600, 7200, 14400, 28800),
                    autocorr_lag_hours=(0.5, 1.0, 2.0, 4.0, 8.0),
                )
                contiguous_dn = compute_contiguous_daynight_detrending(
                    events=packed,
                    block_ranges=block_ranges,
                    dataset=dataset,
                    window_sec=600.0,
                    merge_gap_sec=5.0,
                    min_segment_iei=20,
                )

                all_results[key] = {
                    "dataset": dataset,
                    "subject": sub,
                    "n_events": int(packed.shape[0]),
                    "long_timescale": rate_summary,
                    "contiguous_daynight": contiguous_dn,
                }

                continuity = rate_summary.get("continuity", {})
                longest_run = continuity.get("longest_run_hours", np.nan)
                total_obs = continuity.get("total_observed_hours", np.nan)
                day_r = contiguous_dn.get("day", {}).get("pooled_detrended_r", np.nan)
                night_r = contiguous_dn.get("night", {}).get("pooled_detrended_r", np.nan)
                logger.info(
                    f"    longest_run={longest_run:.2f}h, observed={total_obs:.2f}h, "
                    f"day_r={day_r:.3f}, night_r={night_r:.3f}"
                    if np.isfinite(longest_run) and np.isfinite(total_obs)
                    else "    continuity summary unavailable"
                )
                logger.info(f"  {key}: done ({time.time() - t0:.1f}s)")
            except Exception as e:
                logger.error(f"  {key}: FAILED — {e}", exc_info=True)
                all_results[key] = {"error": str(e)}

    _save(all_results, out_dir / "exp7c_long_timescale.json")
    return all_results


# ==========================================================================
# Main
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 2 periodicity experiments")
    parser.add_argument("--exp", default="all",
                        help="Experiment number(s): 1,2,3,4,5,6,7,7b,7c or 'all'")
    parser.add_argument("--dataset", choices=["yuquan", "epilepsiae", "both"],
                        default="both")
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="Quick test: chengshuai + 548")
    args = parser.parse_args()

    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.exp == "all":
        exps = {"1", "2", "3", "4", "5", "6", "7", "7b", "7c"}
    else:
        exps = {x.strip() for x in args.exp.split(",")}

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

    if "1" in exps:
        run_exp1(yq_subs, out_dir)

    if "2" in exps:
        run_exp2(all_subs, roots, out_dir)

    if "3" in exps:
        run_exp3(all_subs, roots, out_dir)

    if "4" in exps:
        run_exp4(all_subs, roots, out_dir)

    if "5" in exps:
        run_exp5(all_subs, roots, out_dir)

    if "6" in exps:
        run_exp6(all_subs, roots, out_dir)

    if "7" in exps:
        run_exp7(all_subs, roots, out_dir)

    if "7b" in exps:
        run_exp7b(all_subs, roots, out_dir)

    if "7c" in exps:
        run_exp7c(all_subs, roots, out_dir)

    logger.info(f"=== Phase 2 complete: {time.time()-t_total:.0f}s total ===")


if __name__ == "__main__":
    main()
