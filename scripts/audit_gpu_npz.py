#!/usr/bin/env python3
"""
Step 0 + Step 1 of Spatial Modulation PR-1: GPU detection audit + relaxed refine survey.

Iterates all subjects in both Yuquan and Epilepsiae datasets to:
  - Verify gpu.npz file availability and integrity
  - Count total channels, events, nonzero channels per block
  - Compare channel selection under different refine k values
  - Check SOZ channel coverage against gpu channel names
  - Output a combined audit CSV for go/no-go decision

Usage:
    python scripts/audit_gpu_npz.py
    python scripts/audit_gpu_npz.py --min-count 100 --min-rate 5.0

Outputs:
    results/spatial_modulation/gpu_audit.csv
    results/spatial_modulation/relaxed_refine_channel_counts.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.event_periodicity import _try_load_gpu
from src.group_event_analysis import select_core_channels_by_event_count

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("audit_gpu")

YUQUAN_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")
RESULTS_DIR = Path("results/spatial_modulation")
SOZ_FILE_YQ = Path("results/yuquan_soz_core_channels.json")
SOZ_FILE_EPI = Path("results/epilepsiae_soz_core_channels.json")
FOCUS_REL_FILE = Path("results/epilepsiae_electrode_focus_rel.json")

YUQUAN_SUBJECTS = [
    "zhangkexuan", "pengzihang", "chengshuai", "huangwanling",
    "liyouran", "songzishuo", "zhangbichen", "zhaochenxi",
    "zhaojinrui", "zhourongxuan", "zhangjiaqi",
    "chenziyang", "hanyuxuan", "huanghanwen", "litengsheng",
    "xuxinyi", "zhangjinhan", "sunyuanxin",
]

# dongyiming, gaolan, wangyiyang excluded: no lagPat / not in standard analysis set

EPILEPSIAE_SUBJECTS = [
    "1096", "1084", "958", "922", "590", "1150", "442", "1073",
    "253", "1146", "916", "620", "583", "548", "384", "139",
    "1125", "1077", "818", "635",
]

REFINE_K_VALUES = [1.0, 0.5, 0.0, -0.5]
BLOCK_DUR_YUQUAN = 2 * 3600.0
BLOCK_DUR_EPILEPSIAE = 3600.0


def _load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _normalize_channel_name(name: str) -> str:
    """Strip whitespace, uppercase, remove known prefixes."""
    s = name.strip().upper()
    for prefix in ("EEG ", "EEG_"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s


def _build_soz_set(soz_channels: List[str]) -> Set[str]:
    return {_normalize_channel_name(c) for c in soz_channels}


def _match_bipolar_soz(ch_name: str, soz_set: Set[str]) -> str:
    """Match bipolar channel against SOZ set: any contact in SOZ -> 'soz'."""
    normalized = _normalize_channel_name(ch_name)
    parts = normalized.split("-")
    for p in parts:
        p = p.strip()
        if p in soz_set:
            return "soz"
    return "non_soz"


def _match_bipolar_focus_rel(
    ch_name: str, focus_rel: Dict[str, List[str]]
) -> str:
    """Match bipolar channel to Epilepsiae i/l/e: priority i > l > e."""
    normalized = _normalize_channel_name(ch_name)
    parts = [p.strip() for p in normalized.split("-")]
    for label in ("i", "l", "e"):
        label_set = {_normalize_channel_name(c) for c in focus_rel.get(label, [])}
        for p in parts:
            if p in label_set:
                return label
    return "unknown"


def _get_lagpat_channels(subject_dir: Path) -> List[str]:
    """Read chnNames from the first lagPat.npz if available."""
    lagpat_files = sorted(subject_dir.glob("*_lagPat.npz"))
    if not lagpat_files:
        return []
    try:
        tmp = np.load(lagpat_files[0], allow_pickle=True)
        return list(tmp["chnNames"])
    except Exception:
        return []


def audit_subject_yuquan(
    subject: str,
    soz_channels: List[str],
    min_count: int,
    min_rate: float,
) -> Dict[str, Any]:
    subject_dir = YUQUAN_ROOT / subject
    edf_files = sorted(subject_dir.glob("*.edf"))

    lagpat_chns = _get_lagpat_channels(subject_dir)
    soz_set = _build_soz_set(soz_channels)

    total_blocks = len(edf_files)
    valid_blocks = 0
    total_hours = 0.0

    # Accumulate cross-block events_count per channel
    all_ch_names: List[str] = []
    ch_name_set: set = set()
    sum_events: Dict[str, int] = {}

    for edf in edf_files:
        gpu_file = edf.with_name(edf.stem + "_gpu.npz")
        if not gpu_file.exists():
            continue

        gpu = _try_load_gpu(gpu_file)
        if gpu is None:
            continue

        valid_blocks += 1
        total_hours += BLOCK_DUR_YUQUAN / 3600.0

        try:
            chns = list(gpu["chns_names"])
            events_count = np.array(gpu["events_count"], dtype=int).ravel()
        except Exception:
            continue

        for i, ch in enumerate(chns):
            if ch not in ch_name_set:
                ch_name_set.add(ch)
                all_ch_names.append(ch)
            cnt = int(events_count[i]) if i < len(events_count) else 0
            sum_events[ch] = sum_events.get(ch, 0) + cnt

    if not all_ch_names:
        return _empty_row(subject, "yuquan", total_blocks)

    ch_names_arr = all_ch_names
    counts_arr = np.array([sum_events.get(c, 0) for c in ch_names_arr], dtype=float)

    nonzero_total = int(np.sum(counts_arr > 0))

    # Refine at different k values + double threshold
    k_results = {}
    for k in REFINE_K_VALUES:
        selected = select_core_channels_by_event_count(
            events_count=counts_arr,
            ch_names=ch_names_arr,
            method="mean_std",
            k=k,
            min_count=1,
        )
        # Apply double threshold
        if total_hours > 0:
            selected = [
                ch for ch in selected
                if sum_events.get(ch, 0) >= min_count
                and (sum_events.get(ch, 0) / total_hours) >= min_rate
            ]
        k_results[k] = selected

    # SOZ matching for k=0.0 selection (primary candidate)
    primary_chns = k_results.get(0.0, [])
    soz_matched = [ch for ch in primary_chns if _match_bipolar_soz(ch, soz_set) == "soz"]
    nonsoz_matched = [ch for ch in primary_chns if _match_bipolar_soz(ch, soz_set) == "non_soz"]

    return {
        "subject": subject,
        "dataset": "yuquan",
        "total_blocks": total_blocks,
        "valid_blocks": valid_blocks,
        "total_hours": round(total_hours, 1),
        "all_channels": len(ch_names_arr),
        "nonzero_channels": nonzero_total,
        "lagpat_channels": len(lagpat_chns),
        "k1.0_channels": len(k_results.get(1.0, [])),
        "k0.5_channels": len(k_results.get(0.5, [])),
        "k0.0_channels": len(k_results.get(0.0, [])),
        "k-0.5_channels": len(k_results.get(-0.5, [])),
        "soz_defined": len(soz_channels) > 0,
        "soz_in_primary": len(soz_matched),
        "nonsoz_in_primary": len(nonsoz_matched),
        "soz_coverage": f"{len(soz_matched)}/{len(primary_chns)}" if primary_chns else "0/0",
        "has_valid_pair": len(soz_matched) >= 3 and len(nonsoz_matched) >= 3,
    }


def audit_subject_epilepsiae(
    subject: str,
    soz_channels: List[str],
    focus_rel: Dict[str, List[str]],
    min_count: int,
    min_rate: float,
) -> Dict[str, Any]:
    subject_dir = EPILEPSIAE_ROOT / subject / "all_recs"
    if not subject_dir.exists():
        return _empty_row(subject, "epilepsiae", 0)

    lagpat_chns = _get_lagpat_channels(subject_dir)

    lagpat_files = sorted(subject_dir.glob("*_lagPat.npz"))
    gpu_files = sorted(subject_dir.glob("*_gpu.npz"))

    soz_set = _build_soz_set(soz_channels)

    total_blocks = max(len(lagpat_files), len(gpu_files))
    valid_blocks = 0
    total_hours = 0.0

    all_ch_names: List[str] = []
    ch_name_set: set = set()
    sum_events: Dict[str, int] = {}

    for gpu_file in gpu_files:
        gpu = _try_load_gpu(gpu_file)
        if gpu is None:
            continue

        valid_blocks += 1
        total_hours += BLOCK_DUR_EPILEPSIAE / 3600.0

        try:
            chns = list(gpu["chns_names"])
            events_count = np.array(gpu["events_count"], dtype=int).ravel()
        except Exception:
            continue

        for i, ch in enumerate(chns):
            if ch not in ch_name_set:
                ch_name_set.add(ch)
                all_ch_names.append(ch)
            cnt = int(events_count[i]) if i < len(events_count) else 0
            sum_events[ch] = sum_events.get(ch, 0) + cnt

    if not all_ch_names:
        return _empty_row(subject, "epilepsiae", total_blocks)

    ch_names_arr = all_ch_names
    counts_arr = np.array([sum_events.get(c, 0) for c in ch_names_arr], dtype=float)
    nonzero_total = int(np.sum(counts_arr > 0))

    k_results = {}
    for k in REFINE_K_VALUES:
        selected = select_core_channels_by_event_count(
            events_count=counts_arr,
            ch_names=ch_names_arr,
            method="mean_std",
            k=k,
            min_count=1,
        )
        if total_hours > 0:
            selected = [
                ch for ch in selected
                if sum_events.get(ch, 0) >= min_count
                and (sum_events.get(ch, 0) / total_hours) >= min_rate
            ]
        k_results[k] = selected

    primary_chns = k_results.get(0.0, [])
    soz_matched = [ch for ch in primary_chns if _match_bipolar_soz(ch, soz_set) == "soz"]
    nonsoz_matched = [ch for ch in primary_chns if _match_bipolar_soz(ch, soz_set) == "non_soz"]

    # Focus rel matching for primary channels
    focus_counts = {"i": 0, "l": 0, "e": 0, "unknown": 0}
    for ch in primary_chns:
        label = _match_bipolar_focus_rel(ch, focus_rel)
        focus_counts[label] = focus_counts.get(label, 0) + 1

    return {
        "subject": subject,
        "dataset": "epilepsiae",
        "total_blocks": total_blocks,
        "valid_blocks": valid_blocks,
        "total_hours": round(total_hours, 1),
        "all_channels": len(ch_names_arr),
        "nonzero_channels": nonzero_total,
        "lagpat_channels": len(lagpat_chns),
        "k1.0_channels": len(k_results.get(1.0, [])),
        "k0.5_channels": len(k_results.get(0.5, [])),
        "k0.0_channels": len(k_results.get(0.0, [])),
        "k-0.5_channels": len(k_results.get(-0.5, [])),
        "soz_defined": len(soz_channels) > 0,
        "soz_in_primary": len(soz_matched),
        "nonsoz_in_primary": len(nonsoz_matched),
        "soz_coverage": f"{len(soz_matched)}/{len(primary_chns)}" if primary_chns else "0/0",
        "has_valid_pair": len(soz_matched) >= 3 and len(nonsoz_matched) >= 3,
        "focus_i": focus_counts["i"],
        "focus_l": focus_counts["l"],
        "focus_e": focus_counts["e"],
        "focus_unknown": focus_counts["unknown"],
    }


def _empty_row(subject: str, dataset: str, total_blocks: int) -> Dict[str, Any]:
    return {
        "subject": subject,
        "dataset": dataset,
        "total_blocks": total_blocks,
        "valid_blocks": 0,
        "total_hours": 0.0,
        "all_channels": 0,
        "nonzero_channels": 0,
        "lagpat_channels": 0,
        "k1.0_channels": 0,
        "k0.5_channels": 0,
        "k0.0_channels": 0,
        "k-0.5_channels": 0,
        "soz_defined": False,
        "soz_in_primary": 0,
        "nonsoz_in_primary": 0,
        "soz_coverage": "0/0",
        "has_valid_pair": False,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit gpu.npz availability for spatial modulation PR-1")
    parser.add_argument("--min-count", type=int, default=100,
                        help="Minimum total event count per channel (default: 100)")
    parser.add_argument("--min-rate", type=float, default=5.0,
                        help="Minimum event rate in events/hour (default: 5.0)")
    args = parser.parse_args()

    soz_yq = _load_json(SOZ_FILE_YQ)
    soz_epi = _load_json(SOZ_FILE_EPI)
    focus_rel_all = _load_json(FOCUS_REL_FILE)

    rows = []

    logger.info("=== Auditing Yuquan subjects ===")
    for subj in YUQUAN_SUBJECTS:
        logger.info(f"  {subj}...")
        soz_chns = soz_yq.get(subj, [])
        row = audit_subject_yuquan(subj, soz_chns, args.min_count, args.min_rate)
        rows.append(row)
        logger.info(
            f"    blocks={row['valid_blocks']}/{row['total_blocks']} "
            f"all_ch={row['all_channels']} "
            f"k0.0={row['k0.0_channels']} "
            f"soz={row['soz_in_primary']} nonsoz={row['nonsoz_in_primary']}"
        )

    logger.info("=== Auditing Epilepsiae subjects ===")
    for subj in EPILEPSIAE_SUBJECTS:
        logger.info(f"  {subj}...")
        soz_chns = soz_epi.get(subj, [])
        fr = focus_rel_all.get(subj, {})
        row = audit_subject_epilepsiae(subj, soz_chns, fr, args.min_count, args.min_rate)
        rows.append(row)
        logger.info(
            f"    blocks={row['valid_blocks']}/{row['total_blocks']} "
            f"all_ch={row['all_channels']} "
            f"k0.0={row['k0.0_channels']} "
            f"soz={row['soz_in_primary']} nonsoz={row['nonsoz_in_primary']}"
        )

    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "gpu_audit.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {out_path}")

    # Summary statistics
    logger.info("\n=== AUDIT SUMMARY ===")
    yq = df[df["dataset"] == "yuquan"]
    ep = df[df["dataset"] == "epilepsiae"]

    yq_valid = yq[yq["valid_blocks"] > 0]
    ep_valid = ep[ep["valid_blocks"] > 0]
    logger.info(f"Yuquan:     {len(yq_valid)}/{len(yq)} subjects with valid gpu.npz")
    logger.info(f"Epilepsiae: {len(ep_valid)}/{len(ep)} subjects with valid gpu.npz")

    for k in REFINE_K_VALUES:
        col = f"k{k}_channels"
        yq_med = yq_valid[col].median() if len(yq_valid) > 0 else 0
        ep_med = ep_valid[col].median() if len(ep_valid) > 0 else 0
        logger.info(f"  k={k:+.1f}  median channels: Yuquan={yq_med:.0f}  Epilepsiae={ep_med:.0f}")

    valid_pair = df[df["has_valid_pair"]]
    logger.info(f"Subjects with SOZ+nonSOZ ≥3 each (k=0.0): {len(valid_pair)}/{len(df)}")

    # Pass/fail
    yq_pass = len(yq_valid) >= 8
    ep_pass = len(ep_valid) >= 10
    pair_pass = len(valid_pair) >= 15
    all_pass = yq_pass and ep_pass and pair_pass
    logger.info(f"\nPass criteria:")
    logger.info(f"  Yuquan  ≥8  valid:  {'PASS' if yq_pass else 'FAIL'} ({len(yq_valid)})")
    logger.info(f"  Epilepsiae ≥10 valid: {'PASS' if ep_pass else 'FAIL'} ({len(ep_valid)})")
    logger.info(f"  Valid pairs ≥15:     {'PASS' if pair_pass else 'FAIL'} ({len(valid_pair)})")
    logger.info(f"  Overall: {'GO' if all_pass else 'NO-GO — review before proceeding'}")

    # Also save a relaxed refine channel count comparison
    refine_rows = []
    for _, row in df.iterrows():
        for k in REFINE_K_VALUES:
            refine_rows.append({
                "subject": row["subject"],
                "dataset": row["dataset"],
                "refine_k": k,
                "n_channels": row[f"k{k}_channels"],
                "lagpat_channels": row["lagpat_channels"],
            })
    refine_df = pd.DataFrame(refine_rows)
    refine_path = RESULTS_DIR / "relaxed_refine_channel_counts.csv"
    refine_df.to_csv(refine_path, index=False)
    logger.info(f"Saved {refine_path}")


if __name__ == "__main__":
    main()
