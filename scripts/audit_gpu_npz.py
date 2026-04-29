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
EPILEPSIAE_NEW_GPU_ROOT = Path("results/hfo_detection")
RESULTS_DIR = Path("results/spatial_modulation")
SOZ_FILE_YQ = Path("results/yuquan_soz_core_channels.json")
SOZ_FILE_EPI = Path("results/epilepsiae_soz_core_channels.json")
FOCUS_REL_FILE = Path("results/epilepsiae_electrode_focus_rel.json")

# Legacy lagPat files must contain these keys to be considered "complete"
LAGPAT_REQUIRED_KEYS = ("lagPatRaw", "lagPatRank", "eventsBool", "chnNames", "start_t")
# New pipeline gpu_npz must contain these keys for Stage 2 to construct block_ranges + per-channel events
NEW_GPU_REQUIRED_KEYS = ("whole_dets", "chns_names", "events_count", "start_time")
# Stub heuristic: legacy 216B stubs fail the 500B threshold inside _try_load_gpu
STUB_SIZE_THRESHOLD = 500

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


def _inspect_gpu_file(path: Path) -> Dict[str, Any]:
    """Inspect a *_gpu.npz file and report file-size + schema details.

    Used by the artifact census to distinguish between (legacy 216B stub)
    vs (loadable but missing keys) vs (fully OK). Does NOT load whole_dets
    fully — only checks key presence to keep the audit fast.
    """
    info: Dict[str, Any] = {
        "exists": False,
        "size_bytes": 0,
        "is_stub": False,
        "loadable": False,
        "keys": [],
    }
    if not path.exists():
        return info
    info["exists"] = True
    try:
        info["size_bytes"] = path.stat().st_size
    except OSError:
        info["size_bytes"] = 0
    if info["size_bytes"] < STUB_SIZE_THRESHOLD:
        info["is_stub"] = True
        return info
    try:
        with np.load(path, allow_pickle=True) as data:
            info["keys"] = list(data.keys())
        info["loadable"] = True
    except Exception:
        info["loadable"] = False
    return info


def _inspect_lagpat_file(path: Path) -> Dict[str, Any]:
    """Inspect a *_lagPat*.npz file and report key completeness."""
    info: Dict[str, Any] = {
        "exists": False,
        "loadable": False,
        "keys": [],
        "keys_complete": False,
    }
    if not path.exists():
        return info
    info["exists"] = True
    try:
        with np.load(path, allow_pickle=True) as data:
            info["keys"] = list(data.keys())
        info["loadable"] = True
        info["keys_complete"] = all(k in info["keys"] for k in LAGPAT_REQUIRED_KEYS)
    except Exception:
        pass
    return info


def _inspect_simple_npz(path: Path) -> Dict[str, Any]:
    """Light loadable + size check for sub_refineGpu.npz / packedTimes.npy."""
    info: Dict[str, Any] = {"exists": False, "size_bytes": 0, "loadable": False}
    if not path.exists():
        return info
    info["exists"] = True
    try:
        info["size_bytes"] = path.stat().st_size
    except OSError:
        info["size_bytes"] = 0
    try:
        with np.load(path, allow_pickle=True) as _:
            pass
        info["loadable"] = True
    except Exception:
        info["loadable"] = False
    return info


def _classify_legacy_verdict(
    legacy_refine_ok: bool,
    legacy_gpu_real_count: int,
    legacy_gpu_stub_count: int,
    legacy_lagpat_records: int,
    legacy_lagpat_keys_complete: bool,
) -> str:
    """Decide legacy_verdict ∈ {replay_eligible, lagpat_only, refine_only, corrupt_gpu, missing}.

    Track B replay needs three-piece set: refine + real gpu + lagPat with full keys.
    If legacy gpu is all stubs but lagPat exists → lagpat_only (cannot replay refine→pack).
    """
    if (
        legacy_refine_ok
        and legacy_gpu_real_count > 0
        and legacy_lagpat_records > 0
        and legacy_lagpat_keys_complete
    ):
        return "replay_eligible"
    if legacy_lagpat_records > 0 and legacy_gpu_stub_count > 0 and legacy_gpu_real_count == 0:
        return "corrupt_gpu"
    if legacy_lagpat_records > 0 and legacy_refine_ok:
        return "lagpat_only"
    if legacy_refine_ok and legacy_lagpat_records == 0:
        return "refine_only"
    return "missing"


def _classify_new_verdict(
    new_subject_dir_exists: bool,
    new_gpu_records_total: int,
    new_gpu_records_loadable: int,
    new_gpu_start_time_present_count: int,
    new_gpu_schema_complete_count: int,
    new_refine_ok: bool,
) -> str:
    """Decide new_verdict ∈ {ready, partial, broken_time_axis, missing}.

    Stage 2 (per-channel relaxed-refine) requires:
      - all gpu records loadable
      - all records have start_time (block_ranges construction)
      - all records have whole_dets/chns_names/events_count
      - _refineGpu.npz loadable
    """
    if not new_subject_dir_exists or new_gpu_records_total == 0:
        return "missing"
    if (
        new_gpu_records_loadable == new_gpu_records_total
        and new_gpu_start_time_present_count == new_gpu_records_total
        and new_gpu_schema_complete_count == new_gpu_records_total
        and new_refine_ok
    ):
        return "ready"
    if new_gpu_start_time_present_count < new_gpu_records_total:
        return "broken_time_axis"
    return "partial"


def audit_subject_artifact_census_epilepsiae(subject: str) -> Dict[str, Any]:
    """Per-subject deep artifact census: legacy + new pipeline → verdicts.

    Used by the Epilepsiae Topic 3 Stage 0 census (--include-pack-lag mode).
    Distinguishes legacy_verdict (Track B feasibility) from new_verdict
    (Stage 2 per-channel relaxed-refine feasibility).
    """
    # ---- Legacy side ----
    legacy_root = EPILEPSIAE_ROOT / subject / "all_recs"
    legacy_root_exists = legacy_root.exists()

    legacy_refine_info = _inspect_simple_npz(legacy_root / "sub_refineGpu.npz") if legacy_root_exists else {"exists": False, "loadable": False, "size_bytes": 0}
    legacy_refine_ok = bool(legacy_refine_info.get("loadable"))

    legacy_gpu_files = sorted(legacy_root.glob("*_gpu.npz")) if legacy_root_exists else []
    legacy_gpu_real_count = 0
    legacy_gpu_stub_count = 0
    legacy_gpu_unloadable_count = 0
    for f in legacy_gpu_files:
        info = _inspect_gpu_file(f)
        if info["is_stub"]:
            legacy_gpu_stub_count += 1
        elif info["loadable"]:
            legacy_gpu_real_count += 1
        else:
            legacy_gpu_unloadable_count += 1

    legacy_lagpat_files = sorted(legacy_root.glob("*_lagPat.npz")) if legacy_root_exists else []
    legacy_lagpat_with_freq_files = sorted(legacy_root.glob("*_lagPat_withFreqCent.npz")) if legacy_root_exists else []
    legacy_packedtimes_files = sorted(legacy_root.glob("*_packedTimes.npy")) if legacy_root_exists else []

    legacy_lagpat_keys_complete = False
    legacy_lagpat_keys_first: List[str] = []
    if legacy_lagpat_files:
        first_info = _inspect_lagpat_file(legacy_lagpat_files[0])
        legacy_lagpat_keys_complete = bool(first_info.get("keys_complete"))
        legacy_lagpat_keys_first = first_info.get("keys", [])

    legacy_verdict = _classify_legacy_verdict(
        legacy_refine_ok=legacy_refine_ok,
        legacy_gpu_real_count=legacy_gpu_real_count,
        legacy_gpu_stub_count=legacy_gpu_stub_count,
        legacy_lagpat_records=len(legacy_lagpat_files),
        legacy_lagpat_keys_complete=legacy_lagpat_keys_complete,
    )

    # ---- New pipeline side ----
    new_root = EPILEPSIAE_NEW_GPU_ROOT / subject
    new_root_exists = new_root.exists()

    new_gpu_files = sorted(new_root.glob("*_gpu.npz")) if new_root_exists else []
    new_gpu_records_total = len(new_gpu_files)
    new_gpu_records_loadable = 0
    new_gpu_start_time_present_count = 0
    new_gpu_schema_complete_count = 0
    for f in new_gpu_files:
        info = _inspect_gpu_file(f)
        if not info["loadable"]:
            continue
        new_gpu_records_loadable += 1
        keys = set(info["keys"])
        if "start_time" in keys:
            new_gpu_start_time_present_count += 1
        if all(k in keys for k in NEW_GPU_REQUIRED_KEYS):
            new_gpu_schema_complete_count += 1

    new_refine_info = _inspect_simple_npz(new_root / "_refineGpu.npz") if new_root_exists else {"loadable": False, "size_bytes": 0}
    new_refine_ok = bool(new_refine_info.get("loadable"))

    new_verdict = _classify_new_verdict(
        new_subject_dir_exists=new_root_exists,
        new_gpu_records_total=new_gpu_records_total,
        new_gpu_records_loadable=new_gpu_records_loadable,
        new_gpu_start_time_present_count=new_gpu_start_time_present_count,
        new_gpu_schema_complete_count=new_gpu_schema_complete_count,
        new_refine_ok=new_refine_ok,
    )

    # Coverage: does new pipeline cover the legacy record set?
    legacy_record_stems = {f.stem.replace("_gpu", "") for f in legacy_gpu_files}
    new_record_stems = {f.stem.replace("_gpu", "") for f in new_gpu_files}
    new_covers_legacy = legacy_record_stems.issubset(new_record_stems) if legacy_record_stems else False

    return {
        "subject": subject,
        "dataset": "epilepsiae",
        # Legacy side
        "legacy_root_exists": legacy_root_exists,
        "legacy_refine_ok": legacy_refine_ok,
        "legacy_refine_size_bytes": legacy_refine_info.get("size_bytes", 0),
        "legacy_gpu_records_total": len(legacy_gpu_files),
        "legacy_gpu_real_count": legacy_gpu_real_count,
        "legacy_gpu_stub_count": legacy_gpu_stub_count,
        "legacy_gpu_unloadable_count": legacy_gpu_unloadable_count,
        "legacy_lagpat_records": len(legacy_lagpat_files),
        "legacy_lagpat_withfreq_records": len(legacy_lagpat_with_freq_files),
        "legacy_lagpat_keys_complete": legacy_lagpat_keys_complete,
        "legacy_lagpat_keys_first_record": ";".join(legacy_lagpat_keys_first),
        "legacy_packedtimes_records": len(legacy_packedtimes_files),
        "legacy_verdict": legacy_verdict,
        # New pipeline side
        "new_root_exists": new_root_exists,
        "new_gpu_records_total": new_gpu_records_total,
        "new_gpu_records_loadable": new_gpu_records_loadable,
        "new_gpu_start_time_present_count": new_gpu_start_time_present_count,
        "new_gpu_schema_complete_count": new_gpu_schema_complete_count,
        "new_refine_ok": new_refine_ok,
        "new_covers_legacy_records": new_covers_legacy,
        "new_verdict": new_verdict,
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


def _run_artifact_census_epilepsiae() -> None:
    """Stage 0 deep census for Epilepsiae Topic 3 prep.

    Writes a per-subject CSV with both legacy_verdict (Track B feasibility)
    and new_verdict (Stage 2 per-channel feasibility), so future agents can
    decide path-of-action without re-scanning the disk.
    """
    logger.info("=== Epilepsiae artifact census (--include-pack-lag) ===")
    rows = []
    for subj in EPILEPSIAE_SUBJECTS:
        logger.info(f"  {subj}...")
        row = audit_subject_artifact_census_epilepsiae(subj)
        rows.append(row)
        logger.info(
            f"    legacy: refine={row['legacy_refine_ok']} "
            f"gpu_real/stub={row['legacy_gpu_real_count']}/{row['legacy_gpu_stub_count']} "
            f"lagpat={row['legacy_lagpat_records']} keys_complete={row['legacy_lagpat_keys_complete']} "
            f"-> {row['legacy_verdict']}"
        )
        logger.info(
            f"    new:    gpu_loadable={row['new_gpu_records_loadable']}/{row['new_gpu_records_total']} "
            f"start_time={row['new_gpu_start_time_present_count']}/{row['new_gpu_records_total']} "
            f"schema={row['new_gpu_schema_complete_count']}/{row['new_gpu_records_total']} "
            f"refine={row['new_refine_ok']} "
            f"-> {row['new_verdict']}"
        )

    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "epilepsiae_artifact_census.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"\nSaved {out_path}")

    legacy_hist = df["legacy_verdict"].value_counts().to_dict()
    new_hist = df["new_verdict"].value_counts().to_dict()
    logger.info("\n=== CENSUS SUMMARY ===")
    logger.info(f"legacy_verdict: {legacy_hist}")
    logger.info(f"new_verdict:    {new_hist}")
    n_replay = int((df["legacy_verdict"] == "replay_eligible").sum())
    n_stage2_ready = int((df["new_verdict"] == "ready").sum())
    logger.info(f"Track B replay-eligible: {n_replay}/{len(df)}")
    logger.info(f"Stage 2 ready (new pipeline OK): {n_stage2_ready}/{len(df)}")
    if n_replay == 0:
        logger.info("→ Track B SKIP for Epilepsiae cohort (no replay-eligible subjects)")
    if n_stage2_ready < len(df):
        broken = df[df["new_verdict"] != "ready"]["subject"].tolist()
        logger.warning(f"→ Stage 2 BLOCKED for: {broken} (fix detector / start_time before running PR-2)")


def main():
    parser = argparse.ArgumentParser(description="Audit gpu.npz availability for spatial modulation PR-1")
    parser.add_argument("--min-count", type=int, default=100,
                        help="Minimum total event count per channel (default: 100)")
    parser.add_argument("--min-rate", type=float, default=5.0,
                        help="Minimum event rate in events/hour (default: 5.0)")
    parser.add_argument("--include-pack-lag", action="store_true",
                        help="Run Stage 0 Epilepsiae artifact census (legacy refine/gpu/lagPat + new pipeline schema). "
                             "Writes results/spatial_modulation/epilepsiae_artifact_census.csv. "
                             "Skips the standard Yuquan+Epilepsiae k-sweep audit when set.")
    args = parser.parse_args()

    if args.include_pack_lag:
        _run_artifact_census_epilepsiae()
        return

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
