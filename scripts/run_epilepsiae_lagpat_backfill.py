"""Epilepsiae new-pipeline pack + lagPat backfill driver.

Reads:  results/hfo_detection/<subject>/*_gpu.npz       (whole_dets, chns_names, start_time)
        results/hfo_detection/<subject>/_refineGpu.npz   (subject-level refined channel list)
        Raw .data + .head via load_epilepsiae_block       (CAR signal, variable sfreq)
Writes: results/epilepsiae_lagpat_backfill/<subject>/<stem>_packedTimes.npy
        results/epilepsiae_lagpat_backfill/<subject>/<stem>_lagPat.npz
        results/epilepsiae_lagpat_backfill/<subject>/_backfill_log.json

NOT a Track B replay: legacy *_gpu.npz are 216-byte stubs (per artifact census 2026-04-27).
Output is a NEW pack/lag artifact for sensitivity-audit purposes only.

Plan: docs/archive/epilepsiae_lagpat/epilepsiae_lagpat_backfill_plan_2026-04-29.md
"""
from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.epilepsiae_dataset import EpilepsiaePaths, _collect_raw_blocks
from src.preprocessing import _read_epilepsiae_head_for_streaming

NEW_GPU_ROOT = Path("results/hfo_detection")
OUTPUT_ROOT = Path("results/epilepsiae_lagpat_backfill")
RIPPLE_BAND = (80.0, 250.0)
NYQUIST_GATE_HZ = 2.0 * RIPPLE_BAND[1]  # 500 Hz; 256 Hz blocks fail this
SEGMENT_SEC = 200.0  # mirrors Yuquan stitched-segment legacy semantics


def _refine_path_for_subject(subject: str) -> Path:
    return NEW_GPU_ROOT / subject / "_refineGpu.npz"


@lru_cache(maxsize=32)
def load_refine_chns_for_subject(subject: str) -> Tuple[str, ...]:
    """Read subject-level refined channel names (cached per subject).

    The new pipeline writes ONE refine artifact per subject (no per-record
    suffix), schema: {chns_names, events_count}. Returns a tuple so lru_cache
    is safe; convert at call site if mutation is needed.
    """
    refine_path = _refine_path_for_subject(subject)
    if not refine_path.exists():
        raise FileNotFoundError(
            f"Subject-level refine artifact missing: {refine_path}\n"
            "Expected file produced by scripts/run_hfo_detection.py."
        )
    z = np.load(refine_path, allow_pickle=True)
    return tuple(str(c) for c in z["chns_names"])


def _discover_records(subject: str) -> List[Dict]:
    """Cross-reference results/hfo_detection/<subject>/*_gpu.npz with raw .data/.head.

    Returns list of dicts: stem, sfreq, new_gpu_path, raw_data_path, raw_head_path.
    Records where raw .data/.head are missing are skipped with a warning (not error)
    because the new pipeline already filtered Nyquist-failing blocks.
    """
    paths = EpilepsiaePaths()
    raw_blocks = _collect_raw_blocks(subject, paths)  # stem -> RawBlockFiles
    new_gpu_dir = NEW_GPU_ROOT / subject
    if not new_gpu_dir.exists():
        raise FileNotFoundError(f"No new-pipeline gpu dir: {new_gpu_dir}")
    out: List[Dict] = []
    for gpu_path in sorted(new_gpu_dir.glob("*_gpu.npz")):
        stem = gpu_path.stem.replace("_gpu", "")
        if stem.endswith("_refineGpu") or stem.startswith("sub_"):
            continue
        if stem not in raw_blocks:
            print(f"  WARN: {stem} has new gpu but no raw .data/.head; skip")
            continue
        block = raw_blocks[stem]
        if block.head_path is None or block.data_path is None:
            print(f"  WARN: {stem} raw .data or .head missing on disk; skip")
            continue
        head_info = _read_epilepsiae_head_for_streaming(block.head_path)
        sfreq = float(head_info.get("sample_freq", 0))
        out.append(
            {
                "stem": stem,
                "sfreq": sfreq,
                "new_gpu_path": gpu_path,
                "raw_data_path": block.data_path,
                "raw_head_path": block.head_path,
            }
        )
    return out


def _smoke_print(subject: str) -> None:
    """Dry-print first record's metadata; do not write any files."""
    recs = _discover_records(subject)
    if not recs:
        print(f"[smoke] subject={subject} no records discovered")
        return
    first = recs[0]
    z = np.load(first["new_gpu_path"], allow_pickle=True)
    chns_names = [str(c) for c in z["chns_names"]]
    whole_dets = z["whole_dets"]
    n_dets = sum(
        int(np.atleast_2d(np.asarray(d)).shape[0])
        for d in whole_dets
        if np.asarray(d).size
    )
    refine_chns = load_refine_chns_for_subject(subject)
    print(f"[smoke] subject={subject}  total_records={len(recs)}")
    print(f"[smoke] first stem={first['stem']}  sfreq={first['sfreq']}")
    print(f"[smoke]   gpu_path={first['new_gpu_path']}")
    print(f"[smoke]   raw_data={first['raw_data_path']}")
    print(f"[smoke]   raw_head={first['raw_head_path']}")
    print(
        f"[smoke]   n_chns_full={len(chns_names)}  "
        f"n_chns_refined={len(refine_chns)}  n_dets_total={n_dets}"
    )
    print("[smoke] (no files written)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Epilepsiae new-pipeline pack + lagPat backfill (Stage A skeleton).",
    )
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Process first record of given subject only; dry-print, no writes.",
    )
    args = parser.parse_args()

    if args.smoke:
        if not args.subject:
            parser.error("--smoke requires --subject")
        _smoke_print(args.subject)
        return

    raise NotImplementedError(
        "Stage A skeleton only. Stage B (pack + lagPat production) not implemented yet."
    )


if __name__ == "__main__":
    main()
