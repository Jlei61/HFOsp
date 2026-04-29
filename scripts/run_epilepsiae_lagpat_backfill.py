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
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NEW_GPU_ROOT = Path("results/hfo_detection")
OUTPUT_ROOT = Path("results/epilepsiae_lagpat_backfill")
RIPPLE_BAND = (80.0, 250.0)
NYQUIST_GATE_HZ = 2.0 * RIPPLE_BAND[1]  # 500 Hz; 256 Hz blocks fail this
SEGMENT_SEC = 200.0  # mirrors Yuquan stitched-segment legacy semantics


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
    parser.parse_args()

    raise NotImplementedError("Stage A.1 骨架")


if __name__ == "__main__":
    main()
