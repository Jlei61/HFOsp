#!/usr/bin/env python3
"""Build the block-sharded multimodal group-event cache for one or more subjects.

Each block is an independent unit of work with an atomic shard + manifest, so an
interrupted run resumes by skipping finished blocks rather than restarting.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.topic5_group_event_state.cache import BlockSpec, build_block_shard  # noqa: E402
from src.topic5_group_event_state.source_audit import (  # noqa: E402
    write_json_atomic,
)

MAIN_TREE = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_AUDIT = MAIN_TREE / "results/epi_prssm/group_event_state/v0_1/source_audit.json"
DEFAULT_CACHE = Path("/data/hfosp_group_event_state_v0_1/cache")


def specs_for_subject(audit: dict, subject: str) -> list[BlockSpec]:
    record = next(r for r in audit["subjects"] if r["subject"] == subject)
    specs = []
    for block in record["blocks"]:
        if not block["waveform_pointer_reconstructable"]:
            continue
        specs.append(
            BlockSpec(
                dataset=block["dataset"],
                subject=block["subject"].split("_", 1)[1],
                record_name=block["record_name"],
                raw_path=block["raw_path"],
                head_path=block["head_path"] or None,
                gpu_path=block["gpu_path"],
                lagpat_path=block["lagpat_path"],
                packed_path=block["packed_path"],
                block_start_epoch=float(block["block_start_epoch"]),
                native_rate_hz=float(block["native_rate_hz"]),
            )
        )
    return sorted(specs, key=lambda s: s.block_start_epoch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--max-blocks", type=int, default=0, help="0 = all blocks")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--chunk-events", type=int, default=128)
    args = parser.parse_args()

    audit = json.loads(Path(args.audit).read_text())
    try:
        import resource

        def peak_rss_mb() -> float:
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:  # pragma: no cover
        def peak_rss_mb() -> float:
            return float("nan")

    for subject in args.subjects:
        specs = specs_for_subject(audit, subject)
        if args.max_blocks:
            specs = specs[: args.max_blocks]
        out_dir = Path(args.cache_root) / subject
        out_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        manifests, failures = [], []
        for i, spec in enumerate(specs):
            try:
                manifests.append(
                    build_block_shard(
                        spec,
                        out_dir,
                        chunk_events=args.chunk_events,
                        overwrite=args.overwrite,
                    )
                )
                print(
                    f"[{subject}] {i+1}/{len(specs)} {spec.record_name} "
                    f"events={manifests[-1]['n_events_with_waveform']}/{manifests[-1]['n_events']} "
                    f"{manifests[-1]['shard_bytes']/1e6:.1f}MB "
                    f"{manifests[-1]['build_seconds']:.1f}s rss={peak_rss_mb():.0f}MB",
                    flush=True,
                )
            except Exception as exc:
                failures.append(
                    {
                        "record_name": spec.record_name,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=6),
                    }
                )
                print(f"[{subject}] FAILED {spec.record_name}: {exc}", flush=True)

        total_bytes = int(sum(m["shard_bytes"] for m in manifests))
        total_events = int(sum(m["n_events_with_waveform"] for m in manifests))
        summary = {
            "subject": subject,
            "n_blocks_requested": len(specs),
            "n_blocks_built": len(manifests),
            "n_blocks_failed": len(failures),
            "n_events_with_waveform": total_events,
            "n_events_total": int(sum(m["n_events"] for m in manifests)),
            "total_bytes": total_bytes,
            "bytes_per_1000_events": (
                float(total_bytes / max(total_events, 1) * 1000.0)
            ),
            "elapsed_sec": round(time.time() - started, 1),
            "peak_rss_mb": round(peak_rss_mb(), 1),
            "failures": failures,
            "blocks": manifests,
        }
        write_json_atomic(summary, out_dir / "cache_summary.json")
        print(
            f"[{subject}] built {len(manifests)}/{len(specs)} blocks, "
            f"{total_events} events, {total_bytes/1e9:.2f} GB, "
            f"{summary['bytes_per_1000_events']/1e6:.1f} MB/1k events, "
            f"{summary['elapsed_sec']}s, peak RSS {summary['peak_rss_mb']} MB",
            flush=True,
        )


if __name__ == "__main__":
    main()
