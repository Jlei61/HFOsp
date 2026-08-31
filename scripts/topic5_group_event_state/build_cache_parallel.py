#!/usr/bin/env python3
"""Block-level parallel cache builder.

Blocks are the unit of work and each writes an atomic shard + manifest, so the
pool is restartable: a re-run skips finished blocks instead of redoing them.
Worker count is a measured decision -- see ``--workers`` and the RSS reported per
block in the log.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.cache import BlockSpec, build_block_shard  # noqa: E402
from src.topic5_group_event_state.source_audit import write_json_atomic  # noqa: E402

MAIN_TREE = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_AUDIT = MAIN_TREE / "results/epi_prssm/group_event_state/v0_1/source_audit.json"
DEFAULT_CACHE = Path("/data/hfosp_group_event_state_v0_1/cache")

_CACHE_ROOT: Path = DEFAULT_CACHE
_CHUNK = 128


def _init(cache_root: str, chunk: int) -> None:
    global _CACHE_ROOT, _CHUNK
    _CACHE_ROOT = Path(cache_root)
    _CHUNK = int(chunk)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"


def _work(item: tuple[str, dict]) -> dict:
    subject, block = item
    spec = BlockSpec(
        dataset=block["dataset"],
        subject=subject.split("_", 1)[1],
        record_name=block["record_name"],
        raw_path=block["raw_path"],
        head_path=block["head_path"] or None,
        gpu_path=block["gpu_path"],
        lagpat_path=block["lagpat_path"],
        packed_path=block["packed_path"],
        block_start_epoch=float(block["block_start_epoch"]),
        native_rate_hz=float(block["native_rate_hz"]),
    )
    started = time.time()
    try:
        manifest = build_block_shard(spec, _CACHE_ROOT / subject, chunk_events=_CHUNK)
        return {
            "subject": subject,
            "record_name": spec.record_name,
            "ok": True,
            "n_events_with_waveform": manifest["n_events_with_waveform"],
            "n_events": manifest["n_events"],
            "shard_bytes": manifest["shard_bytes"],
            "seconds": round(time.time() - started, 1),
            "rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
        }
    except Exception as exc:
        return {
            "subject": subject,
            "record_name": spec.record_name,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
            "seconds": round(time.time() - started, 1),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-events", type=int, default=128)
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args()

    audit = json.loads(Path(args.audit).read_text())
    by_subject = {r["subject"]: r for r in audit["subjects"]}
    jobs: list[tuple[str, dict]] = []
    for subject in args.subjects:
        record = by_subject[subject]
        for block in sorted(record["blocks"], key=lambda b: float(b["block_start_epoch"])):
            if block["waveform_pointer_reconstructable"]:
                jobs.append((subject, block))
    print(f"{len(jobs)} blocks over {len(args.subjects)} subjects, {args.workers} workers", flush=True)

    started = time.time()
    results: list[dict] = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=args.workers,
        initializer=_init,
        initargs=(str(args.cache_root), args.chunk_events),
    ) as pool:
        for i, result in enumerate(pool.imap_unordered(_work, jobs, chunksize=1), start=1):
            results.append(result)
            if result["ok"]:
                print(
                    f"[{i}/{len(jobs)}] {result['subject']}/{result['record_name']} "
                    f"{result['n_events_with_waveform']}/{result['n_events']} ev "
                    f"{result['shard_bytes']/1e6:.0f}MB {result['seconds']}s "
                    f"rss={result['rss_mb']:.0f}MB",
                    flush=True,
                )
            else:
                print(
                    f"[{i}/{len(jobs)}] FAILED {result['subject']}/{result['record_name']}: "
                    f"{result['error']}",
                    flush=True,
                )
            if i % 25 == 0:
                write_json_atomic(
                    {"partial": True, "n_done": i, "n_total": len(jobs), "results": results},
                    args.out_summary,
                )

    ok = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    per_subject: dict[str, dict] = {}
    for r in ok:
        entry = per_subject.setdefault(
            r["subject"], {"n_blocks": 0, "n_events_with_waveform": 0, "n_events": 0, "bytes": 0}
        )
        entry["n_blocks"] += 1
        entry["n_events_with_waveform"] += r["n_events_with_waveform"]
        entry["n_events"] += r["n_events"]
        entry["bytes"] += r["shard_bytes"]
    summary = {
        "n_blocks": len(jobs),
        "n_blocks_ok": len(ok),
        "n_blocks_failed": len(bad),
        "n_events_with_waveform": int(sum(r["n_events_with_waveform"] for r in ok)),
        "total_bytes": int(sum(r["shard_bytes"] for r in ok)),
        "wall_seconds": round(time.time() - started, 1),
        "workers": args.workers,
        "max_worker_rss_mb": max((r.get("rss_mb", 0) for r in ok), default=0),
        "median_block_seconds": (
            sorted(r["seconds"] for r in ok)[len(ok) // 2] if ok else None
        ),
        "per_subject": per_subject,
        "failures": bad,
    }
    write_json_atomic(summary, args.out_summary)
    print(
        f"DONE {len(ok)}/{len(jobs)} blocks, {summary['n_events_with_waveform']} events, "
        f"{summary['total_bytes']/1e9:.1f} GB, {summary['wall_seconds']}s, "
        f"max worker RSS {summary['max_worker_rss_mb']} MB",
        flush=True,
    )


if __name__ == "__main__":
    main()
