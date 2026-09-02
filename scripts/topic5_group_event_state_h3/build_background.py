#!/usr/bin/env python3
"""Extract the 30 s background-SEEG anchor grid from the block shards, per patient.

CPU-only.  Runs safely while the old v0.1 queue owns both GPUs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state_h3.background import write_background_table  # noqa: E402
from src.topic5_group_event_state_h3.io import write_json_atomic  # noqa: E402

DEFAULT_CACHE = Path("/data/hfosp_group_event_state_v0_1/cache")
DEFAULT_OUT = Path("/data/hfosp_group_event_state_v0_2/agent_c/background")


def _one(subject: str, cache_root: Path, out_root: Path) -> dict:
    started = time.time()
    try:
        meta = write_background_table(cache_root / subject, out_root / f"{subject}.npz")
        meta["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - failures are recorded, never silent
        meta = {"subject": subject, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    meta["seconds"] = round(time.time() - started, 1)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--subjects", nargs="*", default=None)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    subjects = args.subjects or sorted(p.name for p in args.cache_root.iterdir() if p.is_dir())

    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_one, s, args.cache_root, args.out_root): s for s in subjects}
        for fut in as_completed(futures):
            rec = fut.result()
            records.append(rec)
            print(
                f"{rec['subject']:26s} {rec['status']:7s} "
                f"n_anchors={rec.get('n_anchors', 0):7d} {rec['seconds']:6.1f}s",
                flush=True,
            )

    records.sort(key=lambda r: r["subject"])
    write_json_atomic(
        {
            "n_subjects": len(records),
            "n_failed": sum(1 for r in records if r["status"] != "ok"),
            "subjects": records,
        },
        ROOT / "results/epi_prssm/group_event_state/v0_2/h3/support/background_table.json",
    )
    print(f"failed: {sum(1 for r in records if r['status'] != 'ok')}")


if __name__ == "__main__":
    main()
