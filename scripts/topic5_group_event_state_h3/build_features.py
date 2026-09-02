#!/usr/bin/env python3
"""C0: freeze each patient's event count/mark vocabularies to a small cache.

The waveform pass is the expensive part (93 GB of f16 across the cohort) and it
runs once.  Everything downstream then reads a few tens of megabytes per patient,
which is what makes the per-arm, per-seed training grid affordable.

CPU-only and I/O-bound: safe to run while the old v0.1 queue owns both GPUs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state_h3.io import write_json_atomic, write_npz_atomic  # noqa: E402
from src.topic5_group_event_state_h3.stream import build_subject_features  # noqa: E402

DEFAULT_DATASET = Path("/data/hfosp_group_event_state_v0_1/dataset")
DEFAULT_OUT = Path("/data/hfosp_group_event_state_v0_2/agent_c/features")


def _one(subject: str, dataset_root: Path, out_root: Path, overwrite: bool) -> dict:
    started = time.time()
    npz_path = out_root / f"{subject}.npz"
    json_path = out_root / f"{subject}.json"
    if npz_path.exists() and json_path.exists() and not overwrite:
        meta = json.loads(json_path.read_text())
        meta["status"] = "cached"
        meta["seconds"] = 0.0
        return meta
    try:
        feats = build_subject_features(
            dataset_root / subject, waveform_cache=out_root / f"{subject}__waveform_rms.npz"
        )
        write_npz_atomic(
            npz_path,
            t_abs=feats.t_abs,
            count_features=feats.count_features,
            mark_features=feats.mark_features,
            participation=feats.participation,
            size=feats.size,
        )
        meta = {
            "subject": subject,
            "status": "ok",
            "n_events": feats.n_events,
            "n_contacts": int(feats.participation.shape[1]),
            "n_count_features": int(feats.count_features.shape[1]),
            "n_mark_features": int(feats.mark_features.shape[1]),
            "count_feature_names": feats.count_feature_names,
            "mark_feature_names": feats.mark_feature_names,
            "mark_group_slices": {k: list(v) for k, v in feats.mark_group_slices.items()},
            "band_available": feats.band_available.tolist(),
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
        }
        write_json_atomic(meta, json_path)
    except Exception as exc:  # noqa: BLE001 - failures are recorded, never silent
        meta = {"subject": subject, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    meta["seconds"] = round(time.time() - started, 1)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    subjects = args.subjects or sorted(
        p.name for p in args.dataset_root.iterdir() if (p / "index.json").exists()
    )
    # Widest first: long patients dominate wall time, so starting them early keeps
    # the tail from being one 11 GB read after everything else has finished.
    subjects.sort(
        key=lambda s: -(args.dataset_root / s / "waveform.npy").stat().st_size
        if (args.dataset_root / s / "waveform.npy").exists()
        else 0
    )

    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_one, s, args.dataset_root, args.out_root, args.overwrite): s
                   for s in subjects}
        for fut in as_completed(futures):
            rec = fut.result()
            records.append(rec)
            print(
                f"{rec['subject']:26s} {rec['status']:7s} n_ev={rec.get('n_events', 0):8d} "
                f"F={rec.get('n_mark_features', 0):4d} rss={rec.get('peak_rss_gib', 0):5.1f}G "
                f"{rec['seconds']:7.1f}s",
                flush=True,
            )

    records.sort(key=lambda r: r["subject"])
    write_json_atomic(
        {
            "n_subjects": len(records),
            "n_failed": sum(1 for r in records if r["status"] == "failed"),
            "subjects": records,
        },
        ROOT / "results/epi_prssm/group_event_state/v0_2/h3/support/event_features.json",
    )
    print(f"failed: {sum(1 for r in records if r['status'] == 'failed')}")


if __name__ == "__main__":
    main()
