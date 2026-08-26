#!/usr/bin/env python3
"""Build causal, IED-core-masked Bridge features for one or more pilot subjects."""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from src.topic5_continuous_marked_state import contract
from src.topic5_continuous_marked_state.bridge import write_bridge_dataset


def _one(subject: str, overwrite: bool, max_train: int, max_validation: int) -> dict:
    output = contract.RESULT_ROOT / "bridge/features" / f"{subject}.npz"
    manifest = output.with_suffix(".manifest.json")
    if output.exists() and manifest.exists() and not overwrite:
        old = json.loads(manifest.read_text())
        if old.get("contract") == contract.REVISION:
            return {**old, "skipped": True}
    return write_bridge_dataset(
        subject, output, max_train=max_train, max_validation=max_validation
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=list(contract.PILOT_SUBJECTS))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--max-train", type=int, default=contract.MAX_TRAIN_PAIRS)
    parser.add_argument("--max-validation", type=int, default=contract.MAX_VALIDATION_PAIRS)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    unknown = sorted(set(args.subjects) - set(contract.PILOT_SUBJECTS))
    if unknown:
        raise ValueError(f"subjects are not in frozen pilot6: {unknown}")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    results = []
    if args.jobs <= 1:
        for subject in args.subjects:
            result = _one(subject, args.overwrite, args.max_train, args.max_validation)
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(args.subjects))) as pool:
            futures = {
                pool.submit(_one, s, args.overwrite, args.max_train, args.max_validation): s
                for s in args.subjects
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)
    summary = {
        "contract": contract.REVISION,
        "subjects_requested": args.subjects,
        "n_completed": len(results),
        "n_rows": int(sum(r["n_rows"] for r in results)),
        "sealed_opened": False,
    }
    path = contract.RESULT_ROOT / "manifests/BRIDGE_FEATURE_BUILD.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True))
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
