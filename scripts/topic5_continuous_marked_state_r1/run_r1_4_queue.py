#!/usr/bin/env python3
"""Recoverable six-patient R1.4 queue with bounded GPU concurrency."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from src.topic5_continuous_marked_state_r1 import contract


PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
REVISION = "r1_4_six_patient_explicit_primary_raw_residual_v1"
SUBJECTS = (
    "epilepsiae_620",
    "epilepsiae_958",
    "yuquan_huanghanwen",
    "epilepsiae_922",
    "yuquan_pengzihang",
    "yuquan_hanyuxuan",
)
SEEDS = (0, 1, 2)
R1_2_ROOT = contract.RESULT_ROOT / "r1_2"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic(path: Path, value: dict) -> None:
    contract.atomic_json(path, value)


def complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text())
    except Exception:
        return False
    return (
        value.get("status") == "COMPLETE"
        and value.get("sealed_opened") is False
        and value.get("experiment_label", REVISION) == REVISION
    )


def environment() -> dict[str, str]:
    value = os.environ.copy()
    value.update({
        "PYTHONPATH": str(contract.REPO_ROOT),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "CUDA_MODULE_LOADING": "LAZY",
    })
    return value


def run(command: list[str], log: Path) -> dict:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = now()
    with log.open("a") as handle:
        handle.write(f"\n[{started}] {' '.join(command)}\n")
        handle.flush()
        process = subprocess.run(
            command, cwd=contract.REPO_ROOT, env=environment(),
            stdout=handle, stderr=subprocess.STDOUT, text=True,
            start_new_session=True,
        )
    return {
        "command": command,
        "log": str(log),
        "started": started,
        "finished": now(),
        "returncode": int(process.returncode),
    }


def cache_root(subject: str, root: Path) -> Path:
    candidates = (
        contract.RESULT_ROOT / "r1_3" / "cache" / subject / "manifest.json",
        contract.RESULT_ROOT / "r1_3_long_t1_triage" / "cache" / subject / "manifest.json",
        root / "cache" / subject / "manifest.json",
    )
    for manifest in candidates:
        if manifest.exists():
            value = json.loads(manifest.read_text())
            if value.get("status") == "COMPLETE" and value.get("sealed_opened") is False:
                return manifest.parent.parent
    return root / "cache"


def build_cache(subject: str, root: Path) -> dict:
    existing = cache_root(subject, root) / subject / "manifest.json"
    if complete(existing):
        return {"subject": subject, "status": "COMPLETE", "skipped": True,
                "manifest": str(existing)}
    command = [
        str(PYTHON),
        "scripts/topic5_continuous_marked_state_r1/build_r1_3_observation_cache.py",
        "--subject", subject,
        "--output-root", str(root),
    ]
    value = run(command, root / "logs/cache" / f"{subject}.log")
    manifest = root / "cache" / subject / "manifest.json"
    value.update({"subject": subject, "manifest": str(manifest)})
    value["status"] = "COMPLETE" if value["returncode"] == 0 and complete(manifest) else "FAIL"
    return value


def prepare_initialisation(subject: str, seed: int, root: Path) -> dict:
    """Fit a matching-seed R1.2 core before any R1.4 target alignment.

    The original three R1.3 patients only had seed 0 in R1.2.  Reusing their
    later R1.2b checkpoints while the added patients start from R1.2 would give
    the two patient groups different training histories.  R1.4 therefore
    materialises the same R1.2 explicit arm for every patient and every seed.
    """
    output = (
        R1_2_ROOT / "t1_full" / subject
        / f"explicit_d8_seed_{seed}" / "result.json"
    )
    if complete(output):
        return {
            "subject": subject, "seed": seed, "status": "COMPLETE",
            "skipped": True, "output": str(output),
        }
    command = [
        str(PYTHON),
        "scripts/topic5_continuous_marked_state_r1/run_r1_2_t1.py",
        "--subject", subject,
        "--arm", "explicit",
        "--seed", str(seed),
        "--device", "cuda",
        "--epochs", "4",
        "--chunk-anchors", "256",
        "--output-root", str(R1_2_ROOT),
    ]
    value = run(
        command,
        root / "logs/initialisation" / f"{subject}_seed_{seed}.log",
    )
    value.update({"subject": subject, "seed": seed, "output": str(output)})
    value["status"] = (
        "COMPLETE" if value["returncode"] == 0 and complete(output) else "FAIL"
    )
    return value


def fit(subject: str, arm: str, seed: int, root: Path) -> dict:
    output = root / "fits" / subject / f"{arm}_seed_{seed}" / "result.json"
    if complete(output):
        return {"subject": subject, "arm": arm, "seed": seed,
                "status": "COMPLETE", "skipped": True, "output": str(output)}
    observation_root = cache_root(subject, root)
    command = [
        str(PYTHON),
        "scripts/topic5_continuous_marked_state_r1/run_r1_3_target_observer.py",
        "--subject", subject,
        "--arm", arm,
        "--seed", str(seed),
        "--device", "cuda",
        "--experiment-label", REVISION,
        "--observer-epochs", "4",
        "--joint-epochs", "4",
        "--chunk-anchors", "8",
        "--initialisation-source", "r1_2_matching_seed",
        "--matched-wrong-donors", "5",
        "--output-root", str(root),
        "--observation-cache-root", str(observation_root),
    ]
    value = run(
        command,
        root / "logs/fits" / f"{subject}_{arm}_seed_{seed}.log",
    )
    value.update({"subject": subject, "arm": arm, "seed": seed,
                  "output": str(output)})
    value["status"] = "COMPLETE" if value["returncode"] == 0 and complete(output) else "FAIL"
    return value


def parallel(function, tasks: list[tuple], workers: int) -> list[dict]:
    rows = []
    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        futures = {pool.submit(function, *task): task for task in tasks}
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as error:
                rows.append({"task": list(futures[future]), "status": "FAIL",
                             "error": repr(error)})
    return rows


def status(root: Path, stage: str, rows: list[dict] | None = None) -> None:
    fits = list((root / "fits").glob("*/*/result.json"))
    atomic(root / "STATUS.json", {
        "status": "RUNNING" if stage != "complete" else "COMPLETE",
        "stage": stage,
        "revision": REVISION,
        "completed_fits": sum(complete(path) for path in fits),
        "expected_fits": len(SUBJECTS) * len(SEEDS) * 2,
        "last_rows": rows or [],
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "updated_at": now(),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initialisation-workers", type=int, default=3)
    parser.add_argument("--explicit-workers", type=int, default=3)
    parser.add_argument("--raw-workers", type=int, default=2)
    parser.add_argument(
        "--root", type=Path, default=contract.RESULT_ROOT / "r1_4",
    )
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    status(args.root, "initialisation")
    initialisations = parallel(
        prepare_initialisation,
        [(subject, seed, args.root) for subject in SUBJECTS for seed in SEEDS],
        args.initialisation_workers,
    )
    if any(row.get("status") != "COMPLETE" for row in initialisations):
        status(args.root, "initialisation_fail", initialisations)
        raise RuntimeError("R1.4 matching-seed R1.2 initialisation stage failed")
    status(args.root, "cache", initialisations)
    caches = parallel(build_cache, [(subject, args.root) for subject in SUBJECTS], 2)
    if any(row.get("status") != "COMPLETE" for row in caches):
        status(args.root, "cache_fail", caches)
        raise RuntimeError("R1.4 cache stage failed")
    status(args.root, "explicit", caches)
    explicit = parallel(
        fit,
        [(subject, "explicit", seed, args.root) for subject in SUBJECTS for seed in SEEDS],
        args.explicit_workers,
    )
    if any(row.get("status") != "COMPLETE" for row in explicit):
        status(args.root, "explicit_fail", explicit)
        raise RuntimeError("R1.4 explicit stage failed")
    status(args.root, "explicit_raw", explicit)
    raw = parallel(
        fit,
        [(subject, "explicit_raw", seed, args.root) for subject in SUBJECTS for seed in SEEDS],
        args.raw_workers,
    )
    if any(row.get("status") != "COMPLETE" for row in raw):
        status(args.root, "raw_fail", raw)
        raise RuntimeError("R1.4 raw stage failed")
    status(args.root, "aggregate", raw)
    aggregate = run([
        str(PYTHON),
        "scripts/topic5_continuous_marked_state_r1/aggregate_r1_4.py",
        "--root", str(args.root),
    ], args.root / "logs/aggregate.log")
    if aggregate["returncode"] != 0:
        status(args.root, "aggregate_fail", [aggregate])
        raise RuntimeError("R1.4 aggregation failed")
    status(args.root, "complete", [aggregate])


if __name__ == "__main__":
    main()
