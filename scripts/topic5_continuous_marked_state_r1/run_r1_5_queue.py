#!/usr/bin/env python3
"""Recoverable five-seed R1.5 explicit-state extension queue."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from src.topic5_continuous_marked_state_r1 import contract


PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
REVISION = "r1_5_long_support_explicit_extension_v1"
SUBJECTS = contract.R1_5_EXTENSION_SUBJECTS
SEEDS = (0, 1, 2, 3, 4)
R1_2_ROOT = contract.RESULT_ROOT / "r1_2"
TARGET_OBSERVER_RUNNER_REVISION = "r1_3_target_observer_segment_locked_v2"
TARGET_OBSERVER_RUNNER = (
    contract.REPO_ROOT
    / "scripts/topic5_continuous_marked_state_r1/run_r1_3_target_observer.py"
)
TARGET_OBSERVER_RUNNER_SHA256 = contract.sha256_file(TARGET_OBSERVER_RUNNER)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def complete(path: Path, *, experiment: str | None = None,
             runner_sha256: str | None = None) -> bool:
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text())
    except Exception:
        return False
    return bool(
        value.get("status") == "COMPLETE"
        and value.get("sealed_opened") is False
        and (experiment is None or value.get("experiment_label") == experiment)
        and (
            runner_sha256 is None
            or (
                value.get("target_observer_runner_revision")
                == TARGET_OBSERVER_RUNNER_REVISION
                and value.get("target_observer_runner_sha256")
                == runner_sha256
                and value.get("recorded_coverage_segment_lock_required") is True
                and value.get("validation", {}).get(
                    "strict_matched_wrong_time", {}
                ).get("audit", {}).get(
                    "same_recorded_coverage_segment"
                ) is True
            )
        )
    )


def available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024.0 / 1024.0
    return 0.0


def gpu_free_mib() -> float:
    try:
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ], text=True)
        return min(float(value.strip()) for value in output.splitlines())
    except Exception:
        return 0.0


def wait_for_resources(min_ram_gib: float, min_gpu_mib: float) -> None:
    while available_gib() < min_ram_gib or gpu_free_mib() < min_gpu_mib:
        time.sleep(20.0)


def environment() -> dict[str, str]:
    value = os.environ.copy()
    value.update({
        "PYTHONPATH": str(contract.REPO_ROOT),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "CUDA_MODULE_LOADING": "LAZY", "CUDA_VISIBLE_DEVICES": "0",
        "LD_LIBRARY_PATH": (
            "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib:"
            + value.get("LD_LIBRARY_PATH", "")
        ),
    })
    return value


def run(command: list[str], log: Path, *, min_gpu_mib: float = 5000.0) -> dict:
    wait_for_resources(48.0, min_gpu_mib)
    log.parent.mkdir(parents=True, exist_ok=True)
    started = now()
    with log.open("a") as handle:
        handle.write(f"\n[{started}] {' '.join(command)}\n")
        handle.flush()
        process = subprocess.run(
            command, cwd=contract.REPO_ROOT, env=environment(),
            stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            text=True, start_new_session=True,
        )
    return {
        "command": command, "log": str(log), "started": started,
        "finished": now(), "returncode": int(process.returncode),
    }


def prepare_subject(subject: str, root: Path) -> dict:
    tasks = [
        (
            R1_2_ROOT / "baselines" / subject / "seed_0/result.json",
            [str(PYTHON),
             "scripts/topic5_continuous_marked_state_r1/run_r1_2_baseline.py",
             "--subject", subject, "--seed", "0", "--device", "cuda",
             "--mark-batch-size", "512"],
            root / "logs/preparation" / f"{subject}_baseline.log",
        ),
        (
            R1_2_ROOT / "bridge_e1" / subject / "seed_0/result.json",
            [str(PYTHON),
             "scripts/topic5_continuous_marked_state_r1/run_r1_2_bridge.py",
             "--subject", subject, "--seed", "0", "--device", "cuda",
             "--anchor-batch-size", "2", "--max-train-anchors", "64",
             "--max-validation-anchors", "32"],
            root / "logs/preparation" / f"{subject}_bridge.log",
        ),
        (
            R1_2_ROOT / "cache" / subject / "manifest.json",
            [str(PYTHON),
             "scripts/topic5_continuous_marked_state_r1/run_r1_2_cache.py",
             "--subject", subject, "--device", "cuda",
             "--anchor-batch-size", "4"],
            root / "logs/preparation" / f"{subject}_cache.log",
        ),
    ]
    history = []
    for output, command, log in tasks:
        if complete(output):
            history.append({"output": str(output), "skipped": True})
            continue
        value = run(command, log, min_gpu_mib=6500.0)
        value["output"] = str(output); history.append(value)
        if value["returncode"] != 0 or not complete(output):
            return {"subject": subject, "status": "FAIL", "history": history}
    return {"subject": subject, "status": "COMPLETE", "history": history}


def fit_initialisation(subject: str, seed: int, root: Path) -> dict:
    output = (
        R1_2_ROOT / "t1_full" / subject
        / f"explicit_d8_seed_{seed}/result.json"
    )
    if complete(output):
        return {"subject": subject, "seed": seed, "status": "COMPLETE",
                "skipped": True, "output": str(output)}
    command = [
        str(PYTHON),
        "scripts/topic5_continuous_marked_state_r1/run_r1_2_t1.py",
        "--subject", subject, "--arm", "explicit", "--seed", str(seed),
        "--device", "cuda", "--epochs", "4", "--chunk-anchors", "128",
    ]
    value = run(
        command, root / "logs/initialisation" / f"{subject}_seed_{seed}.log",
        min_gpu_mib=5000.0,
    )
    value.update({"subject": subject, "seed": seed, "output": str(output)})
    value["status"] = (
        "COMPLETE" if value["returncode"] == 0 and complete(output) else "FAIL"
    )
    return value


def build_observation_cache(subject: str, root: Path) -> dict:
    candidates = (
        contract.RESULT_ROOT / "r1_3/cache" / subject / "manifest.json",
        contract.RESULT_ROOT / "r1_3_long_t1_triage/cache" / subject / "manifest.json",
        root / "cache" / subject / "manifest.json",
    )
    for path in candidates:
        if complete(path):
            return {"subject": subject, "status": "COMPLETE", "skipped": True,
                    "manifest": str(path)}
    command = [
        str(PYTHON),
        "scripts/topic5_continuous_marked_state_r1/build_r1_3_observation_cache.py",
        "--subject", subject, "--output-root", str(root),
    ]
    value = run(
        command, root / "logs/cache" / f"{subject}.log",
        min_gpu_mib=1000.0,
    )
    path = root / "cache" / subject / "manifest.json"
    value.update({"subject": subject, "manifest": str(path)})
    value["status"] = (
        "COMPLETE" if value["returncode"] == 0 and complete(path) else "FAIL"
    )
    return value


def observation_cache_root(subject: str, root: Path) -> Path:
    for candidate in (
        contract.RESULT_ROOT / "r1_3/cache" / subject / "manifest.json",
        contract.RESULT_ROOT / "r1_3_long_t1_triage/cache" / subject / "manifest.json",
        root / "cache" / subject / "manifest.json",
    ):
        if complete(candidate):
            return candidate.parent.parent
    raise FileNotFoundError(f"no R1.5 observation cache for {subject}")


def fit_r1_5(subject: str, seed: int, root: Path) -> dict:
    output = root / "fits" / subject / f"explicit_seed_{seed}/result.json"
    if complete(
        output, experiment=REVISION,
        runner_sha256=TARGET_OBSERVER_RUNNER_SHA256,
    ):
        return {"subject": subject, "seed": seed, "status": "COMPLETE",
                "skipped": True, "output": str(output)}
    command = [
        str(PYTHON),
        str(TARGET_OBSERVER_RUNNER),
        "--subject", subject, "--arm", "explicit", "--seed", str(seed),
        "--device", "cuda", "--experiment-label", REVISION,
        "--observer-epochs", "4", "--joint-epochs", "4",
        "--chunk-anchors", "8", "--initialisation-source",
        "r1_2_matching_seed", "--matched-wrong-donors", "5",
        "--output-root", str(root), "--observation-cache-root",
        str(observation_cache_root(subject, root)),
    ]
    value = run(
        command, root / "logs/fits" / f"{subject}_seed_{seed}.log",
        min_gpu_mib=7000.0,
    )
    value.update({"subject": subject, "seed": seed, "output": str(output)})
    value["status"] = (
        "COMPLETE" if value["returncode"] == 0
        and complete(
            output, experiment=REVISION,
            runner_sha256=TARGET_OBSERVER_RUNNER_SHA256,
        ) else "FAIL"
    )
    return value


def parallel(function, tasks: list[tuple], workers: int) -> list[dict]:
    rows = []
    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        future = {pool.submit(function, *task): task for task in tasks}
        for item in as_completed(future):
            try:
                rows.append(item.result())
            except Exception as error:
                rows.append({"task": list(future[item]), "status": "FAIL",
                             "error": repr(error)})
    return rows


def write_status(root: Path, stage: str, rows: list[dict] | None = None) -> None:
    fits = list((root / "fits").glob("*/explicit_seed_*/result.json"))
    contract.atomic_json(root / "STATUS.json", {
        "status": "COMPLETE" if stage == "complete" else "RUNNING",
        "stage": stage, "revision": REVISION,
        "subjects": list(SUBJECTS), "seeds": list(SEEDS),
        "completed_fits": sum(complete(
            path, experiment=REVISION,
            runner_sha256=TARGET_OBSERVER_RUNNER_SHA256,
        ) for path in fits),
        "expected_fits": len(SUBJECTS) * len(SEEDS),
        "target_observer_runner_revision": TARGET_OBSERVER_RUNNER_REVISION,
        "target_observer_runner_sha256": TARGET_OBSERVER_RUNNER_SHA256,
        "last_rows": rows or [], "updated_at": now(),
        "formal_test_partition_opened": False, "sealed_opened": False,
    })


def require_complete(root: Path, stage: str, rows: list[dict]) -> None:
    if any(row.get("status") != "COMPLETE" for row in rows):
        write_status(root, f"{stage}_fail", rows)
        raise RuntimeError(f"R1.5 {stage} stage failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--root", type=Path, default=contract.RESULT_ROOT / "r1_5",
    )
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    write_status(args.root, "preparation")
    rows = parallel(
        prepare_subject, [(subject, args.root) for subject in SUBJECTS],
        min(2, args.workers),
    )
    require_complete(args.root, "preparation", rows)
    write_status(args.root, "initialisation", rows)
    rows = parallel(
        fit_initialisation,
        [(subject, seed, args.root) for subject in SUBJECTS for seed in SEEDS],
        args.workers,
    )
    require_complete(args.root, "initialisation", rows)
    write_status(args.root, "cache", rows)
    rows = parallel(
        build_observation_cache,
        [(subject, args.root) for subject in SUBJECTS], min(2, args.workers),
    )
    require_complete(args.root, "cache", rows)
    write_status(args.root, "fit", rows)
    rows = parallel(
        fit_r1_5,
        [(subject, seed, args.root) for subject in SUBJECTS for seed in SEEDS],
        args.workers,
    )
    require_complete(args.root, "fit", rows)
    write_status(args.root, "aggregate", rows)
    value = run([
        str(PYTHON),
        "scripts/topic5_continuous_marked_state_r1/aggregate_r1_5.py",
        "--root", str(args.root),
    ], args.root / "logs/aggregate.log", min_gpu_mib=0.0)
    require_complete(args.root, "aggregate", [{
        **value,
        "status": "COMPLETE" if value["returncode"] == 0 else "FAIL",
    }])
    write_status(args.root, "complete", [value])


if __name__ == "__main__":
    main()
