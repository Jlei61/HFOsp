#!/usr/bin/env python3
"""Recoverable R1.3 explicit T1 triage for long-support development subjects."""
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
REVISION = "r1_3_long_t1_target_trained_triage_v1"
SUBJECTS = (
    "yuquan_hanyuxuan",
    "yuquan_chenziyang",
    "yuquan_chengshuai",
)
SEEDS = (0, 1, 2)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        value = load(path)
    except Exception:
        return False
    return value.get("status") == "COMPLETE" and value.get("sealed_opened") is False


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
        return min(float(row.strip()) for row in output.splitlines())
    except Exception:
        return 0.0


def wait_for_resources(*, ram_gib: float, gpu_mib: float = 0.0) -> None:
    while available_gib() < ram_gib or (gpu_mib and gpu_free_mib() < gpu_mib):
        time.sleep(20.0)


def environment() -> dict[str, str]:
    value = os.environ.copy()
    value.update({
        "PYTHONPATH": str(contract.REPO_ROOT),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "CUDA_MODULE_LOADING": "LAZY",
        "LD_LIBRARY_PATH": (
            "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib:"
            + value.get("LD_LIBRARY_PATH", "")
        ),
    })
    return value


def run(command: list[str], log: Path, *, ram_gib: float,
        gpu_mib: float = 0.0) -> dict:
    wait_for_resources(ram_gib=ram_gib, gpu_mib=gpu_mib)
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


def build_cache(subject: str, root: Path) -> dict:
    output = root / "cache" / subject / "manifest.json"
    if complete(output):
        return {"subject": subject, "status": "COMPLETE", "skipped": True,
                "output": str(output)}
    command = [
        str(PYTHON),
        "scripts/topic5_continuous_marked_state_r1/build_r1_3_observation_cache.py",
        "--subject", subject,
        "--output-root", str(root),
    ]
    value = run(
        command, root / "logs/cache" / f"{subject}.log", ram_gib=48.0,
    )
    value.update({"subject": subject, "output": str(output)})
    value["status"] = (
        "COMPLETE" if value["returncode"] == 0 and complete(output) else "FAIL"
    )
    return value


def fit(subject: str, seed: int, root: Path) -> dict:
    output = root / "fits" / subject / f"explicit_seed_{seed}" / "result.json"
    if complete(output):
        return {"subject": subject, "seed": seed, "status": "COMPLETE",
                "skipped": True, "output": str(output)}
    command = [
        str(PYTHON),
        "scripts/topic5_continuous_marked_state_r1/run_r1_3_target_observer.py",
        "--subject", subject,
        "--arm", "explicit",
        "--seed", str(seed),
        "--device", "cuda",
        "--observer-epochs", "4",
        "--joint-epochs", "4",
        "--chunk-anchors", "8",
        "--r1-2-fallback-seed-mode", "matching_seed",
        "--output-root", str(root),
        "--observation-cache-root", str(root / "cache"),
    ]
    value = run(
        command, root / "logs/fits" / f"{subject}_seed_{seed}.log",
        ram_gib=48.0, gpu_mib=9000.0,
    )
    value.update({"subject": subject, "seed": seed, "output": str(output)})
    value["status"] = (
        "COMPLETE" if value["returncode"] == 0 and complete(output) else "FAIL"
    )
    return value


def parallel(function, tasks: list[tuple], workers: int) -> list[dict]:
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(function, *task): task for task in tasks}
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as error:
                rows.append({
                    "task": list(futures[future]), "status": "FAIL",
                    "error": repr(error),
                })
    return rows


def summarise(root: Path, caches: list[dict], fits: list[dict]) -> dict:
    rows = []
    for subject in SUBJECTS:
        for seed in SEEDS:
            path = root / "fits" / subject / f"explicit_seed_{seed}" / "result.json"
            if not complete(path):
                continue
            value = load(path)
            persistent = value["validation"]["persistent_minus_memoryless"]
            wrong = value["validation"]["strict_matched_wrong_time"][
                "correct_minus_wrong_median"
            ]
            endpoint = value["validation"]["mark_endpoints"][
                "persistent_minus_memoryless"
            ]
            rows.append({
                "subject": subject,
                "seed": seed,
                "selected_total_epoch": value["fit_trace"]["selected_total_epoch"],
                "persistent_minus_memoryless_joint_nll": persistent[
                    "joint_nll_per_event"
                ],
                "correct_minus_wrong_joint_nll": wrong["joint_nll_per_event"],
                "first_subset": endpoint["first_group_subset_nll_per_event"],
                "continuation": endpoint[
                    "same_prefix_continuation_nll_per_event"
                ],
                "stop": endpoint["stop_nll_per_event"],
                "size": endpoint["selecting_group_size_nll_per_event"],
                "target_alignment_selected": bool(
                    value["fit_trace"]["selected_total_epoch"] > 0
                ),
                "persistent_memory_supported": bool(
                    persistent["joint_nll_per_event"] < 0.0
                ),
                "time_specific_supported": bool(
                    wrong["joint_nll_per_event"] < 0.0
                ),
                "initial_checkpoint_sha256": value["initialisation"][
                    "checkpoint_sha256"
                ],
                "checkpoint_sha256": value["checkpoint_sha256"],
                "result": str(path),
            })
    subjects = {}
    for subject in SUBJECTS:
        take = [row for row in rows if row["subject"] == subject]
        subjects[subject] = {
            "completed_seeds": len(take),
            "distinct_initial_payloads": len({
                row["initial_checkpoint_sha256"] for row in take
            }),
            "distinct_final_payloads": len({row["checkpoint_sha256"] for row in take}),
            "target_alignment_selected": sum(
                row["target_alignment_selected"] for row in take
            ),
            "persistent_memory_supported": sum(
                row["persistent_memory_supported"] for row in take
            ),
            "time_specific_supported": sum(
                row["time_specific_supported"] for row in take
            ),
            "eligible_for_h3_support_audit": bool(
                take and sum(
                    row["target_alignment_selected"]
                    and row["persistent_memory_supported"] for row in take
                ) >= 2
            ),
        }
    status = "COMPLETE" if len(rows) == len(SUBJECTS) * len(SEEDS) else "INCOMPLETE"
    payload = {
        "status": status,
        "revision": REVISION,
        "generated_at": now(),
        "subjects": list(SUBJECTS),
        "seeds": list(SEEDS),
        "observer_epochs": 4,
        "joint_epochs": 4,
        "r1_2_fallback_seed_mode": "matching_seed",
        "cache_jobs": caches,
        "fit_jobs": fits,
        "rows": rows,
        "by_subject": subjects,
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "claim_boundary": (
            "fixed three-subject development T1 triage; seeds are optimisation "
            "starts, not biological replicates; H3 is not tested here"
        ),
    }
    contract.atomic_json(root / "summary.json", payload)
    contract.atomic_json(root / "STATUS.json", {
        "status": status,
        "stage": "t1_triage_complete" if status == "COMPLETE" else "incomplete",
        "completed_cache_jobs": sum(row.get("status") == "COMPLETE" for row in caches),
        "completed_fit_jobs": len(rows),
        "expected_fit_jobs": len(SUBJECTS) * len(SEEDS),
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "summary": str(root / "summary.json"),
    })
    return payload


def write_status(root: Path, stage: str, *, caches: list[dict] | None = None,
                 fits: list[dict] | None = None) -> None:
    contract.atomic_json(root / "STATUS.json", {
        "status": "RUNNING",
        "stage": stage,
        "completed_cache_jobs": sum(
            row.get("status") == "COMPLETE" for row in (caches or [])
        ),
        "completed_fit_jobs": sum(
            row.get("status") == "COMPLETE" for row in (fits or [])
        ),
        "expected_fit_jobs": len(SUBJECTS) * len(SEEDS),
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "updated_at": now(),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root", type=Path,
        default=contract.RESULT_ROOT / "r1_3_long_t1_triage",
    )
    args = parser.parse_args()
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    write_status(root, "cache")
    caches = parallel(
        build_cache, [(subject, root) for subject in SUBJECTS], workers=2
    )
    if any(row.get("status") != "COMPLETE" for row in caches):
        summarise(root, caches, [])
        raise SystemExit("R1.3 long T1 cache stage failed")
    write_status(root, "explicit_target_alignment", caches=caches)
    fits = parallel(
        fit,
        [(subject, seed, root) for subject in SUBJECTS for seed in SEEDS],
        workers=2,
    )
    summary = summarise(root, caches, fits)
    if summary["status"] != "COMPLETE":
        raise SystemExit("R1.3 long T1 triage has incomplete fits")
    print(json.dumps({
        "status": summary["status"],
        "root": str(root),
        "by_subject": summary["by_subject"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
