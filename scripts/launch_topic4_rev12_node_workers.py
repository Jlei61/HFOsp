#!/usr/bin/env python3
"""Run a bounded rev12-ND worker queue through transient user services."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_topic4_rev12_node_worker import _validate_scientific_role  # noqa: E402

ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
COMPLETE_STATUS = "REV12ND_NODE_WORKER_COMPLETE"


def _available_memory_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024.0**2
    raise RuntimeError("MemAvailable is absent from /proc/meminfo")


def _available_disk_gib(path: Path) -> float:
    stats = os.statvfs(path)
    return float(stats.f_bavail * stats.f_frsize) / 1024.0**3


def _unit_state(unit: str) -> str:
    result = subprocess.run(
        ["systemctl", "--user", "show", "-p", "ActiveState", "--value", unit],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "inactive"


def _worker_complete(path: Path) -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text())
    if payload.get("status") != COMPLETE_STATUS:
        raise RuntimeError(f"non-complete worker artifact: {path}")
    return True


def _unit_token(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-").lower()


def _stop_units(units: list[str]) -> None:
    for unit in units:
        subprocess.run(
            ["systemctl", "--user", "stop", unit],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def _resource_contract(resources: dict, *, maximum_workers: int | None,
                       estimated_worker_gib: float | None) -> tuple[int, float]:
    workers = (
        int(resources["maximum_workers"])
        if maximum_workers is None else int(maximum_workers)
    )
    estimate = (
        float(resources.get("estimated_worker_gib", 14.0))
        if estimated_worker_gib is None else float(estimated_worker_gib)
    )
    if workers <= 0 or estimate <= 0.0:
        raise RuntimeError("worker count and estimated memory must be positive")
    return workers, estimate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--seed-pool",
        choices=("canary", "fit", "selection", "confirmation"),
        required=True,
    )
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--unit-prefix", default="codex-t4-r12")
    parser.add_argument("--maximum-workers", type=int)
    parser.add_argument("--estimated-worker-gib", type=float)
    args = parser.parse_args()

    config_path = args.config.resolve()
    artifact_root = args.artifact_root.resolve()
    config = json.loads(config_path.read_text())
    _validate_scientific_role(config.get("scientific_role", ""))
    manifest_path = artifact_root / config["candidate_manifest"]
    manifest = json.loads(manifest_path.read_text())
    seeds = [int(seed) for seed in config["search"][f"{args.seed_pool}_network_seeds"]]
    if not seeds:
        raise RuntimeError(f"empty {args.seed_pool} seed pool")
    candidates = [str(row["candidate_id"]) for row in manifest["candidates"]]
    output_root = artifact_root / config["output_root"]
    worker_dir = output_root / "workers"
    worker_dir.mkdir(parents=True, exist_ok=True)

    resources = config["resources"]
    max_workers, estimated_worker_gib = _resource_contract(
        resources, maximum_workers=args.maximum_workers,
        estimated_worker_gib=args.estimated_worker_gib,
    )
    reserve_gib = float(resources["reserved_available_memory_gib"])
    interval = int(resources["monitor_interval_seconds"])
    disk_floor_gib = 40.0
    expected_commit = subprocess.check_output(
        ["git", "rev-parse", args.expected_commit], cwd=ROOT, text=True,
    ).strip()

    queue = []
    for candidate in candidates:
        for seed in seeds:
            stem = f"{candidate}_seed_{seed}"
            output = worker_dir / f"{stem}.json"
            if not _worker_complete(output):
                queue.append((candidate, seed, output))
    active: dict[str, tuple[str, int, Path]] = {}
    launched: list[dict] = []
    started_at = time.time()

    while queue or active:
        for unit, task in list(active.items()):
            if _unit_state(unit) in {"active", "activating"}:
                continue
            candidate, seed, output = task
            if not _worker_complete(output):
                _stop_units(list(active))
                raise RuntimeError(
                    f"worker exited without a complete artifact: {candidate} seed {seed}"
                )
            del active[unit]

        available_gib = _available_memory_gib()
        disk_gib = _available_disk_gib(artifact_root)
        if available_gib < reserve_gib or disk_gib < disk_floor_gib:
            _stop_units(list(active))
            raise RuntimeError(
                f"resource floor crossed: memory={available_gib:.1f} GiB, "
                f"disk={disk_gib:.1f} GiB"
            )
        memory_slots = max(0, int((available_gib - reserve_gib) // estimated_worker_gib))
        slots = min(max_workers - len(active), memory_slots)
        while queue and slots > 0:
            candidate, seed, output = queue.pop(0)
            unit = (
                f"{_unit_token(args.unit_prefix)}-{_unit_token(args.seed_pool)}-"
                f"{_unit_token(candidate)}-{seed}.service"
            )
            command = [
                "systemd-run", "--user", f"--unit={unit}", "--collect",
                f"--working-directory={ROOT}",
                "/usr/bin/nohup", "/usr/bin/env",
                "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
                f"REV12ND_SYSTEMD_UNIT={unit}", str(PYTHON),
                str(ROOT / "scripts/run_topic4_rev12_node_worker.py"),
                "--config", str(config_path), "--candidate-id", candidate,
                "--seed", str(seed), "--expected-commit", expected_commit,
            ]
            subprocess.run(command, cwd=ROOT, check=True)
            active[unit] = (candidate, seed, output)
            launched.append({"unit": unit, "candidate_id": candidate, "seed": seed})
            slots -= 1

        print(json.dumps({
            "elapsed_minutes": round((time.time() - started_at) / 60.0, 1),
            "remaining": len(queue),
            "active": sorted(active),
            "available_memory_gib": round(available_gib, 1),
            "available_disk_gib": round(disk_gib, 1),
        }), flush=True)
        if queue or active:
            time.sleep(interval)

    status_path = output_root / "status" / f"{args.seed_pool}_controller.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps({
        "status": "REV12ND_NODE_WORKER_QUEUE_COMPLETE",
        "seed_pool": args.seed_pool,
        "expected_git_commit": expected_commit,
        "n_candidates": len(candidates),
        "n_seeds": len(seeds),
        "n_launched": len(launched),
        "maximum_workers": max_workers,
        "estimated_worker_gib": estimated_worker_gib,
        "elapsed_minutes": (time.time() - started_at) / 60.0,
        "launched": launched,
    }, indent=2) + "\n")
    print(json.dumps({
        "status": "REV12ND_NODE_WORKER_QUEUE_COMPLETE",
        "status_path": str(status_path),
    }), flush=True)


if __name__ == "__main__":
    main()
