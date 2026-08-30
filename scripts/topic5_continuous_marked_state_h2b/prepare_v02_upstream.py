#!/usr/bin/env python3
"""Rebuild and verify missing R1.7B upstream inputs inside H2b v0.2.

At most one producer is assigned to each GPU.  Rebuilt artifacts are accepted
only if the design and normalised explicit cache match the frozen R1.7B hashes
and the baseline tensors are bitwise equal to the frozen checkpoint baseline.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.topic5_continuous_marked_state_h2b.contract import (  # noqa: E402
    V0_2_RESULT_ROOT,
    atomic_json,
    sha256_file,
    utc_now,
)


PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
SOURCE = Path("/home/honglab/leijiaxin/HFOsp")
R1_ROOT = SOURCE / "results/epi_prssm/continuous_marked_state/r1"
R17B = R1_ROOT / "r1_7b_cohort_extension"


def _complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text())
    except Exception:
        return False
    return bool(
        value.get("status") == "COMPLETE"
        and value.get("formal_test_partition_opened", False) is False
        and value.get("sealed_opened") is False
    )


def _available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024.0 / 1024.0
    return 0.0


def _gpu_free_mib(gpu: int) -> float:
    try:
        value = subprocess.check_output([
            "nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ], text=True)
        return float(value.strip())
    except Exception:
        return 0.0


def _environment(gpu: int) -> dict[str, str]:
    value = os.environ.copy()
    value.update({
        "PYTHONPATH": str(REPO),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "CUDA_MODULE_LOADING": "LAZY",
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "LD_LIBRARY_PATH": (
            "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib:"
            + value.get("LD_LIBRARY_PATH", "")
        ),
    })
    return value


def _run(command: list[str], log: Path, *, gpu: int, min_gpu_mib: float) -> dict:
    while _available_gib() < 32.0 or _gpu_free_mib(gpu) < float(min_gpu_mib):
        time.sleep(20.0)
    log.parent.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{started}] GPU={gpu} {' '.join(command)}\n")
        handle.flush()
        process = subprocess.run(
            command, cwd=REPO, env=_environment(gpu),
            stdout=handle, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, text=True,
        )
    return {
        "command": command,
        "log": str(log),
        "gpu": int(gpu),
        "started_utc": started,
        "finished_utc": utc_now(),
        "returncode": int(process.returncode),
    }


def _baseline_matches_checkpoint(subject: str, baseline_path: Path) -> bool:
    checkpoints = sorted((R17B / "fits" / subject).glob("seed_*/model.pt"))
    if not checkpoints:
        raise ValueError(f"{subject}: no frozen R1.7B checkpoint for baseline audit")
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
    comparisons = []
    for checkpoint in checkpoints:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
        for group, prefix in (
                ("timing", "timing_baseline."), ("mark", "mark_baseline.")):
            for name, value in baseline[group]["history"].items():
                key = prefix + name
                comparisons.append(key in state and torch.equal(value, state[key]))
    return bool(comparisons and all(comparisons))


def _copy_coverage(subject: str, upstream: Path) -> None:
    target_root = upstream / "coverage"
    target_root.mkdir(parents=True, exist_ok=True)
    for suffix in (".npz", ".manifest.json"):
        source = R1_ROOT / "r1_2/coverage" / f"{subject}{suffix}"
        target = target_root / source.name
        if not source.is_file():
            raise FileNotFoundError(source)
        if not target.exists():
            shutil.copy2(source, target)
        if sha256_file(target) != sha256_file(source):
            raise ValueError(f"{subject}: copied coverage hash mismatch")


def prepare_subject(subject: str, root: Path, gpu: int) -> dict:
    upstream = root / "upstream_r1_2"
    verify_root = root / "upstream_r1_3_verification"
    expected_path = R17B / "cache" / subject / "manifest.json"
    expected = json.loads(expected_path.read_text())
    _copy_coverage(subject, upstream)
    commands = [
        (
            "baseline",
            upstream / "baselines" / subject / "seed_0/result.json",
            [str(PYTHON), "scripts/topic5_continuous_marked_state_r1/run_r1_2_baseline.py",
             "--subject", subject, "--seed", "0", "--device", "cuda",
             "--mark-batch-size", "512", "--output-root", str(upstream)],
            9000.0,
        ),
        (
            "bridge",
            upstream / "bridge_e1" / subject / "seed_0/result.json",
            [str(PYTHON), "scripts/topic5_continuous_marked_state_r1/run_r1_2_bridge.py",
             "--subject", subject, "--seed", "0", "--device", "cuda",
             "--anchor-batch-size", "2", "--max-train-anchors", "64",
             "--max-validation-anchors", "32", "--output-root", str(upstream)],
            9000.0,
        ),
        (
            "anchor_cache",
            upstream / "cache" / subject / "manifest.json",
            [str(PYTHON), "scripts/topic5_continuous_marked_state_r1/run_r1_2_cache.py",
             "--subject", subject, "--device", "cuda", "--anchor-batch-size", "4",
             "--output-root", str(upstream)],
            9000.0,
        ),
        (
            "observation_cache_verification",
            verify_root / "cache" / subject / "manifest.json",
            [str(PYTHON),
             "scripts/topic5_continuous_marked_state_r1/build_r1_3_observation_cache.py",
             "--subject", subject, "--r1-2-root", str(upstream),
             "--output-root", str(verify_root)],
            4000.0,
        ),
    ]
    history = []
    for stage, output, command, memory in commands:
        if _complete(output):
            history.append({"stage": stage, "skipped_complete": True,
                            "output": str(output)})
            continue
        result = _run(
            command, root / "logs/upstream_preparation" / f"{subject}_{stage}.log",
            gpu=gpu, min_gpu_mib=memory,
        )
        result.update({"stage": stage, "output": str(output)})
        history.append(result)
        if result["returncode"] != 0 or not _complete(output):
            return {"status": "FAIL", "subject": subject, "history": history}

    design = upstream / "cache" / subject / "full_design.npz"
    baseline = upstream / "baselines" / subject / "seed_0/models.pt"
    verified_explicit = verify_root / "cache" / subject / "explicit_normalised.npy"
    checks = {
        "design_matches_frozen_r1_7b": (
            sha256_file(design) == expected["design_sha256"]
        ),
        "normalised_explicit_matches_frozen_r1_7b": (
            sha256_file(verified_explicit) == expected["explicit_sha256"]
        ),
        "history_baseline_bitwise_matches_frozen_r1_7b_checkpoint": (
            _baseline_matches_checkpoint(subject, baseline)
        ),
    }
    status = "COMPLETE" if all(checks.values()) else "FAIL"
    payload = {
        "status": status,
        "revision": "h2b_v0_2_upstream_rebuild_and_frozen_equivalence_v1",
        "created_utc": utc_now(),
        "subject": subject,
        "gpu": int(gpu),
        "checks": checks,
        "history": history,
        "artifacts": {
            "design": str(design), "design_sha256": sha256_file(design),
            "baseline": str(baseline), "baseline_sha256": sha256_file(baseline),
            "explicit_scaler_result": str(
                upstream / "bridge_e1" / subject / "seed_0/result.json"
            ),
            "explicit_scaler_result_sha256": sha256_file(
                upstream / "bridge_e1" / subject / "seed_0/result.json"
            ),
            "verification_explicit": str(verified_explicit),
            "verification_explicit_sha256": sha256_file(verified_explicit),
        },
        "formal_test_partition_opened": False,
        "sealed_opened": False,
    }
    atomic_json(root / "manifests/upstream_rebuild" / f"{subject}.json", payload)
    return payload


def prepare_gpu_queue(gpu: int, subjects: list[str], root: Path) -> list[dict]:
    """Run one strictly serial queue owned by one physical GPU."""
    rows = []
    for subject in subjects:
        try:
            rows.append(prepare_subject(subject, root, gpu))
        except Exception as exc:
            rows.append({"status": "FAIL", "subject": subject, "error": repr(exc)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--result-root", type=Path, default=V0_2_RESULT_ROOT)
    parser.add_argument("--gpus", type=int, nargs="+", default=(0, 1))
    args = parser.parse_args()
    root = args.result_root.resolve()
    gpu_ids = list(dict.fromkeys(map(int, args.gpus)))
    assignments = {gpu: [] for gpu in gpu_ids}
    for index, subject in enumerate(args.subjects):
        assignments[gpu_ids[index % len(gpu_ids)]].append(subject)
    rows = []
    # One worker per physical GPU prevents the free-memory check from racing
    # another producer onto the same device.
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        futures = {
            pool.submit(prepare_gpu_queue, gpu, subjects, root): gpu
            for gpu, subjects in assignments.items() if subjects
        }
        for future in as_completed(futures):
            try:
                rows.extend(future.result())
            except Exception as exc:
                rows.append({
                    "status": "FAIL", "gpu": futures[future],
                    "error": repr(exc),
                })
    summary = {
        "status": "COMPLETE" if all(row.get("status") == "COMPLETE" for row in rows) else "FAIL",
        "revision": "h2b_v0_2_upstream_rebuild_queue_v1",
        "created_utc": utc_now(),
        "subjects": list(args.subjects),
        "gpu_workers": gpu_ids,
        "one_producer_per_gpu": True,
        "rows": sorted(rows, key=lambda row: str(row.get("subject"))),
        "formal_test_partition_opened": False,
        "sealed_opened": False,
    }
    atomic_json(root / "UPSTREAM_PREPARATION_STATUS.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
