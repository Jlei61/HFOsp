"""Launch the initial spectral-field library with bounded systemd workers."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
MANAGER = ROOT / "scripts/run_topic4_rev10_sa_managed_command.sh"
FREEZER = ROOT / "scripts/freeze_topic4_rev10_sa_spectral_field_candidates.py"
V3_FREEZER = ROOT / "scripts/freeze_topic4_rev10_sa_spectral_field_v3_candidates.py"
V4_FREEZER = ROOT / "scripts/freeze_topic4_rev10_sa_spline_field_v4_candidates.py"
V41_FREEZER = ROOT / "scripts/freeze_topic4_rev10_sa_spline_bridge_v41_candidates.py"
V5_FREEZER = ROOT / "scripts/freeze_topic4_rev10_sa_spline_interpolation_v5_candidates.py"
V51_FREEZER = ROOT / "scripts/freeze_topic4_rev10_sa_v5_selection_candidates.py"
V52_FREEZER = ROOT / "scripts/freeze_topic4_rev10_sa_v52_final_candidates.py"
WORKER = ROOT / "scripts/run_topic4_rev10_sa_spectral_field_worker.py"
AGGREGATOR = ROOT / "scripts/aggregate_topic4_rev10_sa_spectral_field_search.py"
V4_AGGREGATOR = ROOT / "scripts/aggregate_topic4_rev10_sa_spline_field_search.py"
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_sa_observation_invariant_field.json"
NUMERIC_ENV = {
    "BLIS_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _state(path):
    if not path.exists():
        return "MISSING"
    return path.read_text().strip().split(maxsplit=1)[0]


def _complete(job, *, config_sha, commit):
    if (_state(job["status"]) != "SUCCESS" or not job["json"].exists()
            or not job["npz"].exists()):
        return False
    try:
        payload = json.loads(job["json"].read_text())
    except (OSError, json.JSONDecodeError):
        return False
    provenance = payload.get("provenance", {})
    return bool(
        payload.get("status") == "REV10SA_SPECTRAL_FIELD_WORKER_COMPLETE"
        and payload.get("config", {}).get("sha256") == config_sha
        and provenance.get("expected_git_commit") == commit
        and provenance.get("runtime_modules_match_expected_commit") is True
        and not provenance.get("runtime_modules_dirty")
        and payload.get("arrays", {}).get("sha256") == _sha256(job["npz"])
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--commit", required=True)
    parser.add_argument("--max-concurrent", type=int)
    parser.add_argument("--wait-seconds", type=float)
    parser.add_argument("--unit-prefix", default="topic4-rev10sa-spectral")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    role = config["scientific_role"]
    if role == "development_only_observation_invariant_uniform_allocation_refinement":
        freezer, aggregator = V3_FREEZER, AGGREGATOR
    elif role == "development_only_stable_spline_random_field_screen":
        freezer, aggregator = V4_FREEZER, V4_AGGREGATOR
    elif role == "development_only_v3_to_stable_spline_bridge":
        freezer, aggregator = V41_FREEZER, V4_AGGREGATOR
    elif role == "development_only_stable_spline_adaptive_interpolation":
        freezer, aggregator = V5_FREEZER, V4_AGGREGATOR
    elif role == "development_only_stable_spline_selection_confirmation":
        freezer, aggregator = V51_FREEZER, V4_AGGREGATOR
    elif role == "development_only_stable_spline_final_confirmation":
        freezer, aggregator = V52_FREEZER, V4_AGGREGATOR
    else:
        freezer, aggregator = FREEZER, AGGREGATOR
    config_sha = _sha256(config_path)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    commit = subprocess.check_output(
        ["git", "rev-parse", args.commit], cwd=ROOT, text=True,
    ).strip()
    if head != commit:
        raise RuntimeError(f"launcher commit {commit} is not current HEAD {head}")

    output_root = ROOT / config["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "candidate_manifest.json"
    subprocess.run([
        str(PYTHON), str(freezer), "--config", str(config_path),
        "--expected-commit", commit, "--out", str(manifest_path),
    ], cwd=ROOT, check=True, env={**os.environ, **NUMERIC_ENV})
    manifest = json.loads(manifest_path.read_text())
    candidates = [
        row["candidate_id"] for row in manifest["candidate_set"]["candidates"]
    ]
    seeds = [int(value) for value in config["search"]["network_seeds"]]
    maximum = int(args.max_concurrent or min(
        len(candidates) * len(seeds), config["execution"]["max_workers"],
    ))
    wait_seconds = float(args.wait_seconds or config["execution"]["wait_seconds"])
    worker_dir, run_dir = output_root / "workers", output_root / "run_logs"
    worker_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for candidate in candidates:
        for seed in seeds:
            stem = worker_dir / f"{candidate}_seed_{seed}"
            jobs.append({
                "candidate": candidate, "seed": seed,
                "json": stem.with_suffix(".json"),
                "npz": stem.with_suffix(".npz"),
                "status": run_dir / f"{candidate}_seed_{seed}.status",
                "log": run_dir / f"{candidate}_seed_{seed}.log",
            })
    pending, active, completed, failed = [], [], [], []
    for job in jobs:
        if _complete(job, config_sha=config_sha, commit=commit):
            completed.append(job)
        elif _state(job["status"]) == "RUNNING":
            active.append(job)
        else:
            pending.append(job)

    def refresh():
        nonlocal active
        remaining = []
        for job in active:
            if _complete(job, config_sha=config_sha, commit=commit):
                completed.append(job)
            elif _state(job["status"]) == "FAILED":
                failed.append(job)
            else:
                remaining.append(job)
        active = remaining

    while pending or active:
        refresh()
        while pending and len(active) < maximum:
            job = pending.pop(0)
            unit = (f"{args.unit_prefix}-{job['candidate']}-s{job['seed']}-"
                    f"{commit[:8]}")
            command = [
                "systemd-run", "--user", "--collect", f"--unit={unit}",
                "--property=Type=exec", f"--working-directory={ROOT}",
                f"--setenv=REV10SA_SYSTEMD_UNIT={unit}",
                *[f"--setenv={key}={value}" for key, value in NUMERIC_ENV.items()],
                "/usr/bin/nohup", str(MANAGER), str(job["status"]),
                str(job["log"]),
                f"spectral {job['candidate']} seed={job['seed']}", commit[:8],
                str(PYTHON), str(WORKER), "--config", str(config_path),
                "--candidate-id", job["candidate"], "--seed", str(job["seed"]),
                "--expected-commit", commit,
                "--out-json", str(job["json"]), "--out-npz", str(job["npz"]),
            ]
            subprocess.run(command, cwd=ROOT, check=True)
            active.append(job)
            print(json.dumps({
                "progress": "launched", "candidate": job["candidate"],
                "seed": job["seed"], "active": len(active),
                "pending": len(pending),
            }), flush=True)
        if active:
            time.sleep(wait_seconds)
    refresh()
    if failed:
        raise RuntimeError(f"{len(failed)} spectral worker(s) failed")
    subprocess.run([
        str(PYTHON), str(aggregator), "--config", str(config_path),
        "--expected-commit", commit,
    ], cwd=ROOT, check=True, env={**os.environ, **NUMERIC_ENV})
    subprocess.run([
        "notify-send", "Topic 4 rev10-SA",
        f"Observation-invariant spectral search completed: {len(completed)}/{len(jobs)}",
    ], check=False)
    print(json.dumps({
        "status": "REV10SA_SPECTRAL_SEARCH_LAUNCH_COMPLETE",
        "commit": commit, "n_success": len(completed), "n_failed": 0,
        "n_candidates": len(candidates), "n_seeds": len(seeds),
        "max_concurrent": maximum, "wait_seconds": wait_seconds,
    }, indent=2))


if __name__ == "__main__":
    main()
