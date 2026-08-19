#!/usr/bin/env python3
"""Engineering accessibility probe for the patient-specific field search.

Generation 0 of the frozen cohort produced 10/10 ``INVALID_RUNAWAY`` candidates,
so the optimizer received a constant objective and no ranking signal. This probe
asks a purely engineering question before any search parameter is re-frozen:

  Is the runaway caused by the amplitude of the drawn node field, or by the
  local E-source edge redistribution that rides on it?

It walks one amplitude ladder ``alpha`` on two generation-0 base fields with
contrasting core geometry, plus edge-off controls at full amplitude and a flat
field floor. Nothing here feeds selection: outputs live in their own directory,
run on training blocks only, and are labelled as calibration.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic4_patient_specific_field_cohort import (  # noqa: E402
    atomic_json,
    candidate_from_vector,
    load_config,
    projected_field_basis,
)

DEFAULT_CONFIG = ROOT / "config/topic4_patient_specific_field_connectivity_cohort_v2.json"
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
SUBJECT = "epilepsiae_590"
SEED = 1901

# (label, base candidate suffix or None for a flat field, alpha, keep edge terms)
ARMS = (
    ("flat_edge_off", None, 0.0, False),
    ("disperse_a100_edge_off", "c00", 1.0, False),
    ("compact_a100_edge_off", "c09", 1.0, False),
    ("disperse_a060", "c00", 0.60, True),
    ("disperse_a040", "c00", 0.40, True),
    ("disperse_a025", "c00", 0.25, True),
    ("compact_a060", "c09", 0.60, True),
    ("compact_a040", "c09", 0.40, True),
    ("compact_a025", "c09", 0.25, True),
)


def _mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024.0 / 1024.0
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


def _base_vectors(config: dict, cohort_root: Path) -> dict:
    """Read the generation-0 optimizer vectors this probe rescales."""
    root = cohort_root / "per_subject" / SUBJECT / "candidates"
    vectors = {}
    for path in sorted(root.glob("*.json")):
        payload = json.loads(path.read_text())
        key = payload["candidate_id"].split("_")[-2]
        vectors[key] = {
            "field": np.asarray(payload["field_coordinates"], float),
            "edge": np.asarray(payload["edge_coordinates"], float),
            "candidate_id": payload["candidate_id"],
        }
    return vectors


def build_candidates(config: dict, basis: dict, cohort_root: Path,
                     output: Path) -> list[dict]:
    bases = _base_vectors(config, cohort_root)
    directory = output / "candidates"
    directory.mkdir(parents=True, exist_ok=True)
    jobs = []
    for index, (label, base_key, alpha, keep_edge) in enumerate(ARMS):
        if base_key is None:
            field = np.zeros(int(basis["direction_count"]), float)
            edge = np.zeros(int(config["local_connectivity"]["coefficient_count"]), float)
            source = "flat_uniform_field"
        else:
            base = bases[base_key]
            field = float(alpha) * base["field"]
            edge = base["edge"].copy() if keep_edge else np.zeros_like(base["edge"])
            source = base["candidate_id"]
        candidate = candidate_from_vector(
            SUBJECT, np.concatenate([field, edge]), config, basis,
            generation=0, candidate_index=index, restart=0,
        )
        candidate["candidate_id"] = f"calib_{label}"
        candidate["node_field"]["candidate_id"] = f"calib_{label}"
        candidate["calibration"] = {
            "role": "engineering_accessibility_probe_not_selection",
            "arm": label, "alpha": float(alpha), "edge_terms_active": bool(keep_edge),
            "base_candidate_id": source,
        }
        path = directory / f"calib_{label}.json"
        atomic_json(candidate, path)
        jobs.append({"label": label, "candidate_json": path})
    return jobs


def run_probe(job: dict, config_path: Path, output: Path, expected_commit: str,
              gate: threading.Semaphore, floor_gib: float) -> dict:
    out_json = output / "workers" / f"calib_{job['label']}_seed_{SEED}.json"
    out_npz = out_json.with_suffix(".npz")
    log_path = output / "run_logs" / f"calib_{job['label']}_seed_{SEED}.log"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if out_json.exists() and out_npz.exists():
        return json.loads(out_json.read_text())
    with gate:
        while _mem_available_gib() < floor_gib:
            time.sleep(120)
        environment = os.environ.copy()
        for name in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
            environment[f"{name}_NUM_THREADS"] = "1"
        command = [
            str(PYTHON), str(ROOT / "scripts/run_topic4_patient_specific_field_worker_v2.py"),
            "--config", str(config_path), "--subject-id", SUBJECT,
            "--candidate-json", str(job["candidate_json"]), "--seed", str(SEED),
            "--phase", "canary", "--runtime-mode", "active_z_plus_m",
            "--expected-commit", expected_commit,
            "--out-json", str(out_json), "--out-npz", str(out_npz),
        ]
        with log_path.open("a") as log:
            result = subprocess.run(command, cwd=ROOT, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0 or not out_json.exists():
        return {"status": "PROBE_WORKER_FAILED", "label": job["label"],
                "log": str(log_path)}
    return json.loads(out_json.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--max-workers", type=int, default=9)
    parser.add_argument("--memory-floor-gib", type=float, default=30.0)
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    basis = projected_field_basis(config)
    cohort_root = Path(config["output_root"])
    output = cohort_root / "engineering_calibration"
    output.mkdir(parents=True, exist_ok=True)
    status = output / "calibration.status"
    expected_commit = subprocess.check_output(
        ["git", "rev-parse", args.expected_commit], cwd=ROOT, text=True).strip()

    jobs = build_candidates(config, basis, cohort_root, output)
    status.write_text(f"RUNNING arms={len(jobs)} workers={args.max_workers}\n")
    gate = threading.Semaphore(int(args.max_workers))
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = [pool.submit(run_probe, job, args.config.resolve(), output,
                               expected_commit, gate, float(args.memory_floor_gib))
                   for job in jobs]
        payloads = [future.result() for future in futures]

    rows = []
    for (label, base_key, alpha, keep_edge), payload in zip(ARMS, payloads):
        rows.append({
            "arm": label,
            "base_candidate": base_key,
            "alpha": float(alpha),
            "edge_terms_active": bool(keep_edge),
            "status": payload.get("status"),
            "runaway": payload.get("runaway"),
            "runaway_early_stop_ms": payload.get("runaway_early_stop_ms"),
            "simulated_until_ms": payload.get("simulated_until_ms"),
            "n_returned_events": payload.get("n_returned_events"),
            "n_readable_events": (payload.get("score") or {}).get("n_readable_events"),
            "score_status": (payload.get("score") or {}).get("status"),
            "objective": (payload.get("objective") or {}).get("objective"),
            "wall_seconds": payload.get("wall_seconds"),
        })
    atomic_json({
        "status": "ENGINEERING_ACCESSIBILITY_CALIBRATION_COMPLETE",
        "scientific_role": "engineering_accessibility_probe_not_selection",
        "subject_id": SUBJECT, "network_seed": SEED, "target_split": "train",
        "question": "does runaway follow node-field amplitude or the E-source edge redistribution",
        "expected_git_commit": expected_commit,
        "arms": rows,
    }, output / "CALIBRATION_RESULT.json")
    status.write_text("DONE\n")
    subprocess.run(["notify-send", "Topic 4 patient-specific calibration",
                    "Engineering accessibility probe complete"], check=False)


if __name__ == "__main__":
    main()
