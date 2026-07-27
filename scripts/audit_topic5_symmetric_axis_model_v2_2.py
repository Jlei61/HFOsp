#!/usr/bin/env python3
"""Run the frozen analytic model tests and write a Milestone B gate."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "results/topic5_symmetric_axis_propagation_state_v2_2/model_audit"
)
TESTS = [
    "tests/test_topic5_symmetric_axis_operator_v2_2.py",
    "tests/test_topic5_propagation_state_recurrence_v2_2.py",
    "tests/test_topic5_absorbing_rollout_v2_2.py",
    "tests/test_topic5_symmetric_axis_leakage_v2_2.py",
    "tests/test_topic5_symmetric_axis_aggregation_v2_2.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> None:
    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "cuda_env",
        "pytest",
        "-q",
        *TESTS,
    ]
    completed = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    output = (completed.stdout + completed.stderr).strip()
    gate = {
        "contract": "topic5_symmetric_axis_propagation_state_rnn",
        "version": "2.2",
        "status": "pass" if completed.returncode == 0 else "fail",
        "pytest_returncode": completed.returncode,
        "pytest_output": output,
        "scientific_contract": {
            "symmetric_operator_only": True,
            "axis_sign_invariant": True,
            "opposite_source_displacement_tested": True,
            "geometry_incomplete_rejected": True,
            "one_propagation_state_only": True,
            "scalar_stop_only": True,
            "exact_nonempty_set_likelihood": True,
            "absorbing_rollout_mass_conserved": True,
            "no_dense_bypass": True,
            "no_future_head": True,
            "event_reset": True,
            "train_only_node_hazard_bias": True,
            "event_first_aggregation": True,
            "cpu_gpu_operator_checked": True,
            "ictal_target_values_read": False,
        },
        "files": {
            path: sha256(ROOT / path)
            for path in [
                "src/topic5_symmetric_axis_propagation_state_v2_2.py",
                "scripts/train_topic5_symmetric_axis_propagation_state_v2_2.py",
                *TESTS,
            ]
        },
    }
    atomic_json(OUTPUT / "MATHEMATICAL_MODEL_GATE.json", gate)
    print(json.dumps(gate, indent=2))
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
