#!/usr/bin/env python3
"""Persistent gate-aware supervisor for the formal Topic-5 v2.2 pipeline."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, TextIO


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/topic5_symmetric_axis_propagation_state_v2_2"
FORMAL = BASE / "formal"
ANALYSIS = FORMAL / "analysis"
LOGS = FORMAL / "launcher_logs"
STATE = FORMAL / "PIPELINE_SUPERVISOR_STATE.json"
POLL_SECONDS = 300


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def update(stage: str, **extra: Any) -> None:
    atomic_json(
        STATE,
        {
            "status": "RUNNING",
            "stage": stage,
            "pid": os.getpid(),
            "updated_unix": time.time(),
            "target_values_read": False,
            **extra,
        },
    )
    print(json.dumps({"stage": stage, **extra}), flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_logged(command: list[str], log_name: str) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    path = LOGS / log_name
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
            + " ".join(command)
            + "\n"
        )
        handle.flush()
        subprocess.run(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
        )


def require_static_contracts() -> None:
    epoch = read_json(FORMAL / "TRAINER_EPOCH_AUDIT.json")
    sequence = read_json(ANALYSIS / "ALL_SUBJECT_SEQUENCE_STATUS.json")
    target = read_json(BASE / "target_audit/TARGET_METADATA_GATE.json")
    if epoch.get("status") != "PASS":
        raise RuntimeError("formal epoch audit is not PASS")
    if sequence.get("status") != "complete" or sequence.get("n_patients") != 31:
        raise RuntimeError("31-patient sequence sensitivity is incomplete")
    if (
        target.get("energy_values_read")
        or target.get("recruitment_values_read")
        or target.get("early_ictal_transfer_allowed")
    ):
        raise RuntimeError("target seal drifted before interictal gates")


def wait_for_claim2() -> dict[str, Any]:
    while True:
        status_path = ANALYSIS / "CLAIM2_STATUS.json"
        status = read_json(status_path)
        if status.get("status") == "complete":
            return status
        run_states = list((FORMAL / "claim2_runs").glob("*/seed_*/run_state.json"))
        failed = []
        complete = 0
        for path in run_states:
            record = read_json(path)
            if record.get("status") == "FAILED":
                failed.append(str(path.parent.relative_to(ROOT)))
            if record.get("status") == "COMPLETE":
                complete += 1
        if failed:
            raise RuntimeError(f"Claim-2 failed runs: {failed}")
        if complete == 66:
            update("analyze_claim2", complete=complete, expected=66)
            run_logged(
                [
                    "conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    "cuda_env",
                    "python",
                    "scripts/analyze_topic5_symmetric_axis_formal_claim2_v2_2.py",
                ],
                "claim2_analyzer_supervisor_fallback.log",
            )
            continue
        update("wait_claim2", complete=complete, expected=66)
        time.sleep(POLL_SECONDS)


def _nvidia_total_gb() -> float:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return float(output.strip().splitlines()[0]) / 1024.0


def benchmark_claim3() -> dict[str, Any]:
    lock = read_json(FORMAL / "PHYSICAL_AXIS_FORMAL_LOCK.json")
    subject = str(lock["subjects"][0])
    seed = int(lock["seeds"][0])
    run = (
        FORMAL
        / "claim3_random_axis_runs"
        / subject
        / f"seed_{seed}"
        / "chunk_000_031"
    )
    if not (run / "COMPLETE").is_file():
        run_logged(
            [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                "cuda_env",
                "python",
                "scripts/train_topic5_symmetric_axis_formal_claim3_v2_2.py",
                "--subject",
                subject,
                "--seed",
                str(seed),
                "--direction-start",
                "0",
                "--direction-stop",
                "32",
            ],
            "claim3_single_chunk_benchmark.log",
        )
    metrics = read_json(run / "metrics.json")
    peak = float(metrics["resource"]["peak_cuda_reserved_gb"])
    total = _nvidia_total_gb()
    max_by_vram = max(1, int((0.80 * total) // max(peak, 0.1)))
    jobs = min(4, max_by_vram)
    return {
        "subject": subject,
        "seed": seed,
        "directions": 32,
        "peak_cuda_reserved_gb": peak,
        "gpu_total_gb": total,
        "max_jobs_by_80pct_vram": max_by_vram,
        "selected_jobs": jobs,
    }


def finalize(reason: str) -> None:
    update("finalize_interictal_gate", reason=reason)
    run_logged(
        [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "cuda_env",
            "python",
            "scripts/finalize_topic5_interictal_gate_v2_2.py",
        ],
        "interictal_gate_finalize.log",
    )
    summary = read_json(ANALYSIS / "INTERICTAL_CLAIM_SUMMARY.json")
    atomic_json(
        STATE,
        {
            "status": "COMPLETE",
            "stage": "interictal_pipeline_complete",
            "reason": reason,
            "early_ictal_values_unlocked": summary[
                "early_ictal_values_unlocked"
            ],
            "source_metadata_ready": summary["source_metadata_ready"],
            "target_values_read": False,
            "finished_unix": time.time(),
        },
    )


def main() -> None:
    require_static_contracts()
    claim2 = wait_for_claim2()
    if not claim2.get("next_stage_allowed"):
        finalize("claim2_gate_failed")
        return

    update("prepare_claim3_random_axes")
    run_logged(
        [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "cuda_env",
            "python",
            "scripts/prepare_topic5_symmetric_axis_claim3_nulls_v2_2.py",
        ],
        "claim3_prepare_nulls.log",
    )
    update("benchmark_claim3")
    benchmark = benchmark_claim3()
    atomic_json(FORMAL / "CLAIM3_RESOURCE_BENCHMARK.json", benchmark)
    update("run_claim3", **benchmark)
    run_logged(
        [
            "bash",
            "scripts/run_topic5_symmetric_axis_formal_claim3_v2_2.sh",
            "--jobs",
            str(benchmark["selected_jobs"]),
        ],
        "claim3_launcher_supervised.log",
    )
    claim3 = read_json(ANALYSIS / "CLAIM3_STATUS.json")
    if not claim3.get("next_stage_allowed"):
        finalize("claim3_gate_failed")
        return

    update("run_claim4", jobs=6)
    run_logged(
        [
            "bash",
            "scripts/run_topic5_symmetric_axis_formal_claim4_v2_2.sh",
            "--jobs",
            "6",
        ],
        "claim4_launcher_supervised.log",
    )
    claim4 = read_json(ANALYSIS / "CLAIM4_STATUS.json")
    finalize(
        "claim4_complete_pass"
        if claim4.get("next_stage_allowed")
        else "claim4_gate_failed_or_not_estimable"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        atomic_json(
            STATE,
            {
                "status": "FAILED",
                "stage": "supervisor_exception",
                "error": repr(exc),
                "target_values_read": False,
                "finished_unix": time.time(),
            },
        )
        raise
