#!/usr/bin/env python3
"""Durable finalizer for the R1.4/T2-R2.0/H2b goal."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time

from src.topic5_continuous_marked_state_r1 import contract


PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
T2 = contract.RESULT_ROOT / "t2_r2"
H2B = contract.UPSTREAM_ROOT
STATUS = T2 / "FINALIZER_STATUS.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic(stage: str, state: str = "RUNNING", **extra) -> None:
    contract.atomic_json(STATUS, {
        "status": state, "stage": stage, "updated_at": now(),
        "formal_test_partition_opened": False, "sealed_opened": False,
        **extra,
    })


def environment() -> dict[str, str]:
    value = os.environ.copy()
    conda_lib = str(PYTHON.parent.parent / "lib")
    inherited_library_path = value.get("LD_LIBRARY_PATH", "")
    value.update({
        "PYTHONPATH": str(contract.REPO_ROOT), "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "0",
        "LD_LIBRARY_PATH": (
            conda_lib
            if not inherited_library_path
            else f"{conda_lib}:{inherited_library_path}"
        ),
    })
    return value


def run(command: list[str], log: Path) -> subprocess.CompletedProcess:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        handle.write(f"[{now()}] {' '.join(command)}\n")
        return subprocess.run(
            command, cwd=contract.REPO_ROOT, env=environment(),
            stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            text=True,
        )


def read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def main() -> None:
    deadline = time.time() + 14 * 3600
    t2_status = T2 / "PIPELINE_STATUS.json"
    h2b_status = H2B / "manifests/H2B_PSEUDO_FIX_RERUN_STATUS.json"
    atomic("waiting")
    while True:
        t2 = read(t2_status) or {}
        h2b = read(h2b_status) or {}
        if t2.get("status") in {"FAIL", "FAILED"}:
            atomic("t2_failed", "FAIL", t2=t2)
            raise RuntimeError("T2 pipeline failed")
        if h2b.get("status") in {"FAIL", "FAILED"}:
            atomic("h2b_failed", "FAIL", h2b=h2b)
            raise RuntimeError("H2b rerun failed")
        if t2.get("status") == "COMPLETE" and h2b.get("status") == "COMPLETE":
            break
        if time.time() >= deadline:
            atomic("timeout", "FAIL", t2=t2, h2b=h2b)
            raise TimeoutError("goal outputs did not finish inside 14 h")
        time.sleep(60)

    atomic("synthetic_refresh")
    synthetic_log = T2 / "logs/final_synthetic_refresh.log"
    value = run([
        str(PYTHON),
        "scripts/topic5_continuous_marked_state_r1/run_t2_r2_synthetic.py",
        "--device", "cpu", "--output-root", str(T2),
    ], synthetic_log)
    if value.returncode:
        atomic("synthetic_failed", "FAIL", log=str(synthetic_log))
        raise RuntimeError("final synthetic refresh failed")

    atomic("tests")
    test_log = T2 / "logs/final_tests.log"
    test_command = [
        str(PYTHON), "-m", "pytest", "-q",
        "tests/test_raw_seeg_state_conformer.py",
        "tests/test_raw_seeg_state_data_contract.py",
        "tests/test_raw_seeg_state_io.py",
        "tests/test_raw_seeg_state_model.py",
        "tests/test_raw_seeg_state_train.py",
        "tests/topic5_continuous_marked_state",
        "tests/topic5_continuous_marked_state_r1",
        "tests/topic5_epi_prssm",
        "tests/test_topic5_epi_prssm_h2b_pseudo_fix_runner.py",
    ]
    tested = run(test_command, test_log)
    content = test_log.read_text()
    matches = re.findall(r"(\d+) passed", content)
    test_audit = {
        "status": "COMPLETE" if tested.returncode == 0 else "FAIL",
        "returncode": int(tested.returncode),
        "passed": int(matches[-1]) if matches else None,
        "command": " ".join(test_command),
        "log": str(test_log),
        "formal_test_partition_opened": False, "sealed_opened": False,
    }
    contract.atomic_json(T2 / "FINAL_TEST_AUDIT.json", test_audit)
    if tested.returncode:
        atomic("tests_failed", "FAIL", audit=test_audit)
        raise RuntimeError("final test suite failed")

    atomic("reports", tests=test_audit)
    report_log = T2 / "logs/final_reports.log"
    reported = run([
        str(PYTHON),
        "scripts/topic5_continuous_marked_state_r1/finalize_r1_4_t2_r2_reports.py",
    ], report_log)
    if reported.returncode:
        atomic("reports_failed", "FAIL", log=str(report_log))
        raise RuntimeError("final report generation failed")
    atomic("complete", "COMPLETE", tests=test_audit, reports={
        "plain": str(contract.RESULT_ROOT / "final_reports/r1_4_t2_r2_h2b_plain_2026-08-27.md"),
        "technical": str(contract.RESULT_ROOT / "final_reports/r1_4_t2_r2_h2b_technical_2026-08-27.md"),
        "audit": str(contract.RESULT_ROOT / "final_reports/r1_4_t2_r2_h2b_machine_audit.json"),
    })


if __name__ == "__main__":
    main()
