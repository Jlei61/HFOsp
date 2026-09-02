#!/usr/bin/env python3
"""Wait for repair workers, then run all deterministic finalisation steps."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python"
RESULT = ROOT / "results/topic5_dynamical_motif_rnn_v0_1"
CONTACT = RESULT / "contact_selected/GEOMETRY_ONLY_PCA2"
FORMAL = RESULT / "formal/GEOMETRY_ONLY_PCA2"
STATUS = RESULT / "repair_v0_2/FINALIZER_STATUS.json"


def count(pattern: str, root: Path) -> int:
    return sum(1 for _ in root.glob(pattern)) if root.exists() else 0


def write_status(payload: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(STATUS)


def run(label: str, command: list[str]) -> None:
    print(f"[repair-finalizer] {label}", flush=True)
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode:
        raise RuntimeError(f"{label} failed with return code {completed.returncode}")


def main() -> None:
    started = time.time()
    while True:
        contact_done = count("*/*/seed0/DONE.json", CONTACT)
        static_done = count("*/capacity_matched_static_seed*.json", FORMAL)
        failed = count("*/*/seed0/FAILED.json", CONTACT)
        write_status({
            "state": "WAITING", "contact_done": contact_done,
            "contact_expected": 112, "static_done": static_done,
            "static_expected": 84, "contact_failed": failed,
            "elapsed_s": time.time() - started,
        })
        print(f"[repair-finalizer] contact={contact_done}/112 static={static_done}/84 "
              f"failed={failed}", flush=True)
        if failed:
            raise RuntimeError("contact-selected worker failure detected")
        if contact_done == 112 and static_done == 84:
            break
        time.sleep(20)

    steps = [
        ("operator visibility", [
            PYTHON, "scripts/run_topic5_dynamical_motif_operator_visibility_v0_1.py",
            "--device", "cuda:0"]),
        ("hard-transition and capacity analysis", [
            PYTHON, "scripts/analyze_topic5_dynamical_motif_repairs_v0_1.py",
            "--tags", "formal", "contact_selected", "--device", "cuda:0"]),
        ("paper-ready figure", [
            PYTHON, "scripts/paper_figures/plot_topic5_dynamical_motif_rnn_v0_2.py"]),
        ("contract tests", [
            PYTHON, "-m", "pytest", "tests/test_topic5_dynamical_motif_rnn_v0_1.py", "-q"]),
    ]
    completed = []
    try:
        for label, command in steps:
            write_status({"state": "RUNNING", "step": label,
                          "completed": completed, "elapsed_s": time.time() - started})
            run(label, command)
            completed.append(label)
        write_status({"state": "DONE", "completed": completed,
                      "elapsed_s": time.time() - started})
        print("[repair-finalizer] DONE", flush=True)
    except Exception as error:
        write_status({"state": "FAILED", "completed": completed,
                      "error": f"{type(error).__name__}: {error}",
                      "elapsed_s": time.time() - started})
        raise


if __name__ == "__main__":
    main()
