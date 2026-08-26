#!/usr/bin/env python3
"""Autonomous chain for the v0.2 arrival pivot (2026-08-19).

Runs the 80-task plan, then the two model-light analyses that need no training,
then re-aggregates everything the batch touches.  A failing stage never stops the
chain: it is logged and the remaining stages still run.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.topic5_epi_prssm.contracts import OUTPUT_ROOT, atomic_write_json  # noqa: E402

HERE = Path(__file__).resolve().parent
PY_EXE = sys.executable
LOG = OUTPUT_ROOT / "logs/repair_chain.log"
STATUS = OUTPUT_ROOT / "manifests/REPAIR_CHAIN_STATUS.json"


def log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    with LOG.open("a") as handle:
        handle.write(f"{stamp} [repair] {message}\n")
    print(f"{stamp} [repair] {message}", flush=True)


def status(stage: str, state: str, extra: dict | None = None) -> None:
    payload = {}
    if STATUS.exists():
        payload = json.loads(STATUS.read_text())
    payload[stage] = {"state": state, "at": time.time(), **(extra or {})}
    atomic_write_json(STATUS, payload)


def step(name: str, args: list[str], *, timeout: float | None = None) -> bool:
    log(f"step {name}: {' '.join(str(a) for a in args)}")
    status(name, "RUNNING")
    try:
        result = subprocess.run([PY_EXE, str(HERE / args[0])] + [str(a) for a in args[1:]],
                                cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        log(f"step {name}: TIMEOUT")
        status(name, "TIMEOUT")
        return False
    if result.returncode == 0:
        log(f"step {name}: COMPLETE")
        status(name, "COMPLETE")
        return True
    tail = "\n".join((result.stderr or result.stdout or "").strip().splitlines()[-12:])
    log(f"step {name}: FAILED rc={result.returncode}\n{tail}")
    status(name, "FAILED", {"rc": result.returncode, "tail": tail})
    return False


def main() -> None:
    log("repair chain start")
    plan = OUTPUT_ROOT / "manifests/plans/v0_2_arrival_pivot.json"

    # the training batch; the launcher owns concurrency and is itself resumable
    step("batch", ["launch_autonomous.py", "--plan", str(plan), "--cap", "14",
                   "--tag", "repair"])

    # model-light analyses: no training, so they run whatever the batch did
    step("lag_discovery", ["run_exposure_lag_discovery.py", "--cohort", "all34",
                           "--weighting", "both", "--overwrite"])
    step("innovation_frozen", ["run_innovation_controls.py", "--cohort", "all34",
                               "--overwrite", "--tag", "frozen_tau",
                               "--taus", "30,60,120,300,600,1200,1800,3600,7200,"
                                         "14400,28800,86400"])

    # re-aggregate everything the batch can change
    for name, argv in (
        ("aggregate_synthetic", ["aggregate_synthetic.py"]),
        ("aggregate_event_distribution", ["aggregate_event_distribution.py",
                                          "--cohort", "all34"]),
        ("aggregate_exposure", ["aggregate_exposure.py", "--cohort", "all34"]),
        ("aggregate_graph_null", ["aggregate_graph_null.py", "--cohort", "all34"]),
        ("aggregate_arrival", ["aggregate_arrival.py", "--cohort", "all34"]),
    ):
        step(name, argv)

    log("repair chain done")
    status("chain", "COMPLETE")


if __name__ == "__main__":
    main()
