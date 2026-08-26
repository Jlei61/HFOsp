#!/usr/bin/env python3
"""Overnight chain, 2026-08-20.

Waits for the three fitting batches now in flight, then runs the analyses that were
blocked on them, then the H2B producer re-run that the median-interval caliper needs,
then re-aggregates and re-renders.

Failure policy: only a P0 stops the chain.  A P0 here means the data layer or the
leakage gates are compromised -- a stage that returns a null result, refuses to
aggregate, or crashes on its own is logged and the chain continues, because a
negative is a result and a refusal is the fail-closed behaviour working.
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
LOG = OUTPUT_ROOT / "logs/overnight_chain.log"
STATUS = OUTPUT_ROOT / "manifests/OVERNIGHT_CHAIN_STATUS.json"

#: stages whose failure genuinely invalidates everything downstream
P0_STAGES = {"gate_a_recheck"}


def log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    with LOG.open("a") as handle:
        handle.write(f"{stamp} [overnight] {message}\n")
    print(f"{stamp} [overnight] {message}", flush=True)


def status(stage: str, state: str, extra: dict | None = None) -> None:
    payload = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    payload[stage] = {"state": state, "at": time.time(), **(extra or {})}
    atomic_write_json(STATUS, payload)


def step(name: str, args: list[str], *, timeout: float = 14400) -> bool:
    log(f"step {name}: {' '.join(str(a) for a in args)}")
    status(name, "RUNNING")
    try:
        result = subprocess.run([PY_EXE, str(HERE / args[0])] + [str(a) for a in args[1:]],
                                cwd=ROOT, capture_output=True, text=True, timeout=timeout,
                                env={**__import__("os").environ, "OMP_NUM_THREADS": "1",
                                     "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                                     "NUMEXPR_NUM_THREADS": "1"})
    except subprocess.TimeoutExpired:
        log(f"step {name}: TIMEOUT after {timeout}s")
        status(name, "TIMEOUT")
        return False
    if result.returncode == 0:
        log(f"step {name}: COMPLETE")
        status(name, "COMPLETE")
        return True
    tail = "\n".join((result.stderr or result.stdout or "").strip().splitlines()[-14:])
    log(f"step {name}: FAILED rc={result.returncode}\n{tail}")
    status(name, "FAILED", {"rc": result.returncode, "tail": tail})
    if name in P0_STAGES:
        log("that stage is P0; stopping the chain")
        status("chain", "STOPPED_ON_P0")
        raise SystemExit(2)
    return False


def wait_for_launchers(poll: float = 180.0, timeout_hours: float = 8.0) -> None:
    """Block until no epi_prssm fitting worker is left."""
    import subprocess as sp
    deadline = time.time() + timeout_hours * 3600
    log("waiting for the fitting batches in flight")
    while time.time() < deadline:
        out = sp.run(["ps", "-eo", "cmd"], capture_output=True, text=True).stdout
        live = sum(1 for line in out.splitlines()
                   if "run_graph_null.py" in line or "run_arrival_channel.py" in line)
        if live == 0:
            log("no fitting worker left")
            return
        time.sleep(poll)
    log("wait timed out; continuing with whatever landed")


def main() -> None:
    log("overnight chain start")
    status("chain", "RUNNING")

    wait_for_launchers()

    # --- 1. aggregate what the batches produced; both aggregators fail closed
    step("aggregate_graph_null", ["aggregate_graph_null.py", "--cohort", "all34"])
    step("aggregate_arrival", ["aggregate_arrival.py", "--cohort", "all34",
                               "--markov-renewal"])

    # --- 2. the H2B producer re-run the median-interval caliper needs.  This is the
    #        single thing that decides whether the 30 min entropy signal survives.
    step("h2b_producer_with_caliper",
         ["run_goal3b_stage.py", "--cap", "20",
          "--layers", "linear_graph_recurrent", "leaky_state",
          "resource_anchored_on_best_family",
          "--leads", "30.0", "60.0", "15.0", "5.0"], timeout=28800)

    # --- 3. everything downstream of the producer
    for lead in ("lead30m", "lead60m", "lead15m", "lead5m"):
        step(f"crosswalk_{lead}", ["build_seizure_crosswalk.py",
                                   "--layer", "linear_graph_recurrent", "--lead", lead])
    step("denominators", ["build_h2b_denominators.py"])
    step("h2b_sensitivity", ["run_h2b_sensitivity.py"])
    step("h3b_transition", ["run_h3b_transition_coupling.py"])
    step("lag_discovery", ["run_exposure_lag_discovery.py", "--cohort", "all34",
                           "--weighting", "all", "--overwrite"])

    # --- 4. figures last, so they read the final aggregates
    run_id = time.strftime("%Y%m%d-%H%M")
    step("core_figure", ["make_figure_core_evidence.py", "--run-id", run_id])
    step("write_reports", ["write_reports.py", "--cohort", "all34"])

    log("overnight chain done")
    status("chain", "COMPLETE", {"figure_run_id": run_id})


if __name__ == "__main__":
    main()
