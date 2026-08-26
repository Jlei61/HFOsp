#!/usr/bin/env python3
"""Corrected H2B leg: force the producer to actually apply the median-interval caliper,
then rebuild everything downstream of it.

The first overnight pass called ``run_goal3b_stage.py``, which does not pass
``--overwrite``.  Every task therefore matched an existing completed job and was
skipped, so the caliper never ran and the downstream numbers were the two-day-old
ones.  The producer plan here carries ``--overwrite`` explicitly, which also changes
the task key so the launcher cannot adopt the stale status files.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.topic5_epi_prssm.contracts import OUTPUT_ROOT, atomic_write_json  # noqa: E402

HERE = Path(__file__).resolve().parent
PY_EXE = sys.executable
LOG = OUTPUT_ROOT / "logs/caliper_chain.log"
STATUS = OUTPUT_ROOT / "manifests/CALIPER_CHAIN_STATUS.json"
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
       "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}


def log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    with LOG.open("a") as handle:
        handle.write(f"{stamp} [caliper] {message}\n")
    print(f"{stamp} [caliper] {message}", flush=True)


def status(stage: str, state: str, extra: dict | None = None) -> None:
    payload = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    payload[stage] = {"state": state, "at": time.time(), **(extra or {})}
    atomic_write_json(STATUS, payload)


def step(name: str, args: list[str], *, timeout: float = 21600) -> bool:
    log(f"step {name}: {' '.join(str(a) for a in args)}")
    status(name, "RUNNING")
    try:
        result = subprocess.run([PY_EXE, str(HERE / args[0])] + [str(a) for a in args[1:]],
                                cwd=ROOT, capture_output=True, text=True,
                                timeout=timeout, env=ENV)
    except subprocess.TimeoutExpired:
        log(f"step {name}: TIMEOUT"); status(name, "TIMEOUT"); return False
    if result.returncode == 0:
        log(f"step {name}: COMPLETE"); status(name, "COMPLETE"); return True
    tail = "\n".join((result.stderr or result.stdout or "").strip().splitlines()[-14:])
    log(f"step {name}: FAILED rc={result.returncode}\n{tail}")
    status(name, "FAILED", {"rc": result.returncode, "tail": tail})
    return False


def main() -> None:
    log("caliper chain start")
    status("chain", "RUNNING")

    import os as _os
    if _os.environ.get("SKIP_PRODUCER") != "1":
        plan = OUTPUT_ROOT / "manifests/plans/goal3b_caliper.json"
        step("producer_forced", ["launch_autonomous.py", "--plan", str(plan),
                                 "--cap", "18", "--tag", "g3bC"], timeout=28800)
    else:
        log("producer skipped by request; its output is already fresh")

    # verify the caliper actually left evidence before trusting anything downstream.
    # This one IS blocking: everything below it is an interpretation of a balance that
    # would then be unproven, and an unproven balance is exactly the confound this leg
    # exists to remove.
    if not step("verify_caliper", ["verify_caliper_applied.py"]):
        log("caliper verification failed; refusing to compute anything downstream")
        status("chain", "STOPPED_UNVERIFIED_CALIPER")
        raise SystemExit(2)

    # The producer writes per-subject JSON; this is the step that turns it into the
    # per-seizure table every downstream analysis reads.  Omitting it is why four
    # separate "results" tonight were computed from the previous day's aggregation.
    for layer in ("linear_graph_recurrent", "leaky_state",
                  "resource_anchored_on_best_family"):
        for minutes in (30.0, 60.0, 15.0, 5.0):
            step(f"aggregate_goal3b_{layer}_{int(minutes)}m",
                 ["aggregate_goal3b.py", "--layer", layer, "--lead-minutes", minutes])

    # and refuse to interpret anything that is still older than the producer
    if not step("verify_downstream_fresh", ["verify_downstream_fresh.py"]):
        log("downstream artefacts are stale; refusing to interpret them")
        status("chain", "STOPPED_STALE_DOWNSTREAM")
        raise SystemExit(2)

    for lead in ("lead30m", "lead60m", "lead15m", "lead5m"):
        step(f"crosswalk_{lead}", ["build_seizure_crosswalk.py",
                                   "--layer", "linear_graph_recurrent", "--lead", lead])
    step("denominators", ["build_h2b_denominators.py"])
    step("h2b_sensitivity", ["run_h2b_sensitivity.py"])
    step("h3b_transition", ["run_h3b_transition_coupling.py"])

    run_id = time.strftime("%Y%m%d-%H%M")
    step("core_figure", ["make_figure_core_evidence.py", "--run-id", run_id])
    step("write_reports", ["write_reports.py", "--cohort", "all34"])

    log("caliper chain done")
    status("chain", "COMPLETE", {"figure_run_id": run_id})


if __name__ == "__main__":
    main()
