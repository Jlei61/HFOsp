#!/usr/bin/env python3
"""Everything after the running plans: Goal 3b, the reproduction check, and a full
recomputation of every aggregate, figure and report.

Designed to be started with nohup and left alone.  It polls for the work that other
processes are doing, never kills anything, and re-runs every aggregation at the end
so no summary is left holding a partial view.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.topic5_epi_prssm.contracts import OUTPUT_ROOT, atomic_write_json  # noqa: E402
import launch_autonomous as controller  # noqa: E402

PYTHON = os.environ.get("EPI_PRSSM_PYTHON",
                        "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
LOG = OUTPUT_ROOT / "logs/remaining_chain.log"
STATUS = OUTPUT_ROOT / "logs/remaining_chain.status"
TERMINAL = {"COMPLETE", "FAILED", "OOM", "NAN", "INVALID_INPUT", "SKIPPED_EXISTING"}

GOAL1_PLANS = [
    ("goal1_all34.json", "main"),
    ("goal1_static_rerun.json", "goal1fix"),
    ("goal1_timing_baseline.json", "goal1nuis"),
    ("goal1_sensitivity_long_window.json", "goal1sens"),
    ("goal1_resource_on_g1.json", "goal1g1r"),
]


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [chain] {message}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def status(stage: str, state: str, extra: dict | None = None) -> None:
    payload = {}
    if STATUS.exists():
        try:
            payload = json.loads(STATUS.read_text())
        except json.JSONDecodeError:
            payload = {}
    payload.setdefault("stages", {})[stage] = {
        "state": state, "at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"), **(extra or {})}
    payload["heartbeat_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    payload["pid"] = os.getpid()
    atomic_write_json(STATUS, payload)


def step(name: str, args: list[str], *, required: bool = False) -> bool:
    log(f"step {name}: {' '.join(str(a) for a in args)}")
    status(name, "RUNNING")
    result = subprocess.run([PYTHON, *[str(a) for a in args]], cwd=str(HERE),
                            capture_output=True, text=True)
    tail = ((result.stdout or "")[-1200:] + (result.stderr or "")[-1200:])
    if result.returncode == 0:
        log(f"step {name}: COMPLETE")
        status(name, "COMPLETE", {"tail": tail[-400:]})
        return True
    log(f"step {name}: FAILED rc={result.returncode}\n{tail}")
    status(name, "FAILED", {"returncode": result.returncode, "tail": tail[-1200:]})
    return False


def wait_for_plans(poll: float = 60.0) -> None:
    for name, tag in GOAL1_PLANS:
        path = OUTPUT_ROOT / "manifests/plans" / name
        if not path.exists():
            continue
        controller.set_tag(tag)
        tasks = controller.load_plan(path)
        log(f"waiting for {name} ({len(tasks)} tasks)")
        status(f"wait:{name}", "RUNNING")
        while True:
            states = [(controller.read_task(t) or {}).get("state", "PENDING") for t in tasks]
            if all(s in TERMINAL for s in states):
                counts = {s: states.count(s) for s in set(states)}
                log(f"{name} finished: {counts}")
                status(f"wait:{name}", "COMPLETE", {"state_counts": counts})
                break
            time.sleep(poll)


def wait_for_sequencer(timeout_hours: float = 8.0, poll: float = 60.0) -> bool:
    path = OUTPUT_ROOT / "logs/sequencer.status"
    deadline = time.time() + timeout_hours * 3600.0
    log("waiting for the sequencer to finish its stages")
    status("wait:sequencer", "RUNNING")
    while time.time() < deadline:
        if path.exists():
            try:
                payload = json.loads(path.read_text())
            except json.JSONDecodeError:
                payload = {}
            stages = payload.get("stages", {})
            if stages.get("sequencer", {}).get("state") == "COMPLETE":
                log("sequencer reported COMPLETE")
                status("wait:sequencer", "COMPLETE")
                return True
            alive = Path(f"/proc/{payload.get('pid')}").exists() if payload.get("pid") else False
            if not alive and stages:
                log("sequencer process is gone; continuing with whatever it produced")
                status("wait:sequencer", "GONE", {"stages": list(stages)})
                return False
        time.sleep(poll)
    log("sequencer wait timed out; continuing")
    status("wait:sequencer", "TIMEOUT")
    return False


def frozen_layers(limit: int) -> list[str]:
    path = OUTPUT_ROOT / "manifests/INTERICTAL_MODEL_FREEZE.json"
    if not path.exists():
        return []
    freeze = json.loads(path.read_text())
    layers = [r["layer"] for r in freeze["representatives"] if r["status"] == "FROZEN"]
    # the linear graph-CLDS is the best stable recurrent family on this cohort, so it
    # leads; the leaky baseline and the resource-anchored arm follow as contrasts
    preferred = ["linear_graph_recurrent", "leaky_state", "resource_anchored_on_best_family",
                 "nonlinear_graph_recurrent", "resource_anchored",
                 "unconstrained_persistent", "static_repertoire"]
    ordered = [l for l in preferred if l in layers] + [l for l in layers if l not in preferred]
    return ordered[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="all34")
    parser.add_argument("--cap", type=int, default=24)
    parser.add_argument("--matrix-cap", type=int, default=50)
    parser.add_argument("--max-layers", type=int, default=3)
    parser.add_argument("--leads", type=float, nargs="*", default=[30.0, 60.0, 15.0, 5.0])
    args = parser.parse_args()
    log("chain start")
    status("chain", "RUNNING")

    wait_for_plans()
    # the Goal 1 tables and the freeze must see every arm, so recompute them here
    step("aggregate_generator_ladder", ["aggregate_generator_ladder.py", "--cohort", args.cohort])
    step("aggregate_synthetic", ["aggregate_synthetic.py"])
    step("verify_reproduction", ["verify_reproduction.py", "--arm", "ct_ewma_g0",
                                 "--seed", 11, "--cohort", args.cohort, "--max-epochs", 2])
    step("figure_b_interim", ["make_figure_b.py", "--cohort", args.cohort])

    # Goal 2, Goal 4 and the strict Goal 3 arm, driven from here so there is exactly
    # one owner of the remaining matrix
    step("full_matrix", ["run_full_matrix.py", "--cohort", args.cohort,
                         "--epochs", 12, "--cap", args.matrix_cap,
                         "--skip", "goal1_aggregate"])

    # the sequencer writes the base freeze during its Goal 3 stage; make sure it exists
    if not (OUTPUT_ROOT / "manifests/INTERICTAL_MODEL_FREEZE.json").exists():
        step("freeze_interictal_models", ["freeze_interictal_models.py", "--cohort", args.cohort])

    layers = frozen_layers(args.max_layers)
    if layers:
        log(f"Goal 3b layers: {layers}")
        step("goal3b_stage", ["run_goal3b_stage.py", "--cap", args.cap,
                              "--layers", *layers, "--leads", *[str(l) for l in args.leads]])
    else:
        log("no frozen layer available; Goal 3b cannot run")
        status("goal3b_stage", "SKIPPED", {"reason": "no frozen representative"})

    # final recomputation: every aggregate, every figure, every report
    for name, argv in [
        ("aggregate_generator_ladder_final",
         ["aggregate_generator_ladder.py", "--cohort", args.cohort]),
        ("aggregate_event_distribution",
         ["aggregate_event_distribution.py", "--cohort", args.cohort]),
        ("aggregate_exposure", ["aggregate_exposure.py", "--cohort", args.cohort]),
        ("aggregate_synthetic_final", ["aggregate_synthetic.py"]),
        ("figure_a", ["make_figure_a.py"]),
        ("figure_b", ["make_figure_b.py", "--cohort", args.cohort]),
        ("figure_c", ["make_figure_c.py", "--cohort", args.cohort]),
        ("figure_d", ["make_figure_d.py"]),
        ("figure_e", ["make_figure_e.py", "--cohort", args.cohort]),
        ("final_summary", ["write_final_summary.py", "--cohort", args.cohort]),
        ("write_reports", ["write_reports.py", "--cohort", args.cohort]),
    ]:
        step(name, argv)
    status("chain", "COMPLETE")
    log("chain done")


if __name__ == "__main__":
    main()
