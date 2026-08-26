#!/usr/bin/env python3
"""Sequencer for the remaining Epi-PRSSM v0.1 stages.

Runs each stage's worker pool to completion, then the step that depends on it.
Every stage is resumable: a stage whose tasks are already COMPLETE is adopted,
and a stage that produced no usable output records why and lets the independent
stages continue.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from src.topic5_epi_prssm.contracts import (  # noqa: E402
    FROZEN, OUTPUT_ROOT, atomic_write_json, code_revision, package_hash,
)
import launch_autonomous as controller  # noqa: E402

PYTHON = os.environ.get("EPI_PRSSM_PYTHON",
                        "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
STAGE_LOG = OUTPUT_ROOT / "logs/sequencer.log"
STAGE_STATUS = OUTPUT_ROOT / "logs/sequencer.status"


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    line = f"{stamp} [sequencer] {message}"
    print(line, flush=True)
    STAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with STAGE_LOG.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def status(stage: str, state: str, extra: dict | None = None) -> None:
    payload = {}
    if STAGE_STATUS.exists():
        try:
            payload = json.loads(STAGE_STATUS.read_text())
        except json.JSONDecodeError:
            payload = {}
    payload.setdefault("stages", {})[stage] = {"state": state, "at": time.time(),
                                               "at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                               **(extra or {})}
    payload["heartbeat"] = time.time()
    payload["heartbeat_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    payload["pid"] = os.getpid()
    payload["code_revision"] = code_revision()
    payload["package_hash"] = package_hash()
    atomic_write_json(STAGE_STATUS, payload)


def run_step(name: str, args: list[str], *, required: bool = True) -> bool:
    log(f"step {name}: {' '.join(args)}")
    status(name, "RUNNING")
    result = subprocess.run([PYTHON, *args], cwd=str(HERE), capture_output=True, text=True)
    tail = (result.stdout or "")[-1500:] + (result.stderr or "")[-1500:]
    if result.returncode == 0:
        log(f"step {name}: COMPLETE")
        status(name, "COMPLETE", {"tail": tail[-600:]})
        return True
    log(f"step {name}: FAILED rc={result.returncode}\n{tail}")
    status(name, "FAILED", {"returncode": result.returncode, "tail": tail[-1500:]})
    if required:
        log(f"step {name} was required by the stage that follows it; that branch stops here "
            "while the independent branches continue")
    return False


def wait_for_plan(plan: Path, tag: str, poll: float = 30.0) -> None:
    """Wait until every task in an already-running plan has a terminal state."""
    tasks = controller.load_plan(plan)
    controller.set_tag(tag)
    while True:
        states = [(controller.read_task(t) or {}).get("state", "PENDING") for t in tasks]
        if all(s in ("COMPLETE", "FAILED", "OOM", "NAN", "INVALID_INPUT", "SKIPPED_EXISTING")
               for s in states):
            counts = {s: states.count(s) for s in set(states)}
            log(f"plan {plan.name} finished: {counts}")
            return
        time.sleep(poll)


def run_plan(plan: Path, tag: str, cap: int, peak_rss: float = 1.6) -> None:
    tasks = controller.load_plan(plan)
    controller.set_tag(tag)
    log(f"plan {plan.name}: {len(tasks)} tasks, cap {cap}")
    status(f"plan:{tag}", "RUNNING", {"n_tasks": len(tasks)})
    controller.run_pool(tasks, cap=cap, peak_rss_gib=peak_rss)
    counts: dict[str, int] = {}
    for task in tasks:
        state = (controller.read_task(task) or {}).get("state", "PENDING")
        counts[state] = counts.get(state, 0) + 1
    log(f"plan {plan.name} finished: {counts}")
    status(f"plan:{tag}", "COMPLETE", {"state_counts": counts})


def build_plan(stage: str, cohort: str, seeds: list[int], epochs: int) -> Path:
    args = ["build_plan.py", "--stage", stage, "--cohort", cohort, "--epochs", str(epochs),
            "--seeds", *[str(s) for s in seeds]]
    run_step(f"build_plan:{stage}", args)
    return OUTPUT_ROOT / f"manifests/plans/{stage}_{cohort}.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="all34")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--cap", type=int, default=28)
    parser.add_argument("--wait-goal1-plan", default=None,
                        help="path of an already-running Goal 1 plan to wait on")
    parser.add_argument("--skip", nargs="*", default=[])
    args = parser.parse_args()
    seeds = list(FROZEN["breadth_seeds"])
    skip = set(args.skip)

    if args.wait_goal1_plan:
        log("waiting for the running Goal 1 plan")
        status("wait:goal1", "RUNNING")
        wait_for_plan(Path(args.wait_goal1_plan), "main")
        status("wait:goal1", "COMPLETE")
        for tag, name in (("synthetic", "synthetic_rev3"),):
            path = OUTPUT_ROOT / f"manifests/plans/{name}.json"
            if path.exists():
                log(f"waiting for {name}")
                wait_for_plan(path, tag)

    if "goal1_aggregate" not in skip:
        run_step("aggregate_generator_ladder",
                 ["aggregate_generator_ladder.py", "--cohort", args.cohort], required=False)
        run_step("aggregate_synthetic", ["aggregate_synthetic.py"], required=False)
        run_step("figure_b", ["make_figure_b.py", "--cohort", args.cohort], required=False)

    if "goal2" not in skip:
        plan = build_plan("goal2", args.cohort, seeds, args.epochs)
        run_plan(plan, "goal2", args.cap)
        run_step("aggregate_event_distribution",
                 ["aggregate_event_distribution.py", "--cohort", args.cohort], required=False)
        run_step("figure_c", ["make_figure_c.py", "--cohort", args.cohort], required=False)

    if "goal4" not in skip:
        import run_exposure_mechanism as exposure
        stage_a = [{"label": f"goal4a:{arm}:s{seed}:{args.cohort}",
                    "script": "scripts/topic5_epi_prssm/run_exposure_mechanism.py",
                    "workload": "cpu_train",
                    "args": ["--arm", arm, "--seed", seed, "--cohort", args.cohort,
                             "--max-epochs", args.epochs, "--max-train-events", 30000]}
                   for arm in sorted(exposure.STAGE_A) for seed in seeds]
        plan_a = atomic_write_json(OUTPUT_ROOT / "manifests/plans/goal4a.json",
                                   {"stage": "goal4a", "n_tasks": len(stage_a), "tasks": stage_a})
        run_plan(plan_a, "goal4a", args.cap)
        frozen = run_step("freeze_resource_tau", ["freeze_resource_tau.py", "--cohort", args.cohort],
                          required=False)
        if frozen:
            stage_b = [{"label": f"goal4b:{arm}:s{seed}:{args.cohort}",
                        "script": "scripts/topic5_epi_prssm/run_exposure_mechanism.py",
                        "workload": "cpu_train",
                        "args": ["--arm", arm, "--seed", seed, "--cohort", args.cohort,
                                 "--max-epochs", args.epochs, "--max-train-events", 30000]}
                       for arm in sorted(exposure.STAGE_B) for seed in seeds]
            plan_b = atomic_write_json(OUTPUT_ROOT / "manifests/plans/goal4b.json",
                                       {"stage": "goal4b", "n_tasks": len(stage_b),
                                        "tasks": stage_b})
            run_plan(plan_b, "goal4b", args.cap)
        run_step("innovation_controls",
                 ["run_innovation_controls.py", "--cohort", args.cohort], required=False)
        run_step("aggregate_exposure",
                 ["aggregate_exposure.py", "--cohort", args.cohort], required=False)
        run_step("figure_e", ["make_figure_e.py", "--cohort", args.cohort], required=False)

    if "goal3" not in skip:
        frozen = run_step("freeze_interictal_models",
                          ["freeze_interictal_models.py", "--cohort", args.cohort], required=False)
        if frozen or (OUTPUT_ROOT / "manifests/INTERICTAL_MODEL_FREEZE.json").exists():
            freeze = json.loads(
                (OUTPUT_ROOT / "manifests/INTERICTAL_MODEL_FREEZE.json").read_text())
            layers = [r["layer"] for r in freeze["representatives"] if r["status"] == "FROZEN"]
            tasks = [{"label": f"goal3:{layer}:{args.cohort}",
                      "script": "scripts/topic5_epi_prssm/run_seizure_link.py",
                      "workload": "cpu_analysis",
                      "args": ["--layer", layer, "--cohort", args.cohort]} for layer in layers]
            plan = atomic_write_json(OUTPUT_ROOT / "manifests/plans/goal3.json",
                                     {"stage": "goal3", "n_tasks": len(tasks), "tasks": tasks})
            run_plan(plan, "goal3", min(args.cap, len(tasks) or 1))
            run_step("figure_d", ["make_figure_d.py"], required=False)
            run_step("aggregate_exposure_h3b",
                     ["aggregate_exposure.py", "--cohort", args.cohort], required=False)

    run_step("final_summary", ["write_final_summary.py", "--cohort", args.cohort], required=False)
    status("sequencer", "COMPLETE")
    log("sequencer done")


if __name__ == "__main__":
    main()
