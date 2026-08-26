#!/usr/bin/env python3
"""Drive the whole Goal 3b stage: addendum, per-patient jobs, aggregation, figure."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.topic5_epi_prssm.contracts import OUTPUT_ROOT, atomic_write_json  # noqa: E402
from src.topic5_epi_prssm.cohort import cohort_subjects  # noqa: E402
import launch_autonomous as controller  # noqa: E402

PYTHON = os.environ.get("EPI_PRSSM_PYTHON",
                        "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")


def step(name: str, args: list[str]) -> bool:
    print(f"[goal3b] {name}: {' '.join(args)}", flush=True)
    result = subprocess.run([PYTHON, *args], cwd=str(HERE), capture_output=True, text=True)
    print((result.stdout or "")[-2500:], flush=True)
    if result.returncode != 0:
        print((result.stderr or "")[-2500:], flush=True)
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="*", default=None)
    parser.add_argument("--leads", type=float, nargs="*", default=[30.0, 60.0, 15.0, 5.0])
    parser.add_argument("--cap", type=int, default=24)
    args = parser.parse_args()

    step("build_full_event_stream", ["build_full_event_stream.py"])
    step("freeze_goal3b_addendum", ["freeze_goal3b_addendum.py"])

    freeze_path = OUTPUT_ROOT / "manifests/INTERICTAL_MODEL_FREEZE.json"
    if not freeze_path.exists():
        raise SystemExit("the base interictal freeze does not exist yet")
    freeze = json.loads(freeze_path.read_text())
    layers = args.layers or [r["layer"] for r in freeze["representatives"]
                             if r["status"] == "FROZEN"]
    subjects = list(cohort_subjects())

    tasks = [{"label": f"goal3b:{s}:{layer}:lead{int(lead)}m",
              "script": "scripts/topic5_epi_prssm/run_goal3b_preictal.py",
              "workload": "cpu_analysis",
              "args": ["--subject", s, "--layer", layer, "--lead-minutes", lead]}
             for layer in layers for lead in args.leads for s in subjects]
    plan = atomic_write_json(OUTPUT_ROOT / "manifests/plans/goal3b.json",
                             {"stage": "goal3b", "n_tasks": len(tasks), "tasks": tasks,
                              "layers": layers, "leads": args.leads})
    controller.set_tag("goal3b")
    controller.run_pool(controller.load_plan(plan), cap=args.cap, peak_rss_gib=2.5)

    for layer in layers:
        for lead in args.leads:
            step(f"aggregate_goal3b:{layer}:{int(lead)}m",
                 ["aggregate_goal3b.py", "--layer", layer, "--lead-minutes", str(lead)])
    step("figure_d", ["make_figure_d.py"])
    print("[goal3b] done", flush=True)


if __name__ == "__main__":
    main()
