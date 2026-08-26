#!/usr/bin/env python3
"""Emit a controller task plan for one stage of the experiment matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_epi_prssm.contracts import FROZEN, OUTPUT_ROOT, atomic_write_json  # noqa: E402

PLANS = OUTPUT_ROOT / "manifests/plans"


def generator_ladder(cohort: str, seeds, epochs: int, max_train_events: int) -> list[dict]:
    from importlib import import_module
    sys.path.insert(0, str(ROOT / "scripts/topic5_epi_prssm"))
    arms = sorted(import_module("run_generator_ladder").ARMS)
    return [{
        "label": f"goal1:{arm}:s{seed}:{cohort}",
        "script": "scripts/topic5_epi_prssm/run_generator_ladder.py",
        "workload": "cpu_train",
        "args": ["--arm", arm, "--seed", seed, "--cohort", cohort,
                 "--max-epochs", epochs, "--max-train-events", max_train_events],
    } for arm in arms for seed in seeds]


def event_distribution(cohort: str, seeds, epochs: int, max_train_events: int) -> list[dict]:
    from importlib import import_module
    sys.path.insert(0, str(ROOT / "scripts/topic5_epi_prssm"))
    arms = sorted(import_module("run_event_distribution").ARMS)
    return [{
        "label": f"goal2:{arm}:s{seed}:{cohort}",
        "script": "scripts/topic5_epi_prssm/run_event_distribution.py",
        "workload": "cpu_train",
        "args": ["--arm", arm, "--seed", seed, "--cohort", cohort,
                 "--max-epochs", epochs, "--max-train-events", max_train_events],
    } for arm in arms for seed in seeds]


def exposure_mechanism(cohort: str, seeds, epochs: int, max_train_events: int) -> list[dict]:
    from importlib import import_module
    sys.path.insert(0, str(ROOT / "scripts/topic5_epi_prssm"))
    arms = sorted(import_module("run_exposure_mechanism").ARMS)
    return [{
        "label": f"goal4:{arm}:s{seed}:{cohort}",
        "script": "scripts/topic5_epi_prssm/run_exposure_mechanism.py",
        "workload": "cpu_train",
        "args": ["--arm", arm, "--seed", seed, "--cohort", cohort,
                 "--max-epochs", epochs, "--max-train-events", max_train_events],
    } for arm in arms for seed in seeds]


def synthetic(seeds) -> list[dict]:
    from importlib import import_module
    sys.path.insert(0, str(ROOT / "scripts/topic5_epi_prssm"))
    families = sorted(import_module("run_synthetic").TRUTHS)
    return [{
        "label": f"synthetic:{truth}:s{seed}",
        "script": "scripts/topic5_epi_prssm/run_synthetic.py",
        "workload": "cpu_synthetic",
        "args": ["--truth", truth, "--seed", seed],
    } for truth in families for seed in seeds]


BUILDERS = {
    "goal1": generator_ladder,
    "goal2": event_distribution,
    "goal4": exposure_mechanism,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--cohort", default="all34")
    parser.add_argument("--seeds", type=int, nargs="*", default=list(FROZEN["breadth_seeds"]))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--max-train-events", type=int, default=30000)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.stage == "synthetic":
        tasks = synthetic(args.seeds)
    else:
        tasks = BUILDERS[args.stage](args.cohort, args.seeds, args.epochs, args.max_train_events)
    out = Path(args.out) if args.out else PLANS / f"{args.stage}_{args.cohort}.json"
    atomic_write_json(out, {"stage": args.stage, "cohort": args.cohort,
                            "seeds": args.seeds, "n_tasks": len(tasks), "tasks": tasks})
    print(f"{out}  ({len(tasks)} tasks)")


if __name__ == "__main__":
    main()
