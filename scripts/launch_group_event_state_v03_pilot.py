#!/usr/bin/env python3
"""Create the frozen v0.3 task manifest consumed by persistent workers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


SUBJECTS = ("epilepsiae_1146", "yuquan_pengzihang", "yuquan_zhangkexuan")
SEEDS = (20260902, 20260903, 20260904)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("/data/hfosp_group_event_state_v0_3/pilot"))
    parser.add_argument("--workdir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", default="/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--grammar-epochs", type=int, default=12)
    args = parser.parse_args()
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    runner = args.workdir / "scripts/run_group_event_state_v03_pilot.py"
    tasks = []
    for subject in SUBJECTS:
        grammar_id = f"grammar:{subject}"
        grammar_root = root / subject / "grammar"
        tasks.append({
            "id": grammar_id,
            "kind": "grammar",
            "subject": subject,
            "seed": None,
            "status": "pending",
            "depends_on": [],
            "source_commit": args.source_commit,
            "workdir": str(args.workdir),
            "log": str(root / "logs" / f"grammar_{subject}.log"),
            "expected_output": str(grammar_root / "grammar_v03.json"),
            "command": [
                args.python, str(runner), "--mode", "calibrate", "--subject", subject,
                "--device", "cuda:0", "--output-root", str(root),
                "--grammar-epochs", str(args.grammar_epochs),
            ],
        })
        for seed in SEEDS:
            train_id = f"train:{subject}:{seed}"
            run_root = root / subject / f"seed_{seed}"
            tasks.append({
                "id": train_id,
                "kind": "train",
                "subject": subject,
                "seed": seed,
                "status": "pending",
                "depends_on": [grammar_id],
                "source_commit": args.source_commit,
                "workdir": str(args.workdir),
                "log": str(root / "logs" / f"train_{subject}_{seed}.log"),
                "expected_output": str(run_root / "result.json"),
                "command": [
                    args.python, str(runner), "--mode", "train", "--subject", subject,
                    "--seed", str(seed), "--device", "cuda:0", "--output-root", str(root),
                    "--max-epochs", str(args.max_epochs),
                    "--chunk-events", "1024", "--chunk-seconds", "1800",
                ],
            })
            tasks.append({
                "id": f"evaluate:{subject}:{seed}",
                "kind": "evaluate",
                "subject": subject,
                "seed": seed,
                "status": "pending",
                "depends_on": [train_id],
                "source_commit": args.source_commit,
                "workdir": str(args.workdir),
                "log": str(root / "logs" / f"evaluate_{subject}_{seed}.log"),
                "expected_output": str(run_root / "open_loop.json"),
                "command": [
                    args.python, str(runner), "--mode", "evaluate", "--subject", subject,
                    "--seed", str(seed), "--device", "cuda:0", "--output-root", str(root),
                ],
            })
    manifest = root / "task_manifest.json"
    payload = {
        "format": "group_event_state_v0_3_pilot_task_manifest",
        "created_epoch": time.time(),
        "source_commit": args.source_commit,
        "subjects": list(SUBJECTS),
        "seeds": list(SEEDS),
        "tasks": tasks,
    }
    tmp = manifest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, manifest)
    print(manifest)


if __name__ == "__main__":
    main()
