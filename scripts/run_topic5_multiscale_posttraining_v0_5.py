#!/usr/bin/env python3
"""Resume-safe Stage E-analysis through Figure 6 orchestration for v0.5."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import os


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    out = args.out_root.resolve()
    scripts = {
        "driver": Path(__file__).resolve(),
        "embargo": ROOT / "scripts/run_topic5_v0_5_target_free.py",
        "interictal": ROOT / "scripts/analyse_topic5_multiscale_interictal_v0_5.py",
        "stage_f": ROOT / "scripts/run_topic5_multiscale_stage_f_v0_5.py",
        "authorize": ROOT / "scripts/prepare_topic5_multiscale_target_unseal_v0_5.py",
        "score": ROOT / "scripts/score_topic5_multiscale_early_ictal_v0_5.py",
        "figure": ROOT / "scripts/paper_figures/plot_topic5_figure6_multiscale_scaffold_v0_5.py",
    }
    snapshot = {
        "contract": "topic5_multiscale_posttraining_v0_5",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_hashes": {key: sha256_file(path) for key, path in scripts.items()},
        "target_values_read": False,
    }
    write_json(out / "POSTTRAINING_PIPELINE_SNAPSHOT.json", snapshot)
    while not (out / "STAGE_E_TRAINING_COMPLETE.json").exists():
        failed = out / "STAGE_E_TRAINING_FAILED.json"
        if failed.exists():
            raise RuntimeError(f"formal training failed: {failed}")
        time.sleep(max(5, int(args.poll_seconds)))
    commands = (
        ("E_interictal_analysis", out / "STAGE_E_INTERICTAL_ANALYSIS_COMPLETE.json", [
            sys.executable, str(scripts["embargo"]), "--out-root", str(out), "--",
            sys.executable, str(scripts["interictal"]), "--out-root", str(out),
        ]),
        ("F_target_free_freeze", out / "STAGE_F_TARGET_FREE_COMPLETE.json", [
            sys.executable, str(scripts["embargo"]), "--out-root", str(out), "--",
            sys.executable, str(scripts["stage_f"]), "--out-root", str(out),
        ]),
        ("G_target_authorization", out / "TARGET_UNSEAL_AUTHORIZATION.json", [
            sys.executable, str(scripts["authorize"]), "--out-root", str(out),
        ]),
        ("G_locked_scoring", out / "early_ictal/EARLY_ICTAL_V0_5_SUMMARY.json", [
            sys.executable, str(scripts["score"]), "--out-root", str(out),
        ]),
        ("H_figure6", ROOT / "results/paper-ready-figure/fig6_multiscale_scaffold_v0_5/figures/topic5_figure6_multiscale_scaffold_v0_5.png", [
            sys.executable, str(scripts["figure"]), "--out-root", str(out),
        ]),
    )
    logs = out / "posttraining_logs"; logs.mkdir(exist_ok=True)
    completed = []
    child_env = os.environ.copy()
    existing_pythonpath = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = (
        str(ROOT) if not existing_pythonpath
        else f"{ROOT}{os.pathsep}{existing_pythonpath}"
    )
    for label, completion_evidence, command in commands:
        if completion_evidence.exists():
            completed.append(label)
            continue
        log = logs / f"{label}.log"
        with log.open("a") as stream:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=child_env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            write_json(out / "POSTTRAINING_PIPELINE_FAILED.json", {
                "status": "FAILED", "step": label, "returncode": result.returncode,
                "log": str(log), "completed": completed,
            })
            raise SystemExit(result.returncode)
        completed.append(label)
    figure = ROOT / "results/paper-ready-figure/fig6_multiscale_scaffold_v0_5/figures/topic5_figure6_multiscale_scaffold_v0_5.png"
    write_json(out / "PIPELINE_COMPLETE.json", {
        "status": "COMPLETE", "created_utc": datetime.now(timezone.utc).isoformat(),
        "completed": completed, "figure": str(figure), "figure_sha256": sha256_file(figure),
        "target_values_read": True,
    })


if __name__ == "__main__":
    main()
