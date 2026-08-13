#!/usr/bin/env python3
"""Wait for v0.3 postprocessing, then render and audit the final figure package."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    out = args.out_root.resolve()
    snapshot = args.snapshot.resolve()
    figure_dir = out.parents[0] / "paper-ready-figure/fig6_lbss_full_tissue_rnn/figures"
    status = out / "FIGURE_CLOSEOUT_WAIT_STATUS.json"

    primary = out
    while not (primary / "PIPELINE_COMPLETE.json").exists():
        pointer = out / "PRIMARY_ARTIFACT_POINTER.json"
        if pointer.exists():
            payload = json.loads(pointer.read_text())
            primary = Path(payload["artifact_root"]).resolve()
        if (primary / "PIPELINE_FAILED.json").exists() or (out / "FORMAL_TRAINING_FAILED.json").exists():
            atomic(out / "FIGURE_CLOSEOUT_FAILED.json", {
                "status": "UPSTREAM_FAILED", "updated_at": now(), "snapshot": str(snapshot),
                "primary_artifact_root": str(primary),
            })
            raise RuntimeError("upstream v0.3 pipeline failed")
        done = len(list((primary / "per_fit").glob("*/*/seed*/DONE.json")))
        atomic(status, {
            "status": "WAITING_FOR_PIPELINE_COMPLETE", "formal_done": done,
            "formal_total": 465, "updated_at": now(), "pid": os.getpid(),
            "primary_artifact_root": str(primary),
        })
        time.sleep(min(max(args.poll_seconds, 5), 30))

    steps = (
        (
            "claims",
            [
                args.python,
                str(snapshot / "scripts/summarize_topic5_lbss_claims_v0_3.py"),
                "--out-root", str(primary),
            ],
            primary / "LBSS_CLAIM_ADJUDICATION_V0_3_COMPLETE.json",
        ),
        (
            "figure",
            [
                args.python,
                str(snapshot / "scripts/paper_figures/plot_topic5_figure6_lbss_full_tissue_v0_3.py"),
                "--out-root", str(primary),
                "--contact-analysis", str(out.parents[0] / "topic5_rnn_full_cohort_field_transfer_v0_1"),
                "--out-dir", str(figure_dir),
            ],
            figure_dir / "FIGURE6_COMPLETE.json",
        ),
        (
            "audit",
            [
                args.python,
                str(snapshot / "scripts/audit_topic5_lbss_full_tissue_closeout_v0_3.py"),
                "--out-root", str(primary), "--figure-dir", str(figure_dir),
            ],
            primary / "CLOSEOUT_AUDIT.json",
        ),
    )
    logs = out / "run_logs/figure_closeout_v0_3"
    logs.mkdir(parents=True, exist_ok=True)
    completed = []
    for label, command, marker in steps:
        log = logs / f"{label}.log"
        started = time.time()
        with log.open("w") as stream:
            process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, text=True)
        result = {
            "step": label, "returncode": process.returncode,
            "seconds": round(time.time() - started, 2), "log": str(log),
            "marker": str(marker), "marker_exists": marker.exists(),
        }
        completed.append(result)
        if process.returncode != 0 or not marker.exists():
            atomic(out / "FIGURE_CLOSEOUT_FAILED.json", {
                "status": "FAILED", "failed_step": label, "completed": completed,
                "updated_at": now(), "snapshot": str(snapshot),
            })
            raise RuntimeError(f"figure closeout failed at {label}: {log}")
    atomic(out / "FIGURE_CLOSEOUT_COMPLETE.json", {
        "status": "COMPLETE", "completed": completed, "updated_at": now(),
        "snapshot": str(snapshot), "figure_dir": str(figure_dir),
        "primary_artifact_root": str(primary),
    })
    (out / "FIGURE_CLOSEOUT_FAILED.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
