#!/usr/bin/env python3
"""Wait for the formal cohort run, then audit, re-render and report unattended.

Runs under systemd like the cohort controller itself, so a dropped terminal or
a lost network connection cannot leave the run finished but unprocessed. Each
step records its own outcome; a failure is written down rather than swallowed.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
DEFAULT_CONFIG = ROOT / "config/topic4_data_driven_snn_cohort_formal_v1.json"
NUMERIC_ENV = {
    "BLIS_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
}
STEPS = (
    ("conditioning_value_audit", "scripts/audit_topic4_cohort_conditioning_value.py"),
    ("representative_envelope_rerun",
     "scripts/rerun_topic4_cohort_representative_envelope.py"),
    ("cohort_figure", "scripts/paper_figures/build_topic4_cohort_figure.py"),
)


def _first_word(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    text = path.read_text().strip()
    return text.split(maxsplit=1)[0] if text else "EMPTY"


def _wait_for_run(output_root: Path, status: Path, poll_seconds: float,
                  deadline_seconds: float) -> str:
    controller = output_root / "controller.status"
    started = time.time()
    while True:
        state = _first_word(controller)
        if state in {"COMPLETE", "FAILED"}:
            return state
        if time.time() - started > deadline_seconds:
            return "TIMED_OUT"
        failed = sorted((output_root / "run_logs").glob("*.status"))
        broken = [path.name for path in failed if _first_word(path) == "FAILED"]
        if broken:
            status.write_text(f"WORKERS_FAILED {broken[:5]}\n")
            return "WORKER_FAILED"
        status.write_text(
            f"WAITING controller={controller.read_text().strip()} "
            f"checked_at={time.time():.0f}\n"
        )
        time.sleep(poll_seconds)


def _provenance_blockers(expected_commit: str) -> list[str]:
    """Modules a still-pinned worker imports must not have moved on disk."""
    sample = sorted(
        (ROOT / "results/topic4_sef_hfo/data_driven_snn_cohort_v1/formal/workers")
        .glob("*_seed_*.json")
    )
    if not sample:
        return []
    tracked = set(json.loads(sample[0].read_text())["provenance"]["runtime_module_sha256"])
    changed = set(subprocess.check_output(
        ["git", "diff", "--name-only", expected_commit, "HEAD"], cwd=ROOT, text=True,
    ).split())
    dirty = set(subprocess.check_output(
        ["git", "status", "--porcelain", "--"], cwd=ROOT, text=True,
    ).split("\n"))
    dirty = {line.split()[-1] for line in dirty if line.strip()}
    return sorted(tracked & (changed | dirty))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--poll-seconds", type=float, default=120.0)
    parser.add_argument("--deadline-hours", type=float, default=14.0)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output_root = ROOT / config["output_root"]
    status = output_root / "postprocess.status"
    status.parent.mkdir(parents=True, exist_ok=True)

    outcome = _wait_for_run(
        output_root, status, args.poll_seconds, args.deadline_hours * 3600.0,
    )
    if outcome != "COMPLETE":
        status.write_text(f"ABORTED reason={outcome}\n")
        subprocess.run(["notify-send", "Topic 4 cohort post-processing",
                        f"aborted: {outcome}"], check=False)
        print(json.dumps({"status": "ABORTED", "reason": outcome}))
        return

    blockers = _provenance_blockers(args.expected_commit)
    results = []
    for name, script in STEPS:
        if name == "representative_envelope_rerun" and blockers:
            results.append({
                "step": name, "status": "SKIPPED",
                "why": "worker-tracked modules moved since the pinned commit",
                "modules": blockers,
            })
            continue
        status.write_text(f"RUNNING step={name}\n")
        command = [str(PYTHON), str(ROOT / script), "--config", str(args.config)]
        if name != "cohort_figure":
            command += ["--expected-commit", args.expected_commit]
        finished = subprocess.run(
            command, cwd=ROOT, env={**os.environ, **NUMERIC_ENV},
            capture_output=True, text=True,
        )
        results.append({
            "step": name, "status": "OK" if finished.returncode == 0 else "FAILED",
            "returncode": finished.returncode,
            "stdout_tail": finished.stdout[-2000:],
            "stderr_tail": finished.stderr[-2000:],
        })
        if finished.returncode != 0:
            break

    result_path = output_root / "cohort_result.json"
    verdict = (
        json.loads(result_path.read_text())["status"] if result_path.exists()
        else "COHORT_RESULT_ABSENT"
    )
    payload = {
        "status": "POSTPROCESS_COMPLETE" if all(
            row["status"] in {"OK", "SKIPPED"} for row in results
        ) else "POSTPROCESS_FAILED",
        "cohort_status": verdict,
        "provenance_blockers": blockers,
        "steps": results,
        "expected_commit": args.expected_commit,
    }
    (output_root / "postprocess_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    status.write_text(f"{payload['status']} cohort={verdict}\n")
    subprocess.run(["notify-send", "Topic 4 cohort post-processing",
                    f"{payload['status']}; cohort {verdict}"], check=False)
    print(json.dumps({"status": payload["status"], "cohort_status": verdict,
                      "steps": [(row["step"], row["status"]) for row in results]},
                     indent=2))


if __name__ == "__main__":
    main()
