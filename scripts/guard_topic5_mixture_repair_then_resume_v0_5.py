#!/usr/bin/env python3
"""Detached fail-closed repair gate placed before the v0.5 target unseal."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def terminate_group(main_pid: int) -> None:
    try:
        os.killpg(main_pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--main-pid", type=int, required=True)
    parser.add_argument("--freezer-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--timeout-hours", type=float, default=72.0)
    args = parser.parse_args()
    out = args.out_root.resolve()
    started = time.monotonic()
    log = out / "posttraining_logs/G0_train_mixture_repair.log"
    log.parent.mkdir(exist_ok=True)
    try:
        while not (out / "STAGE_F_TARGET_FREE_COMPLETE.json").exists():
            if (out / "TARGET_UNSEAL_AUTHORIZATION.json").exists():
                raise RuntimeError("target authorization preceded mixture repair")
            if (out / "STAGE_F_TARGET_FREE_FAILED.json").exists():
                raise RuntimeError("Stage F failed before mixture repair")
            if time.monotonic() - started > args.timeout_hours * 3600:
                raise TimeoutError("timed out waiting for Stage F")
            time.sleep(max(1, int(args.poll_seconds)))

        # Stage F may have consumed exact-equivalent caches written by the
        # detached hotfill.  Do not begin the repair/unseal chain until that
        # producer has atomically closed its provenance marker.  This is a
        # provenance wait only: the original Stage-F executor remains the
        # authoritative fallback for every cache unit.
        while (out / "ATTENUATION_HOTFILL_ACTIVE.json").exists():
            if (out / "TARGET_UNSEAL_AUTHORIZATION.json").exists():
                raise RuntimeError("target authorization preceded hotfill closeout")
            if time.monotonic() - started > args.timeout_hours * 3600:
                raise TimeoutError("timed out waiting for attenuation hotfill closeout")
            time.sleep(max(1, int(args.poll_seconds)))
        if (
            (out / "ATTENUATION_HOTFILL_EXACT_PARITY.json").exists()
            and not (out / "ATTENUATION_HOTFILL_COMPLETE.json").exists()
        ):
            raise RuntimeError("attenuation hotfill lacks an atomic completion marker")

        command = [
            sys.executable, str(ROOT / "scripts/run_topic5_v0_5_target_free.py"),
            "--out-root", str(out), "--",
            sys.executable,
            str(ROOT / "scripts/repair_topic5_multiscale_train_mixture_v0_5.py"),
            "--out-root", str(out),
        ]
        with log.open("a") as stream:
            result = subprocess.run(
                command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(f"target-free mixture repair failed with rc={result.returncode}")
        marker = json.loads((out / "TRAIN_PREVALENCE_MIXTURE_REPAIR_COMPLETE.json").read_text())
        if not (
            marker.get("status") == "PASS_TARGET_FREE"
            and marker.get("target_values_read") is False
            and marker.get("oracle_ab_vectors_changed") is False
            and marker.get("changed_patient_arm_fields") == 70
        ):
            raise RuntimeError("mixture repair marker is invalid")

        guard = [
            sys.executable, str(ROOT / "scripts/guard_topic5_preunseal_resume_v0_5.py"),
            "--out-root", str(out), "--main-pid", str(args.main_pid),
            "--freezer-pid", str(args.freezer_pid), "--poll-seconds", "2",
            "--timeout-hours", str(args.timeout_hours),
        ]
        with log.open("a") as stream:
            result = subprocess.run(
                guard, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(f"post-repair resume guard failed with rc={result.returncode}")
        write_json(out / "MIXTURE_REPAIR_RESUME_GUARD_COMPLETE.json", {
            "status": "PASS_TARGET_FREE",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "main_pid": args.main_pid,
            "repair_marker": str(out / "TRAIN_PREVALENCE_MIXTURE_REPAIR_COMPLETE.json"),
            "target_values_read": False,
        })
    except Exception as error:
        write_json(out / "MIXTURE_REPAIR_RESUME_GUARD_FAILED.json", {
            "status": "FAIL_CLOSED",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "error": repr(error),
            "target_values_read": False,
        })
        terminate_group(args.main_pid)
        raise


if __name__ == "__main__":
    main()
