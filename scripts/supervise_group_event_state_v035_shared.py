#!/usr/bin/env python3
"""Recoverable two-GPU supervisor for shared S_N/S_G development pilots."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(os.environ.get("HFOSP_PYTHON", sys.executable))
OUT = Path(os.environ.get(
    "HFOSP_GES_V035_SHARED_ROOT",
    "/data/hfosp_group_event_state_v0_3_5_shared",
))
RATE = OUT / "shared_rate"
ADAPTER = OUT / "shared_stepwise_adapter"
PRODUCER = OUT / "shared_producer"
LOG = OUT / "logs"
STATUS = OUT / "supervisor" / "queue_status.json"
SUBJECTS = ("epilepsiae_253", "epilepsiae_1096", "epilepsiae_1125")
SEEDS = ((0, 20260903), (1, 20260904), (2, 20260905))
# Each model uses <1 GB and spends most wall time in the Python event replay.
# Two independent processes per GPU overlap that host-side work without coming
# close to the 24 GB memory limit.  Slot identity is separate from device id so
# two jobs may safely share one device.
GPU_SLOTS = ((0, 0), (0, 1), (1, 0), (1, 1))


def active_producer_commands() -> list[list[str]]:
    """Read active producer argv without relying on self-matching pgrep."""

    commands: list[list[str]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        argv = [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]
        if "scripts/run_group_event_state_v035_shared_producer.py" in argv:
            commands.append(argv)
    return commands


def _arg(argv: list[str], name: str) -> str | None:
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError):
        return None


def job_is_active(job: dict, commands: list[list[str]]) -> bool:
    base = job["base"]
    identity = tuple(_arg(base, key) for key in ("--subject", "--family", "--state-seed"))
    return any(tuple(_arg(argv, key) for key in ("--subject", "--family", "--state-seed"))
               == identity for argv in commands)


def active_on_gpu(commands: list[list[str]], gpu: int) -> int:
    return sum(_arg(argv, "--device") == f"cuda:{gpu}" for argv in commands)


def atomic(payload: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, STATUS)


def run_logged(cmd: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("\nCOMMAND " + " ".join(cmd) + "\n")
        handle.flush()
        return subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT).returncode


def build_rates(state: dict) -> None:
    for subject in SUBJECTS:
        for _decoder_seed, seed in SEEDS:
            card = RATE / subject / f"seed{seed}" / "card.json"
            key = f"rate::{subject}::{seed}"
            if card.exists():
                state["units"][key] = "COMPLETE"; atomic(state); continue
            state["units"][key] = "RUNNING"; atomic(state)
            rc = run_logged([
                str(PYTHON), "scripts/run_group_event_state_v035_dynamic_rate.py",
                "--subject", subject, "--seed", str(seed), "--device", "cpu",
                "--out-root", str(RATE), "--config-json",
                "config/group_event_state_v035_rate_search/shared_state_2h_6h_8h.json",
                "--hold-selection",
            ], LOG / f"rate_{subject}_{seed}.log")
            state["units"][key] = "COMPLETE" if rc == 0 and card.exists() else f"FAILED_RC_{rc}"
            atomic(state)
            if not card.exists():
                raise RuntimeError(f"rate prerequisite failed: {key}")


def adapter_jobs() -> list[dict]:
    jobs = []
    for subject in SUBJECTS:
        for decoder_seed, seed in SEEDS:
            jobs.append({
                "key": f"adapter::{subject}::{seed}",
                "card": ADAPTER / subject / f"decoder_seed{decoder_seed}_state_seed{seed}" / "card.json",
                "log": LOG / f"adapter_{subject}_{seed}.log",
                "base": [str(PYTHON), "-u", "scripts/run_group_event_state_v035_stepwise_decoder.py",
                         "--subject", subject, "--decoder-seed", str(decoder_seed),
                         "--state-seed", str(seed), "--rate-root", str(RATE),
                         "--out-root", str(ADAPTER), "--use-rate-phases"],
                "attempt": 0,
            })
    return jobs


def producer_jobs() -> list[dict]:
    jobs = []
    for subject in SUBJECTS:
        for decoder_seed, seed in SEEDS:
            for family in ("S_N", "S_G"):
                unit = PRODUCER / subject / family / f"decoder_seed{decoder_seed}_state_seed{seed}"
                jobs.append({
                    "key": f"producer::{subject}::{family}::{seed}", "card": unit / "card.json",
                    "log": LOG / f"producer_{subject}_{family}_{seed}.log",
                    "base": [str(PYTHON), "-u", "scripts/run_group_event_state_v035_shared_producer.py",
                             "--subject", subject, "--decoder-seed", str(decoder_seed),
                             "--state-seed", str(seed), "--family", family,
                             "--rate-root", str(RATE), "--adapter-root", str(ADAPTER),
                             "--out-root", str(PRODUCER), "--config-json",
                             "config/group_event_state_v035_shared_producer_full.json",
                             "--chunk-events", "1024" if family == "S_N" else "512"],
                    "attempt": 0,
                })
    return jobs


def run_gpu_stage(jobs: list[dict], state: dict, stage: str) -> None:
    pending = [j for j in jobs if not j["card"].exists()]
    for j in jobs:
        if j["card"].exists():
            state["units"][j["key"]] = "COMPLETE"
    running: dict[tuple[int, int], tuple[subprocess.Popen, dict, object]] = {}
    while pending or running:
        # Manual recovery jobs may already be active, and cards may appear
        # while this supervisor is running.  Never launch a duplicate writer.
        for job in list(pending):
            if job["card"].exists():
                state["units"][job["key"]] = "COMPLETE"
                pending.remove(job)
        commands = active_producer_commands()
        for slot in GPU_SLOTS:
            if slot in running or not pending:
                continue
            gpu = slot[0]
            if active_on_gpu(commands, gpu) >= 2:
                continue
            candidate = next((job for job in pending if not job_is_active(job, commands)), None)
            if candidate is None:
                continue
            job = candidate; pending.remove(job)
            job["log"].parent.mkdir(parents=True, exist_ok=True)
            handle = job["log"].open("a", encoding="utf-8")
            cmd = job["base"] + ["--device", f"cuda:{gpu}"]
            handle.write("\nCOMMAND " + " ".join(cmd) + "\n"); handle.flush()
            process = subprocess.Popen(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            running[slot] = (process, job, handle)
            commands.append(cmd)
            state["units"][job["key"]] = "RUNNING"
            atomic(state)
        time.sleep(10)
        for slot, (process, job, handle) in list(running.items()):
            rc = process.poll()
            if rc is None:
                continue
            handle.close(); del running[slot]
            ok = rc == 0 and job["card"].exists()
            state["units"][job["key"]] = "COMPLETE" if ok else f"FAILED_RC_{rc}"
            atomic(state)
            if not ok and int(job.get("attempt", 0)) == 0:
                # One conservative retry with half the event chunk handles an
                # OOM without silently changing any scientific hyperparameter.
                retry = dict(job); retry["key"] += "::retry"
                base = list(retry["base"])
                if "--chunk-events" in base:
                    i = base.index("--chunk-events") + 1
                    base[i] = str(max(128, int(base[i]) // 2))
                base.append("--overwrite")
                retry["base"] = base
                retry["log"] = Path(str(job["log"]) + ".retry")
                retry["attempt"] = 1
                pending.insert(0, retry)
    state["stages"][stage] = "COMPLETE"
    atomic(state)


def downstream(state: dict) -> None:
    for subject in SUBJECTS:
        for decoder_seed, seed in SEEDS:
            for family in ("S_N", "S_G"):
                unit = PRODUCER / subject / family / f"decoder_seed{decoder_seed}_state_seed{seed}"
                if not (unit / "card.json").exists():
                    continue
                base = OUT / "frozen_evaluator" / subject / family / f"seed{seed}"
                key = f"evaluator::{subject}::{family}::{seed}"
                rc = 0 if (base / "card.json").exists() else run_logged([
                    str(PYTHON), "scripts/run_group_event_state_v035_frozen_shared_evaluator.py",
                    "--producer-unit", str(unit), "--decoder-seed", str(decoder_seed),
                    "--rate-root", str(RATE), "--out-dir", str(base),
                ], LOG / f"evaluator_{subject}_{family}_{seed}.log")
                state["units"][key] = "COMPLETE" if rc == 0 and (base / "card.json").exists() else f"FAILED_RC_{rc}"
                atomic(state)
                h2b = OUT / "shared_h2b" / subject / family / f"seed{seed}"
                key = f"h2b::{subject}::{family}::{seed}"
                rc = 0 if (h2b / "shared_state_binding.json").exists() else run_logged([
                    str(PYTHON), "scripts/run_group_event_state_v035_shared_h2b.py",
                    "--producer-unit", str(unit), "--rate-root", str(RATE),
                    "--out-dir", str(h2b),
                ], LOG / f"h2b_{subject}_{family}_{seed}.log")
                state["units"][key] = "COMPLETE" if rc == 0 and (h2b / "shared_state_binding.json").exists() else f"NOT_ESTIMABLE_OR_FAILED_RC_{rc}"
                atomic(state)
            unit = PRODUCER / subject / "S_G" / f"decoder_seed{decoder_seed}_state_seed{seed}"
            if (unit / "card.json").exists():
                out = OUT / "same_prefix" / subject / f"seed{seed}"
                key = f"same_prefix::{subject}::{seed}"
                rc = 0 if (out / "card.json").exists() else run_logged([
                    str(PYTHON), "scripts/run_group_event_state_v035_shared_same_prefix.py",
                    "--producer-unit", str(unit), "--decoder-seed", str(decoder_seed),
                    "--rate-root", str(RATE), "--out-dir", str(out), "--device", "cuda:0",
                ], LOG / f"same_prefix_{subject}_{seed}.log")
                state["units"][key] = "COMPLETE" if rc == 0 and (out / "card.json").exists() else f"FAILED_RC_{rc}"
                atomic(state)
    state["stages"]["downstream"] = "COMPLETE"; atomic(state)


def main() -> None:
    state = json.loads(STATUS.read_text()) if STATUS.exists() else {
        "format": "group_event_state_v0_3_5_shared_supervisor_v1",
        "subjects": list(SUBJECTS), "seeds": [s for _, s in SEEDS],
        "stages": {"rates": "PENDING", "adapters": "PENDING", "producers": "PENDING", "downstream": "PENDING"},
        "units": {}, "status": "RUNNING", "development_targets_read": False,
        "sealed_partition_opened": False,
    }
    atomic(state)
    build_rates(state); state["stages"]["rates"] = "COMPLETE"; atomic(state)
    run_gpu_stage(adapter_jobs(), state, "adapters")
    run_gpu_stage(producer_jobs(), state, "producers")
    downstream(state)
    state["status"] = "COMPLETE"; atomic(state)


if __name__ == "__main__":
    main()
