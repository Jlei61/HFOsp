#!/usr/bin/env python3
"""Autonomous selection and frozen confirmation for Topic 4 rev5."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python"
CONFIG = ROOT / "config/topic4_data_driven_zm_target_informed_bridge_v1.json"
BASE_CONFIG = "config/topic4_data_driven_zm_ictal_transition_v1.json"


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _git_head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                   text=True).strip()


def _unit_active(unit):
    return subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", unit],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def _launch(unit, command, log_path):
    if _unit_active(unit):
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    shell = "exec /usr/bin/nohup env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 " \
            "OPENBLAS_NUM_THREADS=1 " + " ".join(command) + \
            f" >> {log_path} 2>&1"
    subprocess.run([
        "systemd-run", "--user", f"--unit={unit}", "--collect",
        f"--working-directory={ROOT}", "/bin/bash", "-lc", shell,
    ], check=True, cwd=ROOT)


def _run_paths(stage, candidate_id, seed):
    stem = f"{candidate_id}_seed{seed}"
    base = OUT / stage / stem
    return base.with_suffix(".json"), base.with_suffix(".npz")


def _baseline_paths(stage, seed):
    base = OUT / stage / "paired_baseline" / f"seed{seed}_zmoff"
    return base.with_suffix(".json"), base.with_suffix(".npz")


def _worker_command(parameters, seed, out, *, zm_mode="z_plus_m", duration=10000):
    command = [
        PYTHON, "scripts/run_topic4_zm_joint_morphology_canary.py",
        "--config", BASE_CONFIG, "--seed", str(seed),
        "--i-th-ei", str(parameters["I_th_EI"]),
        "--eta-m", str(parameters["eta_m"]),
        "--tau-z", str(parameters["tau_z"]),
        "--tau-adp", str(parameters["tau_adp"]),
        "--ee-dose", "1", "--etoi-dose", "1", "--zm-mode", zm_mode,
        "--duration-ms", str(duration), "--post-runaway-ms", "2000",
        "--out", str(out.relative_to(ROOT)),
    ]
    if zm_mode == "z_plus_m":
        command.append("--save-spatial-frames")
    return command


def _wait_for(expected, units, interval, status_path, stage):
    while True:
        done = [path.exists() for path in expected]
        available = int(next(line.split()[1] for line in
                             Path("/proc/meminfo").read_text().splitlines()
                             if line.startswith("MemAvailable:"))) / 1024**2
        snapshot = {"status": f"{stage}_RUNNING", "n_done": sum(done),
                    "n_expected": len(done), "memory_available_gib": available,
                    "timestamp_epoch": time.time()}
        _atomic(status_path, snapshot)
        if all(done):
            return
        if available < 32.0:
            raise RuntimeError("memory reserve below 32 GiB")
        if not any(_unit_active(unit) for unit in units):
            missing = [str(path) for path, ok in zip(expected, done) if not ok]
            raise RuntimeError(f"workers exited without artifacts: {missing}")
        time.sleep(interval)


def _score_stage(stage, candidates, seeds, config):
    from scripts.rescore_topic4_fig5_target_informed_candidates import (
        _score_one, build_paired_baseline)
    target_payload = _json(OUT / "clinical_target.json")
    target_npz = np.load(OUT / "clinical_target_vectors.npz", allow_pickle=False)
    records = []
    for seed in seeds:
        baseline = build_paired_baseline(
            _baseline_paths(stage, seed)[1], config["model_readout"])
        for candidate in candidates:
            jpath, npath = _run_paths(stage, candidate["candidate_id"], seed)
            records.append(_score_one(
                jpath, npath, _json(jpath), baseline, target_payload, target_npz,
                config["model_readout"]))
            records[-1]["candidate_id"] = candidate["candidate_id"]
            records[-1]["seed"] = int(seed)
    return records


def _launch_stage(stage, candidates, seeds, interval):
    expected, units = [], []
    log_root = ROOT / "results/run_logs"
    for seed in seeds:
        bjson, bnpz = _baseline_paths(stage, seed)
        if not bnpz.exists():
            unit = f"topic4-rev5-{stage}-baseline-{seed}"
            _launch(unit, _worker_command(
                candidates[0]["parameters"], seed, bnpz.with_suffix(""),
                zm_mode="off", duration=2000),
                log_root / f"{unit}.log")
            units.append(unit)
        expected.append(bnpz)
        for candidate in candidates:
            jpath, npath = _run_paths(stage, candidate["candidate_id"], seed)
            if not jpath.exists():
                unit = f"topic4-rev5-{stage}-{candidate['candidate_id']}-{seed}"
                _launch(unit, _worker_command(
                    candidate["parameters"], seed, npath.with_suffix("")),
                    log_root / f"{unit}.log")
                units.append(unit)
            expected.append(jpath)
    _wait_for(expected, units, interval, OUT / f"{stage}_monitor.json", stage)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-seconds", type=float, default=600.0)
    args = parser.parse_args()
    global OUT
    config = _json(CONFIG)
    OUT = ROOT / config["output_root"]
    status_path = OUT / "selection_controller.json"
    stage1_path = OUT / "existing_candidate_rescore.json"
    while not stage1_path.exists():
        _atomic(status_path, {"status": "WAITING_FOR_STAGE1_RESCORE",
                              "timestamp_epoch": time.time()})
        time.sleep(args.interval_seconds)
    stage1 = _json(stage1_path)
    eligible = [row for row in stage1["records"]
                if row.get("status") == "BRIDGE_EVALUABLE"
                and row.get("primary_zm_only")]
    eligible.sort(key=lambda row: row["J_bridge_without_time"])
    candidates = [{"candidate_id": row["candidate_id"],
                   "parameters": row["parameters"],
                   "stage1_J_bridge": row["J_bridge_without_time"]}
                  for row in eligible[:3]]
    if not candidates:
        _atomic(status_path, {
            "status": "NO_STAGE1_ZM_ONLY_ELIGIBLE",
            "next_action": "bounded adaptation refinement required; selection not launched",
        })
        return
    null_path = OUT / "selection_aware_null.json"
    if not null_path.exists():
        subprocess.run([
            PYTHON, "scripts/compute_topic4_zm_bridge_selection_null.py",
            "--config", str(CONFIG.relative_to(ROOT)), "--draws", "4096",
        ], check=True, cwd=ROOT)
    selection_seeds = list(map(int, config["fit"]["selection_seeds"]))
    _atomic(status_path, {"status": "SELECTION_LAUNCHING", "candidates": candidates,
                          "seeds": selection_seeds, "commit": _git_head()})
    _launch_stage("selection", candidates, selection_seeds, args.interval_seconds)
    selection_records = _score_stage("selection", candidates, selection_seeds, config)
    from src.topic4_fig5_target_informed_bridge import rank_selection_candidates
    ranked = rank_selection_candidates(selection_records, minimum_eligible=2)
    selection_payload = {"status": "SELECTION_COMPLETE", "records": selection_records,
                         "candidate_summary": ranked}
    _atomic(OUT / "selection_results.json", selection_payload)
    viable = [row for row in ranked if row["selection_eligible"]]
    if not viable:
        _atomic(status_path, {"status": "NO_SELECTION_CANDIDATE_CONFIRMED_2_OF_3"})
        return
    winner_id = viable[0]["candidate_id"]
    winner = next(row for row in candidates if row["candidate_id"] == winner_id)
    workpoint = {
        "status": "WORKPOINT_TARGET_INFORMED_FROZEN",
        "candidate_id": winner_id,
        "parameters": winner["parameters"],
        "selection_summary": viable[0],
        "frozen_commit": _git_head(),
        "readout_algorithm": config["model_readout"],
        "claim_boundary": "development-only target-informed Z/M bridge",
    }
    _atomic(OUT / "WORKPOINT_TARGET_INFORMED_FROZEN.json", workpoint)
    confirmation_seeds = list(map(int, config["fit"]["confirmation_seeds"]))
    _launch_stage("confirmation", [winner], confirmation_seeds,
                  args.interval_seconds)
    confirmation_records = _score_stage(
        "confirmation", [winner], confirmation_seeds, config)
    n_eligible = sum(row["status"] == "BRIDGE_EVALUABLE"
                     for row in confirmation_records)
    confirmation = {
        "status": ("FROZEN_CONFIRMATION_PASS" if n_eligible >= 2
                   else "FROZEN_CONFIRMATION_FAIL"),
        "candidate_id": winner_id,
        "n_eligible": n_eligible,
        "n_seeds": len(confirmation_seeds),
        "records": confirmation_records,
    }
    _atomic(OUT / "confirmation_results.json", confirmation)
    _atomic(status_path, {"status": confirmation["status"],
                          "candidate_id": winner_id,
                          "finished_epoch": time.time()})
    if shutil.which("notify-send"):
        subprocess.run(["notify-send", "Topic 4 rev5", confirmation["status"]],
                       check=False)


if __name__ == "__main__":
    main()
