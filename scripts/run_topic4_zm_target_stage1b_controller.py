#!/usr/bin/env python3
"""Run the frozen bounded Z/M adaptation refinement and hand off selection."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python"
CONFIG = ROOT / "config/topic4_data_driven_zm_target_informed_bridge_v1.json"
BASE_CONFIG = "config/topic4_data_driven_zm_ictal_transition_v1.json"


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _memory_available_gib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError("MemAvailable is unavailable")


def _active(unit):
    return subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", unit],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def _launch(unit, command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    shell = ("exec /usr/bin/nohup env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
             "OPENBLAS_NUM_THREADS=1 " + " ".join(command)
             + f" >> {log_path} 2>&1")
    subprocess.run([
        "systemd-run", "--user", f"--unit={unit}", "--collect",
        f"--working-directory={ROOT}", "/bin/bash", "-lc", shell,
    ], check=True, cwd=ROOT)


def _candidates(config):
    fit = config["fit"]
    reference_i_th = float(fit["reference_i_th"])
    s_i = float(fit["stage1b_s_i"])
    tau_z = float(fit["stage1b_tau_z_ms"])
    g_reference = float(fit["stage1b_eta_tau_reference"])
    result = []
    for tau_m in map(float, fit["stage1b_tau_m_ms"]):
        for g_ratio in map(float, fit["stage1b_g_m_ratio"]):
            result.append({
                "candidate_id": f"stage1b_tm{int(tau_m):04d}_gm{int(round(g_ratio * 100)):03d}",
                "parameters": {
                    "I_th_EI": reference_i_th * s_i,
                    "tau_z": tau_z,
                    "tau_adp": tau_m,
                    "eta_m": g_reference * g_ratio / tau_m,
                    "E_to_E_dose": 1.0,
                    "E_to_I_dose": 1.0,
                },
            })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-seconds", type=float, default=600.0)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()
    config = _load(CONFIG)
    out = ROOT / config["output_root"]
    fit_root = out / "fit"
    status_path = out / "stage1b_controller.json"
    candidates = _candidates(config)
    # The centre point was already run in Stage 1 under this exact parameter set.
    centre = next(row for row in candidates
                  if row["parameters"]["tau_adp"] == 500.0
                  and abs(row["parameters"]["eta_m"]
                          - config["fit"]["reference_eta_m"]) < 1e-15)
    centre["reuse"] = "si080_tz2500"
    manifest = {
        "status": "STAGE1B_MANIFEST_FROZEN",
        "selection_role": "model-internal qualification boundary refinement",
        "patient_bridge_score_used_for_design": False,
        "seed": int(config["fit"]["fit_seed"]),
        "candidates": candidates,
    }
    _atomic(out / "stage1b_manifest.json", manifest)
    pending = [row for row in candidates if "reuse" not in row]
    units = {}
    seed = int(config["fit"]["fit_seed"])
    log_root = ROOT / "results/run_logs"
    while True:
        done = {row["candidate_id"]: (fit_root / f"{row['candidate_id']}.json").exists()
                for row in pending}
        if all(done.values()):
            break
        running = [unit for unit in units.values() if _active(unit)]
        available = _memory_available_gib()
        if available < 32.0:
            raise RuntimeError("memory reserve below 32 GiB")
        slots = max(0, min(int(args.max_workers) - len(running),
                           int((available - 32.0) // 5.0)))
        for row in pending:
            if slots <= 0:
                break
            candidate_id = row["candidate_id"]
            if done[candidate_id] or candidate_id in units:
                continue
            parameters = row["parameters"]
            unit = f"topic4-rev5-stage1b-{candidate_id}"
            stem = fit_root / candidate_id
            command = [
                PYTHON, "scripts/run_topic4_zm_joint_morphology_canary.py",
                "--config", BASE_CONFIG, "--seed", str(seed),
                "--i-th-ei", str(parameters["I_th_EI"]),
                "--eta-m", str(parameters["eta_m"]),
                "--tau-z", str(parameters["tau_z"]),
                "--tau-adp", str(parameters["tau_adp"]),
                "--ee-dose", "1", "--etoi-dose", "1", "--zm-mode", "z_plus_m",
                "--duration-ms", "10000", "--post-runaway-ms", "2000",
                "--out", str(stem.relative_to(ROOT)), "--save-spatial-frames",
            ]
            _launch(unit, command, log_root / f"{unit}.log")
            units[candidate_id] = unit
            slots -= 1
        active_now = [unit for unit in units.values() if _active(unit)]
        _atomic(status_path, {
            "status": "STAGE1B_RUNNING",
            "n_done": int(sum(done.values())),
            "n_expected": len(done),
            "n_running": len(active_now),
            "memory_available_gib": available,
            "disk_free_gib": shutil.disk_usage(ROOT).free / 1024**3,
            "timestamp_epoch": time.time(),
        })
        assigned_but_missing = [
            row["candidate_id"] for row in pending
            if not done[row["candidate_id"]] and row["candidate_id"] in units
        ]
        if assigned_but_missing and not active_now and all(
                row["candidate_id"] in units or done[row["candidate_id"]]
                for row in pending):
            raise RuntimeError(
                "refinement workers exited without artifacts: "
                + ", ".join(assigned_but_missing))
        time.sleep(float(args.interval_seconds))

    subprocess.run([
        PYTHON, "scripts/rescore_topic4_fig5_target_informed_candidates.py",
        "--config", str(CONFIG.relative_to(ROOT)),
    ], check=True, cwd=ROOT)
    rescored = _load(out / "existing_candidate_rescore.json")
    eligible = [row for row in rescored["records"]
                if row.get("primary_zm_only")
                and row.get("status") == "BRIDGE_EVALUABLE"]
    if not eligible:
        _atomic(status_path, {
            "status": "NO_STAGE1_ZM_ONLY_ELIGIBLE",
            "n_full_dose_candidates": sum(
                bool(row.get("primary_zm_only")) for row in rescored["records"]),
            "next_action": "bounded Z/M refinement exhausted; no patient selection launched",
        })
        _atomic(out / "selection_controller.json", {
            "status": "NO_STAGE1_ZM_ONLY_ELIGIBLE",
            "source": "stage1b_controller",
        })
    else:
        _atomic(status_path, {
            "status": "STAGE1B_RESCORE_COMPLETE",
            "n_full_dose_eligible": len(eligible),
        })
        unit = "topic4-rev5-selection-controller-r4"
        if not _active(unit):
            _launch(unit, [
                PYTHON, "scripts/run_topic4_zm_target_selection_controller.py",
                "--interval-seconds", str(args.interval_seconds),
            ], log_root / f"{unit}.log")


if __name__ == "__main__":
    main()
