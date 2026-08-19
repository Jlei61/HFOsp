#!/usr/bin/env python3
"""Launcher for the Z/M ictal-transition round.

Worker counts are MEASURED before every round, per job class, and re-measured as
the round proceeds. This machine is shared -- a co-tenant starting mid-round must
shrink our pool, not theirs -- so the launcher never signals a process outside
its own unit prefix.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_zm_ictal_transition import load_round_config  # noqa: E402

PYTHON = sys.executable
WORKER = ROOT / "scripts/run_topic4_zm_ictal_transition_worker.py"
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS")


def _mem_available_gib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / (1024 ** 2)
    raise RuntimeError("MemAvailable not readable")


def _free_cores():
    load1 = float(Path("/proc/loadavg").read_text().split()[0])
    return max(0.0, os.cpu_count() - load1)


def _pool_size(config, job_class, prefix=""):
    """Measured, not fixed. Recomputed before every launch round.

    Our OWN running jobs are added back to the free-core count. Load average
    includes them, so subtracting it without compensation makes every launch
    shrink the next pool -- a negative feedback loop that capped the round at
    two workers while 129 GiB and a fair share of cores were free. The
    co-tenant's load still shrinks our pool, which is the intended direction.
    """
    execution = config["execution"]
    per_worker = float(execution[f"measured_{job_class}_peak_rss_gib"])
    cap = int(execution[f"max_workers_{job_class}"])
    reserve = float(execution["minimum_available_memory_gib"])
    core_margin = int(execution["minimum_free_cores"])
    mine = len(_active(prefix)) if prefix else 0
    by_memory = int((_mem_available_gib() - reserve) // per_worker)
    by_cores = int(_free_cores()) + mine - core_margin
    return max(1, min(cap, by_memory, by_cores))


def _log(path, record):
    record["mem_available_gib"] = round(_mem_available_gib(), 1)
    record["free_cores"] = round(_free_cores(), 1)
    record["disk_free_gib"] = round(shutil.disk_usage(ROOT).free / 2 ** 30, 1)
    record["t"] = time.strftime("%H:%M:%S")
    with open(path, "a") as handle:
        handle.write(json.dumps(record) + "\n")


def _launch(unit, command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = [f"--setenv={name}=1" for name in THREAD_ENV]
    # StandardOutput/Error append: without these the SERVICE's output goes to the
    # journal and only systemd-run's own one-line output reaches log_path, so a
    # crashing worker leaves an empty log file.
    full = ["systemd-run", "--user", f"--unit={unit}", "--quiet", *env,
            "--working-directory", str(ROOT),
            f"--property=StandardOutput=append:{log_path}",
            f"--property=StandardError=append:{log_path}",
            "/usr/bin/nohup", "/usr/bin/time", "-v", *command]
    log_path.touch()
    outcome = subprocess.run(full, cwd=ROOT, capture_output=True, text=True)
    if outcome.returncode != 0:
        # Losing the whole launcher because one unit name collided cost an hour
        # of wall clock. Report and carry on with the rest of the queue.
        return {"unit": unit, "launched": False,
                "reason": (outcome.stderr or outcome.stdout or "").strip()[:200]}
    return {"unit": unit, "launched": True}


def _active(prefix):
    out = subprocess.run(["systemctl", "--user", "list-units", "--no-legend",
                          "--plain", "--state=active", f"{prefix}*"],
                         capture_output=True, text=True).stdout
    return [line.split()[0] for line in out.splitlines()
            if line.strip() and ".service" in line.split()[0]]


def _reset_failed(prefix):
    subprocess.run(["systemctl", "--user", "reset-failed", f"{prefix}*"],
                   capture_output=True)


def _existing_units(prefix):
    out = subprocess.run(["systemctl", "--user", "list-units", "--all",
                          "--no-legend", "--plain", f"{prefix}*"],
                         capture_output=True, text=True).stdout
    return {line.split()[0] for line in out.splitlines() if line.strip()}


def _already_done(command):
    """Skip a job whose --out-json is already on disk. Makes the launcher
    resumable after a restart instead of colliding with live units."""
    if "--out-json" in command:
        return Path(command[command.index("--out-json") + 1]).exists()
    return False


def _run_pool(jobs, config, job_class, prefix, controller_log, poll=20):
    """jobs: list of (unit_suffix, command, log_path). Nothing is killed."""
    # systemctl reports unit names WITH the .service suffix; comparing against
    # the bare name silently matched nothing, so the skip never fired and the
    # launcher died on the first duplicate unit -- twelve of fifteen jobs never
    # started and the chain was about to read that as a science verdict.
    live = _existing_units(prefix)
    pending = [j for j in jobs
               if f"{prefix}{j[0]}.service" not in live and not _already_done(j[1])]
    skipped = len(jobs) - len(pending)
    if skipped:
        _log(controller_log, {"progress": "skipped_already_running_or_done",
                              "n": skipped, "prefix": prefix})
    while pending or _active(prefix):
        size = _pool_size(config, job_class, prefix)
        running = _active(prefix)
        while pending and len(running) < size:
            suffix, command, log_path = pending.pop(0)
            outcome = _launch(f"{prefix}{suffix}", command, log_path)
            running = _active(prefix)
            _log(controller_log, {"progress": "launched" if outcome["launched"]
                                  else "launch_failed", **outcome,
                                  "active": len(running), "pending": len(pending),
                                  "pool": size})
        if not pending and not _active(prefix):
            break
        time.sleep(poll)
    _reset_failed(prefix)


def _prebuild(config, seeds, controller_log, output_root, config_path):
    """Build each network ONCE. Four arms on one seed share one network; letting
    them race would rebuild it four times at 63 s and 2 GiB apiece."""
    cache = output_root / "network_cache"
    jobs = [(f"s{seed}",
             [PYTHON, "-c",
              "import sys; sys.path.insert(0, '.'); "
              "from src.topic4_zm_ictal_transition import build_substrate, load_round_config; "
              f"build_substrate(load_round_config('{config_path}'), "
              f"'node_baseline', {seed}, cache_dir='{cache}')"],
             output_root / "run_logs" / f"prebuild_s{seed}.log")
            for seed in seeds]
    _log(controller_log, {"progress": "prebuild_start", "seeds": list(seeds)})
    _run_pool(jobs, config, "full_run", "topic4-zmitx-prebuild-", controller_log)
    _log(controller_log, {"progress": "prebuild_done"})


def phase_canary(config, args):
    output_root = ROOT / config["output_root"]
    controller_log = output_root / "controller.log"
    seeds = list(config["seeds"]["canary"])
    _prebuild(config, seeds, controller_log, output_root, args.config)

    arms = config["arms"]
    extra = ["--allow-uncommitted-config"] if args.allow_uncommitted_config else []
    jobs = []
    for seed in seeds:
        for arm_name in config["phases"]["canary_arms"]:
            candidate = arms[arm_name]
            slug = candidate.replace("+", "_")
            checkpoints = (["--emit-onset-checkpoints"] if arm_name in
                           config["phases"]["onset_relative_checkpoints_for_arms"] else [])
            jobs.append((f"{slug}-s{seed}",
                         [PYTHON, str(WORKER), "--config", args.config,
                          "--candidate-id", candidate, "--seed", str(seed),
                          "--expected-commit", args.expected_commit,
                          "--zm-mode", "z_plus_m", *checkpoints, *extra],
                         output_root / "run_logs" / f"{slug}_s{seed}.log"))
        if config["phases"]["canary_zm_off_paired"]:
            jobs.append((f"zmoff-s{seed}",
                         [PYTHON, str(WORKER), "--config", args.config,
                          "--candidate-id", arms["Joint"], "--seed", str(seed),
                          "--expected-commit", args.expected_commit,
                          "--zm-mode", "off", *extra],
                         output_root / "run_logs" / f"zmoff_s{seed}.log"))
    _log(controller_log, {"progress": "canary_start", "n_jobs": len(jobs)})
    _run_pool(jobs, config, "full_run", "topic4-zmitx-canary-", controller_log)
    _log(controller_log, {"progress": "canary_done", "n_jobs": len(jobs)})
    atomic_write_json({"phase": "canary", "n_jobs": len(jobs), "seeds": seeds,
                       "arms": config["phases"]["canary_arms"],
                       "zm_off_paired": config["phases"]["canary_zm_off_paired"],
                       "expected_commit": args.expected_commit},
                      str(output_root / "canary_launch.json"))
    print(json.dumps({"phase": "canary", "launched": len(jobs)}))


PERTURB = ROOT / "scripts/run_topic4_zm_perturbation_worker.py"


def _checkpoint(output_root, candidate, seed, label):
    path = output_root / "checkpoints" / f"{candidate}_seed_{seed}_{label}.npz"
    return path if path.exists() else None


def phase_dose(config, args):
    """Ladder x 6 representative sites x 3 canary seeds, BASELINE only.

    The worker itself refuses any label other than baseline for this phase, so
    the dose can never be tuned on a pre-ictal or patient-derived quantity.
    """
    output_root = ROOT / config["output_root"]
    controller_log = output_root / "controller.log"
    joint = config["arms"]["Joint"]
    extra = ["--allow-uncommitted-config"] if args.allow_uncommitted_config else []
    jobs = []
    for rung in config["perturbation"]["dose_ladder_cells"]:
        for seed in config["seeds"]["canary"]:
            checkpoint = _checkpoint(output_root, joint, seed, "low_activity")
            if checkpoint is None:
                continue
            out = output_root / "dose" / f"{joint}_seed_{seed}_baseline_n{rung}"
            jobs.append((f"n{rung}-s{seed}",
                         [PYTHON, str(PERTURB), "--config", args.config,
                          "--candidate-id", joint, "--seed", str(seed),
                          "--checkpoint", str(checkpoint), "--label", "baseline",
                          "--sites", "representative", "--dose-cells", str(rung),
                          "--expected-commit", args.expected_commit,
                          "--out-json", str(out) + ".json",
                          "--out-npz", str(out) + ".npz", *extra],
                         output_root / "run_logs" / f"dose_n{rung}_s{seed}.log"))
    _log(controller_log, {"progress": "dose_start", "n_jobs": len(jobs)})
    _run_pool(jobs, config, "probe", "topic4-zmitx-dose-", controller_log, poll=10)
    _log(controller_log, {"progress": "dose_done", "n_jobs": len(jobs)})
    print(json.dumps({"phase": "dose", "launched": len(jobs)}))


def phase_counterfactual(config, args):
    """Six branches x 6 representative sites x 3 canary seeds at the frozen dose.

    Four of the six are spliced states, which are OFF-MANIFOLD: the dynamics
    never visit 'pre-ictal fast state with baseline z'. They answer which
    variable is consistent with carrying the rise, not what would have happened.
    """
    output_root = ROOT / config["output_root"]
    controller_log = output_root / "controller.log"
    joint = config["arms"]["Joint"]
    dose = json.loads((output_root / "dose_freeze.json").read_text())
    if dose["status"] != "PASS":
        raise SystemExit("no frozen dose -- run the dose gate first")
    cells = int(dose["selected_dose_cells"])
    extra = ["--allow-uncommitted-config"] if args.allow_uncommitted_config else []
    jobs = []
    for seed in config["seeds"]["canary"]:
        baseline = _checkpoint(output_root, joint, seed, "low_activity")
        pre_ictal = _checkpoint(output_root, joint, seed, "pre_ictal")
        if baseline is None or pre_ictal is None:
            continue
        for mode in config["counterfactual_splices"]:
            host, label = (baseline, "baseline") if mode in (
                "native_baseline", "slow_only") else (pre_ictal, "pre_ictal")
            splice = "native" if mode.startswith("native") else mode
            out = output_root / "counterfactual" / f"{joint}_seed_{seed}_{mode}"
            jobs.append((f"{mode}-s{seed}",
                         [PYTHON, str(PERTURB), "--config", args.config,
                          "--candidate-id", joint, "--seed", str(seed),
                          "--checkpoint", str(host),
                          "--baseline-checkpoint", str(baseline),
                          "--label", label, "--splice", splice,
                          "--sites", "representative", "--dose-cells", str(cells),
                          "--expected-commit", args.expected_commit,
                          "--out-json", str(out) + ".json",
                          "--out-npz", str(out) + ".npz", *extra],
                         output_root / "run_logs" / f"cf_{mode}_s{seed}.log"))
    _log(controller_log, {"progress": "counterfactual_start", "n_jobs": len(jobs)})
    _run_pool(jobs, config, "probe", "topic4-zmitx-cf-", controller_log, poll=10)
    _log(controller_log, {"progress": "counterfactual_done", "n_jobs": len(jobs)})
    print(json.dumps({"phase": "counterfactual", "launched": len(jobs)}))


def phase_fig5(config, args):
    """Joint-arm response fields for the paper-facing Figure 5.

    The accepted layout compares the response to one frozen spatial probe at
    low activity and pre-transition. Six geometry-defined representative sites
    are retained in the artifact, so the displayed source-site response is not
    selected after looking at the response. A 7x7 scan asks a different
    susceptibility-mapping question and is deferred to the connectivity round.
    """
    output_root = ROOT / config["output_root"]
    controller_log = output_root / "controller.log"
    joint = config["arms"]["Joint"]
    dose = json.loads((output_root / "dose_freeze.json").read_text())
    if dose["status"] != "PASS":
        raise SystemExit("no frozen dose; Figure 5 D/E/F cannot be built")
    cells = int(dose["selected_dose_cells"])
    extra = ["--allow-uncommitted-config"] if args.allow_uncommitted_config else []
    jobs = []
    for seed in config["seeds"]["canary"]:
        for checkpoint_label, label in (("low_activity", "baseline"),
                                        ("pre_ictal", "pre_ictal")):
            checkpoint = _checkpoint(output_root, joint, seed, checkpoint_label)
            if checkpoint is None:
                continue
            out = output_root / "perturbation" / f"{joint}_seed_{seed}_{checkpoint_label}_representative"
            jobs.append((f"{label}-s{seed}",
                         [PYTHON, str(PERTURB), "--config", args.config,
                          "--candidate-id", joint, "--seed", str(seed),
                          "--checkpoint", str(checkpoint), "--label", label,
                          "--sites", "representative", "--dose-cells", str(cells),
                          "--expected-commit", args.expected_commit,
                          "--out-json", str(out) + ".json",
                          "--out-npz", str(out) + ".npz", *extra],
                         output_root / "run_logs" / f"fig5_{label}_s{seed}.log"))
    _log(controller_log, {"progress": "fig5_start", "n_jobs": len(jobs),
                          "dose_cells": cells})
    _run_pool(jobs, config, "probe", "topic4-zmitx-fig5-", controller_log, poll=15)
    _log(controller_log, {"progress": "fig5_done", "n_jobs": len(jobs)})
    print(json.dumps({"phase": "fig5", "launched": len(jobs), "dose_cells": cells}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", required=True,
                        choices=("canary", "dose", "counterfactual", "fig5"))
    parser.add_argument("--expected-commit", default="HEAD")
    parser.add_argument("--allow-uncommitted-config", action="store_true")
    args = parser.parse_args()
    config = load_round_config(args.config)
    args.expected_commit = subprocess.check_output(
        ["git", "rev-parse", args.expected_commit], cwd=ROOT, text=True).strip()
    (ROOT / config["output_root"] / "run_logs").mkdir(parents=True, exist_ok=True)
    {"canary": phase_canary, "dose": phase_dose,
     "counterfactual": phase_counterfactual,
     "fig5": phase_fig5}[args.phase](config, args)


if __name__ == "__main__":
    main()
