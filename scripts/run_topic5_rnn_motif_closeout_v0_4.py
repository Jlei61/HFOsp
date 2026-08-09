#!/usr/bin/env python3
"""Safely refresh v0.4 summaries and figures after the immutable run finishes.

The primary early-ictal scorer is deliberately absent: it is the single target
unseal operation and must never be invoked twice.  This driver only recomputes
target-free summaries, the prespecified secondary lesion readout, common
observables, figures, and tests with the audited closeout code.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_UNSEAL_SCRIPT = "score_topic5_rnn_motif_early_ictal_v0_4.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def closeout_commands(
    out_root: Path, target_cache_root: Path, snn_readout: Path,
) -> list[tuple[str, list[str]]]:
    """Return the fixed, safe post-unseal closeout sequence."""
    py = sys.executable
    scripts = ROOT / "scripts"
    root_args = ["--out-root", str(out_root)]
    tests = [
        "tests/test_topic5_rnn_motif_v0_4.py",
        "tests/test_topic5_spatial_latent_rnn.py",
        "tests/test_topic5_we_cache.py",
        "tests/test_topic5_we_graph_analysis.py",
        "tests/test_topic5_we_train.py",
        "tests/test_topic5_wiring_economy_rnn.py",
    ]
    return [
        ("export_contracts", [
            py, str(scripts / "export_topic5_rnn_motif_unit_contracts_v0_4.py"),
            *root_args,
        ]),
        ("interictal_summary", [
            py, str(scripts / "analyse_topic5_rnn_motif_interictal_v0_4.py"),
            *root_args, "--device", "cuda",
        ]),
        ("interictal_figure", [
            py, str(scripts / "plot_topic5_rnn_motif_figures_v0_4.py"),
            *root_args, "--stage", "interictal",
        ]),
        ("fields_figure", [
            py, str(scripts / "plot_topic5_rnn_motif_figures_v0_4.py"),
            *root_args, "--stage", "fields",
        ]),
        ("lesion_aggregate", [
            py, str(scripts / "run_topic5_rnn_motif_matched_lesions_v0_4.py"),
            *root_args, "--aggregate-only",
        ]),
        ("theory_summary", [
            py, str(scripts / "summarize_topic5_rnn_motif_theory_v0_4.py"),
            *root_args, "--draws", "1000",
        ]),
        ("motif_figure", [
            py, str(scripts / "plot_topic5_rnn_motif_figures_v0_4.py"),
            *root_args, "--stage", "motif",
        ]),
        ("lesion_early_ictal", [
            py, str(scripts / "score_topic5_rnn_motif_lesion_early_ictal_v0_4.py"),
            *root_args, "--target-cache-root", str(target_cache_root),
            "--n-perm", "5000",
        ]),
        ("early_figure", [
            py, str(scripts / "plot_topic5_rnn_motif_figures_v0_4.py"),
            *root_args, "--stage", "early",
        ]),
        ("common_observables", [
            py, str(scripts / "build_topic5_rnn_motif_common_observables_v0_4.py"),
            *root_args, "--snn-readout", str(snn_readout),
        ]),
        ("final_figure", [
            py, str(scripts / "plot_topic5_rnn_motif_figures_v0_4.py"),
            *root_args, "--stage", "final",
        ]),
        ("focused_tests", [py, "-m", "pytest", *tests, "-q"]),
    ]


def validate_ready(out_root: Path) -> None:
    required = [
        "POSTPROCESS_READY_FOR_VISUAL_QA.json",
        "target_access_audit.json",
        "early_ictal_per_patient_model.csv",
        "MATCHED_LESION_SUMMARY.json",
    ]
    missing = [name for name in required if not (out_root / name).is_file()]
    if missing:
        raise RuntimeError(
            "immutable postprocess is not ready for closeout; missing " + ", ".join(missing)
        )
    access = json.loads((out_root / "target_access_audit.json").read_text())
    if access.get("training_or_model_selection_after_unseal") is not False:
        raise RuntimeError("target access audit does not preserve the frozen-model contract")


def wait_for_ready(out_root: Path, poll_seconds: int, max_wait_hours: float) -> None:
    """Wait without touching scientific artifacts until the immutable run is ready."""
    started = time.monotonic()
    status_path = out_root / "CLOSEOUT_WAIT_STATUS.json"
    while not (out_root / "POSTPROCESS_READY_FOR_VISUAL_QA.json").is_file():
        failure = out_root / "PIPELINE_FAILED.json"
        if failure.is_file():
            raise RuntimeError(f"immutable postprocess failed; see {failure}")
        elapsed_hours = (time.monotonic() - started) / 3600.0
        if elapsed_hours > max_wait_hours:
            raise TimeoutError(
                f"immutable postprocess did not become ready within {max_wait_hours:g} hours"
            )
        atomic_json(status_path, {
            "status": "WAITING_FOR_IMMUTABLE_POSTPROCESS",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_hours": elapsed_hours,
            "target_values_read_by_waiter": False,
        })
        time.sleep(max(10, int(poll_seconds)))
    atomic_json(status_path, {
        "status": "IMMUTABLE_POSTPROCESS_READY",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_hours": (time.monotonic() - started) / 3600.0,
        "target_values_read_by_waiter": False,
    })


def run_step(name: str, command: list[str], out_root: Path, code_hash: str) -> None:
    if any(PRIMARY_UNSEAL_SCRIPT in token for token in command):
        raise RuntimeError("closeout must never invoke the primary target-unseal scorer")
    marker = out_root / "closeout_status" / f"{name}.DONE.json"
    if marker.exists():
        payload = json.loads(marker.read_text())
        if payload.get("closeout_code_hash") != code_hash or payload.get("command") != command:
            raise RuntimeError(f"stale closeout marker for {name}")
        return
    log = out_root / "closeout_logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    with log.open("w") as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        atomic_json(out_root / "CLOSEOUT_FAILED.json", {
            "status": "FAILED", "stage": name, "returncode": result.returncode,
            "log": str(log), "started_utc": started,
        })
        raise RuntimeError(f"closeout stage failed: {name}; see {log}")
    atomic_json(marker, {
        "stage": name, "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "command": command, "closeout_code_hash": code_hash,
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--target-cache-root", type=Path, required=True)
    parser.add_argument("--snn-readout", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait-for-ready", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--max-wait-hours", type=float, default=48.0)
    args = parser.parse_args()
    out_root = args.out_root.resolve()
    target_cache_root = args.target_cache_root.resolve()
    snn_readout = args.snn_readout.resolve()
    commands = closeout_commands(out_root, target_cache_root, snn_readout)
    if any(any(PRIMARY_UNSEAL_SCRIPT in token for token in command)
           for _, command in commands):
        raise RuntimeError("unsafe primary target scorer found in closeout plan")
    if args.dry_run:
        print(json.dumps({"stages": [name for name, _ in commands],
                          "primary_unseal_called": False}, indent=2))
        return 0
    if args.wait_for_ready:
        wait_for_ready(out_root, args.poll_seconds, args.max_wait_hours)
    validate_ready(out_root)
    script_hash = sha256(Path(__file__))
    atomic_json(out_root / "CLOSEOUT_CONTRACT.json", {
        "contract": "topic5_rnn_motif_safe_post_unseal_closeout_v0_4",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "closeout_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "closeout_code_hash": script_hash,
        "primary_target_unseal_repeated": False,
        "stages": [name for name, _ in commands],
    })
    for name, command in commands:
        run_step(name, command, out_root, script_hash)
    atomic_json(out_root / "CLOSEOUT_READY_FOR_VISUAL_QA.json", {
        "status": "READY_FOR_MANUAL_VISUAL_QA",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "primary_target_unseal_repeated": False,
        "test_log": str(out_root / "closeout_logs" / "focused_tests.log"),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
