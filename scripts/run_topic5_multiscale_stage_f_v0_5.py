#!/usr/bin/env python3
"""Run and freeze every target-free v0.5 Stage-F analysis."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
STEPS = (
    ("prefix_template", "analyse_topic5_prefix_template_v0_5.py", (), "PREFIX_TEMPLATE_SUMMARY.json"),
    ("intact_fields", "build_topic5_multiscale_fields_v0_5.py", (), "MODEL_FIELDS_FROZEN.json"),
    ("mechanism", "analyse_topic5_multiscale_mechanism_v0_5.py", ("--workers", "8"), "MECHANISM_ANALYSIS_COMPLETE.json"),
    ("mode_flow_attenuation", "run_topic5_mode_flow_attenuation_v0_5.py", ("--workers", "8"), "MODE_FLOW_ATTENUATION_COMPLETE.json"),
    ("arm_attenuation", "run_topic5_multiscale_attenuation_v0_5.py", ("--workers", "8"), "ATTENUATED_FIELDS_FROZEN.json"),
    # Exact spectral-norm SVDs saturate one GPU.  Multiple independent CUDA
    # processes only contend for the same solver and reduce total throughput;
    # the estimator, prefixes and seeds are unchanged at one worker.
    ("gain_adjustment", "run_topic5_gain_matched_sensitivity_v0_5.py", ("--workers", "1"), "GAIN_ADJUSTED_SENSITIVITY_COMPLETE.json"),
    ("figure", "plot_topic5_multiscale_stage_f_v0_5.py", (),
     "figures/stage_f_v0_5_target_free_mechanism.png"),
)
DEPENDENCIES = (
    "scripts/build_topic5_rnn_motif_fields_v0_4.py",
    "scripts/run_topic5_lbss_attenuation_v0_2.py",
    "scripts/analyse_topic5_rnn_motif_influence_v0_4.py",
    "src/topic5_lbss_analysis_v0_2.py",
    "src/topic5_lbss_rnn_v0_2.py",
    "src/topic5_rnn_motif_v0_4.py",
    "src/topic5_wiring_economy_rnn.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out_root.resolve()
    if os.environ.get("TOPIC5_V0_5_TARGET_SEALED") != "1":
        raise RuntimeError("Stage F must run inside the physical target embargo wrapper")
    if not (out / "STAGE_E_INTERICTAL_ANALYSIS_COMPLETE.json").exists():
        raise RuntimeError("Stage E analysis must finish before Stage F")
    source_hashes = {f"scripts/{script}": sha256_file(ROOT / "scripts" / script)
                     for _, script, _, _ in STEPS}
    source_hashes.update({relative: sha256_file(ROOT / relative) for relative in DEPENDENCIES})
    snapshot = {
        "contract": "topic5_multiscale_stage_f_v0_5",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_hashes": source_hashes, "target_values_read": False,
    }
    (out / "STAGE_F_RUN_SNAPSHOT.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    log_root = out / "stage_f_logs"; log_root.mkdir(exist_ok=True)
    completed = []
    for label, script, extra, completion_relative in STEPS:
        if (out / completion_relative).exists():
            completed.append(label)
            continue
        command = [sys.executable, str(ROOT / "scripts" / script),
                   "--out-root", str(out), *extra]
        destination = log_root / f"{label}.log"
        with destination.open("a") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                    check=False)
        if result.returncode != 0:
            (out / "STAGE_F_TARGET_FREE_FAILED.json").write_text(json.dumps({
                "status": "FAILED", "step": label, "returncode": result.returncode,
                "log": str(destination), "completed": completed,
                "target_values_read": False,
            }, indent=2) + "\n")
            raise SystemExit(result.returncode)
        if sha256_file(ROOT / "scripts" / script) != source_hashes[f"scripts/{script}"]:
            raise RuntimeError(f"Stage F source changed while active: {script}")
        completed.append(label)
    for relative, digest in source_hashes.items():
        if sha256_file(ROOT / relative) != digest:
            raise RuntimeError(f"Stage F dependency changed while active: {relative}")
    required = (
        "PREFIX_TEMPLATE_SUMMARY.json", "MODEL_FIELDS_FROZEN.json",
        "MECHANISM_ANALYSIS_COMPLETE.json", "MODE_FLOW_ATTENUATION_COMPLETE.json",
        "ATTENUATED_FIELDS_FROZEN.json", "GAIN_ADJUSTED_SENSITIVITY_COMPLETE.json",
    )
    artifact_hashes = {relative: sha256_file(out / relative) for relative in required}
    (out / "STAGE_F_TARGET_FREE_COMPLETE.json").write_text(json.dumps({
        "status": "PASS_TARGET_FREE", "created_utc": datetime.now(timezone.utc).isoformat(),
        "completed_steps": completed, "source_hashes": source_hashes,
        "artifact_hashes": artifact_hashes, "target_values_read": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
