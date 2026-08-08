#!/usr/bin/env python3
"""Resumable post-training driver for the locked v0.4 scientific chain."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("core", "dose", "gru")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()


def atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n"); temporary.replace(path)


def git_state() -> tuple[str, str]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    return commit, status


def check_lock(contract: dict[str, Any]) -> None:
    commit, status = git_state()
    if commit != contract["git_commit"] or status:
        raise RuntimeError(f"postprocess worktree changed: commit={commit}, status={status!r}")


def wait_training(out_root: Path, contract: dict[str, Any]) -> None:
    while True:
        check_lock(contract)
        ready = True
        state = {}
        for stage in STAGES:
            path = out_root / f"STAGE_{stage.upper()}_STATUS.json"
            if not path.exists(): ready = False; continue
            payload = json.loads(path.read_text()); state[stage] = payload
            ready &= (int(payload.get("remaining", -1)) == 0 and int(payload.get("failed", -1)) == 0
                      and int(payload.get("oom", -1)) == 0 and int(payload.get("nonfinite", -1)) == 0)
        atomic(out_root / "POSTPROCESS_WAIT_STATUS.json", {
            "updated_utc": datetime.now(timezone.utc).isoformat(), "training": state, "ready": ready,
        })
        if ready: return
        time.sleep(60)


def run_step(name: str, command: list[str], out_root: Path, contract: dict[str, Any]) -> None:
    marker = out_root / "postprocess_status" / f"{name}.DONE.json"
    if marker.exists(): return
    check_lock(contract)
    log_dir = out_root / "postprocess_logs"; log_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    with (log_dir / f"{name}.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        atomic(out_root / "PIPELINE_FAILED.json", {
            "status": "FAILED", "stage": name, "returncode": result.returncode,
            "log": str(log_dir / f"{name}.log"), "started_utc": started,
        })
        raise RuntimeError(f"postprocess stage failed: {name}")
    marker.parent.mkdir(parents=True, exist_ok=True)
    atomic(marker, {"stage": name, "started_utc": started,
                    "finished_utc": datetime.now(timezone.utc).isoformat(), "command": command})


def run_shards(name: str, base: list[str], workers: int, out_root: Path,
               contract: dict[str, Any]) -> None:
    marker = out_root / "postprocess_status" / f"{name}.DONE.json"
    if marker.exists(): return
    check_lock(contract)
    log_dir = out_root / "postprocess_logs"; log_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    for shard in range(workers):
        command = base + ["--shard-index", str(shard), "--n-shards", str(workers)]
        handle = (log_dir / f"{name}_shard{shard:02d}.log").open("w")
        process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        processes.append((process, handle, command))
    failed = []
    for process, handle, command in processes:
        code = process.wait(); handle.close()
        if code: failed.append({"returncode": code, "command": command})
    if failed:
        atomic(out_root / "PIPELINE_FAILED.json", {"status": "FAILED", "stage": name, "failed": failed})
        raise RuntimeError(f"postprocess shards failed: {name}: {failed}")
    marker.parent.mkdir(parents=True, exist_ok=True)
    atomic(marker, {"stage": name, "finished_utc": datetime.now(timezone.utc).isoformat(),
                    "workers": workers, "commands": [item[2] for item in processes]})


def snapshot(contract_path: Path, scripts: list[Path], commit: str) -> dict[str, Any]:
    directory = contract_path.parent / "postprocess_snapshot"
    directory.mkdir(parents=True, exist_ok=True)
    inventory = {}
    for source in scripts:
        target = directory / source.name
        shutil.copy2(source, target)
        inventory[source.name] = {"source": str(source), "snapshot": str(target), "sha256": sha256(target)}
    payload = {"contract": "topic5_rnn_motif_postprocess_snapshot_v0_4",
               "created_utc": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
               "scripts": inventory}
    atomic(contract_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--target-cache-root", type=Path, required=True)
    parser.add_argument("--snn-readout", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    out_root = args.out_root.resolve(); py = sys.executable
    script_names = [
        "run_topic5_rnn_motif_postprocess_v0_4.py",
        "analyse_topic5_rnn_motif_interictal_v0_4.py", "audit_topic5_rnn_motif_target_metadata_v0_4.py",
        "build_topic5_rnn_motif_fields_v0_4.py", "plot_topic5_rnn_motif_figures_v0_4.py",
        "analyse_topic5_rnn_motif_influence_v0_4.py", "run_topic5_rnn_motif_matched_lesions_v0_4.py",
        "summarize_topic5_rnn_motif_theory_v0_4.py", "score_topic5_rnn_motif_early_ictal_v0_4.py",
        "score_topic5_rnn_motif_lesion_early_ictal_v0_4.py",
        "build_topic5_rnn_motif_common_observables_v0_4.py", "finalize_topic5_rnn_motif_v0_4.py",
        "plot_topic5_rnn_motif_figures_v0_4.py",
    ]
    scripts = [ROOT / "scripts" / name for name in dict.fromkeys(script_names)]
    scripts.extend(ROOT / path for path in (
        "scripts/train_topic5_we_unit.py",
        "scripts/plot_topic5_interictal_template_ab_fields.py",
        "scripts/plot_topic5_field_vs_ictal_swap.py",
        "scripts/plot_topic5_interictal_event_envelope_field.py",
        "scripts/paper_figures/plot_fig3b_interictal_ictal_shared_field.py",
        "src/topic5_rnn_motif_v0_4.py",
        "src/topic5_wiring_economy_rnn.py",
        "src/topic5_gradient_grid_field.py",
        "src/topic5_template_axis_field.py",
    ))
    contract_path = out_root / "POSTPROCESS_CONTRACT.json"
    if contract_path.exists():
        contract = json.loads(contract_path.read_text())
    else:
        commit, status = git_state()
        if status: raise RuntimeError(f"refusing to lock dirty worktree: {status}")
        contract = snapshot(contract_path, scripts, commit)
    wait_training(out_root, contract)
    root_args = ["--out-root", str(out_root)]
    run_step("D_interictal", [py, str(ROOT / "scripts/analyse_topic5_rnn_motif_interictal_v0_4.py"),
                              *root_args, "--device", "cuda"], out_root, contract)
    run_step("D_figure", [py, str(ROOT / "scripts/plot_topic5_rnn_motif_figures_v0_4.py"),
                          *root_args, "--stage", "interictal"], out_root, contract)
    run_step("E_metadata", [py, str(ROOT / "scripts/audit_topic5_rnn_motif_target_metadata_v0_4.py"),
                            *root_args, "--target-cache-root", str(args.target_cache_root.resolve())], out_root, contract)
    run_step("E_fields", [py, str(ROOT / "scripts/build_topic5_rnn_motif_fields_v0_4.py"),
                          *root_args], out_root, contract)
    run_step("E_figure", [py, str(ROOT / "scripts/plot_topic5_rnn_motif_figures_v0_4.py"),
                          *root_args, "--stage", "fields"], out_root, contract)

    influence_base = [py, str(ROOT / "scripts/analyse_topic5_rnn_motif_influence_v0_4.py"),
                      *root_args, "--device", "cuda", "--max-prefixes", "32"]
    run_shards("G_influence_shards", influence_base, args.workers, out_root, contract)
    run_step("G_influence_aggregate", influence_base + ["--aggregate-only"], out_root, contract)
    lesion_base = [py, str(ROOT / "scripts/run_topic5_rnn_motif_matched_lesions_v0_4.py"),
                   *root_args, "--device", "cuda", "--target-draws", "500", "--max-events", "32"]
    run_shards("G_lesion_shards", lesion_base, args.workers, out_root, contract)
    run_step("G_lesion_aggregate", lesion_base + ["--aggregate-only"], out_root, contract)
    run_step("G_theory", [py, str(ROOT / "scripts/summarize_topic5_rnn_motif_theory_v0_4.py"),
                          *root_args, "--draws", "1000"], out_root, contract)
    run_step("G_figure", [py, str(ROOT / "scripts/plot_topic5_rnn_motif_figures_v0_4.py"),
                          *root_args, "--stage", "motif"], out_root, contract)

    run_step("F_early_ictal", [py, str(ROOT / "scripts/score_topic5_rnn_motif_early_ictal_v0_4.py"),
                               *root_args, "--target-cache-root", str(args.target_cache_root.resolve()),
                               "--n-perm", "5000"], out_root, contract)
    run_step("F_lesion_early", [py, str(ROOT / "scripts/score_topic5_rnn_motif_lesion_early_ictal_v0_4.py"),
                                *root_args, "--target-cache-root", str(args.target_cache_root.resolve()),
                                "--n-perm", "5000"], out_root, contract)
    run_step("F_figure", [py, str(ROOT / "scripts/plot_topic5_rnn_motif_figures_v0_4.py"),
                          *root_args, "--stage", "early"], out_root, contract)
    run_step("H_common", [py, str(ROOT / "scripts/build_topic5_rnn_motif_common_observables_v0_4.py"),
                          *root_args, "--snn-readout", str(args.snn_readout.resolve())], out_root, contract)
    run_step("I_figure", [py, str(ROOT / "scripts/plot_topic5_rnn_motif_figures_v0_4.py"),
                          *root_args, "--stage", "final"], out_root, contract)
    tests = [
        "tests/test_topic5_rnn_motif_v0_4.py", "tests/test_topic5_spatial_latent_rnn.py",
        "tests/test_topic5_we_cache.py", "tests/test_topic5_we_graph_analysis.py",
        "tests/test_topic5_we_train.py", "tests/test_topic5_wiring_economy_rnn.py",
    ]
    test_log = out_root / "postprocess_logs" / "I_tests.log"
    run_step("I_tests", [py, "-m", "pytest", *tests, "-q"], out_root, contract)
    atomic(out_root / "POSTPROCESS_READY_FOR_VISUAL_QA.json", {
        "status": "READY_FOR_VISUAL_QA", "finished_utc": datetime.now(timezone.utc).isoformat(),
        "figures": [str(out_root / "figures" / name) for name in (
            "stage_interictal_scientific_readout.png", "stage_fields_scientific_readout.png",
            "stage_motif_scientific_readout.png",
            "stage_early_scientific_readout.png", "topic5_figure6_rnn_connectivity_motifs.png")],
        "test_log": str(test_log),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
