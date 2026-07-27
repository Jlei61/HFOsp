#!/usr/bin/env python3
"""Persistent final-artifact watcher for Topic-5 v2.2.

This watcher is deliberately downstream of the gate-aware training supervisor.
It never launches a scientific stage.  Once the interictal pipeline is
finalized, it runs the post-hoc A/B read-back, renders Figure 6, executes the
frozen regression tests, and writes a checksum manifest.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/topic5_symmetric_axis_propagation_state_v2_2"
FORMAL = BASE / "formal"
ANALYSIS = FORMAL / "analysis"
STATE = FORMAL / "FINAL_ARTIFACT_WATCHER_STATE.json"
LOGS = FORMAL / "launcher_logs"
FIGURE_ROOT = (
    ROOT
    / "results/paper-ready-figure"
    / "fig6_symmetric_axis_propagation_state_v2_2"
)
POLL_SECONDS = 300


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def update(stage: str, **extra: Any) -> None:
    payload = {
        "status": "RUNNING",
        "stage": stage,
        "pid": os.getpid(),
        "updated_unix": time.time(),
        "target_values_read": False,
        **extra,
    }
    atomic_json(STATE, payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def run_logged(command: list[str], log_name: str) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    path = LOGS / log_name
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
            + " ".join(command)
            + "\n"
        )
        handle.flush()
        subprocess.run(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
        )


def require_seal() -> None:
    target = read_json(BASE / "target_audit/TARGET_METADATA_GATE.json")
    seal = read_json(BASE / "target_audit/TARGET_VALUES_SEALED.json")
    if (
        target.get("energy_values_read")
        or target.get("recruitment_values_read")
        or seal.get("energy_values_read")
        or seal.get("recruitment_values_read")
    ):
        raise RuntimeError("target value seal was violated")


def wait_for_pipeline() -> dict[str, Any]:
    while True:
        require_seal()
        pipeline = read_json(FORMAL / "PIPELINE_SUPERVISOR_STATE.json")
        if pipeline.get("status") == "FAILED":
            raise RuntimeError(
                f"training supervisor failed: {pipeline.get('error')}"
            )
        if pipeline.get("status") == "COMPLETE":
            summary = read_json(ANALYSIS / "INTERICTAL_CLAIM_SUMMARY.json")
            if summary.get("early_ictal_values_unlocked"):
                transfer = BASE / "early_ictal_transfer/TRANSFER_CLAIM_SUMMARY.json"
                if not transfer.is_file():
                    update("wait_early_ictal_transfer")
                    time.sleep(POLL_SECONDS)
                    continue
            return summary
        update(
            "wait_interictal_pipeline",
            upstream_stage=pipeline.get("stage"),
            upstream_status=pipeline.get("status"),
        )
        time.sleep(POLL_SECONDS)


def build_manifest(summary: dict[str, Any], test_log: Path) -> dict[str, Any]:
    required = [
        BASE / "provenance/upstream_manifest.json",
        BASE / "input_audit/INPUT_AUDIT_GATE.json",
        BASE / "target_audit/TARGET_METADATA_GATE.json",
        BASE / "target_audit/TARGET_VALUES_SEALED.json",
        BASE / "model_audit/MATHEMATICAL_MODEL_GATE.json",
        BASE / "development/DEVELOPMENT_LOCK.json",
        FORMAL / "PHYSICAL_AXIS_FORMAL_LOCK.json",
        FORMAL / "ALL_SUBJECT_SEQUENCE_LOCK.json",
        FORMAL / "TRAINER_EPOCH_AUDIT.json",
        ANALYSIS / "CLAIM1_STATUS.json",
        ANALYSIS / "claim1_sequence_predictability.csv",
        ANALYSIS / "CLAIM2_STATUS.json",
        ANALYSIS / "ALL_SUBJECT_SEQUENCE_STATUS.json",
        ANALYSIS / "AB_AXIS_READBACK_STATUS.json",
        ANALYSIS / "INTERICTAL_CLAIM_SUMMARY.json",
        FIGURE_ROOT / "fig6_symmetric_axis_propagation_state_v2_2_summary.json",
        FIGURE_ROOT
        / "figures/fig6_symmetric_axis_propagation_state_v2_2.png",
        FIGURE_ROOT
        / "figures/fig6_symmetric_axis_propagation_state_v2_2.pdf",
        FIGURE_ROOT / "figures/README.md",
        test_log,
    ]
    if (ANALYSIS / "CLAIM3_STATUS.json").is_file():
        required.extend(
            [
                ANALYSIS / "CLAIM3_STATUS.json",
                ANALYSIS / "claim3_random_axis_specificity.csv",
            ]
        )
    if (ANALYSIS / "CLAIM4_STATUS.json").is_file():
        required.extend(
            [
                ANALYSIS / "CLAIM4_STATUS.json",
                ANALYSIS / "claim4_shared_scaffold.csv",
            ]
        )
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"final artifact manifest is missing: {missing}")
    checksums = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in required
    }
    target = read_json(BASE / "target_audit/TARGET_METADATA_GATE.json")
    return {
        "contract": "topic5_symmetric_axis_propagation_state_rnn",
        "version": "2.2",
        "status": "COMPLETE",
        "interictal_pass": summary["interictal_pass"],
        "early_ictal_values_unlocked": summary[
            "early_ictal_values_unlocked"
        ],
        "source_metadata_ready": summary["source_metadata_ready"],
        "target_values_read": bool(
            target.get("energy_values_read")
            or target.get("recruitment_values_read")
        ),
        "required_artifacts": checksums,
        "figure_summary_sha256": checksums[
            "results/paper-ready-figure/"
            "fig6_symmetric_axis_propagation_state_v2_2/"
            "fig6_symmetric_axis_propagation_state_v2_2_summary.json"
        ],
        "tests": {
            "command": (
                "pytest -q tests/test_topic5_*v2_2.py "
                "tests/test_plot_fig6_symmetric_axis_propagation_state_v2_2.py"
            ),
            "status": "PASS",
            "log": str(test_log.relative_to(ROOT)),
            "log_sha256": sha256(test_log),
        },
        "finished_unix": time.time(),
    }


def main() -> None:
    summary = wait_for_pipeline()
    update("analyze_claim1_node_control")
    run_logged(
        [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "cuda_env",
            "python",
            "scripts/run_topic5_formal_claim1_node_control_v2_2.py",
        ],
        "claim1_node_control_finalize.log",
    )
    update("run_posthoc_ab_readback")
    run_logged(
        [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "cuda_env",
            "python",
            "scripts/analyze_topic5_ab_axis_readback_v2_2.py",
        ],
        "ab_axis_readback.log",
    )
    update("render_figure6")
    run_logged(
        [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "cuda_env",
            "python",
            "scripts/paper_figures/"
            "plot_fig6_symmetric_axis_propagation_state_v2_2.py",
        ],
        "figure6_render.log",
    )
    update("run_delivery_tests")
    test_log = LOGS / "final_delivery_tests.log"
    run_logged(
        [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "cuda_env",
            "pytest",
            "-q",
            "tests/test_topic5_absorbing_rollout_v2_2.py",
            "tests/test_topic5_axis_readback_v2_2.py",
            "tests/test_topic5_formal_node_control_v2_2.py",
            "tests/test_topic5_interictal_gate_v2_2.py",
            "tests/test_topic5_propagation_state_recurrence_v2_2.py",
            "tests/test_topic5_sequence_sensitivity_v2_2.py",
            "tests/test_topic5_symmetric_axis_aggregation_v2_2.py",
            "tests/test_topic5_symmetric_axis_claim4_v2_2.py",
            "tests/test_topic5_symmetric_axis_leakage_v2_2.py",
            "tests/test_topic5_symmetric_axis_operator_v2_2.py",
            "tests/test_topic5_symmetric_axis_random_controls_v2_2.py",
            "tests/test_plot_fig6_symmetric_axis_propagation_state_v2_2.py",
        ],
        test_log.name,
    )
    update("write_reproducibility_manifest")
    manifest = build_manifest(summary, test_log)
    atomic_json(FORMAL / "FINAL_REPRODUCIBILITY_MANIFEST.json", manifest)
    atomic_json(
        STATE,
        {
            "status": "COMPLETE",
            "stage": "final_artifacts_complete",
            "figure": str(
                (
                    FIGURE_ROOT
                    / "figures/fig6_symmetric_axis_propagation_state_v2_2.png"
                ).relative_to(ROOT)
            ),
            "manifest": str(
                (FORMAL / "FINAL_REPRODUCIBILITY_MANIFEST.json").relative_to(
                    ROOT
                )
            ),
            "target_values_read": False,
            "finished_unix": time.time(),
        },
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        atomic_json(
            STATE,
            {
                "status": "FAILED",
                "stage": "final_artifact_watcher_exception",
                "error": repr(exc),
                "target_values_read": False,
                "finished_unix": time.time(),
            },
        )
        raise
