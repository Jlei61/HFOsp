#!/usr/bin/env python3
"""Run the control-repaired v0.3.7 chain in a fresh, immutable result root.

The first v0.3.7 round is retained as provenance and must never be resumed: its
transparent baseline and most H2a adapters were unfitted.  This supervisor
starts from a new root, searches controls and learned models together, stops
closed if either family fails the same INNER-only trainability contract, then
runs the formal H1/H2a/H2b and independent H3 chains.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
DEFAULT_BASE = Path("/data/hfosp_group_event_state_v0_3_7/repaired_v2")


def _atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _run(script: str, *args: str) -> None:
    subprocess.run([str(PYTHON), str(ROOT / "scripts" / script), *args], cwd=ROOT, check=True)


def _spawn(script: str, *args: str) -> subprocess.Popen:
    return subprocess.Popen([str(PYTHON), str(ROOT / "scripts" / script), *args], cwd=ROOT)


def _stage(status_path: Path, base: Path, name: str, **extra: object) -> None:
    _atomic(status_path, {
        "format": "group_event_state_v0_3_7_repaired_completion_v1",
        "status": name,
        "base_root": str(base),
        "old_round_resumed": False,
        "development_targets_read": False,
        "sealed_partition_opened": False,
        "updated_unix": time.time(),
        **extra,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument("--gpus", default="0,1")
    args = parser.parse_args()
    base = args.base_root.resolve()
    status = base / "supervisor/completion_status.json"
    old_marker = Path("/data/hfosp_group_event_state_v0_3_7/SUPERSEDED_UNFITTED_CONTROL_ARMS.md")
    if not old_marker.exists():
        raise RuntimeError("round-one invalidation marker is missing")
    if base == Path("/data/hfosp_group_event_state_v0_3_7"):
        raise RuntimeError("repaired execution must not reuse the superseded root")

    _stage(status, base, "RUNNING_INSTRUMENT_AND_OPTIMIZER_SEARCH")
    _run(
        "audit_group_event_state_v037_ctssm_instrument.py",
        "--device", "cuda:0", "--output", str(base / "instrument/ctssm_instrument_audit.json"),
    )
    _run(
        "supervise_group_event_state_v037_optimizer_search.py",
        "--workers-per-gpu", str(args.workers_per_gpu), "--gpus", args.gpus,
        "--out-root", str(base / "optimizer_search"),
    )
    queue_status = json.loads(
        (base / "optimizer_search/supervisor/queue_status.json").read_text(encoding="utf-8")
    )
    if queue_status.get("status") == "STOPPED_AT_CONTROL_GATE":
        _stage(
            status, base, "STOPPED_AT_TRAINABILITY_GATE",
            failed_families=["B_rate_or_B_mark"],
            baseline_summary=str(base / "optimizer_search/baseline_summary.json"),
        )
        return
    if queue_status.get("status") != "COMPLETE":
        raise RuntimeError(f"optimizer queue did not complete: {queue_status.get('status')}")
    _run(
        "finalize_group_event_state_v037_optimizer_search.py",
        "--root", str(base / "optimizer_search"),
    )
    summary_path = base / "optimizer_search/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    failed = [
        name for name, node in summary["models"].items()
        if node.get("selection_status") != "TRAINABLE_RECIPE_SELECTED"
    ]
    if summary.get("baseline", {}).get("baseline_selection_status") != "TRAINABLE_BASELINE_RECIPE_SELECTED":
        failed.append("transparent_baseline")
    if failed:
        _stage(
            status, base, "STOPPED_AT_TRAINABILITY_GATE",
            failed_families=failed, optimizer_summary=str(summary_path),
        )
        return

    _stage(status, base, "RUNNING_FORMAL_H1", optimizer_summary=str(summary_path))
    _run(
        "supervise_group_event_state_v037_optimized_h1.py",
        "--workers-per-gpu", str(args.workers_per_gpu), "--gpus", args.gpus,
        "--search-summary", str(summary_path),
        "--out-root", str(base / "h1_optimized"),
    )

    _stage(status, base, "RUNNING_FROZEN_H2_AND_H3_INSTRUMENTS")
    h2a = _spawn(
        "supervise_group_event_state_v037_optimized_h2a.py",
        "--workers-per-gpu", str(args.workers_per_gpu), "--gpus", args.gpus,
        "--h1-root", str(base / "h1_optimized"),
        "--out-root", str(base / "h2a_optimized"),
    )
    h2b = _spawn(
        "supervise_group_event_state_v037_optimized_h2b.py",
        "--workers", "2", "--h1-root", str(base / "h1_optimized"),
        "--out-root", str(base / "h2b_optimized"),
    )
    if h2a.wait() != 0 or h2b.wait() != 0:
        raise RuntimeError("formal H2a/H2b failed")

    _run(
        "supervise_group_event_state_v037_h2a_joint.py",
        # These frozen-decoder cells have the same audited memory footprint as
        # primary H2a.  Use the already-approved per-GPU concurrency instead
        # of leaving both devices at roughly 15% utilisation.
        "--workers", str(
            len(tuple(v for v in args.gpus.split(",") if v.strip()))
            * args.workers_per_gpu
        ),
        "--gpus", args.gpus,
        "--h1-root", str(base / "h1_optimized/event"),
        "--primary-h2a-root", str(base / "h2a_optimized/event"),
        "--out-root", str(base / "h2a_joint_sensitivity_optimized"),
    )

    # H3 is a separate Z_phys model; it never reuses S_obs updates as jumps.
    _stage(status, base, "RUNNING_INDEPENDENT_H3")
    _run(
        "audit_group_event_state_v037_h3_instrument.py",
        "--device", "cuda:0", "--out-root", str(base / "h3_instrument"),
    )
    _run(
        "supervise_group_event_state_v037_h3.py",
        "--workers-per-gpu", str(args.workers_per_gpu),
        "--out-root", str(base / "h3_independent_generative"),
        "--instrument-root", str(base / "h3_instrument"),
    )
    _run(
        "audit_and_upgrade_group_event_state_v037_h3_cards.py",
        "--root", str(base / "h3_independent_generative"),
    )
    _run(
        "audit_group_event_state_v037_h3_persistent_instrument.py",
        "--device", "cpu", "--out-root", str(base / "h3_persistent_instrument"),
    )
    _run(
        "supervise_group_event_state_v037_h3_persistent.py",
        "--workers-per-gpu", str(max(1, args.workers_per_gpu // 2)),
        "--gpus", args.gpus,
        "--source-root", str(base / "h3_independent_generative"),
        "--instrument-root", str(base / "h3_persistent_instrument"),
        "--out-root", str(base / "h3_persistent_feedback_v4"),
    )

    _stage(status, base, "RUNNING_FINAL_TESTS_AND_REPORTS")
    subprocess.run([
        str(PYTHON), "-m", "pytest", "-q",
        "tests/test_group_event_state_v037_control_parity.py",
        "tests/test_group_event_state_v037_ctssm.py",
        "tests/test_group_event_state_v037_decoder_audit.py",
        "tests/test_group_event_state_v037_h1_dual.py",
        "tests/test_group_event_state_v037_h1_train.py",
        "tests/test_group_event_state_v037_h2a.py",
        "tests/test_group_event_state_v037_h2b.py",
        "tests/test_group_event_state_v037_h3_generative.py",
        "tests/test_group_event_state_v037_strict_decoder_cache.py",
    ], cwd=ROOT, check=True)
    _run("finalize_group_event_state_v037.py", "--data-root", str(base))
    _stage(
        status, base, "COMPLETE",
        summary=str(base / "final_reports/integrated_summary_v2.json"),
        manifest=str(base / "final_reports/manifest_v3.json"),
    )


if __name__ == "__main__":
    main()
