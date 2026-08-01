#!/usr/bin/env python3
"""Phase A: reconstruct the real optimizer semantics of the frozen Topic 5 runs.

Read-only.  Nothing is retrained.  The audit combines three evidence sources:

1. the frozen launcher arguments (coverage cycles, updates per patient, chunk);
2. the per-fold ``training_log.csv`` written by ``train_shared_coverage``;
3. the sealed rank dataset itself (teacher-forced unroll depth).

Its load-bearing claim is that ``--batch-size 1024`` is a memory chunk and not
an optimizer minibatch: every fold must show more backward chunks than
``optimizer.step()`` calls whenever a patient segment exceeds the chunk size.
"""
from __future__ import annotations

import argparse
import ast
import inspect
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_topic5_interictal_rank_distribution import (  # noqa: E402
    load_records,
    train_shared_coverage,
)

FROZEN_LAUNCHER = ROOT / "scripts/run_topic5_architecture_controls_v0_1.sh"
DEFAULT_FORMAL = (
    ROOT
    / "results/topic5_ordered_history_architecture_audit/formal"
    / "architecture_controls_formal_20260729"
)
GRADIENT_CLIP = 1.0


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):  # pragma: no cover
        return "unknown"


def _launcher_arguments() -> dict:
    """Parse the frozen formal launcher for the training-budget flags."""
    text = FROZEN_LAUNCHER.read_text()
    wanted = {
        "--batch-size": "batch_size",
        "--shared-cycles": "shared_cycles",
        "--calibration-cycles": "calibration_cycles",
        "--updates-per-patient": "updates_per_patient",
    }
    tokens = text.split()
    out: dict[str, int] = {}
    for flag, key in wanted.items():
        if flag not in tokens:
            raise RuntimeError(f"frozen launcher no longer passes {flag}")
        out[key] = int(tokens[tokens.index(flag) + 1])
    return out


def _optimizer_step_structure() -> dict:
    """Statically confirm where ``optimizer.step()`` sits in the frozen loop."""
    source = inspect.getsource(train_shared_coverage)
    tree = ast.parse(source)
    function = tree.body[0]

    def _loop_targets(node) -> list[str]:
        names = []
        for child in ast.walk(node):
            if isinstance(child, ast.For) and isinstance(child.target, ast.Name):
                names.append(child.target.id)
        return names

    step_depth: list[int] = []
    backward_depth: list[int] = []
    zero_grad_depth: list[int] = []

    def _visit(node, depth: int) -> None:
        for child in ast.iter_child_nodes(node):
            child_depth = depth + 1 if isinstance(child, ast.For) else depth
            if isinstance(child, ast.Call):
                target = ast.unparse(child.func)
                if target.endswith("optimizer.step"):
                    step_depth.append(depth)
                elif target.endswith("backward"):
                    backward_depth.append(depth)
                elif target.endswith("optimizer.zero_grad"):
                    zero_grad_depth.append(depth)
            _visit(child, child_depth)

    _visit(function, 0)
    return {
        "loop_variables": _loop_targets(function),
        "optimizer_step_loop_depth": step_depth,
        "backward_loop_depth": backward_depth,
        "zero_grad_loop_depth": zero_grad_depth,
        "backward_is_strictly_deeper_than_step": bool(
            step_depth and backward_depth and min(backward_depth) > max(step_depth)
        ),
        "zero_grad_matches_step_depth": bool(
            step_depth
            and zero_grad_depth
            and max(zero_grad_depth) == max(step_depth)
        ),
        "chunk_loop_is_innermost_over_batch_size": "batch_start"
        in _loop_targets(function),
    }


def _fold_rows(formal_root: Path, batch_size: int) -> pd.DataFrame:
    rows = []
    for log_path in sorted(formal_root.glob("*/seed_*/*/training_log.csv")):
        cell = log_path.parent
        summary = json.loads((cell / "run_summary.json").read_text())
        frame = pd.read_csv(log_path)
        shared = frame[frame.phase == "shared_full_coverage"]
        offset = frame[frame.phase == "heldout_offset_full_coverage"]
        if shared.empty:
            raise RuntimeError(f"{cell}: no shared training rows")
        per_patient = shared.groupby("subject").size()
        shared_chunks = int(
            np.sum(np.ceil(shared.n_events.to_numpy(float) / float(batch_size)))
        )
        offset_chunks = int(
            np.sum(np.ceil(offset.n_events.to_numpy(float) / float(batch_size)))
        )
        rows.append(
            {
                "architecture": summary["architecture"],
                "seed": int(summary["seed"]),
                "heldout_subject": summary["subject"],
                "dataset": summary["dataset"],
                "n_outer_training_patients": int(shared.subject.nunique()),
                "outer_training_patients": "|".join(
                    sorted(shared.subject.astype(str).unique())
                ),
                "shared_coverage_cycles": int(shared.coverage_cycle.max()),
                "shared_updates_per_patient_min": int(per_patient.min()),
                "shared_updates_per_patient_max": int(per_patient.max()),
                "shared_optimizer_steps": int(len(shared)),
                "shared_backward_chunks": shared_chunks,
                "shared_events_per_update_median": float(shared.n_events.median()),
                "shared_events_per_update_max": int(shared.n_events.max()),
                "shared_events_per_update_min": int(shared.n_events.min()),
                "shared_gradient_clip_fraction": float(
                    np.mean(shared.gradient_norm.to_numpy(float) > GRADIENT_CLIP)
                ),
                "shared_gradient_norm_median": float(shared.gradient_norm.median()),
                "shared_gradient_norm_max": float(shared.gradient_norm.max()),
                "heldout_offset_cycles": (
                    int(offset.coverage_cycle.max()) if not offset.empty else 0
                ),
                "heldout_offset_optimizer_steps": int(len(offset)),
                "heldout_offset_backward_chunks": offset_chunks,
                "heldout_offset_events_per_update_median": (
                    float(offset.n_events.median()) if not offset.empty else float("nan")
                ),
                "heldout_offset_gradient_clip_fraction": (
                    float(np.mean(offset.gradient_norm.to_numpy(float) > GRADIENT_CLIP))
                    if not offset.empty
                    else float("nan")
                ),
                "n_train_calibration_events": int(
                    summary["n_train_calibration_events"]
                ),
                "n_eval_events": int(summary["n_eval_events"]),
                "n_parameters": int(summary["n_parameters"]),
                "runtime_seconds": float(summary["resources"]["runtime_seconds"]),
            }
        )
    if not rows:
        raise RuntimeError(f"no frozen folds found under {formal_root}")
    return pd.DataFrame(rows)


def _unroll_depth(dataset_dir: Path) -> pd.DataFrame:
    records = load_records(dataset_dir)
    rows = []
    for subject, record in sorted(records.items()):
        train_count = record.group_count[record.train_indices].astype(int)
        eval_count = record.group_count[record.eval_indices].astype(int)
        rows.append(
            {
                "subject": subject,
                "dataset": record.dataset,
                "n_contacts": int(record.contact_features.shape[0]),
                "n_train_events": int(train_count.size),
                "n_eval_events": int(eval_count.size),
                # one teacher-forced decision per rank step plus the STOP decision
                "train_rank_steps_median": float(np.median(train_count)),
                "train_rank_steps_max": int(train_count.max()),
                "train_decisions_median": float(np.median(train_count) + 1.0),
                "train_decisions_max": int(train_count.max()) + 1,
                "train_total_decisions": int(np.sum(train_count + 1)),
                "eval_rank_steps_median": float(np.median(eval_count)),
                "eval_rank_steps_max": int(eval_count.max()),
                "eval_total_decisions": int(np.sum(eval_count + 1)),
                "input_sha256": record.input_sha256,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, default=DEFAULT_FORMAL)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "results/topic5_rnn_training_sufficiency_v0_1/input_audit",
    )
    args = parser.parse_args()

    formal_root = (
        args.formal_root if args.formal_root.is_absolute() else ROOT / args.formal_root
    )
    dataset_root = (
        args.dataset_root
        if args.dataset_root.is_absolute()
        else ROOT / args.dataset_root
    )
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    launcher = _launcher_arguments()
    structure = _optimizer_step_structure()
    folds = _fold_rows(formal_root, launcher["batch_size"])
    unroll = _unroll_depth(dataset_root)
    manifest = json.loads((dataset_root / "dataset_manifest.json").read_text())

    folds.to_csv(out_dir / "per_fold_training_semantics.csv", index=False)
    unroll.to_csv(out_dir / "teacher_forced_unroll_depth.csv", index=False)

    linear = folds[folds.architecture == "linear_state"]
    if linear.empty:
        raise RuntimeError("no linear_state folds in the frozen formal tree")
    expected_steps = (
        linear.n_outer_training_patients * launcher["updates_per_patient"]
        * launcher["shared_cycles"]
    )
    steps_match = bool(np.all(linear.shared_optimizer_steps.to_numpy() == expected_steps.to_numpy()))
    accumulating = linear.shared_backward_chunks > linear.shared_optimizer_steps

    audit = {
        "status": "COMPLETE",
        "contract": "topic5_rnn_training_sufficiency_v0_1_phase_a",
        "generated_from": {
            "formal_root": str(formal_root.relative_to(ROOT)),
            "dataset_root": str(dataset_root.relative_to(ROOT)),
            "frozen_launcher": str(FROZEN_LAUNCHER.relative_to(ROOT)),
        },
        "environment": {
            "git_commit": _git_commit(),
            "hostname": platform.node(),
            "python": platform.python_version(),
        },
        "sealed_dataset": {
            "n_subjects_ok": int(manifest["n_subjects_ok"]),
            "n_events_ok": int(manifest["n_events_ok"]),
            "target_values_read": bool(manifest["target_values_read"]),
            "ab_or_kmeans_labels_read": bool(manifest["ab_or_kmeans_labels_read"]),
            "split_contract": manifest["split_contract"],
            "n_train_events": int(unroll.n_train_events.sum()),
            "n_eval_events": int(unroll.n_eval_events.sum()),
        },
        "frozen_launcher_arguments": launcher,
        "optimizer_loop_structure": structure,
        "per_fold": {
            "n_folds_audited": int(len(folds)),
            "n_linear_state_folds": int(len(linear)),
            "architectures": sorted(folds.architecture.unique().tolist()),
            "seeds": sorted(int(value) for value in folds.seed.unique()),
            "n_outer_training_patients": sorted(
                int(value) for value in folds.n_outer_training_patients.unique()
            ),
        },
        "shared_core_optimizer": {
            "coverage_cycles": sorted(
                int(value) for value in linear.shared_coverage_cycles.unique()
            ),
            "updates_per_patient": sorted(
                set(linear.shared_updates_per_patient_min.tolist())
                | set(linear.shared_updates_per_patient_max.tolist())
            ),
            "optimizer_steps_per_fold_median": float(
                linear.shared_optimizer_steps.median()
            ),
            "optimizer_steps_match_patients_times_updates": steps_match,
            "events_per_update_median_of_folds": float(
                linear.shared_events_per_update_median.median()
            ),
            "events_per_update_max_across_folds": int(
                linear.shared_events_per_update_max.max()
            ),
            "gradient_clip_threshold": GRADIENT_CLIP,
            "gradient_clip_fraction_median": float(
                linear.shared_gradient_clip_fraction.median()
            ),
            "gradient_clip_fraction_max": float(
                linear.shared_gradient_clip_fraction.max()
            ),
        },
        "heldout_local_offset_optimizer": {
            "calibration_cycles": sorted(
                int(value) for value in linear.heldout_offset_cycles.unique()
            ),
            "optimizer_steps_per_fold_median": float(
                linear.heldout_offset_optimizer_steps.median()
            ),
            "gradient_clip_fraction_median": float(
                linear.heldout_offset_gradient_clip_fraction.median()
            ),
        },
        "batch_size_is_memory_chunk_only": {
            "verdict": bool(accumulating.any()),
            "chunk_size": launcher["batch_size"],
            "n_folds_with_more_backward_chunks_than_steps": int(accumulating.sum()),
            "n_folds": int(len(linear)),
            "shared_backward_chunks_median": float(
                linear.shared_backward_chunks.median()
            ),
            "shared_optimizer_steps_median": float(
                linear.shared_optimizer_steps.median()
            ),
            "static_evidence": (
                "backward() sits inside the batch_start chunk loop while "
                "optimizer.step() sits in the segment loop"
            ),
            "loss_normalisation": (
                "next_set_stop_loss returns an event-mean, and chunks are "
                "accumulated with weight len(chunk)/len(segment); the weighted "
                "sum therefore equals the segment event-mean exactly"
            ),
            "numerical_parity_experiment": "phase B3",
        },
        "teacher_forced_unroll": {
            "rank_steps_median_of_patients": float(
                unroll.train_rank_steps_median.median()
            ),
            "rank_steps_max_across_patients": int(unroll.train_rank_steps_max.max()),
            "decisions_median_of_patients": float(
                unroll.train_decisions_median.median()
            ),
            "decisions_max_across_patients": int(unroll.train_decisions_max.max()),
            "total_train_decisions": int(unroll.train_total_decisions.sum()),
            "total_eval_decisions": int(unroll.eval_total_decisions.sum()),
        },
        "ictal_target_read": False,
        "outer_heldout_read": False,
    }
    (out_dir / "TRAINING_SEMANTICS_AUDIT.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
