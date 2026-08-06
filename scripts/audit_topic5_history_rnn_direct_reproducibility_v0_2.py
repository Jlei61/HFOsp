#!/usr/bin/env python3
"""Build the final reproducibility manifest for HistoryRNN direct transfer v0.2."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"required reproducibility artifact missing: {path}")
    return json.loads(path.read_text())


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, check=False, capture_output=True, text=True
    )
    return result.stdout.strip()


def _hashes(repo: Path, paths: list[Path]) -> dict[str, str]:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"reproducibility inputs missing: {missing}")
    return {
        str(path.resolve().relative_to(repo)): _sha256(path.resolve())
        for path in paths
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("results/topic5_history_rnn_direct_early_ictal_transfer_v0_2"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "docs/archive/topic5/"
            "history_rnn_direct_early_ictal_transfer_v0_2_result_2026-08-02.md"
        ),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    root = (repo / args.result_root).resolve()

    direct = _read(root / "DIRECT_TRANSFER_SUMMARY.json")
    acceptance = _read(root / "FINAL_ACCEPTANCE.json")
    refit = _read(root / "g1_refit_c30/REFIT_SUMMARY.json")
    budget = _read(
        root
        / "training_budget_comparison_c10_to_c30"
        / "DIRECT_TRAINING_BUDGET_COMPARISON.json"
    )
    if int(direct.get("n_completed_folds", 0)) != 16:
        raise RuntimeError("direct fold denominator drifted")
    if int(direct.get("history_checkpoint_cycles", 0)) != 30:
        raise RuntimeError("canonical direct result is not c30")
    if not bool(direct.get("target_values_read", False)):
        raise RuntimeError("direct target access was not recorded")
    if (
        refit.get("status") != "COMPLETE"
        or int(refit.get("n_completed_folds", 0)) != 16
        or int(refit.get("n_failed_folds", 1)) != 0
        or bool(refit.get("target_values_read", True))
        or int(refit.get("history_cycles", 0)) != 30
    ):
        raise RuntimeError("c30 target-blind refit provenance is invalid")
    if int(budget.get("short_history_cycles", 0)) != 10 or int(
        budget.get("long_history_cycles", 0)
    ) != 30:
        raise RuntimeError("c10-to-c30 comparison provenance drifted")
    if acceptance.get("target_reuse_tier") != (
        "INTERNAL_VALIDATION_NOT_INDEPENDENT_CONFIRMATION"
    ):
        raise RuntimeError("target reuse tier drifted")

    refit_rows = []
    refit_root = root / "g1_refit_c30/seed_20260725"
    for done_path in sorted(refit_root.glob("*/DONE.json")):
        done = _read(done_path)
        if bool(done.get("target_values_read", True)):
            raise RuntimeError(f"target seal violation: {done_path.parent}")
        if int(done.get("config", {}).get("history_cycles", 0)) != 30:
            raise RuntimeError(f"history-cycle drift: {done_path.parent}")
        checkpoint = done_path.parent / "checkpoint.pt"
        training_log = done_path.parent / "training_log.csv"
        refit_rows.append({
            "subject": str(done["heldout_subject"]),
            "done_sha256": _sha256(done_path),
            "checkpoint_sha256": _sha256(checkpoint),
            "training_log_sha256": _sha256(training_log),
            "event_checkpoint_sha256": str(done["event_checkpoint_sha256"]),
            "dataset_manifest_sha256": str(done["dataset_manifest_sha256"]),
        })
    if len(refit_rows) != 16:
        raise RuntimeError(f"c30 refit artifacts incomplete: {len(refit_rows)}/16")

    direct_rows = []
    for done_path in sorted(root.glob("epilepsiae_*/DONE.json")):
        done = _read(done_path)
        if not bool(done.get("target_values_read", False)):
            raise RuntimeError(f"direct target unlock missing: {done_path.parent}")
        provenance = done.get("history_checkpoint_provenance") or {}
        if bool(provenance.get("target_values_read", True)) or int(
            provenance.get("history_cycles", 0)
        ) != 30:
            raise RuntimeError(f"direct checkpoint provenance drift: {done_path.parent}")
        direct_rows.append({
            "subject": str(done["heldout_subject"]),
            "done_sha256": _sha256(done_path),
            "predictions_sha256": _sha256(
                done_path.parent / "heldout_contact_predictions.csv"
            ),
            "seizure_metrics_sha256": _sha256(
                done_path.parent / "heldout_seizure_metrics.csv"
            ),
            "wrong_pairing_sha256": _sha256(
                done_path.parent / "heldout_wrong_state_pairing.csv"
            ),
            "residual_sha256": _sha256(
                done_path.parent / "heldout_seizure_specific_residual.csv"
            ),
        })
    if len(direct_rows) != 16:
        raise RuntimeError(f"canonical direct artifacts incomplete: {len(direct_rows)}/16")

    source_paths = [
        repo / "docs/superpowers/specs/2026-08-02-topic5-history-rnn-direct-early-ictal-transfer-v0_2.md",
        repo / "docs/superpowers/plans/2026-08-02-topic5-history-rnn-direct-early-ictal-transfer-v0_2.md",
        repo / "config/topic5_history_rnn_direct_early_ictal_transfer_v0_2.json",
        repo / "src/topic5_history_rnn.py",
        repo / "src/topic5_history_data.py",
        repo / "src/topic5_history_bridge.py",
        repo / "scripts/run_topic5_history_rnn_gate1_sequential_fold_v0_1.py",
        repo / "scripts/run_topic5_history_rnn_early_ictal_fold_v0_1.py",
        repo / "scripts/summarize_topic5_history_rnn_direct_early_ictal_transfer_v0_2.py",
        repo / "scripts/plot_topic5_history_rnn_direct_early_ictal_transfer_v0_2.py",
        repo / "scripts/compare_topic5_history_rnn_direct_training_budget_v0_2.py",
        repo / "scripts/closeout_topic5_history_rnn_direct_v0_2.py",
        repo / "scripts/audit_topic5_history_rnn_direct_reproducibility_v0_2.py",
        repo / "scripts/run_topic5_history_rnn_direct_checkpoint_refit_v0_2.sh",
        repo / "scripts/run_topic5_history_rnn_direct_early_ictal_transfer_v0_2.sh",
        repo / "tests/test_topic5_history_rnn.py",
    ]
    summary_paths = [
        root / "g1_refit_c30/REFIT_SUMMARY.json",
        root / "DIRECT_TRANSFER_SUMMARY.json",
        root / "direct_transfer_patient_metrics.csv",
        root / "direct_transfer_channel_null_patient_metrics.csv",
        root / "state_seizure_pairing_metrics.csv",
        root / "seizure_specific_residual_metrics.csv",
        root / "target_headroom_metrics.csv",
        root
        / "training_budget_comparison_c10_to_c30"
        / "DIRECT_TRAINING_BUDGET_COMPARISON.json",
        root / "FINAL_ACCEPTANCE.json",
    ]
    figure_paths = [
        root / "figures/topic5_history_to_early_ictal_direct_transfer_v0_2.png",
        root / "figures/topic5_history_to_early_ictal_direct_transfer_v0_2.pdf",
        root / "figures/README.md",
    ]
    report_path = (repo / args.report).resolve()
    payload = {
        "status": "REPRODUCIBILITY_AUDIT_PASS",
        "contract": "topic5_history_rnn_direct_early_ictal_transfer_v0_2",
        "direct_status": direct.get("status"),
        "acceptance_status": acceptance.get("status"),
        "training_budget_robustness": budget.get("status"),
        "target_values_read_only_after_refit": True,
        "counts": {
            "c30_refit_folds": len(refit_rows),
            "direct_folds": len(direct_rows),
            "primary_patients": 15,
            "supportive_patients": 16,
            "channel_shuffle_draws_per_patient": int(
                direct["all_contact_channel_shuffle"]["n_draws_per_patient"]
            ),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "git": {
            "branch": _git(repo, "branch", "--show-current"),
            "head": _git(repo, "rev-parse", "HEAD"),
            "status_porcelain": _git(repo, "status", "--short"),
        },
        "source_and_contract_sha256": _hashes(repo, source_paths),
        "summary_sha256": _hashes(repo, summary_paths),
        "figure_sha256": _hashes(repo, figure_paths),
        "report_sha256": _hashes(repo, [report_path]),
        "c30_refit_artifacts": refit_rows,
        "direct_artifacts": direct_rows,
    }
    output = root / "REPRODUCIBILITY_MANIFEST.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
