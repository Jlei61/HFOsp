#!/usr/bin/env python3
"""Build and validate the final reproducibility manifest for HistoryRNN v0.1."""
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


SEEDS = (20260725, 20260726, 20260727)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, check=False, capture_output=True, text=True
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--result-root", type=Path,
        default=Path("results/topic5_history_rnn_early_ictal_field"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    result_root = (repo / args.result_root).resolve()
    formal = result_root / "g1_sequential_formal_v0_1"
    multi = _json(formal / "G1_MULTI_SEED_SUMMARY.json")
    g1_pass = multi["status"] == "G1_MULTI_SEED_PASS_OPEN_G2"

    g1_fold_rows = []
    for seed in SEEDS:
        seed_root = formal / f"seed_{seed}"
        for done_path in sorted(seed_root.glob("*/DONE.json")):
            order_path = done_path.parent / "ORDER_CONTROLS.json"
            checkpoint = done_path.parent / "checkpoint.pt"
            if not order_path.exists() or not checkpoint.exists():
                raise RuntimeError(f"incomplete G1 fold: {done_path.parent}")
            done = _json(done_path)
            order = _json(order_path)
            if bool(done.get("target_values_read", True)) or bool(
                order.get("target_values_read", True)
            ):
                raise RuntimeError(f"G1 target seal violation: {done_path.parent}")
            g1_fold_rows.append(
                {
                    "seed": seed,
                    "subject": done["heldout_subject"],
                    "done_sha256": _sha256(done_path),
                    "order_controls_sha256": _sha256(order_path),
                    "checkpoint_sha256": _sha256(checkpoint),
                    "event_checkpoint_sha256": done["event_checkpoint_sha256"],
                    "dataset_manifest_sha256": done["dataset_manifest_sha256"],
                }
            )
    if len(g1_fold_rows) != 102:
        raise RuntimeError(f"G1 fold denominator drift: {len(g1_fold_rows)}/102")

    g2_root = result_root / "g2_early_ictal_loso_v0_1"
    g2_done = sorted(g2_root.glob("*/DONE.json"))
    if g1_pass and len(g2_done) != 16:
        raise RuntimeError(f"G1 passed but G2 has {len(g2_done)}/16 folds")
    if not g1_pass and g2_done:
        raise RuntimeError("G2 artifacts exist despite failed G1")
    g2_rows = []
    for done_path in g2_done:
        done = _json(done_path)
        if not bool(done.get("target_values_read", False)):
            raise RuntimeError(f"G2 target unlock missing: {done_path}")
        g2_rows.append(
            {
                "subject": done["heldout_subject"],
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
            }
        )

    tracked_inputs = [
        "docs/superpowers/specs/2026-08-01-topic5-history-rnn-early-ictal-field-v0_1.md",
        "docs/superpowers/plans/2026-08-01-topic5-history-rnn-early-ictal-field-v0_1.md",
        "src/topic5_history_rnn.py",
        "src/topic5_history_data.py",
        "src/topic5_history_bridge.py",
        "scripts/audit_topic5_history_rnn_causal_prefix_v0_1.py",
        "scripts/run_topic5_history_rnn_gate1_sequential_fold_v0_1.py",
        "scripts/audit_topic5_history_rnn_gate1_order_controls_v0_1.py",
        "scripts/summarize_topic5_history_rnn_gate1_multiseed_v0_1.py",
        "scripts/run_topic5_history_rnn_early_ictal_fold_v0_1.py",
        "scripts/summarize_topic5_history_rnn_early_ictal_loso_v0_1.py",
        "scripts/plot_topic5_history_rnn_early_ictal_field_v0_1.py",
        "scripts/closeout_topic5_history_rnn_early_ictal_field_v0_1.py",
        "tests/test_topic5_history_rnn.py",
    ]
    input_hashes = {}
    for relative in tracked_inputs:
        path = repo / relative
        if not path.exists():
            raise RuntimeError(f"required source missing: {relative}")
        input_hashes[relative] = _sha256(path)

    summary_paths = [
        result_root / "g0_causal_prefix" / "G0_SUMMARY.json",
        result_root
        / "g1_sequential_development_selection_v0_1"
        / "DEVELOPMENT_SELECTION.json",
        formal / "G1_MULTI_SEED_SUMMARY.json",
        result_root / "FINAL_CLOSEOUT.json",
    ]
    if g1_pass:
        summary_paths.append(g2_root / "G2_G3_SUMMARY.json")
    summary_hashes = {
        str(path.relative_to(repo)): _sha256(path) for path in summary_paths
    }
    figure_paths = [
        result_root / "figures" / "topic5_history_rnn_early_ictal_field_v0_1.png",
        result_root / "figures" / "topic5_history_rnn_early_ictal_field_v0_1.pdf",
        result_root / "figures" / "topic5_history_rnn_early_ictal_field_v0_1.json",
        result_root / "figures" / "README.md",
    ]
    if not all(path.exists() for path in figure_paths):
        raise RuntimeError("final figure package is incomplete")
    figure_hashes = {
        str(path.relative_to(repo)): _sha256(path) for path in figure_paths
    }
    report_paths = [
        repo
        / "docs/archive/topic5/"
        "history_rnn_early_ictal_field_v0_1_closeout_2026-08-02.md",
        repo
        / "docs/paper-draft/"
        "figure6_history_rnn_early_ictal_field_bounded_negative.md",
    ]
    if not all(path.exists() for path in report_paths):
        raise RuntimeError("final report package is incomplete")
    payload = {
        "status": "REPRODUCIBILITY_AUDIT_PASS",
        "contract": "topic5_history_rnn_early_ictal_field_v0_1",
        "g1_status": multi["status"],
        "target_values_read": bool(g1_pass),
        "counts": {
            "g1_seeds": len(SEEDS),
            "g1_folds": len(g1_fold_rows),
            "g1_strict_order_controls": len(g1_fold_rows),
            "g2_folds": len(g2_rows),
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
        "source_and_contract_sha256": input_hashes,
        "summary_sha256": summary_hashes,
        "figure_sha256": figure_hashes,
        "report_sha256": {
            str(path.relative_to(repo)): _sha256(path) for path in report_paths
        },
        "g1_fold_artifacts": g1_fold_rows,
        "g2_fold_artifacts": g2_rows,
    }
    output = result_root / "REPRODUCIBILITY_MANIFEST.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
