#!/usr/bin/env python3
"""Select the smallest Stage-A hidden size within one SE of inner validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def select_smallest_one_se(summary: pd.DataFrame) -> tuple[int, int, float]:
    """Return selected size, best-mean size, and the one-SE threshold."""
    required = {
        "hidden_size",
        "mean_inner_validation_loss",
        "se_inner_validation_loss",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"missing one-SE columns: {missing}")
    best_index = summary.mean_inner_validation_loss.idxmin()
    best = summary.loc[best_index]
    if not np.isfinite(best.se_inner_validation_loss):
        raise ValueError("one-SE selection needs finite uncertainty")
    threshold = float(
        best.mean_inner_validation_loss + best.se_inner_validation_loss
    )
    eligible = summary[
        summary.mean_inner_validation_loss <= threshold + np.finfo(float).eps
    ]
    if eligible.empty:
        raise AssertionError("best hidden size must be one-SE eligible")
    return int(eligible.hidden_size.min()), int(best.hidden_size), threshold


def _load_inner_cell(run_dir: Path) -> dict:
    manifest_path = run_dir / "run_manifest.json"
    epoch_path = run_dir / "epoch_log.csv"
    done_path = run_dir / "DONE.json"
    if not (manifest_path.exists() and epoch_path.exists() and done_path.exists()):
        raise FileNotFoundError(f"incomplete Stage-A cell: {run_dir}")
    manifest = json.loads(manifest_path.read_text())
    if bool(manifest.get("ictal_target_opened", True)):
        raise RuntimeError(f"ictal-target leakage flag in {run_dir}")
    epoch = pd.read_csv(epoch_path)
    inner = epoch[epoch.phase.astype(str) == "shared_initialization"]
    if inner.empty:
        raise RuntimeError(f"missing shared inner-validation log: {run_dir}")
    value = float(inner.selection_loss.min())
    if not np.isfinite(value):
        raise RuntimeError(f"non-finite inner-validation loss: {run_dir}")
    return {
        "run_dir": str(run_dir),
        "subject": str(manifest["heldout_subject"]),
        "seed": int(manifest["seed"]),
        "hidden_size": int(manifest["model_kwargs"]["hidden_size"]),
        "best_inner_validation_loss": value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = [_load_inner_cell(path.resolve()) for path in args.runs]
    cell = pd.DataFrame(rows).sort_values(["hidden_size", "subject", "seed"])
    hidden_sizes = sorted(cell.hidden_size.unique().astype(int).tolist())
    if len(hidden_sizes) < 2:
        raise RuntimeError("one-SE selection needs at least two hidden sizes")

    keys_by_size = {
        int(hidden): set(
            map(tuple, group[["subject", "seed"]].itertuples(index=False, name=None))
        )
        for hidden, group in cell.groupby("hidden_size")
    }
    reference = keys_by_size[hidden_sizes[0]]
    if any(keys != reference for keys in keys_by_size.values()):
        raise RuntimeError("hidden sizes do not have matched subject-by-seed cells")

    rows = []
    for hidden, group in cell.groupby("hidden_size"):
        values = group.best_inner_validation_loss.to_numpy(float)
        rows.append(
            {
                "hidden_size": int(hidden),
                "mean_inner_validation_loss": float(np.mean(values)),
                "se_inner_validation_loss": float(
                    np.std(values, ddof=1) / np.sqrt(values.size)
                )
                if values.size >= 2
                else np.nan,
                "n_cells": int(values.size),
            }
        )
    summary = pd.DataFrame(rows).sort_values("hidden_size")
    try:
        selected, best_hidden, threshold = select_smallest_one_se(summary)
    except ValueError as exc:
        raise RuntimeError(
            "one-SE selection needs at least two matched cells per size"
        ) from exc

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cell.to_csv(args.out_dir / "hidden_size_inner_validation_cells.csv", index=False)
    summary.to_csv(args.out_dir / "hidden_size_one_se_summary.csv", index=False)
    verdict = {
        "selection_rule": "smallest hidden size within one SE of best mean inner-validation loss",
        "selected_hidden_size": selected,
        "best_mean_hidden_size": best_hidden,
        "one_se_threshold": threshold,
        "n_matched_subject_seed_cells_per_size": int(len(reference)),
        "hidden_sizes_compared": hidden_sizes,
        "heldout_last20_metrics_read": False,
        "ictal_target_opened": False,
    }
    (args.out_dir / "hidden_size_selection.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(verdict, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
