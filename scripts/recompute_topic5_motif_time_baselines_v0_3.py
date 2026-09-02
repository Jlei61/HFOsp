#!/usr/bin/env python3
"""Recompute the three time baselines on validation and test, with a ridge.

The first pass scored the baselines on the test split only and then let the
aggregator take the per-patient minimum, which is choosing the comparator after
seeing the test error.  It changed nothing here — 26 of 28 patients pick
STATIC_TARGET either way and every cohort median is identical — but the contract
says the test split is scored once and never selected on, so the selection moves to
validation.

No model is retrained: these are closed-form least squares on cached tensors.
"""
from __future__ import annotations

import os as _os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_dynamical_motif_data_v0_1 import load_frame_unit  # noqa: E402
from src.topic5_motif_time_targets_v0_3 import (  # noqa: E402
    TIME_BASELINES,
    build_event_tensors_with_time,
    time_baseline_scores,
)

MOTIF_ROOT = ROOT / "results/topic5_dynamical_motif_rnn_v0_1"
RESULT_ROOT = ROOT / "results/topic5_motif_time_targets_v0_3"
RIDGE_GRID = (0.0, 0.1, 1.0, 10.0, 100.0)


def main() -> int:
    scores = pd.read_csv(RESULT_ROOT / "PER_ARM_SCORES.csv")
    rows = []
    for patient in sorted(scores["patient"].unique()):
        unit = load_frame_unit(MOTIF_ROOT, "GEOMETRY_ONLY_PCA2", patient)
        tensors = build_event_tensors_with_time(
            unit.ranks, unit.contacts_xy_mm, unit.event_lag_raw)
        delta = tensors["time_delta"].numpy()
        valid = tensors["time_valid"].numpy()
        centroid = tensors["centroid"].numpy()
        distance = np.zeros_like(delta)
        distance[:, :-1] = np.linalg.norm(centroid[:, 1:] - centroid[:, :-1], axis=-1)
        split = np.asarray(unit.split)
        train, validation, test = split == 0, split == 1, split == 2
        logged = np.log1p(delta)
        centre = float(logged[train][valid[train]].mean())
        scale = max(float(logged[train][valid[train]].std()), 1e-9)
        standardised = (logged - centre) / scale
        target = tensors["target"].numpy()

        # the penalty and the comparator are both chosen on validation
        by_ridge = {ridge: time_baseline_scores(standardised, valid, distance, target,
                                                train, validation, ridge=ridge)
                    for ridge in RIDGE_GRID}
        best_ridge = min(
            RIDGE_GRID,
            key=lambda r: min(v for v in by_ridge[r].values() if np.isfinite(v)))
        validation_scores = by_ridge[best_ridge]
        chosen = min((level for level in TIME_BASELINES
                      if np.isfinite(validation_scores[level])),
                     key=lambda level: validation_scores[level])
        test_scores = time_baseline_scores(standardised, valid, distance, target,
                                           train, test, ridge=best_ridge)
        rows.append({
            "patient": patient, "selected_ridge": best_ridge,
            "selected_baseline_on_validation": chosen,
            "comparator_test_mse": test_scores[chosen],
            **{f"validation_{k}": v for k, v in validation_scores.items()},
            **{f"test_{k}": v for k, v in test_scores.items()},
        })
        print(f"  {patient:22s} ridge={best_ridge:<6g} comparator={chosen:14s} "
              f"test={test_scores[chosen]:.5f}", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(RESULT_ROOT / "TIME_BASELINES_VALIDATION_SELECTED.csv", index=False)
    print(f"\nwrote {RESULT_ROOT / 'TIME_BASELINES_VALIDATION_SELECTED.csv'}")
    print("comparator chosen on validation:",
          table["selected_baseline_on_validation"].value_counts().to_dict())
    print("ridge chosen:", table["selected_ridge"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
