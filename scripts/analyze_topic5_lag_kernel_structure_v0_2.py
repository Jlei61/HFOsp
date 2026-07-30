#!/usr/bin/env python3
"""Secondary spatial audit of invariant contact-space lag kernels."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_topic5_interictal_rank_distribution import load_records  # noqa: E402


DATASET = ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4"
FORMAL = ROOT / "results/topic5_minimal_sequence_kernel_closeout/formal_v0_2"


def _shaft(name: str) -> str:
    value = re.sub(r"\d+$", "", str(name)).strip("_- ")
    return value or str(name)


def main() -> None:
    records = load_records(DATASET)
    rows = []
    for path in sorted(FORMAL.glob("seed_*/*/linear_state_lag_kernels.npz")):
        subject = path.parent.name
        seed = int(path.parents[1].name.split("_")[-1])
        record = records[subject]
        with np.load(record.path, allow_pickle=False) as data:
            names = np.asarray(data["contact_names"]).astype(str)
            coords = np.asarray(data["contact_coords"], float)
        shafts = np.asarray([_shaft(name) for name in names])
        same = shafts[:, None] == shafts[None, :]
        diagonal = np.eye(len(names), dtype=bool)
        same &= ~diagonal
        cross = ~same & ~diagonal
        coordinate_valid = np.all(np.isfinite(coords), axis=1)
        pair_valid = coordinate_valid[:, None] & coordinate_valid[None, :] & ~diagonal
        distance = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
        with np.load(path, allow_pickle=False) as data:
            kernels = np.asarray(data["contact_kernels"], float)
        if kernels.shape[1:] != (len(names), len(names)):
            raise RuntimeError(f"{subject}: kernel/contact ordering drift")
        for lag in range(4):
            kernel = kernels[lag]
            absolute = np.abs(kernel)
            rho = (
                float(
                    spearmanr(
                        distance[pair_valid], absolute[pair_valid], nan_policy="omit"
                    ).statistic
                )
                if np.count_nonzero(pair_valid) >= 12
                else np.nan
            )
            rows.append(
                {
                    "subject": subject,
                    "dataset": record.dataset,
                    "seed": seed,
                    "lag": lag,
                    "same_shaft_abs_mean": float(np.mean(absolute[same]))
                    if np.any(same)
                    else np.nan,
                    "cross_shaft_abs_mean": float(np.mean(absolute[cross]))
                    if np.any(cross)
                    else np.nan,
                    "same_minus_cross_abs": float(
                        np.mean(absolute[same]) - np.mean(absolute[cross])
                    )
                    if np.any(same) and np.any(cross)
                    else np.nan,
                    "distance_vs_abs_spearman": rho,
                    "geometry_complete": bool(np.all(coordinate_valid)),
                }
            )
    seed_frame = pd.DataFrame(rows)
    if len(seed_frame) != 102 * 4:
        raise RuntimeError("lag-kernel spatial inventory is incomplete")
    seed_frame.to_csv(FORMAL / "lag_kernel_spatial_structure_seed.csv", index=False)
    patient = (
        seed_frame.groupby(["subject", "dataset", "lag"], as_index=False)
        .agg(
            same_shaft_abs_mean=("same_shaft_abs_mean", "median"),
            cross_shaft_abs_mean=("cross_shaft_abs_mean", "median"),
            same_minus_cross_abs=("same_minus_cross_abs", "median"),
            distance_vs_abs_spearman=("distance_vs_abs_spearman", "median"),
            geometry_complete=("geometry_complete", "all"),
        )
    )
    patient.to_csv(FORMAL / "lag_kernel_spatial_structure_patient.csv", index=False)
    summary_rows = []
    for lag, frame in patient.groupby("lag"):
        values = frame.same_minus_cross_abs.dropna().to_numpy()
        nonzero = values[values != 0]
        summary_rows.append(
            {
                "lag": int(lag),
                "n_same_shaft_estimable": int(len(values)),
                "same_minus_cross_abs_median": float(np.median(values)),
                "same_minus_cross_positive": int(np.sum(values > 0)),
                "same_minus_cross_wilcoxon_two_sided": float(
                    wilcoxon(nonzero, alternative="two-sided").pvalue
                )
                if len(nonzero)
                else 1.0,
                "n_geometry_estimable": int(
                    frame.distance_vs_abs_spearman.notna().sum()
                ),
                "distance_vs_abs_spearman_median": float(
                    frame.distance_vs_abs_spearman.median()
                ),
            }
        )
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(FORMAL / "lag_kernel_spatial_structure_summary.csv", index=False)
    payload = {
        "status": "COMPLETE",
        "contract": "topic5_minimal_sequence_kernel_closeout_v0_2",
        "tier": "secondary_structure_audit",
        "n_patient_seed": int(seed_frame[["subject", "seed"]].drop_duplicates().shape[0]),
        "axis_analysis": "NOT_RUN_NO_INDEPENDENT_AXIS_USED_IN_THIS_CLOSEOUT",
        "summary": summary_frame.to_dict(orient="records"),
        "target_values_read": False,
    }
    (FORMAL / "LAG_KERNEL_STRUCTURE_SUMMARY.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
