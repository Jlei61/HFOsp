#!/usr/bin/env python3
"""Analyze existing train80 transition-axis read-back in the frozen n=9 subgroup."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_axis_positive_static_transfer_v2_4 import (  # noqa: E402
    candidate_alignment_summary,
    sign_invariant_cosine,
)
from src.topic5_transition_decomposition_v0_1 import fibonacci_axes  # noqa: E402


BASE = ROOT / "results/topic5_rnn_axis_positive_static_transfer_v2_4"
AUDIT = BASE / "input_audit"
V23 = ROOT / "results/topic5_symmetric_axis_competitive_propagation_v2_3"
DATASET = ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4"
OUT = BASE / "axis_readback_stage_a0"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def bootstrap_median_ci(
    values: np.ndarray, *, seed: int = 20260727, draws: int = 20_000
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample = rng.choice(values, size=(draws, len(values)), replace=True)
    medians = np.median(sample, axis=1)
    return tuple(np.quantile(medians, [0.025, 0.975]).tolist())


def one_sided_wilcoxon(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if np.allclose(values, 0.0):
        return 1.0
    return float(
        wilcoxon(
            values,
            alternative="greater",
            zero_method="wilcox",
            method="auto",
        ).pvalue
    )


def pca_axis(coords: np.ndarray) -> np.ndarray:
    centered = np.asarray(coords, dtype=np.float64)
    centered = centered - centered.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return vh[0]


def main() -> None:
    audit_status = json.loads(
        (AUDIT / "INPUT_AUDIT_STATUS.json").read_text(encoding="utf-8")
    )
    if audit_status.get("target_values_read"):
        raise SystemExit("target seal failed before Stage A0")
    axis_positive = pd.read_csv(AUDIT / "axis_positive_cohort.csv")
    formal_axes = pd.read_csv(V23 / "input_audit/formal_axis_inventory.csv")
    model_metrics = pd.read_csv(V23 / "formal/patient_model_metrics.csv")
    directions = fibonacci_axes(32)
    rows: list[dict[str, Any]] = []
    for _, axis_row in axis_positive.iterrows():
        subject = str(axis_row.subject)
        selected_row = formal_axes.loc[
            formal_axes.subject.astype(str) == subject
        ]
        if len(selected_row) != 1:
            raise ValueError(f"{subject}: transition axis missing")
        selected = selected_row[
            ["axis_x", "axis_y", "axis_z"]
        ].to_numpy(float)[0]
        reference = axis_row[
            ["shared_axis_x", "shared_axis_y", "shared_axis_z"]
        ].to_numpy(float)
        alignment = candidate_alignment_summary(
            selected, reference, directions
        )
        with np.load(
            DATASET / "per_subject" / f"{subject}.npz",
            allow_pickle=False,
        ) as data:
            coords = np.asarray(data["contact_coords"], dtype=np.float64)
        pca = pca_axis(coords)
        subject_metrics = model_metrics.loc[
            model_metrics.subject.astype(str) == subject
        ].set_index("model")
        isotropic = float(
            subject_metrics.loc[
                "local_isotropic_two_state", "heldout_categorical_nll"
            ]
        )
        no_source = float(
            subject_metrics.loc[
                "axis_two_state_no_source", "heldout_categorical_nll"
            ]
        )
        full = float(
            subject_metrics.loc[
                "axis_two_state_source_full", "heldout_categorical_nll"
            ]
        )
        rows.append(
            {
                "subject": subject,
                "relation": str(axis_row.relation),
                "strict_stability_pass": bool(
                    axis_row.strict_stability_pass
                ),
                **alignment,
                "selected_vs_pca1_abs_cosine": sign_invariant_cosine(
                    selected, pca
                ),
                "shared_vs_pca1_abs_cosine": sign_invariant_cosine(
                    reference, pca
                ),
                "isotropic_heldout_nll": isotropic,
                "axis_no_source_heldout_nll": no_source,
                "axis_full_heldout_nll": full,
                "axis_no_source_over_isotropic_benefit": isotropic - no_source,
                "axis_full_over_isotropic_benefit": isotropic - full,
                "source_over_no_source_benefit": no_source - full,
                "axis_selection_split": str(
                    selected_row.axis_selection_split.iloc[0]
                ),
                "target_values_read": False,
            }
        )

    frame = pd.DataFrame(rows).sort_values("subject")
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "patient_metrics.csv", index=False)
    metric_names = (
        "alignment_margin",
        "axis_no_source_over_isotropic_benefit",
        "axis_full_over_isotropic_benefit",
        "source_over_no_source_benefit",
    )
    summaries: dict[str, Any] = {}
    for name in metric_names:
        values = frame[name].to_numpy(float)
        lower, upper = bootstrap_median_ci(values)
        summaries[name] = {
            "n": len(values),
            "median": float(np.median(values)),
            "bootstrap_ci95": [lower, upper],
            "n_positive": int(np.count_nonzero(values > 0)),
            "wilcoxon_greater_p": one_sided_wilcoxon(values),
        }
    reversed_values = frame.loc[
        frame.relation == "reversed", "source_over_no_source_benefit"
    ].to_numpy(float)
    lower, upper = bootstrap_median_ci(reversed_values)
    summaries["reversed_source_over_no_source_benefit"] = {
        "n": len(reversed_values),
        "median": float(np.median(reversed_values)),
        "bootstrap_ci95": [lower, upper],
        "n_positive": int(np.count_nonzero(reversed_values > 0)),
        "wilcoxon_greater_p": one_sided_wilcoxon(reversed_values),
    }
    payload = {
        "contract": "topic5_axis_readback_stage_a0_v2_4",
        "status": "COMPLETE",
        "interpretation": (
            "transition-selected axis construct read-back; not RNN axis discovery"
        ),
        "n_axis_positive": len(frame),
        "n_reversed": int(np.count_nonzero(frame.relation == "reversed")),
        "metrics": summaries,
        "target_values_read": False,
    }
    atomic_json(OUT / "STAGE_A0_STATUS.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
