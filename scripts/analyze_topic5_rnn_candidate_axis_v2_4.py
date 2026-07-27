#!/usr/bin/env python3
"""Aggregate the v2.4 RNN-selected axis experiment at patient level."""
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
    sign_invariant_cosine,
)


BASE = ROOT / "results/topic5_rnn_axis_positive_static_transfer_v2_4"
AUDIT = BASE / "input_audit/INPUT_AUDIT_STATUS.json"
FORMAL = BASE / "formal"
SEEDS = (17, 29, 43)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def bootstrap_ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    sample = rng.choice(values, size=(20_000, len(values)), replace=True)
    return np.quantile(np.median(sample, axis=1), [0.025, 0.975]).tolist()


def summarize(values: np.ndarray, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    p = (
        1.0
        if np.allclose(values, 0.0)
        else float(wilcoxon(values, alternative="greater").pvalue)
    )
    return {
        "n": len(values),
        "median": float(np.median(values)),
        "bootstrap_ci95": bootstrap_ci(values, seed),
        "n_positive": int(np.count_nonzero(values > 0)),
        "wilcoxon_greater_p": p,
    }


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    rows = []
    for subject in audit["axis_positive_primary_patients"]:
        subject_rows = []
        for seed in SEEDS:
            path = (
                FORMAL
                / "axis_search"
                / subject
                / f"seed_{seed}"
                / "metrics.json"
            )
            if not path.exists():
                raise SystemExit(f"missing candidate-axis result: {path}")
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("status") != "COMPLETE":
                raise SystemExit(f"incomplete candidate-axis result: {path}")
            subject_rows.append(row)
        axes = [np.asarray(row["selected_axis"], dtype=float) for row in subject_rows]
        pairwise = [
            sign_invariant_cosine(axes[first], axes[second])
            for first in range(len(axes))
            for second in range(first + 1, len(axes))
        ]
        rows.append(
            {
                "subject": subject,
                "relation": subject_rows[0]["relation"],
                "selected_axis_indices": ";".join(
                    str(row["selected_axis_index"]) for row in subject_rows
                ),
                "selected_alignment_abs_cosine": float(
                    np.median(
                        [row["selected_abs_cosine"] for row in subject_rows]
                    )
                ),
                "alignment_margin": float(
                    np.median([row["alignment_margin"] for row in subject_rows])
                ),
                "axis_over_isotropic_heldout_benefit": float(
                    np.median(
                        [
                            row["selected_axis_over_isotropic_benefit"]
                            for row in subject_rows
                        ]
                    )
                ),
                "source_over_no_source_heldout_benefit": (
                    float(
                        np.median(
                            [
                                row[
                                    "selected_source_over_no_source_benefit"
                                ]
                                for row in subject_rows
                            ]
                        )
                    )
                    if subject_rows[0]["relation"] == "reversed"
                    else np.nan
                ),
                "seed_axis_consistency_abs_cosine": float(np.median(pairwise)),
                "seed_alignment_sd": float(
                    np.std(
                        [row["selected_abs_cosine"] for row in subject_rows],
                        ddof=0,
                    )
                ),
                "target_values_read": False,
            }
        )

    frame = pd.DataFrame(rows).sort_values("subject")
    frame.to_csv(FORMAL / "axis_selected_patient_metrics.csv", index=False)
    alignment = frame.alignment_margin.to_numpy(float)
    predictive = frame.axis_over_isotropic_heldout_benefit.to_numpy(float)
    source = frame.loc[
        frame.relation == "reversed",
        "source_over_no_source_heldout_benefit",
    ].to_numpy(float)
    summaries = {
        "alignment_margin": summarize(alignment, 20260727),
        "axis_over_isotropic_heldout_benefit": summarize(
            predictive, 20260728
        ),
        "reversed_source_over_no_source_heldout_benefit": summarize(
            source, 20260729
        ),
        "seed_axis_consistency_abs_cosine": summarize(
            frame.seed_axis_consistency_abs_cosine.to_numpy(float),
            20260730,
        ),
    }
    gate = bool(
        summaries["alignment_margin"]["median"] > 0
        and summaries["axis_over_isotropic_heldout_benefit"]["median"] > 0
    )
    payload = {
        "contract": "topic5_rnn_candidate_axis_v2_4",
        "status": "COMPLETE",
        "n_axis_positive": len(frame),
        "n_reversed": int(np.count_nonzero(frame.relation == "reversed")),
        "metrics": summaries,
        "gate_a_axis_positive_construct_validity": "PASS" if gate else "FAIL",
        "claim_scope": (
            "secondary construct validity in a pre-existing axis-positive subgroup"
        ),
        "target_values_read": False,
    }
    atomic_json(FORMAL / "AXIS_SELECTION_GATE_STATUS.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
