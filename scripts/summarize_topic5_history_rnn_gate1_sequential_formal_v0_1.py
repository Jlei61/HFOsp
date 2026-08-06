#!/usr/bin/env python3
"""Patient-first formal G1 inference without opening early-ictal targets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


DEVELOPMENT = {
    "epilepsiae_1073",
    "epilepsiae_1146",
    "yuquan_chenziyang",
}


def _one_sided_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values) or np.allclose(values, 0.0):
        return 1.0
    return float(
        wilcoxon(
            values,
            alternative="greater",
            zero_method="wilcox",
            method="auto",
        ).pvalue
    )


def _bootstrap_median_ci(
    values: np.ndarray, *, seed: int = 20260801, draws: int = 20_000
) -> list[float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(int(draws), len(values)))]
    return [float(value) for value in np.quantile(np.median(sampled, axis=1), [0.025, 0.975])]


def _cohort_summary(frame: pd.DataFrame) -> dict:
    chronology = frame.chronological_increment.to_numpy(float)
    order = frame.prefix_matched_order_shuffle_cost.to_numpy(float)
    dataset_direction = {
        dataset: {
            "n_patients": int(len(group)),
            "median_chronological_increment": float(
                group.chronological_increment.median()
            ),
            "n_chronological_positive": int(
                np.sum(group.chronological_increment > 0)
            ),
            "median_order_shuffle_cost": float(
                group.prefix_matched_order_shuffle_cost.median()
            ),
            "n_order_shuffle_positive": int(
                np.sum(group.prefix_matched_order_shuffle_cost > 0)
            ),
        }
        for dataset, group in frame.groupby("dataset", sort=True)
    }
    return {
        "n_patients": int(len(frame)),
        "median_static_to_matched_gain": float(frame.static_to_matched_gain.median()),
        "median_chronological_increment": float(np.median(chronology)),
        "chronological_increment_bootstrap_median_ci95": _bootstrap_median_ci(
            chronology
        ),
        "chronological_increment_one_sided_wilcoxon_p": _one_sided_p(chronology),
        "n_chronological_positive": int(np.sum(chronology > 0)),
        "median_order_shuffle_cost": float(np.median(order)),
        "order_shuffle_cost_bootstrap_median_ci95": _bootstrap_median_ci(
            order, seed=20260802
        ),
        "order_shuffle_cost_one_sided_wilcoxon_p": _one_sided_p(order),
        "n_order_shuffle_positive": int(np.sum(order > 0)),
        "median_rank_increment": float(frame.rank_increment.median()),
        "median_within_event_rank_shuffle_cost": float(
            frame.within_event_rank_shuffle_cost.median()
        ),
        "n_within_event_rank_shuffle_positive": int(
            np.sum(frame.within_event_rank_shuffle_cost > 0)
        ),
        "dataset_direction": dataset_direction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.input_dir.resolve()
    rows = []
    for path in sorted(root.glob("*/DONE.json")):
        payload = json.loads(path.read_text())
        controls_path = path.parent / "ORDER_CONTROLS.json"
        if not controls_path.exists():
            raise RuntimeError(f"missing formal order controls: {controls_path}")
        controls = json.loads(controls_path.read_text())
        if bool(payload.get("target_values_read", True)):
            raise RuntimeError(f"target seal violated in {path}")
        if bool(controls.get("target_values_read", True)):
            raise RuntimeError(f"target seal violated in {controls_path}")
        contrast = payload["metrics"]["contrasts"]
        control_contrast = controls["metrics"]["contrasts"]
        rows.append(
            {
                "subject": payload["heldout_subject"],
                "dataset": payload["heldout_subject"].split("_", 1)[0],
                "development_patient": payload["heldout_subject"] in DEVELOPMENT,
                "n_events": payload["metrics"]["chronological_history"]["n_events"],
                "static_to_matched_gain": contrast[
                    "static_minus_matched_participation_bce"
                ],
                "chronological_increment": contrast[
                    "matched_minus_chronological_participation_bce"
                ],
                "rank_increment": contrast[
                    "matched_minus_chronological_relative_rank_huber"
                ],
                "order_shuffle_cost": contrast[
                    "shuffle_minus_chronological_participation_bce"
                ],
                "prefix_matched_order_shuffle_cost": control_contrast[
                    "prefix_matched_shuffle_minus_chronological_bce"
                ],
                "within_event_rank_shuffle_cost": control_contrast[
                    "within_event_rank_shuffle_minus_chronological_bce"
                ],
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 34 or frame.subject.nunique() != 34:
        raise RuntimeError(f"formal G1 incomplete: {len(frame)}/34")
    frame.to_csv(root / "g1_patient_metrics.csv", index=False)
    primary = frame.loc[~frame.development_patient].copy()
    supportive = frame.copy()
    primary_summary = _cohort_summary(primary)
    supportive_summary = _cohort_summary(supportive)
    dataset_positive = all(
        values["median_chronological_increment"] > 0
        for values in primary_summary["dataset_direction"].values()
    )
    gate_pass = (
        primary_summary["median_chronological_increment"] > 0
        and primary_summary["chronological_increment_one_sided_wilcoxon_p"] < 0.05
        and dataset_positive
        and primary_summary["median_order_shuffle_cost"] > 0
        and primary_summary["order_shuffle_cost_one_sided_wilcoxon_p"] < 0.05
    )
    result = {
        "status": "G1_PASS_OPEN_G2" if gate_pass else "G1_FAIL_KEEP_ICTAL_TARGET_SEALED",
        "contract": "topic5_history_rnn_early_ictal_field_v0_1_g1_formal",
        "target_values_read": False,
        "primary_cohort": "31 development-excluded patients",
        "gate_definition": {
            "primary": "patient-level BCE(M1)-BCE(HistoryRNN) > 0",
            "requirements": [
                "positive cohort median",
                "one-sided paired Wilcoxon p<0.05",
                "positive median in both Epilepsiae and Yuquan",
                "across-event order shuffle significantly worsens HistoryRNN",
            ],
        },
        "primary": primary_summary,
        "all_34_supportive": supportive_summary,
    }
    (root / "G1_SUMMARY.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
