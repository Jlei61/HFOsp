#!/usr/bin/env python3
"""Summarize frozen v2.4 static-readout models without changing its gates."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/topic5_rnn_axis_positive_static_transfer_v2_4"
STATIC = BASE / "static_readout"
SEED = 20260805


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
    p_value = (
        1.0
        if np.allclose(values, 0.0)
        else float(wilcoxon(values, alternative="greater").pvalue)
    )
    return {
        "n": int(len(values)),
        "median": float(np.median(values)),
        "bootstrap_ci95": bootstrap_ci(values, seed),
        "n_positive": int(np.count_nonzero(values > 0)),
        "wilcoxon_greater_p_uncorrected_descriptive": p_value,
    }


def main() -> None:
    gate_path = STATIC / "STATIC_READOUT_GATE_STATUS.json"
    metrics_path = STATIC / "patient_model_metrics.csv"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(metrics_path)
    if gate.get("status") != "COMPLETE" or not gate.get("target_values_read"):
        raise SystemExit("static readout is not complete")
    if not frame.target_values_read.astype(bool).all():
        raise SystemExit("target-read provenance is inconsistent")

    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for index, (model, model_frame) in enumerate(
        frame.groupby("model", sort=True)
    ):
        all_summary = summarize(
            model_frame.all_contact_margin.to_numpy(float),
            SEED + index * 2,
        )
        shaft_summary = summarize(
            model_frame.within_shaft_margin.to_numpy(float),
            SEED + index * 2 + 1,
        )
        rho_summary = summarize(
            model_frame.spearman_rho.to_numpy(float),
            SEED + 100 + index,
        )
        summaries[str(model)] = {
            "all_contact_margin": all_summary,
            "within_shaft_margin": shaft_summary,
            "spearman_rho": rho_summary,
        }
        rows.append(
            {
                "model": model,
                "n": all_summary["n"],
                "median_rho": rho_summary["median"],
                "median_all_contact_margin": all_summary["median"],
                "all_contact_n_positive": all_summary["n_positive"],
                "all_contact_wilcoxon_p_uncorrected_descriptive": (
                    all_summary[
                        "wilcoxon_greater_p_uncorrected_descriptive"
                    ]
                ),
                "median_within_shaft_margin": shaft_summary["median"],
                "within_shaft_n_positive": shaft_summary["n_positive"],
                "within_shaft_wilcoxon_p_uncorrected_descriptive": (
                    shaft_summary[
                        "wilcoxon_greater_p_uncorrected_descriptive"
                    ]
                ),
            }
        )

    summary_frame = pd.DataFrame(rows).sort_values(
        "median_all_contact_margin", ascending=False
    )
    summary_frame.to_csv(STATIC / "model_absolute_diagnostics.csv", index=False)
    empirical = summaries["empirical_train80"]
    full = summaries["full_fixed_axis"]
    payload = {
        "contract": "topic5_static_readout_diagnostics_v2_4",
        "status": "COMPLETE",
        "analysis_role": (
            "post-gate decomposition of already frozen outputs; "
            "does not alter Gates S/H/X"
        ),
        "n_patients": int(frame.subject.nunique()),
        "models": summaries,
        "frozen_gate_status": {
            "gate_s": gate["gate_s_source_free_static_readout"],
            "gate_h": gate["gate_h_history_contribution"],
            "gate_x": gate["gate_x_axis_contribution"],
        },
        "bounded_interpretation": (
            "The empirical train80 rank distribution showed a positive "
            "patient-median cross-state trend, but its bootstrap interval "
            "included zero and its P values are descriptive and uncorrected. "
            "Every simulated RNN representation failed the frozen Gate S. "
            "The contrast suggests that the current model/rollout compression "
            "may lose information present in the empirical interictal ranks; "
            "it does not by itself establish a new cross-state cohort claim."
        ),
        "empirical_all_contact_margin_median": empirical[
            "all_contact_margin"
        ]["median"],
        "empirical_within_shaft_margin_median": empirical[
            "within_shaft_margin"
        ]["median"],
        "full_all_contact_margin_median": full["all_contact_margin"]["median"],
        "target_values_read": True,
        "no_retraining_or_hyperparameter_selection": True,
        "inference_scope": (
            "descriptive post-gate diagnostic; no multiplicity-corrected "
            "empirical-model claim"
        ),
    }
    atomic_json(STATIC / "STATIC_READOUT_DIAGNOSTICS.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
