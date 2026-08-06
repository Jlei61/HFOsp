#!/usr/bin/env python3
"""Summarize patient-first data-aligned static/RNN maxAB transfer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


TOL = 1e-9
DEVELOPMENT_PATIENT = "epilepsiae_1146"


def _paired(values: np.ndarray, *, alternative: str = "greater") -> dict:
    value = np.asarray(values, dtype=float)
    value = value[np.isfinite(value)]
    tie = np.abs(value) <= TOL
    nonzero = value[~tie]
    result = {
        "n": int(len(value)),
        "median": float(np.median(value)) if len(value) else None,
        "n_positive": int(np.sum(value > TOL)),
        "n_negative": int(np.sum(value < -TOL)),
        "n_tie": int(tie.sum()),
        "alternative": alternative,
    }
    if not len(nonzero):
        result["p"] = 1.0
    else:
        result["p"] = float(
            wilcoxon(nonzero, alternative=alternative, method="exact").pvalue
        )
    return result


def _cohort(frame: pd.DataFrame) -> dict:
    pivot = frame.pivot(index="subject", columns="model", values="observed_patient_median_maxab")
    null_pivot = frame.pivot(index="subject", columns="model", values="channel_null_median")
    output = {
        "n_patients": int(len(pivot)),
        "models": {},
        "contrasts": {},
    }
    for model in pivot:
        values = pivot[model].to_numpy(float)
        margin = values - null_pivot[model].to_numpy(float)
        model_rows = frame.loc[frame.model == model]
        output["models"][model] = {
            "median_maxab_abs_rho": float(np.nanmedian(values)),
            "q25_maxab_abs_rho": float(np.nanpercentile(values, 25)),
            "q75_maxab_abs_rho": float(np.nanpercentile(values, 75)),
            "channel_null_margin": _paired(margin),
            "n_above_own_channel_null_p95": int(model_rows.pass_channel_null_p95.sum()),
        }
    contrasts = {
        "STATIC_LEARNED_minus_RAW_STATIC": ("STATIC_LEARNED", "RAW_STATIC"),
        "STATIC_M1_minus_STATIC_LEARNED": ("STATIC_M1", "STATIC_LEARNED"),
        "STATIC_RNN_minus_STATIC_M1": ("STATIC_RNN", "STATIC_M1"),
        "STATIC_RNN_minus_RAW_STATIC": ("STATIC_RNN", "RAW_STATIC"),
        "STATIC_RNN_minus_ORDER_SHUFFLE": ("STATIC_RNN", "STATIC_RNN_ORDER_SHUFFLE"),
        "STATIC_RNN_minus_ZERO_STATE": ("STATIC_RNN", "STATIC_RNN_ZERO_STATE"),
    }
    for label, (left, right) in contrasts.items():
        if left in pivot and right in pivot:
            output["contrasts"][label] = _paired(
                pivot[left].to_numpy(float) - pivot[right].to_numpy(float)
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.input_dir.resolve()
    done = sorted(root.glob("epilepsiae_*/DONE.json"))
    failed = sorted(root.glob("*.FAILED.json")) + sorted(root.glob("epilepsiae_*.FAILED.json"))
    if len(done) != 16 or failed:
        raise RuntimeError(f"incomplete folds: done={len(done)} failed={len(failed)}")
    metrics = pd.concat(
        [pd.read_csv(path.parent / "heldout_channel_null_metrics.csv") for path in done],
        ignore_index=True,
    )
    metrics.to_csv(root / "data_aligned_patient_metrics.csv", index=False)
    primary = metrics.loc[metrics.subject != DEVELOPMENT_PATIENT].copy()
    summary = {
        "status": "COMPLETE_NO_RESULT_GATE",
        "contract": "topic5_history_rnn_data_aligned_static_transfer_v0_3",
        "scientific_question": "Can a static-first dual-field readout, scored with the paper's sign-free maxAB contract, predict early-ictal contact energy and benefit from frozen HistoryRNN fields?",
        "primary_cohort": _cohort(primary),
        "supportive_cohort": _cohort(metrics),
        "primary_exclusion": DEVELOPMENT_PATIENT,
        "n_completed_folds": len(done),
        "n_failed_folds": len(failed),
        "interpretation_rule": "Report effect sizes and matched nulls; do not convert this exploratory data-aligned screen into a hard gate.",
    }
    (root / "DATA_ALIGNED_TRANSFER_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
