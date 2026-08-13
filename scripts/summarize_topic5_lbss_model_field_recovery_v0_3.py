#!/usr/bin/env python3
"""Patient-first target-free summary of LBSS-generated interictal fields."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


DEFAULT_OUT = Path("results/topic5_lbss_full_tissue_rnn_v0_3")
SHUFFLE = "C_L3_ORDER_SHUFFLED"
ARMS = (
    "L0_LOCAL_ONLY",
    "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
    "L2_LOCAL_PLUS_RANDOM_LR",
    "L3_LOCAL_PLUS_LEARNED_LR",
)
METRICS = (
    "canonical_empirical_r",
    "seed_removed_empirical_r",
    "canonical_contrast_empirical_r",
    "seed_removed_contrast_empirical_r",
)
TOLERANCE = 1e-9


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired(values: np.ndarray) -> dict:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    tied = np.abs(values) <= TOLERANCE
    nonzero = values[~tied]
    p = 1.0 if not len(nonzero) else float(
        wilcoxon(nonzero, alternative="two-sided", method="auto").pvalue
    )
    return {
        "n": int(len(values)),
        "median": float(np.median(values)) if len(values) else float("nan"),
        "n_positive": int((values > TOLERANCE).sum()),
        "n_negative": int((values < -TOLERANCE).sum()),
        "n_tied": int(tied.sum()),
        "wilcoxon_p_two_sided": p,
    }


def holm(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=raw.get)
    output: dict[str, float] = {}
    running = 0.0
    for index, key in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - index) * raw[key]))
        output[key] = running
    return output


def summarize(frame: pd.DataFrame) -> dict:
    expected = set(ARMS) | {SHUFFLE}
    if set(frame.arm) != expected:
        raise RuntimeError(f"field arm denominator changed: {sorted(set(frame.arm))}")
    if frame.subject.nunique() != 21 or len(frame) != 21 * len(expected):
        raise RuntimeError("field patient denominator changed")
    output: dict[str, dict] = {}
    shuffled = frame[frame.arm.eq(SHUFFLE)].set_index("subject")
    for metric in METRICS:
        rows: dict[str, dict] = {}
        raw_p: dict[str, float] = {}
        for arm in ARMS:
            model = frame[frame.arm.eq(arm)].set_index("subject")
            subjects = model.index.intersection(shuffled.index)
            observed = model.loc[subjects, metric].to_numpy(float)
            delta = observed - shuffled.loc[subjects, metric].to_numpy(float)
            comparison = paired(delta)
            raw_p[arm] = comparison["wilcoxon_p_two_sided"]
            rows[arm] = {
                "field_median": float(np.nanmedian(observed)),
                "field_n_positive": int((observed > TOLERANCE).sum()),
                "field_n_negative": int((observed < -TOLERANCE).sum()),
                "field_n_tied": int((np.abs(observed) <= TOLERANCE).sum()),
                "vs_order_shuffle": comparison,
            }
        adjusted = holm(raw_p)
        for arm in ARMS:
            rows[arm]["vs_order_shuffle"]["holm_q_within_metric"] = adjusted[arm]
        output[metric] = rows
    return {
        "contract": "topic5_lbss_model_field_recovery_v0_3",
        "n_patients": 21,
        "patient_first": True,
        "reference": SHUFFLE,
        "inference": "paired two-sided Wilcoxon; Holm across four true-order arms within each field metric",
        "interpretation": (
            "Tests whether frozen recurrent rollouts recover empirical interictal fields beyond an "
            "order-shuffled recurrent control. This target-free summary does not select a spatial arm."
        ),
        "metrics": output,
        "target_values_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out_root.resolve()
    source = out / "model_field_patient_metrics.csv"
    result = summarize(pd.read_csv(source))
    result["input_sha256"] = sha256(source)
    destination = out / "MODEL_FIELD_RECOVERY_SUMMARY.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
