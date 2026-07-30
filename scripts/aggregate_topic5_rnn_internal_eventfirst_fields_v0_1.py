#!/usr/bin/env python3
"""Reaggregate corrected event-first perturbation fields without changing freeze."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/topic5_rnn_internal_state_reduction"


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def summary(values: np.ndarray, seed: int) -> dict:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    rng = np.random.default_rng(int(seed))
    sampled = rng.choice(data, size=(20_000, len(data)), replace=True)
    return {
        "n": int(len(data)),
        "median": float(np.median(data)),
        "bootstrap_ci95": np.quantile(
            np.median(sampled, axis=1), [0.025, 0.975]
        ).tolist(),
        "n_positive": int(np.count_nonzero(data > 0)),
        "wilcoxon_greater_p": (
            1.0
            if np.allclose(data, 0.0)
            else float(wilcoxon(data, alternative="greater").pvalue)
        ),
    }


def main() -> None:
    frames = []
    for path in sorted(
        (BASE / "interictal/perturbation_cells").glob(
            "seed_*/**/direction_contact_fields.csv"
        )
    ):
        frames.append(pd.read_csv(path))
    if len(frames) != 102:
        raise RuntimeError(f"expected 102 corrected cells, found {len(frames)}")
    contact = pd.concat(frames, ignore_index=True)
    required_halves = {"all", "first", "second"}
    if set(contact.event_half.astype(str).unique()) != required_halves:
        raise RuntimeError("event-half field inventory is incomplete")
    contact.to_csv(BASE / "interictal_direction_contact_fields.csv", index=False)

    rows = []
    selected = contact.loc[np.isclose(contact.amplitude_sd, 0.5)]
    keys = ["subject", "control", "direction_type", "direction_index"]
    for key, group in selected.groupby(keys):
        seed_stability = []
        half_stability = []
        for seed_dir, seed in group.groupby("seed_dir"):
            wide = seed.pivot(
                index="contact_name",
                columns="event_half",
                values="probability_contrast",
            ).dropna()
            half_stability.append(
                float(spearmanr(wide["first"], wide["second"]).statistic)
            )
        all_field = group.loc[group.event_half == "all"]
        wide_seed = all_field.pivot(
            index="contact_name",
            columns="seed_dir",
            values="probability_contrast",
        ).dropna()
        seed_names = sorted(wide_seed.columns)
        for left_index, left in enumerate(seed_names):
            for right in seed_names[left_index + 1 :]:
                seed_stability.append(
                    float(spearmanr(wide_seed[left], wide_seed[right]).statistic)
                )
        rows.append(
            {
                **dict(zip(keys, key)),
                "n_contacts": int(len(wide_seed)),
                "median_cross_seed_spearman": float(
                    np.nanmedian(seed_stability)
                ),
                "median_heldout_half_spearman": float(
                    np.nanmedian(half_stability)
                ),
            }
        )
    stability = pd.DataFrame(rows)
    stability.to_csv(
        BASE / "interictal_eventfirst_direction_field_stability.csv", index=False
    )
    metrics = {}
    for key, group in stability.groupby(
        ["control", "direction_type", "direction_index"]
    ):
        control, direction_type, direction_index = key
        prefix = f"{control}__{direction_type}{direction_index}"
        metrics[f"{prefix}__cross_seed_spearman"] = summary(
            group.median_cross_seed_spearman.to_numpy(float),
            2026078000 + len(metrics),
        )
        metrics[f"{prefix}__heldout_half_spearman"] = summary(
            group.median_heldout_half_spearman.to_numpy(float),
            2026078000 + len(metrics),
        )
    atomic_json(
        BASE / "INTERICTAL_EVENTFIRST_FIELD_SUMMARY.json",
        {
            "contract": (
                "topic5_rnn_internal_state_reduction_v0_1_eventfirst_field_fix"
            ),
            "status": "COMPLETE",
            "n_subjects": 34,
            "n_seeds": 3,
            "aggregation": (
                "prefix contrast averaged within event, then across events"
            ),
            "metrics": metrics,
            "direction_selection_revised": False,
            "target_values_used": False,
        },
    )


if __name__ == "__main__":
    main()
