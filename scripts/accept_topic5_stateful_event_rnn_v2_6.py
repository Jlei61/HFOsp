#!/usr/bin/env python3
"""Derived acceptance layer for the frozen Topic 5 stateful event-sequence RNN v2.6.

This script never trains and never reads raw recordings.  It only re-derives
cohort statistics from the already frozen per-patient artifacts, so the frozen
config/module/runner hashes stay valid.

It adds the three quantities the primary runner never wrote:

1. the trained RNN versus the static train-repertoire mean (the artifact behind
   the "RNN beats the static repertoire" claim);
2. the same contrasts stratified by how many non-overlapping formal test windows
   a patient actually contributes;
3. the opposite tail of each chronology null, plus the seed-dispersion and
   training-budget audits needed to bound the negative EWMA comparison.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULT_ROOT = ROOT / "results/topic5_stateful_event_sequence_rnn/v2_6"
CONTRACT = "topic5_stateful_event_sequence_rnn_v2_6"
BOOTSTRAP_SEED = 20260802
BOOTSTRAP_DRAWS = 10000
SUPPORT_THRESHOLDS = (1, 10, 20, 50)
SCIENTIFIC_ADJUDICATION = "ACCEPTED_AS_STATE_TRACKING_PRECURSOR_WITH_KNOWN_TRAINING_BIAS"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def directional_summary(values: Sequence[float], *, favorable: str) -> dict:
    """Cohort summary of a per-patient contrast.

    ``favorable='negative'`` means a lower score is the better model, which is
    the convention of every v2.6 score.  Both tails are always reported: a
    one-sided null that fails is not the same statement as a null that is beaten
    in the opposite direction, and the surrogate arms need that distinction.
    """
    if favorable not in {"negative", "positive"}:
        raise ValueError(f"unknown favorable direction: {favorable}")
    array = np.asarray(values, float)
    array = array[np.isfinite(array)]
    if not len(array):
        raise ValueError("directional_summary needs at least one finite value")
    alternative = "less" if favorable == "negative" else "greater"
    opposite = "greater" if favorable == "negative" else "less"
    count = int(np.sum(array < 0)) if favorable == "negative" else int(np.sum(array > 0))
    nonzero = array[array != 0]
    non_ties = (
        int(np.sum(nonzero < 0)) if favorable == "negative" else int(np.sum(nonzero > 0))
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = np.median(
        rng.choice(array, (BOOTSTRAP_DRAWS, len(array)), replace=True), axis=1
    )
    return {
        "n": int(len(array)),
        "median": float(np.median(array)),
        "bootstrap_median_ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "n_favorable": count,
        "wilcoxon_one_sided_p": float(wilcoxon(array, alternative=alternative).pvalue),
        "wilcoxon_opposite_tail_p": float(
            wilcoxon(array, alternative=opposite).pvalue
        ),
        "n_non_ties": int(len(nonzero)),
        "tie_excluded_sign_p": (
            float(binomtest(non_ties, len(nonzero), 0.5, alternative="greater").pvalue)
            if len(nonzero)
            else None
        ),
    }


def support_strata(
    support: Sequence[int], *, thresholds: Sequence[int] = SUPPORT_THRESHOLDS
) -> dict[int, np.ndarray]:
    counts = np.asarray(support, int)
    return {int(value): counts >= int(value) for value in thresholds}


def _load_json(path: Path):
    with path.open() as stream:
        return json.load(stream)


def load_patient_frame(result_root: Path = RESULT_ROOT) -> pd.DataFrame:
    """Assemble one row per patient from the frozen per-patient artifacts."""
    subjects = sorted(
        path.stem
        for path in (result_root / "per_subject").glob("*.json")
        if not path.stem.endswith("_predictions")
    )
    rows = []
    for subject in subjects:
        primary = _load_json(result_root / "per_subject" / f"{subject}.json")
        dense = _load_json(
            result_root / "dense_test_sensitivity/per_subject" / f"{subject}.json"
        )
        block = _load_json(
            result_root / "chronology_null/block_shuffle/per_subject" / f"{subject}.json"
        )
        reversal = _load_json(
            result_root / "chronology_null/time_reversal/per_subject" / f"{subject}.json"
        )
        seed_scores = [
            float(run["trained_test_score"]["propagation"])
            for run in primary["recurrent_runs"]
        ]
        truncated = [
            run
            for run in primary["recurrent_runs"]
            if int(run["trace"]["best_nested_epoch"]) < 0
        ]
        rows.append(
            {
                "subject": subject,
                "dataset": primary["dataset"],
                "n_formal_test_targets": int(primary["n_formal_test_targets"]),
                "n_dense_test_targets": int(dense["n_dense_test_targets"]),
                "trained_rnn_propagation": float(
                    primary["trained_recurrent_median_test_score"]["propagation"]
                ),
                "ewma_propagation": float(primary["ewma_test_score"]["propagation"]),
                "static_propagation": float(primary["static_test_score"]["propagation"]),
                "trained_rnn_minus_ewma_propagation": float(
                    primary["trained_rnn_minus_ewma"]["propagation"]
                ),
                "trained_rnn_minus_static_propagation": float(
                    primary["trained_recurrent_median_test_score"]["propagation"]
                    - primary["static_test_score"]["propagation"]
                ),
                "ewma_minus_static_propagation": float(
                    primary["ewma_test_score"]["propagation"]
                    - primary["static_test_score"]["propagation"]
                ),
                "dense_rnn_minus_ewma_propagation": float(
                    dense["dense_test_rnn_minus_ewma"]["propagation"]
                ),
                "block_true_minus_null_propagation": float(
                    block["true_minus_null_gain"]["propagation"]
                ),
                "reversal_true_minus_null_propagation": float(
                    reversal["true_minus_reversal_gain"]["propagation"]
                ),
                "seed_score_sd": float(np.std(seed_scores, ddof=1)),
                "n_seeds_at_minimum_budget": int(len(truncated)),
            }
        )
    return pd.DataFrame(rows)


def _stratified(frame: pd.DataFrame, column: str, *, favorable: str) -> dict:
    strata = support_strata(frame["n_formal_test_targets"])
    output = {}
    for threshold, mask in strata.items():
        if not np.any(mask):
            continue
        output[f"min_formal_windows_{threshold}"] = directional_summary(
            frame.loc[mask, column], favorable=favorable
        )
    return output


def build_state(result_root: Path = RESULT_ROOT) -> tuple[dict, pd.DataFrame]:
    frame = load_patient_frame(result_root)
    frozen = _load_json(result_root / "STATEFUL_TEST_STATE.json")
    primary = directional_summary(
        frame["trained_rnn_minus_ewma_propagation"], favorable="negative"
    )
    reproduces = (
        primary["n"] == frozen["trained_primary_propagation"]["n"]
        and primary["median"]
        == frozen["trained_primary_propagation"]["median_rnn_minus_ewma"]
        and primary["n_favorable"]
        == frozen["trained_primary_propagation"]["n_rnn_better"]
        and primary["wilcoxon_one_sided_p"]
        == frozen["trained_primary_propagation"]["wilcoxon_one_sided_less_p"]
    )
    truncated = frame.loc[frame["n_seeds_at_minimum_budget"] > 0, "subject"].tolist()
    untruncated = ~frame["subject"].isin(truncated)
    return {
        "contract": CONTRACT,
        "status": "DERIVED_ACCEPTANCE_COMPLETE"
        if len(frame) == 34 and reproduces
        else "INCOMPLETE",
        "derivation_only_no_training": True,
        "n_patients": int(len(frame)),
        "reproduces_frozen_primary_endpoint": bool(reproduces),
        "scientific_adjudication": {
            "status": SCIENTIFIC_ADJUDICATION,
            "established": [
                "stable_repertoire_has_short_range_cross_event_state",
                "trained_state_uses_recent_event_history",
                "trained_recurrent_model_beats_fixed_repertoire",
            ],
            "not_established": [
                "increment_beyond_leaky_recency_observer",
                "event_innovation_predicts_state_update",
                "activity_dependent_network_shaping",
                "causal_plasticity",
            ],
            "successor_contracts": {
                "v2_7": "repair_only_early_stopping_rerun",
                "v3_0": "event_innovation_low_rank_state_update_test",
            },
        },
        "score_convention": "propagation score is an error; lower is better, so a negative contrast favours the first model",
        "comparisons": {
            "trained_rnn_minus_ewma_formal": primary,
            "trained_rnn_minus_static_formal": directional_summary(
                frame["trained_rnn_minus_static_propagation"], favorable="negative"
            ),
            "ewma_minus_static_formal": directional_summary(
                frame["ewma_minus_static_propagation"], favorable="negative"
            ),
            "trained_rnn_minus_ewma_dense": directional_summary(
                frame["dense_rnn_minus_ewma_propagation"], favorable="negative"
            ),
        },
        "support_strata": {
            "trained_rnn_minus_ewma_formal": _stratified(
                frame, "trained_rnn_minus_ewma_propagation", favorable="negative"
            ),
            "trained_rnn_minus_static_formal": _stratified(
                frame, "trained_rnn_minus_static_propagation", favorable="negative"
            ),
            "n_formal_test_targets": {
                "min": int(frame["n_formal_test_targets"].min()),
                "median": float(frame["n_formal_test_targets"].median()),
                "max": int(frame["n_formal_test_targets"].max()),
                "total": int(frame["n_formal_test_targets"].sum()),
            },
        },
        "dataset_strata": {
            dataset: {
                "n": int(np.sum(frame["dataset"] == dataset)),
                "formal_rnn_minus_ewma_median": float(
                    frame.loc[
                        frame["dataset"] == dataset, "trained_rnn_minus_ewma_propagation"
                    ].median()
                ),
                "dense_rnn_minus_ewma_median": float(
                    frame.loc[
                        frame["dataset"] == dataset, "dense_rnn_minus_ewma_propagation"
                    ].median()
                ),
            }
            for dataset in sorted(frame["dataset"].unique())
        },
        "chronology_nulls_both_tails": {
            "source_coherent_block_shuffle": directional_summary(
                frame["block_true_minus_null_propagation"], favorable="negative"
            ),
            "source_level_time_reversal": directional_summary(
                frame["reversal_true_minus_null_propagation"], favorable="negative"
            ),
        },
        "seed_dispersion": {
            "median_within_patient_seed_sd": float(frame["seed_score_sd"].median()),
            "max_within_patient_seed_sd": float(frame["seed_score_sd"].max()),
            "cohort_median_rnn_minus_ewma_abs": float(abs(primary["median"])),
            "effect_smaller_than_median_seed_sd": bool(
                abs(primary["median"]) < frame["seed_score_sd"].median()
            ),
        },
        "training_budget_audit": {
            "criterion": "early stopping counts staleness against the epoch -1 static initialization, so a run that never beats it stops at the minimum budget",
            "n_runs_at_minimum_budget": int(frame["n_seeds_at_minimum_budget"].sum()),
            "n_runs_total": int(3 * len(frame)),
            "patients_affected": sorted(truncated),
            "trained_rnn_minus_ewma_excluding_affected_patients": directional_summary(
                frame.loc[untruncated, "trained_rnn_minus_ewma_propagation"],
                favorable="negative",
            ),
            "trained_rnn_minus_static_excluding_affected_patients": directional_summary(
                frame.loc[untruncated, "trained_rnn_minus_static_propagation"],
                favorable="negative",
            ),
        },
        "frozen_inputs_sha256": {
            name: sha256(result_root / relative)
            for name, relative in (
                ("primary_test_state", "STATEFUL_TEST_STATE.json"),
                ("frozen_validation_state", "validation_screen/FROZEN_VALIDATION_STATE.json"),
                ("block_null_state", "chronology_null/block_shuffle/BLOCK_NULL_STATE.json"),
                ("time_reversal_state", "chronology_null/time_reversal/TIME_REVERSAL_STATE.json"),
                ("dense_test_state", "dense_test_sensitivity/DENSE_TEST_STATE.json"),
                ("state_reset_state", "state_reset_ablation/STATE_RESET_STATE.json"),
                ("memory_curve_state", "memory_curve/MEMORY_CURVE_STATE.json"),
                ("h40_state", "h40_sensitivity/H40_STATE.json"),
            )
        },
        "runner_sha256": sha256(Path(__file__)),
        "old_heldout20_entered": False,
    }, frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    args = parser.parse_args()
    state, frame = build_state(args.result_root)
    destination = args.result_root / "acceptance"
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "patient_summary.csv", index=False)
    with (destination / "ACCEPTANCE_STATE.json").open("w") as stream:
        json.dump(jsonable(state), stream, indent=2, sort_keys=True)
    print(json.dumps(jsonable(state), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
