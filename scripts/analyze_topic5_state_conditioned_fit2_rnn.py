#!/usr/bin/env python
"""Aggregate sharded Fit-2 RNN runs and adjudicate the subject-first Gate 2."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_topic5_state_conditioned_rnn import (
    grouped_ridge,
    history_summary,
    load_gate0,
    subject_arrays,
)


def nested_history_baseline_choices(
    subjects: list[str],
    targets: pd.DataFrame,
    arrays_by_subject: dict,
    cfg: dict,
) -> pd.DataFrame:
    """Choose EWMA vs linear state space without reading the outer labels."""
    max_events = int(cfg["history"]["max_events_per_history"])
    rows = []
    for outer in subjects:
        train = targets[targets.subject != outer].reset_index(drop=True)
        X = history_summary(train, arrays_by_subject, max_events)
        definitions = {
            "ewma": list(range(4, 8)),
            "linear_state_space": list(range(8, 16)),
        }
        candidates = []
        for name, cols in definitions.items():
            _scaler, _model, alpha, inner = grouped_ridge(
                X[:, cols],
                train[cfg["target"]["primary_label_column"]].to_numpy(float),
                train.subject.astype(str).to_numpy(),
                cfg["validation"]["probe_alpha_grid"],
            )
            candidates.append(
                {
                    "outer_subject": outer,
                    "model": name,
                    "alpha": alpha,
                    "inner_mean_mae": float(inner["best_mean_mae"]),
                }
            )
        chosen = min(candidates, key=lambda row: row["inner_mean_mae"])["model"]
        for row in candidates:
            row["selected"] = row["model"] == chosen
            rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_median(values: np.ndarray, draws: int, seed: int):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, float)
    for draw in range(draws):
        samples[draw] = np.median(rng.choice(values, size=len(values), replace=True))
    return (
        float(np.median(values)),
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def _paired_one_sided(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values) & (np.abs(values) > 1e-12)]
    return (
        float(wilcoxon(values, alternative="greater").pvalue)
        if len(values)
        else np.nan
    )


def history_pairing_null(
    paired: pd.DataFrame,
    target: str,
    *,
    draws: int,
    seed: int,
):
    """Shuffle history-derived RNN predictions across seizures within patient."""
    counts = paired[["subject", "seizure_idx"]].drop_duplicates().subject.value_counts()
    eligible = sorted(counts[counts >= 2].index.astype(str))
    if not eligible:
        return {
            "n_eligible_subjects": 0,
            "observed_median_subject_mae": np.nan,
            "null_median_subject_mae": np.nan,
            "empirical_p_observed_lower": np.nan,
        }
    frame = paired[paired.subject.astype(str).isin(eligible)].copy()
    observed_seed = (
        frame.groupby(["subject", "seed"], as_index=False)
        .rnn_absolute_error.mean()
    )
    observed_subject = observed_seed.groupby("subject").rnn_absolute_error.median()
    observed = float(observed_subject.median())
    rng = np.random.default_rng(seed)
    prepared = []
    n_unique = 1
    for subject_name, subject_frame in frame.groupby("subject"):
        seizure_ids = np.sort(subject_frame.seizure_idx.unique())
        n_unique *= math.factorial(len(seizure_ids))
        seed_groups = []
        for seed_value, group in subject_frame.groupby("seed"):
            group = group.set_index("seizure_idx").loc[seizure_ids]
            seed_groups.append(
                (
                    int(seed_value),
                    group.rnn_prediction.to_numpy(float),
                    group[target].to_numpy(float),
                )
            )
        prepared.append((str(subject_name), seizure_ids, seed_groups))

    # Small repeated-seizure cohorts have a finite, auditable permutation
    # space. Enumerate it exactly instead of presenting repeated Monte Carlo
    # draws as additional statistical information.
    exact = n_unique <= 100_000
    if exact:
        patient_permutations = [
            list(itertools.permutations(range(len(seizure_ids))))
            for _subject, seizure_ids, _seed_groups in prepared
        ]
        permutation_draws = itertools.product(*patient_permutations)
        null = np.empty(n_unique, float)
    else:
        permutation_draws = (
            tuple(rng.permutation(len(seizure_ids)) for _, seizure_ids, _ in prepared)
            for _draw in range(draws)
        )
        null = np.empty(draws, float)

    for draw, patient_draw in enumerate(permutation_draws):
        errors = []
        # A seed is not an independent statistical sample. Apply one seizure
        # permutation per patient identically to every seed.
        for (
            subject_name,
            _seizure_ids,
            seed_groups,
        ), permutation in zip(prepared, patient_draw):
            permutation = np.asarray(permutation, int)
            for seed_value, predictions, targets in seed_groups:
                prediction = predictions[permutation]
                errors.append(
                    {
                        "subject": str(subject_name),
                        "seed": int(seed_value),
                        "mae": float(np.mean(np.abs(prediction - targets))),
                    }
                )
        shuffled = pd.DataFrame(errors)
        null[draw] = float(
            shuffled.groupby("subject").mae.median().median()
        )
    tail_count = int(np.sum(null <= observed + 1e-12))
    empirical_p = (
        tail_count / len(null)
        if exact
        else (1 + tail_count) / (len(null) + 1)
    )
    return {
        "n_eligible_subjects": len(eligible),
        "permutation_mode": "exact" if exact else "monte_carlo",
        "n_unique_patient_pairings": int(n_unique),
        "n_null_draws": int(len(null)),
        "observed_median_subject_mae": observed,
        "null_median_subject_mae": float(np.median(null)),
        "empirical_p_observed_lower": float(empirical_p),
    }


def load_shards(shards: list[Path]):
    selected, selections = [], []
    for shard in shards:
        if not (shard / "DONE.json").exists():
            raise RuntimeError(f"incomplete shard: {shard}")
        selected.append(pd.read_csv(shard / "selected_rank_predictions.csv"))
        selections.append(pd.read_csv(shard / "rank_selection_one_se.csv"))
    return pd.concat(selected, ignore_index=True), pd.concat(selections, ignore_index=True)


def pretext_at_selected_rank(shards: list[Path], selection: pd.DataFrame):
    rows = []
    selected = selection[selection.selected.astype(str).str.lower().isin(("true", "1"))]
    shard_by_outer = {}
    for shard in shards:
        manifest = json.loads((shard / "DONE.json").read_text())
        for outer in manifest["outer_subjects"]:
            shard_by_outer[str(outer)] = shard
    for row in selected.itertuples():
        shard = shard_by_outer[str(row.outer_subject)]
        path = (
            shard
            / "checkpoints/primary"
            / str(row.outer_subject)
            / f"rank_{int(row.rank)}"
            / f"seed_{int(row.seed)}"
            / "DONE.json"
        )
        record = json.loads(path.read_text())
        rows.append(
            {
                "subject": str(row.outer_subject),
                "seed": int(row.seed),
                "rank": int(row.rank),
                **record["pretext"],
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/topic5_state_conditioned_predictor_fit2.yaml",
    )
    parser.add_argument("--shards", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "results/topic5_state_conditioned_predictor/fit2_rnn_final_analysis",
    )
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()
    args.config = args.config if args.config.is_absolute() else ROOT / args.config
    args.shards = [path if path.is_absolute() else ROOT / path for path in args.shards]
    args.out = args.out if args.out.is_absolute() else ROOT / args.out
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(args.config.read_text())
    dataset = ROOT / cfg["outputs"]["dataset"]
    subjects, targets = load_gate0(dataset, cfg)
    arrays = {subject: subject_arrays(dataset, subject) for subject in subjects}
    predictions, selection = load_shards(args.shards)
    target = str(cfg["target"]["primary_label_column"])
    keys = ["dataset", "subject", "seizure_idx", "outer_subject", "seed"]

    choices = nested_history_baseline_choices(subjects, targets, arrays, cfg)
    selected_choice = dict(
        choices[choices.selected].set_index("outer_subject").model.astype(str)
    )
    rnn = predictions[predictions.model.astype(str).str.startswith("lr_ei_ct_rnn")]
    baseline = predictions[
        [
            str(row.model) == selected_choice.get(str(row.outer_subject), "")
            for row in predictions.itertuples()
        ]
    ]
    gru = predictions[predictions.model.astype(str) == "matched_gru"]
    static = predictions[predictions.model.astype(str) == "static_scaffold"]
    rnn = rnn[keys + [target, "rank", "prediction", "absolute_error"]].rename(
        columns={
            "rank": "selected_rank",
            "prediction": "rnn_prediction",
            "absolute_error": "rnn_absolute_error",
        }
    )
    baseline = baseline[
        keys + ["model", "prediction", "absolute_error"]
    ].rename(
        columns={
            "model": "selected_history_baseline",
            "prediction": "baseline_prediction",
            "absolute_error": "baseline_absolute_error",
        }
    )
    gru = gru[keys + ["prediction", "absolute_error"]].rename(
        columns={
            "prediction": "gru_prediction",
            "absolute_error": "gru_absolute_error",
        }
    )
    static = static[keys + ["prediction", "absolute_error"]].rename(
        columns={
            "prediction": "static_prediction",
            "absolute_error": "static_absolute_error",
        }
    )
    paired = rnn.merge(baseline, on=keys, validate="one_to_one").merge(
        gru, on=keys, validate="one_to_one"
    ).merge(
        static, on=keys, validate="one_to_one"
    )
    paired["rnn_increment_over_history_baseline"] = (
        paired.baseline_absolute_error - paired.rnn_absolute_error
    )
    paired["rnn_increment_over_gru"] = (
        paired.gru_absolute_error - paired.rnn_absolute_error
    )
    paired["rnn_increment_over_static"] = (
        paired.static_absolute_error - paired.rnn_absolute_error
    )
    paired.to_csv(args.out / "event_seed_predictions.csv", index=False)
    choices.to_csv(args.out / "nested_history_baseline_selection.csv", index=False)
    selection.to_csv(args.out / "rank_selection_one_se.csv", index=False)

    subject_seed = (
        paired.groupby(["subject", "seed"], as_index=False)
        .agg(
            selected_rank=("selected_rank", "first"),
            n_seizures=("seizure_idx", "nunique"),
            rnn_mae=("rnn_absolute_error", "mean"),
            history_baseline_mae=("baseline_absolute_error", "mean"),
            gru_mae=("gru_absolute_error", "mean"),
            static_mae=("static_absolute_error", "mean"),
            rnn_increment_over_history_baseline=(
                "rnn_increment_over_history_baseline",
                "mean",
            ),
            rnn_increment_over_gru=("rnn_increment_over_gru", "mean"),
            rnn_increment_over_static=("rnn_increment_over_static", "mean"),
        )
    )
    subject_seed.to_csv(args.out / "subject_seed_metrics.csv", index=False)
    subject = (
        subject_seed.groupby("subject", as_index=False)
        .agg(
            n_seizures=("n_seizures", "first"),
            selected_rank_median=("selected_rank", "median"),
            rnn_mae=("rnn_mae", "median"),
            history_baseline_mae=("history_baseline_mae", "median"),
            gru_mae=("gru_mae", "median"),
            static_mae=("static_mae", "median"),
            rnn_increment_over_history_baseline=(
                "rnn_increment_over_history_baseline",
                "median",
            ),
            rnn_increment_over_gru=("rnn_increment_over_gru", "median"),
            rnn_increment_over_static=("rnn_increment_over_static", "median"),
        )
    )
    subject.to_csv(args.out / "subject_level_metrics.csv", index=False)

    pretext = pretext_at_selected_rank(args.shards, selection)
    pretext.to_csv(args.out / "selected_rank_pretext_order_control.csv", index=False)
    pretext_subject = (
        pretext.groupby("subject", as_index=False)
        .shuffle_minus_true.median()
    )
    delta = subject.rnn_increment_over_history_baseline.to_numpy(float)
    median_delta, ci_low, ci_high = _bootstrap_median(
        delta, int(cfg["validation"]["bootstrap_draws"]), args.seed
    )
    gru_delta = subject.rnn_increment_over_gru.to_numpy(float)
    gru_median, gru_low, gru_high = _bootstrap_median(
        gru_delta, int(cfg["validation"]["bootstrap_draws"]), args.seed + 1
    )
    static_delta = subject.rnn_increment_over_static.to_numpy(float)
    static_median, static_low, static_high = _bootstrap_median(
        static_delta, int(cfg["validation"]["bootstrap_draws"]), args.seed + 3
    )
    order_values = pretext_subject.shuffle_minus_true.to_numpy(float)
    order_pass = bool(
        np.median(order_values) > 0 and np.sum(order_values > 0) > len(order_values) / 2
    )
    pairing = history_pairing_null(
        paired,
        target,
        draws=int(cfg["validation"]["bootstrap_draws"]),
        seed=args.seed + 2,
    )
    pairing_pass = bool(
        np.isfinite(pairing["empirical_p_observed_lower"])
        and pairing["empirical_p_observed_lower"] < 0.05
    )
    gate2 = bool(
        median_delta > 0
        and ci_low > 0
        and static_median > 0
        and static_low > 0
        and order_pass
        and pairing_pass
    )
    verdict = {
        "contract": cfg["contract"]["name"],
        "n_subjects": int(subject.subject.nunique()),
        "n_seizures": int(
            paired[["subject", "seizure_idx"]].drop_duplicates().shape[0]
        ),
        "n_seeds": int(subject_seed.seed.nunique()),
        "target": target,
        "primary_dynamic_comparator": (
            "outer-training-only selection between EWMA and linear state space"
        ),
        "rnn_minus_history_baseline_mae_improvement_median": median_delta,
        "patient_bootstrap_95ci": [ci_low, ci_high],
        "paired_wilcoxon_improvement_greater_zero_p": _paired_one_sided(delta),
        "n_subjects_rnn_better_history_baseline": int(np.sum(delta > 0)),
        "rnn_minus_gru_mae_improvement_median": gru_median,
        "rnn_minus_gru_patient_bootstrap_95ci": [gru_low, gru_high],
        "rnn_minus_static_mae_improvement_median": static_median,
        "rnn_minus_static_patient_bootstrap_95ci": [static_low, static_high],
        "event_order_shuffle_minus_true_pretext_loss_median": float(
            np.median(order_values)
        ),
        "n_subjects_true_order_better": int(np.sum(order_values > 0)),
        "event_order_pretext_pass": order_pass,
        "history_pairing_null": pairing,
        "history_pairing_null_pass": pairing_pass,
        "history_pairing_null_status": (
            "underpowered"
            if not pairing_pass and pairing["n_eligible_subjects"] < 8
            else "tested"
        ),
        "gate2_pass": gate2,
        "gate2_claim_boundary": (
            "frozen interictal core predicts seizure-conditioned BB150 "
            "scaffold-margin strength; no signed direction or continuous hazard claim"
        ),
    }
    (args.out / "gate2_verdict.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
