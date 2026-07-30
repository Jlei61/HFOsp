#!/usr/bin/env python3
"""Summarize the target-sealed Topic 5 architecture ladder patient-first."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
OLD = (
    ROOT
    / "results/topic5_interictal_rank_distribution/runs/"
    "formal_multiseed_20260725_v1"
)
LOW_RANK = (
    ROOT
    / "results/topic5_low_rank_dynamics/runs/"
    "low_rank_leaky_multiseed_20260725_v1/"
    "all_seed_rank_subject_summary.csv"
)
SEEDS = (20260725, 20260726, 20260727)
EXPECTED_SUBJECTS = 34
FROZEN_LOW_RANKS = (0, 1, 2, 4)


def bootstrap_median(values: np.ndarray, *, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, float)
    draw = rng.choice(values, (10000, len(values)), replace=True)
    medians = np.median(draw, axis=1)
    return tuple(np.quantile(medians, [0.025, 0.975]).tolist())


def paired_summary(values: pd.Series, *, seed: int) -> dict:
    x = values.dropna().to_numpy(float)
    if not len(x):
        return {
            "n_patients": 0,
            "median_gain": np.nan,
            "ci95": [np.nan, np.nan],
            "n_positive": 0,
            "wilcoxon_greater_p": np.nan,
        }
    try:
        p = float(wilcoxon(x, alternative="greater").pvalue)
    except ValueError:
        p = 1.0
    return {
        "n_patients": int(len(x)),
        "median_gain": float(np.median(x)),
        "ci95": list(bootstrap_median(x, seed=seed)),
        "n_positive": int(np.count_nonzero(x > 0)),
        "wilcoxon_greater_p": p,
    }


def load_new(root: Path) -> pd.DataFrame:
    rows = []
    for path in root.rglob("heldout_metrics.csv"):
        frame = pd.read_csv(path)
        if len(frame) != 1:
            raise RuntimeError(f"{path}: expected one metric row")
        rows.append(frame)
    if not rows:
        raise RuntimeError("no new architecture metrics found")
    out = pd.concat(rows, ignore_index=True)
    expected = EXPECTED_SUBJECTS * len(SEEDS) * 2
    ordinary = out.loc[~out.rank_shuffle.astype(bool)]
    if len(ordinary) != expected:
        raise RuntimeError(
            f"new architecture ladder incomplete: {len(ordinary)}/{expected}"
        )
    return out


def load_old() -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        for path in (OLD / f"seed_{seed}").glob("*/heldout_metrics.csv"):
            frame = pd.read_csv(path)
            rows.append(
                frame.loc[
                    frame.control.isin(
                        [
                            "static_contact_hazard",
                            "unordered_prefix",
                            "last_set_first_order",
                            "full_history_gru",
                            "rank_shuffle_gru",
                        ]
                    )
                ]
            )
    out = pd.concat(rows, ignore_index=True)
    expected = EXPECTED_SUBJECTS * len(SEEDS) * 5
    if len(out) != expected:
        raise RuntimeError(f"accepted architecture artifacts incomplete: {len(out)}")
    return out


def load_low_rank() -> pd.DataFrame:
    frame = pd.read_csv(LOW_RANK)
    frame = frame.loc[frame.recurrent_rank.isin(FROZEN_LOW_RANKS)].copy()
    frame["control"] = frame.recurrent_rank.map(
        lambda value: f"low_rank_r{int(value)}"
    )
    frame["architecture"] = "low_rank_leaky_rnn"
    frame["rank_shuffle"] = False
    if len(frame) != EXPECTED_SUBJECTS * len(SEEDS) * len(FROZEN_LOW_RANKS):
        raise RuntimeError("low-rank artifact denominator drifted")
    return frame


def load_selected_shuffle(root: Path) -> pd.DataFrame:
    rows = []
    for path in root.rglob("heldout_metrics.csv"):
        frame = pd.read_csv(path)
        if len(frame) != 1:
            raise RuntimeError(f"{path}: expected one shuffled metric row")
        rows.append(frame)
    if not rows:
        raise RuntimeError("no selected-architecture shuffle metrics found")
    out = pd.concat(rows, ignore_index=True)
    expected = EXPECTED_SUBJECTS * len(SEEDS)
    if len(out) != expected or not np.all(out.rank_shuffle.astype(bool)):
        raise RuntimeError(
            f"selected shuffle incomplete or mislabeled: {len(out)}/{expected}"
        )
    if out.control.nunique() != 1:
        raise RuntimeError("selected shuffle root contains multiple controls")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--shuffle-root", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT / "results/topic5_ordered_history_architecture_audit/analysis"
        ),
    )
    args = parser.parse_args()
    formal_root = (
        args.formal_root
        if args.formal_root.is_absolute()
        else ROOT / args.formal_root
    )
    shuffle_root = (
        None
        if args.shuffle_root is None
        else args.shuffle_root
        if args.shuffle_root.is_absolute()
        else ROOT / args.shuffle_root
    )
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)

    new = load_new(formal_root)
    selected_shuffle = (
        load_selected_shuffle(shuffle_root)
        if shuffle_root is not None
        else None
    )
    old = load_old()
    low_rank = load_low_rank()
    columns = [
        "subject",
        "dataset",
        "seed",
        "control",
        "heldout_event_nll",
        "participation_mae",
        "rank_wasserstein",
        "precedence_mae",
    ]
    all_seed = pd.concat(
        [
            old.reindex(columns=columns),
            new.reindex(columns=columns),
            low_rank.reindex(columns=columns),
            *(
                [selected_shuffle.reindex(columns=columns)]
                if selected_shuffle is not None
                else []
            ),
        ],
        ignore_index=True,
    )
    all_seed.to_csv(output / "all_seed_architecture_metrics.csv", index=False)
    collapsed = (
        all_seed.groupby(["subject", "dataset", "control"], as_index=False)
        .median(numeric_only=True)
        .drop(columns=["seed"], errors="ignore")
    )
    counts = all_seed.groupby(["subject", "control"]).seed.nunique()
    if not np.all(counts == 3):
        raise RuntimeError("not every patient/control has three seeds")
    collapsed.to_csv(
        output / "patient_seed_collapsed_architecture_metrics.csv", index=False
    )
    nll = collapsed.pivot(
        index="subject", columns="control", values="heldout_event_nll"
    )
    candidates = [
        "linear_state",
        "vanilla_rnn",
        "full_history_gru",
        *[f"low_rank_r{rank}" for rank in FROZEN_LOW_RANKS],
    ]
    comparison_rows = []
    comparison_summary = {}
    for index, candidate in enumerate(candidates):
        for reference_index, reference in enumerate(
            [
                "static_contact_hazard",
                "last_set_first_order",
                "unordered_prefix",
            ]
        ):
            gain = nll[reference] - nll[candidate]
            summary = paired_summary(
                gain, seed=20260729 + index * 10 + reference_index
            )
            comparison_summary[f"{candidate}_vs_{reference}"] = summary
            for subject, value in gain.items():
                comparison_rows.append(
                    {
                        "subject": subject,
                        "candidate": candidate,
                        "reference": reference,
                        "nll_gain_reference_minus_candidate": float(value),
                    }
                )
    gru_order_gain = nll["rank_shuffle_gru"] - nll["full_history_gru"]
    comparison_summary["full_history_gru_vs_rank_shuffle_gru"] = paired_summary(
        gru_order_gain, seed=20260829
    )
    for subject, value in gru_order_gain.items():
        comparison_rows.append(
            {
                "subject": subject,
                "candidate": "full_history_gru",
                "reference": "rank_shuffle_gru",
                "nll_gain_reference_minus_candidate": float(value),
            }
        )
    # The target-facing nongated model is selected from this architecture
    # ladder. Protect the unordered-prefix family against selecting the
    # largest observed median by applying joint patient sign flips and the
    # maximum statistic across recurrent families.
    gain_matrix = np.column_stack(
        [
            (nll["unordered_prefix"] - nll[candidate]).to_numpy(float)
            for candidate in candidates
        ]
    )
    observed_medians = np.median(gain_matrix, axis=0)
    rng = np.random.default_rng(20261729)
    maximum_null = []
    remaining = 50000
    while remaining:
        current = min(5000, remaining)
        signs = rng.choice(
            np.asarray([-1.0, 1.0]), size=(current, gain_matrix.shape[0])
        )
        null_medians = np.median(
            signs[:, :, None] * gain_matrix[None, :, :], axis=1
        )
        maximum_null.append(np.max(null_medians, axis=1))
        remaining -= current
    maximum_null = np.concatenate(maximum_null)
    for candidate, observed in zip(candidates, observed_medians):
        comparison_summary[f"{candidate}_vs_unordered_prefix"][
            "selection_corrected_maxT_p"
        ] = float(
            (1 + np.count_nonzero(maximum_null >= observed))
            / (len(maximum_null) + 1)
        )
    pd.DataFrame(comparison_rows).to_csv(
        output / "patient_paired_nll_gains.csv", index=False
    )

    nongru = [
        "linear_state",
        "vanilla_rnn",
        *[f"low_rank_r{rank}" for rank in FROZEN_LOW_RANKS],
    ]
    selection_table = pd.DataFrame(
        [
            {
                "candidate": candidate,
                **comparison_summary[f"{candidate}_vs_unordered_prefix"],
                "cohort_median_nll": float(np.median(nll[candidate])),
            }
            for candidate in nongru
        ]
    ).sort_values(
        ["median_gain", "cohort_median_nll"],
        ascending=[False, True],
    )
    selection_table.to_csv(
        output / "target_blind_non_gru_selection.csv", index=False
    )
    selected = str(selection_table.iloc[0].candidate)
    if selected_shuffle is not None:
        shuffled_control = str(selected_shuffle.control.iloc[0])
        expected_control = f"{selected}_rank_shuffle"
        if shuffled_control != expected_control:
            raise RuntimeError(
                "target-blind selected model and matched shuffle disagree: "
                f"{selected} vs {shuffled_control}"
            )
        selected_order_gain = nll[shuffled_control] - nll[selected]
        comparison_summary[
            f"{selected}_vs_matched_within_event_rank_shuffle"
        ] = paired_summary(selected_order_gain, seed=20260929)
        for subject, value in selected_order_gain.items():
            comparison_rows.append(
                {
                    "subject": subject,
                    "candidate": selected,
                    "reference": shuffled_control,
                    "nll_gain_reference_minus_candidate": float(value),
                }
            )
        pd.DataFrame(comparison_rows).to_csv(
            output / "patient_paired_nll_gains.csv", index=False
        )
    status = {
        "contract": "topic5_ordered_history_architecture_audit_v0_1",
        "status": (
            "TARGET_SEALED_ARCHITECTURE_AND_MATCHED_SHUFFLE_COMPLETE"
            if selected_shuffle is not None
            else "TARGET_SEALED_ARCHITECTURE_LADDER_COMPLETE"
        ),
        "target_values_read": False,
        "early_ictal_target_arrays_deserialized": False,
        "n_patients": EXPECTED_SUBJECTS,
        "seeds": list(SEEDS),
        "comparisons": comparison_summary,
        "architecture_family_inference": {
            "method": (
                "joint patient sign flips; maximum patient-median NLL gain "
                "across 7 preregistered recurrent families"
            ),
            "n_draws": 50000,
            "reference": "unordered_prefix",
        },
        "target_blind_best_non_gru": {
            "control": selected,
            "selection_rule": (
                "largest patient-median heldout NLL gain over unordered_prefix; "
                "cohort median NLL breaks exact ties"
            ),
            "matched_within_event_rank_shuffle_required": True,
            "matched_within_event_rank_shuffle_complete": (
                selected_shuffle is not None
            ),
        },
        "claim_boundary": (
            "rank-step history is evaluated inside one interictal group event; "
            "no continuous-time or biological slow-state claim is permitted"
        ),
    }
    (output / "ARCHITECTURE_SUMMARY.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
