#!/usr/bin/env python3
"""Systematic interictal analysis of the Topic 5 low-rank leaky RNN sweep.

This script does not read ictal targets.  It combines patient-level behavioral
metrics, a label-free whole-path diagnostic, seed stability, and diagnostics
of the learned recurrent operators.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_topic5_interictal_rank_distribution import load_records  # noqa: E402


SEEDS = [20260725, 20260726, 20260727]
RANKS = [0, 1, 2, 3, 4]
DATASET_COLORS = {"epilepsiae": "#2166AC", "yuquan": "#B66A2B"}
RANK_COLOR = "#B2182B"
REFERENCE_COLOR = "#303030"
EMPIRICAL_COLOR = "#7F7F7F"

LOWER_IS_BETTER = {
    "heldout_event_nll": True,
    "participation_mae": True,
    "rank_wasserstein": True,
    "precedence_mae": True,
    "precedence_correlation": False,
    "path_sliced_wasserstein": True,
}


def _bootstrap_median_ci(
    values: Iterable[float], seed: int = 20260726, n_boot: int = 20_000
) -> tuple[float, float]:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True)
    medians = np.median(draws, axis=1)
    return tuple(np.quantile(medians, [0.025, 0.975]))


def _safe_wilcoxon(values: Iterable[float]) -> float:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0 or np.allclose(values, 0):
        return np.nan
    return float(wilcoxon(values, alternative="two-sided").pvalue)


def _bh_fdr(values: pd.Series) -> pd.Series:
    x = values.to_numpy(dtype=float)
    out = np.full(x.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(x))
    if len(valid) == 0:
        return pd.Series(out, index=values.index)
    order = valid[np.argsort(x[valid])]
    ranked = x[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out[order] = np.minimum(adjusted, 1.0)
    return pd.Series(out, index=values.index)


def _load_behavior_metrics(
    low_rank_root: Path, full_rank_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    low_frames = []
    for path in sorted(low_rank_root.glob("seed_*/rank_*/*/heldout_metrics.csv")):
        frame = pd.read_csv(path)
        frame = frame.loc[frame["control"].str.startswith("low_rank_leaky")]
        low_frames.append(frame)
    low = pd.concat(low_frames, ignore_index=True)
    expected = len(SEEDS) * len(RANKS) * 34
    if len(low) != expected:
        raise RuntimeError(f"Expected {expected} low-rank metric rows, found {len(low)}")

    full_frames = []
    for path in sorted(full_rank_root.glob("seed_*/*/heldout_metrics.csv")):
        frame = pd.read_csv(path)
        full_frames.append(
            frame.loc[
                frame["control"].isin(
                    ["full_history_gru", "empirical_rank_distribution"]
                )
            ]
        )
    full = pd.concat(full_frames, ignore_index=True)
    return low, full


def _event_feature_matrix(group_ids: np.ndarray, group_count: np.ndarray) -> np.ndarray:
    group_ids = np.asarray(group_ids)
    group_count = np.asarray(group_count)
    participating = group_ids >= 0
    denominator = np.maximum(group_count - 1, 1)[:, None]
    normalized_rank = np.where(participating, group_ids / denominator, 0.0)
    return np.concatenate(
        [participating.astype(np.float32), normalized_rank.astype(np.float32)],
        axis=1,
    )


def _projected_quantiles(
    values: np.ndarray,
    directions: np.ndarray,
    *,
    seed: int,
    max_events: int = 1_000,
    n_quantiles: int = 200,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if len(values) > max_events:
        values = values[rng.choice(len(values), max_events, replace=False)]
    quantiles = np.linspace(0.0, 1.0, n_quantiles)
    return np.quantile(values @ directions, quantiles, axis=0)


def _path_diagnostics(
    low_rank_root: Path,
    full_rank_root: Path,
    dataset_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label-free whole-event comparison using fixed random projections."""
    records = load_records(dataset_root)
    low_rows = []
    full_rows = []
    for subject_index, (subject, record) in enumerate(records.items()):
        observed = _event_feature_matrix(
            record.group_ids[record.eval_indices],
            record.group_count[record.eval_indices],
        )
        empirical = _event_feature_matrix(
            record.group_ids[record.train_indices],
            record.group_count[record.train_indices],
        )
        split_at = max(1, len(empirical) // 2)
        rng = np.random.default_rng(91_000 + subject_index)
        directions = rng.normal(size=(observed.shape[1], 48))
        directions /= np.linalg.norm(directions, axis=0, keepdims=True)
        observed_q = _projected_quantiles(
            observed, directions, seed=92_000 + subject_index
        )
        empirical_q = _projected_quantiles(
            empirical, directions, seed=93_000 + subject_index
        )
        first_q = _projected_quantiles(
            empirical[:split_at], directions, seed=94_000 + subject_index
        )
        second_q = _projected_quantiles(
            empirical[split_at:], directions, seed=95_000 + subject_index
        )
        empirical_distance = float(np.mean(np.abs(empirical_q - observed_q)))
        split_half_distance = float(np.mean(np.abs(first_q - second_q)))

        for rank in RANKS:
            for seed in SEEDS:
                generated = np.load(
                    low_rank_root
                    / f"seed_{seed}"
                    / f"rank_{rank}"
                    / subject
                    / "free_rollouts.npz"
                )
                generated_features = _event_feature_matrix(
                    generated["event_group_ids"], generated["event_group_count"]
                )
                generated_q = _projected_quantiles(
                    generated_features,
                    directions,
                    seed=seed + 1009 * rank + 17 * subject_index,
                )
                distance = float(np.mean(np.abs(generated_q - observed_q)))
                low_rows.append(
                    {
                        "subject": subject,
                        "dataset": record.dataset,
                        "seed": seed,
                        "recurrent_rank": rank,
                        "path_sliced_wasserstein": distance,
                        "path_empirical_distance": empirical_distance,
                        "path_split_half_distance": split_half_distance,
                        "path_excess": (
                            distance - empirical_distance - split_half_distance
                        ),
                    }
                )

        for seed in SEEDS:
            generated = np.load(
                full_rank_root
                / f"seed_{seed}"
                / subject
                / "full_history_gru_free_rollouts.npz"
            )
            generated_features = _event_feature_matrix(
                generated["event_group_ids"], generated["event_group_count"]
            )
            generated_q = _projected_quantiles(
                generated_features,
                directions,
                seed=seed + 17 * subject_index,
            )
            distance = float(np.mean(np.abs(generated_q - observed_q)))
            full_rows.append(
                {
                    "subject": subject,
                    "dataset": record.dataset,
                    "seed": seed,
                    "control": "full_history_gru",
                    "path_sliced_wasserstein": distance,
                    "path_empirical_distance": empirical_distance,
                    "path_split_half_distance": split_half_distance,
                    "path_excess": (
                        distance - empirical_distance - split_half_distance
                    ),
                }
            )
    return pd.DataFrame(low_rows), pd.DataFrame(full_rows)


def _hidden_effective_dimension(path: Path) -> float:
    artifact = np.load(path)
    hidden = artifact["hidden_states"]
    mask = artifact["state_mask"]
    valid = hidden[np.broadcast_to(mask[..., None], hidden.shape)].reshape(
        -1, hidden.shape[-1]
    )
    valid = valid.astype(float)
    valid -= valid.mean(axis=0, keepdims=True)
    eigenvalues = np.linalg.eigvalsh(
        valid.T @ valid / max(len(valid) - 1, 1)
    )
    eigenvalues = np.maximum(eigenvalues, 0)
    return float(
        eigenvalues.sum() ** 2 / (np.sum(eigenvalues**2) + 1e-12)
    )


def _mode_diagnostics(low_rank_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(low_rank_root.glob("seed_*/rank_*/*/mode_artifacts.npz")):
        artifact = np.load(path, allow_pickle=True)
        rank = int(artifact["recurrent_rank"])
        seed = int(next(part for part in path.parts if part.startswith("seed_")).split("_")[1])
        subject = path.parent.name
        alpha = float(artifact["alpha"])
        decay = artifact["decay"].astype(float)
        real_dimension = _hidden_effective_dimension(
            path.parent / "real_event_trajectories.npz"
        )
        generated_dimension = _hidden_effective_dimension(
            path.parent / "generated_event_trajectories.npz"
        )
        row = {
            "subject": subject,
            "seed": seed,
            "recurrent_rank": rank,
            "alpha": alpha,
            "median_decay": float(np.median(decay)),
            "real_hidden_effective_dimension": real_dimension,
            "generated_hidden_effective_dimension": generated_dimension,
            "operator_frobenius_norm": np.nan,
            "operator_spectral_norm": np.nan,
            "operator_effective_rank": np.nan,
            "linearized_spectral_radius": float(
                np.max(np.abs((1.0 - alpha) - alpha * decay))
            ),
            "median_mode_to_diagonal_drive_ratio": np.nan,
            "median_mode_drive_energy_fraction": np.nan,
        }
        if rank:
            u = artifact["mode_u"].astype(float)
            v = artifact["mode_v"].astype(float)
            operator = u @ v.T / np.sqrt(float(rank))
            singular = np.linalg.svd(operator, compute_uv=False)
            jacobian = (
                (1.0 - alpha) * np.eye(len(decay))
                + alpha * (-np.diag(decay) + operator)
            )
            trajectory = np.load(path.parent / "real_event_trajectories.npz")
            hidden = trajectory["hidden_states"]
            mask = trajectory["state_mask"]
            valid = hidden[
                np.broadcast_to(mask[..., None], hidden.shape)
            ].reshape(-1, hidden.shape[-1]).astype(float)
            mode_drive = (valid @ v) @ u.T / np.sqrt(float(rank))
            diagonal_drive = -valid * decay
            mode_norm = np.linalg.norm(mode_drive, axis=1)
            diagonal_norm = np.linalg.norm(diagonal_drive, axis=1)
            row.update(
                {
                    "operator_frobenius_norm": float(np.linalg.norm(operator)),
                    "operator_spectral_norm": float(singular[0]),
                    "operator_effective_rank": float(
                        singular.sum() ** 2
                        / (np.sum(singular**2) + 1e-12)
                    ),
                    "linearized_spectral_radius": float(
                        np.max(np.abs(np.linalg.eigvals(jacobian)))
                    ),
                    "median_mode_to_diagonal_drive_ratio": float(
                        np.median(mode_norm / (diagonal_norm + 1e-9))
                    ),
                    "median_mode_drive_energy_fraction": float(
                        np.median(
                            mode_norm**2
                            / (mode_norm**2 + diagonal_norm**2 + 1e-12)
                        )
                    ),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _subspace_basis(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values -= values.mean(axis=0, keepdims=True)
    left, singular, _ = np.linalg.svd(values, full_matrices=False)
    if len(singular) == 0 or singular[0] == 0:
        return left[:, :0]
    tolerance = max(values.shape) * np.finfo(float).eps * singular[0]
    return left[:, singular > tolerance]


def _subspace_similarity(
    left: np.ndarray, right: np.ndarray
) -> tuple[float, float]:
    q_left = _subspace_basis(left)
    q_right = _subspace_basis(right)
    if q_left.shape[1] == 0 or q_right.shape[1] == 0:
        return np.nan, np.nan
    singular = np.linalg.svd(q_left.T @ q_right, compute_uv=False)
    shared_rank = min(q_left.shape[1], q_right.shape[1])
    observed = float(np.mean(singular[:shared_rank] ** 2))
    ambient = max(left.shape[0] - 1, 1)
    chance = min(shared_rank / ambient, 0.999999)
    adjusted = float((observed - chance) / (1.0 - chance))
    return observed, adjusted


def _loading_stability(low_rank_root: Path) -> pd.DataFrame:
    rows = []
    for rank in RANKS[1:]:
        subjects = sorted(
            path.parent.name
            for path in (low_rank_root / f"seed_{SEEDS[0]}" / f"rank_{rank}").glob(
                "*/mode_artifacts.npz"
            )
        )
        for subject in subjects:
            artifacts = [
                np.load(
                    low_rank_root
                    / f"seed_{seed}"
                    / f"rank_{rank}"
                    / subject
                    / "mode_artifacts.npz",
                    allow_pickle=True,
                )
                for seed in SEEDS
            ]
            if not all(
                np.array_equal(artifacts[0]["contact_names"], item["contact_names"])
                for item in artifacts[1:]
            ):
                raise RuntimeError(f"Contact ordering changed across seeds: {subject}")
            for field in ["u_output_loading", "v_output_loading"]:
                observed = []
                adjusted = []
                for left in range(len(artifacts)):
                    for right in range(left + 1, len(artifacts)):
                        raw, corrected = _subspace_similarity(
                            artifacts[left][field], artifacts[right][field]
                        )
                        observed.append(raw)
                        adjusted.append(corrected)
                rows.append(
                    {
                        "subject": subject,
                        "recurrent_rank": rank,
                        "loading_field": field,
                        "median_pairwise_subspace_similarity": float(
                            np.nanmedian(observed)
                        ),
                        "median_chance_adjusted_subspace_similarity": float(
                            np.nanmedian(adjusted)
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _patient_tables(
    low: pd.DataFrame,
    full: pd.DataFrame,
    low_path: pd.DataFrame,
    full_path: pd.DataFrame,
    existing_patient_summary: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    patient = (
        low.groupby(["subject", "dataset", "recurrent_rank"], as_index=False)
        .median(numeric_only=True)
    )
    path_patient = (
        low_path.groupby(["subject", "dataset", "recurrent_rank"], as_index=False)
        .median(numeric_only=True)
    )
    patient = patient.merge(
        path_patient[
            [
                "subject",
                "dataset",
                "recurrent_rank",
                "path_sliced_wasserstein",
                "path_empirical_distance",
                "path_split_half_distance",
                "path_excess",
            ]
        ],
        on=["subject", "dataset", "recurrent_rank"],
        validate="one_to_one",
    )
    excess = pd.read_csv(existing_patient_summary)
    patient = patient.merge(
        excess[
            [
                "subject",
                "dataset",
                "recurrent_rank",
                "participation_excess",
                "rank_wasserstein_excess",
                "precedence_excess",
            ]
        ],
        on=["subject", "dataset", "recurrent_rank"],
        validate="one_to_one",
    )
    patient["participation_within_variability"] = (
        patient["participation_excess"] <= 0
    )
    patient["rank_within_variability"] = (
        patient["rank_wasserstein_excess"] <= 0
    )
    patient["precedence_within_variability"] = (
        patient["precedence_excess"] <= 0
    )
    patient["all_three_within_variability"] = patient[
        [
            "participation_within_variability",
            "rank_within_variability",
            "precedence_within_variability",
        ]
    ].all(axis=1)
    patient["path_within_variability"] = patient["path_excess"] <= 0

    full_patient = (
        full.groupby(["subject", "dataset", "control"], as_index=False)
        .median(numeric_only=True)
    )
    full_path_patient = (
        full_path.groupby(["subject", "dataset", "control"], as_index=False)
        .median(numeric_only=True)
    )
    full_patient = full_patient.merge(
        full_path_patient[
            ["subject", "dataset", "control", "path_sliced_wasserstein", "path_excess"]
        ],
        on=["subject", "dataset", "control"],
        how="left",
        validate="one_to_one",
    )
    return patient, full_patient


def _rank_summary(patient: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rank, frame in patient.groupby("recurrent_rank"):
        row: dict[str, object] = {
            "recurrent_rank": int(rank),
            "n_patients": int(len(frame)),
        }
        for metric in LOWER_IS_BETTER:
            values = frame[metric]
            lo, hi = _bootstrap_median_ci(values, seed=20260726 + int(rank))
            row[f"median_{metric}"] = float(values.median())
            row[f"{metric}_ci95_low"] = lo
            row[f"{metric}_ci95_high"] = hi
        for metric in [
            "participation_excess",
            "rank_wasserstein_excess",
            "precedence_excess",
            "path_excess",
        ]:
            values = frame[metric]
            lo, hi = _bootstrap_median_ci(values, seed=20260800 + int(rank))
            row[f"median_{metric}"] = float(values.median())
            row[f"{metric}_ci95_low"] = lo
            row[f"{metric}_ci95_high"] = hi
        row.update(
            {
                "n_participation_within_variability": int(
                    frame["participation_within_variability"].sum()
                ),
                "n_rank_within_variability": int(
                    frame["rank_within_variability"].sum()
                ),
                "n_precedence_within_variability": int(
                    frame["precedence_within_variability"].sum()
                ),
                "n_all_three_within_variability": int(
                    frame["all_three_within_variability"].sum()
                ),
                "n_path_within_variability": int(
                    frame["path_within_variability"].sum()
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("recurrent_rank")


def _paired_rank_comparisons(
    patient: pd.DataFrame, full_patient: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    vs_zero = []
    vs_full = []
    for metric, lower_is_better in LOWER_IS_BETTER.items():
        wide = patient.pivot(
            index=["subject", "dataset"], columns="recurrent_rank", values=metric
        )
        for rank in RANKS[1:]:
            benefit = (
                wide[0] - wide[rank]
                if lower_is_better
                else wide[rank] - wide[0]
            )
            lo, hi = _bootstrap_median_ci(benefit, seed=203000 + rank)
            vs_zero.append(
                {
                    "metric": metric,
                    "recurrent_rank": rank,
                    "positive_means_rank_better_than_rank0": True,
                    "median_benefit": float(benefit.median()),
                    "bootstrap_ci95_low": lo,
                    "bootstrap_ci95_high": hi,
                    "n_rank_better": int((benefit > 0).sum()),
                    "n_patients": int(len(benefit)),
                    "wilcoxon_p_raw": _safe_wilcoxon(benefit),
                }
            )
        full_values = full_patient.loc[
            full_patient["control"] == "full_history_gru"
        ].set_index(["subject", "dataset"])[metric]
        for rank in RANKS:
            benefit = (
                full_values - wide[rank]
                if lower_is_better
                else wide[rank] - full_values
            )
            lo, hi = _bootstrap_median_ci(benefit, seed=204000 + rank)
            vs_full.append(
                {
                    "metric": metric,
                    "recurrent_rank": rank,
                    "positive_means_low_rank_better_than_full_gru": True,
                    "median_benefit": float(benefit.median()),
                    "bootstrap_ci95_low": lo,
                    "bootstrap_ci95_high": hi,
                    "n_low_rank_better": int((benefit > 0).sum()),
                    "n_patients": int(len(benefit)),
                    "wilcoxon_p_raw": _safe_wilcoxon(benefit),
                }
            )
    zero_frame = pd.DataFrame(vs_zero)
    full_frame = pd.DataFrame(vs_full)
    zero_frame["wilcoxon_p_fdr"] = _bh_fdr(zero_frame["wilcoxon_p_raw"])
    full_frame["wilcoxon_p_fdr"] = _bh_fdr(full_frame["wilcoxon_p_raw"])
    return zero_frame, full_frame


def _seed_stability(low: pd.DataFrame, low_path: pd.DataFrame) -> pd.DataFrame:
    merged = low.merge(
        low_path[
            ["subject", "seed", "recurrent_rank", "path_sliced_wasserstein"]
        ],
        on=["subject", "seed", "recurrent_rank"],
        validate="one_to_one",
    )
    rows = []
    for rank, frame in merged.groupby("recurrent_rank"):
        for metric in LOWER_IS_BETTER:
            wide = frame.pivot(index="subject", columns="seed", values=metric)
            pairwise = []
            for left, seed_a in enumerate(wide.columns):
                for seed_b in wide.columns[left + 1 :]:
                    pairwise.append(
                        float(spearmanr(wide[seed_a], wide[seed_b]).statistic)
                    )
            rows.append(
                {
                    "recurrent_rank": int(rank),
                    "metric": metric,
                    "median_pairwise_seed_spearman": float(np.median(pairwise)),
                    "min_pairwise_seed_spearman": float(np.min(pairwise)),
                    "median_within_patient_sd": float(wide.std(axis=1).median()),
                }
            )
    return pd.DataFrame(rows)


def _reference_medians(full_patient: pd.DataFrame) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for control, frame in full_patient.groupby("control"):
        result[control] = {}
        for metric in LOWER_IS_BETTER:
            if metric in frame and frame[metric].notna().any():
                result[control][metric] = float(frame[metric].median())
    return result


def _build_summary(
    rank_summary: pd.DataFrame,
    vs_zero: pd.DataFrame,
    mode_patient: pd.DataFrame,
    loading: pd.DataFrame,
    references: dict[str, dict[str, float]],
) -> dict[str, object]:
    zero = rank_summary.loc[rank_summary["recurrent_rank"] == 0].iloc[0]
    positive_rank_improvements = vs_zero.loc[
        vs_zero["metric"].isin(
            [
                "participation_mae",
                "rank_wasserstein",
                "precedence_mae",
                "path_sliced_wasserstein",
            ]
        )
    ]
    any_positive_rank_fdr_better = bool(
        (
            (positive_rank_improvements["median_benefit"] > 0)
            & (positive_rank_improvements["wilcoxon_p_fdr"] < 0.05)
        ).any()
    )
    mode_positive = mode_patient.loc[mode_patient["recurrent_rank"] > 0]
    rank_one_loading = loading.loc[
        (loading["recurrent_rank"] == 1)
        & (loading["loading_field"] == "u_output_loading")
    ]
    return {
        "status": "complete",
        "ictal_target_read": False,
        "n_patients": 34,
        "n_seeds": 3,
        "ranks": RANKS,
        "pre_registered_distribution_sufficient_rank": None,
        "rank0_is_diagonal_leaky_rnn_not_nonrecurrent": True,
        "rank0_results": {
            "n_all_three_metrics_within_empirical_variability": int(
                zero["n_all_three_within_variability"]
            ),
            "n_whole_path_within_empirical_variability": int(
                zero["n_path_within_variability"]
            ),
            "median_rank_wasserstein": float(zero["median_rank_wasserstein"]),
            "median_precedence_correlation": float(
                zero["median_precedence_correlation"]
            ),
            "median_path_sliced_wasserstein": float(
                zero["median_path_sliced_wasserstein"]
            ),
        },
        "positive_low_rank_mode_behavioral_support": any_positive_rank_fdr_better,
        "positive_mode_diagnostics": {
            "median_mode_to_diagonal_drive_ratio_range": [
                float(
                    mode_positive.groupby("recurrent_rank")[
                        "median_mode_to_diagonal_drive_ratio"
                    ].median().min()
                ),
                float(
                    mode_positive.groupby("recurrent_rank")[
                        "median_mode_to_diagonal_drive_ratio"
                    ].median().max()
                ),
            ],
            "median_real_hidden_effective_dimension_range": [
                float(
                    mode_patient.groupby("recurrent_rank")[
                        "real_hidden_effective_dimension"
                    ].median().min()
                ),
                float(
                    mode_patient.groupby("recurrent_rank")[
                        "real_hidden_effective_dimension"
                    ].median().max()
                ),
            ],
            "rank1_u_loading_median_chance_adjusted_seed_similarity": float(
                rank_one_loading[
                    "median_chance_adjusted_subspace_similarity"
                ].median()
            ),
        },
        "full_rank_reference_medians": references.get("full_history_gru", {}),
        "empirical_reference_medians": references.get(
            "empirical_rank_distribution", {}
        ),
        "go_no_go": {
            "engineering": "pass",
            "positive_rank_interictal_reproduction": "no_go",
            "ictal_target_read": "defer",
            "reason": (
                "No r>0 model passed the distribution gate or improved the "
                "primary distribution/path metrics over the diagonal leaky "
                "rank-0 model after multiplicity correction."
            ),
        },
    }


def _patient_points(
    ax: plt.Axes,
    patient: pd.DataFrame,
    metric: str,
    ylabel: str,
    *,
    reference: float | None = None,
    empirical: float | None = None,
) -> None:
    rng = np.random.default_rng(20260726)
    for rank in RANKS:
        frame = patient.loc[patient["recurrent_rank"] == rank]
        jitter = rng.uniform(-0.12, 0.12, len(frame))
        ax.scatter(
            rank + jitter,
            frame[metric],
            s=12,
            alpha=0.38,
            color=RANK_COLOR,
            linewidth=0,
        )
        q25, median, q75 = frame[metric].quantile([0.25, 0.5, 0.75])
        ax.vlines(rank, q25, q75, color=RANK_COLOR, lw=3)
        ax.scatter(
            [rank],
            [median],
            s=40,
            color=RANK_COLOR,
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
    if reference is not None:
        ax.axhline(
            reference,
            color=REFERENCE_COLOR,
            lw=1.2,
            ls="--",
            label="Full GRU median",
        )
    if empirical is not None:
        ax.axhline(
            empirical,
            color=EMPIRICAL_COLOR,
            lw=1.2,
            ls=":",
            label="Empirical median",
        )
    ax.set_xticks(RANKS)
    ax.set_xlabel("Added shared recurrent rank")
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)


def _make_rank_sweep_figure(
    patient: pd.DataFrame,
    rank_summary: pd.DataFrame,
    references: dict[str, dict[str, float]],
    figures_dir: Path,
) -> None:
    full = references["full_history_gru"]
    empirical = references["empirical_rank_distribution"]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.4))
    _patient_points(
        axes[0, 0],
        patient,
        "rank_wasserstein",
        "Rank-distribution error (Wasserstein)",
        reference=full["rank_wasserstein"],
        empirical=empirical["rank_wasserstein"],
    )
    axes[0, 0].set_title("A  Rank-distribution fidelity", loc="left", weight="bold")
    _patient_points(
        axes[0, 1],
        patient,
        "precedence_correlation",
        "Pairwise precedence correlation",
        reference=full["precedence_correlation"],
        empirical=empirical["precedence_correlation"],
    )
    axes[0, 1].set_title("B  Pairwise order fidelity", loc="left", weight="bold")

    denominator = rank_summary["n_patients"].to_numpy(float)
    axes[1, 0].plot(
        RANKS,
        rank_summary["n_participation_within_variability"] / denominator,
        "o-",
        label="Participation",
        color="#5B8E7D",
    )
    axes[1, 0].plot(
        RANKS,
        rank_summary["n_rank_within_variability"] / denominator,
        "o-",
        label="Rank distribution",
        color="#B66A2B",
    )
    axes[1, 0].plot(
        RANKS,
        rank_summary["n_precedence_within_variability"] / denominator,
        "o-",
        label="Precedence",
        color="#2166AC",
    )
    axes[1, 0].plot(
        RANKS,
        rank_summary["n_all_three_within_variability"] / denominator,
        "o-",
        label="All three",
        color="#303030",
        lw=2,
    )
    axes[1, 0].set_xticks(RANKS)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_xlabel("Added shared recurrent rank")
    axes[1, 0].set_ylabel("Fraction within empirical variability")
    axes[1, 0].legend(frameon=False, fontsize=7, ncol=2)
    axes[1, 0].set_title(
        "C  Patient-level distribution gate", loc="left", weight="bold"
    )
    axes[1, 0].spines[["top", "right"]].set_visible(False)

    full_path = full["path_sliced_wasserstein"]
    _patient_points(
        axes[1, 1],
        patient,
        "path_sliced_wasserstein",
        "Whole-path sliced-Wasserstein",
        reference=full_path,
    )
    axes[1, 1].set_title(
        "D  Label-free whole-event paths", loc="left", weight="bold"
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.006),
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1), h_pad=2.0, w_pad=1.7)
    for suffix in ["png", "pdf"]:
        fig.savefig(
            figures_dir / f"low_rank_rank_sweep.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def _make_dynamics_figure(
    patient: pd.DataFrame,
    mode_patient: pd.DataFrame,
    loading: pd.DataFrame,
    references: dict[str, dict[str, float]],
    figures_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.3))
    _patient_points(
        axes[0, 0],
        patient,
        "heldout_event_nll",
        "Held-out event NLL",
        reference=references["full_history_gru"]["heldout_event_nll"],
    )
    axes[0, 0].set_title("A  Next-step training task", loc="left", weight="bold")

    _patient_points(
        axes[0, 1],
        mode_patient.rename(
            columns={"real_hidden_effective_dimension": "_dimension"}
        ),
        "_dimension",
        "Hidden-state effective dimension",
    )
    axes[0, 1].set_title(
        "B  Dimensionality of real-event trajectories", loc="left", weight="bold"
    )

    positive = mode_patient.loc[mode_patient["recurrent_rank"] > 0]
    for rank, frame in positive.groupby("recurrent_rank"):
        axes[1, 0].scatter(
            np.full(len(frame), rank),
            frame["median_mode_to_diagonal_drive_ratio"],
            s=12,
            alpha=0.35,
            color=RANK_COLOR,
            linewidth=0,
        )
        q25, median, q75 = frame[
            "median_mode_to_diagonal_drive_ratio"
        ].quantile([0.25, 0.5, 0.75])
        axes[1, 0].vlines(rank, q25, q75, color=RANK_COLOR, lw=3)
        axes[1, 0].scatter(
            rank,
            median,
            s=40,
            color=RANK_COLOR,
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
    axes[1, 0].axhline(1, color="#555555", ls="--", lw=1)
    axes[1, 0].set_xticks(RANKS[1:])
    axes[1, 0].set_xlabel("Added shared recurrent rank")
    axes[1, 0].set_ylabel("Mode-drive / diagonal-drive norm")
    axes[1, 0].set_title(
        "C  Are the added modes dynamically active?", loc="left", weight="bold"
    )
    axes[1, 0].spines[["top", "right"]].set_visible(False)

    for field, label, color in [
        ("u_output_loading", "U output subspace", "#B2182B"),
        ("v_output_loading", "V output subspace", "#2166AC"),
    ]:
        frame = loading.loc[loading["loading_field"] == field]
        summary = frame.groupby("recurrent_rank")[
            "median_chance_adjusted_subspace_similarity"
        ].median()
        axes[1, 1].plot(
            summary.index,
            summary.values,
            "o-",
            label=label,
            color=color,
        )
    axes[1, 1].axhline(0, color="#777777", ls=":", lw=1)
    axes[1, 1].set_xticks(RANKS[1:])
    axes[1, 1].set_xlabel("Added shared recurrent rank")
    axes[1, 1].set_ylabel("Chance-adjusted seed subspace similarity")
    axes[1, 1].set_title(
        "D  Are contact-loading subspaces identifiable?", loc="left", weight="bold"
    )
    axes[1, 1].legend(frameon=False, fontsize=7)
    axes[1, 1].spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.006),
        )
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=2.0, w_pad=1.7)
    for suffix in ["png", "pdf"]:
        fig.savefig(
            figures_dir / f"low_rank_internal_dynamics.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def _write_report(
    output_dir: Path,
    summary: dict[str, object],
    rank_summary: pd.DataFrame,
    vs_zero: pd.DataFrame,
    mode_patient: pd.DataFrame,
    loading: pd.DataFrame,
) -> None:
    rank_zero = rank_summary.loc[rank_summary["recurrent_rank"] == 0].iloc[0]
    rank_one = rank_summary.loc[rank_summary["recurrent_rank"] == 1].iloc[0]
    rank_four = rank_summary.loc[rank_summary["recurrent_rank"] == 4].iloc[0]

    def comparison(metric: str, rank: int) -> pd.Series:
        return vs_zero.loc[
            (vs_zero["metric"] == metric)
            & (vs_zero["recurrent_rank"] == rank)
        ].iloc[0]

    rank4_w1 = comparison("rank_wasserstein", 4)
    path_rank1 = comparison("path_sliced_wasserstein", 1)
    positive_modes = mode_patient.loc[mode_patient["recurrent_rank"] > 0]
    drive_by_rank = positive_modes.groupby("recurrent_rank")[
        "median_mode_to_diagonal_drive_ratio"
    ].median()
    dimension_by_rank = mode_patient.groupby("recurrent_rank")[
        "real_hidden_effective_dimension"
    ].median()
    rank1_loading = loading.loc[
        (loading["recurrent_rank"] == 1)
        & (loading["loading_field"] == "u_output_loading")
    ]["median_chance_adjusted_subspace_similarity"].median()
    text = f"""# Structured low-rank leaky RNN：系统分析

## 分析边界

本次覆盖 34 名患者、3 个 seed、rank 0–4，共 510 个 LOSO fold。全部工程完成且无错误。
分析只使用间期事件；发作期 target 尚未读取。

## 1. 预设分布门

没有任何正 rank 模型通过预设的三项分布门，因此预设的
`minimum_distribution_sufficient_rank` 仍为空。三项指标同时进入患者自身经验变异范围的
人数分别为：rank 0 =
{int(rank_zero['n_all_three_within_variability'])}/34，rank 1 =
{int(rank_one['n_all_three_within_variability'])}/34，rank 4 =
{int(rank_four['n_all_three_within_variability'])}/34。

rank 0 的中位 rank-Wasserstein 为
{rank_zero['median_rank_wasserstein']:.3f}，precedence correlation 为
{rank_zero['median_precedence_correlation']:.3f}。增加 rank 1–3 没有稳定改善，
rank 4 反而使 rank-Wasserstein 变差，中位差
{rank4_w1['median_benefit']:.4f}（正值才表示 rank 4 更好；
FDR p={rank4_w1['wilcoxon_p_fdr']:.3g}）。

## 2. 完整事件路径

探索性的 label-free whole-path 指标把每个事件表示为“触点参与向量 + 归一化
rank 向量”，不使用 A/B 标签。rank 0 有
{int(rank_zero['n_path_within_variability'])}/34 名患者落入经验变异范围；
rank 1 为 {int(rank_one['n_path_within_variability'])}/34。rank 1 相对 rank 0
的中位增益为 {path_rank1['median_benefit']:.4f}
（FDR p={path_rank1['wilcoxon_p_fdr']:.3g}），方向仍是 rank 1 更差。

## 3. 模式是否真正工作

正 rank 模式没有被优化器简单压成零。mode drive 与 diagonal drive 的中位范数比
在各 rank 间为 {drive_by_rank.min():.2f}–{drive_by_rank.max():.2f}，说明附加模式
在状态更新中占有实质量级。然而这种驱动没有转化成更好的 held-out 分布。

真实事件轨迹的 hidden-state effective dimension 在所有 rank 上都约为
{dimension_by_rank.min():.2f}–{dimension_by_rank.max():.2f}，主要表现为共同的低维
progress 方向，而不是随 rank 增加出现更丰富、可重复的传播分支。

rank 1 的 U-contact loading 跨 seed、chance-adjusted subspace similarity 中位数只有
{rank1_loading:.2f}。因此当前单一模式的触点解释不稳定，不能拿去直接解释发作早期场。

## 4. 架构诊断

当前所谓 rank 0 并不是无递归模型。它仍有 32 个可训练的对角 self-decay、leaky
hidden state、非线性 input projection 和 contact decoder，因此本身已经是一个具有
32 条独立记忆通道的 diagonal leaky RNN。`UV^T` 只是附加在这套记忆之上的共享耦合，
并没有真正限制模型只能通过 r 个 latent modes 传递历史。

这解释了为什么 rank 0 可以达到或超过 full GRU，而 rank 1–4 的模式虽动态活跃，
却是冗余且不稳定的。

## 5. 科学裁决

当前结果不支持“一个可识别的正 low-rank recurrent mode 足以解释间期传播”。
因此暂不读取发作期 target，也不做 cross-state mode claim。

这不是对“间期传播具有低维结构”的否定，而是当前参数化没有把 low-rank 变成真正
承重的动力学瓶颈。下一版应把 rank 0 改成真正无历史对照，并把对角项收缩为共享
scalar decay，或直接使用 r 维 latent state 再映射到触点；随后重新测试 rank 1–4。
"""
    (output_dir / "analysis_report.md").write_text(text)


def _write_figure_readme(figures_dir: Path) -> None:
    text = """### low_rank_rank_sweep.png

这张图比较 rank 0–4 在 held-out 自由生成中的 rank 分布、成对先后关系、患者自身经验变异门和完整事件路径。虚线为 full-rank GRU 中位数，点线为直接经验分布中位数；每个红点是一名患者，粗点和竖线为中位数与四分位范围。

**关注点**：增加正 rank 没有产生单调改善，rank 0 的 diagonal leaky dynamics 已达到或超过正 rank 模型。

### low_rank_internal_dynamics.png

这张图检查训练任务、真实事件隐藏轨迹维度、正 rank 模式的实际驱动力，以及触点 loading 子空间的跨 seed 稳定性。mode-drive/diagonal-drive 接近 1 表示附加模式并非被压成零，但这不等同于它们提供了可重复的行为解释。

**关注点**：所有 rank 的轨迹都接近共同的一维 progress 方向；rank 1 的触点 loading 跨 seed 不稳定。
"""
    (figures_dir / "README.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-rank-root", type=Path, required=True)
    parser.add_argument("--full-rank-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 8.0,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    low, full = _load_behavior_metrics(args.low_rank_root, args.full_rank_root)
    low_path, full_path = _path_diagnostics(
        args.low_rank_root, args.full_rank_root, args.dataset_root
    )
    mode = _mode_diagnostics(args.low_rank_root)
    loading = _loading_stability(args.low_rank_root)
    patient, full_patient = _patient_tables(
        low,
        full,
        low_path,
        full_path,
        args.low_rank_root / "patient_seed_collapsed_summary.csv",
    )
    mode_patient = (
        mode.groupby(["subject", "recurrent_rank"], as_index=False)
        .median(numeric_only=True)
    )
    rank_summary = _rank_summary(patient)
    vs_zero, vs_full = _paired_rank_comparisons(patient, full_patient)
    seed_stability = _seed_stability(low, low_path)
    references = _reference_medians(full_patient)
    summary = _build_summary(
        rank_summary, vs_zero, mode_patient, loading, references
    )

    low_path.to_csv(args.output_dir / "whole_path_all_seed.csv", index=False)
    patient.to_csv(args.output_dir / "patient_rank_metrics.csv", index=False)
    full_patient.to_csv(
        args.output_dir / "full_and_empirical_reference_metrics.csv", index=False
    )
    rank_summary.to_csv(args.output_dir / "rank_cohort_summary.csv", index=False)
    vs_zero.to_csv(args.output_dir / "rank_comparisons_vs_rank0.csv", index=False)
    vs_full.to_csv(
        args.output_dir / "rank_comparisons_vs_full_gru.csv", index=False
    )
    seed_stability.to_csv(args.output_dir / "seed_stability.csv", index=False)
    mode.to_csv(args.output_dir / "mode_dynamics_all_seed.csv", index=False)
    mode_patient.to_csv(
        args.output_dir / "mode_dynamics_patient_collapsed.csv", index=False
    )
    loading.to_csv(
        args.output_dir / "contact_loading_seed_stability.csv", index=False
    )
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    _make_rank_sweep_figure(patient, rank_summary, references, figures_dir)
    _make_dynamics_figure(
        patient, mode_patient, loading, references, figures_dir
    )
    _write_report(
        args.output_dir, summary, rank_summary, vs_zero, mode_patient, loading
    )
    _write_figure_readme(figures_dir)
    (args.output_dir / "DONE.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "n_patients": 34,
                "n_seeds": 3,
                "ranks": RANKS,
                "ictal_target_read": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
