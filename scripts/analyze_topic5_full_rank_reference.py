#!/usr/bin/env python3
"""Patient-level analysis of the Topic 5 full-rank GRU reference.

The full-rank model is an unconstrained reference for the structured low-rank
models.  This analysis intentionally stays on the interictal side and does not
read any ictal target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon


LOWER_IS_BETTER = {
    "participation_mae": True,
    "rank_wasserstein": True,
    "precedence_mae": True,
    "precedence_correlation": False,
}

CONTROL_ORDER = [
    "empirical_rank_distribution",
    "full_history_gru",
    "rank_shuffle_gru",
    "last_set_first_order",
    "unordered_prefix",
    "static_contact_hazard",
]

CONTROL_LABELS = {
    "empirical_rank_distribution": "Empirical",
    "full_history_gru": "Full GRU",
    "rank_shuffle_gru": "Rank-shuffled",
    "last_set_first_order": "First-order",
    "unordered_prefix": "Unordered prefix",
    "static_contact_hazard": "Static hazard",
}

DATASET_COLORS = {"epilepsiae": "#2166AC", "yuquan": "#B66A2B"}


def _bootstrap_median_ci(
    values: Iterable[float], seed: int = 20260725, n_boot: int = 20_000
) -> tuple[float, float]:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(x, size=(n_boot, len(x)), replace=True)
    medians = np.median(draws, axis=1)
    return tuple(np.quantile(medians, [0.025, 0.975]))


def _safe_wilcoxon(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0 or np.allclose(x, 0):
        return np.nan
    return float(wilcoxon(x, alternative="two-sided").pvalue)


def _bh_fdr(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    out = np.full(values.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if len(valid) == 0:
        return pd.Series(out, index=p_values.index)
    order = valid[np.argsort(values[valid])]
    ranked = values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out[order] = np.minimum(adjusted, 1.0)
    return pd.Series(out, index=p_values.index)


def _load_metrics(run_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_frames = []
    run_rows = []
    split_rows = []
    for metric_path in sorted(run_root.glob("seed_*/*/heldout_metrics.csv")):
        frame = pd.read_csv(metric_path)
        metric_frames.append(frame)
        run_summary = json.loads((metric_path.parent / "run_summary.json").read_text())
        contacts = pd.read_csv(metric_path.parent / "contact_rank_distributions.csv")
        contacts = contacts.loc[contacts["control"] == "full_history_gru"]
        run_rows.append(
            {
                "subject": run_summary["heldout_subject"],
                "dataset": run_summary["dataset"],
                "seed": int(run_summary["seed"]),
                "n_eval_events": int(run_summary["n_eval_events_used"]),
                "n_contacts": int(contacts["contact_index"].nunique()),
            }
        )
        split = run_summary["empirical_split_half_variability"]
        split_rows.append(
            {
                "subject": run_summary["heldout_subject"],
                "dataset": run_summary["dataset"],
                "seed": int(run_summary["seed"]),
                "split_half_participation_mae": float(split["participation_mae"]),
                "split_half_rank_wasserstein": float(split["rank_wasserstein"]),
                "split_half_precedence_mae": float(split["precedence_mae"]),
            }
        )
    if not metric_frames:
        raise FileNotFoundError(f"No heldout_metrics.csv files under {run_root}")
    metrics = pd.concat(metric_frames, ignore_index=True)
    return metrics, pd.DataFrame(run_rows), pd.DataFrame(split_rows)


def _build_patient_table(
    metrics: pd.DataFrame,
    run_info: pd.DataFrame,
    split_half: pd.DataFrame,
    run_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    collapsed = (
        metrics.groupby(["subject", "dataset", "control"], as_index=False)
        .median(numeric_only=True)
    )
    index = ["subject", "dataset"]
    wide = collapsed.pivot(index=index, columns="control")
    patient = pd.DataFrame(index=wide.index)
    for metric in LOWER_IS_BETTER:
        for control in CONTROL_ORDER:
            patient[f"{control}__{metric}"] = wide[metric][control]

    formal = pd.read_csv(run_root / "patient_seed_collapsed_summary.csv")
    formal = formal.set_index(index)
    for column in [
        "ordered_history_nll_gain",
        "shuffle_minus_gru_participation_mae",
        "shuffle_minus_gru_rank_wasserstein",
    ]:
        patient[column] = formal[column]

    patient = patient.join(
        run_info.groupby(index).median(numeric_only=True)[
            ["n_eval_events", "n_contacts"]
        ]
    )
    patient = patient.join(
        split_half.groupby(index).median(numeric_only=True)[
            [
                "split_half_participation_mae",
                "split_half_rank_wasserstein",
                "split_half_precedence_mae",
            ]
        ]
    )

    for metric in ["participation_mae", "rank_wasserstein", "precedence_mae"]:
        full = patient[f"full_history_gru__{metric}"]
        empirical = patient[f"empirical_rank_distribution__{metric}"]
        margin = patient[f"split_half_{metric}"]
        patient[f"{metric}__noninferiority_excess"] = full - empirical - margin
        patient[f"{metric}__within_empirical_variability"] = (
            patient[f"{metric}__noninferiority_excess"] <= 0
        )
    patient["all_three_distribution_metrics_within_variability"] = patient[
        [
            "participation_mae__within_empirical_variability",
            "rank_wasserstein__within_empirical_variability",
            "precedence_mae__within_empirical_variability",
        ]
    ].all(axis=1)
    return patient.reset_index(), collapsed


def _control_summary(collapsed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for control in CONTROL_ORDER:
        subset = collapsed.loc[collapsed["control"] == control]
        for metric in LOWER_IS_BETTER:
            values = subset[metric].dropna()
            lo, hi = _bootstrap_median_ci(values)
            rows.append(
                {
                    "control": control,
                    "metric": metric,
                    "n_patients": int(len(values)),
                    "median": float(values.median()),
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75)),
                    "bootstrap_ci95_low": lo,
                    "bootstrap_ci95_high": hi,
                }
            )
    return pd.DataFrame(rows)


def _paired_comparisons(patient: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, lower_is_better in LOWER_IS_BETTER.items():
        full = patient[f"full_history_gru__{metric}"]
        for control in CONTROL_ORDER:
            if control == "full_history_gru":
                continue
            other = patient[f"{control}__{metric}"]
            benefit = other - full if lower_is_better else full - other
            lo, hi = _bootstrap_median_ci(benefit)
            rows.append(
                {
                    "metric": metric,
                    "comparison": f"full_history_gru_vs_{control}",
                    "positive_means_full_gru_better": True,
                    "n_patients": int(benefit.notna().sum()),
                    "median_benefit": float(benefit.median()),
                    "bootstrap_ci95_low": lo,
                    "bootstrap_ci95_high": hi,
                    "n_full_gru_better": int((benefit > 0).sum()),
                    "wilcoxon_p_raw": _safe_wilcoxon(benefit),
                }
            )
    result = pd.DataFrame(rows)
    result["wilcoxon_p_fdr"] = _bh_fdr(result["wilcoxon_p_raw"])
    return result


def _seed_stability(metrics: pd.DataFrame) -> pd.DataFrame:
    full = metrics.loc[metrics["control"] == "full_history_gru"]
    rows = []
    for metric in [
        "heldout_event_nll",
        "participation_mae",
        "rank_wasserstein",
        "precedence_mae",
        "precedence_correlation",
    ]:
        pivot = full.pivot(index="subject", columns="seed", values=metric)
        rhos = []
        seeds = list(pivot.columns)
        for i, seed_a in enumerate(seeds):
            for seed_b in seeds[i + 1 :]:
                rho = float(spearmanr(pivot[seed_a], pivot[seed_b]).statistic)
                rhos.append(rho)
                rows.append(
                    {
                        "metric": metric,
                        "seed_a": int(seed_a),
                        "seed_b": int(seed_b),
                        "spearman_rho": rho,
                        "median_within_patient_sd": float(
                            pivot.std(axis=1).median()
                        ),
                        "between_patient_sd_of_seed_mean": float(
                            pivot.mean(axis=1).std()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _primary_summary(
    patient: pd.DataFrame, seed_stability: pd.DataFrame
) -> dict[str, object]:
    def one_sample(values: pd.Series) -> dict[str, object]:
        lo, hi = _bootstrap_median_ci(values)
        return {
            "median": float(values.median()),
            "bootstrap_ci95": [lo, hi],
            "n_positive": int((values > 0).sum()),
            "n_patients": int(values.notna().sum()),
            "wilcoxon_p_two_sided": _safe_wilcoxon(values),
        }

    order = one_sample(patient["ordered_history_nll_gain"])
    shuffle_rank = one_sample(patient["shuffle_minus_gru_rank_wasserstein"])
    shuffle_participation = one_sample(
        patient["shuffle_minus_gru_participation_mae"]
    )
    sequence_p = pd.Series(
        [
            order["wilcoxon_p_two_sided"],
            shuffle_rank["wilcoxon_p_two_sided"],
            shuffle_participation["wilcoxon_p_two_sided"],
        ]
    )
    sequence_fdr = _bh_fdr(sequence_p).tolist()
    order["p_fdr_across_sequence_tests"] = sequence_fdr[0]
    shuffle_rank["p_fdr_across_sequence_tests"] = sequence_fdr[1]
    shuffle_participation["p_fdr_across_sequence_tests"] = sequence_fdr[2]

    reliability = {}
    for metric in ["participation_mae", "rank_wasserstein", "precedence_mae"]:
        excess = patient[f"{metric}__noninferiority_excess"]
        lo, hi = _bootstrap_median_ci(excess)
        reliability[metric] = {
            "median_noninferiority_excess": float(excess.median()),
            "bootstrap_ci95": [lo, hi],
            "n_within_empirical_variability": int((excess <= 0).sum()),
            "n_patients": int(len(excess)),
        }

    contact_rho = spearmanr(
        patient["n_contacts"], patient["ordered_history_nll_gain"]
    )
    event_rho = spearmanr(
        np.log10(patient["n_eval_events"]),
        patient["full_history_gru__rank_wasserstein"],
    )

    return {
        "status": "complete",
        "role": "unconstrained_full_rank_reference_not_primary_model",
        "ictal_target_read": False,
        "n_patients": int(patient["subject"].nunique()),
        "n_seeds": 3,
        "dataset_counts": patient["dataset"].value_counts().to_dict(),
        "ordered_history_next_step_nll": order,
        "rank_shuffle_minus_full_gru_rank_wasserstein": shuffle_rank,
        "rank_shuffle_minus_full_gru_participation_mae": shuffle_participation,
        "empirical_variability": reliability,
        "n_all_three_distribution_metrics_within_variability": int(
            patient["all_three_distribution_metrics_within_variability"].sum()
        ),
        "seed_stability_median_pairwise_spearman": seed_stability.groupby(
            "metric"
        )["spearman_rho"].median().to_dict(),
        "exploratory_covariates": {
            "contact_count_vs_ordered_history_gain_spearman_rho": float(
                contact_rho.statistic
            ),
            "contact_count_vs_ordered_history_gain_p": float(contact_rho.pvalue),
            "log_eval_events_vs_rank_wasserstein_spearman_rho": float(
                event_rho.statistic
            ),
            "log_eval_events_vs_rank_wasserstein_p": float(event_rho.pvalue),
        },
    }


def _draw_distribution_axis(
    ax: plt.Axes,
    collapsed: pd.DataFrame,
    metric: str,
    ylabel: str,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    palette = {
        "empirical_rank_distribution": "#222222",
        "full_history_gru": "#B2182B",
        "rank_shuffle_gru": "#7F7F7F",
        "last_set_first_order": "#A6A6A6",
        "unordered_prefix": "#BDBDBD",
        "static_contact_hazard": "#D0D0D0",
    }
    for x, control in enumerate(CONTROL_ORDER):
        values = collapsed.loc[collapsed["control"] == control, metric].dropna()
        jitter = rng.uniform(-0.14, 0.14, len(values))
        ax.scatter(
            x + jitter,
            values,
            s=11,
            alpha=0.42,
            color=palette[control],
            linewidths=0,
            zorder=1,
        )
        q25, median, q75 = values.quantile([0.25, 0.5, 0.75])
        ax.vlines(x, q25, q75, color=palette[control], lw=3.0, zorder=3)
        ax.scatter(
            [x],
            [median],
            s=40,
            color=palette[control],
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
    ax.set_xticks(range(len(CONTROL_ORDER)))
    ax.set_xticklabels(
        [CONTROL_LABELS[x] for x in CONTROL_ORDER], rotation=32, ha="right"
    )
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.55, len(CONTROL_ORDER) - 0.45)
    ax.spines[["top", "right"]].set_visible(False)


def _draw_sorted_patient_effect(
    ax: plt.Axes,
    patient: pd.DataFrame,
    column: str,
    ylabel: str,
) -> None:
    ordered = patient.sort_values(column).reset_index(drop=True)
    for dataset, group in ordered.groupby("dataset"):
        ax.scatter(
            group.index,
            group[column],
            s=24,
            color=DATASET_COLORS[dataset],
            label=dataset.capitalize(),
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
    ax.axhline(0, color="#444444", lw=0.9, ls="--")
    ax.set_xlabel("Patients, sorted by effect")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.8, len(ordered) - 0.2)
    ax.spines[["top", "right"]].set_visible(False)


def _make_cohort_figure(
    patient: pd.DataFrame, collapsed: pd.DataFrame, figures_dir: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.7))
    _draw_distribution_axis(
        axes[0, 0],
        collapsed,
        "rank_wasserstein",
        "Rank-distribution error (Wasserstein)",
        1,
    )
    axes[0, 0].set_title("A  Free-running rank distributions", loc="left", weight="bold")
    _draw_distribution_axis(
        axes[0, 1],
        collapsed,
        "participation_mae",
        "Contact-participation error (MAE)",
        2,
    )
    axes[0, 1].set_title("B  Contact participation", loc="left", weight="bold")
    _draw_sorted_patient_effect(
        axes[1, 0],
        patient,
        "ordered_history_nll_gain",
        "NLL reduction from ordered history",
    )
    axes[1, 0].set_title(
        "C  Does ordered history improve\nnext-step fit?",
        loc="left",
        weight="bold",
    )
    _draw_sorted_patient_effect(
        axes[1, 1],
        patient,
        "shuffle_minus_gru_rank_wasserstein",
        "Rank error: shuffled − ordered GRU",
    )
    axes[1, 1].set_title(
        "D  Does within-event order improve\ngenerated ranks?",
        loc="left",
        weight="bold",
    )
    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1), h_pad=2.1, w_pad=1.6)
    for suffix in ["png", "pdf"]:
        fig.savefig(
            figures_dir / f"full_rank_reference_cohort.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def _make_reliability_figure(
    patient: pd.DataFrame, seed_stability: pd.DataFrame, figures_dir: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.5))

    ordered = patient.sort_values(
        "rank_wasserstein__noninferiority_excess"
    ).reset_index(drop=True)
    for dataset, group in ordered.groupby("dataset"):
        axes[0, 0].scatter(
            group.index,
            group["rank_wasserstein__noninferiority_excess"],
            s=24,
            color=DATASET_COLORS[dataset],
            edgecolor="white",
            linewidth=0.35,
        )
    axes[0, 0].axhline(0, color="#444444", lw=0.9, ls="--")
    axes[0, 0].set_xlabel("Patients, sorted by excess error")
    axes[0, 0].set_ylabel("GRU rank error − empirical error − variability")
    axes[0, 0].set_title(
        "A  Is rank reconstruction within\nempirical variability?",
        loc="left",
        weight="bold",
    )

    metric_labels = ["Participation", "Rank distribution", "Precedence", "All three"]
    counts = [
        int(patient["participation_mae__within_empirical_variability"].sum()),
        int(patient["rank_wasserstein__within_empirical_variability"].sum()),
        int(patient["precedence_mae__within_empirical_variability"].sum()),
        int(patient["all_three_distribution_metrics_within_variability"].sum()),
    ]
    axes[0, 1].bar(
        range(4), np.asarray(counts) / len(patient), color=["#7B9E87"] * 3 + ["#3B6F57"]
    )
    axes[0, 1].set_xticks(range(4))
    axes[0, 1].set_xticklabels(metric_labels, rotation=25, ha="right")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_ylabel("Fraction of patients")
    for x, count in enumerate(counts):
        axes[0, 1].text(
            x,
            count / len(patient) + 0.025,
            f"{count}/{len(patient)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[0, 1].set_title(
        "B  Distribution-level\nreproducibility", loc="left", weight="bold"
    )

    stability_order = [
        "heldout_event_nll",
        "participation_mae",
        "rank_wasserstein",
        "precedence_mae",
        "precedence_correlation",
    ]
    stability_labels = ["Event NLL", "Participation", "Rank W1", "Precedence MAE", "Precedence r"]
    for x, metric in enumerate(stability_order):
        values = seed_stability.loc[
            seed_stability["metric"] == metric, "spearman_rho"
        ]
        axes[1, 0].scatter(
            np.repeat(x, len(values)), values, color="#555555", s=25, zorder=3
        )
        axes[1, 0].hlines(
            values.median(), x - 0.24, x + 0.24, color="#B2182B", lw=2.5
        )
    axes[1, 0].set_xticks(range(len(stability_order)))
    axes[1, 0].set_xticklabels(stability_labels, rotation=27, ha="right")
    axes[1, 0].set_ylabel("Pairwise seed Spearman ρ")
    axes[1, 0].set_ylim(-0.05, 1.02)
    axes[1, 0].set_title("C  Seed stability", loc="left", weight="bold")

    for dataset, group in patient.groupby("dataset"):
        axes[1, 1].scatter(
            group["n_contacts"],
            group["ordered_history_nll_gain"],
            s=31,
            color=DATASET_COLORS[dataset],
            label=dataset.capitalize(),
            edgecolor="white",
            linewidth=0.4,
        )
    rho = spearmanr(patient["n_contacts"], patient["ordered_history_nll_gain"])
    axes[1, 1].axhline(0, color="#444444", lw=0.9, ls="--")
    axes[1, 1].text(
        0.98,
        0.96,
        f"Spearman ρ={rho.statistic:.2f}\np={rho.pvalue:.3g}",
        ha="right",
        va="top",
        transform=axes[1, 1].transAxes,
        fontsize=8,
    )
    axes[1, 1].set_xlabel("Number of contacts")
    axes[1, 1].set_ylabel("Ordered-history NLL gain")
    axes[1, 1].set_title(
        "D  Where does recurrence help?", loc="left", weight="bold"
    )

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1), h_pad=2.0, w_pad=1.6)
    for suffix in ["png", "pdf"]:
        fig.savefig(
            figures_dir / f"full_rank_reference_reliability.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def _write_report(
    out_dir: Path,
    summary: dict[str, object],
    control_summary: pd.DataFrame,
    paired: pd.DataFrame,
) -> None:
    def metric(control: str, name: str) -> float:
        row = control_summary.loc[
            (control_summary["control"] == control)
            & (control_summary["metric"] == name)
        ]
        return float(row.iloc[0]["median"])

    ordered = summary["ordered_history_next_step_nll"]
    shuffled_rank = summary["rank_shuffle_minus_full_gru_rank_wasserstein"]
    shuffled_part = summary["rank_shuffle_minus_full_gru_participation_mae"]
    reliability = summary["empirical_variability"]
    rank_shuffle_precedence = paired.loc[
        (paired["metric"] == "precedence_correlation")
        & (
            paired["comparison"]
            == "full_history_gru_vs_rank_shuffle_gru"
        )
    ].iloc[0]
    text = f"""# Full-rank GRU reference analysis

## Scope

This is a patient-level, three-seed analysis of all 34 interictal folds
({summary['dataset_counts']['epilepsiae']} Epilepsiae and
{summary['dataset_counts']['yuquan']} Yuquan). The ictal target was not read.
The full-rank GRU is treated as an unconstrained reference, not as the primary
mechanistic model.

## Main results

1. Ordered history did not improve next-step likelihood at cohort level:
median NLL gain {ordered['median']:.4f}, 95% bootstrap CI
[{ordered['bootstrap_ci95'][0]:.4f}, {ordered['bootstrap_ci95'][1]:.4f}],
{ordered['n_positive']}/{ordered['n_patients']} patients positive,
Wilcoxon p={ordered['wilcoxon_p_two_sided']:.3g}.

2. Event order nevertheless contributed to free-running spatial structure.
Rank shuffling increased contact-participation MAE by a median
{shuffled_part['median']:.4f}
({shuffled_part['n_positive']}/{shuffled_part['n_patients']} patients;
p={shuffled_part['wilcoxon_p_two_sided']:.3g}) and reduced pairwise precedence
correlation in {int(rank_shuffle_precedence['n_full_gru_better'])}/34 patients
(median ordered-GRU benefit
{rank_shuffle_precedence['median_benefit']:.3f},
p={rank_shuffle_precedence['wilcoxon_p_raw']:.3g}). The rank-Wasserstein
advantage was weaker: median {shuffled_rank['median']:.4f},
{shuffled_rank['n_positive']}/{shuffled_rank['n_patients']} patients,
p={shuffled_rank['wilcoxon_p_two_sided']:.3g}.

3. The full GRU did not reproduce the complete held-out distribution as well
as the direct empirical reference. Median rank-Wasserstein error was
{metric('full_history_gru', 'rank_wasserstein'):.3f} for the GRU versus
{metric('empirical_rank_distribution', 'rank_wasserstein'):.3f} for the
empirical distribution; precedence correlation was
{metric('full_history_gru', 'precedence_correlation'):.3f} versus
{metric('empirical_rank_distribution', 'precedence_correlation'):.3f}.

4. Relative to patient-specific split-half variability, the GRU was within
range for participation in
{reliability['participation_mae']['n_within_empirical_variability']}/34,
rank distribution in
{reliability['rank_wasserstein']['n_within_empirical_variability']}/34,
precedence error in
{reliability['precedence_mae']['n_within_empirical_variability']}/34, and all
three simultaneously in
{summary['n_all_three_distribution_metrics_within_variability']}/34 patients.

5. Rank-distribution and precedence-error estimates were stable across seeds
(median pairwise Spearman rho
{summary['seed_stability_median_pairwise_spearman']['rank_wasserstein']:.2f}
and
{summary['seed_stability_median_pairwise_spearman']['precedence_mae']:.2f}),
whereas the magnitude of precedence correlation was less stable
(rho
{summary['seed_stability_median_pairwise_spearman']['precedence_correlation']:.2f}).

## Scientific interpretation

The safe conclusion is not that the full-rank GRU is a superior predictor.
Instead, the ordered model contains reproducible information about within-event
contact ordering, especially pairwise precedence, while its unconstrained
free-running dynamics do not recover the complete patient distribution for
most patients. This makes it a useful reference ceiling and diagnostic model.
The structured low-rank analysis must test whether a simpler recurrent system
can preserve the order-sensitive component while improving distributional
stability and interpretability.

The positive association between contact count and ordered-history NLL gain
is exploratory and may reflect task complexity or dataset differences; it is
not a mechanistic result.
"""
    (out_dir / "analysis_report.md").write_text(text)


def _write_figure_readme(figures_dir: Path) -> None:
    text = """### full_rank_reference_cohort.png

这张图先比较 full-rank GRU、经验分布和四个对照在自由生成后的患者级误差，再分别显示逐步历史和事件内顺序带来的增量。A/B 回答模型能否重建完整触点分布，C/D 回答递归历史与真实顺序是否提供额外信息。

**关注点**：整体一步预测没有稳定增益，但打乱事件内顺序后，部分空间传播结构会丢失。

### full_rank_reference_reliability.png

这张图检查 full-rank 结果是否落在患者自身 split-half 变异范围内、三项分布指标能覆盖多少患者、不同 seed 是否稳定，以及递归增益与触点数的关系。最后一项只作患者异质性的探索性诊断。

**关注点**：rank/precedence 误差本身跨 seed 稳定，但只有少数患者同时达到三项经验变异范围。
"""
    (figures_dir / "README.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.out_dir / "figures"
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

    metrics, run_info, split_half = _load_metrics(args.run_root)
    patient, collapsed = _build_patient_table(
        metrics, run_info, split_half, args.run_root
    )
    control_summary = _control_summary(collapsed)
    paired = _paired_comparisons(patient)
    seed_stability = _seed_stability(metrics)
    summary = _primary_summary(patient, seed_stability)

    patient.to_csv(args.out_dir / "full_rank_patient_metrics.csv", index=False)
    collapsed.to_csv(
        args.out_dir / "full_rank_patient_control_metrics.csv", index=False
    )
    control_summary.to_csv(
        args.out_dir / "full_rank_control_summary.csv", index=False
    )
    paired.to_csv(
        args.out_dir / "full_rank_paired_comparisons.csv", index=False
    )
    seed_stability.to_csv(
        args.out_dir / "full_rank_seed_stability.csv", index=False
    )
    (args.out_dir / "full_rank_reference_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    _make_cohort_figure(patient, collapsed, figures_dir)
    _make_reliability_figure(patient, seed_stability, figures_dir)
    _write_report(args.out_dir, summary, control_summary, paired)
    _write_figure_readme(figures_dir)
    (args.out_dir / "DONE.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "n_patients": int(patient["subject"].nunique()),
                "n_seeds": int(metrics["seed"].nunique()),
                "ictal_target_read": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
