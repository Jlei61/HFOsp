#!/usr/bin/env python3
"""Build subject- and cohort-level summaries for T_spectral v1.2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    ROOT
    / "results/topic5_ictal_recruitment/peri_onset_energy_timing/epilepsiae/refinement_v1p2"
)

STATUS_ORDER = (
    "phenotype_present",
    "phenotype_absent",
    "prior_candidate_manual_only",
)
STATUS_COLORS = {
    "phenotype_present": "#2C7FB8",
    "phenotype_absent": "#D0D0D0",
    "prior_candidate_manual_only": "#7A4F9A",
}


def _finite(values: pd.Series | np.ndarray) -> np.ndarray:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return x[np.isfinite(x)]


def _q(values: pd.Series | np.ndarray, probs=(0.25, 0.5, 0.75)) -> list[float]:
    x = _finite(values)
    if not x.size:
        return [float("nan") for _ in probs]
    return [float(v) for v in np.quantile(x, probs)]


def _bool_series(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def build_subject_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for subject, use in events.groupby("subject", sort=True):
        status = use["phenotype_status"].value_counts()
        timing = use["timing_status"].value_counts()
        candidates = use[_bool_series(use["has_candidate_t"])]
        accepted = use[_bool_series(use["has_accepted_t_best"])]
        times = _finite(accepted["t_spectral_best_rel_eeg_sec"])
        clinical_times = _finite(accepted["t_spectral_best_rel_clinical_sec"])
        widths = _finite(accepted["bootstrap_width_sec"])
        consistency = _finite(accepted["selection_consistency_1s"])
        margins = _finite(accepted["score_margin"])
        shifts = _finite(accepted["delta_t_best_minus_v1p1_sec"])
        prototype = accepted[_bool_series(accepted["prototype_used"])]
        q25, med, q75 = _q(accepted["t_spectral_best_rel_eeg_sec"])
        rows.append(
            {
                "subject": subject,
                "n_seizures": len(use),
                "n_phenotype_present": int(status.get("phenotype_present", 0)),
                "n_phenotype_absent": int(status.get("phenotype_absent", 0)),
                "n_prior_candidate_manual_only": int(
                    status.get("prior_candidate_manual_only", 0)
                ),
                "n_accepted_t_best": len(accepted),
                "n_candidate_t": len(candidates),
                "n_candidate_no_subject_timing_template": int(
                    timing.get("candidate_no_subject_timing_template", 0)
                ),
                "n_candidate_temporally_unanchored": int(
                    timing.get("candidate_temporally_unanchored", 0)
                ),
                "n_accepted_subject_recurrent": int(
                    timing.get("accepted_subject_recurrent", 0)
                ),
                "fraction_phenotype_present": float(
                    status.get("phenotype_present", 0) / len(use)
                ),
                "t_best_q25_rel_eeg_sec": q25,
                "t_best_median_rel_eeg_sec": med,
                "t_best_q75_rel_eeg_sec": q75,
                "fraction_t_best_within_eeg_1s": (
                    float(np.mean(np.abs(times) <= 1.0)) if times.size else float("nan")
                ),
                "fraction_t_best_within_eeg_2s": (
                    float(np.mean(np.abs(times) <= 2.0)) if times.size else float("nan")
                ),
                "fraction_t_best_within_eeg_5s": (
                    float(np.mean(np.abs(times) <= 5.0)) if times.size else float("nan")
                ),
                "t_best_median_rel_clinical_sec": (
                    float(np.median(clinical_times))
                    if clinical_times.size
                    else float("nan")
                ),
                "median_abs_distance_to_eeg_sec": (
                    float(np.median(np.abs(times))) if times.size else float("nan")
                ),
                "median_abs_distance_to_clinical_sec": (
                    float(np.median(np.abs(clinical_times)))
                    if clinical_times.size
                    else float("nan")
                ),
                "fraction_t_best_within_clinical_5s": (
                    float(np.mean(np.abs(clinical_times) <= 5.0))
                    if clinical_times.size
                    else float("nan")
                ),
                "n_accepted_t_best_with_prototype": len(prototype),
                "prototype_coherence_median": (
                    float(np.median(_finite(prototype["prototype_coherence"])))
                    if len(prototype)
                    else float("nan")
                ),
                "bootstrap_width_median_sec": (
                    float(np.median(widths)) if widths.size else float("nan")
                ),
                "selection_consistency_1s_median": (
                    float(np.median(consistency)) if consistency.size else float("nan")
                ),
                "score_margin_median": (
                    float(np.median(margins)) if margins.size else float("nan")
                ),
                "median_abs_shift_from_v1p1_sec": (
                    float(np.median(np.abs(shifts))) if shifts.size else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def build_cohort_summary(events: pd.DataFrame, subjects: pd.DataFrame) -> dict:
    counts = events["phenotype_status"].value_counts()
    timing_counts = events["timing_status"].value_counts()
    candidates = events[_bool_series(events["has_candidate_t"])]
    accepted = events[_bool_series(events["has_accepted_t_best"])]
    accepted_times = _finite(accepted["t_spectral_best_rel_eeg_sec"])
    frac_q = _q(subjects["fraction_phenotype_present"])
    subject_t_q = _q(subjects["t_best_median_rel_eeg_sec"])
    width_q = _q(subjects["bootstrap_width_median_sec"])
    consistency_q = _q(subjects["selection_consistency_1s_median"])
    has_clinical = (
        bool(_bool_series(events["clinical_onset_available"]).any())
        if "clinical_onset_available" in events
        else True
    )
    subject_clinical_t_q = (
        _q(subjects["t_best_median_rel_clinical_sec"])
        if has_clinical
        else None
    )
    clinical_distance = _finite(subjects["median_abs_distance_to_clinical_sec"])
    compare = subjects[
        np.isfinite(subjects["median_abs_distance_to_eeg_sec"])
        & np.isfinite(subjects["median_abs_distance_to_clinical_sec"])
    ]
    return {
        "analysis_version": "topic5_tspectral_subject_v1p2",
        "status": "algorithmic_broadband_onset_manual_review_pending",
        "unit_for_cohort_summary": "subject",
        "annotation_mode_counts": (
            {str(k): int(v) for k, v in events["annotation_mode"].value_counts().items()}
            if "annotation_mode" in events
            else {"eeg_and_clinical": int(len(events))}
        ),
        "cache_tier_counts": (
            {str(k): int(v) for k, v in events["cache_tier"].value_counts().items()}
            if "cache_tier" in events
            else {}
        ),
        "claim_boundary": (
            "Accepted T_spectral requires both a sustained broadband episode and "
            "support from a recurrent within-patient timing mode; phenotype-absent, "
            "prior-only, temporally unanchored, and no-template seizures retain no "
            "accepted time."
        ),
        "n_subjects": int(len(subjects)),
        "n_seizures": int(len(events)),
        "pooled_event_counts_descriptive": {
            key: int(counts.get(key, 0)) for key in STATUS_ORDER
        },
        "pooled_timing_status_counts_descriptive": {
            str(key): int(value) for key, value in timing_counts.items()
        },
        "n_candidate_t": int(len(candidates)),
        "n_accepted_t_best": int(len(accepted)),
        "subject_fraction_phenotype_present_q25_median_q75": frac_q,
        "subject_t_best_median_rel_eeg_q25_median_q75_sec": subject_t_q,
        "subject_t_best_median_rel_clinical_q25_median_q75_sec": subject_clinical_t_q,
        "subject_bootstrap_width_median_q25_median_q75_sec": width_q,
        "subject_selection_consistency_median_q25_median_q75": consistency_q,
        "accepted_event_timing_descriptive": {
            "q25_median_q75_rel_eeg_sec": _q(
                accepted["t_spectral_best_rel_eeg_sec"]
            ),
            "fraction_within_eeg_1s": (
                float(np.mean(np.abs(accepted_times) <= 1.0))
                if accepted_times.size
                else float("nan")
            ),
            "fraction_within_eeg_2s": (
                float(np.mean(np.abs(accepted_times) <= 2.0))
                if accepted_times.size
                else float("nan")
            ),
            "fraction_within_eeg_5s": (
                float(np.mean(np.abs(accepted_times) <= 5.0))
                if accepted_times.size
                else float("nan")
            ),
        },
        "annotation_alignment_descriptive_not_independent": {
            "reason": (
                "EEG and clinical annotations participate in episode-to-seizure assignment; "
                "distances describe offsets but are not independent validation."
                if has_clinical
                else "Yuquan has EEG-only annotation; no clinical-onset distance or EEG-vs-clinical comparison is defined."
            ),
            "n_subjects_with_accepted_t": int(
                np.sum(np.isfinite(subjects["t_best_median_rel_eeg_sec"]))
            ),
            "median_subject_abs_distance_to_eeg_sec": float(
                np.nanmedian(subjects["median_abs_distance_to_eeg_sec"])
            ),
            "median_subject_abs_distance_to_clinical_sec": float(
                np.median(clinical_distance)
            ) if clinical_distance.size else None,
            "n_subjects_eeg_closer": int(
                np.sum(
                    compare["median_abs_distance_to_eeg_sec"]
                    < compare["median_abs_distance_to_clinical_sec"]
                )
            ),
            "n_subjects_clinical_closer": int(
                np.sum(
                    compare["median_abs_distance_to_clinical_sec"]
                    < compare["median_abs_distance_to_eeg_sec"]
                )
            ),
        },
        "prototype": {
            "n_accepted_with_loso_prototype": int(
                np.sum(_bool_series(accepted["prototype_used"]))
            ),
            "n_accepted_generic_fallback": int(
                np.sum(~_bool_series(accepted["prototype_used"]))
            ),
        },
    }


def _subject_label(value: str) -> str:
    return str(value).replace("epilepsiae_", "E").replace("yuquan_", "Y-")


def _dataset_tag(events: pd.DataFrame) -> str:
    datasets = sorted({str(value).split("_", 1)[0] for value in events["subject"]})
    return datasets[0] if len(datasets) == 1 else "combined"


def plot_subject_raster(
    events: pd.DataFrame, subjects: pd.DataFrame, path: Path, *, dataset_tag: str
) -> None:
    order = subjects.sort_values(
        ["t_best_median_rel_eeg_sec", "subject"], na_position="last"
    )["subject"].tolist()
    fig, ax = plt.subplots(figsize=(11.5, max(6.0, 0.40 * len(order))))
    accepted_labeled = False
    manual_labeled = False
    for yi, subject in enumerate(order):
        use = events[events["subject"] == subject]
        accepted = use[_bool_series(use["has_accepted_t_best"])]
        x = _finite(accepted["t_spectral_best_rel_eeg_sec"])
        if x.size:
            ax.scatter(
                x,
                np.full(x.size, yi),
                s=25,
                alpha=0.78,
                color="#2C7FB8",
                label="accepted subject-recurrent" if not accepted_labeled else None,
            )
            accepted_labeled = True
        manual = use[
            _bool_series(use["has_candidate_t"])
            & ~_bool_series(use["has_accepted_t_best"])
        ]
        manual_x = _finite(manual["t_spectral_candidate_rel_eeg_sec"])
        if manual_x.size:
            ax.scatter(
                manual_x,
                np.full(manual_x.size, yi),
                s=28,
                alpha=0.85,
                color="#E17C05",
                marker="x",
                label="manual-only candidate" if not manual_labeled else None,
            )
            manual_labeled = True
        ax.text(
            1.005,
            yi,
            f"no accepted {len(use) - len(accepted)}/{len(use)}",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=7,
            color="0.35",
        )
    ax.axvline(0.0, color="#7A4F9A", ls=":", lw=1.1, label="EEG onset")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([_subject_label(x) for x in order], fontsize=8)
    ax.set_xlabel("accepted patient-specific T_spectral relative to EEG onset (s)")
    ax.set_title(f"{dataset_tag.title()} v1.2 broadband-onset times; NA events are retained", loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_cohort_overview(
    subjects: pd.DataFrame, path: Path, *, dataset_tag: str
) -> None:
    use = subjects.sort_values(
        ["fraction_phenotype_present", "subject"], ascending=[False, True]
    ).reset_index(drop=True)
    labels = [_subject_label(x) for x in use["subject"]]
    x = np.arange(len(use))
    fig, axs = plt.subplots(2, 2, figsize=(13.2, 9.3), gridspec_kw={"hspace": 0.42})

    ax = axs[0, 0]
    bottom = np.zeros(len(use))
    for status in STATUS_ORDER:
        vals = use[f"n_{status}"].to_numpy(dtype=float) / use["n_seizures"].to_numpy(
            dtype=float
        )
        ax.bar(x, vals, bottom=bottom, color=STATUS_COLORS[status], label=status)
        bottom += vals
    ax.set_ylim(0, 1.02)
    ax.set_xticks(x, labels, rotation=90, fontsize=7)
    ax.set_ylabel("within-subject seizure fraction")
    ax.set_title("a  Broadband phenotype eligibility", loc="left")
    ax.legend(frameon=False, fontsize=7)

    ax = axs[0, 1]
    for xi, row in use.iterrows():
        if np.isfinite(row["t_best_median_rel_eeg_sec"]):
            ax.plot(
                [xi, xi],
                [row["t_best_q25_rel_eeg_sec"], row["t_best_q75_rel_eeg_sec"]],
                color="0.5",
                lw=1,
            )
            ax.scatter(xi, row["t_best_median_rel_eeg_sec"], color="#2C7FB8", s=27)
    ax.axhline(0, color="#7A4F9A", ls=":", lw=1)
    ax.set_xticks(x, labels, rotation=90, fontsize=7)
    ax.set_ylabel("T_spectral relative to EEG onset (s)")
    ax.set_title("b  Patient median and within-patient IQR", loc="left")

    ax = axs[1, 0]
    ax.scatter(
        use["fraction_phenotype_present"],
        use["fraction_t_best_within_eeg_5s"],
        color="#2C7FB8",
        s=35,
    )
    for _, row in use.iterrows():
        if np.isfinite(row["fraction_t_best_within_eeg_5s"]):
            ax.text(
                row["fraction_phenotype_present"],
                row["fraction_t_best_within_eeg_5s"],
                _subject_label(row["subject"]),
                fontsize=6,
            )
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("fraction with broadband phenotype")
    ax.set_ylabel("accepted T within +/-5 s of EEG onset")
    ax.set_title("c  Eligibility versus EEG-time concentration", loc="left")

    ax = axs[1, 1]
    ax.scatter(
        use["bootstrap_width_median_sec"],
        use["selection_consistency_1s_median"],
        color="#E17C05",
        s=35,
    )
    for _, row in use.iterrows():
        if np.isfinite(row["bootstrap_width_median_sec"]):
            ax.text(
                row["bootstrap_width_median_sec"],
                row["selection_consistency_1s_median"],
                _subject_label(row["subject"]),
                fontsize=6,
            )
    ax.set_xlabel("subject median resampling interval width (s)")
    ax.set_ylabel("subject median selection consistency within 1 s")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("d  Timing-selection stability", loc="left")

    for ax in axs.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{dataset_tag.title()} patient-specific T_spectral v1.2", fontsize=14, y=0.99)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_readmes(root: Path, *, dataset_tag: str) -> None:
    fig_dir = root / "figures/cohort"
    fig_dir.mkdir(parents=True, exist_ok=True)
    (fig_dir / "README.md").write_text(
        "# Patient-specific T_spectral cohort figures\n\n"
        f"### {dataset_tag}_subject_tbest_raster.png\n\n"
        "逐患者展示患者内时间模式复现后接受的 T_spectral（蓝点），以及有宽带 episode 但缺少患者内时间支持的 manual-only candidate（橙色叉）。每行右侧明确标出未接受时刻的 seizure 数。\n\n"
        "**关注点**：同一患者内时刻是否形成复现模式，以及孤立晚期宽带状态是否被保守降级。\n\n"
        f"### {dataset_tag}_subject_tbest_cohort_overview.png\n\n"
        "Panel a 展示宽带表型是否存在，panel b 展示患者内 T_spectral 中位数和 IQR，panel c 比较表型覆盖率与 EEG onset 附近 5 s 的集中度，panel d 展示候选选择的重采样稳定性。cohort 推断单位为 subject，pooled seizure 数仅作描述。\n\n"
        "**关注点**：没有宽带增强的发作不得被强行赋时刻；该时刻是算法性宽带能量起点，不等同于普适的临床 seizure onset。\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        f"# {dataset_tag.title()} patient-specific T_spectral v1.2\n\n"
        "逐 seizure 主表为 `per_seizure_subject_refined_onset.csv`。`phenotype_present` 事件先获得候选时刻；只有患者内时间模式复现的事件才写入 `t_spectral_best_rel_eeg_sec`。无表型、prior-only、缺少患者时间模板和时间上孤立的事件都不强制写入 accepted onset。\n\n"
        "`subject_cohort_summary.csv` 和 `cohort_summary.json` 分别给出患者级与 cohort 级描述。`figures/per_seizure/` 保存原始通道波形和多频带诊断图，`figures/cohort/` 保存汇总图。当前结果仍标记为 algorithmic/manual-review-pending。\n",
        encoding="utf-8",
    )


def run(root: Path) -> Path:
    event_path = root / "per_seizure_subject_refined_onset.csv"
    events = pd.read_csv(event_path)
    dataset_tag = _dataset_tag(events)
    subjects = build_subject_summary(events)
    cohort = build_cohort_summary(events, subjects)
    subjects.to_csv(root / "subject_cohort_summary.csv", index=False)
    (root / "cohort_summary.json").write_text(
        json.dumps(cohort, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    fig_dir = root / "figures/cohort"
    plot_subject_raster(
        events,
        subjects,
        fig_dir / f"{dataset_tag}_subject_tbest_raster.png",
        dataset_tag=dataset_tag,
    )
    plot_cohort_overview(
        subjects,
        fig_dir / f"{dataset_tag}_subject_tbest_cohort_overview.png",
        dataset_tag=dataset_tag,
    )
    write_readmes(root, dataset_tag=dataset_tag)
    return root / "cohort_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(run(args.root.resolve()))


if __name__ == "__main__":
    main()
