#!/usr/bin/env python3
"""Summarize algorithmic-v1 T_spectral per-seizure and cohort outputs."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_ROOT = (
    ROOT
    / "results/topic5_ictal_recruitment/peri_onset_energy_timing/epilepsiae/refinement_v1p2/seed_v1p1"
)
TIER_ORDER = (
    "primary_precise",
    "sensitivity_stable_candidate",
    "exploratory_unstable_candidate",
    "separate_prior_episode",
    "no_detectable_broadband_transition",
)
TIER_COLORS = {
    "primary_precise": "#2CA02C",
    "sensitivity_stable_candidate": "#ECA82C",
    "exploratory_unstable_candidate": "#C77C2B",
    "separate_prior_episode": "#7A4F9A",
    "no_detectable_broadband_transition": "#D0D0D0",
}


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _float_or_nan(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _bool(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def timing_tier(row: dict) -> str:
    status = str(row["auto_status"])
    if status == "confirmed_precise_T":
        return "primary_precise"
    if status == "broadband_but_imprecise_T":
        return (
            "sensitivity_stable_candidate"
            if _bool(row.get("auto_stable_candidate_time"))
            else "exploratory_unstable_candidate"
        )
    if status == "separate_prior_episode":
        return status
    return "no_detectable_broadband_transition"


def build_refined_rows(manifest_rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for row in manifest_rows:
        tier = timing_tier(row)
        candidate = _float_or_nan(row.get("auto_t_spectral_rel_eeg_sec"))
        eeg_rel_clin = _float_or_nan(row.get("eeg_onset_rel_clinical_sec"))
        clinical_rel_eeg = -eeg_rel_clin
        has_candidate = np.isfinite(candidate)
        primary = tier == "primary_precise"
        sensitivity = tier in {"primary_precise", "sensitivity_stable_candidate"}
        out.append(
            {
                "analysis_version": row.get("analysis_version", "topic5_tspectral_v1p1"),
                "subject": row["subject"],
                "seizure_idx": int(row["seizure_idx"]),
                "seizure_id": row["seizure_id"],
                "timing_tier": tier,
                "manual_adjudication_pending": True,
                "candidate_t_spectral_rel_eeg_sec": candidate if has_candidate else "",
                "candidate_t_spectral_rel_clinical_sec": (
                    candidate + eeg_rel_clin if has_candidate else ""
                ),
                "candidate_ci_q05_rel_eeg_sec": row.get("auto_bootstrap_q05_rel_eeg_sec", ""),
                "candidate_ci_q95_rel_eeg_sec": row.get("auto_bootstrap_q95_rel_eeg_sec", ""),
                "candidate_ci_width_sec": row.get("auto_bootstrap_ci_width_sec", ""),
                "primary_t_spectral_rel_eeg_sec": candidate if primary else "",
                "sensitivity_t_spectral_rel_eeg_sec": candidate if sensitivity else "",
                "distance_to_eeg_sec": abs(candidate) if has_candidate else "",
                "distance_to_clinical_sec": (
                    abs(candidate - clinical_rel_eeg) if has_candidate else ""
                ),
                "eeg_onset_rel_clinical_sec": eeg_rel_clin,
                "complete_change_gate": row.get("auto_complete_change_gate", ""),
                "n_step_bands": row.get("auto_n_step_bands", ""),
                "n_step_contacts": row.get("auto_n_step_contacts", ""),
                "low_step_supported": row.get("auto_low_step_supported", ""),
                "high_step_supported": row.get("auto_high_step_supported", ""),
                "n_detected_episodes": int(row["auto_n_episodes"]),
                "n_prior_episodes": int(row["auto_n_prior_episodes"]),
                "blind_figure": row.get("blind_figure", ""),
                "revealed_figure": row.get("revealed_figure", ""),
            }
        )
    return sorted(out, key=lambda row: (row["subject"], row["seizure_idx"]))


def _quartiles(values) -> tuple[float, float, float]:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if not x.size:
        return float("nan"), float("nan"), float("nan")
    return tuple(float(v) for v in np.quantile(x, [0.25, 0.5, 0.75]))


def build_subject_summary(refined_rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in refined_rows:
        grouped[str(row["subject"])].append(row)
    out: list[dict] = []
    for subject in sorted(grouped):
        rows = grouped[subject]
        count = Counter(row["timing_tier"] for row in rows)
        primary = np.asarray(
            [_float_or_nan(row["primary_t_spectral_rel_eeg_sec"]) for row in rows], dtype=float
        )
        primary = primary[np.isfinite(primary)]
        sensitivity = np.asarray(
            [_float_or_nan(row["sensitivity_t_spectral_rel_eeg_sec"]) for row in rows], dtype=float
        )
        sensitivity = sensitivity[np.isfinite(sensitivity)]
        p_q25, p_med, p_q75 = _quartiles(primary)
        s_q25, s_med, s_q75 = _quartiles(sensitivity)
        primary_rows = [row for row in rows if row["timing_tier"] == "primary_precise"]
        med_abs_eeg = (
            float(np.median([float(row["distance_to_eeg_sec"]) for row in primary_rows]))
            if primary_rows
            else float("nan")
        )
        med_abs_clin = (
            float(np.median([float(row["distance_to_clinical_sec"]) for row in primary_rows]))
            if primary_rows
            else float("nan")
        )
        out.append(
            {
                "subject": subject,
                "n_seizures": len(rows),
                **{f"n_{tier}": count[tier] for tier in TIER_ORDER},
                **{f"fraction_{tier}": count[tier] / len(rows) for tier in TIER_ORDER},
                "primary_t_rel_eeg_q25_sec": p_q25,
                "primary_t_rel_eeg_median_sec": p_med,
                "primary_t_rel_eeg_q75_sec": p_q75,
                "primary_t_rel_eeg_iqr_sec": p_q75 - p_q25 if primary.size else float("nan"),
                "sensitivity_t_rel_eeg_q25_sec": s_q25,
                "sensitivity_t_rel_eeg_median_sec": s_med,
                "sensitivity_t_rel_eeg_q75_sec": s_q75,
                "primary_median_abs_distance_to_eeg_sec": med_abs_eeg,
                "primary_median_abs_distance_to_clinical_sec": med_abs_clin,
                "primary_fraction_within_eeg_1s": (
                    float(np.mean(np.abs(primary) <= 1.0)) if primary.size else float("nan")
                ),
                "primary_fraction_within_eeg_2s": (
                    float(np.mean(np.abs(primary) <= 2.0)) if primary.size else float("nan")
                ),
                "primary_fraction_within_eeg_5s": (
                    float(np.mean(np.abs(primary) <= 5.0)) if primary.size else float("nan")
                ),
            }
        )
    return out


def _bootstrap_subject_mean(values, *, n_boot=10000, seed=20260714) -> list[float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not x.size:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    boot = np.mean(rng.choice(x, size=(int(n_boot), x.size), replace=True), axis=1)
    return [float(v) for v in np.quantile(boot, [0.025, 0.975])]


def _bootstrap_subject_median(values, *, n_boot=10000, seed=20260714) -> list[float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not x.size:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    boot = np.median(rng.choice(x, size=(int(n_boot), x.size), replace=True), axis=1)
    return [float(v) for v in np.quantile(boot, [0.025, 0.975])]


def build_cohort_summary(refined_rows: list[dict], subject_rows: list[dict]) -> dict:
    count = Counter(row["timing_tier"] for row in refined_rows)
    primary_fraction = [row["fraction_primary_precise"] for row in subject_rows]
    sensitivity_fraction = [
        row["fraction_primary_precise"] + row["fraction_sensitivity_stable_candidate"]
        for row in subject_rows
    ]
    primary_subject_medians = [row["primary_t_rel_eeg_median_sec"] for row in subject_rows]
    sensitivity_subject_medians = [
        row["sensitivity_t_rel_eeg_median_sec"] for row in subject_rows
    ]
    paired = [
        row
        for row in subject_rows
        if row["n_primary_precise"] >= 2
        and np.isfinite(row["primary_median_abs_distance_to_eeg_sec"])
        and np.isfinite(row["primary_median_abs_distance_to_clinical_sec"])
    ]
    eeg_distance = np.asarray(
        [row["primary_median_abs_distance_to_eeg_sec"] for row in paired], dtype=float
    )
    clinical_distance = np.asarray(
        [row["primary_median_abs_distance_to_clinical_sec"] for row in paired], dtype=float
    )
    delta = clinical_distance - eeg_distance
    nonzero = delta[np.abs(delta) > 1e-12]
    if nonzero.size:
        wilcoxon_p = float(wilcoxon(clinical_distance, eeg_distance, alternative="greater").pvalue)
        n_eeg_closer = int(np.sum(nonzero > 0.0))
        sign_p = float(binomtest(n_eeg_closer, nonzero.size, 0.5, alternative="greater").pvalue)
    else:
        wilcoxon_p = sign_p = 1.0
        n_eeg_closer = 0
    return {
        "status": "algorithmic_v1p1_manual_adjudication_pending",
        "unit_for_cohort_summary": "subject",
        "n_subjects": len(subject_rows),
        "n_seizures": len(refined_rows),
        "pooled_event_counts_descriptive": {tier: count[tier] for tier in TIER_ORDER},
        "primary_precise": {
            "subject_fraction_q25_median_q75": list(_quartiles(primary_fraction)),
            "subject_fraction_mean": float(np.mean(primary_fraction)),
            "subject_fraction_mean_bootstrap_95ci": _bootstrap_subject_mean(primary_fraction),
            "n_subjects_with_at_least_one": int(sum(row["n_primary_precise"] >= 1 for row in subject_rows)),
            "n_subjects_with_at_least_two": int(sum(row["n_primary_precise"] >= 2 for row in subject_rows)),
        },
        "primary_plus_stable_sensitivity": {
            "subject_fraction_q25_median_q75": list(_quartiles(sensitivity_fraction)),
            "subject_fraction_mean": float(np.mean(sensitivity_fraction)),
            "subject_fraction_mean_bootstrap_95ci": _bootstrap_subject_mean(
                sensitivity_fraction, seed=20260715
            ),
        },
        "timing_relative_to_eeg_onset_subject_level": {
            "primary_subject_median_q25_median_q75_sec": list(
                _quartiles(primary_subject_medians)
            ),
            "primary_median_of_subject_medians_bootstrap_95ci_sec": (
                _bootstrap_subject_median(primary_subject_medians, seed=20260716)
            ),
            "primary_n_subjects_with_defined_median": int(
                np.sum(np.isfinite(np.asarray(primary_subject_medians, dtype=float)))
            ),
            "sensitivity_subject_median_q25_median_q75_sec": list(
                _quartiles(sensitivity_subject_medians)
            ),
            "sensitivity_median_of_subject_medians_bootstrap_95ci_sec": (
                _bootstrap_subject_median(sensitivity_subject_medians, seed=20260717)
            ),
            "sensitivity_n_subjects_with_defined_median": int(
                np.sum(np.isfinite(np.asarray(sensitivity_subject_medians, dtype=float)))
            ),
        },
        "annotation_alignment_descriptive_not_independent": {
            "reason": "annotations are used only for episode-to-seizure assignment, so distance comparison is descriptive",
            "n_subjects_with_at_least_two_primary": len(paired),
            "median_subject_abs_distance_to_eeg_sec": (
                float(np.median(eeg_distance)) if eeg_distance.size else float("nan")
            ),
            "median_subject_abs_distance_to_clinical_sec": (
                float(np.median(clinical_distance)) if clinical_distance.size else float("nan")
            ),
            "n_eeg_closer": n_eeg_closer,
            "n_clinical_closer": int(np.sum(nonzero < 0.0)),
            "wilcoxon_eeg_closer_p": wilcoxon_p,
            "sign_eeg_closer_p": sign_p,
        },
    }


def _plot_subject_raster(refined_rows: list[dict], subject_rows: list[dict], out_path: Path) -> None:
    order = sorted(subject_rows, key=lambda row: (row["primary_t_rel_eeg_median_sec"], row["subject"]))
    subjects = [row["subject"] for row in order]
    fig, ax = plt.subplots(figsize=(11.5, max(6.5, 0.38 * len(subjects))))
    marker = {
        "primary_precise": "o",
        "sensitivity_stable_candidate": "^",
        "exploratory_unstable_candidate": "x",
    }
    labeled: set[str] = set()
    for yi, subject in enumerate(subjects):
        rows = [row for row in refined_rows if row["subject"] == subject]
        for tier in marker:
            values = [
                _float_or_nan(row["candidate_t_spectral_rel_eeg_sec"])
                for row in rows
                if row["timing_tier"] == tier
            ]
            values = [value for value in values if np.isfinite(value)]
            if values:
                ax.scatter(
                    values,
                    np.full(len(values), yi),
                    color=TIER_COLORS[tier],
                    marker=marker[tier],
                    s=25,
                    alpha=0.78,
                    label=tier if tier not in labeled else None,
                )
                labeled.add(tier)
    ax.axvline(0.0, color="#7A4F9A", ls=":", lw=1.1, label="EEG onset")
    ax.set_yticks(np.arange(len(subjects)))
    ax.set_yticklabels([subject.replace("epilepsiae_", "E") for subject in subjects], fontsize=8)
    ax.set_xlabel("candidate T_spectral relative to EEG onset (s)")
    ax.set_title("Epilepsiae algorithmic-v1.1 spectral-onset candidates by subject", loc="left")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_cohort_overview(subject_rows: list[dict], out_path: Path) -> None:
    order = sorted(subject_rows, key=lambda row: (-row["fraction_primary_precise"], row["subject"]))
    labels = [row["subject"].replace("epilepsiae_", "E") for row in order]
    x = np.arange(len(order))
    fig, axs = plt.subplots(2, 2, figsize=(13.0, 9.4), gridspec_kw={"hspace": 0.40, "wspace": 0.32})

    ax = axs[0, 0]
    bottom = np.zeros(len(order), dtype=float)
    for tier in TIER_ORDER:
        values = np.asarray([row[f"fraction_{tier}"] for row in order], dtype=float)
        ax.bar(x, values, bottom=bottom, color=TIER_COLORS[tier], label=tier, width=0.82)
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("within-subject seizure fraction")
    ax.set_title("a  Event-level outcome composition", loc="left")
    ax.legend(frameon=False, fontsize=6.8, ncol=2, loc="upper right")

    ax = axs[0, 1]
    for xi, row in enumerate(order):
        med = row["primary_t_rel_eeg_median_sec"]
        if np.isfinite(med):
            ax.plot(
                [xi, xi],
                [row["primary_t_rel_eeg_q25_sec"], row["primary_t_rel_eeg_q75_sec"]],
                color="0.45",
                lw=1.0,
            )
            ax.scatter(xi, med, color=TIER_COLORS["primary_precise"], s=28)
    ax.axhline(0.0, color="#7A4F9A", ls=":", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("primary T_spectral rel. EEG (s)")
    ax.set_title("b  Subject median and IQR", loc="left")

    ax = axs[1, 0]
    paired = [row for row in order if row["n_primary_precise"] >= 2]
    for idx, row in enumerate(paired):
        eeg = row["primary_median_abs_distance_to_eeg_sec"]
        clin = row["primary_median_abs_distance_to_clinical_sec"]
        ax.plot([0, 1], [eeg, clin], color="0.70", lw=0.9)
        ax.scatter(0, eeg, color="#7A4F9A", s=25)
        ax.scatter(1, clin, color="#C23B22", s=25)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["EEG onset", "clinical onset"])
    ax.set_ylabel("subject median absolute distance (s)")
    ax.set_title("c  Annotation distance (descriptive)", loc="left")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axs[1, 1]
    primary_fraction = [row["fraction_primary_precise"] for row in order]
    sensitivity_fraction = [
        row["fraction_primary_precise"] + row["fraction_sensitivity_stable_candidate"]
        for row in order
    ]
    ax.scatter(primary_fraction, sensitivity_fraction, color="#4C78A8", s=35)
    ax.plot([0, 1], [0, 1], color="0.75", ls="--", lw=0.8)
    for px, py, label in zip(primary_fraction, sensitivity_fraction, labels):
        ax.text(px, py, label, fontsize=6, ha="left", va="bottom")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("primary precise fraction")
    ax.set_ylabel("primary + stable sensitivity fraction")
    ax.set_title("d  Precision-gate sensitivity", loc="left")

    for ax in axs.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Epilepsiae T_spectral algorithmic-v1.1 cohort summary", fontsize=14, y=0.99)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _write_readme(fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    (fig_dir / "README.md").write_text(
        "# T_spectral cohort figures\n\n"
        "### epilepsiae_tspectral_subject_raster.png\n\n"
        "逐患者展示 primary precise、稳定 sensitivity candidate 和不稳定 candidate 的数据定义时刻，横轴为相对 EEG onset。未形成宽带 episode 和 prior-only 事件不强行放置时间点。\n\n"
        "**关注点**：检查同一患者内时刻是否聚集，以及不同患者是否存在系统性 annotation offset。\n\n"
        "### epilepsiae_tspectral_cohort_overview.png\n\n"
        "Panel a 为逐患者事件结局构成，panel b 为 primary 时刻的患者内中位数与 IQR，panel c 描述其距 EEG/clinical onset 的差异，panel d 展示 precision gate 对覆盖率的影响。所有 cohort 汇总以 subject 为单位；pooled seizure 数只作描述。\n\n"
        "**关注点**：automatic v1.1 尚待人工 adjudication；annotation distance 不是独立验证，因为 annotation 参与 episode assignment。\n",
        encoding="utf-8",
    )


def _write_root_readme(root: Path) -> None:
    (root / "README.md").write_text(
        "# Epilepsiae T_spectral algorithmic-v1.1 cohort\n\n"
        "本目录包含逐 seizure 的宽带频谱转变检测、精修时刻和审查图。"
        "`per_seizure_refined_onset.csv` 是逐发作主表；`subject_refined_onset_summary.csv` "
        "和 `cohort_summary.json` 分别给出患者级与队列级汇总。\n\n"
        "`figures/blind/` 隐藏 EEG/clinical onset，供盲审；`figures/revealed/` 显示标注与自动"
        " episode，供最终 adjudication。`figures/cohort/` 汇总患者内时间规律和检测覆盖率。\n\n"
        "当前状态为 algorithmic v1.1，人工逐事件 adjudication 尚未完成。只有 "
        "`primary_precise` 写入 primary 精修时刻；`sensitivity_stable_candidate` 仅进入"
        "敏感性分析，不强制升级为 onset。\n",
        encoding="utf-8",
    )


def run(root: Path) -> None:
    manifest_path = root / "review_manifest.csv"
    manifest_rows = list(csv.DictReader(manifest_path.open(encoding="utf-8")))
    refined = build_refined_rows(manifest_rows)
    subjects = build_subject_summary(refined)
    cohort = build_cohort_summary(refined, subjects)
    _write_csv(root / "per_seizure_refined_onset.csv", refined)
    _write_csv(root / "subject_refined_onset_summary.csv", subjects)
    (root / "cohort_summary.json").write_text(
        json.dumps(cohort, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    fig_dir = root / "figures/cohort"
    _plot_subject_raster(refined, subjects, fig_dir / "epilepsiae_tspectral_subject_raster.png")
    _plot_cohort_overview(subjects, fig_dir / "epilepsiae_tspectral_cohort_overview.png")
    _write_readme(fig_dir)
    _write_root_readme(root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    run(args.root)
    print(args.root / "cohort_summary.json")


if __name__ == "__main__":
    main()
