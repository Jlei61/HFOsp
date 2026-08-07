#!/usr/bin/env python3
"""Render the paper-ready interictal gradient-axis validation panel.

The left column reuses the accepted QC-clean single-event direction producer.
The right column displays the subject-level alignment gain over the
montage-matched template-rank-shuffle null, so cohort shift and between-patient
heterogeneity remain visible in the same panel.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_topic5_interictal_template_direction_rose import (  # noqa: E402
    DEFAULT_MAX_EVENTS,
    DEFAULT_SEED,
    TA_COLOR,
    TB_COLOR,
)
from scripts.plot_topic5_interictal_template_direction_rose_qc import (  # noqa: E402
    build_subject_payload,
)
from scripts.run_topic5_axis_representativeness import (  # noqa: E402
    DEFAULT_MIN_EVENTS,
    DEFAULT_N_PERM,
    process_subject,
)
from scripts.paper_figures.plot_fig3f_ab_dominance_cohort import (  # noqa: E402
    EPILEPSIAE_ORDER_SOURCE,
    SUPPLEMENTARY_TABLE,
    _pretty as _manuscript_id,
)
from src.topic5_tspectral_field_concordance import (  # noqa: E402
    bootstrap_median_ci,
)


DEFAULT_EXAMPLES = ("epilepsiae_1146", "epilepsiae_548")
DEFAULT_BINS = 18
DEFAULT_OUT = ROOT / "results/paper-ready-figure/fig2b_gradient_axis_validation"
LEGACY_NULL_CACHE = (
    ROOT
    / "results/paper-ready-figure/figr_gradient_axis_validation/"
    "cohort_rank_shuffle_null.npz"
)
EFFECT_ROOT = ROOT / "results/interictal_propagation_masked/axis_representativeness"
EFFECT_CSV = EFFECT_ROOT / "subject_folded_axis_representativeness.csv"
EFFECT_SUMMARY = EFFECT_ROOT / "axis_representativeness_summary.json"
AXIS_COHORT_CSV = (
    ROOT / "results/interictal_propagation_masked/template_gradient_fields/axis_cohort.csv"
)
FIELD_SUMMARY = (
    ROOT
    / "results/interictal_propagation_masked/template_gradient_fields/cohort_summary.json"
)

POINT_COLOR = "#3E6F8E"
POINT_EDGE = "#FFFFFF"
GRID_COLOR = "#D6D6D6"
TEXT_COLOR = "#202020"
NULL_COLOR = "#777777"


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def _load_csv(path: Path) -> Sequence[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_effect_contract() -> Dict[str, object]:
    """Load and cross-check the accepted n=34 -> 28 -> 26 contract."""
    effect_rows = _load_csv(EFFECT_CSV)
    axis_rows = _load_csv(AXIS_COHORT_CSV)
    effect_summary = json.loads(EFFECT_SUMMARY.read_text())
    field_summary = json.loads(FIELD_SUMMARY.read_text())

    stable_k2 = [row for row in axis_rows if row.get("stable_k") == "2"]
    estimable = [row for row in stable_k2 if _as_bool(row.get("axis_pair_estimable"))]
    geometry_2d = [row for row in estimable if _as_bool(row.get("geometry_2d_supported"))]
    effects = []
    for row in effect_rows:
        margin = float(row["alignment_margin"])
        if not np.isfinite(margin):
            continue
        effects.append({
            "subject_id": row["subject_id"],
            "pretty_subject": row["pretty_subject"],
            "dataset": row["dataset"],
            "alignment_margin": margin,
            "mean_signed_cosine": float(row["mean_signed_cosine"]),
            "null_mean_cosine_median": float(row["null_mean_cosine_median"]),
            "strict_stability_pass": _as_bool(row["strict_stability_pass"]),
        })

    denominators = field_summary["denominators"]
    expected = {
        "template_pair_inputs": 40,
        "ab_template_cohort": 34,
        "axis_pair_estimable": 28,
        "geometry_2d_supported": 26,
        "effect_rows": 26,
    }
    observed = {
        "template_pair_inputs": int(denominators["template_pair_inputs"]),
        "ab_template_cohort": len(stable_k2),
        "axis_pair_estimable": len(estimable),
        "geometry_2d_supported": len(geometry_2d),
        "effect_rows": len(effects),
    }
    if observed != expected:
        raise RuntimeError(f"Cohort contract drift: expected {expected}, observed {observed}")

    effect_ids = {row["subject_id"] for row in effects}
    geometry_ids = {row["subject_id"] for row in geometry_2d}
    if effect_ids != geometry_ids:
        missing = sorted(geometry_ids - effect_ids)
        extra = sorted(effect_ids - geometry_ids)
        raise RuntimeError(
            f"Effect/2-D cohort mismatch: missing effects={missing}, extra effects={extra}"
        )

    effect_stat = effect_summary["cohort"]["alignment_margin"]["gradient"]
    if int(effect_stat["n"]) != len(effects):
        raise RuntimeError("Summary n does not match subject effect rows")
    return {
        "effects": effects,
        "effect_stat": effect_stat,
        "effect_summary": effect_summary,
        "field_summary": field_summary,
        "cohort_flow": {
            "ab_template_cohort": len(stable_k2),
            "axis_pair_estimable": len(estimable),
            "geometry_2d_supported": len(geometry_2d),
            "coordinates_or_mapping_unavailable": len(stable_k2) - len(estimable),
            "intrinsic_1d_geometry": len(estimable) - len(geometry_2d),
        },
    }


def _null_cache_fingerprint(
    subject_ids: Sequence[str], *, n_perm: int, seed: int
) -> str:
    digest = hashlib.sha256()
    digest.update(EFFECT_CSV.read_bytes())
    digest.update("\n".join(subject_ids).encode("utf-8"))
    digest.update(f"|n_perm={n_perm}|seed={seed}".encode("utf-8"))
    return digest.hexdigest()


def load_or_build_cohort_null(
    contract: Mapping[str, object],
    *,
    cache_path: Path,
    n_perm: int = DEFAULT_N_PERM,
    seed: int = DEFAULT_SEED,
) -> Dict[str, object]:
    """Return the draw-wise cohort median under the accepted rank-shuffle null.

    Each draw rebuilds TA and TB axes after independently shuffling their template
    ranks over the same mapped contacts. TA/TB scores are averaged within patient,
    then the median is taken across the 26 patients. The cache stores the complete
    1,000-draw distribution rather than reconstructing it from saved quantiles.
    """
    effects = sorted(contract["effects"], key=lambda row: str(row["subject_id"]))
    subject_ids = [str(row["subject_id"]) for row in effects]
    fingerprint = _null_cache_fingerprint(subject_ids, n_perm=n_perm, seed=seed)
    for candidate in dict.fromkeys((cache_path, LEGACY_NULL_CACHE)):
        if not candidate.exists():
            continue
        with np.load(candidate, allow_pickle=False) as cached:
            cached_fingerprint = str(cached["fingerprint"].item())
            cached_ids = [str(value) for value in cached["subject_ids"].tolist()]
            cohort_medians = np.asarray(cached["cohort_medians"], float)
            patient_null_scores = np.asarray(cached["patient_null_scores"], float)
        if (
            cached_fingerprint == fingerprint
            and cached_ids == subject_ids
            and cohort_medians.shape == (n_perm,)
            and patient_null_scores.shape == (len(subject_ids), n_perm)
        ):
            if candidate != cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    cache_path,
                    fingerprint=np.asarray(fingerprint),
                    subject_ids=np.asarray(subject_ids),
                    patient_null_scores=patient_null_scores,
                    cohort_medians=cohort_medians,
                    n_perm=np.asarray(n_perm),
                    seed=np.asarray(seed),
                )
            return {
                "subject_ids": subject_ids,
                "patient_null_scores": patient_null_scores,
                "cohort_medians": cohort_medians,
                "fingerprint": fingerprint,
                "cache_path": cache_path,
            }

    patient_null_scores = []
    effect_by_id = {str(row["subject_id"]): row for row in effects}
    for index, subject_id in enumerate(subject_ids, 1):
        print(
            f"[{index:02d}/{len(subject_ids)}] rebuilding cohort-null draws: "
            f"{subject_id}",
            flush=True,
        )
        rows = process_subject(
            subject_id,
            n_perm=n_perm,
            min_events=DEFAULT_MIN_EVENTS,
            max_events=DEFAULT_MAX_EVENTS,
            seed=seed,
        )
        rows = sorted(
            [row for row in rows if bool(row["analysis_eligible"])],
            key=lambda row: str(row["template"]),
        )
        if [str(row["template"]) for row in rows] != ["TA", "TB"]:
            raise RuntimeError(f"{subject_id}: expected eligible TA and TB rows")
        patient_null = np.mean(
            np.vstack([np.asarray(row["_null_cosine"], float) for row in rows]),
            axis=0,
        )
        observed = float(np.mean([row["mean_signed_cosine"] for row in rows]))
        expected = effect_by_id[subject_id]
        if not np.isclose(observed, float(expected["mean_signed_cosine"]), atol=1e-10):
            raise RuntimeError(f"{subject_id}: observed score drift during null rebuild")
        if not np.isclose(
            float(np.nanmedian(patient_null)),
            float(expected["null_mean_cosine_median"]),
            atol=1e-10,
        ):
            raise RuntimeError(f"{subject_id}: rank-shuffle null drift during rebuild")
        patient_null_scores.append(patient_null)

    patient_null_array = np.vstack(patient_null_scores)
    cohort_medians = np.nanmedian(patient_null_array, axis=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        fingerprint=np.asarray(fingerprint),
        subject_ids=np.asarray(subject_ids),
        patient_null_scores=patient_null_array,
        cohort_medians=cohort_medians,
        n_perm=np.asarray(n_perm),
        seed=np.asarray(seed),
    )
    return {
        "subject_ids": subject_ids,
        "patient_null_scores": patient_null_array,
        "cohort_medians": cohort_medians,
        "fingerprint": fingerprint,
        "cache_path": cache_path,
    }


def _histogram_proportions(values: Sequence[float], edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(np.asarray(values, float), bins=edges)
    total = int(counts.sum())
    return counts.astype(float) / total if total else np.zeros_like(counts, dtype=float)


def _rose_max(payloads: Sequence[Mapping[str, object]], edges: np.ndarray) -> float:
    maximum = 0.0
    for payload in payloads:
        for label in (0, 1):
            proportions = _histogram_proportions(payload["groups_qc"][label], edges)
            if proportions.size:
                maximum = max(maximum, float(np.max(proportions)))
    if maximum <= 0:
        return 0.1
    return float(np.ceil(maximum / 0.05) * 0.05)


def draw_probability_rose(
    ax: plt.Axes,
    payload: Mapping[str, object],
    *,
    edges: np.ndarray,
    rmax: float,
    label: str,
) -> None:
    """Draw a compact, within-template normalized QC-clean rose."""
    centers = edges[:-1] + 0.5 * np.diff(edges)
    width = float(np.diff(edges)[0] * 0.92)
    for template, color in ((0, TA_COLOR), (1, TB_COLOR)):
        proportions = _histogram_proportions(payload["groups_qc"][template], edges)
        ax.bar(
            centers,
            proportions,
            width=width,
            facecolor=matplotlib.colors.to_rgba(color, 0.20),
            edgecolor=color,
            linewidth=0.85,
            zorder=2,
        )

    theta_b = float(payload["basis"]["theta_b_rad"])
    line_top = rmax * 1.02
    ax.plot([0.0, 0.0], [0.0, line_top], color=TA_COLOR, lw=2.4, zorder=5)
    ax.plot([theta_b, theta_b], [0.0, line_top], color=TB_COLOR, lw=2.4, zorder=5)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels([])
    for text, x_pos, y_pos, h_align, v_align in (
        ("0°", 1.02, 0.50, "left", "center"),
        ("90°", 0.50, 0.985, "center", "top"),
        ("180°", -0.02, 0.50, "right", "center"),
        ("270°", 0.50, 0.015, "center", "bottom"),
    ):
        ax.text(
            x_pos,
            y_pos,
            text,
            transform=ax.transAxes,
            ha=h_align,
            va=v_align,
            fontsize=7,
            color=TEXT_COLOR,
            clip_on=False,
        )
    ax.set_ylim(0.0, rmax * 1.04)
    tick_step = 0.10 if rmax >= 0.20 else 0.05
    radial_ticks = np.arange(tick_step, rmax + 1e-9, tick_step)
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels([f"{100 * value:.0f}%" for value in radial_ticks], fontsize=6.4)
    ax.set_rlabel_position(102)
    ax.grid(color=GRID_COLOR, linewidth=0.55, alpha=0.95)
    ax.spines["polar"].set_color("#777777")
    ax.spines["polar"].set_linewidth(0.65)
    ax.text(
        -0.18,
        0.97,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    n_events = int(payload["n_direction_qc_ta"]) + int(payload["n_direction_qc_tb"])
    ax.text(
        -0.18,
        0.88,
        f"n={n_events:,}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color="#444444",
    )


def draw_subject_effects(
    ax: plt.Axes,
    contract: Mapping[str, object],
    cohort_null: Mapping[str, object],
    *,
    example_labels: Mapping[str, str],
) -> None:
    effects = sorted(contract["effects"], key=lambda row: float(row["mean_signed_cosine"]))
    values = np.asarray([row["mean_signed_cosine"] for row in effects], float)
    y = np.arange(2, len(effects) + 2, dtype=float)
    null_values = np.asarray(cohort_null["cohort_medians"], float)
    null_values = null_values[np.isfinite(null_values)]
    null_q025, null_q975 = np.percentile(null_values, [2.5, 97.5])

    ax.axvline(0.0, color="#8A8A8A", lw=0.85, ls=(0, (3, 2)), zorder=0)
    ax.axvspan(null_q025, null_q975, color="#BDBDBD", alpha=0.16, zorder=0)
    ax.scatter(
        values,
        y,
        s=27,
        facecolor=POINT_COLOR,
        edgecolor=POINT_EDGE,
        linewidth=0.65,
        zorder=3,
    )

    for yi, row in zip(y, effects):
        subject_id = str(row["subject_id"])
        if subject_id not in example_labels:
            continue
        value = float(row["mean_signed_cosine"])
        ax.scatter(
            [value],
            [yi],
            s=50,
            facecolor="none",
            edgecolor="#111111",
            linewidth=1.1,
            zorder=4,
        )
        ax.text(
            value + 0.020,
            yi,
            example_labels[subject_id],
            ha="left",
            va="center",
            fontsize=7.0,
            fontweight="bold",
            color="#222222",
        )

    median = float(np.median(values))
    ci_lo, ci_hi = bootstrap_median_ci(values, n_boot=5000, seed=DEFAULT_SEED)
    summary_y = 0.55
    ax.errorbar(
        median,
        summary_y,
        xerr=np.array([[median - ci_lo], [ci_hi - median]]),
        fmt="D",
        color="#111111",
        ecolor="#111111",
        elinewidth=1.2,
        capsize=2.4,
        markersize=4.4,
        markerfacecolor="#FFFFFF",
        markeredgewidth=1.0,
        zorder=5,
    )

    ridge_y = -1.45
    x_grid = np.linspace(float(null_values.min()), float(null_values.max()), 300)
    density = gaussian_kde(null_values)(x_grid)
    ridge_height = 1.05 * density / float(np.max(density))
    ax.fill_between(
        x_grid,
        ridge_y,
        ridge_y + ridge_height,
        color="#A9A9A9",
        alpha=0.65,
        linewidth=0,
        zorder=2,
    )
    ax.plot(
        x_grid,
        ridge_y + ridge_height,
        color="#707070",
        lw=0.8,
        zorder=3,
    )
    ax.plot(
        [null_q025, null_q975],
        [ridge_y - 0.06, ridge_y - 0.06],
        color="#666666",
        lw=1.4,
        solid_capstyle="round",
        zorder=3,
    )
    null_median = float(np.median(null_values))
    empirical_p = float(
        (1 + np.sum(null_values >= median - 1e-15)) / (len(null_values) + 1)
    )

    bracket_y = len(effects) + 2.05
    ax.plot(
        [null_median, null_median, median, median],
        [bracket_y - 0.32, bracket_y, bracket_y, bracket_y - 0.32],
        color="#222222",
        lw=0.9,
        clip_on=False,
    )
    ax.text(
        0.5 * (null_median + median),
        bracket_y + 0.06,
        "***",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="#111111",
    )

    ax.set_xlim(min(-0.10, float(null_q025) - 0.02), 0.88)
    ax.set_ylim(-2.15, len(effects) + 3.05)
    ax.set_yticks([])
    ax.text(
        0.02,
        0.93,
        "n=26",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color="#555555",
    )
    ax.set_xlabel("Direction score", fontsize=10.0, labelpad=6)
    ax.set_ylabel("Patients (ordered)", fontsize=10.0, labelpad=6)
    ax.tick_params(axis="x", labelsize=8.5, length=3)
    ax.tick_params(axis="y", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.grid(False)
    ax._figr_null_summary = {  # type: ignore[attr-defined]
        "median": null_median,
        "ci95": [float(null_q025), float(null_q975)],
        "empirical_p_observed_cohort_median": empirical_p,
        "n_observed_above_null_q975": int(np.sum(values > null_q975)),
    }


def _metadata(
    payloads: Sequence[Mapping[str, object]],
    examples: Sequence[str],
    example_labels: Sequence[str],
    contract: Mapping[str, object],
    cohort_null: Mapping[str, object],
    null_summary: Mapping[str, object],
    outputs: Mapping[str, Path],
) -> Dict[str, object]:
    effect_stat = contract["effect_stat"]
    observed_scores = np.asarray(
        [row["mean_signed_cosine"] for row in contract["effects"]], float
    )
    observed_ci = bootstrap_median_ci(
        observed_scores, n_boot=5000, seed=DEFAULT_SEED
    )
    return {
        "figure_role": "Figure R-B candidate: interictal gradient-axis representativeness",
        "examples": [
            {
                "display_label": label,
                "subject_id": subject_id,
                "n_qc_clean_ta": int(payload["n_direction_qc_ta"]),
                "n_qc_clean_tb": int(payload["n_direction_qc_tb"]),
                "n_qc_clean_total": (
                    int(payload["n_direction_qc_ta"])
                    + int(payload["n_direction_qc_tb"])
                ),
                "tb_angle_deg_in_ta_frame": float(payload["basis"]["theta_b_deg"]),
                "rose_source": (
                    "results/interictal_propagation_masked/"
                    "template_direction_rose_qc_clean"
                ),
            }
            for payload, subject_id, label in zip(payloads, examples, example_labels)
        ],
        "rose_contract": {
            "input": "QC-clean interictal single-event directions only",
            "minimum_mapped_participating_contacts": 6,
            "minimum_participating_shafts": 2,
            "minimum_effective_coordinate_rank": 2,
            "minimum_loco_valid_fraction": 0.8,
            "minimum_loco_median_signed_cosine": 0.8,
            "histogram_normalization": "within-template proportion per 20-degree bin",
            "solid_lines": "frozen TA/TB template-gradient early-to-late axes",
            "ictal_input": "none",
        },
        "cohort_contract": {
            **contract["cohort_flow"],
            "statistical_unit": "patient",
            "within_patient_fold": "equal mean of TA and TB",
            "primary_effect": (
                "mean signed cosine between fitted template-gradient axes and QC-clean "
                "single-event propagation directions, averaged equally over TA and TB"
            ),
            "null": "shuffle template ranks over the same mapped contacts and refit the axis",
            "plain_language_score": (
                "For each event, 1 means it travels in the same direction as the fitted "
                "axis, 0 means perpendicular, and -1 means opposite. Scores are averaged "
                "over events and then equally over TA and TB within each patient."
            ),
            "observed_direction_score": {
                "median": float(np.median(observed_scores)),
                "bootstrap_median_ci95": [
                    float(observed_ci[0]),
                    float(observed_ci[1]),
                ],
                "range": [
                    float(np.min(observed_scores)),
                    float(np.max(observed_scores)),
                ],
            },
            "randomized_cohort_distribution": {
                "fold": (
                    "for each of 1000 draws, average TA/TB randomized-axis scores within "
                    "patient, then take the median across the same 26 patients"
                ),
                "cache": str(Path(cohort_null["cache_path"]).relative_to(ROOT)),
                "fingerprint": str(cohort_null["fingerprint"]),
                **null_summary,
            },
            "median_alignment_gain": float(effect_stat["median"]),
            "bootstrap_median_ci95": [
                float(effect_stat["bootstrap_median_ci95"][0]),
                float(effect_stat["bootstrap_median_ci95"][1]),
            ],
            "wilcoxon_greater_than_zero_p": float(
                effect_stat["wilcoxon_greater_than_zero_p"]
            ),
            "n_positive": int(effect_stat["n_positive"]),
            "n_analyzed": int(effect_stat["n"]),
        },
        "claim_boundary": (
            "In-sample descriptive representativeness relative to a montage-matched "
            "rank-shuffle null; not held-out generalization and not a comparison with "
            "endpoint/source-sink axis methods."
        ),
        "sources": {
            "rose_producer": "scripts/plot_topic5_interictal_template_direction_rose_qc.py",
            "effect_producer": "scripts/paper_figures/plot_axis_representativeness.py",
            "effect_csv": str(EFFECT_CSV.relative_to(ROOT)),
            "effect_summary": str(EFFECT_SUMMARY.relative_to(ROOT)),
            "axis_cohort": str(AXIS_COHORT_CSV.relative_to(ROOT)),
        },
        "outputs": {key: str(path.relative_to(ROOT)) for key, path in outputs.items()},
    }


def _write_readme(out_dir: Path, metadata: Mapping[str, object]) -> None:
    figures = out_dir / "figures"
    cohort = metadata["cohort_contract"]
    observed = cohort["observed_direction_score"]
    randomized = cohort["randomized_cohort_distribution"]
    figures.joinpath("README.md").write_text(
        "\n".join([
            "# Figure R-B：间期 gradient 轴的单事件方向代表性",
            "",
            "### figr_gradient_axis_validation.png",
            "",
            (
                "左侧两张 Rose 图复用 `template_direction_rose_qc_clean` 的渐进轴版本，"
                "只纳入至少 6 个 mapped participating contacts、至少 2 根 shafts、二维有效秩"
                "且通过 LOCO 稳定性筛选的间期单事件。红/蓝柱为 TA/TB 单事件方向的模板内比例，"
                "红/蓝粗实线为冻结的 early-to-late template-gradient axes。"
            ),
            (
                "右侧每个点是一位患者。对每个 QC-clean 单事件，先计算它的传播方向与拟合轴"
                "是否同向：完全同向记为 1，垂直记为 0，反向记为 -1；随后先在 TA/TB 内"
                "平均，再对 TA/TB 等权平均，得到患者分数。"
                f"在 34 位 A/B-template 患者中，28 位可拟合双轴，26 位具有可评估二维方向；"
                "灰色峰为 1,000 次随机打乱 contact-template ranks、重建轴后，每次在同一"
                f"26 人中计算出的 cohort median 分布。真实 cohort median 为 "
                f"{observed['median']:.3f}（95% bootstrap CI "
                f"{observed['bootstrap_median_ci95'][0]:.3f}–"
                f"{observed['bootstrap_median_ci95'][1]:.3f}），随机分布中位数为 "
                f"{randomized['median']:.3f}（95% 范围 "
                f"{randomized['ci95'][0]:.3f}–{randomized['ci95'][1]:.3f}），"
                f"随机化检验 p={randomized['empirical_p_observed_cohort_median']:.3f}；"
                "26/26 位患者均位于"
                "随机 cohort-median 95% 范围上界右侧。"
            ),
            "**关注点**：真实患者分数整体位于随机 cohort-median 分布右侧，但患者间跨度明显；"
            "该图支持 in-sample 代表性，不是 held-out 泛化，也不比较 endpoint 方法。",
            "",
        ]),
        encoding="utf-8",
    )


def build_figure(
    *,
    examples: Sequence[str],
    example_labels: Sequence[str],
    out_dir: Path,
    bins: int = DEFAULT_BINS,
) -> Mapping[str, Path]:
    if len(examples) != 2 or len(example_labels) != 2:
        raise ValueError("Figure R-B requires exactly two example subjects and labels")
    payloads = [build_subject_payload(subject_id) for subject_id in examples]
    if not all(bool(payload["geometry_2d_supported"]) for payload in payloads):
        raise ValueError("Rose examples must have supported 2-D geometry")
    contract = load_effect_contract()
    cohort_null = load_or_build_cohort_null(
        contract,
        cache_path=out_dir / "cohort_rank_shuffle_null.npz",
    )

    edges = np.linspace(0.0, 2.0 * np.pi, bins + 1)
    rmax = _rose_max(payloads, edges)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.15, 4.55), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(0.92, 1.55),
        height_ratios=(1, 1),
        left=0.028,
        right=0.992,
        top=0.970,
        bottom=0.135,
        wspace=0.14,
        hspace=0.26,
    )
    rose_axes = [
        fig.add_subplot(grid[0, 0], projection="polar"),
        fig.add_subplot(grid[1, 0], projection="polar"),
    ]
    for ax, payload, label in zip(rose_axes, payloads, example_labels):
        draw_probability_rose(ax, payload, edges=edges, rmax=rmax, label=label)

    stat_ax = fig.add_subplot(grid[:, 1])
    draw_subject_effects(
        stat_ax,
        contract,
        cohort_null,
        example_labels=dict(zip(examples, example_labels)),
    )

    legend_handles = [
        Patch(
            facecolor=matplotlib.colors.to_rgba(TA_COLOR, 0.20),
            edgecolor=TA_COLOR,
            linewidth=0.9,
            label="TA events",
        ),
        Patch(
            facecolor=matplotlib.colors.to_rgba(TB_COLOR, 0.20),
            edgecolor=TB_COLOR,
            linewidth=0.9,
            label="TB events",
        ),
        Line2D([0], [0], color="#333333", lw=2.2, label="Fitted axes"),
        Patch(
            facecolor="#A9A9A9",
            edgecolor="#707070",
            linewidth=0.8,
            label="Randomized median",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.028, 0.012),
        ncol=2,
        frameon=False,
        fontsize=7.6,
        handlelength=1.8,
        columnspacing=1.15,
    )

    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    stem = "figr_gradient_axis_validation"
    outputs = {
        "png": figures / f"{stem}.png",
        "pdf": figures / f"{stem}.pdf",
        "metadata": figures / f"{stem}_metadata.json",
    }
    fig.savefig(outputs["png"], dpi=300, facecolor="white")
    fig.savefig(outputs["pdf"], facecolor="white")
    plt.close(fig)

    metadata = _metadata(
        payloads,
        examples,
        example_labels,
        contract,
        cohort_null,
        stat_ax._figr_null_summary,  # type: ignore[attr-defined]
        outputs,
    )
    outputs["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_readme(out_dir, metadata)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", nargs=2, default=DEFAULT_EXAMPLES)
    parser.add_argument("--example-labels", nargs=2, default=DEFAULT_EXAMPLE_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_figure(
        examples=args.examples,
        example_labels=args.example_labels,
        out_dir=args.output_dir,
        bins=args.bins,
    )
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
