#!/usr/bin/env python3
"""TA/TB own-field negative-similarity null analysis.

Two subject-specific spatial nulls are evaluated on the same 2D-eligible
cohort and the same estimator:

* channel: jointly permute TB earliness and support across all contacts;
* within_shaft: jointly permute them only within each electrode shaft.

TA/TB axes, own planes, bandwidths and contact sets remain frozen.  TB fields
are rebuilt from each permuted contact payload; already-smoothed field values
are never shuffled directly.  The resulting inference is conditional on the
frozen template-specific geometry.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

from paper_figures.plot_fig3_field_concordance_cohort_stat import (
    _add_sig_bracket,
    _add_violin_box_points,
    _fmt_p,
    _p_stars,
)
from plot_topic5_interictal_template_ab_fields import (
    DEFAULT_YUQUAN_CROSSWALK,
    _display_name,
    _load_yuquan_crosswalk,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results/interictal_propagation_masked/template_gradient_fields"
DEFAULT_OUTPUT = DEFAULT_INPUT
NULL_MODES = ("channel", "within_shaft")
NULL_LABELS = {"channel": "Channel shuffle", "within_shaft": "Within-shaft shuffle"}
RELATION_COLORS = {"reversed": "#2F7F72", "same": "#A97828", "different": "#667B8A"}


def _stable_seed(text: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{text}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if int(keep.sum()) < 3:
        return float("nan")
    x = x[keep] - float(np.mean(x[keep]))
    y = y[keep] - float(np.mean(y[keep]))
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(x @ y / denom) if denom > 0 else float("nan")


def _bh_fdr(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    q = np.full_like(p, np.nan)
    finite = np.where(np.isfinite(p))[0]
    if not len(finite):
        return q
    order = finite[np.argsort(p[finite])]
    running = 1.0
    m = len(order)
    for reverse_index, original_index in enumerate(order[::-1]):
        rank = m - reverse_index
        running = min(running, float(p[original_index]) * m / rank)
        q[original_index] = running
    return q


def _group_indices(shafts: list[str], mode: str) -> list[np.ndarray]:
    if mode == "channel":
        return [np.arange(len(shafts), dtype=int)]
    if mode != "within_shaft":
        raise ValueError(mode)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, shaft in enumerate(shafts):
        grouped[str(shaft)].append(index)
    return [np.asarray(indices, dtype=int) for indices in grouped.values()]


def _permutation_indices(
    shafts: list[str], mode: str, n_perm: int, rng: np.random.Generator
) -> tuple[np.ndarray, int, bool]:
    groups = _group_indices(shafts, mode)
    n_unique = math.prod(math.factorial(len(group)) for group in groups)
    n_contacts = len(shafts)
    if n_unique <= n_perm:
        group_permutations = [list(itertools.permutations(group.tolist())) for group in groups]
        result = np.tile(np.arange(n_contacts, dtype=int), (n_unique, 1))
        for row_index, combination in enumerate(itertools.product(*group_permutations)):
            for target, source in zip(groups, combination):
                result[row_index, target] = np.asarray(source, dtype=int)
        return result, int(n_unique), True

    result = np.tile(np.arange(n_contacts, dtype=int), (n_perm, 1))
    for group in groups:
        if len(group) < 2:
            continue
        orders = np.argsort(rng.random((n_perm, len(group))), axis=1)
        result[:, group] = group[orders]
    return result, int(n_unique), False


def _gaussian_kernel(points: np.ndarray, sigma: float) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    distance2 = ((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-distance2 / (2.0 * float(sigma) ** 2))


def _null_correlations(
    field_a: np.ndarray,
    earliness_b: np.ndarray,
    support_b: np.ndarray,
    points_b: np.ndarray,
    sigma_b: float,
    permutation_indices: np.ndarray,
    *,
    chunk_size: int = 500,
) -> np.ndarray:
    kernel = _gaussian_kernel(points_b, sigma_b)
    centered_a = np.asarray(field_a, dtype=float) - float(np.mean(field_a))
    norm_a = float(np.linalg.norm(centered_a))
    output = np.full(len(permutation_indices), np.nan, dtype=float)
    for start in range(0, len(permutation_indices), chunk_size):
        stop = min(start + chunk_size, len(permutation_indices))
        permutations = permutation_indices[start:stop]
        support = support_b[permutations]
        values = earliness_b[permutations]
        weighted_support = support * values
        numerator = np.einsum("jk,bk->bj", kernel, weighted_support, optimize=True)
        denominator = np.einsum("jk,bk->bj", kernel, support, optimize=True)
        fields = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > 0,
        )
        centered_b = fields - np.nanmean(fields, axis=1, keepdims=True)
        norm_b = np.linalg.norm(centered_b, axis=1)
        valid = np.isfinite(centered_b).all(axis=1) & (norm_b > 0) & (norm_a > 0)
        output[start:stop][valid] = centered_b[valid] @ centered_a / (norm_b[valid] * norm_a)
    return output[np.isfinite(output)]


def _subject_record(
    artifact_path: Path,
    *,
    n_perm: int,
    base_seed: int,
    yuquan_labels: dict[str, str],
) -> tuple[dict, dict[str, np.ndarray]] | None:
    record = json.loads(artifact_path.read_text())
    pair = record.get("axis_pair") or {}
    if record.get("status") != "ok" or not pair.get("geometry_2d_supported"):
        return None
    field = record["interictal_field"]
    models = field["field_models"]
    own_a = models["own_a"]
    own_b = models["own_b"]
    field_a = np.asarray(own_a["template_field"], dtype=float)
    field_b = np.asarray(own_b["template_field"], dtype=float)
    earliness_b = np.asarray(field["earliness_b"], dtype=float)
    support_b = np.asarray(field["support_b"], dtype=float)
    points_b = np.asarray(own_b["points"], dtype=float)
    sigma_b = float(own_b["sigma"])
    rebuilt_b = _null_correlations(
        field_a,
        earliness_b,
        support_b,
        points_b,
        sigma_b,
        np.arange(len(field_b), dtype=int)[None, :],
    )
    observed = _pearson_r(field_a, field_b)
    if len(rebuilt_b) != 1 or not np.isclose(rebuilt_b[0], observed, atol=1e-10, rtol=1e-10):
        raise ValueError(f"frozen-field rebuild mismatch for {record['subject_id']}")

    subject_row = {
        "subject_id": record["subject_id"],
        "display_id": _display_name(record["subject_id"], yuquan_labels),
        "dataset": record["dataset"],
        "subject": record["subject"],
        "relation": pair["relation"]["relation"],
        "cos_ta_tb": float(pair["relation"]["cosine"]),
        "strict_stability_pass": bool(pair.get("strict_stability_pass")),
        "n_contacts": int(len(field_a)),
        "observed_own_field_r": float(observed),
    }
    null_arrays: dict[str, np.ndarray] = {}
    shafts = [str(value) for value in field["shafts"]]
    for mode in NULL_MODES:
        rng = np.random.default_rng(_stable_seed(f"{record['subject_id']}:{mode}", base_seed))
        permutations, n_unique, exact = _permutation_indices(shafts, mode, n_perm, rng)
        null = _null_correlations(
            field_a,
            earliness_b,
            support_b,
            points_b,
            sigma_b,
            permutations,
        )
        if exact:
            p_negative = float(np.mean(null <= observed + 1e-12))
        else:
            p_negative = float((1 + np.sum(null <= observed)) / (len(null) + 1))
        null_arrays[mode] = null
        subject_row.update(
            {
                f"{mode}_null_median": float(np.median(null)),
                f"{mode}_null_q05": float(np.percentile(null, 5)),
                f"{mode}_null_q95": float(np.percentile(null, 95)),
                f"{mode}_p_negative": p_negative,
                f"{mode}_n_draws": int(len(null)),
                f"{mode}_n_unique": int(n_unique),
                f"{mode}_exact": bool(exact),
                f"{mode}_resolution_adequate": bool(n_unique >= 20),
            }
        )
    return subject_row, null_arrays


def _bh_rejection_counts(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    ordered = np.sort(np.asarray(p_values, dtype=float), axis=1)
    thresholds = alpha * np.arange(1, ordered.shape[1] + 1) / ordered.shape[1]
    passes = ordered <= thresholds[None, :]
    return np.max(
        np.where(passes, np.arange(1, ordered.shape[1] + 1)[None, :], 0), axis=1
    )


def _cohort_summary(
    rows: list[dict],
    null_arrays: dict[str, dict[str, np.ndarray]],
    mode: str,
    *,
    n_cohort_perm: int,
    base_seed: int,
) -> dict:
    observed = np.asarray([row["observed_own_field_r"] for row in rows], dtype=float)
    arrays = [null_arrays[row["subject_id"]][mode] for row in rows]
    centers = np.asarray([np.median(values) for values in arrays], dtype=float)
    observed_delta = float(np.median(observed - centers))
    rng = np.random.default_rng(_stable_seed(f"cohort:{mode}", base_seed))
    n_subjects = len(rows)
    exceed_shift = 0
    null_nominal_counts = []
    null_fdr_counts = []
    batch_size = 5_000
    for start in range(0, n_cohort_perm, batch_size):
        n_batch = min(batch_size, n_cohort_perm - start)
        samples = np.column_stack(
            [values[rng.integers(0, len(values), size=n_batch)] for values in arrays]
        )
        null_delta = np.median(samples - centers[None, :], axis=1)
        exceed_shift += int(np.sum(null_delta <= observed_delta))
        pseudo_p = np.column_stack(
            [
                np.searchsorted(np.sort(values), samples[:, index], side="right") / len(values)
                for index, values in enumerate(arrays)
            ]
        )
        null_nominal_counts.append(np.sum(pseudo_p < 0.05, axis=1))
        null_fdr_counts.append(_bh_rejection_counts(pseudo_p))
    nominal_counts = np.concatenate(null_nominal_counts)
    fdr_counts = np.concatenate(null_fdr_counts)
    observed_p = np.asarray([row[f"{mode}_p_negative"] for row in rows], dtype=float)
    observed_q = _bh_fdr(observed_p)
    observed_nominal = int(np.sum(observed_p < 0.05))
    observed_fdr = int(np.sum(observed_q < 0.05))
    return {
        "n_subjects": n_subjects,
        "n_observed_negative": int(np.sum(observed < 0)),
        "sign_test_p_more_negative_than_half": float(
            binomtest(int(np.sum(observed < 0)), n_subjects, 0.5, alternative="greater").pvalue
        ),
        "observed_r_median": float(np.median(observed)),
        "null_median_across_subject_medians": float(np.median(centers)),
        "observed_median_delta_vs_subject_null": observed_delta,
        "cohort_shift_p_negative": float((1 + exceed_shift) / (n_cohort_perm + 1)),
        "n_cohort_permutations": int(n_cohort_perm),
        "n_nominal_subject_p_lt_0_05": observed_nominal,
        "n_bh_fdr_q_lt_0_05": observed_fdr,
        "prevalence_p_nominal_count": float(
            (1 + np.sum(nominal_counts >= observed_nominal)) / (n_cohort_perm + 1)
        ),
        "prevalence_p_bh_fdr_count": float(
            (1 + np.sum(fdr_counts >= observed_fdr)) / (n_cohort_perm + 1)
        ),
        "null_nominal_count_q95": float(np.percentile(nominal_counts, 95)),
        "null_bh_fdr_count_q95": float(np.percentile(fdr_counts, 95)),
    }


def _add_fdr(rows: list[dict]) -> None:
    for mode in NULL_MODES:
        q_values = _bh_fdr(row[f"{mode}_p_negative"] for row in rows)
        for row, q_value in zip(rows, q_values):
            row[f"{mode}_q_bh"] = float(q_value)
            row[f"{mode}_fdr_significant_negative"] = bool(
                row["observed_own_field_r"] < 0 and q_value < 0.05
            )


def _plot_paired(
    rows: list[dict], summaries: dict[str, dict], out_png: Path, out_pdf: Path, *, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(7.1, 4.45))
    positions = {"channel": (1.0, 1.72), "within_shaft": (3.15, 3.87)}
    for mode in NULL_MODES:
        x_data, x_null = positions[mode]
        observed = np.asarray([row["observed_own_field_r"] for row in rows], dtype=float)
        null = np.asarray([row[f"{mode}_null_median"] for row in rows], dtype=float)
        jitter = rng.normal(0.0, 0.035, size=len(rows))
        data_x = _add_violin_box_points(
            ax,
            observed,
            x_data,
            facecolor="#65B7A6",
            edgecolor="#2F7F72",
            rng=rng,
            point_face="#2F7F72",
            point_edge="white",
            jitter=jitter,
        )
        null_x = _add_violin_box_points(
            ax,
            null,
            x_null,
            facecolor="#D8B56A",
            edgecolor="#9B7430",
            rng=rng,
            point_face="#9B7430",
            point_edge="white",
            jitter=jitter,
        )
        for x0, y0, x1, y1 in zip(data_x, observed, null_x, null):
            ax.plot([x0, x1], [y0, y1], color="0.5", linewidth=0.65, alpha=0.25, zorder=3)
        summary = summaries[mode]
        y_bracket = max(float(np.max(observed)), float(np.max(null))) + 0.075
        _add_sig_bracket(
            ax, x_data, x_null, y_bracket, _p_stars(summary["cohort_shift_p_negative"])
        )
        ax.text(
            (x_data + x_null) / 2,
            -0.155,
            f"{NULL_LABELS[mode]}\nn={len(rows)}, P={_fmt_p(summary['cohort_shift_p_negative'])}\n"
            f"FDR negative: {summary['n_bh_fdr_q_lt_0_05']}/{len(rows)}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8.4,
        )
    ax.axhline(0.0, color="0.5", linestyle="--", linewidth=1.0, zorder=0)
    ax.text(4.27, 0.015, "r = 0", ha="left", va="bottom", fontsize=8.5, color="0.45")
    ax.set_ylabel("TA–TB own-field similarity (signed r)", fontsize=11)
    ax.set_xticks([value for mode in NULL_MODES for value in positions[mode]])
    ax.set_xticklabels(["Data", "Null", "Data", "Null"], fontsize=10)
    ax.set_xlim(0.45, 4.55)
    ax.set_ylim(-1.05, 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", width=1.0)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.97, bottom=0.23)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def _plot_subjects(
    rows: list[dict], summaries: dict[str, dict], out_png: Path, out_pdf: Path
) -> None:
    ordered = sorted(rows, key=lambda row: row["observed_own_field_r"])
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 0.31 * len(ordered) + 1.8), sharey=True)
    for ax, mode in zip(axes, NULL_MODES):
        for index, row in enumerate(ordered):
            y = len(ordered) - 1 - index
            q05 = row[f"{mode}_null_q05"]
            q95 = row[f"{mode}_null_q95"]
            median = row[f"{mode}_null_median"]
            observed = row["observed_own_field_r"]
            color = RELATION_COLORS[row["relation"]]
            significant = row[f"{mode}_fdr_significant_negative"]
            ax.plot([q05, q95], [y, y], color="0.72", linewidth=2.2, zorder=1)
            ax.plot([median, median], [y - 0.18, y + 0.18], color="0.43", linewidth=1.0, zorder=2)
            ax.scatter(
                [observed],
                [y],
                s=43,
                facecolors=(color if significant else "white"),
                edgecolors=color,
                linewidths=1.25,
                zorder=3,
            )
            if ax is axes[0]:
                ax.text(
                    -1.03,
                    y,
                    row["display_id"],
                    ha="right",
                    va="center",
                    fontsize=7.6,
                    transform=ax.transData,
                )
        summary = summaries[mode]
        ax.axvline(0.0, color="0.55", linestyle="--", linewidth=0.9, zorder=0)
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-0.75, len(ordered) - 0.25)
        ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.set_yticks([])
        ax.set_title(
            f"{NULL_LABELS[mode]}\ncohort P={_fmt_p(summary['cohort_shift_p_negative'])}; "
            f"FDR {summary['n_bh_fdr_q_lt_0_05']}/{len(rows)}",
            fontsize=10.2,
            fontweight="bold",
        )
        ax.set_xlabel("Signed field r", fontsize=9.5)
        ax.spines[["top", "right", "left"]].set_visible(False)
    handles = [
        mlines.Line2D([], [], marker="o", linestyle="none", markerfacecolor=color,
                      markeredgecolor=color, label=relation.capitalize())
        for relation, color in RELATION_COLORS.items()
    ]
    handles.extend(
        [
            mlines.Line2D([], [], color="0.72", linewidth=2.2, label="Null 5–95%"),
            mlines.Line2D([], [], marker="o", linestyle="none", markerfacecolor="0.35",
                          markeredgecolor="0.35", label="BH-FDR q<0.05"),
        ]
    )
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8.2)
    fig.subplots_adjust(left=0.14, right=0.99, top=0.92, bottom=0.10, wspace=0.22)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def _plot_patient_distribution(
    rows: list[dict], out_png: Path, out_pdf: Path, *, seed: int
) -> None:
    """Compact patient-level TA/TB distribution; null statistics stay in the caption."""
    rng = np.random.default_rng(seed)
    observed = np.asarray([row["observed_own_field_r"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(3.55, 4.05))
    point_x = _add_violin_box_points(
        ax,
        observed,
        1.0,
        facecolor="#D8DDE1",
        edgecolor="#777F86",
        rng=rng,
        point_face="none",
        point_edge="none",
    )
    for relation in ("reversed", "same", "different"):
        indices = [index for index, row in enumerate(rows) if row["relation"] == relation]
        ax.scatter(
            point_x[indices],
            observed[indices],
            s=31,
            facecolors=RELATION_COLORS[relation],
            edgecolors="white",
            linewidths=0.8,
            alpha=0.95,
            zorder=5,
        )
    ax.axhline(0.0, color="0.5", linestyle="--", linewidth=1.0, zorder=0)
    ax.text(1.31, 0.012, "r = 0", ha="left", va="bottom", fontsize=8.3, color="0.45")
    ax.text(
        0.055,
        0.965,
        f"median r = {np.median(observed):+.2f}\n{int(np.sum(observed < 0))}/{len(rows)} below 0",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
    )
    handles = [
        mlines.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=6.0,
            markerfacecolor=RELATION_COLORS[relation],
            markeredgecolor="white",
            label={"reversed": "Reversed", "same": "Same", "different": "Different"}[relation],
        )
        for relation in ("reversed", "same", "different")
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.56, -0.13),
        ncol=3,
        frameon=False,
        fontsize=7.8,
        handletextpad=0.3,
        columnspacing=0.8,
    )
    ax.set_ylabel("TA–TB own-field similarity (signed r)", fontsize=10.5)
    ax.set_xticks([1.0])
    ax.set_xticklabels([f"Patients\nn={len(rows)}"], fontsize=9.5)
    ax.set_xlim(0.58, 1.44)
    ax.set_ylim(-1.05, 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", width=1.0)
    fig.subplots_adjust(left=0.26, right=0.96, top=0.97, bottom=0.24)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def _plot_compact_distribution(
    rows: list[dict], out_png: Path, out_pdf: Path, *, seed: int
) -> None:
    """Small descriptive panel: one point per subject, no null and no violin."""
    order = ("reversed", "same", "different")
    labels = {
        "reversed": "Reversed-collinear",
        "same": "Same-collinear",
        "different": "Different-axis",
    }
    y_position = {relation: float(len(order) - 1 - index) for index, relation in enumerate(order)}
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(5.15, 2.75))
    for relation in order:
        values = np.asarray(
            [row["observed_own_field_r"] for row in rows if row["relation"] == relation],
            dtype=float,
        )
        y = y_position[relation]
        color = RELATION_COLORS[relation]
        q25, median, q75 = np.percentile(values, [25, 50, 75])
        ax.plot([q25, q75], [y, y], color=color, linewidth=7.0, alpha=0.24,
                solid_capstyle="round", zorder=1)
        ax.scatter(
            [median], [y], marker="D", s=35, facecolor="black", edgecolor="white",
            linewidth=0.7, zorder=4,
        )
        jitter = np.clip(rng.normal(0.0, 0.085, size=len(values)), -0.16, 0.16)
        ax.scatter(
            values,
            np.full(len(values), y) + jitter,
            s=43,
            facecolor=color,
            edgecolor="white",
            linewidth=0.9,
            alpha=0.94,
            zorder=3,
        )
    all_values = np.asarray([row["observed_own_field_r"] for row in rows], dtype=float)
    ax.axvline(0.0, color="0.52", linestyle="--", linewidth=1.0, zorder=0)
    fig.text(
        0.98,
        0.955,
        f"Overall: {int(np.sum(all_values < 0))}/{len(all_values)} negative; "
        f"median r={np.median(all_values):+.2f}",
        ha="right",
        va="center",
        fontsize=8.4,
    )
    ax.set_yticks([y_position[relation] for relation in order])
    ax.set_yticklabels(
        [
            f"{labels[relation]}\n"
            f"{sum(row['observed_own_field_r'] < 0 for row in rows if row['relation'] == relation)}/"
            f"{sum(row['relation'] == relation for row in rows)} negative"
            for relation in order
        ],
        fontsize=8.5,
    )
    for tick, relation in zip(ax.get_yticklabels(), order):
        tick.set_color(RELATION_COLORS[relation])
    ax.set_xlabel("TA–TB own-field similarity (signed r)", fontsize=10.2)
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-0.48, 2.48)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.subplots_adjust(left=0.28, right=0.98, top=0.88, bottom=0.20)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def _write_csv(rows: list[dict], path: Path) -> None:
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--yuquan-crosswalk", type=Path, default=DEFAULT_YUQUAN_CROSSWALK)
    parser.add_argument("--n-perm", type=int, default=10_000)
    parser.add_argument("--n-cohort-perm", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()

    yuquan_labels = _load_yuquan_crosswalk(args.yuquan_crosswalk)
    rows = []
    null_arrays: dict[str, dict[str, np.ndarray]] = {}
    for artifact_path in sorted((args.input_root / "per_subject").glob("*.json")):
        result = _subject_record(
            artifact_path,
            n_perm=args.n_perm,
            base_seed=args.seed,
            yuquan_labels=yuquan_labels,
        )
        if result is None:
            continue
        row, subject_nulls = result
        rows.append(row)
        null_arrays[row["subject_id"]] = subject_nulls
    if len(rows) != 26:
        raise ValueError(f"expected 26 geometry_2d_supported subjects, found {len(rows)}")
    _add_fdr(rows)
    summaries = {
        mode: _cohort_summary(
            rows,
            null_arrays,
            mode,
            n_cohort_perm=args.n_cohort_perm,
            base_seed=args.seed,
        )
        for mode in NULL_MODES
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / "field_similarity_negative_null_subjects.csv"
    json_path = args.output_root / "field_similarity_negative_null_statistics.json"
    npz_path = args.output_root / "field_similarity_negative_null_draws.npz"
    paired_png = figure_dir / "template_field_negative_null_comparison.png"
    paired_pdf = figure_dir / "template_field_negative_null_comparison.pdf"
    subject_png = figure_dir / "template_field_negative_null_subjects.png"
    subject_pdf = figure_dir / "template_field_negative_null_subjects.pdf"
    compact_png = figure_dir / "template_field_similarity_subject_distribution.png"
    compact_pdf = figure_dir / "template_field_similarity_subject_distribution.pdf"
    distribution_png = figure_dir / "template_field_patient_distribution.png"
    distribution_pdf = figure_dir / "template_field_patient_distribution.pdf"
    _write_csv(rows, csv_path)
    np.savez_compressed(
        npz_path,
        **{
            f"{row['subject_id']}__{mode}": null_arrays[row["subject_id"]][mode]
            for row in rows
            for mode in NULL_MODES
        },
    )
    json_path.write_text(
        json.dumps(
            {
                "contract": "topic5_interictal_template_fields_v1",
                "cohort": "geometry_2d_supported",
                "n_subjects": len(rows),
                "field_metric": "signed Pearson r between contact-evaluated own_a and own_b fields",
                "permutation_payload": "TB earliness and support jointly permuted; TB field rebuilt",
                "frozen": ["contact set", "TA/TB own axes", "own planes", "sigma"],
                "nulls": {
                    "channel": "permute TB payload across all contacts",
                    "within_shaft": "permute TB payload only within electrode shaft",
                },
                "n_permutations_requested": args.n_perm,
                "n_cohort_permutations": args.n_cohort_perm,
                "seed": args.seed,
                "cohort_summary": summaries,
                "interpretation_boundary": (
                    "Tests negative TA/TB field placement conditional on frozen template-specific "
                    "geometry. It does not rerun KMeans template discovery or refit axes, and therefore "
                    "does not establish that negative field structure is independent of template "
                    "selection or gradient direction."
                ),
                "per_subject_csv": str(csv_path.relative_to(ROOT)),
                "null_draws_npz": str(npz_path.relative_to(ROOT)),
            },
            indent=2,
        )
        + "\n"
    )
    _plot_paired(rows, summaries, paired_png, paired_pdf, seed=args.seed)
    _plot_subjects(rows, summaries, subject_png, subject_pdf)
    _plot_compact_distribution(rows, compact_png, compact_pdf, seed=args.seed)
    _plot_patient_distribution(rows, distribution_png, distribution_pdf, seed=args.seed)
    print(json.dumps(summaries, indent=2))
    print(f"[done] wrote {paired_png}")
    print(f"[done] wrote {subject_png}")
    print(f"[done] wrote {compact_png}")
    print(f"[done] wrote {distribution_png}")
    print(f"[done] wrote {csv_path}")
    print(f"[done] wrote {json_path}")


if __name__ == "__main__":
    main()
