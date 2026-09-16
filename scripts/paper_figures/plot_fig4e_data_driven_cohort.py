#!/usr/bin/env python3
"""Render the compact Figure 4E cohort-level matched-null panel.

The panel consumes the frozen formal cohort result. It does not rerun the SNN,
refit KMeans, or change any acceptance rule. Network-seed repeatability and
same-network KMeans remain in metadata/caption rather than the main panel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT = (
    ROOT
    / "results/topic4_sef_hfo/data_driven_snn_cohort_v1/formal/cohort_result.json"
)
DEFAULT_OUTPUT = ROOT / "results/paper-ready-figure/fig4/figures"

CANONICAL_COLOR = "#2166AC"
REAL_COLOR = "#D97706"
NULL_COLOR = "#8F8F8F"
KMEANS_PASS_COLOR = "#00897B"
KMEANS_FAIL_COLOR = "#B7B7B7"
ZERO_COLOR = "#333333"
GROUP_ORDER = ("both", "kmeans_only", "heldout_only", "neither")
GROUP_LABELS = {
    "both": "both criteria",
    "kmeans_only": "two modes only",
    "heldout_only": "contact-null only",
    "neither": "neither",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _significance_label(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def _short_subject(subject_id: str, yuquan_aliases: dict[str, str]) -> str:
    if subject_id.startswith("epilepsiae_"):
        return "E" + subject_id.removeprefix("epilepsiae_")
    if subject_id.startswith("yuquan_"):
        return yuquan_aliases[subject_id]
    return subject_id


def _subject_group(row: dict) -> str:
    heldout = bool(row["subject_endpoint_pass"])
    kmeans = bool(row["natural_kmeans"]["same_network_k2"])
    if heldout and kmeans:
        return "both"
    if kmeans:
        return "kmeans_only"
    if heldout:
        return "heldout_only"
    return "neither"


def cohort_payload(result: dict) -> dict:
    """Validate and flatten the frozen result into the exact display payload."""
    canonical = list(result["canonical_subjects"])
    real = {row["subject_id"]: row for row in result["real_geometry_subjects"]}
    if len(canonical) != int(result["denominators"]["primary_canonical_layout"]):
        raise ValueError("canonical subject count disagrees with frozen denominator")
    if len(real) != int(result["denominators"]["real_geometry_sensitivity"]):
        raise ValueError("real-geometry subject count disagrees with frozen denominator")
    if len(set(row["subject_id"] for row in canonical)) != len(canonical):
        raise ValueError("canonical subject IDs are not unique")
    yuquan_ids = sorted(
        row["subject_id"] for row in canonical
        if row["subject_id"].startswith("yuquan_")
    )
    yuquan_aliases = {
        subject_id: f"Y{index + 1}" for index, subject_id in enumerate(yuquan_ids)
    }

    rows = []
    for row in canonical:
        n_seeds = int(row["n_seeds"])
        kmeans_count = int(
            row["natural_kmeans"]["n_seeds_with_same_network_k2"]
        )
        if n_seeds != len(result["confirmation_seeds"]):
            raise ValueError("subject confirmation seed count changed")
        if not 0 <= kmeans_count <= n_seeds:
            raise ValueError("invalid same-network KMeans count")
        real_row = real.get(row["subject_id"])
        rows.append({
            "subject_id": row["subject_id"],
            "short_id": _short_subject(row["subject_id"], yuquan_aliases),
            "group": _subject_group(row),
            "heldout_pass": bool(row["subject_endpoint_pass"]),
            "canonical_advantage": float(
                row["delta_null_median_minus_observed"]
            ),
            "kmeans_positive_networks": kmeans_count,
            "kmeans_subject_pass": bool(
                row["natural_kmeans"]["same_network_k2"]
            ),
            "real_geometry_advantage": (
                None if real_row is None else float(
                    real_row["delta_null_median_minus_observed"]
                )
            ),
        })

    group_rank = {group: index for index, group in enumerate(GROUP_ORDER)}
    rows.sort(key=lambda row: (
        group_rank[row["group"]],
        -row["kmeans_positive_networks"],
        -row["canonical_advantage"],
        row["subject_id"],
    ))
    group_counts = {
        group: sum(row["group"] == group for row in rows)
        for group in GROUP_ORDER
    }
    real_shared = [
        row for row in rows if row["real_geometry_advantage"] is not None
    ]
    sign_agreement = sum(
        np.sign(row["canonical_advantage"])
        == np.sign(row["real_geometry_advantage"])
        for row in real_shared
    )
    both_with_real = [
        row for row in rows
        if row["group"] == "both" and row["real_geometry_advantage"] is not None
    ]
    return {
        "rows": rows,
        "group_counts": group_counts,
        "n_subjects": len(rows),
        "n_heldout_pass": sum(row["heldout_pass"] for row in rows),
        "n_kmeans_pass": sum(row["kmeans_subject_pass"] for row in rows),
        "n_both": group_counts["both"],
        "n_real_geometry": len(real_shared),
        "n_real_geometry_pass": sum(
            bool(row["subject_endpoint_pass"])
            for row in result["real_geometry_subjects"]
        ),
        "real_geometry_sign_agreement": int(sign_agreement),
        "n_both_with_real_geometry": len(both_with_real),
        "n_both_with_positive_real_geometry": sum(
            row["real_geometry_advantage"] > 0 for row in both_with_real
        ),
        "confirmation_seeds": list(result["confirmation_seeds"]),
        "status": result["status"],
        "failed_gates": list(result["verdict"].get("failed_gates", [])),
    }


def _draw_group_boundaries(axes: list, payload: dict) -> None:
    start = 0
    for index, group in enumerate(GROUP_ORDER):
        count = payload["group_counts"][group]
        if count == 0:
            continue
        stop = start + count
        midpoint = (start + stop - 1) / 2
        axes[0].text(
            midpoint, 1.025, f"{GROUP_LABELS[group]} ({count})",
            transform=axes[0].get_xaxis_transform(), ha="center", va="bottom",
            fontsize=7.0, fontweight="bold", color="#333333",
        )
        if index > 0:
            for axis in axes:
                axis.axvline(start - 0.5, color="#D0D0D0", linewidth=0.8)
        start = stop


def render(
    result: dict,
    output_dir: Path,
    *,
    stem_name: str = "fig4-panele",
) -> dict:
    payload = cohort_payload(result)
    observed_loss = np.asarray([
        row["observed_weakest_mode_loss"]
        for row in result["canonical_subjects"]
    ], float)
    null_loss = np.asarray([
        row["null_median"] for row in result["canonical_subjects"]
    ], float)
    primary = result["cohort"]["primary_test"]
    real_test = result["cohort"]["sensitivity"]["real_geometry_test"]

    fig = plt.figure(figsize=(8.4, 3.55), facecolor="white")
    grid = fig.add_gridspec(
        1, 2, width_ratios=(1.12, 0.88),
        left=0.19, right=0.985, bottom=0.22, top=0.88, wspace=0.38,
    )
    ax_count = fig.add_subplot(grid[0, 0])
    ax_null = fig.add_subplot(grid[0, 1])

    counts = np.asarray([
        payload["n_subjects"], payload["n_heldout_pass"],
        payload["n_kmeans_pass"], payload["n_both"],
    ])
    y = np.asarray([3, 2, 1, 0])
    count_colors = ["#A7A7A7", CANONICAL_COLOR, "#6A51A3", KMEANS_PASS_COLOR]
    bars = ax_count.barh(y, counts, height=0.62, color=count_colors, edgecolor="none")
    for bar, count in zip(bars, counts):
        percentage = 100.0 * float(count) / payload["n_subjects"]
        label = str(int(count)) if count == payload["n_subjects"] else (
            f"{int(count)} ({percentage:.0f}%)"
        )
        ax_count.text(
            count + 0.65, bar.get_y() + bar.get_height() / 2, label,
            ha="left", va="center", fontsize=12.2, fontweight="bold",
            color="#222222",
        )
    ax_count.set_xlim(0, 42)
    ax_count.set_xticks([0, 10, 20, 30])
    ax_count.set_yticks(
        y,
        ["Eligible", "Loss < null", "Two modes", "Both"],
        fontsize=12.5,
    )
    ax_count.set_xlabel("Patients", fontsize=13.5)
    ax_count.set_title("Cohort gates", fontsize=15.5, fontweight="bold", pad=8)
    ax_count.spines[["top", "right"]].set_visible(False)
    ax_count.tick_params(length=3.5, width=0.9, labelsize=12.0)

    violin = ax_null.violinplot(
        [observed_loss, null_loss], positions=[0, 1], widths=0.72,
        showmeans=False, showmedians=False, showextrema=False,
    )
    for body, color in zip(violin["bodies"], [CANONICAL_COLOR, NULL_COLOR]):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.20)
        body.set_zorder(1)
    for observed, shuffled in zip(observed_loss, null_loss):
        ax_null.plot(
            [0, 1], [observed, shuffled],
            color="#777777", linewidth=0.75, alpha=0.34, zorder=2,
        )
        ax_null.scatter(
            [0, 1], [observed, shuffled], s=21,
            facecolors=[CANONICAL_COLOR, NULL_COLOR],
            edgecolors="white", linewidths=0.35, alpha=0.78, zorder=3,
        )
    boxes = ax_null.boxplot(
        [observed_loss, null_loss], positions=[0, 1], widths=0.24,
        patch_artist=True, showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.6},
        whiskerprops={"color": "#555555", "linewidth": 1.1},
        capprops={"color": "#555555", "linewidth": 1.1},
    )
    for patch, color in zip(boxes["boxes"], [CANONICAL_COLOR, NULL_COLOR]):
        patch.set_facecolor(color)
        patch.set_alpha(0.58)
        patch.set_edgecolor(color)
        patch.set_zorder(4)
    ax_null.set_xlim(-0.55, 1.55)
    ax_null.set_ylim(0.10, 0.27)
    ax_null.set_yticks([0.10, 0.15, 0.20, 0.25])
    ax_null.set_xticks(
        [0, 1], ["Data-driven", "Shuffled null"],
        fontsize=12.0,
    )
    ax_null.set_ylabel("Held-out loss ↓", fontsize=13.5)
    ax_null.set_title("Matched-null test", fontsize=15.5,
                      fontweight="bold", pad=8)
    bracket_y, bracket_h = 0.257, 0.004
    ax_null.plot(
        [0, 0, 1, 1],
        [bracket_y, bracket_y + bracket_h, bracket_y + bracket_h, bracket_y],
        color="#333333", linewidth=0.9, clip_on=False,
    )
    ax_null.text(
        0.5, bracket_y + bracket_h + 0.001,
        _significance_label(float(primary["wilcoxon_p"])),
        ha="center", va="bottom", fontsize=15.0, fontweight="bold",
        color=CANONICAL_COLOR,
    )
    ax_null.spines[["top", "right"]].set_visible(False)
    ax_null.tick_params(length=3.5, width=0.9, labelsize=12.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / stem_name
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return {
        "schema_version": "fig4e_data_driven_snn_cohort_v1",
        "source": {
            "path": str(DEFAULT_RESULT.relative_to(ROOT)),
            "sha256": _sha256(DEFAULT_RESULT),
        },
        "display_contract": {
            "left": "nested cohort counts: eligible, model loss below matched null, and the subset also recovering both patient modes in one network",
            "right": "paired canonical-layout held-out weakest-mode loss for the data-driven model and each subject's within-shaft contact-identity permutation null median; unlabeled points and lines are subjects",
            "right_p": "pre-registered two-sided subject-level Wilcoxon signed-rank test",
            "significance_symbols": "ns: P>=0.05; *: P<0.05; **: P<0.01; ***: P<0.001",
        },
        "summary": {key: value for key, value in payload.items() if key != "rows"},
        "subject_order": payload["rows"],
        "scientific_boundary": (
            "The 11/34 joint count is descriptive: no joint KMeans-plus-loss null "
            "was pre-registered. P=0.043 applies only to the continuous canonical "
            "paired model-versus-null loss comparison. The real-geometry sensitivity "
            f"is not displayed (P={float(real_test['wilcoxon_p']):.2f}); the same-network "
            "KMeans fraction and observation-layout sensitivity gates were not met."
        ),
        "files": {
            "png": str(png.relative_to(ROOT)),
            "png_sha256": _sha256(png),
            "pdf": str(pdf.relative_to(ROOT)),
            "pdf_sha256": _sha256(pdf),
        },
    }


def build(
    output_dir: Path = DEFAULT_OUTPUT,
    result_path: Path = DEFAULT_RESULT,
    *,
    stem_name: str = "fig4-panele",
) -> dict:
    result_path = Path(result_path)
    result = json.loads(result_path.read_text())
    metadata = render(result, Path(output_dir), stem_name=stem_name)
    metadata["source"] = {
        "path": str(result_path.relative_to(ROOT)),
        "sha256": _sha256(result_path),
    }
    metadata_path = Path(output_dir) / f"{stem_name}-metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    metadata = build(args.output_dir, args.result)
    print(json.dumps(metadata["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
