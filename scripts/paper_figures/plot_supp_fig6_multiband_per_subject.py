#!/usr/bin/env python3
"""Build Supplementary Figure 6 on one aligned seven-band canvas.

Panel A preserves the accepted cohort subject-delta/violin/maxT-FWER grammar.
Panel B shows the same 17 subjects as a subject-by-band own-null heatmap.
The two panels share identical band columns; their stars deliberately retain
different, explicitly documented inferential meanings.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_figures.patient_public_labels import public_patient_label  # noqa: E402
CALC = (
    ROOT
    / "results/topic5_ictal_recruitment/field_concordance_grid_method_sensitivity/"
    "n161_subject_fixed"
)
OUT_ROOT = ROOT / "results/paper-ready-figure/supp_fig6_multiband_field_concordance"
FIG_DIR = OUT_ROOT / "figures"
NULL_NPZ = CALC / "multiband_subject_null_draws.npz"
SUBJECT_CSV = CALC / "multiband_subject.csv"
COHORT_CSV = CALC / "multiband_cohort.csv"
OMNIBUS_JSON = CALC / "multiband_band_omnibus.json"

BAND_ORDER = [
    "delta_HYP_slow",
    "theta_preictal_PAC",
    "alpha_sharp_leq13",
    "beta_LVFA_low",
    "gamma_LVFA",
    "hg_low_ripple",
    "ripple_high",
]
BAND_LABELS = [
    "δ\n1–4",
    "θ\n4–8",
    "α\n8–13",
    "β\n13–30",
    "γ\n30–80",
    "R\n80–150",
    "FR\n150–250",
]
SIG_COLOR = "#B64F4F"
NS_COLOR = "#D7D7D7"
POINT_COLOR = "#333333"


def _display_subject(subject: str) -> str:
    if subject.startswith("epilepsiae_"):
        return public_patient_label("epilepsiae", subject.split("_", 1)[1])
    if subject.startswith("yuquan_"):
        return public_patient_label("yuquan", subject.split("_", 1)[1])
    return subject


def _load() -> dict:
    z = np.load(NULL_NPZ)
    subjects = [str(value) for value in z["subjects"]]
    bands = [str(value) for value in z["bands"]]
    if bands != BAND_ORDER:
        raise RuntimeError(f"band order mismatch: {bands}")
    observed = np.asarray(z["D"], dtype=float)
    null = np.asarray(z["N"], dtype=float)
    null_median = np.nanmedian(null, axis=2)
    delta = observed - null_median
    subject_p = (1 + np.sum(null >= observed[:, :, None], axis=2)) / (
        null.shape[2] + 1
    )

    table = pd.read_csv(SUBJECT_CSV)
    pivot = (
        table.pivot(index="subject", columns="band", values="delta")
        .reindex(index=subjects, columns=BAND_ORDER)
        .to_numpy(dtype=float)
    )
    if not np.allclose(delta, pivot, equal_nan=True, atol=1e-10):
        raise RuntimeError("NPZ delta does not match multiband_subject.csv")
    cohort = pd.read_csv(COHORT_CSV).set_index("band").reindex(BAND_ORDER)
    cohort_median = cohort["delta_cohort_median"].to_numpy(dtype=float)
    pfwer = cohort["seven_band_maxt_pfwer"].to_numpy(dtype=float)

    order = sorted(
        range(len(subjects)),
        key=lambda idx: (
            -int(np.sum(subject_p[idx] < 0.05)),
            -float(np.nanmedian(delta[idx])),
            subjects[idx],
        ),
    )
    return {
        "subjects": subjects,
        "delta": delta,
        "subject_p": subject_p,
        "cohort_median": cohort_median,
        "cohort_pfwer": pfwer,
        "order": np.asarray(order, dtype=int),
        "n_null": int(null.shape[2]),
    }


def _draw_cohort(ax: plt.Axes, payload: dict) -> None:
    rng = np.random.default_rng(17)
    all_values = payload["delta"][np.isfinite(payload["delta"])]
    for col in range(len(BAND_ORDER)):
        values = payload["delta"][:, col]
        values = values[np.isfinite(values)]
        significant = bool(payload["cohort_pfwer"][col] < 0.05)
        color = SIG_COLOR if significant else NS_COLOR
        if values.size >= 2 and np.nanmax(values) > np.nanmin(values):
            body = ax.violinplot(
                [values],
                positions=[col],
                widths=0.72,
                showmedians=False,
                showextrema=False,
            )["bodies"][0]
            body.set_facecolor(color)
            body.set_edgecolor(SIG_COLOR if significant else "#9B9B9B")
            body.set_linewidth(0.8)
            body.set_alpha(0.52 if significant else 0.42)
        ax.scatter(
            col + rng.uniform(-0.070, 0.070, values.size),
            values,
            s=15,
            color=POINT_COLOR,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.28,
            zorder=4,
        )
        ax.hlines(
            payload["cohort_median"][col],
            col - 0.27,
            col + 0.27,
            color="black",
            lw=2.0,
            zorder=5,
        )

    data_min = float(np.nanmin(all_values))
    data_max = float(np.nanmax(all_values))
    y_lo = min(-0.62, data_min - 0.04)
    y_hi = max(0.48, data_max + 0.13)
    star_y = data_max + 0.075
    for col, p_value in enumerate(payload["cohort_pfwer"]):
        if p_value < 0.05:
            ax.text(
                col,
                star_y,
                "*",
                ha="center",
                va="center",
                color=SIG_COLOR,
                fontsize=15,
                fontweight="bold",
            )
    ax.axhline(0.0, color="0.48", lw=0.75)
    ax.set_xlim(-0.5, len(BAND_ORDER) - 0.5)
    ax.set_ylim(y_lo, y_hi)
    ax.set_ylabel("Field concordance − own null (Δ)", fontsize=8)
    ax.set_xticks(np.arange(len(BAND_ORDER)))
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", labelsize=7, length=2.5)
    ax.grid(axis="y", color="0.91", lw=0.55, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_linewidth(0.7)


def _draw_heatmap(
    fig: plt.Figure, ax: plt.Axes, cax: plt.Axes, payload: dict
) -> tuple[list[str], np.ndarray]:
    order = payload["order"]
    values = payload["delta"][order]
    pvalues = payload["subject_p"][order]
    subjects = [payload["subjects"][idx] for idx in order]
    image_values = np.vstack([values, payload["cohort_median"][None, :]])
    max_abs = float(np.nanmax(np.abs(image_values)))
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    image = ax.imshow(
        image_values,
        aspect="auto",
        cmap="RdBu_r",
        norm=norm,
        interpolation="nearest",
        extent=(-0.5, len(BAND_ORDER) - 0.5, len(subjects) + 0.5, -0.5),
        rasterized=True,
    )
    ax.axhline(len(subjects) - 0.5, color="black", lw=1.0)
    for row in range(len(subjects)):
        for col in range(len(BAND_ORDER)):
            if np.isfinite(pvalues[row, col]) and pvalues[row, col] < 0.05:
                artist = ax.text(
                    col,
                    row,
                    "*",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8.8,
                    fontweight="bold",
                )
                artist.set_path_effects(
                    [path_effects.withStroke(linewidth=0.9, foreground="black")]
                )
    for col, value in enumerate(payload["cohort_median"]):
        color = "white" if abs(value) > 0.42 * max_abs else "black"
        ax.text(
            col,
            len(subjects),
            f"{value:+.02f}",
            ha="center",
            va="center",
            fontsize=6.0,
            color=color,
        )
    ax.set_xlim(-0.5, len(BAND_ORDER) - 0.5)
    ax.set_xticks(np.arange(len(BAND_ORDER)), BAND_LABELS)
    ax.set_yticks(
        np.arange(len(subjects) + 1),
        [_display_subject(subject) for subject in subjects] + ["Cohort median"],
    )
    ax.tick_params(
        axis="x",
        top=True,
        labeltop=True,
        bottom=False,
        labelbottom=False,
        labelsize=7.2,
        length=0,
        pad=3,
    )
    ax.tick_params(axis="y", labelsize=6.8, length=0)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label(
        "Field concordance Δ\n(observed − own-null median)",
        fontsize=6.8,
        labelpad=7,
    )
    cbar.ax.tick_params(labelsize=6.2, length=2)
    cbar.outline.set_linewidth(0.55)
    return subjects, pvalues


def main() -> None:
    payload = _load()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    fig = plt.figure(figsize=(6.65, 5.85), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 0.032],
        height_ratios=[1.05, 2.80],
        hspace=0.16,
        wspace=0.065,
        left=0.155,
        right=0.910,
        bottom=0.060,
        top=0.935,
    )
    ax_cohort = fig.add_subplot(grid[0, 0])
    ax_heat = fig.add_subplot(grid[1, 0], sharex=ax_cohort)
    fig.add_subplot(grid[0, 1]).axis("off")
    cax = fig.add_subplot(grid[1, 1])
    _draw_cohort(ax_cohort, payload)
    subjects, _ = _draw_heatmap(fig, ax_heat, cax, payload)

    fig.canvas.draw()
    fig.text(
        0.012,
        0.982,
        "A",
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
    )
    fig.text(
        0.012,
        ax_heat.get_position().y1 + 0.048,
        "B",
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
    )

    stem = FIG_DIR / "supp_fig6_multiband_cohort_and_subject_heatmap"
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary_rows = []
    for rank, idx in enumerate(payload["order"]):
        summary_rows.append(
            {
                "display_order": rank + 1,
                "patient_id": _display_subject(payload["subjects"][idx]),
                "n_self_null_p_lt_0_05": int(
                    np.sum(payload["subject_p"][idx] < 0.05)
                ),
                "median_delta": float(np.nanmedian(payload["delta"][idx])),
                "delta_by_band": {
                    band: float(payload["delta"][idx, col])
                    for col, band in enumerate(BAND_ORDER)
                },
                "subject_self_null_p_by_band": {
                    band: float(payload["subject_p"][idx, col])
                    for col, band in enumerate(BAND_ORDER)
                },
            }
        )
    omnibus = json.loads(OMNIBUS_JSON.read_text(encoding="utf-8"))
    caption_title = (
        "Multiband sensitivity analysis of ictal-to-interictal field concordance."
    )
    caption_body = (
        f"**A,** Patient-level field-concordance difference (\u0394) across seven "
        f"frequency bands in {len(subjects)} patients. For each patient and "
        f"band, D is the median early-ictal dense-grid spatial concordance with "
        f"the better-matching of two predefined interictal template fields "
        f"after resolving sign and transverse-mirror symmetry, and \u0394 is D minus the "
        f"median of that patient's all-contact permutation null "
        f"({payload['n_null']} draws); points denote patients, violins the "
        f"distributions and black horizontal lines the cohort medians. Red "
        f"violins and stars denote bands significant under a coherent "
        f"seven-band maxT family-wise error-rate correction (*P_FWER < 0.05). "
        f"**B,** Heat map of the same patient-by-band \u0394 values, with rows "
        f"ordered by the number of bands exceeding the patient's own null and "
        f"the bottom row showing cohort medians; outlined stars denote "
        f"one-sided empirical patient-versus-own-null P < 0.05 and are not "
        f"cohort-level or multiplicity-corrected. Bands are \u03b4 (1\u20134 Hz), "
        f"\u03b8 (4\u20138 Hz), \u03b1 (8\u201313 Hz), \u03b2 (13\u201330 Hz), "
        f"\u03b3 (30\u201380 Hz), R (80\u2013150 Hz) and FR (150\u2013250 Hz)."
    )
    metadata = {
        "figure": "Supplementary Figure 6",
        "caption": (
            f"Supplementary Fig. 6 | {caption_title} "
            f"{caption_body.replace('**', '')}"
        ),
        "source": str(NULL_NPZ.relative_to(ROOT)),
        "n_subjects": len(subjects),
        "n_bands": len(BAND_ORDER),
        "n_null_draws_per_subject_band": payload["n_null"],
        "panel_a": {
            "cell_source": str(SUBJECT_CSV.relative_to(ROOT)),
            "estimand": "subject-level D minus own all-contact null median",
            "star": "seven-band coherent maxT-FWER P<0.05",
            "cohort_pfwer_by_band": {
                band: float(payload["cohort_pfwer"][col])
                for col, band in enumerate(BAND_ORDER)
            },
        },
        "panel_b": {
            "cell_value": "D - median(subject all-contact null draws)",
            "cell_star": (
                "one-sided empirical subject-vs-own-null P<0.05; "
                "not cohort maxT-FWER"
            ),
            "row_sort": (
                "descending number of subject-self-null P<0.05 cells, then "
                "descending median delta, then subject identifier"
            ),
            "subjects": summary_rows,
        },
        "direct_between_band_test": omnibus,
        "claim_boundary": (
            "Bandwise significance against each band's own null does not "
            "establish a difference between bands; use the direct omnibus "
            "test for that question."
        ),
        "outputs": {
            "png": str(stem.with_suffix(".png").relative_to(ROOT)),
            "pdf": str(stem.with_suffix(".pdf").relative_to(ROOT)),
        },
    }
    (OUT_ROOT / "supp_fig6_multiband_combined_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (FIG_DIR / "README.md").write_text(
        "### supp_fig6_multiband_cohort_and_subject_heatmap.png\n\n"
        f"**Supplementary Fig. 6 | {caption_title}**\n\n"
        f"{caption_body}\n\n"
        "**关注点**：A 的星号是队列 maxT-FWER，B 的 cell 星号是患者自身 "
        "null P<0.05，二者不可互换。频带间直接 omnibus 检验为非显著，因此"
        "不能由 δ/θ 的星号声称它们显著强于其他频带。\n",
        encoding="utf-8",
    )
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
