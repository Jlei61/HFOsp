#!/usr/bin/env python3
"""Paper-ready fixed-window T_spectral field-concordance supplement.

This renderer reads only closed subject/cohort tables and the subject-level
median activation vectors written by ``run_topic5_tspectral_field_concordance``.
It reuses the accepted Topic-5 field-atlas, margin-board and Fig3-Sup1
multiband painters.  It never rebuilds an axis, refits a plane, or recomputes a
spatial null.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from scripts.plot_topic5_field_concordance import plot_atlas  # noqa: E402
from scripts.plot_topic5_field_concordance_best_board import (  # noqa: E402
    plot_or_margin_board,
)
from scripts.plot_topic5_v2_phase1_figures import (  # noqa: E402
    paint_observed_maxab_heatmap,
    paint_null_per_band_axis,
    plot_null_per_band_figure,
)
from scripts.paper_figures.plot_fig3_field_concordance_cohort_stat import (  # noqa: E402
    _add_sig_bracket,
    _fmt_p,
    _p_stars,
    plot_paired_data_null_groups,
)
from src.topic5_template_axis_field import score_field, scorers_from_interictal_record  # noqa: E402


DEFAULT_ANALYSIS = ROOT / "results/topic5_ictal_recruitment/tspectral_field_concordance"
FIELD_ROOT = ROOT / "results/interictal_propagation_masked/template_gradient_fields/per_subject"
PAPER_ROOT = ROOT / "results/paper-ready-figure/fig3-sup-tspectral-field-concordance"
FIGURES = PAPER_ROOT / "figures"

WINDOW_ORDER = ("distal", "pre20", "pre10", "post10", "post20", "late20_30")
WINDOW_LABEL = {
    "distal": "distal",
    "pre20": "−20–0 s",
    "pre10": "−10–0 s",
    "post10": "0–10 s",
    "post20": "0–20 s",
    "late20_30": "20–30 s",
}
WINDOW_TILE = {
    "distal": "D",
    "pre20": "P20",
    "pre10": "P10",
    "post10": "A10",
    "post20": "A20",
    "late20_30": "L30",
}
WINDOW_COLOR = {
    "distal": "#6F7B83",
    "pre20": "#4C78A8",
    "pre10": "#72A0C1",
    "post10": "#D0644D",
    "post20": "#B4473A",
    "late20_30": "#8E5E9D",
}
BAND_ORDER = ("broadband_1_150", "hfa_60_100", "gamma_30_80_sensitivity")
BAND_SHORT = {
    "broadband_1_150": "BB\n1–150",
    "hfa_60_100": "HFA\n60–100",
    "gamma_30_80_sensitivity": "Gamma sens.\n30–80",
}
PRIMARY_ATLAS = {
    "broadband_1_150": ("broadband", "Broadband phenotype · 1–150 Hz"),
    "hfa_60_100": ("hfa", "Gamma non-broadband phenotype · HFA 60–100 Hz"),
}
STEMS = {
    "atlas_broadband": "field_concordance_atlas_broadband",
    "atlas_hfa": "field_concordance_atlas_hfa",
    "board": "field_concordance_or_margin_board",
    "observed": "fig3sup1_A_observed_maxAB",
    "null": "fig3sup1_B_null_per_band",
    "pooled": "phenotype_matched_cohort_by_window",
    "subject_data_null": "phenotype_matched_subject_data_vs_null_by_window",
    "relation": "phenotype_matched_relation_by_window",
    "band_exploratory": "phenotype_matched_exploratory_band_by_window",
}


def _rank01(values) -> np.ndarray:
    values = np.asarray(values, float)
    out = np.full(values.shape, np.nan)
    finite = np.isfinite(values)
    if not finite.any():
        return out
    ranked = pd.Series(values[finite]).rank(method="average").to_numpy(float)
    if len(ranked) == 1:
        out[finite] = 0.5
    else:
        out[finite] = (ranked - 1.0) / (len(ranked) - 1.0)
    return out


def _limits(points: np.ndarray, sigma: float) -> tuple[tuple[float, float], tuple[float, float]]:
    points = np.asarray(points, float)
    rx = max(float(np.ptp(points[:, 0])), 1e-3)
    ry = max(float(np.ptp(points[:, 1])), 1e-3)
    padx = max(2.5 * float(sigma), 0.08 * rx)
    pady = max(2.5 * float(sigma), 0.08 * ry)
    return ((float(points[:, 0].min() - padx), float(points[:, 0].max() + padx)),
            (float(points[:, 1].min() - pady), float(points[:, 1].max() + pady)))


def _atlas_rows(analysis: Path, subject_table: pd.DataFrame, band: str) -> list[dict]:
    rows = []
    frame = subject_table[subject_table.band == band].copy()
    meta = (frame[["subject", "dataset", "axis_relation"]].drop_duplicates("subject")
            .sort_values(["dataset", "axis_relation", "subject"], na_position="last"))
    for subject in meta.subject:
        subject_json = json.loads((analysis / "per_subject" / f"{subject}.json").read_text())
        display = subject_json["fixed_window_activation_subject_median"].get(band)
        if not display:
            continue
        field_record = json.loads((FIELD_ROOT / f"{subject}.json").read_text())
        scorers = scorers_from_interictal_record(field_record)
        field = field_record["interictal_field"]
        subject_stats = frame[frame.subject == subject].set_index("fixed_window")
        for window in WINDOW_ORDER:
            if window not in subject_stats.index or window not in display["windows"]:
                continue
            activation = np.asarray(display["windows"][window], float)
            scored = {label: score_field(scorers[f"own_{label}"], activation)
                      for label in ("a", "b")}
            best = max(scored, key=lambda label: scored[label]["abs_r"])
            model = field["field_models"][f"own_{best}"]
            points = np.asarray(model["points"], float)
            sigma = float(model["sigma"])
            xlim, ylim = _limits(points, sigma)
            stat = subject_stats.loc[window]
            rows.append({
                "ds_sid": f"{subject} · {WINDOW_TILE[window]} · T{best.upper()}",
                "xs": points[:, 0], "ys": points[:, 1],
                "inter": _rank01(model["template_field"]),
                "ict": _rank01(activation), "support": np.asarray(model["support"], float),
                "soz": np.zeros(len(points), bool), "xlim": xlim, "ylim": ylim,
                "sigma": sigma, "sign_neg": bool(scored[best]["signed_r"] < 0),
                "r": float(stat.own_maxab),
                "p95": float(stat.own_within_shaft_null_p95_folded),
                "passed": bool(stat.own_within_shaft_exceeds_p95),
                "margin": float(stat.own_within_shaft_margin_to_p95),
                "n_ch": int(stat.n_finite),
            })
    return rows


def _plot_atlases(analysis: Path, subject_table: pd.DataFrame) -> dict:
    made = {}
    for band, (activation, title) in PRIMARY_ATLAS.items():
        rows = [row for row in _atlas_rows(analysis, subject_table, band)
                if " · A10 · " in row["ds_sid"]]
        for row in rows:
            row["ds_sid"] = row["ds_sid"].split(" · ", 1)[0]
        stem_key = "atlas_broadband" if band == "broadband_1_150" else "atlas_hfa"
        output = FIGURES / f"{STEMS[stem_key]}.png"
        plot_atlas(
            rows, activation, ncols=6, preserve_order=False,
            title_text=f"Narrow own-field concordance · {title} · 0–10 s after T_spectral",
            subtitle_text=("per subject: frozen interictal TA/TB field vs phenotype-matched ictal "
                           "energy field · |r| = seizure-median own maxAB · dark frame = above "
                           "subject within-shaft null Q95"),
            output_path=output, save_pdf=True, tile_fontsize=9.0,
        )
        made[stem_key] = {"rows": len(rows), "subjects": int(subject_table[
            subject_table.band == band].subject.nunique())}
    return made


def _plot_board(pooled_table: pd.DataFrame) -> int:
    candidates = [{"name": WINDOW_LABEL[window], "color": WINDOW_COLOR[window]}
                  for window in WINDOW_ORDER]
    rows = []
    for subject, group in pooled_table.groupby("subject", sort=True):
        vals = {}
        for record in group.itertuples():
            vals[WINDOW_LABEL[record.fixed_window]] = {
                "margin": float(record.own_within_shaft_margin_to_p95),
                "pass": bool(record.own_within_shaft_exceeds_p95),
            }
        best_name = max(vals, key=lambda name: vals[name]["margin"])
        best = vals[best_name]
        rows.append({
            "subject_id": str(subject), "or_pass": any(v["pass"] for v in vals.values()),
            "margin": best["margin"], "color": next(c["color"] for c in candidates
                                                       if c["name"] == best_name),
            "vals": vals, "best": best_name,
        })
    rows.sort(key=lambda row: (-row["margin"], row["subject_id"]))
    output = FIGURES / f"{STEMS['board']}.png"
    plot_or_margin_board(
        rows, candidates, output,
        "Pooled phenotype-matched narrow own-field concordance by fixed T_spectral window",
        xlabel="descriptive best-window margin: subject observed maxAB − within-shaft null Q95",
        candidate_title="above Q95?",
        footer_text=" ",
        save_pdf=True, open_label="open square = at/below Q95",
    )
    return len(rows)


def _plot_pooled_by_window(pooled_subject: pd.DataFrame,
                           pooled_cohort: pd.DataFrame) -> None:
    stats = pooled_cohort[
        (pooled_cohort.dataset_stratum == "combined") &
        (pooled_cohort.field_plane == "own") &
        (pooled_cohort.null_type == "within_shaft")
    ].set_index("fixed_window")
    deltas = {
        window: pd.to_numeric(
            pooled_subject.loc[pooled_subject.fixed_window == window,
                               "own_within_shaft_delta_null_median"],
            errors="coerce",
        ).dropna().to_numpy(float)
        for window in WINDOW_ORDER
    }
    tested_means = {window: float(stats.loc[window, "mean"]) for window in WINDOW_ORDER}
    pvalues = {
        window: float(stats.loc[window, "two_sided_sign_flip_maxt_p"])
        for window in WINDOW_ORDER
    }
    sample_sizes = {window: int(stats.loc[window, "n_subjects"]) for window in WINDOW_ORDER}
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.3), sharey=True)
    for panel, (ax, windows) in enumerate(zip(axes, (WINDOW_ORDER[:3], WINDOW_ORDER[3:]))):
        annotations = {
            window: (f"{_p_stars(pvalues[window])}\n"
                     f"p={_fmt_p(pvalues[window])}, n={sample_sizes[window]}")
            for window in windows
        }
        paint_null_per_band_axis(
            ax, windows, WINDOW_LABEL, deltas, tested_means, annotations,
            seed=20260716 + 10 * panel,
            ylabel=("own maxAB − null median (Δ)"
                    if panel == 0 else None),
        )
        ax.text(0.0, 1.055, "baseline / pre-T_spectral" if panel == 0
                else "post-T_spectral", transform=ax.transAxes, ha="left", va="bottom",
                fontsize=11, fontweight="bold")
    handles = [
        Line2D([0], [0], marker="o", ls="none", color="#333333", ms=6,
               label="per-subject Δ"),
        Line2D([0], [0], color="black", lw=2.8, label="cohort mean (tested)"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.985),
               framealpha=0.92)
    fig.suptitle(
        "Phenotype-matched own-field concordance · combined subject cohort\n"
        "within-shaft spatial null · two-sided sign-flip maxT across six windows",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84), w_pad=2.5)
    output = FIGURES / f"{STEMS['pooled']}.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_subject_data_vs_null_by_window(
        pooled_subject: pd.DataFrame, pooled_cohort: pd.DataFrame) -> None:
    """Accepted Fig3 paired Data-vs-Null grammar, one subject value per window."""
    stats = pooled_cohort[
        (pooled_cohort.dataset_stratum == "combined") &
        (pooled_cohort.field_plane == "own") &
        (pooled_cohort.null_type == "within_shaft")
    ].set_index("fixed_window")
    groups = []
    for window in WINDOW_ORDER:
        frame = pooled_subject[pooled_subject.fixed_window == window].copy()
        frame = frame.sort_values("subject")
        rows = [
            {
                "subject_id": str(row.subject),
                "data": float(row.own_maxab),
                "null": float(row.own_within_shaft_null_median_folded),
            }
            for row in frame.itertuples()
            if np.isfinite(row.own_maxab)
            and np.isfinite(row.own_within_shaft_null_median_folded)
        ]
        groups.append({
            "label": WINDOW_LABEL[window],
            "rows": rows,
            "summary": {"n": len(rows)},
            "display_p": float(stats.loc[window, "two_sided_sign_flip_maxt_p"]),
            "p_label": "maxT p",
        })
    output = FIGURES / f"{STEMS['subject_data_null']}.png"
    plot_paired_data_null_groups(
        groups, output, output.with_suffix(".pdf"),
        ylabel="Phenotype-matched field concordance |r|",
        seed=20260716,
    )
    metadata = {
        "figure": output.name,
        "unit": "subject",
        "grouping_dimension": "fixed_window",
        "band_dimension": None,
        "data": "subject median phenotype-matched own maxAB",
        "null": "subject draw-wise seizure-folded within-shaft null median",
        "p_value": "two-sided subject sign-flip maxT across six fixed windows",
        "groups": [
            {
                "fixed_window": window,
                "label": group["label"],
                "n_subjects": len(group["rows"]),
                "maxT_p": group["display_p"],
                "data_median": float(np.median([row["data"] for row in group["rows"]])),
                "null_median": float(np.median([row["null"] for row in group["rows"]])),
                "subjects": [row["subject_id"] for row in group["rows"]],
            }
            for window, group in zip(WINDOW_ORDER, groups)
        ],
    }
    output.with_name(f"{STEMS['subject_data_null']}_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


def _plot_relation_by_window(pooled_subject: pd.DataFrame,
                             relation_stats: pd.DataFrame) -> None:
    subject = pooled_subject[
        pooled_subject.geometry_2d_supported.fillna(False).astype(bool)
    ].copy()
    stats = relation_stats[
        (relation_stats.quality_scope == "geometry_2d_supported") &
        (relation_stats.dataset_stratum == "combined")
    ].set_index("fixed_window")
    relations = ("reversed", "different", "same")
    labels = {"reversed": "reversed\ncollinear", "different": "different /\nnon-collinear",
              "same": "same\ncollinear"}
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.6), sharey=True)
    for j, (ax, window) in enumerate(zip(axes.flat, WINDOW_ORDER)):
        values = {
            relation: pd.to_numeric(subject.loc[
                (subject.fixed_window == window) & (subject.axis_relation == relation),
                "own_within_shaft_delta_null_median"], errors="coerce"
            ).dropna().to_numpy(float)
            for relation in relations
        }
        row = stats.loc[window]
        means = {relation: float(row[f"{relation}_mean"]) for relation in relations}
        annotations = {relation: f"n={len(values[relation])}" for relation in relations}
        paint_null_per_band_axis(
            ax, relations, labels, values, means, annotations,
            seed=20260716 + 10 * j,
            ylabel=("observed − within-shaft null median" if j % 3 == 0 else None),
        )
        ax.text(0.0, 1.055, WINDOW_LABEL[window], transform=ax.transAxes,
                ha="left", va="bottom", fontsize=11, fontweight="bold")
        finite = np.concatenate([v for v in values.values() if len(v)])
        if len(finite):
            y0, y1 = ax.get_ylim()
            span = max(y1 - y0, 0.2)
            y = float(np.nanmax(finite) + 0.08 * span)
            p = float(row.two_sided_label_permutation_maxt_p)
            _add_sig_bracket(ax, 0, 1, y, f"{_p_stars(p)}  p={_fmt_p(p)}")
            ax.set_ylim(y0, max(y1, y + 0.16 * span))
    y0, y1 = axes[0, 0].get_ylim()
    for ax in axes.flat:
        ax.set_ylim(y0, y1 + 0.16 * (y1 - y0))
    handles = [
        Line2D([0], [0], marker="o", ls="none", color="#333333", ms=6,
               label="per-subject Δ"),
        Line2D([0], [0], color="black", lw=2.8, label="group mean (tested)"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.985),
               framealpha=0.92)
    fig.suptitle(
        "Pre-existing TA/TB relation strata · phenotype-matched own field\n"
        "reversed vs different: two-sided label-permutation maxT; same retained descriptively",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    output = FIGURES / f"{STEMS['relation']}.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_exploratory_band_by_window(subject_table: pd.DataFrame,
                                     exploratory_stats: pd.DataFrame) -> None:
    stats = exploratory_stats[
        exploratory_stats.dataset_stratum == "combined"
    ].set_index(["fixed_window", "band"])
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.6), sharey=True)
    for j, (ax, window) in enumerate(zip(axes.flat, WINDOW_ORDER)):
        values = {
            band: pd.to_numeric(subject_table.loc[
                (subject_table.fixed_window == window) & (subject_table.band == band),
                "own_within_shaft_delta_null_median"], errors="coerce"
            ).dropna().to_numpy(float)
            for band in BAND_ORDER
        }
        means = {band: float(stats.loc[(window, band), "mean"]) for band in BAND_ORDER}
        annotations = {
            band: (f"{_p_stars(float(stats.loc[(window, band), 'two_sided_sign_flip_maxt_p']))}  "
                   f"p={float(stats.loc[(window, band), 'two_sided_sign_flip_maxt_p']):.3g}\n"
                   f"n={int(stats.loc[(window, band), 'n_subjects'])}")
            for band in BAND_ORDER
        }
        paint_null_per_band_axis(
            ax, BAND_ORDER, BAND_SHORT, values, means, annotations,
            seed=20260716 + 10 * j,
            ylabel=("observed − within-shaft null median" if j % 3 == 0 else None),
        )
        ax.set_title(WINDOW_LABEL[window], loc="left", pad=18,
                     fontsize=11, fontweight="bold")
    handles = [
        Line2D([0], [0], marker="o", ls="none", color="#333333", ms=6,
               label="per-subject Δ"),
        Line2D([0], [0], color="black", lw=2.8, label="cohort mean (tested)"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.985),
               framealpha=0.92)
    fig.suptitle(
        "Exploratory phenotype-separated narrow-band concordance by fixed window\n"
        "two-sided subject sign-flip maxT across six windows within each readout",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    output = FIGURES / f"{STEMS['band_exploratory']}.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _condition(window: str, band: str) -> str:
    return f"{window}|{band}"


def _plot_multiband_observed(subject_table: pd.DataFrame) -> int:
    frame = subject_table[subject_table.fixed_window == "post10"].copy()
    pivot = frame.pivot(index="subject", columns="band", values="own_maxab").reindex(columns=BAND_ORDER)
    significance = {(str(row.subject), row.band): bool(row.own_within_shaft_exceeds_p95)
                    for row in frame.itertuples()}
    n_sig = pd.Series({subject: sum(significance.get((str(subject), band), False)
                                    for band in BAND_ORDER) for subject in pivot.index})
    order = pd.DataFrame({"n_sig": n_sig, "med": pivot.median(axis=1)})
    pivot = pivot.loc[order.sort_values(["n_sig", "med"], ascending=[False, False]).index]
    output = FIGURES / f"{STEMS['observed']}.png"
    paint_observed_maxab_heatmap(
        pivot, significance, BAND_SHORT,
        f"F1 · narrow own field · 0–10 s after T_spectral (n={len(pivot)}) — observed maxAB |corr|\n"
        "rows ↓ by # above-null bands · * = subject observed > within-shaft null Q95",
        output, group_boundaries=(), figsize=(9.5, 0.46 * (len(pivot) + 1) + 2.0), save_pdf=True,
        median_label="combined descriptive median",
    )
    return int(len(pivot))


def _plot_multiband_null(subject_table: pd.DataFrame,
                         exploratory_stats: pd.DataFrame) -> None:
    subject = subject_table[subject_table.fixed_window == "post10"].copy()
    stats = exploratory_stats[
        (exploratory_stats.dataset_stratum == "combined") &
        (exploratory_stats.fixed_window == "post10")
    ].set_index("band")
    deltas = {band: pd.to_numeric(subject.loc[subject.band == band,
                                               "own_within_shaft_delta_null_median"],
                                  errors="coerce").dropna().to_numpy(float)
              for band in BAND_ORDER}
    tested_means = {band: float(stats.loc[band, "mean"]) for band in BAND_ORDER}
    pvalues = {band: float(stats.loc[band, "two_sided_sign_flip_maxt_p"])
               for band in BAND_ORDER}
    sample_sizes = {band: int(stats.loc[band, "n_subjects"]) for band in BAND_ORDER}
    output = FIGURES / f"{STEMS['null']}.png"
    plot_null_per_band_figure(
        BAND_ORDER, BAND_SHORT, deltas, tested_means, pvalues, sample_sizes,
        "F2 · 0–10 s narrow own-field concordance\n"
        "combined subject cohort · within-shaft null · six-window maxT",
        output, ylabel="observed own maxAB − within-shaft null median (Δ)",
        save_pdf=True, seed=20260716,
    )


def _write_readme() -> None:
    text = f"""# Fig3 supplement：T_spectral 固定时间窗 field concordance

### {STEMS['atlas_broadband']}.png / .pdf
严格复用既有 `field_concordance_atlas_broadband` 的成对 field-map painter，主窗口固定为 T_spectral 后 0–10 s。每个 tile 左图是冻结间期 TA/TB own field，右图是 strict-broadband seizure 的 1–150 Hz `delta_E`；深色边框表示该患者 observed own-maxAB 高于 within-shaft null Q95。

**关注点**：看不同窗口内间期与发作场是否相似并超过保留杆间整体热度的解剖 null；不要把边框数量解释成 cohort gate。

### {STEMS['atlas_hfa']}.png / .pdf
绘图语法与 broadband atlas 完全相同，窗口同样固定为 0–10 s；纳入互斥的 gamma-nonbroadband seizure，主 readout 为 HFA 60–100 Hz。每个 tile 的 TA/TB 与 mirror 选择只用于 frozen own-maxAB 诊断，未在 own/shared 之间再取最大值。

**关注点**：比较同一患者在 distal、pre、post 和 20–30 s 窗口的空间场一致性，而不是检验筛选后能量是否升高。

### {STEMS['board']}.png / .pdf
复用既有 margin-board painter。每次 seizure 先按表型选唯一 readout（strict broadband→1–150 Hz；gamma-nonbroadband→HFA 60–100 Hz），再在患者内取中位数；右侧六列方块表示各固定窗口 observed 是否超过患者自己的 within-shaft null Q95。左侧大点仅显示最大窗口 margin 以便排版，不构成 OR 检验，也不用于选择“最佳窗口”。

**关注点**：读取右侧逐窗口方块和所有彩色刻度；不要把左侧描述性最大值当正式跨窗口推断。

### {STEMS['observed']}.png / .pdf
严格复用旧 Fig3-Sup1A 的 observed-maxAB heatmap painter，只画冻结的 narrow own fields 和 0–10 s 主窗口。三列依次为 Broadband 1–150 Hz、HFA 60–100 Hz、Gamma 30–80 Hz sensitivity；星号表示该 subject×band observed 高于 within-shaft null Q95。

**关注点**：HFA 60–100 Hz 是 gamma-nonbroadband 主结果，30–80 Hz 只作 label-matched sensitivity；空白格来自互斥表型分母，不是缺失后填补。

### {STEMS['null']}.png / .pdf
严格复用旧 Fig3-Sup1B 的单轴 violin、subject point、黑色 cohort 统计横杠和顶部 `* / n.s.` 语法，窗口固定为 0–10 s。纵轴为 subject observed own-maxAB 减去其 draw-wise seizure→subject folded within-shaft null median；同时显示 combined cohort subject-level 双侧 sign-flip 的 maxT 校正 p 和 n。

**关注点**：这是 combined cohort 的探索性频带分层；黑色横杠是实际检验的 subject mean delta，p 值对六个固定窗口做 maxT 校正，Yuquan 已纳入并在数值表中保留 dataset 标识。

### {STEMS['pooled']}.png / .pdf
严格复用旧 Fig3-Sup1B 的 violin、逐 subject 点、黑色统计横杠和显著性标注语法。六列是预先固定的 distal、pre20、pre10、post10、post20 和 20–30 s；每名患者每个窗口只有一个表型匹配分数，黑色横杠为实际检验的 subject mean delta，顶部显示六窗口 maxT 校正 p 与 n。

**关注点**：这是本轮不再按频带拆分的主要 cohort 图；判断冻结间期 own field 与表型匹配 ictal field 的相似性是否整体高于 within-shaft null，而不是比较能量幅度。

### {STEMS['subject_data_null']}.png / .pdf
严格复用 `field_concordance_cohort_stat.png` 的成对 Data–Null painter：每个固定窗口都是一组 violin、box、逐 subject 点和同一 subject 的灰色配对线。这里没有频带维度；每名患者的 Data 是其所有目标发作按表型匹配 readout 后折叠得到的 own-maxAB，Null 是同一患者 draw-wise 折叠的 within-shaft null median，括号和下方文字显示六窗口 maxT 校正 p 与 n。

**关注点**：直接读取同一患者的 Data 是否系统性高于其空间 null；六组只代表预先固定的时间窗，不代表六个频带或六个独立 cohort。

### {STEMS['relation']}.png / .pdf
每个 panel 继续调用旧 Fig3-Sup1B 的 violin、subject point 和黑色 group-mean painter，并调用既有显著性 bracket。只在预先定义的 `geometry_2d_supported` 患者中比较 reversed 与 different；same 作为第三组完整显示，括号给出 reversed-vs-different 的六窗口 maxT 校正 p。

**关注点**：relation 是冻结间期轴的预先存在分层，不按当前 concordance 强弱筛人；geometry-unsupported 患者只从这项二维 relation 比较排除，仍在总体 pooled 图中。

### {STEMS['band_exploratory']}.png / .pdf
六个固定窗口分别显示 strict-broadband×1–150 Hz、gamma-nonbroadband×HFA 60–100 Hz，以及以 30–80 Hz 完整替换 gamma readout 的 sensitivity。每轴复用旧 violin、subject point 与黑色 tested-mean 横杠，标注对应 readout 内六窗口 maxT 校正 p 和 n。

**关注点**：三个 readout 仅作探索性跨 subject 分层；30–80 Hz 不增加 gamma seizure 数，也不与 HFA 取最大值。
"""
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / "README.md").write_text(text)


def _write_analysis_readme(analysis: Path) -> None:
    text = """# T_spectral-aligned fixed-window field concordance

本目录保存完整事件漏斗、逐 seizure 轨迹及固定时间窗的 subject-first 数值结果。当前核心表是 `phenotype_matched_fixed_window_subject.csv` 与 `phenotype_matched_fixed_window_cohort.csv`：strict-broadband seizure 只贡献 1–150 Hz，gamma-nonbroadband seizure 只贡献 HFA 60–100 Hz，再在患者内跨其 target seizures 取中位数，cohort 不再按频带拆分。`phenotype_matched_fixed_window_*gamma30_80_sensitivity.csv` 用 30–80 Hz 完整替换 gamma readout，不新增或复制 seizure。

`cohort_subject_funnel.csv` 每名患者一行，集中列出 broadband/gamma seizure 数、二维几何质量、TA/TB relation、shared-field availability 与通过校验的 fingerprint；完整 event 级分类和合同排除仍以 `event_inventory.csv`、`drop_inventory.csv` 为准。

`phenotype_matched_relation_statistics.csv` 只在 `geometry_2d_supported` 层比较预先存在的 reversed 与 different，并单列 same；`exploratory_band_fixed_window_statistics.csv` 保存 broadband、HFA 和 Gamma 30–80 Hz sensitivity 的探索性分频带统计。所有固定窗口均为预先指定的探索性 family，p 值使用保留同一 subject 跨窗口依赖的 maxT 校正。

每行显式记录 `axis_quality_tier`：`strict_2d`、`non_strict_2d` 或 `geometry_unsupported`。注册的全 field-ready 分母不因质量层级而改写；另有 `*_strict_2d_sensitivity.csv` 两张表提供只保留 strict、二维几何充分患者的独立 sensitivity，不能与主表静默混用。field artifact 必须通过版本化 fingerprint 校验才能进入任何一张结果表。

正式 paper-ready 图、图释和对应表副本位于 `results/paper-ready-figure/fig3-sup-tspectral-field-concordance/`。within-shaft 是主 null，all-contact 仅作弱参考；shared field 单独留在数值表中，未与 own field 混选。
"""
    (analysis / "README.md").write_text(text)


def plot_all(analysis_dir: str | Path = DEFAULT_ANALYSIS) -> dict:
    analysis = Path(analysis_dir).resolve()
    subject_table = pd.read_csv(analysis / "fixed_window_field_concordance_subject.csv")
    pooled_subject = pd.read_csv(analysis / "phenotype_matched_fixed_window_subject.csv")
    pooled_cohort = pd.read_csv(analysis / "phenotype_matched_fixed_window_cohort.csv")
    relation_stats = pd.read_csv(analysis / "phenotype_matched_relation_statistics.csv")
    exploratory_stats = pd.read_csv(
        analysis / "exploratory_band_fixed_window_statistics.csv"
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    for obsolete in FIGURES.glob("fig3sup_tspectral_field_concordance_*"):
        if obsolete.is_file():
            obsolete.unlink()
    atlas = _plot_atlases(analysis, subject_table)
    board_n = _plot_board(pooled_subject)
    heatmap_n = _plot_multiband_observed(subject_table)
    _plot_multiband_null(subject_table, exploratory_stats)
    _plot_pooled_by_window(pooled_subject, pooled_cohort)
    _plot_subject_data_vs_null_by_window(pooled_subject, pooled_cohort)
    _plot_relation_by_window(pooled_subject, relation_stats)
    _plot_exploratory_band_by_window(subject_table, exploratory_stats)
    shutil.copy2(analysis / "fixed_window_field_concordance_subject.csv",
                 PAPER_ROOT / "fixed_window_subject_statistics.csv")
    shutil.copy2(analysis / "fixed_window_field_concordance_cohort.csv",
                 PAPER_ROOT / "fixed_window_cohort_statistics.csv")
    shutil.copy2(analysis / "fixed_window_field_concordance_subject_strict_2d_sensitivity.csv",
                 PAPER_ROOT / "fixed_window_subject_statistics_strict_2d_sensitivity.csv")
    shutil.copy2(analysis / "fixed_window_field_concordance_cohort_strict_2d_sensitivity.csv",
                 PAPER_ROOT / "fixed_window_cohort_statistics_strict_2d_sensitivity.csv")
    for filename in (
        "phenotype_matched_fixed_window_subject.csv",
        "phenotype_matched_fixed_window_cohort.csv",
        "phenotype_matched_fixed_window_subject_gamma30_80_sensitivity.csv",
        "phenotype_matched_fixed_window_cohort_gamma30_80_sensitivity.csv",
        "phenotype_matched_fixed_window_subject_strict_2d_sensitivity.csv",
        "phenotype_matched_fixed_window_cohort_strict_2d_sensitivity.csv",
        "phenotype_matched_relation_statistics.csv",
        "exploratory_band_fixed_window_statistics.csv",
        "cohort_subject_funnel.csv",
    ):
        shutil.copy2(analysis / filename, PAPER_ROOT / filename)
    _write_readme()
    _write_analysis_readme(analysis)
    manifest = {
        "analysis": str(analysis), "paper_ready": str(PAPER_ROOT),
        "atlas": atlas, "board_rows": board_n, "heatmap_subjects": heatmap_n,
        "pooled_subjects": int(pooled_subject.subject.nunique()),
        "stems": STEMS, "field_plane": "own narrow only",
        "primary_null": "within_shaft", "reference_null": "all_contact",
    }
    (analysis / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (PAPER_ROOT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_dir", nargs="?", type=Path, default=DEFAULT_ANALYSIS)
    args = parser.parse_args()
    print(json.dumps(plot_all(args.analysis_dir), indent=2))


if __name__ == "__main__":
    main()
