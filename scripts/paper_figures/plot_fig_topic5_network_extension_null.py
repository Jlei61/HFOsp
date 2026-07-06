#!/usr/bin/env python3
"""Paper-ready Topic 5 network-extension and added-advantage statistic.

Panel A tests network extension:
core-derived interictal field prediction on hidden contacts versus a matched
channel-shuffle null.

Panel B tests the stricter added-advantage claim:
core-derived prediction versus the hidden contacts' own interictal order.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_field_extrapolation import (  # noqa: E402
    DEF_BROAD_POOL,
    DEF_NARROW_POOL,
    broad_minus_narrow,
    channel_names_from_pool,
    field_null_p,
    ictal_paired_features,
    load_broad_axis_record,
    maxabscorr_series,
    predicted_interictal_order,
)


IN_DIR = ROOT / "results/topic5_ictal_recruitment/field_extrapolation"
PER_SUBJECT = IN_DIR / "cohort_per_subject"
FINAL_JSON = IN_DIR / "energy_field_extrapolation_FINAL.json"
OUT_DIR = ROOT / "results/paper-ready-figure/fig_topic5_network_extension_null/figures"

BANDS = [("bb_auc", "Broadband energy"), ("hfa_auc", "HFA energy")]
OWN_TIE_DELTA = 0.03
PALETTES = {
    "bb_auc": {
        "field": "#4E79A7",
        "field_edge": "#345F82",
        "field_point": "#4E79A7",
        "null": "#C9D9E6",
        "null_edge": "#8EA9BE",
        "null_point": "#7E96AA",
        "own": "#59A14F",
        "own_edge": "#3E7737",
        "own_point": "#59A14F",
    },
    "hfa_auc": {
        "field": "#C8744A",
        "field_edge": "#9B5635",
        "field_point": "#C8744A",
        "null": "#E9CDBD",
        "null_edge": "#C49D89",
        "null_point": "#A88472",
        "own": "#B07AA1",
        "own_edge": "#815778",
        "own_point": "#B07AA1",
    },
}


def _median(series) -> float:
    values = [float(v) for v in series if np.isfinite(v)]
    return float(np.median(values)) if values else float("nan")


def _p_stars(p_value: float) -> str:
    if p_value < 1e-3:
        return "***"
    if p_value < 1e-2:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def _fmt_p(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "n/a"
    if p_value < 1e-4:
        return f"{p_value:.1e}"
    return f"{p_value:.4f}".rstrip("0").rstrip(".")


def _fmt_q(q_value: float | None) -> str:
    if q_value is None or not np.isfinite(q_value):
        return "q=n/a"
    if q_value < 1e-4:
        return "q<1e-4"
    return f"q={q_value:.3f}".rstrip("0").rstrip(".")


def _paired_p(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 3:
        return float("nan")
    if np.allclose(left, right):
        return 1.0
    try:
        return float(wilcoxon(left, right, alternative="greater").pvalue)
    except ValueError:
        return float("nan")


def _load_final() -> dict[tuple[str, str], dict]:
    rows = json.loads(FINAL_JSON.read_text())
    return {(row["band"], row["hypothesis"]): row for row in rows}


def _load_subject_jsons(band: str) -> dict[str, dict]:
    rows = {}
    for path in sorted(PER_SUBJECT.glob(f"*__{band}.json")):
        row = json.loads(path.read_text())
        if row.get("status") == "ok":
            rows[str(row["subject"])] = row
    return rows


def _compute_subject_null_pair(
    ds_sid: str,
    activation: str,
    *,
    n_null: int,
    sigma_xy: float | None,
    seed: int,
) -> dict:
    """Mirror run_topic5_energy_field_cohort.evaluate_subject for channel null."""
    rec_a = load_broad_axis_record(ds_sid, template="t_a")
    if rec_a is None:
        return {"subject": ds_sid, "band": activation, "status": "no_broad_geometry"}
    rec_b = load_broad_axis_record(ds_sid, template="t_b")
    recs = [rec_a] + ([rec_b] if rec_b is not None else [])

    narrow = set(channel_names_from_pool(ds_sid, DEF_NARROW_POOL))
    hidden = broad_minus_narrow(channel_names_from_pool(ds_sid, DEF_BROAD_POOL), list(narrow))
    cache, paired = ictal_paired_features(ds_sid, "bact", activation)
    if not paired:
        return {"subject": ds_sid, "band": activation, "status": "no_seizure_cache"}
    sz = [target for _bact, target in paired]
    cache_set = set(cache)

    def _valid_names(record: dict) -> set[str]:
        by_name = {c["name"]: c for c in record["channels"]}
        return {
            name
            for name in hidden
            if name in by_name and np.isfinite(by_name[name].get("typical_rank", np.nan))
        }

    names = [
        name
        for name in hidden
        if name in cache_set and all(name in _valid_names(record) for record in recs)
    ]
    pred_core = [
        predicted_interictal_order(
            record,
            names,
            loo=True,
            sigma_xy=sigma_xy,
            core_names=narrow,
        )
        for record in recs
    ]
    names = [name for name in names if all(np.isfinite(pred.get(name, np.nan)) for pred in pred_core)]
    if len(names) < 3:
        return {
            "subject": ds_sid,
            "band": activation,
            "status": "insufficient_hidden",
            "n_hidden": len(names),
        }

    cache_index = {name: i for i, name in enumerate(cache)}
    ci = np.array([cache_index[name] for name in names], dtype=int)
    xcore = [np.array([pred[name] for name in names], dtype=float) for pred in pred_core]
    s_core = maxabscorr_series(xcore, ci, sz)
    field_prediction = _median(s_core)

    channel_labels = [np.zeros(len(names), dtype=int)] * len(sz)
    null = field_null_p(xcore, ci, sz, field_prediction, channel_labels, n=n_null, seed=seed)
    return {
        "subject": ds_sid,
        "band": activation,
        "status": "ok",
        "field_prediction": field_prediction,
        "channel_shuffle_null": float(null["null_median"]),
        "subject_channel_p": float(null["p_value"]),
        "subject_channel_p95": float(null["p95"]),
        "n_hidden": len(names),
        "n_seizures": len([value for value in s_core if np.isfinite(value)]),
    }


def _build_group(
    band: str,
    label: str,
    *,
    n_null: int,
    sigma_xy: float | None,
    seed: int,
    final: dict[tuple[str, str], dict],
) -> dict:
    rows = []
    subject_jsons = _load_subject_jsons(band)
    for subject, stored in subject_jsons.items():
        row = _compute_subject_null_pair(
            subject,
            band,
            n_null=n_null,
            sigma_xy=sigma_xy,
            seed=seed,
        )
        if row.get("status") != "ok":
            continue
        row["hidden_own_order"] = float(stored.get("C1", np.nan))
        row["stored_F_core_only"] = float(stored.get("F_core_only", np.nan))
        row["core_minus_own"] = row["field_prediction"] - row["hidden_own_order"]
        if all(
            np.isfinite(row[key])
            for key in ("field_prediction", "channel_shuffle_null", "hidden_own_order")
        ):
            rows.append(row)

    field = np.array([row["field_prediction"] for row in rows], dtype=float)
    null = np.array([row["channel_shuffle_null"] for row in rows], dtype=float)
    own = np.array([row["hidden_own_order"] for row in rows], dtype=float)
    diff = field - own

    formal = final.get((band, "F_core>channel_null"), {})
    summary = {
        "label": label,
        "band": band,
        "n": len(rows),
        "n_null_per_subject": n_null,
        "wilcoxon_p_core_gt_channel_shuffle_null": _paired_p(field, null),
        "wilcoxon_p_own_gt_channel_shuffle_null": _paired_p(own, null),
        "wilcoxon_p_core_gt_hidden_own_order": _paired_p(field, own),
        "field_prediction_median": float(np.median(field)),
        "field_prediction_iqr": [float(np.percentile(field, 25)), float(np.percentile(field, 75))],
        "channel_shuffle_null_median": float(np.median(null)),
        "channel_shuffle_null_iqr": [float(np.percentile(null, 25)), float(np.percentile(null, 75))],
        "hidden_own_order_median": float(np.median(own)),
        "hidden_own_order_iqr": [float(np.percentile(own, 25)), float(np.percentile(own, 75))],
        "core_minus_own_median": float(np.median(diff)),
        "n_core_gt_null": int(np.sum(field > null)),
        "n_own_gt_null": int(np.sum(own > null)),
        "n_core_gt_own_delta": int(np.sum(diff > OWN_TIE_DELTA)),
        "n_own_gt_core_delta": int(np.sum(diff < -OWN_TIE_DELTA)),
        "n_tie_delta": int(np.sum(np.abs(diff) <= OWN_TIE_DELTA)),
        "own_tie_delta": OWN_TIE_DELTA,
        "formal_subject_p_pass": int(formal.get("n_pass", -1)),
        "formal_subject_p_total": int(formal.get("n_subjects", -1)),
        "formal_binomial_p": float(formal.get("cohort_p", np.nan)),
        "formal_fdr_q": float(formal.get("fdr_q", np.nan)),
    }
    return {"label": label, "band": band, "rows": rows, "summary": summary}


def _add_violin_box_points(
    ax: plt.Axes,
    values: np.ndarray,
    x: float,
    *,
    facecolor: str,
    edgecolor: str,
    point_face: str,
    point_edge: str,
    jitter: np.ndarray,
) -> np.ndarray:
    parts = ax.violinplot(
        [values],
        positions=[x],
        widths=0.58,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    body = parts["bodies"][0]
    body.set_facecolor(facecolor)
    body.set_edgecolor("none")
    body.set_alpha(0.72)

    ax.boxplot(
        [values],
        positions=[x],
        widths=0.34,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.45},
        boxprops={
            "facecolor": facecolor,
            "edgecolor": edgecolor,
            "linewidth": 1.1,
            "alpha": 0.8,
        },
        whiskerprops={"color": edgecolor, "linewidth": 1.0},
        capprops={"color": edgecolor, "linewidth": 1.0},
    )
    point_x = np.full(len(values), x) + jitter
    ax.scatter(
        point_x,
        values,
        s=25,
        facecolors=point_face,
        edgecolors=point_edge,
        linewidths=0.8,
        alpha=0.9,
        zorder=4,
    )
    return point_x


def _add_sig_bracket(ax: plt.Axes, x1: float, x2: float, y: float, text: str) -> None:
    h = 0.035
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", lw=1.25, clip_on=False)
    ax.text(
        (x1 + x2) / 2,
        y + h + 0.008,
        text,
        ha="center",
        va="bottom",
        fontsize=12.5,
        fontweight="bold",
    )


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.085,
        1.03,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12.5,
        fontweight="bold",
    )


def _plot(groups: list[dict], out_paths: list[Path]) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    rng = np.random.default_rng(20260706)
    fig, ax = plt.subplots(figsize=(7.9, 4.55))

    positions = [(1.0, 1.72, 2.44), (4.05, 4.77, 5.49)]
    for group, (x_field, x_own, x_null) in zip(groups, positions):
        palette = PALETTES[group["band"]]
        field = np.array([row["field_prediction"] for row in group["rows"]], dtype=float)
        own = np.array([row["hidden_own_order"] for row in group["rows"]], dtype=float)
        null = np.array([row["channel_shuffle_null"] for row in group["rows"]], dtype=float)
        paired_jitter = rng.normal(0.0, 0.032, size=len(field))

        field_x = _add_violin_box_points(
            ax,
            field,
            x_field,
            facecolor=palette["field"],
            edgecolor=palette["field_edge"],
            point_face=palette["field_point"],
            point_edge="white",
            jitter=paired_jitter,
        )
        own_x = _add_violin_box_points(
            ax,
            own,
            x_own,
            facecolor=palette["own"],
            edgecolor=palette["own_edge"],
            point_face=palette["own_point"],
            point_edge="white",
            jitter=paired_jitter,
        )
        null_x = _add_violin_box_points(
            ax,
            null,
            x_null,
            facecolor=palette["null"],
            edgecolor=palette["null_edge"],
            point_face=palette["null_point"],
            point_edge="white",
            jitter=paired_jitter,
        )
        for xs, ys in zip(zip(field_x, own_x, null_x), zip(field, own, null)):
            ax.plot(xs, ys, color="0.45", linewidth=0.6, alpha=0.22, zorder=3)

        ymax = max(float(np.nanmax(field)), float(np.nanmax(own)), float(np.nanmax(null)))
        summary = group["summary"]
        _add_sig_bracket(
            ax,
            x_field,
            x_own,
            ymax + 0.04,
            _p_stars(summary["wilcoxon_p_core_gt_hidden_own_order"]),
        )
        _add_sig_bracket(
            ax,
            x_own,
            x_null,
            ymax + 0.12,
            _p_stars(summary["wilcoxon_p_own_gt_channel_shuffle_null"]),
        )
        _add_sig_bracket(
            ax,
            x_field,
            x_null,
            ymax + 0.20,
            _p_stars(summary["wilcoxon_p_core_gt_channel_shuffle_null"]),
        )
        ax.text(
            (x_field + x_null) / 2,
            -0.17,
            f"{group['label']}\ncore>own/own>core/tie={summary['n_core_gt_own_delta']}/"
            f"{summary['n_own_gt_core_delta']}/{summary['n_tie_delta']}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8.1,
        )

    ax.set_title("Hidden-contact seizure-energy prediction", loc="left", pad=7, fontsize=11)
    ax.set_ylabel("Prediction strength |r|", fontsize=11)
    ax.set_xticks([x for triplet in positions for x in triplet])
    ax.set_xticklabels(
        ["Core-field\nprediction", "Hidden own\norder", "Channel-shuffle\nnull"] * len(positions),
        fontsize=8.3,
    )
    ax.set_xlim(0.45, 6.05)
    ax.set_ylim(0.0, 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", width=1.0)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)

    fig.subplots_adjust(left=0.12, right=0.985, top=0.9, bottom=0.25)
    for out_path in out_paths:
        fig.savefig(out_path, dpi=300 if out_path.suffix.lower() == ".png" else None)
    plt.close(fig)


def _write_outputs(groups: list[dict]) -> None:
    metadata = {
        "source": {
            "per_subject": str(PER_SUBJECT.relative_to(ROOT)),
            "final_json": str(FINAL_JSON.relative_to(ROOT)),
        },
        "statistics": {
            "core_vs_channel_shuffle_null": "paired F_core_only prediction vs matched channel-shuffle null median",
            "own_vs_channel_shuffle_null": "paired hidden contacts' own interictal order C1 vs matched channel-shuffle null median",
            "core_vs_hidden_own_order": "paired F_core_only prediction vs hidden contacts' own interictal order C1",
            "own_order_tie_delta": OWN_TIE_DELTA,
        },
        "interpretation_boundary": (
            "The three-way plot supports network extension because both core-derived field prediction "
            "and hidden contacts' own interictal order are above channel-shuffle null. It does not "
            "support the stricter added-advantage claim because core-derived prediction does not "
            "systematically exceed hidden contacts' own interictal order."
        ),
        "groups": [
            {
                "label": group["label"],
                "band": group["band"],
                "summary": group["summary"],
                "per_subject": group["rows"],
            }
            for group in groups
        ],
    }
    for filename in (
        "topic5_network_extension_three_way_comparison_summary.json",
        "topic5_network_extension_core_vs_null_and_own_order_summary.json",
        "topic5_network_extension_channel_null_summary.json",
    ):
        (OUT_DIR / filename).write_text(json.dumps(metadata, indent=2) + "\n")

    panel_a_lines = []
    panel_b_lines = []
    for group in groups:
        summary = group["summary"]
        panel_a_lines.append(
            f"{group['label']}：Core-field prediction > channel-shuffle null "
            f"Wilcoxon one-sided p={_fmt_p(summary['wilcoxon_p_core_gt_channel_shuffle_null'])}，"
            f"{summary['n_core_gt_null']}/{summary['n']} subjects above null；formal subject-pass "
            f"{summary['formal_subject_p_pass']}/{summary['formal_subject_p_total']}，"
            f"{_fmt_q(summary['formal_fdr_q'])}"
        )
        panel_a_lines.append(
            f"{group['label']}：Hidden own-order > channel-shuffle null "
            f"Wilcoxon one-sided p={_fmt_p(summary['wilcoxon_p_own_gt_channel_shuffle_null'])}，"
            f"{summary['n_own_gt_null']}/{summary['n']} subjects above null"
        )
        panel_b_lines.append(
            f"{group['label']}：Core>Own/Own>Core/Tie={summary['n_core_gt_own_delta']}/"
            f"{summary['n_own_gt_core_delta']}/{summary['n_tie_delta']} "
            f"(tie=|Δ|≤{OWN_TIE_DELTA})，Core-field > Own-order Wilcoxon one-sided "
            f"p={_fmt_p(summary['wilcoxon_p_core_gt_hidden_own_order'])}"
        )

    readme = f"""# Topic 5 network-extension three-way statistic

### topic5_network_extension_three_way_comparison.png / .pdf

正式三联版。每个频段放在同一组里：`Core-field prediction`、`Hidden own order`、`Channel-shuffle null`。`Core-field prediction` 是 core-only interictal field 对 hidden contacts seizure-energy pattern 的 per-subject median |r|；`Hidden own order` 是 hidden contacts 自身间期顺序 C1 对同一发作能量的预测；`Channel-shuffle null` 是同一 subject、同一 hidden-contact set、同一发作集合下的通道打乱 null median。

三条 bracket 对应三个问题：`Core-field prediction` vs `Channel-shuffle null` = network extension；`Hidden own order` vs `Channel-shuffle null` = hidden 自身间期顺序是否也有预测力；`Core-field prediction` vs `Hidden own order` = 核心外推是否有 added advantage。

**关注点：null 对比**：{"；".join(panel_a_lines)}。Core-field 和 hidden own-order 都显著高于 channel-shuffle null。

**关注点：added advantage**：{"；".join(panel_b_lines)}。Core-field 没有系统性赢过 hidden own-order，但这不是严格等价性检验。

### topic5_network_extension_core_vs_null_and_own_order.png / .pdf

兼容上一版 combined 文件名，内容与正式三联版相同。

### topic5_network_extension_channel_null.png / .pdf

兼容旧文件名，内容与正式三联版相同。

### topic5_network_extension_three_way_comparison_summary.json

Machine-readable core-field/null/own-order medians, paired Wilcoxon statistics, formal subject-pass binomial/FDR results, and per-subject rows.
"""
    (OUT_DIR / "README.md").write_text(readme)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-null", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sigma-xy", type=float, default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    final = _load_final()
    groups = [
        _build_group(
            band,
            label,
            n_null=args.n_null,
            sigma_xy=args.sigma_xy,
            seed=args.seed,
            final=final,
        )
        for band, label in BANDS
    ]

    out_paths = [
        OUT_DIR / "topic5_network_extension_three_way_comparison.png",
        OUT_DIR / "topic5_network_extension_three_way_comparison.pdf",
        OUT_DIR / "topic5_network_extension_core_vs_null_and_own_order.png",
        OUT_DIR / "topic5_network_extension_core_vs_null_and_own_order.pdf",
        OUT_DIR / "topic5_network_extension_channel_null.png",
        OUT_DIR / "topic5_network_extension_channel_null.pdf",
    ]
    _plot(groups, out_paths)
    _write_outputs(groups)
    for out_path in out_paths:
        print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
