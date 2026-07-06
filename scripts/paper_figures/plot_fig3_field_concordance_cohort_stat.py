#!/usr/bin/env python3
"""Paper Fig3 panel: field concordance Data-vs-Null cohort statistic.

The panel is intentionally not a per-subject board. It compresses maxAB-eligible
subjects into three Data-vs-Null comparisons:

1. Legacy broadband 1-45 Hz maxAB.
2. Line-noise-masked broadband 1-150 Hz maxAB.
3. HFA 60-100 Hz maxAB.

Null is the matched channel-shuffle null median of the selected candidate. This
visualizes a cohort-level shift above null; the formal pass gate remains the
selection-corrected p95/p-value table upstream.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[2]
ALIGN_DIR = ROOT / "results/topic5_ictal_recruitment/axis_alignment"
OUT_DIR = (
    ROOT / "results/paper-ready-figure/fig3_field_concordance_cohort_stat/figures"
)

CANDIDATES = {
    "BB 1-45 maxAB": "axis_alignment_broadband_max_ab_B1000.json",
    "BB 1-150 maxAB": "axis_alignment_broadband150_max_ab_B1000.json",
    "HFA 60-100 maxAB": "axis_alignment_hfa_max_ab_B1000.json",
}
COMPARISONS = [
    ("BB 1-45 maxAB", ["BB 1-45 maxAB"]),
    ("BB 1-150 maxAB", ["BB 1-150 maxAB"]),
    ("HFA 60-100 maxAB", ["HFA 60-100 maxAB"]),
]


def _load_candidate_records() -> dict[str, dict[str, dict]]:
    records = {}
    for name, filename in CANDIDATES.items():
        data = json.loads((ALIGN_DIR / filename).read_text())
        records[name] = {
            row["subject_id"]: row
            for row in data.get("per_subject", [])
            if row.get("status") == "ok"
            and row.get("real_median_abs_corr") is not None
            and row.get("channel_null_median") is not None
        }
    return records


def _best_rows(records: dict[str, dict[str, dict]], candidate_names: list[str]) -> list[dict]:
    subjects = sorted(set().union(*[set(records[name]) for name in candidate_names]))
    rows = []
    for subject_id in subjects:
        candidates = []
        for name in candidate_names:
            row = records[name].get(subject_id)
            if row is None:
                continue
            real = float(row["real_median_abs_corr"])
            null = float(row["channel_null_median"])
            candidates.append(
                {
                    "subject_id": subject_id,
                    "candidate": name,
                    "data": real,
                    "null": null,
                    "margin_vs_null_median": real - null,
                    "n_seizures": int(row.get("n_seizures", 0)),
                }
            )
        if candidates:
            rows.append(max(candidates, key=lambda c: c["margin_vs_null_median"]))
    return rows


def _p_stars(p_value: float) -> str:
    if p_value < 1e-3:
        return "***"
    if p_value < 1e-2:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def _fmt_p(p_value: float) -> str:
    if p_value < 1e-4:
        return f"{p_value:.1e}"
    return f"{p_value:.4f}".rstrip("0").rstrip(".")


def _summary(label: str, rows: list[dict]) -> dict:
    data = np.array([r["data"] for r in rows], dtype=float)
    null = np.array([r["null"] for r in rows], dtype=float)
    p_value = float(wilcoxon(data, null, alternative="greater").pvalue)
    return {
        "label": label,
        "n": len(rows),
        "wilcoxon_p_data_gt_null_median": p_value,
        "data_median": float(np.median(data)),
        "data_iqr": [float(np.percentile(data, 25)), float(np.percentile(data, 75))],
        "null_median": float(np.median(null)),
        "null_iqr": [float(np.percentile(null, 25)), float(np.percentile(null, 75))],
        "n_data_gt_null": int(np.sum(data > null)),
        "selected_candidates": {
            name: sum(r["candidate"] == name for r in rows) for name in CANDIDATES
        },
    }


def _add_violin_box_points(
    ax: plt.Axes,
    values: np.ndarray,
    x: float,
    *,
    facecolor: str,
    edgecolor: str,
    rng: np.random.Generator,
    point_face: str,
    point_edge: str,
    jitter: np.ndarray | None = None,
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
        medianprops={"color": "black", "linewidth": 1.5},
        boxprops={"facecolor": facecolor, "edgecolor": edgecolor, "linewidth": 1.1, "alpha": 0.8},
        whiskerprops={"color": edgecolor, "linewidth": 1.0},
        capprops={"color": edgecolor, "linewidth": 1.0},
    )
    if jitter is None:
        jitter = rng.normal(0.0, 0.045, size=len(values))
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
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", lw=1.3, clip_on=False)
    ax.text((x1 + x2) / 2, y + h + 0.008, text, ha="center", va="bottom", fontsize=13, fontweight="bold")


def _plot(groups: list[dict], out_png: Path, out_pdf: Path) -> None:
    rng = np.random.default_rng(20260626)
    fig, ax = plt.subplots(figsize=(7.2, 4.1))

    positions = [(1.0, 1.72), (3.05, 3.77), (5.1, 5.82)]
    for group, (x_data, x_null) in zip(groups, positions):
        label = group["label"]
        data = np.array([r["data"] for r in group["rows"]], dtype=float)
        null = np.array([r["null"] for r in group["rows"]], dtype=float)
        paired_jitter = rng.normal(0.0, 0.035, size=len(data))
        data_x = _add_violin_box_points(
            ax,
            data,
            x_data,
            facecolor="#9fbdcf",
            edgecolor="#6f8fa3",
            rng=rng,
            point_face="#5f86a3",
            point_edge="white",
            jitter=paired_jitter,
        )
        null_x = _add_violin_box_points(
            ax,
            null,
            x_null,
            facecolor="#d8d8d8",
            edgecolor="#9a9a9a",
            rng=rng,
            point_face="#888888",
            point_edge="white",
            jitter=paired_jitter,
        )
        for x0, y0, x1, y1 in zip(data_x, data, null_x, null):
            ax.plot(
                [x0, x1],
                [y0, y1],
                color="0.45",
                linewidth=0.65,
                alpha=0.28,
                zorder=3,
            )
        ymax = max(float(np.nanmax(data)), float(np.nanmax(null)))
        _add_sig_bracket(ax, x_data, x_null, ymax + 0.055, _p_stars(group["summary"]["wilcoxon_p_data_gt_null_median"]))
        ax.text(
            (x_data + x_null) / 2,
            -0.145,
            f"{label}\nn={group['summary']['n']}, p={_fmt_p(group['summary']['wilcoxon_p_data_gt_null_median'])}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8.2,
        )

    ax.set_ylabel("Field concordance |r|", fontsize=11)
    ax.set_xticks([x for pair in positions for x in pair])
    ax.set_xticklabels(["Data", "Null"] * len(positions), fontsize=10)
    ax.set_xlim(0.45, 6.35)
    ax.set_ylim(0.0, 1.12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", width=1.0)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.94, bottom=0.18)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def _write_readme(groups: list[dict], out_dir: Path) -> None:
    lines = []
    for g in groups:
        s = g["summary"]
        lines.append(
            f"{g['label']}：n={s['n']}，Wilcoxon one-sided "
            f"p={_fmt_p(s['wilcoxon_p_data_gt_null_median'])}，"
            f"{s['n_data_gt_null']}/{s['n']} data>null"
        )
    summary_text = "；".join(lines)
    text = f"""# Fig3 field concordance Data-vs-Null statistic

### field_concordance_cohort_stat.png / field_concordance_cohort_stat.pdf

按参考图风格绘制：每一组都是 `Data` vs `Null` 的 violin + box + subject 点，并用浅灰线连接同一 subject 的配对 Data/Null 值，不显示 subject 名字。三组分别是 `BB 1-45 maxAB`、`BB 1-150 maxAB` 和 `HFA 60-100 maxAB`；都使用当前 maxAB artifact 中可评估的 subject，不写 `All candidates`，也不混入 broad fallback。

**关注点**：{summary_text}。这张图展示 cohort-level shift above null；formal pass 仍以 selection-corrected p95/p-value 表为准。`BB 1-150 maxAB` 是新增 sensitivity，原 `bb_auc` 仍是 legacy 1-45 Hz。
"""
    (out_dir / "README.md").write_text(text)


def main() -> None:
    records = _load_candidate_records()
    groups = []
    for label, candidate_names in COMPARISONS:
        rows = _best_rows(records, candidate_names)
        groups.append({"label": label, "rows": rows, "summary": _summary(label, rows)})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "field_concordance_cohort_stat.png"
    out_pdf = OUT_DIR / "field_concordance_cohort_stat.pdf"
    _plot(groups, out_png, out_pdf)

    metadata = {
        "source": str(ALIGN_DIR.relative_to(ROOT)),
        "statistic": "paired Data vs selected candidate channel-null median",
        "selection_rules": {
            label: candidate_names for label, candidate_names in COMPARISONS
        },
        "rows": [
            {
                "label": g["label"],
                "summary": g["summary"],
                "per_subject": g["rows"],
            }
            for g in groups
        ],
        "interpretation_boundary": (
            "Visualizes cohort-level field-concordance shift above channel-null center. "
            "Formal pass claims still require the upstream selection-corrected null."
        ),
    }
    (OUT_DIR / "field_concordance_cohort_stat_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )
    _write_readme(groups, OUT_DIR)
    print(f"[done] wrote {out_png}")
    print(f"[done] wrote {out_pdf}")


if __name__ == "__main__":
    main()
