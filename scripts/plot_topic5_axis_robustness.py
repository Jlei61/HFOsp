#!/usr/bin/env python3
"""Topic 5 -- TA/TB field-reversal §6a axis-robustness supplement: null-comparison plotter.

Reads results/topic5_ictal_recruitment/field_reversal/axis_robustness/{cohort_summary.json,
per_subject/{broad,narrow}/*.json} written by scripts/run_topic5_axis_robustness.py -- this
script renders ONLY, it does not recompute the pipeline's own axis fits or held-out splits.

Three figures:
  1. axis_three_way_comparison.png/.pdf -- per substrate (broad | narrow, NEVER pooled), per
     subject held-out rho for Coordinate axis (raw_contact) | Shaft-order axis (sequence) |
     Random-axis null, paired violin+box+points with gray lines connecting each subject's
     triplet. Mirrors scripts/paper_figures/plot_fig_topic5_network_extension_null.py's idiom
     verbatim (_add_violin_box_points / _add_sig_bracket / triplet x-positions / per-group
     palette / left title / y-lim headroom) -- reused, not reinvented. This figure's own
     Wilcoxon brackets are computed on the subset of ok subjects where ALL THREE quantities
     resolve (needed so every gray line connects a real triplet); cohort_summary.json's own
     `*_beats_*_wilcoxon` entries use the maximal pairwise-available subject set per comparison
     (a subject missing only `sequence` still counts toward the coord-vs-null comparison) and
     remain the pipeline's authoritative numbers -- the two will differ only when a subject's
     sequence_axis is structurally degenerate (single-shaft montage), which the module's own
     tests document as rare.
  2. divergence_distribution.png -- angle(sequence_axis, raw_contact_axis) per substrate,
     45 deg / 90 deg reference lines, n_gt45/frac_gt45 annotated (straight from cohort_summary).
  3. case_axes.png -- the three axis arrows overlaid on the shared statistical (normalized
     readout) frame for the worst-divergence subject per substrate (data-driven: the ok
     subject with the largest subject-level angle(sequence, raw_contact) -- NOT hardcoded).

Honesty red-line (spec §6a, unchanged): "use real coordinates, not electrode/shaft order;
coordinate-blind reading captures less held-out order and diverges badly in a fraction of
subjects (narrow/compact-core worse); smoothing (field) adds nothing beyond coordinates."
Never write "field denoises" or "true axis". broad/narrow never pooled anywhere below.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.topic5_event_resolved_alignment import (
    load_event_labels_ranks, class_aggregate_contact_values, build_plane_xy)

IN_DIR = _ROOT / "results/topic5_ictal_recruitment/field_reversal/axis_robustness"
OUT_DIR = IN_DIR / "figures"
# Geometry lives in the main results tree (gitignored, not the worktree) -- same default the
# runner uses for --input-results-root, and the same paths plot_topic5_field_reversal.py uses.
GEOM = {
    "broad": Path("/home/honglab/leijiaxin/HFOsp/results/spatial_modulation/propagation_geometry_broad"
                   "/observation_readout/real_subjects"),
    "narrow": Path("/home/honglab/leijiaxin/HFOsp/results/spatial_modulation/propagation_geometry"
                    "/observation_readout/real_subjects"),
}
SUBSTRATES = ("broad", "narrow")
TIE_DELTA = 0.03   # |coord - shaft| held-out rho tie band (figure footnote only)

PALETTES = {
    "broad": {"coord": "#4C72B0", "coord_edge": "#33507A", "coord_point": "#4C72B0",
              "shaft": "#9CB2CC", "shaft_edge": "#7186A3", "shaft_point": "#7186A3",
              "null": "#D8D8D8", "null_edge": "#9A9A9A", "null_point": "#888888"},
    "narrow": {"coord": "#DD8452", "coord_edge": "#A05F39", "coord_point": "#DD8452",
               "shaft": "#EFC3A5", "shaft_edge": "#C99872", "shaft_point": "#C1875C",
               "null": "#D8D8D8", "null_edge": "#9A9A9A", "null_point": "#888888"},
}


# --------------------------------------------------------------------------- loading
def _load_per_subject(substrate: str) -> dict:
    d = IN_DIR / "per_subject" / substrate
    return {f.stem: json.loads(f.read_text()) for f in sorted(d.glob("*.json"))}


def _ok_records(records: dict) -> dict:
    return {k: v for k, v in records.items() if v.get("reason") == "ok"}


def _p_stars(p):
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _fmt_p(p):
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}".rstrip("0").rstrip(".")


def _paired_p_greater(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size < 3:
        return float("nan")
    if np.allclose(a, b):
        return 1.0
    try:
        return float(wilcoxon(a, b, alternative="greater").pvalue)
    except ValueError:
        return float("nan")


# --------------------------------------------------------------------------- Fig 1 idiom
# Adapted verbatim from scripts/paper_figures/plot_fig_topic5_network_extension_null.py
# (_add_violin_box_points / _add_sig_bracket) -- reused, not reinvented, per the redirect.
def _add_violin_box_points(ax, values, x, *, facecolor, edgecolor, point_face, point_edge, jitter):
    parts = ax.violinplot([values], positions=[x], widths=0.58, showmeans=False, showmedians=False,
                          showextrema=False)
    body = parts["bodies"][0]
    body.set_facecolor(facecolor)
    body.set_edgecolor("none")
    body.set_alpha(0.72)

    ax.boxplot([values], positions=[x], widths=0.34, patch_artist=True, showfliers=False,
              medianprops={"color": "black", "linewidth": 1.45},
              boxprops={"facecolor": facecolor, "edgecolor": edgecolor, "linewidth": 1.1, "alpha": 0.8},
              whiskerprops={"color": edgecolor, "linewidth": 1.0},
              capprops={"color": edgecolor, "linewidth": 1.0})
    point_x = np.full(len(values), x) + jitter
    ax.scatter(point_x, values, s=25, facecolors=point_face, edgecolors=point_edge, linewidths=0.8,
              alpha=0.9, zorder=4)
    return point_x


def _add_sig_bracket(ax, x1, x2, y, text):
    h = 0.035
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", lw=1.25, clip_on=False)
    ax.text((x1 + x2) / 2, y + h + 0.008, text, ha="center", va="bottom", fontsize=12.5, fontweight="bold")


def _build_group(substrate: str, per_subject_ok: dict) -> dict:
    """Coord/shaft/null triplet restricted to subjects where all three subject_summary means
    resolve (needed so every gray pairing line in Fig 1 connects a real triplet). Computes its
    OWN Wilcoxon + tie counts on exactly this subset (self-contained, mirrors the reference
    script) -- cohort_summary.json's own axis-vs-null / axis-vs-sequence entries remain the
    pipeline's authoritative maximal-pairwise-available numbers (see module docstring)."""
    rows = []
    for ds_sid, rec in per_subject_ok.items():
        s = rec["subject_summary"]
        coord, shaft, null = (s["held_out_raw_mean"], s["held_out_sequence_mean"],
                              s["held_out_null_mean"])
        if np.isfinite(coord) and np.isfinite(shaft) and np.isfinite(null):
            rows.append({"ds_sid": ds_sid, "coord": coord, "shaft": shaft, "null": null})
    rows.sort(key=lambda r: r["ds_sid"])
    coord = np.array([r["coord"] for r in rows], float)
    shaft = np.array([r["shaft"] for r in rows], float)
    null = np.array([r["null"] for r in rows], float)
    diff = coord - shaft
    summary = {
        "substrate": substrate,
        "n": len(rows),
        "n_ok_total": len(per_subject_ok),
        "wilcoxon_p_coord_gt_shaft": _paired_p_greater(coord, shaft),
        "wilcoxon_p_shaft_gt_null": _paired_p_greater(shaft, null),
        "wilcoxon_p_coord_gt_null": _paired_p_greater(coord, null),
        "n_coord_gt_shaft_delta": int(np.sum(diff > TIE_DELTA)),
        "n_shaft_gt_coord_delta": int(np.sum(diff < -TIE_DELTA)),
        "n_tie_delta": int(np.sum(np.abs(diff) <= TIE_DELTA)),
        "tie_delta": TIE_DELTA,
        "coord_median": float(np.median(coord)) if len(rows) else float("nan"),
        "shaft_median": float(np.median(shaft)) if len(rows) else float("nan"),
        "null_median": float(np.median(null)) if len(rows) else float("nan"),
    }
    return {"substrate": substrate, "rows": rows, "coord": coord, "shaft": shaft, "null": null,
           "summary": summary}


def plot_three_way(groups: list, cohort_summary: dict, out_paths: list[Path]) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5, "axes.labelsize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    rng = np.random.default_rng(20260706)
    fig, ax = plt.subplots(figsize=(7.9, 4.85))
    positions = [(1.0, 1.72, 2.44), (4.05, 4.77, 5.49)]

    group_ymax = []
    for group, (x_coord, x_shaft, x_null) in zip(groups, positions):
        substrate = group["substrate"]
        palette = PALETTES[substrate]
        coord, shaft, null = group["coord"], group["shaft"], group["null"]
        jitter = rng.normal(0.0, 0.032, size=len(coord))

        coord_x = _add_violin_box_points(ax, coord, x_coord, facecolor=palette["coord"],
                                         edgecolor=palette["coord_edge"], point_face=palette["coord_point"],
                                         point_edge="white", jitter=jitter)
        shaft_x = _add_violin_box_points(ax, shaft, x_shaft, facecolor=palette["shaft"],
                                         edgecolor=palette["shaft_edge"], point_face=palette["shaft_point"],
                                         point_edge="white", jitter=jitter)
        null_x = _add_violin_box_points(ax, null, x_null, facecolor=palette["null"],
                                        edgecolor=palette["null_edge"], point_face=palette["null_point"],
                                        point_edge="white", jitter=jitter)
        for xs, ys in zip(zip(coord_x, shaft_x, null_x), zip(coord, shaft, null)):
            ax.plot(xs, ys, color="0.45", linewidth=0.6, alpha=0.22, zorder=3)

        ymax = max(float(np.nanmax(coord)), float(np.nanmax(shaft)), float(np.nanmax(null)))
        group_ymax.append(ymax)
        s = group["summary"]
        _add_sig_bracket(ax, x_coord, x_shaft, ymax + 0.04, _p_stars(s["wilcoxon_p_coord_gt_shaft"]))
        _add_sig_bracket(ax, x_shaft, x_null, ymax + 0.12, _p_stars(s["wilcoxon_p_shaft_gt_null"]))
        _add_sig_bracket(ax, x_coord, x_null, ymax + 0.20, _p_stars(s["wilcoxon_p_coord_gt_null"]))

        ax.text((x_coord + x_null) / 2, -0.17,
                f"{substrate} (n={s['n']})\ncoord>shaft/shaft>coord/tie="
                f"{s['n_coord_gt_shaft_delta']}/{s['n_shaft_gt_coord_delta']}/{s['n_tie_delta']}",
                transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8.1)

    ax.axhline(0.0, color="0.85", lw=0.8, zorder=1)
    ax.set_title("Coordinate vs shaft-order propagation-axis readout", loc="left", pad=7, fontsize=11)
    ax.set_ylabel("Held-out prediction strength (ρ)", fontsize=11)
    ax.set_xticks([x for triplet in positions for x in triplet])
    ax.set_xticklabels(["Coordinate\naxis", "Shaft-order\naxis", "Random-axis\nnull"] * len(positions),
                       fontsize=8.3)
    ax.set_xlim(0.45, 6.05)

    flat = np.concatenate([np.concatenate([g["coord"], g["shaft"], g["null"]]) for g in groups])
    y_lo = min(-1.05, float(np.nanmin(flat)) - 0.08)
    y_hi = max(group_ymax) + 0.20 + 0.16   # top bracket sits at ymax+0.20; headroom for its label
    ax.set_ylim(y_lo, y_hi)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", width=1.0)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)

    cos_b = cohort_summary["broad"]["cos_raw_field_median"]
    cos_n = cohort_summary["narrow"]["cos_raw_field_median"]
    fig.text(0.5, 0.015,
             f"field ≈ coordinate axis (cos≈{cos_b:.3f} broad / {cos_n:.3f} narrow) "
             "— smoothing adds nothing beyond coordinates",
             ha="center", fontsize=8.0, color="0.35")

    fig.subplots_adjust(left=0.11, right=0.985, top=0.90, bottom=0.27)
    for out_path in out_paths:
        fig.savefig(out_path, dpi=300 if out_path.suffix.lower() == ".png" else None)
    plt.close(fig)


# --------------------------------------------------------------------------- Fig 2
def plot_divergence_distribution(cohort_summary: dict, out_png: Path) -> None:
    """angle(sequence_axis, raw_contact_axis) per substrate -- how far does the coordinate-blind
    reading diverge from the coordinate-aware one, and in what fraction of subjects."""
    rng = np.random.default_rng(20260706)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    positions = {"broad": 1.0, "narrow": 2.3}
    colors = {"broad": ("#4C72B0", "#33507A", "#4C72B0"),
             "narrow": ("#DD8452", "#A05F39", "#DD8452")}

    for substrate in SUBSTRATES:
        d = cohort_summary[substrate]["divergence_sequence_vs_raw_contact_deg"]
        vals = np.array(list(d["values_by_subject"].values()), float)
        x = positions[substrate]
        face, edge, point = colors[substrate]
        jitter = rng.normal(0.0, 0.05, size=len(vals))
        _add_violin_box_points(ax, vals, x, facecolor=face, edgecolor=edge, point_face=point,
                               point_edge="white", jitter=jitter)
        ax.text(x, -0.06, f"n={d['n']}\n>45°: {d['n_gt45']} ({d['frac_gt45']*100:.0f}%)\n"
                f">90°: {d['n_gt90']} ({d['frac_gt90']*100:.0f}%)",
                transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8.3)

    ax.axhline(45, color="0.35", lw=1.1, ls="--", zorder=1)
    ax.text(2.95, 45, "45°", fontsize=8.2, va="bottom", ha="right", color="0.3")
    ax.axhline(90, color="0.15", lw=1.1, ls="--", zorder=1)
    ax.text(2.95, 90, "90°", fontsize=8.2, va="bottom", ha="right", color="0.2")

    ax.set_xticks([1.0, 2.3])
    ax.set_xticklabels(["broad", "narrow"], fontsize=10.5)
    ax.set_xlim(0.35, 3.05)
    ax.set_ylim(-5, 190)
    ax.set_ylabel("angle(sequence_axis, raw_contact_axis)  [deg]", fontsize=10.5)
    ax.set_title("How far does the coordinate-blind reading diverge from the coordinate-aware one?",
                loc="left", fontsize=10.0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(bottom=0.30, top=0.90, left=0.13, right=0.97)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- Fig 3
def _rank01(vals):
    v = np.asarray(vals, float)
    out = np.full(v.shape, np.nan)
    ok = np.isfinite(v)
    if ok.sum() >= 2:
        out[ok] = np.argsort(np.argsort(v[ok])) / (ok.sum() - 1)
    elif ok.sum() == 1:
        out[ok] = 0.5
    return out


def _case_subject(cohort_summary: dict, substrate: str) -> str:
    """Data-driven: the ok subject with the LARGEST subject-level angle(sequence, raw_contact)
    for this substrate -- re-derived from cohort_summary.json every run (NOT hardcoded), so a
    future re-run with a different cohort still picks the correct worst-divergence case."""
    vals = cohort_summary[substrate]["divergence_sequence_vs_raw_contact_deg"]["values_by_subject"]
    return max(vals, key=vals.get)


def _draw_axis_arrow(ax, cx, cy, unit, length, **kwargs):
    ux, uy = float(unit[0]), float(unit[1])
    if not (np.isfinite(ux) and np.isfinite(uy)):
        return
    ax.annotate("", xy=(cx + ux * length, cy + uy * length), xytext=(cx, cy),
               arrowprops=dict(arrowstyle="-|>", mutation_scale=18, **kwargs), zorder=5)


def plot_case_axes(per_subject_ok: dict, cohort_summary: dict, out_png: Path) -> None:
    """The three axis arrows (raw_contact / field / sequence) on the shared statistical
    (normalized readout, x_norm/y_norm) frame -- NOT the mm display frame used elsewhere -- for
    the worst-divergence subject per substrate. Dots = per-contact class-aggregate value
    (rank01, display only) for whichever of TA/TB has the LARGER divergence for that subject
    (both classes are the same case subject; this just picks the more dramatic of its two
    classes to draw, noted in the panel title)."""
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.6))
    for ax, substrate in zip(axes, SUBSTRATES):
        ds_sid = _case_subject(cohort_summary, substrate)
        record = per_subject_ok[substrate][ds_sid]
        dataset, subject = ds_sid.split("_", 1)

        plane_a = json.loads((GEOM[substrate] / f"{ds_sid}_t_a.json").read_text())
        bundle = load_event_labels_ranks(dataset, subject, broad=(substrate == "broad"))
        plane_xy = build_plane_xy(plane_a)

        cluster_map = {int(k): v for k, v in record["cluster_map"].items()}
        inv = {v: k for k, v in cluster_map.items()}
        ta_ang = record["per_class"]["TA"]["angle_sequence_raw"]
        tb_ang = record["per_class"]["TB"]["angle_sequence_raw"]
        cls_name, cls_label = (("TA", inv["t_a"]) if ta_ang >= tb_ang else ("TB", inv["t_b"]))
        cls_rec = record["per_class"][cls_name]

        cav = class_aggregate_contact_values(bundle, cls_label)
        names = [c["name"] for c in plane_a["channels"] if c["name"] in plane_xy]
        vals = _rank01([cav.get(n, {}).get("value", np.nan) for n in names])
        xs = np.array([plane_xy[n][0] for n in names], float)
        ys = np.array([plane_xy[n][1] for n in names], float)

        sca = ax.scatter(xs, ys, c=vals, cmap="viridis", vmin=0, vmax=1, s=95, edgecolors="0.25",
                         linewidths=0.7, zorder=3)
        cx, cy = float(np.nanmean(xs)), float(np.nanmean(ys))
        x_range = float(xs.max() - xs.min()) if xs.size else 1.0
        y_range = float(ys.max() - ys.min()) if ys.size else 1.0
        length = 0.42 * max(x_range, y_range, 1e-6)

        _draw_axis_arrow(ax, cx, cy, cls_rec["sequence_unit"], length,
                         color="#C1121F", linewidth=2.4, linestyle="-")
        _draw_axis_arrow(ax, cx, cy, cls_rec["field_unit"], length * 0.88,
                         color="#2B8A3E", linewidth=2.0, linestyle="--")
        _draw_axis_arrow(ax, cx, cy, cls_rec["raw_contact_unit"], length,
                         color="#1C3FAA", linewidth=2.4, linestyle="-")

        # pad_x/pad_y must contain a length-`length` arrow tip regardless of DIRECTION -- an
        # arrow can point almost purely along x even when the point cloud (and so `length`,
        # scaled by max(x_range, y_range)) is elongated along y (real case: epilepsiae_1077
        # broad's contacts span y_range=2.7 vs x_range=1.0, but raw_contact/field there point
        # nearly due -x -- a 0.3*length pad clipped the arrow tip clean off, silently dropping
        # the raw_contact arrow from the figure). Since cx/cy (the arrow origin) sits INSIDE
        # [xs.min(),xs.max()]x[ys.min(),ys.max()] by construction (it's the centroid), padding
        # by >= length on every side is sufficient to contain any unit-direction tip.
        pad_x = max(0.25 * max(x_range, 1e-6), length) + 0.08
        pad_y = max(0.25 * max(y_range, 1e-6), length) + 0.08
        ax.set_xlim(xs.min() - pad_x, xs.max() + pad_x)
        ax.set_ylim(ys.min() - pad_y, ys.max() + pad_y)
        ax.set_aspect("equal", adjustable="box")

        pretty = ds_sid.replace("epilepsiae_", "E").replace("yuquan_", "Y-")
        ax.set_title(f"{substrate}: {pretty} ({cls_name}), divergence={cls_rec['angle_sequence_raw']:.0f}°",
                    fontsize=10.8)
        ax.set_xlabel("x_norm (shared statistical frame)")
        ax.set_ylabel("y_norm")
        plt.colorbar(sca, ax=ax, fraction=0.045, pad=0.03,
                    label="early(0)→late(1) [rank01, display only]")

    legend_elems = [
        plt.Line2D([0], [0], color="#1C3FAA", lw=2.4, label="raw_contact (coordinate LS)"),
        plt.Line2D([0], [0], color="#2B8A3E", lw=2.0, ls="--", label="field (smoothed, ≈raw_contact)"),
        plt.Line2D([0], [0], color="#C1121F", lw=2.4, label="sequence (shaft-collapse, coordinate-blind)"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3, fontsize=9, frameon=False,
              bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Worst-divergence case per substrate: coordinate-blind (sequence) vs "
                "coordinate-aware (raw_contact / field) axis reads", fontsize=11.3)
    fig.subplots_adjust(top=0.86, bottom=0.18, wspace=0.32)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


# --------------------------------------------------------------------------- README
def _write_readme(groups: list, cohort_summary: dict, out_dir: Path) -> None:
    g = {grp["substrate"]: grp["summary"] for grp in groups}
    b, n = g["broad"], g["narrow"]

    def _bracket_line(label, s):
        return (f"{label}：coord-vs-shaft p={_fmt_p(s['wilcoxon_p_coord_gt_shaft'])} "
                f"{_p_stars(s['wilcoxon_p_coord_gt_shaft'])}，"
                f"shaft-vs-null p={_fmt_p(s['wilcoxon_p_shaft_gt_null'])} "
                f"{_p_stars(s['wilcoxon_p_shaft_gt_null'])}，"
                f"coord-vs-null p={_fmt_p(s['wilcoxon_p_coord_gt_null'])} "
                f"{_p_stars(s['wilcoxon_p_coord_gt_null'])}（n={s['n']}）")

    div_b = cohort_summary["broad"]["divergence_sequence_vs_raw_contact_deg"]
    div_n = cohort_summary["narrow"]["divergence_sequence_vs_raw_contact_deg"]
    case_b = _case_subject(cohort_summary, "broad")
    case_n = _case_subject(cohort_summary, "narrow")

    text = f"""# TA/TB 传播轴鲁棒性补充 — vs 随机方向零假设 图说明

**测了什么**：上一版 axis-robustness 补充只比较了三种"读传播方向"的办法——用真实坐标做直线拟合（`raw_contact`）、先把值摊成空间场再拟合（`field`）、只看触点在哪根电极杆上（`sequence`，完全不看杆内位置，模拟"按电极顺序读"这种最朴素的做法）。这次补的是一个此前缺的零假设：每一种读法算出来的方向，到底比"随便瞎猜一个方向"强多少？如果一个读法给出的方向其实跟乱猜差不多，那这个读法就没有真的抓住传播顺序。

**怎么测的**：对每个方向（不管是真读出的还是随便猜的），都用同一把尺子打分——把每个触点的真实坐标投影到这个方向上，看投影值和"留出来没用于拟合的另一半事件"的触点顺序有多少 Spearman 相关（留一半训练、留一半打分，重复几十次取中位数）。"随机方向零假设"就是对同一个病人、同一次留出打分，换成 200 个随机角度各打一次分，取中位数——这就是"这个病人的电极摆放本身、什么都不做也能蒙对多少"的本底水平（不是理论上假设的 0，因为触点摆放本身不对称时蒙对的本底也会偏离 0）。然后看：真实拟合出的方向（`raw_contact`/`field`/`sequence`）比这个本底高多少，配对 Wilcoxon 单尾检验。

**揭示了什么**：`raw_contact`（用真坐标）和 `field`（先摊场再拟合）都远超随机方向零假设，两个触点覆盖范围（`broad` = 覆盖更广、`narrow` = 更贴近临床 SOZ 核心）里都是这样；`sequence`（只看电极杆、不看杆内位置）也超过零假设，但超得**没那么多**——换句话说，纯按电极顺序读多少还是比瞎猜强，但明显不如用真坐标。`sequence` 相对 `raw_contact` 的劣势在 `narrow`（触点更少、更贴近核心）比在 `broad` 更明显——杆越少、越贴近核心，只看杆序丢掉的信息比例越大。三个 bracket 的配对检验（{_bracket_line("broad", b)}；{_bracket_line("narrow", n)}）都指向同一件事：**要读传播方向，必须用真实坐标，不能只看电极/杆的顺序**；而"先摊成场再拟合"（`field`）没有比"直接对坐标做拟合"（`raw_contact`）多带来任何东西（两者拟合出的方向几乎重合，夹角余弦中位数 broad≈{cohort_summary['broad']['cos_raw_field_median']:.3f}、narrow≈{cohort_summary['narrow']['cos_raw_field_median']:.3f}，1.0 = 完全同向）——**这不是"场去噪"，只是"场没有比坐标本身多给什么"**。下文按图说明，`broad`/`narrow` 全程分开报告、从不合并。

---

### axis_three_way_comparison.png / .pdf — 头条：三种读法 vs 随机方向零假设

`broad` | `narrow` 两组并排，每组内三列 `Coordinate axis`（`raw_contact`）| `Shaft-order axis`（`sequence`）| `Random-axis null`，violin+箱线+散点，浅灰线连接同一个病人在三列上的取值（同一病人一条折线）。三条 bracket 对应三个问题（从下到上）：`Coordinate` vs `Shaft-order`（真坐标是不是真的比只看杆序强）、`Shaft-order` vs `Null`（只看杆序好歹是不是也比瞎猜强）、`Coordinate` vs `Null`（真坐标是不是明显比瞎猜强，预期最强）。纵轴 = 留出数据上的 Spearman 预测强度（`ρ`），0 线标出"完全没有预测力"的参照。底部脚注给出这次配对比较里"真坐标赢/杆序赢/打平（差距≤0.03）"的病人数。图内数字用的是本图自己那个"三个量都算得出来"的病人子集（配对折线需要三个值同时存在）；`cohort_summary.json` 里对应的 `*_beats_*_wilcoxon` 字段是流水线的正式数字，用的是每次两两比较各自能用的最大病人数（只有当某个病人的 `sequence` 因为触点全部在同一根电极杆上而算不出来时，两边数字才会略有差异）。

**关注点**：`Coordinate` 和 `Shaft-order` 两列都清楚地站在 `Random-axis null` 之上（最右侧 bracket 星号最多）——说明不管用什么方法读，多少都比瞎猜强；但 `Coordinate` vs `Shaft-order` 那条 bracket 显示真坐标显著强于只看杆序，尤其在 `narrow` 更明显。底部小字说明 `field` 和 `raw_contact` 幅度几乎相同（cos≈{cohort_summary['broad']['cos_raw_field_median']:.3f}/{cohort_summary['narrow']['cos_raw_field_median']:.3f}）——这不是图里单独画出来的第四列，是因为再画一列几乎会和 `Coordinate` 重叠，属于冗余面板，所以只用一句话带过。

### divergence_distribution.png — 杆序读法偏离真坐标读法有多远

`broad`/`narrow` 并排的 violin+箱线+散点，纵轴 = `sequence_axis` 与 `raw_contact_axis` 之间的夹角（度），横虚线标 45°/90° 两条参照线。脚注给出每个底物里夹角 >45°、>90° 的病人数和占比：`broad` n={div_b['n']}，>45° {div_b['n_gt45']}/{div_b['n']}（{div_b['frac_gt45']*100:.0f}%），>90° {div_b['n_gt90']}/{div_b['n']}（{div_b['frac_gt90']*100:.0f}%）；`narrow` n={div_n['n']}，>45° {div_n['n_gt45']}/{div_n['n']}（{div_n['frac_gt45']*100:.0f}%），>90° {div_n['n_gt90']}/{div_n['n']}（{div_n['frac_gt90']*100:.0f}%）。

**关注点**：`narrow`（贴近核心、触点更少）里夹角 >45° 的比例明显比 `broad` 高——触点越少、越贴近核心，只看杆序这种朴素读法误导得越厉害，这和上面 `axis_three_way_comparison.png` 里 `narrow` 的 `Coordinate` vs `Shaft-order` bracket 更显著是同一件事的两个角度（分布形状 + 假设检验），不是重复信息。

### case_axes.png — 两个底物各自偏离最狠的病人，三条箭头摆在一起看

左：`broad` 偏离最大的病人（数据驱动选出，当前是 `{case_b.replace('epilepsiae_', 'E').replace('yuquan_', 'Y-')}`，夹角≈{div_b['values_by_subject'].get(case_b, float('nan')):.0f}°）；右：`narrow` 偏离最大的病人（当前是 `{case_n.replace('epilepsiae_', 'E').replace('yuquan_', 'Y-')}`，夹角≈{div_n['values_by_subject'].get(case_n, float('nan')):.0f}°）。每个病人只画 TA/TB 两类里偏离更夸张的那一类（标题标注 TA 还是 TB），触点按早（0，深色）到晚（1，亮色）上色（仅用于展示，不是统计输入），三条箭头从触点重心出发：蓝色实线 = `raw_contact`（真坐标直线拟合）、绿色虚线 = `field`（先摊场再拟合）、红色实线 = `sequence`（只看电极杆、不看杆内位置）。坐标轴是这条流水线做统计用的共享 normalized 坐标系，不是另一套图用的 mm 展示坐标系。

**关注点**：蓝色和绿色箭头（`raw_contact`/`field`）几乎重合在一起，红色箭头（`sequence`）明显指向不同方向——这就是"只看杆序会把方向读偏"这件事在两个具体病人身上长什么样子。这两个病人是当前队列里偏离最狠的两个（脚本每次都会重新从 `cohort_summary.json` 里挑，不是写死的名字），如果以后重跑队列换了别的病人，这张图会跟着换。
"""
    (out_dir / "README.md").write_text(text)


# --------------------------------------------------------------------------- main
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cohort_summary = json.loads((IN_DIR / "cohort_summary.json").read_text())
    per_subject = {s: _load_per_subject(s) for s in SUBSTRATES}
    ok = {s: _ok_records(per_subject[s]) for s in SUBSTRATES}

    print("[1/3] axis_three_way_comparison ...")
    groups = [_build_group(s, ok[s]) for s in SUBSTRATES]
    out_paths = [OUT_DIR / "axis_three_way_comparison.png", OUT_DIR / "axis_three_way_comparison.pdf"]
    plot_three_way(groups, cohort_summary, out_paths)
    for p in out_paths:
        print(f"[fig] {p}")

    print("[2/3] divergence_distribution ...")
    plot_divergence_distribution(cohort_summary, OUT_DIR / "divergence_distribution.png")
    print(f"[fig] {OUT_DIR / 'divergence_distribution.png'}")

    print("[3/3] case_axes ...")
    plot_case_axes(ok, cohort_summary, OUT_DIR / "case_axes.png")
    print(f"[fig] {OUT_DIR / 'case_axes.png'}")

    metadata = {
        "source": str(IN_DIR.relative_to(_ROOT)),
        "axis_three_way_comparison": {
            "statistic": (
                "self-contained paired Wilcoxon (alternative='greater') on the subset of ok "
                "subjects where raw_contact/sequence/random-axis-null held-out rho ALL resolve "
                "(needed for the gray pairing lines to connect a real triplet); "
                "cohort_summary.json's own axis-vs-null and axis-vs-sequence Wilcoxon entries "
                "use the maximal pairwise-available subject set per comparison and are the "
                "pipeline's authoritative numbers."
            ),
            "groups": [g["summary"] for g in groups],
        },
        "case_axes_subjects": {"broad": _case_subject(cohort_summary, "broad"),
                              "narrow": _case_subject(cohort_summary, "narrow")},
        "interpretation_boundary": (
            "use real coordinates, not electrode/shaft order: coordinate-blind (sequence) "
            "captures less held-out propagation order than coordinate-aware (raw_contact) and "
            "diverges badly (>45 deg) in a fraction of subjects, worse in narrow (compact-core) "
            "than broad; field (smoothed) adds nothing beyond raw_contact's plain coordinates. "
            "None of this is 'field denoises' or a claim of a 'true axis'. broad/narrow never "
            "pooled."
        ),
    }
    (OUT_DIR / "axis_robustness_supplement_metadata.json").write_text(json.dumps(metadata, indent=2))
    _write_readme(groups, cohort_summary, OUT_DIR)
    print(f"[done] figures + metadata + README under {OUT_DIR}")


if __name__ == "__main__":
    main()
