#!/usr/bin/env python3
"""Topic 5 -- TA/TB interictal field-reversal gate: plotter + figures README data.

Spec: docs/superpowers/specs/2026-07-06-topic5-tatb-field-reversal-design.md
Plan: docs/superpowers/plans/2026-07-06-topic5-tatb-field-reversal.md (Task 9)
Data: scripts/run_topic5_field_reversal.py (Task 8) COHORT output -- this script renders
ONLY, it does not run within_shaft_reversal_gate / channel_floor / loo_reproducibility etc.
again; every number here is read straight from the already-written per-subject JSONs and
cohort_summary.json under results/topic5_ictal_recruitment/field_reversal/.

Panel-1 (per-subject TA/TB field maps) is the one exception that touches raw geometry +
event data again, because the per-subject JSON does not persist the full 2D field -- it
mirrors scripts/plot_topic5_event_resolved_fields.py: shared-frame DISPLAY via
_subject_display_frame/_smooth_rank_field_mm (VIS_* constants, rank01 -- display-only).
The statistical gate shown alongside it (right-hand panel) is read verbatim from the
per-subject JSON's `gate`/`channel_floor`/`random_split` -- those never touch VIS constants.

1146 mechanism panel (spec Sec9 / brief panel 7): DEFERRED, not rendered. epilepsiae_1146's
broad t_a record does carry poor_planarity=True (consistent with the spec's own caveat), but
the "raw contact values + fitted direction" LEFT panel the brief calls for needs a bespoke
naive electrode-order axis fit that is not computed anywhere else in this pipeline (the
existing along_axis_mm/x_norm fields are themselves already geometry+smoothing aware -- they
ARE the right-hand-panel readout, not a naive contrast to it). Inventing that statistic here
would not be "quickly verifiable" (brief's own bar for keeping this panel); switching subject
does not remove that need. Skipped per brief Sec9's explicit permission; noted in the README.
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

from scripts.plot_contact_plane_static import (
    _subject_display_frame, _display_points, _smooth_rank_field_mm, _attach_real_coords)
from src.topic5_event_resolved_alignment import load_event_labels_ranks, class_aggregate_contact_values

IN_DIR = _ROOT / "results/topic5_ictal_recruitment/field_reversal"
OUT_DIR = IN_DIR / "figures"
# Geometry lives in the main results tree (gitignored, not the worktree) -- same default the
# Task 8 runner uses for --input-results-root.
GEOM = {
    "broad": Path("/home/honglab/leijiaxin/HFOsp/results/spatial_modulation/propagation_geometry_broad"
                   "/observation_readout/real_subjects"),
    "narrow": Path("/home/honglab/leijiaxin/HFOsp/results/spatial_modulation/propagation_geometry"
                    "/observation_readout/real_subjects"),
}
SUBSTRATES = ("broad", "narrow")
REASON_ORDER = ("no_planes", "load_error", "c1_violation", "plane_not_built",
                "cluster_map_ambiguous", "insufficient_overlap", "degenerate_null", "ok")
COLOR_BROAD, COLOR_NARROW = "#4C72B0", "#DD8452"
# Two informative examples per substrate (one clean pass, one where the raw signed_corr looks
# strongly negative but fails its OWN within-shaft null -- the pedagogically important case that
# shows why the null matters, not just the raw number).
EXAMPLE_SUBJECTS = {
    "broad": ["epilepsiae_1077", "epilepsiae_1125"],
    "narrow": ["epilepsiae_1096", "yuquan_zhaochenxi"],
}


# --------------------------------------------------------------------------- loading
def _load_per_subject(substrate: str) -> dict:
    d = IN_DIR / "per_subject" / substrate
    return {f.stem: json.loads(f.read_text()) for f in sorted(d.glob("*.json"))}


def _ok_records(records: dict) -> dict:
    return {k: v for k, v in records.items() if v.get("reason") == "ok"}


# --------------------------------------------------------------------------- fig3 idiom
# Adapted from scripts/paper_figures/plot_fig3_field_concordance_cohort_stat.py. Kept the same
# violin+box+jittered-points / bracket / stars / p-format construction; the only functional
# addition is an optional shared `jitter` (so Data and Null columns can be connected pointwise
# by a light-gray line per subject -- this figure's pairing is the whole point, unlike fig3's
# group-level shift).
def _p_stars(p):
    if p is None or not np.isfinite(p):
        return "NA"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _fmt_p(p):
    if p is None or not np.isfinite(p):
        return "NA"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}".rstrip("0").rstrip(".")


def _safe_wilcoxon(a, b, alternative):
    try:
        stat, p = wilcoxon(a, b, alternative=alternative)
        return float(stat), float(p)
    except ValueError:
        return None, None


def _add_violin_box_points(ax, values, x, *, facecolor, edgecolor, rng, point_face, point_edge,
                            jitter=None):
    parts = ax.violinplot([values], positions=[x], widths=0.58, showmeans=False,
                           showmedians=False, showextrema=False)
    body = parts["bodies"][0]
    body.set_facecolor(facecolor)
    body.set_edgecolor("none")
    body.set_alpha(0.72)

    ax.boxplot([values], positions=[x], widths=0.34, patch_artist=True, showfliers=False,
               medianprops={"color": "black", "linewidth": 1.5},
               boxprops={"facecolor": facecolor, "edgecolor": edgecolor, "linewidth": 1.1, "alpha": 0.8},
               whiskerprops={"color": edgecolor, "linewidth": 1.0},
               capprops={"color": edgecolor, "linewidth": 1.0})
    if jitter is None:
        jitter = rng.normal(0.0, 0.045, size=len(values))
    ax.scatter(np.full(len(values), x) + jitter, values, s=25, facecolors=point_face,
               edgecolors=point_edge, linewidths=0.8, alpha=0.9, zorder=3)
    return jitter


def _add_sig_bracket(ax, x1, x2, y, text, h=0.06):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", lw=1.3, clip_on=False)
    ax.text((x1 + x2) / 2, y + h + 0.015, text, ha="center", va="bottom", fontsize=13, fontweight="bold")


# --------------------------------------------------------------------------- Fig A (headline)
def plot_cohort_stat(per_subject_by_substrate: dict, cohort_summary: dict, out_dir: Path):
    """Brief panel 2 -- field_reversal_cohort_stat.png. Mirrors fig3's Data-vs-Null idiom with
    the three mandatory departures: signed r (not |r|), less-tail Wilcoxon, grouped by substrate.
    """
    rng = np.random.default_rng(20260706)
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    positions = {"broad": (1.0, 1.75), "narrow": (3.05, 3.8)}
    groups_meta = {}

    for substrate in SUBSTRATES:
        ok = _ok_records(per_subject_by_substrate[substrate])
        ds_sids = sorted(ok)
        data = np.array([ok[s]["gate"]["signed_corr"] for s in ds_sids], float)
        null = np.array([ok[s]["gate"]["null_p50"] for s in ds_sids], float)
        stat, p = _safe_wilcoxon(data, null, alternative="less")
        n_lt = int(np.sum(data < null))
        acc = cohort_summary[substrate]["accountability"]
        excluded = {r: c for r, c in acc.items() if r != "ok" and c > 0}
        binom = cohort_summary[substrate]["binomial"]

        # Sanity: this script's own "ok" count and binomial n/k must reconcile with the
        # cohort_summary Task 8 already wrote -- a mismatch means this script mis-derived
        # something and must not be shipped silently.
        assert len(ds_sids) == cohort_summary[substrate]["n_ok"], (
            f"{substrate}: plotted n_ok={len(ds_sids)} != cohort_summary n_ok="
            f"{cohort_summary[substrate]['n_ok']}")
        assert binom["n"] == len(ds_sids) and binom["k"] == int(np.sum(
            [ok[s]["gate"]["passed"] for s in ds_sids])), (
            f"{substrate}: binomial k/n does not reconcile with per-subject gate.passed")

        x_data, x_null = positions[substrate]
        jitter = rng.normal(0.0, 0.045, size=len(data))
        for i in range(len(data)):
            ax.plot([x_data + jitter[i], x_null + jitter[i]], [data[i], null[i]],
                     color="0.78", lw=0.8, zorder=2, alpha=0.85)
        _add_violin_box_points(ax, data, x_data, facecolor="#9fbdcf", edgecolor="#6f8fa3",
                               rng=rng, point_face="#5f86a3", point_edge="white", jitter=jitter)
        _add_violin_box_points(ax, null, x_null, facecolor="#d8d8d8", edgecolor="#9a9a9a",
                               rng=rng, point_face="#888888", point_edge="white", jitter=jitter)

        ymax = max(float(np.nanmax(data)), float(np.nanmax(null)))
        _add_sig_bracket(ax, x_data, x_null, ymax + 0.11, _p_stars(p), h=0.07)
        excl_str = ", ".join(f"{k}={v}" for k, v in excluded.items()) or "none"
        footnote = (f"{substrate}\nn={len(data)} (excluded: {excl_str})\n"
                    f"Wilcoxon (less) p={_fmt_p(p)} {_p_stars(p)}, n(data<null)={n_lt}/{len(data)}\n"
                    f"cohort_binomial pass {binom['k']}/{binom['n']} (p={_fmt_p(binom['p_binom'])})")
        ax.text((x_data + x_null) / 2, -0.32, footnote, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7.6)

        groups_meta[substrate] = {
            "n": len(data), "wilcoxon_statistic": stat, "wilcoxon_p_data_lt_null_alt_less": p,
            "n_data_lt_null": n_lt, "data_median": float(np.median(data)),
            "null_median": float(np.median(null)), "cohort_binomial": binom,
            "excluded_accountability": excluded,
            "per_subject": [
                {"ds_sid": s, "data_signed_corr": float(ok[s]["gate"]["signed_corr"]),
                 "null_p50": float(ok[s]["gate"]["null_p50"]),
                 "passed": bool(ok[s]["gate"]["passed"])}
                for s in ds_sids],
        }

    ax.axhline(0.0, color="0.3", lw=1.0, ls="--", zorder=1)
    ax.text(4.32, 0.03, "r = 0", fontsize=8, color="0.3", va="bottom", ha="right")
    ax.set_ylabel("TA–TB field reversal (signed r)", fontsize=11)
    ax.set_xticks([1.0, 1.75, 3.05, 3.8])
    ax.set_xticklabels(["Data", "Null", "Data", "Null"], fontsize=10)
    ax.set_xlim(0.45, 4.35)
    ax.set_ylim(-1.15, 1.35)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("TA–TB field reversal: cohort Data vs within-shaft Null (broad | narrow, never pooled)\n"
                 "reversal = signed r below 0 AND below its own null", fontsize=10.3, y=0.998)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.34)

    out_png = out_dir / "field_reversal_cohort_stat.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "source": "results/topic5_ictal_recruitment/field_reversal",
        "statistic": "paired Wilcoxon, per-subject gate.signed_corr vs gate.null_p50, alternative='less'",
        "groups": groups_meta,
        "interpretation_boundary": (
            "visualizes cohort-level reversal shift below the within-shaft null; formal pass "
            "still requires the per-subject binomial gate."),
    }
    (out_dir / "field_reversal_cohort_stat_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[fig] {out_png}")
    return groups_meta


# --------------------------------------------------------------------------- diagnostic supplement
def plot_null_forest(per_subject: dict, substrate: str, out_png: Path):
    """Brief panel 2 diagnostic supplement (optional, not a paper panel): observed vs own null
    p05, sorted, black=passed."""
    ok = _ok_records(per_subject)
    rows = sorted(ok.values(), key=lambda r: r["gate"]["signed_corr"])
    if not rows:
        print(f"[skip] {substrate} null_forest: no ok subjects"); return
    obs = [r["gate"]["signed_corr"] for r in rows]
    p05 = [r["gate"]["null_p05"] for r in rows]
    passed = [bool(r["gate"]["passed"]) for r in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(max(8.0, 0.32 * len(rows)), 4.4))
    for i, (o, p) in enumerate(zip(obs, p05)):
        ax.plot([i, i], [o, p], color="0.75", lw=1.0, zorder=1)
    ax.scatter(x, p05, marker="_", s=160, color="0.35", linewidths=1.8, zorder=2, label="own within-shaft null p05")
    face = ["black" if pk else "white" for pk in passed]
    ax.scatter(x, obs, s=40, facecolors=face, edgecolors="black", linewidths=1.0, zorder=3,
               label="observed signed r (black = passed)")
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.set_xticks([])
    ax.set_xlabel(f"{len(rows)} ok subjects, sorted by observed signed r")
    ax.set_ylabel("signed r (TA_field, TB_field)")
    ax.set_title(f"{substrate}: observed reversal vs own within-shaft null (diagnostic, not a paper panel)",
                 fontsize=10.5)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[fig] {out_png}")


# --------------------------------------------------------------------------- Fig B (accountability)
def plot_accountability(cohort_summary: dict, out_png: Path):
    """Brief panel 5 -- who entered narrow vs broad and why. ONE figure, both substrates."""
    reasons = list(REASON_ORDER)
    b = [cohort_summary["broad"]["accountability"].get(r, 0) for r in reasons]
    n_ = [cohort_summary["narrow"]["accountability"].get(r, 0) for r in reasons]

    x = np.arange(len(reasons))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    bars_b = ax.bar(x - w / 2, b, width=w, color=COLOR_BROAD, label=f"broad (n={sum(b)})")
    bars_n = ax.bar(x + w / 2, n_, width=w, color=COLOR_NARROW, label=f"narrow (n={sum(n_)})")
    for bars in (bars_b, bars_n):
        for rect in bars:
            h = rect.get_height()
            if h > 0:
                ax.text(rect.get_x() + rect.get_width() / 2, h + 0.35, str(int(h)),
                        ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(reasons, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("n subjects")
    ax.set_title("who entered the TA/TB field-reversal gate, and why (per substrate)", fontsize=11.5)
    ax.legend(fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[fig] {out_png}")


# --------------------------------------------------------------------------- Fig C (head-to-head)
def plot_field_vs_contact_headtohead(ok: dict, substrate: str, out_png: Path):
    """Brief panel 3 -- field passes vs contact passes (paired binary concordance)."""
    n = len(ok)
    field_pass = np.array([r["gate"]["passed"] for r in ok.values()], bool)
    contact_pass = np.array([r["contact_gate"]["passed"] for r in ok.values()], bool)
    grid = np.array([[int(np.sum(field_pass & contact_pass)), int(np.sum(field_pass & ~contact_pass))],
                     [int(np.sum(~field_pass & contact_pass)), int(np.sum(~field_pass & ~contact_pass))]])

    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    ax.imshow(grid, cmap="Blues", vmin=0, vmax=max(n, 1))
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(grid[i, j]), ha="center", va="center", fontsize=17, fontweight="bold",
                    color="white" if grid[i, j] > n / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["contact pass", "contact fail"], fontsize=9.5)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["field pass", "field fail"], fontsize=9.5)
    ax.set_title(f"{substrate}: field vs contact pass concordance (n={n})\n"
                 "“does the field buy robustness?” (binary pass/fail layer)", fontsize=10)
    fig.text(0.5, 0.015, f"field {int(field_pass.sum())}/{n} pass; contact {int(contact_pass.sum())}/{n} "
             "pass — each vs its own within-shaft null", ha="center", fontsize=8.3)
    fig.subplots_adjust(bottom=0.16, top=0.82)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[fig] {out_png}")


# --------------------------------------------------------------------------- Fig D (LOO supplement)
def plot_loo_reproducibility(ok: dict, wilcoxon_stats: dict, substrate: str, out_png: Path):
    """Brief panel 4 -- the denoising supplement. field_rho vs contact_rho paired, per subject.
    MUST show contact honestly beating field where that is what the data says (spec Sec6/Sec2:
    "field more robust?" is a tested sub-question, not an assumption)."""
    rng = np.random.default_rng(20260706)
    ds_sids = sorted(ok)
    field = np.array([ok[s]["loo"]["field_rho"] for s in ds_sids], float)
    contact = np.array([ok[s]["loo"]["contact_rho"] for s in ds_sids], float)
    n = len(field)
    n_contact_beats_field = int(np.sum(contact > field))
    p = wilcoxon_stats.get("p_value")

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    x_field, x_contact = 1.0, 2.0
    jitter = rng.normal(0.0, 0.045, size=n)
    for i in range(n):
        ax.plot([x_field + jitter[i], x_contact + jitter[i]], [field[i], contact[i]],
                 color="0.78", lw=0.8, zorder=2, alpha=0.85)
    ax.scatter(x_field + jitter, field, s=32, facecolors="#5f86a3", edgecolors="white",
               linewidths=0.8, zorder=3, label="field (LOO-smoothed)")
    ax.scatter(x_contact + jitter, contact, s=32, facecolors="#c1121f", edgecolors="white",
               linewidths=0.8, zorder=3, label="contact (raw train-half mean)")
    ax.set_xticks([x_field, x_contact])
    ax.set_xticklabels(["Field", "Contact"], fontsize=10.5)
    ax.set_xlim(0.55, 2.45)
    ax.set_ylabel("held-out prediction accuracy (Spearman ρ)")
    ax.set_title(f"{substrate}: does the field denoise? (n={n})", fontsize=11)
    ax.legend(fontsize=8, loc="lower center")
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.015, f"contact beats field in {n_contact_beats_field}/{n} subjects; "
             f"paired Wilcoxon p={_fmt_p(p)} {_p_stars(p)}", ha="center", fontsize=8.6)
    fig.subplots_adjust(bottom=0.20, top=0.90)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[fig] {out_png}  (n_contact_beats_field={n_contact_beats_field}/{n})")
    return n_contact_beats_field, n


# --------------------------------------------------------------------------- Fig E (sensitivity)
def plot_broad_vs_narrow(cohort_summary: dict, out_png: Path):
    """Brief panel 6 -- for subjects ok in both, signed_corr broad vs narrow + 2x2 inset."""
    sens = cohort_summary["sensitivity_broad_vs_narrow"]
    rows = sens["per_subject"]
    b = np.array([r["signed_corr_broad"] for r in rows], float)
    n_ = np.array([r["signed_corr_narrow"] for r in rows], float)
    pb = np.array([r["passed_broad"] for r in rows], bool)
    pn = np.array([r["passed_narrow"] for r in rows], bool)
    cats = np.where(pb & pn, "both_pass", np.where(pb & ~pn, "broad_only",
                    np.where(~pb & pn, "narrow_only", "neither")))
    colors = {"both_pass": "#2b8a3e", "broad_only": COLOR_BROAD, "narrow_only": COLOR_NARROW,
              "neither": "0.7"}

    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    for cat, c in colors.items():
        m = cats == cat
        if m.any():
            ax.scatter(b[m], n_[m], s=60, facecolors=c, edgecolors="0.25", linewidths=0.6,
                       label=f"{cat} (n={int(m.sum())})", zorder=3)
    lim = 1.05
    ax.plot([-lim, lim], [-lim, lim], color="0.6", lw=1.0, ls="--", zorder=1, label="broad = narrow")
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("broad: signed r(TA_field, TB_field)")
    ax.set_ylabel("narrow: signed r(TA_field, TB_field)")
    ax.set_title(f"broad vs narrow reversal, subjects ok in BOTH (n={len(rows)})\n"
                 "does reversal survive at the compact core, or is it coarse/distal?", fontsize=10.5)
    ax.legend(fontsize=7.3, loc="lower right")

    cc = sens["pass_concordance_2x2"]
    grid = np.array([[cc["both_pass"], cc["broad_only"]], [cc["narrow_only"], cc["neither"]]])
    ins = ax.inset_axes([0.04, 0.70, 0.30, 0.28])
    ins.imshow(grid, cmap="Greys", vmin=0, vmax=max(len(rows), 1))
    for i in range(2):
        for j in range(2):
            ins.text(j, i, str(grid[i, j]), ha="center", va="center", fontsize=10.5, fontweight="bold",
                     color="white" if grid[i, j] > len(rows) / 2 else "black")
    ins.set_xticks([0, 1]); ins.set_xticklabels(["narrow pass", "narrow fail"], fontsize=6.3)
    ins.set_yticks([0, 1]); ins.set_yticklabels(["broad pass", "broad fail"], fontsize=6.3)
    ins.set_title("2×2 pass concordance", fontsize=7.2)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[fig] {out_png}")


# --------------------------------------------------------------------------- per-subject panel
def _rank01(vals):
    v = np.asarray(vals, float)
    out = np.full(v.shape, np.nan)
    ok = np.isfinite(v)
    if ok.sum() >= 2:
        out[ok] = np.argsort(np.argsort(v[ok])) / (ok.sum() - 1)
    elif ok.sum() == 1:
        out[ok] = 0.5
    return out


def _field_panel(ax, xs, ys, vals, support, xlim, ylim, sigma, title, soz):
    X, Y, T, _, _ = _smooth_rank_field_mm(xs, ys, vals, support, xlim, ylim, sigma)
    im = ax.imshow(T, origin="lower", extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                   aspect="equal", cmap="viridis", vmin=0, vmax=1)
    okv = np.isfinite(vals)
    ax.scatter(np.asarray(xs)[okv], np.asarray(ys)[okv], c=np.asarray(vals)[okv], cmap="viridis",
               vmin=0, vmax=1, s=70, zorder=3,
               edgecolors=["k" if z else "white" for z, v in zip(soz, vals) if np.isfinite(v)],
               linewidths=[1.6 if z else 0.5 for z, v in zip(soz, vals) if np.isfinite(v)])
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel("along template-A axis (mm, display frame)")
    ax.set_ylabel("transverse (mm, display frame)")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal", adjustable="box")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="early(0)->late(1) [display rank01]")
    return im


def _draw_null_panel(ax, record):
    gate = record["gate"]
    null = np.asarray(gate["null_corrs"], float)
    obs = gate["signed_corr"]
    if null.size:
        ax.hist(null, bins=24, color="0.75", edgecolor="0.4", alpha=0.85,
                 label=f"within-shaft null (n={null.size})")
    color = "#2b8a3e" if gate["passed"] else "#c1121f"
    if obs is not None:
        ax.axvline(obs, color=color, lw=2.2, label=f"observed r={obs:.3f}")
    cf = record.get("channel_floor", {})
    if np.isfinite(cf.get("null_p50", np.nan)):
        ax.axvline(cf["null_p50"], color="0.25", lw=1.4, ls="--",
                   label=f"channel-floor null median={cf['null_p50']:.2f}")
    rs = record.get("random_split", {})
    if np.isfinite(rs.get("split_median", np.nan)):
        ax.axvline(rs["split_median"], color="#1c7ed6", lw=1.4, ls=":",
                   label=f"random-split median={rs['split_median']:.2f} (non-inferential)")
    ax.axvline(0, color="0.88", lw=0.8, zorder=0)
    ax.set_xlabel("signed r (TA_field, TB_field)")
    ax.set_ylabel("count (within-shaft permutations)")
    txt = (f"percentile={gate['percentile']:.1f}  eff_n={gate['effective_n']}  "
           f"gate sigma={gate['sigma']:.3f}\n"
           f"{'PASSED' if gate['passed'] else 'not passed'} "
           f"(degenerate_null={gate['degenerate_null']})")
    ax.text(0.02, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.88))
    ax.legend(fontsize=6.6, loc="upper right", frameon=True)
    ax.set_title("observed reversal vs within-shaft / channel-floor / random-split", fontsize=10)


def plot_per_subject_panel(ds_sid: str, substrate: str, record: dict, out_png: Path) -> bool:
    """Brief panel 1 -- TA_field | TB_field (shared display frame, VIS constants, display-only)
    | observed signed_corr on its within-shaft null histogram + channel-floor + random-split."""
    dataset, subject = ds_sid.split("_", 1)
    geom_dir = GEOM[substrate]
    ta_f = geom_dir / f"{ds_sid}_t_a.json"
    if not ta_f.exists():
        print(f"[skip] {substrate}/{ds_sid}: no t_a geometry file"); return False
    plane_a = json.loads(ta_f.read_text())
    if "channels" not in plane_a:
        print(f"[skip] {substrate}/{ds_sid}: t_a plane not built (status-only record)"); return False

    try:
        bundle = load_event_labels_ranks(dataset, subject, broad=(substrate == "broad"))
    except Exception as e:
        print(f"[skip] {substrate}/{ds_sid}: bundle load failed: {e}"); return False

    cluster_map = {int(k): v for k, v in record["cluster_map"].items()}
    ta_label = next(k for k, v in cluster_map.items() if v == "t_a")
    tb_label = next(k for k, v in cluster_map.items() if v == "t_b")
    cav_ta = class_aggregate_contact_values(bundle, ta_label)
    cav_tb = class_aggregate_contact_values(bundle, tb_label)

    _attach_real_coords([plane_a])
    frame = _subject_display_frame([plane_a])
    if frame is None:
        print(f"[skip] {substrate}/{ds_sid}: no display frame"); return False
    xs, ys = _display_points(plane_a, frame)
    names = [c["name"] for c in plane_a["channels"]]
    soz = np.array([bool(c.get("is_soz")) for c in plane_a["channels"]])

    ta_vals = _rank01([cav_ta.get(n, {}).get("value", np.nan) for n in names])
    ta_sup = np.array([cav_ta.get(n, {}).get("support", 0.0) for n in names], float)
    tb_vals = _rank01([cav_tb.get(n, {}).get("value", np.nan) for n in names])
    tb_sup = np.array([cav_tb.get(n, {}).get("support", 0.0) for n in names], float)

    xlim, ylim, sigma = frame["xlim"], frame["ylim"], frame["sigma_mm"]
    n_ta = int(np.sum(np.asarray(bundle["labels"]) == ta_label))
    n_tb = int(np.sum(np.asarray(bundle["labels"]) == tb_label))

    fig, ax = plt.subplots(1, 3, figsize=(19.5, 6.2))
    _field_panel(ax[0], xs, ys, ta_vals, ta_sup, xlim, ylim, sigma,
                f"TA field ({n_ta} interictal events)", soz)
    _field_panel(ax[1], xs, ys, tb_vals, tb_sup, xlim, ylim, sigma,
                f"TB field ({n_tb} interictal events)", soz)
    _draw_null_panel(ax[2], record)

    pretty = ds_sid.replace("epilepsiae_", "E").replace("yuquan_", "Y-")
    gate = record["gate"]
    verdict = "PASSED beyond within-shaft null" if gate["passed"] else "not passed"
    fig.suptitle(f"{pretty}  |  substrate={substrate}  |  signed r={gate['signed_corr']:.3f}  |  {verdict}",
                 fontsize=13)
    fig.text(0.5, 0.005,
             "black ring = clinical SOZ contact (overlay only, not metric input). Field panels use "
             "display-only rank01 + VIS smoothing constants for visualization; the gate (right panel) "
             "is computed from the raw class-mean value on the primary (non-VIS) statistical sigma.",
             ha="center", fontsize=8.0, color="0.35")
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"[fig] {out_png}")
    return True


# --------------------------------------------------------------------------- main
def main():
    for s in SUBSTRATES:
        (OUT_DIR / s / "per_subject").mkdir(parents=True, exist_ok=True)

    per_subject = {s: _load_per_subject(s) for s in SUBSTRATES}
    cohort_summary = json.loads((IN_DIR / "cohort_summary.json").read_text())

    print("[1/6] cohort_stat (headline) ...")
    plot_cohort_stat(per_subject, cohort_summary, OUT_DIR)

    print("[2/6] accountability ...")
    plot_accountability(cohort_summary, OUT_DIR / "accountability.png")

    for s in SUBSTRATES:
        ok = _ok_records(per_subject[s])
        print(f"[3/6] {s}/field_vs_contact_headtohead ...")
        plot_field_vs_contact_headtohead(ok, s, OUT_DIR / s / "field_vs_contact_headtohead.png")
        print(f"[4/6] {s}/loo_reproducibility ...")
        plot_loo_reproducibility(ok, cohort_summary[s]["field_vs_contact_wilcoxon"], s,
                                  OUT_DIR / s / "loo_reproducibility.png")
        print(f"[bonus] {s}/field_reversal_null_forest (diagnostic, optional) ...")
        plot_null_forest(per_subject[s], s, OUT_DIR / s / "field_reversal_null_forest.png")

    print("[5/6] field_reversal_broad_vs_narrow ...")
    plot_broad_vs_narrow(cohort_summary, OUT_DIR / "field_reversal_broad_vs_narrow.png")

    print("[6/6] per-subject example panels ...")
    for s, examples in EXAMPLE_SUBJECTS.items():
        for ds_sid in examples:
            rec = per_subject[s].get(ds_sid)
            if rec is None:
                print(f"[skip-example] {s}/{ds_sid}: not found in per_subject records"); continue
            out_png = OUT_DIR / s / "per_subject" / f"{ds_sid}.png"
            try:
                plot_per_subject_panel(ds_sid, s, rec, out_png)
            except Exception as e:
                print(f"[fig-error] {s}/per_subject/{ds_sid}: {e}")

    print("[deferred] case_1146_mechanism.png -- see module docstring for rationale; not rendered.")
    print(f"[done] figures written under {OUT_DIR}")


if __name__ == "__main__":
    main()
