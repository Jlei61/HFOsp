"""Topic5 contact-similarity — 3 paper-grade figures (method -> result -> robustness).

Plan: docs/superpowers/plans/2026-07-02-topic5-contact-similarity-paper-figures.md

Narrow 口径 (verbatim discipline, all captions/titles): spatially-weighted contact
rank captures the same coarse interictal<->ictal spatial scaffold as the gridded
field. native-3D adds no distinguishable info beyond the 2D-plane rank (CI within
±SESOI -> equivalence PASSES); the grid step shows no distinguishable gain over
the same-plane contact rank, but its CI is wider than ±SESOI, so this is NOT a
strict-zero claim. NEVER "predicts / characterizes the pathological network."
Single-subject panels (fig1, fig2-left,
fig3-left) are illustrative; the cohort null-context (fig2-right, fig3-right) is
the honest counterweight.

Reuses (import, do NOT reinvent): scripts.run_topic5_contact_similarity._ctx for
the representative subject's matched channels / plane / frozen sigma / per-seizure
bb_auc vectors; src.topic5_contact_similarity.kernel_smooth_at_contacts for
contact-level spatial weighting; src.topic5_axis_alignment.make_field_record +
src.propagation_contact_plane_readout.{R_smooth_rank, make_plane_grid, S_THRESH}
for the gridded field. Cohort statistics are read straight from
cohort_summary_{activation}.json / r2b_summary_{activation}.json — no recompute.

Three figures, one shared loader `_load_subject_ctx`:
  fig1 — spatial-weighting method schematic (input dots -> Gaussian-kernel
         illustration -> smoothed output), representative subject only.
  fig2 — LEFT: rankdisp-style per-contact rank comparison (spatially-weighted
         interictal rank A/B vs ictal early-broadband-energy rank), sorted
         source->sink along template A. RIGHT: cohort null context — R2 obs vs
         its within-shaft-shuffle null p95, R1 obs as a light contrast.
  fig3 — LEFT: same weighted-rank quantity at contacts vs the gridded field
         (visually the same shape). RIGHT: cohort R2-vs-R3 (grid: no
         distinguishable gain, CI wider than ±SESOI, not zero) and R2b-vs-R2_nm
         (native-3D: equivalence PASSES within ±SESOI) scatters.
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
from scipy.stats import pearsonr

# Chinese font: matplotlib indexes the Noto Sans CJK .ttc under the JP family
# name (glyphs for common Hanzi are shared across SC/TC/JP/KR), matching the
# convention in scripts/plot_pr25_split_half_schematic.py. Panel titles in
# fig1/fig3 use Chinese method labels ("① 输入", "不铺网格" ...).
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.style"] = "normal"   # Noto Sans CJK has no italic variant

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.plot_style import DPI_PUB, FS_LABEL, FS_TICK, FS_TITLE, style_panel
from src.topic5_contact_similarity import kernel_smooth_at_contacts
from src.topic5_axis_alignment import make_field_record
from src.propagation_contact_plane_readout import (
    R_smooth_rank, S_THRESH, X_LO, X_HI, Y_EXT,
)
from scripts.run_topic5_contact_similarity import _ctx, DEF_ROOT, DEF_OUT, SESOI

REPRESENTATIVE_SUBJECT = "epilepsiae_1146"   # 15 contacts, 2 shafts, passes R1/R2/R3 within-shaft null
COL_TEMPLATE_A = "#B71C2B"   # red — matches scripts/plot_rank_displacement.py convention
COL_TEMPLATE_B = "#1F4E9C"   # blue


# --------------------------------------------------------------------------- shared loader

def _load_subject_ctx(ds_sid: str, activation: str, root: str) -> dict:
    """Representative-subject context: `_ctx` fields + a per-contact ictal energy
    vector (mean of bb_auc over the subject's ok seizures — a plain per-subject
    summary for illustration, NOT the per-seizure null-fold used by the cohort
    R1/R2/R3 statistics)."""
    ctx = _ctx(ds_sid, activation, input_results_root=root)
    if ctx is None:
        raise RuntimeError(f"{ds_sid}: no eligible T0/axis context for activation={activation} "
                           f"under root={root}")
    sz_arr = np.array(list(ctx["sz_vals"].values()), dtype=float)
    ctx["ictal_mean"] = np.nanmean(sz_arr, axis=0)
    ctx["subject_id"] = ds_sid
    ctx["activation"] = activation
    return ctx


def _plane_bounds(pts: np.ndarray, pad: float = 0.15):
    return ((pts[:, 0].min() - pad, pts[:, 0].max() + pad),
            (pts[:, 1].min() - pad, pts[:, 1].max() + pad))


def save_fig(fig: plt.Figure, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI_PUB, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- Fig 1: method schematic

def fig1(ctx: dict) -> plt.Figure:
    """3-step method cartoon: input dots -> Gaussian-kernel spatial weighting -> smoothed output.
    Pure method illustration — no statistics, no claim.

    Uses an EXAGGERATED display-only sigma (~4x median nearest-neighbor contact
    spacing) for the ② kernel-illustration radius AND the ③ smoothed output, so
    the smoothing effect is visible on sparse contacts (at ~2-3x the blending is
    too subtle to see at this contact density). This sigma_display is NOT the
    frozen analysis sigma (ctx["sigma"]) used everywhere else in the paper — it
    exists only to make this schematic legible."""
    pts = ctx["source_pts"]
    sup = np.asarray(ctx["support"], float)
    rank_a = np.asarray(ctx["rank_a"], float)
    nn = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(nn, np.inf)
    sigma_display = 4.0 * float(np.median(nn.min(axis=1)))
    smoothed = kernel_smooth_at_contacts(rank_a, pts, pts, sup, sigma_display)
    xlim, ylim = _plane_bounds(pts)
    n = pts.shape[0]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6))
    cmap = plt.cm.viridis

    ax = axes[0]
    ax.scatter(pts[:, 0], pts[:, 1], c=rank_a, cmap=cmap, vmin=0, vmax=1, s=170,
               edgecolors="black", linewidths=1.0, zorder=3)
    ax.set_title("① 输入：一个序列 = 每触点一个值，\n放到它的空间位置", fontsize=FS_LABEL - 1)

    ax = axes[1]
    ax.scatter(pts[:, 0], pts[:, 1], c=rank_a, cmap=cmap, vmin=0, vmax=1, s=110,
               alpha=0.35, edgecolors="none", zorder=2)
    # focal contacts = the two extremes along x_norm (illustrative, not the axis endpoints)
    focal_idx = sorted({int(np.argmin(pts[:, 0])), int(np.argmax(pts[:, 0]))})
    for fi in focal_idx:
        circ = plt.Circle((pts[fi, 0], pts[fi, 1]), sigma_display, facecolor="0.6",
                          edgecolor="0.4", alpha=0.18, zorder=1)
        ax.add_patch(circ)
        d2 = ((pts - pts[fi]) ** 2).sum(axis=1)
        w = sup * np.exp(-d2 / (2.0 * sigma_display ** 2))
        wn = w / w.max() if w.max() > 0 else w
        for j in range(n):
            if j == fi:
                continue
            ax.plot([pts[fi, 0], pts[j, 0]], [pts[fi, 1], pts[j, 1]],
                    color="0.45", alpha=float(min(wn[j], 1.0)) * 0.85,
                    linewidth=0.6 + 3.0 * wn[j], zorder=1)
        ax.scatter([pts[fi, 0]], [pts[fi, 1]], c=[rank_a[fi]], cmap=cmap, vmin=0, vmax=1,
                   s=220, edgecolors="black", linewidths=1.6, zorder=4)
    ax.set_title("② 空间加权：新值 = 自己 + 邻近触点\n按距离(高斯核)加权平均", fontsize=FS_LABEL - 1)

    ax = axes[2]
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=smoothed, cmap=cmap, vmin=0, vmax=1, s=170,
                    edgecolors="black", linewidths=1.0, zorder=3)
    ax.set_title("③ 输出：空间平滑后的形状", fontsize=FS_LABEL - 1)

    for ax in axes:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("contact plane (x)", fontsize=FS_TICK - 2)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("contact plane (y)", fontsize=FS_TICK - 2)

    cbar_ax = fig.add_axes((0.955, 0.15, 0.014, 0.68))
    cb = fig.colorbar(sc, cax=cbar_ax)
    cb.set_label("interictal rank (0=early/source → 1=late)", fontsize=FS_LABEL - 2)

    fig.suptitle(f"Spatial-weighting method schematic — {ctx['subject_id']} "
                 "(illustrative, no statistics)", fontsize=FS_TITLE, y=1.03)
    fig.text(0.47, -0.01,
             f"示意用 σ(放大)={sigma_display:.3f} ≈ 4×中位最近邻触点间距,非分析所用 σ",
             ha="center", fontsize=FS_TICK - 2, color="#999")
    fig.tight_layout(rect=(0.0, 0.02, 0.94, 1.0))
    return fig


# --------------------------------------------------------------------------- Fig 2: rank comparison

def fig2(ctx: dict, cohort_summary: dict) -> plt.Figure:
    """LEFT: rankdisp-style direct contact-rank comparison for the representative subject.
    RIGHT: cohort within-shaft null context (null-比-null, mode a)."""
    pts = ctx["source_pts"]
    sup = np.asarray(ctx["support"], float)
    sigma = float(ctx["sigma"])
    rank_a = np.asarray(ctx["rank_a"], float)
    rank_b = np.asarray(ctx["rank_b"], float) if ctx["rank_b"] is not None else None
    names = ctx["names_m"]
    order = np.argsort(rank_a)          # T_a source -> sink, plot_rank_displacement.py convention

    w_a = kernel_smooth_at_contacts(rank_a, pts, pts, sup, sigma)
    w_b = kernel_smooth_at_contacts(rank_b, pts, pts, sup, sigma) if rank_b is not None else None
    w_e = kernel_smooth_at_contacts(ctx["ictal_mean"], pts, pts, sup, sigma)

    fin_e = np.isfinite(w_e)
    e_rank = np.full_like(w_e, np.nan)
    if fin_e.sum() >= 2:
        dense = np.argsort(np.argsort(w_e[fin_e]))
        denom = max(int(fin_e.sum()) - 1, 1)
        e_rank[fin_e] = dense / denom

    def _corr(w):
        if w is None:
            return np.nan
        m = np.isfinite(w) & fin_e
        if m.sum() < 3:
            return np.nan
        return float(pearsonr(w[m], w_e[m])[0])

    cands = [(lab, r) for lab, r in (("A", _corr(w_a)), ("B", _corr(w_b))) if np.isfinite(r)]
    max_lab, max_r = max(cands, key=lambda t: abs(t[1])) if cands else ("n/a", float("nan"))

    fig = plt.figure(figsize=(17.0, 6.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.55)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])

    # ---- LEFT: representative-subject rank comparison ----
    x = np.arange(len(order))
    axL.plot(x, w_a[order], "-o", color=COL_TEMPLATE_A, ms=6, lw=1.8, zorder=3,
             label="template A — spatially-weighted interictal rank")
    if w_b is not None:
        axL.plot(x, w_b[order], "-o", color=COL_TEMPLATE_B, ms=6, lw=1.8, zorder=3,
                 label="template B — spatially-weighted interictal rank")
    axL.set_ylim(-0.03, 1.03)
    axL.set_xticks(x)
    axL.set_xticklabels([names[i] for i in order], rotation=60, ha="right", fontsize=FS_TICK - 3)
    axL.set_xlabel("contact (sorted source → sink along template A)", fontsize=FS_LABEL - 1)
    axL.set_ylabel("interictal rank\n(spatially-weighted, 0=early/source → 1=late)",
                   fontsize=FS_LABEL - 3)
    style_panel(axL)

    axL2 = axL.twinx()
    axL2.plot(x, e_rank[order], "--D", color="black", ms=5, lw=1.4, alpha=0.85, zorder=2,
              label="ictal early-broadband-energy rank\n(rank of spatially-weighted bb_auc)")
    axL2.set_ylim(-0.03, 1.03)
    axL2.set_ylabel("ictal energy rank\n(0=lowest → 1=highest, spatially-weighted)",
                    fontsize=FS_LABEL - 3)
    axL2.spines["top"].set_visible(False)

    h1, l1 = axL.get_legend_handles_labels()
    h2, l2 = axL2.get_legend_handles_labels()
    axL.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=FS_TICK - 4, frameon=False)
    axL.set_title(
        f"{ctx['subject_id']} — direct contact-rank comparison\n"
        f"maxAB match: template {max_lab}, |r|={abs(max_r):.2f} "
        "(illustrative same-plane correlation, single subject)",
        fontsize=FS_LABEL - 1,
    )

    # ---- RIGHT: cohort within-shaft null context ----
    rows = []
    for s in cohort_summary.get("per_subject", []):
        if s.get("status") != "ok":
            continue
        r2 = s.get("R2", {}).get("within_shaft", {})
        r1 = s.get("R1", {}).get("within_shaft", {})
        if r2.get("status") != "ok":
            continue
        rows.append({"subject_id": s["subject_id"], "r2_obs": r2["obs_subject"],
                     "r2_p95": r2["null_q"]["p95"], "r1_obs": r1.get("obs_subject", np.nan),
                     "passed": bool(r2.get("passed"))})
    rows.sort(key=lambda r: r["r2_obs"])
    n = len(rows)
    y = np.arange(n)
    r2_obs = [r["r2_obs"] for r in rows]
    r1_obs = [r["r1_obs"] for r in rows]

    for yi, r in zip(y, rows):
        axR.plot([r["r2_p95"], r["r2_p95"]], [yi - 0.32, yi + 0.32], color="0.35", lw=2.2, zorder=2)
    axR.scatter(r2_obs, y, s=70, color="#fdae61", edgecolors="black", linewidths=0.6, zorder=3,
               label="R2 obs (spatially-weighted, in-plane)")
    axR.scatter(r1_obs, y, s=46, facecolors="none", edgecolors="0.55", linewidths=1.2, zorder=1,
               label="R1 obs (raw, no geometry)")
    axR.plot([], [], color="0.35", lw=2.2, label="within-shaft-shuffle null p95")
    for yi, r in zip(y, rows):
        if r["subject_id"] == ctx["subject_id"]:
            axR.scatter([r["r2_obs"]], [yi], s=190, facecolors="none", edgecolors="black",
                       linewidths=1.8, zorder=4, marker="*",
                       label=f"{ctx['subject_id']} (left panel)")

    axR.set_yticks(y)
    ylabels = axR.set_yticklabels([r["subject_id"] for r in rows], fontsize=FS_TICK - 4)
    for lbl, r in zip(ylabels, rows):
        if r["passed"]:
            lbl.set_fontweight("bold")
    axR.set_xlabel("|maxAB similarity|  (interictal ↔ ictal)", fontsize=FS_LABEL - 1)
    n_ok = cohort_summary.get("n_ok", n)
    n_pass_r1 = cohort_summary.get("n_pass_R1_within_shaft")
    n_pass_r2 = cohort_summary.get("n_pass_R2_within_shaft")
    axR.set_title(
        f"cohort within-shaft null context (n={n_ok})\n"
        f"clears null: R1={n_pass_r1}/{n_ok}  R2={n_pass_r2}/{n_ok}  "
        "(bold label = R2 clears null)",
        fontsize=FS_LABEL - 2,
    )
    handles, labels = axR.get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    axR.legend(seen.values(), seen.keys(), loc="lower right", fontsize=FS_TICK - 5, frameon=False)
    style_panel(axR)

    # NOTE: fig.tight_layout() is not compatible with twinx() axes (axL2) and
    # visibly collapses the gap between axL2's right-side ylabel and axR's
    # y-tick subject labels; rely on the gridspec wspace + explicit margins.
    fig.subplots_adjust(left=0.06, right=0.965, top=0.84, bottom=0.28, wspace=0.42)
    return fig


# --------------------------------------------------------------------------- Fig 3: vs field + native-3D

def _scatter_vs_diag(ax, xv, yv, color, xlabel, ylabel, title, annotation=None):
    lo = float(min(xv.min(), yv.min())) - 0.05
    hi = float(max(xv.max(), yv.max())) + 0.05
    ax.plot([lo, hi], [lo, hi], color="0.4", lw=1.4, ls="--", zorder=1, label="y = x")
    ax.fill_between([lo, hi], [lo - SESOI, hi - SESOI], [lo + SESOI, hi + SESOI],
                    color="0.4", alpha=0.12, zorder=0, label=f"±SESOI ({SESOI:.2f})")
    ax.scatter(xv, yv, s=55, color=color, edgecolors="black", linewidths=0.4, zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel, fontsize=FS_TICK - 3)
    ax.set_ylabel(ylabel, fontsize=FS_TICK - 3)
    ax.set_title(title, fontsize=FS_TICK - 2, pad=34)
    if annotation:
        # top-left corner: both panels hug the y=x diagonal, so the off-diagonal
        # corner is data-sparse and safe for a compact stats box (matches the
        # AUC-box convention in scripts/plot_rank_displacement.py).
        ax.text(0.04, 0.97, annotation, transform=ax.transAxes, fontsize=FS_TICK - 5,
               ha="left", va="top",
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                        edgecolor="0.80", linewidth=0.5, alpha=0.92))
    style_panel(ax)


def fig3(ctx: dict, cohort_summary: dict, r2b_summary: dict) -> plt.Figure:
    """LEFT: same weighted-interictal-rank quantity at contacts vs the gridded field
    for the representative subject (visually the same shape). RIGHT: cohort R2-vs-R3
    (grid step: no distinguishable gain, CI wider than ±SESOI — not a strict-zero
    claim) and R2b-vs-R2_nm (native-3D: equivalence PASSES within ±SESOI)."""
    pts = ctx["source_pts"]
    sup = np.asarray(ctx["support"], float)
    sigma = float(ctx["sigma"])
    rank_a = np.asarray(ctx["rank_a"], float)
    w_contacts = kernel_smooth_at_contacts(rank_a, pts, pts, sup, sigma)
    X, Y = ctx["X"], ctx["Y"]
    field = R_smooth_rank(make_field_record(ctx["matched"], ctx["rank_a"]), X, Y, sigma, S_THRESH)
    T_show = np.where(field["mask"], field["T"], np.nan)
    xlim, ylim = _plane_bounds(pts)

    fig = plt.figure(figsize=(17.5, 7.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.32,
                         left=0.02, right=0.98, top=0.82, bottom=0.10)
    gsL = gs[0, 0].subgridspec(1, 2, wspace=0.08)
    axL1 = fig.add_subplot(gsL[0, 0])
    axL2 = fig.add_subplot(gsL[0, 1])
    gsR = gs[0, 1].subgridspec(2, 1, hspace=0.95)
    axR1 = fig.add_subplot(gsR[0, 0])
    axR2 = fig.add_subplot(gsR[1, 0])

    # ---- LEFT: contact-weighted vs gridded field, same quantity ----
    sc = axL1.scatter(pts[:, 0], pts[:, 1], c=w_contacts, cmap="viridis", vmin=0, vmax=1, s=170,
                      edgecolors="black", linewidths=1.0, zorder=3)
    axL1.set_title("不铺网格：触点上的空间加权",
                   fontsize=FS_LABEL - 2)

    cmap_field = plt.cm.viridis.copy()
    cmap_field.set_bad(color="white")
    im2 = axL2.imshow(T_show, origin="lower", extent=(X_LO, X_HI, -Y_EXT, Y_EXT), aspect="equal",
               cmap=cmap_field, vmin=0, vmax=1)
    axL2.scatter(pts[:, 0], pts[:, 1], c=w_contacts, cmap="viridis", vmin=0, vmax=1, s=45,
                edgecolors="black", linewidths=0.5, zorder=3)
    axL2.set_title("铺网格：81×81 场（同一量）",
                   fontsize=FS_LABEL - 2)

    for ax in (axL1, axL2):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # colorbar attaches directly to axL2's gridded-field imshow (right edge), not
    # floating between the left and right panel groups; scatter (sc) and imshow
    # (im2) share vmin=0/vmax=1/cmap so either mappable reads the same scale.
    pos2 = axL2.get_position()
    cbar_ax = fig.add_axes((pos2.x1 + 0.008, pos2.y0, 0.011, pos2.height))
    cb = fig.colorbar(im2, cax=cbar_ax)
    cb.set_label("interictal rank\n(0=early/source → 1=late)", fontsize=FS_LABEL - 4)
    fig.text(0.235, 0.94, f"{ctx['subject_id']} — same quantity, contact vs grid",
             ha="center", fontsize=FS_LABEL, fontweight="bold")

    # ---- RIGHT top: cohort R2 vs R3 ----
    r2v, r3v = [], []
    for s in cohort_summary.get("per_subject", []):
        if s.get("status") != "ok":
            continue
        r2 = s.get("R2", {}).get("within_shaft", {})
        r3 = s.get("R3", {}).get("within_shaft", {})
        if r2.get("status") != "ok" or r3.get("status") != "ok":
            continue
        r2v.append(r2["obs_subject"])
        r3v.append(r3["obs_subject"])
    r2v, r3v = np.array(r2v), np.array(r3v)
    gd_med = cohort_summary.get("grid_delta_median")
    gd_ci = cohort_summary.get("grid_delta_ci")
    ann1 = (f"grid_delta median={gd_med:+.3f}\nCI=[{gd_ci[0]:+.3f},{gd_ci[1]:+.3f}]  n={len(r2v)}"
            if gd_med is not None else "grid_delta: n/a")
    _scatter_vs_diag(axR1, r2v, r3v, "#d73027",
                     "R2 obs (in-plane smoothed contact similarity)",
                     "R3 obs (gridded field similarity)",
                     "grid vs contact rank (R2 vs R3):\nno distinguishable gain "
                     "(CI wider than ±SESOI, not zero)", ann1)
    axR1.legend(fontsize=FS_TICK - 5, frameon=False, loc="lower right")

    # ---- RIGHT bottom: cohort R2b vs R2_nm ----
    nmv, bv = [], []
    for s in r2b_summary.get("per_subject", []):
        if s.get("r2b_status") != "ok":
            continue
        r2nm, r2b = s.get("R2_nm", {}), s.get("R2b", {})
        if "obs_subject" not in r2nm or "obs_subject" not in r2b:
            continue
        nmv.append(r2nm["obs_subject"])
        bv.append(r2b["obs_subject"])
    nmv, bv = np.array(nmv), np.array(bv)
    rb_med = r2b_summary.get("r2b_minus_r2nm_median")
    rb_ci = r2b_summary.get("r2b_minus_r2nm_ci")
    ann2 = (f"r2b_minus_r2nm median={rb_med:+.4f}\nCI=[{rb_ci[0]:+.3f},{rb_ci[1]:+.3f}]  n={len(nmv)}"
            if rb_med is not None else "r2b_minus_r2nm: n/a")
    _scatter_vs_diag(axR2, nmv, bv, "#1F4E9C",
                     "R2_nm obs (2D-plane, no mirror)", "R2b obs (native-3D mm, no mirror)",
                     "native-3D vs 2D-plane (R2_nm vs R2b):\n"
                     "equivalence PASSES (CI within ±SESOI)", ann2)

    fig.suptitle("触点加权 ≈ 铺网格场的同一形状；native-3D 等价通过（无可分辨增益），"
                 "网格未见可分辨增益（CI 宽于 SESOI，非零）", fontsize=FS_TITLE - 1, y=1.02)
    return fig


# --------------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-results-root", default=DEF_ROOT,
                    help="root holding the gitignored T0 cache + axis records (default: results)")
    ap.add_argument("--out-dir", default=DEF_OUT,
                    help="contact-similarity results dir (holds cohort_summary_*.json + figures/)")
    ap.add_argument("--subject", default=REPRESENTATIVE_SUBJECT)
    ap.add_argument("--activation", choices=["broadband", "hfa"], default="broadband")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ctx = _load_subject_ctx(args.subject, args.activation, args.input_results_root)
    cohort_summary = json.load(open(out_dir / f"cohort_summary_{args.activation}.json"))
    r2b_summary = json.load(open(out_dir / f"r2b_summary_{args.activation}.json"))

    p1 = save_fig(fig1(ctx), fig_dir / "fig1_spatial_weighting_schematic.png")
    p2 = save_fig(fig2(ctx, cohort_summary), fig_dir / "fig2_rank_comparison.png")
    p3 = save_fig(fig3(ctx, cohort_summary, r2b_summary), fig_dir / "fig3_vs_field.png")
    for p in (p1, p2, p3):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
