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
from src.topic5_contact_similarity import kernel_smooth_at_contacts, median_nn_spacing
from src.topic5_axis_alignment import make_field_record
from src.propagation_contact_plane_readout import (
    R_smooth_rank, S_THRESH, X_LO, X_HI, Y_EXT,
)
from src.seeg_coord_loader import (
    load_subject_coords, assert_coord_result_is_mm_for_main_analysis,
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

def fig2(ctx: dict) -> plt.Figure:
    """Vertical rank ladder for the representative subject on the natural integer rank
    axis (1..n; all contacts participate, so no [0,1] renormalization). Three per-
    contact rank sequences, contacts sorted by the ictal (seizure) order so the seizure
    is a clean 1..n reference:
      - 发作 (black)               : ictal early-broadband-energy rank
      - 间期·空间加权最像 (red)    : whichever WEIGHTED template A/B best matches ictal
      - 间期·非空间加权最像 (blue) : whichever RAW template A/B best matches ictal
    "best matches" = max sign-free |corr| (the ladder's maxAB rule); the winning
    template is oriented so its rank runs WITH the seizure direction (propagation
    direction is arbitrary / sign-free). Illustrative single subject; cohort null in
    fig2_sup."""
    pts = ctx["source_pts"]
    sup = np.asarray(ctx["support"], float)
    sigma = float(ctx["sigma"])
    rank_a = np.asarray(ctx["rank_a"], float)
    rank_b = np.asarray(ctx["rank_b"], float) if ctx["rank_b"] is not None else None
    ictal = np.asarray(ctx["ictal_mean"], float)
    names = ctx["names_m"]

    def _int_rank(v):                       # dense integer rank 1..m over finite entries
        v = np.asarray(v, float)
        out = np.full(v.shape, np.nan)
        fin = np.isfinite(v)
        if int(fin.sum()) >= 2:
            out[fin] = np.argsort(np.argsort(v[fin])) + 1.0
        return out

    def _best_and_orient(cands, ref):       # max |corr| over A/B, then orient to +corr with ref
        m = np.isfinite(ref)
        best = None
        for lab, v in cands:
            mm = m & np.isfinite(v)
            if int(mm.sum()) < 3:
                continue
            r = float(pearsonr(v[mm], ref[mm])[0])
            if best is None or abs(r) > abs(best[2]):
                best = (lab, v, r)
        lab, v, r = best
        return lab, (v if r >= 0 else -v), r

    raw_cands = [("A", rank_a)] + ([("B", rank_b)] if rank_b is not None else [])
    wtd_cands = [("A", kernel_smooth_at_contacts(rank_a, pts, pts, sup, sigma))]
    if rank_b is not None:
        wtd_cands.append(("B", kernel_smooth_at_contacts(rank_b, pts, pts, sup, sigma)))

    raw_lab, raw_v, _ = _best_and_orient(raw_cands, ictal)
    wtd_lab, wtd_v, _ = _best_and_orient(wtd_cands, ictal)

    ri = _int_rank(ictal)                   # 发作
    rraw = _int_rank(raw_v)                 # 间期·非空间加权最像
    rwtd = _int_rank(wtd_v)                 # 间期·空间加权最像

    order = np.argsort(np.where(np.isfinite(ri), ri, np.inf))   # contacts in seizure order
    y = np.arange(len(order))
    n = int(np.isfinite(ri).sum())

    fig, ax = plt.subplots(figsize=(7.8, 8.8))
    ax.plot(ri[order], y, "--D", color="black", ms=6, lw=1.6, alpha=0.9, zorder=2,
            label="发作（发作早期能量 rank）")
    ax.plot(rwtd[order], y, "-o", color=COL_TEMPLATE_A, ms=7, lw=2.0, zorder=4,
            label=f"间期·空间加权最像（模板 {wtd_lab}）")
    ax.plot(rraw[order], y, "-s", color=COL_TEMPLATE_B, ms=6, lw=1.7, zorder=3,
            label=f"间期·非空间加权最像（模板 {raw_lab}）")

    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order], fontsize=FS_TICK - 2)
    ax.invert_yaxis()                       # seizure-earliest (rank 1) at top
    ax.set_xlim(0.3, n + 0.7)
    ax.set_xticks(range(1, n + 1))
    ax.set_xlabel("rank（1 … n；发作=按发作早期能量，间期模板方向已按发作对齐）",
                  fontsize=FS_LABEL - 2)
    ax.set_ylabel("contact（按发作 rank 自上而下）", fontsize=FS_LABEL - 2)
    ax.legend(loc="upper right", fontsize=FS_TICK - 2, frameon=True, facecolor="white",
              framealpha=0.92, edgecolor="0.8")
    ax.set_title(f"{ctx['subject_id']} — 发作 vs 间期空间加权最像 vs 间期非空间加权最像（示意，单被试）",
                 fontsize=FS_LABEL - 2)
    style_panel(ax)
    fig.tight_layout()
    return fig


def fig2_sup(cohort_summary: dict, star_subject: str) -> plt.Figure:
    """Cohort per-subject maxAB (R2 spatially-weighted obs) vs its within-shaft
    shuffle null p95 (null-比-null): spatial weighting raises the observed
    similarity but also raises the null, so only a minority clear it."""
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

    fig, axR = plt.subplots(figsize=(8.4, 6.8))
    for yi, r in zip(y, rows):
        axR.plot([r["r2_p95"], r["r2_p95"]], [yi - 0.32, yi + 0.32], color="0.35", lw=2.2, zorder=2)
    axR.scatter([r["r2_obs"] for r in rows], y, s=70, color="#fdae61", edgecolors="black",
                linewidths=0.6, zorder=3, label="R2 obs (spatially-weighted, in-plane)")
    axR.scatter([r["r1_obs"] for r in rows], y, s=46, facecolors="none", edgecolors="0.55",
                linewidths=1.2, zorder=1, label="R1 obs (raw, no geometry)")
    axR.plot([], [], color="0.35", lw=2.2, label="within-shaft-shuffle null p95")
    for yi, r in zip(y, rows):
        if r["subject_id"] == star_subject:
            axR.scatter([r["r2_obs"]], [yi], s=190, facecolors="none", edgecolors="black",
                        linewidths=1.8, zorder=4, marker="*", label=star_subject)
    axR.set_yticks(y)
    ylabels = axR.set_yticklabels([r["subject_id"] for r in rows], fontsize=FS_TICK - 3)
    for lbl, r in zip(ylabels, rows):
        if r["passed"]:
            lbl.set_fontweight("bold")
    axR.set_xlabel("|maxAB similarity|  (interictal ↔ ictal)", fontsize=FS_LABEL - 1)
    n_ok = cohort_summary.get("n_ok", n)
    n_pass_r1 = cohort_summary.get("n_pass_R1_within_shaft")
    n_pass_r2 = cohort_summary.get("n_pass_R2_within_shaft")
    axR.set_title(
        f"per-subject maxAB vs within-shaft null (n={n_ok})\n"
        f"clears null: R1={n_pass_r1}/{n_ok}  R2={n_pass_r2}/{n_ok}  "
        "(bold label = R2 clears null)",
        fontsize=FS_LABEL - 2,
    )
    handles, labels = axR.get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    axR.legend(seen.values(), seen.keys(), loc="lower right", fontsize=FS_TICK - 4,
               frameon=True, facecolor="white", framealpha=0.9, edgecolor="0.8")
    style_panel(axR)
    fig.tight_layout()
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


def _dist_2d_vs_3d_panel(ax, ctx: dict) -> None:
    """σ-normalized inter-contact distance scatter: 2D-plane vs native-3D (mm).
    Each distance is divided by its own space's median nearest-neighbor spacing
    (σ_xy for the normalized plane, σ_3D for mm) — the same bandwidth the Gaussian
    kernel weights on — so the y=x diagonal is meaningful. Points hugging the
    diagonal ⇒ the 2D plane preserves the relative neighbor structure the kernel
    uses ⇒ the 2D-plane and native-3D weightings are ~identical ⇒ why R2b ≈ R2_nm.
    Same common-subset + mm hard gate as scripts/augment_topic5_r2b_3d.py."""
    from itertools import combinations
    ds, subj = ctx["subject_id"].split("_", 1)
    cr = load_subject_coords(ds, subj, ctx["names_m"], allow_voxel_fallback=False)
    assert_coord_result_is_mm_for_main_analysis(cr)
    coords_all = np.asarray(cr.coords_array_in_requested_order, float)
    mask = np.asarray(cr.mapped_mask_in_requested_order, bool) & np.isfinite(coords_all).all(axis=1)
    idx = np.where(mask)[0]
    c3 = coords_all[idx]
    p2 = np.asarray(ctx["source_pts"], float)[idx]
    sigma_xy = float(ctx["sigma"])
    sigma_3d = float(median_nn_spacing(c3))

    ij = list(combinations(range(len(idx)), 2))
    d2 = np.array([np.linalg.norm(p2[i] - p2[j]) for i, j in ij]) / sigma_xy
    d3 = np.array([np.linalg.norm(c3[i] - c3[j]) for i, j in ij]) / sigma_3d
    r = float(pearsonr(d2, d3)[0])

    hi = float(max(d2.max(), d3.max())) * 1.05
    ax.plot([0, hi], [0, hi], color="0.4", lw=1.4, ls="--", zorder=1, label="y = x")
    ax.scatter(d2, d3, s=26, color="#4a7db5", edgecolors="black", linewidths=0.3,
               alpha=0.75, zorder=3)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("2D-plane 触点间距离 / σ_xy", fontsize=FS_TICK - 2)
    ax.set_ylabel("native-3D 触点间距离 / σ_3D", fontsize=FS_TICK - 2)
    ax.set_title(f"{ctx['subject_id']}：触点间距离 2D vs 3D\n（各按自身 σ 归一,{len(idx)} 触点）",
                 fontsize=FS_TICK - 1, pad=10)
    ax.text(0.04, 0.97, f"Pearson r = {r:.3f}\n触点近共面 → 平面近乎无损保留三维几何",
            transform=ax.transAxes, fontsize=FS_TICK - 4, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.80",
                      linewidth=0.5, alpha=0.92))
    ax.legend(fontsize=FS_TICK - 4, frameon=False, loc="lower right")
    style_panel(ax)


def fig3(ctx: dict, cohort_summary: dict, r2b_summary: dict) -> plt.Figure:
    """(a) same weighted-interictal-rank quantity at contacts vs the gridded field
    (same shape); (b) 2D-plane vs native-3D σ-normalized inter-contact distances
    (why 2D ≈ 3D); (c) cohort R2-vs-R3 (grid step: no distinguishable gain, CI wider
    than ±SESOI — not strict-zero) and R2b-vs-R2_nm (native-3D: equivalence PASSES
    within ±SESOI)."""
    pts = ctx["source_pts"]
    sup = np.asarray(ctx["support"], float)
    sigma = float(ctx["sigma"])
    rank_a = np.asarray(ctx["rank_a"], float)
    w_contacts = kernel_smooth_at_contacts(rank_a, pts, pts, sup, sigma)
    X, Y = ctx["X"], ctx["Y"]
    field = R_smooth_rank(make_field_record(ctx["matched"], ctx["rank_a"]), X, Y, sigma, S_THRESH)
    T_show = np.where(field["mask"], field["T"], np.nan)
    xlim, ylim = _plane_bounds(pts)

    fig = plt.figure(figsize=(16.5, 9.2))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.05], hspace=0.42,
                             left=0.045, right=0.925, top=0.90, bottom=0.075)
    gsT = outer[0].subgridspec(1, 2, wspace=0.14)
    axMap1 = fig.add_subplot(gsT[0, 0])
    axMap2 = fig.add_subplot(gsT[0, 1])
    gsB = outer[1].subgridspec(1, 3, wspace=0.42)
    axDist = fig.add_subplot(gsB[0, 0])
    axR1 = fig.add_subplot(gsB[0, 1])
    axR2 = fig.add_subplot(gsB[0, 2])

    # (a) TOP: contact-weighted map vs gridded field. Same xlim/ylim + aspect='auto'
    # so both fill their cells identically — the comparison is contact-scatter vs
    # grid-field of the SAME subject (both stretched the same), so "same shape" reads
    # directly without the elongated-plane whitespace that equal-aspect would leave.
    axMap1.scatter(pts[:, 0], pts[:, 1], c=w_contacts, cmap="viridis", vmin=0, vmax=1, s=170,
                   edgecolors="black", linewidths=1.0, zorder=3)
    axMap1.set_title(f"① 不铺网格：触点上的空间加权（{ctx['subject_id']}）", fontsize=FS_LABEL - 2)
    cmap_field = plt.cm.viridis.copy()
    cmap_field.set_bad(color="white")
    im2 = axMap2.imshow(T_show, origin="lower", extent=(X_LO, X_HI, -Y_EXT, Y_EXT), aspect="auto",
                        cmap=cmap_field, vmin=0, vmax=1)
    axMap2.scatter(pts[:, 0], pts[:, 1], c=w_contacts, cmap="viridis", vmin=0, vmax=1, s=55,
                   edgecolors="black", linewidths=0.6, zorder=3)
    axMap2.set_title("② 铺网格：81×81 场（同一量）", fontsize=FS_LABEL - 2)
    for ax in (axMap1, axMap2):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    # shared colorbar to the right of the map pair; scatter and imshow share
    # vmin=0/vmax=1/cmap so im2 reads the common scale.
    cb = fig.colorbar(im2, ax=[axMap1, axMap2], location="right", shrink=0.78,
                      pad=0.015, aspect=26)
    cb.set_label("interictal rank (0=source → 1=sink)", fontsize=FS_LABEL - 4)

    # (b) 2D-plane vs native-3D σ-normalized inter-contact distances
    _dist_2d_vs_3d_panel(axDist, ctx)

    # (c) cohort equivalence: R2 vs R3 (grid) and R2b vs R2_nm (native-3D)
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
                     "R2 obs (in-plane smoothed contact)",
                     "R3 obs (gridded field)",
                     "网格 vs 触点 (R2 vs R3)：\n未见可分辨增益 (CI 宽于 ±SESOI, 非零)", ann1)
    axR1.legend(fontsize=FS_TICK - 5, frameon=False, loc="lower right")

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
    ann2 = (f"r2b−r2nm median={rb_med:+.4f}\nCI=[{rb_ci[0]:+.3f},{rb_ci[1]:+.3f}]  n={len(nmv)}"
            if rb_med is not None else "r2b_minus_r2nm: n/a")
    _scatter_vs_diag(axR2, nmv, bv, "#1F4E9C",
                     "R2_nm obs (2D-plane)", "R2b obs (native-3D mm)",
                     "native-3D vs 2D-plane (R2_nm vs R2b)：\n等价通过 (CI 落在 ±SESOI 内)", ann2)

    fig.suptitle("上：触点加权 ≈ 铺网格场（同一形状）　·　"
                 "下：2D 平面保留三维邻居结构 → 网格与 native-3D 均无可分辨增益 / 等价通过",
                 fontsize=FS_TITLE - 2, y=0.975)
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
    p2 = save_fig(fig2(ctx), fig_dir / "fig2_rank_comparison.png")
    p2s = save_fig(fig2_sup(cohort_summary, args.subject), fig_dir / "fig2_sup_maxab_vs_null.png")
    p3 = save_fig(fig3(ctx, cohort_summary, r2b_summary), fig_dir / "fig3_vs_field.png")
    for p in (p1, p2, p2s, p3):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
