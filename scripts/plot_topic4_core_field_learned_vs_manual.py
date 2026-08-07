"""Diagnostic: what spatial pathology field did the search actually learn?

Four independent questions, one panel each:
  A  along the propagation axis, does the learned field have two lobes like the
     hand-placed one, or one?
  B  which shape did the search actually chase, generation by generation?
  C  in the objective's own landscape, does a two-lobed field score better?
  D  in the sheet plane, what does the learned field look like next to the
     hand-placed cores and the recording contacts?

Visual diagnostic, not a mechanism claim figure (figure_style_guide Topic 4,
diagnostic variant). Substrate colouring uses plasma, matching the mechanism
panel convention.
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, os.getcwd())
from src.topic4_core_field import (axis_coords, manual_mask, project_to_budget,
                                   two_core_q)
from src.topic4_core_field_runner import cache_key, connectivity_config
from src.topic4_core_field_scoring import candidate_key
from src.topic4_core_field_stage2 import (LOG_RHO_LIMIT, _unpack, params_to_h,
                                          uniform_theta)

OUT = "results/topic4_sef_hfo/data_driven_core_field"
FIGDIR = os.path.join(OUT, "figures")
POP = 10          # CMA-ES population size, for slicing history into generations
BIN_MM = 1.0

C_MANUAL = "#c0392b"      # hand-placed two-core target
C_LEARNED = "#1f6f8b"     # learned field
C_UNIFORM = "#8d8d8d"     # flat corridor reference
C_WIDTH = "#e67e22"       # transverse width, the shape the search did chase


def load():
    cfg = json.load(open(os.path.join(OUT, "config", "stage_config.json")))
    e = cfg["engine"]
    sys.path.insert(0, os.path.join("src", "snn_engine"))
    from params import Params

    from scripts.run_topic4_core_field_stage2 import _load_cmrun, _placement
    cmrun = _load_cmrun()
    cmrun.KDIR, cmrun.PART_MIN = int(e["k_dir"]), 2 * int(e["k_dir"]) + 1
    reg = _placement(cfg)
    p = Params(g=e["g"], L=e["L"], density=e["density"], T=cfg["duration_ms"],
               dt=e["dt"], nu_ext_ratio=cmrun.DRIVE, seed=1)
    key = cache_key(connectivity_config(p, reg["theta_deg"], e["AR"]))
    payload = pickle.load(open(os.path.join(OUT, "network_cache", f"{key}.pkl"), "rb"))
    posE = payload["net"]["pos"][:int(payload["NE"])]
    s, r = axis_coords(posE, reg["center"], reg["axis_unit_vec"])
    geom = dict(sep=float(np.linalg.norm(reg["sink_centroid"] - reg["source_centroid"])),
                s_support=(float(s.min()) + cfg["field"]["AXIAL_MARGIN"],
                           float(s.max()) - cfg["field"]["AXIAL_MARGIN"]),
                M=cfg["field"]["M"], sigma_perp=e["core_r"],
                shift_mm=cfg["field"]["SHIFT_MM"])
    hist = json.load(open(os.path.join(OUT, "stage2_optimization",
                                       "checkpoint.json")))["history"]
    return cfg, e, reg, posE, s, r, geom, hist


def main():
    cfg, e, reg, posE, s, r, geom, hist = load()
    N = float(cfg["N_core_manual"])
    sep, core_r = geom["sep"], float(e["core_r"])

    edges = np.arange(np.floor(s.min()), np.ceil(s.max()) + BIN_MM, BIN_MM)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    binid = np.clip(np.digitize(s, edges) - 1, 0, len(ctr) - 1)

    def profile(h):
        out = np.zeros(len(ctr))
        np.add.at(out, binid, h)
        return out

    def two_lobe_index(pr):
        """How empty is the trough between the two hand-placed core centres.

        1 = the two lobes are fully separated; 0 = a single lobe with no dip."""
        lo, hi = np.abs(ctr + sep / 2).argmin(), np.abs(ctr - sep / 2).argmin()
        a, b = min(lo, hi), max(lo, hi)
        pk = min(pr[a], pr[b])
        return float(1.0 - pr[a + 1:b].min() / pk) if pk > 0 else np.nan

    h_hard = manual_mask(posE, reg["source_centroid"],
                         reg["sink_centroid"], core_r).astype(float)
    h_smooth, _ = project_to_budget(two_core_q(s, r, sep), N)
    h_uni = params_to_h(uniform_theta(geom["M"]), s, r, geom, N)
    best = max(hist, key=lambda x: candidate_key(x["n_dir"], x["S_rank"]))
    h_learn = params_to_h(np.asarray(best["theta"], float), s, r, geom, N)

    pr_hard, pr_smooth = profile(h_hard), profile(h_smooth)
    pr_uni, pr_learn = profile(h_uni), profile(h_learn)

    bm_all = np.array([two_lobe_index(profile(params_to_h(
        np.asarray(x["theta"], float), s, r, geom, N))) for x in hist])
    nd_all = np.array([x["n_dir"] for x in hist])
    sr_all = np.array([x["S_rank"] for x in hist], float)

    def width_mm(theta):
        """Transverse extent of the field: sigma_perp / rho, in mm."""
        return float(geom["sigma_perp"]) / _unpack(theta, geom["M"])[1]

    gens = sorted({i // POP for i in range(len(hist))})
    gb = [max(hist[g * POP:(g + 1) * POP],
              key=lambda x: candidate_key(x["n_dir"], x["S_rank"])) for g in gens]
    bm_gen = np.array([two_lobe_index(profile(params_to_h(
        np.asarray(x["theta"], float), s, r, geom, N))) for x in gb])
    w_gen = np.array([width_mm(np.asarray(x["theta"], float)) for x in gb])
    w_floor = float(geom["sigma_perp"]) / np.exp(LOG_RHO_LIMIT)

    # ---------------------------------------------------------------- figure
    fig = plt.figure(figsize=(15.0, 7.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.0, 1.0],
                          hspace=0.42, wspace=0.26,
                          left=0.065, right=0.975, top=0.90, bottom=0.085)
    axA, axC = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axB, axD = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # -- A: axial profile --------------------------------------------------
    for c in (-sep / 2, sep / 2):
        axA.axvspan(c - core_r, c + core_r, color=C_MANUAL, alpha=0.07, zorder=0)
    axA.plot(ctr, pr_hard, color=C_MANUAL, lw=2.4, label="hand-placed cores")
    axA.plot(ctr, pr_smooth, color=C_MANUAL, lw=1.6, ls="--", alpha=0.8,
             label="hand-placed, soft edge")
    axA.plot(ctr, pr_uni, color=C_UNIFORM, lw=1.8, ls=":", label="flat corridor")
    axA.plot(ctr, pr_learn, color=C_LEARNED, lw=2.6, label="learned from data")
    axA.set_xlim(ctr.min(), ctr.max())
    axA.set_ylim(0, max(pr_hard.max(), pr_learn.max()) * 1.30)
    axA.set_xlabel("position along the propagation axis (mm)")
    axA.set_ylabel(f"pathological cells per {BIN_MM:.0f} mm")
    axA.set_title("A   where the pathology sits along the axis", loc="left",
                  fontweight="bold", fontsize=11)
    axA.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2,
               handlelength=1.8, columnspacing=1.2)
    axA.text(0.985, 0.965,
             f"gap between the two lobes\nhand-placed {two_lobe_index(pr_hard):.2f}"
             f"   ·   learned {two_lobe_index(pr_learn):.2f}",
             transform=axA.transAxes, va="top", ha="right", fontsize=8.5,
             color="0.25")

    # -- B: which shape did the search chase? ------------------------------
    axB.axhline(two_lobe_index(pr_hard), color=C_MANUAL, lw=1.6, zorder=1)
    axB.text(0.2, two_lobe_index(pr_hard) - 0.03,
             "hand-placed: the two lobes are fully separated",
             ha="left", va="top", fontsize=8.5, color=C_MANUAL)
    axB.plot(gens, bm_gen, "-o", color=C_LEARNED, lw=1.9, ms=4.5, zorder=3)
    axB.set_xlim(-0.4, max(gens) + 0.4)
    axB.set_ylim(-0.04, 1.10)
    axB.set_xlabel("optimisation generation")
    axB.set_ylabel("gap between the two lobes", color=C_LEARNED)
    axB.tick_params(axis="y", colors=C_LEARNED)

    axB2 = axB.twinx()
    axB2.axhline(core_r, color=C_MANUAL, lw=1.6, ls="--", zorder=1)
    axB2.text(0.2, core_r - 0.045, "hand-placed: 1.5 mm across the axis",
              ha="left", va="top", fontsize=8.5, color=C_MANUAL)
    axB2.axhline(w_floor, color="0.55", lw=1.2, ls=":", zorder=1)
    axB2.annotate("narrowest the search was allowed",
                  xy=(max(gens) * 0.72, w_floor), xytext=(max(gens) * 0.55, 0.95),
                  fontsize=8, color="0.45", ha="left", va="bottom",
                  arrowprops=dict(arrowstyle="-", color="0.6", lw=0.9,
                                  shrinkA=2, shrinkB=2))
    axB2.plot(gens, w_gen, "-s", color=C_WIDTH, lw=1.9, ms=4.0, zorder=3)
    axB2.set_ylim(0, core_r * 1.35)
    axB2.set_ylabel("width across the axis (mm)", color=C_WIDTH)
    axB2.tick_params(axis="y", colors=C_WIDTH)
    axB.set_title("B   both shape choices were settled by the first generation",
                  loc="left", fontweight="bold", fontsize=11)
    axB.legend(handles=[
        Line2D([], [], color=C_LEARNED, marker="o", ms=4.5, lw=1.9,
               label="gap between the two lobes  (left axis)"),
        Line2D([], [], color=C_WIDTH, marker="s", ms=4.0, lw=1.9,
               label="width across the axis  (right axis)")],
        frameon=False, fontsize=8.5, loc="center left", ncol=1,
        bbox_to_anchor=(0.02, 0.50))

    # -- C: does the objective pay for two lobes? --------------------------
    ok = np.isfinite(sr_all)
    m2 = ok & (nd_all == 2)
    axC.scatter(bm_all[m2], sr_all[m2], s=28, marker="s", facecolor=C_LEARNED,
                edgecolor="white", linewidth=0.4, alpha=0.8, zorder=2,
                label="a candidate field")
    axC.scatter([two_lobe_index(pr_learn)], [best["S_rank"]], s=150, marker="*",
                color=C_WIDTH, edgecolor="black", linewidth=0.5, zorder=4,
                label="the one the search kept")
    axC.axvline(two_lobe_index(pr_hard), color=C_MANUAL, lw=1.8, zorder=1)
    rho = spearmanr(bm_all[m2], sr_all[m2])
    axC.set_xlim(-0.06, 1.22)
    lo, hi = sr_all[m2].min(), sr_all[m2].max()
    axC.set_ylim(lo - 0.10 * (hi - lo), hi + 0.28 * (hi - lo))
    axC.text(two_lobe_index(pr_hard) + 0.03, lo - 0.06 * (hi - lo),
             " hand-placed cores", rotation=90, va="bottom", ha="left",
             fontsize=8.5, color=C_MANUAL)
    axC.set_xlabel("gap between the two lobes of the candidate field")
    axC.set_ylabel("template match score")
    axC.set_title("C   a two-lobed field earns no better score", loc="left",
                  fontweight="bold", fontsize=11)
    axC.legend(frameon=False, fontsize=8.5, loc="lower left")
    axC.text(0.36, 0.975,
             f"rho = {rho.correlation:+.2f},  p = {rho.pvalue:.2f}   "
             f"(n = {m2.sum()} fields)\nthe search did reach two-lobed fields, "
             f"and dropped them",
             transform=axC.transAxes, ha="left", va="top", fontsize=8.5,
             color="0.25")

    # -- D: the learned field in the plane ---------------------------------
    xe = np.arange(np.floor(s.min()), np.ceil(s.max()) + 0.5, 0.5)
    ye = np.arange(-12.0, 12.001, 0.5)
    cnt = np.histogram2d(s, r, bins=[xe, ye])[0]
    tot = np.histogram2d(s, r, bins=[xe, ye], weights=h_learn)[0]
    img = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    im = axD.imshow(img.T, origin="lower", cmap="plasma",
                    extent=[xe[0], xe[-1], ye[0], ye[-1]], aspect="equal",
                    vmin=0, vmax=float(np.nanmax(img)), interpolation="nearest")
    th = np.linspace(0, 2 * np.pi, 200)
    for c in (-sep / 2, sep / 2):
        axD.plot(c + core_r * np.cos(th), core_r * np.sin(th),
                 color="white", lw=2.6, zorder=3)
        axD.plot(c + core_r * np.cos(th), core_r * np.sin(th),
                 color=C_MANUAL, lw=1.6, zorder=4)
    cs, cr = axis_coords(np.asarray(reg["montage_sheet"].contacts, float),
                         reg["center"], reg["axis_unit_vec"])
    axD.scatter(cs, cr, s=44, marker="v", facecolor="white",
                edgecolor="black", linewidth=0.9, zorder=5)
    axD.set_xlim(ctr.min(), ctr.max())
    axD.set_ylim(-12, 12)
    axD.set_xlabel("position along the propagation axis (mm)")
    axD.set_ylabel("distance from the axis (mm)")
    axD.set_title("D   the learned field in the sheet", loc="left",
                  fontweight="bold", fontsize=11)
    axD.legend(handles=[
        Line2D([], [], color=C_MANUAL, lw=1.8, label="hand-placed cores"),
        Line2D([], [], marker="v", ls="none", mfc="white", mec="black",
               ms=7, label="recording contacts")],
        fontsize=8.5, loc="upper left", framealpha=0.92, edgecolor="0.8",
        borderpad=0.5)
    cb = fig.colorbar(im, ax=axD, fraction=0.030, pad=0.02)
    cb.set_label("chance a cell is pathological", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle("Fitting only the contact orderings gives a thin filament along "
                 "the whole axis, not two separated cores",
                 fontsize=13, fontweight="bold", x=0.065, ha="left", y=0.965)

    os.makedirs(FIGDIR, exist_ok=True)
    base = os.path.join(FIGDIR, "learned_vs_manual_core_field")
    fig.savefig(base + ".png", dpi=200)
    fig.savefig(base + ".pdf")
    plt.close(fig)

    u = mannwhitneyu(bm_all[nd_all == 2], bm_all[nd_all == 1])
    json.dump(dict(
        n_candidates=len(hist), pop_size=POP, n_generations=len(gens),
        two_lobe_index=dict(
            manual_hard=two_lobe_index(pr_hard),
            manual_smooth=two_lobe_index(pr_smooth),
            uniform_axial=two_lobe_index(pr_uni),
            learned_best=two_lobe_index(pr_learn),
            candidates_median=float(np.median(bm_all)),
            candidates_max=float(np.nanmax(bm_all))),
        n_dir_counts={str(k): int((nd_all == k).sum()) for k in sorted(set(nd_all))},
        two_lobe_by_n_dir_p=float(u.pvalue),
        match_vs_two_lobe_within_bidirectional=dict(
            rho=float(rho.correlation), p=float(rho.pvalue), n=int(m2.sum())),
        transverse_width_mm=dict(
            manual=core_r, learned_best=float(w_gen[-1]),
            allowed_minimum=float(w_floor),
            candidates_median=float(np.median(
                [width_mm(np.asarray(x["theta"], float)) for x in hist]))),
        best_theta=best["theta"], best_S_rank=best["S_rank"],
        config_checksum=cfg["checksum"]),
        open(base + "_metadata.json", "w"), indent=1)
    print(f"wrote {base}.png / .pdf / _metadata.json")


if __name__ == "__main__":
    main()
