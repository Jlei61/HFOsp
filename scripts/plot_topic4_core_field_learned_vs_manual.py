"""What the learned pathology field is, and what it produces.

Four columns, each a separate question:

  A  where the pathology sits along the propagation axis -- learned field
     against the hand-placed cores it is meant to replace
  B  the model's forward event on the sheet
  C  the model's reverse event on the sheet
  D  the virtual-SEEG readout of the same run

B/C/D are drawn by the accepted subject-SNN renderers so the layout, colour
scale and registration are identical to the hand-placed figure and the two can
be read side by side.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.getcwd())
from scripts.paper_figures.plot_fig_subject_snn import (  # noqa: E402
    _plot_event, _plot_interictal_sample_readout, _registered_axis_display,
    _shaft)
from src.topic4_core_field import axis_coords, manual_mask  # noqa: E402
from src.topic4_core_field_scoring import candidate_key  # noqa: E402
from src.topic4_core_field_stage2 import (params_to_h,  # noqa: E402
                                          uniform_theta)

OUT = "results/topic4_sef_hfo/data_driven_core_field"
RUN = "results/topic4_sef_hfo/field_swap_subject_snn"
LEARNED_TAG = "epilepsiae_1146_learned_core_field_cr1p5_s5"
BIN_MM = 1.0

C_MANUAL = "#c0392b"
C_LEARNED = "#1f6f8b"
C_UNIFORM = "#9a9a9a"


def _axial_profiles(cfg, fd):
    """Axial mass profile of the hand-placed, flat and learned fields.

    Positions come from the rendered run itself, so the learned curve is the
    field that was actually simulated, not a reconstruction on another network.
    """
    e = cfg["engine"]
    reg_raw = fd["reg"].item()
    src = np.asarray(reg_raw["source_centroid"], float)
    snk = np.asarray(reg_raw["sink_centroid"], float)
    center = np.asarray(reg_raw["center"], float)
    axis_u = (snk - src) / np.linalg.norm(snk - src)
    posE = np.asarray(fd["posE"], float)
    s, r = axis_coords(posE, center, axis_u)
    sep = float(np.linalg.norm(snk - src))
    geom = dict(sep=sep, M=cfg["field"]["M"], sigma_perp=e["core_r"],
                s_support=(float(s.min()) + cfg["field"]["AXIAL_MARGIN"],
                           float(s.max()) - cfg["field"]["AXIAL_MARGIN"]),
                shift_mm=cfg["field"]["SHIFT_MM"])
    N = float(cfg["N_core_manual"])

    edges = np.arange(np.floor(s.min()), np.ceil(s.max()) + BIN_MM, BIN_MM)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    binid = np.clip(np.digitize(s, edges) - 1, 0, len(ctr) - 1)

    def prof(h):
        out = np.zeros(len(ctr))
        np.add.at(out, binid, h)
        return out

    hist = json.load(open(os.path.join(OUT, "stage2_optimization",
                                       "checkpoint.json")))["history"]
    best = max(hist, key=lambda x: candidate_key(x["n_dir"], x["S_rank"]))
    h_hard = manual_mask(posE, src, snk, e["core_r"]).astype(float)
    h_flat = params_to_h(uniform_theta(geom["M"]), s, r, geom, N)
    h_learn = params_to_h(np.asarray(best["theta"], float), s, r, geom, N)

    def gap(pr):
        lo, hi = np.abs(ctr + sep / 2).argmin(), np.abs(ctr - sep / 2).argmin()
        a, b = min(lo, hi), max(lo, hi)
        pk = min(pr[a], pr[b])
        return float(1.0 - pr[a + 1:b].min() / pk) if pk > 0 else np.nan

    p_hard, p_flat, p_learn = prof(h_hard), prof(h_flat), prof(h_learn)
    return dict(ctr=ctr, sep=sep, core_r=float(e["core_r"]),
                hard=p_hard, flat=p_flat, learned=p_learn,
                gap_hard=gap(p_hard), gap_learned=gap(p_learn),
                theta=best["theta"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=LEARNED_TAG)
    ap.add_argument("--out", default=os.path.join(OUT, "figures"))
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(OUT, "config", "stage_config.json")))
    ro = json.load(open(os.path.join(RUN, f"readout_{a.tag}.json")))
    fd = np.load(os.path.join(RUN, f"figdata_{a.tag}.npz"), allow_pickle=True)
    pf = _axial_profiles(cfg, fd)
    reg = fd["reg"].item()
    core_a, core_b = list(reg["source_names"]), list(reg["sink_names"])
    names = [str(x) for x in fd["names"]]
    shafts = sorted({_shaft(n) for n in names})
    display = _registered_axis_display(fd)
    rep_f, rep_r = fd["rep_fwd"].item(), fd["rep_rev"].item()

    fig = plt.figure(figsize=(18.8, 4.55), facecolor="white")
    outer = gridspec.GridSpec(1, 3, width_ratios=[1.15, 2.0, 2.75],
                              left=0.055, right=0.988, bottom=0.155, top=0.885,
                              wspace=0.185)
    maps = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, 1],
                                            wspace=0.045)
    axA = fig.add_subplot(outer[0, 0])
    axF, axR = fig.add_subplot(maps[0, 0]), fig.add_subplot(maps[0, 1])
    axRO = fig.add_subplot(outer[0, 2])

    # -- A: where the pathology sits ---------------------------------------
    for c in (-pf["sep"] / 2, pf["sep"] / 2):
        axA.axvspan(c - pf["core_r"], c + pf["core_r"], color=C_MANUAL,
                    alpha=0.08, lw=0, zorder=0)
    axA.plot(pf["ctr"], pf["hard"], color=C_MANUAL, lw=2.2,
             label="hand-placed cores")
    axA.plot(pf["ctr"], pf["flat"], color=C_UNIFORM, lw=1.6, ls=":",
             label="flat corridor")
    axA.plot(pf["ctr"], pf["learned"], color=C_LEARNED, lw=2.4,
             label="learned field")
    axA.set_xlim(pf["ctr"].min(), pf["ctr"].max())
    axA.set_ylim(0, pf["hard"].max() * 1.22)
    axA.set_xlabel("TA–TB shared axis (mm)", fontsize=10.5)
    axA.set_ylabel("pathological cells per mm", fontsize=10.5)
    axA.set_title("pathology along the axis", fontsize=12, fontweight="bold",
                  color="0.2", pad=8)
    axA.legend(frameon=False, fontsize=9, loc="upper center", ncol=1,
               handlelength=1.5, borderpad=0.1, labelspacing=0.35)
    axA.tick_params(labelsize=9.5)
    for side in ("top", "right"):
        axA.spines[side].set_visible(False)

    # -- B/C: the two directions on the sheet ------------------------------
    _plot_event(axF, fd, rep_f, "model forward", shafts, core_a, core_b,
                source_index=0, normalize_color=True, display=display,
                formal=True, show_ylabel=True)
    mappable = _plot_event(axR, fd, rep_r, "model reverse", shafts, core_a,
                           core_b, source_index=1, normalize_color=True,
                           display=display, formal=True, show_ylabel=False)
    # the two maps are butted together, so the shared +10/-10 boundary label
    # would print twice and collide
    right_labels = axF.get_xticklabels()
    if right_labels:
        right_labels[-1].set_visible(False)
    if mappable is not None:
        cax = axR.inset_axes([1.035, 0.0, 0.052, 1.0])
        cb = fig.colorbar(mappable, cax=cax)
        cb.set_ticks([0.0, 1.0])
        cb.set_ticklabels(["early", "late"])
        cb.ax.set_title("relative\nfiring onset", fontsize=10.0, pad=5.0)
        cb.ax.tick_params(labelsize=9.5, length=2.8)

    # -- D: the readout of the same run ------------------------------------
    stats = _plot_interictal_sample_readout(
        axRO, fd, ro.get("readout_window_events", ro["events"]), names, shafts,
        window_ms=1200.0)

    fig.canvas.draw()
    box_r, box_ro = axR.get_position(), axRO.get_position()
    axRO.set_position([box_ro.x0, box_r.y0, box_ro.width, box_r.height])
    axA.set_position([axA.get_position().x0, box_r.y0,
                      axA.get_position().width, box_r.height])

    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, "learned_core_field_readout")
    fig.savefig(stem + ".png", dpi=200, facecolor="white",
                bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem + ".pdf", facecolor="white", bbox_inches="tight",
                pad_inches=0.03)
    plt.close(fig)

    json.dump(dict(
        figure="learned_core_field_readout", subject=cfg["subject"],
        learned_tag=a.tag, plotting_only=True,
        reference_hand_placed_figure=(
            "results/paper-ready-figure/fig4_subject_snn_e1146/figures/"
            "fig4_panel_b_bidirectional_readout"),
        field=dict(theta=pf["theta"], budget_cells=cfg["N_core_manual"],
                   gap_between_lobes_hand_placed=pf["gap_hard"],
                   gap_between_lobes_learned=pf["gap_learned"]),
        run=dict(seed=ro["seed"], k_dir=ro["k_dir"], placement=ro["placement"],
                 theta_deg=ro["theta_deg"], n_events=ro["n_events"],
                 n_clean=ro["n_clean"], dir_forward=ro["dir_forward"],
                 dir_reverse=ro["dir_reverse"],
                 bidirectional=ro["bidirectional"]),
        readout_events=stats, config_checksum=cfg["checksum"],
        claim_boundary=("single-run rendering of the learned field on one network; "
                        "not a cohort claim and not evidence that the learned field "
                        "reproduces the hand-placed mechanism")),
        open(stem + "_metadata.json", "w"), indent=1)
    print(f"wrote {stem}.png / .pdf / _metadata.json")
    print(f"  run: {ro['n_events']} events, directional fwd/rev = "
          f"{ro['dir_forward']}/{ro['dir_reverse']}, clean = {ro['n_clean']}")


if __name__ == "__main__":
    main()
