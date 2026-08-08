"""The flexible field: where it sits, and what it produces.

Four columns on one measuring stick. The map and the fit are scored by the same
distance to the patient's per-event profile distribution, so the field's own
iso-region can be laid straight over the map of what a fixed blob achieves at
each position.

Columns two to four reuse the accepted subject-SNN renderers, so this figure and
the published hand-placed one can be read side by side. The hand-placed cores
appear nowhere as a mechanism marker: this run never had them.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.getcwd())
from scripts.paper_figures.plot_fig_subject_snn import (  # noqa: E402
    _plot_event, _plot_interictal_sample_readout, _registered_axis_display,
    _shaft)
from scripts.plot_topic4_core_field_learned_vs_manual import (  # noqa: E402
    _field_extent, _ignition_xy)
from src.topic4_core_field import project_to_budget  # noqa: E402
from src.topic4_core_field_stage3 import (K_COMPONENTS, params_to_q,  # noqa: E402
                                          unpack)

STAGE2 = "results/topic4_sef_hfo/data_driven_core_field"
OUT = "results/topic4_sef_hfo/data_driven_core_field_stage3"
RUN = "results/topic4_sef_hfo/field_swap_subject_snn"
TAG = "epilepsiae_1146_stage3_flexible_field_s5"

C_FIELD = "#c0392b"
C_GREY = "#c9c9c9"


def _field_density(pos_xy, h, mass=0.90, cell=0.35):
    """Density grid and iso-level enclosing the requested field mass."""
    xy = np.asarray(pos_xy, float)
    xe = np.arange(xy[:, 0].min(), xy[:, 0].max() + cell, cell)
    ye = np.arange(xy[:, 1].min(), xy[:, 1].max() + cell, cell)
    total = np.histogram2d(xy[:, 0], xy[:, 1], bins=[xe, ye], weights=h)[0]
    count = np.histogram2d(xy[:, 0], xy[:, 1], bins=[xe, ye])[0]
    density = np.where(count > 0, total / np.maximum(count, 1), 0.0)
    order = np.argsort(density.ravel())[::-1]
    keep = np.cumsum(total.ravel()[order]) <= float(mass) * total.sum()
    level = density.ravel()[order][keep].min() if keep.any() else density.max()
    return (0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:]),
            density, float(level))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--out", default=os.path.join(OUT, "figures"))
    a = ap.parse_args()

    cfg = json.load(open(f"{STAGE2}/config/stage_config.json"))
    sweep = json.load(open(f"{OUT}/config/sweep_config.json"))
    r1 = json.load(open(f"{OUT}/profile_round1.json"))
    conf = json.load(open(f"{OUT}/fit/confirmation_K3_r0.json"))
    ro = json.load(open(f"{RUN}/readout_{a.tag}.json"))
    fd = np.load(f"{RUN}/figdata_{a.tag}.npz", allow_pickle=True)

    names = [str(x) for x in fd["names"]]
    shafts = sorted({_shaft(n) for n in names})
    display = _registered_axis_display(fd)
    posE = np.asarray(fd["posE"], float)
    theta = np.asarray(conf["best_theta"], float)
    h, _ = project_to_budget(
        params_to_q(theta, posE, K_COMPONENTS, float(cfg["engine"]["L"])),
        float(cfg["N_core_manual"]))

    fig = plt.figure(figsize=(21.6, 5.15), facecolor="white")
    outer = gridspec.GridSpec(1, 3, width_ratios=[1.90, 2.0, 2.75],
                              left=0.045, right=0.99, bottom=0.245, top=0.875,
                              wspace=0.17)
    maps = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, 1],
                                            wspace=0.045)
    axA = fig.add_subplot(outer[0, 0], projection="3d")
    axF, axR = fig.add_subplot(maps[0, 0]), fig.add_subplot(maps[0, 1])
    axRO = fig.add_subplot(outer[0, 2])

    # ---- A: the same distance, everywhere a fixed blob could sit -----------
    by_center = {}
    for row in r1["arms"]:
        if row["arm"].startswith("single_blob@"):
            x, y = row["arm"].split("@(")[1].rstrip(")").split(",")
            by_center[(round(float(x), 1), round(float(y), 1))] = row["distance_train"]
    n = int(sweep["grid"]["n"])
    lo, hi = sweep["grid"]["lo"], sweep["grid"]["hi"]
    step = (hi - lo) / (n - 1)
    edges = np.linspace(lo - step / 2, hi + step / 2, n + 1)
    vals = np.full((n, n), np.nan)
    for i, c in enumerate(sweep["grid"]["centers"]):
        r, col = divmod(i, n)
        vals[r, col] = by_center.get((round(c[0], 1), round(c[1], 1)), np.nan)

    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    cm = plt.get_cmap("magma_r")
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    # A discrete measured landscape. Missing cells are not interpolated: they
    # remain grey floor tiles, so perspective cannot turn "no readable events"
    # into an invented ridge or basin.
    z = 1.0 - vals
    z0 = 0.0
    for r in range(n):
        for c in range(n):
            valid_cell = np.isfinite(vals[r, c])
            face = cm(norm(vals[r, c])) if valid_cell else C_GREY
            height = float(z[r, c]) if valid_cell else 0.008
            axA.bar3d(edges[c] + 0.08 * step, edges[r] + 0.08 * step, z0,
                      0.84 * step, 0.84 * step, height,
                      color=face, alpha=0.96 if valid_cell else 0.55,
                      edgecolor="white", linewidth=0.35, shade=True)
    xc, yc, field_density, lvl = _field_density(posE, h)
    axA.contour(xc, yc, field_density.T, levels=[lvl], zdir="z", offset=z0 + 0.012,
                colors=[C_FIELD], linewidths=2.2)
    contacts = np.asarray(fd["contacts"], float)
    axA.scatter(contacts[:, 0], contacts[:, 1], np.full(len(contacts), z0 + 0.018),
                s=22, marker="v", facecolor="white", edgecolor="black",
                linewidth=0.65, depthshade=False)
    axA.set_xlim(edges[0], edges[-1]); axA.set_ylim(edges[0], edges[-1])
    axA.set_zlim(z0, max(0.46, float(np.nanmax(z)) * 1.12))
    axA.set_xlabel("sheet x (mm)", fontsize=8.5, labelpad=3)
    axA.set_ylabel("sheet y (mm)", fontsize=8.5, labelpad=3)
    axA.tick_params(labelsize=7.5, pad=0)
    axA.view_init(elev=31, azim=-57)
    axA.set_box_aspect((1.0, 1.0, 0.58))
    for axis in (axA.xaxis, axA.yaxis, axA.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor("0.85")
    axA.set_title("fixed-blob match landscape", fontsize=11.5, fontweight="bold",
                  color="0.2", pad=8)
    axA.text2D(0.5, 0.965, "bar height = 1 - distance", transform=axA.transAxes,
               ha="center", va="top", fontsize=7.6, color="0.35")
    box = axA.get_position()
    cax = fig.add_axes([box.x0 - 0.018, box.y0 + 0.24 * box.height,
                        0.009, 0.48 * box.height])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cm), cax=cax)
    cb.set_label("distance to patient", fontsize=8.0, labelpad=3)
    cb.ax.tick_params(labelsize=7.5)
    axA.legend(handles=[
        Line2D([], [], color=C_FIELD, lw=2.2, label="learned field: 90% mass"),
        Patch(facecolor=C_GREY, edgecolor="white",
              label="too few events there to score"),
        Line2D([], [], marker="v", ls="none", mfc="white", mec="black", ms=6,
               label="recording contacts")],
        fontsize=7.2, loc="upper center", bbox_to_anchor=(0.50, -0.12), ncol=3,
        frameon=False, borderpad=0.1, handlelength=1.4, columnspacing=0.9)
    # the floor must use the same event count: a total-variation distance
    # between histograms is biased upward at small samples, so the all-events
    # figure of 0.03 is not what 80 model events should be compared against
    # the producer renamed this when the floor's structure was corrected; accept
    # either so a figure can still be drawn from an artifact of either vintage
    fl = conf.get("floor_structure_matched") or conf["floor_matched_to_n_events"]
    axA.text2D(0.5, -0.25,
             f"learned field {conf['confirm_distance_train']:.2f} "
             f"(held-out recordings {conf['confirm_distance_heldout']:.2f})   ·   "
             f"best fixed blob {min(by_center.values()):.2f}   ·   "
             f"patient vs itself at the same {fl.get('model_n', fl.get('n'))} events "
             f"{fl['median']:.2f} [{fl['p05']:.2f}-{fl['p95']:.2f}]",
             transform=axA.transAxes, ha="center", va="top", fontsize=8.2,
             color="0.25")

    # ---- B/C: the two directions this field produces -----------------------
    fd_nc = {k: fd[k] for k in fd.files}
    fd_nc["foci"] = np.full((2, 2), np.nan)          # this run had no placed cores
    rep_f, rep_r = fd["rep_fwd"].item(), fd["rep_rev"].item()
    _plot_event(axF, fd_nc, rep_f, "model forward", shafts, [], [],
                source_index=0, normalize_color=True, display=display,
                formal=True, show_ylabel=True)
    mappable = _plot_event(axR, fd_nc, rep_r, "model reverse", shafts, [], [],
                           source_index=1, normalize_color=True,
                           display=display, formal=True, show_ylabel=False)
    lab = axF.get_xticklabels()
    if lab:
        lab[-1].set_visible(False)
    for ax_, rep in ((axF, rep_f), (axR, rep_r)):
        _field_extent(ax_, posE, h, display)
        ig = _ignition_xy(rep, posE, display)
        if ig is not None:
            ax_.scatter([ig[0]], [ig[1]], marker="*", s=170, c="black",
                        ec="white", lw=0.9, zorder=8)
    if mappable is not None:
        cax = axR.inset_axes([1.035, 0.0, 0.052, 1.0])
        cbe = fig.colorbar(mappable, cax=cax)
        cbe.set_ticks([0.0, 1.0]); cbe.set_ticklabels(["early", "late"])
        cbe.ax.set_title("relative\nfiring onset", fontsize=10.0, pad=5.0)
        cbe.ax.tick_params(labelsize=9.5, length=2.8)
    axF.legend(handles=[
        Patch(facecolor=C_FIELD, alpha=0.28, edgecolor=C_FIELD, lw=1.4,
              label="the learned field"),
        Line2D([], [], marker="*", ls="none", mfc="black", mec="white", ms=11,
               label="where the event started")],
        frameon=True, framealpha=0.9, edgecolor="0.85", fontsize=8.2,
        loc="lower left", borderpad=0.35, handlelength=1.3)

    # ---- D: the readout of the same run ------------------------------------
    stats = _plot_interictal_sample_readout(
        axRO, fd, ro.get("readout_window_events", ro["events"]), names, shafts,
        window_ms=1200.0)
    axRO.set_title("model virtual-SEEG", fontsize=12, fontweight="bold",
                   color="0.2", pad=7, loc="left")

    fig.canvas.draw()
    box_r, box_ro = axR.get_position(), axRO.get_position()
    axRO.set_position([box_ro.x0, box_r.y0, box_ro.width, box_r.height])

    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, "stage3_flexible_field")
    fig.savefig(stem + ".png", dpi=200, facecolor="white",
                bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem + ".pdf", facecolor="white", bbox_inches="tight",
                pad_inches=0.03)
    plt.close(fig)

    json.dump(dict(
        figure="stage3_flexible_field", plotting_only=True, learned_tag=a.tag,
        field=dict(theta=conf["best_theta"], K=K_COMPONENTS,
                   components=[{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                for k, v in comp.items()}
                               for comp in unpack(theta, K_COMPONENTS,
                                                  float(cfg["engine"]["L"]))],
                   iso_level=lvl),
        distance=dict(fit=conf["fit_value"],
                      confirmed_independent_networks=conf["confirm_distance_train"],
                      confirmed_held_out_recordings=conf["confirm_distance_heldout"],
                      best_fixed_blob=min(by_center.values()),
                      floor_structure_matched=fl,
                      patient_train_vs_heldout_full=conf["patient_train_vs_heldout_full"],
                      winners_curse=conf["winners_curse"],
                      **conf["reference"]),
        landscape=dict(rendering="discrete measured 7x7 bars; no interpolation",
                       z="1 - first-round one-dimensional distance",
                       n_valid=int(np.isfinite(vals).sum()),
                       n_missing=int((~np.isfinite(vals)).sum()),
                       learned_field_overlay="90% field-mass contour projected to z floor",
                       view=dict(elevation_deg=31, azimuth_deg=-57)),
        run=dict(seed=ro["seed"], n_events=ro["n_events"], n_clean=ro["n_clean"],
                 dir_forward=ro["dir_forward"], dir_reverse=ro["dir_reverse"],
                 bidirectional=ro["bidirectional"]),
        readout_events=stats,
        component_mass_fraction=[
            float(c["weight"] * c["sigma_par"] * c["sigma_perp"]) /
            sum(float(d["weight"] * d["sigma_par"] * d["sigma_perp"])
                for d in unpack(theta, K_COMPONENTS, float(cfg["engine"]["L"])))
            for c in unpack(theta, K_COMPONENTS, float(cfg["engine"]["L"]))],
        caveats=[
            "the objective run was a one-dimensional marginal, not the "
            "two-dimensional joint distance frozen in spec 9.3; the marginal is "
            "satisfiable by a single mid-array generator and the fitted field's "
            "two directions correlate at +0.65 rather than being opposite",
            "one restart at K=3 only, so the field's shape is not yet "
            "identifiable; the search did not converge either, with sigma "
            "growing after generation five",
            "axial mass within 2 mm is 0.686, below the 0.70 threshold, so "
            "AXIS_REDISCOVERED is NOT met",
            "component weight is not component mass: the widest component "
            "carries 0.04 of the weight but 15.5% of the mass, and the compact "
            "one near the sink core carries 7.0%",
            "one network for the event panels; distances come from six "
            "independent networks and held-out patient recordings",
        ]),
        open(stem + "_metadata.json", "w"), indent=1)
    print(f"wrote {stem}.png / .pdf / _metadata.json")
    print(f"  run: {ro['n_events']} events, fwd/rev {ro['dir_forward']}/{ro['dir_reverse']}")


if __name__ == "__main__":
    main()
