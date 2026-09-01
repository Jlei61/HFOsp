"""Leg A's maps.

Each cell carries its own denominator on the canvas. A cell where one of four
networks produced a readable direction is not the same evidence as one where
all four did, and a plain average would draw them identically. Cells with no
valid seed are grey rather than blank, so an unreadable cell is never mistaken
for a low-scoring one, and cells with one or two are hatched.

Nothing is interpolated: the sweep has 2.67 mm spacing and smoothing would
invent spatial precision the design does not have.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, os.getcwd())
from src.topic4_core_field_runner import _placement  # noqa: E402

OUT = "results/topic4_sef_hfo/data_driven_core_field_stage3"
STAGE2 = "results/topic4_sef_hfo/data_driven_core_field"
C_REF = "#c0392b"
C_GREY = "#c9c9c9"


def _geometry():
    cfg = json.load(open(os.path.join(STAGE2, "config", "stage_config.json")))
    reg = _placement(cfg)
    return dict(contacts=np.asarray(reg["montage_sheet"].contacts, float),
                src=np.asarray(reg["source_centroid"], float),
                snk=np.asarray(reg["sink_centroid"], float),
                center=np.asarray(reg["center"], float),
                core_r=float(cfg["engine"]["core_r"]), L=float(cfg["engine"]["L"]))


def _edges(cfg):
    n, lo, hi = int(cfg["grid"]["n"]), cfg["grid"]["lo"], cfg["grid"]["hi"]
    step = (hi - lo) / (n - 1)
    return np.linspace(lo - step / 2, hi + step / 2, n + 1)


def _draw_cells(ax, arr, denom, edges, *, cmap, vmin, vmax, show_counts=True,
                n_total=4):
    """One rectangle per cell -- no interpolation, and the denominator on top.

    `denom` must be the denominator OF THIS LAYER. The match score is undefined
    wherever a cell produced one direction, so it uses n_valid; direction and
    recruitment are defined for every completed run, so they use n_runs. Passing
    n_valid to those layers would grey out measured zeros as if unmeasured.
    """
    A = np.asarray(arr, float)
    V = np.asarray(denom, int)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cm = plt.get_cmap(cmap)
    for r in range(A.shape[0]):
        for c in range(A.shape[1]):
            x0, x1 = edges[c], edges[c + 1]
            y0, y1 = edges[r], edges[r + 1]
            unreadable = (not np.isfinite(A[r, c])) or V[r, c] == 0
            face = C_GREY if unreadable else cm(norm(A[r, c]))
            hatch = ("///" if (not unreadable and 0 < V[r, c] <= n_total // 2)
                     else None)
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=face,
                                   edgecolor="white", linewidth=0.5, hatch=hatch,
                                   zorder=1))
            if show_counts:
                # dark fills need light text or the denominator disappears --
                # and the denominator is the whole point of drawing it
                lum = (0.45 if unreadable else
                       0.299 * face[0] + 0.587 * face[1] + 0.114 * face[2])
                ax.text((x0 + x1) / 2, y0 + 0.12 * (y1 - y0),
                        f"{V[r, c]}/{n_total}",
                        ha="center", va="bottom", fontsize=6.8,
                        fontweight="bold" if not unreadable else "normal",
                        color="white" if lum < 0.5 else "0.2", zorder=3)
    return plt.cm.ScalarMappable(norm=norm, cmap=cm)


def _overlay(ax, g, edges, *, label_axis=True):
    u = (g["snk"] - g["src"]) / np.linalg.norm(g["snk"] - g["src"])
    t = np.array([-14.0, 14.0])
    line = g["center"][None, :] + t[:, None] * u[None, :]
    ax.plot(line[:, 0], line[:, 1], color="0.35", lw=1.1, ls="--", zorder=4)
    th = np.linspace(0, 2 * np.pi, 200)
    for c in (g["src"], g["snk"]):
        ax.plot(c[0] + g["core_r"] * np.cos(th), c[1] + g["core_r"] * np.sin(th),
                color=C_REF, lw=1.3, zorder=5)
    ax.scatter(g["contacts"][:, 0], g["contacts"][:, 1], s=34, marker="v",
               facecolor="white", edgecolor="black", linewidth=0.8, zorder=6)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(edges[0], edges[-1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("sheet x (mm)", fontsize=10)
    if label_axis:
        ax.set_ylabel("sheet y (mm)", fontsize=10)
    ax.tick_params(labelsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.out, "config", "sweep_config.json")))
    summ = json.load(open(os.path.join(a.out, "sweep_summary.json")))
    conf_path = os.path.join(a.out, "region_confirmation.json")
    conf = json.load(open(conf_path)) if os.path.exists(conf_path) else None
    g, edges = _geometry(), _edges(cfg)
    figdir = os.path.join(a.out, "figures")
    os.makedirs(figdir, exist_ok=True)

    legend = [
        Patch(facecolor=C_GREY, edgecolor="white", label="one direction only, match score undefined"),
        Patch(facecolor="white", edgecolor="0.4", hatch="///", label="1-2 of 4 networks contribute a score"),
        Line2D([], [], color=C_REF, lw=1.3, label="hand-placed cores (reference, not an input)"),
        Line2D([], [], color="0.35", lw=1.1, ls="--", label="frozen E->E anisotropy axis; no field-location constraint"),
        Line2D([], [], marker="v", ls="none", mfc="white", mec="black", ms=6, label="recording contacts"),
    ]

    # ---------------------------------------------------- main map + sensitivity
    have = [k for k in ("primary", "sensitivity") if k in summ["maps"]]
    fig, axes = plt.subplots(1, len(have), figsize=(6.6 * len(have), 6.9),
                             squeeze=False)
    all_S = np.concatenate([np.asarray(summ["maps"][k]["S_rank"], float).ravel()
                            for k in have])
    lim = float(np.nanmax(np.abs(all_S))) if np.isfinite(all_S).any() else 1.0
    for ax, key in zip(axes[0], have):
        m = summ["maps"][key]
        sm = _draw_cells(ax, m["S_rank"], m["n_valid"], edges,
                         cmap="RdBu_r", vmin=-lim, vmax=lim)
        _overlay(ax, g, edges, label_axis=(key == have[0]))
        if conf and key == "primary":
            # both numbers, never the map value alone
            for row in conf["cells"]:
                cx, cy = row["center"]
                mv, cv = row["map_value"], row["confirmed_mean"]
                if mv is None:
                    continue
                ax.annotate(f"map {mv:+.2f}\nconfirmed {cv:+.2f}"
                            if cv is not None else f"map {mv:+.2f}",
                            xy=(cx, cy), xytext=(cx, cy - 4.2), ha="center",
                            fontsize=8.2, color="0.15", zorder=8,
                            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                      ec="0.6", lw=0.6, alpha=0.92),
                            arrowprops=dict(arrowstyle="-", color="0.4", lw=0.9))
        sigma = cfg["sigmas"][key]
        ax.set_title(f"probe width {sigma:g} mm"
                     + ("   (pre-registered primary)" if key == "primary"
                        else "   (sensitivity)"),
                     fontsize=11.5, fontweight="bold", color="0.2", pad=9)
    cb = fig.colorbar(sm, ax=axes[0].tolist(), fraction=0.030, pad=0.02)
    cb.set_label("template match", fontsize=10)
    fig.suptitle("Where a fixed-budget heterogeneity scores across the sheet",
                 fontsize=13.5, fontweight="bold", x=0.045, ha="left", y=0.975)
    fig.legend(handles=legend, frameon=False, fontsize=8.6, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    stem = os.path.join(figdir, "stage3_sweep_match_map")
    fig.savefig(stem + ".png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(stem + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ------------------------------------------- what the score cannot show
    m = summ["maps"]["primary"]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.9))
    runs = m["n_runs"]
    sm0 = _draw_cells(axes[0], m["bidirectional_fraction"], runs, edges,
                      cmap="viridis", vmin=0, vmax=1, show_counts=False)
    _overlay(axes[0], g, edges)
    axes[0].set_title("fraction of networks producing both directions",
                      fontsize=11.5, fontweight="bold", color="0.2", pad=9)
    fig.colorbar(sm0, ax=axes[0], fraction=0.040, pad=0.02)

    rec = np.asarray(m["recruited_min"], float)
    sm1 = _draw_cells(axes[1], rec, runs, edges, cmap="viridis",
                      vmin=0, vmax=float(np.nanmax(rec)) if np.isfinite(rec).any() else 1,
                      show_counts=False)
    _overlay(axes[1], g, edges, label_axis=False)
    axes[1].set_title("contacts recruited by the weaker direction",
                      fontsize=11.5, fontweight="bold", color="0.2", pad=9)
    fig.colorbar(sm1, ax=axes[1], fraction=0.040, pad=0.02)
    fig.suptitle(f"What the match score cannot show   (probe width "
                 f"{cfg['sigmas']['primary']:g} mm)",
                 fontsize=13, fontweight="bold", x=0.045, ha="left", y=0.975)
    stem2 = os.path.join(figdir, "stage3_sweep_direction_and_recruitment")
    fig.savefig(stem2 + ".png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(stem2 + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    json.dump(dict(figures=[os.path.basename(stem), os.path.basename(stem2)],
                   config_checksum=cfg["checksum"],
                   summary_complete=summ["complete"],
                   artefact_audit=summ["artefact_audit"],
                   undefined_fraction={k: summ["maps"][k]["undefined_fraction"]
                                       for k in have},
                   n_valid_histogram={k: summ["maps"][k]["n_valid_histogram"]
                                      for k in have},
                   high_scoring_region=summ["high_scoring_region"],
                   region_confirmation=conf,
                   degradations=["sensitivity map (probe width 2.4 mm) was not run: "
                                 "the plan's pre-registered degradation order cuts "
                                 "that tier first when the window is short"],
                   plotting_only=True,
                   claim_boundary=("descriptive map from a fixed-shape probe; the "
                                   "highest cell is not a determined optimum -- see "
                                   "the confirmation pass on independent seeds")),
              open(stem + "_metadata.json", "w"), indent=1)
    print(f"wrote {stem}.png / .pdf\nwrote {stem2}.png / .pdf")


if __name__ == "__main__":
    main()
