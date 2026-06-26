#!/usr/bin/env python3
"""Topic 5 — interictal two-template field comparison with the SWAP NODES marked by role.

For each subject whose two interictal propagation templates show a role-swap (PR-6 supplement
rank-displacement swap_sweep, swap_class strict/candidate), draw the two templates side by side on
the subject's contact plane (template A order | template B order) and ring the SWAP NODES, colored
by which template they are the SOURCE (early) end in — the role is bound to the channel, so the ring
color is the SAME contact in both panels:
  RED  = source in template A  (early/leads in A, late/trails in B)
  BLUE = source in template B  (early/leads in B, late/trails in A)
(src.rank_displacement.swap_node_groups_at_k at the swap's decision_k.) Contacts are labelled with
their channel name (legible halo). One figure per swap-positive subject + one combined overview.

Default substrate = NARROW (the PR-6 masked rank-displacement + its A-line t_a/t_b records).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from src.rank_displacement import swap_node_groups_at_k
from scripts.plot_contact_plane_static import (_subject_display_frame, _display_points,
                                               _smooth_rank_field_mm, _attach_real_coords)

SUBSTRATE = {
    "narrow": (_ROOT / "results/interictal_propagation_masked/rank_displacement/per_subject",
               _ROOT / "results/spatial_modulation/propagation_geometry/observation_readout/real_subjects"),
    "broad": (_ROOT / "results/interictal_propagation_masked_broad/rank_displacement/per_subject",
              _ROOT / "results/spatial_modulation/propagation_geometry_broad/observation_readout/real_subjects"),
}
OUT = _ROOT / "results/topic5_ictal_recruitment/axis_alignment/figures/field_concordance/swap_nodes"
SRC_A_COL = "#d62728"   # source-in-A (red)
SRC_B_COL = "#1f77b4"   # source-in-B (blue)
LABEL_HALO = [pe.withStroke(linewidth=1.0, foreground="black")]   # subtle, not eye-grabbing


def _ring(ax, xx, yy, col, *, compact):
    """A colored swap ring with a black outline so it reads on any viridis background."""
    lw = 2.0 if compact else 2.6
    pc = ax.scatter(xx, yy, s=80 if compact else 200, facecolors="none",
                    edgecolors=col, linewidths=lw, zorder=5)
    pc.set_path_effects([pe.withStroke(linewidth=lw + 2.6, foreground="black")])


def _rank01(vals):
    v = np.asarray(vals, float); out = np.full(v.shape, np.nan); ok = np.isfinite(v)
    if ok.sum() >= 2:
        out[ok] = np.argsort(np.argsort(v[ok])) / (ok.sum() - 1)
    elif ok.sum() == 1:
        out[ok] = 0.5
    return out


def _subject_data(ds_sid, rd_dir, geo_dir):
    rd = rd_dir / f"{ds_sid}.json"
    ta_f, tb_f = geo_dir / f"{ds_sid}_t_a.json", geo_dir / f"{ds_sid}_t_b.json"
    if not (rd.exists() and ta_f.exists() and tb_f.exists()):
        return None
    d = json.load(open(rd))
    pp = d.get("primary_pair") or (d.get("pairs") or [{}])[0]
    ss = pp.get("swap_sweep", {})
    k = ss.get("decision_k")
    if ss.get("swap_class") not in ("strict", "candidate") or not k:
        return None
    jv = np.asarray(pp["joint_valid"], bool)
    ra = np.asarray(pp["rank_a_dense_full"], float); rb = np.asarray(pp["rank_b_dense_full"], float)
    src_a_i, src_b_i = swap_node_groups_at_k(ra, rb, jv, jv, int(k))
    names_rd = pp["channel_names"]
    src_a = {names_rd[i] for i in src_a_i}; src_b = {names_rd[i] for i in src_b_i}
    if not (src_a or src_b):
        return None
    ta = json.loads(ta_f.read_text()); tb = json.loads(tb_f.read_text())
    if not ta.get("channels") or not tb.get("channels"):
        return None
    _attach_real_coords([ta, tb])
    frame = _subject_display_frame([ta, tb])
    if frame is None:
        return None
    return {"ds_sid": ds_sid, "ta": ta, "tb": tb, "frame": frame, "ss": ss,
            "src_a": src_a, "src_b": src_b}


def _arrays(rec, frame):
    names = [c["name"] for c in rec["channels"]]
    xs, ys = _display_points(rec, frame)
    vals = _rank01([c.get("typical_rank", np.nan) for c in rec["channels"]])
    sup = np.array([c.get("support", 1.0) for c in rec["channels"]], float)
    soz = np.array([bool(c.get("is_soz")) for c in rec["channels"]])
    return names, xs, ys, vals, sup, soz


def _panel(ax, dat, rec, title, *, compact=False, labels=True, cbar=True):
    names, xs, ys, vals, sup, soz = _arrays(rec, dat["frame"])
    xlim, ylim, sigma = dat["frame"]["xlim"], dat["frame"]["ylim"], dat["frame"]["sigma_mm"]
    _, _, T, _, _ = _smooth_rank_field_mm(xs, ys, vals, sup, xlim, ylim, sigma)
    im = ax.imshow(T, origin="lower", extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                   aspect="equal", cmap="viridis", vmin=0, vmax=1)
    ok = np.isfinite(vals)
    xx, yy, vv = np.asarray(xs)[ok], np.asarray(ys)[ok], np.asarray(vals)[ok]
    nn = [n for n, o in zip(names, ok) if o]
    soz_ok = np.asarray(soz)[ok]
    base_lw = 1.0 if compact else 1.7        # thicker electrode frame
    ax.scatter(xx, yy, c=vv, cmap="viridis", vmin=0, vmax=1, s=40 if compact else 80, zorder=3,
               edgecolors=["k" if z else "white" for z in soz_ok],
               linewidths=[base_lw + 0.4 if z else base_lw for z in soz_ok])
    # swap node rings, colored by source role (same contact -> same color in both panels), black-outlined
    for grp, col in ((dat["src_a"], SRC_A_COL), (dat["src_b"], SRC_B_COL)):
        mask = np.array([n in grp for n in nn])
        if mask.any():
            _ring(ax, xx[mask], yy[mask], col, compact=compact)
    if labels and not compact:
        for x, y, n in zip(xx, yy, nn):
            ax.text(x, y + (ylim[1] - ylim[0]) * 0.026, n, ha="center", va="bottom",
                    fontsize=5.5, color="0.92", path_effects=LABEL_HALO, zorder=6)
    ax.set_title(title, fontsize=9 if compact else 10.5)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal", adjustable="box")
    if compact:
        ax.set_xticks([]); ax.set_yticks([])
    else:
        ax.set_xlabel("along propagation axis (mm)"); ax.set_ylabel("transverse (mm)")
        if cbar:
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label("early (0) -> late (1)", fontsize=9)


def _legend_handles():
    return [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor=SRC_A_COL,
                       markeredgewidth=2.5, markersize=12, label="source in template A (leads A, trails B)"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor=SRC_B_COL,
                       markeredgewidth=2.5, markersize=12, label="source in template B (leads B, trails A)"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="0.6", markeredgecolor="k",
                       markersize=10, label="clinical seizure-onset contact (overlay)")]


def plot_subject(dat, substrate):
    ds_sid = dat["ds_sid"]; ss = dat["ss"]
    pretty = ds_sid.replace("epilepsiae_", "E").replace("yuquan_", "Y-")
    fig, ax = plt.subplots(1, 2, figsize=(13.6, 6.6), constrained_layout=True)
    _panel(ax[0], dat, dat["ta"], "interictal propagation order — template A")
    _panel(ax[1], dat, dat["tb"], "interictal propagation order — template B")
    nodes = sorted(dat["src_a"]) + sorted(dat["src_b"])
    fig.suptitle(f"Patient {pretty} — two interictal templates, role-SWAP nodes "
                 f"(swap={ss.get('swap_class')}, k={ss.get('decision_k')}, p_fw={ss.get('p_fw'):.3f}; "
                 f"{len(nodes)} nodes)", fontsize=10.5)
    fig.legend(handles=_legend_handles(), loc="lower center", ncol=3, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.03))     # legend OUTSIDE the panels
    OUT.mkdir(parents=True, exist_ok=True)
    fp = OUT / f"{ds_sid}_swap_nodes_{substrate}.png"
    fig.savefig(fp, dpi=135, bbox_inches="tight"); plt.close(fig)
    return fp


def plot_all(data, substrate):
    n = len(data)
    # Grid of subjects (each = an adjacent A|B axis pair) instead of one tall
    # single column, so the contact sheet stays compact rather than 3*n inches long.
    ncols_sub = 3 if n >= 9 else (2 if n >= 4 else 1)
    nrows = int(np.ceil(n / ncols_sub))
    fig, axes = plt.subplots(nrows, 2 * ncols_sub, figsize=(6.6 * ncols_sub, 3.1 * nrows),
                             squeeze=False)
    for i, dat in enumerate(data):
        r, c = i // ncols_sub, (i % ncols_sub) * 2
        pretty = dat["ds_sid"].replace("epilepsiae_", "E").replace("yuquan_", "Y-")
        ss = dat["ss"]
        _panel(axes[r, c], dat, dat["ta"], f"{pretty}  A  (swap={ss.get('swap_class')}, k={ss.get('decision_k')})",
               compact=True, labels=False)
        _panel(axes[r, c + 1], dat, dat["tb"], f"{pretty}  B", compact=True, labels=False)
    for j in range(n, nrows * ncols_sub):          # blank the unused subject slots
        r, c = j // ncols_sub, (j % ncols_sub) * 2
        axes[r, c].axis("off"); axes[r, c + 1].axis("off")
    fig.legend(handles=_legend_handles(), loc="upper center", ncol=3, fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"All swap-positive subjects ({substrate}): interictal template A | B, role-SWAP nodes "
                 f"(red=source in A, blue=source in B)", fontsize=11, y=1.005)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    OUT.mkdir(parents=True, exist_ok=True)
    fp = OUT / f"ALL_swap_nodes_{substrate}.png"
    fig.savefig(fp, dpi=120, bbox_inches="tight"); plt.close(fig)
    return fp


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--substrate", choices=list(SUBSTRATE), default="narrow")
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()
    rd_dir, geo_dir = SUBSTRATE[args.substrate]
    subs = args.subjects or sorted(p.stem for p in rd_dir.glob("*.json"))
    data = []
    for ds_sid in subs:
        dat = _subject_data(ds_sid, rd_dir, geo_dir)
        if dat is None:
            continue
        fp = plot_subject(dat, args.substrate)
        print(f"[fig] {fp.name}  swap={dat['ss'].get('swap_class')} k={dat['ss'].get('decision_k')} "
              f"src_A={sorted(dat['src_a'])} src_B={sorted(dat['src_b'])}")
        data.append(dat)
    if data:
        fp = plot_all(data, args.substrate)
        print(f"[fig] {fp.name}  (combined {len(data)} subjects)")
    print(f"[done] {len(data)} per-subject + 1 combined ({args.substrate}) -> {OUT}")


if __name__ == "__main__":
    main()
