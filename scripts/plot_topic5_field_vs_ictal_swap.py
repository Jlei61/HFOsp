#!/usr/bin/env python3
"""Topic 5 — interictal propagation field vs SEIZURE-ONSET activation field, with the
role-SWAP SOURCE nodes ringed (same paper-ready annotation as the swap_nodes / class_vs_template
figures).

This is the field-concordance comparison (interictal axis field | ictal activation field) carrying
the swap-source annotation that the other paper-ready field figures already have:
  left  = interictal propagation order (early -> late)                      [template A]
  right = seizure-onset activation (low -> high, mean broadband 0-10 s)      [the ictal gradient,
          sign-oriented to the interictal axis so "same colour in the same place = concordant"]
On BOTH panels the SWAP SOURCE contacts are ringed, coloured by which template they are the SOURCE
(early) end in — the role is bound to the contact, so the ring colour is the SAME contact in both
panels:
  RED  = source in template A (leads A, trails B)
  BLUE = source in template B (leads B, trails A)

Per swap-positive subject one figure + one combined atlas. Two substrates (broad / narrow); the swap
nodes and the display geometry are substrate-matched. Ictal activation comes from the (Epilepsiae)
t0 feature cache, matched to the geometry contacts by name (Yuquan has no ictal cache -> skipped).

Reuses scripts.plot_topic5_swap_nodes_fields for the substrate map, subject loader (rank-displacement
+ geometry + swap groups), ring style and legend, so the annotation is byte-for-byte the same style.
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

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.plot_contact_plane_static import _smooth_rank_field_mm
from scripts.plot_topic5_swap_nodes_fields import (
    SUBSTRATE, _subject_data, _arrays, _ring, _legend_handles,
    SRC_A_COL, SRC_B_COL, LABEL_HALO,
)

CACHE_DIR = _ROOT / "results/topic5_ictal_recruitment/t0_feature_cache"   # Epilepsiae-only ictal cache
OUT = _ROOT / "results/topic5_ictal_recruitment/axis_alignment/figures/field_concordance/field_vs_ictal_swap"
ACTIVATION_KEY = {"broadband": "bb_auc", "hfa": "hfa_auc", "ramp": "ramp", "ei": "ei_like"}
ACTIVATION_LABEL = {"broadband": "broadband power, 0-10 s", "hfa": "fast activity 60-100 Hz, 0-10 s",
                    "ramp": "activation ramp slope, 0-10 s", "ei": "EI-like (fast-activity / delay)"}

# Enlarged paper-grade font sizes (xy-labels + colorbar labels emphasized, per user request).
FS_AXLABEL = 17        # x/y axis labels (largest)
FS_CBAR_LABEL = 17     # colorbar labels (largest)
FS_TICK = 13           # axis + colorbar tick numbers
FS_TITLE_PANEL = 14    # per-subject panel title
FS_TITLE_COMPACT = 11  # atlas compact panel title
FS_TITLE_SUP = 15      # figure suptitle
FS_LEGEND = 14         # legend


def _ictal_activation(ds_sid, key="bb_auc"):
    """Per-contact mean early-ictal activation across the subject's eligible seizures -> {name: z}."""
    npz, mj = CACHE_DIR / f"{ds_sid}.npz", CACHE_DIR / f"{ds_sid}.json"
    if not npz.exists():
        return {}
    data = np.load(npz, allow_pickle=True)
    meta = json.load(open(mj))
    names = [str(x) for x in data["channels"]]
    arrs = [data[f"{key}__{i}"] for i in meta["eligible_idxs"] if f"{key}__{i}" in data.files]
    if not arrs:
        return {}
    mean_act = np.nanmean(np.vstack([np.asarray(a, float) for a in arrs]), axis=0)
    return {n: float(v) for n, v in zip(names, mean_act)}


def _rank01(vals):
    v = np.asarray(vals, float); out = np.full(v.shape, np.nan); ok = np.isfinite(v)
    if ok.sum() >= 2:
        out[ok] = np.argsort(np.argsort(v[ok])) / (ok.sum() - 1)
    elif ok.sum() == 1:
        out[ok] = 0.5
    return out


def _payload(ds_sid, rd_dir, geo_dir, activation):
    """Swap-positive subject with ictal activation -> drawing payload, else None."""
    dat = _subject_data(ds_sid, rd_dir, geo_dir)        # None unless swap strict/candidate + geometry
    if dat is None:
        return None
    names, xs, ys, inter, sup, soz = _arrays(dat["ta"], dat["frame"])
    act = _ictal_activation(ds_sid, ACTIVATION_KEY[activation])
    if not act:                                          # no eligible ictal seizures (e.g. Yuquan) -> skip
        return None
    ict_raw = _rank01([act.get(n, np.nan) for n in names])
    # Statistic is sign-free: orient the ictal field to the interictal axis sign (flip if anti-correlated)
    # so "same colour in the same place = concordant gradient" stays readable.
    m = np.isfinite(inter) & np.isfinite(ict_raw)
    flip = bool(m.sum() >= 3 and np.corrcoef(inter[m], ict_raw[m])[0, 1] < 0)
    ict = (1.0 - ict_raw) if flip else ict_raw
    dat.update(names=names, xs=np.asarray(xs), ys=np.asarray(ys), sup=sup, soz=soz,
               inter=inter, ict=ict, flip=flip)
    return dat


def _field_panel(ax, dat, vals, title, cbar_label, *, compact, labels=False, cbar=False):
    """viridis field + contacts + swap-source rings (red=source in A, blue=source in B) + SOZ rings."""
    xlim, ylim, sigma = dat["frame"]["xlim"], dat["frame"]["ylim"], dat["frame"]["sigma_mm"]
    xs, ys, names, soz = dat["xs"], dat["ys"], dat["names"], dat["soz"]
    _, _, T, _, _ = _smooth_rank_field_mm(xs, ys, vals, dat["sup"], xlim, ylim, sigma)
    im = ax.imshow(T, origin="lower", extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                   aspect="equal", cmap="viridis", vmin=0, vmax=1)
    ok = np.isfinite(vals)
    xx, yy, vv = xs[ok], ys[ok], np.asarray(vals)[ok]
    nn = [n for n, o in zip(names, ok) if o]
    soz_ok = np.asarray(soz)[ok]
    base_lw = 1.0 if compact else 1.7
    ax.scatter(xx, yy, c=vv, cmap="viridis", vmin=0, vmax=1, s=40 if compact else 80, zorder=3,
               edgecolors=["k" if z else "white" for z in soz_ok],
               linewidths=[base_lw + 0.4 if z else base_lw for z in soz_ok])
    for grp, col in ((dat["src_a"], SRC_A_COL), (dat["src_b"], SRC_B_COL)):
        mask = np.array([n in grp for n in nn])
        if mask.any():
            _ring(ax, xx[mask], yy[mask], col, compact=compact)
    if labels and not compact:
        for x, y, n in zip(xx, yy, nn):
            ax.text(x, y + (ylim[1] - ylim[0]) * 0.026, n, ha="center", va="bottom",
                    fontsize=5.5, color="0.92", path_effects=LABEL_HALO, zorder=6)
    ax.set_title(title, fontsize=FS_TITLE_COMPACT if compact else FS_TITLE_PANEL)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal", adjustable="box")
    if compact:
        ax.set_xticks([]); ax.set_yticks([])
    else:
        ax.set_xlabel("along propagation axis (mm)", fontsize=FS_AXLABEL)
        ax.set_ylabel("transverse (mm)", fontsize=FS_AXLABEL)
        ax.tick_params(labelsize=FS_TICK)
        if cbar:
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.set_label(cbar_label, fontsize=FS_CBAR_LABEL)
            cb.ax.tick_params(labelsize=FS_TICK)


def _ict_labels(dat, activation):
    # Colorbar text = what the field actually is: a within-subject RANK of the mean 0-10 s
    # baseline-robust-z activation (not absolute energy, not a peak). The flip note is kept
    # because the ictal field is sign-oriented to the interictal axis for the concordance read.
    base = "activation rank (within subject, mean 0–10s)"
    flip_note = "  [flipped to match axis]" if dat["flip"] else ""
    return (f"seizure-onset activation — {ACTIVATION_LABEL[activation]}", base + flip_note)


def plot_subject(dat, substrate, activation):
    ds_sid = dat["ds_sid"]; ss = dat["ss"]
    pretty = ds_sid.replace("epilepsiae_", "E").replace("yuquan_", "Y-")
    ict_title, ict_lbl = _ict_labels(dat, activation)
    fig, ax = plt.subplots(1, 2, figsize=(14.0, 7.4), layout="constrained")
    _field_panel(ax[0], dat, dat["inter"], "interictal propagation order — template A",
                 "early (0) -> late (1)", compact=False, labels=True, cbar=True)
    _field_panel(ax[1], dat, dat["ict"], ict_title, ict_lbl, compact=False, labels=True, cbar=True)
    nodes = sorted(dat["src_a"]) + sorted(dat["src_b"])
    fig.suptitle(f"Patient {pretty} — interictal propagation field vs seizure-onset activation, "
                 f"role-SWAP source nodes\n(swap={ss.get('swap_class')}, k={ss.get('decision_k')}, "
                 f"p_fw={ss.get('p_fw'):.3f}; {len(nodes)} nodes; red=source in A, blue=source in B)",
                 fontsize=FS_TITLE_SUP)
    # Legend in its own reserved band below the panels (separate, never overlapping the fields).
    fig.legend(handles=_legend_handles(), loc="outside lower center", ncol=3,
               fontsize=FS_LEGEND, frameon=False)
    OUT.mkdir(parents=True, exist_ok=True)
    fp = OUT / f"{ds_sid}_field_vs_ictal_{substrate}.png"
    fig.savefig(fp, dpi=135, bbox_inches="tight"); plt.close(fig)
    return fp


def plot_all(data, substrate, activation):
    n = len(data)
    # Grid of subjects (each = an adjacent interictal|ictal axis pair) so the atlas stays compact.
    ncols_sub = 3 if n >= 9 else (2 if n >= 4 else 1)
    nrows = int(np.ceil(n / ncols_sub))
    fig, axes = plt.subplots(nrows, 2 * ncols_sub, figsize=(6.6 * ncols_sub, 3.1 * nrows),
                             squeeze=False, layout="constrained")
    for i, dat in enumerate(data):
        r, c = i // ncols_sub, (i % ncols_sub) * 2
        pretty = dat["ds_sid"].replace("epilepsiae_", "E").replace("yuquan_", "Y-")
        ss = dat["ss"]
        _field_panel(axes[r, c], dat, dat["inter"],
                     f"{pretty}  interictal  (swap={ss.get('swap_class')}, k={ss.get('decision_k')})",
                     "", compact=True)
        _field_panel(axes[r, c + 1], dat, dat["ict"], f"{pretty}  seizure-onset", "", compact=True)
    for j in range(n, nrows * ncols_sub):          # blank the unused subject slots
        r, c = j // ncols_sub, (j % ncols_sub) * 2
        axes[r, c].axis("off"); axes[r, c + 1].axis("off")
    # Legend in its own reserved band (bottom), separate from the panels and the suptitle.
    fig.legend(handles=_legend_handles(), loc="outside lower center", ncol=3, fontsize=FS_LEGEND,
               framealpha=0.9)
    fig.suptitle(f"Swap-positive subjects ({substrate}): interictal propagation field | seizure-onset "
                 f"activation ({ACTIVATION_LABEL[activation]}), role-SWAP source nodes ringed "
                 f"(red=source in A, blue=source in B)", fontsize=FS_TITLE_SUP)
    OUT.mkdir(parents=True, exist_ok=True)
    fp = OUT / f"ALL_field_vs_ictal_{substrate}.png"
    fig.savefig(fp, dpi=120, bbox_inches="tight"); plt.close(fig)
    return fp


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--substrate", choices=list(SUBSTRATE), default="broad")
    ap.add_argument("--activation", choices=list(ACTIVATION_KEY), default="broadband")
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()
    rd_dir, geo_dir = SUBSTRATE[args.substrate]
    subs = args.subjects or sorted(p.stem for p in rd_dir.glob("*.json"))
    data = []
    for ds_sid in subs:
        dat = _payload(ds_sid, rd_dir, geo_dir, args.activation)
        if dat is None:
            continue
        fp = plot_subject(dat, args.substrate, args.activation)
        print(f"[fig] {fp.name}  swap={dat['ss'].get('swap_class')} k={dat['ss'].get('decision_k')} "
              f"src_A={sorted(dat['src_a'])} src_B={sorted(dat['src_b'])}", flush=True)
        data.append(dat)
    if data:
        fp = plot_all(data, args.substrate, args.activation)
        print(f"[fig] {fp.name}  (combined {len(data)} subjects)")
    print(f"[done] {len(data)} per-subject + 1 combined ({args.substrate}) -> {OUT}")


if __name__ == "__main__":
    main()
