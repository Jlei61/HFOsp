#!/usr/bin/env python3
"""Diagnostic subject plot for Topic 5 A-line field concordance.

This figure is intentionally NOT the paper atlas contract:
  - ictal activation is min-max normalized raw robust-z energy, not rank-normalized;
  - ictal panel is not sign-flipped to match the interictal axis;
  - red colormap maps low energy to light red and high energy to dark red.

Use it to inspect whether the visual field match depends on rank-display or sign alignment.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.plot_contact_plane_static import (
    _attach_real_coords,
    _display_points,
    _smooth_rank_field_mm,
    _subject_display_frame,
)
from scripts.plot_topic5_axis_alignment_fields import _ictal_activation
from src.topic5_axis_alignment import matched_channels


REAL_DIR = _ROOT / "results/spatial_modulation/propagation_geometry/observation_readout/real_subjects"
CACHE_DIR = _ROOT / "results/topic5_ictal_recruitment/t0_feature_cache"
OUT = _ROOT / "results/topic5_ictal_recruitment/axis_alignment/figures/field_concordance"

ACTIVATION_KEY = {"broadband": "bb_auc", "hfa": "hfa_auc"}
ACTIVATION_LABEL = {
    "broadband": "broadband robust-z power, 0-10 s",
    "hfa": "HFA robust-z power, 0-10 s",
}


def _minmax01(vals: np.ndarray) -> tuple[np.ndarray, float, float]:
    vals = np.asarray(vals, float)
    out = np.full(vals.shape, np.nan)
    ok = np.isfinite(vals)
    if not ok.any():
        return out, float("nan"), float("nan")
    vmin = float(np.nanmin(vals[ok]))
    vmax = float(np.nanmax(vals[ok]))
    if vmax - vmin <= 1e-12:
        out[ok] = 0.5
    else:
        out[ok] = (vals[ok] - vmin) / (vmax - vmin)
    return out, vmin, vmax


def _panel(ax, xs, ys, vals, support, xlim, ylim, sigma, cmap, title, cbar_label, soz):
    _, _, field, _, _ = _smooth_rank_field_mm(xs, ys, vals, support, xlim, ylim, sigma)
    im = ax.imshow(
        field,
        origin="lower",
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        aspect="equal",
        cmap=cmap,
        vmin=0,
        vmax=1,
    )
    ax.scatter(
        xs,
        ys,
        c=vals,
        cmap=cmap,
        vmin=0,
        vmax=1,
        s=62,
        zorder=3,
        edgecolors=["black" if z else "white" for z in soz],
        linewidths=[1.6 if z else 0.55 for z in soz],
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("along propagation axis (mm)")
    ax.set_ylabel("transverse (mm)")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(cbar_label, fontsize=9)
    return im


def plot_subject(ds_sid: str, activation: str) -> Path:
    axis_f = REAL_DIR / f"{ds_sid}_t_a.json"
    cache_f = CACHE_DIR / f"{ds_sid}.npz"
    if not axis_f.exists():
        raise FileNotFoundError(axis_f)
    if not cache_f.exists():
        raise FileNotFoundError(cache_f)

    rec = json.loads(axis_f.read_text())
    _attach_real_coords([rec])
    frame = _subject_display_frame([rec])
    if frame is None:
        raise RuntimeError(f"{ds_sid}: no display frame")

    cache = np.load(cache_f, allow_pickle=True)
    cache_names = [str(x) for x in cache["channels"]]
    matched = matched_channels(rec, {n: 0.0 for n in cache_names})
    names = [c["name"] for c in matched]
    if len(names) < 6:
        raise RuntimeError(f"{ds_sid}: insufficient matched channels ({len(names)})")

    all_xs, all_ys = _display_points(rec, frame)
    rec_index = {c["name"]: i for i, c in enumerate(rec["channels"])}
    idx = np.array([rec_index[n] for n in names])
    xs, ys = all_xs[idx], all_ys[idx]
    inter = np.array([float(c["typical_rank"]) for c in matched], float)
    support = np.array([float(c.get("support", 1.0)) for c in matched], float)
    soz = np.array([bool(c.get("is_soz")) for c in matched])

    act = _ictal_activation(ds_sid, ACTIVATION_KEY[activation])
    raw = np.array([act.get(n, np.nan) for n in names], float)
    ict01, raw_min, raw_max = _minmax01(raw)

    xlim, ylim, sigma = frame["xlim"], frame["ylim"], frame["sigma_mm"]
    red_light_low_to_dark_high = LinearSegmentedColormap.from_list(
        "red_light_low_to_dark_high",
        ["#fff7f3", "#fee0d2", "#ef3b2c", "#7f0000"],
    )

    pretty = ds_sid.replace("epilepsiae_", "E").replace("yuquan_", "Y-")
    fig, ax = plt.subplots(1, 2, figsize=(13.2, 6.0), constrained_layout=True)
    _panel(
        ax[0],
        xs,
        ys,
        inter,
        support,
        xlim,
        ylim,
        sigma,
        "viridis",
        "interictal propagation order - template A",
        "early (0) -> late (1)",
        soz,
    )
    _panel(
        ax[1],
        xs,
        ys,
        ict01,
        support,
        xlim,
        ylim,
        sigma,
        red_light_low_to_dark_high,
        f"seizure-onset activation - {ACTIVATION_LABEL[activation]}",
        "low energy (0, light red) -> high energy (1, dark red)",
        soz,
    )
    fig.suptitle(
        f"{pretty}: raw ictal energy display, min-max normalized, no rank, no sign flip\n"
        f"raw robust-z range across matched contacts: {raw_min:.3g} to {raw_max:.3g}",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.004,
        "Right panel is raw robust-z energy min-max normalized for display only; "
        "black ring = clinical SOZ overlay.",
        ha="center",
        fontsize=8.5,
        color="0.35",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{ds_sid}_raw_{activation}_minmax_no_flip.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="epilepsiae_1146")
    ap.add_argument("--activation", choices=sorted(ACTIVATION_KEY), default="broadband")
    args = ap.parse_args()
    out = plot_subject(args.subject, args.activation)
    print(out)


if __name__ == "__main__":
    main()
