#!/usr/bin/env python3
"""Topic5 V3d — exploratory per-seizure C_AB raincloud (axis-present windows only).

Purely descriptive: no statistics, no null. For a subject (default epilepsiae_1146),
render one horizontal row per seizure showing the distribution of the fixed-orientation
A/B scaffold contrast C_AB (+1 = A source side, -1 = B source side) over that seizure's
axis-present peri-onset windows, with each window point colored by its time from onset
(dark = far-preictal, bright = onset). Purpose: eyeball whether "switch"-labeled
seizures are truly bimodal at the +/-extremes (far-pre on one side, near-onset on the
other) versus just smeared near 0.

Reads (does NOT recompute — see scripts/run_topic5_scaffold_ab_switching.py for the
producer):
  results/topic5_ictal_recruitment/scaffold_ab_switching/per_subject/
    <ds_sid>_scaffold_ab_contrast.npz    (grid_centers, cab, present, align_sign,
                                           h1_valid, seizure_idx)
    <ds_sid>_scaffold_ab_per_seizure.csv (event_class + far_side/near_side labels)

NOTE on align_sign: the npz's `cab` array is already the TRUE fixed-orientation C_AB
(the same convention plot_topic5_scaffold_ab_contrast_timecourse.py calls its "thin
lines"). `align_sign` (sign of each seizure's near-onset mean) exists only to build
that other figure's aggregate near-onset-aligned median; it is deliberately NOT applied
here, because flipping by align_sign would force every seizure's near-onset windows
positive by construction and make "is it bimodal at the extremes" untestable by eye.

Output: results/topic5_ictal_recruitment/scaffold_ab_switching/figures/
        <ds_sid>_scaffold_ab_perseizure_dist.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from src.plot_style import savefig_pub
    _HAVE_STYLE = True
except Exception:  # pragma: no cover - styling best-effort
    _HAVE_STYLE = False

OUT_DIR = _ROOT / "results/topic5_ictal_recruitment/scaffold_ab_switching"
SUB_DIR = OUT_DIR / "per_subject"
FIG_DIR = OUT_DIR / "figures"

DELTA_SIDE = 0.2          # side-label threshold; matches src.topic5_scaffold_ab_contrast DELTA_SIDE
MIN_N_VIOLIN = 6          # minimum axis-present points required to draw a half-violin
BIMODAL_FRAC_MIN = 0.25   # both-side fraction threshold for the descriptive bimodal_extremes flag
T_MIN, T_MAX = -120.0, 15.0  # peri-onset window range (spec-locked grid, s from onset)

COL_A = "#b2182b"   # A source side (matches plot_topic5_scaffold_ab_contrast_timecourse.py)
COL_B = "#2166ac"   # B source side (matches plot_topic5_scaffold_ab_contrast_timecourse.py)

GROUP_ORDER = {"switch": 0, "selection": 1, "persistent": 2, "none": 3}


def _pretty(ds_sid: str) -> str:
    return ds_sid.replace("epilepsiae_", "E").replace("yuquan_", "Y-")


def _load_rows(ds_sid: str) -> list[dict]:
    npz = np.load(SUB_DIR / f"{ds_sid}_scaffold_ab_contrast.npz")
    grid_centers = npz["grid_centers"]
    cab = npz["cab"]
    present = npz["present"]
    seizure_idx_npz = [int(v) for v in npz["seizure_idx"]]
    by_idx = {si: i for i, si in enumerate(seizure_idx_npz)}

    df = pd.read_csv(SUB_DIR / f"{ds_sid}_scaffold_ab_per_seizure.csv")

    rows = []
    for _, r in df.iterrows():
        si = int(r["seizure_idx"])
        if si not in by_idx:  # defensive: npz/csv are written together by the same producer run
            print(f"  WARNING: seizure_idx={si} in CSV but not in npz -- skipped", file=sys.stderr)
            continue
        i = by_idx[si]
        mask = present[i]
        c_present = cab[i][mask]
        t_present = grid_centers[mask]
        n_present = int(mask.sum())
        if n_present != int(r["n_axis_present_win"]):
            print(f"  WARNING: seizure_idx={si} n_present mismatch npz={n_present} "
                  f"csv={int(r['n_axis_present_win'])}", file=sys.stderr)

        if n_present > 0:
            frac_on_A = float(np.mean(c_present >= DELTA_SIDE))
            frac_on_B = float(np.mean(c_present <= -DELTA_SIDE))
            frac_near_zero = float(np.mean(np.abs(c_present) < DELTA_SIDE))
        else:
            frac_on_A = frac_on_B = frac_near_zero = float("nan")
        bimodal_extremes = bool(frac_on_A >= BIMODAL_FRAC_MIN and frac_on_B >= BIMODAL_FRAC_MIN)

        rows.append({
            "seizure_idx": si,
            "event_class": str(r["event_class"]),
            "far_side": str(r["far_side"]),
            "near_side": str(r["near_side"]),
            "n_present": n_present,
            "frac_on_A": frac_on_A,
            "frac_on_B": frac_on_B,
            "frac_near_zero": frac_near_zero,
            "bimodal_extremes": bimodal_extremes,
            "c_present": c_present,
            "t_present": t_present,
        })

    rows.sort(key=lambda d: (GROUP_ORDER.get(d["event_class"], 4), d["seizure_idx"]))
    return rows


def _print_table(rows: list[dict]) -> None:
    print(f"{'sz':>4}  {'event_class':<11} {'n':>3}  {'fracA':>6} {'fracB':>6} "
          f"{'fracZero':>8}  {'bimodal':>7}  far->near")
    for d in rows:
        print(f"{d['seizure_idx']:>4}  {d['event_class']:<11} {d['n_present']:>3}  "
              f"{d['frac_on_A']:>6.2f} {d['frac_on_B']:>6.2f} {d['frac_near_zero']:>8.2f}  "
              f"{str(d['bimodal_extremes']):>7}  {d['far_side']}->{d['near_side']}")

    switch_rows = [d for d in rows if d["event_class"] == "switch"]
    n_switch = len(switch_rows)
    n_bimodal = sum(1 for d in switch_rows if d["bimodal_extremes"])
    n_smeared = n_switch - n_bimodal
    print(f"\nSWITCH VERDICT: {n_bimodal}/{n_switch} switch-labeled seizures are truly "
          f"bimodal-at-extremes (>={int(BIMODAL_FRAC_MIN * 100)}% of axis-present windows on "
          f"EACH side, |C_AB|>={DELTA_SIDE}); {n_smeared}/{n_switch} are smeared near 0 / "
          f"lopsided instead.")


def plot_subject(ds_sid: str, rows: list[dict]) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_png = FIG_DIR / f"{ds_sid}_scaffold_ab_perseizure_dist.png"

    n_rows = len(rows)
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=T_MIN, vmax=T_MAX)

    row_h = 1.0
    violin_h = 0.34
    jitter_h = 0.28
    y = -row_h * np.arange(n_rows, dtype=float)  # row 0 -> y=0 (top row)

    fig_h = max(6.0, n_rows * 0.40 + 2.4)
    fig, ax = plt.subplots(figsize=(11.0, fig_h))

    ax.set_xlim(-1.05, 1.05)
    y_top = 0.55
    y_bot = y[-1] - jitter_h - 0.25
    ax.set_ylim(y_bot, y_top)

    ax.axvspan(0, 1.05, color=COL_A, alpha=0.06, lw=0, zorder=0)
    ax.axvspan(-1.05, 0, color=COL_B, alpha=0.06, lw=0, zorder=0)
    ax.axvline(0, color="0.25", lw=1.2, zorder=1)
    ax.axvline(DELTA_SIDE, color="0.45", lw=0.8, ls="--", alpha=0.7, zorder=1)
    ax.axvline(-DELTA_SIDE, color="0.45", lw=0.8, ls="--", alpha=0.7, zorder=1)
    ax.text(0.5, y_top - 0.03, "A source side", ha="center", va="top",
            fontsize=9, color=COL_A, alpha=0.85)
    ax.text(-0.5, y_top - 0.03, "B source side", ha="center", va="top",
            fontsize=9, color=COL_B, alpha=0.85)

    rng = np.random.default_rng(0)  # visual jitter only, fixed for a reproducible render
    for i, d in enumerate(rows):
        yi = y[i]
        c_vals, t_vals, n = d["c_present"], d["t_present"], d["n_present"]

        if n >= MIN_N_VIOLIN and np.std(c_vals) > 1e-9:
            kde = gaussian_kde(c_vals, bw_method="silverman")
            xs = np.linspace(-1.0, 1.0, 200)
            dens = kde(xs)
            dens = dens / dens.max() * violin_h
            ax.fill_between(xs, yi, yi + dens, color="0.35", alpha=0.35,
                             lw=0.7, edgecolor="0.2", zorder=3)

        if n > 0:
            jit = rng.uniform(-jitter_h, -0.03, size=n)
            ax.scatter(c_vals, yi + jit, c=t_vals, cmap=cmap, norm=norm,
                       s=16, alpha=0.85, linewidths=0.3, edgecolors="white", zorder=4)

        ax.text(-0.012, yi, f"sz{d['seizure_idx']} · {d['event_class']} · n={n}",
                transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8,
                clip_on=False)

        if i > 0 and rows[i]["event_class"] != rows[i - 1]["event_class"]:
            ax.axhline((y[i] + y[i - 1]) / 2.0, color="0.6", lw=0.7, alpha=0.6, zorder=2)

    ax.set_yticks([])
    for spine in ("left", "top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_xticks([-1, -0.5, -0.2, 0, 0.2, 0.5, 1])
    ax.set_xlabel(r"$C_{AB}$   (A source side  +1  $\leftrightarrow$  $-1$  B source side)",
                  fontsize=11)
    ax.set_title(f"{_pretty(ds_sid)} · per-seizure $C_{{AB}}$ distribution "
                 "(axis-present windows only), colored by peri-onset time", fontsize=12)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.022, aspect=45)
    cbar.set_label("window center (s from onset)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.text(0.5, 0.005,
              "Half-violins: per-row peak-normalized KDE of axis-present C_AB (shape only; "
              f"skipped when n<{MIN_N_VIOLIN}). Points: every axis-present window, vertically "
              "jittered, colored by window center time (dark=far-preictal, bright=onset). "
              f"Dashed lines at ±{DELTA_SIDE} = side-label threshold.",
              ha="center", va="bottom", fontsize=7.5, color="0.35")

    if _HAVE_STYLE:
        savefig_pub(fig, out_png, dpi=200)
    else:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return out_png


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", default="epilepsiae_1146")
    args = ap.parse_args()

    rows = _load_rows(args.subject)
    _print_table(rows)
    out_png = plot_subject(args.subject, rows)
    print(f"\nSaved {out_png}")


if __name__ == "__main__":
    main()
