#!/usr/bin/env python3
"""Topic5 V3d — figure 1: the fixed-orientation A/B contrast ``C_AB(t)`` timecourse.

Reads the producer's ``<ds_sid>_scaffold_ab_contrast.npz`` + ``..._summary.json`` and
renders ``figures/<ds_sid>_scaffold_ab_timecourse.png``. This is the figure that
carries the H1 statistic (spec §8 fig1).

Per spec §8 (P9): a naive signed median across seizures cancels (opposite-side
seizures average to ~0), so it is NOT plotted. Instead:
  - thin lines  = each seizure's TRUE ``C_AB(t)`` (+ = A source side, − = B source side),
                  drawn only where the window is axis_present (gaps elsewhere);
  - bold line   = the near-onset-side-aligned median: each seizure is multiplied by the
                  sign of its own near-onset signed mean (``align_sign`` from the producer)
                  so every seizure points its near-onset side up, then median across the
                  H1-valid seizures. This is what the H1 |mean| (locking) statistic sees;
  - bottom strip= fraction of seizures that are axis_present per window;
  - 0 line = no lateral preference; dashed vertical = clinical onset.

Standalone-clean preview figure (no PR/§ codes in axes); a text box reports the
subject's tier / ρ_AB / valid-seizure count / L_obs / subject_locked / p.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from src.plot_style import FS_LABEL, FS_TICK, savefig_pub  # noqa: E402
    _HAVE_STYLE = True
except Exception:  # pragma: no cover - styling best-effort
    _HAVE_STYLE = False
    FS_LABEL = FS_TICK = 11

OUT_DIR = _ROOT / "results/topic5_ictal_recruitment/scaffold_ab_switching"
SUB_DIR = OUT_DIR / "per_subject"
FIG_DIR = OUT_DIR / "figures"

COL_A = "#b2182b"   # A source side (diverging red, style-guide convention)
COL_B = "#2166ac"   # B source side (diverging blue)
COL_SZ = "0.72"     # individual seizures
COL_MED = "#222222" # aligned median (bold)
MIN_N_MED = 4       # min axis_present seizures per window to draw the aligned median


def _pretty(ds_sid: str) -> str:
    return ds_sid.replace("epilepsiae_", "E").replace("yuquan_", "Y-")


def _save(fig, out_png: Path) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if _HAVE_STYLE:
        savefig_pub(fig, out_png, dpi=200)
    else:
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _drop_figure(ds_sid: str, summary: dict, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axis("off")
    ax.text(0.5, 0.5, f"{_pretty(ds_sid)}\nno C_AB timecourse\n"
            f"drop reason: {summary.get('drop_reason', 'unknown')}",
            ha="center", va="center", fontsize=13, color="0.25")
    _save(fig, out_png)


def plot_subject(ds_sid: str) -> Path:
    summary = json.loads((SUB_DIR / f"{ds_sid}_scaffold_ab_summary.json").read_text())
    out_png = FIG_DIR / f"{ds_sid}_scaffold_ab_timecourse.png"
    npz_fp = SUB_DIR / f"{ds_sid}_scaffold_ab_contrast.npz"
    if summary.get("status") != "ok" or not npz_fp.exists():
        _drop_figure(ds_sid, summary, out_png)
        return out_png

    z = np.load(npz_fp)
    centers = z["grid_centers"]
    cab = z["cab"]                       # (n_sz, n_win)
    present = z["present"]               # (n_sz, n_win) bool
    align = z["align_sign"]              # (n_sz,)
    h1_valid = z["h1_valid"]             # (n_sz,) bool
    n_sz = cab.shape[0]

    # bold aligned median: over the H1-valid seizures if there are enough to form a stable
    # median, else a descriptive fall-back over ALL kept seizures (labelled as such).
    enough_valid = int(h1_valid.sum()) >= MIN_N_MED
    use = h1_valid if enough_valid else np.ones(n_sz, bool)
    aligned = np.where(present, cab * align[:, None], np.nan)   # flip each seizure's side up
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN columns -> NaN
        n_present_use = present[use].sum(axis=0)
        med = np.nanmedian(aligned[use], axis=0)
    med = np.where(n_present_use >= MIN_N_MED, med, np.nan)      # only where well-supported
    frac_present = present.mean(axis=0)                          # over all kept seizures

    fig, (ax, axf) = plt.subplots(
        2, 1, figsize=(7.4, 4.8), sharex=True,
        gridspec_kw={"height_ratios": [4.0, 1.0]}, layout="constrained")

    # side cue: faint A(+)/B(-) bands (interpretation of the thin true-C_AB lines).
    ax.axhspan(0, 1.05, color=COL_A, alpha=0.045, lw=0, zorder=0)
    ax.axhspan(-1.05, 0, color=COL_B, alpha=0.045, lw=0, zorder=0)

    # thin lines: each seizure's TRUE C_AB where axis_present (gaps where not).
    cab_present = np.where(present, cab, np.nan)
    for i in range(n_sz):
        ax.plot(centers, cab_present[i], color=COL_SZ, lw=0.7, alpha=0.35, zorder=2)
    ax.plot([], [], color=COL_SZ, lw=1.0, alpha=0.6, label=f"individual seizures (n={n_sz})")

    # bold aligned median.
    med_label = (f"near-onset-side-aligned median (H1-valid, n={int(use.sum())})" if enough_valid
                 else f"near-onset-side-aligned median (all sz, n={int(use.sum())}; H1 not eligible)")
    ax.plot(centers, med, color=COL_MED, lw=2.6, zorder=5, label=med_label)

    ax.axhline(0, color="0.35", lw=1.0, zorder=1)
    ax.axvline(0, color="0.30", ls="--", lw=1.0, zorder=1)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(float(centers.min()), float(centers.max()))
    ax.set_ylabel("A/B contrast  $C_{AB}$", fontsize=FS_LABEL)
    ax.text(0.006, 0.985, "A source side", transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color=COL_A, alpha=0.85)
    ax.text(0.006, 0.015, "B source side", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8, color=COL_B, alpha=0.85)
    ax.set_title(f"{_pretty(ds_sid)}  ·  interictal A/B scaffold contrast, peri-onset",
                 fontsize=FS_LABEL + 1)
    ax.legend(frameon=False, loc="lower right", fontsize=8, handlelength=1.6)

    h1 = summary["H1"]
    locked = "yes" if h1["subject_locked"] else "no"
    p = h1["p"]
    lobs = h1["L_obs"]
    box = (f"pair: {summary['template_pair_tier']}  (ρ$_{{AB}}$={summary['rho_AB']:+.2f}, "
           f"n$_{{joint}}$={summary['n_joint']})\n"
           f"H1 valid seizures: {h1['n_valid_seizures']} / {n_sz}"
           f"{'' if h1['H1_eligible'] else '  (< 3: not eligible)'}\n"
           f"L$_{{obs}}$={lobs:.3f}   subject-locked: {locked}"
           f"{'' if p is None else f'   p={p:.3f}'}")
    ax.text(0.006, 0.02 + 0.115, box, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7.6, color="0.15",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", alpha=0.9))

    # bottom: axis_present fraction per window (all kept seizures).
    axf.fill_between(centers, 0, frac_present, color="#6a51a3", alpha=0.30, lw=0, step="mid")
    axf.step(centers, frac_present, where="mid", color="#6a51a3", lw=1.1)
    axf.axvline(0, color="0.30", ls="--", lw=1.0)
    axf.set_ylim(0, 1.0)
    axf.set_ylabel("axis-present\nfraction", fontsize=FS_TICK - 1)
    axf.set_xlabel("time from clinical onset (s), window center", fontsize=FS_LABEL)
    if _HAVE_STYLE:
        ax.tick_params(labelsize=FS_TICK - 1)
        axf.tick_params(labelsize=FS_TICK - 1)

    _save(fig, out_png)
    return out_png


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", default="epilepsiae_1146")
    args = ap.parse_args()
    out = plot_subject(args.subject)
    print(out)


if __name__ == "__main__":
    main()
