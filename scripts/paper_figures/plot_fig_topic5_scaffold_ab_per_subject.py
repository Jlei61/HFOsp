#!/usr/bin/env python3
"""Paper-ready PER-SUBJECT figures for Topic5 V3d scaffold A/B lateral switching.

Two figure types per subject, restyled to the locked palette (docs/figure_style_guide.md
§0 + §5a) and to match scripts/paper_figures/plot_fig_topic5_scaffold_ab_cohort.py so the
per-subject and cohort figures read as one family:

  1. <ds_sid>_perseizure_dist.png  -- one row per seizure, the distribution of axis-present
     C_AB values (raincloud). Adapted from scripts/plot_topic5_scaffold_ab_perseizure_distribution.py.
     Shows whether a subject's seizures split into A-dominant vs B-dominant states.
  2. <ds_sid>_timecourse.png       -- C_AB(t) peri-onset, thin per-seizure lines + bold
     near-onset-side-aligned median. Adapted from scripts/plot_topic5_scaffold_ab_contrast_timecourse.py.
     Shows whether lateral polarization strengthens approaching seizure onset.

This is a read-only downstream figure script: it does not touch
src/topic5_scaffold_ab_contrast.py or the two source plotters above, and does not
recompute anything -- every number is read verbatim from
results/topic5_ictal_recruitment/scaffold_ab_switching/per_subject/<ds_sid>_scaffold_ab_{contrast.npz,per_seizure.csv,summary.json}.

Output: results/paper-ready-figure/fig_topic5_scaffold_ab/figures/per_subject/
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
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.plot_style import FS_LABEL, FS_TICK, savefig_pub  # noqa: E402

DATA_DIR = ROOT / "results/topic5_ictal_recruitment/scaffold_ab_switching"
SUB_DIR = DATA_DIR / "per_subject"
OUT_DIR = ROOT / "results/paper-ready-figure/fig_topic5_scaffold_ab/figures/per_subject"

# ---------------------------------------------------------------------------
# Locked palette (docs/figure_style_guide.md §0 + this figure family's own spec).
# Same hex values as scripts/paper_figures/plot_fig_topic5_scaffold_ab_cohort.py
# so per-subject and cohort figures read as one family.
# ---------------------------------------------------------------------------
COL_A = "#B2182B"        # A source side / template A
COL_B = "#2166AC"        # B source side / template B
COL_BIMODAL = "#762A83"  # both sides seen (matches cohort figure's "bimodal" category)
COL_LOWDATA = "#BDBDBD"  # too few usable windows (matches cohort figure's "low_data" category)
COL_SZ = "0.72"          # individual seizures, timecourse
COL_MED = "#222222"      # near-onset-aligned median, timecourse

DELTA_SIDE = 0.2         # |C_AB| side-label threshold (spec-locked)
MIN_N_VIOLIN = 6         # min axis-present windows to draw a half-violin / not be "low data"
BIMODAL_FRAC_MIN = 0.25  # both-side mass fraction threshold for the "bimodal" row group
MIN_N_MED = 4            # min axis_present seizures per window to draw the aligned median/IQR

# Row-group taxonomy for the per-seizure distribution figure: which side does THIS
# seizure's own whole-window C_AB distribution favor. Independent of (and complementary
# to) the far-vs-near event_class taxonomy used for the per-row text label below.
GROUP_ORDER = {"A_dominant": 0, "bimodal": 1, "B_dominant": 2, "low_data": 3}
GROUP_LABEL = {"A_dominant": "A-dominant", "bimodal": "bimodal", "B_dominant": "B-dominant",
               "low_data": "low data"}
GROUP_COLOR = {"A_dominant": COL_A, "bimodal": COL_BIMODAL, "B_dominant": COL_B,
               "low_data": COL_LOWDATA}
# Text-label variant of GROUP_COLOR: COL_LOWDATA ("#BDBDBD") is legible as a marker/patch
# fill but too pale for a bold section-header string on a white background, so the "low
# data" section tag uses a darker neutral gray instead (marker/patch color unchanged).
GROUP_TEXT_COLOR = {**GROUP_COLOR, "low_data": "#8C8C8C"}

# Plain-language stand-ins for the far-vs-near event_class taxonomy (no code-words on
# any axis/legend/title -- see task instructions / CLAUDE.md §8).
EVENT_CLASS_PLAIN = {
    "persistent": "single-side",
    "switch": "bimodal (both sides)",
    "selection": "onset-side selection",
    "none": "no clear side",
}

DEFAULT_SUBJECTS = [
    "epilepsiae_1084", "epilepsiae_916", "epilepsiae_922", "epilepsiae_548",
    "epilepsiae_1146", "epilepsiae_442", "epilepsiae_635", "epilepsiae_1125",
    "epilepsiae_590", "epilepsiae_958", "epilepsiae_1096", "epilepsiae_1150",
]


def _pretty(ds_sid: str) -> str:
    """'epilepsiae_1146' -> 'E1146', 'yuquan_xuxinyi' -> 'Y-xuxinyi'."""
    if ds_sid.startswith("epilepsiae_"):
        return "E" + ds_sid[len("epilepsiae_"):]
    if ds_sid.startswith("yuquan_"):
        return "Y-" + ds_sid[len("yuquan_"):]
    return ds_sid


def _save(fig: plt.Figure, out_png: Path) -> Path:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    return savefig_pub(fig, out_png, dpi=200)


# ============================================================================
# Data loading (mirrors the two source plotters' loaders; read-only, no recompute)
# ============================================================================


def _load(ds_sid: str):
    """Load npz + per_seizure csv + summary json for one subject. None if any is missing."""
    npz_fp = SUB_DIR / f"{ds_sid}_scaffold_ab_contrast.npz"
    csv_fp = SUB_DIR / f"{ds_sid}_scaffold_ab_per_seizure.csv"
    json_fp = SUB_DIR / f"{ds_sid}_scaffold_ab_summary.json"
    if not (npz_fp.exists() and csv_fp.exists() and json_fp.exists()):
        return None
    npz = np.load(npz_fp)
    df = pd.read_csv(csv_fp)
    summary = json.loads(json_fp.read_text())
    return npz, df, summary


def _build_rows(npz, df: pd.DataFrame) -> list[dict]:
    """Per-seizure axis-present C_AB values + row group, for the distribution figure.

    Adapted from scripts/plot_topic5_scaffold_ab_perseizure_distribution.py::_load_rows:
    same frac_on_A / frac_on_B / bimodal_extremes fields, but sorted into the row-group
    taxonomy (A_dominant/bimodal/B_dominant/low_data) instead of the far-vs-near
    event_class taxonomy.
    """
    grid_centers = npz["grid_centers"]
    cab = npz["cab"]
    present = npz["present"]
    seizure_idx_npz = [int(v) for v in npz["seizure_idx"]]
    by_idx = {si: i for i, si in enumerate(seizure_idx_npz)}

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

        if n_present > 0:
            frac_on_A = float(np.mean(c_present >= DELTA_SIDE))
            frac_on_B = float(np.mean(c_present <= -DELTA_SIDE))
        else:
            frac_on_A = frac_on_B = 0.0
        bimodal_extremes = bool(frac_on_A >= BIMODAL_FRAC_MIN and frac_on_B >= BIMODAL_FRAC_MIN)

        if n_present < MIN_N_VIOLIN:
            group = "low_data"
        elif bimodal_extremes:
            group = "bimodal"
        elif frac_on_A >= frac_on_B:
            group = "A_dominant"
        else:
            group = "B_dominant"

        rows.append({
            "seizure_idx": si,
            "event_class": str(r["event_class"]),
            "n_present": n_present,
            "group": group,
            "c_present": c_present,
            "t_present": t_present,
        })

    rows.sort(key=lambda d: (GROUP_ORDER[d["group"]], d["seizure_idx"]))
    return rows


# ============================================================================
# Figure type 1 -- per-seizure distribution (raincloud)
# ============================================================================


def plot_distribution(ds_sid: str, rows: list[dict]) -> Path:
    out_png = OUT_DIR / f"{ds_sid}_perseizure_dist.png"

    n_rows = len(rows)
    cmap = plt.get_cmap("viridis")
    all_t = [d["t_present"] for d in rows if d["n_present"] > 0]
    if all_t:
        t_cat = np.concatenate(all_t)
        norm = Normalize(vmin=float(t_cat.min()), vmax=float(t_cat.max()))
    else:
        norm = Normalize(vmin=-115.0, vmax=15.0)

    row_h = 1.0
    violin_h = 0.34
    jitter_h = 0.28
    y = -row_h * np.arange(n_rows, dtype=float)  # row 0 -> y=0 (top row)

    fig_h = max(6.0, n_rows * 0.40 + 2.4)
    fig, ax = plt.subplots(figsize=(11.8, fig_h))

    ax.set_xlim(-1.05, 1.05)
    y_top = 0.70  # extra headroom above row 0 so the A/B header text and the first
                  # row-group tag (drawn just above row 0) don't crowd each other
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

        plain_ec = EVENT_CLASS_PLAIN.get(d["event_class"], d["event_class"])
        ax.text(-0.012, yi, f"sz{d['seizure_idx']} · {plain_ec} · n={n}",
                transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8,
                clip_on=False)

        new_group = (i == 0) or (d["group"] != rows[i - 1]["group"])
        if new_group:
            ax.text(-1.0, yi + 0.42, GROUP_LABEL[d["group"]], ha="left", va="center",
                    fontsize=9.5, fontweight="bold", color=GROUP_TEXT_COLOR[d["group"]], zorder=5)
            if i > 0:
                ax.axhline((y[i] + y[i - 1]) / 2.0, color="0.6", lw=0.7, alpha=0.6, zorder=2)

    ax.set_yticks([])
    for spine in ("left", "top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_xticks([-1, -0.5, -0.2, 0, 0.2, 0.5, 1])
    ax.set_xlabel(r"$C_{AB}$   (A source side  +1  $\leftrightarrow$  $-1$  B source side)",
                  fontsize=11)
    ax.set_title(f"{_pretty(ds_sid)} · per-seizure A/B state distribution",
                 fontsize=13, fontweight="bold", pad=10)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.022, aspect=45)
    cbar.set_label("time from onset (s)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.text(0.5, 0.005,
              "Half-violins: per-seizure peak-normalized KDE of axis-present C_AB (shape only; "
              f"skipped when n<{MIN_N_VIOLIN}). Points: every axis-present window, vertically "
              "jittered, colored by time from onset (dark=far pre-ictal, bright=onset). Dashed "
              f"lines at ±{DELTA_SIDE}: side-label threshold. Rows grouped by which side each "
              "seizure's own distribution favors (left margin: per-seizure far→near label).",
              ha="center", va="bottom", fontsize=7.5, color="0.35")

    return _save(fig, out_png)


# ============================================================================
# Figure type 2 -- C_AB(t) timecourse
# ============================================================================


def plot_timecourse(ds_sid: str, npz, summary: dict) -> Path:
    out_png = OUT_DIR / f"{ds_sid}_timecourse.png"

    centers = npz["grid_centers"]
    cab = npz["cab"]                    # (n_sz, n_win)
    present = npz["present"]            # (n_sz, n_win) bool
    align = npz["align_sign"]           # (n_sz,)
    h1_valid = npz["h1_valid"]          # (n_sz,) bool
    n_sz = cab.shape[0]

    # Bold aligned median + IQR band: over the near-onset-testable seizures if there are
    # enough to form a stable summary, else a descriptive fall-back over ALL kept seizures
    # (labelled as such). Do NOT plot a naive signed median (P9 in the spec) -- opposite-
    # side seizures would cancel; instead each seizure is flipped by the sign of its own
    # near-onset mean before combining.
    enough_valid = int(h1_valid.sum()) >= MIN_N_MED
    use = h1_valid if enough_valid else np.ones(n_sz, bool)
    aligned = np.where(present, cab * align[:, None], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN columns -> NaN
        n_present_use = present[use].sum(axis=0)
        med = np.nanmedian(aligned[use], axis=0)
        q25 = np.nanpercentile(aligned[use], 25, axis=0)
        q75 = np.nanpercentile(aligned[use], 75, axis=0)
    well_supported = n_present_use >= MIN_N_MED
    med = np.where(well_supported, med, np.nan)
    q25 = np.where(well_supported, q25, np.nan)
    q75 = np.where(well_supported, q75, np.nan)

    fig, ax = plt.subplots(figsize=(7.6, 4.6), layout="constrained")

    # side cue: faint A(+)/B(-) bands (interpretation of the thin true-C_AB lines).
    ax.axhspan(0, 1.05, color=COL_A, alpha=0.05, lw=0, zorder=0)
    ax.axhspan(-1.05, 0, color=COL_B, alpha=0.05, lw=0, zorder=0)

    # thin lines: each seizure's TRUE C_AB where axis_present (gaps where not).
    cab_present = np.where(present, cab, np.nan)
    for i in range(n_sz):
        ax.plot(centers, cab_present[i], color=COL_SZ, lw=0.7, alpha=0.35, zorder=2)

    # shaded band: cross-seizure IQR of the aligned quantity (style guide §5a convention).
    ax.fill_between(centers, q25, q75, color=COL_MED, alpha=0.15, lw=0, zorder=3)

    med_label = (f"near-onset-side-aligned median ({int(use.sum())} onset-testable seizures)"
                 if enough_valid else
                 f"near-onset-side-aligned median (all {int(use.sum())} seizures; too few pass "
                 "the near/far data gate)")
    ax.plot(centers, med, color=COL_MED, lw=2.6, zorder=5)

    ax.axhline(0, color="0.30", lw=1.1, zorder=1)
    ax.axhline(DELTA_SIDE, color="0.5", lw=0.8, ls="--", alpha=0.7, zorder=1)
    ax.axhline(-DELTA_SIDE, color="0.5", lw=0.8, ls="--", alpha=0.7, zorder=1)
    ax.axvline(0, color="0.30", ls="--", lw=1.2, zorder=1)

    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(float(centers.min()), float(centers.max()))
    ax.set_ylabel(r"A/B contrast  $C_{AB}$", fontsize=FS_LABEL - 2)
    ax.set_xlabel("time from clinical onset (s)", fontsize=FS_LABEL - 2)
    ax.text(0.994, 0.985, "A source side", transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color=COL_A, alpha=0.9)
    ax.text(0.006, 0.015, "B source side", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.5, color=COL_B, alpha=0.9)
    ax.set_title(f"{_pretty(ds_sid)} · peri-onset A/B contrast, per seizure",
                 fontsize=FS_LABEL + 1, fontweight="bold")
    ax.tick_params(labelsize=FS_TICK - 3)

    legend_handles = [
        Line2D([0], [0], color=COL_SZ, lw=1.2, alpha=0.7, label=f"individual seizures (n={n_sz})"),
        Patch(facecolor=COL_MED, alpha=0.15, label="aligned-median IQR"),
        Line2D([0], [0], color=COL_MED, lw=2.6, label=med_label),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="lower right", fontsize=7.8,
              handlelength=1.6)

    tier = summary.get("template_pair_tier", "?")
    rho = summary.get("rho_AB")
    h1 = summary["H1"]
    n_valid = int(h1.get("n_valid_seizures") or 0)
    if h1.get("H1_eligible"):
        locked_txt = "yes" if h1["subject_locked"] else "no"
        p = h1.get("p")
        lock_line = f"near-onset side-locking: {locked_txt}" + (f"  (p={p:.3f})" if p is not None else "")
    else:
        lock_line = f"near-onset side-locking: not testable (need ≥ 3 usable seizures, have {n_valid})"
    rho_txt = f"  (ρ={rho:+.2f})" if rho is not None else ""
    box = (f"template pair: {tier}{rho_txt}\n"
           f"seizures: {n_sz} total, {n_valid} usable for the near-onset test\n"
           f"{lock_line}")
    ax.text(0.006, 0.965, box, transform=ax.transAxes, ha="left", va="top", fontsize=7.6,
            color="0.15", bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", alpha=0.92),
            zorder=6)

    return _save(fig, out_png)


# ============================================================================
# Driver
# ============================================================================


def render_subject(ds_sid: str) -> dict:
    loaded = _load(ds_sid)
    if loaded is None:
        print(f"{ds_sid}: SKIPPED (missing per-subject artifacts)")
        return {"subject": ds_sid, "status": "skipped_missing_files"}
    npz, df, summary = loaded

    if summary.get("status") != "ok":
        print(f"{ds_sid}: SKIPPED (status={summary.get('status')}, "
              f"drop_reason={summary.get('drop_reason')})")
        return {"subject": ds_sid, "status": f"skipped_{summary.get('status')}"}

    n_present_total = int(npz["present"].sum())
    if n_present_total == 0:
        print(f"{ds_sid}: SKIPPED (0 axis-present windows across all seizures)")
        return {"subject": ds_sid, "status": "skipped_zero_windows"}

    rows = _build_rows(npz, df)
    p1 = plot_distribution(ds_sid, rows)
    p2 = plot_timecourse(ds_sid, npz, summary)
    print(f"{ds_sid}: wrote {p1.name}, {p2.name}  (n_seizures={len(df)}, "
          f"n_present_total={n_present_total})")
    return {"subject": ds_sid, "status": "ok", "n_seizures": int(len(df)),
            "n_present_total": n_present_total}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--subject", help="single ds_sid, e.g. epilepsiae_1146")
    g.add_argument("--all", action="store_true",
                   help=f"render the locked informative-pool subjects ({len(DEFAULT_SUBJECTS)})")
    args = ap.parse_args()

    subjects = DEFAULT_SUBJECTS if args.all else [args.subject]
    results = [render_subject(s) for s in subjects]

    n_ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n{n_ok}/{len(results)} subjects rendered ({2 * n_ok} PNGs) -> {OUT_DIR}")


if __name__ == "__main__":
    main()
