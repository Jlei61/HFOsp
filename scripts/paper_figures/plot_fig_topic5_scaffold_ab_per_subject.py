#!/usr/bin/env python3
"""Paper-ready PER-SUBJECT figures for Topic5 V3d scaffold A/B lateral switching.

Two figure types per subject, restyled to the locked palette (docs/figure_style_guide.md
§0 + §5a) and to match scripts/paper_figures/plot_fig_topic5_scaffold_ab_cohort.py so the
per-subject and cohort figures read as one family:

  1. <ds_sid>_perseizure_dist.png  -- one row per seizure: two small box-and-whisker
     summaries of that seizure's axis-present C_AB values, one for the early (pre-ictal)
     time bin and one for the late (near-onset) time bin. Row data loading adapted from
     scripts/plot_topic5_scaffold_ab_perseizure_distribution.py; rendering is bespoke.
     Shows whether a subject's seizures split into A-dominant vs B-dominant states, and
     whether any given seizure's side shifts between its early and late windows.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

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
COL_EARLY = "#440154"    # dark viridis -- early (pre-ictal) per-seizure time-bin box
COL_LATE = "#FDE725"     # bright viridis -- late (near-onset) per-seizure time-bin box

DELTA_SIDE = 0.2         # |C_AB| side-label threshold (spec-locked)
MIN_N_VIOLIN = 6         # min axis-present windows to not be "low data" (row-group taxonomy)
BIMODAL_FRAC_MIN = 0.25  # both-side mass fraction threshold for the "bimodal" row group
MIN_N_MED = 4            # min axis_present seizures per window to draw the aligned median/IQR
EARLY_LATE_SPLIT_S = -30.0  # window-center split: early (pre-ictal) < -30s <= late (near-onset)
MIN_N_BOX = 3            # min windows in a time-bin to draw a box; below this, draw dots

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

# The timecourse figure needs a contiguous run of onset-adjacent, well-supported (>=4
# seizures/window) time windows to read as a trend rather than disconnected fragments.
# Only these two subjects have that: E442 (near-onset-locked, dense) and E1146 (dense,
# not locked -- the honest contrast). The other 10 subjects' axis-present windows are too
# sparse for a legible timecourse; render_subject() still renders their distribution
# figure (the all-subject deliverable), just not the timecourse.
TIMECOURSE_SUBJECTS = ["epilepsiae_442", "epilepsiae_1146"]


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
# Figure type 1 -- per-seizure distribution (early-vs-late paired boxes)
# ============================================================================

BOX_OFFSET = 0.16   # |y| offset of each time-bin box/dots from the row center
BOX_HALF_H = 0.085  # box half-height (IQR rectangle)


def _draw_time_bin(ax, vals: np.ndarray, y_center: float, color: str, rng: np.random.Generator) -> None:
    """Draw one seizure's one time-bin (early or late) at y_center: a 5/25/50/75/95
    box-and-whisker when n>=MIN_N_BOX, else individual dots (too few points for a box
    to mean anything).
    """
    n = len(vals)
    if n == 0:
        return
    if n < MIN_N_BOX:
        jit = rng.uniform(-0.025, 0.025, size=n) if n > 1 else np.zeros(1)
        ax.scatter(vals, y_center + jit, s=13, color=color, edgecolors="0.15",
                   linewidths=0.4, alpha=0.95, zorder=5)
        return
    q5, q25, med, q75, q95 = np.percentile(vals, [5, 25, 50, 75, 95])
    cap_h = BOX_HALF_H * 0.55
    ax.plot([q5, q95], [y_center, y_center], color="0.3", lw=0.9, zorder=4)
    ax.plot([q5, q5], [y_center - cap_h, y_center + cap_h], color="0.3", lw=0.9, zorder=4)
    ax.plot([q95, q95], [y_center - cap_h, y_center + cap_h], color="0.3", lw=0.9, zorder=4)
    ax.add_patch(Rectangle((q25, y_center - BOX_HALF_H), max(q75 - q25, 1e-3), 2 * BOX_HALF_H,
                            facecolor=color, edgecolor="0.15", lw=0.7, alpha=0.9, zorder=5))
    tick_color = "white" if color == COL_EARLY else "0.1"
    ax.plot([med, med], [y_center - BOX_HALF_H, y_center + BOX_HALF_H], color=tick_color,
            lw=1.4, zorder=6)


def plot_distribution(ds_sid: str, rows: list[dict]) -> Path:
    out_png = OUT_DIR / f"{ds_sid}_perseizure_dist.png"

    n_rows = len(rows)
    row_h = 0.85  # must stay large enough that a row-group tag line (text-only, drawn in the
                  # outside-axes label margin, see below) never touches this row's own label
                  # line or the previous row's -- smaller row_h makes text overlap at fixed font size
    y = -row_h * np.arange(n_rows, dtype=float)  # row 0 -> y=0 (top row)

    fig_h = max(2.4, n_rows * 0.335 + 1.2)
    fig, ax = plt.subplots(figsize=(10.6, fig_h))

    ax.set_xlim(-1.05, 1.05)
    y_top = 0.70  # headroom above row 0 so the A/B header text clears row 0's own late
                  # box (offset up to +BOX_OFFSET+BOX_HALF_H=0.245) and the first row-group tag
    y_bot = y[-1] - 0.40
    ax.set_ylim(y_bot, y_top)

    ax.axvspan(0, 1.05, color=COL_A, alpha=0.06, lw=0, zorder=0)
    ax.axvspan(-1.05, 0, color=COL_B, alpha=0.06, lw=0, zorder=0)
    ax.axvline(0, color="0.25", lw=1.2, zorder=1)
    ax.axvline(DELTA_SIDE, color="0.45", lw=0.8, ls="--", alpha=0.7, zorder=1)
    ax.axvline(-DELTA_SIDE, color="0.45", lw=0.8, ls="--", alpha=0.7, zorder=1)
    ax.text(0.5, y_top - 0.025, "A source side", ha="center", va="top",
            fontsize=9, color=COL_A, alpha=0.85)
    ax.text(-0.5, y_top - 0.025, "B source side", ha="center", va="top",
            fontsize=9, color=COL_B, alpha=0.85)

    rng = np.random.default_rng(0)  # visual jitter only (sparse-bin dots), fixed for reproducibility
    for i, d in enumerate(rows):
        yi = y[i]
        c_vals, t_vals, n = d["c_present"], d["t_present"], d["n_present"]

        if n > 0:
            early_mask = t_vals < EARLY_LATE_SPLIT_S
            _draw_time_bin(ax, c_vals[early_mask], yi - BOX_OFFSET, COL_EARLY, rng)
            _draw_time_bin(ax, c_vals[~early_mask], yi + BOX_OFFSET, COL_LATE, rng)

        ax.text(-0.012, yi, f"sz{d['seizure_idx']} · n={n}",
                transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8,
                clip_on=False)

        new_group = (i == 0) or (d["group"] != rows[i - 1]["group"])
        if new_group:
            # Drawn in the same outside-axes label margin as the "sz<idx> ..." row labels
            # (not at x=-1.0 inside the data area) so it can never collide with a B-dominant
            # row's own box, which legitimately sits near C_AB=-1.
            ax.text(-0.012, yi + row_h / 2.0, GROUP_LABEL[d["group"]],
                    transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=9.5,
                    fontweight="bold", color=GROUP_TEXT_COLOR[d["group"]], zorder=5, clip_on=False)
            if i > 0:
                ax.axhline((y[i] + y[i - 1]) / 2.0, color="0.6", lw=0.7, alpha=0.6, zorder=2)

    ax.set_yticks([])
    for spine in ("left", "top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_xticks([-1, -0.5, -0.2, 0, 0.2, 0.5, 1])
    ax.set_xlabel(r"$C_{AB}$   (A source side  +1  $\leftrightarrow$  $-1$  B source side)",
                  fontsize=11)
    ax.set_title(f"{_pretty(ds_sid)} · per-seizure A/B state distribution",
                 fontsize=13, fontweight="bold", pad=28)

    # Anchored just above the axes (outside the data area, in axes-fraction coordinates) so
    # it can never collide with a row's boxes, however close row 0's own values sit to +-1
    # (a data-anchored corner legend does collide for subjects like E916 sz6, whose early/late
    # boxes sit right at the top-right corner near C_AB=+1).
    legend_handles = [
        Patch(facecolor=COL_EARLY, edgecolor="0.15", label="early (pre-ictal)"),
        Patch(facecolor=COL_LATE, edgecolor="0.15", label="late (near-onset)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", bbox_to_anchor=(1.0, 1.01),
              frameon=False, fontsize=8, handlelength=1.2, ncol=2, columnspacing=1.2,
              borderaxespad=0)

    return _save(fig, out_png)


# ============================================================================
# Figure type 2 -- C_AB(t) timecourse
# ============================================================================


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Index (start, stop) pairs (stop exclusive) for each contiguous run of True in mask."""
    runs = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i + 1
            while j < n and mask[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


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

    # thin lines: each seizure's TRUE C_AB where axis_present (gaps where not). Kept very
    # faint -- these are background context, not the readable signal (that's the median).
    cab_present = np.where(present, cab, np.nan)
    for i in range(n_sz):
        ax.plot(centers, cab_present[i], color=COL_SZ, lw=0.5, alpha=0.12, zorder=2)

    # Bold median + IQR band: draw ONLY across contiguous runs of well-supported windows
    # (>=MIN_N_MED seizures AND consecutive grid steps). A window that drops below support
    # or a real time gap breaks the line -- never bridge the median across sparse/missing
    # windows, which would fabricate a trend between unrelated seizure subsets.
    for i0, i1 in _contiguous_runs(well_supported):
        if i1 - i0 < 2:
            continue  # single isolated window -- no segment to draw
        seg_t = centers[i0:i1]
        ax.fill_between(seg_t, q25[i0:i1], q75[i0:i1], color=COL_MED, alpha=0.15, lw=0, zorder=3)
        ax.plot(seg_t, med[i0:i1], color=COL_MED, lw=2.6, zorder=5)

    med_label = (f"near-onset-side-aligned median ({int(use.sum())} onset-testable seizures)"
                 if enough_valid else
                 f"near-onset-side-aligned median (all {int(use.sum())} seizures; too few pass "
                 "the near/far data gate)")

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


def render_subject(ds_sid: str, render_timecourse: bool = True) -> dict:
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
    if render_timecourse:
        p2 = plot_timecourse(ds_sid, npz, summary)
        print(f"{ds_sid}: wrote {p1.name}, {p2.name}  (n_seizures={len(df)}, "
              f"n_present_total={n_present_total})")
    else:
        print(f"{ds_sid}: wrote {p1.name}  (timecourse skipped -- axis-present windows too "
              f"sparse for a legible trend; n_seizures={len(df)}, n_present_total={n_present_total})")
    return {"subject": ds_sid, "status": "ok", "n_seizures": int(len(df)),
            "n_present_total": n_present_total, "timecourse": render_timecourse}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--subject", help="single ds_sid, e.g. epilepsiae_1146")
    g.add_argument("--all", action="store_true",
                   help=f"render the locked informative-pool subjects ({len(DEFAULT_SUBJECTS)}); "
                        f"timecourse only for {TIMECOURSE_SUBJECTS}")
    args = ap.parse_args()

    if args.all:
        results = [render_subject(s, render_timecourse=(s in TIMECOURSE_SUBJECTS))
                   for s in DEFAULT_SUBJECTS]
    else:
        results = [render_subject(args.subject, render_timecourse=True)]

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_tc = sum(1 for r in results if r.get("timecourse"))
    print(f"\n{n_ok}/{len(results)} subjects rendered (distribution: {n_ok}, "
          f"timecourse: {n_tc}) -> {OUT_DIR}")


if __name__ == "__main__":
    main()
