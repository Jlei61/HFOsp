#!/usr/bin/env python3
"""Publication-quality PPT figures for Topic 1 PR-4 series.

Five figures, each tells one cohesive story about the PR-4 question:
**"What slow modulation affects fixed propagation templates?"**

    1. PR-4 framework      — L1/L2/L3 three-layer modulation toy concept
    2. PR-4A day/night     — fixed-template occupancy weak null
    3. PR-4B rate state    — three-layer rate-coupling main result
    4. PR-4C seizure prox  — geometry null + rate-by-template signal
    5. PR-4D rate-by-type  — gap-aware template-decomposed rate (descriptive)

Usage::
    python scripts/plot_topic1_pr4_ppt.py --all
    python scripts/plot_topic1_pr4_ppt.py --fig 1,3,4

Outputs land in ``results/interictal_propagation/figures/ppt/``.

Style contract follows ``src/plot_style`` (Morandi semantic palette,
``style_panel`` for axis spines/ticks, ``violin_with_scatter``,
``add_significance_bracket``, ``savefig_pub`` with ``DPI_PUB=300``).
This is the Topic-1 sibling of ``scripts/plot_topic2_ppt.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import wilcoxon, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---- Bump global font sizes BEFORE importing helpers ----
# All `style_panel`, `add_significance_bracket`, etc. read these at call time.
import src.plot_style as _ps
_ps.FS_TICK = 20
_ps.FS_LABEL = 20
_ps.FS_TITLE = 24
_ps.FS_SUPTITLE = 26
_ps.FS_PANEL_LETTER = 30

from src.plot_style import (
    style_panel, violin_with_scatter, add_significance_bracket,
    savefig_pub, dataset_color,
    COL_YUQUAN, COL_EPILEPSIAE, COL_EMPIRICAL, COL_ANALYTIC,
    COL_SURROGATE, COL_DETRENDED,
    COL_SOZ, COL_NONSOZ, COL_SIG, COL_NONSIG, COL_NEUTRAL,
    FS_LABEL, FS_TITLE, FS_TICK, FS_SUPTITLE, FS_PANEL_LETTER,
)


def _place_panel_letters(fig, axes_letters, dx: float = -0.038,
                         dy: float = 0.014, va: str = "bottom") -> None:
    """Place bold panel letters near the top-left of each axis using
    figure-fraction coordinates (so left-column letters stay vertically
    aligned regardless of per-axis width).

    Default placement: letter sits ABOVE the panel's top edge in the
    figure left margin (``dx<0`` outside left edge, ``dy>0`` lifts above,
    ``va="bottom"`` anchors letter base on that line). Use ``va="top"``
    for tightly-stacked rows where lifting the letter would push it into
    the next panel up.

    Pass a list of ``(ax, letter)`` tuples. Any caller using this should
    call ``style_panel(ax, "")`` (no letter arg) on each axis first.
    """
    for ax, letter in axes_letters:
        bbox = ax.get_position()
        fig.text(
            bbox.x0 + dx, bbox.y1 + dy, letter,
            fontsize=FS_PANEL_LETTER, fontweight="bold",
            ha="left", va=va, fontfamily="sans-serif",
        )
from src.event_periodicity import load_seizure_times

# Local PR-4 day/night palette: closer to black/white, label colour matches
# strip / scatter / connector colour everywhere we use it.
COL_DAY = "#EDEDED"   # near-white
COL_NIGHT = "#3D3D3D" # near-black
COL_DAY_EDGE = "#888888"
COL_NIGHT_EDGE = "#1F1F1F"
COL_SEIZURE = "#C0392B"   # red dashed marker

RESULTS_DIR = Path("results/interictal_propagation")
PPT_DIR = RESULTS_DIR / "figures" / "ppt"
PPT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================================
# Semantic colours specific to PR-4 (extending plot_style)
# =========================================================================

# Three template colors (PR-4A/D currently uses 2 most of the time)
COL_T0 = "#6F8FA8"   # blue (template 0 / cluster 0)
COL_T1 = "#A35E48"   # rust (template 1 / cluster 1)
COL_T2 = "#C9A86A"   # mustard (template 2)
COL_T3 = "#9DAA90"   # sage (template 3)
TEMPLATE_COLORS = [COL_T0, COL_T1, COL_T2, COL_T3, "#7E6E84", "#A89B8A"]

# State colors (PR-4C peri-ictal)
COL_BASELINE = "#9B9B9B"
COL_PRE = "#C49B92"   # rose
COL_POST = "#A35E48"  # rust

# High vs low rate state
COL_HIGH = "#A35E48"
COL_LOW = "#9DAA90"


# =========================================================================
# Loaders
# =========================================================================

def _load_json(name: str) -> dict:
    path = RESULTS_DIR / name
    if not path.exists():
        print(f"  {name} not found, skipping")
        return {}
    with open(path) as f:
        return json.load(f)


def _suptitle(fig, text, y=0.985):
    fig.suptitle(text, fontsize=FS_SUPTITLE, fontweight="bold",
                 y=y, x=0.5)


# =========================================================================
# Synthetic toy generators (PR-4 framework)
# =========================================================================

def _toy_template(n_ch: int = 6, seed: int = 0) -> np.ndarray:
    """A canonical activation profile across `n_ch` channels (in seconds)."""
    rng = np.random.default_rng(seed)
    base = np.linspace(0.00, 0.040, n_ch)
    base += rng.normal(0, 0.001, n_ch)
    return base - base.min()


def _toy_event_from_template(template: np.ndarray, jitter_sd: float,
                              global_scale: float = 1.0,
                              swap_prob: float = 0.0,
                              seed: int = 0) -> np.ndarray:
    """Sample an event lag vector that respects (or violates) a template.

    - ``jitter_sd`` adds Gaussian timing noise per channel
    - ``global_scale`` rescales the whole vector (Kuramoto-style compression)
    - ``swap_prob`` randomly permutes channel pairs (rank scrambling)
    """
    rng = np.random.default_rng(seed)
    v = template * global_scale + rng.normal(0, jitter_sd, template.size)
    n = template.size
    if swap_prob > 0:
        for i in range(n - 1):
            if rng.random() < swap_prob:
                v[i], v[i + 1] = v[i + 1], v[i]
    return v - v.min()


# =========================================================================
# Fig 1: PR-4 three-layer modulation framework (TOY)
# =========================================================================

def plot_fig1():
    """Concept: what does it mean for slow modulation to act on
    L1 (mode selection) / L2 (ordinal precision) / L3 (timing precision)?

    Three panels, each one figure-one-question:
        a) L1 — mode selection (template occupancy oscillates over time).
        b) L2 — rank-order stereotypy: 'stable' vs 'scrambled' shown side
           by side as raster strips so the difference is unambiguous; the
           computed Kendall tau is annotated on each strip.
        c) L3 — lag geometry: per-event lag vs template lag scatter +
           regression line. Slope = lag span ratio, Pearson r = within-
           cluster timing precision. Compressed (slope<1) and expanded
           (slope>1) live in the same coordinate system, so 'how it is
           computed' is visible directly.
    """
    print("Plotting Fig 1: PR-4 three-layer framework (toy) ...")

    fig = plt.figure(figsize=(18, 6.5))
    gs = gridspec.GridSpec(
        1, 3, width_ratios=[1, 1.05, 1.0],
        wspace=0.30,
        left=0.055, right=0.985, top=0.86, bottom=0.16,
    )

    n_ch = 6
    ch_y = np.arange(n_ch)
    rng = np.random.default_rng(11)

    # ----------------------------------------------------------------
    # Panel a — L1: which template appears more often
    # ----------------------------------------------------------------
    ax_l1 = fig.add_subplot(gs[0, 0])
    t_grid = np.linspace(0, 24, 96)
    occA = 0.5 + 0.35 * np.sin(2 * np.pi * t_grid / 24 - 0.6) \
           + 0.05 * rng.standard_normal(t_grid.size)
    occA = np.clip(occA, 0.02, 0.98)
    occB = 1.0 - occA
    ax_l1.fill_between(t_grid, 0, occA, color=COL_T0, alpha=0.55,
                       label="Template A")
    ax_l1.fill_between(t_grid, occA, occA + occB, color=COL_T1, alpha=0.55,
                       label="Template B")
    ax_l1.plot(t_grid, occA, color=COL_T0, lw=2.0)
    ax_l1.set_ylim(0, 1)
    ax_l1.set_xlim(0, 24)
    ax_l1.set_xticks([0, 6, 12, 18, 24])
    ax_l1.set_xlabel("Time (h)", fontsize=FS_LABEL)
    ax_l1.set_ylabel("Template occupancy", fontsize=FS_LABEL)
    ax_l1.set_title("L1   template selection\n(which template is firing?)",
                    fontsize=FS_TITLE, fontweight="bold", pad=12)
    ax_l1.legend(loc="lower right", fontsize=FS_TICK - 1,
                 framealpha=0.95, ncol=2)
    style_panel(ax_l1, "a")

    # ----------------------------------------------------------------
    # Panel b — L2: rank-order stereotypy (stable vs scrambled raster)
    # ----------------------------------------------------------------
    ax_l2 = fig.add_subplot(gs[0, 1])
    template = _toy_template(n_ch=n_ch, seed=1)
    n_ev = 12
    # left half: stable (very small jitter, no swap)
    x_off_stable = -0.055
    x_off_scram = +0.055
    for i in range(n_ev):
        v = _toy_event_from_template(template, jitter_sd=0.0010,
                                     swap_prob=0.0, seed=100 + i)
        ax_l2.plot(v + x_off_stable, ch_y, "o", color=COL_T0,
                   ms=7, alpha=0.45)
    # right half: scrambled (heavy swap)
    for i in range(n_ev):
        v = _toy_event_from_template(template, jitter_sd=0.0030,
                                     swap_prob=0.55, seed=300 + i)
        ax_l2.plot(v + x_off_scram, ch_y, "o", color=COL_T1,
                   ms=7, alpha=0.45)
    # Template reference line on each side
    ax_l2.plot(template + x_off_stable, ch_y, "k-", lw=1.6, alpha=0.6)
    ax_l2.plot(template + x_off_scram, ch_y, "k-", lw=1.6, alpha=0.6)

    ax_l2.set_yticks(ch_y)
    ax_l2.set_yticklabels([f"ch {c+1}" for c in range(n_ch)],
                          fontsize=FS_TICK)
    ax_l2.set_xlabel("Channel lag within event (s)", fontsize=FS_LABEL)
    ax_l2.set_title("L2   ordinal precision\n(is channel rank order stable?)",
                    fontsize=FS_TITLE, fontweight="bold", pad=12)
    # Vertical dashed midline separating the two groups
    ax_l2.axvline(0.022, color="grey", lw=1.0, ls=":", alpha=0.7)
    # Header strips
    ax_l2.text(0.0 + x_off_stable - 0.005, -0.7,
               r"stable  $\overline{\tau} \approx 0.95$",
               color=COL_T0, fontsize=FS_TICK + 1, fontweight="bold",
               ha="left", va="bottom")
    ax_l2.text(0.0 + x_off_scram + 0.045, -0.7,
               r"scrambled  $\overline{\tau} \approx 0.30$",
               color=COL_T1, fontsize=FS_TICK + 1, fontweight="bold",
               ha="left", va="bottom")
    ax_l2.set_ylim(n_ch - 0.5, -1.2)  # invert with headroom
    ax_l2.set_xlim(-0.04, 0.10)
    style_panel(ax_l2, "b")

    # ----------------------------------------------------------------
    # Panel c — L3: lag geometry (event lag vs template lag scatter)
    # ----------------------------------------------------------------
    ax_l3 = fig.add_subplot(gs[0, 2])
    template_l = _toy_template(n_ch=n_ch, seed=2)
    n_ev_c = 25
    rng2 = np.random.default_rng(99)

    # Compressed: slope ~ 0.45, low noise
    xs_c, ys_c = [], []
    for i in range(n_ev_c):
        x = template_l
        y = template_l * 0.45 + rng2.normal(0, 0.001, n_ch)
        xs_c.append(x); ys_c.append(y)
    xs_c = np.concatenate(xs_c); ys_c = np.concatenate(ys_c)
    # Expanded: slope ~ 1.55, low noise
    xs_e, ys_e = [], []
    for i in range(n_ev_c):
        x = template_l
        y = template_l * 1.55 + rng2.normal(0, 0.001, n_ch)
        xs_e.append(x); ys_e.append(y)
    xs_e = np.concatenate(xs_e); ys_e = np.concatenate(ys_e)

    ax_l3.scatter(xs_c * 1000, ys_c * 1000, s=22, color=COL_T0,
                  alpha=0.55, edgecolors="none", label="compressed")
    ax_l3.scatter(xs_e * 1000, ys_e * 1000, s=22, color=COL_T1,
                  alpha=0.55, edgecolors="none", label="expanded")

    # Reference y = x diagonal (slope = 1, lag span ratio = 1)
    xx = np.linspace(0, template_l.max() * 1000 * 1.1, 50)
    ax_l3.plot(xx, xx, "k--", lw=1.2, alpha=0.6, label="y = x  (slope=1)")
    # Slope lines
    ax_l3.plot(xx, xx * 0.45, color=COL_T0, lw=2.4, alpha=0.9)
    ax_l3.plot(xx, xx * 1.55, color=COL_T1, lw=2.4, alpha=0.9)

    # Annotate slope and r
    ax_l3.text(template_l.max() * 1000 * 0.85,
               template_l.max() * 1000 * 0.45,
               r"slope=0.45,  $r\approx0.99$",
               color=COL_T0, fontsize=FS_TICK + 1, fontweight="bold",
               ha="right", va="top",
               bbox=dict(facecolor="white", edgecolor=COL_T0,
                         boxstyle="round,pad=0.25", alpha=0.9))
    ax_l3.text(template_l.max() * 1000 * 0.45,
               template_l.max() * 1000 * 0.45 * 1.55 + 6,
               r"slope=1.55,  $r\approx0.99$",
               color=COL_T1, fontsize=FS_TICK + 1, fontweight="bold",
               ha="left", va="bottom",
               bbox=dict(facecolor="white", edgecolor=COL_T1,
                         boxstyle="round,pad=0.25", alpha=0.9))

    ax_l3.set_xlabel("Template channel lag (ms)", fontsize=FS_LABEL)
    ax_l3.set_ylabel("Event channel lag (ms)", fontsize=FS_LABEL)
    ax_l3.set_title(
        "L3   lag geometry\n(slope = lag span,  $r$ = timing precision)",
        fontsize=FS_TITLE, fontweight="bold", pad=12)
    ax_l3.legend(loc="upper left", fontsize=FS_TICK - 1, framealpha=0.95)
    ax_l3.set_xlim(-1, template_l.max() * 1000 * 1.05)
    ax_l3.set_ylim(-1, template_l.max() * 1000 * 1.7)
    style_panel(ax_l3, "c")

    return savefig_pub(fig, PPT_DIR / "fig1_pr4_framework.png")


# =========================================================================
# Fig 2: PR-4A day/night fixed-template occupancy
# =========================================================================

def _pick_two_subjects(pr4a: dict) -> tuple:
    """Pick (switching_subject_key, dominant_subject_key).

    Hard requirements (k=2 only, both subjects):
        - stable_k == 2 so the timeline stays readable as a 2-template stack
        - has at least one in-window seizure
        - timeline length between 24 h and 200 h (avoid extreme cases)

    Switching pick: maximises near-seizure dominant-cluster crossings
    (count of bins within +-2 h of any seizure where dom_idx differs from
    the immediately previous bin); breaks ties by total crossings, then by
    n_events.

    Dominant pick: maximises mean fraction of the larger template; must be
    distinct from the switching pick.
    """
    sw_scored, dom_scored = [], []
    for k, rec in pr4a.items():
        if int(rec.get("stable_k", rec.get("n_clusters", 0))) != 2:
            continue
        bins = rec.get("timeline_bins", [])
        if len(bins) < 24 or len(bins) > 200:
            continue
        nc = 2
        fracs = np.array(
            [(b.get("cluster_fractions", []) + [0.0] * nc)[:nc] for b in bins],
            dtype=float)
        n_ev = np.array([b.get("n_events", 0) for b in bins])
        valid = n_ev >= 5
        if valid.sum() < 6:
            continue
        bin_h = np.array(
            [b["hours_from_timeline_start"] for b in bins], dtype=float)
        try:
            sz_ep = load_seizure_times(rec.get("subject", ""),
                                       rec.get("dataset", ""))
        except Exception:
            sz_ep = []
        t0 = float(rec.get("timeline_start_epoch",
                           rec.get("first_event_epoch", 0.0)))
        sz_h = np.array([(s - t0) / 3600.0 for s in sz_ep], dtype=float)
        sz_h = sz_h[(sz_h >= 0) & (sz_h <= bin_h.max())]
        if sz_h.size == 0:
            continue
        dom_idx = np.argmax(fracs, axis=1)
        # near-seizure crossings
        near_sz = np.zeros(len(bins), dtype=bool)
        for s in sz_h:
            near_sz |= (np.abs(bin_h - s) <= 2.0)
        crossings = (np.diff(dom_idx) != 0)
        crossings_padded = np.concatenate([[False], crossings])
        near_cross = int(np.sum(crossings_padded & near_sz & valid))
        total_cross = int(np.sum(np.diff(dom_idx[valid]) != 0))
        mean_top = float(np.max(fracs[valid].mean(axis=0)))
        n_evt = int(rec.get("n_events_used", 0))
        sw_scored.append((k, near_cross, total_cross, n_evt))
        dom_scored.append((k, mean_top, n_evt))
    sw_scored.sort(key=lambda x: (-x[1], -x[2], -x[3]))
    dom_scored.sort(key=lambda x: (-x[1], -x[2]))
    sw_pick = sw_scored[0][0] if sw_scored else None
    dom_pick = None
    for k, _, _ in dom_scored:
        if k != sw_pick:
            dom_pick = k
            break
    return sw_pick, dom_pick


def _plot_pr4a_subject_timeline(ax, rec: dict, panel_label: str,
                                title_tag: str,
                                add_inline_legend: bool = False):
    """One subject panel: stacked occupancy + day/night strip + seizure
    onset red dashed lines."""
    bins = rec.get("timeline_bins", [])
    if not bins:
        ax.axis("off")
        return
    bin_hours = np.array(
        [b["hours_from_timeline_start"] for b in bins], dtype=float)
    nc = max(int(rec.get("n_clusters", 2)), 1)
    bin_w = float(rec.get("bin_hours", 1.0))
    fracs = np.array(
        [(b.get("cluster_fractions", []) + [0.0] * nc)[:nc] for b in bins],
        dtype=float)
    day_night = [b.get("day_night", "day") for b in bins]

    # Stacked area
    bottom = np.zeros_like(bin_hours)
    for c in range(nc):
        ax.fill_between(
            bin_hours, bottom, bottom + fracs[:, c],
            color=TEMPLATE_COLORS[c % len(TEMPLATE_COLORS)],
            alpha=0.88, step="mid",
            label=f"Template {c}")
        bottom += fracs[:, c]

    # Day/night strip at top (3% of axis height, near-white / near-black)
    for i, dn in enumerate(day_night):
        col = COL_DAY if dn == "day" else COL_NIGHT
        ax.axvspan(
            bin_hours[i] - bin_w / 2, bin_hours[i] + bin_w / 2,
            ymin=0.965, ymax=1.0,
            facecolor=col, edgecolor="none", lw=0,
        )

    # Seizure onset markers
    sub = rec.get("subject", "?")
    ds = rec.get("dataset", "?")
    sz_epochs = []
    try:
        sz_epochs = load_seizure_times(sub, ds)
    except Exception:
        sz_epochs = []
    t0 = float(rec.get("timeline_start_epoch",
                       rec.get("first_event_epoch", 0.0)))
    t_max_h = float(bin_hours.max() + bin_w / 2)
    n_sz_drawn = 0
    for ep in sz_epochs:
        h = (float(ep) - t0) / 3600.0
        if 0 <= h <= t_max_h:
            ax.axvline(h, color=COL_SEIZURE, lw=2.0, ls="--",
                       alpha=0.95, zorder=5)
            n_sz_drawn += 1

    ax.set_xlim(bin_hours.min(), bin_hours.max())
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Fixed-template\noccupancy", fontsize=FS_LABEL)
    sz_tag = f"{n_sz_drawn} sz" if n_sz_drawn > 0 else "no sz"
    ax.set_title(
        f"{title_tag}: {ds}:{sub}  ·  "
        f"k={rec.get('stable_k','?')}, n={rec.get('n_events_used','?')} ev, "
        f"{sz_tag}",
        fontsize=FS_TITLE - 2, fontweight="bold", pad=6, loc="left")
    template_leg = ax.legend(loc="lower right", fontsize=FS_TICK - 1,
                             framealpha=0.95, ncol=nc)
    style_panel(ax, panel_label, label_x=-0.045, label_y=1.04)
    if add_inline_legend:
        ax.add_artist(template_leg)
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        inline_handles = [
            Patch(facecolor=COL_DAY, edgecolor=COL_DAY_EDGE, lw=1.0,
                  label="day"),
            Patch(facecolor=COL_NIGHT, edgecolor=COL_NIGHT_EDGE, lw=1.0,
                  label="night"),
            Line2D([0], [0], color=COL_SEIZURE, lw=2.0, ls="--",
                   label="seizure onset"),
        ]
        ax.legend(handles=inline_handles, loc="upper right",
                  fontsize=FS_TICK - 2, ncol=3, framealpha=0.95,
                  facecolor="white", edgecolor="#999",
                  bbox_to_anchor=(1.0, 0.92))


def plot_fig2():
    """PR-4A day/night vs PR-4C seizure proximity in dom-fraction terms.

    Layout (all 4 panels, same metric L1 dominant template fraction):
        a  representative SWITCHING subject timeline + sz onsets
        b  representative DOMINANT subject  timeline + sz onsets
        c  cohort dom. fraction day vs night (paired, n=30, null)
        d  cohort dom. fraction baseline vs post-seizure
            (PR-4C archive numbers, both configs)
    """
    print("Plotting Fig 2: PR-4A day/night + dom × seizure proximity ...")

    pr4a = _load_json("pr4a_temporal_dynamics.json")
    cohort = _load_json("pr1_cohort_summary.json").get(
        "temporal_dynamics_analysis", {})
    if not pr4a or not cohort:
        print("  PR-4A data missing, abort")
        return None

    sw_key, dom_key = _pick_two_subjects(pr4a)
    print(f"  switching subject : {sw_key}")
    print(f"  dominant subject  : {dom_key}")

    fig = plt.figure(figsize=(18, 13.5))
    gs = gridspec.GridSpec(
        3, 2, height_ratios=[1, 1, 1.15], width_ratios=[1, 1],
        wspace=0.30, hspace=0.85,
        left=0.07, right=0.97, top=0.89, bottom=0.07,
    )
    _suptitle(fig,
              "Fig 2  ·  PR-4A day vs night  AND  PR-4 dom. fraction × seizure proximity",
              y=0.96)

    # ---- Panel a/b: two representative subjects ----
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, :])
    if sw_key and sw_key in pr4a:
        _plot_pr4a_subject_timeline(
            ax_a, pr4a[sw_key], "a",
            "switching subject", add_inline_legend=True)
    if dom_key and dom_key in pr4a:
        _plot_pr4a_subject_timeline(
            ax_b, pr4a[dom_key], "b",
            "dominant-template subject", add_inline_legend=True)
    ax_b.set_xlabel("Hours from recording start", fontsize=FS_LABEL)

    # ---- Build cohort paired arrays ----
    dom_day, dom_night = [], []
    for k, r in pr4a.items():
        ds = r.get("day_night_summary", {})
        d = ds.get("day", {})
        n = ds.get("night", {})
        if "dominant_fraction" in d and "dominant_fraction" in n:
            dom_day.append(d["dominant_fraction"])
            dom_night.append(n["dominant_fraction"])
    dom_day = np.array(dom_day, dtype=float)
    dom_night = np.array(dom_night, dtype=float)

    # ---- Panel c: dom fraction day vs night paired ----
    ax_c = fig.add_subplot(gs[2, 0])
    n = len(dom_day)
    for d, nn in zip(dom_day, dom_night):
        col = "#bbbbbb" if abs(d - nn) < 0.02 \
            else (COL_DAY_EDGE if d > nn else COL_NIGHT_EDGE)
        ax_c.plot([0, 1], [d, nn], "-", color=col, lw=1.2, alpha=0.55)
    ax_c.scatter(np.zeros(n), dom_day, s=80, color=COL_DAY,
                 edgecolors=COL_DAY_EDGE, linewidths=1.2, zorder=3)
    ax_c.scatter(np.ones(n), dom_night, s=80, color=COL_NIGHT,
                 edgecolors=COL_NIGHT_EDGE, linewidths=1.2, zorder=3)
    ax_c.set_xticks([0, 1])
    ax_c.set_xticklabels(["day", "night"], fontsize=FS_LABEL)
    ax_c.set_xlim(-0.4, 1.4)
    ax_c.set_ylabel("Dominant template fraction", fontsize=FS_LABEL)
    p = float(cohort["dominant_fraction"]["wilcoxon_p"])
    add_significance_bracket(
        ax_c, 0, 1, max(dom_day.max(), dom_night.max()) + 0.05,
        p, dy=0.018, fontsize=FS_TITLE)
    ax_c.set_title("PR-4A  day vs night  (cohort null)",
                   fontsize=FS_TITLE, fontweight="bold", pad=22)
    ax_c.text(0.5, -0.22, f"Wilcoxon paired p = {p:.3f},  n = {n}",
              ha="center", va="top", fontsize=FS_TICK,
              transform=ax_c.transAxes, color="#444")
    style_panel(ax_c, "c")

    # ---- Panel d: per-subject Spearman rho(bin dom-fraction, |Δt to nearest seizure|) ----
    # Inline computation: for each subject with >=1 in-range seizure, take
    # the per-bin dominant template fraction, compute the distance from the
    # bin centre to the nearest seizure onset, then Spearman rho. Cohort
    # Wilcoxon test rho != 0 across subjects.
    ax_d = fig.add_subplot(gs[2, 1])
    rhos, ds_labels = [], []
    n_skip_no_sz = 0
    for k, rec in pr4a.items():
        bins = rec.get("timeline_bins", [])
        if not bins:
            continue
        nc = max(int(rec.get("n_clusters", 2)), 1)
        fracs = np.array(
            [(b.get("cluster_fractions", []) + [0.0] * nc)[:nc] for b in bins],
            dtype=float)
        n_ev = np.array([b.get("n_events", 0) for b in bins])
        bin_center_h = np.array(
            [b["hours_from_timeline_start"] for b in bins], dtype=float)
        t0 = float(rec.get("timeline_start_epoch",
                           rec.get("first_event_epoch", 0.0)))
        try:
            sz_ep = load_seizure_times(rec.get("subject", ""),
                                       rec.get("dataset", ""))
        except Exception:
            sz_ep = []
        if not sz_ep:
            n_skip_no_sz += 1
            continue
        sz_h = np.array([(s - t0) / 3600.0 for s in sz_ep], dtype=float)
        # only keep seizure onsets within the timeline (with a small slack)
        sz_h = sz_h[(sz_h >= -2) & (sz_h <= bin_center_h.max() + 2)]
        if sz_h.size == 0:
            n_skip_no_sz += 1
            continue
        # bin-wise distance to nearest seizure (hours)
        dist = np.min(np.abs(bin_center_h[:, None] - sz_h[None, :]), axis=1)
        # per-bin dominant template fraction = max across templates
        dom_frac = fracs.max(axis=1)
        # only use bins with non-trivial event counts
        keep = (n_ev >= 5) & np.isfinite(dom_frac) & np.isfinite(dist)
        if keep.sum() < 6:
            continue
        rho, _ = spearmanr(dist[keep], dom_frac[keep])
        if np.isnan(rho):
            continue
        rhos.append(float(rho))
        ds_labels.append(rec.get("dataset", "?"))
    rhos = np.array(rhos, dtype=float)
    n_used = rhos.size
    if n_used >= 5:
        try:
            w_stat, w_p = wilcoxon(rhos, alternative="two-sided")
        except Exception:
            w_stat, w_p = (np.nan, np.nan)
        med_rho = float(np.median(rhos))
        n_neg = int(np.sum(rhos < 0))

        # Violin + scatter coloured by dataset
        violin_with_scatter(ax_d, rhos, pos=0.0, color=COL_NEUTRAL,
                            width=0.55, scatter_size=0)
        rng3 = np.random.default_rng(31)
        jit = rng3.normal(0, 0.05, n_used)
        cols_arr = np.array([dataset_color(d) for d in ds_labels])
        ax_d.scatter(jit, rhos, s=80, c=cols_arr,
                     edgecolors="black", linewidths=0.8, zorder=3,
                     alpha=0.92)
        ax_d.axhline(0, color="black", lw=1.2, ls="--", alpha=0.7)
        ax_d.set_xticks([0])
        ax_d.set_xticklabels([f"cohort (n={n_used})"], fontsize=FS_LABEL)
        ax_d.set_xlim(-0.55, 0.55)
        ax_d.set_ylim(-1.05, 1.05)
        ax_d.set_ylabel("Spearman ρ", fontsize=FS_LABEL)
        sig_tag = "*" if (w_p is not None and w_p < 0.05) else "n.s."
        ax_d.set_title(
            f"dom. fraction × |Δt to seizure|  ({sig_tag})",
            fontsize=FS_TITLE, fontweight="bold", pad=22)
        ax_d.text(
            0.5, -0.22,
            f"ρ med = {med_rho:+.3f},  Wilcoxon vs ρ=0  p = {w_p:.3f},  "
            f"{n_neg}/{n_used} ρ < 0",
            ha="center", va="top", fontsize=FS_TICK,
            transform=ax_d.transAxes, color="#444")
        # dataset legend (top-right of panel d)
        legend_handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=COL_YUQUAN, markeredgecolor="black",
                       markersize=11, label="Yuquan"),
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=COL_EPILEPSIAE,
                       markeredgecolor="black",
                       markersize=11, label="Epilepsiae"),
        ]
        ax_d.legend(handles=legend_handles, loc="lower right",
                    fontsize=FS_TICK - 1, framealpha=0.95)
    else:
        ax_d.text(0.5, 0.5,
                  f"insufficient subjects with in-window seizures\n"
                  f"(used={n_used}, skipped={n_skip_no_sz})",
                  ha="center", va="center", fontsize=FS_TICK,
                  transform=ax_d.transAxes)
    style_panel(ax_d, "d")

    return savefig_pub(fig, PPT_DIR / "fig2_pr4a_daynight.png")


# =========================================================================
# Fig 3: PR-4B rate state × three-layer modulation (main result)
# =========================================================================

def _extract_pr4b_subject_deltas(coupling: dict) -> dict:
    """Pull subject-level high-minus-low deltas across L1/L2/L3."""
    def _f(v):
        try:
            return float(v) if v is not None else np.nan
        except (TypeError, ValueError):
            return np.nan
    out = {
        "subjects": [], "datasets": [],
        "raw_tau_delta": [], "centered_tau_delta": [],
        "lag_span_delta": [], "pearson_r_delta": [],
        "occupancy_rho": [],
        "dominant_r": [],
        "n_clusters": [],
    }
    for k, r in coupling.items():
        out["subjects"].append(r.get("subject", k))
        out["datasets"].append(r.get("dataset", "?"))
        out["raw_tau_delta"].append(_f(r.get("subject_raw_delta")))
        out["centered_tau_delta"].append(_f(r.get("subject_centered_delta")))
        out["lag_span_delta"].append(_f(r.get("subject_lag_span_delta")))
        out["pearson_r_delta"].append(_f(r.get("subject_pearson_r_delta")))
        l1 = r.get("l1") or {}
        dc = (l1 or {}).get("dominant_cluster") or {}
        out["occupancy_rho"].append(_f(dc.get("occupancy_rate_spearman_rho")))
        out["n_clusters"].append(int(r.get("n_clusters", 0)))
    return {k: (np.array(v, dtype=float) if k not in ("subjects", "datasets")
                else np.array(v))
            for k, v in out.items()}


def _pull_dominant_r(lag_validation: dict) -> dict:
    out = {}
    for k, r in lag_validation.items():
        out[k] = float(r.get("dominant_cluster_median_r", np.nan))
    return out


def _hviolin_row(ax, vals, color=COL_NEUTRAL, width=0.7,
                 hc_vals=None, hc_color=COL_SIG):
    """Horizontal violin (single row) at y=0 with optional HC overlay."""
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    parts = ax.violinplot([vals], positions=[0.0], vert=False,
                          widths=width, showextrema=False, showmedians=False)
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.45)
    rng = np.random.default_rng(13)
    jit = rng.normal(0, 0.07, vals.size)
    ax.scatter(vals, jit, s=55, color=color,
               edgecolors="white", linewidths=0.7, zorder=3, alpha=0.92)
    if hc_vals is not None and hc_vals.size:
        jit_hc = rng.normal(0, 0.07, hc_vals.size)
        ax.scatter(hc_vals, jit_hc, s=110, color=hc_color,
                   edgecolors="black", linewidths=1.0, zorder=4)
    # median tick
    med = float(np.median(vals))
    ax.plot([med, med], [-0.32, 0.32], color="black", lw=2.4)


def plot_fig3():
    """PR-4B: rate state × L1/L2/L3 — collapsed to 3 panels.

        a  rate-binning + matched subsampling toy
        b  five-row horizontal violin: L1 ρ, L2 raw τ Δ, L2 centered τ Δ,
           L3 lag span Δ, L3 Pearson r Δ. HC subset overlaid in red.
        c  PR-4B Step-0 absolute-lag validation: subject-level dom_r with
           HC threshold (so the reader sees why only n=8 is HC).
    """
    print("Plotting Fig 3: PR-4B rate state coupling (compact) ...")

    coupling = _load_json("pr4b_coupling_summary.json")
    lag_val = _load_json("pr4b_lag_validation.json")
    cohort = _load_json("pr1_cohort_summary.json").get(
        "rate_state_coupling_analysis", {})
    if not coupling or not cohort:
        print("  PR-4B data missing, abort")
        return None

    deltas = _extract_pr4b_subject_deltas(coupling)
    dom_r = _pull_dominant_r(lag_val)
    dom_r_arr = np.array(
        [dom_r.get(k, np.nan) for k in coupling.keys()], dtype=float)
    hc_mask = dom_r_arr > 0.7
    n_total = len(deltas["subjects"])
    n_hc = int(np.sum(hc_mask))

    fig = plt.figure(figsize=(20, 13.5))
    gs = gridspec.GridSpec(
        2, 2, height_ratios=[0.95, 1.0], width_ratios=[1, 1.6],
        wspace=0.22, hspace=0.60,
        left=0.055, right=0.985, top=0.89, bottom=0.10,
    )
    _suptitle(fig,
              "Fig 3  ·  PR-4B  rate-state coupling on L1 / L2 / L3",
              y=0.965)

    # ----------------------------------------------------------------
    # Panel a — rate-binning + matched subsampling toy
    # ----------------------------------------------------------------
    ax_toy = fig.add_subplot(gs[0, 0])
    rates = np.array([12, 35, 48, 22, 9, 14, 60, 75, 30, 18, 41, 28], float)
    t_bins = np.arange(0, 12)
    median_rate = float(np.median(rates))
    bar_colors = [COL_HIGH if r > median_rate else COL_LOW for r in rates]
    ax_toy.bar(t_bins, rates, width=0.85, color=bar_colors,
               edgecolor="white", linewidth=0.6)
    ax_toy.axhline(median_rate, color="black", lw=1.8, ls="--")
    ax_toy.text(11.4, median_rate + 1.5, "subject median",
                fontsize=FS_TICK, color="black",
                ha="right", va="bottom")
    ax_toy.set_xlim(-0.6, 11.6)
    ax_toy.set_xlabel("Time (rate bin index)", fontsize=FS_LABEL)
    ax_toy.set_ylabel("Local event rate (events/h)", fontsize=FS_LABEL)
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=COL_HIGH, edgecolor="white", label="high state"),
        Patch(facecolor=COL_LOW, edgecolor="white", label="low state"),
    ]
    ax_toy.legend(handles=handles, loc="upper left", fontsize=FS_TICK,
                  framealpha=0.95)
    ax_toy.set_title("Rate binning + matched subsampling",
                     fontsize=FS_TITLE, fontweight="bold", pad=10)
    style_panel(ax_toy, "a", label_y=1.04)

    # ----------------------------------------------------------------
    # Panel c (top-right placement) — HC threshold bar
    # ----------------------------------------------------------------
    # Move HC bar to top-right so the big metric panel can occupy bottom row
    ax_hc = fig.add_subplot(gs[0, 1])
    order = np.argsort(dom_r_arr)
    labels = [f"{deltas['datasets'][i][:1].upper()}:{deltas['subjects'][i]}"
              for i in order]
    bar_cols = [COL_SIG if dom_r_arr[i] > 0.7 else (
        COL_NEUTRAL if np.isfinite(dom_r_arr[i]) else "#dddddd"
    ) for i in order]
    ax_hc.bar(np.arange(len(order)), dom_r_arr[order],
              color=bar_cols, edgecolor="white", linewidth=0.4,
              alpha=0.95)
    ax_hc.axhline(0.7, color="black", lw=1.6, ls="--")
    ax_hc.text(len(order) - 0.5, 0.73, "HC threshold (dom_r > 0.7)",
               ha="right", va="bottom", fontsize=FS_TICK, color="black",
               fontweight="bold")
    ax_hc.set_xticks(np.arange(len(order)))
    ax_hc.set_xticklabels(labels, rotation=70, fontsize=FS_TICK - 4,
                          ha="right")
    ax_hc.set_ylabel("Dominant cluster\nmedian Pearson r", fontsize=FS_LABEL)
    ax_hc.set_title(
        f"Step-0 absolute-lag validation: {n_hc}/{n_total} subjects qualify HC",
        fontsize=FS_TITLE, fontweight="bold", pad=10)
    ax_hc.set_ylim(0, 1)
    style_panel(ax_hc, "c", label_y=1.04)

    # ----------------------------------------------------------------
    # Panel b — five vertical violins (L1 + L2 + L3) side-by-side. Each
    # column is its own axis with its own y-unit (units differ across
    # metrics). HC subset overlaid as red dots where applicable.
    # ----------------------------------------------------------------
    sub_gs = gridspec.GridSpecFromSubplotSpec(
        1, 5, subplot_spec=gs[1, :], wspace=0.50,
    )

    rho   = deltas["occupancy_rho"]
    raw   = deltas["raw_tau_delta"]
    cen   = deltas["centered_tau_delta"]
    ls    = deltas["lag_span_delta"]
    pr_d  = deltas["pearson_r_delta"]
    ls_hc_mask = hc_mask & np.isfinite(ls)
    pr_hc_mask = hc_mask & np.isfinite(pr_d)

    rho_med = float(cohort["l1"]["dominant_cluster_rho_median"])
    p_raw   = float(cohort["raw_tau"]["wilcoxon_p"])
    p_cen   = float(cohort["centered_tau"]["wilcoxon_p"])
    p_ls    = float(cohort["l3"]["lag_span"]["wilcoxon_p"])
    d_ls    = float(cohort["l3"]["lag_span"]["delta_high_minus_low_median"])
    n_pos_ls = int(cohort["l3"]["lag_span"]["n_subjects_high_gt_low"])
    p_pr     = float(cohort["l3"]["pearson_r_exploratory"]["wilcoxon_p"])
    p_pr_hc  = float(cohort["l3"]["pearson_r_high_confidence"]["wilcoxon_p"])
    d_pr_hc  = float(cohort["l3"]["pearson_r_high_confidence"]["delta_high_minus_low_median"])
    n_pr_hc_pos = int(cohort["l3"]["pearson_r_high_confidence"]["n_subjects_high_gt_low"])

    n_rho_pos = int(np.sum(rho[np.isfinite(rho)] > 0))
    n_rho_fin = int(np.sum(np.isfinite(rho)))
    n_ls_fin  = int(np.sum(np.isfinite(ls)))

    cols = [
        # (vals, hc_mask_col, top_label, y_label, color,
        #  annot_line1, annot_line2, verdict)
        (rho, None, "L1  ρ\noccupancy ↔ rate", "Spearman ρ", COL_T0,
         f"ρ med = {rho_med:+.3f}",
         f"{n_rho_pos}/{n_rho_fin} ρ>0", "null"),
        (raw, None, "L2  Δ raw τ\nhigh − low", "Δ raw τ", COL_T1,
         f"Δ med = {float(np.nanmedian(raw)):+.3f}",
         f"Wilcoxon p = {p_raw:.3f}", "null"),
        (cen, None, "L2  Δ centered τ\nhigh − low", "Δ centered τ", COL_T1,
         f"Δ med = {float(np.nanmedian(cen)):+.3f}",
         f"Wilcoxon p = {p_cen:.3f}", "null"),
        (ls, ls_hc_mask, "L3  Δ lag span\nhigh − low", "Δ lag span (s)",
         COL_T2,
         f"Δ med = {d_ls:+.4f} s",
         f"Wilcoxon p = {p_ls:.3f}  ({n_pos_ls}/{n_ls_fin})", "trend"),
        (pr_d, pr_hc_mask, "L3  Δ Pearson r\nhigh − low",
         "Δ Pearson r", COL_T3,
         f"full p = {p_pr:.3f}",
         f"HC Δ = {d_pr_hc:+.3f},  HC p = {p_pr_hc:.3f}",
         f"HC sig*  ({n_pr_hc_pos}/{n_hc})"),
    ]

    verdict_color = {"null": "#888888", "trend": "#C97D27",
                     "HC sig*  (7/8)": "#C0392B"}
    for i, (vals, hc_m, top_lbl, y_lbl, color,
            line1, line2, verdict) in enumerate(cols):
        ax = fig.add_subplot(sub_gs[0, i])
        v_fin = vals[np.isfinite(vals)]
        violin_with_scatter(ax, v_fin, pos=0.0, color=color,
                            width=0.55, scatter_size=42,
                            alpha_body=0.22)
        # HC overlay
        if hc_m is not None and hc_m.any():
            v_hc = vals[hc_m]
            rng_hc = np.random.default_rng(7)
            jit_hc = rng_hc.normal(0, 0.06, v_hc.size)
            ax.scatter(jit_hc, v_hc, s=110,
                       facecolors=COL_SIG, edgecolors="black",
                       linewidths=1.2, zorder=5, alpha=0.95,
                       label=f"HC (n={int(hc_m.sum())})")
            ax.legend(loc="upper right", fontsize=FS_TICK - 3,
                      framealpha=0.9, handlelength=1.0)
        ax.axhline(0, color="black", lw=1.2, ls="--", alpha=0.7)
        ax.set_xticks([])
        ax.set_xlim(-0.55, 0.55)
        ax.set_ylabel(y_lbl, fontsize=FS_LABEL - 1)
        ax.set_title(top_lbl, fontsize=FS_TITLE - 2,
                     fontweight="bold", pad=10)
        ax.tick_params(labelsize=FS_TICK - 1)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        # symmetric y range so the 0 line sits centred
        v_max = float(np.nanmax(np.abs(v_fin))) if v_fin.size else 0.1
        ax.set_ylim(-v_max * 1.15, v_max * 1.15)
        # annotation underneath
        ax.text(0.5, -0.12, line1, transform=ax.transAxes,
                ha="center", va="top", fontsize=FS_TICK - 2, color="#222")
        ax.text(0.5, -0.20, line2, transform=ax.transAxes,
                ha="center", va="top", fontsize=FS_TICK - 2, color="#222")
        v_col = ("#C0392B" if "sig" in verdict else
                 ("#C97D27" if verdict == "trend" else "#888888"))
        ax.text(0.5, -0.30, f"→ {verdict}", transform=ax.transAxes,
                ha="center", va="top", fontsize=FS_TICK - 1,
                color=v_col, fontweight="bold")
        if i == 0:
            style_panel(ax, "b", label_x=-0.30, label_y=1.30)
            ax.text(-0.30, 1.20,
                    f"n = {n_rho_fin} subjects   ·   "
                    f"HC subset n = {n_hc} (red)",
                    transform=ax.transAxes,
                    ha="left", va="top", fontsize=FS_TICK,
                    fontstyle="italic", color="#444")

    return savefig_pub(fig, PPT_DIR / "fig3_pr4b_rate_coupling.png")


def _binom(n, k):
    """Two-sided binomial p approx via simple combinations - returns one tail probs.

    Used only for inline sign-test annotation; not the formal stat.
    """
    from math import comb
    p = 0.5
    out = []
    for j in range(k, n + 1):
        out.append(comb(n, j) * p ** n)
    return out


# =========================================================================
# Fig 4: PR-4C seizure proximity — geometry null + rate signal
# =========================================================================

def plot_fig4():
    """PR-4C: cohort propagation geometry is null around seizures, but
    fixed-template event rate (rate_by_template) is robustly elevated post-ictally.

    Numbers come from ``docs/archive/topic1/pr4c_seizure_proximity_review_2026-04-17.md``
    §9 (P0 fix re-run, both main + auxiliary configs).
    """
    print("Plotting Fig 4: PR-4C seizure proximity ...")

    # ---- documented PR-4C numbers ----
    metrics = ["raw_tau", "centered_tau", "lag_span",
               "pearson_r", "dom_cluster_frac"]
    metric_labels = ["raw τ", "centered τ", "lag span",
                     "Pearson r", "dom. fraction"]
    pairs = ["pre vs base", "post vs pre", "post vs base"]
    # Wilcoxon p main config (after P0 fix, archive §9.3)
    p_main = np.array([
        [0.264, 0.121, 0.853, 0.303, 0.895],   # pre vs baseline
        [0.855, 0.643, 0.229, 0.877, 0.020],   # post vs pre
        [0.846, 0.264, 1.000, 0.252, 0.252],   # post vs baseline
    ])
    p_aux = np.array([
        [0.141, 0.893, 0.916, 0.229, 0.491],
        [0.160, 0.539, 0.508, 0.197, 0.178],
        [0.390, 0.656, 0.768, 0.684, 0.002],
    ])

    # rate_by_template numbers (archive §9.4)
    rate_state_main = {
        "baseline": {"total": 72.2, "dominant": 36.4},
        "pre":      {"total": 176.8, "dominant": 82.3},
        "post":     {"total": 170.5, "dominant": 107.2},
    }
    rate_state_aux = {
        "baseline": {"total": 109.4, "dominant": 52.2},
        "pre":      {"total": 155.7, "dominant": 85.8},
        "post":     {"total": 156.7, "dominant": 84.4},
    }
    rate_compare = {
        "pre_vs_base":  {"main_d": 4.8,  "main_p": 0.300,
                         "aux_d":  9.9,  "aux_p":  0.099},
        "post_vs_pre":  {"main_d": 11.3, "main_p": 0.034,
                         "aux_d": -1.0,  "aux_p":  0.635},
        "post_vs_base": {"main_d": 39.9, "main_p": 0.0009,
                         "aux_d": 22.7,  "aux_p":  0.0067},
    }

    fig = plt.figure(figsize=(20, 14.5))
    gs = gridspec.GridSpec(
        2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.05],
        hspace=0.55, wspace=0.24,
        left=0.07, right=0.97, top=0.87, bottom=0.07,
    )
    _suptitle(fig,
              "Fig 4  ·  PR-4C  seizure proximity:  geometry null  +  recruitment-rate signal",
              y=0.96)

    # ----------------------------------------------------------------
    # Panel a (top-left): window contract toy
    # ----------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    sz_t = 0.0
    main_windows = {
        "baseline":   (-4.0, -1.0, COL_BASELINE),
        "pre-ictal":  (-1.0, -0.25, COL_PRE),
        "post-ictal": (0.25, 1.0, COL_POST),
    }
    aux_windows = {
        "baseline":   (-2.0, -0.5),
        "pre-ictal":  (-0.5, -1/12),
        "post-ictal": (1/12, 1.0),
    }
    y_main = 1.6
    for name, (a, b, col) in main_windows.items():
        ax_a.barh(y_main, b - a, left=a, height=0.55,
                  color=col, alpha=0.85, edgecolor="black",
                  linewidth=1.0)
        # baseline label moved further right inside the bar so it isn't
        # blocked by the recording gap rect
        if name == "baseline":
            tx = b - 0.05
            ha = "right"
            tcol = "black"
        else:
            tx = (a + b) / 2
            ha = "center"
            tcol = "white"
        ax_a.text(tx, y_main, name, ha=ha, va="center",
                  fontsize=FS_TICK, fontweight="bold", color=tcol)
    y_aux = 0.5
    for (name, (a, b)), col in zip(aux_windows.items(),
                                    [COL_BASELINE, COL_PRE, COL_POST]):
        ax_a.barh(y_aux, b - a, left=a, height=0.55,
                  color=col, alpha=0.50, edgecolor="black", linewidth=1.0,
                  hatch="//")

    # Recording-gap example inside baseline (label simplified)
    ax_a.add_patch(plt.Rectangle((-3.0, y_main - 0.30), 0.5, 0.60,
                                  facecolor="white",
                                  edgecolor="#888", lw=0.8,
                                  hatch="xxx", zorder=3))
    ax_a.text(-2.75, y_main - 0.50, "recording gap",
              ha="center", va="top", fontsize=FS_TICK - 1, color="#444")

    # Seizure marker
    ax_a.axvline(sz_t, color=COL_SEIZURE, lw=2.5, ls="--")
    ax_a.text(sz_t, 2.55, "seizure onset", ha="center", va="bottom",
              fontsize=FS_LABEL, color=COL_SEIZURE, fontweight="bold")

    # Inline (in-axis) labels for main / aux on the LEFT INSIDE of axis
    ax_a.text(-3.95, y_main, "main", ha="left", va="center",
              fontsize=FS_LABEL, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                        edgecolor="#888", alpha=0.95))
    ax_a.text(-3.95, y_aux, "aux", ha="left", va="center",
              fontsize=FS_LABEL, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                        edgecolor="#888", alpha=0.95))

    ax_a.set_yticks([])
    ax_a.set_xlim(-4.0, 1.3)
    ax_a.set_ylim(-0.2, 2.95)
    ax_a.set_xlabel("Time relative to seizure onset (h)", fontsize=FS_LABEL)
    ax_a.set_title("Seizure-proximity window contract",
                   fontsize=FS_TITLE, fontweight="bold", pad=14)
    style_panel(ax_a, "a")

    # ----------------------------------------------------------------
    # Panel b (bottom row, full width): 5×3×2 Wilcoxon p heatmap
    # ----------------------------------------------------------------
    ax_b = fig.add_subplot(gs[1, :])
    combined = np.zeros((3, len(metrics) * 2))
    for i, m_arr in enumerate([p_main, p_aux]):
        for j in range(len(metrics)):
            combined[:, j * 2 + i] = m_arr[:, j]
    log_p = -np.log10(np.clip(combined, 1e-4, 1.0))
    im = ax_b.imshow(log_p, cmap="RdYlGn_r", aspect="auto",
                     vmin=0, vmax=2.5)

    # Less text: only annotate p < 0.05 with bold p; n.s. cells stay blank
    for ii in range(combined.shape[0]):
        for jj in range(combined.shape[1]):
            p = combined[ii, jj]
            if p < 0.05:
                txt = f"{p:.3f}" if p >= 0.005 else f"{p:.4f}"
                color = "white" if log_p[ii, jj] > 1.0 else "black"
                ax_b.text(jj, ii, txt, ha="center", va="center",
                          fontsize=FS_TICK, color=color, fontweight="bold")

    # Cleaner xtick labels: metric name on row 1, 'main / aux' on row 2
    xt_labels = []
    for m in metric_labels:
        xt_labels.append(f"{m}\nmain")
        xt_labels.append("aux")
    ax_b.set_xticks(np.arange(combined.shape[1]))
    ax_b.set_xticklabels(xt_labels, fontsize=FS_TICK)
    ax_b.set_yticks([0, 1, 2])
    ax_b.set_yticklabels(pairs, fontsize=FS_LABEL)
    # Group separators between metrics
    for k in range(1, len(metrics)):
        ax_b.axvline(k * 2 - 0.5, color="white", lw=2.5)
    ax_b.set_title(
        "Cohort Wilcoxon p:  5 geometry metrics × 3 pairs × 2 configs",
        fontsize=FS_TITLE, fontweight="bold", pad=12)
    cbar = plt.colorbar(im, ax=ax_b, fraction=0.025, pad=0.015,
                        shrink=0.85)
    cbar.set_label("$-\\log_{10}$ p", fontsize=FS_LABEL)
    cbar.ax.tick_params(labelsize=FS_TICK - 1)
    cbar.ax.axhline(-np.log10(0.05), color="black", lw=1.2)
    style_panel(ax_b, "b")
    for s in ax_b.spines.values():
        s.set_visible(False)

    # ----------------------------------------------------------------
    # Panel c (top-right): dominant-template rate per state — baseline
    # in MIDDLE, significance brackets within each config band only.
    # ----------------------------------------------------------------
    ax_c = fig.add_subplot(gs[0, 1])
    # Reorder states: pre / baseline / post  (baseline middle)
    states = ["pre", "baseline", "post"]
    state_cols = {"pre": COL_PRE, "baseline": COL_BASELINE, "post": COL_POST}
    width = 0.36
    x = np.arange(len(states))
    main_dom = [rate_state_main[s]["dominant"] for s in states]
    aux_dom  = [rate_state_aux[s]["dominant"]  for s in states]
    bar_main = ax_c.bar(x - width/2, main_dom, width,
                        color=[state_cols[s] for s in states],
                        alpha=0.95, edgecolor="black", linewidth=1.0,
                        label="main config")
    bar_aux  = ax_c.bar(x + width/2, aux_dom, width,
                        color=[state_cols[s] for s in states],
                        alpha=0.55, edgecolor="black", linewidth=1.0,
                        hatch="//", label="auxiliary config")

    # Value labels on top of bars
    for bars, vals in [(bar_main, main_dom), (bar_aux, aux_dom)]:
        for bar, v in zip(bars, vals):
            ax_c.text(bar.get_x() + bar.get_width()/2,
                      v + 3, f"{v:.0f}", ha="center", va="bottom",
                      fontsize=FS_TICK, color="black")

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(["pre-ictal", "baseline", "post-ictal"],
                         fontsize=FS_LABEL)
    ax_c.set_ylabel("Cohort median dominant-template rate\n"
                    "(events/h, gap-aware)",
                    fontsize=FS_LABEL)
    ax_c.set_title(
        "Dominant rate: post-ictal > baseline",
        fontsize=FS_TITLE, fontweight="bold", pad=14)

    # Significance brackets — only baseline-vs-post within each config; bars
    # do not span across configs.
    y_main_top = max(main_dom) + 22
    y_aux_top  = max(aux_dom)  + 22
    add_significance_bracket(
        ax_c, x[1] - width/2, x[2] - width/2, y_main_top,
        rate_compare["post_vs_base"]["main_p"], dy=4, fontsize=FS_TITLE)
    add_significance_bracket(
        ax_c, x[1] + width/2, x[2] + width/2, y_aux_top,
        rate_compare["post_vs_base"]["aux_p"], dy=4, fontsize=FS_TITLE)
    ax_c.text(0.02, 0.97,
              f"main  Δ_med = +{rate_compare['post_vs_base']['main_d']:.1f} ev/h,  "
              f"p = {rate_compare['post_vs_base']['main_p']:.4f}\n"
              f" aux  Δ_med = +{rate_compare['post_vs_base']['aux_d']:.1f} ev/h,  "
              f"p = {rate_compare['post_vs_base']['aux_p']:.4f}",
              transform=ax_c.transAxes, ha="left", va="top",
              fontsize=FS_TICK - 1,
              bbox=dict(boxstyle="round,pad=0.4", facecolor="#F4F1EA",
                        edgecolor="#888", alpha=0.95))
    ax_c.legend(loc="upper right", fontsize=FS_TICK, framealpha=0.95)
    ax_c.set_ylim(0, max(y_main_top, y_aux_top) + 35)
    style_panel(ax_c, "c")

    return savefig_pub(fig, PPT_DIR / "fig4_pr4c_seizure_proximity.png")


# =========================================================================
# Fig 5: PR-4D template-decomposed rate (descriptive)
# =========================================================================

def _score_rate_cluster_seizure(pr4a: dict) -> list:
    """For each subject, compute two coupled scores describing the
    'rate-burst window enriches seizure occurrence + dominant-template
    fraction is modulated by seizure proximity' pattern visible in
    Fig 5 / Fig 2.

    Returns a list of dicts sorted by enrichment, each with:
        subject, dataset, k, n_sz, hours, enrich, rho_dom, hi_frac_time
        match  ('strict' / 'loose' / '-')

        - enrich       = N(sz in rate>p75 bin) / Expected_uniform
        - rho_dom      = Spearman rho(per-bin dominant fraction,
                                       |Δt to nearest seizure|)
        - hi_frac_time = fraction of recording time in the rate>p75 window

    Inclusion criteria for "match":
        strict  enrich >= 1.5  AND  |rho_dom| >= 0.15
        loose   enrich >= 1.5  AND  |rho_dom| <  0.15
    """
    out = []
    for k, r in pr4a.items():
        bins = r.get("timeline_bins", [])
        if not bins:
            continue
        bin_h = np.array(
            [b["hours_from_timeline_start"] for b in bins], dtype=float)
        n_ev = np.array([b["n_events"] for b in bins], dtype=float)
        bin_w = float(r.get("bin_hours", 1.0))
        rate = n_ev / bin_w
        nc = max(int(r.get("n_clusters", 2)), 1)
        fracs = np.array(
            [(b.get("cluster_fractions", []) + [0.0] * nc)[:nc] for b in bins],
            dtype=float)
        dom_frac = fracs.max(axis=1)
        try:
            sz_ep = load_seizure_times(r.get("subject", ""),
                                       r.get("dataset", ""))
        except Exception:
            sz_ep = []
        if not sz_ep:
            continue
        t0 = float(r.get("timeline_start_epoch",
                         r.get("first_event_epoch", 0.0)))
        sz_h = np.array([(s - t0) / 3600.0 for s in sz_ep], dtype=float)
        sz_h = sz_h[(sz_h >= 0) & (sz_h <= bin_h.max())]
        if sz_h.size < 2:
            continue
        pos_rate = rate[rate > 0]
        if pos_rate.size < 5:
            continue
        p75 = float(np.percentile(pos_rate, 75))
        hi_mask = rate > p75
        if hi_mask.sum() == 0:
            continue
        hi_bins = bin_h[hi_mask]
        hi_frac_time = float(hi_mask.sum() / len(rate))
        sz_in_hi = 0
        for s in sz_h:
            if np.any(np.abs(hi_bins - s) <= bin_w / 2 + 0.5):
                sz_in_hi += 1
        expected = sz_h.size * hi_frac_time
        enrich = float(sz_in_hi / expected) if expected > 0 else float("nan")
        dist = np.min(np.abs(bin_h[:, None] - sz_h[None, :]), axis=1)
        keep = (n_ev >= 5) & np.isfinite(dom_frac) & np.isfinite(dist)
        if keep.sum() >= 6:
            rho, _ = spearmanr(dist[keep], dom_frac[keep])
            rho = float(rho) if np.isfinite(rho) else float("nan")
        else:
            rho = float("nan")
        if enrich >= 1.5 and abs(rho) >= 0.15:
            tag = "strict"
        elif enrich >= 1.5:
            tag = "loose"
        else:
            tag = "-"
        out.append({
            "subject": r.get("subject", k),
            "dataset": r.get("dataset", "?"),
            "key": k,
            "k": int(r.get("stable_k", nc)),
            "n_sz": int(sz_h.size),
            "hours": int(bin_h.max()),
            "enrich": enrich,
            "rho_dom": rho,
            "hi_frac_time": hi_frac_time,
            "match": tag,
        })
    out.sort(key=lambda x: -x["enrich"])
    return out


def _print_rate_cluster_seizure_table():
    pr4a = _load_json("pr4a_temporal_dynamics.json")
    if not pr4a:
        return
    rows = _score_rate_cluster_seizure(pr4a)
    print("\n  cohort rate-cluster ↔ seizure-cluster pattern table:")
    print(f"  {'subject':<32s} {'k':>2} {'nsz':>4} {'hrs':>5} "
          f"{'enrich':>7} {'rho_dom':>8} {'match':>7}")
    print("  " + "-" * 76)
    for r in rows:
        print(f"  {r['dataset']+':'+r['subject']:<32s} {r['k']:>2} "
              f"{r['n_sz']:>4} {r['hours']:>5} {r['enrich']:>7.2f} "
              f"{r['rho_dom']:>8.3f} {r['match']:>7s}")
    n_strict = sum(1 for r in rows if r["match"] == "strict")
    n_loose = sum(1 for r in rows if r["match"] == "loose")
    print(f"  → strict match (enrich≥1.5 & |ρ|≥0.15): {n_strict}/{len(rows)}")
    print(f"  → loose  match (enrich≥1.5 only):       {n_loose}/{len(rows)}")


def _plot_pr4d_subject(ax_top, ax_bot, rec: dict, sz_h=None):
    nc = int(rec.get("n_clusters", rec.get("chosen_k", 2)))
    rc = rec.get("rate_curve", {})
    hist = rec.get("histogram", {})
    summary = rec.get("summary", {})
    grid = np.array(rc.get("grid_hours", []), dtype=float)
    ptr = np.array(rc.get("per_template_rate", []), dtype=float)
    bin_h = np.array(hist.get("bin_center_hours", []), dtype=float)
    ptc = np.array(hist.get("per_template_count", []), dtype=int)
    bin_w = float(rec.get("bin_hours", 1.0))
    if grid.size == 0:
        return
    # rate envelope: fill + line so the median band is visible even when a
    # single tail-bin pushes max to >10x the bulk
    nc_use = min(nc, ptr.shape[0])
    all_finite = ptr[:nc_use][np.isfinite(ptr[:nc_use])]
    if all_finite.size:
        p95 = float(np.percentile(all_finite, 95))
        p100 = float(np.nanmax(all_finite))
        ymax = p95 * 1.20 if p95 > 0 else max(p100, 1.0)
    else:
        ymax = 1.0
        p100 = 1.0
    spike_msgs = []
    for c in range(nc_use):
        col = TEMPLATE_COLORS[c % len(TEMPLATE_COLORS)]
        v = ptr[c]
        # rate_curve grid has interleaved NaN entries (recording gaps); we
        # must drop them before fill / plot or every segment becomes
        # zero-length and the curve never shows up.
        mask = np.isfinite(grid) & np.isfinite(v)
        if mask.sum() == 0:
            continue
        g = grid[mask]
        vv = v[mask]
        ax_top.fill_between(g, 0, vv, color=col, alpha=0.30,
                            linewidth=0, step="mid")
        ax_top.plot(g, vv, lw=1.8, color=col, drawstyle="steps-mid",
                    label=f"Template {c}")
        if vv.max() > ymax:
            spike_msgs.append(f"T{c}: peak = {vv.max():.0f} ev/h")
    ax_top.set_ylim(0, ymax)
    ax_top.set_ylabel("Rate (events/h)", fontsize=FS_LABEL)
    dom_frac = summary.get("dominant_rate_fraction", float("nan"))
    title = (f"{rec.get('dataset','?')}:{rec.get('subject','?')}  "
             f"dom_frac = {dom_frac:.3f},  n = {rec.get('n_events_used','?')}")
    ax_top.set_title(title, fontsize=FS_TITLE - 1, fontweight="bold", pad=8)
    ax_top.legend(loc="upper left", fontsize=FS_TICK - 2, framealpha=0.95,
                  ncol=min(4, max(1, nc_use)))
    if spike_msgs:
        ax_top.text(0.99, 0.06, "off-axis: " + " · ".join(spike_msgs),
                    transform=ax_top.transAxes, ha="right", va="bottom",
                    fontsize=FS_TICK - 3, color="#A35E48",
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor="white", edgecolor="#888", alpha=0.9))
    if bin_h.size and ptc.size:
        bottom = np.zeros(ptc.shape[1], dtype=float)
        for c in range(min(nc, ptc.shape[0])):
            ax_bot.bar(
                bin_h, ptc[c], width=bin_w * 0.9, bottom=bottom,
                color=TEMPLATE_COLORS[c % len(TEMPLATE_COLORS)], alpha=0.85,
                edgecolor="white", linewidth=0.2,
            )
            bottom += ptc[c].astype(float)
    ax_bot.set_ylim(bottom=0)
    ax_bot.set_ylabel("Count / 1-h bin", fontsize=FS_LABEL)
    ax_bot.set_xlabel("Hours from recording start", fontsize=FS_LABEL)
    x_max = float(np.nanmax(grid)) * 1.005
    ax_top.set_xlim(0, x_max)
    ax_bot.set_xlim(0, x_max)

    # Seizure onset markers on BOTH rate envelope and stacked count panels
    if sz_h is not None and len(sz_h):
        sz_h = np.asarray(sz_h, dtype=float)
        sz_h = sz_h[(sz_h >= 0) & (sz_h <= x_max)]
        for h in sz_h:
            ax_top.axvline(h, color=COL_SEIZURE, lw=1.6, ls="--",
                           alpha=0.85, zorder=4)
            ax_bot.axvline(h, color=COL_SEIZURE, lw=1.6, ls="--",
                           alpha=0.85, zorder=4)
        # one dashed-line legend entry on the rate panel only
        from matplotlib.lines import Line2D
        cur_handles, cur_labels = ax_top.get_legend_handles_labels()
        cur_handles.append(Line2D([0], [0], color=COL_SEIZURE, lw=1.6,
                                  ls="--"))
        cur_labels.append(f"sz onset (n={sz_h.size})")
        ax_top.legend(cur_handles, cur_labels, loc="upper left",
                      fontsize=FS_TICK - 2, framealpha=0.95,
                      ncol=min(4, len(cur_handles)))


def plot_fig5():
    """PR-4D: gap-aware template-decomposed rate (descriptive layer)."""
    print("Plotting Fig 5: PR-4D template-rate decomposition ...")

    pr4d = _load_json("pr4a_followup_template_mix_dynamics.json")
    fol = _load_json("pr1_cohort_summary.json").get(
        "temporal_dynamics_followup_analysis", {})

    if not pr4d:
        print("  PR-4D data missing, abort")
        return None

    # Pick two representative subjects: 1 short Epilepsiae, 1 longer Yuquan
    candidates_epi = [k for k in pr4d if k.startswith("epilepsiae/")]
    candidates_yq = [k for k in pr4d if k.startswith("yuquan/")]
    sub_a_key = "epilepsiae/548" if "epilepsiae/548" in pr4d \
        else (candidates_epi[0] if candidates_epi else None)
    sub_b_key = "yuquan/chenziyang" if "yuquan/chenziyang" in pr4d \
        else (candidates_yq[0] if candidates_yq else None)

    fig = plt.figure(figsize=(18, 14.5))
    gs = gridspec.GridSpec(
        3, 2, height_ratios=[0.95, 0.95, 1.05], width_ratios=[1, 1],
        wspace=0.24, hspace=0.55,
        left=0.07, right=0.98, top=0.92, bottom=0.10,
    )

    def _sz_h_for(key):
        if key not in pr4d:
            return np.array([])
        rec = pr4d[key]
        try:
            sz_ep = load_seizure_times(rec.get("subject", ""),
                                       rec.get("dataset", ""))
        except Exception:
            sz_ep = []
        if not sz_ep:
            return np.array([])
        t0 = float(rec.get("timeline_start_epoch",
                           rec.get("first_event_epoch", 0.0)))
        return np.array([(s - t0) / 3600.0 for s in sz_ep], dtype=float)

    if sub_a_key and sub_a_key in pr4d:
        ax_a_top = fig.add_subplot(gs[0, 0])
        ax_a_bot = fig.add_subplot(gs[1, 0], sharex=ax_a_top)
        _plot_pr4d_subject(ax_a_top, ax_a_bot, pr4d[sub_a_key],
                           sz_h=_sz_h_for(sub_a_key))
        style_panel(ax_a_top, "a")
        style_panel(ax_a_bot, "")

    if sub_b_key and sub_b_key in pr4d:
        ax_b_top = fig.add_subplot(gs[0, 1])
        ax_b_bot = fig.add_subplot(gs[1, 1], sharex=ax_b_top)
        _plot_pr4d_subject(ax_b_top, ax_b_bot, pr4d[sub_b_key],
                           sz_h=_sz_h_for(sub_b_key))
        style_panel(ax_b_top, "b")
        style_panel(ax_b_bot, "")

    _print_rate_cluster_seizure_table()

    # ---- Panel c: cohort dominant rate fraction distribution ----
    ax_c = fig.add_subplot(gs[2, 0])
    dom_fracs = []
    datasets = []
    for k, r in pr4d.items():
        s = r.get("summary", {})
        df = s.get("dominant_rate_fraction", None)
        if df is not None:
            dom_fracs.append(df)
            datasets.append(r.get("dataset", "?"))
    dom_fracs = np.array(dom_fracs, dtype=float)
    datasets = np.array(datasets)
    if dom_fracs.size:
        violin_with_scatter(ax_c, dom_fracs, pos=0.0, color=COL_NEUTRAL,
                            width=0.6, scatter_size=0)
        # Custom scatter colored by dataset
        rng = np.random.default_rng(7)
        jit = np.zeros(dom_fracs.size) + rng.normal(0, 0.05, dom_fracs.size)
        cols = np.array([dataset_color(d) for d in datasets])
        ax_c.scatter(jit, dom_fracs, s=70, c=cols,
                     edgecolors="white", linewidths=0.8, zorder=3,
                     alpha=0.95)
        med = float(np.median(dom_fracs))
        ax_c.axhline(med, color="black", lw=1.4, ls="--", alpha=0.85)
        ax_c.set_xticks([0])
        ax_c.set_xticklabels(["cohort (n=30)"])
        ax_c.set_xlim(-0.55, 0.55)
        ax_c.set_ylim(0, 1)
        ax_c.set_ylabel("Dominant template rate fraction", fontsize=FS_LABEL)
        ax_c.set_title("Cohort dominant rate fraction (n=30)",
                       fontsize=FS_TITLE - 1, fontweight="bold", pad=12)
        ax_c.text(0.5, -0.18,
                   f"median = {med:.3f},  range [{dom_fracs.min():.2f}, {dom_fracs.max():.2f}]",
                   ha="center", va="top", fontsize=FS_TICK,
                   transform=ax_c.transAxes, color="#444")
        # Legend
        legend_handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=COL_YUQUAN, markeredgecolor="white",
                       markersize=10, label="Yuquan"),
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=COL_EPILEPSIAE, markeredgecolor="white",
                       markersize=10, label="Epilepsiae"),
        ]
        ax_c.legend(handles=legend_handles, loc="lower right",
                    fontsize=FS_TICK - 1, framealpha=0.95)
        style_panel(ax_c, "c", label_y=1.04)

    # ---- Panel d: dominant template crossing summary ----
    ax_d = fig.add_subplot(gs[2, 1])
    # Documented numbers (Topic 1 doc §3.1c / §7.3)
    cats = ["any\ncrossing",
            "meaningful\n(≥ 25% peak)",
            "repeated\n(≥ 3)"]
    counts = [25, 17, 6]
    n_total = 30
    bars = ax_d.bar(cats, counts, color=[COL_T0, COL_T1, COL_T2],
                    edgecolor="white", linewidth=0.8, alpha=0.92)
    for bar, c in zip(bars, counts):
        ax_d.text(bar.get_x() + bar.get_width() / 2,
                  c + 0.4, f"{c}/{n_total}",
                  ha="center", va="bottom",
                  fontsize=FS_LABEL, fontweight="bold", color="black")
    ax_d.set_ylim(0, n_total + 3)
    ax_d.set_ylabel("# subjects", fontsize=FS_LABEL)
    ax_d.set_title("Dominant-template switching counts (n=30)",
                   fontsize=FS_TITLE - 1, fontweight="bold", pad=12)
    ax_d.tick_params(axis="x", labelsize=FS_TICK - 1)
    style_panel(ax_d, "d", label_y=1.04)

    _suptitle(fig,
              "Fig 5  ·  PR-4D: gap-aware template-decomposed rate "
              "(rate × type, descriptive)",
              y=0.97)
    return savefig_pub(fig, PPT_DIR / "fig5_pr4d_template_rate.png")


# =========================================================================
# CLI
# =========================================================================

def _plot_cluster_rank_curves_semantic(
    ax, ranks, bools, valid_events, labels, channel_order,
    channel_names, title="", show_legend: bool = False):
    """Per-cluster mean rank +/- std curves on fixed channel order, using
    TEMPLATE_COLORS so cluster id matches rate-panel template id colour.
    """
    n_ch = len(channel_order)
    ordered_names = [channel_names[idx] for idx in channel_order]
    unique_k = np.unique(labels)
    y_pos = np.arange(n_ch, dtype=float)
    xs_for_lim: list[float] = []
    for cid in unique_k:
        mask_cluster = labels == cid
        eidx = valid_events[mask_cluster]
        means = np.full(n_ch, np.nan)
        stds = np.full(n_ch, np.nan)
        for ci_plot, ci_raw in enumerate(channel_order):
            vals = np.asarray(ranks[ci_raw, eidx], dtype=float)
            bmask = np.asarray(bools[ci_raw, eidx], dtype=bool)
            vals = vals[bmask & np.isfinite(vals)]
            if vals.size > 0:
                means[ci_plot] = float(np.mean(vals))
                stds[ci_plot] = float(np.std(vals))
        valid = np.isfinite(means)
        col = TEMPLATE_COLORS[int(cid) % len(TEMPLATE_COLORS)]
        lo = (means - stds)[valid]
        hi = (means + stds)[valid]
        m = means[valid]
        if lo.size:
            xs_for_lim.extend(lo.tolist())
            xs_for_lim.extend(hi.tolist())
            xs_for_lim.extend(m.tolist())
        ax.fill_betweenx(y_pos[valid], lo, hi,
                         color=col, alpha=0.18, linewidth=0)
        ax.plot(m, y_pos[valid], "-o",
                color=col, lw=2.4, ms=6, zorder=10,
                label=f"C{int(cid)} (n={int(mask_cluster.sum())})")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ordered_names,
                       fontsize=FS_TICK if n_ch <= 20 else FS_TICK - 3)
    ax.set_ylim(-0.5, n_ch - 0.5)
    ax.invert_yaxis()
    ax.set_xlabel("Rank", fontsize=FS_LABEL)
    if xs_for_lim:
        xmin = float(np.min(xs_for_lim))
        xmax = float(np.max(xs_for_lim))
        span = xmax - xmin
        pad = max(0.03 * span, 0.12) if span > 0 else 0.25
        ax.set_xlim(xmin - pad, xmax + pad)
    else:
        ax.set_xlim(-0.5, 0.5)
    ax.margins(x=0, y=0)
    if title:
        ax.set_title(title, fontsize=FS_TITLE - 2, fontweight="bold", pad=8)
    if show_legend:
        ax.legend(fontsize=FS_TICK - 2, loc="upper center",
                  bbox_to_anchor=(0.5, -0.10),
                  ncol=max(1, len(unique_k)), framealpha=0.95)


def _draw_daynight_strip(ax, bin_h, is_day, bin_w):
    """Paint a thin coloured day/night strip ribbon at the very bottom of
    the axis (axes-fraction y=0..0.025). Uses semantic COL_DAY / COL_NIGHT.
    """
    for i in range(len(bin_h)):
        col = COL_DAY if is_day[i] else COL_NIGHT
        ax.axvspan(bin_h[i] - bin_w / 2.0, bin_h[i] + bin_w / 2.0,
                   ymin=0.0, ymax=0.025,
                   facecolor=col, edgecolor="none", zorder=4)


def _draw_full_daynight_axis(ax, bin_h, is_day, bin_w, x_max):
    """Draw a stand-alone day/night strip on its own axis."""
    for i in range(len(bin_h)):
        col = COL_DAY if is_day[i] else COL_NIGHT
        ax.axvspan(bin_h[i] - bin_w / 2.0, bin_h[i] + bin_w / 2.0,
                   facecolor=col, edgecolor="none")
    ax.set_xlim(0, x_max)
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    for sp in ax.spines.values():
        sp.set_visible(False)


def plot_pr_per_subject_combined():
    """Single-PNG per-subject figure integrating PR-3 (rank heatmaps +
    cluster geometry) and PR-4D (rate envelope + stacked count). Day/night
    background and seizure-onset markers are drawn on every time-axis panel
    using the unified semantic palette (Morandi day/night, COL_SEIZURE,
    TEMPLATE_COLORS shared between cluster ids and template ids).

    Output: ``results/interictal_propagation/figures/ppt/per_subject/{ds}_{sub}.png``
    """
    print("Plotting integrated PR-3 + PR-4D per-subject figures ...")

    # Lazy import: only paid when --per-subject is requested
    from matplotlib.colors import ListedColormap
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.plot_interictal_propagation import (
        _load_pr3_subject_records, _resolve_subject_dir, _load_lagpat,
        _formal_day_mask, _fixed_channel_order, _sample_event_indices,
        _plot_rank_heatmap, _plot_rank_histogram,
    )
    from src.interictal_propagation import _valid_event_indices

    pr4d = _load_json("pr4a_followup_template_mix_dynamics.json")
    pr4a = _load_json("pr4a_temporal_dynamics.json")
    if not pr4d:
        print("  PR-4D data missing, abort")
        return None

    pr3_records = _load_pr3_subject_records("both", None)
    pr3_by_key = {f"{r['dataset']}/{r['subject']}": r for r in pr3_records}

    score_table = {}
    if pr4a:
        for r in _score_rate_cluster_seizure(pr4a):
            score_table[r["key"]] = r

    # day/night info per subject (1-h bins from PR-4A timeline_bins)
    daynight_by_key = {}
    if pr4a:
        for k, r in pr4a.items():
            bins = r.get("timeline_bins", [])
            if not bins:
                continue
            bh = np.array([b["hours_from_timeline_start"] for b in bins],
                          dtype=float)
            isd = np.array([b.get("day_night") == "day" for b in bins],
                           dtype=bool)
            bw = float(r.get("bin_hours", 1.0))
            daynight_by_key[k] = (bh, isd, bw)

    TAG_COL = {
        "strict": "#5A8C6E",       # Morandi sage
        "loose":  COL_DETRENDED,   # Morandi mustard
        "-":      COL_NONSIG,
    }
    TAG_TEXT = {
        "strict": "STRICT match",
        "loose":  "LOOSE match",
        "-":      "no-match",
    }

    out_dir = PPT_DIR / "per_subject"
    out_dir.mkdir(parents=True, exist_ok=True)

    keys = sorted(set(pr4d.keys()) & set(pr3_by_key.keys()))
    only_4d = sorted(set(pr4d.keys()) - set(pr3_by_key.keys()))
    if only_4d:
        print(f"  {len(only_4d)} subjects without PR-3 record → "
              f"PR-4D-only figure: {only_4d}")

    for k in sorted(set(pr4d.keys())):
        rec_4d = pr4d[k]
        rec_3 = pr3_by_key.get(k)
        ds = rec_4d.get("dataset", "?")
        sub = rec_4d.get("subject", "?")

        try:
            sz_ep = load_seizure_times(sub, ds)
        except Exception:
            sz_ep = []
        t0 = float(rec_4d.get("timeline_start_epoch",
                              rec_4d.get("first_event_epoch", 0.0)))
        sz_h_4d = (np.array([(s - t0) / 3600.0 for s in sz_ep], dtype=float)
                   if sz_ep else np.array([], dtype=float))

        # NOTE: score_table is still consulted by _print_rate_cluster_seizure_table
        # at the end of this function. Per-figure STRICT/enrich annotations
        # have been removed at user request, so we no longer build a per-
        # subject score_line / tag here.

        # ----- attempt PR-3 prep -----
        pr3_ok = False
        if rec_3 is not None:
            try:
                subject_dir = _resolve_subject_dir(ds, sub)
                loaded = _load_lagpat(subject_dir)
                ranks = np.asarray(loaded["ranks"], dtype=float)
                bools = np.asarray(loaded["bools"], dtype=bool)
                channel_names = list(loaded["channel_names"])
                n_ch = ranks.shape[0]
                valid_events = _valid_event_indices(
                    bools, min_participating=3)
                if valid_events.size > 0:
                    channel_order = _fixed_channel_order(ranks, bools)
                    ordered_names = [channel_names[i] for i in channel_order]
                    original_order = np.arange(n_ch, dtype=int)
                    display_events = _sample_event_indices(
                        valid_events, max_events=2000)
                    day_mask_event = _formal_day_mask(
                        ds, loaded["event_abs_times"][display_events])
                    event_h = (np.asarray(
                        loaded["event_abs_times"][display_events],
                        dtype=float) - t0) / 3600.0
                    adaptive = rec_3.get("adaptive_cluster", {})
                    adaptive_labels = np.asarray(
                        adaptive.get("labels", []), dtype=int)
                    has_adaptive = (
                        adaptive_labels.size == valid_events.size
                        and adaptive_labels.size > 0)
                    if has_adaptive:
                        best_k = int(adaptive.get("chosen_k", 2))
                        best_labels = adaptive_labels
                        within_tau = adaptive.get(
                            "within_cluster_tau_mean", float("nan"))
                        corr_mat = adaptive.get(
                            "inter_cluster_corr_matrix", [])
                        inter_corr = (
                            float(corr_mat[0][1])
                            if (corr_mat and best_k == 2
                                and len(corr_mat) >= 2
                                and len(corr_mat[0]) >= 2)
                            else float("nan"))
                    else:
                        k2 = rec_3.get("cluster", {})
                        best_k = 2
                        best_labels = np.asarray(
                            k2.get("labels", []), dtype=int)
                        within_tau = k2.get(
                            "within_cluster_tau_mean", float("nan"))
                        inter_corr = k2.get(
                            "inter_cluster_corr", float("nan"))
                    has_best = (best_labels.size == valid_events.size
                                and best_labels.size > 0)
                    all_tau = rec_3.get("propagation_stereotypy", {}).get(
                        "all", {}).get("mean_tau", float("nan"))
                    repro_grade = rec_3.get(
                        "time_split_reproducibility", {}).get(
                        "reproducibility_grade", "unknown")
                    pr3_ok = True
            except Exception as e:
                print(f"  {k}: PR-3 panels unavailable ({e})")

        # ===== build figure =====
        # New ordering (top → bottom):
        #   a  rate envelope        (real time, hours)
        #   b  stacked event count  (real time, hours)
        #      day/night strip      (shared time axis, no legend)
        #   c  raw rank heatmap     (event index)
        #   d  per-channel rank distribution
        #   e  k-clustered rank heatmap
        #   f  cluster rank curves
        # Rationale: present what the recording actually looks like in real
        # wall-clock time first (a/b share the day/night strip), then drill
        # down into the propagation structure on the event-index axis.
        if pr3_ok:
            # h_unit = HEATMAP-only height for c/e (in relative units).
            # The c row reserves an extra ``strip_extra`` slice for the
            # day/night sub-strip below the raw heatmap, so c's heatmap
            # ends up the SAME physical height as e's heatmap (and d/f
            # match because they mirror the sub-gridspec). Without this,
            # c was ~4% shorter than e and the ▼ sz markers above c made
            # the visual gap obvious.
            h_unit = max(3.4, 0.28 * n_ch)
            strip_extra = 0.18
            fig = plt.figure(figsize=(22, 2 * h_unit + 14.5))

            outer = gridspec.GridSpec(
                2, 1,
                height_ratios=[4.2 + 4.2 + 0.32,
                               2 * h_unit + strip_extra + 1.4],
                left=0.06, right=0.985, top=0.93, bottom=0.05,
                hspace=0.30,
            )

            # ---- TOP block: a (rate) + b (count) + day/night strip ----
            top_gs = gridspec.GridSpecFromSubplotSpec(
                3, 1, subplot_spec=outer[0],
                height_ratios=[4.2, 4.2, 0.32], hspace=0.06,
            )
            ax_a = fig.add_subplot(top_gs[0])
            ax_b = fig.add_subplot(top_gs[1], sharex=ax_a)
            _plot_pr4d_subject(ax_a, ax_b, rec_4d, sz_h=sz_h_4d)
            ax_a.set_title("")
            ax_a.tick_params(axis="x", labelbottom=False)
            ax_b.set_xlabel("")
            ax_b.tick_params(axis="x", labelbottom=False)
            # Larger ticks on the time-axis block (a/b + day-night strip)
            # so peak rates / counts are easy to read at PPT scale.
            for _ax in (ax_a, ax_b):
                _ax.tick_params(axis="y", labelsize=FS_TICK + 2)
                _ax.yaxis.label.set_size(FS_LABEL + 1)
            style_panel(ax_a, "")
            style_panel(ax_b, "")

            ax_dn = fig.add_subplot(top_gs[2], sharex=ax_a)
            dn_info = daynight_by_key.get(k)
            if dn_info is not None:
                bh, isd, bw = dn_info
                x_max = float(ax_a.get_xlim()[1])
                _draw_full_daynight_axis(ax_dn, bh, isd, bw, x_max)
                ax_dn.set_xlabel("Hours from recording start",
                                 fontsize=FS_LABEL + 1)
                ax_dn.tick_params(axis="x", labelsize=FS_TICK + 2)
            else:
                ax_dn.axis("off")

            # ---- BOTTOM block: c|d  →  rank colorbar (between c and e)  →  e|f
            cbar_h = 0.20
            bot_gs = gridspec.GridSpecFromSubplotSpec(
                3, 2, subplot_spec=outer[1],
                width_ratios=[5.0, 0.6],
                # Row 0: c|d (c heatmap slice = h_unit, matches e row).
                # Row 1: shared horizontal colorbar for c + e.
                # Row 2: e|f.
                height_ratios=[h_unit + strip_extra, cbar_h, h_unit],
                hspace=0.52, wspace=0.13,
            )

            # --- c: raw rank heatmap + per-event day/night sub-strip ---
            # The right column (d) uses a MIRRORED 2-row sub-gridspec so
            # that d's plotting region matches c's HEATMAP height exactly.
            c_sub_ratios = [h_unit, strip_extra]
            c_sub = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=bot_gs[0, 0],
                height_ratios=c_sub_ratios, hspace=0.06,
            )
            ax_c = fig.add_subplot(c_sub[0])
            display_ranks_raw = ranks[channel_order][:, display_events]
            im_rank = _plot_rank_heatmap(
                ax_c, display_ranks_raw, ordered_names, title="")
            ax_c.text(
                0.5, 1.06,
                f"raw rank heatmap  ·  n={valid_events.size}, "
                f"τ={all_tau:.3f}",
                transform=ax_c.transAxes, ha="center", va="bottom",
                fontsize=FS_TITLE - 3, fontweight="bold", color="#222",
            )
            ax_c.tick_params(axis="x", labelbottom=False)
            ax_c.set_xlabel("")
            sz_idx_c = []
            if sz_h_4d.size and event_h.size:
                for sz in sz_h_4d:
                    idx = int(np.argmin(np.abs(event_h - sz)))
                    if abs(event_h[idx] - sz) <= 2.0:
                        ax_c.axvline(idx + 0.5, color=COL_SEIZURE,
                                     lw=3.0, ls="--", alpha=1.0,
                                     zorder=10)
                        sz_idx_c.append(idx + 0.5)
            ax_c_strip = fig.add_subplot(c_sub[1], sharex=ax_c)
            strip = np.where(day_mask_event, 1, 0)[None, :]
            ax_c_strip.imshow(
                strip, aspect="auto", interpolation="nearest",
                cmap=ListedColormap([COL_NIGHT, COL_DAY]),
                vmin=0, vmax=1,
            )
            for x in sz_idx_c:
                ax_c_strip.axvline(x, color=COL_SEIZURE, lw=3.0,
                                   ls="--", alpha=1.0, zorder=10)
            if sz_idx_c:
                ax_c.scatter(
                    sz_idx_c, [n_ch + 0.25] * len(sz_idx_c),
                    marker="v", s=110, color=COL_SEIZURE,
                    edgecolors="white", linewidths=0.8,
                    zorder=11, clip_on=False,
                )
            ax_c_strip.set_yticks([])
            ax_c_strip.set_xticks([])
            for sp in ax_c_strip.spines.values():
                sp.set_visible(False)
            style_panel(ax_c, "")

            # --- d: per-channel rank distribution ---
            # Mirror c's sub-gridspec so d's drawing region has the same
            # height as c's heatmap (and not c+strip).
            d_sub = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=bot_gs[0, 1],
                height_ratios=c_sub_ratios, hspace=0.06,
            )
            ax_d = fig.add_subplot(d_sub[0])
            _plot_rank_histogram(
                ax_d, ranks, bools, valid_events,
                original_order, channel_names,
                title="",
            )
            ax_d.text(
                0.5, 1.05, "Per-channel ranks",
                transform=ax_d.transAxes, ha="center", va="bottom",
                fontsize=FS_TITLE - 4, fontweight="bold", color="#222",
            )
            style_panel(ax_d, "")

            # --- e: clustered rank heatmap ---
            ax_e = fig.add_subplot(bot_gs[2, 0])
            if has_best:
                disp_best_labels = best_labels[
                    np.isin(valid_events, display_events)]
                order_best = np.argsort(disp_best_labels, kind="stable")
                best_events_sorted = display_events[order_best]
                best_labels_sorted = disp_best_labels[order_best]
                display_ranks_best = ranks[channel_order][:, best_events_sorted]
                _plot_rank_heatmap(
                    ax_e, display_ranks_best, ordered_names, title="")
                cursor = 0
                for cid in np.unique(best_labels_sorted):
                    cnt = int(np.sum(best_labels_sorted == cid))
                    col = TEMPLATE_COLORS[int(cid) % len(TEMPLATE_COLORS)]
                    if cursor > 0:
                        ax_e.axvline(cursor, color=col, lw=2.0, ls="--",
                                     zorder=4)
                    ax_e.text(
                        (cursor + cnt / 2.0) / max(1, len(best_events_sorted)),
                        0.96,
                        f"C{int(cid)}  n={cnt}",
                        transform=ax_e.transAxes,
                        color="white", ha="center", va="top",
                        fontsize=FS_TICK, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25",
                                  facecolor=col, edgecolor="white",
                                  alpha=0.92),
                    )
                    cursor += cnt
                ax_e.text(
                    0.5, 1.05,
                    f"k={best_k} clustered  ·  "
                    f"within-τ={within_tau:.3f}  ·  "
                    f"inter-corr={inter_corr:.2f}",
                    transform=ax_e.transAxes, ha="center", va="bottom",
                    fontsize=FS_TITLE - 3, fontweight="bold", color="#222",
                )
            else:
                _plot_rank_heatmap(
                    ax_e, display_ranks_raw, ordered_names,
                    title="(no cluster labels)")
            ax_e.set_xlabel("Pop events (sorted by cluster)",
                            fontsize=FS_LABEL, labelpad=2)
            style_panel(ax_e, "")

            # --- f: cluster rank curves (TEMPLATE_COLORS) ---
            ax_f = fig.add_subplot(bot_gs[2, 1])
            if has_best:
                _plot_cluster_rank_curves_semantic(
                    ax_f, ranks, bools, valid_events, best_labels,
                    channel_order, channel_names,
                    title="Cluster ranks",
                )
            style_panel(ax_f, "")

            # --- rank colorbar between c and e: same *length* as before (inset
            # under e used width=32% of the heatmap column); host is left col only.
            ax_cbar_host = fig.add_subplot(bot_gs[1, 0])
            ax_cbar_host.set_axis_off()
            # Top of this row (toward c): loc=upper center on host; do not pass
            # bbox_to_anchor with %-sized inset (matplotlib requires a Bbox then).
            cax = inset_axes(
                ax_cbar_host, width="32%", height="58%",
                loc="upper center", borderpad=0,
            )
            cbar = fig.colorbar(im_rank, cax=cax, orientation="horizontal")
            cbar.set_label("rank: First → Last", fontsize=FS_LABEL,
                           labelpad=2)
            cbar.ax.tick_params(labelsize=FS_TICK - 1)

            # Panel letter for a sits inside its own top-left (va="top")
            # because a is tightly stacked above b — lifting it would push
            # it off the figure / above the suptitle. b's letter is
            # intentionally omitted (panel b inherits its identity from
            # the shared time axis with a). Letters c/d/e/f are lifted
            # ABOVE their panels (default va="bottom") so they read more
            # like classic panel labels.
            _place_panel_letters(fig, [(ax_a, "a")], dy=0.0, va="top")
            _place_panel_letters(fig, [
                (ax_c, "c"), (ax_d, "d"),
                (ax_e, "e"), (ax_f, "f"),
            ])

            title_main = (
                f"{ds}:{sub}  ·  repro = {repro_grade}    "
                f"dom_frac = "
                f"{rec_4d.get('summary',{}).get('dominant_rate_fraction', float('nan')):.3f},  "
                f"n = {rec_4d.get('n_events_used','?')}"
            )

        else:
            # PR-3 missing → fall back to a 2-panel PR-4D-only layout
            fig = plt.figure(figsize=(18, 11))
            outer = gridspec.GridSpec(
                3, 1, height_ratios=[1.0, 1.0, 0.15], hspace=0.10,
                left=0.07, right=0.985, top=0.92, bottom=0.07,
            )
            ax_a = fig.add_subplot(outer[0])
            ax_b = fig.add_subplot(outer[1], sharex=ax_a)
            _plot_pr4d_subject(ax_a, ax_b, rec_4d, sz_h=sz_h_4d)
            ax_a.set_title("")
            ax_a.tick_params(axis="x", labelbottom=False)
            ax_b.tick_params(axis="x", labelbottom=False)
            ax_b.set_xlabel("")
            style_panel(ax_a, "")
            style_panel(ax_b, "")
            ax_dn = fig.add_subplot(outer[2], sharex=ax_a)
            dn_info = daynight_by_key.get(k)
            if dn_info is not None:
                bh, isd, bw = dn_info
                x_max = float(ax_a.get_xlim()[1])
                _draw_full_daynight_axis(ax_dn, bh, isd, bw, x_max)
                ax_dn.set_xlabel("Hours from recording start",
                                 fontsize=FS_LABEL)
            else:
                ax_dn.axis("off")
            _place_panel_letters(fig, [(ax_a, "a")], dy=0.0, va="top")
            title_main = (
                f"{ds}:{sub}    "
                f"dom_frac = "
                f"{rec_4d.get('summary',{}).get('dominant_rate_fraction', float('nan')):.3f},  "
                f"n = {rec_4d.get('n_events_used','?')}"
            )

        # ----- shared title (single line; STRICT/enrich rows removed) -----
        _suptitle(fig, title_main, y=0.97)

        savefig_pub(fig, out_dir / f"{ds}_{sub}.png")

    print(f"  per-subject figures written to {out_dir}")
    _print_rate_cluster_seizure_table()
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="Generate all 5 figures")
    parser.add_argument("--fig", type=str, default="",
                        help="Comma-separated figure indices, e.g. 1,3,4")
    parser.add_argument("--per-subject", action="store_true",
                        help="Also write a Fig-5-style image for every "
                             "subject into per_subject/")
    args = parser.parse_args()

    fns = {1: plot_fig1, 2: plot_fig2, 3: plot_fig3,
           4: plot_fig4, 5: plot_fig5}

    targets = []
    if args.all:
        targets = sorted(fns.keys())
    elif args.fig:
        targets = [int(x.strip()) for x in args.fig.split(",") if x.strip()]
    elif not args.per_subject:
        parser.error("specify --all, --fig 1,2,..., or --per-subject")

    for i in targets:
        if i not in fns:
            print(f"  unknown figure {i}, skipping")
            continue
        fns[i]()

    if args.per_subject:
        plot_pr_per_subject_combined()


if __name__ == "__main__":
    main()
