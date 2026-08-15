#!/usr/bin/env python
"""Figures for the multimode propagation-grammar audit.

Figure contract
---------------
Core conclusion: the extra modes of the K>2 subjects track the size of the
implanted contact set rather than the recording; each mode marks a distinct
earliest/latest contact pair on a near-exhausted rank lattice; no two-direction
family survives a random-ordering null; and the event-to-event mode sequence is
memoryless.  The patient data therefore resolve neither a fixed-direction
pathway nor a universal mode selector.

Panel = one question (CLAUDE.md §7):
  A  does the number of modes track the brain process or the electrode array?
  B  what is a mode, concretely, in the six K>2 subjects?
  C  do mode pairs differ by propagation order or by recruitment?
  D  is there any temporal mode selector?
Supplement:
  S1 transition matrices for the K=4 and K=6 subjects, observed and vs null
  S2 the random-ordering null that removes the two-direction reading

Style (docs/figure_style_guide.md §0):
  - propagation order / rank -> viridis, First -> Last, one shared colourbar
  - no jet / rainbow; K uses an ordinal single-hue violet ramp, distinct from
    both viridis and the reserved red-blue diverging scale
  - individual panels carry no corner letter (identity is the filename); the
    assembled `*_complete_layout` file carries the shared A-D letters
  - no internal jargon in reader-facing text
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "legend.frameon": False,
    "figure.dpi": 150,
})

# ordinal K ramp: grey for the K=2 majority, single-hue violet for K=3..6
K_COLORS = {2: "#B4B4B4", 3: "#D9BCE2", 4: "#B486CB", 5: "#8B4E9E", 6: "#5C2A6B"}
NEUTRAL_DARK = "#4D4D4D"
NEUTRAL_MID = "#767676"
ACCENT = "#B64342"

KGT2_ORDER = [
    "yuquan_huangwanling", "epilepsiae_818", "epilepsiae_916",
    "yuquan_zhangjinhan", "yuquan_zhourongxuan", "yuquan_zhaojinrui",
]


def pretty(sid: str) -> str:
    ds, sub = sid.split("_", 1)
    return f"{ds}:{sub}"


# ---------------------------------------------------------------------------
def load_all():
    rows = json.load(open(HERE / "cohort_summary.json"))["rows"]
    add = json.load(open(HERE / "cohort_addendum.json"))
    deep = json.load(open(HERE / "kgt2_deep_dive.json"))
    subs = {p.stem: json.load(open(p)) for p in (HERE / "per_subject").glob("*.json")}
    return rows, add, deep, subs


# ---------------------------------------------------------------------------
# A - number of modes vs size of the implanted contact set
# ---------------------------------------------------------------------------
def draw_panel_a(ax, ax_marg, rows, add, show_legend=True, legend_ncol=5):
    rng = np.random.default_rng(3)
    nch = np.array([r["n_channels"] for r in rows], float)
    K = np.array([r["chosen_k"] for r in rows], float)
    ids = [r["subject_id"] for r in rows]
    jitter = rng.uniform(-0.16, 0.16, size=K.size)
    ax.axvspan(3.4, 6.45, color="#F0EAF3", zorder=0)
    for k in sorted(set(K.astype(int))):
        m = K == k
        ax.scatter(nch[m], K[m] + jitter[m], s=26, c=K_COLORS[k],
                   edgecolors="white", linewidths=0.5, zorder=3,
                   label=f"K = {k}")
    # the six K>2 subjects carry the index used by panel b
    for n, sid in enumerate(KGT2_ORDER, start=1):
        i = ids.index(sid)
        ax.annotate(str(n), (nch[i], K[i] + jitter[i]), textcoords="offset points",
                    xytext=(7, 0), va="center", ha="left", fontsize=6.2,
                    color=NEUTRAL_DARK, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xticks([4, 6, 8, 12, 16, 24, 36, 52])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlim(3.4, 64)
    ax.set_ylim(1.45, 6.75)
    ax.set_yticks([2, 3, 4, 5, 6])
    ax.set_xlabel("Contacts contributing to propagation events")
    ax.set_ylabel("Number of modes, K")
    ax.text(4.55, 1.53, "≤ 6 contacts", fontsize=5.6, ha="center",
            color=NEUTRAL_MID)
    pr = add["chosen_k_predictors_spearman"]
    brho, bp = _blocks_rho(rows)
    # plain text, no mathtext subscripts: a script glyph from a 5.8 pt parent
    # would render below the 5 pt journal floor
    ax.text(0.985, 0.97,
            "Spearman correlation with K\n"
            "contacts   %+.2f  (P = %.0e)\n"
            "events      %+.2f  (P = %.2f)\n"
            "blocks      %+.2f  (P = %.2f)"
            % (pr["n_channels"]["spearman_rho_vs_chosen_k"], pr["n_channels"]["p"],
               pr["n_valid_events"]["spearman_rho_vs_chosen_k"], pr["n_valid_events"]["p"],
               brho, bp),
            transform=ax.transAxes, ha="right", va="top", fontsize=5.8,
            color=NEUTRAL_DARK, linespacing=1.5)
    if show_legend:
        ax.legend(loc="center left", bbox_to_anchor=(0.24, 0.60), handletextpad=0.35,
                  labelspacing=0.28, ncol=legend_ncol, columnspacing=0.9)

    cnt = {k: int(np.sum(K == k)) for k in sorted(set(K.astype(int)))}
    ax_marg.barh(list(cnt.keys()), list(cnt.values()), height=0.62,
                 color=[K_COLORS[k] for k in cnt], edgecolor="white", linewidth=0.5)
    for k, v in cnt.items():
        ax_marg.text(v + 1.2, k, str(v), va="center", fontsize=5.8, color=NEUTRAL_DARK)
    ax_marg.set_ylim(1.45, 6.75)
    ax_marg.set_yticks([])
    ax_marg.set_xlim(0, 44)
    ax_marg.set_xticks([0, 34])
    ax_marg.set_xlabel("Subjects")
    ax_marg.spines["left"].set_visible(False)


def _blocks_rho(rows):
    from scipy import stats
    r = stats.spearmanr(np.array([x["n_blocks_used"] for x in rows], float),
                        np.array([x["chosen_k"] for x in rows], float))
    return float(r.statistic), float(r.pvalue)


# ---------------------------------------------------------------------------
# B - what a mode is: masked rank prototypes of the six K>2 subjects
# ---------------------------------------------------------------------------
def draw_panel_b(axes, cax, ax_key, subs):
    for n, (ax, sid) in enumerate(zip(axes, KGT2_ORDER), start=1):
        d = subs[sid]
        a2 = d["analysis2_direction_extent"]
        names = d["engineering_audit"]["channel_names"]
        k = d["engineering_audit"]["chosen_k"]
        P = np.array([[np.nan if v is None else v for v in m["prototype_masked_rank"]]
                      for m in a2["modes"]]).T          # contacts x modes
        im = ax.imshow(P, cmap="viridis", vmin=0, vmax=1, aspect="auto",
                       interpolation="nearest")
        for j in range(k):
            col = P[:, j]
            if np.all(np.isnan(col)):
                continue
            ax.plot(j, int(np.nanargmin(col)), marker="v", ms=4.2, mfc="white",
                    mec="#272727", mew=0.5, ls="none")
            ax.plot(j, int(np.nanargmax(col)), marker="^", ms=4.2, mfc="white",
                    mec="#272727", mew=0.5, ls="none")
        ax.set_xticks(range(k))
        ax.set_xticklabels(range(1, k + 1))
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=5.6)
        ax.set_xlabel("Mode")
        ds, sub = sid.split("_", 1)
        ax.set_title(f"{n}   {ds}\n{sub}", fontsize=5.9, color=NEUTRAL_DARK, pad=3)
        ax.tick_params(length=1.8, pad=1.2)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = plt.colorbar(im, cax=cax)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["First", "Last"])
    cb.set_label("Firing order within event", labelpad=2, fontsize=6)
    cb.outline.set_visible(False)
    cax.tick_params(length=0, pad=1.5)
    ax_key.axis("off")
    ax_key.legend(
        handles=[
            Line2D([], [], marker="v", ls="none", ms=4.2, mfc="white", mec="#272727",
                   mew=0.5, label="earliest contact of the mode"),
            Line2D([], [], marker="^", ls="none", ms=4.2, mfc="white", mec="#272727",
                   mew=0.5, label="latest contact of the mode"),
        ],
        loc="center", ncol=2, handletextpad=0.3, columnspacing=2.2)


# ---------------------------------------------------------------------------
# C - direction versus recruitment for every mode pair
# ---------------------------------------------------------------------------
def draw_panel_c(ax, subs, show_legend=True):
    import csv
    pairs = list(csv.DictReader(open(HERE / "mode_pairs.csv")))
    xs, ys, cs, ks = [], [], [], []
    for p in pairs:
        rho = p["spearman_rho"]
        if rho in ("", "nan", "None"):
            continue
        k = int(float(p["chosen_k"]))
        xs.append(float(rho))
        ys.append(abs(float(p["recruited_fraction_diff"])))
        cs.append(K_COLORS[k])
        ks.append(k)
    xs, ys, ks = np.array(xs), np.array(ys), np.array(ks)
    rng = np.random.default_rng(11)
    yj = ys + rng.uniform(-0.005, 0.005, ys.size)
    y_lo, y_hi = -0.118, 0.44
    strip_top = -0.033
    ax.add_patch(mpl.patches.Rectangle((-1.06, y_lo), 2.12, strip_top - y_lo,
                                       facecolor="#F4F4F4", edgecolor="none", zorder=0))
    ax.axvspan(-1.06, -0.5, ymin=0, ymax=1, color="#F3EFEA", zorder=0)
    ax.axvline(0, color=NEUTRAL_MID, lw=0.6, ls=(0, (3, 3)), zorder=1)

    order = np.argsort(ks)                      # K=2 drawn first, K>2 on top
    ax.scatter(xs[order], yj[order], s=24, c=[cs[i] for i in order],
               edgecolors="white", linewidths=0.5, zorder=3)

    # 95% range of the prototype rank correlation between two RANDOM contact
    # orderings, for the two array sizes that carry the K>2 subjects
    for m, y, lab in [(4, -0.058, "4 contacts"), (6, -0.093, "6 contacts")]:
        half = min(1.96 / np.sqrt(m - 1), 1.0)
        ax.plot([-half, half], [y, y], color=NEUTRAL_DARK, lw=1.0,
                solid_capstyle="butt", zorder=4)
        ax.plot([-half, half], [y, y], marker="|", ms=3.6, color=NEUTRAL_DARK,
                ls="none", zorder=4)
        ax.text(-1.04, y + 0.014, lab, fontsize=5.2, va="center", ha="left",
                color=NEUTRAL_DARK, zorder=5)
    ax.text(1.04, -0.033, "95% range for random contact orderings",
            fontsize=5.4, va="top", ha="right", color=NEUTRAL_DARK)
    ax.text(-1.02, 0.425, "opposite order", fontsize=5.4, va="top", color=NEUTRAL_MID)

    ax.set_xlim(-1.06, 1.06)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.spines["left"].set_bounds(0, 0.4)
    ax.set_xlabel("Prototype rank correlation between two modes")
    ax.set_ylabel("| difference in recruited-contact fraction |")
    if show_legend:
        ax.legend(handles=[Line2D([], [], marker="o", ls="none", ms=4.4,
                                  mfc=K_COLORS[k], mec="white", mew=0.5, label=f"K = {k}")
                           for k in sorted(K_COLORS)],
                  loc="upper right", handletextpad=0.3, labelspacing=0.22, ncol=2,
                  columnspacing=0.8)


# ---------------------------------------------------------------------------
# D - is the mode sequence anything other than memoryless?
# ---------------------------------------------------------------------------
def draw_panel_d(ax, rows, show_legend=True):
    H = np.array([r["normalized_entropy"] for r in rows], float)
    E = np.array([r["excess_switch_rate"] for r in rows], float)
    K = np.array([r["chosen_k"] for r in rows], int)
    ids = [r["subject_id"] for r in rows]
    ax.axhline(0, color=NEUTRAL_DARK, lw=0.8, zorder=2)
    for k in sorted(set(K)):
        m = K == k
        ax.scatter(H[m], E[m], s=26, c=K_COLORS[k], edgecolors="white",
                   linewidths=0.5, zorder=3, label=f"K = {k}")
    offsets = {1: (-9.0, -3.0), 2: (0, -9.0), 3: (0, 9.0),
               4: (9.5, 1.5), 5: (0, 9.0), 6: (-11.0, -5.0)}
    for n, sid in enumerate(KGT2_ORDER, start=1):
        i = ids.index(sid)
        dx, dy = offsets[n]
        ax.annotate(str(n), (H[i], E[i]), textcoords="offset points", xytext=(dx, dy),
                    ha="center", va="center", fontsize=6.2, fontweight="bold",
                    color=NEUTRAL_DARK, zorder=4)
    ax.text(0.8235, 0.0028, "no memory (block-shuffled null)", fontsize=5.6,
            color=NEUTRAL_DARK, va="bottom")
    ax.set_xlim(0.822, 1.014)
    ax.set_ylim(-0.078, 0.026)
    ax.set_xticks([0.85, 0.90, 0.95, 1.00])
    ax.set_xlabel("Evenness of mode use  (entropy / log K)")
    ax.set_ylabel("Excess switch rate vs null")
    if show_legend:
        ax.legend(loc="lower left", handletextpad=0.3, labelspacing=0.22, ncol=2,
                  columnspacing=0.8)


# ---------------------------------------------------------------------------
# supplement
# ---------------------------------------------------------------------------
def _null_expected_transitions(per_block, k):
    """E[count(i->j)] under the within-block occupancy-preserving permutation.

    For one block with class counts n_i and n events, a uniformly random
    arrangement puts n_i (n_j - delta_ij) / n of the n-1 adjacent pairs on the
    ordered pair (i, j).  Summing over blocks gives the null expectation used
    as the reference for the observed transition matrix.
    """
    exp = np.zeros((k, k))
    for b in per_block:
        c = np.asarray(b["counts"], float)
        n = c.sum()
        if n < 2:
            continue
        exp += (np.outer(c, c) - np.diag(c)) / n
    return exp


def draw_supp_s1(subs):
    ids = ["epilepsiae_818", "epilepsiae_916", "yuquan_zhaojinrui"]
    P_all, D_all = {}, {}
    for sid in ids:
        a1 = subs[sid]["analysis1_occupancy_transitions"]
        k = subs[sid]["engineering_audit"]["chosen_k"]
        obs_c = np.array(a1["transition_counts"], float)
        exp_c = _null_expected_transitions(a1["per_block"], k)
        P = obs_c / obs_c.sum(1, keepdims=True)
        P_all[sid] = P
        D_all[sid] = P - exp_c / exp_c.sum(1, keepdims=True)
    pmax = max(float(P.max()) for P in P_all.values())
    lim = float(max(np.abs(D).max() for D in D_all.values()))

    fig = plt.figure(figsize=(7.09, 4.15))
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 0.05],
                  wspace=0.34, hspace=0.40, left=0.072, right=0.855,
                  top=0.885, bottom=0.105)
    im1 = im2 = None
    for c, sid in enumerate(ids):
        k = subs[sid]["engineering_audit"]["chosen_k"]
        ax = fig.add_subplot(gs[0, c])
        im1 = ax.imshow(P_all[sid], cmap="Greys", vmin=0, vmax=pmax,
                        interpolation="nearest")
        _tm_axes(ax, k, f"{pretty(sid)}   K = {k}")
        ax.set_ylabel("Mode of event $n$" if c == 0 else "")
        ax2 = fig.add_subplot(gs[1, c])
        im2 = ax2.imshow(D_all[sid], cmap="RdBu_r", vmin=-lim, vmax=lim,
                         interpolation="nearest")
        _tm_axes(ax2, k, "")
        ax2.set_xlabel("Mode of event $n+1$")
        ax2.set_ylabel("Mode of event $n$" if c == 0 else "")
    cb1 = fig.colorbar(im1, cax=fig.add_subplot(gs[0, 3]))
    cb1.set_label("Transition probability", labelpad=3, fontsize=6.4)
    cb1.outline.set_visible(False)
    cb2 = fig.colorbar(im2, cax=fig.add_subplot(gs[1, 3]))
    cb2.set_label("Observed − memoryless null", labelpad=3, fontsize=6.4)
    cb2.outline.set_visible(False)
    fig.text(0.072, 0.955, "Observed within-block transitions", fontsize=7,
             color=NEUTRAL_DARK)
    fig.text(0.072, 0.487, "Departure from the memoryless null", fontsize=7,
             color=NEUTRAL_DARK)
    _save(fig, "supp_s1_transition_matrices")


def _tm_axes(ax, k, title):
    ax.set_xticks(range(k)); ax.set_xticklabels(range(1, k + 1))
    ax.set_yticks(range(k)); ax.set_yticklabels(range(1, k + 1))
    ax.tick_params(length=1.8, pad=1.2)
    if title:
        ax.set_title(title, fontsize=6.4, color=NEUTRAL_DARK, pad=3)
    for s in ax.spines.values():
        s.set_visible(False)


def draw_supp_s2(deep):
    fig, axes = plt.subplots(1, 6, figsize=(7.09, 2.15), sharey=True)
    fig.subplots_adjust(left=0.098, right=0.992, top=0.775, bottom=0.185, wspace=0.14)
    for n, (ax, sid) in enumerate(zip(axes, KGT2_ORDER), start=1):
        s = deep["subjects"][sid]
        no = s["random_ordering_null"]
        obs = no["observed_best_split_separation"]
        samples = np.asarray(no["null_separation_samples"], float)
        parts = ax.violinplot([samples], positions=[0.42], widths=0.55,
                              showextrema=False, showmedians=False)
        for b in parts["bodies"]:
            b.set_facecolor("#DCDCDC")
            b.set_edgecolor("none")
            b.set_alpha(1.0)
        q = no["null_separation_percentiles"]
        ax.vlines(0.42, q["2.5"], q["97.5"], color=NEUTRAL_MID, lw=0.8, zorder=2)
        ax.vlines(0.42, q["25"], q["75"], color=NEUTRAL_DARK, lw=3.0, zorder=3)
        ax.plot([0.42], [q["50"]], marker="o", ms=2.0, mfc="white", mec="white",
                mew=0, ls="none", zorder=4)
        ax.plot([0.80], [obs], marker="o", ms=5.4, mfc=ACCENT, mec="white",
                mew=0.7, ls="none", zorder=5)
        ax.set_xlim(0, 1.15)
        ax.set_xticks([])
        ax.set_title(f"{n}   {pretty(sid)}\nK = {s['chosen_k']}", fontsize=5.9,
                     color=NEUTRAL_DARK, pad=3)
        ax.text(0.5, 0.015, "P = %.2f" % no["p_separation"], fontsize=6,
                ha="center", color=NEUTRAL_DARK, transform=ax.transAxes)
        ax.spines["bottom"].set_visible(False)
        if n > 1:
            ax.spines["left"].set_visible(False)
            ax.tick_params(left=False)
    axes[0].set_ylabel("Two-family separation\nof mode prototypes")
    axes[0].set_ylim(0, 1.72)
    fig.legend(handles=[
        Line2D([], [], marker="o", ls="none", ms=4.8, mfc=ACCENT, mec="white",
               mew=0.6, label="observed best two-way split"),
        Line2D([], [], color=NEUTRAL_DARK, lw=3.0,
               label="random contact orderings (median, quartiles, 95%)"),
    ], loc="lower center", ncol=2, bbox_to_anchor=(0.54, -0.01),
        handletextpad=0.4, columnspacing=1.8)
    _save(fig, "supp_s2_superfamily_random_ordering_null")


# ---------------------------------------------------------------------------
def _save(fig, stem):
    fig.savefig(FIG / f"{stem}.png", dpi=600)
    fig.savefig(FIG / f"{stem}.pdf")
    fig.savefig(FIG / f"{stem}.svg")
    plt.close(fig)
    print(f"  wrote figures/{stem}.png/.pdf/.svg")


def main() -> None:
    rows, add, deep, subs = load_all()

    # ---- individual panels, no corner letters (style guide §0.3) --------
    fig = plt.figure(figsize=(3.9, 2.85))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 0.2], wspace=0.08,
                  left=0.135, right=0.975, top=0.975, bottom=0.135)
    draw_panel_a(fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), rows, add,
                 legend_ncol=2)
    _save(fig, "panel_a_modes_vs_array_size")

    ks = [subs[s]["engineering_audit"]["chosen_k"] for s in KGT2_ORDER]
    fig = plt.figure(figsize=(7.09, 2.4))
    gs = GridSpec(2, 7, figure=fig, width_ratios=ks + [0.34],
                  height_ratios=[1, 0.10], wspace=0.62, hspace=0.55,
                  left=0.045, right=0.885, top=0.735, bottom=0.055)
    draw_panel_b([fig.add_subplot(gs[0, i]) for i in range(6)],
                 fig.add_subplot(gs[0, 6]), fig.add_subplot(gs[1, :]), subs)
    _save(fig, "panel_b_mode_prototypes")

    fig, ax = plt.subplots(figsize=(3.5, 2.85))
    fig.subplots_adjust(left=0.155, right=0.985, top=0.975, bottom=0.145)
    draw_panel_c(ax, subs)
    _save(fig, "panel_c_order_vs_recruitment")

    fig, ax = plt.subplots(figsize=(3.5, 2.85))
    fig.subplots_adjust(left=0.165, right=0.985, top=0.975, bottom=0.145)
    draw_panel_d(ax, rows)
    _save(fig, "panel_d_evenness_vs_switching")

    # ---- assembled layout: one shared K legend for the whole figure -----
    fig = plt.figure(figsize=(7.09, 7.9))
    outer = GridSpec(3, 1, figure=fig, height_ratios=[1.0, 0.86, 1.0],
                     hspace=0.42, left=0.078, right=0.885, top=0.955, bottom=0.058)
    ga = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], width_ratios=[1, 0.16],
                                 wspace=0.07)
    draw_panel_a(fig.add_subplot(ga[0]), fig.add_subplot(ga[1]), rows, add,
                 show_legend=True)
    gb = GridSpecFromSubplotSpec(2, 7, subplot_spec=outer[1], width_ratios=ks + [0.34],
                                 height_ratios=[1, 0.10], wspace=0.62, hspace=0.55)
    draw_panel_b([fig.add_subplot(gb[0, i]) for i in range(6)],
                 fig.add_subplot(gb[0, 6]), fig.add_subplot(gb[1, :]), subs)
    gc = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2], wspace=0.32)
    draw_panel_c(fig.add_subplot(gc[0]), subs, show_legend=False)
    draw_panel_d(fig.add_subplot(gc[1]), rows, show_legend=False)
    for lab, x, y in [("a", 0.004, 0.988), ("b", 0.004, 0.625),
                      ("c", 0.004, 0.345), ("d", 0.478, 0.345)]:
        fig.text(x, y, lab, fontsize=9, fontweight="bold", va="top")
    _save(fig, "multimode_grammar_complete_layout")

    draw_supp_s1(subs)
    draw_supp_s2(deep)


if __name__ == "__main__":
    main()
