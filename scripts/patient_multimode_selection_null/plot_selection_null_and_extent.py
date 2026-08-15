#!/usr/bin/env python
"""Figures for the selection-corrected null and the 916 recruitment-extent endpoint.

Figure contract
---------------
F1 core conclusion: once the mode-internal ordering concentration is compared
against a null that keeps every recruitment mask and re-runs the identical fixed-K
clustering on order-randomised events, only part of the cohort keeps an excess,
and none of it survives a null that additionally preserves each contact's own
tendency to fire early or late.

F2 core conclusion: a recruitment-extent split of `epilepsiae_916` fitted on
train blocks alone transfers to held-out blocks with the same class proportion,
the same contact participation profile and an identical firing order, so the two
strata differ in how far the recruitment reaches and not in which way it runs.

Panel = one question:
  F1     does the observed dominant-order concentration exceed each null?
  F2 a   which contacts take part in each stratum, train vs held-out?
  F2 b   in what order do they fire, train vs held-out?
  F2 c   is the cross-shaft recruitment above or below a size-matched draw?

Style (docs/figure_style_guide.md §0): relative firing order keeps the viridis
`First -> Last` sense on the shared axis; no jet/rainbow; one shared legend per
figure; individual panels carry no corner letter, the assembled layout does; no
internal status codes, no PASS/FAIL banner, no long conclusions inside the axes.
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
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "legend.frameon": False, "figure.dpi": 150,
})

NEUTRAL_DARK = "#4D4D4D"
NEUTRAL_MID = "#767676"
ACCENT = "#B64342"
NULL_ORDER = "#C9C9C9"      # order-randomised null
NULL_MARGINAL = "#8FA8B8"   # marginal-preserving sensitivity null
LOW_EXTENT = "#42949E"      # teal
HIGH_EXTENT = "#9A4D8E"     # violet

KGT2_ORDER = ["yuquan_huangwanling", "epilepsiae_818", "epilepsiae_916",
              "yuquan_zhangjinhan", "yuquan_zhourongxuan", "yuquan_zhaojinrui"]


def pretty(sid):
    ds, sub = sid.split("_", 1)
    return f"{ds}:{sub}"


def _save(fig, stem):
    fig.savefig(FIG / f"{stem}.png", dpi=600)
    fig.savefig(FIG / f"{stem}.pdf")
    fig.savefig(FIG / f"{stem}.svg")
    plt.close(fig)
    print(f"  wrote figures/{stem}.png/.pdf/.svg")


# ---------------------------------------------------------------------------
# F1  observed concentration vs the two selection-corrected nulls
# ---------------------------------------------------------------------------
def draw_f1(axes, data, maxent, show_legend=True):
    for n, (ax, sid) in enumerate(zip(axes, KGT2_ORDER), start=1):
        d = data[sid]
        obs = d["observed"]["equal_mode_weighted_dominant_order_concentration"]
        for pos, key, colour in ((0.30, "order", NULL_ORDER),
                                 (0.70, "maxent", NULL_MARGINAL)):
            null = (d["nulls"]["order"] if key == "order" else maxent[sid])
            draws = np.asarray(null.get("draws_concentration") or [], float)
            if draws.size:
                parts = ax.violinplot([draws], positions=[pos], widths=0.30,
                                      showextrema=False)
                for b in parts["bodies"]:
                    b.set_facecolor(colour); b.set_edgecolor("none"); b.set_alpha(1.0)
                q = np.percentile(draws, [2.5, 25, 50, 75, 97.5])
            else:
                s = null["concentration_null"]
                q = [s["q05"], s["q05"], s["q50"], s["q95"], s["q95"]]
            ax.vlines(pos, q[0], q[4], color=NEUTRAL_MID, lw=0.8, zorder=2)
            ax.vlines(pos, q[1], q[3], color=NEUTRAL_DARK, lw=2.6, zorder=3)
            ax.plot([pos], [q[2]], marker="o", ms=1.9, mfc="white", mec="white",
                    mew=0, ls="none", zorder=4)
        ax.axhline(obs, color=ACCENT, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.plot([0.30, 0.70], [obs, obs], marker="o", ms=4.6, mfc=ACCENT,
                mec="white", mew=0.7, ls="none", zorder=5)
        ax.set_xlim(0, 1.0)
        ax.set_xticks([])
        ax.set_title(f"{n}   {pretty(sid)}\nK = {d['chosen_k']}", fontsize=5.9,
                     color=NEUTRAL_DARK, pad=3)
        # the two P values are stacked and colour-keyed to their null, because
        # side by side they collide at this subplot width
        po = d["nulls"]["order"]["empirical_p_concentration_ge"]
        pm = maxent[sid]["empirical_p_concentration_ge"]
        for row, (pv, colour) in enumerate(((po, NULL_ORDER), (pm, NULL_MARGINAL))):
            y = -0.085 - 0.085 * row
            ax.plot([0.13], [y], marker="s", ms=3.2, mfc=colour, mec="none",
                    ls="none", transform=ax.get_xaxis_transform(), clip_on=False)
            ax.text(0.24, y, f"P = {pv:.3f}", fontsize=5.4, ha="left", va="center",
                    color=NEUTRAL_DARK, transform=ax.get_xaxis_transform())
        ax.tick_params(length=1.8, pad=1.5)
        ax.spines["bottom"].set_visible(False)
        if n > 1:
            ax.spines["left"].set_visible(False)
            ax.tick_params(left=False, labelleft=False)
    axes[0].set_ylabel("Dominant-order concentration\n(equal weight per mode)")
    axes[0].set_ylim(0, 0.72)
    if show_legend:
        axes[0].legend(handles=[
            Line2D([], [], marker="o", ls="none", ms=4.4, mfc=ACCENT, mec="white",
                   mew=0.6, label="observed"),
            mpl.patches.Patch(facecolor=NULL_ORDER, edgecolor="none",
                              label="order randomised"),
            mpl.patches.Patch(facecolor=NULL_MARGINAL, edgecolor="none",
                              label="marginals kept"),
        ], loc="upper left", bbox_to_anchor=(-0.03, 1.03), handletextpad=0.4,
            labelspacing=0.22).set_zorder(20)


# ---------------------------------------------------------------------------
# F1b  construction check: does each null keep/destroy what it claims?
# ---------------------------------------------------------------------------
def draw_f1b(ax, audit, maxent, show_legend=True):
    subs = KGT2_ORDER
    x = np.arange(len(subs))
    obs = np.array([audit["subjects"][s]["observed_per_contact_order_spread"] for s in subs])
    ordn = np.array([audit["subjects"][s]["order_null_spread"]["mean"] for s in subs])
    shuf = np.array([audit["subjects"][s]["marginal_null_spread"]["mean"] for s in subs])
    mx = np.array([maxent[s]["construction_check"]["null_per_contact_order_spread_mean"]
                   for s in subs])
    w = 0.2
    ax.bar(x - 1.5 * w, obs, w, color=ACCENT, edgecolor="white", linewidth=0.4,
           label="observed")
    ax.bar(x - 0.5 * w, ordn, w, color=NULL_ORDER, edgecolor="white", linewidth=0.4,
           label="order randomised")
    ax.bar(x + 0.5 * w, shuf, w, color="#D8B7A0", edgecolor="white", linewidth=0.4,
           label="shuffle-and-repair")
    ax.bar(x + 1.5 * w, mx, w, color=NULL_MARGINAL, edgecolor="white", linewidth=0.4,
           label="max-entropy")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}" for i in range(1, len(subs) + 1)], fontsize=6)
    ax.set_xlabel("Subject (numbering as above)")
    ax.set_ylabel("Spread of per-contact\nmean firing order")
    ax.set_ylim(0, max(obs.max(), shuf.max(), mx.max()) * 1.62)
    ax.tick_params(length=1.8, pad=1.5)
    if show_legend:
        ax.legend(loc="upper left", ncol=2, handletextpad=0.4, labelspacing=0.22,
                  columnspacing=1.0, fontsize=5.6)


# ---------------------------------------------------------------------------
# F2  the 916 extent endpoint, train vs held-out
# ---------------------------------------------------------------------------
def draw_f2(ax_prof, ax_rank, ax_am, ext, show_legend=True):
    names = ext["channel_names"]
    v = ext["variants"]["primary_1d_recruited_fraction"]
    x = np.arange(len(names))
    style = {"train": dict(ls="-", marker="o", ms=3.4),
             "heldout": dict(ls="--", marker="s", ms=3.2)}
    colour = {"low_extent": LOW_EXTENT, "high_extent": HIGH_EXTENT}

    for split in ("train", "heldout"):
        s = v["splits"][split]
        for lab in ("low_extent", "high_extent"):
            st = s[lab]
            ax_prof.plot(x, st["participation_profile"], color=colour[lab],
                         mfc=colour[lab], mec="white", mew=0.4, lw=1.0, **style[split])
            proto = [np.nan if p is None else p for p in st["prototype_masked_rank"]]
            ax_rank.plot(x, proto, color=colour[lab], mfc=colour[lab], mec="white",
                         mew=0.4, lw=1.0, **style[split])

    shafts = ext["shaft_of_channel"]
    for a in (ax_prof, ax_rank):
        a.set_xticks(x)
        a.set_xticklabels(names, fontsize=5.6)
        a.set_xlim(-0.4, len(names) - 0.6)
        b = next(i for i in range(1, len(shafts)) if shafts[i] != shafts[i - 1])
        a.axvline(b - 0.5, color=NEUTRAL_MID, lw=0.6, ls=(0, (2, 2)), zorder=0)
        a.tick_params(length=1.8, pad=1.5)
    ax_prof.set_ylabel("Contact participation rate")
    ax_prof.set_ylim(0, 1.05)
    ax_rank.set_ylabel("Relative firing order\n(0 = first, 1 = last)")
    ax_rank.set_ylim(0, 1.0)
    ax_prof.text(0.5, 1.02, "AM", transform=ax_prof.get_xaxis_transform(),
                 fontsize=5.6, ha="center", color=NEUTRAL_MID)
    ax_prof.text(3.5, 1.02, "AH", transform=ax_prof.get_xaxis_transform(),
                 fontsize=5.6, ha="center", color=NEUTRAL_MID)

    # cross-shaft recruitment against a size-matched draw
    pos, obs, chance, cols = [], [], [], []
    labels = []
    for i, split in enumerate(("train", "heldout")):
        s = v["splits"][split]
        for j, lab in enumerate(("low_extent", "high_extent")):
            st = s[lab]
            pos.append(i * 2.6 + j)
            obs.append(st["am_shaft_participation"])
            chance.append(st["size_matched_chance_am_participation"])
            cols.append(colour[lab])
            labels.append("low" if lab == "low_extent" else "high")
    ax_am.bar(pos, obs, width=0.72, color=cols, edgecolor="white", linewidth=0.5)
    for p, c in zip(pos, chance):
        ax_am.plot([p - 0.46, p + 0.46], [c, c], color="#272727", lw=1.6,
                   solid_capstyle="butt", zorder=5)
        ax_am.plot([p - 0.46, p + 0.46], [c, c], color="white", lw=3.2,
                   solid_capstyle="butt", zorder=4)
    ax_am.set_xticks(pos)
    ax_am.set_xticklabels(labels, fontsize=5.8)
    ax_am.set_ylim(0, 1.15)
    ax_am.set_ylabel("AM-shaft recruitment")
    ax_am.text(np.mean(pos[:2]), -0.155, "train", transform=ax_am.get_xaxis_transform(),
               ha="center", fontsize=6, color=NEUTRAL_DARK)
    ax_am.text(np.mean(pos[2:]), -0.155, "held-out",
               transform=ax_am.get_xaxis_transform(), ha="center", fontsize=6,
               color=NEUTRAL_DARK)
    ax_am.tick_params(length=1.8, pad=1.5)

    if show_legend:
        ax_prof.legend(handles=[
            Line2D([], [], color=LOW_EXTENT, lw=1.2, marker="o", ms=3.4,
                   mec="white", mew=0.4, label="low-extent"),
            Line2D([], [], color=HIGH_EXTENT, lw=1.2, marker="o", ms=3.4,
                   mec="white", mew=0.4, label="high-extent"),
            Line2D([], [], color=NEUTRAL_DARK, lw=1.0, ls="-", label="train"),
            Line2D([], [], color=NEUTRAL_DARK, lw=1.0, ls="--", label="held-out"),
            Line2D([], [], color=NEUTRAL_DARK, lw=1.2, ls="-",
                   label="size-matched draw"),
        ], loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=2,
            handletextpad=0.4, labelspacing=0.25, columnspacing=1.0)


# ---------------------------------------------------------------------------
def main():
    scn_dir = HERE / "selection_corrected_null"
    data = {}
    for sid in KGT2_ORDER:
        p = scn_dir / f"{sid}.json"
        if p.exists():
            data[sid] = json.load(open(p))
    mx_dir = HERE / "marginal_maxent_null"
    maxent = {sid: json.load(open(mx_dir / f"{sid}.json"))
              for sid in KGT2_ORDER if (mx_dir / f"{sid}.json").exists()}
    audit_path = HERE / "null_construction_audit.json"
    audit = json.load(open(audit_path)) if audit_path.exists() else None
    ext_path = HERE / "extent_endpoint_916" / "extent_endpoint_916.json"
    ext = json.load(open(ext_path)) if ext_path.exists() else None

    ready = len(data) == len(KGT2_ORDER) and len(maxent) == len(KGT2_ORDER)
    if ready:
        fig, axes = plt.subplots(1, 6, figsize=(7.09, 2.35), sharey=True)
        fig.subplots_adjust(left=0.112, right=0.992, top=0.775, bottom=0.235,
                            wspace=0.16)
        draw_f1(axes, data, maxent)
        _save(fig, "f1_dominant_order_vs_selection_corrected_null")
    else:
        print(f"  [skip F1] order nulls {len(data)}/6, max-entropy nulls {len(maxent)}/6")

    if ready and audit is not None:
        fig, ax = plt.subplots(figsize=(3.6, 2.4))
        fig.subplots_adjust(left=0.20, right=0.985, top=0.975, bottom=0.185)
        draw_f1b(ax, audit, maxent)
        _save(fig, "f1b_null_construction_check")

    if ext is not None:
        fig = plt.figure(figsize=(7.09, 2.5))
        gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.62], wspace=0.36,
                      left=0.075, right=0.99, top=0.95, bottom=0.20)
        draw_f2(fig.add_subplot(gs[0]), fig.add_subplot(gs[1]),
                fig.add_subplot(gs[2]), ext)
        _save(fig, "f2_916_extent_endpoint_train_heldout")

    if ready and ext is not None and audit is not None:
        fig = plt.figure(figsize=(7.09, 7.6))
        outer = GridSpec(3, 1, figure=fig, height_ratios=[1.0, 0.85, 1.0], hspace=0.62,
                         left=0.098, right=0.99, top=0.955, bottom=0.062)
        g1 = GridSpecFromSubplotSpec(1, 6, subplot_spec=outer[0], wspace=0.16)
        a1 = [fig.add_subplot(g1[i]) for i in range(6)]
        for a in a1[1:]:
            a.sharey(a1[0])
        draw_f1(a1, data, maxent, show_legend=True)
        gb = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], width_ratios=[1, 1.1],
                                     wspace=0.30)
        draw_f1b(fig.add_subplot(gb[0]), audit, maxent)
        fig.add_subplot(gb[1]).axis("off")
        g2 = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2],
                                     width_ratios=[1, 1, 0.62], wspace=0.36)
        draw_f2(fig.add_subplot(g2[0]), fig.add_subplot(g2[1]),
                fig.add_subplot(g2[2]), ext)
        for lab, x, y in [("a", 0.004, 0.988), ("b", 0.004, 0.640),
                          ("c", 0.004, 0.352), ("d", 0.352, 0.352), ("e", 0.700, 0.352)]:
            fig.text(x, y, lab, fontsize=9, fontweight="bold", va="top")
        _save(fig, "selection_null_and_extent_complete_layout")


if __name__ == "__main__":
    main()
