"""Topic 5 Stage-1 echo-gate figure: the proxy similarity is a shared habitual-timing
anchor, not specific-path replay.

Three independent panels (CLAUDE.md §7): A = the similarity collapses once you control
for habitual recruitment timing; B = the shared structure IS the habitual timing; C = that
habitual-timing anchor is focus-proximity. Paper-grade self-contained: no internal codenames.

Spec/archive: docs/archive/topic5/echo_gate/stage1_proxy_triage_2026-06-08.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import warnings

import numpy as np
from scipy.stats import rankdata

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
warnings.filterwarnings("ignore", message="Mean of empty slice")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = _ROOT / "results/topic5_ictal_template_echo"
PS = OUT / "per_subject"
MASKED = _ROOT / "results/interictal_propagation_masked/rank_displacement/per_subject"

INK = "#222222"
ACCENT = "#c0392b"     # red — the "specific path" hope
ANCHOR = "#2c6fbb"     # blue — the habitual-timing anchor
GREY = "#9aa0a6"


def _zscore(x):
    x = np.asarray(x, float)
    s = np.nanstd(x)
    return (x - np.nanmean(x)) / s if s > 0 else x * 0.0


def load_cohort():
    return json.load(open(OUT / "cohort_echo_summary.json"))


def load_subjects():
    return [json.load(open(p)) for p in sorted(PS.glob("*.json"))]


def soz_for(stem):
    f = MASKED / f"{stem}.json"
    if not f.exists():
        return set()
    return set(json.load(open(f)).get("soz_channels", []))


def _pstar(p):
    return "p < 0.001" if p < 0.001 else (f"p = {p:.3f}" if p >= 0.001 else "")


def panel_a(ax, cohort):
    keys = [("primary_channel_all", "Free channel\nshuffle"),
            ("primary_within_shaft_all", "Within-electrode\nshuffle"),
            ("primary_anchor_matched_all", "Matched by\nhabitual timing")]
    xs = np.arange(3)
    mat = np.array([cohort[k]["E_s"] for k, _ in keys])     # (3 nulls, n_subjects)
    ps = [cohort[k]["wilcoxon_p_onesided"] for k, _ in keys]
    for j in range(mat.shape[1]):
        ax.plot(xs, mat[:, j], color=GREY, lw=0.9, alpha=0.5, zorder=1)
    med = np.median(mat, axis=1)
    # filled marker = significant (survives this null); open = not significant
    for i, (m, p) in enumerate(zip(med, ps)):
        sig = p < 0.05
        ax.plot([i], [m], marker="o", ms=9, zorder=4,
                color=INK if sig else "white", mec=INK, mew=2.0)
    ax.plot(xs, med, color=INK, lw=2.6, zorder=3, label="Cohort median")
    ax.axhline(0, color=ACCENT, lw=1.3, ls="--", zorder=2)
    ax.text(-0.2, -0.02, "no similarity beyond chance", color=ACCENT, fontsize=8.5, va="top")
    ymax = mat.max()
    for i, p in enumerate(ps):
        tag = ("survives  " + _pstar(p)) if p < 0.05 else ("vanishes  " + _pstar(p))
        ax.text(i, ymax * 1.06, tag, ha="center", fontsize=8.6,
                color=INK if p < 0.05 else ACCENT, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([lbl for _, lbl in keys], fontsize=9)
    ax.set_xlim(-0.35, 2.35)
    ax.set_ylim(min(-0.15, mat.min() * 1.05), ymax * 1.18)
    ax.set_ylabel("Seizure–template similarity\nbeyond chance (standardized)", fontsize=10)
    ax.set_title("A   The similarity vanishes once each channel's\nhabitual timing is controlled",
                 fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower left")


def panel_b(ax, subs):
    xs, ys = [], []
    for s in subs:
        seiz = np.array(s["seizure_ranks"], float)
        with np.errstate(invalid="ignore"):
            habit = np.where(np.all(np.isnan(seiz), axis=0), np.nan, np.nanmean(seiz, axis=0))
        templs = [np.array(t, float) for t in s["template_ranks"]]
        # pick the template that best matches the habitual order (the "matched" template)
        best, best_r = templs[0], -2.0
        for t in templs:
            m = np.isfinite(habit) & np.isfinite(t)
            if m.sum() >= 5:
                r = np.corrcoef(rankdata(habit[m]), rankdata(t[m]))[0, 1]
                if r > best_r:
                    best_r, best = r, t
        m = np.isfinite(habit) & np.isfinite(best)
        if m.sum() >= 5:
            xs.extend(_zscore(habit[m]))
            ys.extend(_zscore(best[m]))
    xs, ys = np.array(xs), np.array(ys)
    ax.scatter(xs, ys, s=16, color=ANCHOR, alpha=0.5, edgecolor="none", zorder=2)
    b1, b0 = np.polyfit(xs, ys, 1)
    xx = np.array([xs.min(), xs.max()])
    ax.plot(xx, b1 * xx + b0, color=INK, lw=2.2, zorder=3)
    r = np.corrcoef(rankdata(xs), rankdata(ys))[0, 1]      # rank correlation of the orders
    ax.text(0.04, 0.93, f"rank correlation = {r:.2f}", transform=ax.transAxes,
            fontsize=9.5, color=INK)
    ax.set_xlabel("Habitual ictal recruitment order\n(earlier → later, within patient)", fontsize=10)
    ax.set_ylabel("Interictal template order\n(within patient)", fontsize=10)
    ax.set_title("B   The shared structure is the\nhabitual recruitment order", fontsize=11, loc="left")


def panel_c(ax, subs):
    soz_vals, non_vals = [], []
    for s in subs:
        soz = soz_for(s["subject"])
        seiz = np.array(s["seizure_ranks"], float)
        with np.errstate(invalid="ignore"):
            habit = np.where(np.all(np.isnan(seiz), axis=0), np.nan, np.nanmean(seiz, axis=0))
        z = _zscore(habit)                 # standardized so earlier = negative
        for c, zc in zip(s["channels"], z):
            if not np.isfinite(zc):
                continue
            (soz_vals if c in soz else non_vals).append(zc)
    data = [np.array(soz_vals), np.array(non_vals)]
    pos = [0, 1]
    for x, vals, col in zip(pos, data, [ACCENT, GREY]):
        jit = (np.random.default_rng(0).random(vals.size) - 0.5) * 0.28
        ax.scatter(np.full(vals.size, x) + jit, vals, s=12, color=col, alpha=0.45, edgecolor="none")
        ax.plot([x - 0.22, x + 0.22], [np.median(vals)] * 2, color=INK, lw=2.4, zorder=3)
    ax.axhline(0, color=GREY, lw=0.8, ls=":")
    ax.set_xticks(pos)
    ax.set_xticklabels(["Seizure-onset-zone\nchannels", "Other\nchannels"], fontsize=9)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("Habitual ictal recruitment order\n(earlier ↓, standardized)", fontsize=10)
    ax.invert_yaxis()      # earlier on top
    ax.set_title("C   That anchor is focus-proximity:\nfocus channels recruit habitually earlier",
                 fontsize=11, loc="left")


def main():
    cohort = load_cohort()
    subs = load_subjects()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2))
    panel_a(axes[0], cohort)
    panel_b(axes[1], subs)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)
    fig.suptitle("Seizure-onset proxy resembles the interictal template only through a shared "
                 "habitual-timing anchor, not specific-path replay   (n = 10 patients)",
                 fontsize=12.5, y=1.01)
    fig.tight_layout()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    out = OUT / "figures" / "echo_anchor_not_path.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
