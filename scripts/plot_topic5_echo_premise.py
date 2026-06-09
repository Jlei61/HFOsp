"""Topic 5 Stage-1 PREMISE figure: is the seizure-onset proxy similar to the interictal
template at all — and is that resemblance solid enough to build on?

Two independent panels (CLAUDE.md §7), honest about fragility:
  A = what "resemblance" looks like inside a patient (loose cloud, even at best);
  B = is the resemblance consistent across patients (per-subject + 3 cohort readouts).

This is the PREMISE, not the conclusion. Paper-grade self-contained, no codenames.
Archive: docs/archive/topic5/echo_gate/stage1_proxy_triage_2026-06-08.md
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

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

INK = "#222222"
POS = "#2c6fbb"      # blue — positive resemblance
NEG = "#c0392b"      # red — negative / no resemblance
GREY = "#9aa0a6"


def load_cohort():
    return json.load(open(OUT / "cohort_echo_summary.json"))


def load_subjects():
    return [json.load(open(p)) for p in sorted(PS.glob("*.json"))]


def _habit_and_template(s):
    """Return (habitual seizure order, best-matched template order) over common channels."""
    seiz = np.array(s["seizure_ranks"], float)
    with np.errstate(invalid="ignore"):
        habit = np.where(np.all(np.isnan(seiz), axis=0), np.nan, np.nanmean(seiz, axis=0))
    best, best_r = None, -2.0
    for t in s["template_ranks"]:
        t = np.array(t, float)
        m = np.isfinite(habit) & np.isfinite(t)
        if m.sum() >= 5:
            r = np.corrcoef(rankdata(habit[m]), rankdata(t[m]))[0, 1]
            if r > best_r:
                best_r, best = r, t
    return habit, best, best_r


def panel_a_examples(ax, subs):
    """Concrete PREMISE, rawest form: pool EVERY seizure's raw rank correlation with its
    best-matching template (per-seizure, not the averaged habitual order). The histogram
    shows the resemblance is modest and spreads down to / below zero — no cherry-pick."""
    rs = []
    for s in subs:
        _, t, _ = _habit_and_template(s)
        if t is None:
            continue
        t = np.asarray(t, float)
        seiz = np.array(s["seizure_ranks"], float)
        for k in range(seiz.shape[0]):
            m = np.isfinite(t) & np.isfinite(seiz[k])
            if m.sum() >= 8:
                rs.append(np.corrcoef(rankdata(t[m]), rankdata(seiz[k][m]))[0, 1])
    rs = np.array(rs)
    ax.hist(rs, bins=np.arange(-1.0, 1.05, 0.15), color=POS, alpha=0.75, edgecolor="white")
    ax.axvline(0, color=NEG, lw=1.4, ls="--", zorder=3)
    ax.axvline(np.median(rs), color=INK, lw=2.2, zorder=4)
    ax.text(np.median(rs), ax.get_ylim()[1] * 0.96, f" median {np.median(rs):.2f}",
            color=INK, fontsize=9, va="top")
    ax.text(0.02, 0.04, f"{rs.size} seizures pooled\n{int(np.mean(rs > 0) * 100)}% positive\n"
            f"range {rs.min():.2f} to {rs.max():.2f}", transform=ax.transAxes,
            fontsize=8.6, va="bottom", color=INK)
    ax.set_xlabel("Single-seizure resemblance to the template\n(rank correlation, raw)", fontsize=10)
    ax.set_ylabel("Number of seizures", fontsize=10)
    ax.set_title("A   Per seizure, the raw resemblance is modest\nand spreads down toward zero",
                 fontsize=11, loc="left")


def panel_b_solidity(ax, subs, cohort):
    """Per-subject resemblance-beyond-chance (per-seizure dots + median), sorted, with
    the three cohort readouts — exposes the fragility."""
    rows = []
    for s in subs:
        evs = [ps["channel"]["e_k"] for ps in s["per_seizure"]
               if ps.get("channel") and ps["channel"]["e_k"] is not None
               and np.isfinite(ps["channel"]["e_k"])]
        if evs:
            rows.append((np.median(evs), evs))
    rows.sort(key=lambda x: x[0])
    rng = np.random.default_rng(0)
    for i, (med, evs) in enumerate(rows):
        col = POS if med > 0 else NEG
        jit = (rng.random(len(evs)) - 0.5) * 0.32
        ax.scatter(evs, np.full(len(evs), i) + jit, s=14, color=col, alpha=0.4,
                   edgecolor="none", zorder=2)
        ax.plot([med, med], [i - 0.3, i + 0.3], color=col, lw=3.0, zorder=3)
    ax.axvline(0, color=INK, lw=1.3, ls="--", zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"patient {i+1}" for i in range(len(rows))], fontsize=8)
    ax.set_xlabel("Seizure–template resemblance beyond chance\n(per seizure ●, per-patient median ▮)",
                  fontsize=10)
    ax.set_title("B   8 of 10 positive, but 2 negative and a few\nlarge values carry it — not solid",
                 fontsize=11, loc="left")
    p = cohort["primary_channel_all"]
    txt = (f"cohort readouts\n"
           f"by magnitude:  p = {p['wilcoxon_p_onesided']:.3f}\n"
           f"by direction:  p = {p['sign_p_onesided']:.3f}  (8/10)\n"
           f"resample 95%:  [{p['boot_ci95'][0]:.2f}, {p['boot_ci95'][1]:.2f}]")
    ax.text(0.98, 0.03, txt, transform=ax.transAxes, fontsize=8.4, va="bottom", ha="right",
            family="monospace", bbox=dict(boxstyle="round", fc="white", ec=GREY, alpha=0.9))


def main():
    cohort = load_cohort()
    subs = load_subjects()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.3))
    panel_a_examples(axes[0], subs)
    panel_b_solidity(axes[1], subs, cohort)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)
    fig.suptitle("Premise check: the seizure-onset proxy resembles the interictal template "
                 "only weakly, and not yet solidly   (n = 10 patients)", fontsize=12.5, y=1.01)
    fig.tight_layout()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    out = OUT / "figures" / "echo_premise_resemblance.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
