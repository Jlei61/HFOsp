"""Cohort figure for MZ gradient-corridor stimulation — 2 independent panels, paper-ready.

Reads subject_effects.csv + per_seed_effects.csv. The result must be readable from the DISTRIBUTION of
points, not from text: no panel titles, no stat blocks on the axes, no stats inside legends (those live in
cohort_statistics.json and the figures/README). The cross-corridor-spread readout is intentionally NOT a
panel — on a common decoupled window it is ~+/-0.01 for every subject (no window-robust site difference),
so it is reported in text only, not drawn as if it were a result.

  A  site effect: per-subject middle-minus-endpoint runaway-free time (C_run filled, C_best open) with the
     zero line and BOTH cohort medians (all-available solid, complete-case dashed) — the two analysis sets
     shown side by side, neither crowned. Dots above 0 = middle better; below = endpoints better.
  B  seed stability: per-subject C_run at each seed (one hue per seed) — tight same-side cluster = stable
     site preference; a cluster straddling 0 = seed-dependent.
"""
import argparse
import csv
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_gradient_corridor_stimulation")

C_RUN = "#2a9d8f"
C_BEST = "#8856a7"
SEED_C = {"1": "#4c78a8", "3": "#f28e2b", "4": "#59a14f"}


def _read_csv(p):
    return list(csv.DictReader(open(p))) if os.path.isfile(p) else []


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _short(s):
    return s.replace("epilepsiae_", "E").replace("yuquan_", "Y-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=RES)
    args = ap.parse_args()
    figdir = os.path.join(args.res, "figures")
    os.makedirs(figdir, exist_ok=True)
    subj = [r for r in _read_csv(os.path.join(args.res, "subject_effects.csv")) if r.get("tier") == "primary"]
    seed_rows = _read_csv(os.path.join(args.res, "per_seed_effects.csv"))
    if not subj:
        print("no primary subject_effects; run aggregate first")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    subjects = [r["subject"] for r in subj]
    x = np.arange(len(subjects))
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax in (axA, axB):
        ax.axhline(0, color="#b0b0b0", lw=1.0, ls="--", zorder=1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks(x)
        ax.set_xticklabels([_short(s) for s in subjects], rotation=25, ha="right", fontsize=9)

    # ---- A: site effect, both analysis sets ----
    cru = np.array([_f(r.get("all_avail_c_run")) for r in subj])
    cbe = np.array([_f(r.get("all_avail_c_best")) for r in subj])
    axA.scatter(x, cru, color=C_RUN, s=70, zorder=3)
    axA.scatter(x, cbe, facecolors="none", edgecolors=C_BEST, s=64, lw=1.6, zorder=3)
    axA.axhline(np.nanmedian(cru), color=C_RUN, lw=2.0, alpha=0.7)
    cc = np.array([_f(r.get("complete_case_c_run")) for r in subj])
    axA.axhline(np.nanmedian(cc), color=C_RUN, lw=1.6, ls=(0, (4, 3)), alpha=0.7)
    axA.set_ylabel("middle − endpoint runaway-free time (ms)")
    axA.text(0.02, 0.94, "A", transform=axA.transAxes, fontsize=13, fontweight="bold")
    axA.legend(handles=[Line2D([], [], marker="o", ls="", color=C_RUN, label="C_run (vs avg endpoint)"),
                        Line2D([], [], marker="o", ls="", mfc="none", mec=C_BEST, label="C_best (vs best endpoint)"),
                        Line2D([], [], color=C_RUN, lw=2, label="cohort median (all seeds)"),
                        Line2D([], [], color=C_RUN, lw=1.6, ls="--", label="cohort median (complete-case)")],
               fontsize=7.4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)

    # ---- B: seed stability ----
    by = {s: {} for s in subjects}
    for r in seed_rows:
        if r["subject"] in by:
            by[r["subject"]][r["seed"]] = _f(r.get("c_run"))
    for i, s in enumerate(subjects):
        for sd, val in by[s].items():
            axB.scatter(i, val, color=SEED_C.get(sd, "#666"), s=60, zorder=3)
    axB.set_ylabel("C_run per seed (ms)")
    axB.text(0.02, 0.94, "B", transform=axB.transAxes, fontsize=13, fontweight="bold")
    axB.legend(handles=[Line2D([], [], marker="o", ls="", color=SEED_C[k], label=f"seed {k}")
                        for k in ("1", "3", "4")], fontsize=8, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(figdir, f"mz_gradient_corridor_stimulation_cohort.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] -> {figdir}/mz_gradient_corridor_stimulation_cohort.png")


if __name__ == "__main__":
    main()
