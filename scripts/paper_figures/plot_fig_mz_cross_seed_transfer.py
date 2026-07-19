"""Cross-seed template transfer for the MZ early-ictal bridge (E1146 model).

Plotting + light analysis only — consumes results/topic4_sef_hfo/mz_early_field_bridge/per_seed/*.
No simulation. Question (2026-07-19 review 1st/3rd priority): is the interictal timing template a
property of the FIXED scaffold, or just a same-seed noise-replay coincidence? Test by transfer:

  cell(i,j) = seed_i slow-off direction templates (TA_i, TB_i)  vs  seed_j pre-t120 runaway energy field,
              scored by the mirror-invariant maxAB, with a maxAB-matched WITHIN-SHAFT null that permutes
              seed_j's energy (the target field) within shaft (design §9 reuse).

If maxAB is set by the ENERGY seed (column) and not by the TEMPLATE seed (row), the ordering template is
a fixed-scaffold property. Convergence across non-correlation metrics (field cosine, quartile contrast)
guards against a single-metric (Spearman) artifact. Model proxy, not seizure; direction-free maxAB.
"""
import csv
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
import numpy as np                                           # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys                                                   # noqa: E402
sys.path.insert(0, ROOT)
import src.topic4_mz_early_field_bridge as B                 # noqa: E402

OUT = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_early_field_bridge")
FIGDIR = os.path.join(ROOT, "results", "paper-ready-figure", "fig_mz_early_bridge", "figures")
SEEDS = (1, 3, 4)
PRIMARY_WK = "early_0_50_ms"
N_PERM, SEED = 10000, 0


def _shaft(name):
    m = re.match(r"[A-Za-z]+", str(name))
    return m.group(0) if m else str(name)


def _load_seed(seed):
    d = os.path.join(OUT, "per_seed", f"seed{seed}")
    tm = np.load(os.path.join(d, "templates.npz"), allow_pickle=True)
    na = np.load(os.path.join(d, "native.npz"), allow_pickle=True)
    names = [str(x) for x in na["names"]]
    return {"A": np.asarray(tm["contact_A"], float), "B": np.asarray(tm["contact_B"], float),
            "energy": np.asarray(na[f"contact_energy__{PRIMARY_WK}"], float), "names": names}


def compute():
    data = {s: _load_seed(s) for s in SEEDS}
    names0 = data[SEEDS[0]]["names"]
    for s in SEEDS:                                          # montage is seed-independent -> identical order
        assert data[s]["names"] == names0, f"contact order differs for seed {s}"
    shafts = np.array([_shaft(n) for n in names0], object)

    cells = {}                                               # (i_template, j_energy) -> metrics
    for i in SEEDS:
        ra, rb = data[i]["A"], data[i]["B"]
        for j in SEEDS:
            e = data[j]["energy"]
            sa, sb = np.isfinite(ra), np.isfinite(rb)
            obs = B.maxab_observed(ra, rb, e, support_a=sa, support_b=sb)
            null = B.maxab_permutation_null(ra, rb, e, support_a=sa, support_b=sb, groups=shafts,
                                            n_permutations=N_PERM, seed=SEED)
            assoc_a = B.associate(ra, e)
            assoc_b = B.associate(rb, e)
            win = assoc_a if (obs.get("rho_a") or -9) >= (obs.get("rho_b") or -9) else assoc_b
            cells[(i, j)] = {
                "template_seed": i, "energy_seed": j, "same_seed": i == j,
                "rho_maxab": obs.get("rho_maxab"), "rho_a": obs.get("rho_a"), "rho_b": obs.get("rho_b"),
                "within_shaft_p": null.get("p_one_sided"), "within_shaft_null_p95": null.get("null_p95"),
                "field_cosine": win.get("field_cosine"), "quartile_contrast": win.get("quartile_contrast"),
                "top_k_overlap": win.get("top_k_overlap"),
            }
    return cells, names0


def _matrix(cells, key):
    M = np.full((len(SEEDS), len(SEEDS)), np.nan)
    for a, i in enumerate(SEEDS):
        for b, j in enumerate(SEEDS):
            v = cells[(i, j)].get(key)
            if v is not None:
                M[a, b] = v
    return M


def save(cells):
    rows = [cells[(i, j)] for i in SEEDS for j in SEEDS]
    maxab = _matrix(cells, "rho_maxab")
    row_spread = float(np.nanmean(np.nanstd(maxab, axis=0)))     # spread across TEMPLATE seeds (within a column)
    col_spread = float(np.nanstd(np.nanmean(maxab, axis=0)))     # spread across ENERGY seeds (column means)
    diag = [cells[(s, s)]["rho_maxab"] for s in SEEDS]
    offdiag = [cells[(i, j)]["rho_maxab"] for i in SEEDS for j in SEEDS if i != j]
    summary = {
        "experiment": "MZ early-ictal bridge cross-seed template transfer (E1146 model)",
        "framing": "template_seed x energy_seed; mirror-invariant maxAB; maxAB-matched within-shaft null "
                   "permuting the target energy field; model proxy, not seizure",
        "seeds": list(SEEDS), "window": PRIMARY_WK, "n_perm": N_PERM,
        "maxab_matrix_rows_template_cols_energy": maxab.tolist(),
        "within_shaft_p_matrix": _matrix(cells, "within_shaft_p").tolist(),
        "mean_within_column_spread_across_template_seeds": row_spread,
        "spread_across_energy_seed_column_means": col_spread,
        "energy_seed_dominates_template_seed": bool(col_spread > 2 * row_spread + 1e-9),
        "diagonal_same_seed_maxab": diag, "offdiagonal_transfer_maxab": offdiag,
        "cells": rows,
    }
    json.dump(B.to_jsonable(summary), open(os.path.join(OUT, "cross_seed_transfer.json"), "w"), indent=2)
    keys = ["template_seed", "energy_seed", "same_seed", "rho_maxab", "rho_a", "rho_b",
            "within_shaft_p", "field_cosine", "quartile_contrast", "top_k_overlap"]
    with open(os.path.join(OUT, "cross_seed_transfer.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(B.to_jsonable(r))
    return summary


def plot(cells, summary):
    maxab = _matrix(cells, "rho_maxab")
    pmat = _matrix(cells, "within_shaft_p")
    seed_cols = {1: "#1f77b4", 3: "#2ca02c", 4: "#d62728"}
    fig = plt.figure(figsize=(15.5, 5.0), facecolor="white")
    gs = fig.add_gridspec(1, 3, wspace=0.34, left=0.055, right=0.975, top=0.82, bottom=0.14)

    # Panel A: 3x3 transfer heatmap (maxAB), rows=template seed, cols=energy seed
    axA = fig.add_subplot(gs[0, 0])
    im = axA.imshow(maxab, cmap="viridis", vmin=0.0, vmax=1.0, aspect="equal")
    for a in range(len(SEEDS)):
        for b in range(len(SEEDS)):
            star = "*" if (np.isfinite(pmat[a, b]) and pmat[a, b] < 0.05) else ""
            axA.text(b, a, f"{maxab[a, b]:.2f}{star}", ha="center", va="center",
                     color="white" if maxab[a, b] < 0.6 else "black", fontsize=11, fontweight="bold")
    axA.set_xticks(range(len(SEEDS))); axA.set_xticklabels([f"seed {s}" for s in SEEDS])
    axA.set_yticks(range(len(SEEDS))); axA.set_yticklabels([f"seed {s}" for s in SEEDS])
    axA.set_xlabel("energy seed (target field)"); axA.set_ylabel("template seed")
    axA.set_title("A · transfer maxAB  (* = within-shaft p<0.05)", fontsize=11, fontweight="bold")
    cb = plt.colorbar(im, ax=axA, fraction=0.046, pad=0.04); cb.set_label("mirror-inv maxAB", fontsize=9)

    # Panel B: maxAB by ENERGY seed; 3 template-seed points per column converge -> scaffold property
    axB = fig.add_subplot(gs[0, 1])
    for b, j in enumerate(SEEDS):
        vals = [cells[(i, j)]["rho_maxab"] for i in SEEDS]
        for i, v in zip(SEEDS, vals):
            axB.scatter(b, v, s=70, facecolors="none", edgecolors=seed_cols[i], linewidths=1.6, zorder=3,
                        label=f"template seed {i}" if b == 0 else None)
        axB.scatter(b, cells[(j, j)]["rho_maxab"], marker="*", s=200, color=seed_cols[j],
                    edgecolors="k", linewidths=0.5, zorder=4)      # same-seed diagonal
        p95 = cells[(SEEDS[0], j)]["within_shaft_null_p95"]
        if p95 is not None:
            axB.plot([b - 0.25, b + 0.25], [p95, p95], color="0.4", lw=2, zorder=2)
    axB.set_xticks(range(len(SEEDS))); axB.set_xticklabels([f"seed {s}" for s in SEEDS])
    axB.set_xlabel("energy seed (target field)"); axB.set_ylabel("mirror-inv maxAB")
    axB.set_ylim(-0.05, 1.02); axB.legend(fontsize=7.5, loc="lower left")
    rs = summary["mean_within_column_spread_across_template_seeds"]
    csp = summary["spread_across_energy_seed_column_means"]
    axB.set_title(f"B · maxAB clusters by energy seed\n(template-seed spread {rs:.02f} << energy-seed spread {csp:.02f})",
                  fontsize=10, fontweight="bold")
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)

    # Panel C: metric convergence across the 9 cells (maxAB vs cosine and vs quartile contrast)
    axC = fig.add_subplot(gs[0, 2])
    mx = [cells[(i, j)]["rho_maxab"] for i in SEEDS for j in SEEDS]
    cos = [cells[(i, j)]["field_cosine"] for i in SEEDS for j in SEEDS]
    quart = [cells[(i, j)]["quartile_contrast"] for i in SEEDS for j in SEEDS]
    axC.scatter(mx, cos, s=55, color="#4C72B0", edgecolors="k", linewidths=0.4, label="field cosine")
    axC.scatter(mx, quart, s=55, color="#DD8452", edgecolors="k", linewidths=0.4, marker="^",
                label="quartile contrast (energy IQR)")
    axC.axhline(0, color="0.7", lw=0.8); axC.axvline(0, color="0.7", lw=0.8)
    axC.set_xlabel("mirror-inv maxAB (all 9 cells)"); axC.set_ylabel("non-correlation metric")
    axC.legend(fontsize=8, loc="upper left")
    axC.set_title("C · metric convergence\n(cosine & quartile agree with maxAB, all cells)",
                  fontsize=10, fontweight="bold")
    for s in ("top", "right"):
        axC.spines[s].set_visible(False)

    fig.suptitle("MZ early-ictal bridge — cross-seed template transfer (E1146 model)",
                 fontsize=13.5, fontweight="bold", y=0.965)
    os.makedirs(FIGDIR, exist_ok=True)
    png = os.path.join(FIGDIR, "fig_mz_cross_seed_transfer.png")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(FIGDIR, "fig_mz_cross_seed_transfer.pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[transfer] wrote {png} (+ .pdf)")


def main():
    cells, _ = compute()
    summary = save(cells)
    print("maxAB matrix (rows=template, cols=energy seed 1/3/4):")
    print(np.array(summary["maxab_matrix_rows_template_cols_energy"]).round(3))
    print("template-seed spread (within column):", round(summary["mean_within_column_spread_across_template_seeds"], 3),
          "| energy-seed spread (column means):", round(summary["spread_across_energy_seed_column_means"], 3),
          "| energy dominates:", summary["energy_seed_dominates_template_seed"])
    plot(cells, summary)


if __name__ == "__main__":
    main()
