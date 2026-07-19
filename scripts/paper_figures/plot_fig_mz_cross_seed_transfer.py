"""Cross-seed template transfer diagnostic for the MZ early-field bridge.

Plotting + light analysis only — consumes results/topic4_sef_hfo/mz_early_field_bridge/per_seed/*.
No simulation. Question: does a slow-off timing template predict a target pre-t120 field only when
template and target share the same random-noise seed, or does that readout transfer across seeds?

  cell(i,j) = seed_i slow-off direction templates (TA_i, TB_i)  vs  seed_j pre-t120 runaway energy field,
              scored by the mirror-invariant maxAB, with a maxAB-matched WITHIN-SHAFT null that permutes
              seed_j's energy (the target field) within shaft (design §9 reuse).

The 3 x 3 cells are descriptive repeated evaluations on three target fields, not nine independent
replicates. A lack of same-seed advantage weakens a pure common-random-number replay explanation. It
does not by itself prove that both directional branches are seed-invariant. Model proxy, not seizure.
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
            winner = "A_to_B" if (obs.get("rho_a") or -9) >= (obs.get("rho_b") or -9) else "B_to_A"
            win = assoc_a if winner == "A_to_B" else assoc_b
            cells[(i, j)] = {
                "template_seed": i, "energy_seed": j, "same_seed": i == j,
                "rho_maxab": obs.get("rho_maxab"), "rho_a": obs.get("rho_a"), "rho_b": obs.get("rho_b"),
                "winner_direction": winner,
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


def summarize(cells):
    rows = [cells[(i, j)] for i in SEEDS for j in SEEDS]
    maxab = _matrix(cells, "rho_maxab")
    template_spread = float(np.nanmean(np.nanstd(maxab, axis=0)))
    target_spread = float(np.nanstd(np.nanmean(maxab, axis=0)))
    diag = [cells[(s, s)]["rho_maxab"] for s in SEEDS]
    offdiag = [cells[(i, j)]["rho_maxab"] for i in SEEDS for j in SEEDS if i != j]
    matched = []
    for target in SEEDS:
        same = float(cells[(target, target)]["rho_maxab"])
        foreign = [float(cells[(source, target)]["rho_maxab"]) for source in SEEDS if source != target]
        matched.append({
            "target_energy_seed": target,
            "same_seed_maxab": same,
            "foreign_template_median_maxab": float(np.median(foreign)),
            "same_minus_foreign_median": float(same - np.median(foreign)),
        })
    winners = [str(row["winner_direction"]) for row in rows]
    return {
        "experiment": "MZ early-field bridge cross-seed template transfer diagnostic (E1146 model)",
        "framing": "template_seed x energy_seed; mirror-invariant maxAB; maxAB-matched within-shaft null "
                   "permuting the target energy field; three target fields are the repeated units; "
                   "model proxy, not seizure",
        "scientific_tier": "exploratory descriptive diagnostic; not a nine-replicate inferential test",
        "safe_interpretation": "Cross-seed templates predict each target field almost as well as its "
                               "same-seed template, weakening a pure same-noise replay explanation. "
                               "Because every cell selected B_to_A in this run, this does not establish "
                               "seed-invariance of both directional branches.",
        "seeds": list(SEEDS), "window": PRIMARY_WK, "n_perm": N_PERM,
        "n_independent_target_fields": len(SEEDS),
        "n_descriptive_template_target_cells": len(rows),
        "maxab_matrix_rows_template_cols_energy": maxab.tolist(),
        "within_shaft_p_matrix": _matrix(cells, "within_shaft_p").tolist(),
        "descriptive_mean_within_target_spread_across_template_seeds": template_spread,
        "descriptive_spread_across_target_field_means": target_spread,
        "matched_same_seed_vs_foreign_template": matched,
        "winner_direction_counts": {direction: winners.count(direction) for direction in sorted(set(winners))},
        "diagonal_same_seed_maxab": diag, "offdiagonal_transfer_maxab": offdiag,
        "cells": rows,
    }


def save(cells):
    summary = summarize(cells)
    rows = summary["cells"]
    json.dump(B.to_jsonable(summary), open(os.path.join(OUT, "cross_seed_transfer.json"), "w"), indent=2)
    keys = ["template_seed", "energy_seed", "same_seed", "winner_direction", "rho_maxab", "rho_a", "rho_b",
            "within_shaft_p", "field_cosine", "quartile_contrast", "top_k_overlap"]
    with open(os.path.join(OUT, "cross_seed_transfer.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(B.to_jsonable(r))
    return summary


def plot(cells, summary):
    maxab = _matrix(cells, "rho_maxab")
    seed_cols = {1: "#1f77b4", 3: "#2ca02c", 4: "#d62728"}
    fig = plt.figure(figsize=(15.5, 5.0), facecolor="white")
    gs = fig.add_gridspec(1, 3, wspace=0.34, left=0.055, right=0.975, top=0.82, bottom=0.14)

    # Panel A: 3x3 transfer heatmap (maxAB), rows=template seed, cols=energy seed
    axA = fig.add_subplot(gs[0, 0])
    im = axA.imshow(maxab, cmap="viridis", vmin=0.0, vmax=1.0, aspect="equal")
    for a in range(len(SEEDS)):
        for b in range(len(SEEDS)):
            axA.text(b, a, f"{maxab[a, b]:.2f}", ha="center", va="center",
                     color="white" if maxab[a, b] < 0.6 else "black", fontsize=11, fontweight="bold")
    axA.set_xticks(range(len(SEEDS))); axA.set_xticklabels([f"seed {s}" for s in SEEDS])
    axA.set_yticks(range(len(SEEDS))); axA.set_yticklabels([f"seed {s}" for s in SEEDS])
    axA.set_xlabel("energy seed (target field)"); axA.set_ylabel("template seed")
    axA.set_title("A · descriptive transfer maxAB\n(all 9 winners: B→A)", fontsize=11, fontweight="bold")
    cb = plt.colorbar(im, ax=axA, fraction=0.046, pad=0.04); cb.set_label("mirror-inv maxAB", fontsize=9)

    # Panel B: exact matched null threshold for each template-target cell.
    axB = fig.add_subplot(gs[0, 1])
    for b, j in enumerate(SEEDS):
        for offset, i in zip((-0.10, 0.0, 0.10), SEEDS):
            v = cells[(i, j)]["rho_maxab"]
            x = b + offset
            axB.scatter(x, v, s=70, facecolors="none", edgecolors=seed_cols[i], linewidths=1.6, zorder=3,
                        label=f"template seed {i}" if b == 0 else None)
            p95 = cells[(i, j)]["within_shaft_null_p95"]
            if p95 is not None:
                axB.plot([x - 0.035, x + 0.035], [p95, p95], color=seed_cols[i], lw=2, zorder=2)
            if i == j:
                axB.scatter(x, v, marker="*", s=190, color=seed_cols[i], edgecolors="k",
                            linewidths=0.5, zorder=4)
    axB.set_xticks(range(len(SEEDS))); axB.set_xticklabels([f"seed {s}" for s in SEEDS])
    axB.set_xlabel("energy seed (target field)"); axB.set_ylabel("mirror-inv maxAB")
    axB.set_ylim(-0.05, 1.02); axB.legend(fontsize=7.5, loc="lower left")
    rs = summary["descriptive_mean_within_target_spread_across_template_seeds"]
    csp = summary["descriptive_spread_across_target_field_means"]
    axB.set_title(f"B · transfer by target field\n(template spread {rs:.02f}; target spread {csp:.02f})",
                  fontsize=10, fontweight="bold")
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)

    # Panel C: direct matched question — is same-seed transfer stronger than foreign-seed transfer?
    axC = fig.add_subplot(gs[0, 2])
    matched = summary["matched_same_seed_vs_foreign_template"]
    delta = [row["same_minus_foreign_median"] for row in matched]
    axC.axhline(0, color="0.35", lw=1.0)
    axC.scatter(range(len(SEEDS)), delta, s=95, color=[seed_cols[s] for s in SEEDS],
                edgecolors="k", linewidths=0.5, zorder=3)
    for x, value in enumerate(delta):
        axC.plot([x, x], [0, value], color="0.65", lw=1.2, zorder=2)
        axC.text(x, value + (0.002 if value >= 0 else -0.002), f"{value:+.3f}",
                 ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    axC.set_xticks(range(len(SEEDS))); axC.set_xticklabels([f"seed {s}" for s in SEEDS])
    axC.set_xlabel("target energy seed")
    axC.set_ylabel("same-seed maxAB − median foreign-template maxAB")
    axC.set_ylim(-0.025, 0.025)
    axC.set_title("C · no descriptive same-seed advantage\n(three target fields; not n=9 inference)",
                  fontsize=10, fontweight="bold")
    for s in ("top", "right"):
        axC.spines[s].set_visible(False)

    fig.suptitle("MZ early-field bridge — cross-seed template transfer diagnostic (E1146 model)",
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
    print("template-seed spread (within target):",
          round(summary["descriptive_mean_within_target_spread_across_template_seeds"], 3),
          "| target-field spread:", round(summary["descriptive_spread_across_target_field_means"], 3))
    print("matched same-minus-foreign:",
          [round(x["same_minus_foreign_median"], 3)
           for x in summary["matched_same_seed_vs_foreign_template"]])
    plot(cells, summary)


if __name__ == "__main__":
    main()
