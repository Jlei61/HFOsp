"""Topic5 V2 Phase-1-v2 / W2 Task 2.3 — subject phenotype hunt + descriptive band-subgroups.

Pure post-processing of the Task 2.1/2.2 phenotype csv (phase1_v2_subject_phenotype.csv):
NO new nulls, NO simulation, NO KMeans / statistical subtype claim. Two INDEPENDENT
scientific questions -> two panels (CLAUDE.md §7); this figure must NOT re-draw the
existing Phase-1 F1 (subject x band heatmap) / F2 (per-band null violins) / F3
(per-subject n_sig bars).

  Panel A  band-gradient lean per subject, colored by tier: do subjects lean
           low-freq / HFA, or sit band-generic near 0? (expected: near 0, heterogeneous)
  Panel B  does any single feature predict multi-band positivity? |Spearman r| of the
           significant-band count vs each candidate trait, with the LOCKED gate |r|=0.4.
           (expected: no INDEPENDENT trait passes -- "no single clean phenotype")

Also writes two companion CSVs next to the phenotype csv:
  phase1_v2_subject_band_profile.csv        descriptive band_profile_group roster
  phase1_v2_phenotype_hunt_spearman.csv     full Spearman r/p table (both targets, both pools)

Style: figure_style_guide §0 -- tight axes, one shared legend per panel, colorblind-safe
qualitative tier palette (Okabe-Ito, NOT viridis / jet); the signed gradient sign is shown
by x-position vs the 0 line, NOT also red/blue color-encoded. English only (no CJK).
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_topic5_v2_subject_phenotype import (  # noqa: E402
    assign_band_profile_group,
    spearman_phenotype_gate,
    PHENOTYPE_TARGETS,
    PHENOTYPE_PREDICTORS,
    GATE_R,
)

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "results/topic5_ictal_recruitment/v2_band_scan"
FIGDIR = V2 / "figures"
PHENO_CSV = V2 / "phase1_v2_subject_phenotype.csv"
SUBS = [("narrow", 20), ("broad", 17)]

# Okabe-Ito colorblind-safe qualitative palette (NOT viridis / jet). Tier = color in A.
TIER_C = {"strong": "#0072B2", "directional": "#E69F00", "weak_absent": "#999999"}
TIER_ORDER = ["strong", "directional", "weak_absent"]
TIER_LABEL = {"strong": "strong (>=4 sig. bands)",
              "directional": "directional (>=5 positive)",
              "weak_absent": "weak / absent"}
POOL_C = {"narrow": "#CC79A7", "broad": "#009E73"}  # reddish-purple / bluish-green (CB-safe)

# Panel B candidate features: two groups. INDEP = genuinely independent subject traits;
# DERIVED = re-descriptions of the SAME per-band spatial-delta vector that the significant-
# band count is computed from (they co-vary by construction, so passing the gate does NOT
# make them an independent predictor). Display names only -- no raw column names on the fig.
INDEP_TRAITS = [
    ("n_sz", "# seizures"),
    ("n_contacts", "# contacts"),
    ("maxab_primary", "raw alignment magnitude"),
    ("within_subject_seizure_consistency", "cross-seizure consistency"),
    ("HF_minus_low", "band-gradient (HFA - low)"),
]
DERIVED_FEATS = [
    ("profile_entropy", "positive-band spread"),
    ("low_band_score", "low-band (1-13 Hz) alignment"),
    ("LVFA_band_score", "fast band (13-80 Hz) alignment"),
    ("HFA_ripple_score", "HFA/ripple (80-250 Hz) alignment"),
]


def _short(sid):
    return str(sid).replace("epilepsiae_", "E").replace("yuquan_", "Y:")


def _load():
    df = pd.read_csv(PHENO_CSV)
    df["subject"] = df["subject"].astype(str)
    df["band_profile_group"] = [
        assign_band_profile_group(r.HF_minus_low, r.band_genericity_index, r.tier)
        for r in df.itertuples()
    ]
    return df


# ---------------------------------------------------------------------------
# companion CSVs
# ---------------------------------------------------------------------------
def write_band_profile_csv(df):
    cols = ["subject", "substrate", "tier", "HF_minus_low",
            "band_genericity_index", "band_profile_group"]
    out = V2 / "phase1_v2_subject_band_profile.csv"
    df[cols].to_csv(out, index=False)
    print("wrote", out)


def spearman_table(df):
    rows = []
    for substrate, _ in SUBS:
        sub = df[df["substrate"] == substrate]
        for target in PHENOTYPE_TARGETS:
            for pred in PHENOTYPE_PREDICTORS:
                r, p, n, passes = spearman_phenotype_gate(sub[target].to_numpy(),
                                                          sub[pred].to_numpy())
                rows.append(dict(substrate=substrate, target=target, predictor=pred,
                                 r=r, p=p, n=n, passes_gate=passes))
    return pd.DataFrame(rows)


def write_spearman_csv(table):
    out = V2 / "phase1_v2_phenotype_hunt_spearman.csv"
    table.to_csv(out, index=False)
    print("wrote", out)


# ---------------------------------------------------------------------------
# beeswarm helper (deterministic; n small -> O(n^2) is fine)
# ---------------------------------------------------------------------------
def _swarm_y(xvals, x_tol, y_gap=1.0):
    xvals = np.asarray(xvals, dtype=float)
    order = np.argsort(xvals, kind="stable")
    ys = np.zeros(len(xvals))
    placed = []  # (x, y)
    levels = [0]
    for k in range(1, len(xvals) + 1):
        levels += [k, -k]
    for i in order:
        xi = xvals[i]
        for lv in levels:
            yc = lv * y_gap
            if all(abs(xi - px) >= x_tol or abs(yc - py) >= y_gap * 0.85
                   for px, py in placed):
                ys[i] = yc
                placed.append((xi, yc))
                break
    return ys


# ---------------------------------------------------------------------------
# Panel A — band-gradient lean per subject (colored by tier)
# ---------------------------------------------------------------------------
def _panel_A(ax, sub, substrate, n, show_xlabel):
    x = sub["HF_minus_low"].to_numpy(dtype=float)
    ys = _swarm_y(x, x_tol=0.012, y_gap=1.0)
    for tier in TIER_ORDER:
        m = sub["tier"].to_numpy() == tier
        if m.any():
            ax.scatter(x[m], ys[m], s=95, c=TIER_C[tier], edgecolors="black",
                       linewidths=0.6, alpha=0.92, zorder=4, label=TIER_LABEL[tier])
    # descriptive bucket cutoffs: 0 = band-generic (solid), +/-0.05 = descriptive edges (dotted)
    ax.axvline(0.0, color="black", lw=1.4, zorder=2)
    ax.axvline(-0.05, color="0.45", lw=1.0, ls=":", zorder=2)
    ax.axvline(+0.05, color="0.45", lw=1.0, ls=":", zorder=2)
    ax.set_yticks([])
    ax.set_ylim(ys.min() - 1.4, ys.max() + 2.2)
    ax.set_xlim(-0.30, 0.16)
    ax.set_ylabel(f"{substrate}\n(n={n})", fontsize=11, rotation=0, ha="right", va="center", labelpad=26)
    # region labels: x in data coords, y pinned near the top of the axes (blended transform)
    trans = ax.get_xaxis_transform()
    ax.text(-0.165, 0.93, "low-freq leaning", ha="center", va="center", fontsize=8.5,
            style="italic", color="0.30", transform=trans)
    ax.text(0.0, 0.93, "band-generic", ha="center", va="center", fontsize=8.5,
            style="italic", color="0.10", transform=trans)
    ax.text(0.105, 0.93, "HFA leaning", ha="center", va="center", fontsize=8.5,
            style="italic", color="0.30", transform=trans)
    ax.grid(alpha=0.18, axis="x")
    if show_xlabel:
        ax.set_xlabel("HFA (80-250 Hz) minus low-band (1-13 Hz) early-ictal alignment  "
                      "($\\Delta$;  <0 low-freq leaning,  >0 HFA leaning)", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)


# ---------------------------------------------------------------------------
# Panel B — does any single feature predict multi-band positivity?
# ---------------------------------------------------------------------------
def _panel_B(ax, df):
    target = "n_sig_7bands"  # multi-band positivity = # significant bands (of 7)
    stats = {}  # (substrate, col) -> (absr, p, passes)
    for substrate, _ in SUBS:
        sub = df[df["substrate"] == substrate]
        for col, _disp in INDEP_TRAITS + DERIVED_FEATS:
            r, p, n, passes = spearman_phenotype_gate(sub[target].to_numpy(),
                                                      sub[col].to_numpy())
            stats[(substrate, col)] = (abs(r) if np.isfinite(r) else 0.0, p, passes)

    # sort each group by narrow |r| descending, independent group on top
    def _sorted(group):
        return sorted(group, key=lambda cd: stats[("narrow", cd[0])][0], reverse=True)
    indep = _sorted(INDEP_TRAITS)
    derived = _sorted(DERIVED_FEATS)
    rows = [(disp, col, "indep") for col, disp in indep] + \
           [(disp, col, "derived") for col, disp in derived]

    n_rows = len(rows)
    yy = {i: n_rows - i for i in range(n_rows)}  # row 0 at top
    off = 0.19
    h = 0.36
    for i, (disp, col, grp) in enumerate(rows):
        y = yy[i]
        for substrate, dy in (("narrow", +off), ("broad", -off)):
            absr, p, passes = stats[(substrate, col)]
            ax.barh(y + dy, absr, height=h, color=POOL_C[substrate],
                    edgecolor="black", linewidth=0.4, alpha=0.92, zorder=3)
            if np.isfinite(p) and p < 0.05:
                ax.text(absr + 0.015, y + dy, "*", ha="left", va="center",
                        fontsize=14, fontweight="bold", color="black", zorder=5)

    ax.axvline(GATE_R, color="crimson", ls="--", lw=1.6, zorder=2)
    ax.text(GATE_R + 0.005, n_rows + 0.55, f"gate |r| = {GATE_R}", color="crimson",
            fontsize=9, ha="left", va="bottom")

    # group separator + labels
    sep = yy[len(indep) - 1] - 0.5
    ax.axhline(sep, color="0.5", lw=0.9, ls="-", zorder=1)
    ax.text(1.03, (yy[0] + yy[len(indep) - 1]) / 2, "independent\nsubject traits",
            ha="left", va="center", fontsize=8.5, color="0.20", rotation=0)
    ax.text(1.03, (yy[len(indep)] + yy[n_rows - 1]) / 2,
            "same alignment $\\Delta$\n(co-vary by\nconstruction)",
            ha="left", va="center", fontsize=8.5, color="0.20", rotation=0)

    ax.set_yticks([yy[i] for i in range(n_rows)])
    ax.set_yticklabels([disp for disp, _, _ in rows], fontsize=9)
    ax.set_ylim(0.3, n_rows + 1.1)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("| Spearman r |  with  # significant bands (of 7)", fontsize=10)
    ax.grid(alpha=0.22, axis="x")
    ax.tick_params(axis="x", labelsize=9)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=POOL_C["narrow"], ec="k"),
               plt.Rectangle((0, 0), 1, 1, fc=POOL_C["broad"], ec="k"),
               plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="k",
                          markeredgecolor="k", markersize=12, linestyle="none")]
    ax.legend(handles, ["narrow (n=20)", "broad (n=17)", "p < 0.05"],
              loc="upper right", fontsize=8.5, frameon=True, framealpha=0.9)


def make_figure(df):
    fig = plt.figure(figsize=(15.2, 8.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.18], height_ratios=[1, 1],
                          hspace=0.28, wspace=0.36)
    axA0 = fig.add_subplot(gs[0, 0])
    axA1 = fig.add_subplot(gs[1, 0], sharex=axA0)
    axB = fig.add_subplot(gs[:, 1])

    for ax, (substrate, n), show_x in ((axA0, SUBS[0], False), (axA1, SUBS[1], True)):
        _panel_A(ax, df[df["substrate"] == substrate], substrate, n, show_x)
    axA0.set_title("A  Band-gradient lean per subject (color = subject tier)",
                   fontsize=12, loc="left", fontweight="bold")
    handlesA, labelsA = axA0.get_legend_handles_labels()
    order = [labelsA.index(TIER_LABEL[t]) for t in TIER_ORDER if TIER_LABEL[t] in labelsA]
    axA0.legend([handlesA[i] for i in order], [labelsA[i] for i in order],
                loc="lower left", fontsize=8.5, frameon=True, framealpha=0.9, ncol=1)

    _panel_B(axB, df)
    axB.set_title("B  Does any single feature predict multi-band positivity?",
                  fontsize=12, loc="left", fontweight="bold")

    fig.suptitle("Topic 5 interictal-HFO / early-ictal alignment: subject band phenotype  -  "
                 "gradient is band-generic & heterogeneous, no independent trait predicts "
                 "multi-band positivity", fontsize=12.5, y=0.985)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / "phase1_v2_W2_subject_phenotype.png"
    fig.savefig(out, dpi=135, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return out


def _print_summary(df, table):
    print("\n########## band_profile_group rosters ##########")
    for substrate, _ in SUBS:
        sub = df[df["substrate"] == substrate]
        print(f"\n=== {substrate} (n={len(sub)}) ===")
        for g in ["low_leaning", "flat_generic", "hf_leaning", "weak_absent", "flat_lowgeneric"]:
            gg = sub[sub["band_profile_group"] == g]
            if len(gg):
                print(f"  {g:<16} n={len(gg):>2}  {[_short(s) for s in sorted(gg.subject)]}")
    print("\n########## phenotype-hunt gate (target = # significant bands) ##########")
    for substrate, _ in SUBS:
        t = table[(table.substrate == substrate) & (table.target == "n_sig_7bands")]
        passed = t[t.passes_gate].predictor.tolist()
        print(f"  {substrate}: gate-passing = {passed if passed else 'NONE'}")


def main():
    df = _load()
    write_band_profile_csv(df)
    table = spearman_table(df)
    write_spearman_csv(table)
    make_figure(df)
    _print_summary(df, table)


if __name__ == "__main__":
    main()
