"""Topic5 V2 Phase-1 paper-ready 图（3 张，每图一个独立科学问题；样式复刻
`plot_topic5_energy_field_cohort.py` + figure_style_guide §0）。

F1 观测层：subject×band maxAB 热图（viridis 顺序量）——有没有共结构、narrow>broad、band-generic。
F2 形式化 null：每 primary band 的 cohort delta（obs−弱空间 null）+ FWER 显著（crimson）——超没超过 null、ripple_high 最弱。
F3 per-subject caveat：每 subject 显著频带数 n_sig（of 7 primary）——cohort 6/7 是不是聚合（暴露 per-subject 弱）。

读现成 CSV，不重算。输出 results/.../v2_band_scan/figures/。
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon


def _holm(pvals):
    """Holm step-down FWER adjustment (inline; avoids a statsmodels dep)."""
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "results/topic5_ictal_recruitment/v2_band_scan"
FIGDIR = V2 / "figures"
cfg = yaml.safe_load(open(ROOT / "config/topic5_v2_phase1.yaml"))
PRIMARY = [b[0] for b in cfg["bands"]["primary"]]
COMPOSITE = [b[0] for b in cfg["bands"]["composites"]]
SHORT = {"delta_HYP_slow": "δ\n1-4", "theta_preictal_PAC": "θ\n4-8", "alpha_sharp_leq13": "α\n8-13",
         "beta_LVFA_low": "β\n13-30", "gamma_LVFA": "γ\n30-80", "hg_low_ripple": "R\n80-150",
         "ripple_high": "FR\n150-250", "low_HYP_1_13": "low\n1-13", "LVFA_13_80": "LVFA\n13-80",
         "ripple_full_80_250": "Rf\n80-250", "ripple_safe_80_220": "Rs\n80-220"}
STRENGTH_C = {"within_shaft_strong": "#08519c", "distance_bin_fallback": "#6baed6",
              "subject_wide_weak": "#c6dbef"}   # 深=强 null，浅=弱 null
SUBS = [("narrow", 20), ("broad", 17)]


def _short(sid):
    return sid.replace("epilepsiae_", "E").replace("yuquan_", "Y:")


def _load(sub):
    a = pd.read_csv(V2 / sub / "phase1_alignment_raw_subject_summary.csv")
    a = a[a.used_fixed_mask == True]
    n = pd.read_csv(V2 / sub / "phase1_null_raw_subject_summary.csv")
    g = pd.read_csv(V2 / sub / "phase1_gate_summary.csv")
    return a, n, g


# ---------------------------------------------------------------- F1 observed heatmap
def fig1_observed():
    bands = PRIMARY + COMPOSITE
    norm = TwoSlopeNorm(vmin=0.35, vcenter=0.5, vmax=0.90)                    # < 0.5 blue, > 0.5 red
    for sub, n in SUBS:                                                       # narrow / broad = 独立图
        a, nl, _ = _load(sub)
        piv = a.pivot_table(index="subject", columns="band", values="align_abs_maxab", aggfunc="median")
        piv = piv.reindex(columns=bands)
        sp = nl[nl.null_type == "spatial"].copy()                                         # per-cell 显著: 该 subject 自身空间 null
        sp["delta"] = pd.to_numeric(sp.delta, errors="coerce")
        sp["empirical_p"] = pd.to_numeric(sp.empirical_p, errors="coerce")
        sig = {(str(r.subject), r.band): bool(r.delta > 0 and r.empirical_p < 0.05) for r in sp.itertuples()}
        # 行序：显著 primary band 数降序（并列按 primary maxAB 中位降序），显著多的在上
        n_sig = pd.Series({s: sum(sig.get((str(s), b), False) for b in PRIMARY) for s in piv.index})
        order = pd.DataFrame({"n_sig": n_sig, "med": piv[PRIMARY].median(axis=1)})
        piv = piv.loc[order.sort_values(["n_sig", "med"], ascending=[False, False]).index]
        med = piv.median(axis=0)
        M = np.vstack([piv.values, med.values])
        ylabs = [_short(s) for s in piv.index] + ["cohort median"]
        fig, ax = plt.subplots(figsize=(9.5, 0.46 * len(ylabs) + 2.0))                    # 高度随行数
        im = ax.imshow(M, aspect="auto", cmap="RdBu_r", norm=norm)
        for i, subj in enumerate(piv.index):
            for j, b in enumerate(bands):
                if sig.get((str(subj), b), False):
                    ax.text(j, i, "*", ha="center", va="center", fontsize=14, color="white", fontweight="bold",
                            path_effects=[pe.withStroke(linewidth=1.6, foreground="black")])
        ax.set_xticks(range(len(bands))); ax.set_xticklabels([SHORT[b] for b in bands], fontsize=13)
        ax.set_yticks(range(len(ylabs))); ax.set_yticklabels(ylabs, fontsize=12)
        ax.axhline(len(piv) - 0.5, color="k", lw=1.5)
        ax.axvline(len(PRIMARY) - 0.5, color="k", ls="--", lw=2.5)                        # primary(7) | composite(4)
        ax.set_title(f"F1 · {sub} (n={n}) — observed maxAB |corr| (subject × band)\n"
                     "rows ↓ by # significant primary bands   ·   * = subject self-null p<0.05", fontsize=13)
        for j in range(M.shape[1]):
            if np.isfinite(M[-1, j]):
                ax.text(j, M.shape[0] - 1, f"{M[-1,j]:.2f}", ha="center", va="center", fontsize=9, color="k")
        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cbar.set_label("maxAB |corr|  (blue < 0.5 < red)", fontsize=13)
        cbar.ax.tick_params(labelsize=11)
        fig.tight_layout()
        _save(fig, f"phase1_F1_observed_maxAB_heatmap_{sub}.png")


# ---------------------------------------------------------------- F2 per-band null result
def fig2_null_perband():
    SIG_C, NS_C = "#c44e52", "#cfcfcf"                                        # muted red / light gray (柔和)
    for sub, n in SUBS:                                                       # narrow / broad = 独立图
        _, nl, g = _load(sub)
        sp = nl[(nl.null_type == "spatial") & (nl.band.isin(PRIMARY))].copy()
        sp["delta"] = pd.to_numeric(sp.delta, errors="coerce")
        gg = g.set_index("band")
        fig, ax = plt.subplots(figsize=(9.5, 6.8))
        stars = []                                                                        # (xi, sig); drawn after ylim headroom so they stay in the box
        for xi, b in enumerate(PRIMARY):
            d = sp[sp.band == b].delta.dropna().values
            sig = float(gg.loc[b, "max_over_bands_p"]) < 0.05
            col = SIG_C if sig else NS_C
            if len(d) >= 2:
                vp = ax.violinplot([d], positions=[xi], widths=0.85, showmedians=False, showextrema=False)
                vp["bodies"][0].set_facecolor(col); vp["bodies"][0].set_edgecolor("gray")
                vp["bodies"][0].set_alpha(0.40)
            jit = np.random.default_rng(xi).uniform(-0.085, 0.085, len(d))                # 背景点(per-subject Δ)
            ax.scatter(xi + jit, d, s=24, c="#333333", alpha=0.75, edgecolors="none", zorder=4)
            cd = float(gg.loc[b, "cohort_perm_delta_spatial"])
            ax.hlines(cd, xi - 0.33, xi + 0.33, color="k", lw=2.8, zorder=6)              # cohort Δ (tested)
            stars.append((xi, sig))
        ax.axhline(0, color="gray", lw=0.7, zorder=1)                                     # 0 线减细
        y0, y1 = ax.get_ylim()
        rng = y1 - y0
        ax.set_ylim(y0, y1 + 0.12 * rng)                                                  # 比例 headroom: star row 不贴顶/不超框
        for xi, sig in stars:
            ax.annotate("*" if sig else "n.s.", (xi, y1 + 0.035 * rng), ha="center", va="bottom",
                        fontsize=21 if sig else 12, color=SIG_C if sig else "gray",
                        weight="bold" if sig else "normal", annotation_clip=True)
        ax.set_xticks(range(len(PRIMARY)))
        ax.set_xticklabels([SHORT[b] for b in PRIMARY], fontsize=15)
        ax.tick_params(axis="y", labelsize=14)
        nsig = int((pd.to_numeric(g[g.in_primary_family == True].max_over_bands_p, errors="coerce") < 0.05).sum())
        ax.set_title(f"F2 · {sub} (n={n})  ·  {nsig}/7 pass FWER", fontsize=15)
        ax.grid(alpha=0.25, axis="y")
        ax.spines[["top", "right"]].set_visible(False)                                     # 去右上框
        ax.set_ylabel("cohort alignment − spatial-null median   (Δ per subject)", fontsize=15)
        handles = [Patch(facecolor=SIG_C, alpha=0.4, edgecolor="gray", label="band passes FWER"),
                   Patch(facecolor=NS_C, alpha=0.4, edgecolor="gray", label="n.s. band"),
                   Line2D([0], [0], color="k", lw=2.8, label="cohort Δ (tested)"),
                   Line2D([0], [0], marker="o", ls="none", color="#333333", ms=7, label="per-subject Δ")]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.005, 1.0),
                  fontsize=11, framealpha=0.92)                                            # legend 右上（框外，不遮 star 行）
        fig.tight_layout()
        _save(fig, f"phase1_F2_null_per_band_{sub}.png")


# ---------------------------------------------------------------- F3 per-subject stability
def fig3_subject_stability():
    fig, axes = plt.subplots(1, 2, figsize=(13, 8), sharex=True)
    for ax, (sub, n) in zip(axes, SUBS):
        _, nl, _ = _load(sub)
        sp = nl[(nl.null_type == "spatial") & (nl.band.isin(PRIMARY))].copy()
        for c in ("delta", "empirical_p"):
            sp[c] = pd.to_numeric(sp[c], errors="coerce")
        rows = []
        for subj, gb in sp.groupby("subject"):
            n_sig = int(((gb.delta > 0) & (gb.empirical_p < 0.05)).sum())
            rows.append((subj, n_sig, gb.spatial_null_strength.iloc[0]))
        df = pd.DataFrame(rows, columns=["subject", "n_sig", "strength"]).sort_values("n_sig")
        y = range(len(df))
        ax.barh(list(y), df.n_sig, color=[STRENGTH_C.get(s, "#c6dbef") for s in df.strength],
                edgecolor="k", linewidth=0.5)
        ax.set_yticks(list(y)); ax.set_yticklabels([_short(s) for s in df.subject], fontsize=7)
        med = int(df.n_sig.median())
        ax.axvline(med, color="crimson", ls="--", lw=1.5)
        ax.axvline(4, color="gray", ls=":", lw=1.2)
        ax.annotate(f"median {med}/7", (med, len(df) - 0.5), color="crimson", fontsize=8, ha="left")
        ge5 = int((df.n_sig >= 5).sum()); ge4 = int((df.n_sig >= 4).sum())
        ax.set_title(f"{sub}  (n={n})  ·  ≥5/7: {ge5}  ≥4/7: {ge4}", fontsize=11)
        ax.set_xlim(0, 7); ax.grid(alpha=0.3, axis="x")
    axes[0].set_xlabel("# primary bands significant per subject  (Δ>0 & self-null p<0.05, of 7)", fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=STRENGTH_C[k], ec="k") for k in
               ("within_shaft_strong", "distance_bin_fallback", "subject_wide_weak")]
    fig.legend(handles, ["within-shaft strong", "distance-bin", "subject-wide weak (weak null)"],
               loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("F3 · Per-subject stability: cohort 6/7 is an AGGREGATE, not per-subject robust\n"
                 "median subject ≈ 2/7 (narrow); multi-band-positive subjects are mostly NOT within-shaft-strong",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    _save(fig, "phase1_F3_per_subject_stability.png")


# ------------------------------------------------ F3 between-subject consistency (one-sided Wilcoxon)
def fig3_between_subject_consistency():
    """Per-band between-subject consistency: fraction of subjects with Δ>0 + one-sided Wilcoxon
    signed-rank (H1: Δ>0). Complements F2 — F2 asks "does the cohort beat the weak SPATIAL null"
    (permutation, per-subject spatial shuffle); THIS asks "is the per-subject Δ robustly positive
    ACROSS subjects" (between-subject). The gap quantifies the subject-heterogeneity: the permutation
    aggregates within-subject significance (6/7), but between subjects only ~65-80% of subjects are
    positive and a standard test is marginal → per-patient bias, not a strong cohort-general effect."""
    HOLM_C, RAW_C, NS_C = "#c44e52", "#dd8452", "#cfcfcf"                     # Holm-sig / raw-only / n.s.
    for sub, n in SUBS:                                                       # narrow / broad = 独立图
        _, nl, _ = _load(sub)
        sp = nl[(nl.null_type == "spatial") & (nl.band.isin(PRIMARY))].copy()
        sp["delta"] = pd.to_numeric(sp.delta, errors="coerce")
        frac, npos, ntot, wraw = [], [], [], []
        for b in PRIMARY:
            d = sp[sp.band == b]["delta"].dropna().values
            dz = d[d != 0]                                                    # wilcoxon drops exact zeros
            p = float(wilcoxon(dz, alternative="greater").pvalue) if len(dz) else float("nan")
            frac.append(float((d > 0).mean())); npos.append(int((d > 0).sum()))
            ntot.append(len(d)); wraw.append(p)
        wholm = _holm(wraw)
        fig, ax = plt.subplots(figsize=(9.5, 6.2))
        x = np.arange(len(PRIMARY))
        for xi, b in enumerate(PRIMARY):
            col = HOLM_C if wholm[xi] < 0.05 else (RAW_C if wraw[xi] < 0.05 else NS_C)
            ax.bar(xi, frac[xi], width=0.7, color=col, edgecolor="gray", alpha=0.9, zorder=2)
            ax.text(xi, frac[xi] + 0.015, f"p={wraw[xi]:.3f}" + ("★" if wholm[xi] < 0.05 else ""),
                    ha="center", va="bottom", fontsize=11, fontweight="bold" if wraw[xi] < 0.05 else "normal")
            ax.text(xi, 0.03, f"{npos[xi]}/{ntot[xi]}", ha="center", va="bottom", fontsize=11,
                    color="white" if frac[xi] > 0.18 else "black", zorder=3)
        ax.axhline(0.5, ls="--", color="#888888", lw=1.3, zorder=1)
        ax.text(len(PRIMARY) - 0.55, 0.5, " 50% (Δ symmetric)", va="bottom", ha="right", fontsize=10, color="#777777")
        ax.set_ylim(0, 1.1)
        ax.set_xticks(x); ax.set_xticklabels([SHORT[b] for b in PRIMARY], fontsize=15)
        ax.tick_params(axis="y", labelsize=13)
        ax.set_ylabel("fraction of subjects with Δ > 0\n(between-subject, per band)", fontsize=14)
        nr = int((np.asarray(wraw) < 0.05).sum()); nh = int((wholm < 0.05).sum())
        ax.set_title(f"F3 · {sub} (n={n}) — between-subject consistency of the per-band alignment\n"
                     f"one-sided Wilcoxon signed-rank (Δ>0): {nr}/7 raw p<0.05, {nh}/7 Holm  ·  "
                     "complements F2's spatial-null permutation (per-subject weak / heterogeneous)", fontsize=11.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.2, axis="y")
        handles = [Patch(fc=HOLM_C, alpha=0.9, ec="gray", label="Wilcoxon Holm p<0.05"),
                   Patch(fc=RAW_C, alpha=0.9, ec="gray", label="raw p<0.05 only (Holm n.s.)"),
                   Patch(fc=NS_C, alpha=0.9, ec="gray", label="n.s.")]
        ax.legend(handles=handles, loc="upper right", fontsize=11, framealpha=0.92)
        fig.tight_layout()
        _save(fig, f"phase1_F3_between_subject_consistency_{sub}.png")


def _save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / name
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig1_observed()
    fig2_null_perband()
    fig3_subject_stability()
    fig3_between_subject_consistency()
