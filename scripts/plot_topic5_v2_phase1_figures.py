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
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "results/topic5_ictal_recruitment/v2_band_scan"
FIGDIR = V2 / "figures"
cfg = yaml.safe_load(open(ROOT / "config/topic5_v2_phase1.yaml"))
PRIMARY = [b[0] for b in cfg["bands"]["primary"]]
COMPOSITE = [b[0] for b in cfg["bands"]["composites"]]
SHORT = {"delta_HYP_slow": "δ\n1-4", "theta_preictal_PAC": "θ\n4-8", "alpha_sharp_leq13": "α\n8-13",
         "beta_LVFA_low": "β\n13-30", "gamma_LVFA": "γ\n30-80", "hg_low_ripple": "hgR\n80-150",
         "ripple_high": "R\n150-250", "low_HYP_1_13": "low\n1-13", "LVFA_13_80": "LVFA\n13-80",
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
    # diverging red-blue centered at the pooled cohort median (red = above-typical alignment, blue = below);
    # center is the SAME across both panels so narrow/broad are directly comparable.
    allv = []
    for sub, _ in SUBS:
        a, _, _ = _load(sub)
        allv += pd.to_numeric(a[a.band.isin(PRIMARY)].align_abs_maxab, errors="coerce").dropna().tolist()
    vc = round(float(np.median(allv)), 2)
    norm = TwoSlopeNorm(vmin=0.40, vcenter=vc, vmax=0.95)
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [20, 17]})
    im = None
    for ax, (sub, n) in zip(axes, SUBS):
        a, _, _ = _load(sub)
        piv = a.pivot_table(index="subject", columns="band", values="align_abs_maxab", aggfunc="median")
        piv = piv.reindex(columns=bands)
        piv = piv.loc[piv[PRIMARY].median(axis=1).sort_values(ascending=False).index]   # 高在上
        med = piv.median(axis=0)
        M = np.vstack([piv.values, med.values])                                          # 末行=cohort median
        ylabs = [_short(s) for s in piv.index] + ["cohort median"]
        im = ax.imshow(M, aspect="auto", cmap="RdBu_r", norm=norm)
        ax.set_xticks(range(len(bands))); ax.set_xticklabels([SHORT[b] for b in bands], fontsize=8)
        ax.set_yticks(range(len(ylabs))); ax.set_yticklabels(ylabs, fontsize=7)
        ax.axhline(len(piv) - 0.5, color="k", lw=1.5)                                     # 分隔 cohort median
        ax.axvline(len(PRIMARY) - 0.5, color="k", ls="--", lw=2.5)                        # primary(7) | composite(4)
        ax.set_title(f"{sub}  (n={n})", fontsize=12)
        for j in range(M.shape[1]):                                                       # 数值标注 cohort median 行
            if np.isfinite(M[-1, j]):
                ax.text(j, M.shape[0] - 1, f"{M[-1,j]:.2f}", ha="center", va="center", fontsize=6, color="k")
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label(f"maxAB |corr|   (red = higher / blue = lower; diverging at pooled median {vc})", fontsize=10)
    fig.suptitle("F1 · Observed alignment: early-ictal multi-band energy field <-> interictal HFO-derived geometry\n"
                 "descriptive magnitude (smoothed-field |corr|); narrow > broad; band-generic.  "
                 "dashed line = primary (7) | composite (4) bands", fontsize=11)
    _save(fig, "phase1_F1_observed_maxAB_heatmap.png")


# ---------------------------------------------------------------- F2 per-band null result
def fig2_null_perband():
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    for ax, (sub, n) in zip(axes, SUBS):
        _, nl, g = _load(sub)
        sp = nl[(nl.null_type == "spatial") & (nl.band.isin(PRIMARY))].copy()
        sp["delta"] = pd.to_numeric(sp.delta, errors="coerce")
        gg = g.set_index("band")
        for xi, b in enumerate(PRIMARY):
            d = sp[sp.band == b].delta.dropna().values
            sig = float(gg.loc[b, "max_over_bands_p"]) < 0.05
            col = "crimson" if sig else "lightgray"
            if len(d) >= 2:                                                                # per-subject Δ violin
                vp = ax.violinplot([d], positions=[xi], widths=0.85, showmedians=False, showextrema=True)
                vp["bodies"][0].set_facecolor(col); vp["bodies"][0].set_edgecolor("k")
                vp["bodies"][0].set_alpha(0.55)
                for key in ("cbars", "cmins", "cmaxes"):
                    if vp.get(key) is not None:
                        vp[key].set_color("gray"); vp[key].set_linewidth(1.0)
            else:
                ax.scatter([xi] * len(d), d, s=30, c=col, edgecolors="k", zorder=4)
            cd = float(gg.loc[b, "cohort_perm_delta_spatial"])
            ax.hlines(cd, xi - 0.35, xi + 0.35, color="k", lw=3, zorder=6)                 # cohort Δ (tested stat)
            top = np.nanmax(d) if len(d) else cd
            ax.annotate("*" if sig else "n.s.", (xi, top + 0.03), ha="center",
                        fontsize=17 if sig else 10, color="crimson" if sig else "gray",
                        weight="bold" if sig else "normal")
        ax.axhline(0, color="k", lw=1.0)
        ax.set_xticks(range(len(PRIMARY)))
        ax.set_xticklabels([SHORT[b] for b in PRIMARY], fontsize=12)          # 两行(symbol/freq)避免横向重叠
        ax.tick_params(axis="y", labelsize=13)
        nsig = int((pd.to_numeric(g[g.in_primary_family == True].max_over_bands_p, errors="coerce") < 0.05).sum())
        ax.set_title(f"{sub}  (n={n})  ·  {nsig}/7 primary pass family-wise correction", fontsize=14)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("cohort alignment - spatial-null median   (Δ per subject)", fontsize=15)
    fig.suptitle("F2 · Formal-null: per-subject Δ distribution (violin) vs the weak / subject-wide spatial null\n"
                 "violin = per-subject Δ; black bar = cohort Δ (tested); crimson * = band passes the max-over-bands "
                 "family-wise (FWER) correction.\nripple_high weakest -> NOT ripple-specific.  "
                 "WEAK null -> anti-conservative, likely inflated; formal within-shaft Gate A unresolved (2/20).",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    _save(fig, "phase1_F2_null_per_band.png")


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
