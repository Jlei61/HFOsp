#!/usr/bin/env python
"""Stage-1 paper-ready staging figures for the Figure 3 ictal R3 grid rebuild.

Reads the parallel calculation root written by
``scripts/run_topic5_figure3_ictal_grid_rebuild.py`` and renders, without
overwriting any existing paper-ready figure, into
``results/paper-ready-figure/fig3_ictal_field_concordance_grid_rebuild/``:

    figures/field_concordance_cohort_stat.{png,pdf}(+metadata)
    figures/multiband_field_concordance_stat.{png,pdf}(+metadata)
    figures/r2_vs_r3_sensitivity.{png,pdf}
    figures/multiband_within_shaft_sensitivity.{png,pdf}
    figures/README.md

It also derives the paired R2–R3 diagnostic (§K) into the calculation root.
Stars correspond only to the coherent seven-band pFWER / one-sided Wilcoxon that
the runner computed; real n is always labelled; no old F2 / 184-seizure / mixed
null wording is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon

REPO = Path(__file__).resolve().parents[2]
import sys
for p in (REPO, REPO / "scripts", REPO / "scripts" / "paper_figures"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from plot_fig3_field_concordance_cohort_stat import plot_paired_data_null_groups
from plot_topic5_v2_phase1_figures import plot_null_per_band_figure

CALC = REPO / "results/topic5_ictal_recruitment/field_concordance_grid_parent_matched"
STAGE = REPO / "results/paper-ready-figure/fig3_ictal_field_concordance_grid_rebuild"
FIGDIR = STAGE / "figures"

BAND_ORDER = ["delta_HYP_slow", "theta_preictal_PAC", "alpha_sharp_leq13",
              "beta_LVFA_low", "gamma_LVFA", "hg_low_ripple", "ripple_high"]
BAND_LABELS = {"delta_HYP_slow": "δ\n1–4", "theta_preictal_PAC": "θ\n4–8",
               "alpha_sharp_leq13": "α\n8–13", "beta_LVFA_low": "β\n13–30",
               "gamma_LVFA": "γ\n30–80", "hg_low_ripple": "R\n80–150",
               "ripple_high": "FR\n150–250"}
GROUP_LABELS = {"all_phenotype_matched": "Pooled\nphenotype-matched",
                "strict_broadband": "Broadband\nBB 1–150",
                "gamma_nonbroadband": "Gamma\n30–80"}
ROUTE_COLOR = {"shared": "#B2182B", "own_fallback": "#2166AC"}


def prim_method():
    manifest = json.loads((CALC / "contract_manifest.json").read_text())
    return f"R3_{manifest['grids'][0]}"


def fig_cohort(subj, cohort, method):
    # Primary cohort test is the coherent cohort spatial-null PERMUTATION p, not
    # the subject-vs-own-null Wilcoxon (which is only a sidecar).
    has_perm = "coherent_cohort_spatial_null_p" in cohort.columns
    groups = []
    for gid, label in GROUP_LABELS.items():
        rows = subj[(subj.group_id == gid) & (subj.method == method)]
        c = cohort[(cohort.group_id == gid) & (cohort.method == method)].iloc[0]
        n = int(c.n_subjects)
        nsz = int(c.n_seizures)
        perm_p = float(c.coherent_cohort_spatial_null_p) if has_perm else float("nan")
        groups.append({
            "label": label, "x_label": f"{label}\nn={n} · {nsz} sz",
            "rows": [{"data": float(r.data), "null": float(r.null_median)} for r in rows.itertuples()],
            "summary": {"n": n, "wilcoxon_p_data_gt_null_median": float(c.wilcoxon_one_sided_data_gt_null_p)},
            "display_p": perm_p if has_perm else float(c.wilcoxon_one_sided_data_gt_null_p),
            "p_label": "cohort permutation p" if has_perm else "Wilcoxon p"})
    plot_paired_data_null_groups(
        groups, FIGDIR / "field_concordance_cohort_stat.png",
        FIGDIR / "field_concordance_cohort_stat.pdf",
        ylabel="R3 grid-field concordance |r|", xaxis_mode="group",
        figsize=(8.0, 4.5),
        annotation="early-ictal onset 0–10 s\nstars = coherent cohort\nspatial-null permutation p",
        annotation_xy=(0.30, 0.14))
    def _c(g):
        return cohort[(cohort.group_id == g) & (cohort.method == method)].iloc[0]
    meta = {"figure": "field_concordance_cohort_stat",
            "estimand": "R3 dense-grid support-gated maxAB field concordance",
            "primary_method": method, "window": "clinical onset [0,10] s",
            "null": "coherent all-contact channel shuffle (1000)",
            "primary_cohort_test": "coherent_cohort_spatial_null_p (permutation); stars use this",
            "sidecar_tests": "one-sided Wilcoxon (subject vs own null) + two-sided subject sign-flip",
            "groups": [{"group_id": g, "n_subject": int(_c(g).n_subjects),
                        "n_seizures": int(_c(g).n_seizures),
                        "data_median": float(_c(g).data_median), "null_median": float(_c(g).null_median),
                        "margin_median": float(_c(g).margin_median),
                        "coherent_cohort_spatial_null_p": (float(_c(g).coherent_cohort_spatial_null_p)
                                                           if "coherent_cohort_spatial_null_p" in cohort.columns else None),
                        "wilcoxon_one_sided_p_sidecar": float(_c(g).wilcoxon_one_sided_data_gt_null_p)}
                       for g in GROUP_LABELS],
            "claim_boundary": "coarse patient-specific field/scaffold concordance above an all-contact label-shuffle null; not per-contact replay, direction, causal, or ripple-specific."}
    (FIGDIR / "field_concordance_cohort_stat_metadata.json").write_text(json.dumps(meta, indent=2))


def fig_multiband(mb_subj, mb_cohort, *, panel_label="", significance_mode="fwer"):
    """Render the main-figure seven-band gradient-R3 FWER panel.

    The statistical contract is unchanged from the calculation artifact: each
    dot is one subject-level observed-minus-own-null margin, the black bar is
    the cohort median, and colour/star status comes only from the coherent
    seven-band maxT pFWER.  This compact renderer deliberately omits a title
    and exact p-value prose so the panel can be read at journal column scale.
    """
    subject_deltas = {b: mb_subj[mb_subj.band == b].delta.dropna().tolist() for b in BAND_ORDER}
    cohort_medians = {r.band: float(r.delta_cohort_median) for r in mb_cohort.itertuples()}
    if significance_mode == "fwer":
        pvalue_column = "seven_band_maxt_pfwer"
        output_stem = "multiband_field_concordance_stat"
    elif significance_mode == "raw-coherent":
        pvalue_column = "coherent_cohort_spatial_null_p"
        output_stem = "multiband_field_concordance_raw_p95_sensitivity"
    else:
        raise ValueError(f"unknown significance_mode: {significance_mode}")
    pvalues = {r.band: float(getattr(r, pvalue_column)) for r in mb_cohort.itertuples()}
    n = int(mb_cohort.n_subjects.max())
    npass = int((mb_cohort[pvalue_column] < 0.05).sum())

    sig_color = "#B64F4F"
    ns_color = "#D7D7D7"
    point_color = "#333333"
    rng = np.random.default_rng(17)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    all_values = []
    for xi, band in enumerate(BAND_ORDER):
        values = np.asarray(subject_deltas[band], dtype=float)
        values = values[np.isfinite(values)]
        all_values.extend(values.tolist())
        significant = bool(np.isfinite(pvalues[band]) and pvalues[band] < 0.05)
        color = sig_color if significant else ns_color

        if len(values) >= 2 and np.nanmax(values) > np.nanmin(values):
            violin = ax.violinplot(
                [values], positions=[xi], widths=0.76,
                showmedians=False, showextrema=False,
            )["bodies"][0]
            violin.set_facecolor(color)
            violin.set_edgecolor(sig_color if significant else "#9B9B9B")
            violin.set_linewidth(1.0)
            violin.set_alpha(0.52 if significant else 0.42)

        jitter = rng.uniform(-0.075, 0.075, len(values))
        ax.scatter(
            xi + jitter, values, s=22, color=point_color, alpha=0.78,
            edgecolors="white", linewidths=0.35, zorder=4,
        )
        ax.hlines(
            cohort_medians[band], xi - 0.28, xi + 0.28,
            color="black", linewidth=2.6, zorder=5,
        )

    data_min = float(np.nanmin(all_values))
    data_max = float(np.nanmax(all_values))
    y_lo = min(-0.62, data_min - 0.04)
    y_hi = max(0.48, data_max + 0.13)
    star_y = data_max + 0.075
    for xi, band in enumerate(BAND_ORDER):
        if pvalues[band] < 0.05:
            ax.text(
                xi, star_y, "*", ha="center", va="center",
                color=sig_color, fontsize=22, fontweight="bold", zorder=7,
            )

    ax.axhline(0, color="#777777", linewidth=0.9, zorder=1)
    ax.set_xlim(-0.62, len(BAND_ORDER) - 0.38)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xticks(range(len(BAND_ORDER)))
    ax.set_xticklabels([BAND_LABELS[b] for b in BAND_ORDER])
    ax.set_ylabel("Field concordance − null (Δ)", fontsize=13.5, labelpad=7)
    ax.tick_params(axis="x", labelsize=13, width=1.1, length=4, pad=5)
    ax.tick_params(axis="y", labelsize=12.5, width=1.1, length=4)
    ax.grid(axis="y", color="#E7E7E7", linewidth=0.65, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_linewidth(1.1)

    if panel_label:
        ax.text(
            0.985, 0.985, panel_label, transform=ax.transAxes,
            ha="right", va="top", fontsize=15, fontweight="bold",
            color="#111111",
        )

    fig.tight_layout(pad=0.45)
    out_png = FIGDIR / f"{output_stem}.png"
    fig.savefig(out_png, dpi=400, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)

    significant_bands = [b for b in BAND_ORDER if pvalues[b] < 0.05]
    meta = {"figure": output_stem,
            "axis": "gradient R3 dense grid",
            "window": "clinical onset [0,10] s",
            "n_subjects": n,
            "significance_mode": significance_mode,
            "pvalue_column_used_for_stars": pvalue_column,
            "n_significant": npass,
            "n_significant_fwer": (npass if significance_mode == "fwer" else None),
            "significant_bands": significant_bands,
            "panel_label": panel_label,
            "stars": ("seven-band coherent maxT pFWER (< 0.05)" if significance_mode == "fwer"
                      else "coherent cohort spatial-null P < 0.05, uncorrected across seven bands"),
            "per_band": [{"band": r.band, "delta_cohort_median": float(r.delta_cohort_median),
                          "n_positive": int(r.n_positive), "wilcoxon_one_sided_p": float(r.wilcoxon_one_sided_p),
                          "coherent_cohort_spatial_null_p": float(r.coherent_cohort_spatial_null_p),
                          "seven_band_maxt_pfwer": float(r.seven_band_maxt_pfwer)} for r in mb_cohort.itertuples()],
            "claim_boundary": "band inheritance shown per band vs its own all-contact null; a star on one band and not another is NOT evidence that the two bands differ (see direct band omnibus)."}
    (FIGDIR / f"{output_stem}_metadata.json").write_text(json.dumps(meta, indent=2))
    return npass


def fig_r2_r3(subj, method):
    r3 = subj[(subj.group_id == "all_phenotype_matched") & (subj.method == method)][["subject", "data", "null_median", "margin"]]
    r2 = subj[(subj.group_id == "all_phenotype_matched") & (subj.method == "R2")][["subject", "data", "null_median", "margin"]]
    m = r3.merge(r2, on="subject", suffixes=("_r3", "_r2"))
    route = pd.read_csv(CALC / "field_routing_sigma_grid_inventory.csv").set_index("subject").route.to_dict()
    fig, ax = plt.subplots(figsize=(4.9, 4.7))
    lo = float(min(m.data_r2.min(), m.data_r3.min())) - 0.03
    hi = float(max(m.data_r2.max(), m.data_r3.max())) + 0.03
    ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.9, ls="--", zorder=1)
    for r in m.itertuples():
        ax.scatter(r.data_r2, r.data_r3, s=42, color=ROUTE_COLOR.get(route.get(r.subject), "#555"),
                   edgecolor="white", linewidth=0.6, zorder=3)
    diff = (m.data_r3 - m.data_r2).values
    rho = spearmanr(m.data_r2, m.data_r3).correlation
    try:
        pw = wilcoxon(m.data_r3, m.data_r2, alternative="two-sided").pvalue
    except ValueError:
        pw = float("nan")
    conc = float(np.mean(np.sign(m.margin_r3.values) == np.sign(m.margin_r2.values)))
    ax.set_xlabel("R2 contact-evaluated concordance |r|")
    ax.set_ylabel("R3 dense-grid concordance |r|")
    ax.set_title("Paired R2 vs R3 (pooled, per subject)", fontsize=11)
    ax.text(0.03, 0.97, f"median R3−R2 = {np.median(diff):+.3f}\nSpearman ρ = {rho:.2f}\n"
                        f"paired Wilcoxon p = {pw:.3f}\nmargin sign concordance = {conc:.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.6,
            bbox=dict(boxstyle="round", fc="white", ec="0.8"))
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="none", color=ROUTE_COLOR["shared"], label="shared route"),
                       Line2D([0], [0], marker="o", ls="none", color=ROUTE_COLOR["own_fallback"], label="own fallback")],
              loc="lower right", frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    fig.tight_layout()
    fig.savefig(FIGDIR / "r2_vs_r3_sensitivity.png", dpi=300)
    fig.savefig(FIGDIR / "r2_vs_r3_sensitivity.pdf")
    plt.close(fig)
    # §K paired diagnostic into the calculation root
    diag = {"median_r3_minus_r2_pooled": float(np.median(diff)), "spearman_rho": float(rho),
            "paired_two_sided_wilcoxon_p": float(pw), "margin_sign_concordance": conc,
            "note": "R2 rerun on identical inputs in the same pass; no equivalence claim."}
    (CALC / "r2_r3_cohort_comparison.json").write_text(json.dumps(diag, indent=2))
    return diag


def fig_within_shaft(ws_subj, ws_cohort, ws_cohort_summary):
    subject_deltas = {b: ws_subj[ws_subj.band == b].delta.dropna().tolist() for b in BAND_ORDER}
    cohort_medians = {r.band: float(r.delta_cohort_median) for r in ws_cohort.itertuples()}
    pvalues = {r.band: float(r.wilcoxon_one_sided_p) for r in ws_cohort.itertuples()}
    sizes = {r.band: int(r.n_subjects) for r in ws_cohort.itertuples()}
    n_sub = int(ws_cohort_summary["eligible_subjects"])
    n_ev = int(ws_cohort_summary["eligible_events"])
    plot_null_per_band_figure(
        BAND_ORDER, BAND_LABELS, subject_deltas, cohort_medians, pvalues, sizes,
        f"Pure within-shaft sensitivity (min group 4, no fallback) · eligible n={n_sub} subjects / {n_ev} events",
        FIGDIR / "multiband_within_shaft_sensitivity.png",
        ylabel="R3 grid-field concordance − within-shaft null median\n(subject-level Δ)",
        save_pdf=True, show_exact_annotations=True, figsize=(11.8, 6.6))


def write_readme(pooled, npass, diag, ws_summary):
    txt = f"""# Figure 3 · ictal gradient R3 dense-grid field-concordance (staging)

计算根：`results/topic5_ictal_recruitment/field_concordance_grid_parent_matched/`
母清单：17 名被试 / 167 次发作（`all_phenotype_matched`），主统计量 = R3 dense-grid
support-gated maxAB 场一致性；R2 为同输入配对敏感性。

### field_concordance_cohort_stat
把"发作早期 0–10 s 能量场"与"患者间期 HFO timing 场"的空间一致性，和"打乱触点标签"
的零分布并排画出来。横轴三组：Pooled（17/167，逐发作按表型取 BB150 或 30–80）、
Broadband（16/106，BB 1–150）、Gamma（11/61，30–80）。蓝点=真实，灰点=同一被试的
all-contact 洗牌零假设中位，连线为配对。星号对应单侧配对 Wilcoxon。Pooled 真实中位
{pooled['data_median']:.3f}、零假设 {pooled['null_median']:.3f}、差 {pooled['margin_median']:+.3f}。
**关注点**：看真实点是否系统性高于自己的零假设点，而不是某一组是否单独显著。

### multiband_field_concordance_stat
七个频带各自的 subject-level Δ（真实 − 自己的 all-contact 零假设中位），黑横杠是队列
中位，星号=七带同步 maxT pFWER（<0.05）。{npass}/7 个频带过 FWER。
**关注点**：效应是否铺开在多个频带（band-generic），单个频带有没有星号不能推断"这个
频带比那个频带强"——频带间差异看直接检验（`multiband_band_omnibus.json`）。

### r2_vs_r3_sensitivity
每名被试的 R2（触点评估）与 R3（密网格评估）在同一批输入、同一 mask、同一 sigma、
同一 permutation 下的配对散点，虚线为 y=x。中位 R3−R2={diag['median_r3_minus_r2_pooled']:+.3f}，
Spearman ρ={diag['spearman_rho']:.2f}。
**关注点**：两种评估层是否给出一致的被试排序；不主张"哪一种更真实"，也没有做等价检验。

### multiband_within_shaft_sensitivity
纯 within-shaft 零假设（每根电极杆内洗牌，min group 4，无 fallback）的七带 Δ。合格
denominator 只有 {ws_summary['eligible_subjects']} 名被试 / {ws_summary['eligible_events']} 次发作，
远小于 17/167——这是严格解剖零假设的代价，只作 anatomical sensitivity。
**关注点**：这是二级敏感性，样本很小；若这里没有证据，主张上限仍是 coarse patient-specific
scaffold，不能升级为 within-shaft-specific。
"""
    (FIGDIR / "README.md").write_text(txt)


def main():
    global FIGDIR, CALC
    ap = argparse.ArgumentParser()
    ap.add_argument("--calc", default=str(CALC))
    ap.add_argument("--stage", default=str(STAGE))
    ap.add_argument("--only", choices=("all", "multiband"), default="all",
                    help="Render the full staging package or only the seven-band main panel.")
    ap.add_argument("--panel-label", default="",
                    help="Lower-case panel label placed inside the upper-right corner; use '' to omit.")
    ap.add_argument("--significance-mode", choices=("fwer", "raw-coherent"), default="fwer",
                    help="Formal seven-band maxT-FWER or explicitly uncorrected coherent-null sensitivity.")
    args = ap.parse_args()
    calc = Path(args.calc)
    CALC = calc
    FIGDIR = Path(args.stage) / "figures"
    FIGDIR.mkdir(parents=True, exist_ok=True)
    mb_subj = pd.read_csv(calc / "multiband_subject.csv")
    mb_cohort = pd.read_csv(calc / "multiband_cohort.csv")
    if args.only == "multiband":
        npass = fig_multiband(
            mb_subj, mb_cohort, panel_label=args.panel_label,
            significance_mode=args.significance_mode,
        )
        print(f"[figures] wrote multiband panel -> {FIGDIR} | {npass}/7 {args.significance_mode}")
        return

    method = prim_method()
    subj = pd.read_csv(calc / "parent_anchor_subject.csv")
    cohort = pd.read_csv(calc / "parent_anchor_cohort.csv")
    ws_subj = pd.read_csv(calc / "within_shaft_multiband_subject.csv")
    ws_cohort = pd.read_csv(calc / "within_shaft_multiband_cohort.csv")
    ws_summary = json.loads((calc / "summary.json").read_text())["within_shaft"]

    fig_cohort(subj, cohort, method)
    if args.significance_mode != "fwer":
        raise ValueError("--significance-mode raw-coherent is only supported with --only multiband")
    fig_multiband(mb_subj, mb_cohort, panel_label=args.panel_label, significance_mode="fwer")
    diag = fig_r2_r3(subj, method)
    fig_within_shaft(ws_subj, ws_cohort, ws_summary)
    pooled = cohort[(cohort.group_id == "all_phenotype_matched") & (cohort.method == method)].iloc[0].to_dict()
    npass = int((mb_cohort.seven_band_maxt_pfwer < 0.05).sum())
    write_readme(pooled, npass, diag, ws_summary)
    print(f"[figures] wrote staging -> {FIGDIR}")
    print(f"[figures] pooled data={pooled['data_median']:.4f} null={pooled['null_median']:.4f} "
          f"margin={pooled['margin_median']:+.4f} | multiband {npass}/7 FWER | "
          f"R2-R3 rho={diag['spearman_rho']:.2f}")


if __name__ == "__main__":
    main()
