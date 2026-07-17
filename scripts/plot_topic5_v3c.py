"""V3c figures (paper-grade, self-contained): coverage null forest, latency AUC
forest, assay-QC bars, surplus-spatial distance forest. One PNG per question
(CLAUDE.md §7). Reads the cohort CSVs/JSONs; render -> eyeball -> commit.

All three forests use PER-SUBJECT null intervals so the significance colour
(red = observed beyond that subject's own null in the hypothesised direction)
matches the dot's position relative to its interval.

SOZ is an intended metric input here (coverage of / recruitment vs SOZ), so the
Topic-1/3 'SOZ overlay only, not metric input' disclaimer does NOT apply.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_ROOT = Path(__file__).resolve().parents[1]
OUT = _ROOT / "results/topic5_ictal_recruitment/v3c_soz_axis_coverage"

C_OBS = "#1b1b1b"       # observed, not significant
C_NULL = "#b8b8b8"      # per-subject null interval
C_SIG = "#c0392b"       # observed, significant (p<0.05) in the hypothesised direction
C_INVAL = "#bbbbbb"     # assay-invalid (latency)
C_FIN = "#2e7d32"       # finite (green)
C_T0 = "#e08a1e"        # t0 left-censored (orange)
C_CENS = "#9e9e9e"      # right-censored (grey)


def _sid(s):  # "epilepsiae_1146" -> "1146"
    return s.split("_", 1)[1]


def _load_json(p):
    return json.loads(p.read_text()) if p.exists() else {}


def _forest(ax, df, obs_col, q05_col, q95_col, p_col, *, sig_label, ref=None, ref_label=None,
            invalid_mask=None):
    """Shared per-subject forest: null interval bar + observed dot (red if p<0.05)."""
    y = np.arange(len(df))
    for i, r in df.reset_index(drop=True).iterrows():
        inval = bool(invalid_mask.iloc[i]) if invalid_mask is not None else False
        if np.isfinite(r[q05_col]) and np.isfinite(r[q95_col]):
            if abs(r[q95_col] - r[q05_col]) < 1e-9:
                # degenerate within-shaft null (no valid permutation: covered/surplus
                # on disjoint shafts) -> point null, drawn as a short vertical tick
                ax.plot([r[q05_col], r[q05_col]], [i - 0.28, i + 0.28], color=C_NULL, lw=3, zorder=1)
            else:
                ax.plot([r[q05_col], r[q95_col]], [i, i], color=C_NULL, lw=7,
                        solid_capstyle="butt", zorder=1)
        if inval:
            ax.scatter(r[obs_col], i, s=70, color=C_INVAL, marker="x", zorder=3)
            continue
        sig = np.isfinite(r[p_col]) and r[p_col] < 0.05
        ax.scatter(r[obs_col], i, s=70, color=C_SIG if sig else C_OBS, zorder=3)
    if ref is not None:
        ax.axvline(ref, color="#555", ls="--", lw=1, zorder=2)
    ax.set_yticks(y); ax.invert_yaxis(); ax.margins(y=0.08)


def _legend_handles(sig_label, *, invalid=False, ref_label=None):
    h = [Line2D([0], [0], marker="o", color="w", markerfacecolor=C_SIG, markersize=8, label=sig_label),
         Line2D([0], [0], marker="o", color="w", markerfacecolor=C_OBS, markersize=8, label="not significant"),
         Line2D([0], [0], color=C_NULL, lw=7, label="per-subject null 5–95%")]
    if invalid:
        h.append(Line2D([0], [0], marker="x", color="w", markeredgecolor=C_INVAL,
                        markerfacecolor=C_INVAL, markersize=8, label="assay-invalid (excluded)"))
    if ref_label:
        h.append(Line2D([0], [0], color="#555", ls="--", lw=1, label=ref_label))
    return h


def coverage_forest(cohort):
    base = OUT / cohort
    df = pd.read_csv(base / "coverage_subject.csv")
    df = df[df["eligible"] == True].reset_index(drop=True)  # noqa: E712
    ch = _load_json(base / "coverage_cohort.json")
    fig, ax = plt.subplots(figsize=(6.6, 0.55 * len(df) + 1.7))
    _forest(ax, df, "coverage", "null_q05", "null_q95", "coverage_null_p",
            sig_label="covers SOZ beyond geometry (p<0.05)")
    for i, r in df.iterrows():
        if r["n_missed"] > 0:
            ax.annotate(f"{int(r['n_missed'])} SOZ not on axis", (r["coverage"], i),
                        textcoords="offset points", xytext=(-9, 9), ha="right",
                        fontsize=7.5, color=C_SIG)
    ax.set_yticklabels(df["subject"].map(_sid))
    ax.set_xlabel("SOZ coverage by interictal axis   |A∩S| / |S|")
    ax.set_xlim(0, 1.06)
    cp = ch.get("p_value")
    ax.set_title("Interictal axis coverage of clinical SOZ  ·  " + cohort
                 + (f"  (cohort-median null p = {cp:.3f})" if cp is not None else ""))
    ax.legend(handles=_legend_handles("covers SOZ beyond geometry (p<0.05)"),
              loc="lower left", fontsize=8, framealpha=0.9)
    _save(fig, base, "coverage_null_forest.png")


def latency_auc_forest(cohort):
    base = OUT / cohort
    latp = base / "latency" / "latency_subject.csv"
    if not latp.exists():
        return
    df = pd.read_csv(latp)
    df = df[np.isfinite(df["auc_primary"])].reset_index(drop=True)
    ch = _load_json(base / "latency" / "latency_cohort.json")
    sm = _load_json(base / "v3c_summary.json")
    fig, ax = plt.subplots(figsize=(6.8, 0.55 * len(df) + 2.0))
    _forest(ax, df, "auc_primary", "auc_null_q05", "auc_null_q95", "auc_null_p",
            sig_label="surplus later than SOZ core (p<0.05)", ref=0.5,
            invalid_mask=~df["eligible"].astype(bool))
    ax.set_yticklabels([f"{_sid(s)}" + ("" if e else "  (assay-invalid)")
                        for s, e in zip(df["subject"], df["eligible"])])
    ax.set_xlabel("AUC_late   P(surplus recruited later than SOZ core)   ·   >0.5 = downstream")
    ax.set_xlim(0, 1.0)
    obs = ch.get("obs_cohort_median_auc"); interp = sm.get("latency_interpretation", "")
    ttl = "Ictal recruitment of axis-surplus vs SOZ core  ·  " + cohort
    if obs is not None:
        ttl += f"\ncohort AUC = {obs:.3f},  Δt = {ch.get('delta_t_med', float('nan')):+.1f}s  →  {interp}"
    ax.set_title(ttl, fontsize=10)
    ax.legend(handles=_legend_handles("surplus later than SOZ core (p<0.05)", invalid=True,
                                      ref_label="AUC = 0.5 (synchronous)"),
              loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=7.5, framealpha=0.9)
    _save(fig, base, "latency_auc_forest.png")


def qc_bars(cohort):
    base = OUT / cohort
    qcp = base / "latency_qc" / "qc_subject.csv"
    if not qcp.exists():
        return
    df = pd.read_csv(qcp)
    df = df[np.isfinite(df["finite_frac"])].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.9, 0.55 * len(df) + 1.9))
    y = np.arange(len(df))
    fin, t0, cens = df["finite_frac"], df["t0_frac"], df["cens_frac"]
    ax.barh(y, fin, color=C_FIN, label="finite latency")
    ax.barh(y, t0, left=fin, color=C_T0, label="t0 (already hot at onset)")
    ax.barh(y, cens, left=fin + t0, color=C_CENS, label="never crosses (censored)")
    for i, r in df.iterrows():
        tag = "✓ valid" if bool(r["assay_valid"]) else "✗ invalid"
        if bool(r.get("cens_flag", False)):
            tag += " ⚑cens"
        ax.annotate(tag, (1.01, i), va="center", fontsize=8,
                    color=C_FIN if bool(r["assay_valid"]) else C_SIG)
    ax.set_yticks(y); ax.set_yticklabels(df["subject"].map(_sid))
    ax.set_xlabel("fraction of axis contacts")
    ax.set_xlim(0, 1.30); ax.invert_yaxis()
    ax.set_title("Label-blind latency assay-QC on axis contacts  ·  " + cohort
                 + "\nassay valid needs t0 fraction ≤ 0.50 (the orange segment)", fontsize=10)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.margins(y=0.08)
    _save(fig, base, "qc_bars.png")


def surplus_spatial(cohort):
    base = OUT / cohort
    sp = base / "surplus_spatial" / "surplus_subject.csv"
    if not sp.exists():
        return
    df = pd.read_csv(sp)
    df = df[np.isfinite(df["mean_min_dist_to_soz"])].reset_index(drop=True)
    ch = _load_json(base / "surplus_spatial" / "surplus_spatial_cohort.json")
    fig, ax = plt.subplots(figsize=(6.8, 0.55 * len(df) + 1.8))
    _forest(ax, df, "mean_min_dist_to_soz", "dist_null_q05", "dist_null_q95", "dist_null_p",
            sig_label="surplus close to SOZ (p<0.05)")
    ax.set_yticklabels(df["subject"].map(_sid))
    ax.set_xlabel("mean distance from axis-surplus to nearest SOZ (mm)   ·   lower = closer / structured")
    cp = ch.get("p_value")
    ax.set_title("Spatial organization of axis-surplus  ·  " + cohort
                 + (f"  (cohort-median distance null p = {cp:.3f})" if cp is not None else ""),
                 fontsize=10)
    ax.legend(handles=_legend_handles("surplus close to SOZ (p<0.05)"),
              loc="lower right", fontsize=8, framealpha=0.9)
    _save(fig, base, "surplus_spatial.png")


def _save(fig, base, name):
    (base / "figures").mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(base / "figures" / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {base.name}/figures/{name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", choices=["broad", "narrow"], required=True)
    args = ap.parse_args()
    coverage_forest(args.cohort)
    latency_auc_forest(args.cohort)
    qc_bars(args.cohort)
    surplus_spatial(args.cohort)
    print(f"[done] figures for {args.cohort}")


if __name__ == "__main__":
    main()
