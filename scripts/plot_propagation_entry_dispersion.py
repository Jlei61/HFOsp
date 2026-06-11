#!/usr/bin/env python3
"""Figures for the propagation entry-dispersion analysis.

Cohort figure (3 panels, one independent question each, CLAUDE.md §7):
  A. Is the entry more spread than ONE noisy propagation template predicts?
     (observed vs single-template-noise-null effective number of entry channels)
  B. Does the leading channel just track a single template's early end?
     (alignment of per-channel leading-frequency with template order)
  C. Is the downstream order the same regardless of which channel led?
     (downstream-order agreement across different entry channels)

Per-subject diagnostic (per cluster): leading-channel frequency ordered by
template position with the single-template-noise-null overlaid (monotonic decay
matching the null = one noisy template), plus a spatial map of where the entry
mass sits along the propagation axis (where coordinates exist).

Paper-grade: plain-English axis labels (no Neff / spearman / cluster_id), shared
legend, tight axes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path("results/spatial_modulation/propagation_geometry/components/entry_variability")
FIG = OUT / "figures"
DS_COLOR = {"epilepsiae": "#3b6ea5", "yuquan": "#c0504d"}
DS_LABEL = {"epilepsiae": "Epilepsiae cohort", "yuquan": "Yuquan cohort"}


def _load():
    summary = json.load(open(OUT / "cohort_summary.json"))
    return summary["summary"], summary["clusters"]


# --------------------------------------------------------------------------
VERDICT_COLOR = {
    "robust_excess": "#2e7d32",      # green: exceeds both nulls
    "fragile_excess": "#f0a202",     # amber: pooled-only (heteroscedasticity)
    "concentrated": "#3b6ea5",       # blue: more concentrated than one template
    "consistent_one_template": "#9e9e9e",  # grey
}
VERDICT_LABEL = {
    "robust_excess": "entry spread exceeds both nulls",
    "fragile_excess": "exceeds liberal null only",
    "concentrated": "more concentrated than one template",
    "consistent_one_template": "consistent with one noisy template",
}


def cohort_figure(rows: List[Dict]):
    an = [r for r in rows if r.get("p_neff_excess_pooled") is not None]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))

    # ---- Panel A: FIRM result -- ignition is NOT a single fixed point ----
    ax = axes[0]
    ts = [r["top_share"] for r in an if r.get("top_share") is not None]
    ax.hist(ts, bins=np.linspace(0, 1, 21), color="#6a6a6a", edgecolor="white")
    ax.axvline(1.0, color="k", ls=":", lw=1.4)
    ax.axvline(np.median(ts), color="#c0504d", lw=2, label=f"median = {np.median(ts):.2f}")
    ax.annotate("a single fixed\nignition point", (1.0, ax.get_ylim()[1]*0.9),
                ha="right", va="top", fontsize=8.5, color="k")
    ax.set_xlabel("fraction of events led by the single most frequent entry channel")
    ax.set_ylabel("number of templates")
    ax.set_title("A. Is ignition a single fixed point? (no)", fontsize=10.5)
    ax.set_xlim(0, 1.02)
    ax.legend(loc="upper right", fontsize=8.5)

    # ---- Panel B: excess BRACKET -- observed vs conservative null ----
    ax = axes[1]
    for v, col in VERDICT_COLOR.items():
        sub = [r for r in an if r["verdict"] == v and r.get("null_neff_gauss_mean")]
        if sub:
            ax.scatter([r["null_neff_gauss_mean"] for r in sub], [r["neff"] for r in sub],
                       s=34, alpha=0.85, color=col, edgecolor="white", linewidth=0.5,
                       label=f"{VERDICT_LABEL[v]} (n={sum(1 for r in an if r['verdict']==v)})")
    allv = [r["neff"] for r in an] + [r["null_neff_gauss_mean"] for r in an if r.get("null_neff_gauss_mean")]
    lim = [0.9, max(allv) * 1.15]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("entry spread from one noisy template, per-channel noise\n(conservative null; effective # of entry channels, log)")
    ax.set_ylabel("observed entry spread\n(effective # of entry channels)")
    ax.set_title("B. Entry spread vs one noisy template (bracketed verdict)", fontsize=10.5)
    ax.legend(loc="upper left", fontsize=7.2, framealpha=0.9)

    # ---- Panel C: CONFOUND -- does 'excess' track cluster quality / montage? ----
    ax = axes[2]
    for v, col in VERDICT_COLOR.items():
        sub = [r for r in an if r["verdict"] == v and r.get("silhouette_k2") is not None
               and r.get("n_channels") is not None]
        if sub:
            ax.scatter([r["silhouette_k2"] for r in sub], [r["n_channels"] for r in sub],
                       s=34, alpha=0.85, color=col, edgecolor="white", linewidth=0.5,
                       label=VERDICT_LABEL[v])
    ax.set_xlabel("cluster separation quality\n(silhouette; low = loose blend / continuum)")
    ax.set_ylabel("number of recording channels")
    ax.set_title("C. Does 'excess' track cluster quality / montage size?", fontsize=10.5)
    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=7.2, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(FIG / "cohort_entry_dispersion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / "cohort_entry_dispersion.png")


# --------------------------------------------------------------------------
def per_subject_figure(stem: str):
    p = OUT / "per_subject" / f"{stem}.json"
    if not p.exists():
        print("skip (no file):", stem); return
    d = json.load(open(p))
    clusters = [c for c in d["clusters"] if "null" in c and c["null"].get("obs_earliest_prob")]
    if not clusters:
        print("skip (no analyzable cluster):", stem); return
    chn = d["channel_names"]
    coords = np.array(d["coords_mm"]) if d.get("coords_mm") else None
    mapped = np.array(d["mapped"], bool) if d.get("mapped") else None
    has_space = coords is not None and mapped is not None and mapped.sum() >= 2

    ncol = 2 if has_space else 1
    fig, axes = plt.subplots(len(clusters), ncol,
                             figsize=(5.4 * ncol, 3.1 * len(clusters)),
                             squeeze=False)
    for ri, c in enumerate(clusters):
        tr = np.array(c["template_rank"], float)
        order = np.argsort(tr)                      # channels by template position
        nn = c["null"]
        obs = np.array(nn["obs_earliest_prob"])
        nul_pool = np.array(nn["null_earliest_prob"])             # liberal null
        nul_gauss = np.array(nn.get("null_earliest_prob_gauss", nn["null_earliest_prob"]))

        # left: leading-channel frequency by template order, with BOTH null
        # overlays (liberal + conservative) so the line and the bracketed verdict
        # come from the same place.
        ax = axes[ri][0]
        xpos = np.arange(len(order))
        ax.bar(xpos, obs[order], color="#3b6ea5", edgecolor="white",
               label="observed", zorder=2)
        ax.plot(xpos, nul_pool[order], "o-", color="#f0a202", ms=3.5, lw=1.4,
                label="one noisy template — liberal null", zorder=3)
        ax.plot(xpos, nul_gauss[order], "s--", color="#c0504d", ms=3.5, lw=1.4,
                label="one noisy template — conservative null", zorder=4)
        if len(order) <= 20:
            ax.set_xticks(xpos)
            ax.set_xticklabels([chn[i] for i in order], rotation=60, ha="right", fontsize=7)
        else:
            ax.set_xticks([])           # too many channels to label legibly
        ax.set_ylabel("fraction of events\nthis channel leads")
        ax.set_xlabel("channel (ordered: template-early → template-late)")
        # verdict mirrors runner's _verdict (both nulls) -> line & title agree
        pe_p = nn.get("p_neff_excess", 1.0)
        pe_g = nn.get("p_neff_excess_gauss", 1.0)
        pc_p = nn.get("p_neff_concentrated", 1.0)
        if pe_p is not None and pe_p < 0.05 and pe_g is not None and pe_g < 0.05:
            tag = "spread beyond BOTH nulls"
        elif pe_p is not None and pe_p < 0.05:
            tag = "beyond liberal null only (fragile)"
        elif pc_p is not None and pc_p < 0.05:
            tag = "more concentrated than one template"
        else:
            tag = "≈ one noisy template"
        ax.set_title(f"template {ri+1}: ≈{c['neff']:.1f} entry channels — {tag}",
                     fontsize=9.5)
        if ri == 0:
            ax.legend(fontsize=7, loc="upper right")

        # right: spatial map of entry mass along the propagation axis
        if has_space:
            ax = axes[ri][1]
            xyz = coords[mapped]
            # propagation axis = template-rank gradient in space (PCA-free: use
            # template_rank-weighted direction). Project onto 2D via PCA of coords.
            xyz_c = xyz - xyz.mean(0)
            u, s, vt = np.linalg.svd(xyz_c, full_matrices=False)
            proj = xyz_c @ vt[:2].T
            obs_m = obs[mapped]
            tr_m = tr[mapped]
            sc = ax.scatter(proj[:, 0], proj[:, 1], c=tr_m, cmap="viridis",
                            s=40 + 900 * obs_m, edgecolor="k", linewidth=0.6,
                            zorder=2)
            for k, ci in enumerate(np.where(mapped)[0]):
                ax.annotate(chn[ci], (proj[k, 0], proj[k, 1]), fontsize=6.5,
                            ha="center", va="center")
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
            cb.set_label("template position\n(0 = earliest)", fontsize=8)
            ax.set_xlabel("electrode space (mm, projected)")
            ax.set_ylabel("electrode space (mm, projected)")
            ax.set_title("marker size = how often that channel leads", fontsize=9.5)
            ax.set_aspect("equal", "datalim")

    sp = d.get("coord_space") or "no coordinates"
    fig.suptitle(f"{DS_LABEL[d['dataset']]} — subject {d['subject']} "
                 f"({d['n_mapped_channels']}/{d['n_channels']} contacts in {sp})",
                 fontsize=11, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    (FIG / "per_subject").mkdir(parents=True, exist_ok=True)
    out = FIG / "per_subject" / f"{stem}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--showcase", type=str, default="",
                    help="comma list of stems for per-subject figs; "
                         "default = auto (one representative per verdict)")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    summary, rows = _load()
    cohort_figure(rows)

    if args.showcase:
        stems = args.showcase.split(",")
    else:
        # auto: one representative per verdict (prefer small-montage, well-separated
        # clusters so the showcase is the cleanest case of each kind).
        an = [r for r in rows if r.get("p_neff_excess_pooled") is not None]
        stems = []
        for v in ("robust_excess", "fragile_excess", "concentrated", "consistent_one_template"):
            cand = sorted([r for r in an if r["verdict"] == v],
                          key=lambda r: (r.get("n_channels") or 99))
            for r in cand:
                s = f"{r['dataset']}_{r['subject']}"
                if s not in stems:
                    stems.append(s); break
    for s in stems:
        per_subject_figure(s)


if __name__ == "__main__":
    main()
