#!/usr/bin/env python3
"""D_AB gradient axis vs frozen source->sink ("swap") axis: per-subject maxAB field concordance.

Same pool (narrow), same statistic (maxAB), same activation (broadband), same B=1000 — ONLY the
axis differs. Answers: does the seizure-early field similarity get MORE or FEWER significant
subjects on the D_AB axis? Stratified by multi-shaft vs single-shaft (effective_rank) because the
D_AB cross-shaft axis is only well-conditioned for multi-shaft subjects.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
OLD = REPO / "results/topic5_ictal_recruitment/axis_alignment/axis_alignment_broadband_max_ab_B1000.json"
NEW_GLOB = str(REPO / "results/topic5_ictal_recruitment/axis_alignment/dab_axis/maxab_grp*.json")
DABREC = REPO / "results/spatial_modulation/propagation_geometry/observation_readout/real_subjects_dab"
OUT = REPO / "results/topic5_ictal_recruitment/axis_alignment/figures/field_concordance/dab_vs_sourcesink_maxab.png"


def _per_subject(path):
    return {r["subject_id"]: r for r in json.loads(Path(path).read_text()).get("per_subject", [])
            if r.get("status") == "ok"}


def _merge_new():
    out = {}
    for f in sorted(glob.glob(NEW_GLOB)):
        out.update(_per_subject(f))
    return out


def _nshafts(sid):
    f = DABREC / f"{sid}_t_a.json"
    if not f.exists():
        return None
    q = (json.loads(f.read_text()).get("axis_qc") or {})
    return q.get("n_shafts"), q.get("effective_rank")


def main():
    old = _per_subject(OLD)
    new = _merge_new()
    if not new:
        print("NEW maxAB group JSONs not found/empty — has Stage 2 finished?")
        sys.exit(1)
    shared = sorted(set(old) & set(new))
    rows = []
    for sid in shared:
        o, n = old[sid], new[sid]
        ns, er = _nshafts(sid)
        rows.append(dict(
            sid=sid, old_r=o["real_median_abs_corr"], new_r=n["real_median_abs_corr"],
            old_pass=bool(o.get("pass_channel_null")), new_pass=bool(n.get("pass_channel_null")),
            n_shafts=ns, effective_rank=er, multishaft=(ns is not None and ns >= 2)))

    n_old = sum(r["old_pass"] for r in rows)
    n_new = sum(r["new_pass"] for r in rows)
    gained = [r for r in rows if r["new_pass"] and not r["old_pass"]]
    lost = [r for r in rows if r["old_pass"] and not r["new_pass"]]
    dr = np.array([r["new_r"] - r["old_r"] for r in rows])

    print(f"=== D_AB axis vs source->sink axis, maxAB field concordance (n={len(rows)}) ===")
    print(f"significant (pass channel-null):  source->sink {n_old}/{len(rows)}  ->  D_AB {n_new}/{len(rows)}"
          f"   (Δ = {n_new - n_old:+d})")
    print(f"gained significance: {len(gained)}  {[r['sid'] for r in gained]}")
    print(f"lost significance:   {len(lost)}  {[r['sid'] for r in lost]}")
    print(f"per-subject |r| shift (D_AB - src/sink): median {np.median(dr):+.3f}, "
          f"{int((dr > 0).sum())}/{len(dr)} increased")
    ms = [r for r in rows if r["multishaft"]]
    ss = [r for r in rows if not r["multishaft"]]
    print(f"-- multi-shaft (n={len(ms)}): sig {sum(r['old_pass'] for r in ms)} -> "
          f"{sum(r['new_pass'] for r in ms)}; median |r| shift {np.median([r['new_r']-r['old_r'] for r in ms]):+.3f}")
    print(f"-- single-shaft (n={len(ss)}): sig {sum(r['old_pass'] for r in ss)} -> "
          f"{sum(r['new_pass'] for r in ss)}; median |r| shift {np.median([r['new_r']-r['old_r'] for r in ss]):+.3f}")
    print("\nper-subject:")
    print(f"{'subject':16s} {'shafts':6s} {'src->sink |r|':13s} {'D_AB |r|':9s} {'src pass':8s} {'D_AB pass':9s}")
    for r in sorted(rows, key=lambda x: -(x["new_r"] - x["old_r"])):
        print(f"{r['sid']:16s} {str(r['n_shafts']):6s} {r['old_r']:.3f}{'':8s} {r['new_r']:.3f}{'':4s} "
              f"{str(r['old_pass']):8s} {str(r['new_pass'])}")

    # paired figure
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    for r in rows:
        col = "#2c7fb8" if r["multishaft"] else "0.6"
        ax.plot([0, 1], [r["old_r"], r["new_r"]], "-", color=col, alpha=0.5, lw=1, zorder=1)
        for xx, pr, rr in ((0, r["old_pass"], r["old_r"]), (1, r["new_pass"], r["new_r"])):
            ax.scatter(xx, rr, s=70, facecolor=(col if pr else "white"), edgecolor=col,
                       linewidths=1.6, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"source→sink axis\n({n_old}/{len(rows)} sig)", f"D_AB axis\n({n_new}/{len(rows)} sig)"])
    ax.set_ylabel("maxAB field concordance |r|  (seizure-early vs interictal)")
    ax.set_title("Per-subject seizure-early field similarity: source→sink vs D_AB axis\n"
                 f"filled = passes channel-null; blue = multi-shaft, grey = single-shaft (n={len(rows)})",
                 fontsize=10.5)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(handles=[Line2D([0], [0], color="#2c7fb8", marker="o", ls="-", label="multi-shaft"),
                       Line2D([0], [0], color="0.6", marker="o", ls="-", label="single-shaft"),
                       Line2D([0], [0], color="k", marker="o", ls="none", markerfacecolor="k", label="passes null"),
                       Line2D([0], [0], color="k", marker="o", ls="none", markerfacecolor="white", label="fails null")],
              fontsize=8, loc="lower left")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
