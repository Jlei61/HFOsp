#!/usr/bin/env python3
"""Per-template propagation axis pair: axis_A = gradient(eA), axis_B = gradient(eB).

For every k=2 subject (both datasets) with >=6 coord-mapped joint contacts, build each template's
OWN propagation axis (the single-template least-squares gradient — the building block; D_AB is the
special case axis_A - axis_B) and classify the A/B relationship by the SPATIAL axis angle
cos(u_A, u_B), NOT by the channel-count-sensitive rank correlation rho_AB:
  reversed  cos <= -0.5  -> one bidirectional pathological axis (D_AB-like)
  same      cos >= +0.5  -> redundant modes, one axis
  different |cos| < 0.5  -> two distinct pathological propagation directions
Single-shaft subjects (effective_rank<2) have a trivial cos ~ +-1 (both axes = the shaft) and are
flagged; the cohort verdict is read on the multi-shaft subjects.
"""
import argparse
import csv
import glob
import json
import os
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
sys.path.insert(0, str(REPO))
from src.topic5_template_axis_field import compute_template_axis_pair  # noqa: E402
from src.seeg_coord_loader import load_subject_coords  # noqa: E402
from src.propagation_skeleton_geometry import parse_shaft  # noqa: E402

RANKDISP = REPO / "results/interictal_propagation_masked/rank_displacement/per_subject"
OUTCSV = REPO / "results/topic5_ictal_recruitment/axis_alignment/template_axis_pair_cohort.csv"
OUTFIG = REPO / "results/topic5_ictal_recruitment/axis_alignment/figures/field_concordance/template_axis_pair_cohort.png"


def _classify(cos):
    if cos <= -0.5:
        return "reversed"
    if cos >= 0.5:
        return "same"
    return "different"


def analyze(sid):
    ds, subj = sid.split("_", 1)
    d = json.loads((RANKDISP / f"{sid}.json").read_text())
    if d.get("stable_k") != 2:
        return {"subject": sid, "status": "k!=2", "stable_k": d.get("stable_k")}
    p = d["pairs"][0]
    nm = p["channel_names"]
    jv = np.array(p["joint_valid"], bool)
    ra = np.array(p["rank_a_dense_full"], float)
    rb = np.array(p["rank_b_dense_full"], float)
    jn = [nm[i] for i in np.where(jv)[0]]
    try:
        cr = load_subject_coords(ds, subj, jn)
    except Exception as e:
        return {"subject": sid, "status": f"no_coords:{str(e)[:30]}", "stable_k": 2}
    C = np.asarray(cr.coords_array_in_requested_order, float)
    mp = np.asarray(cr.mapped_mask_in_requested_order, bool)
    if int(mp.sum()) < 6:
        return {"subject": sid, "status": "insufficient_mapped", "stable_k": 2}
    Cm = C[mp]
    raj, rbj = ra[jv][mp], rb[jv][mp]
    sh = [parse_shaft(jn[i])[0] for i in np.where(mp)[0]]
    pair_axes = compute_template_axis_pair(
        Cm, raj, rbj, sh, n_axis_boot=200, n_pair_boot=500, seed=0,
    )
    axA, axB = pair_axes["axis_a"], pair_axes["axis_b"]
    if axA["status"] != "ok" or axB["status"] != "ok":
        return {"subject": sid, "status": "degenerate_axis", "stable_k": 2}
    # Both u vectors are explicitly propagation-positive (early -> late).
    cos = float(np.asarray(axA["u"]) @ np.asarray(axB["u"]))
    rho = float(np.corrcoef(raj, rbj)[0, 1])
    er = min(axA["effective_rank"], axB["effective_rank"])
    multishaft = bool(axA["n_shafts"] >= 2 and er >= 2)
    return {"subject": sid, "dataset": ds, "status": "ok", "stable_k": 2,
            "n_joint": len(Cm), "n_shafts": axA["n_shafts"],
            "within_shaft_frac": round(min(axA["within_shaft_frac"], axB["within_shaft_frac"]), 3),
            "effective_rank": er, "rho_AB": round(rho, 3), "cos_uAuB": round(cos, 3),
            "type": _classify(cos), "multishaft": multishaft}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None)
    a = ap.parse_args()
    sids = a.subjects or sorted(os.path.basename(f)[:-5] for f in glob.glob(str(RANKDISP / "*.json")))
    rows = [analyze(s) for s in sids]
    ok = [r for r in rows if r.get("status") == "ok"]
    ms = [r for r in ok if r["multishaft"]]

    OUTCSV.parent.mkdir(parents=True, exist_ok=True)
    cols = ["subject", "dataset", "status", "stable_k", "n_joint", "n_shafts",
            "within_shaft_frac", "effective_rank", "rho_AB", "cos_uAuB", "type", "multishaft"]
    with open(OUTCSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    def _counts(rr):
        from collections import Counter
        return dict(Counter(r["type"] for r in rr))
    print(f"total rank-disp subjects: {len(rows)}; k=2 with axis pair (ok): {len(ok)}; "
          f"of those multi-shaft: {len(ms)}")
    print(f"  by dataset (ok): " + ", ".join(f"{ds}={sum(1 for r in ok if r['dataset']==ds)}"
                                             for ds in ("epilepsiae", "yuquan")))
    print(f"  ALL ok      types: {_counts(ok)}")
    print(f"  MULTI-SHAFT types: {_counts(ms)}   <- the meaningful cohort verdict")
    print(f"  single-shaft (cos trivially +-1): {len(ok)-len(ms)}")
    from scipy import stats
    if len(ms) >= 4:
        rr = np.array([[r["rho_AB"], r["cos_uAuB"]] for r in ms])
        print(f"  multi-shaft corr(rho_AB, cos_uAuB) = {stats.spearmanr(rr[:,0],rr[:,1])[0]:.2f} "
              f"p={stats.spearmanr(rr[:,0],rr[:,1])[1]:.3f}  (low => rank-corr and spatial-axis disagree)")

    # figure: rho_AB vs cos(uA,uB)
    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    dcol = {"epilepsiae": "#1f77b4", "yuquan": "#d62728"}
    for r in ok:
        mk = "o" if r["multishaft"] else "x"
        ax.scatter(r["rho_AB"], r["cos_uAuB"], marker=mk, s=90 if r["multishaft"] else 55,
                   facecolor=(dcol[r["dataset"]] if r["multishaft"] else "none"),
                   edgecolor=dcol[r["dataset"]], linewidths=1.6, zorder=3, alpha=0.9)
    ax.axhline(-0.5, color="0.7", ls="--", lw=1)
    ax.axhline(0.5, color="0.7", ls="--", lw=1)
    ax.axvline(-0.5, color="0.85", ls=":", lw=1)
    ax.plot([-1, 1], [-1, 1], color="0.85", lw=0.8, zorder=1)  # y=x reference
    ax.set_xlabel("rho_AB  (template rank correlation — channel-count sensitive)")
    ax.set_ylabel("cos(axis_A, axis_B)  (spatial propagation-axis angle)")
    ax.set_title("A/B relationship: rank correlation vs spatial axis angle\n"
                 "off the diagonal = the two disagree; y-bands = reversed / different / same axis",
                 fontsize=11)
    ax.text(-0.95, 0.9, "same axis", fontsize=8, color="0.4")
    ax.text(-0.95, 0.0, "two different axes", fontsize=8, color="0.4")
    ax.text(-0.95, -0.95, "reversed (one bidirectional axis)", fontsize=8, color="0.4")
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.2)
    ax.legend(handles=[Line2D([0], [0], marker="o", color=dcol["epilepsiae"], ls="none", label="epilepsiae"),
                       Line2D([0], [0], marker="o", color=dcol["yuquan"], ls="none", label="yuquan"),
                       Line2D([0], [0], marker="o", color="0.4", ls="none", markerfacecolor="0.4", label="multi-shaft"),
                       Line2D([0], [0], marker="x", color="0.4", ls="none", label="single-shaft (trivial cos)")],
              fontsize=8.5, loc="upper right")
    OUTFIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTFIG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {OUTCSV}\nwrote {OUTFIG}")


if __name__ == "__main__":
    main()
