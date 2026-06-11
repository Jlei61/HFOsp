#!/usr/bin/env python3
"""Do the two k=2 templates enter from the SAME or OPPOSITE channels? — 3D + axis.

Per subject (k=2, with coords), each template's "entry group" = smallest channel
set covering >=70% of that template's per-event entries (the small early group,
archive §4.6). Then, spatially:
  - OVERLAP: do the two entry groups share channels (Jaccard) / how far apart are
    their entry-prob-weighted centroids (mm)?
  - COMPACTNESS (confirm test): is each entry group geometrically tighter than k
    random recorded contacts in the same subject (within-subject random-k null,
    1000x, one-sided p)? CAVEAT: "compact" partly just reflects that the
    template's early end is at one spatial end (true for any axial propagation),
    so it confirms spatial localization of the entry, NOT a distinct entry
    mechanism. (Distinct object from the skeleton-geometry source/sink CORE null:
    that = template-mean top-k; this = per-event entry-prob group.)
  - render in 3D real space + projected onto the template propagation axis (axis
    from the template_rank spatial gradient, NOT from entry labels -> not
    circular). Epilepsiae MNI mm / Yuquan native RAS mm NEVER pooled.

Outputs: results/spatial_modulation/propagation_geometry/components/entry_variability/entry_overlap_summary.{json,csv}
         results/spatial_modulation/propagation_geometry/components/entry_variability/figures/entry_overlap_3d/<stem>.png
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

OUT = Path("results/spatial_modulation/propagation_geometry/components/entry_variability")
FIG = OUT / "figures" / "entry_overlap_3d"
COVERAGE = 0.70
N_NULL = 1000


def entry_group(prob: np.ndarray, coverage: float = COVERAGE) -> set:
    order = np.argsort(prob)[::-1]
    cum, grp = 0.0, set()
    for c in order:
        if prob[c] <= 0:
            break
        grp.add(int(c)); cum += prob[c]
        if cum >= coverage:
            break
    return grp


def weighted_centroid(idx, prob, coords, mapped):
    use = [c for c in idx if mapped[c]]
    if not use:
        return None
    w = prob[use]; w = w / w.sum()
    return (w[:, None] * coords[use]).sum(axis=0)


def _rms_radius(idx, coords) -> float:
    """Unweighted rms distance of the channel set to its own centroid (mm)."""
    xyz = coords[list(idx)]
    return float(np.sqrt(((xyz - xyz.mean(0)) ** 2).sum(1).mean()))


def compactness_null(group, coords, mapped, rng, n=N_NULL):
    """Is the entry group geometrically tighter than k random recorded contacts?
    Unweighted radius for BOTH observed and null (fair: random channels have no
    entry weight). Returns (obs_radius_mm, p_compact, k). p small = compact."""
    use = [c for c in group if mapped[c]]
    pool = np.where(mapped)[0]
    k = len(use)
    if k < 2 or len(pool) <= k:
        return (float("nan"), float("nan"), k)
    obs = _rms_radius(use, coords)
    null = np.array([_rms_radius(rng.choice(pool, size=k, replace=False), coords)
                     for _ in range(n)])
    return (round(obs, 2), round(float(np.mean(null <= obs)), 3), k)


def propagation_axis(template_rank, coords, mapped):
    """Spatial direction along which template rank increases (propagation axis),
    from the template (NOT entry labels). Regress each coord dim on rank over
    mapped channels; axis = normalized slope vector. Fallback to PCA-1."""
    use = np.where(mapped)[0]
    r = np.asarray(template_rank, float)[use]
    xyz = coords[use]
    if r.std() > 1e-9:
        rc = r - r.mean()
        slope = (xyz - xyz.mean(0)).T @ rc / (rc @ rc)  # (3,)
        if np.linalg.norm(slope) > 1e-9:
            return slope / np.linalg.norm(slope)
    # fallback: PCA-1 of mapped contacts
    u, s, vt = np.linalg.svd(xyz - xyz.mean(0), full_matrices=False)
    return vt[0]


def analyze(d: Dict, rng) -> Dict:
    chn = d["channel_names"]
    coords = np.array(d["coords_mm"]); mapped = np.array(d["mapped"], bool)
    cl = [c for c in d["clusters"] if "null" in c and c["null"].get("obs_earliest_prob")]
    pA = np.array(cl[0]["null"]["obs_earliest_prob"])
    pB = np.array(cl[1]["null"]["obs_earliest_prob"])
    gA, gB = entry_group(pA), entry_group(pB)
    shared, union = gA & gB, gA | gB
    jacc = len(shared) / len(union) if union else float("nan")
    cA = weighted_centroid(gA, pA, coords, mapped)
    cB = weighted_centroid(gB, pB, coords, mapped)
    dist = float(np.linalg.norm(cA - cB)) if (cA is not None and cB is not None) else float("nan")
    radA, pcA, kA = compactness_null(gA, coords, mapped, rng)
    radB, pcB, kB = compactness_null(gB, coords, mapped, rng)
    axis = propagation_axis(cl[0]["template_rank"], coords, mapped)
    tr1 = np.asarray(cl[0]["template_rank"], int)   # per-channel typical order in T1 (0=avg-first)
    tr2 = np.asarray(cl[1]["template_rank"], int)   # ... in T2
    return {
        "dataset": d["dataset"], "subject": d["subject"],
        "_tr1": tr1, "_tr2": tr2,
        "coord_space": d["coord_space"], "n_channels": d["n_channels"],
        "groupA": sorted(chn[c] for c in gA), "groupB": sorted(chn[c] for c in gB),
        "groupA_size": len(gA), "groupB_size": len(gB),
        "n_shared": len(shared), "jaccard": round(jacc, 3) if np.isfinite(jacc) else None,
        "centroid_dist_mm": round(dist, 2) if np.isfinite(dist) else None,
        "groupA_radius_mm": radA, "groupA_p_compact": pcA,
        "groupB_radius_mm": radB, "groupB_p_compact": pcB,
        "_pA": pA, "_pB": pB, "_coords": coords, "_mapped": mapped, "_chn": chn,
        "_gA": gA, "_gB": gB, "_shared": shared, "_cA": cA, "_cB": cB, "_axis": axis,
    }


def figure(r: Dict):
    coords, mapped, chn, axis = r["_coords"], r["_mapped"], r["_chn"], r["_axis"]
    pmax = np.maximum(r["_pA"], r["_pB"])
    mp = np.where(mapped)[0]
    fig = plt.figure(figsize=(13.5, 5.8))
    sp = "MNI" if "mni" in r["coord_space"] else "native RAS"
    fig.suptitle(f"{r['dataset']} — subject {r['subject']}  ({sp} mm)   "
                 f"entry-group overlap Jaccard={r['jaccard']}, centroid dist={r['centroid_dist_mm']} mm   "
                 f"| compactness p: T1={r['groupA_p_compact']}, T2={r['groupB_p_compact']}",
                 fontsize=10)
    # typical firing order within each template (1 = on-average first to fire).
    tr1, tr2 = r["_tr1"], r["_tr2"]
    ord_lab = {  # per-channel order tag, by which group the channel is in
        "A": lambda c: f"#{int(tr1[c]) + 1}",
        "B": lambda c: f"#{int(tr2[c]) + 1}",
        "S": lambda c: f"T1#{int(tr1[c]) + 1}/T2#{int(tr2[c]) + 1}",
    }
    groups = [(r["_gA"] - r["_shared"], "#c0504d", "template 1 entry group", "A"),
              (r["_gB"] - r["_shared"], "#3b6ea5", "template 2 entry group", "B"),
              (r["_shared"], "#7e57c2", "shared by both", "S")]

    # --- left: 3D real space ---
    ax = fig.add_subplot(121, projection="3d")
    ax.scatter(coords[mp, 0], coords[mp, 1], coords[mp, 2], s=10, c="0.8",
               edgecolor="none", depthshade=False, label="other contacts")
    for idx, color, lab, _ in groups:
        use = [c for c in idx if mapped[c]]
        if use:
            ax.scatter(coords[use, 0], coords[use, 1], coords[use, 2],
                       s=35 + 1200 * pmax[use], c=color, edgecolor="k", linewidth=0.5,
                       depthshade=False, label=lab)
    if r["_cA"] is not None and r["_cB"] is not None:
        cc = np.vstack([r["_cA"], r["_cB"]])
        ax.plot(cc[:, 0], cc[:, 1], cc[:, 2], "k--", lw=1.1)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_zlabel("z (mm)")
    ax.set_title("3D electrode space", fontsize=9.5)
    ax.legend(loc="upper left", fontsize=7)

    # --- right: projection onto template propagation axis ---
    ax = fig.add_subplot(122)
    ctr = coords[mp].mean(0)
    e2 = np.cross(axis, [0, 0, 1.0])
    if np.linalg.norm(e2) < 1e-6:
        e2 = np.cross(axis, [0, 1.0, 0])
    e2 = e2 / np.linalg.norm(e2)
    along = (coords - ctr) @ axis
    off = (coords - ctr) @ e2
    ax.scatter(along[mp], off[mp], s=10, c="0.8", edgecolor="none", label="other contacts")
    for idx, color, lab, gk in groups:
        use = [c for c in idx if mapped[c]]
        if use:
            ax.scatter(along[use], off[use], s=35 + 1200 * pmax[use], c=color,
                       edgecolor="k", linewidth=0.5, label=lab)
            for c in use:
                # annotate channel name + its typical firing order in that template
                ax.annotate(f"{chn[c]} {ord_lab[gk](c)}", (along[c], off[c]),
                            fontsize=6.3, color=color)
    ax.axhline(0, color="0.6", lw=0.7, ls=":")
    ax.set_xlabel("along propagation axis (mm)  [template-early → template-late]")
    ax.set_ylabel("off-axis (mm)")
    ax.set_title("projection onto template propagation axis\n"
                 "(label '#k' = channel is on average the k-th to fire in that template)",
                 fontsize=9)
    ax.set_aspect("equal", "datalim")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / f"{r['dataset']}_{r['subject']}.png"
    fig.savefig(out, dpi=145, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--showcase", type=str, default="ALL",
                    help="'ALL' (full cohort) or comma list dataset_subject")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    rows: List[Dict] = []
    for f in sorted((OUT / "per_subject").glob("*.json")):
        d = json.load(open(f))
        if d["chosen_k"] != 2 or not d.get("coords_mm"):
            continue
        cl = [c for c in d["clusters"] if "null" in c and c["null"].get("obs_earliest_prob")]
        if len(cl) != 2:
            continue
        rows.append(analyze(d, rng))

    def med(xs):
        xs = [x for x in xs if x is not None and np.isfinite(x)]
        return round(float(np.median(xs)), 3) if xs else None
    p_compact_all = [r["groupA_p_compact"] for r in rows] + [r["groupB_p_compact"] for r in rows]
    p_compact_all = [p for p in p_compact_all if p is not None and np.isfinite(p)]
    summary = {
        "n_subjects": len(rows), "coverage_threshold": COVERAGE, "n_null": N_NULL,
        "overlap": {
            "median_jaccard": med([r["jaccard"] for r in rows]),
            "fraction_disjoint_entry_groups": round(float(np.mean([r["n_shared"] == 0 for r in rows])), 3),
            "median_centroid_dist_mm_all": med([r["centroid_dist_mm"] for r in rows]),
            "by_coord_space": {sp: {"n": sum(1 for r in rows if r["coord_space"] == sp),
                                    "median_centroid_dist_mm": med([r["centroid_dist_mm"] for r in rows if r["coord_space"] == sp])}
                               for sp in sorted({r["coord_space"] for r in rows})},
        },
        "compactness": {
            "n_entry_groups": len(p_compact_all),
            "fraction_compact_p_lt_0.05": round(float(np.mean(np.array(p_compact_all) < 0.05)), 3),
            "median_p_compact": med(p_compact_all),
            "caveat": "'compact' partly reflects the template's localized early end "
                      "(any axial propagation) -> confirms spatial localization, not a "
                      "distinct entry mechanism.",
        },
    }
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    with open(OUT / "entry_overlap_summary.json", "w") as fh:
        json.dump({"summary": summary, "subjects": clean}, fh, indent=2)
    cols = ["dataset", "subject", "coord_space", "n_channels", "groupA_size", "groupB_size",
            "n_shared", "jaccard", "centroid_dist_mm",
            "groupA_radius_mm", "groupA_p_compact", "groupB_radius_mm", "groupB_p_compact"]
    with open(OUT / "entry_overlap_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in clean:
            w.writerow(r)
    print("SUMMARY:", json.dumps(summary, indent=1))

    sel = None if args.showcase == "ALL" else set(args.showcase.split(","))
    for r in rows:
        if sel is None or f"{r['dataset']}_{r['subject']}" in sel:
            figure(r)


if __name__ == "__main__":
    main()
