#!/usr/bin/env python3
"""Topic 5 发作早期方向无监督两类聚类 ↔ 间期 A/B 方向 runner。

设计 spec: docs/superpowers/specs/2026-06-27-topic5-ictal-direction-clustering-design.md
口径: ictal-only k=2 (盲于间期) -> 三道预锁门 -> 描述性分档。无锚点/无预设/无队列断言。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.plot_topic5_axis_direction_rose import (_load_frame, _seizure_angles,
                                                     _electrode_kind)
from src.topic5_directional_replay import (
    SEED, plane_fit_direction, coord_aspect, cluster_directions_k2,
    unimodal_null_pvalue, bootstrap_label_stability, two_class_eligible,
    axis_quality_tier, angular_distance, best_pair_residual, best_pair_rotation_null)
from src.topic5_axis_direction import axial_mean, axial_distance

REAL_DIR = _ROOT / "results/spatial_modulation/propagation_geometry/observation_readout/real_subjects"
OUT_DIR = _ROOT / "results/topic5_ictal_recruitment/directional_clustering"
ACTIVATION_KEY = {"broadband": "bb_auc", "hfa": "hfa_auc"}
PRIMARY_COHORT = ["epilepsiae_442", "epilepsiae_548", "epilepsiae_583",
                  "epilepsiae_1084", "epilepsiae_384", "epilepsiae_958"]
ASPECT_MIN = 0.15


def template_direction(ds_sid, x, y, names, which):
    f = REAL_DIR / f"{ds_sid}_t_{which}.json"
    if not f.exists():
        return (np.nan, 0.0, 0.0, 0)
    j = json.loads(f.read_text())
    tr = {c["name"]: c.get("typical_rank") for c in j.get("channels", [])}
    vals = np.array([tr[n] if (n in tr and tr[n] is not None) else np.nan for n in names], float)
    return plane_fit_direction(x, y, vals)


def _report_tier(axis_tier, eligible, p_align, geometry_clean):
    if not geometry_clean:                       # P1 review: unclean geometry -> never two-class
        return "diagnostic_only"
    if axis_tier == "diagnostic_only":
        return "diagnostic_only"
    if not eligible:
        return "single_axis"
    if p_align is not None and np.isfinite(p_align) and p_align < 0.05:
        return "two_class_mapped"
    return "two_class_unmapped"


def process_subject(ds_sid, activation, *, n_perm=2000, n_boot=500, seed=SEED):
    loaded = _load_frame(ds_sid)
    if loaded is None:
        return {"subject": ds_sid, "activation": activation, "status": "skip",
                "reason": "no_frame"}
    rec, x, y, names = loaded
    ds, subj = ds_sid.split("_", 1)
    kind, _ = _electrode_kind(ds, subj, names)
    asp = coord_aspect(x, y)
    sz = _seizure_angles(ds_sid, x, y, names, activation)
    if sz.size < 4:
        return {"subject": ds_sid, "activation": activation, "status": "skip",
                "reason": "too_few_seizures", "n_sz": int(sz.size)}
    clus = cluster_directions_k2(sz, seed=0)
    p_bimodal, s_obs = unimodal_null_pvalue(sz, B=n_perm, seed=seed)
    stability = bootstrap_label_stability(sz, B=n_boot, seed=seed)
    eligible, reasons = two_class_eligible(clus["n"], clus["sizes"], p_bimodal, stability)
    thA, gnA, r2A, nvA = template_direction(ds_sid, x, y, names, "a")
    thB, gnB, r2B, nvB = template_direction(ds_sid, x, y, names, "b")
    delta_ab = angular_distance(thA, thB) if (np.isfinite(thA) and np.isfinite(thB)) else np.nan
    axis_tier = axis_quality_tier(delta_ab, nvA, nvB)
    bp = best_pair_residual(clus["means"], [thA, thB])               # dict or None
    p_align = best_pair_rotation_null(clus["means"], [thA, thB], B=n_perm, seed=seed)
    geom_clean = bool(kind == "ECoG" and np.isfinite(asp) and asp >= ASPECT_MIN)
    report_tier = _report_tier(axis_tier, eligible, p_align, geom_clean)
    m0, m1 = clus["means"]
    delta_ictal = angular_distance(m0, m1) if (np.isfinite(m0) and np.isfinite(m1)) else np.nan
    axis_offset = (axial_distance(axial_mean(clus["angles"]), axial_mean(np.array([thA, thB])))
                   if (np.isfinite(thA) and np.isfinite(thB) and clus["n"] >= 2) else np.nan)
    return {
        "subject": ds_sid, "activation": activation, "status": "ok",
        "geometry_clean": geom_clean,
        "electrode_kind": kind, "coord_aspect": None if not np.isfinite(asp) else float(asp),
        "n_sz": clus["n"], "sizes": clus["sizes"], "class_R": clus["class_R"],
        "means_deg": [None if not np.isfinite(m) else float(np.degrees(m)) for m in clus["means"]],
        "R_dir": clus["R_dir"], "R_axial": clus["R_axial"],
        "delta_ictal_deg": None if not np.isfinite(delta_ictal) else float(np.degrees(delta_ictal)),
        "axis_offset_deg": None if not np.isfinite(axis_offset) else float(np.degrees(axis_offset)),
        "silhouette": s_obs, "p_bimodal": p_bimodal, "stability": stability,
        "two_class_eligible": eligible, "two_class_reasons": reasons,
        "theta_A": None if not np.isfinite(thA) else float(np.degrees(thA)),
        "theta_B": None if not np.isfinite(thB) else float(np.degrees(thB)),
        "template_quality": {"grad_norm_a": gnA, "r2_a": r2A, "n_valid_a": nvA,
                             "grad_norm_b": gnB, "r2_b": r2B, "n_valid_b": nvB},
        "delta_ab_deg": None if not np.isfinite(delta_ab) else float(np.degrees(delta_ab)),
        "axis_tier": axis_tier,
        "best_pair_resid_sum_deg": None if bp is None else float(np.degrees(bp["sum"])),
        "best_pair_resid_each_deg": None if bp is None else [float(np.degrees(d)) for d in bp["matched"]],
        "best_pair_pairing": None if bp is None else bp["pairing"],
        "p_align": None if (p_align is None or not np.isfinite(p_align)) else float(p_align),
        "report_tier": report_tier,
        "provenance": {"spec": "docs/superpowers/specs/2026-06-27-topic5-ictal-direction-clustering-design.md",
                       "n_perm": n_perm, "n_boot": n_boot, "seed": seed},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--activation", choices=list(ACTIVATION_KEY), default="broadband")
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()
    subs = args.subjects or PRIMARY_COHORT
    (OUT_DIR / "per_subject").mkdir(parents=True, exist_ok=True)
    rows = []
    for sid in subs:
        rec = process_subject(sid, args.activation, n_perm=args.n_perm, n_boot=args.n_boot)
        (OUT_DIR / "per_subject" / f"{sid}__dir_cluster_{args.activation}.json").write_text(
            json.dumps(rec, indent=2))
        rows.append(rec)
        print(f"  {sid}: {rec.get('status')} "
              f"{rec.get('report_tier', '')} {rec.get('axis_tier', '')}", flush=True)
    summ = OUT_DIR / f"cohort_summary_{args.activation}.json"
    summ.write_text(json.dumps(rows, indent=2))
    ok = [r for r in rows if r["status"] == "ok"]
    cols = ["subject", "n_sz", "sizes", "R_dir", "R_axial", "p_bimodal", "stability",
            "two_class_eligible", "delta_ictal_deg", "delta_ab_deg", "axis_tier",
            "best_pair_resid_sum_deg", "best_pair_resid_each_deg", "best_pair_pairing",
            "p_align", "report_tier", "geometry_clean", "electrode_kind", "coord_aspect"]
    with open(OUT_DIR / f"cohort_summary_{args.activation}.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for r in ok:
            w.writerow([r.get(c) for c in cols])
    print(f"wrote {summ}", flush=True)


if __name__ == "__main__":
    main()
