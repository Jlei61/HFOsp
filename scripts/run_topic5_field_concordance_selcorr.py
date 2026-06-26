#!/usr/bin/env python3
"""Topic 5 — field-concordance with SELECTION-CORRECTED null (board upgrade, formal pass).

The best-only board is a screen: it picks the best of N candidates (substrate × band) without
correcting for that selection in the null. This runner makes it formal: per subject it recomputes
every candidate's real field-alignment AND its per-draw channel-null, then applies the max-statistic
selection-corrected p-value (null repeats "take the best candidate" each draw). See
src/topic5_field_selcorr.py.

Candidates (match the board): bb/HFA × {maxAB (narrow t_a/t_b, stat=max over the two),
broad (broad t_a)}. A subject contributes only the candidates whose axis record + ictal activation
exist. Field smoothing is matmul-accelerated; corr is the A-line mirror-invariant, sign-free stat.
EXPLORATORY secondary; no cohort verdict.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from src.topic5_axis_alignment import matched_channels
from src.propagation_contact_plane_readout import (make_plane_grid, R_smooth_rank,
                                                   corr_pair_mirror_invariant, S_THRESH, OVERLAP_MIN)
import src.topic5_field_selcorr as sc

NARROW = _ROOT / "results/spatial_modulation/propagation_geometry/observation_readout/real_subjects"
BROAD = _ROOT / "results/spatial_modulation/propagation_geometry_broad/observation_readout/real_subjects"
CACHES = [_ROOT / "results/topic5_ictal_recruitment/t0_feature_cache",
          _ROOT / "results/topic5_ictal_recruitment/t0_feature_cache_v2_windows"]
OUT = _ROOT / "results/topic5_ictal_recruitment/axis_alignment/field_concordance_selcorr"
MIN_CH = 6
BANDS = {"bb": "bb_auc", "HFA": "hfa_auc"}
RNG_SEED = 20260626


def _cache(ds_sid):
    for c in CACHES:
        if (c / f"{ds_sid}.npz").exists() and (c / f"{ds_sid}.json").exists():
            return np.load(c / f"{ds_sid}.npz", allow_pickle=True), json.load(open(c / f"{ds_sid}.json"))
    return None, None


def _abs_corr(Fi, Fj):
    r = corr_pair_mirror_invariant(Fi["T"], Fi["S"], Fj["T"], Fj["S"], S_THRESH, OVERLAP_MIN)["corr"]
    return abs(r) if r is not None and np.isfinite(r) else np.nan


def _candidate_planes(ds_sid):
    """{candidate_substrate: [axis_record_paths]} for the substrates whose records exist."""
    out = {}
    if (NARROW / f"{ds_sid}_t_a.json").exists() and (NARROW / f"{ds_sid}_t_b.json").exists():
        out["maxAB"] = [NARROW / f"{ds_sid}_t_a.json", NARROW / f"{ds_sid}_t_b.json"]
    if (BROAD / f"{ds_sid}_t_a.json").exists():
        out["broad"] = [BROAD / f"{ds_sid}_t_a.json"]
    return out


def _prep_plane(axis_path, cache_names, X, Y):
    """Build the interictal rank field + the ictal smoother (matmul) for one plane.
    Returns (F_inter, smoother, matched_cache_idx) or None."""
    axis = json.load(open(axis_path))
    if not axis.get("channels"):
        return None
    matched = matched_channels(axis, {n: 0.0 for n in cache_names})
    if len(matched) < MIN_CH:
        return None
    cidx = {n: i for i, n in enumerate(cache_names)}
    m_in_cache = np.array([cidx[c["name"]] for c in matched])
    inter = [float(c["typical_rank"]) for c in matched]
    rec = {"channels": [dict(c, typical_rank=float(v)) for c, v in zip(matched, inter)]}
    F_inter = R_smooth_rank(rec, X, Y, None, S_THRESH)
    sm = sc.precompute_smoother(matched, X, Y, F_inter["sigma_xy"])
    if sm is None:
        return None
    return F_inter, sm, m_in_cache


def _run_subject(ds_sid, *, B, rng, n_perm_cap=None):
    data, meta = _cache(ds_sid)
    if data is None:
        return {"subject_id": ds_sid, "status": "no_ictal_cache"}
    cache_names = [str(x) for x in data["channels"]]
    elig = meta.get("eligible_idxs") or []
    if not elig:
        return {"subject_id": ds_sid, "status": "no_eligible_idxs"}
    X, Y = make_plane_grid()
    planes = _candidate_planes(ds_sid)
    real_by_cand, nulldist_by_cand, cand_meta = {}, {}, {}
    for substrate, paths in planes.items():
        prepped = [_prep_plane(p, cache_names, X, Y) for p in paths]
        prepped = [p for p in prepped if p is not None]
        if not prepped:
            continue
        for band, key in BANDS.items():
            name = f"{band} {substrate}"
            per_sz_real, per_sz_null = [], []
            for idx in elig:
                k = f"{key}__{idx}"
                if k not in data.files:
                    continue
                ict = np.asarray(data[k], float)
                # per-plane real + null, take max over planes (maxAB) or single (broad)
                plane_real, plane_null = [], []
                for (F_inter, sm, m_in_cache) in prepped:
                    v = ict[m_in_cache]
                    if np.isfinite(v).sum() < MIN_CH:
                        plane_real = []; break
                    plane_real.append(_abs_corr(F_inter, sc.field_from_values(sm, v)))
                    # B permuted value vectors -> vectorized mirror-invariant |corr| null draws
                    V = np.array([v[rng.permutation(v.size)] for _ in range(B)])      # (B, nch)
                    plane_null.append(sc.null_aligns_vectorized(F_inter, sm, V, S_THRESH, OVERLAP_MIN))
                if not plane_real:
                    continue
                per_sz_real.append(float(np.nanmax(plane_real)))
                per_sz_null.append(np.nanmax(np.vstack(plane_null), axis=0))   # (B,) max over planes
            if not per_sz_real:
                continue
            real_by_cand[name] = float(np.nanmedian(per_sz_real))
            nulldist_by_cand[name] = np.nanmedian(np.vstack(per_sz_null), axis=0).tolist()  # median over sz, (B,)
            cand_meta[name] = {"n_seizures": len(per_sz_real)}
    if not real_by_cand:
        return {"subject_id": ds_sid, "status": "no_candidates"}
    selc = sc.selection_corrected_pvalue(real_by_cand, nulldist_by_cand)
    # per-candidate summary (real + this candidate's OWN channel-null p95) — reusable for the
    # screen board's legacy per-candidate JSONs (backfill missing maxAB rows).
    per_cand = {c: {"real_median_maxab": float(real_by_cand[c]),
                    "channel_null_p95": float(np.nanpercentile(nulldist_by_cand[c], 95)),
                    "channel_null_median": float(np.nanmedian(nulldist_by_cand[c])),
                    "n_seizures": cand_meta[c]["n_seizures"]} for c in real_by_cand}
    return {"subject_id": ds_sid, "dataset": ds_sid.split("_", 1)[0], "status": "ok",
            "B": B, "candidate_n_seizures": cand_meta, "selcorr": selc, "per_candidate": per_cand}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    outdir = Path(args.out); psdir = outdir / "per_subject"; psdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    if args.subjects:
        subs = args.subjects
    else:                                            # all subjects with any candidate plane + cache
        subs = sorted({p.name[:-len("_t_a.json")] for d in (NARROW, BROAD) for p in d.glob("*_t_a.json")})
    for ds_sid in subs:
        if not args.force and (psdir / f"{ds_sid}.json").exists():
            print(f"[skip-exists] {ds_sid}", flush=True); continue
        print(f"[run] {ds_sid} ...", flush=True)
        res = _run_subject(ds_sid, B=args.B, rng=rng)
        json.dump(res, open(psdir / f"{ds_sid}.json", "w"), indent=2)
        s = res.get("selcorr", {})
        if res["status"] == "ok":
            print(f"    best={s.get('best_candidate')} obs={s.get('observed_max'):.3f} "
                  f"p_selcorr={s.get('p_selcorr'):.4f} pass={s.get('pass_selcorr')} "
                  f"n_cand={s.get('n_candidates')}", flush=True)
        else:
            print(f"    status={res['status']}", flush=True)
    # cohort summary from disk
    rows = []
    for f in sorted(psdir.glob("*.json")):
        r = json.load(open(f))
        if r.get("status") == "ok":
            s = r["selcorr"]
            rows.append({"subject_id": r["subject_id"], "dataset": r["dataset"],
                         "best_candidate": s["best_candidate"], "observed_max": s["observed_max"],
                         "p_selcorr": s["p_selcorr"], "pass_selcorr": s["pass_selcorr"],
                         "n_candidates": s["n_candidates"]})
    for ds in ("epilepsiae", "yuquan", None):
        dr = [r for r in rows if (ds is None or r["dataset"] == ds)]
        if dr:
            npass = sum(r["pass_selcorr"] for r in dr)
            print(f"  [{ds or 'all'}] selection-corrected pass: {npass}/{len(dr)} (p<0.05)")
    json.dump({"note": "selection-corrected (max-statistic family-wise) field concordance; formal pass "
               "(best-of-N selection repeated in the null). EXPLORATORY secondary, no cohort verdict.",
               "B": args.B, "rows": rows}, open(outdir / "cohort_summary.json", "w"), indent=2)
    print(f"[done] -> {outdir/'cohort_summary.json'}")


if __name__ == "__main__":
    main()
