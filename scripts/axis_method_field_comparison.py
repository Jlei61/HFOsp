#!/usr/bin/env python3
"""Systematic axis-building-method comparison via seizure-early field concordance (narrow pool).

For every k=2 subject with an ictal cache, for each band and each per-template axis method, build
template A's and B's OWN axis, project onto the axis plane, smooth the interictal rank field and the
seizure activation field on the SAME plane, and take maxAB = max_T |corr_pair_mirror_invariant|.
Channel-shuffle null (B) gives the margin (real - null median) — the honest measure.

Axis methods (per template T, rank r_T on coord-mapped joint contacts):
  gradient          early->late axis -gradient(e_T), e_T=-zscore(r_T), over 3D coords (endpoint-free).
  endpoint_fixed    build_endpoint_cores(r_T, k=3/2) source->sink centroid axis (frozen main / figure).
  endpoint_decisionk source=bottom-k, sink=top-k of r_T at k=swap decision_k (the swap-node axis).

Bands: broadband=bb_auc (1-45 Hz), hfa=hfa_auc (60-100 Hz). broadband150 (1-150) is unavailable
(feature absent from the current t0 cache) and is reported as blocked, not run.
"""
import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.topic5_template_axis_field import compute_template_propagation_axis  # noqa: E402
from src.seeg_coord_loader import load_subject_coords  # noqa: E402
from src.propagation_skeleton_geometry import (build_endpoint_cores, compute_axis_frame,  # noqa: E402
                                               parse_shaft)
from src import propagation_contact_plane_readout as R  # noqa: E402

RANKDISP = REPO / "results/interictal_propagation_masked/rank_displacement/per_subject"
CACHE = REPO / "results/topic5_ictal_recruitment/t0_feature_cache"
BANDS = {"broadband": "bb_auc", "hfa": "hfa_auc"}
METHODS = ["gradient", "endpoint_fixed", "endpoint_decisionk"]
GRID_N = 61
_GX, _GY = R.make_plane_grid(GRID_N)


def _z(x):
    x = np.asarray(x, float); s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else x * 0.0


def _rank01(v):
    v = np.asarray(v, float); out = np.full(v.shape, np.nan); ok = np.isfinite(v)
    if ok.sum() >= 2:
        out[ok] = np.argsort(np.argsort(v[ok])) / (ok.sum() - 1)
    elif ok.sum() == 1:
        out[ok] = 0.5
    return out


def _ictal(sid, names, key):
    npz, mj = CACHE / f"{sid}.npz", CACHE / f"{sid}.json"
    if not npz.exists():
        return None
    data = np.load(npz, allow_pickle=True); meta = json.load(open(mj))
    ch = [str(x) for x in data["channels"]]
    arrs = [data[f"{key}__{i}"] for i in meta["eligible_idxs"] if f"{key}__{i}" in data.files]
    if not arrs:
        return None
    by = {n: float(v) for n, v in zip(ch, np.nanmean(np.vstack([np.asarray(a, float) for a in arrs]), 0))}
    return np.array([by.get(n, np.nan) for n in names], float)


def _plane(coords, u, w, xbar):
    along = (coords - xbar) @ u
    resid = (coords - xbar) - np.outer(along, u)
    trans = resid @ w
    part = np.isfinite(along)
    if part.sum() < 2:
        return None
    scale = np.percentile(along[part], 97.5) - np.percentile(along[part], 2.5)
    scale = scale if scale > 1e-9 else 1.0
    return along / scale, trans / scale


def _perp_dir(coords, u, origin):
    perp = (coords - origin) - np.outer((coords - origin) @ u, u)
    fin = np.isfinite(perp).all(1)
    if fin.sum() < 2:
        return np.zeros(3)
    return np.linalg.svd(perp[fin] - perp[fin].mean(0), full_matrices=False)[2][0]


def _axis(method, rank, coords, shafts, dk):
    if method == "gradient":
        ax = compute_template_propagation_axis(coords, -_z(rank), shafts)
        return (np.asarray(ax["u"]), np.asarray(ax["w"]), np.asarray(ax["xbar"])) if ax["status"] == "ok" else None
    if method == "endpoint_fixed":
        c = build_endpoint_cores(rank, np.ones(len(coords), bool), k_primary=3)
        if c["tier"] == "descriptive_only":
            return None
        src_idx, snk_idx = c["source_idx"], c["sink_idx"]
    else:  # endpoint_decisionk
        if not dk or 2 * int(dk) > len(rank):
            return None
        order = np.argsort(rank, kind="stable")
        src_idx, snk_idx = order[: int(dk)].tolist(), order[-int(dk):].tolist()
    src = np.nanmean(coords[src_idx], 0); snk = np.nanmean(coords[snk_idx], 0)
    u = snk - src; n = np.linalg.norm(u)
    if n < 1e-9:
        return None
    u = u / n
    return u, _perp_dir(coords, u, coords.mean(0)), coords.mean(0)


def _smooth(xn, yn, vals):
    rec = {"channels": [{"x_norm": xn[i], "y_norm": yn[i], "typical_rank": vals[i], "support": 1.0}
                        for i in range(len(xn)) if np.isfinite(xn[i]) and np.isfinite(vals[i])]}
    return R.smooth_field(rec, _GX, _GY)


def _corr(Ti, Si, xn, yn, sval):
    fs = _smooth(xn, yn, sval)
    res = R.corr_pair_mirror_invariant(Ti, Si, fs["T"], fs["S"])
    c = res.get("corr")
    return abs(float(c)) if c is not None else np.nan


def _maxab(method, coords, raj, rbj, sh, seiz, dk, B, seed):
    rng = np.random.default_rng(seed)
    best = None
    for rank in (raj, rbj):
        ax = _axis(method, rank, coords, sh, dk)
        if ax is None:
            continue
        pl = _plane(coords, *ax)
        if pl is None:
            continue
        xn, yn = pl
        fi = _smooth(xn, yn, _rank01(rank))          # interictal field smoothed ONCE
        Ti, Si = fi["T"], fi["S"]
        sz = _rank01(seiz)
        real = _corr(Ti, Si, xn, yn, sz)
        if not np.isfinite(real):
            continue
        null = np.array([_corr(Ti, Si, xn, yn, _rank01(rng.permutation(seiz))) for _ in range(B)])
        if not np.isfinite(null).any():
            continue
        cand = (real, real - np.nanmedian(null), bool(real > np.nanpercentile(null, 95)))
        if best is None or cand[0] > best[0]:
            best = cand
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None, help="ds_sid tokens; default all k=2")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sids = a.subjects or [os.path.basename(f)[:-5] for f in sorted(glob.glob(str(RANKDISP / "*.json")))]
    rows = []
    for sid in sids:
        f = RANKDISP / f"{sid}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        if d.get("stable_k") != 2:
            continue
        ds, subj = sid.split("_", 1)
        p = d["pairs"][0]
        sw = (p.get("swap_sweep") or {})
        swc, dk = sw.get("swap_class", "none"), sw.get("decision_k")
        nm = p["channel_names"]; jv = np.array(p["joint_valid"], bool)
        ra = np.array(p["rank_a_dense_full"], float); rb = np.array(p["rank_b_dense_full"], float)
        jn = [nm[i] for i in np.where(jv)[0]]
        if not any((CACHE / f"{sid}.npz").exists() for _ in [0]):
            continue
        try:
            cr = load_subject_coords(ds, subj, jn)
        except Exception:
            continue
        C = np.asarray(cr.coords_array_in_requested_order, float)
        mp = np.asarray(cr.mapped_mask_in_requested_order, bool)
        if mp.sum() < 7:
            continue
        Cm, raj, rbj = C[mp], ra[jv][mp], rb[jv][mp]
        sh = [parse_shaft(jn[i])[0] for i in np.where(mp)[0]]
        gax = compute_template_propagation_axis(Cm, -_z(raj), sh)
        n_shafts = gax.get("n_shafts") if gax["status"] == "ok" else None
        er = gax.get("effective_rank") if gax["status"] == "ok" else None
        for band, key in BANDS.items():
            seiz_full = _ictal(sid, jn, key)
            if seiz_full is None:
                continue
            sz = seiz_full[mp]
            for method in METHODS:
                r = _maxab(method, Cm, raj, rbj, sh, sz, dk, a.B, 0)
                if r is None:
                    rows.append(dict(subject=sid, dataset=ds, swap_class=swc, n_shafts=n_shafts,
                                     effective_rank=er, decision_k=dk, band=band, method=method,
                                     maxab_r="", margin="", pass_null=""))
                else:
                    rows.append(dict(subject=sid, dataset=ds, swap_class=swc, n_shafts=n_shafts,
                                     effective_rank=er, decision_k=dk, band=band, method=method,
                                     maxab_r=round(r[0], 4), margin=round(r[1], 4), pass_null=r[2]))
                print(f"[{sid}] {band} {method}: {rows[-1]['maxab_r']} marg={rows[-1]['margin']} "
                      f"pass={rows[-1]['pass_null']}", flush=True)
    cols = ["subject", "dataset", "swap_class", "n_shafts", "effective_rank", "decision_k",
            "band", "method", "maxab_r", "margin", "pass_null"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"wrote {a.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
