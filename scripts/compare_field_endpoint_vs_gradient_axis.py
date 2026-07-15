#!/usr/bin/env python3
"""Swap-positive subjects: is the endpoint(swap) axis a BETTER calibration than the gradient axis
for seizure-early field concordance? Same field pipeline, ONLY the per-template axis differs.

For each template T in {A,B} and each axis method:
  endpoint : build_endpoint_cores(rank_T) source->sink  (the swap/main-analysis axis)
  gradient : -gradient(eT), eT=-z(rankT)                 (early-to-late propagation axis)
project contacts onto the axis plane, smooth interictal rank field + seizure broadband energy on
the SAME plane, |corr_pair_mirror_invariant|; maxAB = max over templates. Channel-shuffle null
(B) gives the margin (real - null median) — the honest measure, since raw |r| co-inflates with null.
"""
import argparse
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
GRID_N = 61


def _z(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else x * 0.0


def _rank01(v):
    v = np.asarray(v, float)
    out = np.full(v.shape, np.nan)
    ok = np.isfinite(v)
    if ok.sum() >= 2:
        out[ok] = np.argsort(np.argsort(v[ok])) / (ok.sum() - 1)
    elif ok.sum() == 1:
        out[ok] = 0.5
    return out


def _ictal(sid, names):
    npz, mj = CACHE / f"{sid}.npz", CACHE / f"{sid}.json"
    if not npz.exists():
        return None
    data = np.load(npz, allow_pickle=True)
    meta = json.load(open(mj))
    ch = [str(x) for x in data["channels"]]
    arrs = [data[f"bb_auc__{i}"] for i in meta["eligible_idxs"] if f"bb_auc__{i}" in data.files]
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


def _axis_endpoint(rank, coords):
    c = build_endpoint_cores(rank, np.ones(len(coords), bool), k_primary=3)
    if c["tier"] == "descriptive_only":
        return None
    fr = compute_axis_frame(coords, c["source_idx"], c["sink_idx"])
    if fr["degenerate_axis"]:
        return None
    src = np.array(fr["source_centroid"])
    u = (np.array(fr["sink_centroid"]) - src)
    u = u / max(np.linalg.norm(u), 1e-12)
    along = np.asarray(fr["along_axis"], float)
    perp = (coords - src) - np.outer(np.where(np.isnan(along), 0.0, along), u)
    w = np.linalg.svd(perp[np.isfinite(perp).all(1)] - perp[np.isfinite(perp).all(1)].mean(0),
                      full_matrices=False)[2][0]
    return u, w, coords.mean(0)


def _axis_gradient(scalar, coords, shafts):
    ax = compute_template_propagation_axis(coords, scalar, shafts)
    if ax["status"] != "ok":
        return None
    return np.asarray(ax["u"]), np.asarray(ax["w"]), np.asarray(ax["xbar"])


def _field_corr(xn, yn, inter, seiz, rng=None):
    """|corr_pair_mirror_invariant| of the smoothed interictal vs seizure field on one plane."""
    X, Y = R.make_plane_grid(GRID_N)
    sup = np.ones(len(xn))
    rec_i = {"channels": [{"x_norm": xn[i], "y_norm": yn[i], "typical_rank": inter[i], "support": 1.0}
                          for i in range(len(xn)) if np.isfinite(xn[i]) and np.isfinite(inter[i])]}
    s = seiz if rng is None else _rank01(rng.permutation(seiz))
    rec_s = {"channels": [{"x_norm": xn[i], "y_norm": yn[i], "typical_rank": s[i], "support": 1.0}
                          for i in range(len(xn)) if np.isfinite(xn[i]) and np.isfinite(s[i])]}
    fi = R.smooth_field(rec_i, X, Y)
    fs = R.smooth_field(rec_s, X, Y)
    res = R.corr_pair_mirror_invariant(fi["T"], fi["S"], fs["T"], fs["S"])
    c = res.get("corr")
    return abs(float(c)) if c is not None else np.nan


def _maxab_on_axis(method, coords, raj, rbj, sh, seiz, B, seed):
    rng = np.random.default_rng(seed)
    vals = []
    for rank in (raj, rbj):
        ax = (_axis_endpoint(rank, coords) if method == "endpoint"
              else _axis_gradient(-_z(rank), coords, sh))
        if ax is None:
            continue
        pl = _plane(coords, *ax)
        if pl is None:
            continue
        xn, yn = pl
        inter = _rank01(rank)
        sz = _rank01(seiz)
        real = _field_corr(xn, yn, inter, sz)
        if not np.isfinite(real):
            continue
        null = np.array([_field_corr(xn, yn, inter, seiz, rng=rng) for _ in range(B)])
        if not np.isfinite(null).any():
            continue
        vals.append((real, real - np.nanmedian(null), bool(real > np.nanpercentile(null, 95))))
    if not vals:
        return None
    # maxAB = template with the larger real |r|
    return max(vals, key=lambda v: v[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swap", nargs="*", default=["strict", "candidate"])
    ap.add_argument("--B", type=int, default=200)
    a = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(str(RANKDISP / "*.json"))):
        d = json.load(open(f))
        if d.get("stable_k") != 2:
            continue
        sid = os.path.basename(f)[:-5]
        ds, subj = sid.split("_", 1)
        p = d["pairs"][0]
        swc = (p.get("swap_sweep") or {}).get("swap_class", "none")
        if swc not in a.swap:
            continue
        nm = p["channel_names"]
        jv = np.array(p["joint_valid"], bool)
        ra = np.array(p["rank_a_dense_full"], float)
        rb = np.array(p["rank_b_dense_full"], float)
        jn = [nm[i] for i in np.where(jv)[0]]
        seiz = _ictal(sid, jn)
        if seiz is None:
            rows.append((sid, swc, "no_ictal_cache"))
            continue
        try:
            cr = load_subject_coords(ds, subj, jn)
        except Exception:
            rows.append((sid, swc, "no_coords"))
            continue
        C = np.asarray(cr.coords_array_in_requested_order, float)
        mp = np.asarray(cr.mapped_mask_in_requested_order, bool)
        if mp.sum() < 7:
            rows.append((sid, swc, "insufficient"))
            continue
        Cm, raj, rbj = C[mp], ra[jv][mp], rb[jv][mp]
        sz = seiz[mp]
        sh = [parse_shaft(jn[i])[0] for i in np.where(mp)[0]]
        e = _maxab_on_axis("endpoint", Cm, raj, rbj, sh, sz, a.B, 0)
        g = _maxab_on_axis("gradient", Cm, raj, rbj, sh, sz, a.B, 0)
        rows.append((sid, swc, e, g))
        print(f"[{sid}] {swc}: endpoint {e}  gradient {g}", flush=True)

    print("\n=== TABLE: seizure-early field concordance, endpoint(swap) axis vs gradient axis ===")
    print(f"{'subject':16s} {'swap':10s} {'endpt |r|':9s} {'endpt marg':10s} {'grad |r|':8s} "
          f"{'grad marg':9s} {'endpt>grad?':s}")
    ok = [r for r in rows if len(r) == 4 and r[2] and r[3]]
    for sid, swc, e, g in ok:
        better = "endpoint" if e[1] > g[1] else "gradient"
        print(f"{sid:16s} {swc:10s} {e[0]:.3f}{'':4s} {e[1]:+.3f}{'':4s} {g[0]:.3f}{'':3s} "
              f"{g[1]:+.3f}{'':3s} {better} ({'+' if e[1]>g[1] else ''}{e[1]-g[1]:.3f})")
    for r in rows:
        if len(r) == 3:
            print(f"{r[0]:16s} {r[1]:10s} SKIP: {r[2]}")
    if ok:
        de = np.array([e[1] for _, _, e, g in ok])
        dg = np.array([g[1] for _, _, e, g in ok])
        print(f"\nmargin: endpoint median {np.median(de):+.3f} vs gradient median {np.median(dg):+.3f}; "
              f"endpoint better in {sum(e[1]>g[1] for _,_,e,g in ok)}/{len(ok)}")
        print(f"pass channel-null: endpoint {sum(e[2] for _,_,e,g in ok)}/{len(ok)}, "
              f"gradient {sum(g[2] for _,_,e,g in ok)}/{len(ok)}")


if __name__ == "__main__":
    main()
