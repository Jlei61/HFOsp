#!/usr/bin/env python3
"""Per-subject field concordance on the D_AB gradient axis: template A | template B | seizure.

Reads the migrated D_AB-axis readout records (axis_definition=dab_gradient_v1) produced by
run_contact_plane_readout.py --axis dab_gradient, so the plane shown IS the one the maxAB
statistic runs on. Layout mirrors epilepsiae_1146_raw_broadband_minmax_no_flip.png:
  A, B  = interictal propagation order (viridis, rank 0=early -> 1=late), the two templates.
  right = seizure-onset broadband energy (Reds, raw robust-z min-max normalized, NO rank, NO flip).
Seizure field = mean over the subject's eligible seizures. Black ring = clinical SOZ.
D_AB defines the AXIS only; the fields are the template order and the ictal energy.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.plot_contact_plane_static import _smooth_rank_field_mm  # noqa: E402

DAB_DIR = REPO / "results/spatial_modulation/propagation_geometry/observation_readout/real_subjects_dab"
CACHE = REPO / "results/topic5_ictal_recruitment/t0_feature_cache"
OUT = REPO / "results/topic5_ictal_recruitment/axis_alignment/figures/field_concordance/abseizure_dab_axis"


def _rank01(v):
    v = np.asarray(v, float)
    out = np.full(v.shape, np.nan)
    ok = np.isfinite(v)
    if ok.sum() >= 2:
        out[ok] = np.argsort(np.argsort(v[ok])) / (ok.sum() - 1)
    elif ok.sum() == 1:
        out[ok] = 0.5
    return out


def _minmax(v):
    v = np.asarray(v, float)
    out = np.full(v.shape, np.nan)
    ok = np.isfinite(v)
    if ok.sum() >= 1:
        lo, hi = np.nanmin(v[ok]), np.nanmax(v[ok])
        out[ok] = (v[ok] - lo) / (hi - lo) if hi > lo else 0.5
    return out


def _ictal_mean(ds_sid, names, key="bb_auc"):
    npz, mj = CACHE / f"{ds_sid}.npz", CACHE / f"{ds_sid}.json"
    if not npz.exists():
        return None
    data = np.load(npz, allow_pickle=True)
    meta = json.load(open(mj))
    ch = [str(x) for x in data["channels"]]
    arrs = [data[f"{key}__{i}"] for i in meta["eligible_idxs"] if f"{key}__{i}" in data.files]
    if not arrs:
        return None
    by = {n: float(v) for n, v in zip(ch, np.nanmean(np.vstack([np.asarray(a, float) for a in arrs]), 0))}
    return np.array([by.get(n, np.nan) for n in names], float)


def _record_arrays(rec):
    ch = rec.get("channels", [])
    names = [c["name"] for c in ch]
    x = np.array([c.get("along_axis_mm", np.nan) for c in ch], float)
    y = np.array([c.get("signed_transverse_mm", np.nan) for c in ch], float)
    rank = np.array([c.get("typical_rank", np.nan) for c in ch], float)
    sup = np.array([c.get("support", 1.0) for c in ch], float)
    soz = np.array([bool(c.get("is_soz")) for c in ch])
    return names, x, y, rank, sup, soz


def _panel(ax, x, y, vals, sup, soz, names, xlim, ylim, sigma, cmap, title, cbar_label):
    _, _, T, _, _ = _smooth_rank_field_mm(x, y, vals, sup, xlim, ylim, sigma)
    im = ax.imshow(T, origin="lower", extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                   aspect="equal", cmap=cmap, vmin=0, vmax=1)
    ok = np.isfinite(vals) & np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[ok], y[ok], c=vals[ok], cmap=cmap, vmin=0, vmax=1, s=85, zorder=3,
               edgecolors=["k" if z else "white" for z in soz[ok]],
               linewidths=[1.9 if z else 1.0 for z in soz[ok]])
    for i in np.where(ok)[0]:
        ax.annotate(names[i], (x[i], y[i]), fontsize=5.6, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points", color="0.9")
    ax.set_title(title, fontsize=11.5)
    ax.set_xlabel("along D_AB axis (mm)", fontsize=11)
    ax.set_ylabel("transverse (mm)", fontsize=11)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(cbar_label, fontsize=10)


def make(dataset, subject, out=None):
    ds_sid = f"{dataset}_{subject}"
    fa, fb = DAB_DIR / f"{ds_sid}_t_a.json", DAB_DIR / f"{ds_sid}_t_b.json"
    if not (fa.exists() and fb.exists()):
        print(f"[skip] {ds_sid}: missing D_AB record")
        return None
    ra, rb = json.loads(fa.read_text()), json.loads(fb.read_text())
    if not ra.get("channels") or not rb.get("channels"):
        print(f"[skip] {ds_sid}: degenerate D_AB record ({ra.get('status')})")
        return None
    names, x, y, rank_a, sup, soz = _record_arrays(ra)
    _, _, _, rank_b, _, _ = _record_arrays(rb)     # same plane (shared axis), template-B field
    act = _ictal_mean(ds_sid, names)
    if act is None:
        print(f"[skip] {ds_sid}: no ictal cache")
        return None
    fA, fB, fS = _rank01(rank_a), _rank01(rank_b), _minmax(act)

    pad = 6.0
    fin = np.isfinite(x) & np.isfinite(y)
    xlim = (float(np.nanmin(x[fin])) - pad, float(np.nanmax(x[fin])) + pad)
    ylim = (float(np.nanmin(y[fin])) - pad, float(np.nanmax(y[fin])) + pad)
    pts = np.column_stack([x[fin], y[fin]])
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    sigma = float(np.median(d.min(1))) if len(pts) > 1 else 5.0

    qc = ra.get("axis_qc") or {}
    fig, ax = plt.subplots(1, 3, figsize=(19.5, 6.6), layout="constrained")
    _panel(ax[0], x, y, fA, sup, soz, names, xlim, ylim, sigma, "viridis",
           "interictal propagation order — template A", "early (0) -> late (1)")
    _panel(ax[1], x, y, fB, sup, soz, names, xlim, ylim, sigma, "viridis",
           "interictal propagation order — template B", "early (0) -> late (1)")
    _panel(ax[2], x, y, fS, sup, soz, names, xlim, ylim, sigma, "Reds",
           "seizure-onset activation — broadband 0-10 s",
           "low energy (0) -> high energy (1)")
    soz_h = Line2D([0], [0], marker="o", markerfacecolor="none", markeredgecolor="k",
                   linestyle="none", markersize=9, markeredgewidth=1.7, label="clinical SOZ")
    fig.legend(handles=[soz_h], loc="outside lower center", frameon=False, fontsize=11)
    r2 = qc.get("R2"); wsf = qc.get("within_shaft_frac"); nsh = qc.get("n_shafts")
    loso = qc.get("loso_cosine")
    fig.suptitle(
        f"{dataset} {subject} — field concordance on the D_AB gradient axis "
        f"(A, B interictal order | mean seizure energy)\n"
        f"axis QC: R²={r2:.2f}, within-shaft var={wsf:.0%}, shafts={nsh}, "
        f"leave-one-shaft-out cos={('%.2f' % loso) if loso == loso else 'n/a'}   "
        f"[D_AB defines the axis only; fields = template order & ictal energy]",
        fontsize=12)
    out = Path(out) if out else OUT / f"{ds_sid}_abseizure_dab.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=135, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out.name}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None, help="ds:subj tokens; default = all with a D_AB record")
    a = ap.parse_args()
    if a.subjects:
        toks = [s.split(":", 1) for s in a.subjects]
    else:
        toks = sorted({tuple(f.stem.rsplit("_t_", 1)[0].split("_", 1))
                       for f in DAB_DIR.glob("*_t_a.json")})
    n = 0
    for ds, subj in toks:
        if make(ds, subj):
            n += 1
    print(f"[done] {n} figures -> {OUT}")


if __name__ == "__main__":
    main()
