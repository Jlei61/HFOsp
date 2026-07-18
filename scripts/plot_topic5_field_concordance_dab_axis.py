#!/usr/bin/env python3
"""Field concordance on the D_AB gradient AXIS.

Architecture (user-locked 2026-07-13):
  axis  = D_AB 3D least-squares gradient — endpoint-free, no source/sink, no decision-k
          (reuses scripts.plot_topic5_dab_axis_subject.compute). D_AB defines the axis ONLY.
  left  = interictal propagation order (template-A rank = real interictal event timing),
          smoothed on the D_AB-axis plane. This is the field value; D_AB is NOT the field value.
  right = seizure-onset broadband energy (mean 0-10 s), same plane, sign-oriented for readability.
  statistic = maxAB (frozen A-line primary): resemblance of the seizure field to the better of the
              two interictal templates. This figure is the concordance VISUAL on the D_AB axis.
"""
import argparse
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
sys.path.insert(0, str(REPO))
from scripts.plot_topic5_dab_axis_subject import compute  # noqa: E402
from scripts.plot_contact_plane_static import _smooth_rank_field_mm  # noqa: E402

CACHE = REPO / "results/topic5_ictal_recruitment/t0_feature_cache"
OUT = REPO / "results/topic5_ictal_recruitment/axis_alignment/figures/field_concordance/field_concordance_dab_axis"


def _rank01(v):
    v = np.asarray(v, float)
    out = np.full(v.shape, np.nan)
    ok = np.isfinite(v)
    if ok.sum() >= 2:
        out[ok] = np.argsort(np.argsort(v[ok])) / (ok.sum() - 1)
    elif ok.sum() == 1:
        out[ok] = 0.5
    return out


def _ictal(ds_sid, names, key="bb_auc"):
    """Per-contact mean early-ictal activation over eligible seizures, aligned to `names`."""
    npz, mj = CACHE / f"{ds_sid}.npz", CACHE / f"{ds_sid}.json"
    if not npz.exists():
        return None
    data = np.load(npz, allow_pickle=True)
    meta = json.load(open(mj))
    ch = [str(x) for x in data["channels"]]
    arrs = [data[f"{key}__{i}"] for i in meta["eligible_idxs"] if f"{key}__{i}" in data.files]
    if not arrs:
        return None
    mean_act = np.nanmean(np.vstack([np.asarray(a, float) for a in arrs]), axis=0)
    by = {n: float(v) for n, v in zip(ch, mean_act)}
    return np.array([by.get(n, np.nan) for n in names], float)


def _median_nn(pts):
    if len(pts) < 2:
        return 5.0
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))


def _panel(ax, along, perp, vals, sup, soz_mask, names, xlim, ylim, sigma, title, cbar_label):
    _, _, T, _, _ = _smooth_rank_field_mm(along, perp, vals, sup, xlim, ylim, sigma)
    im = ax.imshow(T, origin="lower", extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                   aspect="equal", cmap="viridis", vmin=0, vmax=1)
    ok = np.isfinite(vals)
    ax.scatter(along[ok], perp[ok], c=vals[ok], cmap="viridis", vmin=0, vmax=1, s=95, zorder=3,
               edgecolors=["k" if z else "white" for z in soz_mask[ok]],
               linewidths=[1.9 if z else 1.0 for z in soz_mask[ok]])
    for i in np.where(ok)[0]:
        ax.annotate(names[i], (along[i], perp[i]), fontsize=5.8, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points", color="0.92")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("along D_AB axis (mm)", fontsize=12)
    ax.set_ylabel("transverse (mm)", fontsize=12)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(cbar_label, fontsize=11)


def make(dataset, subject, pool, out=None):
    r = compute(dataset, subject, pool)
    names = list(r["names"])
    along, perp = np.asarray(r["along"]), np.asarray(r["perp"])
    act = _ictal(f"{dataset}_{subject}", names)
    if act is None:
        print(f"[skip] {dataset}_{subject}: no ictal cache")
        return None
    ict_raw = _rank01(act)
    rank_a01, rank_b01 = _rank01(r["rank_a"]), _rank01(r["rank_b"])
    ma = np.isfinite(rank_a01) & np.isfinite(ict_raw)
    mb = np.isfinite(rank_b01) & np.isfinite(ict_raw)
    if ma.sum() < 3:
        print(f"[skip] {dataset}_{subject}: <3 matched contacts")
        return None
    r_a = abs(np.corrcoef(rank_a01[ma], ict_raw[ma])[0, 1])
    r_b = abs(np.corrcoef(rank_b01[mb], ict_raw[mb])[0, 1]) if mb.sum() >= 3 else np.nan
    # maxAB shows the template the statistic actually uses = the better-matching one.
    use_b = bool(np.nan_to_num(r_b) > np.nan_to_num(r_a))
    inter = rank_b01 if use_b else rank_a01
    which = "B" if use_b else "A"
    m = mb if use_b else ma
    max_ab = np.nanmax([r_a, r_b])
    # sign-orient seizure field to the shown interictal field ("same colour, same place = concordant")
    flip = bool(np.corrcoef(inter[m], ict_raw[m])[0, 1] < 0)
    ict = (1.0 - ict_raw) if flip else ict_raw

    sup = np.ones(len(names))
    soz_mask = np.array([n in r["soz"] for n in names])
    pad = 6.0
    xlim = (float(np.nanmin(along)) - pad, float(np.nanmax(along)) + pad)
    ylim = (float(np.nanmin(perp)) - pad, float(np.nanmax(perp)) + pad)
    sigma = _median_nn(np.column_stack([along, perp]))

    fig, ax = plt.subplots(1, 2, figsize=(14.0, 7.2), layout="constrained")
    _panel(ax[0], along, perp, inter, sup, soz_mask, names, xlim, ylim, sigma,
           f"interictal propagation order — template {which} (maxAB match)", "early (0) -> late (1)")
    _panel(ax[1], along, perp, ict, sup, soz_mask, names, xlim, ylim, sigma,
           "seizure-onset activation — broadband power, 0-10 s",
           "activation rank" + ("  [flipped to match axis]" if flip else ""))
    soz_h = Line2D([0], [0], marker="o", markerfacecolor="none", markeredgecolor="k",
                   linestyle="none", markersize=9, markeredgewidth=1.7, label="clinical SOZ")
    fig.legend(handles=[soz_h], loc="outside lower center", frameon=False, fontsize=12)
    fig.suptitle(
        f"{dataset} {subject} — field concordance on the D_AB gradient axis  "
        f"(interictal timing field vs seizure onset)\n"
        f"tier={r['tier']}, rho(A,B)={r['rho_AB']:+.2f}; statistic = maxAB = "
        f"max(|r_A|={r_a:.2f}, |r_B|={r_b:.2f}) = {max_ab:.2f}   "
        f"[axis = D_AB gradient; D_AB defines the axis, not the field]",
        fontsize=12.5)
    out = Path(out) if out else OUT / f"{dataset}_{subject}_{pool}_field_concordance_dab_axis.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out.name}  maxAB={max_ab:.3f} (r_A={r_a:.3f} r_B={r_b:.3f}) flip={flip}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="epilepsiae")
    ap.add_argument("--subject", default="139")
    ap.add_argument("--pool", default="broad", choices=["narrow", "broad"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    make(a.dataset, a.subject, a.pool, a.out)


if __name__ == "__main__":
    main()
