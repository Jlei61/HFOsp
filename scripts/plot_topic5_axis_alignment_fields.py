#!/usr/bin/env python3
"""Topic 5 A-line field figures — does the interictal propagation AXIS line up with the
SEIZURE-ONSET activation gradient? Per subject, two panels on the SAME subject-fixed mm plane:
  left  = interictal propagation order (early -> late)   [the axis]
  right = seizure-onset activation (low -> high, mean broadband 0-10 s) [the ictal gradient]
If the two spatial gradients are collinear, the early-interictal end coincides with one end of
the activation gradient (sign-free). The mirror-invariant |alignment| in the title quantifies it.

Reuses the mature contact-plane machinery (subject-fixed display frame + mm field smoothing)
from scripts.plot_contact_plane_static; the interictal record + ictal feature cache supply the
two scalars. Paper-grade: scientific labels only (no internal field names), tight equal axes.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.plot_contact_plane_static import (_subject_display_frame, _display_points,
                                               _smooth_rank_field_mm, _attach_real_coords)

REAL_DIR = _ROOT / "results/spatial_modulation/propagation_geometry/observation_readout/real_subjects"
CACHE_DIR = _ROOT / "results/topic5_ictal_recruitment/t0_feature_cache"
ALIGN_DIR = _ROOT / "results/topic5_ictal_recruitment/axis_alignment"
OUT = ALIGN_DIR / "figures/fields"
ACTIVATION_KEY = {"broadband": "bb_auc", "hfa": "hfa_auc", "ramp": "ramp", "ei": "ei_like"}
ACTIVATION_LABEL = {"broadband": "broadband power, 0-10 s", "hfa": "fast activity 60-100 Hz, 0-10 s",
                    "ramp": "activation ramp slope, 0-10 s", "ei": "EI-like (fast-activity / delay)"}


def _ictal_activation(ds_sid, key="bb_auc"):
    """Per-contact mean early-ictal activation across the subject's eligible seizures -> {name: z}."""
    npz, mj = CACHE_DIR / f"{ds_sid}.npz", CACHE_DIR / f"{ds_sid}.json"
    if not npz.exists():
        return {}
    data = np.load(npz, allow_pickle=True)
    meta = json.load(open(mj))
    names = [str(x) for x in data["channels"]]
    arrs = [data[f"{key}__{i}"] for i in meta["eligible_idxs"] if f"{key}__{i}" in data.files]
    if not arrs:
        return {}
    mean_act = np.nanmean(np.vstack([np.asarray(a, float) for a in arrs]), axis=0)
    return {n: float(v) for n, v in zip(names, mean_act)}


def _rank01(vals):
    """Min-max-ish display normalization to [0,1] by rank (robust, monotone)."""
    v = np.asarray(vals, float)
    out = np.full(v.shape, np.nan)
    ok = np.isfinite(v)
    if ok.sum() >= 2:
        order = np.argsort(np.argsort(v[ok]))
        out[ok] = order / (ok.sum() - 1)
    elif ok.sum() == 1:
        out[ok] = 0.5
    return out


def _align_lookup(activation):
    f = ALIGN_DIR / f"axis_alignment_{activation}_B1000.json"
    if not f.exists():
        return {}
    d = json.load(open(f))
    return {r["subject_id"]: r for r in d.get("per_subject", []) if r.get("status") == "ok"}


def _verdict(rec):
    """Plain-language null-pass summary for the caption."""
    if rec is None:
        return ""
    beat = [name for name, k in [("coarse", "pass_channel_null"), ("within-shaft", "pass_within_shaft_null"),
                                 ("activity", "pass_anchor_matched_null"), ("joint", "pass_joint_null")]
            if rec.get(k)]
    return ("beats " + ", ".join(beat)) if beat else "does not beat the coarse null"


def _panel(ax, xs, ys, vals, support, xlim, ylim, sigma, cmap, title, cbar_label, soz):
    X, Y, T, _, _ = _smooth_rank_field_mm(xs, ys, vals, support, xlim, ylim, sigma)
    im = ax.imshow(T, origin="lower", extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                   aspect="equal", cmap=cmap, vmin=0, vmax=1)
    ax.scatter(xs, ys, c=vals, cmap=cmap, vmin=0, vmax=1, s=70, zorder=3,
               edgecolors=["k" if z else "white" for z in soz],
               linewidths=[1.6 if z else 0.5 for z in soz])
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("along propagation axis (mm)")
    ax.set_ylabel("transverse (mm)")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(cbar_label, fontsize=9)


def plot_subject(ds_sid, align, activation):
    rec_f = REAL_DIR / f"{ds_sid}_t_a.json"
    if not rec_f.exists():
        return None
    rec = json.loads(rec_f.read_text())
    if not rec.get("channels"):
        return None
    _attach_real_coords([rec])
    frame = _subject_display_frame([rec])
    if frame is None:
        return None
    xs, ys = _display_points(rec, frame)
    names = [c["name"] for c in rec["channels"]]
    support = np.array([c["support"] for c in rec["channels"]], float)
    soz = np.array([bool(c.get("is_soz")) for c in rec["channels"]])
    inter = np.array([c["typical_rank"] for c in rec["channels"]], float)         # 0..1 already
    act = _ictal_activation(ds_sid, ACTIVATION_KEY[activation])
    if not act:                       # axis record exists but no eligible ictal seizures -> skip
        return None
    ict_raw = _rank01([act.get(n, np.nan) for n in names])
    # SAME colormap for both panels so "same colour in the same place = consistent" is readable.
    # The statistic is sign-free (a reverse-collinear subject is still aligned), so orient the
    # ictal field to the axis sign — flip when negatively correlated — and say so in the colorbar.
    m = np.isfinite(inter) & np.isfinite(ict_raw)
    flip = bool(m.sum() >= 3 and np.corrcoef(inter[m], ict_raw[m])[0, 1] < 0)
    ict = (1.0 - ict_raw) if flip else ict_raw
    ict_lbl = ("activation high (0) -> low (1)  [flipped to match axis]" if flip
               else "activation low (0) -> high (1)")
    ict_title = f"seizure-onset activation — {ACTIVATION_LABEL[activation]}"
    xlim, ylim, sigma = frame["xlim"], frame["ylim"], frame["sigma_mm"]

    ar = align.get(ds_sid)
    rval = ar.get("real_median_abs_corr") if ar else None
    pretty = ds_sid.replace("epilepsiae_", "E").replace("yuquan_", "Y-")
    fig, ax = plt.subplots(1, 2, figsize=(13.0, 6.0), constrained_layout=True)
    _panel(ax[0], xs, ys, inter, support, xlim, ylim, sigma, "viridis",
           "interictal propagation order — template A", "early (0) -> late (1)", soz)
    _panel(ax[1], xs, ys, ict, support, xlim, ylim, sigma, "viridis",
           ict_title, ict_lbl, soz)
    l1 = f"Patient {pretty} — interictal propagation axis vs seizure-onset activation"
    l2 = (f"mirror-invariant axis alignment |r| = {rval:.2f}   ·   {_verdict(ar)}"
          if rval is not None else "")
    fig.suptitle(l1 + ("\n" + l2 if l2 else ""), fontsize=11.5)
    fig.text(0.5, 0.004, "black ring = clinical seizure-onset contact (overlay only, not used in scoring)",
             ha="center", fontsize=8.5, color="0.35")
    OUT.mkdir(parents=True, exist_ok=True)
    out_png = OUT / f"{ds_sid}_axis_vs_{activation}.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--activation", choices=list(ACTIVATION_KEY), default="broadband")
    args = ap.parse_args()
    align = _align_lookup(args.activation)
    all_subs = sorted(p.stem.replace("_t_a", "") for p in REAL_DIR.glob("*_t_a.json"))
    subs = args.subjects or [s for s in all_subs if (CACHE_DIR / f"{s}.npz").exists()]
    for ds_sid in subs:
        p = plot_subject(ds_sid, align, args.activation)
        if p:
            print(f"  wrote {p.name}", flush=True)


if __name__ == "__main__":
    main()
