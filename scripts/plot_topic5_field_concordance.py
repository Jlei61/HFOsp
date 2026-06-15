#!/usr/bin/env python3
"""Topic 5 A-line FIELD CONCORDANCE figures (subject-level, NOT cohort average).

Cohort-level here means: lay out EVERY patient's paired field side by side, each judged against
its OWN channel-shuffle null — never average fields across subjects (different montage / axis /
focus per subject; an average field is a brain that does not exist).

  field_concordance_atlas_broadband.png   -- Panel A method schematic + Panel B 18-subject
      paired-field atlas (left = interictal propagation-order field, right = seizure-onset
      broadband activation field, same colormap, seizure shown in its best sign/mirror so
      "same colour in the same place = aligned"); sorted by margin |r| - Q95(null); tile border
      dark = beats the subject's coarse channel-shuffle null, grey = not.
  field_concordance_null_forest_broadband.png -- Panel C: per subject, the channel-shuffle null
      distribution (grey) + observed |r| (black dot) + null 95th pct (line). Same sort order.

Primary scope only: Epilepsiae 18, template A, broadband 0-10 s, sign-free/mirror-invariant field
alignment, coarse channel null. HFA + the 4-null ladder stay in the supplement.

Reuses the A-line field construction (R_smooth_rank + corr_pair_mirror_invariant_signed) and the
mm-plane display smoothing from the per-subject field figure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.plot_contact_plane_static import (_attach_real_coords, _subject_display_frame,
                                               _display_points, _smooth_rank_field_mm)
from scripts.plot_topic5_axis_alignment_fields import _ictal_activation, _rank01
from src.topic5_axis_alignment import matched_channels, make_field_record
from src.propagation_contact_plane_readout import (make_plane_grid, R_smooth_rank,
                                                   corr_pair_mirror_invariant_signed, S_THRESH, OVERLAP_MIN)
from run_topic5_axis_alignment import _abs_corr
from src.topic5_axis_alignment import channel_shuffle

REAL_DIR = _ROOT / "results/spatial_modulation/propagation_geometry/observation_readout/real_subjects"
CACHE_DIR = _ROOT / "results/topic5_ictal_recruitment/t0_feature_cache"
ALIGN_DIR = _ROOT / "results/topic5_ictal_recruitment/axis_alignment"
OUT = ALIGN_DIR / "figures/field_concordance"
ACTIVATION_KEY = {"broadband": "bb_auc", "hfa": "hfa_auc"}


def _subject_data(ds_sid, activation):
    """Per-subject paired fields + orientation + margin, or None if not usable."""
    axis_f = REAL_DIR / f"{ds_sid}_t_a.json"
    npz = CACHE_DIR / f"{ds_sid}.npz"
    if not axis_f.exists() or not npz.exists():
        return None
    axis = json.loads(axis_f.read_text())
    if not axis.get("channels"):
        return None
    cache = np.load(npz, allow_pickle=True)
    cache_names = [str(x) for x in cache["channels"]]
    matched = matched_channels(axis, {n: 0.0 for n in cache_names})
    if len(matched) < 6:
        return None
    names = [c["name"] for c in matched]
    act = _ictal_activation(ds_sid, ACTIVATION_KEY[activation])
    if not act:
        return None
    inter = np.array([float(c["typical_rank"]) for c in matched], float)
    ict = _rank01([act.get(n, np.nan) for n in names])
    support = np.array([float(c.get("support", 1.0)) for c in matched], float)
    soz = np.array([bool(c.get("is_soz")) for c in matched])

    # orientation decision on the A-line's own smoothed fields (R_smooth_rank), so the displayed
    # mirror/sign matches the |r| the cohort stat reports
    X, Y = make_plane_grid()
    F_inter = R_smooth_rank(make_field_record(matched, list(inter)), X, Y, None, S_THRESH)
    sigma = F_inter["sigma_xy"]
    F_ict = R_smooth_rank(make_field_record(matched, list(ict)), X, Y, sigma, S_THRESH)
    o = corr_pair_mirror_invariant_signed(F_inter["T"], F_inter["S"], F_ict["T"], F_ict["S"],
                                          S_THRESH, OVERLAP_MIN)
    mirror = (o.get("mirror_choice") == "mirror")
    signed = o.get("signed_corr")
    sign_neg = (signed is not None and signed < 0)

    # display frame (mm) for the nice viridis panels
    rec = json.loads(axis_f.read_text())
    _attach_real_coords([rec])
    frame = _subject_display_frame([rec])
    xs_all, ys_all = _display_points(rec, frame)
    pos = {c["name"]: i for i, c in enumerate(rec["channels"])}
    idx = np.array([pos[n] for n in names])
    xs, ys = xs_all[idx], ys_all[idx]

    al = _align_lookup(activation).get(ds_sid, {})
    r = al.get("real_median_abs_corr")
    p95 = al.get("channel_null_p95")
    passed = bool(al.get("pass_channel_null"))
    margin = (r - p95) if (r is not None and p95 is not None) else -np.inf
    return {"ds_sid": ds_sid, "xs": xs, "ys": ys, "inter": inter, "ict": ict, "support": support,
            "soz": soz, "xlim": frame["xlim"], "ylim": frame["ylim"], "sigma": frame["sigma_mm"],
            "mirror": mirror, "sign_neg": sign_neg, "r": r, "p95": p95, "passed": passed,
            "margin": float(margin), "n_ch": len(matched)}


def _align_lookup(activation):
    f = ALIGN_DIR / f"axis_alignment_{activation}_B1000.json"
    d = json.loads(f.read_text())
    return {r["subject_id"]: r for r in d.get("per_subject", []) if r.get("status") == "ok"}


def _field_panel(ax, xs, ys, vals, support, xlim, ylim, sigma, soz):
    X, Y, T, _, _ = _smooth_rank_field_mm(xs, ys, vals, support, xlim, ylim, sigma)
    ax.imshow(T, origin="lower", extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
              aspect="equal", cmap="viridis", vmin=0, vmax=1)
    ax.scatter(xs, ys, c=vals, cmap="viridis", vmin=0, vmax=1, s=14, zorder=3,
               edgecolors=["k" if z else "white" for z in soz],
               linewidths=[1.0 if z else 0.3 for z in soz])
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def plot_atlas(rows, activation, ncols=6):
    rows = [r for r in rows if r is not None]
    rows.sort(key=lambda d: -d["margin"])
    n = len(rows)
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(ncols * 2.7, 2.0 + nrows * 2.35))
    # Panel A schematic strip on top
    gs = fig.add_gridspec(nrows + 1, ncols * 2, height_ratios=[0.5] + [1] * nrows,
                          hspace=0.62, wspace=0.05, left=0.012, right=0.905, top=0.965, bottom=0.03)
    axA = fig.add_subplot(gs[0, :]); axA.axis("off")
    axA.text(0.5, 0.80, "A-line field concordance across subjects", ha="center", va="center",
             fontsize=17, fontweight="bold", transform=axA.transAxes)
    axA.text(0.5, 0.40, "per subject:  interictal template-rank field   vs   seizure-onset broadband "
             "activation field          r$_s$ = | corr$_{mirror}$( F$_{interictal}$ , F$_{seizure}$ ) |",
             ha="center", va="center", fontsize=10.5, transform=axA.transAxes, color="0.2")
    axA.text(0.5, 0.10, "same colour in the same place (after sign alignment) = same spatial gradient "
             "— this is field similarity, not direction and not replay",
             ha="center", va="center", fontsize=9.2, transform=axA.transAxes, color="0.4", style="italic")
    last_im = None
    for k, d in enumerate(rows):
        rr, cc = k // ncols, k % ncols
        axL = fig.add_subplot(gs[rr + 1, 2 * cc])
        axR = fig.add_subplot(gs[rr + 1, 2 * cc + 1])
        _field_panel(axL, d["xs"], d["ys"], d["inter"], d["support"], d["xlim"], d["ylim"], d["sigma"], d["soz"])
        ys_s = -d["ys"] if d["mirror"] else d["ys"]
        vals_s = (1.0 - d["ict"]) if d["sign_neg"] else d["ict"]
        _field_panel(axR, d["xs"], ys_s, vals_s, d["support"], d["xlim"], d["ylim"], d["sigma"], d["soz"])
        last_im = axR.images[-1]
        # tile border = passes coarse null?
        edge = "#1a1a1a" if d["passed"] else "#b8b8b8"
        lw = 2.4 if d["passed"] else 1.0
        bb = axL.get_position(); bb2 = axR.get_position()
        x0, y0 = bb.x0 - 0.004, bb.y0 - 0.004
        x1, y1 = bb2.x1 + 0.004, bb.y1 + 0.004
        fig.add_artist(Rectangle((x0, y0), x1 - x0, y1 - y0, transform=fig.transFigure,
                                 fill=False, edgecolor=edge, lw=lw, zorder=10))
        pretty = d["ds_sid"].replace("epilepsiae_", "E")
        over = f"+{d['margin']:.2f}" if d["passed"] else "n.s."
        fig.text((x0 + x1) / 2, y1 + 0.0015, f"{pretty}   |r| {d['r']:.2f}   {over}",
                 ha="center", va="bottom", fontsize=9.0,
                 fontweight=("bold" if d["passed"] else "normal"), color=("#1a1a1a" if d["passed"] else "0.5"))
    if last_im is not None:
        cax = fig.add_axes([0.915, 0.06, 0.014, 0.5])
        cb = fig.colorbar(last_im, cax=cax)
        cb.set_label("early / low  ->  late / high  (rank-normalized)", fontsize=8.5)
        cb.set_ticks([0, 1]); cb.set_ticklabels(["0", "1"])
    fig.text(0.46, 0.005, "left = interictal propagation order   |   right = seizure-onset activation "
             "(best sign/mirror)   ·   black ring = clinical SOZ contact   ·   dark frame = beats the "
             "subject's own channel-shuffle null   ·   sorted by margin   ·   Epilepsiae, template A, "
             f"{activation} 0-10 s", ha="center", fontsize=7.6, color="0.4")
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"field_concordance_atlas_{activation}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out, len(rows)


def _channel_null_forest(ds_sid, activation, B, rng):
    """Recompute the subject's channel-shuffle null DISTRIBUTION + observed |r| (the JSON keeps only
    median+p95). Mirrors run_topic5_axis_alignment._subject's channel-null path exactly."""
    axis = json.loads((REAL_DIR / f"{ds_sid}_t_a.json").read_text())
    data = np.load(CACHE_DIR / f"{ds_sid}.npz", allow_pickle=True)
    meta = json.loads((CACHE_DIR / f"{ds_sid}.json").read_text())
    cache_names = [str(x) for x in data["channels"]]
    cidx = {n: i for i, n in enumerate(cache_names)}
    matched = matched_channels(axis, {n: 0.0 for n in cache_names})
    names_m = [c["name"] for c in matched]
    m_in_cache = np.array([cidx[n] for n in names_m])
    X, Y = make_plane_grid()
    F_inter = R_smooth_rank(make_field_record(matched, [float(c["typical_rank"]) for c in matched]),
                            X, Y, None, S_THRESH)
    sigma = F_inter["sigma_xy"]
    fld = lambda v: R_smooth_rank(make_field_record(matched, v), X, Y, sigma, S_THRESH)
    key = ACTIVATION_KEY[activation]
    real, draws = [], []
    for idx in meta["eligible_idxs"]:
        k = f"{key}__{idx}"
        if k not in data.files:
            continue
        iv = data[k][m_in_cache].astype(float)
        if np.isfinite(iv).sum() < 6:
            continue
        r = _abs_corr(F_inter, fld(iv))
        if not np.isfinite(r):
            continue
        real.append(r)
        draws.append([_abs_corr(F_inter, fld(channel_shuffle(iv, rng))) for _ in range(B)])
    if not real:
        return None
    return float(np.median(real)), np.nanmedian(np.asarray(draws, float), axis=0)  # observed, [B]


def plot_forest(rows, activation, B=500, seed=20260615):
    rows = [r for r in rows if r is not None]
    rows.sort(key=lambda d: -d["margin"])
    rng = np.random.default_rng(seed)
    data = []
    for d in rows:
        nd = _channel_null_forest(d["ds_sid"], activation, B, rng)
        if nd is None:
            continue
        obs, null = nd
        data.append((d, obs, null))
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(data) + 1.4), constrained_layout=True)
    for i, (d, obs, null) in enumerate(data):
        y = len(data) - 1 - i
        nn = null[np.isfinite(null)]
        ax.scatter(nn, np.full(nn.size, y), s=5, color="0.80", alpha=0.45, zorder=1)
        # authoritative observed / 95th / verdict come from the cohort JSON (B=1000), so the forest
        # matches the atlas borders and the FINAL table; the grey cloud is the recomputed shape only.
        p95, robs, passed = d["p95"], d["r"], d["passed"]
        ax.plot([p95, p95], [y - 0.32, y + 0.32], color="0.4", lw=1.5, zorder=2)
        ax.scatter([robs], [y], s=48, color=("#1a1a1a" if passed else "#c0392b"), zorder=3)
        ax.text(-0.02, y, d["ds_sid"].replace("epilepsiae_", "E"), ha="right", va="center",
                fontsize=8.2, transform=ax.get_yaxis_transform())
    ax.set_yticks([]); ax.set_ylim(-0.7, len(data) - 0.3)
    ax.set_xlabel("mirror-invariant field alignment  |r|", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_title("Epilepsiae subjects · broadband 0-10 s · template A\n"
                 "observed |r| (●) vs each subject's own channel-shuffle null (grey) + null 95th pct (│)",
                 fontsize=10.5)
    ax.text(0.99, 0.01, "black ● = beats own null 95%   ·   red ● = does not   ·   sorted by margin",
            ha="right", va="bottom", transform=ax.transAxes, fontsize=7.8, color="0.4")
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"field_concordance_null_forest_{activation}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out, len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activation", default="broadband", choices=list(ACTIVATION_KEY))
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--atlas-only", action="store_true")
    args = ap.parse_args()
    subs = args.subjects or [p.stem for p in sorted(REAL_DIR.glob("epilepsiae_*_t_a.json"))]
    subs = [s.replace("_t_a", "") for s in subs]
    rows = []
    for sid in subs:
        if not sid.startswith("epilepsiae_"):
            continue
        d = _subject_data(sid, args.activation)
        if d:
            rows.append(d)
    out, n = plot_atlas(rows, args.activation)
    print(f"wrote {out.name}  ({n} subjects)", flush=True)
    if not args.atlas_only:
        fout, fn = plot_forest(rows, args.activation)
        print(f"wrote {fout.name}  ({fn} subjects)", flush=True)


if __name__ == "__main__":
    main()
