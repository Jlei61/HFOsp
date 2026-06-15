#!/usr/bin/env python3
"""Topic 5 A-line DIRECTION ROSE (per subject + cohort-pooled).

Reference-frame convention (user-locked 2026-06-15):
  - BLACK = the subject's seizure early-activation gradient direction, NORMALIZED to 0 deg
    (bold line = circular mean over eligible seizures; faint ticks = individual seizures).
  - Two HOLLOW histograms (full 360 deg) = the propagation directions of the individual
    interictal group EVENTS, split by which template (A / B) each event belongs to.
Two templates that are opposite ends of ONE bidirectional axis appear as two lobes ~180 deg
apart; if the axis is collinear with the seizure axis, one lobe sits near 0 deg and the other
near 180 deg. This is the directional, per-event companion to the sign-free |corr| A-line
statistic (a 180 deg lobe is reverse-collinear = still 'aligned' under that statistic).

ECoG grids give a physically planar 2D direction; SEEG depth subjects are shown with a
projected-plane caveat (electrode type read from Epilepsiae SQL electrode_array.type).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.plot_contact_plane_static import (_attach_real_coords, _display_points,
                                               _subject_display_frame)
from scripts.run_contact_plane_readout import _subject_dir, _load_accepted_templates
from src.interictal_propagation import load_subject_propagation_events
from src.lagpat_rank_audit import mask_phantom_ranks
from src import propagation_skeleton_geometry as G
from src.topic5_axis_direction import (event_angles_by_template, resultant_length,
                                       rotate_to_reference, gradient_angle,
                                       axial_mean, axial_resultant_length, axial_distance)

REAL_DIR = _ROOT / "results/spatial_modulation/propagation_geometry/observation_readout/real_subjects"
CACHE_DIR = _ROOT / "results/topic5_ictal_recruitment/t0_feature_cache"
OUT = _ROOT / "results/topic5_ictal_recruitment/axis_alignment/figures/rose"
SQL_DIR = Path("/mnt/epilepsia_data/all_data_sqls")
ALIGN_DIR = _ROOT / "results/topic5_ictal_recruitment/axis_alignment"
ACTIVATION_KEY = {"broadband": "bb_auc", "hfa": "hfa_auc", "ramp": "ramp", "ei": "ei_like"}
A_COLOR, B_COLOR = "#1f77b4", "#d95f02"


def _align_lookup(activation):
    """A-line cohort statistic per subject (template-A field |corr_pair_mirror_invariant| vs the 4
    nulls) — the test that already established interictal-axis vs seizure-activation SIGNIFICANCE."""
    f = ALIGN_DIR / f"axis_alignment_{activation}_B1000.json"
    if not f.exists():
        return {}
    d = json.load(open(f))
    return {r["subject_id"]: r for r in d.get("per_subject", []) if r.get("status") == "ok"}


def _aline_caption(rec):
    if not rec or rec.get("real_median_abs_corr") is None:
        return ""
    beats = [nm for nm, k in [("coarse", "pass_channel_null"), ("within-shaft", "pass_within_shaft_null"),
                              ("activity", "pass_anchor_matched_null"), ("joint", "pass_joint_null")]
             if rec.get(k)]
    tail = (" · beats " + ", ".join(beats)) if beats else " · beats no null"
    return f"A-line field alignment |r| = {rec['real_median_abs_corr']:.2f}{tail}"
MAX_EVENTS = 6000          # subjects have 90k-240k events; a few thousand fixes the direction
EVENT_SEED = 20260615      # histogram exactly. Seeded subsample keeps it reproducible + fast.


# ----- electrode type (ECoG grid/strip vs SEEG depth) from Epilepsiae SQL metadata -----
def _epilepsiae_electrode_types(subj, names):
    """{channel_name: 'grid'|'strip'|'depth'|...} from electrode_array.type in the subject SQL."""
    sql = next(iter(SQL_DIR.glob(f"pat_{subj}02_*.sql")), None)
    if sql is None:
        return {}
    txt = sql.read_text(errors="ignore")
    arr_type = {aid: typ for aid, _nm, typ in
                re.findall(r"electrode_array \([^)]*\) VALUES \((\d+), \d+, '([^']+)', '([^']+)'", txt)}
    name_type = {}
    for eid, aid, enm in re.findall(r"INSERT INTO electrode \([^)]*\) VALUES \((\d+), (\d+), '([^']+)'", txt):
        if aid in arr_type:
            name_type[enm] = arr_type[aid]
    return {n: name_type.get(n.split("-")[0], "?") for n in names}


def _electrode_kind(ds, subj, names):
    """('ECoG'|'SEEG', detail) for the title/caveat, from real metadata where available."""
    if ds == "yuquan":
        return "SEEG", "depth (Yuquan SEEG)"
    types = _epilepsiae_electrode_types(subj, names)
    vals = [t for t in types.values() if t not in ("?", None)]
    if not vals:                                     # metadata miss -> naming fallback
        gfrac = np.mean([n[:1] == "G" for n in names])
        return ("ECoG", "grid (by naming)") if gfrac >= 0.6 else ("SEEG", "depth (by naming)")
    grid_frac = np.mean([t in ("grid", "strip") for t in vals])
    if grid_frac >= 0.6:
        return "ECoG", f"grid {grid_frac:.0%}"
    return "SEEG", f"depth {1 - grid_frac:.0%}"


# ----- per-event interictal directions split by template -----
def _interictal_event_vals(ds, subj, names_rec):
    """(n_rec, n_ev) per-event masked rank aligned to record channels + (n_ev,) template labels."""
    ev = load_subject_propagation_events(_subject_dir(ds, subj))
    full_names = list(ev["channel_names"])
    if not full_names or np.asarray(ev["ranks"]).size == 0:
        return None, None, None
    ranks = np.asarray(ev["ranks"], float)
    bools = np.asarray(ev["bools"]) > 0
    masked_full = mask_phantom_ranks(ranks, bools, normalize=True)       # (n_full, n_ev)
    if masked_full.shape[1] > MAX_EVENTS:                                 # seeded subsample
        sel = np.sort(np.random.default_rng(EVENT_SEED).choice(masked_full.shape[1],
                                                               MAX_EVENTS, replace=False))
        masked_full = masked_full[:, sel]
    ta, tb, swap = _load_accepted_templates(ds, subj, full_names)
    labels = G.assign_events_to_templates(masked_full, ta, tb)           # 0/1/-1
    fidx = {n: i for i, n in enumerate(full_names)}
    rows = [masked_full[fidx[n]] if n in fidx else np.full(masked_full.shape[1], np.nan)
            for n in names_rec]
    return np.vstack(rows), labels, swap


def _seizure_angles(ds_sid, x, y, names_rec, activation):
    npz = CACHE_DIR / f"{ds_sid}.npz"
    mj = CACHE_DIR / f"{ds_sid}.json"
    if not npz.exists():
        return np.array([])
    data = np.load(npz, allow_pickle=True)
    meta = json.load(open(mj))
    cidx = {str(n): i for i, n in enumerate(data["channels"])}
    key = ACTIVATION_KEY[activation]
    angs = []
    for sz in meta.get("eligible_idxs", []):
        k = f"{key}__{sz}"
        if k not in data.files:
            continue
        arr = np.asarray(data[k], float)
        vals = np.array([arr[cidx[n]] if n in cidx else np.nan for n in names_rec])
        a = gradient_angle(x, y, vals)
        if np.isfinite(a):
            angs.append(a)
    return np.asarray(angs, float)


def _load_frame(ds_sid):
    f = REAL_DIR / f"{ds_sid}_t_a.json"
    if not f.exists():
        return None
    rec = json.loads(f.read_text())
    if not rec.get("channels"):
        return None
    _attach_real_coords([rec])
    frame = _subject_display_frame([rec])
    if frame is None:
        return None
    x, y = _display_points(rec, frame)
    return rec, x, y, [c["name"] for c in rec["channels"]]


def _rose_axes(ax, grp, ref, sz_angles, bins):
    edges = np.linspace(0, 2 * np.pi, bins + 1)
    centers = edges[:-1] + (edges[1] - edges[0]) / 2
    width = (edges[1] - edges[0]) * 0.95
    rmax = 1
    for lbl, color, name in [(0, A_COLOR, "template A"), (1, B_COLOR, "template B")]:
        a = rotate_to_reference(grp.get(lbl, np.array([])), ref)
        a = a[np.isfinite(a)]
        if a.size == 0:
            continue
        counts, _ = np.histogram(a, bins=edges)
        rmax = max(rmax, int(counts.max()))
        R = resultant_length(a)
        ax.bar(centers, counts, width=width, facecolor="none", edgecolor=color,
               linewidth=2.0, alpha=0.95, label=f"{name}  (n={a.size}, R={R:.2f})")
    # seizure reference = AXIAL mean (sign-free; stable under a 0/180 bimodal seizure set, where a
    # plain circular mean would cancel). Black line drawn BOTH ways = the bidirectional axis at 0/180.
    sz_rot = rotate_to_reference(sz_angles, ref)
    for a in sz_rot[np.isfinite(sz_rot)]:
        ax.plot([a, a], [0, rmax * 0.92], color="0.45", lw=1.0, alpha=0.55, zorder=2)
    n_sz = int(np.isfinite(sz_angles).sum())
    r_ax, r_dir = axial_resultant_length(sz_angles), resultant_length(sz_angles)
    ax.plot([0, 0], [0, rmax * 1.12], color="black", lw=3.2, zorder=4,
            label=f"seizure axis (axial mean of {n_sz} sz; R_axial={r_ax:.2f}, dir-bias R={r_dir:.2f})")
    ax.plot([np.pi, np.pi], [0, rmax * 1.12], color="black", lw=3.2, zorder=4)   # other end of axis
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rlabel_position(112.5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=False, fontsize=8.8)


def plot_subject(ds_sid, activation, align=None, bins=18):
    loaded = _load_frame(ds_sid)
    if loaded is None:
        return None
    rec, x, y, names_rec = loaded
    ds, subj = ds_sid.split("_", 1)
    try:
        event_vals, labels, swap = _interictal_event_vals(ds, subj, names_rec)
    except FileNotFoundError:
        return None
    if event_vals is None:
        return None
    grp = event_angles_by_template(event_vals, x, y, labels)
    sz_angles = _seizure_angles(ds_sid, x, y, names_rec, activation)
    if sz_angles.size == 0 or (grp[0].size + grp[1].size) == 0:
        return None
    ref = axial_mean(sz_angles)          # sign-free axial reference (P1: stable under 0/180 bimodal)
    if not np.isfinite(ref):
        return None
    kind, detail = _electrode_kind(ds, subj, names_rec)
    pretty = ds_sid.replace("epilepsiae_", "E").replace("yuquan_", "Y-")

    fig = plt.figure(figsize=(7.4, 7.6), constrained_layout=True)
    ax = fig.add_subplot(111, projection="polar")
    _rose_axes(ax, grp, ref, sz_angles, bins)
    fig.suptitle(f"Patient {pretty} — {kind} ({detail})\n"
                 f"per-event interictal directions per template vs seizure axis  ({activation})",
                 fontsize=12, y=1.0)
    cap = ("DIRECTIONAL view — TWO roles: (1) a supplement to the A-line (whose main stat is the "
           "mirror-invariant FIELD correlation; field maps are the primary A-line visual), and (2) the "
           "C-line's main figure (per-seizure directions by subtype). direction = scalar-field "
           "gradient, NOT a wavefront velocity; black = seizure axis (axial mean) 0°/180°; hollow bars "
           "= interictal event directions per template. Field & direction agree on clean subjects — see "
           "aline_direction/ for the clean-subject directional A-line.")
    if kind == "SEEG":
        cap = "SEEG: projected-plane estimate, read with care.  " + cap
    fig.text(0.5, -0.01, cap, ha="center", fontsize=7.6, color="0.35", wrap=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{ds_sid}_direction_rose_{activation}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out, kind


def plot_pooled(results, activation, bins=24):
    """Cohort-pooled rose: every subject rotated to its own seizure axis (0 deg), event
    directions pooled per template. Shows whether, across subjects, template directions
    concentrate near 0 deg (forward-collinear) / 180 deg (reverse-collinear)."""
    pooled = {0: [], 1: []}
    for ds_sid in results:
        loaded = _load_frame(ds_sid)
        if loaded is None:
            continue
        rec, x, y, names_rec = loaded
        ds, subj = ds_sid.split("_", 1)
        try:
            event_vals, labels, _ = _interictal_event_vals(ds, subj, names_rec)
        except FileNotFoundError:
            continue
        if event_vals is None:
            continue
        grp = event_angles_by_template(event_vals, x, y, labels)
        sz = _seizure_angles(ds_sid, x, y, names_rec, activation)
        ref = axial_mean(sz)             # same sign-free axial reference as the per-subject roses
        if not np.isfinite(ref):
            continue
        for lbl in (0, 1):
            pooled[lbl].append(rotate_to_reference(grp.get(lbl, np.array([])), ref))
    pooled = {k: (np.concatenate(v) if v else np.array([])) for k, v in pooled.items()}
    if pooled[0].size + pooled[1].size == 0:
        return None
    edges = np.linspace(0, 2 * np.pi, bins + 1)
    centers = edges[:-1] + (edges[1] - edges[0]) / 2
    width = (edges[1] - edges[0]) * 0.95
    fig = plt.figure(figsize=(7.2, 7.4), constrained_layout=True)
    ax = fig.add_subplot(111, projection="polar")
    rmax = 1
    for lbl, color, name in [(0, A_COLOR, "template A"), (1, B_COLOR, "template B")]:
        a = pooled[lbl][np.isfinite(pooled[lbl])]
        if a.size == 0:
            continue
        counts, _ = np.histogram(a, bins=edges)
        rmax = max(rmax, int(counts.max()))
        ax.bar(centers, counts, width=width, facecolor="none", edgecolor=color, linewidth=2.0,
               label=f"{name}  (n={a.size}, R={resultant_length(a):.2f})")
    ax.plot([0, 0], [0, rmax * 1.1], color="black", lw=3.2,
            label="seizure axis (axial mean, each subject set to 0°/180°)")
    ax.plot([np.pi, np.pi], [0, rmax * 1.1], color="black", lw=3.2)
    ax.set_theta_zero_location("E"); ax.set_theta_direction(1); ax.set_rlabel_position(112.5)
    ax.set_title(f"Cohort-pooled interictal event directions vs seizure axis — {activation}\n"
                 "each subject rotated so its seizure early-activation axis = 0°", fontsize=11.5, pad=18)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=1, frameon=False, fontsize=9)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"cohort_pooled_direction_rose_{activation}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--activation", choices=list(ACTIVATION_KEY), default="broadband")
    ap.add_argument("--bins", type=int, default=18)
    ap.add_argument("--pooled", action="store_true", help="also render the cohort-pooled rose")
    ap.add_argument("--pooled-only", action="store_true", help="render ONLY the cohort-pooled rose")
    args = ap.parse_args()
    align = _align_lookup(args.activation)
    all_subs = sorted(p.stem for p in CACHE_DIR.glob("*.npz"))
    subs = args.subjects or all_subs
    done = []
    if not args.pooled_only:
        for sid in subs:
            r = plot_subject(sid, args.activation, align=align, bins=args.bins)
            if r:
                out, kind = r
                done.append(sid)
                print(f"  wrote {out.name}  [{kind}]", flush=True)
            else:
                print(f"  skip {sid} (insufficient data)", flush=True)
    if args.pooled_only:
        p = plot_pooled(subs, args.activation)
        print(f"  wrote {p.name}  [cohort-pooled]" if p else "  pooled: insufficient data", flush=True)
    elif args.pooled and done:
        p = plot_pooled(done, args.activation)
        if p:
            print(f"  wrote {p.name}  [cohort-pooled]", flush=True)


if __name__ == "__main__":
    main()
