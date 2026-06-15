#!/usr/bin/env python3
"""Topic 5 DIRECTIONAL A-line figure (clean subjects only).

The A-line cohort statistic is `|corr_pair_mirror_invariant|` — a y-mirror-invariant correlation
of the two smoothed scalar FIELDS (interictal typical-rank vs seizure activation). It is NOT a
gradient-direction comparison: it folds out the transverse component, so a high |r| does not
require the gradient directions to line up. The per-event rose can therefore disagree with |r|.

This figure shows the DIRECTIONAL view of the A-line, restricted to subjects where the direction
is reliably measured ("clean": >= 8 contacts AND seizure R_axial >= 0.6):
  - interictal axis = gradient of each template's typical-rank field (the A-line's interictal axis),
  - seizure axis    = axial mean of per-seizure activation-gradient directions (normalized to 0°),
  - Delta_axis      = axial angle between them (0° = directionally collinear, 90° = orthogonal).
A cohort scatter (Delta_axis vs field |r|) shows the directional view tracks the tested field stat
on clean subjects, while making the mirror-invariance caveat explicit.

Reuses helpers from plot_topic5_axis_direction_rose (loading, seizure angles, electrode type, the
A-line |r| lookup). Clean gate is the honest scope, not cherry-picking: non-clean subjects keep the
FIELD maps (fields/) as their A-line evidence.
"""
from __future__ import annotations

import argparse
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
from scripts.plot_topic5_axis_direction_rose import (_seizure_angles, _electrode_kind, _align_lookup,
                                                     CACHE_DIR, REAL_DIR, ACTIVATION_KEY,
                                                     A_COLOR, B_COLOR)
from src.topic5_axis_direction import (gradient_angle, axial_mean, axial_resultant_length,
                                       axial_distance, rotate_to_reference)
import json

OUT = _ROOT / "results/topic5_ictal_recruitment/axis_alignment/figures/aline_direction"
MIN_CONTACTS_CLEAN = 8     # too few contacts -> 2D gradient fit is noise
RAXIAL_MIN = 0.6           # seizure axis must be reasonably concentrated to be a meaningful target


def _template_axis(ds_sid, tid):
    """(typical-rank gradient angle, x, y, n_contacts) for one template record, or (None,...)."""
    f = REAL_DIR / f"{ds_sid}_{tid}.json"
    if not f.exists():
        return None, None, None, 0
    rec = json.loads(f.read_text())
    if not rec.get("channels"):
        return None, None, None, 0
    _attach_real_coords([rec])
    frame = _subject_display_frame([rec])
    if frame is None:
        return None, None, None, 0
    x, y = _display_points(rec, frame)
    tr = np.array([c.get("typical_rank", np.nan) for c in rec["channels"]], float)
    return gradient_angle(x, y, tr), x, y, len(rec["channels"])


def _subject_dir_aline(ds_sid, activation, align):
    """Compute the clean-gate + directional quantities for one subject (no plotting)."""
    ga, x, y, nC = _template_axis(ds_sid, "t_a")
    gb, *_ = _template_axis(ds_sid, "t_b")
    if x is None:
        return None
    sz = _seizure_angles(ds_sid, x, y, _names(ds_sid), activation)
    if sz.size == 0:
        return None
    ref = axial_mean(sz)
    r_ax = axial_resultant_length(sz)
    if not np.isfinite(ref):
        return None
    clean = (nC >= MIN_CONTACTS_CLEAN) and (r_ax >= RAXIAL_MIN)
    dA = np.degrees(axial_distance(ga, ref)) if ga is not None and np.isfinite(ga) else np.nan
    dB = np.degrees(axial_distance(gb, ref)) if gb is not None and np.isfinite(gb) else np.nan
    rec = (align or {}).get(ds_sid) or {}
    return {"ds_sid": ds_sid, "n_contacts": nC, "r_axial": float(r_ax), "clean": bool(clean),
            "seizure_ref": float(ref), "ga": ga, "gb": gb, "x": x, "y": y, "sz": sz,
            "dA": float(dA) if np.isfinite(dA) else None, "dB": float(dB) if np.isfinite(dB) else None,
            "field_r": rec.get("real_median_abs_corr"),
            "beats": [nm for nm, k in [("coarse", "pass_channel_null"),
                                       ("within-shaft", "pass_within_shaft_null"),
                                       ("activity", "pass_anchor_matched_null"),
                                       ("joint", "pass_joint_null")] if rec.get(k)]}


def _names(ds_sid):
    f = REAL_DIR / f"{ds_sid}_t_a.json"
    rec = json.loads(f.read_text())
    return [c["name"] for c in rec.get("channels", [])]


def _axis_spoke(ax, angle, ref, color, rmax, label):
    rot = float(np.mod(angle - ref, 2 * np.pi))
    for end in (rot, rot + np.pi):
        ax.plot([end, end], [0, rmax], color=color, lw=3.4, zorder=5, alpha=0.92)
    ax.plot([], [], color=color, lw=3.4, label=label)   # legend proxy (single entry)


def plot_subject(d, activation):
    if d is None or not d["clean"]:
        return None
    ds_sid = d["ds_sid"]
    ds, subj = ds_sid.split("_", 1)
    kind, detail = _electrode_kind(ds, subj, _names(ds_sid))
    pretty = ds_sid.replace("epilepsiae_", "E").replace("yuquan_", "Y-")
    ref = d["seizure_ref"]
    fig = plt.figure(figsize=(7.2, 7.6), constrained_layout=True)
    ax = fig.add_subplot(111, projection="polar")
    rmax = 1.0
    # seizure axis (axial mean) at 0/180 + faint per-seizure ticks
    for a in rotate_to_reference(d["sz"], ref):
        if np.isfinite(a):
            ax.plot([a, a], [0, rmax * 0.9], color="0.45", lw=1.0, alpha=0.5, zorder=2)
    ax.plot([0, 0], [0, rmax * 1.12], color="black", lw=3.2, zorder=4,
            label=f"seizure axis (axial mean of {d['sz'].size} sz; R_axial={d['r_axial']:.2f})")
    ax.plot([np.pi, np.pi], [0, rmax * 1.12], color="black", lw=3.2, zorder=4)
    if d["ga"] is not None and d["dA"] is not None:
        _axis_spoke(ax, d["ga"], ref, A_COLOR, rmax, f"interictal axis A  (Δ={d['dA']:.0f}° from seizure)")
    if d["gb"] is not None and d["dB"] is not None:
        _axis_spoke(ax, d["gb"], ref, B_COLOR, rmax, f"interictal axis B  (Δ={d['dB']:.0f}° from seizure)")
    ax.set_theta_zero_location("E"); ax.set_theta_direction(1)
    ax.set_yticklabels([]); ax.set_rlabel_position(112.5)
    fr = d["field_r"]
    aline = (f"A-line field |r| = {fr:.2f}" + (" · beats " + ", ".join(d["beats"]) if d["beats"] else "")) if fr is not None else ""
    ax.set_title(f"Patient {pretty} — {kind} ({detail})\n"
                 f"interictal axis vs seizure axis — directional view ({activation})"
                 + (f"\n{aline}" if aline else ""), fontsize=11.5, pad=16)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.20), ncol=1, frameon=False, fontsize=8.8)
    fig.text(0.5, -0.005, "interictal axis = gradient of template typical-rank field. Δ = its angle off "
             "the seizure axis (0°=collinear). NOTE: A-line |r| is a MIRROR-INVARIANT field correlation "
             "(folds out transverse) — |r| can exceed what Δ alone suggests.  Clean subjects only "
             "(≥8 contacts, seizure R_axial≥0.6).", ha="center", fontsize=7.2, color="0.4", wrap=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{ds_sid}_aline_direction_{activation}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_cohort_scatter(rows, activation):
    clean = [d for d in rows if d and d["clean"] and d["dA"] is not None and d["field_r"] is not None]
    if len(clean) < 2:
        return None
    fig, ax = plt.subplots(figsize=(6.6, 5.6), constrained_layout=True)
    xs = [d["field_r"] for d in clean]
    ys = [d["dA"] for d in clean]
    ax.scatter(xs, ys, s=70, c="#2c3e50", zorder=3)
    for d in clean:
        ax.annotate(d["ds_sid"].replace("epilepsiae_", "E").replace("yuquan_", "Y-"),
                    (d["field_r"], d["dA"]), fontsize=7.5, xytext=(4, 3), textcoords="offset points")
    if len(clean) >= 3:
        r = np.corrcoef(xs, ys)[0, 1]
        ax.set_title(f"Directional vs field alignment — clean subjects ({activation})\n"
                     f"interictal-axis-vs-seizure-axis angle Δ vs A-line field |r|  (Pearson r={r:.2f}, n={len(clean)})",
                     fontsize=11)
    ax.set_xlabel("A-line field |r| (mirror-invariant, the tested stat)")
    ax.set_ylabel("Δ interictal-axis vs seizure-axis (°, directional)")
    ax.axhline(45, color="0.7", ls="--", lw=1)
    ax.set_ylim(0, 92); ax.grid(alpha=0.25)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"cohort_aline_direction_scatter_{activation}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activation", choices=list(ACTIVATION_KEY), default="broadband")
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()
    align = _align_lookup(args.activation)
    subs = args.subjects or sorted(p.stem for p in CACHE_DIR.glob("*.npz"))
    rows = []
    for sid in subs:
        try:
            d = _subject_dir_aline(sid, args.activation, align)
        except Exception as e:
            print(f"  skip {sid}: {type(e).__name__} {e}", flush=True)
            continue
        rows.append(d)
        if d and d["clean"]:
            p = plot_subject(d, args.activation)
            print(f"  wrote {p.name}  (Δa={d['dA']}, |r|={d['field_r']})" if p else f"  {sid} clean but no plot",
                  flush=True)
        elif d:
            print(f"  skip {sid}: not clean (nC={d['n_contacts']}, R_axial={d['r_axial']:.2f})", flush=True)
    sc = plot_cohort_scatter(rows, args.activation)
    if sc:
        print(f"  wrote {sc.name}  [cohort scatter]", flush=True)


if __name__ == "__main__":
    main()
