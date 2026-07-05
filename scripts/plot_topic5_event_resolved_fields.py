#!/usr/bin/env python3
"""Topic 5 event-resolved interictal FIELDS — the per-class version of the A-line left panel.

User pivot (2026-06-25): instead of a single subject-averaged interictal field, take ALL the
class-A interictal events, project them onto the contact plane (our 2D field projection) and
weight-normalize -> one class-A field; same for class-B; show the seizure-onset field for
comparison. Rendered in the form of the A-line field figure's LEFT panel
(results/topic5_ictal_recruitment/axis_alignment/figures/fields/*_axis_vs_broadband.png).

Per subject, 3 panels on ONE shared physical contact plane (the broad template-A axis frame):
  class-A interictal propagation field  |  class-B field  |  seizure-onset activation field
All viridis, contacts as dots, clinical seizure-onset contacts ringed black (overlay only).
Broad substrate (20 contacts). The per-class field value at each contact = mean of that
contact's normalized propagation order over that class's events (weight = participation fraction);
display rank-normalized to early(0)->late(1).
"""
from __future__ import annotations

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
import src.topic5_event_resolved_alignment as erm
from scripts.plot_contact_plane_static import (_subject_display_frame, _display_points,
                                               _smooth_rank_field_mm, _attach_real_coords)

GEOM_BROAD = _ROOT / "results/spatial_modulation/propagation_geometry_broad/observation_readout/real_subjects"
ICTAL_CACHE = _ROOT / "results/topic5_ictal_recruitment/t0_feature_cache"
OUT = _ROOT / "results/topic5_ictal_recruitment/event_resolved_alignment/figures/fields"


def _rank01(vals):
    v = np.asarray(vals, float); out = np.full(v.shape, np.nan); ok = np.isfinite(v)
    if ok.sum() >= 2:
        out[ok] = np.argsort(np.argsort(v[ok])) / (ok.sum() - 1)
    elif ok.sum() == 1:
        out[ok] = 0.5
    return out


def _ictal_by_channel(ds_sid, key="bb_auc"):
    npz, mj = ICTAL_CACHE / f"{ds_sid}.npz", ICTAL_CACHE / f"{ds_sid}.json"
    if not npz.exists() or not mj.exists():
        return {}
    data = np.load(npz, allow_pickle=True); meta = json.load(open(mj))
    names = [str(x) for x in data["channels"]]
    arrs = [np.asarray(data[f"{key}__{i}"], float) for i in meta.get("eligible_idxs", [])
            if f"{key}__{i}" in data.files]
    if not arrs:
        return {}
    mean_act = np.nanmean(np.vstack(arrs), axis=0)
    return {n: float(v) for n, v in zip(names, mean_act) if np.isfinite(v)}


def _panel(ax, xs, ys, vals, support, xlim, ylim, sigma, title, cbar_label, soz):
    X, Y, T, _, _ = _smooth_rank_field_mm(xs, ys, vals, support, xlim, ylim, sigma)
    im = ax.imshow(T, origin="lower", extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                   aspect="equal", cmap="viridis", vmin=0, vmax=1)
    okv = np.isfinite(vals)
    ax.scatter(np.asarray(xs)[okv], np.asarray(ys)[okv], c=np.asarray(vals)[okv], cmap="viridis",
               vmin=0, vmax=1, s=70, zorder=3,
               edgecolors=["k" if z else "white" for z, v in zip(soz, vals) if np.isfinite(v)],
               linewidths=[1.6 if z else 0.5 for z, v in zip(soz, vals) if np.isfinite(v)])
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel("along template-A propagation axis (mm)")
    ax.set_ylabel("transverse (mm)")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal", adjustable="box")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label(cbar_label, fontsize=9)


def plot_subject(ds_sid):
    dataset, subject = ds_sid.split("_", 1)
    rec_f = GEOM_BROAD / f"{ds_sid}_t_a.json"
    if not rec_f.exists():
        print(f"[skip] {ds_sid}: no broad t_a record"); return None
    rec = json.loads(rec_f.read_text())
    if not rec.get("channels"):
        print(f"[skip] {ds_sid}: empty record"); return None
    try:
        bundle = erm.load_event_labels_ranks(dataset, subject)
    except Exception as e:
        print(f"[skip] {ds_sid}: load failed {e}"); return None

    # map cluster label -> template, to know which class is "A" (t_a) vs "B" (t_b)
    order = bundle["channel_names"]
    name_to_ta = {c["name"]: c.get("typical_rank") for c in rec["channels"]}
    tb_f = GEOM_BROAD / f"{ds_sid}_t_b.json"
    rec_b = json.loads(tb_f.read_text()) if tb_f.exists() else None
    name_to_tb = {c["name"]: c.get("typical_rank") for c in (rec_b["channels"] if rec_b else [])}
    ta_rank = np.array([name_to_ta.get(n, np.nan) for n in order], float)
    tb_rank = np.array([name_to_tb.get(n, np.nan) for n in order], float)
    cmap = erm.map_clusters_to_templates(np.array(bundle["cluster_template_ranks"][0], float),
                                         np.array(bundle["cluster_template_ranks"][1], float),
                                         ta_rank, tb_rank)
    if cmap["ambiguous"]:
        print(f"[skip] {ds_sid}: ambiguous cluster->template map"); return None
    label_A = [k for k, t in cmap["map"].items() if t == "t_a"][0]
    label_B = [k for k, t in cmap["map"].items() if t == "t_b"][0]

    valA = erm.class_aggregate_contact_values(bundle, label_A)
    valB = erm.class_aggregate_contact_values(bundle, label_B)
    ict = _ictal_by_channel(ds_sid)

    _attach_real_coords([rec])
    frame = _subject_display_frame([rec])
    if frame is None:
        print(f"[skip] {ds_sid}: no display frame"); return None
    xs, ys = _display_points(rec, frame)
    names = [c["name"] for c in rec["channels"]]
    soz = np.array([bool(c.get("is_soz")) for c in rec["channels"]])

    A_vals = _rank01([valA.get(n, {}).get("value", np.nan) for n in names])
    A_sup = np.array([valA.get(n, {}).get("support", 0.0) for n in names], float)
    B_vals = _rank01([valB.get(n, {}).get("value", np.nan) for n in names])
    B_sup = np.array([valB.get(n, {}).get("support", 0.0) for n in names], float)
    ict_raw = _rank01([ict.get(n, np.nan) for n in names])
    ict_sup = np.where(np.isfinite(ict_raw), 1.0, 0.0)

    xlim, ylim, sigma = frame["xlim"], frame["ylim"], frame["sigma_mm"]
    nA_ev = int(np.sum(bundle["labels"] == label_A)); nB_ev = int(np.sum(bundle["labels"] == label_B))
    fig, ax = plt.subplots(1, 3, figsize=(18.5, 6.0), constrained_layout=True)
    _panel(ax[0], xs, ys, A_vals, A_sup, xlim, ylim, sigma,
           f"interictal field — class A (all {nA_ev} A-events, weight-normalized)",
           "early (0) -> late (1)", soz)
    _panel(ax[1], xs, ys, B_vals, B_sup, xlim, ylim, sigma,
           f"interictal field — class B (all {nB_ev} B-events, weight-normalized)",
           "early (0) -> late (1)", soz)
    _panel(ax[2], xs, ys, ict_raw, ict_sup, xlim, ylim, sigma,
           "seizure-onset activation — broadband power 0-10 s",
           "activation low (0) -> high (1)", soz)
    pretty = ds_sid.replace("epilepsiae_", "E").replace("yuquan_", "Y-")
    fig.suptitle(f"Patient {pretty} — two interictal propagation classes vs seizure onset "
                 f"(broad 20-contact plane; EXPLORATORY secondary, NOT a replay claim)", fontsize=12)
    fig.text(0.5, 0.005, "black ring = clinical seizure-onset contact (overlay only). All three fields "
             "on the same physical plane (template-A axis frame); class-B order shown on the same layout.",
             ha="center", fontsize=8.5, color="0.35")
    OUT.mkdir(parents=True, exist_ok=True)
    fp = OUT / f"{ds_sid}_class_fields.png"
    fig.savefig(fp, dpi=140); plt.close(fig)
    return fp


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()
    subs = args.subjects or sorted(p.stem.replace("_t_a", "") for p in GEOM_BROAD.glob("*_t_a.json"))
    made = []
    for ds_sid in subs:
        fp = plot_subject(ds_sid)
        if fp:
            print(f"[fig] {fp}"); made.append(fp.name)
    print(f"[done] {len(made)} field figures -> {OUT}")


if __name__ == "__main__":
    main()
