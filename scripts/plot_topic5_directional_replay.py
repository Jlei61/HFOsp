#!/usr/bin/env python3
"""Topic 5 发作方向两类聚类 ↔ 间期 A/B 方向 — 每被试玫瑰图。

黑虚线 = 间期模板 A/B 方向; 彩色实线 = 发作两类堆内平均方向; 彩色 ticks = 逐发作方向(按类着色)。
角注 = report_tier / axis_tier / p_bimodal / p_align。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.plot_topic5_axis_direction_rose import (_load_frame, _seizure_angles,
                                                     _interictal_event_vals)
from scripts.run_topic5_directional_replay import template_direction, OUT_DIR, PRIMARY_COHORT
from src.topic5_directional_replay import cluster_directions_k2
from src.topic5_axis_direction import (event_angles_by_template, axial_mean,
                                       rotate_to_reference, resultant_length)

FIG_DIR = OUT_DIR / "figures"
C1, C2 = "#1b9e77", "#7570b3"             # ictal direction classes (green / purple)
A_COLOR, B_COLOR = "#1f77b4", "#d95f02"   # interictal templates (match mature rose)


def plot_subject(ds_sid, activation):
    loaded = _load_frame(ds_sid)
    if loaded is None:
        return None
    rec, x, y, names = loaded
    sz = _seizure_angles(ds_sid, x, y, names, activation)
    if sz.size < 4:
        return None
    clus = cluster_directions_k2(sz, seed=0)
    labels, means = clus["labels"], clus["means"]
    thA, *_ = template_direction(ds_sid, x, y, names, "a")
    thB, *_ = template_direction(ds_sid, x, y, names, "b")
    rj = OUT_DIR / "per_subject" / f"{ds_sid}__dir_cluster_{activation}.json"
    meta = json.loads(rj.read_text()) if rj.exists() else {}

    fig = plt.figure(figsize=(7.2, 7.6), constrained_layout=True)
    ax = fig.add_subplot(111, projection="polar")
    for c, col in ((0, C1), (1, C2)):
        for a in clus["angles"][labels == c]:
            ax.plot([a, a], [0, 0.82], color=col, lw=1.1, alpha=0.6, zorder=2)
        if np.isfinite(means[c]):
            ax.plot([means[c], means[c]], [0, 1.05], color=col, lw=3.4, zorder=4,
                    label=f"ictal class {c+1} (n={clus['sizes'][c]}, R={clus['class_R'][c]:.2f})")
    for th, nm in ((thA, "interictal A"), (thB, "interictal B")):
        if np.isfinite(th):
            ax.plot([th, th], [0, 1.12], color="black", lw=2.2, ls="--", zorder=3, label=nm)
    ax.set_theta_zero_location("E"); ax.set_theta_direction(1)
    ax.set_rticks([]); ax.set_rlim(0, 1.2)
    tier = meta.get("report_tier", "?"); axt = meta.get("axis_tier", "?")
    pb = meta.get("p_bimodal"); pa = meta.get("p_align")
    pretty = ds_sid.replace("epilepsiae_", "E").replace("yuquan_", "Y-")
    ax.set_title(f"{pretty} — ictal direction k=2 vs interictal A/B  ({activation})\n"
                 f"report_tier={tier} · axis={axt} · "
                 f"p_bimodal={pb if pb is None else round(pb, 3)} · "
                 f"p_align={pa if pa is None else round(pa, 3)}", fontsize=11)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=8.6)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"{ds_sid}__dir_cluster_{activation}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def plot_class_interictal_rose(ds_sid, activation, bins=18):
    """Cohort-pooled-rose style (hollow histograms): BOTH the two unsupervised ictal
    direction classes AND the two interictal templates drawn as hollow histogram bars.
    Interictal events outnumber seizures ~1000x, so each curve is normalized to its own
    peak bin -> the DIRECTIONAL SHAPES are comparable; raw n + concentration R in legend.
    Seizure axis rotated to 0 deg / 180 deg (mature convention)."""
    loaded = _load_frame(ds_sid)
    if loaded is None:
        return None
    rec, x, y, names = loaded
    ds, subj = ds_sid.split("_", 1)
    sz = _seizure_angles(ds_sid, x, y, names, activation)
    if sz.size < 4:
        return None
    clus = cluster_directions_k2(sz, seed=0)
    ref = axial_mean(sz)                          # seizure axis -> 0 deg / 180 deg
    try:
        event_vals, ev_labels, _ = _interictal_event_vals(ds, subj, names)
        grp = event_angles_by_template(event_vals, x, y, ev_labels) if event_vals is not None else {0: [], 1: []}
    except FileNotFoundError:
        grp = {0: np.array([]), 1: np.array([])}

    fig = plt.figure(figsize=(7.8, 8.0), constrained_layout=True)
    ax = fig.add_subplot(111, projection="polar")
    edges = np.linspace(0, 2 * np.pi, bins + 1)
    centers = edges[:-1] + (edges[1] - edges[0]) / 2
    width = (edges[1] - edges[0]) * 0.95

    def hollow_hist(angles, color, label, lw=2.0):
        a = np.asarray(angles, float); a = a[np.isfinite(a)]
        if a.size == 0:
            return
        counts, _ = np.histogram(rotate_to_reference(a, ref), bins=edges)
        if counts.max() == 0:
            return
        ax.bar(centers, counts / counts.max(), width=width, facecolor="none",
               edgecolor=color, linewidth=lw, alpha=0.95,
               label=f"{label}  n={a.size}, R={resultant_length(a):.2f}")

    # interictal templates (thousands of events) — thin hollow bars
    hollow_hist(grp.get(0, []), A_COLOR, "interictal template A", lw=1.6)
    hollow_hist(grp.get(1, []), B_COLOR, "interictal template B", lw=1.6)
    # two ictal direction classes (units) — thick hollow bars (the seizure 'hist')
    for c, col in ((0, C1), (1, C2)):
        hollow_hist(clus["angles"][clus["labels"] == c], col, f"ictal direction class {c+1}", lw=2.8)
    # seizure axis (black, both ends) — same reference as the cohort-pooled rose
    ax.plot([0, 0], [0, 1.12], color="black", lw=3.0, zorder=1, label="seizure axis (0 deg / 180 deg)")
    ax.plot([np.pi, np.pi], [0, 1.12], color="black", lw=3.0, zorder=1)

    ax.set_theta_zero_location("E"); ax.set_theta_direction(1)
    ax.set_rlim(0, 1.18); ax.set_rticks([0.5, 1.0]); ax.set_rlabel_position(100)
    pretty = ds_sid.replace("epilepsiae_", "E").replace("yuquan_", "Y-")
    ax.set_title(f"{pretty} — two ictal direction classes vs interictal templates (hollow histograms)\n"
                 f"each curve normalized to its own peak; seizure axis = 0 deg / 180 deg  ({activation})",
                 fontsize=11.0, pad=16)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.26), ncol=1, frameon=False, fontsize=8.6)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"{ds_sid}__classes_vs_interictal_hist_{activation}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--activation", default="broadband")
    ap.add_argument("--hist-rose", action="store_true",
                    help="mature-style rose: interictal event hist + class-colored seizure ticks")
    args = ap.parse_args()
    subs = args.subjects or PRIMARY_COHORT
    fn = plot_class_interictal_rose if args.hist_rose else plot_subject
    for sid in subs:
        out = fn(sid, args.activation)
        print(f"  {'wrote ' + out.name if out else 'skip ' + sid}", flush=True)


if __name__ == "__main__":
    main()
