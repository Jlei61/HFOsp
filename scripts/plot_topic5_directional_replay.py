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
from matplotlib.colors import to_rgba
import numpy as np

plt.rcParams.update({"font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14,
                     "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12.5})

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.plot_topic5_axis_direction_rose import (_load_frame, _seizure_angles,
                                                     _interictal_event_vals)
from scripts.run_topic5_directional_replay import template_direction, OUT_DIR, PRIMARY_COHORT
from src.topic5_directional_replay import cluster_directions_k2, cohort_alignment_rotation_test
from src.topic5_axis_direction import (event_angles_by_template, axial_mean,
                                       rotate_to_reference, resultant_length, circular_mean)

FIG_DIR = OUT_DIR / "figures"
C1, C2 = "#1b9e77", "#7570b3"             # ictal direction classes (green / purple) — per-subject fig
A_COLOR, B_COLOR = "#d62728", "#1f77b4"   # interictal templates: A=red / B=blue (combined rose)


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

    fig = plt.figure(figsize=(8.8, 8.6), constrained_layout=True)
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
                 f"p_align={pa if pa is None else round(pa, 3)}", fontsize=12)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=11.5)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"{ds_sid}__dir_cluster_{activation}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def plot_class_interictal_rose(ds_sid, activation, bins=18, subdir=None):
    """简洁版：间期模板 A/B 事件方向画成空心直方图（A=红 / B=蓝，原始计数），
    发作只画两类的**平均方向**（黑线，实线=类1 / 虚线=类2）。只两色直方图 + 黑均向线，
    避免多色拥挤。发作轴旋到 0 deg / 180 deg（mature convention）。"""
    loaded = _load_frame(ds_sid)
    if loaded is None:
        return None
    rec, x, y, names = loaded
    ds, subj = ds_sid.split("_", 1)
    sz = _seizure_angles(ds_sid, x, y, names, activation)
    if sz.size < 4:
        return None
    clus = cluster_directions_k2(sz, seed=0)
    # align so the seizure MAIN direction (mean of the larger ictal class) sits at 0 deg
    _dom = 0 if clus["sizes"][0] >= clus["sizes"][1] else 1
    _main = circular_mean(clus["angles"][clus["labels"] == _dom])
    ref = _main if np.isfinite(_main) else axial_mean(sz)
    try:
        event_vals, ev_labels, _ = _interictal_event_vals(ds, subj, names)
        grp = event_angles_by_template(event_vals, x, y, ev_labels) if event_vals is not None else {0: [], 1: []}
    except FileNotFoundError:
        grp = {0: np.array([]), 1: np.array([])}

    fig = plt.figure(figsize=(9.2, 9.0), constrained_layout=True)
    ax = fig.add_subplot(111, projection="polar")
    edges = np.linspace(0, 2 * np.pi, bins + 1)
    centers = edges[:-1] + (edges[1] - edges[0]) / 2
    width = (edges[1] - edges[0]) * 0.95
    rmax = 1
    for lbl, color, nm in [(0, A_COLOR, "interictal template A"), (1, B_COLOR, "interictal template B")]:
        a = rotate_to_reference(np.asarray(grp.get(lbl, []), float), ref)
        a = a[np.isfinite(a)]
        if a.size == 0:
            continue
        counts, _ = np.histogram(a, bins=edges)
        rmax = max(rmax, int(counts.max()))
        ax.bar(centers, counts, width=width, facecolor=to_rgba(color, 0.22), edgecolor=color,
               linewidth=2.0, label=f"{nm}  n={a.size}, R={resultant_length(a):.2f}")
    # seizure: two class MEAN directions only (black solid=class1, dashed=class2)
    for c, ls in ((0, "-"), (1, "--")):
        m = circular_mean(clus["angles"][clus["labels"] == c])
        if np.isfinite(m):
            mr = float(rotate_to_reference(np.array([m]), ref)[0])
            ax.plot([mr, mr], [0, rmax * 1.1], color="black", lw=3.0, ls=ls, zorder=5,
                    label=f"ictal class {c+1} mean  n={clus['sizes'][c]}, R={clus['class_R'][c]:.2f}")
    ax.set_theta_zero_location("E"); ax.set_theta_direction(1); ax.set_rlabel_position(100)
    pretty = ds_sid.replace("epilepsiae_", "E").replace("yuquan_", "Y-")
    ax.set_title(f"{pretty}: ictal class means + interictal A/B event directions\n"
                 f"seizure main direction → 0°  ({activation})", fontsize=14, pad=14)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=False, fontsize=11.5)
    out_dir = (FIG_DIR / subdir) if subdir else FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{ds_sid}__classes_vs_interictal_hist_{activation}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def plot_cohort_pooled_main_aligned(subjects, activation, bins=24, subdir=None):
    """全队列汇总：每被试旋到自己的发作主要方向(占多数类均向)=0 deg, 再把每被试的间期事件
    方向直方图**归一化后等权平均**(避免事件多的被试主导), 看跨被试间期事件相对发作主方向有没有
    一致偏好。toward-main = 落在主方向 ±90 deg 内的占比(~0.5 = 无方向偏好, 只共享轴)。"""
    edges = np.linspace(0, 2 * np.pi, bins + 1)
    centers = edges[:-1] + (edges[1] - edges[0]) / 2
    width = (edges[1] - edges[0]) * 0.95
    dens, used = [], []
    for ds_sid in subjects:
        loaded = _load_frame(ds_sid)
        if loaded is None:
            continue
        rec, x, y, names = loaded
        ds, subj = ds_sid.split("_", 1)
        sz = _seizure_angles(ds_sid, x, y, names, activation)
        if sz.size < 4:
            continue
        clus = cluster_directions_k2(sz, seed=0)
        dom = 0 if clus["sizes"][0] >= clus["sizes"][1] else 1
        ref = circular_mean(clus["angles"][clus["labels"] == dom])
        if not np.isfinite(ref):
            continue
        try:
            ev, lab, _ = _interictal_event_vals(ds, subj, names)
            grp = event_angles_by_template(ev, x, y, lab) if ev is not None else {0: [], 1: []}
        except FileNotFoundError:
            continue
        a = np.concatenate([np.asarray(grp.get(0, []), float), np.asarray(grp.get(1, []), float)])
        a = rotate_to_reference(a, ref); a = a[np.isfinite(a)]
        if a.size == 0:
            continue
        counts, _ = np.histogram(a, bins=edges)
        if counts.sum() == 0:
            continue
        dens.append(counts / counts.sum())
        used.append(ds_sid)
    if not dens:
        return None
    mean_dens = np.mean(dens, axis=0)
    d0 = np.minimum(centers, 2 * np.pi - centers)            # angular distance to 0 deg
    toward = float(mean_dens[d0 < np.pi / 2].sum() / mean_dens.sum())

    fig = plt.figure(figsize=(9.2, 9.0), constrained_layout=True)
    ax = fig.add_subplot(111, projection="polar")
    ax.bar(centers, mean_dens / mean_dens.max(), width=width, facecolor=to_rgba("#6a3d9a", 0.22),
           edgecolor="#6a3d9a", linewidth=2.2,
           label=f"pooled interictal events ({len(used)} subj, equal weight)")
    ax.plot([0, 0], [0, 1.12], color="black", lw=3.0, label="seizure main direction (each subj = 0 deg)")
    ax.set_theta_zero_location("E"); ax.set_theta_direction(1); ax.set_rlabel_position(100)
    ax.set_rlim(0, 1.18); ax.set_rticks([0.5, 1.0])
    ax.set_title(f"Cohort-pooled interictal directions vs seizure MAIN direction ({activation})\n"
                 f"each subject's seizure main direction = 0°  (n={len(used)}; toward-main = {toward:.2f})",
                 fontsize=13, pad=14)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=False, fontsize=12)
    out_dir = (FIG_DIR / subdir) if subdir else FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"cohort_pooled_main_aligned_{activation}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def plot_cohort_axis_alignment(bands=("broadband", "hfa"), subdir=None):
    """Cohort SIGN-FREE axis-alignment test: per subject, angular gap from the seizure MAIN
    direction (dominant ictal class mean) to the NEARER interictal template direction; vs a
    per-subject uniform-rotation null. One figure, one panel per band (broadband primary)."""
    fig, axes = plt.subplots(1, len(bands), figsize=(14.5, 7.2), constrained_layout=True, sharey=True)
    if len(bands) == 1:
        axes = [axes]
    for ax, band in zip(axes, bands):
        mains, pairs, meta = [], [], []
        for sid in PRIMARY_COHORT:
            jp = OUT_DIR / "per_subject" / f"{sid}__dir_cluster_{band}.json"
            if not jp.exists():
                continue
            j = json.loads(jp.read_text())
            if j.get("status") != "ok":
                continue
            means, sizes, thA, thB = j["means_deg"], j["sizes"], j["theta_A"], j["theta_B"]
            if thA is None or thB is None or None in means:
                continue
            dom = 0 if sizes[0] >= sizes[1] else 1
            mains.append(np.radians(means[dom])); pairs.append((np.radians(thA), np.radians(thB)))
            meta.append((sid.replace("epilepsiae_", "E"), j.get("delta_ab_deg")))
        if not mains:
            continue
        r = cohort_alignment_rotation_test(mains, pairs, B=20000, seed=20260627)
        gaps = np.degrees(r["gaps"]); lo = np.degrees(r["null_lo"])
        md = np.degrees(r["null_md"]); hi = np.degrees(r["null_hi"])
        order = np.argsort(gaps)
        for k, i in enumerate(order):
            ax.bar(k, hi[i] - lo[i], bottom=lo[i], width=0.55, color="0.82", zorder=0)
            ax.hlines(md[i], k - 0.28, k + 0.28, color="0.45", lw=1.5, zorder=1)
            col = "#d62728" if r["per_pct"][i] < 0.05 else "black"
            ax.plot(k, gaps[i], "o", color=col, ms=9, zorder=5)
            if meta[i][1] is not None and meta[i][1] < 60:
                ax.annotate("ΔAB<60°\n(degenerate)", (k, gaps[i]), textcoords="offset points",
                            xytext=(0, 9), ha="center", fontsize=9.5, color="0.4")
        ax.axhline(45, ls="--", color="0.6", lw=1, zorder=0)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([meta[i][0] for i in order], fontsize=13)
        ax.set_ylim(0, 188)
        ax.set_title(f"{band}: median gap {np.degrees(r['T_obs']):.1f}°, "
                     f"rotation-null p={r['p']:.4f} (n={len(mains)})", fontsize=13)
    axes[0].set_ylabel("gap: seizure main dir → nearest interictal template dir (deg)")
    fig.suptitle("Cohort sign-free axis alignment: seizure main direction vs nearest interictal template\n"
                 "gray = rotation null (5–95%, median tick) · dot = observed · dashed = 45° chance",
                 fontsize=13)
    out_dir = (FIG_DIR / subdir) if subdir else FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "cohort_axis_alignment_test.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--activation", default="broadband")
    ap.add_argument("--hist-rose", action="store_true",
                    help="combined rose: interictal A/B event hist + ictal class mean directions")
    ap.add_argument("--pooled", action="store_true",
                    help="cohort-pooled rose aligned to each subject's seizure main direction")
    ap.add_argument("--out-subdir", default=None, help="figures/<subdir>/ for caveat sets (e.g. seeg_caveat)")
    ap.add_argument("--alignment", action="store_true",
                    help="cohort sign-free axis-alignment test figure (both bands)")
    args = ap.parse_args()
    if args.alignment:
        out = plot_cohort_axis_alignment()
        print(f"  wrote {out.name}", flush=True)
        return
    if args.pooled:
        out = plot_cohort_pooled_main_aligned(args.subjects or PRIMARY_COHORT, args.activation)
        print(f"  {'wrote ' + out.name if out else 'pooled: insufficient data'}", flush=True)
        return
    subs = args.subjects or PRIMARY_COHORT
    fn = plot_class_interictal_rose if args.hist_rose else plot_subject
    for sid in subs:
        out = (fn(sid, args.activation, subdir=args.out_subdir) if args.hist_rose
               else fn(sid, args.activation))
        print(f"  {'wrote ' + out.name if out else 'skip ' + sid}", flush=True)


if __name__ == "__main__":
    main()
