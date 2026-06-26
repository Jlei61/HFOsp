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

from scripts.plot_topic5_axis_direction_rose import _load_frame, _seizure_angles
from scripts.run_topic5_directional_replay import template_direction, OUT_DIR, PRIMARY_COHORT
from src.topic5_directional_replay import cluster_directions_k2

FIG_DIR = OUT_DIR / "figures"
C1, C2 = "#1b9e77", "#7570b3"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--activation", default="broadband")
    args = ap.parse_args()
    subs = args.subjects or PRIMARY_COHORT
    for sid in subs:
        out = plot_subject(sid, args.activation)
        print(f"  {'wrote ' + out.name if out else 'skip ' + sid}", flush=True)


if __name__ == "__main__":
    main()
