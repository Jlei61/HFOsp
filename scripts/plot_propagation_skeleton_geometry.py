#!/usr/bin/env python3
"""Figures for propagation skeleton geometry. Spec §8.

Three figures, each answering one question (CLAUDE.md §7):
  Fig 1 — per-subject axis-frame scatter (what a skeleton looks like).
  Fig 2 — cohort geometric scalars, dataset-stratified (no pooled mm).
  Fig 3 — pooled along-axis stereotypy-excess profile (boundary sanity).

The temporal metric is the n-INVARIANT RAW EXCESS (obs - null_mean), NOT the
rate-inflated matched-null z. Every label says so.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

_ROOT = Path(__file__).resolve().parents[1]
BASE = _ROOT / "results/topic4_sef_hfo/skeleton_geometry"
FIG = BASE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

recs = [json.loads(p.read_text())
        for p in sorted((BASE / "per_subject").glob("*.json"))]
# Cohort-stat records: primary + fallback with a measured axis (not 1D-excluded
# is handled per panel). Descriptive-only / errored records are dropped.
ok = [r for r in recs
      if r.get("eligibility_tier") in ("primary", "fallback")
      and "axis_length_mm" in r]

_DS_TAG = {"epilepsiae": "(MNI mm)", "yuquan": "(native mm)"}
_EXCESS_LABEL = "stereotypy excess (raw, obs−null_mean)"

# ---------------------------------------------------------------------------
# Fig 1 — per-subject axis-frame scatter (key "what is a skeleton" figure)
# ---------------------------------------------------------------------------
# Up to 6 PRIMARY + distributed subjects (off-axis is only meaningful when the
# sampling is distributed, not a single 1D shaft).
fig1_subjects = [
    r for r in recs
    if r.get("eligibility_tier") == "primary"
    and r.get("sampling_geometry", {}).get("geometry") == "distributed"
    and r.get("channels")
][:6]

# Shared, symmetric diverging color scale across ALL plotted channels.
all_excess = [c["stereotypy_excess"]
              for r in fig1_subjects for c in r["channels"]
              if np.isfinite(c["stereotypy_excess"])]
M = max((abs(v) for v in all_excess), default=1.0) or 1.0
norm = Normalize(vmin=-M, vmax=M)
cmap = plt.get_cmap("coolwarm")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes_flat = axes.ravel()
src_handle = snk_handle = None
for ax, r in zip(axes_flat, fig1_subjects):
    chans = r["channels"]
    xs = np.array([c["along_axis_mm"] for c in chans])
    ys = np.array([c["off_axis_mm"] for c in chans])
    cs = np.array([c["stereotypy_excess"] for c in chans])
    roles = [c["role"] for c in chans]
    interior = [i for i, ro in enumerate(roles) if ro == "interior"]
    source = [i for i, ro in enumerate(roles) if ro == "source_core"]
    sink = [i for i, ro in enumerate(roles) if ro == "sink_core"]
    L = float(r["axis_length_mm"])
    # axis line at off=0 from source (along=0) to sink (along=L)
    ax.plot([0, L], [0, 0], color="0.4", lw=1.2, zorder=0,
            label="source→sink axis")
    if interior:
        ax.scatter(xs[interior], ys[interior], c=cs[interior], cmap=cmap,
                   norm=norm, marker="o", s=70, edgecolors="0.3",
                   linewidths=0.5, zorder=2)
    hs = hk = None
    if source:
        hs = ax.scatter(xs[source], ys[source], c=cs[source], cmap=cmap,
                        norm=norm, marker="^", s=160, edgecolors="k",
                        linewidths=1.2, zorder=3)
    if sink:
        hk = ax.scatter(xs[sink], ys[sink], c=cs[sink], cmap=cmap, norm=norm,
                        marker="s", s=130, edgecolors="k", linewidths=1.2,
                        zorder=3)
    if hs is not None and src_handle is None:
        src_handle = hs
    if hk is not None and snk_handle is None:
        snk_handle = hk
    ax.set_title(f"{r['dataset']}:{r['subject']}  L={L:.0f}mm", fontsize=10)
    ax.set_xlabel("along-axis distance from source (mm)")
    ax.set_ylabel("off-axis distance from axis (mm)")
for ax in axes_flat[len(fig1_subjects):]:
    ax.axis("off")

# shared legend (markers) + single shared colorbar
legend_handles = []
if src_handle is not None:
    legend_handles.append(
        plt.Line2D([], [], marker="^", color="w", markerfacecolor="0.6",
                   markeredgecolor="k", markersize=12, label="source core"))
if snk_handle is not None:
    legend_handles.append(
        plt.Line2D([], [], marker="s", color="w", markerfacecolor="0.6",
                   markeredgecolor="k", markersize=11, label="sink core"))
legend_handles.append(
    plt.Line2D([], [], marker="o", color="w", markerfacecolor="0.7",
               markeredgecolor="0.3", markersize=10, label="interior channel"))
fig.legend(handles=legend_handles, loc="lower center", ncol=3,
           frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("Per-subject interictal propagation skeleton — axis-coordinate "
             "frame\n(color = temporal stereotypy excess; source/sink cores "
             "marked; primary + distributed-sampling subjects)", fontsize=12)
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
cbar.set_label(_EXCESS_LABEL)
fig.subplots_adjust(bottom=0.10, top=0.90)
fig.savefig(FIG / "axis_frame_examples.png", dpi=140, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 2 — cohort geometric scalars, dataset-stratified (no pooled mm)
# ---------------------------------------------------------------------------
DATASETS = [("yuquan", "tab:blue"), ("epilepsiae", "tab:orange")]
panels = [
    ("source-sink axis length (mm)",
     lambda r: r["axis_length_mm"], False),
    ("source-core radius RMS (mm)",
     lambda r: r["source_radius"]["rms_mm"], False),
    ("sink-core radius RMS (mm)",
     lambda r: r["sink_radius"]["rms_mm"], False),
    ("perpendicular spread RMS (mm)",
     lambda r: r["perp_spread"]["rms_mm"], True),  # measurable-only
]
rng = np.random.default_rng(0)
fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
for ax, (lab, getter, measurable_only) in zip(axes, panels):
    for i, (ds, color) in enumerate(DATASETS):
        vals = []
        for r in ok:
            if r["dataset"] != ds:
                continue
            if measurable_only and not r.get("perp_width_measurable"):
                continue
            v = getter(r)
            if v is not None and np.isfinite(v):
                vals.append(v)
        jitter = rng.uniform(-0.08, 0.08, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, c=color, alpha=0.75,
                   s=45, edgecolors="0.3", linewidths=0.4,
                   label=f"{ds} {_DS_TAG[ds]}")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Yuquan\nSEEG", "Epilepsiae\nSEEG+ECoG"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel(lab)
    ax.legend(fontsize=7, loc="upper right")
axes[3].set_title("(1D-sampling subjects excluded)", fontsize=8)
fig.suptitle("Interictal propagation skeleton — geometric scalars, per dataset\n"
             "(Epilepsiae in MNI-space mm, NOT pooled with Yuquan native mm)",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(FIG / "skeleton_scalars_by_dataset.png", dpi=140)
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 3 — pooled along-axis stereotypy-excess profile (boundary sanity)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 5))
xs, ys = [], []
for r in ok:  # primary AND fallback
    for b in r.get("along_axis_profile", []):
        if b["n"] > 0 and np.isfinite(b.get("mean_excess", np.nan)):
            xs.append((b["a_lo"] + b["a_hi"]) / 2.0)
            ys.append(b["mean_excess"])
ax.axhline(0, ls=":", c="grey", lw=1.2, label="chance (matched null)")
ax.scatter(xs, ys, alpha=0.45, c="tab:purple", s=40, edgecolors="0.3",
           linewidths=0.3)
ax.set_xlabel("along-axis distance from source (mm)")
ax.set_ylabel(_EXCESS_LABEL)
ax.set_title("Temporal stereotypy along the propagation axis\n"
             "(pooled across subjects; descriptive boundary sanity)",
             fontsize=12)
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "along_axis_stereotypy_profile.png", dpi=140)
plt.close(fig)

print("Fig 1 subjects:", [f"{r['dataset']}:{r['subject']}" for r in fig1_subjects])
print("excess color scale: +/-", round(M, 4))
print("wrote 3 figures to", FIG)
