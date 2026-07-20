"""FCXR-RC1 Stage A clip-audit figure: is the arm-C recurrent-conductance clip a localized mode?
  A  spatial map of clipping E cells (colored by clip_count) over the low-Vth cores -> localized vs spread
  B  per-cell peak RAW recurrent conductance (pre-clip) distribution + the cap -> how heavy is the tail
  C  audit metrics (core enrichment / persistence / spatial radius / leading-mode IPR) + verdict
Reads a clip-audit run dir (default: latest_clip_audit_dt0.1_cap99.json).
Output: results/topic4_sef_hfo/mz_full_conductance_spatial_relay/figures/clip_audit.png
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_ROOT = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_full_conductance_spatial_relay")
FIGDIR = os.path.join(OUT_ROOT, "figures")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run-dir", default=None); args = ap.parse_args()
    rd = args.run_dir or json.load(open(os.path.join(OUT_ROOT, "latest_clip_audit_dt0.1_cap99.json")))["path"]
    s = json.load(open(os.path.join(rd, "summary.json")))
    z = np.load(os.path.join(rd, "clip_identity.npz"))
    pos = np.asarray(z["posE"], float); clip = np.asarray(z["clip_count"]); vth = np.asarray(z["vth_E"], float)
    mraw = np.asarray(z["max_raw_gErec"], float)
    a = s.get("audit") or {}
    cap = float(s["cap"]); clipped = clip > 0; core = vth < 18.0

    os.makedirs(FIGDIR, exist_ok=True)
    fig = plt.figure(figsize=(15, 4.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 0.9], wspace=0.28)

    # ---- A: spatial clip map ----
    axA = fig.add_subplot(gs[0, 0])
    axA.scatter(pos[~clipped, 0], pos[~clipped, 1], s=1, c="0.85", label="E (no clip)", rasterized=True)
    axA.scatter(pos[core, 0], pos[core, 1], s=3, c="#88CCEE", alpha=0.5, label="low-Vth core", rasterized=True)
    if clipped.any():
        sc = axA.scatter(pos[clipped, 0], pos[clipped, 1], s=18, c=clip[clipped], cmap="autumn_r",
                         edgecolors="k", linewidths=0.2, zorder=5, label=f"clip ({int(clipped.sum())})")
        plt.colorbar(sc, ax=axA, fraction=0.045, label="clip_count")
    axA.set_aspect("equal"); axA.set_xlabel("x (mm)"); axA.set_ylabel("y (mm)")
    axA.set_title(f"A · clipping cells over cores  (core_enrich={a.get('core_enrichment')})", fontsize=10, loc="left")
    axA.legend(fontsize=7, loc="upper right", markerscale=2)

    # ---- B: raw gErec tail ----
    axB = fig.add_subplot(gs[0, 1])
    pos_g = mraw[mraw > 0]
    if pos_g.size:
        axB.hist(pos_g, bins=np.logspace(np.log10(max(pos_g.min(), 1e-3)), np.log10(pos_g.max() + 1e-9), 60),
                 color="#4C72B0", alpha=0.85)
        axB.set_xscale("log")
    axB.axvline(cap, color="crimson", ls="--", lw=1.4, label=f"cap={cap:g}")
    axB.set_xlabel("per-cell peak RAW recurrent conductance g_Erec"); axB.set_ylabel("# E cells")
    axB.set_title(f"B · raw gErec tail  (P95={np.percentile(mraw,95):.2f}, max={mraw.max():.1f})", fontsize=10, loc="left")
    axB.legend(fontsize=8)

    # ---- C: audit metrics + verdict ----
    axC = fig.add_subplot(gs[0, 2]); axC.axis("off")
    sp = a.get("spatial") or {}
    lines = [
        ("n clip cells", a.get("n_clip_cells")),
        ("clip frames", a.get("n_clip_frames")),
        ("persistent share", round(a.get("persistent_share", float("nan")), 3)),
        ("core enrichment ×", round(a.get("core_enrichment", float("nan")), 2)),
        ("clip rms radius mm", round(sp.get("rms_radius_mm", float("nan")), 2) if sp else None),
        ("clip p90 radius mm", round(sp.get("p90_radius_mm", float("nan")), 2) if sp else None),
        ("leading right IPR", a.get("leading_right_ipr")),
        ("corr(clip,-Vth)", round(a.get("corr_clipcount_neg_vth", float("nan")), 3)),
        ("settled clip %", round(100 * s["row"]["settled_max_clip"], 4)),
        ("n returning (ref)", s["row"]["event_profile"]["n_returning"]),
    ]
    y = 0.95
    for k, v in lines:
        axC.text(0.02, y, f"{k}:", fontsize=9, va="top"); axC.text(0.62, y, f"{v}", fontsize=9, va="top", weight="bold")
        y -= 0.075
    axC.text(0.02, y - 0.02, "verdict:", fontsize=9, va="top")
    axC.text(0.02, y - 0.09, a.get("verdict", "?"), fontsize=8.5, va="top", color="#8B0000", wrap=True)
    axC.set_title("C · clip audit", fontsize=10, loc="left")

    fig.suptitle(f"MZ-FCXR-RC1 Stage A clip audit — arm C (rec-cond), seed{s['seed']}, dt={s['dt']}, cap={cap:g}",
                 fontsize=11, y=1.02)
    out = os.path.join(FIGDIR, "clip_audit.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); print("wrote", out)


if __name__ == "__main__":
    main()
