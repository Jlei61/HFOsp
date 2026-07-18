"""Plot the Topic 4 state-conditioned spatial susceptibility diagnostic figure (design §9).

Reads artifacts ONLY (no simulation). Layout (fixed coords/limits/colormaps across state columns):
  row 1: coarse z(x) slow field  (source ^, sink v, scaffold axis, core outlines)
  row 2: phase-paired susceptibility in kx-ky at T_primary
  row 3: registered strongest probe -> finite-time E RESPONSE field (NOT an eigenmode)
  row 4: within-seed (faint) + median (bold) trajectories of axial / perpendicular / global gain,
         and gain persistence, across all five states.

Usage: python scripts/plot_topic4_state_conditioned_susceptibility.py --candidate zA_q50_tz10000
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.gridspec import GridSpec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results", "topic4_sef_hfo", "state_conditioned_susceptibility")

# 4 field columns = the RESOLVED pre-onset trajectory (onset is the unresolved boundary, shown as the
# blank endpoint in row 4a). Row 4 shows all 5 states.
COL_STATES = ["baseline_1000ms", "mid_fraction", "pre_onset_500ms", "pre_onset_100ms"]
COL_TITLES = ["baseline (1000 ms)", "mid (0.5x onset)", "pre-onset (-500 ms)", "pre-onset (-100 ms)"]
ALL_STATES = ["baseline_1000ms", "mid_fraction", "pre_onset_500ms", "pre_onset_100ms", "onset"]
STATE_SHORT = {"baseline_1000ms": "base", "mid_fraction": "mid", "pre_onset_500ms": "pre500",
               "pre_onset_100ms": "pre100", "onset": "onset"}


def _cfg():
    import yaml
    return yaml.safe_load(open(os.path.join(ROOT, "config", "topic4_state_conditioned_susceptibility.yaml")))


def _affine(xy, cfg):
    return (np.asarray(xy, float) - np.asarray(cfg["center_phys"], float)) * (cfg["L_norm"] / cfg["L_phys"])


def _median_field(arrays, seeds, state, key):
    fields = [arrays[f"{s}__{state}__{key}"] for s in seeds if f"{s}__{state}__{key}" in arrays.files]
    if not fields:
        return None
    return np.median(np.stack(fields), axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="zA_q50_tz10000")
    args = ap.parse_args()
    cfg = _cfg()
    L = float(cfg["L_norm"]); half = L / 2.0
    atlas = json.load(open(os.path.join(OUT_DIR, "susceptibility_atlas.json")))
    arrays = np.load(os.path.join(OUT_DIR, "susceptibility_arrays.npz"), allow_pickle=True)
    seeds = [str(s) for s in atlas["seeds"]]
    Tp = float(cfg["T_primary"])

    # geometry overlay (normalized) from the primary seed snapshot
    snap0 = json.load(open(os.path.join(OUT_DIR, "snapshots", args.candidate, f"seed_{atlas['seeds'][0]}.json")))
    snp = np.load(os.path.join(OUT_DIR, "snapshots", args.candidate, f"seed_{atlas['seeds'][0]}.npz"), allow_pickle=True)
    src_n = _affine(snp["src_xy"], cfg); snk_n = _affine(snp["snk_xy"], cfg)
    core_r = float(cfg["core_radius_norm"])
    pmax = int(cfg["p_max"]); kmax = 2 * np.pi * pmax / L

    fig = plt.figure(figsize=(15.5, 15.0))
    gs = GridSpec(4, 4, figure=fig, hspace=0.34, wspace=0.14,
                  height_ratios=[1, 1, 1, 1.05], top=0.935, bottom=0.055, left=0.055, right=0.965)

    # ---- shared color norms per row (fixed across state columns) ----
    z_fields = {s: _median_field(arrays, seeds, s, "zbar_field") for s in COL_STATES}
    k_fields = {s: _median_field(arrays, seeds, s, "gain_kxky") for s in COL_STATES}
    r_fields = {s: _median_field(arrays, seeds, s, "peak_probe_output_rE") for s in COL_STATES}
    zvals = np.concatenate([f.ravel() for f in z_fields.values() if f is not None]) if any(v is not None for v in z_fields.values()) else np.array([0.0, 1.0])
    zlo, zhi = float(np.nanmin(zvals)), float(np.nanmax(zvals))
    kmaxv = max([np.nanmax(f) for f in k_fields.values() if f is not None] + [1e-9])
    rmaxv = max([np.nanmax(f) for f in r_fields.values() if f is not None] + [1e-9])

    def _mark_geometry(ax):
        ax.plot([src_n[0], snk_n[0]], [src_n[1], snk_n[1]], "-", color="white", lw=1.0, alpha=0.7)
        ax.plot(*src_n, "^", color="white", ms=9, mec="k", mew=0.6)      # source
        ax.plot(*snk_n, "v", color="white", ms=9, mec="k", mew=0.6)      # sink
        for c in (src_n, snk_n):
            ax.add_patch(Circle(c, core_r, fill=False, ec="white", lw=1.0, ls="--", alpha=0.8))

    def _unresolved_tag(ax, per_state_seed_status):
        n_res = sum(st == "resolved" for st in per_state_seed_status)
        if n_res < len(seeds):
            ax.text(0.5, 0.5, f"{len(seeds)-n_res}/{len(seeds)} seed(s)\nnot resolved",
                    transform=ax.transAxes, ha="center", va="center", color="crimson", fontsize=9,
                    bbox=dict(boxstyle="round", fc="white", ec="crimson", alpha=0.85))

    for j, (st, title) in enumerate(zip(COL_STATES, COL_TITLES)):
        statuses = [atlas["per_seed"][s].get(st, {}).get("op_status", "missing") for s in seeds]
        # row 1: z field
        ax = fig.add_subplot(gs[0, j])
        f = z_fields[st]
        if f is not None:
            im = ax.imshow(f.T, origin="lower", extent=[-half, half, -half, half], cmap="viridis",
                           vmin=zlo, vmax=zhi, aspect="equal")
            _mark_geometry(ax)
        else:
            ax.set_facecolor("0.9")
        _unresolved_tag(ax, statuses)
        ax.set_title(title, fontsize=11)
        if j == 0:
            ax.set_ylabel("row 1: slow field  z(x)\n(median over seeds)", fontsize=9)
        if f is not None and j == 3:
            _cbar(fig, im, ax, "z_bar")
        # row 2: kx-ky susceptibility
        ax = fig.add_subplot(gs[1, j])
        f = k_fields[st]
        if f is not None:
            im = ax.imshow(f.T, origin="lower", extent=[-kmax, kmax, -kmax, kmax], cmap="magma",
                           vmin=0, vmax=kmaxv, aspect="equal")
        else:
            ax.set_facecolor("0.9"); _unresolved_tag(ax, statuses)
        ax.axhline(0, color="w", lw=0.4, alpha=0.4); ax.axvline(0, color="w", lw=0.4, alpha=0.4)
        if j == 0:
            ax.set_ylabel("row 2: susceptibility\nprobe gain in kx-ky  (T=%.0f ms)" % Tp, fontsize=9)
        if f is not None and j == 3:
            _cbar(fig, im, ax, "paired gain")
        # row 3: finite-time RESPONSE field
        ax = fig.add_subplot(gs[2, j])
        f = r_fields[st]
        if f is not None:
            im = ax.imshow(f.T, origin="lower", extent=[-half, half, -half, half], cmap="inferno",
                           vmin=0, vmax=rmaxv, aspect="equal")
            _mark_geometry(ax)
        else:
            ax.set_facecolor("0.9"); _unresolved_tag(ax, statuses)
        if j == 0:
            ax.set_ylabel("row 3: finite-time RESPONSE\nC exp(J T) b_peak  (NOT an eigenmode)", fontsize=9)
        if f is not None and j == 3:
            _cbar(fig, im, ax, "|rE| response")

    # ---- row 4: gain trajectories + persistence across ALL five states ----
    ax_g = fig.add_subplot(gs[3, 0:2])
    ax_p = fig.add_subplot(gs[3, 2:4])
    xs = np.arange(len(ALL_STATES))
    colors = {"axial_gain": "#1b7837", "perp_gain": "#762a83", "global_gain": "#2166ac"}
    for metric, col in colors.items():
        per_seed_vals = []
        for s in seeds:
            ys = [_dig(atlas["per_seed"][s].get(st), ["atlas", "per_T", str(Tp), metric])
                  if _dig(atlas["per_seed"][s].get(st), ["atlas", "per_T", str(Tp), metric]) is not None
                  else _dig(atlas["per_seed"][s].get(st), ["atlas", "per_T", ("%.1f" % Tp), metric]) for st in ALL_STATES]
            per_seed_vals.append(ys)
            ax_g.plot(xs, ys, "-", color=col, alpha=0.28, lw=1.0)
        med = [np.nanmedian([v[i] for v in per_seed_vals if v[i] is not None]) if any(v[i] is not None for v in per_seed_vals) else np.nan for i in range(len(ALL_STATES))]
        ax_g.plot(xs, med, "-o", color=col, lw=2.4, ms=5, label=metric.replace("_gain", ""))
    ax_g.set_xticks(xs); ax_g.set_xticklabels([STATE_SHORT[s] for s in ALL_STATES], fontsize=9)
    ax_g.set_ylabel("finite-time gain  (T=%.0f ms)" % Tp, fontsize=9)
    ax_g.set_title("row 4a: axial / perpendicular / global gain trajectory  (faint = 3 seeds, bold = median)", fontsize=9.5)
    ax_g.legend(fontsize=9, loc="best"); ax_g.grid(alpha=0.25)

    # persistence panel: axial gain vs T at baseline vs the last resolved pre-onset state (median)
    T_windows = [float(t) for t in atlas["T_windows"]]
    for st, ls, lab in (("baseline_1000ms", "-", "baseline"), ("pre_onset_100ms", "--", "pre-onset (-100 ms)")):
        med = []
        for T in T_windows:
            vals = [_dig(atlas["per_seed"][s].get(st), ["atlas", "per_T", str(T), "axial_gain"]) for s in seeds]
            vals = [v for v in vals if v is not None]
            med.append(np.median(vals) if vals else np.nan)
        ax_p.plot(T_windows, med, ls + "o", color="#1b7837", lw=2.0, label="axial %s" % lab)
    ax_p.set_xlabel("finite-time window T (ms)", fontsize=9)
    ax_p.set_ylabel("median axial gain", fontsize=9)
    ax_p.set_title("row 4b: axial gain vs T (self-limiting persistence)", fontsize=9.5)
    ax_p.legend(fontsize=9); ax_p.grid(alpha=0.25)

    fig.suptitle("Topic 4 — state-conditioned spatial susceptibility along the MZ z-depletion trajectory  "
                 "(candidate %s; model-side diagnostic, NOT a seizure)" % args.candidate, fontsize=12.5, y=0.975)
    fig.text(0.5, 0.014, "Columns = the resolved pre-onset trajectory; onset (runoff boundary) is "
             "unresolved/saturated for all 3 seeds (fail-closed) -> blank endpoint in row 4a. "
             "eigenmode (row-4 text) / probe / finite-time response are distinct objects.",
             ha="center", fontsize=8.3, style="italic", color="0.35")
    os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, "figures", f"state_conditioned_susceptibility_diagnostic.{ext}"),
                    dpi=150, bbox_inches="tight")
    print("[plot] wrote figures/state_conditioned_susceptibility_diagnostic.{png,pdf}")
    _plot_controls(cfg, atlas)


def _plot_controls(cfg, atlas):
    """Companion (design §9 optional): real vs uniform/rotate/shuffle/z-blocked/AR1 controls, at
    baseline and the strongest resolved pre-onset state; shows spatial-pattern -> magnitude and
    scaffold-anisotropy -> axial direction."""
    cpath = os.path.join(OUT_DIR, "control_summary.json")
    if not os.path.exists(cpath):
        return
    c = json.load(open(cpath))
    seeds = [str(s) for s in atlas["seeds"]]
    Tp = str(float(cfg["T_primary"]))
    controls = ["real", "uniform_mean", "rotated_90", "spatial_shuffle", "z_blocked", "AR1_isotropic"]

    def gain(cn, st, metric, s):
        if cn == "real":
            d = atlas["per_seed"][s].get(st)
        elif cn == "AR1_isotropic":
            d = c["ar1_isotropic"].get(s, {}).get(st)
        else:
            d = c["per_seed"][s].get(st, {}).get(cn)
        pt = (d or {}).get("atlas", {})
        pt = pt.get("per_T", {}).get(Tp) if pt else None
        return pt[metric] if pt else np.nan

    def med(cn, st, metric):
        vals = [gain(cn, st, metric, s) for s in seeds]
        vals = [v for v in vals if not (v is None or np.isnan(v))]
        return np.median(vals) if vals else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.4))
    for ax, (st, title) in zip(axes, [("baseline_1000ms", "baseline (1000 ms)"),
                                       ("pre_onset_100ms", "pre-onset (-100 ms)")]):
        x = np.arange(len(controls)); w = 0.38
        axial = [med(cn, st, "axial_gain") for cn in controls]
        perp = [med(cn, st, "perp_gain") for cn in controls]
        ax.bar(x - w / 2, axial, w, label="axial", color="#1b7837")
        ax.bar(x + w / 2, perp, w, label="perpendicular", color="#762a83")
        for i, (a, p) in enumerate(zip(axial, perp)):
            if np.isnan(a):
                ax.text(i, 0.01, "unresolved", rotation=90, fontsize=7, ha="center", va="bottom", color="crimson")
        ax.set_xticks(x); ax.set_xticklabels([cn.replace("_isotropic", "").replace("_", "\n") for cn in controls],
                                             fontsize=8.2)
        ax.set_title("%s: finite-time gain (T=%.0f ms), median/3 seeds" % (title, cfg["T_primary"]), fontsize=10)
        ax.set_ylabel("finite-time gain"); ax.legend(fontsize=9, loc="upper right"); ax.grid(alpha=0.25, axis="y")
    fig.suptitle("Controls — real vs uniform-mean / rotate-90 / shuffle / z-blocked / AR1   "
                 "(spatial pattern sets MAGNITUDE: real>>uniform;  anisotropic scaffold sets axial DIRECTION: "
                 "rotate/shuffle keep the axial>perp margin)", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, "figures", f"state_conditioned_susceptibility_controls.{ext}"),
                    dpi=150, bbox_inches="tight")
    print("[plot] wrote figures/state_conditioned_susceptibility_controls.{png,pdf}")


def _cbar(fig, im, ax, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=7); cb.set_label(label, fontsize=8)


def _dig(d, path):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur or cur[k] is None:
            return None
        cur = cur[k]
    return cur if isinstance(cur, (int, float)) else None


if __name__ == "__main__":
    main()
