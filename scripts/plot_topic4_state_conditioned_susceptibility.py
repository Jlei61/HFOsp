"""Plot the Topic 4 state-conditioned spatial susceptibility diagnostic figure.

Structure (review 2026-07-19): keep the three DISTINCT linear objects separate, in the order
  z(x)  ->  leading eigenmode (asymptotic)  ->  V1 optimal finite-time INPUT  ->  U1 optimal OUTPUT
        ->  G(k_par, k_perp) susceptibility atlas.
Reads artifacts ONLY (no simulation). Columns = the resolved pre-onset trajectory; onset is the
unresolved equilibrium boundary (shown as a hatched panel, NOT gain=0). "axial/perp" gains are gains
for input WAVEVECTOR parallel / perpendicular to the scaffold axis (k_par / k_perp), NOT output
propagation direction — the propagation-direction evidence is the U1 output-field elongation (row 4).
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
sys.path.insert(0, ROOT)
from src.topic4_m3b_spectral_phase import Grid  # noqa: E402

OUT_DIR = os.path.join(ROOT, "results", "topic4_sef_hfo", "state_conditioned_susceptibility")
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


def _stack(arrays, seeds, state, key):
    fs = [arrays[f"{s}__{state}__{key}"] for s in seeds if f"{s}__{state}__{key}" in arrays.files]
    return [f for f in fs if np.isfinite(f).any()]


def _median_field(arrays, seeds, state, key):
    fs = _stack(arrays, seeds, state, key)
    return np.median(np.stack(fs), axis=0) if fs else None


def _sign_aligned_median(arrays, seeds, state, key, align_mask):
    """Median of a SIGNED field (SVD singular vectors have arbitrary sign): flip each seed so its mean
    over `align_mask` (source-core region) is >= 0, then median. Makes the +/- lobe structure additive."""
    fs = _stack(arrays, seeds, state, key)
    if not fs:
        return None
    out = []
    for f in fs:
        s = np.nanmean(f[align_mask]) if align_mask.any() else np.nanmean(f)
        out.append(f if s >= 0 else -f)
    return np.median(np.stack(out), axis=0)


def _dig(d, path):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur or cur[k] is None:
            return None
        cur = cur[k]
    return cur if isinstance(cur, (int, float)) else None


def _cbar(fig, im, ax, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=7); cb.set_label(label, fontsize=8)


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

    snp = np.load(os.path.join(OUT_DIR, "snapshots", args.candidate, f"seed_{atlas['seeds'][0]}.npz"),
                  allow_pickle=True)
    src_n = _affine(snp["src_xy"], cfg); snk_n = _affine(snp["snk_xy"], cfg)
    core_r = float(cfg["core_radius_norm"])
    grid = Grid(n=int(cfg["grid_n"]), L=L)
    X, Y = grid.coords()
    src_mask = ((X - src_n[0]) ** 2 + (Y - src_n[1]) ** 2) <= core_r ** 2   # sign-align region for V1/U1
    pmax = int(cfg["p_max"]); kmax = 2 * np.pi * pmax / L

    # ---- collect per-state fields (median; V1/U1 sign-aligned) ----
    z_f = {s: _median_field(arrays, seeds, s, "zbar_field") for s in COL_STATES}
    eig_f = {s: _median_field(arrays, seeds, s, "eigen_field") for s in COL_STATES}
    v1_f = {s: _sign_aligned_median(arrays, seeds, s, "v1_optimal_input", src_mask) for s in COL_STATES}
    u1_f = {s: _sign_aligned_median(arrays, seeds, s, "u1_optimal_output", src_mask) for s in COL_STATES}
    g_f = {s: _median_field(arrays, seeds, s, "gain_kxky") for s in COL_STATES}

    def _rng(fdict, absval=False):
        vals = np.concatenate([(np.abs(f) if absval else f).ravel() for f in fdict.values() if f is not None]) \
            if any(v is not None for v in fdict.values()) else np.array([0.0, 1.0])
        return float(np.nanmin(vals)), float(np.nanmax(vals))

    zlo, zhi = _rng(z_f)
    eighi = _rng(eig_f, absval=True)[1]
    v1a = max(_rng(v1_f, absval=True)[1], 1e-12); u1a = max(_rng(u1_f, absval=True)[1], 1e-12)
    ghi = max(_rng(g_f, absval=True)[1], 1e-12)

    fig = plt.figure(figsize=(15.5, 15.5))
    gs = GridSpec(5, 4, figure=fig, hspace=0.36, wspace=0.14,
                  height_ratios=[1, 1, 1, 1, 1], top=0.94, bottom=0.06, left=0.055, right=0.965)

    def _geom(ax):
        ax.plot([src_n[0], snk_n[0]], [src_n[1], snk_n[1]], "-", color="w", lw=1.0, alpha=0.65)
        ax.plot(*src_n, "^", color="w", ms=8, mec="k", mew=0.6); ax.plot(*snk_n, "v", color="w", ms=8, mec="k", mew=0.6)
        for c in (src_n, snk_n):
            ax.add_patch(Circle(c, core_r, fill=False, ec="w", lw=0.9, ls="--", alpha=0.75))

    ROWS = [
        ("row 1: slow field  z(x)\n(median/seeds)", z_f, "viridis", (zlo, zhi), "z_bar", True, False),
        ("row 2: leading EIGENMODE |phi_E|\n(asymptotic mode shape)", eig_f, "magma", (0, eighi), "eigenmode", True, False),
        ("row 3: optimal INPUT V1\n(most-amplified pattern, T=30ms)", v1_f, "RdBu_r", (-v1a, v1a), "V1 (signed)", True, True),
        ("row 4: optimal OUTPUT U1\n(response; axis elongation = propagation)", u1_f, "RdBu_r", (-u1a, u1a), "U1 (signed)", True, True),
        ("row 5: susceptibility G(k||, k_perp)\n(probe gain, T=30ms)", g_f, "magma", (0, ghi), "paired gain", False, False),
    ]
    for r, (ylab, fdict, cmap, (vlo, vhi), clab, geom, kspace_none) in enumerate(ROWS):
        for j, (st, title) in enumerate(zip(COL_STATES, COL_TITLES)):
            ax = fig.add_subplot(gs[r, j])
            f = fdict[st]
            is_k = (r == 4)
            ext = [-kmax, kmax, -kmax, kmax] if is_k else [-half, half, -half, half]
            if f is not None:
                im = ax.imshow(f.T, origin="lower", extent=ext, cmap=cmap, vmin=vlo, vmax=vhi, aspect="equal")
                if geom and not is_k:
                    _geom(ax)
                if is_k:
                    ax.axhline(0, color="w", lw=0.4, alpha=0.4); ax.axvline(0, color="w", lw=0.4, alpha=0.4)
                    ax.set_xlabel("k|| (along axis)", fontsize=7); ax.set_ylabel("k_perp", fontsize=7)
                if j == 3:
                    _cbar(fig, im, ax, clab)
            else:
                ax.set_facecolor("0.92")
                ax.text(0.5, 0.5, "unresolved", transform=ax.transAxes, ha="center", va="center",
                        color="crimson", fontsize=8)
            if r == 0:
                ax.set_title(title, fontsize=11)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=8.5)

    fig.suptitle("Topic 4 — state-conditioned spatial susceptibility along the MZ z-depletion trajectory  "
                 "(candidate %s; static state maps; model-side diagnostic, NOT a seizure)" % args.candidate,
                 fontsize=12.5, y=0.965)
    fig.text(0.5, 0.02, "eigenmode (asymptotic mode) / V1 (optimal finite-time input) / U1 (optimal output) / "
             "G (probe scan) are DISTINCT objects. k||/k_perp = input WAVEVECTOR direction (not propagation). "
             "onset = unresolved equilibrium boundary (omitted). Propagation DYNAMICS are in the separate "
             "time_response figure (fixed-kick sigma1(T) + evolution + kymograph). peak_k rail-limited "
             "(low-k = whole-sheet; see convergence figure).", ha="center", fontsize=8.0, style="italic", color="0.35")
    os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, "figures", f"state_conditioned_susceptibility_diagnostic.{ext}"),
                    dpi=150, bbox_inches="tight")
    print("[plot] wrote figures/state_conditioned_susceptibility_diagnostic.{png,pdf}")
    _plot_controls(cfg, atlas)


def _plot_controls(cfg, atlas):
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
        kpar = [med(cn, st, "axial_gain") for cn in controls]
        kperp = [med(cn, st, "perp_gain") for cn in controls]
        ax.bar(x - w / 2, kpar, w, label="k|| (wavevector along axis)", color="#1b7837")
        ax.bar(x + w / 2, kperp, w, label="k_perp", color="#762a83")
        for i, a in enumerate(kpar):
            if np.isnan(a):
                ax.text(i, 0.01, "unresolved", rotation=90, fontsize=7, ha="center", va="bottom", color="crimson")
        ax.set_xticks(x); ax.set_xticklabels([cn.replace("_isotropic", "").replace("_", "\n") for cn in controls],
                                             fontsize=8.2)
        ax.set_title("%s: finite-time gain (T=%.0f ms), median/3 seeds" % (title, cfg["T_primary"]), fontsize=10)
        ax.set_ylabel("finite-time gain"); ax.legend(fontsize=8.5, loc="upper right"); ax.grid(alpha=0.25, axis="y")
    fig.suptitle("Controls — real vs uniform-mean / rotate-90 / shuffle / z-blocked / AR1   "
                 "(spatial pattern sets MAGNITUDE: real>>uniform;  anisotropic scaffold sets the k|| preference: "
                 "rotate/shuffle keep the k||>k_perp margin, AR1 halves it)", fontsize=10.3)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, "figures", f"state_conditioned_susceptibility_controls.{ext}"),
                    dpi=150, bbox_inches="tight")
    print("[plot] wrote figures/state_conditioned_susceptibility_controls.{png,pdf}")


if __name__ == "__main__":
    main()
