"""Paper-ready candidate figures for Topic 4 MZ direct-SNN spatial mode dynamics.

Design contract: docs/superpowers/specs/2026-07-19-topic4-mz-direct-spatial-modes-design.md §8/§11.
Reads results/topic4_sef_hfo/mz_direct_spatial_modes/ (per-seed summaries + arrays) and renders:
  Supplementary 1  fixed-kick spatial response  (baseline vs pre_onset, SAME localized kick)
  Supplementary 2  operator identifiability + axial-recruitment quantification

This is the direct current-based MZ spiking network — NOT a rate-field surrogate, NOT exact
eigenmodes. At the registered amplitudes the empirical operator is nonlinear_response_only
(no linear V1/U1/sigma_hat_1), so Supplementary 2 is the linearity/nonlinearity diagnostic, not
eigenmodes. Fail-closed when a required sidecar is missing (spec §7 C13). Never overwrite the
rate-field Figure 5 directory (spec §8).

Visual contract (spec §11): width ~7.2in, 300 dpi, editable PDF text (fonttype=42), no suptitle,
no background grid, short left panel letters, baseline = grey #555555, pre_onset = ochre #C88719,
NO red/blue (reserved for template A/B). Signed fields -> PuOr (purple/orange) diverging.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_direct_spatial_modes")
CAND_DIR = os.path.join(ROOT, "results", "paper-ready-figure",
                        "fig5_mz_direct_snn_spatial_modes_candidate", "figures")

BASE, MID, PRE = "#555555", "#9aa0a6", "#C88719"
STATE_COLOR = {"baseline": BASE, "midpoint": MID, "pre_onset": PRE}
STATE_LABEL = {"baseline": "baseline", "midpoint": "midpoint", "pre_onset": "pre-onset"}
SIGNED, MAG = "PuOr", "magma"
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8,
                     "axes.linewidth": 0.6, "figure.dpi": 150})


def load_state(label, seed, state):
    j = os.path.join(OUT, "per_seed", f"state_{label}_seed{seed}_{state}.json")
    npz = os.path.join(OUT, "per_seed", f"arrays_{label}_seed{seed}_{state}.npz")
    if not os.path.exists(j) or not os.path.exists(npz):
        return None
    return dict(summary=json.load(open(j)), arr=dict(np.load(npz, allow_pickle=True)))


def seeds_present(label, state="baseline"):
    out = []
    for f in sorted(glob.glob(os.path.join(OUT, "per_seed", f"state_{label}_seed*_{state}.json"))):
        out.append(int(os.path.basename(f).split("seed")[1].split("_")[0]))
    return out


def _letter(ax, s):
    ax.text(-0.02, 1.06, s, transform=ax.transAxes, fontsize=11, fontweight="bold", ha="right", va="bottom")


def _norm_to_grididx(xy, n):
    return (xy[0] / 5.0 + 0.5) * (n - 1), (xy[1] / 5.0 + 0.5) * (n - 1)


def _field(ax, field, *, vmax, title=None, src=None, snk=None):
    f = np.asarray(field, float)
    im = ax.imshow(f.T, origin="lower", cmap=SIGNED, vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8, pad=2)
    n = f.shape[0]
    for xy, mk in ((src, "o"), (snk, "s")):
        if xy is not None:
            gx, gy = _norm_to_grididx(xy, n)
            ax.plot(gx, gy, mk, mfc="none", mec="k", mew=0.9, ms=5)
    return im


# =============================================================== Supplementary 1: fixed-kick response
def figure_supp1(label, seed):
    sts = ["baseline", "pre_onset"]
    data = {s: load_state(label, seed, s) for s in sts}
    if any(data[s] is None for s in sts):
        raise SystemExit(f"[fig1] missing fixed-kick sidecars for seed{seed} {sts} — run + aggregate first")
    centers = data["baseline"]["arr"]["fk_map_centers"].tolist()
    src = data["baseline"]["summary"].get("src_g"); snk = data["baseline"]["summary"].get("snk_g")
    dmax = max(np.nanmax(np.abs(d["arr"]["fk_dmaps"])) for d in data.values()) or 1.0
    kmax = max(np.nanmax(np.abs(d["arr"]["fk_kymo"])) for d in data.values()) or 1.0

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = GridSpec(3, 5, figure=fig, height_ratios=[1.0, 1.05, 0.95], width_ratios=[1, 1, 1, 1, 0.9],
                  hspace=0.5, wspace=0.2, left=0.07, right=0.965, top=0.9, bottom=0.1)
    im = None
    for ri, st in enumerate(sts):
        dmaps = data[st]["arr"]["fk_dmaps"]
        for ci, c in enumerate(centers[:4]):
            ax = fig.add_subplot(gs[ri, ci])
            im = _field(ax, dmaps[ci], vmax=dmax, title=(f"{int(c)} ms" if ri == 0 else None),
                        src=src, snk=snk)
            if ci == 0:
                ax.set_ylabel(STATE_LABEL[st], color=STATE_COLOR[st], fontsize=9, fontweight="bold")
            if ri == 0 and ci == 0:
                _letter(ax, "a")
    cax = fig.add_subplot(gs[0:2, 4]); cax.set_axis_off()
    fig.colorbar(im, ax=cax, fraction=0.55, pad=0.0).set_label("Δ E-rate (Hz)\nkick − control", fontsize=7)

    cmap_k = plt.get_cmap(MAG).copy(); cmap_k.set_bad(plt.get_cmap(MAG)(0.0))   # empty axial bins -> min colour
    for ci, st in enumerate(sts):
        ax = fig.add_subplot(gs[2, ci])
        ky = data[st]["arr"]["fk_kymo"]
        ext = [data[st]["arr"]["fk_kymo_times"][0], data[st]["arr"]["fk_kymo_times"][-1],
               0, data[st]["arr"]["fk_kymo_dist"][-1]]
        imk = ax.imshow(np.ma.masked_invalid(np.abs(ky).T), origin="lower", aspect="auto",
                        cmap=cmap_k, vmin=0, vmax=kmax, extent=ext)
        ax.set_xlabel("time (ms)", fontsize=7.5)
        ax.set_title(STATE_LABEL[st], color=STATE_COLOR[st], fontsize=8)
        if ci == 0:
            ax.set_ylabel("axial dist.\n(src→sink)", fontsize=7.5); _letter(ax, "b")
    cax2 = fig.add_subplot(gs[2, 2]); cax2.set_axis_off()
    fig.colorbar(imk, ax=cax2, fraction=0.55, pad=0.0).set_label("|Δ E-rate| (Hz)", fontsize=7)

    # response norm: baseline vs pre_onset (primary comparison; midpoint is transitional/seed-variable
    # and per spec only enters the linearity diagnostic, not the primary comparison)
    axC = fig.add_subplot(gs[2, 3:5])
    order = ["baseline", "pre_onset"]
    for xi, st in enumerate(order):
        norms = [load_state(label, s, st)["summary"]["fixed_kick"]["response_norm"]
                 for s in seeds_present(label) if load_state(label, s, st)]
        axC.scatter([xi] * len(norms), norms, color=STATE_COLOR[st], s=22, zorder=3)
        if norms:
            axC.plot([xi - 0.22, xi + 0.22], [np.mean(norms)] * 2, color=STATE_COLOR[st], lw=1.8)
    axC.set_xticks(range(2)); axC.set_xticklabels(["baseline", "pre-onset"], fontsize=7.5)
    axC.set_xlim(-0.5, 1.5)
    axC.set_ylabel("fixed-kick response norm", fontsize=7.5)
    axC.spines[["top", "right"]].set_visible(False); _letter(axC, "c")

    _save(fig, "figure5_supplementary_1_direct_snn_spatial_response")


# =============================================================== Supplementary 2: identifiability + axial
def figure_supp2(label, seed, cfg):
    aud_path = os.path.join(OUT, "linearity_audit.json")
    if not os.path.exists(aud_path):
        raise SystemExit("[fig2] missing linearity_audit.json — run first")
    aud = json.load(open(aud_path))
    ladder = [float(a) for a in aud["per_state"]["baseline"]["ladder"]]
    tol = float(cfg["linearity_tol"])

    fig = plt.figure(figsize=(7.2, 3.1))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.25, 1.05, 1.15], wspace=0.42,
                  left=0.08, right=0.975, top=0.86, bottom=0.2)

    # Panel A: linearity discrepancy vs amplitude -> operator NOT identifiable (all above the 15% gate)
    axA = fig.add_subplot(gs[0, 0])
    for st in ["baseline", "midpoint", "pre_onset"]:
        disc = np.array(aud["per_state"][st]["discrepancies"], float)
        axA.plot(ladder, disc, "o-", color=STATE_COLOR[st], ms=4, lw=1.3, label=STATE_LABEL[st])
        nan = ~np.isfinite(disc)
        if nan.any():
            axA.scatter(np.array(ladder)[nan], [tol * 0.35] * nan.sum(), marker="x",
                        color=STATE_COLOR[st], s=22)     # x = no measurable response (below floor)
    axA.axhline(tol, color="k", lw=0.8, ls="--")
    axA.text(ladder[0], tol * 1.12, "15% linearity gate", fontsize=6.5, va="bottom")
    axA.set_xscale("log"); axA.set_xlabel("perturbation amplitude (×I_EE)", fontsize=7.5)
    axA.set_ylabel("linearity discrepancy\n‖K(ε)−K(ε/2)‖ / ‖K(ε/2)‖", fontsize=7.5)
    axA.legend(fontsize=6.3, frameon=False, loc="upper right")
    axA.spines[["top", "right"]].set_visible(False); _letter(axA, "a")

    # Panel B: axial-corridor response magnitude, baseline vs pre_onset, all seeds (robust — no
    # source-division; source can be 0 at the quiet baseline). Near-0 at baseline -> consistent at pre_onset.
    axB = fig.add_subplot(gs[0, 1])
    order = ["baseline", "pre_onset"]
    for xi, st in enumerate(order):
        vals = [load_state(label, s, st)["summary"]["fixed_kick"]["region"]["axis_corridor"]
                for s in seeds_present(label) if load_state(label, s, st)]
        axB.scatter([xi] * len(vals), vals, color=STATE_COLOR[st], s=22, zorder=3)
        if vals:
            axB.plot([xi - 0.22, xi + 0.22], [np.mean(vals)] * 2, color=STATE_COLOR[st], lw=1.8)
    axB.set_xticks(range(2)); axB.set_xticklabels(["baseline", "pre-onset"], fontsize=7.5)
    axB.set_xlim(-0.5, 1.5)
    axB.set_ylabel("axial-corridor response\n|Δ E-rate| (Hz)", fontsize=7.5)
    axB.spines[["top", "right"]].set_visible(False); _letter(axB, "b")

    # Panel C: arrival-time-vs-distance at pre_onset (axial recruitment) for the representative seed
    axC = fig.add_subplot(gs[0, 2])
    d = load_state(label, seed, "pre_onset")
    fk = d["summary"]["fixed_kick"] if d else {}
    fit = fk.get("arrival_fit", {})
    arr = d["arr"] if d else {}
    if d and "fk_kymo" in arr:
        ky = arr["fk_kymo"]; dist = arr["fk_kymo_dist"]; times = arr["fk_kymo_times"]
        thr = 0.1 * np.nanmax(np.abs(ky))
        arrivals = np.array([times[np.argmax(np.abs(ky[:, p]) >= thr)] if np.any(np.abs(ky[:, p]) >= thr)
                             else np.nan for p in range(ky.shape[1])])
        ok = np.isfinite(arrivals)
        axC.scatter(dist[ok], arrivals[ok], color=PRE, s=20, zorder=3)
        if fit.get("eligible"):
            xs = np.array([dist[ok].min(), dist[ok].max()])
            axC.plot(xs, fit["slope"] * xs + (arrivals[ok].mean() - fit["slope"] * dist[ok].mean()),
                     color=PRE, lw=1.4)
            v = fit.get("velocity_proxy")
            axC.text(0.04, 0.93, f"eligible · R²={fit['r2']:.2f}\nv≈{v:.2f} u/ms", transform=axC.transAxes,
                     fontsize=6.5, va="top")
        else:
            axC.text(0.5, 0.5, "arrival fit\nnot eligible", ha="center", va="center", transform=axC.transAxes,
                     fontsize=8)
    axC.set_xlabel("axial distance (src→sink)", fontsize=7.5)
    axC.set_ylabel("first-arrival time (ms)", fontsize=7.5)
    axC.set_title("pre-onset kick", color=PRE, fontsize=8)
    axC.spines[["top", "right"]].set_visible(False); _letter(axC, "c")

    _save(fig, "figure5_supplementary_2_direct_snn_empirical_modes")


def _save(fig, stem):
    os.makedirs(CAND_DIR, exist_ok=True)
    p = os.path.join(CAND_DIR, stem)
    fig.savefig(p + ".png", dpi=300); fig.savefig(p + ".pdf"); plt.close(fig)
    print(f"[fig] wrote {p}.png/.pdf", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="zA_q75_tz5000")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import yaml
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "topic4_mz_direct_spatial_modes.yaml")))
    figure_supp1(args.label, args.seed)
    figure_supp2(args.label, args.seed, cfg)


if __name__ == "__main__":
    main()
