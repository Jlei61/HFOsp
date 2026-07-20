"""Paper-ready candidate figures for Topic 4 MZ direct-SNN spatial mode dynamics.

Design contract: docs/superpowers/specs/2026-07-19-topic4-mz-direct-spatial-modes-design.md §8/§11
(+ review 2026-07-20). Reads results/topic4_sef_hfo/mz_direct_spatial_modes/ and renders:
  Supplementary 1  fixed-kick STATE-CONDITIONED finite-time susceptibility
  Supplementary 2  operator IDENTIFIABILITY diagnostic + identified low-k modes (where identifiable)

This is the direct current-based MZ spiking network — NOT a rate-field surrogate, NOT exact
eigenmodes. The empirical operator is identifiable (ensemble / RMS-matched / low-k audit) only at
some (seed,state) — chiefly the intermediate-activity midpoint; V1/U1 are shown only where the audit
passes the linearity gate. Fail-closed when a required sidecar is missing (spec §7 C13). Never
overwrite the rate-field Figure 5 directory (spec §8).

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
STATE_SHORT = {"baseline": "base", "midpoint": "mid", "pre_onset": "pre"}
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


def _grididx(xy, n):
    return (xy[0] / 5.0 + 0.5) * (n - 1), (xy[1] / 5.0 + 0.5) * (n - 1)


def _signed_field(ax, field, *, vmax, title=None, src=None, snk=None):
    f = np.asarray(field, float)
    im = ax.imshow(f.T, origin="lower", cmap=SIGNED, vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8, pad=2)
    n = f.shape[0]
    for xy, mk in ((src, "o"), (snk, "s")):
        if xy is not None:
            gx, gy = _grididx(xy, n)
            ax.plot(gx, gy, mk, mfc="none", mec="k", mew=0.9, ms=5)
    return im


# =============================================================== Supplementary 1: fixed-kick susceptibility
def figure_supp1(label, seed):
    sts = ["baseline", "pre_onset"]
    data = {s: load_state(label, seed, s) for s in sts}
    if any(data[s] is None for s in sts):
        raise SystemExit(f"[fig1] missing fixed-kick sidecars for seed{seed} {sts}")
    centers = data["baseline"]["arr"]["fk_map_centers"].tolist()
    src = data["baseline"]["summary"].get("src_g"); snk = data["baseline"]["summary"].get("snk_g")
    dmax = max(np.nanmax(np.abs(d["arr"]["fk_dmaps"])) for d in data.values()) or 1.0
    kmax = max(np.nanmax(np.abs(d["arr"]["fk_kymo"])) for d in data.values()) or 1.0

    fig = plt.figure(figsize=(7.2, 6.0))
    gs_top = GridSpec(2, 5, figure=fig, width_ratios=[1, 1, 1, 1, 0.9], hspace=0.28, wspace=0.22,
                      left=0.07, right=0.965, top=0.95, bottom=0.44)
    im = None
    for ri, st in enumerate(sts):
        dmaps = data[st]["arr"]["fk_dmaps"]
        for ci, c in enumerate(centers[:4]):
            ax = fig.add_subplot(gs_top[ri, ci])
            im = _signed_field(ax, dmaps[ci], vmax=dmax, title=(f"{int(c)} ms" if ri == 0 else None),
                               src=src, snk=snk)
            if ci == 0:
                ax.set_ylabel(STATE_LABEL[st], color=STATE_COLOR[st], fontsize=9, fontweight="bold")
            if ri == 0 and ci == 0:
                _letter(ax, "a")
    cax = fig.add_subplot(gs_top[0:2, 4]); cax.set_axis_off()
    fig.colorbar(im, ax=cax, fraction=0.55, pad=0.0).set_label("Δ E-rate (Hz)\nkick − control", fontsize=7)

    # bottom row: kymographs (shared cbar) + corridor magnitude + arrival — own column layout
    gs_bot = GridSpec(1, 6, figure=fig, width_ratios=[1, 1, 0.09, 0.72, 0.18, 1.5], wspace=0.32,
                      left=0.07, right=0.965, top=0.33, bottom=0.1)
    cmap_k = plt.get_cmap(MAG).copy(); cmap_k.set_bad(plt.get_cmap(MAG)(0.0))
    imk = None
    for ci, st in enumerate(sts):
        ax = fig.add_subplot(gs_bot[0, ci])
        ky = data[st]["arr"]["fk_kymo"]
        ext = [data[st]["arr"]["fk_kymo_times"][0], data[st]["arr"]["fk_kymo_times"][-1],
               0, data[st]["arr"]["fk_kymo_dist"][-1]]
        imk = ax.imshow(np.ma.masked_invalid(np.abs(ky).T), origin="lower", aspect="auto",
                        cmap=cmap_k, vmin=0, vmax=kmax, extent=ext)
        ax.set_xlabel("time (ms)", fontsize=7.5)
        ax.set_title(STATE_LABEL[st], color=STATE_COLOR[st], fontsize=8)
        if ci == 0:
            ax.set_ylabel("axial dist.\n(src→sink)", fontsize=7.5); _letter(ax, "b")
    ckax = fig.add_subplot(gs_bot[0, 2])
    fig.colorbar(imk, cax=ckax).set_label("|Δrate| (Hz)", fontsize=6.5)

    # axial-corridor response magnitude, baseline vs pre_onset, all seeds (robust reproducible finding)
    axC = fig.add_subplot(gs_bot[0, 3])
    for xi, st in enumerate(["baseline", "pre_onset"]):
        vals = [load_state(label, s, st)["summary"]["fixed_kick"]["region"]["axis_corridor"]
                for s in seeds_present(label) if load_state(label, s, st)]
        axC.scatter([xi] * len(vals), vals, color=STATE_COLOR[st], s=20, zorder=3)
        if vals:
            axC.plot([xi - 0.28, xi + 0.28], [np.mean(vals)] * 2, color=STATE_COLOR[st], lw=1.8)
    axC.set_xticks(range(2)); axC.set_xticklabels(["base", "pre"], fontsize=7.5); axC.set_xlim(-0.6, 1.6)
    axC.set_ylabel("axial-corridor\n|Δ E-rate| (Hz)", fontsize=7.5)
    axC.spines[["top", "right"]].set_visible(False); _letter(axC, "c")

    # arrival-time vs distance at pre_onset (axial recruitment) for the representative seed
    axD = fig.add_subplot(gs_bot[0, 5])
    d = load_state(label, seed, "pre_onset")
    fit = d["summary"]["fixed_kick"].get("arrival_fit", {})
    arr = d["arr"]
    ky = arr["fk_kymo"]; dist = arr["fk_kymo_dist"]; times = arr["fk_kymo_times"]
    thr = 0.1 * np.nanmax(np.abs(ky))
    arrivals = np.array([times[np.argmax(np.abs(ky[:, p]) >= thr)] if np.any(np.abs(ky[:, p]) >= thr)
                         else np.nan for p in range(ky.shape[1])])
    ok = np.isfinite(arrivals)
    axD.scatter(dist[ok], arrivals[ok], color=PRE, s=20, zorder=3)
    if fit.get("eligible"):
        xs = np.array([dist[ok].min(), dist[ok].max()])
        axD.plot(xs, fit["slope"] * xs + (arrivals[ok].mean() - fit["slope"] * dist[ok].mean()), color=PRE, lw=1.4)
        axD.text(0.04, 0.93, f"eligible · R²={fit['r2']:.2f}", transform=axD.transAxes, fontsize=6.8, va="top")
    axD.set_xlabel("axial distance (src→sink)", fontsize=7.5)
    axD.set_ylabel("first-arrival (ms)", fontsize=7.5)
    axD.set_title("pre-onset kick", color=PRE, fontsize=8)
    axD.spines[["top", "right"]].set_visible(False); _letter(axD, "d")

    _save(fig, "figure5_supplementary_1_direct_snn_spatial_response")


# =============================================================== Supplementary 2: identifiability + modes
def figure_supp2(label, cfg):
    ca_path = os.path.join(OUT, "corrected_audit_summary.json")
    if not os.path.exists(ca_path):
        raise SystemExit("[fig2] missing corrected_audit_summary.json")
    ca = json.load(open(ca_path))
    tol = float(ca["tol"])
    rows = ca["rows"]
    # pick the cleanest identifiable state (lowest discrepancy) for the mode fields
    ident = [r for r in rows if r["identifiable"]]
    best = min(ident, key=lambda r: r["discrepancy"]) if ident else None

    fig = plt.figure(figsize=(7.2, 3.1))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.35, 1.0, 1.0], wspace=0.42,
                  left=0.085, right=0.965, top=0.86, bottom=0.2)

    # Panel A: corrected identifiability map — discrepancy per state x seed vs the 15% gate
    axA = fig.add_subplot(gs[0, 0])
    order = ["baseline", "midpoint", "pre_onset"]
    for xi, st in enumerate(order):
        vals = [r["discrepancy"] for r in rows if r["state"] == st]
        idf = [r["identifiable"] for r in rows if r["state"] == st]
        for v, i in zip(vals, idf):
            axA.scatter([xi], [v], s=30, zorder=3, facecolor=(STATE_COLOR[st] if i else "none"),
                        edgecolor=STATE_COLOR[st], linewidth=1.2)
    axA.axhline(tol, color="k", lw=0.9, ls="--")
    axA.text(2.15, tol, "15% gate", fontsize=6.6, va="center", ha="left")
    axA.text(0.02, 0.97, "ensemble · RMS-matched · low-k\n(orig. thin-input audit: 0.5–2.6, all states)",
             transform=axA.transAxes, fontsize=6.2, va="top")
    axA.set_xticks(range(3)); axA.set_xticklabels([STATE_SHORT[s] for s in order], fontsize=7.5)
    axA.set_xlim(-0.4, 2.4); axA.set_ylim(0, max(0.5, max(r["discrepancy"] for r in rows) * 1.1))
    axA.set_ylabel("linearity discrepancy\n(filled = identifiable)", fontsize=7.5)
    axA.spines[["top", "right"]].set_visible(False); _letter(axA, "a")

    # Panels B/C: identified V1 input + U1 output at the cleanest identifiable state
    Tmid = int(round(cfg["T_windows_ms"][1]))
    v1 = u1 = None
    if best is not None:
        npz = os.path.join(OUT, "per_seed", f"corrected_audit_arrays_{label}_seed{best['seed']}_{best['state']}.npz")
        if os.path.exists(npz):
            a = np.load(npz)
            v1 = a.get(f"corr_v1_T{Tmid}"); u1 = a.get(f"corr_u1_T{Tmid}")
    st0 = load_state(label, best["seed"], best["state"]) if best else None
    src = st0["summary"].get("src_g") if st0 else None
    snk = st0["summary"].get("snk_g") if st0 else None
    lab = f"{STATE_LABEL[best['state']]} seed{best['seed']}" if best else ""
    for ax_i, (kind, field) in enumerate([("V₁ input", v1), ("U₁ output", u1)]):
        ax = fig.add_subplot(gs[0, 1 + ax_i])
        if field is not None:
            vmax = np.nanmax(np.abs(field)) or 1.0
            _signed_field(ax, field, vmax=vmax, src=src, snk=snk)
            ax.set_title(f"{kind}\n{lab}", fontsize=7.5, color=MID)
        else:
            ax.set_axis_off(); ax.text(0.5, 0.5, "no identifiable\noperator", ha="center", va="center", fontsize=8)
        _letter(ax, "b" if ax_i == 0 else "c")

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
    figure_supp2(args.label, cfg)


if __name__ == "__main__":
    main()
