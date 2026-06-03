# scripts/run_sef_hfo_step0a_stability.py
"""Step 0a: delayed dispersion + n-convergence + low-heterogeneity screen, with the
framework Step-0a output contract (topic4_sef_itp_framework.md:810-822):
phase diagram (+candidate boundary + data-locked family), growth/k* heatmap, gain & low-het shift."""
import argparse, json
from dataclasses import replace
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on path (run as `python scripts/X.py`)
from src.sef_hfo_field import SEFParams
from src.sef_hfo_stability import (self_consistent_operating_point, eta_lin, leading_mode, gain,
                                   build_dispersion_matrix, erlang_n_convergence,
                                   screen_low_heterogeneity_effect)
OUT = Path("results/topic4_sef_hfo/linear_stability")
PHASE_CMAP = ListedColormap(["#3182bd", "#fee08b", "#de2d26"])   # stable / candidate / unstable

def phase_idx(eta, tol=1e-2):
    return 0 if eta > tol else (2 if eta < -tol else 1)

def phase_label(eta, tol=1e-2):
    return ["stable", "candidate_excitable", "unstable"][phase_idx(eta, tol)]

def plot_phase_diagram(p, family, k, out):                       # framework: phase diagram + boundary
    ie = np.linspace(0.1, 0.7, 13); ii = np.linspace(0.05, 0.30, 11)
    M = np.full((len(ii), len(ie)), np.nan)
    for a, I_I in enumerate(ii):
        for b, I_E in enumerate(ie):
            op = self_consistent_operating_point(p, I_E, I_I)
            if op.get("converged"): M[a, b] = phase_idx(eta_lin(p, op, k))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(M, origin="lower", aspect="auto", cmap=PHASE_CMAP, vmin=-0.5, vmax=2.5,
              extent=[ie[0], ie[-1], ii[0], ii[-1]])
    ax.plot([f[0] for f in family], [f[1] for f in family], "ko", ms=7)
    ax.set_xlabel("background drive I_E"); ax.set_ylabel("inhibitory drive I_I"); ax.set_title("Step 0a phase diagram")
    handles = [plt.Rectangle((0, 0), 1, 1, color=PHASE_CMAP(i)) for i in range(3)] + \
              [plt.Line2D([], [], marker="o", color="k", ls="")]
    ax.legend(handles, ["stable", "candidate excitable", "unstable", "data-locked family"], loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "figures" / "step0a_phase_diagram.png", dpi=140); plt.close(fig)

def plot_growth_kmap(p, op, out):                                # framework: max Re(lambda) heatmap + k*
    kk = np.linspace(-3, 3, 61); G = np.empty((len(kk), len(kk)))
    for a, kperp in enumerate(kk):
        for b, kpar in enumerate(kk):
            G[a, b] = np.linalg.eigvals(build_dispersion_matrix(p, op, float(kpar), float(kperp))).real.max()
    a0, b0 = np.unravel_index(np.argmax(G), G.shape); lim = float(np.abs(G).max())
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(G, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                   extent=[kk[0], kk[-1], kk[0], kk[-1]])
    ax.plot(kk[b0], kk[a0], "k*", ms=15, label=f"k*=({kk[b0]:.2f}, {kk[a0]:.2f})")
    ax.set_xlabel("k_parallel (along axis)"); ax.set_ylabel("k_perp (across axis)"); ax.legend(fontsize=8)
    ax.set_title("max Re(lambda) over wavevector"); fig.colorbar(im, ax=ax, label="max Re(lambda)")
    fig.tight_layout(); fig.savefig(out / "figures" / "step0a_growth_kmap.png", dpi=140); plt.close(fig)

def plot_gain_lowhet_shift(p, family, sigma_patch, k, fraction_closer, out):   # framework: gain map (+patch A)
    pp = replace(p, sigma_phi=sigma_patch); rows = []
    for I_E, I_I in family:
        ob = self_consistent_operating_point(p, I_E, I_I); opp = self_consistent_operating_point(pp, I_E, I_I)
        if not (ob.get("converged") and opp.get("converged")): continue
        rows.append((I_E, gain(ob["h_E0"], p.phi_bar, p.sigma_phi, p.beta),
                     gain(opp["h_E0"], pp.phi_bar, pp.sigma_phi, pp.beta),
                     eta_lin(p, ob, k), eta_lin(pp, opp, k)))
    x = [r[0] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(x, [r[1] for r in rows], "o-", label=f"baseline (sigma_phi={p.sigma_phi})")
    ax[0].plot(x, [r[2] for r in rows], "s--", label=f"low-het patch (sigma_phi={sigma_patch})")
    ax[0].set_xlabel("I_E"); ax[0].set_ylabel("population gain G_E"); ax[0].set_title("Gain at operating point"); ax[0].legend(fontsize=8)
    ax[1].plot(x, [r[3] for r in rows], "o-", label="baseline"); ax[1].plot(x, [r[4] for r in rows], "s--", label="low-het patch")
    ax[1].axhline(0, color="k", lw=.8); ax[1].set_xlabel("I_E"); ax[1].set_ylabel("eta_lin (distance to boundary)")
    ax[1].set_title(f"Does lowering heterogeneity drop eta_lin? (not automatic) — closer at {fraction_closer:.0%} of points")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "figures" / "step0a_gain_and_lowhet_shift.png", dpi=140); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/sef_hfo_operating_points.json")
    ap.add_argument("--sigma-phi-patch", type=float, default=0.5)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    fam = [(d["I_E"], d["I_I"]) for d in cfg["operating_points"]]
    p = SEFParams(); k = np.linspace(-3, 3, 61); kc = np.linspace(-3, 3, 31)   # kc: coarser for the 2-D phase grid
    rows = []
    for I_E, I_I in fam:
        op = self_consistent_operating_point(p, I_E, I_I)
        if not op.get("converged"):
            rows.append({"I_E": I_E, "I_I": I_I, "converged": False}); continue
        eta = eta_lin(p, op, k); lm = leading_mode(p, op, k)
        rows.append({"I_E": I_E, "I_I": I_I, "converged": True, "bistable": op["bistable"],
                     "eta_lin": eta, "phase": phase_label(eta), "k_star": lm["k_star"], "omega_star": lm["omega_star"]})
    conv = erlang_n_convergence(p, self_consistent_operating_point(p, *fam[len(fam) // 2]), k)
    screen = screen_low_heterogeneity_effect(p, fam, args.sigma_phi_patch, k)
    candidates = [{"I_E": r["I_E"], "I_I": r["I_I"]} for r in rows if r.get("phase") == "candidate_excitable"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "step0a_stability.json").write_text(json.dumps(
        {"provenance": cfg["provenance"], "sigma_phi_baseline": p.sigma_phi, "sigma_phi_patch": args.sigma_phi_patch,
         "per_point": rows, "erlang_n_convergence": conv, "low_heterogeneity_screen": screen,
         "candidate_operating_points": candidates}, indent=2, default=float))
    plot_phase_diagram(p, fam, kc, OUT)
    rep = (candidates[0]["I_E"], candidates[0]["I_I"]) if candidates else fam[len(fam) // 2]
    plot_growth_kmap(p, self_consistent_operating_point(p, rep[0], rep[1]), OUT)
    plot_gain_lowhet_shift(p, fam, args.sigma_phi_patch, k, screen["fraction_closer"], OUT)
    print(f"[step0a] fraction_closer={screen['fraction_closer']:.3f}; "
          f"erlang converged={conv['converged']} rec_n={conv['recommended_n']}; candidates={len(candidates)}")

if __name__ == "__main__":
    main()
