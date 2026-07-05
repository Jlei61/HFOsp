#!/usr/bin/env python3
"""Linear-algebra Λ_eff(μ) screen on SYNTHETIC propagation operators (spec C2).

WHAT THIS IS (and IS NOT):
- IS: the cheap linear-algebra predictor the spec calls a screen (C2 "线代筛查/便宜预测"),
  run on SYNTHETIC anisotropic operators. Pure NumPy. No SNN engine. Not the gated
  pipeline. Commits nothing. Freezes nothing.
- IS NOT: an SNN dynamics result, the calibration, or the Λ₀×μ phase map (Task 7). Those
  run on MEASURED W and are gated behind the user's prereg-freeze decision.

PURPOSE:
  (a) validate that the W_step / h / Λ₀ + D_μ(h) prediction layer behaves sensibly;
  (b) build intuition for how the slow permissivity μ pushes the recruitment branching
      ratio Λ_eff = ρ(D_μ(h)·W_step) across 1 (sub- -> super-critical);
  (c) preview the headline C5 control at the cheap layer: does SHAPED h (real coupling)
      reach Λ_eff=1 at a different μ than UNIFORM-μ (h_eff=1) or SHUFFLED-h?
      A real difference at the SNN layer is the actual test; this only screens it.

Run: python3 scripts/explore_lambda_eff_linalg_screen.py
"""
from __future__ import annotations
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, ROOT)
from src.topic4_propagation_operator import make_step_operator, h_field, spectral_radius


def synth_w_resp(n_per_axis=6, ar=2.0, theta_deg=45.0, J=1.0, sigma=1.3):
    """Synthetic off-diagonal W_resp on an n×n bin grid with local anisotropic coupling.

    Source q recruits target p with weight J·exp(-½ dᵀ Σ⁻¹ d) where d = center[p]-center[q]
    is rotated into the axis frame and the along-axis std is `ar`× the cross-axis std
    (events travel preferentially along θ). Diagonal is 0 (self-response handled by
    `src_mass`, mirroring build_w_resp). Returns (W_resp_offdiag, src_mass, centers).
    """
    xs, ys = np.meshgrid(np.arange(n_per_axis), np.arange(n_per_axis))
    centers = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    n = centers.shape[0]
    th = np.radians(theta_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    # along-axis variance larger by ar^2 -> elongated coupling along θ
    inv_cov = R @ np.diag([1.0 / (ar * sigma) ** 2, 1.0 / sigma ** 2]) @ R.T
    W = np.zeros((n, n))
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            d = centers[p] - centers[q]
            W[p, q] = J * np.exp(-0.5 * d @ inv_cov @ d)
    src_mass = np.full(n, 1.0)  # fixed source self-response (the Λ₀ knob is J on off-diag)
    return W, src_mass, centers


def _tune_J_for_lambda0(target, **kw):
    """Bisection on J so that ρ(W_step) ≈ target (Λ₀ knob = off-diagonal gain)."""
    lo, hi = 1e-3, 50.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        W, sm, _ = synth_w_resp(J=mid, **kw)
        rho = spectral_radius(make_step_operator(W, sm))
        if rho < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def lambda_eff(W_step, h_eff, mu):
    """Λ_eff(μ) = ρ(diag(1+μ·h_eff)·W_step) — μ amplifies each target row by its h."""
    D = np.diag(1.0 + mu * np.asarray(h_eff, float))
    return spectral_radius(D @ W_step)


def first_crossing(mus, lam, level=1.0):
    """First μ where Λ_eff crosses `level` (linear interp); None if never."""
    lam = np.asarray(lam, float)
    for i in range(1, len(lam)):
        if lam[i - 1] < level <= lam[i]:
            t = (level - lam[i - 1]) / (lam[i] - lam[i - 1] + 1e-12)
            return float(mus[i - 1] + t * (mus[i] - mus[i - 1]))
    return None


def main():
    out_dir = os.path.join(ROOT, "results/topic4_sef_hfo/m3_local_w/lambda_eff_linalg_screen")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    rng = np.random.default_rng(0)
    lambda0_targets = [0.70, 0.85, 0.95]
    mus = np.linspace(0.0, 2.0, 41)
    grid_kw = dict(n_per_axis=6, ar=2.0, theta_deg=45.0, sigma=1.3)

    results = {"note": "linear-algebra screen on synthetic operators; NOT an SNN result",
               "grid": grid_kw, "mu_grid": [float(m) for m in mus], "levels": []}
    curves = {}

    n_shuffle = 20
    for tgt in lambda0_targets:
        J = _tune_J_for_lambda0(tgt, **grid_kw)
        W, sm, _ = synth_w_resp(J=J, **grid_kw)
        W_step = make_step_operator(W, sm)
        lam0 = spectral_radius(W_step)
        h = h_field(W, "post")
        level_rec = {"lambda0_target": tgt, "lambda0_actual": float(lam0),
                     "J": float(J), "h_cv": float(np.std(h) / (np.mean(h) + 1e-12)),
                     "n_shuffle": n_shuffle, "mu_star": {}}
        curves[tgt] = {}
        for name, h_eff in (("shaped", h), ("uniform", np.ones_like(h))):
            lam = [lambda_eff(W_step, h_eff, m) for m in mus]
            curves[tgt][name] = lam
            level_rec["mu_star"][name] = first_crossing(mus, lam)
        # shuffled: a DISTRIBUTION over n_shuffle permutations (a single draw is noisy and
        # could spuriously beat/lose to shaped). Report mean + min/max band.
        shuf = np.array([[lambda_eff(W_step, h[rng.permutation(len(h))], m) for m in mus]
                         for _ in range(n_shuffle)])
        curves[tgt]["shuffled_mean"] = shuf.mean(0).tolist()
        curves[tgt]["shuffled_lo"] = shuf.min(0).tolist()
        curves[tgt]["shuffled_hi"] = shuf.max(0).tolist()
        shuf_mu = [m for m in (first_crossing(mus, row) for row in shuf) if m is not None]
        level_rec["mu_star"]["shuffled_mean"] = float(np.mean(shuf_mu)) if shuf_mu else None
        level_rec["mu_star"]["shuffled_min"] = float(np.min(shuf_mu)) if shuf_mu else None
        level_rec["mu_star"]["shuffled_max"] = float(np.max(shuf_mu)) if shuf_mu else None
        results["levels"].append(level_rec)

    with open(os.path.join(out_dir, "lambda_eff_screen_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- figure: Λ_eff vs μ, one panel per Λ₀ level, lines per h-control ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(lambda0_targets), figsize=(4.2 * len(lambda0_targets), 3.6),
                             sharey=True)
    colors = {"shaped": "#1f77b4", "uniform": "#7f7f7f", "shuffled": "#d62728"}
    for ax, rec in zip(axes, results["levels"]):
        tgt = rec["lambda0_target"]
        for name in ("shaped", "uniform"):
            ax.plot(mus, curves[tgt][name], color=colors[name], lw=2, label=name)
            ms = rec["mu_star"][name]
            if ms is not None:
                ax.axvline(ms, color=colors[name], ls=":", lw=1, alpha=0.7)
        # shuffled: mean curve + min/max band over n_shuffle permutations
        ax.plot(mus, curves[tgt]["shuffled_mean"], color=colors["shuffled"], lw=1.5,
                label=f"shuffled (mean/{rec['n_shuffle']})")
        ax.fill_between(mus, curves[tgt]["shuffled_lo"], curves[tgt]["shuffled_hi"],
                        color=colors["shuffled"], alpha=0.15)
        ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.6)
        ax.set_title(f"Λ₀ ≈ {rec['lambda0_actual']:.2f}")
        ax.set_xlabel("μ (permissivity)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Λ_eff = ρ(D_μ(h)·W_step)")
    axes[-1].legend(frameon=False, fontsize=8, title="h field")
    fig.suptitle("Linear-algebra screen (synthetic operators, NOT an SNN result): "
                 "μ pushes Λ_eff across 1", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "lambda_eff_vs_mu.png"), dpi=130)
    plt.close(fig)

    # ---- C5 power preview: does the shaped-vs-uniform μ* gap grow with h heterogeneity? ----
    # At fixed Λ₀≈0.85, vary the coupling reach (sigma) to span a range of recruitment-map
    # heterogeneity (h CV) and record the (uniform − shaped) μ* gap. Intuition for whether the
    # C5 control has POWER given the expected real-h heterogeneity (a homogeneous h would make
    # shaped/uniform/shuffled hard to separate even at the SNN layer).
    cv_rows = []
    for sig in [0.7, 1.0, 1.3, 1.8, 2.5]:
        kw = dict(n_per_axis=6, ar=2.0, theta_deg=45.0, sigma=sig)
        Jc = _tune_J_for_lambda0(0.85, **kw)
        Wc, smc, _ = synth_w_resp(J=Jc, **kw)
        Wsc = make_step_operator(Wc, smc)
        hc = h_field(Wc, "post")
        lam_sh = [lambda_eff(Wsc, hc, m) for m in mus]
        lam_un = [lambda_eff(Wsc, np.ones_like(hc), m) for m in mus]
        ms_sh = first_crossing(mus, lam_sh)
        ms_un = first_crossing(mus, lam_un)
        gap = (ms_un - ms_sh) if (ms_sh is not None and ms_un is not None) else None
        cv_rows.append({"sigma": sig, "lambda0": float(spectral_radius(Wsc)),
                        "h_cv": float(np.std(hc) / (np.mean(hc) + 1e-12)),
                        "mu_star_shaped": ms_sh, "mu_star_uniform": ms_un, "gap": gap})
    results["cv_sensitivity"] = {"lambda0_target": 0.85, "rows": cv_rows}
    with open(os.path.join(out_dir, "lambda_eff_screen_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    fig2, ax2 = plt.subplots(figsize=(4.8, 3.6))
    cvs = [r["h_cv"] for r in cv_rows if r["gap"] is not None]
    gaps = [r["gap"] for r in cv_rows if r["gap"] is not None]
    ax2.plot(cvs, gaps, "o-", color="#1f77b4", lw=2)
    ax2.axhline(0.0, color="k", ls="--", lw=1, alpha=0.5)
    ax2.set_xlabel("h heterogeneity (CV of recruitment map)")
    ax2.set_ylabel("μ* gap  (uniform − shaped)")
    ax2.set_title("C5 power preview @ Λ₀≈0.85\n(synthetic operators, NOT an SNN result)", fontsize=9)
    ax2.grid(alpha=0.25)
    fig2.tight_layout()
    fig2.savefig(os.path.join(fig_dir, "c5_gap_vs_h_cv.png"), dpi=130)
    plt.close(fig2)

    # console summary
    print("Λ_eff linear-algebra screen (synthetic; NOT an SNN result):")
    for rec in results["levels"]:
        ms = rec["mu_star"]
        print(f"  Λ₀≈{rec['lambda0_actual']:.3f} (h CV={rec['h_cv']:.2f}): "
              f"μ* shaped={_fmt(ms['shaped'])} uniform={_fmt(ms['uniform'])} "
              f"shuffled(mean/{rec['n_shuffle']})={_fmt(ms['shuffled_mean'])} "
              f"[{_fmt(ms['shuffled_min'])},{_fmt(ms['shuffled_max'])}]")
    print(f"Wrote {out_dir}/lambda_eff_screen_summary.json + figures/lambda_eff_vs_mu.png")


def _fmt(x):
    return "none" if x is None else f"{x:.3f}"


if __name__ == "__main__":
    main()
