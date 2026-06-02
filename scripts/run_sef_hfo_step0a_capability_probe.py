"""Step 0a capability probe: does the dispersion machinery STRUCTURALLY support the
framework's three-region (stable/candidate/unstable) phase structure AND a finite-k
(propagation-axis) leading mode -- as opposed to only global k=0 instability?

This backs the step0_results writeup's capability claim with a reproducible artifact.
It sweeps CONNECTIVITY regimes (short-range excitation / long-range inhibition, the
'Mexican-hat' ingredient) over a background-drive grid and reports, per regime:
  - is eta_lin<0 reachable (an unstable region exists)?
  - is a finite-k leading mode (k_star>0) reachable, and beating k=0?
  - is a finite-k *Hopf* (k_star>0 AND omega_star>0, i.e. a traveling wave) reachable?
The SCAFFOLD regime is the locked SEFParams default (test-only); the others are
clearly-labeled exploratory connectivity, NOT a re-lock of any default.

Run: python scripts/run_sef_hfo_step0a_capability_probe.py
"""
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on path
from src.sef_hfo_field import SEFParams
from src.sef_hfo_stability import self_consistent_operating_point, eta_lin, leading_mode

OUT = Path("results/topic4_sef_hfo/linear_stability")

# Connectivity regimes: scaffold (locked default) + progressively stronger Mexican-hat
# (narrower excitation ell, wider/stronger inhibition sigma_I/J_EI). EXPLORATORY, not locked.
REGIMES = {
    "scaffold_default": dict(),
    "moderate_hat": dict(ell_par=2.0, ell_perp=1.0, sigma_I=12.0, J_EI=2.0, J_IE=2.0, sigma_phi=0.3),
    "aggressive_hat": dict(ell_par=1.0, ell_perp=0.4, sigma_I=18.0, J_EE=2.0, J_EI=3.0, beta=6.0, sigma_phi=0.3),
}
IE_GRID = np.linspace(0.1, 1.5, 15)
II_GRID = np.linspace(0.05, 0.40, 8)
K = np.linspace(-3, 3, 61)


def probe_regime(kw):
    p = replace(SEFParams(), **kw)
    rows = []
    for I_E in IE_GRID:
        for I_I in II_GRID:
            op = self_consistent_operating_point(p, float(I_E), float(I_I))
            if not op.get("converged"):
                continue
            eta = eta_lin(p, op, K)
            lm = leading_mode(p, op, K)
            rows.append({"I_E": float(I_E), "I_I": float(I_I), "eta_lin": float(eta),
                         "k_star": float(lm["k_star"]), "omega_star": float(lm["omega_star"]),
                         "max_re": float(lm["max_re"])})
    unstable = [r for r in rows if r["eta_lin"] < 0]
    finite_k = [r for r in rows if abs(r["k_star"]) > 1e-6]
    finite_k_unstable = [r for r in finite_k if r["max_re"] > 0]
    finite_k_hopf = [r for r in finite_k_unstable if r["omega_star"] > 1e-6]
    # the single most illustrative finite-k point (max k_star among unstable finite-k, else any finite-k)
    pool = finite_k_unstable or finite_k
    example = max(pool, key=lambda r: r["max_re"]) if pool else None
    return {"n_converged": len(rows),
            "eta_lt0_reachable": bool(unstable), "n_eta_lt0": len(unstable),
            "finite_k_leading_reachable": bool(finite_k), "n_finite_k": len(finite_k),
            "finite_k_unstable_reachable": bool(finite_k_unstable), "n_finite_k_unstable": len(finite_k_unstable),
            "finite_k_hopf_reachable": bool(finite_k_hopf), "n_finite_k_hopf": len(finite_k_hopf),
            "illustrative_finite_k_point": example}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"note": "EXPLORATORY connectivity sweep; only scaffold_default is the locked SEFParams. "
                   "Backs the step0_results capability claim.",
           "regimes": {name: probe_regime(kw) for name, kw in REGIMES.items()},
           "regime_params": {name: kw for name, kw in REGIMES.items()}}
    (OUT / "capability_probe.json").write_text(json.dumps(out, indent=2, default=float))
    for name, r in out["regimes"].items():
        ex = r["illustrative_finite_k_point"]
        exs = (f"k*={ex['k_star']:.2f} omega*={ex['omega_star']:.2f} maxRe={ex['max_re']:.3f} "
               f"@(I_E={ex['I_E']:.2f},I_I={ex['I_I']:.2f})") if ex else "none"
        print(f"[{name}] n={r['n_converged']} eta<0:{r['n_eta_lt0']} "
              f"finite-k:{r['n_finite_k']} finite-k-unstable:{r['n_finite_k_unstable']} "
              f"finite-k-Hopf:{r['n_finite_k_hopf']} | example: {exs}")


if __name__ == "__main__":
    main()
