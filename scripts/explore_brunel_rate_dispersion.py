"""Exploration 1 (2026-06-03): does the RATE-level dispersion REPRODUCE & EXPLAIN the
mechanism of coworker1's empirical Brunel-LIF traveling beta wave?

Ground truth = coworker1's spiking sim (finite-k ~29 Hz traveling wave at Brunel
Table-1 params). This is a cheap rate-reduction CONFIRMATION/EXPLANATION, NOT an
independent quantitative predictor (single-pole synaptic filter, static injected gain,
Gaussian approx of the rho=0.6 elliptical kernel, constant/zero delay -- all approximate).

Reuses the SEF-HFO rate-dispersion structure (src.sef_hfo_stability) with Brunel-mapped
params. Reports:
  (1) gain scan (diagonal G_E=G_I): leading mode k*, frequency near criticality;
  (2) MECHANISM CONTROL: slow GABA (tau_d=18) vs scaffold-like fast GABA (tau_d=2);
  (3) gain-ratio robustness G_I/G_E in {0.5,1,2} (guards the diagonal-scan blind spot);
  (4) Re lam(k) profile (shows the finite-k peak is SHALLOW; k=0 nearly co-unstable).

Run: python scripts/explore_brunel_rate_dispersion.py
"""
import json
from pathlib import Path
import numpy as np
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scipy.optimize import fsolve
from src.sef_hfo_stability import gaussian_hat

OUT = Path("results/topic4_sef_hfo/linear_stability")

# --- Brunel Table-1 (coworker1 params.py) mapped to the 2x2 rate dispersion ---
TAU_E, TAU_I, TAU_AMPA = 20.0, 10.0, 3.5     # ms (membrane E/I; AMPA decay)
# weight ratios J_ab = C_ab * w_ab, normalized to J_EE
JEE = 1.0
JIE = (800 * 0.2625) / (800 * 0.1575)            # 1.667
JEI = (200 * 1.07 * 3.6 * 0.1575) / (800 * 0.1575)  # 0.963
JII = (200 * 3.6 * 0.2625) / (800 * 0.1575)      # 1.500
ELL_PAR, ELL_PERP = 0.54, 0.27   # E->E anisotropic (l_EE=0.38mm, rho_EE=0.6 -> ratio ~2; Gaussian approx)
L_INH = 0.25                     # inhibitory widths (l_EI=l_IE=l_II=0.25mm), isotropic


def char_det(lam, kpar, GE, GI, tau_gaba, d_const):
    def H(tau):
        return np.exp(-lam * d_const) / (1.0 + lam * tau)
    WEE = JEE * gaussian_hat(kpar, 0.0, ELL_PAR, ELL_PERP)
    WEI = JEI * gaussian_hat(kpar, 0.0, L_INH, L_INH)
    WIE = JIE * gaussian_hat(kpar, 0.0, L_INH, L_INH)
    WII = JII * gaussian_hat(kpar, 0.0, L_INH, L_INH)
    a = (1 + TAU_E * lam) - GE * WEE * H(TAU_AMPA)
    b = GE * WEI * H(tau_gaba)
    c = -GI * WIE * H(TAU_AMPA)
    dd = (1 + TAU_I * lam) + GI * WII * H(tau_gaba)
    return a * dd - b * c


def rightmost(kpar, GE, GI, tau_gaba=18.0, d_const=1.3, re_lo=-1.0, re_hi=0.5, im_hi=0.7, n=14):
    """Rightmost root of char_det over a bounded complex box (multi-start fsolve)."""
    best = (-np.inf, 0.0)
    def fr(v):
        z = char_det(complex(v[0], v[1]), kpar, GE, GI, tau_gaba, d_const)
        return [z.real, z.imag]
    for r0 in np.linspace(re_lo, re_hi, n):
        for i0 in np.linspace(0.0, im_hi, n):
            sol, _, ier, _ = fsolve(fr, [r0, i0], full_output=True)
            if ier != 1:
                continue
            re, im = float(sol[0]), float(sol[1])
            if (re_lo - 1e-6 <= re <= re_hi + 1e-6 and abs(im) <= im_hi + 1e-6
                    and abs(complex(*fr(sol))) < 1e-7 and re > best[0]):
                best = (re, abs(im))
    return best


def leading_over_k(GE, GI, tau_gaba=18.0, d_const=1.3, kmax=3.5, nk=29):
    kk = np.linspace(0.0, kmax, nk)
    prof = [rightmost(float(k), GE, GI, tau_gaba, d_const) for k in kk]
    re = np.array([p[0] for p in prof]); im = np.array([p[1] for p in prof])
    j = int(np.argmax(re))
    return {"k_grid": kk.tolist(), "re": re.tolist(), "im": im.tolist(),
            "k_star": float(kk[j]), "re_max": float(re[j]), "im_at_star": float(im[j]),
            "freq_hz": float(1000.0 * im[j] / (2 * np.pi)),
            "re_at_k0": float(re[0]), "im_at_k0": float(im[0]),
            "finite_k_preferred": bool(kk[j] > 1e-3 and re[j] > re[0] + 1e-3),
            "is_hopf": bool(im[j] > 1e-3)}


def main():
    out = {"provenance": "EXPLORATORY rate-reduction of coworker1 Brunel-LIF params; "
                         "approximate (single-pole synapse, static gain, Gaussian kernel). "
                         "Confirms/explains mechanism; ground truth = coworker1 spiking sim."}

    # (1) diagonal gain scan
    scan = []
    for g in [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]:
        lo = leading_over_k(g, g)
        scan.append({"g": g, "re_max": lo["re_max"], "k_star": lo["k_star"],
                     "freq_hz": lo["freq_hz"], "is_hopf": lo["is_hopf"],
                     "finite_k_preferred": lo["finite_k_preferred"]})
    out["gain_scan_diagonal"] = scan

    # (2) mechanism control: slow vs fast GABA at fixed gain
    out["mechanism_control"] = {
        "slow_gaba_tau18": leading_over_k(4.0, 4.0, tau_gaba=18.0),
        "fast_gaba_tau2": leading_over_k(4.0, 4.0, tau_gaba=2.0),
    }

    # (3) gain-ratio robustness (guard diagonal-scan blind spot): vary G_I/G_E
    ratios = {}
    GE = 4.0
    for r in [0.5, 1.0, 2.0]:
        lo = leading_over_k(GE, GE * r)
        ratios[f"GI/GE={r}"] = {"k_star": lo["k_star"], "freq_hz": lo["freq_hz"],
                                "is_hopf": lo["is_hopf"], "re_max": lo["re_max"],
                                "finite_k_preferred": lo["finite_k_preferred"]}
    out["gain_ratio_robustness"] = ratios

    # (4) profile at a representative near-critical setting
    out["profile_g4_diagonal"] = leading_over_k(4.0, 4.0)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exploration1_brunel_dispersion.json").write_text(json.dumps(out, indent=2, default=float))

    print("=== (1) diagonal gain scan (G_E=G_I=g), slow GABA tau=18, d=1.3 ===")
    for s in scan:
        print(f"  g={s['g']:.1f}: reMax={s['re_max']:+.4f} k*={s['k_star']:.2f} "
              f"freq={s['freq_hz']:.1f}Hz hopf={s['is_hopf']} finite_k_pref={s['finite_k_preferred']}")
    mc = out["mechanism_control"]
    print("=== (2) mechanism control @g=4 ===")
    print(f"  slow GABA(18): k*={mc['slow_gaba_tau18']['k_star']:.2f} freq={mc['slow_gaba_tau18']['freq_hz']:.1f}Hz "
          f"hopf={mc['slow_gaba_tau18']['is_hopf']}")
    print(f"  fast GABA(2):  k*={mc['fast_gaba_tau2']['k_star']:.2f} freq={mc['fast_gaba_tau2']['freq_hz']:.1f}Hz "
          f"hopf={mc['fast_gaba_tau2']['is_hopf']}  <- scaffold-like: Hopf should vanish")
    print("=== (3) gain-ratio robustness (G_I/G_E) ===")
    for k, v in ratios.items():
        print(f"  {k}: k*={v['k_star']:.2f} freq={v['freq_hz']:.1f}Hz hopf={v['is_hopf']} finite_k_pref={v['finite_k_preferred']}")
    pr = out["profile_g4_diagonal"]
    print(f"=== (4) profile @g=4: Re(k0)={pr['re_at_k0']:+.4f} -> Re(k*={pr['k_star']:.2f})={pr['re_max']:+.4f} "
          f"(swing {pr['re_max']-pr['re_at_k0']:+.4f}; shallow => long-wavelength, k=0 nearly co-unstable) ===")


if __name__ == "__main__":
    main()
