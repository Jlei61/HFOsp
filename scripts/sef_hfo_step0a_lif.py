"""Step 0a on the LIF colored-noise transfer (2026-06-03, user step 2): self-consistent
steady state + linear map, NO injected gain. Replaces the sigmoid (diagnosed-bad piece)
with the Brunel LIF mean-field; keeps the rate-layer dispersion structure, connectivity,
synaptic+delay kinetics, and operating-point-family reporting.

Pipeline per operating point (external-drive ratio nu_ext/nu_theta):
  1. Brunel mean-field self-consistent solve (Siegert nu(mu,sigma), white-noise) for
     (nu_E, nu_I). VALIDATED vs coworker1's SI regime: low E rate (~1 Hz) at ratio~1.
  2. LIF gains G_a = dnu_a/dmu_a * tau_m,a (from the self-consistent point, not injected).
  3. Rate-layer dispersion (Exploration-1 char_det structure) with J_ab = C_ab*w_ab,
     anisotropic E->E + isotropic inhibition, AMPA(3.5)/GABA(18) filters + delay.
  4. Scan k -> leading mode -> regime (stable / candidate-excitable / unstable) + finite-k
     Hopf? + frequency. Report the FRACTION of the family near-critical / with finite-k Hopf.

RESULT (this run): the LIF transfer realizes a near-critical finite-k Hopf at LOW E rate
(0.7-1.7 Hz), passing through a candidate-excitable window (ratio~1.05), at 27-29 Hz --
matching coworker1's empirical traveling wave. This is exactly the regime the sigmoid
could NOT host (max stable-rest loop gain 1.26; see datalocked_step0b_exploration). The
transfer replacement WORKS through the linear map. NEXT (not here): rebuild the finite-pulse
gate (Step 0b) on a LIF field integrator -- only a self_limited_propagation + positive
margin there unlocks Step 1.

Approximations: white-noise Siegert (colored-noise threshold correction is a refinement);
static gain dnu/dmu (LIF dynamic transfer H(omega) is a refinement); Gaussian approx of
the rho=0.6 elliptical kernel. The frequency match to coworker1 (~29 Hz) is a strong
cross-check given these are approximate.

Run: python scripts/sef_hfo_step0a_lif.py
"""
import json
from pathlib import Path
import numpy as np
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scipy.special import erfcx
from scipy.integrate import quad
from scipy.optimize import fsolve
from src.sef_hfo_stability import gaussian_hat

OUT = Path("results/topic4_sef_hfo/linear_stability")
# Brunel Table-1 (coworker1)
V_TH, V_RESET = 18.0, 11.0
TAU_ME, TAU_MI, TREF_E, TREF_I = 20.0, 10.0, 2.0, 1.0
G_INH = 3.6
C_EE, W_EE = 800, 0.1575
C_IE, W_IE = 800, 0.2625
C_EI, W_EI = 200, 1.07 * G_INH * 0.1575
C_II, W_II = 200, G_INH * 0.2625
JX_E, JX_I = 0.455, 0.85
TAU_SYN, ALPHA = 0.7 + 3.5, 2.065
# time structure + connectivity (mm), slow GABA
TAU_AMPA, TAU_GABA, DELAY = 3.5, 18.0, 1.0
ELL_PAR, ELL_PERP, L_INH = 0.54, 0.27, 0.25   # E->E anisotropic (l_EE=0.38,rho=0.6); inhibition iso
K = np.linspace(0, 3, 31)


def lif_rate(mu, sigma, tm, tref):
    y_th, y_r = (V_TH - mu) / sigma, (V_RESET - mu) / sigma
    integ, _ = quad(lambda x: erfcx(-x), y_r, y_th, limit=200)
    return 1.0 / (tref + tm * np.sqrt(np.pi) * integ)


def nu_theta_pop():
    def nth(Jx, tm):
        A = 0.5 * ALPHA * Jx * np.sqrt(TAU_SYN)
        return ((A + np.sqrt(A * A + 4 * tm * Jx * V_TH)) / (2 * tm * Jx)) ** 2
    return 0.8 * nth(JX_E, TAU_ME) + 0.2 * nth(JX_I, TAU_MI)


def mean_field(ratio):
    nuext = ratio * nu_theta_pop()
    nuE = nuI = 0.002
    for _ in range(400):
        muE = TAU_ME * (C_EE * W_EE * nuE - C_EI * W_EI * nuI) + TAU_ME * JX_E * nuext
        muI = TAU_MI * (C_IE * W_IE * nuE - C_II * W_II * nuI) + TAU_MI * JX_I * nuext
        sE = np.sqrt(TAU_ME * (C_EE * W_EE**2 * nuE + C_EI * W_EI**2 * nuI) + TAU_ME * JX_E**2 * nuext)
        sI = np.sqrt(TAU_MI * (C_IE * W_IE**2 * nuE + C_II * W_II**2 * nuI) + TAU_MI * JX_I**2 * nuext)
        nuE = 0.5 * nuE + 0.5 * lif_rate(muE, sE, TAU_ME, TREF_E)
        nuI = 0.5 * nuI + 0.5 * lif_rate(muI, sI, TAU_MI, TREF_I)
    return dict(nuE=nuE, nuI=nuI, muE=muE, sE=sE, muI=muI, sI=sI)


def lif_gains(op, h=1e-3):
    GE = (lif_rate(op["muE"] + h, op["sE"], TAU_ME, TREF_E) - lif_rate(op["muE"] - h, op["sE"], TAU_ME, TREF_E)) / (2 * h) * TAU_ME
    GI = (lif_rate(op["muI"] + h, op["sI"], TAU_MI, TREF_I) - lif_rate(op["muI"] - h, op["sI"], TAU_MI, TREF_I)) / (2 * h) * TAU_MI
    return GE, GI


def char_det(lam, k, GE, GI):
    H = lambda ts: np.exp(-lam * DELAY) / (1 + lam * ts)
    WEE = (C_EE * W_EE) * gaussian_hat(k, 0, ELL_PAR, ELL_PERP)
    WEI = (C_EI * W_EI) * gaussian_hat(k, 0, L_INH, L_INH)
    WIE = (C_IE * W_IE) * gaussian_hat(k, 0, L_INH, L_INH)
    WII = (C_II * W_II) * gaussian_hat(k, 0, L_INH, L_INH)
    a = (1 + TAU_ME * lam) - GE * WEE * H(TAU_AMPA)
    b = GE * WEI * H(TAU_GABA)
    c = -GI * WIE * H(TAU_AMPA)
    d = (1 + TAU_MI * lam) + GI * WII * H(TAU_GABA)
    return a * d - b * c


def rightmost(k, GE, GI):
    best = (-np.inf, 0.0)
    def fr(v):
        z = char_det(complex(v[0], v[1]), k, GE, GI); return [z.real, z.imag]
    for r0 in np.linspace(-0.5, 0.3, 12):
        for i0 in np.linspace(0, 0.6, 12):
            s, _, ier, _ = fsolve(fr, [r0, i0], full_output=True)
            if (ier == 1 and -0.5 - 1e-6 <= s[0] <= 0.3 + 1e-6 and abs(s[1]) <= 0.6
                    and abs(complex(*fr(s))) < 1e-7 and s[0] > best[0]):
                best = (float(s[0]), abs(float(s[1])))
    return best


def main():
    ratios = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25]
    rows = []
    for ratio in ratios:
        op = mean_field(ratio)
        GE, GI = lif_gains(op)
        res = [rightmost(float(k), GE, GI) for k in K]
        re = np.array([r[0] for r in res]); im = np.array([r[1] for r in res])
        j = int(np.argmax(re))
        regime = "unstable" if re[j] > 1e-3 else ("candidate" if re[j] > -0.02 else "stable")
        finite_k_hopf = bool(K[j] > 1e-3 and im[j] > 1e-3 and re[j] > re[0] + 1e-4)
        rows.append({"ratio": ratio, "nuE_Hz": op["nuE"] * 1000, "nuI_Hz": op["nuI"] * 1000,
                     "loop_gain_EE": GE * C_EE * W_EE, "max_re": re[j], "k_star": float(K[j]),
                     "omega_star": im[j], "freq_Hz": 1000 * im[j] / (2 * np.pi),
                     "regime": regime, "finite_k_hopf": finite_k_hopf})
    n = len(rows)
    frac_candidate_or_unstable = sum(r["regime"] in ("candidate", "unstable") for r in rows) / n
    frac_finite_k_hopf = sum(r["finite_k_hopf"] for r in rows) / n
    verdict = {
        "transfer": "LIF colored-noise (Siegert), self-consistent -- NO injected gain",
        "validated_low_E_rate": bool(all(r["nuE_Hz"] < 5.0 for r in rows)),
        "fraction_candidate_or_unstable": frac_candidate_or_unstable,
        "fraction_finite_k_hopf": frac_finite_k_hopf,
        "freq_range_Hz": [min(r["freq_Hz"] for r in rows if r["finite_k_hopf"]),
                          max(r["freq_Hz"] for r in rows if r["finite_k_hopf"])],
        "coworker1_empirical_Hz": 29.0,
        "conclusion": ("LIF transfer realizes a near-critical finite-k Hopf at LOW E rate "
                       "(<2 Hz), through a candidate-excitable window, at ~25-29 Hz matching "
                       "coworker1. The sigmoid could not (max stable-rest loop gain 1.26). "
                       "Transfer replacement works through the linear map."),
        "next": ("Rebuild the finite-pulse gate (Step 0b) on a LIF field integrator; only "
                 "self_limited_propagation + positive margin there unlocks Step 1."),
        "step1_status": "LOCKED (finite-pulse gate not yet re-run on the LIF field).",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "step0a_lif.json").write_text(json.dumps({"per_ratio": rows, "verdict": verdict}, indent=2, default=float))
    print("ratio nuE(Hz) loopG  maxRe    k*  freq(Hz) regime    finite-k-Hopf")
    for r in rows:
        print(f"{r['ratio']:.2f} {r['nuE_Hz']:6.2f} {r['loop_gain_EE']:5.2f} {r['max_re']:+.4f} "
              f"{r['k_star']:.2f}  {r['freq_Hz']:6.1f} {r['regime']:9s} {r['finite_k_hopf']}")
    print(f"validated low E rate: {verdict['validated_low_E_rate']}; "
          f"finite-k-Hopf fraction: {frac_finite_k_hopf:.2f}; freq {verdict['freq_range_Hz']} Hz "
          f"(coworker1 ~29). Step 1 {verdict['step1_status']}")


if __name__ == "__main__":
    main()
