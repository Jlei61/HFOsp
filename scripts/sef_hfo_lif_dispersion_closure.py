"""Closure #1 — is the rate-field's Hopf real at the SELF-CONSISTENT operating point, or
was "stable / no Hopf" a white-noise gain-underestimate artifact?

The spiking ground truth oscillates at a genuine finite-k Hopf (26.7 Hz, N-scaling-confirmed).
Our analyses disagreed: the committed Step-0a (damped-iteration mean-field, higher operating
point) FOUND a 27-29 Hz finite-k Hopf; the fsolve re-run (lower operating point, nuE~0.2 Hz)
found it stable. The advisor's point: the white-noise mean-field places nuE too low
(0.2 Hz vs the spiking 0.8 Hz) -> underestimates the f-I gain -> may wrongly conclude stable.

This runs the SAME rate dispersion (char_det from sef_hfo_step0a_lif: anisotropic E->E,
isotropic inhibition, AMPA/GABA single-pole filters + conduction delay, slow GABA 18 ms),
across drive, with TWO gain settings:
  A. white-noise mean-field gain (canonical fsolve operating point)        [our "stable" path]
  B. gain at the SPIKING-measured E/I rate (results/.../data/drive_sweep.json)  [corrected]
If B gives a finite-k Hopf at ~27 Hz where A does not, the rate-field reproduces the spiking
Hopf at the true operating rate; "no Hopf" was a gain underestimate. Honest either way.

Run: python scripts/sef_hfo_lif_dispersion_closure.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, fsolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sef_hfo_lif import (  # noqa: E402
    lif_rate, mean_field, lif_gains,
    TAU_ME, TAU_MI, TREF_E, TREF_I,
    C_EE, W_EE, C_EI, W_EI, C_IE, W_IE, C_II, W_II,
    TAU_AMPA, TAU_GABA,
)
from src.sef_hfo_stability import gaussian_hat  # noqa: E402

DELAY = 1.0                       # conduction delay (ms) — matches step0a
ELL_PAR, ELL_PERP, L_INH = 0.54, 0.27, 0.25
K = np.linspace(0.0, 3.0, 31)
DATA = Path("results/topic4_sef_hfo/lif_snn/data")


def char_det(lam, k, GE, GI):
    """2x2 rate-dispersion determinant (verbatim structure from sef_hfo_step0a_lif)."""
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
        z = char_det(complex(v[0], v[1]), k, GE, GI)
        return [z.real, z.imag]
    for r0 in np.linspace(-0.5, 0.3, 12):
        for i0 in np.linspace(0.0, 0.6, 12):
            s, _, ier, _ = fsolve(fr, [r0, i0], full_output=True)
            if (ier == 1 and -0.5 - 1e-6 <= s[0] <= 0.3 + 1e-6 and abs(s[1]) <= 0.6
                    and abs(complex(*fr(s))) < 1e-7 and s[0] > best[0]):
                best = (float(s[0]), abs(float(s[1])))
    return best


def leading(GE, GI):
    res = [rightmost(float(k), GE, GI) for k in K]
    re = np.array([r[0] for r in res])
    im = np.array([r[1] for r in res])
    j = int(np.argmax(re))
    return dict(k_star=float(K[j]), re_max=float(re[j]), omega=float(im[j]),
                freq_Hz=float(1000.0 * im[j] / (2 * np.pi)),
                is_hopf=bool(K[j] > 1e-3 and im[j] > 1e-3 and re[j] > re[0] + 1e-4),
                regime=("unstable" if re[j] > 1e-3 else "candidate" if re[j] > -0.02 else "stable"))


def gain_at_rate(target_kHz, sigma, tm, tref, h=1e-3):
    """G_mu = dnu/dmu * tau_m at the mu that yields target rate, at fixed sigma."""
    f = lambda mu: lif_rate(mu, sigma, tm, tref) - target_kHz
    mu = brentq(f, -25.0, 30.0, xtol=1e-7)
    g = (lif_rate(mu + h, sigma, tm, tref) - lif_rate(mu - h, sigma, tm, tref)) / (2 * h) * tm
    return g, mu


def main():
    drive = json.load(open(DATA / "drive_sweep.json"))
    ratios, spkE, spkI = drive["ratios"], drive["rateE_mean"], drive["rateI_mean"]
    rows = []
    for ratio, rE, rI in zip(ratios, spkE, spkI):
        op = mean_field(ratio)
        g = lif_gains(op)
        A = leading(g["E"], g["I"])                                  # white-noise mean-field gain
        try:
            GE_B, _ = gain_at_rate(rE / 1000.0, op["sE"], TAU_ME, TREF_E)
            GI_B, _ = gain_at_rate(rI / 1000.0, op["sI"], TAU_MI, TREF_I)
            B = leading(GE_B, GI_B)                                  # gain at spiking-measured rate
        except Exception as e:
            GE_B = GI_B = float("nan")
            B = dict(error=str(e), is_hopf=False, regime="err", re_max=float("nan"),
                     freq_Hz=float("nan"), k_star=float("nan"))
        rows.append(dict(ratio=ratio, spkE_Hz=rE, spkI_Hz=rI, mf_nuE_Hz=op["nuE"] * 1000.0,
                         loopG_A=g["E"] * C_EE * W_EE, loopG_B=GE_B * C_EE * W_EE,
                         A=A, B=B))

    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "dispersion_closure.json").write_text(json.dumps({"per_ratio": rows}, indent=2, default=float))

    print("                  white-noise gain (A)            spiking-rate gain (B)")
    print("ratio mf_nuE spkE | loopG  reMax   freq  Hopf reg | loopG  reMax   freq  Hopf reg")
    for r in rows:
        A, B = r["A"], r["B"]
        print(f"{r['ratio']:.2f} {r['mf_nuE_Hz']:5.2f} {r['spkE_Hz']:4.2f} | "
              f"{r['loopG_A']:5.2f} {A['re_max']:+.3f} {A['freq_Hz']:5.1f} {str(A['is_hopf'])[0]} {A['regime'][:4]:4s} | "
              f"{r['loopG_B']:5.2f} {B['re_max']:+.3f} {B['freq_Hz']:5.1f} {str(B['is_hopf'])[0]} {B['regime'][:4]:4s}")
    nA = sum(r["A"]["is_hopf"] for r in rows)
    nB = sum(r["B"]["is_hopf"] for r in rows)
    print(f"\nfinite-k Hopf: white-noise gain {nA}/{len(rows)} ratios; spiking-rate gain {nB}/{len(rows)} ratios")
    print("(spiking ground-truth Hopf ~26.7 Hz; coworker1 ~29 Hz)")


if __name__ == "__main__":
    main()
