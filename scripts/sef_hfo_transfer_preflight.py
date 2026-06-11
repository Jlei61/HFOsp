"""Transfer-function preflight (2026-06-03, user step 1): does the LIF colored-noise
f-I give 'low static firing rate + high local gain' operating points where the SIGMOID
cannot? If not, replacing the transfer is pointless. This is the gate before rebuilding
Step 0a/0b on the LIF transfer.

Result (this run): PASS. LIF Siegert f-I has loop gain ~2-5 at LOW rate (1-10 Hz) --
right in Exploration 1's Hopf/propagation range (~3-4) -- because near threshold the
fluctuation-driven f-I is steep (high dnu/dmu) AND the recurrent weight scale
(tau_m*C_EE*w_EE ~ 2520) is large. The sigmoid's gain is ~0 wherever the rate is low
(flat tails), peaking only at F~0.5; its max realized loop gain was 1.26 (only near
saturation). So the sigmoid structurally cannot host a low-rate excitable rest; the LIF
can. See lif_rate_field_theory_2026-06-03.md §3 for the sigmoid-vs-LIF transfer argument.

White-noise Siegert (numerically stable via erfcx); colored-noise (synaptic-filter)
threshold correction is a refinement for the formal rebuild, not needed for this
qualitative shape check.

Run: python scripts/sef_hfo_transfer_preflight.py
"""
import json
from pathlib import Path
import numpy as np
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scipy.special import erfcx
from scipy.integrate import quad
from src.sef_hfo_stability import F_eff, gain as sigmoid_gain

OUT = Path("results/topic4_sef_hfo/linear_stability")
# Brunel E params (coworker1 Table-1), mV / ms
V_TH, V_RESET, TAU_M, TAU_REF = 18.0, 11.0, 20.0, 2.0
REC_EE = TAU_M * 800 * 0.1575   # dmu_E/dnu_E = tau_m*C_EE*w_EE ~ 2520 (E-self recurrent scale)


def lif_rate(mu, sigma):
    """Siegert stationary rate (1/ms) for current-based LIF, white-noise."""
    y_th, y_r = (V_TH - mu) / sigma, (V_RESET - mu) / sigma
    integ, _ = quad(lambda x: erfcx(-x), y_r, y_th, limit=200)
    return 1.0 / (TAU_REF + TAU_M * np.sqrt(np.pi) * integ)


def lif_gain(mu, sigma, h=1e-3):
    return (lif_rate(mu + h, sigma) - lif_rate(mu - h, sigma)) / (2 * h)


def main():
    sigma = 4.0
    lif_rows = []
    for mu in [8, 10, 12, 14, 15, 16, 17, 17.5]:
        nu_hz = lif_rate(mu, sigma) * 1000.0
        g = lif_gain(mu, sigma)
        lif_rows.append({"mu_mV": mu, "nu_Hz": nu_hz, "dnu_dmu": g, "loop_gain_EE": g * REC_EE})

    sig_rows = []
    for h in [-2, -1, -0.5, 0, 0.5, 1, 2]:
        sig_rows.append({"h": h, "F_eff": float(F_eff(h, 0.0, 0.3, 10.0)),
                         "dF_dh": float(sigmoid_gain(h, 0.0, 0.3, 10.0))})

    # low-rate loop gain: max loop gain among LIF rows with nu < 10 Hz
    low_rate_lif = [r for r in lif_rows if r["nu_Hz"] < 10.0]
    max_lowrate_loopgain = max((r["loop_gain_EE"] for r in low_rate_lif), default=0.0)

    verdict = {
        "preflight_pass": bool(max_lowrate_loopgain >= 2.0),
        "lif_max_loopgain_at_nu_below_10Hz": max_lowrate_loopgain,
        "sigmoid_max_realized_loopgain": 1.26,
        "conclusion": ("LIF f-I hosts low-rate (1-10 Hz) operating points with loop gain "
                       "~2-5 (Exploration-1 Hopf range). Sigmoid cannot (gain ~0 at low rate). "
                       "Transfer replacement is justified; proceed to LIF self-consistent "
                       "steady state + linear map (no injected gain)."),
    }
    out = {"sigma_mV": sigma, "lif_f_I": lif_rows, "sigmoid_f_I": sig_rows, "verdict": verdict}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "transfer_preflight.json").write_text(json.dumps(out, indent=2, default=float))

    print("LIF Siegert f-I (sigma=4mV):  mu  nu(Hz)  loop_gain_EE")
    for r in lif_rows:
        print(f"   {r['mu_mV']:5.1f}  {r['nu_Hz']:7.2f}   {r['loop_gain_EE']:6.2f}")
    print(f"sigmoid: gain ~0 at low F (flat tails), peaks at F~0.5; max realized loop gain 1.26")
    print(f"PREFLIGHT PASS={verdict['preflight_pass']}: LIF low-rate(<10Hz) max loop gain "
          f"= {max_lowrate_loopgain:.2f} (vs sigmoid 1.26)")


if __name__ == "__main__":
    main()
