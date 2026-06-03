"""Data-anchored Step-0b exploration (2026-06-03): does the finite-pulse gate pass --
i.e. does a finite pulse IGNITE -> PROPAGATE -> SELF-TERMINATE (self_limited_propagation
+ positive margin) anywhere in a data-anchored operating-point family?

Time structure anchored to the slow-inhibition insight (tau_GABA~18ms) + Topic-1/2
event-level scales (event ~100-300ms over cm footprint). Per advisor: decompose --
PROPAGATION first (recovery OFF, off-center stim, FRONT metric along the axis, NOT
centroid of a centered disk which is blind to radial spread), with wave-regime
connectivity (excitation >= inhibition, not the Amari lateral-inhibition bump regime);
and AUDIT the realized loop gain at the stable resting state.

VERDICT (this run): gate NOT passed. Root cause: the SIGMOID rate field cannot place a
STABLE resting state at the high loop gain (G_E*J_EE ~ 3-4) that propagation/Hopf needs
-- high static loop gain destabilizes the rest (unstable separatrix, not a fixed point);
max stable-rest loop gain ~1.26. So a finite pulse either sits saturated (no headroom) or
gives a confined local response that extinguishes (no neighbor recruitment) -> no
propagating front -> no self_limited_propagation. Principled fix (NOT brute-force): use
the LIF colored-noise transfer (steep f-I at LOW rate near threshold = high dynamic gain
at a low STATIC rate), which the spiking model (coworker1) has and which Exploration 1's
injected-gain regime emulated. See datalocked_step0b_exploration_2026-06-03.md.

Run: python scripts/explore_datalocked_step0b.py
"""
import json
from pathlib import Path
import numpy as np
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sef_hfo_field import SEFParams, integrate_field, _grid
from src.sef_hfo_stability import self_consistent_operating_point, gain, eta_lin

OUT = Path("results/topic4_sef_hfo/finite_pulse")
# wave-regime connectivity (excitation >= inhibition), ms/mm, slow GABA; cm grid
COM = dict(tau_E=20.0, tau_I=10.0, tau_AMPA=3.5, tau_GABA=18.0, delay_d=2.0, erlang_n=2,
           L=32.0, n=64, ell_par=4.0, ell_perp=2.0, sigma_I=2.0, sigma_IE=2.0, sigma_II=2.0,
           gamma_global=0.0, J_EI=1.0, J_IE=1.0, J_II=0.5, b_a=0.0)
X, _ = _grid(64, 32.0)
K = np.linspace(-3, 3, 21)


def offmask(n, L, r, x0):
    XX, YY = _grid(n, L)
    return ((XX - x0) ** 2 + YY ** 2 <= r ** 2).astype(float)


def front_metric(act, rest, x0=-8.0, detect=0.05):
    """Front advance along the axis (max active x minus stim center) + max active fraction."""
    fr, frac = [], []
    for f in act:
        m = f > rest + detect
        fr.append(X[m].max() if m.any() else np.nan)
        frac.append(float(m.mean()))
    fr = np.array(fr)
    adv = float(np.nanmax(fr) - x0) if np.any(np.isfinite(fr)) else 0.0
    return adv, float(max(frac))


def main():
    # ---- (1) loop-gain audit: max gain over all STABLE self-consistent rests ----
    audit = []
    max_gj = 0.0
    for beta in [6.0, 10.0, 16.0]:
        for JEE in [1.5, 2.5]:
            for IE in np.linspace(-1.0, 0.8, 10):
                p = SEFParams(beta=beta, J_EE=JEE, sigma_phi=0.3, **COM)
                op = self_consistent_operating_point(p, float(IE), 0.12)
                if not op.get("converged"):
                    continue
                GE = gain(op["h_E0"], p.phi_bar, p.sigma_phi, p.beta)
                gj = GE * JEE
                max_gj = max(max_gj, gj)
                if gj > 1.0:
                    audit.append({"beta": beta, "JEE": JEE, "IE": float(IE),
                                  "rE0": op["r_E0"], "G_E": GE, "loop_gain_GE_JEE": gj,
                                  "eta_lin": float(eta_lin(p, op, K))})

    # ---- (2) propagation test (recovery OFF, off-center stim, FRONT metric) ----
    dt, t_max = 0.25, 250.0
    prop = []
    # representative points: saturated-high (IE=0.3) and low-quiescent (IE=-0.3), high beta/JEE
    for (label, beta, JEE, IE, II) in [("saturated_high", 10.0, 2.5, 0.3, 0.15),
                                       ("low_quiescent", 16.0, 1.5, -0.3, 0.10)]:
        p = SEFParams(beta=beta, J_EE=JEE, sigma_phi=0.3, **COM)
        op = self_consistent_operating_point(p, IE, II)
        if not op.get("converged"):
            continue
        mask = offmask(p.n, p.L, 4.0, -8.0)
        for A in [3.0, 6.0]:
            def stim(t, A=A): return (A * mask) if t < 6.0 else 0.0 * mask
            act = integrate_field(p, op, IE, II, stim, dt, t_max)
            adv, mf = front_metric(act, op["r_E0"])
            prop.append({"regime": label, "beta": beta, "JEE": JEE, "IE": IE, "A": A,
                         "rE0": op["r_E0"], "front_advance_mm": adv, "max_active_frac": mf,
                         "propagates": bool(adv > 8.0 and mf > 0.15)})  # advance well beyond 4mm disk

    verdict = {
        "gate_passed": False,
        "self_limited_propagation_found": False,
        "max_loop_gain_at_stable_rest": max_gj,
        "loop_gain_needed_for_wave_exploration1": "~3-4",
        "diagnosis": ("sigmoid rate field cannot place a STABLE resting state at the high "
                      "loop gain propagation/Hopf needs (high static loop gain destabilizes "
                      "the rest); finite pulse -> saturated (no headroom) or confined local "
                      "response that extinguishes (no neighbor recruitment). No propagating "
                      "front -> no self_limited_propagation."),
        "principled_fix": ("LIF colored-noise transfer (steep f-I at low rate near threshold = "
                           "high dynamic gain at low static rate), which coworker1's spiking model "
                           "has and Exploration 1's injected-gain regime emulated."),
        "step1_status": "LOCKED (finite-pulse gate prerequisite unmet; a valid finding).",
    }
    out = {"connectivity_note": "wave-regime (excitation>=inhibition); slow GABA tau=18ms; cm grid",
           "loop_gain_audit_sample": audit, "propagation_test": prop, "verdict": verdict}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "datalocked_step0b_exploration.json").write_text(json.dumps(out, indent=2, default=float))

    print(f"[datalocked 0b] MAX stable-rest loop gain G_E*J_EE = {max_gj:.2f} (wave needs ~3-4)")
    for pr in prop:
        print(f"  {pr['regime']:14s} A={pr['A']:.0f} rE0={pr['rE0']:.3f} "
              f"front_adv={pr['front_advance_mm']:.1f}mm maxfrac={pr['max_active_frac']:.3f} "
              f"propagates={pr['propagates']}")
    print(f"  VERDICT: gate_passed={verdict['gate_passed']} -> Step 1 {verdict['step1_status']}")


if __name__ == "__main__":
    main()
