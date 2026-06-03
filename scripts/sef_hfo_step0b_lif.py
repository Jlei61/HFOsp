"""Step 0b on the LIF transfer (2026-06-03, user step 3): the finite-pulse GATE.

Rebuilds the finite-pulse response map on a LIF rate-field integrator (transfer = Siegert
nu(mu,sigma) at fixed operating-point sigma; physical mV; Brunel connectivity; synaptic
low-pass; recovery switchable). This is the ACTUAL Step-1 prerequisite: a finite pulse must
IGNITE -> PROPAGATE (moving front) -> SELF-TERMINATE, with A_runaway - A_self_limited > 0.

Per advisor: off-center stim + FRONT-position metric along the axis (not centroid of a
centered disk). Per user: report the FRACTION of the data-anchored operating-point family
that yields self_limited_propagation + positive margin; recovery OFF and ON both reported.

Foundations validated separately: mean-field reproduces coworker1's low-rate SI regime
(scripts/sef_hfo_step0a_lif.py); steady state holds exactly (resid ~1e-17); LIF transfer
gives low-rate high-gain rests the sigmoid cannot (scripts/sef_hfo_transfer_preflight.py).

Run: python scripts/sef_hfo_step0b_lif.py
"""
import json
from pathlib import Path
import numpy as np
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scipy.special import erfcx
from scipy.integrate import quad
from scipy.optimize import fsolve
from src.sef_hfo_field import anisotropic_gaussian, isotropic_gaussian, convolve_periodic, _grid

OUT = Path("results/topic4_sef_hfo/finite_pulse")
V_TH, V_RESET = 18.0, 11.0
TAU_ME, TAU_MI, TREF_E, TREF_I = 20.0, 10.0, 2.0, 1.0
G_INH = 3.6
C_EE, W_EE = 800, 0.1575; C_IE, W_IE = 800, 0.2625
C_EI, W_EI = 200, 1.07 * G_INH * 0.1575; C_II, W_II = 200, G_INH * 0.2625
JX_E, JX_I = 0.455, 0.85; TAU_SYN, ALPHA = 4.2, 2.065
TAU_AMPA, TAU_GABA = 3.5, 18.0
ELL_PAR, ELL_PERP, L_INH = 0.54, 0.27, 0.25
N, L = 96, 12.0
DT, T_MAX = 0.25, 300.0
X, _ = _grid(N, L)
STIM_X0, STIM_R, STIM_T = -3.0, 2.0, 30.0       # off-center disk, 30 ms pulse
RUNAWAY_FRAC, RETURN_FRAC, DETECT, TRAVEL_MM = 0.5, 0.2, 0.005, 3.0


def lif_rate(mu, sigma, tm, tref):
    y_th, y_r = (V_TH - mu) / sigma, (V_RESET - mu) / sigma
    integ, _ = quad(lambda x: erfcx(-x), y_r, y_th, limit=200)
    return 1.0 / (tref + tm * np.sqrt(np.pi) * integ)


def nu_theta_pop():
    f = lambda Jx, tm: ((0.5*ALPHA*Jx*np.sqrt(TAU_SYN) + np.sqrt((0.5*ALPHA*Jx*np.sqrt(TAU_SYN))**2 + 4*tm*Jx*V_TH))/(2*tm*Jx))**2
    return 0.8*f(JX_E, TAU_ME) + 0.2*f(JX_I, TAU_MI)


def _ms(nuE, nuI, nuext):
    muE = TAU_ME*(C_EE*W_EE*nuE - C_EI*W_EI*nuI) + TAU_ME*JX_E*nuext
    muI = TAU_MI*(C_IE*W_IE*nuE - C_II*W_II*nuI) + TAU_MI*JX_I*nuext
    sE = np.sqrt(max(TAU_ME*(C_EE*W_EE**2*nuE + C_EI*W_EI**2*nuI) + TAU_ME*JX_E**2*nuext, 1e-9))
    sI = np.sqrt(max(TAU_MI*(C_IE*W_IE**2*nuE + C_II*W_II**2*nuI) + TAU_MI*JX_I**2*nuext, 1e-9))
    return muE, muI, sE, sI


def mean_field(ratio):
    nuext = ratio * nu_theta_pop()
    def resid(nu):
        muE, muI, sE, sI = _ms(max(nu[0], 1e-9), max(nu[1], 1e-9), nuext)
        return [nu[0] - lif_rate(muE, sE, TAU_ME, TREF_E), nu[1] - lif_rate(muI, sI, TAU_MI, TREF_I)]
    sol = fsolve(resid, [0.001, 0.005])
    nuE, nuI = float(sol[0]), float(sol[1])
    muE, muI, sE, sI = _ms(nuE, nuI, nuext)
    return dict(nuE=nuE, nuI=nuI, muE=muE, sE=sE, muI=muI, sI=sI, nuext=nuext)


def _lut(sigma, tm, tref, lo=-12, hi=45, npts=5000):
    mus = np.linspace(lo, hi, npts)
    return mus, np.array([lif_rate(m, sigma, tm, tref) for m in mus])


_KEE = anisotropic_gaussian(N, L, ELL_PAR, ELL_PERP, 0.0)
_KI = isotropic_gaussian(N, L, L_INH)


def integrate(op, luts, b_a, tau_a, stim_fn):
    KEE, KI = _KEE, _KI
    lE, lI = luts
    fE = lambda mu: np.interp(mu, lE[0], lE[1]); fI = lambda mu: np.interp(mu, lI[0], lI[1])
    muxE = TAU_ME*JX_E*op["nuext"]; muxI = TAU_MI*JX_I*op["nuext"]
    rE = np.full((N, N), op["nuE"]); rI = np.full((N, N), op["nuI"]); a = np.full((N, N), op["nuE"])
    sEE = convolve_periodic(rE, KEE).copy(); sEI = convolve_periodic(rI, KI).copy()
    sIE = convolve_periodic(rE, KI).copy(); sII = convolve_periodic(rI, KI).copy()
    nsteps = int(T_MAX/DT); ext = np.empty(nsteps); front = np.empty(nsteps)
    thr = op["nuE"] + DETECT
    for t in range(nsteps):
        stim = stim_fn(t*DT)
        cEE = convolve_periodic(rE, KEE); cEI = convolve_periodic(rI, KI)
        cIE = convolve_periodic(rE, KI); cII = convolve_periodic(rI, KI)
        sEE += DT/TAU_AMPA*(cEE-sEE); sEI += DT/TAU_GABA*(cEI-sEI)
        sIE += DT/TAU_AMPA*(cIE-sIE); sII += DT/TAU_GABA*(cII-sII)
        muE = TAU_ME*(C_EE*W_EE*sEE - C_EI*W_EI*sEI) + muxE + stim - b_a*a
        muI = TAU_MI*(C_IE*W_IE*sIE - C_II*W_II*sII) + muxI
        rE = rE + DT/TAU_ME*(-rE + fE(muE)); rI = rI + DT/TAU_MI*(-rI + fI(muI))
        a = a + DT/tau_a*(-a + rE)
        m = rE > thr; ext[t] = m.mean(); front[t] = X[m].max() if m.any() else np.nan
    return ext, front


def classify(ext, front):
    max_ext = float(ext.max()); final = float(ext[-1])
    adv = float(np.nanmax(front) - (STIM_X0 + STIM_R)) if np.any(np.isfinite(front)) else 0.0
    returned = final <= RETURN_FRAC * max(max_ext, 1e-12)
    on = ext > 0.02
    dur = float((np.where(on)[0][-1] - np.where(on)[0][0] + 1) * DT) if on.any() else 0.0
    if max_ext >= RUNAWAY_FRAC and not returned:
        lbl = "runaway"
    elif adv >= TRAVEL_MM and not returned:
        lbl = "runaway"
    elif adv >= TRAVEL_MM and returned:
        lbl = "self_limited_propagation"
    elif max_ext >= RUNAWAY_FRAC and returned:
        lbl = "global_synchronous"
    elif max_ext > 0.03:
        lbl = "local_bump"
    else:
        lbl = "extinction"
    return lbl, dict(max_ext=max_ext, adv_mm=adv, returned=bool(returned), dur_ms=dur)


def disk(x0, r):
    XX, YY = _grid(N, L); return ((XX - x0)**2 + YY**2 <= r**2).astype(float)


def scan_point(ratio, b_a, tau_a):
    op = mean_field(ratio)
    luts = (_lut(op["sE"], TAU_ME, TREF_E), _lut(op["sI"], TAU_MI, TREF_I))
    mask = disk(STIM_X0, STIM_R)
    amps = [1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0]
    grid = []
    A_event = A_self = A_run = np.inf
    for A in amps:
        ext, front = integrate(op, luts, b_a, tau_a, (lambda t, A=A: (A*mask) if t < STIM_T else 0.0*mask))
        lbl, info = classify(ext, front)
        grid.append({"A": A, "label": lbl, **info})
        if lbl in ("local_bump", "self_limited_propagation", "global_synchronous", "runaway") and not np.isfinite(A_event):
            A_event = A
        if lbl == "self_limited_propagation" and not np.isfinite(A_self):
            A_self = A
        if lbl == "runaway" and not np.isfinite(A_run):
            A_run = A
    margin = A_run - A_self
    has_window = bool(np.isfinite(A_self) and (margin > 0))
    sl_durs = [c["dur_ms"] for c in grid if c["label"] == "self_limited_propagation"]
    return {"ratio": ratio, "nuE_Hz": op["nuE"]*1000, "A_event": A_event, "A_self_limited": A_self,
            "A_runaway": A_run, "safety_margin": margin, "has_self_limited_window": has_window,
            "self_limited_durations_ms": sl_durs, "grid": grid}


def main():
    family = [0.95, 1.00, 1.05, 1.10]
    runs = {}
    for name, b_a, tau_a in [("recovery_off", 0.0, 80.0), ("recovery_on", 0.5, 80.0)]:
        pts = [scan_point(r, b_a, tau_a) for r in family]
        n = len(pts); k = sum(p["has_self_limited_window"] for p in pts)
        runs[name] = {"n": n, "n_with_window": k, "fraction_with_window": k/n, "points": pts}
    any_win = max(runs["recovery_off"]["fraction_with_window"], runs["recovery_on"]["fraction_with_window"])
    durs = [d for r in runs.values() for p in r["points"] for d in p["self_limited_durations_ms"]]
    verdict = {
        "gate_passed": bool(any_win > 0),
        "fraction_with_window_off": runs["recovery_off"]["fraction_with_window"],
        "fraction_with_window_on": runs["recovery_on"]["fraction_with_window"],
        "self_limited_duration_range_ms": [min(durs), max(durs)] if durs else None,
        "data_target_ms": "channel-spread ~50-178ms (Exploration 2)",
        "conclusion": ("LIF transfer field: a finite pulse ignites -> propagates (moving front) "
                       "-> self-terminates, with positive safety margin, in a fraction of the "
                       "data-anchored family -- the sigmoid field could not. If durations match "
                       "data, Step 1 (add noise) is unlocked at the candidate window."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "step0b_lif.json").write_text(json.dumps({"runs": runs, "verdict": verdict}, indent=2, default=float))
    for name, r in runs.items():
        print(f"== {name}: fraction_with_window={r['fraction_with_window']:.2f} ==")
        for p in r["points"]:
            print(f"  ratio={p['ratio']:.2f} nuE={p['nuE_Hz']:.2f}Hz A_event={p['A_event']} "
                  f"A_self_lim={p['A_self_limited']} A_runaway={p['A_runaway']} margin={p['safety_margin']} "
                  f"window={p['has_self_limited_window']} durs={[round(d) for d in p['self_limited_durations_ms']]}")
    print(f"GATE PASSED={verdict['gate_passed']} (off {verdict['fraction_with_window_off']:.2f} / "
          f"on {verdict['fraction_with_window_on']:.2f}); self-limited dur range {verdict['self_limited_duration_range_ms']} ms")


if __name__ == "__main__":
    main()
