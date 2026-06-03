"""Step 0d (2026-06-03): anisotropy ROTATION control -- the framework's load-bearing
discriminator, on the LIF rate field.

Claim to test: the propagation-template direction is set by the E->E connectivity
anisotropy axis theta_EE, NOT by electrode/grid geometry. So:
  - rotate theta_EE -> propagation direction theta_prop rotates WITH it (slope~1);
  - ISOTROPIC E->E (ell_par=ell_perp) MUST FAIL to produce a directional template
    (activated region ~circular, no preferred axis) -- the negative control.

Method: fire a CENTERED finite pulse (so direction is not biased by boundary), integrate
the LIF rate field, and at peak activity measure the principal axis of the active region
(2nd-moment / covariance eigenvector) = theta_prop, plus the anisotropy ratio
sqrt(lambda1/lambda2). theta_prop/theta_EE compared mod 180 deg (they are axes).

Reuses the validated LIF mean-field + Siegert transfer from sef_hfo_step0b_lif (fold of
Phi_LIF into a reusable form; full src-module refactor is task 21). Uses the Brunel-gain
regime (robust self_limited_propagation, recovery-off).

Run: python scripts/sef_hfo_step0d_anisotropy_control.py
"""
import json
from pathlib import Path
import numpy as np
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.sef_hfo_step0b_lif as G
from src.sef_hfo_field import anisotropic_gaussian, isotropic_gaussian, convolve_periodic, _grid

OUT = Path("results/topic4_sef_hfo/finite_pulse")
N, L = 96, 16.0
X, Y = _grid(N, L)
DT, T_MAX = 0.25, 150.0
ELL_PAR, ELL_PERP, L_INH = 0.9, 0.45, 0.45   # anisotropic E->E (ratio 2); inhibition iso
A_PULSE, T_PULSE, R_PULSE = 8.0, 30.0, 1.5


def integrate_dir(op, KEE, KI, b_a=0.0, tau_a=80.0):
    """Centered pulse; return per-frame active fraction + the active-cell field at peak."""
    lE, lI = G._lut(op["sE"], G.TAU_ME, G.TREF_E), G._lut(op["sI"], G.TAU_MI, G.TREF_I)
    fE = lambda m: np.interp(m, lE[0], lE[1]); fI = lambda m: np.interp(m, lI[0], lI[1])
    muxE = G.TAU_ME * G.JX_E * op["nuext"]; muxI = G.TAU_MI * G.JX_I * op["nuext"]
    mask = (X**2 + Y**2 <= R_PULSE**2).astype(float)
    rE = np.full((N, N), op["nuE"]); rI = np.full((N, N), op["nuI"]); a = np.full((N, N), op["nuE"])
    sEE = convolve_periodic(rE, KEE).copy(); sEI = convolve_periodic(rI, KI).copy()
    sIE = convolve_periodic(rE, KI).copy(); sII = convolve_periodic(rI, KI).copy()
    thr = op["nuE"] + G.DETECT
    ns = int(T_MAX / DT); ext = np.empty(ns); peak_field = None; peak_ext = -1.0
    for t in range(ns):
        st = (A_PULSE * mask) if t * DT < T_PULSE else 0.0 * mask
        cEE = convolve_periodic(rE, KEE); cEI = convolve_periodic(rI, KI)
        cIE = convolve_periodic(rE, KI); cII = convolve_periodic(rI, KI)
        sEE += DT/G.TAU_AMPA*(cEE-sEE); sEI += DT/G.TAU_GABA*(cEI-sEI)
        sIE += DT/G.TAU_AMPA*(cIE-sIE); sII += DT/G.TAU_GABA*(cII-sII)
        muE = G.TAU_ME*(G.C_EE*G.W_EE*sEE - G.C_EI*G.W_EI*sEI) + muxE + st - b_a*a
        muI = G.TAU_MI*(G.C_IE*G.W_IE*sIE - G.C_II*G.W_II*sII) + muxI
        rE = rE + DT/G.TAU_ME*(-rE + fE(muE)); rI = rI + DT/G.TAU_MI*(-rI + fI(muI))
        a = a + DT/tau_a*(-a + rE)
        m = rE > thr; ext[t] = m.mean()
        if ext[t] > peak_ext:
            peak_ext = ext[t]; peak_field = (rE - op["nuE"]).copy()
    return ext, peak_field


def principal_axis(field, op):
    """Angle (deg, mod 180) of the principal axis of the active region + anisotropy ratio."""
    thr = G.DETECT
    m = field > thr
    if m.sum() < 5:
        return float("nan"), 1.0
    w = field[m]
    xs, ys = X[m], Y[m]
    xm = np.average(xs, weights=w); ym = np.average(ys, weights=w)
    cxx = np.average((xs-xm)**2, weights=w); cyy = np.average((ys-ym)**2, weights=w)
    cxy = np.average((xs-xm)*(ys-ym), weights=w)
    C = np.array([[cxx, cxy], [cxy, cyy]])
    ev, evec = np.linalg.eigh(C)
    v = evec[:, -1]   # largest eigenvalue eigenvector
    ang = np.degrees(np.arctan2(v[1], v[0])) % 180.0
    ratio = float(np.sqrt(ev[-1] / max(ev[0], 1e-12)))
    return float(ang), ratio


def axis_diff(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def main():
    op = G.mean_field(1.0)
    KI = isotropic_gaussian(N, L, L_INH)
    rows = []
    # anisotropic E->E, rotate theta_EE
    for th in [0.0, 30.0, 60.0, 90.0, 135.0]:
        KEE = anisotropic_gaussian(N, L, ELL_PAR, ELL_PERP, np.radians(th))
        _, pf = integrate_dir(op, KEE, KI)
        thp, ratio = principal_axis(pf, op)
        rows.append({"kernel": "anisotropic", "theta_EE": th, "theta_prop": thp,
                     "axis_err_deg": axis_diff(thp, th), "anisotropy_ratio": ratio})
    # isotropic control (no preferred axis) -- rotate the (meaningless) "theta_EE", expect NO tracking
    iso = []
    for th in [0.0, 45.0, 90.0]:
        KEEiso = anisotropic_gaussian(N, L, 0.6, 0.6, np.radians(th))  # ell_par=ell_perp -> isotropic
        _, pf = integrate_dir(op, KEEiso, KI)
        thp, ratio = principal_axis(pf, op)
        iso.append({"kernel": "isotropic", "theta_EE": th, "theta_prop": thp,
                    "axis_err_deg": axis_diff(thp, th), "anisotropy_ratio": ratio})

    aniso_tracks = all(r["axis_err_deg"] < 20.0 for r in rows) and all(r["anisotropy_ratio"] > 1.3 for r in rows)
    iso_fails = all(r["anisotropy_ratio"] < 1.3 for r in iso)   # isotropic -> ~circular, no axis
    verdict = {
        "anisotropic_theta_prop_tracks_theta_EE": bool(aniso_tracks),
        "isotropic_control_fails_to_produce_axis": bool(iso_fails),
        "discriminator_passed": bool(aniso_tracks and iso_fails),
        "note": ("Step-0d load-bearing discriminator: propagation axis set by E->E connectivity "
                 "anisotropy (theta_prop tracks theta_EE), NOT electrode geometry; isotropic "
                 "control produces no directional template. PASS supports SEF-HFO vs geometry artifact."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "step0d_anisotropy_control.json").write_text(
        json.dumps({"anisotropic": rows, "isotropic_control": iso, "verdict": verdict}, indent=2, default=float))
    print("=== anisotropic E->E: does theta_prop track theta_EE? ===")
    for r in rows:
        print(f"  theta_EE={r['theta_EE']:5.0f}  theta_prop={r['theta_prop']:6.1f}  "
              f"axis_err={r['axis_err_deg']:5.1f}deg  anisotropy_ratio={r['anisotropy_ratio']:.2f}")
    print("=== isotropic control (must NOT produce a tracking axis) ===")
    for r in iso:
        print(f"  theta_EE={r['theta_EE']:5.0f}  theta_prop={r['theta_prop']:6.1f}  "
              f"axis_err={r['axis_err_deg']:5.1f}deg  anisotropy_ratio={r['anisotropy_ratio']:.2f}")
    print(f"DISCRIMINATOR PASSED = {verdict['discriminator_passed']} "
          f"(aniso tracks={aniso_tracks}, iso fails={iso_fails})")


if __name__ == "__main__":
    main()
