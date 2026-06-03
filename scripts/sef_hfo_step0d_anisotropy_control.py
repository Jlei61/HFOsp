"""Step 0d (2026-06-03): anisotropy ROTATION control -- the framework's load-bearing
discriminator, on the canonical LIF rate field (src/sef_hfo_lif.py).

Claim to test: the propagation-template direction is set by the E->E connectivity
anisotropy axis theta_EE, NOT by electrode/grid geometry. So:
  - rotate theta_EE -> propagation direction theta_prop rotates WITH it (slope~1);
  - ISOTROPIC E->E (ell_par=ell_perp) MUST FAIL to produce a directional template
    (activated region ~circular, no preferred axis) -- the negative control.

Method: fire a CENTERED finite pulse (so direction is not biased by boundary), integrate
the LIF rate field, and at PEAK activity measure the principal axis of the active region
(2nd-moment / covariance eigenvector) = theta_prop, plus the anisotropy ratio
sqrt(lambda1/lambda2). theta_prop/theta_EE compared mod 180 deg (they are axes).

Now imports the canonical src/sef_hfo_lif (task 21 fold-in complete): mean_field +
integrate_lif_field (theta_EE rotation + return_peak_field) replace the prior private
copy of the dynamics, so this script and the test suite share ONE integrator. Only the
0d-specific measurement (principal_axis) stays local. Brunel-gain regime (w_ee_mult=1.0,
robust self_limited_propagation, recovery-off).

Run: python scripts/sef_hfo_step0d_anisotropy_control.py
"""
import json
from pathlib import Path
import numpy as np
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sef_hfo_lif import mean_field, integrate_lif_field, _grid, DETECT

OUT = Path("results/topic4_sef_hfo/finite_pulse")
N, L = 96, 16.0
X, Y = _grid(N, L)
DT, T_MAX = 0.25, 150.0
ELL_PAR, ELL_PERP, L_INH = 0.9, 0.45, 0.45   # anisotropic E->E (ratio 2); inhibition iso
ELL_ISO = 0.6                                 # ell_par=ell_perp -> isotropic control
A_PULSE, T_PULSE, R_PULSE = 8.0, 30.0, 1.5


def peak_field(op, theta_EE, ell_par, ell_perp):
    """Centered finite pulse; return the active-cell deviation field at PEAK extent."""
    mask = (X ** 2 + Y ** 2 <= R_PULSE ** 2).astype(float)

    def stim_fn(t):
        return (A_PULSE * mask) if t < T_PULSE else (0.0 * mask)

    _ext, _front, pf = integrate_lif_field(
        op, stim_fn, dt=DT, t_max=T_MAX, b_a=0.0, theta_EE=theta_EE,
        n=N, L=L, ell_par=ell_par, ell_perp=ell_perp, l_inh=L_INH,
        return_peak_field=True,
    )
    return pf - op["nuE"]


def principal_axis(field):
    """Angle (deg, mod 180) of the principal axis of the active region + anisotropy ratio."""
    m = field > DETECT
    if m.sum() < 5:
        return float("nan"), 1.0
    w = field[m]
    xs, ys = X[m], Y[m]
    xm = np.average(xs, weights=w); ym = np.average(ys, weights=w)
    cxx = np.average((xs - xm) ** 2, weights=w); cyy = np.average((ys - ym) ** 2, weights=w)
    cxy = np.average((xs - xm) * (ys - ym), weights=w)
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
    op = mean_field(1.0)
    rows = []
    # anisotropic E->E, rotate theta_EE
    for th in [0.0, 30.0, 60.0, 90.0, 135.0]:
        pf = peak_field(op, np.radians(th), ELL_PAR, ELL_PERP)
        thp, ratio = principal_axis(pf)
        rows.append({"kernel": "anisotropic", "theta_EE": th, "theta_prop": thp,
                     "axis_err_deg": axis_diff(thp, th), "anisotropy_ratio": ratio})
    # isotropic control (no preferred axis) -- rotate the (meaningless) "theta_EE", expect NO tracking
    iso = []
    for th in [0.0, 45.0, 90.0]:
        pf = peak_field(op, np.radians(th), ELL_ISO, ELL_ISO)
        thp, ratio = principal_axis(pf)
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
                 "control produces no directional template. PASS supports SEF-HFO vs geometry artifact. "
                 "Dynamics from canonical src/sef_hfo_lif.integrate_lif_field (task 21 fold-in)."),
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
