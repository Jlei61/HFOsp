"""[WIP / STALLED — does NOT yet work in the rate field; see §9.8.]

!!! The SNN onset window (first ~8 ms, pre-saturation) is TOO EARLY for the rate
    field: it is slow (tau_m=20 ms), so a kicked event only develops over ~20-50 ms
    (peak coh ~50 ms; see /tmp kick diagnostic in session log). calibrate() at
    ONSET_MS=8 returns n_active=0 on every angle. To make this work: (1) adapt the
    onset window to the rate-field timescale (measure the front in the ~15-40 ms
    pre-saturation growth phase, not 8 ms), and (2) for NOISE events, add per-event
    isolation (the front of EACH event, not a whole-frame aggregate). This is the
    pre-registered next measure (§9.6/§9.8) -- NOT a measure #4, the adaptation of
    the SNN's onset-front to the rate field's slower dynamics. Committed as a WIP
    marker for clean resume; calibrate() is currently non-functional.

Step 1 direction discriminator via ONSET-FRONT (adopted from the spiking GT,
spiking_gt_validation_2026-06-03.md §anisotropy-onset-front).

The onset-front = principal axis of the field active in the FIRST ~8 ms after a
kick (pre-saturation leading edge), NOT the whole-event shape or a whole-frame
aggregate (those were grid-contaminated, §9.5). PER-EVENT by construction.

This module has two entry points:
  * calibrate()  -- §9.6-mandated FIRST step: validate the measure on the 0d
    DETERMINISTIC pulse at OFF-AXIS theta=30/60 (+ 0/45/90 SNN set) + isotropic
    control, BEFORE touching noise. If a clean deterministic kick's onset-front
    does not track theta_EE off-axis, the measure is broken -- fix here, never on
    noise.
  * (noise application is added only AFTER calibrate() passes.)

Measure: kick a small localized disk at quiet rest (drive ~0.6), integrate to
T_PULSE + ONSET_MS, take the field at that time as the onset front, smooth
(coh_len=ELL_PAR), threshold, and compute the covariance principal axis +
elongation ratio. Honest caveat inherited from the SNN: the isotropic control is
a near-fizzle, so the claim is "anisotropic E->E reach is NECESSARY for both
propagation and a directional front", not "direction read from a strong isotropic
wave".

Run: python scripts/run_sef_hfo_step1_onsetfront.py
"""
import json
from pathlib import Path

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sef_hfo_lif import DETECT, ELL_PAR, ELL_PERP, _grid, integrate_lif_field, mean_field
from src.sef_hfo_field import convolve_periodic, isotropic_gaussian

OUT = Path("results/topic4_sef_hfo/step1_noise")
N, L, DT = 64, 16.0, 0.25
DRIVE = 0.6              # SNN-corrected interictal operating point
ONSET_MS = 8.0          # SNN onset-front window (pre-saturation)
R_KICK, A_KICK, T_PULSE = 0.8, 10.0, 2.0   # small localized brief kick
X, Y = _grid(N, L)
KCOH = isotropic_gaussian(N, L, ELL_PAR)


def onset_front_axis(op, theta_EE_deg, ell_par, ell_perp):
    """Principal axis (deg mod 180) + elongation ratio of the onset front."""
    mask = (X ** 2 + Y ** 2 <= R_KICK ** 2).astype(float)

    def stim(t):
        return (A_KICK * mask) if t < T_PULSE else (0.0 * mask)

    _ext, _front, pf = integrate_lif_field(
        op, stim, dt=DT, t_max=T_PULSE + ONSET_MS, theta_EE=np.radians(theta_EE_deg),
        ell_par=ell_par, ell_perp=ell_perp, n=N, L=L, return_field=True,
    )
    dev = convolve_periodic(pf - op["nuE"], KCOH)
    m = dev > DETECT
    if m.sum() < 5:
        return float("nan"), 1.0, int(m.sum())
    w = dev[m]
    xs, ys = X[m], Y[m]
    xm = np.average(xs, weights=w)
    ym = np.average(ys, weights=w)
    cxx = np.average((xs - xm) ** 2, weights=w)
    cyy = np.average((ys - ym) ** 2, weights=w)
    cxy = np.average((xs - xm) * (ys - ym), weights=w)
    ev, evec = np.linalg.eigh(np.array([[cxx, cxy], [cxy, cyy]]))
    v = evec[:, -1]
    ang = float(np.degrees(np.arctan2(v[1], v[0])) % 180.0)
    return ang, float(np.sqrt(ev[-1] / max(ev[0], 1e-12))), int(m.sum())


def axis_diff(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def calibrate():
    op = mean_field(DRIVE)
    aniso, iso = [], []
    for theta in [0.0, 30.0, 45.0, 60.0, 90.0]:
        ang, ratio, npix = onset_front_axis(op, theta, ELL_PAR, ELL_PERP)
        aniso.append(dict(theta_EE=theta, theta_front=ang, axis_err_deg=axis_diff(ang, theta),
                          ratio=ratio, n_active=npix))
    ang, ratio, npix = onset_front_axis(op, 0.0, 0.45, 0.45)  # isotropic E->E
    iso.append(dict(theta_EE="isotropic", theta_front=ang, ratio=ratio, n_active=npix))

    aniso_tracks = all(np.isfinite(r["theta_front"]) and r["axis_err_deg"] < 20.0 and r["ratio"] > 1.3
                       for r in aniso)
    iso_no_axis = all(r["ratio"] < 1.3 for r in iso)
    verdict = dict(calibration_aniso_tracks=bool(aniso_tracks),
                   isotropic_no_axis=bool(iso_no_axis),
                   calibration_passed=bool(aniso_tracks))  # iso fizzle is expected, not required for calib gate
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "onsetfront_calibration_0d.json").write_text(json.dumps(
        dict(params=dict(N=N, L=L, drive=DRIVE, onset_ms=ONSET_MS, r_kick=R_KICK,
                         a_kick=A_KICK, ell_par=ELL_PAR, ell_perp=ELL_PERP),
             anisotropic=aniso, isotropic=iso, verdict=verdict), indent=2, default=float))
    print(f"=== onset-front 0d calibration (drive={DRIVE}, deterministic kick) ===")
    for r in aniso:
        print(f"  theta_EE={r['theta_EE']:5.0f} theta_front={r['theta_front']:6.1f} "
              f"axis_err={r['axis_err_deg']:5.1f} ratio={r['ratio']:.2f} n_active={r['n_active']}")
    for r in iso:
        print(f"  isotropic         theta_front={r['theta_front']:6.1f} ratio={r['ratio']:.2f} n_active={r['n_active']}")
    print(f"CALIBRATION PASSED = {verdict['calibration_passed']} "
          f"(aniso tracks={aniso_tracks}; iso no-axis={iso_no_axis})")
    return verdict


if __name__ == "__main__":
    calibrate()
