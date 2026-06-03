"""TDD tests for src/sef_hfo_lif — canonical LIF-derived rate field.

Tests lock known-good behavior validated in scripts/sef_hfo_step0b_lif.py and
scripts/sef_hfo_step0d_anisotropy_control.py.

Test 1: mean_field at ratio=1.0 → LOW E rate (validated vs coworker1's SI regime).
Test 2: LIF f-I has HIGH gain at LOW rate (property the sigmoid lacks — preflight gate).
Test 3: steady state holds exactly (fsolve op is a true fixed point of the field).
Test 4: finite off-center disk pulse → self_limited_propagation (Step-0b gate behavior).

Note: tests 3 and 4 run a field simulation; each takes ~10–60s.
"""
import numpy as np
import pytest

from src.sef_hfo_lif import (
    C_EE,
    DETECT,
    TAU_ME,
    TAU_MI,
    TREF_E,
    TREF_I,
    W_EE,
    _DEFAULT_L,
    _DEFAULT_N,
    _STIM_R,
    _STIM_T,
    _STIM_X0,
    _grid,
    _lut,
    classify_response,
    integrate_lif_field,
    lif_gains,
    lif_rate,
    mean_field,
)


# ---------------------------------------------------------------------------
# Test 1: mean_field self-consistency at ratio=1.0
# ---------------------------------------------------------------------------

def test_mean_field_low_E_rate():
    """mean_field(1.0) must produce a LOW E firing rate and satisfy self-consistency.

    Validated against coworker1's SI (sub-threshold irregular) regime:
    nuE should be well below 5 Hz (in kHz: < 5e-3).
    Self-consistency check: lif_rate recomputed at (muE, sE) must match nuE to < 1e-4 kHz.
    """
    op = mean_field(1.0)
    nuE_Hz = op["nuE"] * 1000.0
    assert 0.05 < nuE_Hz < 5.0, (
        f"E firing rate {nuE_Hz:.3f} Hz not in LOW regime (0.05–5 Hz)"
    )
    # Self-consistency residual
    nu_recomputed = lif_rate(op["muE"], op["sE"], TAU_ME, TREF_E)
    assert abs(nu_recomputed - op["nuE"]) < 1e-4, (
        f"Self-consistency residual {abs(nu_recomputed - op['nuE']):.2e} kHz exceeds 1e-4"
    )


# ---------------------------------------------------------------------------
# Test 2: LIF f-I curve — high gain at low rate
# ---------------------------------------------------------------------------

def test_lif_high_gain_at_low_rate():
    """The LIF f-I curve must have HIGH GAIN at LOW firing rate.

    This is the preflight property the sigmoid (F_eff) lacks: at the low-rate
    operating point the recurrent E→E loop gain (∂ν_E/∂μ_E)·τ_m,E·C_EE·W_EE
    must exceed 1.5.  The sigmoid saturates and cannot provide this.

    Method: scan mu to find a value giving ν_E ≈ 5 Hz (0.005 kHz), then compute
    the loop gain at that point.
    """
    target_nu = 0.005  # kHz ≈ 5 Hz
    # Use a representative sigma from the ratio=1.0 operating point
    op = mean_field(1.0)
    sigma = op["sE"]
    # Scan mu to find the point closest to target_nu
    mus = np.linspace(op["muE"] - 5.0, op["muE"] + 20.0, 500)
    rates = np.array([lif_rate(m, sigma, TAU_ME, TREF_E) for m in mus])
    idx = int(np.argmin(np.abs(rates - target_nu)))
    mu_5hz = mus[idx]
    # Finite-difference gain at that mu
    h = 1e-3
    dnu = (lif_rate(mu_5hz + h, sigma, TAU_ME, TREF_E)
           - lif_rate(mu_5hz - h, sigma, TAU_ME, TREF_E)) / (2 * h)
    loop_gain = dnu * TAU_ME * C_EE * W_EE
    assert loop_gain > 1.5, (
        f"LIF loop gain at ~5 Hz = {loop_gain:.3f}, expected > 1.5"
    )


# ---------------------------------------------------------------------------
# Test 3: fsolve operating point is a true fixed point of the field
# ---------------------------------------------------------------------------

def test_steady_state_holds():
    """Integrating the field with NO stim from the fsolve op must stay at the fixed point.

    Criterion: max |rE(x,t) - op["nuE"]| over all space at t=80 ms < 5e-3 kHz.
    This verifies the fsolve solution is a genuine fixed point of the spatial PDE.
    Uses return_field=True to directly check field deviation (not a proxy via ext).
    """
    op = mean_field(1.0)
    ext, front, rE_final = integrate_lif_field(
        op,
        stim_fn=lambda t: 0.0,
        dt=0.25,
        t_max=80.0,
        b_a=0.0,
        tau_a=80.0,
        theta_EE=0.0,
        n=_DEFAULT_N,
        L=_DEFAULT_L,
        return_field=True,
    )
    max_dev = float(np.max(np.abs(rE_final - op["nuE"])))
    assert max_dev < 5e-3, (
        f"Fixed-point violated: max |rE - nuE| = {max_dev:.2e} kHz (expected < 5e-3)"
    )


# ---------------------------------------------------------------------------
# Test 4: finite off-center disk pulse → self_limited_propagation
# ---------------------------------------------------------------------------

def test_self_limited_propagation():
    """Off-center disk pulse (A=8mV, step0b defaults) must yield self_limited_propagation.

    Validated behavior from scripts/sef_hfo_step0b_lif.py scan_point(ratio=1.0, b_a=0.0):
    A=8.0 → 'self_limited_propagation' with front advance > 3mm and active fraction < 0.5.

    Assertions:
      - label == 'self_limited_propagation'
      - front advance > 3mm  (propagation happened)
      - max active fraction < 0.5  (localized, did not go runaway)
    """
    op = mean_field(1.0)
    n, L = _DEFAULT_N, _DEFAULT_L
    X_grid, Y_grid = _grid(n, L)
    mask = ((X_grid - _STIM_X0) ** 2 + Y_grid ** 2 <= _STIM_R ** 2).astype(float)
    A = 8.0

    def stim_fn(t):
        return A * mask if t < _STIM_T else 0.0 * mask

    ext, front = integrate_lif_field(
        op,
        stim_fn=stim_fn,
        dt=0.25,
        t_max=300.0,
        b_a=0.0,
        tau_a=80.0,
        theta_EE=0.0,
        n=n,
        L=L,
    )
    lbl, info = classify_response(ext, front, stim_x0=_STIM_X0, stim_r=_STIM_R, dt=0.25)

    assert lbl == "self_limited_propagation", (
        f"Expected 'self_limited_propagation', got '{lbl}' "
        f"(adv_mm={info['adv_mm']:.2f}, max_ext={info['max_ext']:.3f}, returned={info['returned']})"
    )
    assert info["adv_mm"] >= 3.0, (
        f"Front advance {info['adv_mm']:.2f} mm < 3mm — propagation did not occur"
    )
    assert info["max_ext"] < 0.5, (
        f"Max active fraction {info['max_ext']:.3f} >= 0.5 — response went runaway"
    )


# ---------------------------------------------------------------------------
# Test 5: anisotropy ROTATION discriminator (load-bearing; tripwire for Step 0d)
# ---------------------------------------------------------------------------

def _principal_axis_deg_ratio(field, X, Y, thr):
    """Principal-axis angle (deg, mod 180) + anisotropy ratio of the active region."""
    m = field > thr
    if m.sum() < 5:
        return float("nan"), 1.0
    w = field[m]
    xs, ys = X[m], Y[m]
    xm = np.average(xs, weights=w)
    ym = np.average(ys, weights=w)
    cxx = np.average((xs - xm) ** 2, weights=w)
    cyy = np.average((ys - ym) ** 2, weights=w)
    cxy = np.average((xs - xm) * (ys - ym), weights=w)
    ev, evec = np.linalg.eigh(np.array([[cxx, cxy], [cxy, cyy]]))
    v = evec[:, -1]
    ang = np.degrees(np.arctan2(v[1], v[0])) % 180.0
    return float(ang), float(np.sqrt(ev[-1] / max(ev[0], 1e-12)))


def _axis_diff(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def test_anisotropy_rotation_discriminator():
    """Load-bearing discriminator: propagation axis tracks E→E connectivity, not geometry.

    Fast tripwire for scripts/sef_hfo_step0d_anisotropy_control.py (the durable
    5-angle + 3-isotropic regression).  Locks both halves on the canonical
    integrate_lif_field:
      - anisotropic E→E at theta_EE=60° (non-axis-aligned, so a pass cannot be a
        grid-alignment artifact) → active-region principal axis tracks 60°
        (err < 20°) and is clearly elongated (ratio > 1.3);
      - isotropic E→E → no preferred axis (ratio < 1.3).
    """
    op = mean_field(1.0)
    N, L = 96, 16.0
    X, Y = _grid(N, L)
    ell_par, ell_perp, l_inh = 0.9, 0.45, 0.45
    R, A, Tp = 1.5, 8.0, 30.0
    mask = (X ** 2 + Y ** 2 <= R ** 2).astype(float)

    def stim(t):
        return (A * mask) if t < Tp else (0.0 * mask)

    # anisotropic E→E rotated to 60°
    _e, _f, pf = integrate_lif_field(
        op, stim, dt=0.25, t_max=150.0, b_a=0.0, theta_EE=np.radians(60.0),
        n=N, L=L, ell_par=ell_par, ell_perp=ell_perp, l_inh=l_inh,
        return_peak_field=True,
    )
    ang, ratio = _principal_axis_deg_ratio(pf - op["nuE"], X, Y, DETECT)
    assert _axis_diff(ang, 60.0) < 20.0, (
        f"theta_prop {ang:.1f}° does not track theta_EE=60° (err {_axis_diff(ang, 60.0):.1f}°)"
    )
    assert ratio > 1.3, f"anisotropic ratio {ratio:.2f} <= 1.3 (no directional elongation)"

    # isotropic control: must NOT produce a preferred axis
    _e, _f, pf2 = integrate_lif_field(
        op, stim, dt=0.25, t_max=150.0, b_a=0.0, theta_EE=0.0,
        n=N, L=L, ell_par=0.6, ell_perp=0.6, l_inh=l_inh,
        return_peak_field=True,
    )
    _ang2, ratio2 = _principal_axis_deg_ratio(pf2 - op["nuE"], X, Y, DETECT)
    assert ratio2 < 1.3, f"isotropic ratio {ratio2:.2f} >= 1.3 (spurious directional axis)"
