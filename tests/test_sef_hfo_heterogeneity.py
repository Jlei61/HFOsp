import numpy as np
import pytest
from src.sef_hfo_lif import lif_rate, V_TH, TAU_ME, TREF_E, mean_field, integrate_lif_field
from src.sef_hfo_field import _grid
from scripts.sef_hfo_step0d_anisotropy_control import principal_axis
from src.sef_hfo_heterogeneity import phi_eff_vth


def test_lif_rate_vth_default_matches_global():
    """Adding v_th param must not change existing behavior at the default."""
    r_default = lif_rate(5.0, 4.0, TAU_ME, TREF_E)
    r_explicit = lif_rate(5.0, 4.0, TAU_ME, TREF_E, v_th=V_TH)
    assert r_explicit == r_default


def test_lif_rate_vth_lower_threshold_raises_rate():
    """Lowering threshold (easier to fire) raises rate at fixed mu,sigma —
    a monotone single-neuron sanity, NOT the closed-loop direction claim."""
    r_hi = lif_rate(5.0, 4.0, TAU_ME, TREF_E, v_th=V_TH)
    r_lo = lif_rate(5.0, 4.0, TAU_ME, TREF_E, v_th=V_TH - 2.0)
    assert r_lo > r_hi


def _offcenter_pulse(A=8.0, T=30.0, R=2.0, x0=-3.0, n=96, L=12.0):
    """Off-center seed matching the Step-0b validated chain."""
    X, Y = _grid(n, L)
    disk = ((X - x0) ** 2 + Y ** 2) <= R ** 2
    return lambda t: (A * disk if t < T else 0.0)


def test_geometry_rho_controls_directionality():
    """ρ<1 (anisotropic) gives a stronger directional template than ρ=1 (isotropic)."""
    op = mean_field(1.0)
    def ratio_for(ell_par, ell_perp):
        out = integrate_lif_field(op, _offcenter_pulse(), theta_EE=0.0,
                                  ell_par=ell_par, ell_perp=ell_perp,
                                  return_peak_field=True, t_max=150.0)
        _angle, ratio = principal_axis(out[2] - op["nuE"])
        return ratio
    r_aniso = ratio_for(0.9, 0.45)   # rho = 0.5
    r_iso = ratio_for(0.6, 0.6)      # rho = 1.0
    assert r_iso < 1.5                       # isotropic: weak directional bias (off-center seed)
    assert r_aniso > r_iso + 0.3             # anisotropy creates a directional template


def test_phi_eff_zero_std_reduces_to_lif_rate():
    """vth_std=0 → no spread → exactly lif_rate at vth_mean."""
    r = phi_eff_vth(5.0, 4.0, TAU_ME, TREF_E, vth_mean=V_TH, vth_std=0.0)
    assert abs(r - lif_rate(5.0, 4.0, TAU_ME, TREF_E, v_th=V_TH)) < 1e-9


def test_phi_eff_quadrature_matches_dense_average():
    """Gauss–Hermite must match a dense trapezoid average."""
    mu, sig, vm, vs = 5.0, 4.0, V_TH, 2.0
    grid = np.linspace(vm - 5 * vs, vm + 5 * vs, 4001)
    w = np.exp(-0.5 * ((grid - vm) / vs) ** 2); w /= w.sum()
    dense = float(np.sum(w * np.array([lif_rate(mu, sig, TAU_ME, TREF_E, v_th=v) for v in grid])))
    gh = phi_eff_vth(mu, sig, TAU_ME, TREF_E, vth_mean=vm, vth_std=vs)
    assert abs(gh - dense) < 1e-4


def test_closed_loop_leading_canonical_op_stable():
    """Canonical white-noise operating point is robustly STABLE.
    Framework banner: max Re λ≈−0.05, k=0. Locks the promotion."""
    from src.sef_hfo_lif import mean_field, lif_gains, closed_loop_leading
    op = mean_field(1.0)
    g = lif_gains(op)
    res = closed_loop_leading(g["E"], g["I"])
    assert res["re_max"] < 0.0           # stable
    assert res["re_max"] > -0.15         # ~ -0.05, not wildly off
    assert res["k_star"] < 0.3           # dominant mode near k=0 (no finite-k Hopf)
