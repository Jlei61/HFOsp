"""activity-excess K layer contract tests (plan section 2)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src", "snn_engine"))
import src.snn_engine.activity_excess_k as AK        # noqa: E402
import src.topic4_fcxr_hyb1 as H                     # noqa: E402

NG = 8
NV = NG * NG


def _cfg(**over):
    d = dict(b_v=np.full(NV, 4.0), eps=0.4, q_K=H.Q_K, n_grid=NG, dx_mm=0.625, dt_ion_ms=0.5)
    d.update(over)
    return AK.ActivityExcessKConfig(**d)


def _layer(N=256, **over):
    cfg = _cfg(**over)
    voxel = (np.arange(N) % NV).astype(np.int32)
    return AK.ActivityExcessK(N, voxel, cfg), cfg


# ---------------------------------------------------------------- source is strictly zero below bg
def test_source_is_exactly_zero_at_and_below_the_registered_background():
    src = AK.excess_source(np.array([0.0, 2.0, 3.999, 4.0]), np.full(4, 4.0), 0.4, H.Q_K)
    assert np.all(src == 0.0)


def test_source_turns_on_only_above_background_and_grows_monotonically():
    b = np.full(4, 4.0)
    src = AK.excess_source(np.array([4.5, 5.0, 8.0, 20.0]), b, 0.4, H.Q_K)
    assert src[0] > 0.0 and np.all(np.diff(src) > 0)


def test_rhs_is_identically_zero_at_rest():
    """dK = 0 with sub-background load must give EXACTLY 0.0 -- the structural fixed point."""
    cfg = _cfg()
    rhs = AK.d_dK_dt(np.zeros((NG, NG)), np.full((NG, NG), 3.0), cfg)
    assert np.all(rhs == 0.0)


def test_rest_is_a_fixed_point_even_with_a_spatially_varying_background():
    rng = np.random.default_rng(3)
    b = rng.uniform(1.0, 9.0, NV)
    cfg = _cfg(b_v=b, eps=0.3)
    rhs = AK.d_dK_dt(np.zeros((NG, NG)), (b * 0.5).reshape(NG, NG), cfg)
    assert np.all(rhs == 0.0)


def test_a_softplus_source_would_break_the_fixed_point():
    """Records the failure mode the deadband exists to prevent."""
    leak = 0.4 * np.log1p(np.exp((3.0 - 4.0) / 0.4))
    assert leak > 0.0
    assert AK.deadband_positive(np.array([3.0 - 4.0]), 0.4)[0] == 0.0


# ---------------------------------------------------------------- clearance / diffusion numerics
def test_clearance_relaxes_toward_zero_with_the_registered_tau():
    cfg = _cfg()
    dK = np.full((NG, NG), 1.0)
    rhs = AK.d_dK_dt(dK, np.zeros((NG, NG)), cfg)      # uniform -> laplacian is 0
    assert rhs[0, 0] == pytest.approx(-1.0 / cfg.tau_K_s)


def test_diffusion_conserves_total_dK_zero_flux_boundary():
    cfg = _cfg()
    rng = np.random.default_rng(1)
    dK = rng.random((NG, NG))
    from src.topic4_fcxr_ion import diffusion_term
    assert float(diffusion_term(dK, dx_mm=cfg.dx_mm, D=cfg.D_K).sum()) == pytest.approx(0.0, abs=1e-15)


def test_explicit_step_is_far_inside_the_stability_limit():
    cfg = _cfg()
    dt_s = cfg.dt_ion_ms * 1e-3
    assert dt_s / cfg.tau_K_s < 0.01
    assert 4.0 * cfg.D_K * dt_s / cfg.dx_mm ** 2 < 0.01


def test_steady_state_matches_the_analytic_value_for_a_held_excess():
    lay, cfg = _layer()
    excess = 10.0
    load = np.full(NV, cfg.b_v[0] + excess)
    for _ in range(40000):                                     # ~20 s at 0.5 ms
        lay.dK_grid += (cfg.dt_ion_ms * 1e-3) * AK.d_dK_dt(
            lay.dK_grid, load.reshape(NG, NG), cfg)
    want = cfg.tau_K_s * AK.excess_source(load, cfg.b_v, cfg.eps, cfg.q_K)[0]
    assert float(lay.dK_grid.mean()) == pytest.approx(want, rel=2e-3)


# ---------------------------------------------------------------- membrane coupling
def test_membrane_current_is_zero_when_dK_is_zero():
    assert np.all(AK.membrane_current_from_dK(np.zeros(10), 1.0) == 0.0)


def test_membrane_current_is_positive_and_monotone_in_dK():
    c = AK.membrane_current_from_dK(np.array([0.0, 0.2, 1.0, 3.0]), 1.0)
    assert c[0] == 0.0 and np.all(np.diff(c) > 0)


def test_membrane_current_reaches_BOTH_E_and_I_cells():
    """The plan requires E and I to both receive it; the current is per-cell with no E/I split, so
    the test asserts the adapter adds the same vector to the whole drive."""
    lay, _ = _layer(N=16)
    lay.dK_grid += 0.5
    lay._cur = AK.membrane_current_from_dK(lay.dK_grid.ravel()[lay.cell_voxel], lay.cfg.g_dK)

    class _MZ:
        NE = 10
        def membrane_terms(self, n):
            return np.zeros(n), None, None
        def step(self, *a):
            pass
    a = AK.ExcessKMZAdapter(_MZ(), lay)
    drive, g_rel, g_rev = a.membrane_terms(16)
    assert np.all(drive > 0) and drive.size == 16
    assert g_rel is None and g_rev is None          # never a conductance


def test_adapter_is_a_noop_when_the_layer_is_disabled():
    lay, _ = _layer(N=16, enabled=False)

    class _MZ:
        NE = 10
        def membrane_terms(self, n):
            return np.full(n, 7.0), None, None
    a = AK.ExcessKMZAdapter(_MZ(), lay)
    assert np.all(a.membrane_terms(16)[0] == 7.0)


def test_adapter_does_not_synthesise_absent_engine_attributes():
    """The engine guards branches with hasattr(slow, 'q_I' / 'uses_shunt'); inventing them would
    silently switch execution paths."""
    lay, _ = _layer(N=16)

    class _MZ:
        NE = 10
    a = AK.ExcessKMZAdapter(_MZ(), lay)
    assert not hasattr(a, "q_I") and not hasattr(a, "uses_shunt") and not hasattr(a, "nE")


# ---------------------------------------------------------------- accumulation / determinism
def test_voxel_load_is_a_per_cell_rate_not_a_raw_count():
    lay, cfg = _layer(N=NV * 4)                 # exactly 4 cells per voxel
    spk = np.zeros(lay.N, bool)
    spk[lay.cell_voxel == 0] = True             # all 4 cells of voxel 0 fire once
    lay.accumulate(spk)
    lay.update()
    # 4 spikes / 4 cells / 0.5 ms  ->  2000 Hz, hugely above background -> voxel 0 rises, others not
    assert lay.dK_grid.ravel()[0] > 0.0
    assert np.all(lay.dK_grid.ravel()[1:] == 0.0)


def test_empty_voxels_never_produce_a_source():
    N = 8
    voxel = np.zeros(N, np.int32)               # every cell in voxel 0; all others empty
    lay = AK.ActivityExcessK(N, voxel, _cfg())
    lay.accumulate(np.ones(N, bool))
    lay.update()
    assert np.all(lay.dK_grid.ravel()[1:] == 0.0)


def test_two_identical_layers_step_identically():
    a, _ = _layer(); b, _ = _layer()
    rng = np.random.default_rng(11)
    for _ in range(50):
        spk = rng.random(a.N) < 0.02
        for L in (a, b):
            L.accumulate(spk); L.update()
    assert np.array_equal(a.dK_grid, b.dK_grid)


def test_snapshot_restart_reproduces_the_continuous_run():
    a, _ = _layer(); b, _ = _layer()
    rng = np.random.default_rng(5)
    spikes = [rng.random(a.N) < 0.03 for _ in range(40)]
    for s in spikes:
        a.accumulate(s); a.update()
    for s in spikes[:17]:
        b.accumulate(s); b.update()
    st = b.state_dict()
    c, _ = _layer()
    c.load_state_dict(st)
    for s in spikes[17:]:
        c.accumulate(s); c.update()
    assert np.allclose(a.dK_grid, c.dK_grid, rtol=0, atol=0)
    assert c.n_updates == a.n_updates


def test_out_of_band_dK_fails_closed_instead_of_clamping():
    lay, _ = _layer(dk_bounds=(-1e-9, 1e-3))
    lay.accumulate(np.ones(lay.N, bool))
    with pytest.raises(AK.ExcessKSafetyError):
        for _ in range(50):
            lay.accumulate(np.ones(lay.N, bool)); lay.update()


def test_mismatched_voxel_map_fails_closed():
    with pytest.raises(ValueError):
        AK.ActivityExcessK(10, np.zeros(9, np.int32), _cfg())


def test_mismatched_background_length_fails_closed():
    with pytest.raises(ValueError):
        AK.ActivityExcessK(10, np.zeros(10, np.int32), _cfg(b_v=np.full(NV + 1, 4.0)))


def test_duty_cycle_counts_only_occupied_voxels():
    N = 8
    voxel = np.zeros(N, np.int32)
    lay = AK.ActivityExcessK(N, voxel, _cfg(dk_bounds=(-1e-9, 1e6)))
    lay.accumulate(np.ones(N, bool)); lay.update()      # voxel 0 above background
    assert lay.duty_cycle() == pytest.approx(1.0)       # 1 of 1 OCCUPIED voxel, not 1 of 64


def test_background_envelope_is_the_registered_quantile_per_voxel():
    load = np.stack([np.arange(100.0), np.arange(100.0) * 2], axis=1)
    b = AK.background_envelope(load, 0.99)
    assert b[1] == pytest.approx(2 * b[0], rel=1e-9)
