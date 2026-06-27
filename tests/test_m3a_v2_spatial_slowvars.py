"""TDD (red scaffold) for M3A-v2 spatial slow-variable field.

Spatial generalization of M3A-A2 (scalar q_core/q_global). Encodes the §B5 contracts:
  - off-by-default == slow=None (byte parity) + uniform field == scalar RegionalResource
  - q_I / g_K field dynamics: depletion/recovery, SPATIAL locality, bounds, sigma_q>sigma_K
  - source-space onset-gradient axis score (methodological lock) + the four-state classifier
  - proxy phase plane

Spec:  docs/snn_core_model_equations.md §B5
Plan:  docs/superpowers/plans/2026-06-28-sef-hfo-m3a-v2-spatial-slowvar-field-plan.md

STATUS: every test is RED -- the implementation modules are stubs (NotImplementedError).
Implement task-by-task per the plan; each task turns its tests green. Run the fast set with
`pytest tests/test_m3a_v2_spatial_slowvars.py -m "not slow"`.
"""
import hashlib
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

from params import Params                              # noqa: E402
from connectivity import place_neurons                 # noqa: E402
from connectivity_rot import build_connectivity_rot    # noqa: E402
from kick_probe import simulate_kick                    # noqa: E402
from slow_vars import RegionalResource, RegionalResourceConfig  # noqa: E402  (v1 reduction target)
from slow_field import (                                # noqa: E402
    SpatialSlowField, SpatialSlowFieldConfig, saturation, aq_drive, firing_rate_field,
    sample_field_at,
)
from src.topic4_m3a_v2_phenotype import (              # noqa: E402
    recruitment_area, axis_score, offaxis_fraction, participation_ratio, event_recovery,
    classify_event, PhenotypeGates, region_pressure, proxy_phase_point,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _synthetic_positions(L=6.0, nE=400, nI=100, seed=0):
    rng = np.random.default_rng(seed)
    posE = rng.uniform(0.0, L, size=(nE, 2))
    posI = rng.uniform(0.0, L, size=(nI, 2))
    labels = np.concatenate([np.zeros(nE, int), np.ones(nI, int)])
    return posE, posI, labels


def _build(L=6.0, density=100.0, T=300.0, nu=0.6, seed=1):
    """Small deterministic SNN (mirrors tests/test_m3a_quasistatic_slowvars.py::_build)."""
    p = Params(L=L, density=density, T=T, dt=0.1, nu_ext_ratio=nu, seed=seed)
    rng = np.random.default_rng(seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=np.radians(45), AR=2.0)
    return p, net, NE, NI


def _run(p, net, NE, NI, *, slow=None, V_th_per_neuron=None):
    net["rng"] = np.random.default_rng(1)
    return simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, slow=slow,
                         V_th_per_neuron=V_th_per_neuron)


def _sha(res):
    return hashlib.sha1(res["E_spk_bool"].tobytes()).hexdigest()[:16]


# ===========================================================================
# Task 1 -- config structural invariants (sigma_q > sigma_K, eta_I >= eta_E)
# ===========================================================================
def test_config_validate_accepts_locked_defaults():
    SpatialSlowFieldConfig().validate()                # defaults: sigma_q=1.5 > sigma_K=0.5; eta_I=1>=eta_E=0.3


def test_config_validate_rejects_sigma_q_not_greater_than_sigma_K():
    with pytest.raises(ValueError):
        SpatialSlowFieldConfig(sigma_q=0.5, sigma_K=0.5).validate()   # WIDE>NARROW invariant breached


def test_config_validate_rejects_eta_I_below_eta_E():
    with pytest.raises(ValueError):
        SpatialSlowFieldConfig(eta_E=1.0, eta_I=0.3).validate()


# ===========================================================================
# Task 2 -- saturation f(a) = [a-a0]_+ / (a50 + [a-a0]_+)
# ===========================================================================
def test_saturation_zero_at_and_below_onset():
    assert saturation(0.0, a0=0.0, a50=1.0) == 0.0
    assert saturation(-5.0, a0=0.0, a50=1.0) == 0.0            # [.]_+ floors below onset


def test_saturation_half_at_a50():
    assert abs(saturation(1.0, a0=0.0, a50=1.0) - 0.5) < 1e-12  # a = a0 + a50 -> 1/2


def test_saturation_approaches_one_and_is_monotone():
    assert saturation(1e6, a0=0.0, a50=1.0) > 0.999
    xs = np.linspace(0.0, 10.0, 50)
    f = saturation(xs, a0=0.0, a50=1.0)
    assert np.all(np.diff(f) >= -1e-15) and np.all((f >= 0.0) & (f < 1.0))


# ===========================================================================
# Task 3 -- firing-rate field (bin + isotropic-Gaussian convolution) + sampling
# ===========================================================================
def test_firing_rate_field_single_spike_peaks_near_its_position():
    L, n_grid = 6.0, 32
    pos = np.array([[1.5, 4.5]])                                # one neuron
    spk = np.array([True])
    field = firing_rate_field(spk, pos, L, n_grid, sigma=0.5)
    iy, ix = np.unravel_index(int(np.argmax(field)), field.shape)
    cx, cy = (ix + 0.5) * L / n_grid, (iy + 0.5) * L / n_grid
    assert abs(cx - 1.5) < L / n_grid and abs(cy - 4.5) < L / n_grid
    assert field.shape == (n_grid, n_grid)


def test_firing_rate_field_wider_sigma_spreads_more():
    L, n_grid = 6.0, 32
    pos = np.array([[3.0, 3.0]]); spk = np.array([True])
    narrow = firing_rate_field(spk, pos, L, n_grid, sigma=0.3)
    wide = firing_rate_field(spk, pos, L, n_grid, sigma=1.2)
    # wider kernel -> lower peak, larger above-half-max footprint
    assert wide.max() < narrow.max()
    assert (wide > 0.5 * wide.max()).sum() > (narrow > 0.5 * narrow.max()).sum()


def test_sample_field_at_recovers_grid_values():
    L, n_grid = 6.0, 8
    field = np.arange(n_grid * n_grid, dtype=float).reshape(n_grid, n_grid)
    centers = (np.arange(n_grid) + 0.5) * L / n_grid
    pos = np.array([[centers[3], centers[5]]])                  # a known cell center (ix=3, iy=5)
    val = sample_field_at(field, pos, L, n_grid)
    assert val.shape == (1,) and abs(val[0] - field[5, 3]) < 1e-9


# ===========================================================================
# Task 4 -- apply_currents: parity, coupling sign, v1 reduction
# ===========================================================================
def test_apply_currents_off_is_IE_minus_II():
    posE, posI, _ = _synthetic_positions()
    N = posE.shape[0] + posI.shape[0]
    fld = SpatialSlowField(N, 18.0, posE, posI, L=6.0,
                           cfg=SpatialSlowFieldConfig(k_q=0.0, k_K=0.0, q_init=1.0))
    I_E = np.arange(N, dtype=float) + 1.0
    I_I = np.arange(N, dtype=float) * 0.5
    assert np.array_equal(fld.apply_currents(I_E, I_I, None), I_E - I_I)   # q_I=1, g_K=0 -> exact


def test_apply_currents_uniform_qI_matches_scalar_regionalresource():
    """v2 reduces to v1 (§B5.2): a spatially-uniform q_I=0.5 field == RegionalResource with
    q_global=0.5, q_core=1.0 (global-only)."""
    posE, posI, _ = _synthetic_positions()
    nE, N = posE.shape[0], posE.shape[0] + posI.shape[0]
    fld = SpatialSlowField(N, 18.0, posE, posI, L=6.0,
                           cfg=SpatialSlowFieldConfig(q_init=0.5, k_q=0.0, k_K=0.0, use_gK=False))
    rr = RegionalResource(N, 18.0, np.zeros(N, bool),
                          RegionalResourceConfig(q_global_init=0.5, q_core_init=1.0), NE=nE)
    I_E = np.linspace(1.0, 3.0, N); I_I = np.linspace(0.2, 1.0, N)
    assert np.allclose(fld.apply_currents(I_E, I_I, None), rr.apply_currents(I_E, I_I, None))


def test_apply_currents_gK_subtracts_on_E_only():
    posE, posI, _ = _synthetic_positions(nE=50, nI=10)
    nE, N = 50, 60
    fld = SpatialSlowField(N, 18.0, posE, posI, L=6.0,
                           cfg=SpatialSlowFieldConfig(k_q=0.0, k_K=0.0, eta_K=1.0))
    fld.g_K[:] = 0.3                                            # uniform fatigue field
    out = fld.apply_currents(np.ones(N), np.zeros(N), None)     # I_I=0 isolates the g_K term
    assert np.allclose(out[:nE], 1.0 - 0.3)                     # E cells: -eta_K*g_K
    assert np.allclose(out[nE:], 1.0)                           # I cells untouched


# ===========================================================================
# Task 5 -- step dynamics (field-level parity, sign, SPATIAL locality, bounds, kernels)
# ===========================================================================
def test_step_off_holds_fields():
    posE, posI, labels = _synthetic_positions()
    N = labels.size
    fld = SpatialSlowField(N, 18.0, posE, posI, L=6.0,
                           cfg=SpatialSlowFieldConfig(k_q=0.0, k_K=0.0, q_init=1.0))
    spk = np.zeros(N, bool); spk[:50] = True
    for _ in range(100):
        fld.step(spk, labels, 0.1)
    assert np.all(fld.q_I == 1.0) and np.all(fld.g_K == 0.0)    # k_q=k_K=0 -> nothing moves


def test_qI_depletes_with_activity_then_refills():
    posE, posI, labels = _synthetic_positions()
    N = labels.size
    fld = SpatialSlowField(N, 18.0, posE, posI, L=6.0,
                           cfg=SpatialSlowFieldConfig(k_q=0.05, tau_q=2000.0, tau_a=1.0,
                                                      q_min=0.0, use_gK=False))
    spk = np.zeros(N, bool); spk[:200] = True                  # all E fire
    for _ in range(500):
        fld.step(spk, labels, 0.1)
    depleted = float(fld.q_I.mean())
    assert depleted < 0.99                                      # activity drains q_I
    for _ in range(3000):
        fld.step(np.zeros(N, bool), labels, 0.1)               # quiet -> refill toward 1
    assert fld.q_I.mean() > depleted


def test_aq_drive_weights_inhibition_at_least_excitation():
    """a_q drive = eta_E*r_E + eta_I*r_I with eta_I >= eta_E (inhibitory resource tracks inhibitory
    USE, §B5.2). Pins the exact weighted formula so step() cannot silently drop r_I."""
    r = np.ones((3, 3)); z = np.zeros((3, 3))
    assert np.allclose(aq_drive(r, z, eta_E=0.3, eta_I=1.0), 0.3)            # E-only -> eta_E
    assert np.allclose(aq_drive(z, r, eta_E=0.3, eta_I=1.0), 1.0)            # I-only -> eta_I
    assert np.all(aq_drive(z, r, 0.3, 1.0) > aq_drive(r, z, 0.3, 1.0))      # eta_I > eta_E
    assert np.allclose(aq_drive(r, r, 0.3, 1.0), 0.3 + 1.0)                  # exact weighted sum
    assert np.allclose(aq_drive(r, z, 0.5, 0.5), aq_drive(z, r, 0.5, 0.5))  # equal weights symmetric


def test_qI_depletes_from_inhibitory_activity():
    """q_I tracks inhibitory USE (the eta_I*r_I term wired into step, §B5.2): firing ONLY I neurons
    (r_E stays 0) still depletes q_I. An implementation that drops r_I and depletes on r_E alone
    leaves q_I==1 here and fails."""
    posE, posI, labels = _synthetic_positions()
    N = labels.size; nE = posE.shape[0]
    fld = SpatialSlowField(N, 18.0, posE, posI, L=6.0,
                           cfg=SpatialSlowFieldConfig(k_q=0.05, tau_q=1e9, tau_a=1.0,
                                                      q_min=0.0, use_gK=False))
    spk = np.zeros(N, bool); spk[nE:] = True                   # ONLY I neurons fire
    for _ in range(500):
        fld.step(spk, labels, 0.1)
    assert fld.q_I.mean() < 0.99                               # inhibitory use alone drains q_I


def test_qI_depletes_locally_not_globally():
    """The whole point of v2: spatial history. A LEFT burst depletes q_I on the left more
    than on the right (§B5.0)."""
    L = 6.0
    posE, posI, labels = _synthetic_positions(L=L)
    N = labels.size
    fld = SpatialSlowField(N, 18.0, posE, posI, L=L,
                           cfg=SpatialSlowFieldConfig(k_q=0.05, tau_q=1e9, tau_a=1.0,
                                                      q_min=0.0, sigma_q=0.8, use_gK=False))
    spk = np.zeros(N, bool); spk[:posE.shape[0]] = posE[:, 0] < L / 3.0   # only left E fire
    for _ in range(500):
        fld.step(spk, labels, 0.1)
    probe = np.array([[L / 6.0, L / 2.0], [5.0 * L / 6.0, L / 2.0]])      # left vs right
    q_left, q_right = sample_field_at(fld.q_I, probe, L, fld.q_I.shape[0])
    assert q_left < q_right and q_right > 0.95                  # local drain, far side intact


def test_qI_bounded_floor():
    posE, posI, labels = _synthetic_positions()
    N = labels.size
    fld = SpatialSlowField(N, 18.0, posE, posI, L=6.0,
                           cfg=SpatialSlowFieldConfig(k_q=50.0, q_min=0.25, tau_a=1.0, use_gK=False))
    spk = np.zeros(N, bool); spk[:200] = True
    for _ in range(2000):
        fld.step(spk, labels, 0.1)
    assert fld.q_I.min() >= 0.25 - 1e-12 and fld.q_I.max() <= 1.0 + 1e-12


def test_gK_builds_on_E_activity_and_decays_bounded():
    posE, posI, labels = _synthetic_positions()
    N = labels.size
    fld = SpatialSlowField(N, 18.0, posE, posI, L=6.0,
                           cfg=SpatialSlowFieldConfig(k_q=0.0, k_K=0.05, gK_max=1.0,
                                                      tau_K=2000.0, tau_a=1.0))
    spk = np.zeros(N, bool); spk[:200] = True
    for _ in range(500):
        fld.step(spk, labels, 0.1)
    built = float(fld.g_K.max())
    assert 0.0 < built <= 1.0
    for _ in range(3000):
        fld.step(np.zeros(N, bool), labels, 0.1)               # quiet -> decay
    assert fld.g_K.max() < built


def test_gK_zero_kK_does_not_build():
    """k_K is a STRENGTH knob, not just a switch: k_K=0 -> g_K never builds under strong E activity,
    while q_I (k_q>0) still evolves -> the two slow vars gate independently."""
    posE, posI, labels = _synthetic_positions()
    N = labels.size
    fld = SpatialSlowField(N, 18.0, posE, posI, L=6.0,
                           cfg=SpatialSlowFieldConfig(k_q=0.05, k_K=0.0, tau_q=1e9,
                                                      tau_a=1.0, q_min=0.0))
    spk = np.zeros(N, bool); spk[:200] = True
    for _ in range(500):
        fld.step(spk, labels, 0.1)
    assert np.all(fld.g_K == 0.0)                              # k_K=0 -> no fatigue ever
    assert fld.q_I.mean() < 1.0                                # but q_I (k_q>0) evolved -> independent


def test_gK_larger_kK_builds_more():
    """k_K scales the build rate AND the bounded steady state (§B5.3): same spikes/steps, larger k_K
    -> more g_K. Catches a formula that drops the k_K factor and only multiplies gK_max*f."""
    posE, posI, labels = _synthetic_positions()
    N = labels.size

    def built(kK):
        fld = SpatialSlowField(N, 18.0, posE, posI, L=6.0,
                               cfg=SpatialSlowFieldConfig(k_q=0.0, k_K=kK, gK_max=1.0,
                                                          tau_K=20.0, tau_a=1.0))
        spk = np.zeros(N, bool); spk[:200] = True
        for _ in range(500):
            fld.step(spk, labels, 0.1)
        return float(fld.g_K.max())
    assert built(0.10) > built(0.02) > 0.0                     # monotone in k_K (graded steady state)


def test_gK_bounded_ceiling():
    """g_K never exceeds gK_max even under aggressive build (large k_K), mirroring qI_bounded_floor."""
    posE, posI, labels = _synthetic_positions()
    N = labels.size
    fld = SpatialSlowField(N, 18.0, posE, posI, L=6.0,
                           cfg=SpatialSlowFieldConfig(k_q=0.0, k_K=50.0, gK_max=1.0, tau_a=1.0))
    spk = np.zeros(N, bool); spk[:200] = True
    for _ in range(2000):
        fld.step(spk, labels, 0.1)
    assert fld.g_K.max() <= 1.0 + 1e-12 and fld.g_K.min() >= 0.0 - 1e-12


def test_kernel_q_footprint_wider_than_kernel_K():
    """sigma_q > sigma_K behaviorally: identical localized burst, the q_I depletion footprint
    must be wider than the g_K buildup footprint (§B5.3). Footprint = full-width-half-max cell
    count (relative 0.5*peak on BOTH fields) -- an amplitude-INDEPENDENT spatial-width measure.
    (The firing-rate field is normalized so absolute depletion is small, ~0.007; an absolute
    q-threshold would measure amplitude, not width, and read 0.)"""
    L = 6.0
    posE, posI, labels = _synthetic_positions(L=L)
    N = labels.size
    fld = SpatialSlowField(N, 18.0, posE, posI, L=L,
                           cfg=SpatialSlowFieldConfig(k_q=0.05, k_K=0.05, tau_q=1e9, tau_K=1e9,
                                                      tau_a=1.0, q_min=0.0,
                                                      sigma_q=1.5, sigma_K=0.5))
    spk = np.zeros(N, bool)
    spk[:posE.shape[0]] = np.linalg.norm(posE - L / 2.0, axis=1) < 0.4    # central blob
    for _ in range(500):
        fld.step(spk, labels, 0.1)
    q_dep = 1.0 - fld.q_I
    q_footprint = (q_dep > 0.5 * q_dep.max()).sum()            # FWHM disinhibition footprint (~sigma_q)
    gk_footprint = (fld.g_K > 0.5 * fld.g_K.max()).sum()       # FWHM fatigue footprint (~sigma_K)
    assert q_footprint > gk_footprint


def test_step_no_nan_long_random():
    posE, posI, labels = _synthetic_positions()
    N = labels.size
    fld = SpatialSlowField(N, 18.0, posE, posI, L=6.0,
                           cfg=SpatialSlowFieldConfig(k_q=0.03, k_K=0.03))
    rng = np.random.default_rng(0)
    for _ in range(2000):
        fld.step(rng.random(N) < 0.1, labels, 0.1)
    assert np.all(np.isfinite(fld.q_I)) and np.all(np.isfinite(fld.g_K))
    assert fld.q_I.min() >= fld.cfg.q_min - 1e-12 and fld.g_K.max() <= fld.cfg.gK_max + 1e-12


@pytest.mark.slow
def test_offparity_byte_identical_to_slow_none():
    """k_q=0,k_K=0,q_init=1 -> q_I==1,g_K==0 -> the engine is byte-identical to slow=None
    (the BASELINE_SHA regression, cf tests/test_m3a_quasistatic_slowvars.py)."""
    p, net, NE, NI = _build()
    pos = net["pos"]; N = NE + NI
    vth = np.full(N, 18.0)
    res_none = _run(p, net, NE, NI, slow=None, V_th_per_neuron=vth)
    fld = SpatialSlowField(N, 18.0, pos[:NE], pos[NE:], p.L,
                           cfg=SpatialSlowFieldConfig(k_q=0.0, k_K=0.0, q_init=1.0))
    res_v2 = _run(p, net, NE, NI, slow=fld, V_th_per_neuron=vth)
    assert _sha(res_v2) == _sha(res_none)


# ===========================================================================
# Tasks 6-8 -- per-event source-space metrics
# ===========================================================================
def test_recruitment_area_fraction_above_threshold():
    A = np.zeros((10, 10)); A[:3, :] = 1.0                      # 30 of 100 cells hot
    assert abs(recruitment_area(A, theta_A=0.5) - 0.30) < 1e-12


def test_axis_score_source_space_onset_gradient():
    """Methodological lock: S_axis from the per-cell onset gradient (onset_axis)."""
    rng = np.random.default_rng(0)
    posE = rng.uniform(0.0, 6.0, size=(200, 2))
    u_axis = np.array([1.0, 0.0])
    onset_along = posE[:, 0] * 2.0 + 5.0                        # onset increases ALONG axis -> S~1
    onset_perp = posE[:, 1] * 2.0 + 5.0                        # onset increases PERP to axis -> S~0
    assert axis_score(posE, onset_along, u_axis) > 0.9
    assert axis_score(posE, onset_perp, u_axis) < 0.2


def test_axis_score_nan_when_too_few_onsets():
    posE = np.random.default_rng(1).uniform(0, 6, size=(200, 2))
    onset = np.full(200, np.nan); onset[:5] = 1.0              # only 5 finite onsets (< min_n=20)
    assert np.isnan(axis_score(posE, onset, np.array([1.0, 0.0])))


def test_offaxis_fraction_on_axis_vs_off_axis():
    L, n = 6.0, 24
    xs = (np.arange(n) + 0.5) * L / n
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    grid_xy = np.stack([gx, gy], axis=-1)
    center = np.array([L / 2, L / 2]); u_axis = np.array([1.0, 0.0])
    on_axis = (np.abs(gy - L / 2) < 0.3).astype(float)         # a stripe along the axis
    off_axis = (np.abs(gy - L / 2) > 1.5).astype(float)        # mass away from the axis
    assert offaxis_fraction(on_axis, grid_xy, center, u_axis, corridor_halfwidth=0.6) < 0.1
    assert offaxis_fraction(off_axis, grid_xy, center, u_axis, corridor_halfwidth=0.6) > 0.8


def test_participation_ratio_bounds():
    single = np.zeros((10, 10)); single[5, 5] = 1.0
    uniform = np.ones((10, 10))
    assert participation_ratio(single) < 0.02                  # ~1/100
    assert participation_ratio(uniform) > 0.99                 # ~1


def test_event_recovery_returned_vs_runaway():
    dt = 0.1
    t = np.arange(0, 400, dt)
    decaying = 2.0 + 30.0 * np.exp(-(t - 150.0) / 20.0) * (t >= 150)
    sustained = np.where(t >= 150, 40.0, 2.0)
    assert event_recovery(decaying, dt, t_post0=300.0, baseline=2.0, sigma_base=0.5) is True
    assert event_recovery(sustained, dt, t_post0=300.0, baseline=2.0, sigma_base=0.5) is False


# ===========================================================================
# Task 9 -- four-state classifier (the science-contract gate STRUCTURE)
# ===========================================================================
def _metrics(n_onsets=200, R_area=0.1, S_axis=0.8, F_offaxis=0.1, G_PR=0.1, recovery=True):
    return dict(n_onsets=n_onsets, R_area=R_area, S_axis=S_axis,
                F_offaxis=F_offaxis, G_PR=G_PR, recovery=recovery)


def test_classify_interictal_axial():
    assert classify_event(_metrics(R_area=0.02, S_axis=0.85)) == "interictal_axial"


def test_classify_expanded_axial():
    assert classify_event(_metrics(R_area=0.40, S_axis=0.80, F_offaxis=0.10)) == "expanded_axial"


def test_classify_ictal_like_candidate():
    assert classify_event(_metrics(R_area=0.50, S_axis=0.20, F_offaxis=0.55, G_PR=0.45)) \
        == "ictal_like_candidate"


def test_classify_runaway_overrides_when_not_recovered():
    assert classify_event(_metrics(R_area=0.6, S_axis=0.2, F_offaxis=0.6, recovery=False)) == "runaway"


def test_classify_large_axial_is_NOT_ictal_like():
    """KEY boundary (§B5.6): large + axis-dominant + low off-axis is expanded_axial, never
    ictal_like. Size alone must not trigger the seizure-candidate label."""
    assert classify_event(_metrics(R_area=0.7, S_axis=0.85, F_offaxis=0.05, G_PR=0.2)) \
        == "expanded_axial"


def test_classify_small_offaxis_is_NOT_ictal_like():
    """KEY size gate (§B5.6): a SMALL event, even with broken axis + high off-axis + recovered, is
    NOT ictal_like. R_area large is a NECESSARY condition -- a local off-axis blip must not read as
    seizure-like (the complement of the size boundary above)."""
    out = classify_event(_metrics(R_area=0.02, S_axis=0.15, F_offaxis=0.60, G_PR=0.50, recovery=True))
    assert out != "ictal_like_candidate"


def test_classify_insufficient_fails_closed():
    # too few onsets -> INSUFFICIENT even if everything else screams ictal-like
    assert classify_event(_metrics(n_onsets=5, S_axis=0.1, F_offaxis=0.9, G_PR=0.9)) == "INSUFFICIENT"
    # undefined axis -> INSUFFICIENT
    assert classify_event(_metrics(S_axis=float("nan"))) == "INSUFFICIENT"


# ===========================================================================
# Task 10 -- proxy phase plane
# ===========================================================================
def test_region_pressure_formula():
    q = np.full(4, 0.5); g = np.full(4, 0.2)
    expected = np.log(2.0) - np.mean(np.log(0.5 + 1e-9)) - 0.3 * 0.2
    assert abs(region_pressure(q, g, lgr=2.0, beta_K=0.3) - expected) < 1e-9


def test_proxy_phase_point_axis_dominant_has_positive_X():
    """Sign lock (§B5.7): q_I down = disinhibited = HIGHER region pressure. AXIS-DOMINANT means the
    axis is the MORE disinhibited channel (q_axis < q_off) -> X = P_axis - P_offaxis > 0. When
    off-axis catches up (equal q_I) X drops to 0 = axis-breaking."""
    n = 16
    axis_mask = np.zeros((n, n), bool); axis_mask[:, n // 2 - 1:n // 2 + 1] = True
    off_mask = ~axis_mask
    masks = {"axis": axis_mask, "offaxis": off_mask, "global": np.ones((n, n), bool)}

    class _F:                                                    # minimal field stand-in
        def __init__(self, q_axis, q_off):
            self.q_I = np.where(axis_mask, q_axis, q_off)
            self.g_K = np.zeros((n, n))
    # axis-dominant: axis MORE disinhibited (q_axis=0.4) than off-axis (q_off=0.9) -> P_axis > P_off -> X>0
    X_dom, _ = proxy_phase_point(_F(0.4, 0.9), masks, lgr=1.0, beta_K=0.3)
    # equalized: q_axis == q_off -> X == 0 (axis no longer dominant; off-axis has caught up)
    X_eq, _ = proxy_phase_point(_F(0.4, 0.4), masks, lgr=1.0, beta_K=0.3)
    assert X_dom > X_eq
    assert X_dom > 0.0 and abs(X_eq) < 1e-9


def test_proxy_phase_point_Y_is_global_pressure():
    """Y is the GLOBAL region pressure (whole sheet; matches the spectral Y=alpha_global for overlay),
    NOT P_offaxis. With the axis disinhibited (low q) but off-axis fully inhibited (q=1), P_offaxis==0
    while P_global averages in the disinhibited axis cells -> Y must be > 0. Pins Y=P_global and proves
    the 'global' mask is used (not a dead arg)."""
    n = 16
    axis_mask = np.zeros((n, n), bool); axis_mask[:, n // 2 - 1:n // 2 + 1] = True
    masks = {"axis": axis_mask, "offaxis": ~axis_mask, "global": np.ones((n, n), bool)}

    class _F:
        def __init__(self):
            self.q_I = np.where(axis_mask, 0.2, 1.0)   # axis disinhibited, off-axis fully inhibited
            self.g_K = np.zeros((n, n))
    _, Y = proxy_phase_point(_F(), masks, lgr=1.0, beta_K=0.3)
    expected_global = -np.mean(np.log(np.where(axis_mask, 0.2, 1.0) + 1e-9))   # lgr=1 -> log term only
    assert abs(Y - expected_global) < 1e-6                     # Y is P_global exactly
    assert Y > 1e-3                                            # > 0 (would be ~0 if Y were P_offaxis)
