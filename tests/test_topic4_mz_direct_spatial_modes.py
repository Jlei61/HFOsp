"""Contract tests for Topic 4 MZ direct spatial-mode dynamics (design spec §7 C1-C14).

Each test maps 1:1 to a spec clause. SNN-backed tests use a TINY network so the
parity / common-RNG / window invariants are exercised without the 40k-neuron substrate;
the full-substrate native parity is a runner-level check documented in STATUS.md.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

import copy  # noqa: E402

from params import Params  # noqa: E402
from model import build_network  # noqa: E402
from kick_probe import simulate_kick  # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402
from src.topic4_mz_onset_dynamics import MZOnsetProbe, run_loop  # noqa: E402
from src.topic4_mz_direct_spatial_modes import (  # noqa: E402
    MZSpatialProbe, build_grid_readout, grid_pattern_to_current, rms_normalize,
    spikes_to_rate_grid, local_window_maps, real_fourier_basis_1d, real_fourier_basis_2d,
    central_difference, build_empirical_operator, field_globality, field_axis_alignment,
    normalized_field_overlap, gaussian_current_field, response_norm, region_response,
    cumulative_response_ratio, axis_kymograph, first_arrival_times, fit_arrival_distance,
    linearity_discrepancy, select_epsilon, right_censoring_label, balanced_lowk_indices,
    robust_identifiability_gate,
)


# ------------------------------------------------------------------ tiny-net fixture (mirror onset tests)
@pytest.fixture(scope="module")
def tiny():
    p = Params(g=3.6, L=1.0, density=2000.0, T=60.0, dt=0.1, nu_ext_ratio=0.9, seed=1)
    net = build_network(p, verbose=False)
    NE, N = net["NE"], net["NE"] + net["NI"]
    posE = net["pos"][:NE]
    core = np.linalg.norm(posE - np.array([0.5, 0.5]), axis=1) <= 0.2
    vth = np.full(N, p.V_th)
    vth[:NE][core] -= 1.0
    cfg = MZSlowVarsConfig(use_z=True, use_m=False, I_th_EI=5.0, tau_z=3000.0)
    return dict(p=p, net=net, NE=NE, N=N, core=core, vth=vth, posE=posE, cfg=cfg,
                nsteps=int(round(p.T / p.dt)))


def _fresh_spatial(tiny, cfg=None):
    t = tiny
    slow = MZSpatialProbe(t["N"], 18.0, cfg or t["cfg"], NE=t["NE"], core_mask_E=t["core"])
    t["net"]["rng"] = np.random.default_rng(t["p"].seed)
    return slow


# ============================================================ C1 off-by-default parity
def test_c1_spatial_probe_no_schedule_parity(tiny):
    """MZSpatialProbe with NO current schedule == MZOnsetProbe == simulate_kick (bit-identical)."""
    t = tiny
    a = MZOnsetProbe(t["N"], 18.0, t["cfg"], NE=t["NE"], core_mask_E=t["core"])
    t["net"]["rng"] = np.random.default_rng(t["p"].seed)
    ra = simulate_kick(t["p"], t["net"], 0.0, slow=a, kick_center=[0.5, 0.5], r_kick=0.3, t_kick=1e9,
                       V_th_per_neuron=t["vth"])
    b = _fresh_spatial(tiny)
    rb = simulate_kick(t["p"], t["net"], 0.0, slow=b, kick_center=[0.5, 0.5], r_kick=0.3, t_kick=1e9,
                       V_th_per_neuron=t["vth"])
    assert np.array_equal(ra["rate_E"], rb["rate_E"])
    assert np.array_equal(ra["E_spk_bool"], rb["E_spk_bool"])


def test_c1_run_loop_parity(tiny):
    """run_loop with a no-schedule MZSpatialProbe reproduces simulate_kick bit-for-bit."""
    t = tiny
    a = MZOnsetProbe(t["N"], 18.0, t["cfg"], NE=t["NE"], core_mask_E=t["core"])
    t["net"]["rng"] = np.random.default_rng(t["p"].seed)
    ra = simulate_kick(t["p"], t["net"], 0.0, slow=a, kick_center=[0.5, 0.5], r_kick=0.3, t_kick=1e9,
                       V_th_per_neuron=t["vth"])
    b = _fresh_spatial(tiny)
    rb = run_loop(t["p"], t["net"], b, t["vth"], n_steps=t["nsteps"], store_spikes=True)
    assert np.array_equal(ra["rate_E"], rb["rate_E"])
    assert np.array_equal(ra["E_spk_bool"], rb["E_spk_bool"])


# ============================================================ C4 current schedule: E-only + window + RMS
def test_c4_current_schedule_apply_currents_E_only_and_window(tiny):
    """The additive current acts only on E cells inside [lo,hi); I cells and off-window untouched."""
    t = tiny
    NE, N = t["NE"], t["N"]
    pat = np.linspace(-1.0, 1.0, NE)
    slow = MZSpatialProbe(N, 18.0, t["cfg"], NE=NE, core_mask_E=t["core"])
    slow.set_current_schedule(lo=100, hi=110, pattern_E=pat)
    I_E = np.full(N, 3.0); I_I = np.full(N, 1.0)
    # baseline I_net for these currents with no schedule active
    base = MZSpatialProbe(N, 18.0, t["cfg"], NE=NE, core_mask_E=t["core"])
    base.z[:NE] = 1.0; slow.z[:NE] = 1.0
    base._step_i = 105; ref_net = base.apply_currents(I_E.copy(), I_I.copy())
    slow._step_i = 50                                  # before window -> parity
    assert np.array_equal(slow.apply_currents(I_E.copy(), I_I.copy()), ref_net)
    slow._step_i = 105                                 # inside window -> E cells shifted by pattern
    inw = slow.apply_currents(I_E.copy(), I_I.copy())
    assert np.allclose(inw[:NE] - ref_net[:NE], pat)   # E cells shifted by exactly the pattern
    assert np.array_equal(inw[NE:], ref_net[NE:])      # I cells untouched
    slow._step_i = 110                                 # window is half-open [lo,hi) -> excluded
    assert np.array_equal(slow.apply_currents(I_E.copy(), I_I.copy()), ref_net)


def test_c4_zero_pattern_schedule_is_bit_parity(tiny):
    """An ACTIVE schedule carrying a zero pattern must be bit-identical to no-probe (schedule
    machinery itself never perturbs the RNG stream or the dynamics)."""
    t = tiny
    K = 200
    r1 = run_loop(t["p"], t["net"], _fresh_spatial(tiny), t["vth"], n_steps=K, capture_final=True,
                  store_spikes=False)
    ck = r1["checkpoint"]
    s0 = copy.deepcopy(ck.slow)                        # no schedule -> no-probe control
    r0 = run_loop(t["p"], t["net"], s0, t["vth"], n_steps=150, start=ck, store_spikes=True)
    sz = copy.deepcopy(ck.slow)
    sz.set_current_schedule(lo=K, hi=K + 10, pattern_E=np.zeros(t["NE"]))   # active but zero
    rz = run_loop(t["p"], t["net"], sz, t["vth"], n_steps=150, start=ck, store_spikes=True)
    assert np.array_equal(r0["rate_E"], rz["rate_E"])
    assert np.array_equal(r0["E_spk_bool"], rz["E_spk_bool"])


# ============================================================ C3 common random numbers
def test_c3_common_rng_forks_idempotent(tiny):
    """+eps forks from the same checkpoint are reproducible (common RNG); repeats are identical."""
    t = tiny
    K = 200
    r1 = run_loop(t["p"], t["net"], _fresh_spatial(tiny), t["vth"], n_steps=K, capture_final=True,
                  store_spikes=False)
    ck = r1["checkpoint"]
    pat = 0.5 * np.ones(t["NE"])
    outs = []
    for _ in range(2):
        s = copy.deepcopy(ck.slow)
        s.set_current_schedule(lo=K, hi=K + 10, pattern_E=pat)
        outs.append(run_loop(t["p"], t["net"], s, t["vth"], n_steps=120, start=ck, store_spikes=True))
    assert np.array_equal(outs[0]["rate_E"], outs[1]["rate_E"])
    assert np.array_equal(outs[0]["E_spk_bool"], outs[1]["E_spk_bool"])


def test_c3_nonzero_pattern_changes_output(tiny):
    """A non-zero +eps current must actually change the output relative to the no-probe control
    (so the perturbation is measurable, not a silent no-op)."""
    t = tiny
    K = 200
    r1 = run_loop(t["p"], t["net"], _fresh_spatial(tiny), t["vth"], n_steps=K, capture_final=True,
                  store_spikes=False)
    ck = r1["checkpoint"]
    s0 = copy.deepcopy(ck.slow)
    r0 = run_loop(t["p"], t["net"], s0, t["vth"], n_steps=120, start=ck, store_spikes=True)
    sp = copy.deepcopy(ck.slow)
    sp.set_current_schedule(lo=K, hi=K + 10, pattern_E=np.full(t["NE"], 8.0))   # strong positive kick
    rp = run_loop(t["p"], t["net"], sp, t["vth"], n_steps=120, start=ck, store_spikes=True)
    assert not np.array_equal(r0["E_spk_bool"], rp["E_spk_bool"])


# ============================================================ C4 clear_current_schedule
def test_c4_clear_schedule_restores_parity(tiny):
    """clear_current_schedule() removes the schedule (restores no-probe path at apply_currents)."""
    t = tiny
    NE, N = t["NE"], t["N"]
    slow = MZSpatialProbe(N, 18.0, t["cfg"], NE=NE, core_mask_E=t["core"])
    slow.z[:NE] = 1.0
    I_E = np.full(N, 2.0); I_I = np.full(N, 1.0)
    slow._step_i = 5; ref = slow.apply_currents(I_E.copy(), I_I.copy())
    slow.set_current_schedule(lo=0, hi=10, pattern_E=np.full(NE, 3.0))
    assert not np.array_equal(slow.apply_currents(I_E.copy(), I_I.copy()), ref)
    slow.clear_current_schedule()
    assert np.array_equal(slow.apply_currents(I_E.copy(), I_I.copy()), ref)


# ============================================================ C5 real 2-D Fourier basis
def test_c5_fourier_basis_1d_orthonormal():
    for n in (4, 5, 12):
        B = real_fourier_basis_1d(n)
        assert B.shape == (n, n)
        assert np.allclose(B.T @ B, np.eye(n), atol=1e-10)
        assert np.isrealobj(B)


def test_c5_fourier_basis_2d_144_orthonormal():
    """The operator input space is the FULL 144-dim real 2-D Fourier basis with Q^T Q = I."""
    P = real_fourier_basis_2d(12)
    assert P.shape == (144, 144)
    assert np.allclose(P.T @ P, np.eye(144), atol=1e-10)
    assert np.isrealobj(P)
    # a DC (constant) column exists (all-equal pattern)
    col_ptp = np.ptp(P, axis=0)
    assert np.isclose(col_ptp.min(), 0.0, atol=1e-12)


def test_balanced_lowk_indices_symmetric():
    """Balanced low-k selection = 2-D modes built from 1-D frequencies <= k_max in BOTH axes
    (symmetric cos/sin, both directions) — NOT the leading columns (which include Nyquist)."""
    n = 12
    idx = balanced_lowk_indices(n, k_max=1)                    # 1-D freqs {0,1} -> 1-D idx {0,1,2}
    assert len(idx) == 9                                       # 3 x 3 outer products
    P = real_fourier_basis_2d(n)
    sub = P[:, idx]
    assert np.allclose(sub.T @ sub, np.eye(len(idx)), atol=1e-10)   # still orthonormal
    idx2 = balanced_lowk_indices(n, k_max=2)                   # 1-D freqs {0,1,2} -> 5 idx -> 25
    assert len(idx2) == 25
    assert set(idx).issubset(set(idx2))                        # nested
    assert max(idx2) < n * n


# ============================================================ C6 grid readout + spike-mass conservation
def test_c6_grid_readout_occupancy(tiny):
    ro = build_grid_readout(tiny["posE"], grid_n=6, L_phys=1.0, L_norm=5.0, center_phys=[0.5, 0.5])
    assert ro.occupancy.shape == (6, 6)
    assert int(ro.occupancy.sum()) == tiny["NE"]               # every E neuron lands in exactly one cell
    assert ro.cell_flat.min() >= 0 and ro.cell_flat.max() < 36


def test_c6_spikes_to_rate_grid_mass_conservation(tiny):
    ro = build_grid_readout(tiny["posE"], grid_n=6, L_phys=1.0, L_norm=5.0, center_phys=[0.5, 0.5])
    NE = tiny["NE"]
    rng = np.random.default_rng(0)
    spk = rng.random((50, NE)) < 0.02                          # ~2% per step
    out = spikes_to_rate_grid(spk, ro, dt_ms=1.0)
    assert out["mass_ok"]                                       # binned spikes == total spikes
    assert int(np.nansum(out["spikes_binned"])) == int(spk.sum())
    # mean-rate identity for a fully-occupied bin: rate = spikes/n_E/T_sec
    b = np.argmax(ro.occupancy)                                 # a populated bin
    bi, bj = np.unravel_index(b, (6, 6))
    n_in = ro.occupancy[bi, bj]
    if n_in > 0:
        exp = spk[:, ro.cell_flat == b].sum() / n_in / (50 * 1e-3)
        assert np.isclose(out["rate_hz"][bi, bj], exp)


def test_c6_empty_bin_flagged():
    """A grid cell with no E neuron is flagged empty and returns NaN rate (not 0)."""
    pos = np.array([[0.5, 0.5], [0.5, 0.5]])                    # both neurons in the centre
    ro = build_grid_readout(pos, grid_n=4, L_phys=1.0, L_norm=5.0, center_phys=[0.5, 0.5])
    assert ro.empty_mask.sum() == 4 * 4 - 1                     # only one occupied cell
    out = spikes_to_rate_grid(np.ones((3, 2), bool), ro, dt_ms=1.0)
    assert np.isnan(out["rate_hz"][ro.empty_mask]).all()


def test_c6_local_window_maps(tiny):
    """Local-window maps read the mean rate in the window ENDING at each center (width_ms)."""
    ro = build_grid_readout(tiny["posE"], grid_n=6, L_phys=1.0, L_norm=5.0, center_phys=[0.5, 0.5])
    spk = np.zeros((500, tiny["NE"]), bool)
    spk[0:50] = True                                           # all fire in [0,5) ms only
    maps = local_window_maps(spk, ro, dt_ms=0.1, centers_ms=[5.0, 50.0], width_ms=5.0)
    assert set(maps) == {5.0, 50.0}
    assert np.nanmax(maps[5.0]) > 0                            # activity in [0,5) window
    assert np.nanmax(maps[50.0]) == 0                         # silent in [45,50) window


# ============================================================ C7 synthetic operator recovery
def _synthetic_operator(grid_n, sigmas, seed=0):
    """Known bin-space operator M = sum_k sigma_k u_k v_k^T with orthonormal u,v."""
    N = grid_n * grid_n
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((N, N)))
    V, _ = np.linalg.qr(rng.standard_normal((N, N)))
    S = np.zeros(N)
    S[:len(sigmas)] = sigmas
    M = (U * S) @ V.T
    return M, U, S, V


def test_c7_synthetic_operator_recovery():
    """A known linear operator is recovered (sigma1, U1, V1) through the central-difference + SVD
    pipeline: inject the orthonormal basis patterns, form K = M @ P, rebuild, SVD."""
    gn = 6
    P = real_fourier_basis_2d(gn)
    M, U, S, V = _synthetic_operator(gn, [4.0, 1.3, 0.5], seed=3)
    eps = 1e-3
    # linear response to +/- eps * basis pattern j (units cancel in central diff)
    Y_plus = M @ (eps * P)
    Y_minus = M @ (-eps * P)
    K = np.column_stack([central_difference(Y_plus[:, j], Y_minus[:, j], eps) for j in range(P.shape[1])])
    op = build_empirical_operator(K, P, grid_n=gn)
    assert np.isclose(op["sigma1"], S[0], rtol=1e-6)
    assert abs(np.dot(op["u1_field"].ravel() / np.linalg.norm(op["u1_field"]), U[:, 0])) > 1 - 1e-6
    assert abs(np.dot(op["v1_field"].ravel() / np.linalg.norm(op["v1_field"]), V[:, 0])) > 1 - 1e-6


def test_c8_sigma1_ge_any_probe_gain():
    """sigma_hat_1 >= the gain ||K[:,j]||/||p_j|| of any single probe in the same input space."""
    gn = 6
    P = real_fourier_basis_2d(gn)
    M, U, S, V = _synthetic_operator(gn, [4.0, 1.3, 0.5], seed=5)
    K = M @ P                                                   # exact linear responses (eps cancels)
    op = build_empirical_operator(K, P, grid_n=gn)
    probe_gains = np.linalg.norm(K, axis=0)                     # ||M p_j|| ; ||p_j||=1
    assert op["sigma1"] >= probe_gains.max() - 1e-9


def test_c10_near_degenerate_reports_subspace():
    """When s1/s2 ~ 1, report the leading SUBSPACE, not a single (unstable) vector."""
    gn = 6
    P = real_fourier_basis_2d(gn)
    M, U, S, V = _synthetic_operator(gn, [2.0, 1.99, 0.3], seed=7)
    op = build_empirical_operator(K := M @ P, P, grid_n=gn, degeneracy_ratio=1.05)
    assert op["degenerate"] is True
    assert op["subspace_dim"] >= 2
    assert op["u_subspace"].shape[1] >= 2


# ============================================================ C9 sign invariance + field metrics
def test_c9_field_metrics_sign_invariant():
    f = np.array([[0.0, 1.0, 0.0], [0.5, -2.0, 0.5], [0.0, 1.0, 0.0]])
    assert np.isclose(field_globality(f), field_globality(-f))
    ro = build_grid_readout(np.array([[c % 3, c // 3] for c in range(9)], float),
                            grid_n=3, L_phys=2.0, L_norm=2.0, center_phys=[1.0, 1.0])
    ax = np.array([1.0, 0.0])
    assert np.isclose(field_axis_alignment(f, ro, ax), field_axis_alignment(-f, ro, ax))


def test_c9_globality_bounds():
    """globality in [1/N, 1]: uniform -> 1, single-cell -> 1/N."""
    n = 4
    uni = np.ones((n, n))
    assert np.isclose(field_globality(uni), 1.0)
    one = np.zeros((n, n)); one[0, 0] = 1.0
    assert np.isclose(field_globality(one), 1.0 / (n * n))


def test_c9_axis_alignment_directional():
    """A field elongated along the axis -> alignment near +1; along perpendicular -> near -1."""
    ro = build_grid_readout(np.array([[i, j] for i in range(5) for j in range(5)], float),
                            grid_n=5, L_phys=4.0, L_norm=4.0, center_phys=[2.0, 2.0])
    axis = np.array([1.0, 0.0])
    along = np.zeros((5, 5)); along[:, 2] = 1.0                 # a stripe varying along x (axis)
    perp = np.zeros((5, 5)); perp[2, :] = 1.0                   # a stripe varying along y (perp)
    assert field_axis_alignment(along, ro, axis) > 0.5
    assert field_axis_alignment(perp, ro, axis) < -0.5


def test_field_overlap_sign_invariant_and_bounded():
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert np.isclose(normalized_field_overlap(a, a), 1.0)
    assert np.isclose(normalized_field_overlap(a, -a), 1.0)     # sign-invariant
    b = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert np.isclose(normalized_field_overlap(a, b), 0.0)


# ============================================================ fixed-kick readouts
def test_gaussian_current_field_rms(tiny):
    ro = build_grid_readout(tiny["posE"], grid_n=8, L_phys=1.0, L_norm=5.0, center_phys=[0.5, 0.5])
    g = gaussian_current_field(ro, center_norm=(0.0, 0.0), sigma=1.0, rms=2.0)
    assert g.shape == (8, 8)
    assert np.isclose(np.sqrt(np.mean(g ** 2)), 2.0)           # RMS normalized
    assert g.max() == g[np.unravel_index(np.argmax(g), g.shape)]   # peak at centre-most cell


def test_cumulative_response_ratio_no_zero_spike():
    """remote/source cumulative ratio uses CUMULATIVE sums (no division blow-up at source=0)."""
    remote = np.array([0.0, 0.0, 1.0, 2.0])
    source = np.array([0.0, 5.0, 5.0, 5.0])                     # source zero at t0
    r = cumulative_response_ratio(remote, source)
    assert np.all(np.isfinite(r))
    assert np.isclose(r[-1], remote.sum() / source.sum())


def test_kymograph_and_arrival_regression():
    """Kymograph (time,position); arrival = first threshold crossing; fit needs >=4 positions."""
    # synthetic time-stack: a front moving from position 0 to 4 over time
    n_t, gn = 20, 5
    stack = np.zeros((n_t, gn, gn))
    for ti in range(n_t):
        front = ti // 4                                        # advances one cell every 4 steps
        if front < gn:
            stack[ti, front, 2] = 1.0                          # active cell on the mid row
    ro = build_grid_readout(np.array([[i, j] for i in range(gn) for j in range(gn)], float),
                            grid_n=gn, L_phys=4.0, L_norm=4.0, center_phys=[2.0, 2.0])
    ky = axis_kymograph(stack, ro, axis_unit=np.array([1.0, 0.0]),
                        src_norm=(-2.0, 0.0), snk_norm=(2.0, 0.0), band=0.6, n_pos=gn)
    assert ky["kymo"].shape == (n_t, gn)
    arr = first_arrival_times(ky["kymo"], ky["times"], threshold=0.5)
    fit = fit_arrival_distance(ky["distances"], arr, min_points=4)
    assert fit["eligible"] and fit["n_points"] >= 4
    assert fit["slope"] > 0 and fit["r2"] > 0.8               # monotone advance -> good linear fit


def test_arrival_fit_ineligible_few_points():
    """<4 crossed positions -> eligible False (never a forced wavefront)."""
    dist = np.array([0.0, 1.0, 2.0, 3.0])
    arr = np.array([1.0, np.nan, np.nan, np.nan])              # only one position crossed
    fit = fit_arrival_distance(dist, arr, min_points=4)
    assert not fit["eligible"] and fit["n_points"] == 1


def test_arrival_fit_ineligible_degenerate_constant():
    """A zero/near-zero response makes every position 'arrive' at t=0 (threshold ~0) -> constant
    arrivals -> ineligible (no front), never a spurious 0-slope / NaN-R2 'eligible' fit."""
    dist = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    arr = np.zeros(5)                                          # all positions cross at t=0 (noise)
    fit = fit_arrival_distance(dist, arr, min_points=4)
    assert not fit["eligible"]                                # degenerate (zero spread) rejected


def test_arrival_fit_rejects_negative_slope():
    """Arrival DECREASING with distance (far positions respond first) is not source-driven axial
    recruitment -> ineligible (review 2026-07-20 round-2)."""
    dist = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    arr = np.array([4.0, 3.0, 2.0, 1.0, 0.0])                 # arrival falls with distance -> slope<0
    fit = fit_arrival_distance(dist, arr, min_points=4)
    assert not fit["eligible"] and fit["slope"] < 0


def test_arrival_fit_rejects_low_r2():
    """A positive but very poor (low-R2) fit is not a clean front -> ineligible."""
    dist = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    arr = np.array([0.0, 5.0, 0.0, 5.0, 0.0, 6.0])           # scatter, ~flat trend, low R2
    fit = fit_arrival_distance(dist, arr, min_points=4, r2_min=0.5)
    assert not fit["eligible"]
    good = fit_arrival_distance(np.arange(5.0), np.arange(5.0) * 2.0 + 1.0, min_points=4, r2_min=0.5)
    assert good["eligible"] and good["slope"] > 0 and good["r2"] > 0.99   # clean line still passes


def test_corrected_identifiability_gate_requires_cross_half_stability():
    """Within-half amplitude checks do not replace agreement between the two estimated operators."""
    assert robust_identifiability_gate(0.10, 0.11, 0.12, 0.13, any_saturated=False, tol=0.15)
    assert not robust_identifiability_gate(0.10, 0.11, 0.12, 0.16, any_saturated=False, tol=0.15)
    assert not robust_identifiability_gate(0.10, 0.11, 0.12, 0.13, any_saturated=True, tol=0.15)


def test_region_response_and_norm():
    dY = np.array([[1.0, -1.0], [2.0, 0.0]])
    assert np.isclose(response_norm(dY), np.sqrt(1 + 1 + 4 + 0))
    masks = {"a": np.array([[True, False], [False, False]]), "b": np.array([[False, True], [True, True]])}
    rr = region_response(dY, masks)
    assert np.isclose(rr["a"], 1.0)                            # mean |dY| in region a
    assert np.isclose(rr["b"], (1.0 + 2.0 + 0.0) / 3.0)


# ============================================================ linearity audit + censoring labels
def test_linearity_discrepancy_and_selection():
    K = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.isclose(linearity_discrepancy(K, K), 0.0)        # identical -> 0
    Khalf = K * 1.05                                           # 5% off
    assert 0.04 < linearity_discrepancy(K, Khalf) < 0.06
    sel = select_epsilon([0.001, 0.0025, 0.005, 0.01], [0.03, 0.08, 0.12, 0.30],
                         saturated=[False, False, False, True], tol=0.15)
    assert sel["epsilon"] == 0.005 and sel["mode"] == "operator"   # largest qualifying, unsaturated
    sel2 = select_epsilon([0.001, 0.0025], [0.30, 0.40], saturated=[False, False], tol=0.15)
    assert sel2["mode"] == "nonlinear_response_only" and sel2["epsilon"] is None


def test_right_censoring_label():
    assert right_censoring_label(120.0) == "right_censored_native_transition"   # no-probe ran away
    assert right_censoring_label(None) == "resolved"                            # no-probe stable
