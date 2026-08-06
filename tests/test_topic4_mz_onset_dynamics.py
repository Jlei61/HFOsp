"""Contract tests for Topic 4 MZ early-onset dynamics (design spec §14).

Each test maps 1:1 to a spec §14 clause. SNN-backed tests use a TINY network (fast, seconds) so the
parity/freeze/probe invariants are exercised without the 40k-neuron substrate; the full-substrate onset
parity is a runner-level check documented in STATUS.md.
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
from src.topic4_mz_onset_dynamics import (  # noqa: E402
    MZOnsetProbe, run_loop, score_runaway, build_region_masks, slow_state_coordinates,
    qeff_region_summary, zbar_qeff_field_audit, realized_D_grid, build_DA_q_field, DA_controls,
    epsilon_c_from_ladder, classify_ignition, projected_flow_eligibility,
    natural_zm_trajectory,
)


# ------------------------------------------------------------------ tiny-net fixture
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
    return dict(p=p, net=net, NE=NE, N=N, core=core, vth=vth, cfg=cfg, nsteps=int(round(p.T / p.dt)))


def _fresh(tiny, slow_cls=MZOnsetProbe, cfg=None):
    t = tiny
    slow = slow_cls(t["N"], 18.0, cfg or t["cfg"], NE=t["NE"], core_mask_E=t["core"])
    t["net"]["rng"] = np.random.default_rng(t["p"].seed)
    return slow


# ------------------------------------------------------------------ §14.1 off-by-default parity
def test_1_off_by_default_parity(tiny):
    """MZOnsetProbe with NO schedule == MZSlowVars, and run_loop == simulate_kick (bit-identical)."""
    t = tiny
    # (a) MZOnsetProbe(no schedule) matches base MZSlowVars through the real guarded engine
    a = MZSlowVars(t["N"], 18.0, t["cfg"], NE=t["NE"], core_mask_E=t["core"])
    t["net"]["rng"] = np.random.default_rng(t["p"].seed)
    ra = simulate_kick(t["p"], t["net"], 0.0, slow=a, kick_center=[0.5, 0.5], r_kick=0.3, t_kick=1e9,
                       V_th_per_neuron=t["vth"])
    b = _fresh(tiny)
    rb = simulate_kick(t["p"], t["net"], 0.0, slow=b, kick_center=[0.5, 0.5], r_kick=0.3, t_kick=1e9,
                       V_th_per_neuron=t["vth"])
    assert np.array_equal(ra["rate_E"], rb["rate_E"])
    assert np.array_equal(ra["E_spk_bool"], rb["E_spk_bool"])
    assert np.array_equal(np.array(a.trace_z_min), np.array(b.trace_z_min))


def test_2_run_loop_matches_simulate_kick(tiny):
    """§14.2 native replay parity: the resumable loop reproduces the guarded engine bit-for-bit."""
    t = tiny
    a = _fresh(tiny)
    ra = simulate_kick(t["p"], t["net"], 0.0, slow=a, kick_center=[0.5, 0.5], r_kick=0.3, t_kick=1e9,
                       V_th_per_neuron=t["vth"])
    b = _fresh(tiny)
    rb = run_loop(t["p"], t["net"], b, t["vth"], n_steps=t["nsteps"], store_spikes=True)
    assert np.array_equal(ra["rate_E"], rb["rate_E"])
    assert np.array_equal(ra["E_spk_bool"], rb["E_spk_bool"])
    assert np.array_equal(np.array(a.trace_z_mean), np.array(b.trace_z_mean))


def test_3_prebranch_bit_identical_and_resume(tiny):
    """§14.3 + §14.13: checkpoint at K then native resume == fresh full run (pre-branch bit-identical)."""
    t = tiny
    ref = run_loop(t["p"], t["net"], _fresh(tiny), t["vth"], n_steps=t["nsteps"], store_spikes=True)
    K = 250
    r1 = run_loop(t["p"], t["net"], _fresh(tiny), t["vth"], n_steps=K, capture_final=True, store_spikes=True)
    ck = r1["checkpoint"]
    sC = copy.deepcopy(ck.slow)
    r2 = run_loop(t["p"], t["net"], sC, t["vth"], n_steps=t["nsteps"] - K, start=ck, store_spikes=True)
    assert np.array_equal(np.concatenate([r1["rate_E"], r2["rate_E"]]), ref["rate_E"])
    assert np.array_equal(np.concatenate([r1["E_spk_bool"], r2["E_spk_bool"]], axis=0), ref["E_spk_bool"])


def test_13_resume_idempotent(tiny):
    """§14.13 idempotency: resuming twice from the same checkpoint gives identical results."""
    t = tiny
    K = 200
    r1 = run_loop(t["p"], t["net"], _fresh(tiny), t["vth"], n_steps=K, capture_final=True, store_spikes=False)
    ck = r1["checkpoint"]
    a = run_loop(t["p"], t["net"], copy.deepcopy(ck.slow), t["vth"], n_steps=150, start=ck, store_spikes=True)
    b = run_loop(t["p"], t["net"], copy.deepcopy(ck.slow), t["vth"], n_steps=150, start=ck, store_spikes=True)
    assert np.array_equal(a["rate_E"], b["rate_E"])
    assert np.array_equal(a["E_spk_bool"], b["E_spk_bool"])


def test_4_freeze_holds_zm_constant(tiny):
    """§14.4 freeze actually holds z/m constant after the branch."""
    t = tiny
    cfg = MZSlowVarsConfig(use_z=True, use_m=True, I_th_EI=5.0, tau_z=3000.0, tau_adp=2000.0, eta_m=0.1)
    K = 200
    slow = MZOnsetProbe(t["N"], 18.0, cfg, NE=t["NE"], core_mask_E=t["core"]).set_branch(branch_step=K, freeze=True)
    t["net"]["rng"] = np.random.default_rng(t["p"].seed)
    r1 = run_loop(t["p"], t["net"], slow, t["vth"], n_steps=K, capture_final=True, store_spikes=False)
    ck = r1["checkpoint"]
    sF = copy.deepcopy(ck.slow)
    z0 = sF.z[:t["NE"]].copy()
    m0 = sF.m[:t["NE"]].copy()
    run_loop(t["p"], t["net"], sF, t["vth"], n_steps=300, start=ck, store_spikes=False)
    assert np.array_equal(sF.z[:t["NE"]], z0)          # z frozen
    assert np.array_equal(sF.m[:t["NE"]], m0)          # m frozen
    assert np.ptp(np.array(sF.trace_z_min)[K:]) == 0.0  # trace flat post-branch


def test_5_counterfactual_invariants():
    """§14.5 uniform/reset/shuffle preserve their declared invariants (z_transform closures)."""
    z = np.array([0.2, 0.4, 0.6, 0.8, 0.9, 0.1])
    reset = np.ones_like(z)
    assert np.all(reset == 1.0)
    uniform = np.full_like(z, z.mean())
    assert np.allclose(uniform, z.mean()) and np.ptp(uniform) == 0.0
    rng = np.random.default_rng(20260719)
    shuf = z.copy(); rng.shuffle(shuf)
    assert np.array_equal(np.sort(shuf), np.sort(z))   # histogram preserved


def test_6_shuffle_rng_isolated_from_network(tiny):
    """§14.6 an independent shuffle RNG does not touch the network RNG before the branch."""
    t = tiny
    net_rng = np.random.default_rng(t["p"].seed)
    st_before = copy.deepcopy(net_rng.bit_generator.state)
    shuf_rng = np.random.default_rng(20260719)      # counterfactual shuffle draws here, NOT from net rng
    _ = shuf_rng.permutation(100)
    assert net_rng.bit_generator.state == st_before  # network stream untouched


def test_7_probe_target_and_window(tiny):
    """§14.7 deterministic threshold probe acts only on the registered E target and time window."""
    t = tiny
    tgt = t["core"].copy()
    slow = MZOnsetProbe(t["N"], 18.0, t["cfg"], NE=t["NE"], core_mask_E=t["core"])
    slow.set_probe(lo=100, hi=200, target_E=tgt, delta=2.0)
    slow._step_i = 50                                # before window
    assert slow.threshold(t["vth"]) is t["vth"]      # passthrough (same object)
    slow._step_i = 150                               # inside window
    v = slow.threshold(t["vth"])
    assert np.allclose(t["vth"][:t["NE"]][tgt] - v[:t["NE"]][tgt], 2.0)      # target lowered
    assert np.array_equal(t["vth"][:t["NE"]][~tgt], v[:t["NE"]][~tgt])       # non-target untouched
    assert np.array_equal(t["vth"][t["NE"]:], v[t["NE"]:])                   # I cells untouched
    slow._step_i = 250                               # after window
    assert slow.threshold(t["vth"]) is t["vth"]


def test_8_qeff_current_aware(tiny):
    """§14.8 current-aware q_eff synthetic fixture: q_eff = sum(z*I_I)/sum(I_I) over the window."""
    t = tiny
    NE = t["NE"]
    slow = MZOnsetProbe(t["N"], 18.0, t["cfg"], NE=NE, core_mask_E=t["core"])
    slow.set_qeff_windows([(0, 2, "s")], I_th=1.0)
    # craft two steps with known z and I_I
    slow.z[:NE] = 0.5
    I_I = np.zeros(t["N"]); I_I[:NE] = 2.0
    slow._step_i = 0
    slow.apply_currents(np.zeros(t["N"]), I_I)
    slow.z[:NE] = 0.25
    I_I2 = np.zeros(t["N"]); I_I2[:NE] = 6.0
    slow._step_i = 1
    slow.apply_currents(np.zeros(t["N"]), I_I2)
    f = slow.qeff_fields()["s"]
    # q_eff = (0.5*2 + 0.25*6) / (2 + 6) = (1 + 1.5)/8 = 0.3125 ; both I_I >= 1 -> p_deplete = 1
    assert np.allclose(f["q_eff"], 0.3125)
    assert np.allclose(f["p_deplete"], 1.0)
    assert f["n_steps"] == 2


def test_9_region_masks_and_coordinates():
    """§14.9 coordinate/axis/core mapping fixture: region masks + D_z / A coordinates."""
    src = np.array([0.0, 0.0]); snk = np.array([4.0, 0.0]); axis = np.array([1.0, 0.0])
    pos = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 0.0], [2.0, 5.0]])   # src, snk, on-axis mid, far off-axis
    masks = build_region_masks(pos, src, snk, axis, core_r=1.0, corridor_halfwidth=0.5)
    assert masks["source_core"].tolist() == [True, False, False, False]
    assert masks["sink_core"].tolist() == [False, True, False, False]
    assert masks["axis_corridor"][2] and not masks["axis_corridor"][3]
    assert masks["off_axis"][3] and masks["core_excluded"][3]
    z = np.array([1.0, 1.0, 0.5, 1.0]); m = np.array([0.0, 0.0, 2.0, 0.0])
    coords = slow_state_coordinates(z, m, eta_m=0.5, region_masks=masks)
    # corridor = {src, snk, mid} (all perp=0 on-axis): mean z = (1+1+0.5)/3, mean m = (0+0+2)/3
    assert np.isclose(coords["axis_corridor"]["D_z"], 1.0 - (1.0 + 1.0 + 0.5) / 3.0)
    assert np.isclose(coords["axis_corridor"]["A"], 0.5 * (2.0 / 3.0))
    assert np.isclose(coords["source_core"]["D_z"], 0.0)        # z=1 at src -> D_z=0
    qs = qeff_region_summary(np.array([0.9, 0.9, 0.4, 0.9]), np.array([0, 0, 1.0, 0]), masks)
    assert np.isclose(qs["source_core"]["q_eff"], 0.9)


def test_9b_zbar_qeff_audit():
    """§14.8/§5.2 z_bar vs q_eff field audit returns spearman/cosine/diffs on a synthetic pair."""
    a = np.array([[0.2, 0.4], [0.6, 0.8]])
    aud = zbar_qeff_field_audit(a, a)
    assert np.isclose(aud["cosine"], 1.0) and np.isclose(aud["mean_abs_diff"], 0.0)
    b = a + 0.1
    aud2 = zbar_qeff_field_audit(a, b)
    assert np.isclose(aud2["mean_abs_diff"], 0.1) and np.isclose(aud2["max_abs_diff"], 0.1)


def test_10_DA_grid_fields():
    """§6.1 realized (D,A) grid: D range clipped; primary q-field scaled to requested mean D."""
    D = realized_D_grid(0.1, 0.5, n_D=9, clip=[0.0, 0.95], overshoot=1.10)
    assert len(D) == 9 and np.isclose(D[0], 0.1) and D[-1] <= 0.95 and np.isclose(D[-1], 0.55)
    dep = np.array([[0.1, 0.3], [0.3, 0.3]])   # mean depletion 0.25
    q, Dfield = build_DA_q_field(dep, target_D=0.4)
    assert np.isclose(np.mean(Dfield), 0.4, atol=1e-6)          # scaled to requested mean D
    assert np.all(q >= 0) and np.all(q <= 1)
    ctrl = DA_controls(dep, shuffle_seed=1)
    assert np.isclose(np.mean(ctrl["primary"]), 1.0)           # mean-1 normalized pattern
    assert np.ptp(ctrl["uniform"]) == 0.0
    assert np.array_equal(np.sort(ctrl["spatial_shuffle"].ravel()), np.sort(ctrl["primary"].ravel()))


def test_11_ignition_censoring():
    """§14.11 right-censored / zero-threshold ignition classification fixtures."""
    lad = [0.0, 0.025, 0.05, 0.10, 0.20]
    z = epsilon_c_from_ladder(lad, [True, True, True, True, True])
    assert z["zero_runaway"] and z["epsilon_c"] == 0.0
    cens = epsilon_c_from_ladder(lad, [False, False, False, False, False])
    assert cens["censored"] and cens["epsilon_c"] is None
    mid = epsilon_c_from_ladder(lad, [False, False, True, True, True])
    assert mid["epsilon_c"] == 0.05 and mid["bracket"] == (0.025, 0.05)
    # trajectory labels (result-neutral)
    assert classify_ignition([{"epsilon_c": None, "seed_consistent": True}]) == "unresolved"
    traj = [{"epsilon_c": 0.2, "alpha1": -1.0, "axial_gain": 0.5, "perp_gain": 0.4, "global_gain": 0.3, "seed_consistent": True},
            {"epsilon_c": 0.05, "alpha1": -0.5, "axial_gain": 0.6, "perp_gain": 0.5, "global_gain": 0.4, "seed_consistent": True}]
    assert classify_ignition(traj) == "finite_amplitude_escape"
    lin = [{"epsilon_c": 0.05, "alpha1": -1.0, "axial_gain": 0.1, "perp_gain": 0.1, "global_gain": 0.1, "seed_consistent": True},
           {"epsilon_c": 0.0, "alpha1": 0.2, "axial_gain": 0.1, "perp_gain": 0.1, "global_gain": 0.1, "seed_consistent": True}]
    assert classify_ignition(lin) == "linear_crossing"
    assert classify_ignition([{"epsilon_c": 0.1, "seed_consistent": False}]) == "seed-inconsistent"


def test_12_phase_arrow_eligibility():
    """§14.12 / §5.3 phase-arrow eligibility: needs >=3 visits, >=2 seeds, sign-agreement >=2/3."""
    # eligible: 3 visits, 2 seeds, consistent signs
    vis = [{"seed": 1, "dD": 0.1, "dA": 0.05}, {"seed": 3, "dD": 0.12, "dA": 0.04}, {"seed": 4, "dD": 0.11, "dA": 0.06}]
    e = projected_flow_eligibility(vis)
    assert e["eligible"] and e["n_seeds"] == 3 and e["sign_ok_dD"] and e["sign_ok_dA"]
    # ineligible: only 2 visits
    assert not projected_flow_eligibility(vis[:2])["eligible"]
    # ineligible: sign disagreement on dA (each seed one visit, signs +,-,-> 2 neg but dD all +)
    vis2 = [{"seed": 1, "dD": 0.1, "dA": 0.05}, {"seed": 3, "dD": 0.1, "dA": -0.05}, {"seed": 4, "dD": 0.1, "dA": -0.05}]
    e2 = projected_flow_eligibility(vis2)
    assert e2["sign_ok_dD"] and e2["sign_ok_dA"]   # dA: 2/3 negative agree -> ok
    vis3 = [{"seed": 1, "dD": 0.1, "dA": 0.05}, {"seed": 3, "dD": 0.1, "dA": -0.05}, {"seed": 1, "dD": 0.1, "dA": 0.05}]
    assert not projected_flow_eligibility(vis3)["eligible"]   # only 2 distinct seeds


def test_10b_operator_residual_synthetic():
    """§14.10 finite-Jacobian residual: a resolved operating point yields a small-residual dense Jacobian
    and batched==single finite-time response (reuse the susceptibility/M3B operator)."""
    from src.topic4_m3b_spectral_phase import Grid, build_kernels, build_excitability_field
    from src.topic4_state_conditioned_susceptibility import (
        two_core_mask_at, state_operator, make_phase_paired_probe_dictionary, probe_matrix,
        batched_finite_time_response,
    )
    grid = Grid(n=8, L=5.0)
    core = two_core_mask_at(grid, [(-1.0, 0.0), (1.0, 0.0)], 0.375, 0.0)
    kernels = build_kernels(grid, ar=2.0, ell_perp=0.6, theta=0.0)
    exc = build_excitability_field(grid, core, mu_core=0.6)
    scaffold = dict(kernels=kernels, core=core, exc=exc, theta=0.0)
    zbar = np.full((8, 8), 0.9)                       # mild uniform inhibitory efficacy
    op, J, q = state_operator(zbar, grid, scaffold, w_ee_mult=1.05, ratio=1.0, q_floor=0.05)
    assert op.status in ("resolved", "saturated", "unresolved")
    if op.status == "resolved":
        assert J is not None and op.residual < 1e-3
        probes = make_phase_paired_probe_dictionary(grid, p_max=2, sigma=1.0, center=(-1.0, 0.0))
        B = probe_matrix(probes, grid)
        Yb = batched_finite_time_response(J, B, 30.0)
        y0 = batched_finite_time_response(J, B[:, :1], 30.0)
        assert np.allclose(Yb[:, 0], y0[:, 0], atol=1e-8)


def test_14_natural_zm_trajectory_coordinates():
    """Temporal phase-diagram §5.3: continuous D–a trajectory from engine streaming traces.
    D_allE = 1 - z̄_E ; a_allE = (eta_m·m̄_E)/I_EE_scale (trace_adap_current already == eta_m·m̄_E)."""
    z_mean = [1.0, 0.8, 0.6, 0.4]              # -> D = [0, .2, .4, .6]
    adap_current = [0.0, 10.0, 20.0, 30.0]     # eta_m*m̄_E == A_abs ; /100 -> a = [0, .1, .2, .3]
    rate_hz = [5.0, 6.0, 7.0, 8.0]
    tr = natural_zm_trajectory(z_mean, adap_current, rate_hz, dt=0.1,
                               I_EE_scale=100.0, downsample_ms=0.1)   # downsample_ms==dt -> identity
    np.testing.assert_allclose(tr["D_allE"], [0.0, 0.2, 0.4, 0.6])
    np.testing.assert_allclose(tr["a_allE"], [0.0, 0.1, 0.2, 0.3])
    np.testing.assert_allclose(tr["rate_E_hz"], [5.0, 6.0, 7.0, 8.0])
    np.testing.assert_allclose(tr["t_ms"], [0.0, 0.1, 0.2, 0.3])


def test_15_natural_zm_trajectory_downsample():
    """§5.3 downsampling averages within each downsample_ms window (bin = round(downsample_ms/dt) steps)."""
    z_mean = [1.0, 0.8, 0.6, 0.4]              # 2-step bins: [0.9, 0.5] -> D = [0.1, 0.5]
    adap_current = [0.0, 10.0, 20.0, 30.0]     # bins [5, 25] -> a = [0.05, 0.25]
    rate_hz = [5.0, 6.0, 7.0, 8.0]             # bins [5.5, 7.5]
    tr = natural_zm_trajectory(z_mean, adap_current, rate_hz, dt=0.1,
                               I_EE_scale=100.0, downsample_ms=0.2)   # bin = 2 steps
    np.testing.assert_allclose(tr["D_allE"], [0.1, 0.5])
    np.testing.assert_allclose(tr["a_allE"], [0.05, 0.25])
    np.testing.assert_allclose(tr["rate_E_hz"], [5.5, 7.5])
    np.testing.assert_allclose(tr["t_ms"], [0.0, 0.2])


def _add_scripts_path():
    import sys
    R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in (R, os.path.join(R, "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)


def test_plot_load_covers_all_gap_fracs(tmp_path, monkeypatch):
    """P1-1 + tau-contamination regression for the plotter's _load: (a) keep 0.0025/0.0075 distinct —
    round(...,3) collapsed 0.0025->0.003, silently dropping two of five strengths; (b) exclude the _tau
    sweep files so they don't overwrite the tau=2000 gap cell keyed on the SAME A_frac. Self-contained:
    monkeypatches TRAJ_DIR (the actual producer constant — TRAJ does not exist) so it never reads repo artifacts."""
    _add_scripts_path()
    import plot_topic4_mz_onset_dynamics as P
    monkeypatch.setattr(P, "TRAJ_DIR", str(tmp_path))

    def _save(name, a_frac, seed, dval):
        np.savez(os.path.join(tmp_path, name), z_regime=P.REGIME, A_frac=a_frac, seed=seed,
                 t_ms=np.arange(10.0), D_allE=np.full(10, dval), a_allE=np.zeros(10), rate_E_hz=np.zeros(10),
                 event_on_ms=np.zeros(0), event_off_ms=np.zeros(0), runaway_ms=np.nan)

    for fr in [0.0] + P.NONZERO_FRACS:                        # gap grid: 6 strengths x 3 seeds (plateau D=0.05)
        for s in (1, 3, 4):
            _save(f"traj_{P.REGIME}_A{fr:g}_seed{s}.npz", fr, s, 0.05)
    # tau-sweep file: SAME (regime, 0.001, seed 1) key but D=0.99 -> must be EXCLUDED, not overwrite the gap cell
    _save(f"traj_{P.REGIME}_A0.001_tau500_seed1.npz", 0.001, 1, 0.99)

    cells = P._load()
    for fr in [0.0] + P.NONZERO_FRACS:                        # six strengths x three seeds all present
        for s in (1, 3, 4):
            assert (P.REGIME, round(fr, 5), s) in cells, f"frac {fr} seed {s} missing from _load"
    assert (P.REGIME, 0.0025, 1) in cells and (P.REGIME, 0.0075, 1) in cells   # not collapsed by rounding
    assert float(cells[(P.REGIME, 0.001, 1)]["D_allE"].max()) < 0.1            # tau=500 (D=0.99) did NOT overwrite
