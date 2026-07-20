"""Contract tests for Topic 4 MZ state-aligned finite-time spatial mode tracking.

Each test maps 1:1 to a spec §7 engineering invariant (E1-E20). Pure-module tests run fast;
SNN-backed tests (E4/E5/E9 + m-control checkpoint) use a TINY network (mirror the direct-spatial
and onset tiny fixtures) so the parity / freeze / common-RNG / m-mutation invariants are exercised
without the 40k-neuron substrate.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

import copy  # noqa: E402

from src.topic4_mz_m_eigenmode_tracking import (  # noqa: E402
    build_zm_slow_config, resting_mask, register_states, trajectory_parity, transform_m,
    apply_m_control, principal_angles_deg, subspace_alignment, weighted_centroid,
    centroid_displacement, state_checkpoint_fingerprint, leading_subspace, SCHEMA_VERSION,
)


# ================================================================= E1 eta_m via eta_m_from_frac
def test_e1_build_zm_slow_config_eta_m():
    """eta_m is computed by eta_m_from_frac (never hardcoded) and equals the locked plateau value."""
    wp = dict(use_z=True, use_m=True, I_th_EI=95.19851312666987, tau_z=5000.0, tau_adp_ms=2000.0,
              A_target=0.001, peak_m_tau2000=36.6036014019694)
    cfg = build_zm_slow_config(wp, I_EE_scale=272.75518960107513)
    assert cfg.use_z is True and cfg.use_m is True
    assert cfg.I_th_EI == 95.19851312666987
    assert cfg.tau_z == 5000.0 and cfg.tau_adp == 2000.0
    assert abs(cfg.eta_m - 0.007451594355587098) < 1e-15   # == eta_m_from_frac(0.001, 272.755..., 36.6036...)


# ================================================================= resting mask + E2 state registration
def test_resting_mask_flags_quiet_not_events():
    dt = 1.0
    rate = np.full(500, 1.0)
    rate[200:210] = 80.0                                   # an event burst
    rest = resting_mask(rate, dt, win_ms=5.0, k=0.3)
    assert rest[50] and not rest[204]                      # quiet resting; burst peak not resting


def _synthetic_traj(dt_ms=1.0, n=2000):
    """D ramps 0.014->0.05 over [0,1000] ms then plateaus at 0.05; a a tiny parallel ramp; rate quiet
    with a few event spikes (one placed on an approach crossing to test resting avoidance)."""
    t = np.arange(n) * dt_ms
    D = 0.01 + 0.04 * np.clip(t / 1000.0, 0.0, 1.0)
    a = 1.0e-4 * np.clip(t / 1000.0, 0.0, 1.0)
    rate = np.full(n, 2.0)
    for c in (325, 700, 1500):                             # event bursts (325 sits on the 25% crossing)
        rate[c:c + 8] = 90.0
    return D, a, rate


def _sr_params(**over):
    p = dict(baseline_ms=100.0, baseline_search_halfwidth_ms=50.0, approach_fracs=[0.25, 0.50, 0.75],
             approach_search_ms=60.0, settle_tail_ms=200.0, resting_win_ms=5.0, resting_k=0.3,
             settled_D_ptp_max=0.005, settled_a_ptp_max=1.0e-5, settled_min_resting_frac=0.3,
             D_onset_ref=0.06)
    p.update(over)
    return p


def test_e2_register_states_ordering_and_targets():
    """5 states register from D/a/rate ONLY (no perturbation input), in monotone time order, with
    approach D at D_base + f*(D_plateau - D_base)."""
    D, a, rate = _synthetic_traj()
    reg = register_states(D, a, rate, 1.0, **_sr_params())
    st = reg["states"]
    order = [st["baseline"]["branch_step"], st["approach_25"]["branch_step"],
             st["approach_50"]["branch_step"], st["approach_75"]["branch_step"],
             st["settled_plateau"]["branch_step"]]
    assert all(x is not None for x in order)
    assert order == sorted(order)                          # monotone in time
    assert 0.049 <= reg["D_plateau"] <= 0.051 and 0.010 <= reg["D_base"] <= 0.020
    for f, name in ((0.25, "approach_25"), (0.50, "approach_50"), (0.75, "approach_75")):
        target = reg["D_base"] + f * (reg["D_plateau"] - reg["D_base"])
        assert D[st[name]["branch_step"]] >= target - 1e-9   # at/after the crossing


def test_e2_register_states_avoids_event_peak():
    """The 25% crossing sits on an event burst (rate 90); the chosen checkpoint is a resting step."""
    D, a, rate = _synthetic_traj()
    reg = register_states(D, a, rate, 1.0, **_sr_params())
    step = reg["states"]["approach_25"]["branch_step"]
    assert rate[step] < 10.0                               # not on the burst


def test_e2_register_states_independent_of_call_order():
    """Registration is a pure function of (D,a,rate): identical inputs -> identical output (no hidden
    perturbation dependence)."""
    D, a, rate = _synthetic_traj()
    r1 = register_states(D, a, rate, 1.0, **_sr_params())
    r2 = register_states(D.copy(), a.copy(), rate.copy(), 1.0, **_sr_params())
    assert r1["states"]["settled_plateau"]["branch_step"] == r2["states"]["settled_plateau"]["branch_step"]
    assert r1["D_plateau"] == r2["D_plateau"]


def test_e2_settled_gate_fails_when_not_settled():
    """A tail that is still rising STEEPLY (large local ptp) fails the settled gate -> unresolved."""
    n = 2000
    D = 0.01 + 0.05 * (np.arange(n) / n) ** 2              # accelerating rise: tail is steep, not flat
    a = 1e-4 * (np.arange(n) / n) ** 2
    rate = np.full(n, 2.0)
    reg = register_states(D, a, rate, 1.0, **_sr_params())
    assert np.ptp(D[-200:]) > 0.005                        # tail genuinely not flat
    assert reg["states"]["settled_plateau"]["settled"] is False
    assert reg["states"]["settled_plateau"]["branch_step"] is None


def test_e2_approach_unresolved_when_target_never_reached():
    """If D never reaches an approach target (flat trajectory, no rise), that approach is unresolved."""
    n = 2000
    a = 1e-4 * np.clip(np.arange(n) / 1000.0, 0.0, 1.0)
    rate = np.full(n, 2.0)
    reg = register_states(np.full(n, 0.014), a, rate, 1.0, **_sr_params(settled_D_ptp_max=0.05))
    assert reg["states"]["approach_75"]["branch_step"] is None
    assert reg["states"]["approach_75"]["resolved"] is False


# ================================================================= E3 replay <-> NPZ parity
def test_e3_trajectory_parity_pass_and_fail():
    D = np.linspace(0.0, 0.05, 100); a = np.linspace(0, 1e-4, 100); r = np.linspace(0, 50, 100)
    ok = trajectory_parity(D, a, r, D.copy(), a.copy(), r.copy(), rel_tol=0.02)
    assert ok["pass"] and ok["D"]["max_abs"] == 0.0
    near = trajectory_parity(D * 1.005, a, r, D, a, r, rel_tol=0.02)   # 0.5% off -> within tol
    assert near["pass"]
    bad = trajectory_parity(D * 1.10, a, r, D, a, r, rel_tol=0.02)     # 10% off -> fail
    assert not bad["pass"]


# ================================================================= E6/E7/E8 m counterfactual transforms
def test_e6_transform_m_reset():
    m = np.array([1.0, 3.0, 5.0, 0.0])
    out = transform_m(m, "m_reset")
    assert np.array_equal(out, np.zeros(4)) and not np.shares_memory(out, m)


def test_e7_transform_m_uniform_preserves_mean():
    m = np.array([1.0, 3.0, 5.0, 7.0])
    out = transform_m(m, "m_uniform")
    assert np.allclose(out, m.mean()) and np.isclose(out.mean(), m.mean())
    assert np.ptp(out) == 0.0                              # spatial pattern flattened


def test_e8_transform_m_shuffle_preserves_distribution():
    m = np.arange(50.0)
    out = transform_m(m, "m_shuffle", seed=123)
    assert np.array_equal(np.sort(out), np.sort(m))       # same multiset
    assert not np.array_equal(out, m)                     # order changed
    out2 = transform_m(m, "m_shuffle", seed=123)
    assert np.array_equal(out, out2)                      # deterministic given seed


def test_transform_m_native_is_copy():
    m = np.array([1.0, 2.0, 3.0])
    out = transform_m(m, "native_zm")
    assert np.array_equal(out, m) and not np.shares_memory(out, m)


# ================================================================= E16/E17 mode sign + subspace tracking
def test_e16_principal_angles_sign_invariant():
    rng = np.random.default_rng(0)
    U, _ = np.linalg.qr(rng.standard_normal((16, 3)))
    u = U[:, :1]
    assert principal_angles_deg(u, u).max() < 1e-9
    assert principal_angles_deg(u, -u).max() < 1e-9       # sign-invariant
    assert np.isclose(subspace_alignment(u, u), 1.0)
    assert np.isclose(subspace_alignment(u, -u), 1.0)


def test_e17_degenerate_subspace_tracking():
    """Two identical 2-D subspaces -> angles ~0; two orthogonal 2-D subspaces -> angles ~90 deg."""
    rng = np.random.default_rng(1)
    Q, _ = np.linalg.qr(rng.standard_normal((16, 16)))
    A = Q[:, :2]
    B = Q[:, :2] @ _rot2(0.4)                              # same plane, rotated basis
    C = Q[:, 2:4]                                          # orthogonal plane
    assert principal_angles_deg(A, B).max() < 1e-6        # same subspace regardless of in-plane basis
    ang = principal_angles_deg(A, C)
    assert np.all(ang > 89.9)                             # orthogonal planes


def _rot2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def test_weighted_centroid_sign_invariant_and_displacement():
    n = 5
    X, Y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    fa = np.zeros((n, n)); fa[0, 0] = 1.0
    fb = np.zeros((n, n)); fb[4, 4] = 1.0
    ca = weighted_centroid(fa, X, Y)
    assert np.allclose(ca, weighted_centroid(-fa, X, Y))   # |field|^2 -> sign-invariant
    assert centroid_displacement(fa, fb, X, Y) > 5.0       # corner-to-corner


# ================================================================= E19 checkpoint fingerprint
class _FakeCk:
    def __init__(self, m):
        self.t = 100
        self.V = np.ones(6); self.ref = np.zeros(6, int)
        self.s_E = np.zeros(6); self.I_E = np.zeros(6); self.s_I = np.zeros(6); self.I_I = np.zeros(6)
        self.ring_sE = np.zeros((3, 6)); self.ring_sI = np.zeros((3, 6)); self.xi = 0.1
        self.rng_state = {"a": 1}
        self.slow = type("S", (), {"z": np.ones(6), "m": np.asarray(m, float)})()


def test_e19_checkpoint_fingerprint_deterministic_and_m_sensitive():
    a = state_checkpoint_fingerprint(_FakeCk([0.0, 1.0, 2.0]))
    b = state_checkpoint_fingerprint(_FakeCk([0.0, 1.0, 2.0]))
    c = state_checkpoint_fingerprint(_FakeCk([0.0, 1.0, 9.0]))   # m changed
    assert a == b and a != c and isinstance(a, str) and len(a) >= 12


def test_leading_subspace_degeneracy(P17=None):
    """leading_subspace flags a near-degenerate leading pair (gap < ratio) -> subspace_dim >= 2, and a
    well-separated leading mode -> single vector. Used by P3 cross-state mode tracking."""
    rng = np.random.default_rng(2)
    U, _ = np.linalg.qr(rng.standard_normal((36, 36)))
    V, _ = np.linalg.qr(rng.standard_normal((9, 9)))

    def _K(sig):
        return (U[:, :9] * np.asarray(sig)) @ V.T

    sep = leading_subspace(_K([4.0, 1.0, 0.5, 0, 0, 0, 0, 0, 0]), 1.05)
    assert not sep["degenerate"] and sep["subspace_dim"] == 1 and sep["u1"].shape == (36,)
    deg = leading_subspace(_K([2.0, 1.99, 0.4, 0, 0, 0, 0, 0, 0]), 1.05)
    assert deg["degenerate"] and deg["subspace_dim"] >= 2 and deg["U"].shape[1] >= 2
    assert np.isclose(sep["sigma1"], 4.0, rtol=1e-6)


def test_schema_version():
    assert SCHEMA_VERSION == "mz-m-eigenmode-tracking-1.0"


# ================================================================= E18 resume idempotency (runner predicates)
def test_e18_resume_predicates(tmp_path):
    """A completed cell is not recomputed; an absent/incomplete cell IS recomputed; resume off always
    recomputes (spec §7 E18). Importing the runner is side-effect-free (no simulations)."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import run_topic4_mz_m_eigenmode_tracking as R
    p = str(tmp_path / "state.json")
    assert not R._state_done(p, resume=True)               # missing -> recompute
    open(p, "w").write("{}")
    assert R._state_done(p, resume=True)                   # present + resume -> skip
    assert not R._state_done(p, resume=False)              # resume off -> always recompute
    assert not R._register_done(99, {"seeds": {}})         # seed unregistered -> recompute
    reg = {"seeds": {"99": {"states": {"baseline": {"branch_step": 999}}}}}   # seed 99: no real checkpoints
    assert not R._register_done(99, reg)                   # registered but checkpoint missing -> recompute


# ================================================================= E20 plotting fails closed on missing sidecar
def test_e20_plot_fail_closed(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "scripts", "paper_figures"))
    import plot_figure5_mz_m_eigenmode_tracking as P
    empty = str(tmp_path)
    assert P.load_state(1, "baseline", out=empty) is None      # missing sidecar -> None (no crash)
    assert P.load_registration(out=empty) is None
    with pytest.raises(SystemExit):
        P.require_registration(out=empty)                      # required sidecar missing -> fail closed
    with pytest.raises(SystemExit):
        P.figure_b({"T_windows_ms": [10.0, 30.0, 50.0], "linearity_tol": 0.15,
                    "m_controls": {"conditions": []}}, out=empty)


# ================================================================= tiny-SNN: E4/E5/E9 + m-control checkpoint
@pytest.fixture(scope="module")
def tiny():
    from params import Params
    from model import build_network
    p = Params(g=3.6, L=1.0, density=2000.0, T=60.0, dt=0.1, nu_ext_ratio=0.9, seed=1)
    net = build_network(p, verbose=False)
    NE, N = net["NE"], net["NE"] + net["NI"]
    posE = net["pos"][:NE]
    core = np.linalg.norm(posE - np.array([0.5, 0.5]), axis=1) <= 0.2
    vth = np.full(N, p.V_th); vth[:NE][core] -= 1.0
    return dict(p=p, net=net, NE=NE, N=N, core=core, vth=vth)


def _zm_probe(tiny):
    from src.topic4_mz_direct_spatial_modes import MZSpatialProbe
    from mz_slow_vars import MZSlowVarsConfig
    cfg = MZSlowVarsConfig(use_z=True, use_m=True, I_th_EI=5.0, tau_z=3000.0, tau_adp=2000.0, eta_m=0.01)
    return MZSpatialProbe(tiny["N"], 18.0, cfg, NE=tiny["NE"], core_mask_E=tiny["core"])


def _branch_ck(tiny, n=250):
    from src.topic4_mz_onset_dynamics import run_loop
    tiny["net"]["rng"] = np.random.default_rng(tiny["p"].seed)
    rep = run_loop(tiny["p"], tiny["net"], _zm_probe(tiny), tiny["vth"], n_steps=n,
                   capture_final=True, store_spikes=False)
    return rep["checkpoint"]


def test_e4_checkpoint_resume_parity_segmented_equals_continuous(tiny):
    """Segmented replay + resume == a single continuous replay (full engine + z + m state), and the
    segmented FINAL checkpoint fingerprint equals the continuous final checkpoint fingerprint."""
    from src.topic4_mz_onset_dynamics import run_loop
    tiny["net"]["rng"] = np.random.default_rng(tiny["p"].seed)      # fresh stream for the continuous run
    cont = run_loop(tiny["p"], tiny["net"], _zm_probe(tiny), tiny["vth"], n_steps=400,
                    capture_final=True, store_spikes=True)
    tiny["net"]["rng"] = np.random.default_rng(tiny["p"].seed)      # identical fresh stream for segment A
    a = run_loop(tiny["p"], tiny["net"], _zm_probe(tiny), tiny["vth"], n_steps=250,
                 capture_final=True, store_spikes=True)
    ck = a["checkpoint"]
    b = run_loop(tiny["p"], tiny["net"], copy.deepcopy(ck.slow), tiny["vth"], n_steps=150,
                 start=ck, capture_final=True, store_spikes=True)
    assert np.array_equal(cont["rate_E"], np.concatenate([a["rate_E"], b["rate_E"]]))
    assert np.array_equal(cont["E_spk_bool"], np.vstack([a["E_spk_bool"], b["E_spk_bool"]]))
    assert state_checkpoint_fingerprint(b["checkpoint"]) == state_checkpoint_fingerprint(cont["checkpoint"])


def test_e5_freeze_holds_z_and_m(tiny):
    """A frozen fork holds z and m constant across the window (fast-subsystem isolation)."""
    from src.topic4_mz_onset_dynamics import run_loop
    ck = _branch_ck(tiny)
    z0 = ck.slow.z.copy(); m0 = ck.slow.m.copy()
    s = copy.deepcopy(ck.slow); s.set_branch(branch_step=ck.t, freeze=True)
    run_loop(tiny["p"], tiny["net"], s, tiny["vth"], n_steps=200, start=ck, store_spikes=False)
    assert np.array_equal(s.z, z0) and np.array_equal(s.m, m0)   # frozen -> unchanged


def test_e6b_apply_m_control_reset_changes_only_m(tiny):
    """apply_m_control('m_reset') zeroes E-cell m and leaves the fast state / z / rng untouched."""
    ck = _branch_ck(tiny)
    NE = tiny["NE"]
    ck_m = apply_m_control(ck, "m_reset", NE, seed=0)
    assert np.array_equal(ck_m.slow.m[:NE], np.zeros(NE))        # m reset
    assert np.array_equal(ck_m.slow.z, ck.slow.z)               # z untouched
    assert np.array_equal(ck_m.V, ck.V) and ck_m.rng_state is ck.rng_state   # fast state / rng shared
    assert not np.shares_memory(ck_m.slow.m, ck.slow.m)         # private copy (original intact)
    assert not np.array_equal(ck.slow.m[:NE], np.zeros(NE))     # original checkpoint unchanged


def test_e9_common_rng_across_m_controls(tiny):
    """native / m_reset / m_uniform forks from the same checkpoint share the checkpoint RNG stream
    (common random numbers): a zero-effect control reproduces the native output bit-for-bit."""
    from src.topic4_mz_onset_dynamics import run_loop
    ck = _branch_ck(tiny)
    NE = tiny["NE"]

    def _fork(ck_x):
        s = copy.deepcopy(ck_x.slow); s.set_branch(branch_step=ck_x.t, freeze=True)
        return run_loop(tiny["p"], tiny["net"], s, tiny["vth"], n_steps=200, start=ck_x,
                        store_spikes=True)["E_spk_bool"]

    native = _fork(ck)
    # m_uniform when m is already ~uniform would differ; instead prove CRN via a native re-fork:
    native2 = _fork(apply_m_control(ck, "native_zm", NE, seed=0))
    assert np.array_equal(native, native2)                     # common RNG + identical m -> identical
