"""Initial-state probe v1: engine interface, membership, digests and statistics.

Only real risks are tested: the default path must stay byte-identical, the
initial voltage must actually enter the engine and nothing else, the observer
must not perturb the trajectory, and the pre-registered statistics must match
hand-computed values.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "snn_engine"))
sys.path.insert(0, str(ROOT))

from kick_probe import simulate_kick  # noqa: E402
from model import build_network  # noqa: E402
from params import Params  # noqa: E402
from src.topic4_initial_state import (  # noqa: E402
    ProbeNotApplicable, RunObserver, block_matched_reference, bootstrap_mean_ci,
    build_initial_voltage, check_arm_symmetry, compare_input_digests,
    core_membership, exchange_test_p, holm_adjust, initial_voltage_summary,
    instability_flags, missing_pair_bounds, off_diagonal_terms, paired_effect,
    screen_verdict, validate_initial_voltage, window_mode_counts,
)


def _tiny(T=300.0, seed=11):
    p = Params(g=3.6, L=1.0, density=4000.0, T=T, dt=0.1, nu_ext_ratio=0.9, seed=seed)
    net = build_network(p, verbose=False)
    return p, net


def _run(p, net, seed=None, **kwargs):
    net["rng"] = np.random.default_rng(p.seed if seed is None else seed)
    return simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, **kwargs)


class _Tape:
    """Reference observer that records the raw per-step inputs."""

    def __init__(self):
        self.ext, self.xi = [], []

    def observe(self, t, tm, xi, nu_now, ext, delta_rate, V, I_E, I_I, spk):
        self.ext.append(np.array(ext, copy=True))
        self.xi.append(float(xi))


# ---------------------------------------------------------------- engine (E1-E8)
def test_default_none_equals_explicit_reset_voltage():
    p, net = _tiny()
    a = _run(p, net)
    n = net["NE"] + net["NI"]
    b = _run(p, net, initial_voltage=np.full(n, p.V_reset))
    assert np.array_equal(a["rate_E"], b["rate_E"])
    assert np.array_equal(a["E_spk_bool"], b["E_spk_bool"])
    assert np.array_equal(a["initial_V"], b["initial_V"])


def test_initial_voltage_enters_the_engine_and_changes_only_v():
    p, net = _tiny()
    n = net["NE"] + net["NI"]
    v0 = np.full(n, p.V_reset)
    v0[:50] += 1.0
    keep = v0.copy()
    a = _run(p, net)
    b = _run(p, net, initial_voltage=v0)
    assert np.array_equal(v0, keep)                     # E3: input not modified
    assert b["initial_V"][:50].tolist() == [p.V_reset + 1.0] * 50
    assert np.all(b["initial_V"][50:] == p.V_reset)
    assert not np.array_equal(a["E_spk_bool"], b["E_spk_bool"])
    v0[:50] += 5.0                                      # engine holds its own copy
    assert b["initial_V"][0] == p.V_reset + 1.0


def test_initial_voltage_validation():
    p, net = _tiny(T=20.0)
    n = net["NE"] + net["NI"]
    with pytest.raises(ValueError):
        _run(p, net, initial_voltage=np.full(n + 1, p.V_reset))
    bad = np.full(n, p.V_reset); bad[0] = np.nan
    with pytest.raises(ValueError):
        _run(p, net, initial_voltage=bad)
    high = np.full(n, p.V_reset); high[3] = p.V_th
    with pytest.raises(ValueError):
        _run(p, net, initial_voltage=high)
    vth = np.full(n, p.V_th); vth[7] = 12.0
    near = np.full(n, p.V_reset); near[7] = 12.5
    with pytest.raises(ValueError):
        _run(p, net, initial_voltage=near, V_th_per_neuron=vth)


def test_initial_voltage_excludes_resume_state():
    p, net = _tiny(T=100.0)
    captured = {}
    net["rng"] = np.random.default_rng(p.seed)
    simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, checkpoint_steps=[200],
                  checkpoint_sink=lambda step, state: captured.setdefault(step, state))
    n = net["NE"] + net["NI"]
    with pytest.raises(ValueError):
        simulate_kick(Params(g=3.6, L=1.0, density=4000.0, T=50.0, dt=0.1,
                             nu_ext_ratio=0.9, seed=p.seed), net,
                      KICK_BOOST=0.0, t_kick=1e9, resume_state=captured[200],
                      time_offset_ms=20.0, initial_voltage=np.full(n, p.V_reset))


def test_observer_does_not_perturb_the_trajectory_and_sees_identical_inputs():
    p, net = _tiny(T=250.0)
    n_e, n = net["NE"], net["NE"] + net["NI"]
    groups = {"A": np.arange(0, 30), "B": np.arange(30, 60),
              "surround": np.arange(60, n_e), "I": np.arange(n_e, n)}
    steps = int(round(p.T / p.dt))
    plain = _run(p, net)
    obs0 = RunObserver(dt_ms=p.dt, n_e=n_e, n_total=n, groups=groups, n_steps=steps,
                       segment_ms=50.0, trace_ms=1.0)
    with_obs = _run(p, net, step_observer=obs0)
    assert np.array_equal(plain["rate_E"], with_obs["rate_E"])       # E7
    assert np.array_equal(plain["E_spk_bool"], with_obs["E_spk_bool"])
    out0 = obs0.finish()
    assert len(out0["segments"]) == 5 and all(s["complete"] for s in out0["segments"])
    assert out0["trace"]["time_ms"].shape[0] == 250
    assert out0["trace"]["E_spikes"].sum() == plain["rate_E"].sum() * n_e * p.dt / 1e3
    # a perturbed initial state receives identical external inputs (D2)
    v0 = np.full(n, p.V_reset); v0[groups["A"]] += 1.0
    obs1 = RunObserver(dt_ms=p.dt, n_e=n_e, n_total=n, groups=groups, n_steps=steps,
                       segment_ms=50.0, trace_ms=1.0)
    tape = _Tape()
    perturbed = _run(p, net, initial_voltage=v0, step_observer=obs1)
    _run(p, net, initial_voltage=v0, step_observer=tape)
    out1 = obs1.finish()
    comparison = compare_input_digests(out0["segments"], out1["segments"])
    assert comparison["all_complete_segments_equal"] and not comparison["prefix_only"]
    assert np.array_equal(out0["ext_segment_sums"], out1["ext_segment_sums"])
    assert not np.array_equal(plain["E_spk_bool"], perturbed["E_spk_bool"])
    # the digest is the actual tape: the exact per-segment sums match the raw counts
    raw = np.stack(tape.ext).reshape(5, 500, n).sum(axis=1)
    assert np.array_equal(raw.astype(np.int32), out1["ext_segment_sums"])
    # different dynamics seed -> different digests (D3)
    obs2 = RunObserver(dt_ms=p.dt, n_e=n_e, n_total=n, groups=groups, n_steps=steps,
                       segment_ms=50.0, trace_ms=1.0)
    _run(p, net, seed=p.seed + 1, step_observer=obs2)
    other = compare_input_digests(out0["segments"], obs2.finish()["segments"])
    assert not other["all_complete_segments_equal"]


def test_digest_prefix_only_when_lengths_differ():
    a = [{"n_steps": 10, "stream_sha256": "x", "ext_sum_sha256": "y", "delta_sum_sha256": None}] * 3
    b = a[:2] + [{**a[0], "n_steps": 4}]
    out = compare_input_digests(a, b)
    assert out["prefix_only"] and out["n_complete_segments_compared"] == 2
    assert out["all_complete_segments_equal"]


# ---------------------------------------------------------- membership (M1-M5)
def test_core_membership_rules():
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [9.0, 0.0], [10.0, 0.0],
                          [5.0, 0.0], [4.5, 0.0], [10.0, 5.0]])
    h = np.array([1, 1, 1, 1, 1, 1, 0, 1], float)
    centers = [[0.0, 0.0], [10.0, 0.0]]
    out = core_membership(positions, h, centers)
    # cell 5 is equidistant (5 mm) -> lower core index (core 0); cell 6 has h=0
    assert out["n_A"] == 4 and out["n_B"] == 3 and out["K"] == 3
    assert out["S_A"].tolist() == [0, 1, 2]              # nearest three of core A
    assert out["S_B"].tolist() == [4, 3, 7]              # (distance, index) order
    assert out["dropped_from_larger_group"] == 1
    with pytest.raises(ProbeNotApplicable):
        core_membership(positions, np.array([1, 1, 1, 0, 0, 0, 0, 0], float), centers)


def test_index_tie_break_within_core():
    positions = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [30.0, 0.0], [31.0, 0.0]])
    h = np.ones(5)
    out = core_membership(positions, h, [[0.0, 0.0], [30.0, 0.0]])
    assert out["S_A"].tolist() == [0, 1]                 # equal distance -> lower index first


def test_arm_construction_and_validation():
    membership = {"S_A": np.array([0, 1]), "S_B": np.array([4, 5])}
    b0 = build_initial_voltage(8, 11.0, membership, 1.0, "B0")
    b1 = build_initial_voltage(8, 11.0, membership, 1.0, "B1")
    b2 = build_initial_voltage(8, 11.0, membership, 1.0, "B2")
    assert np.all(b0 == 11.0)
    assert b1.tolist() == [12, 12, 11, 11, 11, 11, 11, 11]
    assert b2.tolist() == [11, 11, 11, 11, 12, 12, 11, 11]
    s1, s2 = initial_voltage_summary(b1, 11.0), initial_voltage_summary(b2, 11.0)
    assert check_arm_symmetry(s1, s2)
    with pytest.raises(RuntimeError):
        check_arm_symmetry(s1, initial_voltage_summary(b0, 11.0))
    vth = np.full(8, 14.0)
    audit = validate_initial_voltage(b1, vth, 11.0, 0.5)
    assert audit["pass"] and audit["minimum_margin_perturbed_mV"] == 2.0
    vth[1] = 12.3
    with pytest.raises(ProbeNotApplicable):
        validate_initial_voltage(b1, vth, 11.0, 0.5)


# ---------------------------------------------------------- windows (O2-O4)
def test_window_mode_counts_use_midpoints_and_keep_unclassifiable_separate():
    times = np.array([11990.0, 12000.0, 15000.0, 23999.0, 24000.0])
    labels = np.array([0, 1, -1, 0, 0])
    out = window_mode_counts(times, labels, [0, 1, 2, 3, 4], [12000.0, 24000.0])
    assert out["mode_counts"] == [1, 1] and out["n_unclassifiable"] == 1
    assert out["n_primary_in_window"] == 3 and out["mode0_proportion"] == 0.5
    empty = window_mode_counts(times, labels, [0], [12000.0, 24000.0])
    assert empty["mode0_proportion"] is None


# ------------------------------------------------------- statistics (S1-S7)
def test_exchange_test_matches_enumeration():
    d = np.array([0.2, 0.1, 0.3, -0.05])
    out = exchange_test_p(d)
    signs = np.array(list(itertools.product((-1, 1), repeat=4)))
    expected = np.mean(np.abs(signs @ d) / 4 >= abs(d.mean()) - 1e-12)
    assert out["p_two_sided"] == pytest.approx(expected) and out["n_assignments"] == 16
    assert exchange_test_p(np.ones(12))["p_two_sided"] == pytest.approx(2 / 4096)


def test_bootstrap_is_seeded_and_covers_the_mean():
    d = np.linspace(-0.2, 0.4, 12)
    a = bootstrap_mean_ci(d, n_resamples=2000, seed=5)
    b = bootstrap_mean_ci(d, n_resamples=2000, seed=5)
    assert a == b and a["ci"][0] <= d.mean() <= a["ci"][1]


def test_instability_flags():
    dominated = np.array([0.9, 0.01, 0.01, 0.01])
    flags = instability_flags(dominated, [0.1, 0.5], 0.01)
    assert flags["single_pair_dominates"] and flags["unstable"]
    flip = np.array([0.5, -0.1, -0.1, -0.1])
    assert instability_flags(flip, [-0.1, 0.4], 0.5)["leave_one_out_sign_flip"]
    steady = np.array([0.2, 0.25, 0.18, 0.22])
    calm = instability_flags(steady, [0.15, 0.25], 0.01)
    assert not calm["unstable"]
    assert instability_flags(steady, [-0.1, 0.4], 0.01)["bootstrap_exchange_disagree"]
    assert instability_flags(np.zeros(4), [0.0, 0.0], 1.0)["max_single_pair_contribution"] == 0.0


def test_screen_verdicts_and_missing_bounds():
    strong = paired_effect(np.full(12, 0.2) + np.linspace(-0.02, 0.02, 12),
                           n_resamples=500, seed=1)
    assert screen_verdict(strong) == "PERSISTENT_INITIAL_STATE_EFFECT_SINGLE_GRAPH"
    tiny = paired_effect(np.linspace(-0.03, 0.03, 12), n_resamples=500, seed=1)
    assert screen_verdict(tiny) == "SMALL_EFFECT_BOUNDED_FOR_TESTED_PROBE"
    wide = paired_effect(np.array([0.5, -0.4, 0.3, -0.35, 0.45, -0.2, 0.1, -0.5, 0.4, 0.3, -0.3, 0.2]),
                         n_resamples=500, seed=1)
    assert screen_verdict(wide) == "INCONCLUSIVE"
    assert screen_verdict(wide, early_effect=strong) == "STARTUP_SENSITIVITY_ONLY"
    assert screen_verdict(None) == "NOT_ESTIMABLE"
    assert missing_pair_bounds([0.1] * 10, 2) == pytest.approx([(1.0 - 2) / 12, (1.0 + 2) / 12])
    assert holm_adjust([0.01, 0.04]) == pytest.approx([0.02, 0.04])


# ------------------------------------------------------ propagation quality (Q3, Q5)
def test_off_diagonal_terms_match_explicit_identity():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(7, 4)); target = rng.normal(size=4)
    out = off_diagonal_terms(x, target)
    centered = x - target
    cross = centered @ centered.T
    explicit = (cross.sum() - np.trace(cross)) / (7 * 6)
    assert out["D_off"] == pytest.approx(explicit)
    assert out["D_off"] == pytest.approx(out["A_mean_target_squared_distance"] - out["B_finite_event_subtraction"])
    assert off_diagonal_terms(x[:1], target)["status"] == "NOT_ESTIMABLE"


def test_block_matched_reference_is_seeded_and_uses_blocks():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 3)); blocks = np.repeat([1, 2, 3, 4], 10)
    a = block_matched_reference(x, blocks, x.mean(0), n_model=5, n_resamples=50, seed=3)
    b = block_matched_reference(x, blocks, x.mean(0), n_model=5, n_resamples=50, seed=3)
    assert a == b and a["n_blocks_available"] == 4 and len(a["D_off_quantiles"]) == 3
    assert block_matched_reference(x, blocks, x.mean(0), n_model=1, n_resamples=5, seed=3)["status"] == "NOT_ESTIMABLE"
