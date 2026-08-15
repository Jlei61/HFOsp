"""FCXR-LC6B contracts: the D/H freeze, its off-by-default parity, and the clamp-window classifier."""
from __future__ import annotations

import copy
import inspect
import os
import sys

import numpy as np
import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

from model import build_network  # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402
from params import Params  # noqa: E402
from src.topic4_fcxr_lc3 import run_fcxr_loop, state_hash  # noqa: E402
from src.topic4_fcxr_lc6b_clamp import (  # noqa: E402
    ARMS, apply_slow_clamp, cell_rate_distribution, classify_clamp_window,
    slow_field_constancy,
)


def _cfg(**over):
    """The LC6A runtime shape: full conductance, RC1 saturation, live LC2 H, frozen X, live Z."""
    base = dict(
        membrane_mode="full_conductance", E_E=58.0, c_E=1.0,
        ff_conductance=False, rec_conductance=True, rec_sat_g=21.6,
        v_match=18.0, e_gaba=0.0, e_k=0.0,
        max_total_conductance=99.0, fail_on_clip=False,
        use_h_lc2=True, tau_h_lc2=80.0, theta_h_lc2=0.03, k_h_lc2=0.02, rho_h_lc2=0.2,
        use_x=True, tau_y=120.0, tau_x=800.0, x_min=0.1, y_gate=2.0, K_y=4.0, hill_n=4,
        use_z=True, I_th_EI=1.0, tau_z=3000.0, z_scope="local_only",
        global_gaba_mode="additive",
    )
    base.update(over)
    return MZSlowVarsConfig(**base)


def _case(seed=17, **over):
    p = Params(L=4.0, density=80.0, T=120.0, dt=0.1, nu_ext_ratio=0.92, seed=seed)
    net = build_network(p, verbose=False)
    ne, n = net["NE"], net["NE"] + net["NI"]
    vth = np.full(n, p.V_th)
    vth[: min(8, ne)] -= 0.8
    cfg = _cfg(x_relay_frozen_E=np.ones(ne), **over)
    slow = MZSlowVars(n, 18.0, cfg, NE=ne, core_mask_E=np.zeros(ne, bool))
    net["rng"] = np.random.default_rng(seed)
    return p, net, slow, vth


def _warm(seed=17, n_steps=400, **over):
    """A live state with H and z genuinely moved off their initial values."""
    p, net, slow, vth = _case(seed, **over)
    out = run_fcxr_loop(p, net, slow=slow, n_steps=n_steps, capture_final=True,
                        store_spikes=False, v_th_per_neuron=vth)
    state = out["checkpoint"]
    ne = int(state.slow.NE)
    assert state.slow.h_lc2_E.max() > 0.0, "warmup must move H"
    assert state.slow.z[:ne].min() < 1.0, "warmup must move z"
    return p, net, state, vth


def _step_states(p, net, state, *, n_steps, every=1, vth=None):
    """Advance one step at a time, capturing the state after each step."""
    states, cur = [], state
    for _ in range(n_steps):
        out = run_fcxr_loop(p, net, start=cur, n_steps=every, capture_final=True,
                            store_spikes=False, v_th_per_neuron=vth)
        cur = out["checkpoint"]
        states.append(cur)
    return states


# ------------------------------------------------------------------ TDD-1 off-by-default parity

def test_h_freeze_off_is_byte_identical_to_the_unfrozen_engine():
    p, net_a, slow_a, vth = _case()
    ref = run_fcxr_loop(p, net_a, slow=slow_a, n_steps=600, capture_final=True,
                        store_spikes=True, v_th_per_neuron=vth)
    p, net_b, slow_b, vth = _case()
    assert slow_b.cfg.h_lc2_frozen_E is None
    got = run_fcxr_loop(p, net_b, slow=slow_b, n_steps=600, capture_final=True,
                        store_spikes=True, v_th_per_neuron=vth)
    assert np.array_equal(ref["E_spk_bool"], got["E_spk_bool"])
    assert np.array_equal(ref["rate_E"], got["rate_E"])
    assert state_hash(ref["checkpoint"]) == state_hash(got["checkpoint"])
    assert ref["checkpoint"].slow.h_lc2_E.tobytes() == got["checkpoint"].slow.h_lc2_E.tobytes()


def test_h_evolves_when_not_frozen():
    """Guards the parity test above: a frozen-off run must still be a run in which H moves."""
    _p, _net, state, _vth = _warm()
    h0 = state.slow.h_lc2_E.copy()
    _p, net, state2, _vth = _warm(n_steps=800)
    assert not np.array_equal(h0, state2.slow.h_lc2_E)


# ------------------------------------------------------------------ TDD-2/3/5/6 constancy

def test_frozen_h_is_bitwise_constant_every_step():
    p, net, state, vth = _warm()
    child, record = apply_slow_clamp(state, clamp_d=False, clamp_h=True)
    h0 = child.slow.h_lc2_E.copy()
    states = _step_states(p, net, child, n_steps=25, vth=vth)
    for later in states:
        assert later.slow.h_lc2_E.tobytes() == h0.tobytes()
    assert slow_field_constancy([child, *states])["h_lc2_E"]["bitwise_constant"] is True
    assert record["clamp_h"] is True and record["use_h_lc2"] is True


def test_frozen_h_keeps_recording_its_source_and_trace():
    p, net, state, vth = _warm()
    child, _record = apply_slow_clamp(state, clamp_d=False, clamp_h=True)
    h0 = child.slow.h_lc2_E.copy()
    out = run_fcxr_loop(p, net, start=child, n_steps=60, capture_final=True,
                        store_spikes=False, v_th_per_neuron=vth)
    slow = out["checkpoint"].slow
    # the source cache is still written every frame ...
    assert np.any(np.asarray(slow._h_source_lc2_E) > 0.0)
    assert len(slow.trace_gA_raw_lc2_mean) >= 60
    # ... but never reaches the held state, and the trace reports the held value
    assert slow.h_lc2_E.tobytes() == h0.tobytes()
    assert all(value == pytest.approx(float(h0.mean()), abs=0.0)
               for value in slow.trace_h_lc2_mean[-60:])


def test_frozen_z_is_bitwise_constant_and_still_modulates_the_membrane():
    p, net, state, vth = _warm()
    ne = int(state.slow.NE)
    child, record = apply_slow_clamp(state, clamp_d=True, clamp_h=False)
    assert child.slow.cfg.use_z is False
    assert child.slow.cfg.z_frozen_E is not None
    z0 = child.slow.z[:ne].copy()
    states = _step_states(p, net, child, n_steps=25, vth=vth)
    for later in states:
        assert later.slow.z[:ne].tobytes() == z0.tobytes()
    assert record["clamp_d"] is True and record["use_z"] is False
    # The frozen field is APPLIED, not bypassed.  On this substrate e_gaba == 0 in the engine's V_L=0
    # coordinates, so GABA is a pure shunt: z scales the inhibitory CONDUCTANCE and therefore shows up
    # in g_rel, while g_rev (which multiplies gI by e_gaba) is insensitive to it by construction.
    n = int(child.slow.N)
    I_E = np.full(n, 1.7); I_I = np.full(n, 1.3); I_E_rec = np.full(n, 0.9)
    _d0, rel0, _rev0 = child.slow.membrane_terms(I_E, I_I, np.asarray(net["labels"]), I_E_rec=I_E_rec)
    weaker = copy.deepcopy(child)
    weaker.slow.z[:ne] = z0 * 0.5
    _d1, rel1, _rev1 = weaker.slow.membrane_terms(I_E, I_I, np.asarray(net["labels"]), I_E_rec=I_E_rec)
    assert not np.allclose(rel0[:ne], rel1[:ne])
    assert np.all(rel1[:ne] < rel0[:ne])          # less inhibitory efficacy -> less shunt


def test_both_fields_constant_under_dh_clamp():
    p, net, state, vth = _warm()
    child, record = apply_slow_clamp(state, clamp_d=True, clamp_h=True)
    states = _step_states(p, net, child, n_steps=20, vth=vth)
    constancy = slow_field_constancy([child, *states])
    assert constancy["z"]["bitwise_constant"] is True
    assert constancy["h_lc2_E"]["bitwise_constant"] is True
    assert set(record["frozen_field_sha256"]) == {"z", "h_lc2_E"}


def test_unclamped_arm_lets_both_fields_move():
    """Guards the constancy tests: the same window must move both fields when nothing is clamped."""
    p, net, state, vth = _warm()
    child, _record = apply_slow_clamp(state, clamp_d=False, clamp_h=False)
    states = _step_states(p, net, child, n_steps=25, vth=vth)
    constancy = slow_field_constancy([child, *states])
    assert constancy["z"]["bitwise_constant"] is False
    assert constancy["h_lc2_E"]["bitwise_constant"] is False


# ------------------------------------------------------------------ TDD-4 validation

def test_frozen_h_field_validation():
    # The config is a plain dataclass; every check lives in MZSlowVars.__init__/_validate_config, so a
    # test that only builds the dataclass would pass while the guard was missing.
    p = Params(L=4.0, density=80.0, T=120.0, dt=0.1, seed=3)
    net = build_network(p, verbose=False)
    ne, n = net["NE"], net["NE"] + net["NI"]

    def _build(**over):
        return MZSlowVars(n, 18.0, _cfg(**over), NE=ne)

    with pytest.raises(ValueError, match="require use_h_lc2=True"):
        _build(use_h_lc2=False, rho_h_lc2=0.0, h_lc2_frozen_E=np.zeros(ne))
    with pytest.raises(ValueError, match="h_lc2_frozen_E must be a finite 1-D field"):
        _build(h_lc2_frozen_E=np.full(ne, np.nan))
    with pytest.raises(ValueError, match="h_lc2_frozen_E must be a finite 1-D field"):
        _build(h_lc2_frozen_E=np.full(ne, -1.0))
    with pytest.raises(ValueError, match="h_lc2_frozen_E must be a finite 1-D field"):
        _build(h_lc2_frozen_E=np.zeros((2, ne)))


def test_frozen_h_field_shape_is_checked_against_ne():
    p = Params(L=4.0, density=80.0, T=120.0, dt=0.1, seed=3)
    net = build_network(p, verbose=False)
    ne, n = net["NE"], net["NE"] + net["NI"]
    with pytest.raises(ValueError, match="h_lc2_frozen_E must have length NE"):
        MZSlowVars(n, 18.0, _cfg(h_lc2_frozen_E=np.zeros(ne + 1)), NE=ne)
    slow = MZSlowVars(n, 18.0, _cfg(h_lc2_frozen_E=np.full(ne, 0.75)), NE=ne)
    assert np.array_equal(slow.h_lc2_E, np.full(ne, 0.75))


# ------------------------------------------------------------------ TDD-7/8/10 helper contracts

def test_nat_arm_changes_nothing():
    _p, _net, state, _vth = _warm()
    before = state_hash(state)
    child, record = apply_slow_clamp(state, clamp_d=False, clamp_h=False)
    assert state_hash(child) == before
    assert child.slow.cfg.use_z is state.slow.cfg.use_z
    assert child.slow.cfg.z_frozen_E is None and state.slow.cfg.z_frozen_E is None
    assert child.slow.cfg.h_lc2_frozen_E is None
    assert record["frozen_field_sha256"] == {}


def test_clamp_does_not_write_through_to_the_source_state():
    _p, _net, state, _vth = _warm()
    before = state_hash(state)
    child, _record = apply_slow_clamp(state, clamp_d=True, clamp_h=True)
    child.slow.z[:] = 0.123
    child.slow.h_lc2_E[:] = 9.0
    child.slow.cfg.use_z = True
    assert state_hash(state) == before
    assert state.slow.cfg.use_z is True and state.slow.cfg.z_frozen_E is None
    assert state.slow.cfg.h_lc2_frozen_E is None


def test_config_difference_is_not_reported_as_state_difference():
    """Arms differing only by clamp config share a future input but must not share a state hash."""
    p, net, state, vth = _warm()
    hashes, inputs = {}, {}
    for arm, (clamp_d, clamp_h) in ARMS.items():
        child, record = apply_slow_clamp(state, clamp_d=clamp_d, clamp_h=clamp_h)
        seen = []
        net["rng"] = np.random.default_rng(4242)
        out = run_fcxr_loop(p, net, start=child, n_steps=40, capture_final=True,
                            store_spikes=False, v_th_per_neuron=vth,
                            input_sink=lambda t, xi, ext: seen.append((t, float(xi), ext.sum())))
        hashes[arm] = state_hash(out["checkpoint"])
        inputs[arm] = seen
        assert record["clamp_config_sha256"]
    # identical future external input across arms (the checkpoint restores the generator state) ...
    for arm in ARMS:
        assert inputs[arm] == inputs["NAT"], arm
    # ... while the clamps genuinely change the trajectory
    assert len(set(hashes.values())) == len(ARMS), hashes


def test_clamp_config_digest_separates_the_four_arms():
    _p, _net, state, _vth = _warm()
    digests = {
        arm: apply_slow_clamp(state, clamp_d=d, clamp_h=h)[1]["clamp_config_sha256"]
        for arm, (d, h) in ARMS.items()
    }
    assert len(set(digests.values())) == 4


# ------------------------------------------------------------------ TDD-9 save/load round trip

def test_clamped_continuation_round_trips_through_an_exact_checkpoint(tmp_path):
    from src.topic4_fcxr_lc3_statefork import load_into, save_loop_state

    p, net, state, vth = _warm()
    child, _record = apply_slow_clamp(state, clamp_d=True, clamp_h=True)
    out = run_fcxr_loop(p, net, start=child, n_steps=30, capture_final=True,
                        store_spikes=False, v_th_per_neuron=vth)
    final = out["checkpoint"]
    path = tmp_path / "clamped.npz"
    written = save_loop_state(str(path), final)
    assert written == state_hash(final)
    reloaded = load_into(str(path), child)
    assert state_hash(reloaded) == state_hash(final)
    assert reloaded.slow.h_lc2_E.tobytes() == final.slow.h_lc2_E.tobytes()


# ------------------------------------------------------------------ TDD-11/12 classifier

def _cells(second_means, n_cells=64, *, ceiling_fraction=0.0):
    """Per-cell rates whose per-second mean matches ``second_means``."""
    rows = []
    for mean in second_means:
        row = np.full(n_cells, float(mean))
        if ceiling_fraction:
            k = max(1, int(round(ceiling_fraction * n_cells)))
            row[:k] = 460.0
        rows.append(row)
    return np.asarray(rows, float)


def _rate(values_per_second, bins_per_second=50):
    return np.repeat(np.asarray(values_per_second, float), bins_per_second)


def test_classifier_refuses_slow_variable_traces_by_signature():
    """LC6A's classifier reads D/H slopes; under a clamp those are constant by construction."""
    names = set(inspect.signature(classify_clamp_window).parameters)
    assert not any("d_trace" in n or "h_trace" in n or n.startswith(("d_", "h_")) for n in names)
    assert {"rate_bins_hz", "cell_rates_hz"} <= names


def test_classifier_covers_every_registered_label():
    flat = [60.0] * 6
    seen = {}

    seen["NUMERICAL_FAIL"] = classify_clamp_window(
        rate_bins_hz=_rate(flat), cell_rates_hz=_cells(flat),
        completed_ms=6000.0, registered_ms=6000.0, numerical_fail=True)
    seen["RIGHT_CENSORED_INCOMPLETE"] = classify_clamp_window(
        rate_bins_hz=_rate(flat[:3]), cell_rates_hz=_cells(flat[:3]),
        completed_ms=3000.0, registered_ms=6000.0)
    seen["ESCALATING_SATURATION"] = classify_clamp_window(
        rate_bins_hz=_rate([40.0, 80.0, 130.0, 190.0, 240.0, 300.0]),
        cell_rates_hz=_cells([40.0, 80.0, 130.0, 190.0, 240.0, 300.0]),
        completed_ms=6000.0, registered_ms=6000.0)
    seen["ESCALATING_SATURATION_LOCAL"] = classify_clamp_window(
        rate_bins_hz=_rate(flat),
        cell_rates_hz=_cells(flat, ceiling_fraction=0.08),
        completed_ms=6000.0, registered_ms=6000.0)
    seen["SILENCE"] = classify_clamp_window(
        rate_bins_hz=_rate([60.0, 40.0, 10.0, 0.0, 0.0, 0.0]),
        cell_rates_hz=_cells([60.0, 40.0, 10.0, 0.0, 0.0, 0.0]),
        completed_ms=6000.0, registered_ms=6000.0)
    seen["AFTER_DISCHARGE"] = classify_clamp_window(
        rate_bins_hz=_rate([60.0, 3.0, 3.0, 3.0, 3.0, 3.0]),
        cell_rates_hz=_cells([60.0, 3.0, 3.0, 3.0, 3.0, 3.0]),
        completed_ms=6000.0, registered_ms=6000.0)
    seen["LOW_STATE"] = classify_clamp_window(
        rate_bins_hz=_rate([60.0, 60.0, 60.0, 3.0, 3.0, 3.0]),
        cell_rates_hz=_cells([60.0, 60.0, 60.0, 3.0, 3.0, 3.0]),
        completed_ms=6000.0, registered_ms=6000.0)
    seen["RIGHT_CENSORED_ESCALATING"] = classify_clamp_window(
        rate_bins_hz=_rate([40.0, 50.0, 60.0, 70.0, 90.0, 120.0]),
        cell_rates_hz=_cells([40.0, 50.0, 60.0, 70.0, 90.0, 120.0]),
        completed_ms=6000.0, registered_ms=6000.0)
    seen["BOUNDED_STATIONARY"] = classify_clamp_window(
        rate_bins_hz=_rate(flat), cell_rates_hz=_cells(flat),
        completed_ms=6000.0, registered_ms=6000.0)
    bursty = np.tile(np.concatenate([np.full(15, 240.0), np.full(35, 1.0)]), 6)
    seen["BOUNDED_OSCILLATORY"] = classify_clamp_window(
        rate_bins_hz=bursty, cell_rates_hz=_cells([72.7] * 6),
        completed_ms=6000.0, registered_ms=6000.0)

    assert seen["NUMERICAL_FAIL"]["label"] == "NUMERICAL_FAIL"
    assert seen["RIGHT_CENSORED_INCOMPLETE"]["label"] == "RIGHT_CENSORED"
    assert seen["RIGHT_CENSORED_INCOMPLETE"]["reason"] == "INCOMPLETE_REGISTERED_WINDOW"
    assert seen["ESCALATING_SATURATION"]["label"] == "ESCALATING_SATURATION"
    assert seen["ESCALATING_SATURATION_LOCAL"]["label"] == "ESCALATING_SATURATION"
    assert seen["ESCALATING_SATURATION_LOCAL"]["local_saturated"] is True
    assert seen["ESCALATING_SATURATION_LOCAL"]["global_saturated"] is False
    assert seen["SILENCE"]["label"] == "SILENCE"
    assert seen["AFTER_DISCHARGE"]["label"] == "AFTER_DISCHARGE"
    assert seen["LOW_STATE"]["label"] == "LOW_STATE"
    assert seen["RIGHT_CENSORED_ESCALATING"]["label"] == "RIGHT_CENSORED"
    assert seen["RIGHT_CENSORED_ESCALATING"]["reason"] == "STILL_ESCALATING_AT_WINDOW_END"
    assert seen["BOUNDED_STATIONARY"]["label"] == "BOUNDED_STATIONARY"
    assert seen["BOUNDED_OSCILLATORY"]["label"] == "BOUNDED_OSCILLATORY"
    assert {row["label"] for row in seen.values()} == {
        "NUMERICAL_FAIL", "RIGHT_CENSORED", "ESCALATING_SATURATION", "SILENCE",
        "AFTER_DISCHARGE", "LOW_STATE", "BOUNDED_STATIONARY", "BOUNDED_OSCILLATORY",
    }


def test_reignition_train_is_not_called_stationary():
    """LC3's 'unstoppable seizure' was a train re-igniting from full silence every 86 ms."""
    burst = np.concatenate([np.full(2, 300.0), np.full(3, 0.0)])          # 40 ms on / 60 ms off
    rate = np.tile(burst, 60)
    row = classify_clamp_window(
        rate_bins_hz=rate, cell_rates_hz=_cells([120.0] * 6),
        completed_ms=6000.0, registered_ms=6000.0)
    assert row["label"] == "BOUNDED_OSCILLATORY"
    assert row["silence_bin_fraction_tail"] >= 0.25
    assert row["longest_sub_band_run_ms_tail"] >= 60.0
    assert row["bounded_candidate"] is True


def test_bounded_labels_never_claim_a_perturbation_return():
    row = classify_clamp_window(
        rate_bins_hz=_rate([60.0] * 6), cell_rates_hz=_cells([60.0] * 6),
        completed_ms=6000.0, registered_ms=6000.0)
    assert row["bounded_candidate"] is True
    assert row["perturbation_return_tested"] is False


def test_saturation_is_read_before_the_still_escalating_censor():
    rising_into_saturation = [40.0, 90.0, 150.0, 210.0, 260.0, 320.0]
    row = classify_clamp_window(
        rate_bins_hz=_rate(rising_into_saturation),
        cell_rates_hz=_cells(rising_into_saturation),
        completed_ms=6000.0, registered_ms=6000.0)
    assert row["label"] == "ESCALATING_SATURATION"
    assert row["rate_drift_ok"] is False


def test_recovering_low_state_is_reported_not_censored():
    """A slowly recovering low state has a positive slope but is a resolved outcome, not a censor."""
    recovering = [60.0, 60.0, 60.0, 1.0, 2.0, 4.0]
    row = classify_clamp_window(
        rate_bins_hz=_rate(recovering), cell_rates_hz=_cells(recovering),
        completed_ms=6000.0, registered_ms=6000.0)
    assert row["label"] == "LOW_STATE"


def test_cell_rate_distribution_reports_every_registered_band():
    cells = _cells([60.0] * 3, ceiling_fraction=0.10)
    out = cell_rate_distribution(cells)
    assert set(out["fraction_above_hz"]) == {"250", "300", "350", "400", "450"}
    assert set(out["quantiles_hz"]) == {"q50", "q75", "q90", "q95", "q99"}
    assert out["near_refractory_fraction"][0] == pytest.approx(0.10, abs=0.02)
    assert out["fraction_above_hz"]["450"][0] == pytest.approx(0.10, abs=0.02)
    assert out["fraction_above_hz"]["250"][0] == pytest.approx(0.10, abs=0.02)
