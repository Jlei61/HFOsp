from types import SimpleNamespace

import numpy as np
import pytest

from src.topic4_zm_phasec_observation import (
    PhaseCEffectiveSlowObserver,
    PhaseCCurrentRecorder,
    free_e_threshold_margin_snapshot,
    raw_synaptic_lag,
    reconstruct_effective_snapshot,
)


class _DummySlow:
    def __init__(self):
        self.nE = 4
        self.z = np.array([0.5, 0.6, 0.7, 0.8])
        self.m = np.array([1.0, 2.0, 3.0, 4.0])
        self.S_G = 0.1
        self.H = 0.0
        self.cfg = SimpleNamespace(
            alpha_G=2.0,
            alpha_H=0.0,
            use_H=False,
            eta_m=0.1,
            beta_SG=0.0,
        )
        self.calls = 0

    def apply_currents(self, I_E, I_I, labels=None, I_E_rec=None):
        self.calls += 1
        removed = I_E_rec[:4] * (0.2 / 1.2)
        out = np.asarray(I_E, float) - np.asarray(I_I, float)
        out[:4] = I_E[:4] - removed - self.z * I_I[:4] - 0.1 * self.m
        return out

    def threshold(self, value):
        return value


def test_effective_slow_observer_is_transparent_and_checkpoint_writes_delegate():
    inner = _DummySlow()
    obs = PhaseCEffectiveSlowObserver(
        inner, np.array([True, True, False, False]), stride_steps=1
    )
    exc = np.arange(6, dtype=float) + 2
    inh = np.ones(6)
    rec = np.ones(6) * 0.6
    expected = inner.apply_currents(exc, inh, I_E_rec=rec)
    inner.calls = 0
    actual = obs.apply_currents(exc, inh, I_E_rec=rec)
    np.testing.assert_array_equal(actual, expected)
    assert inner.calls == 1
    obs.S_G = 0.25
    assert inner.S_G == 0.25
    traces = obs.traces(dt_ms=0.1)
    assert traces["evidence_label"] == "effective_membrane_drive"
    assert traces["effective_net_drive_core_mean_mV"].shape == (1,)
    assert traces["sample_dt_ms"] == 0.1


class _InnerRecorder:
    def __init__(self):
        self.sites = np.asarray([[0.0, 0.0], [1.0, 1.0]])
        self.NE = 4
        self.n_calls = 0

    def sample(self, I_E, I_I):
        self.n_calls += 1
        # Float64 result with nontrivial signs: the wrapper must return this
        # exact array object, not recompute/cast/copy it.
        return np.asarray([
            I_E[0] + 2.0 * I_I[0],
            I_E[3] - I_I[3],
        ], dtype=np.float64)


def test_wrapped_recorder_is_transparent_and_fixed_stride():
    inner = _InnerRecorder()
    rec = PhaseCCurrentRecorder(
        inner, np.asarray([True, True, False, False]), stride_steps=2
    )
    returned = []
    expected = []
    for step in range(5):
        exc = np.arange(6, dtype=float) + step
        inh = np.arange(6, dtype=float) * 0.5 + 2 * step
        want = np.asarray([exc[0] + 2 * inh[0], exc[3] - inh[3]], dtype=np.float64)
        got = rec.sample(exc, inh)
        returned.append(got)
        expected.append(want)
        assert np.array_equal(got, want)
        assert got.dtype == want.dtype
    assert inner.n_calls == 5
    tr = rec.traces(dt_ms=0.1)
    np.testing.assert_array_equal(tr["sample_step"], [0, 2, 4])
    assert tr["sample_dt_ms"] == pytest.approx(0.2)
    assert tr["evidence_label"] == "raw_synaptic"
    assert tr["raw_ampa_core_mean_mV"][0] == pytest.approx(0.5)
    assert tr["raw_ampa_surround_mean_mV"][0] == pytest.approx(2.5)
    assert tr["raw_gaba_to_ampa_core_ratio"][0] == pytest.approx(0.25 / 0.5)


def test_raw_synaptic_lag_sign_is_gaba_trails_ampa():
    rng = np.random.default_rng(2)
    amp = rng.standard_normal(500)
    gaba = np.zeros_like(amp)
    gaba[3:] = amp[:-3]
    traces = {
        "evidence_label": "raw_synaptic",
        "sample_dt_ms": 0.5,
        "raw_ampa_core_mean_mV": amp,
        "raw_gaba_core_mean_mV": gaba,
    }
    out = raw_synaptic_lag(traces, region="core", max_lag_ms=5.0)
    assert out["status"] == "ok"
    assert out["lag_steps"] == 3
    assert out["lag_ms"] == pytest.approx(1.5)
    assert out["peak_correlation"] > 0.99
    assert "GABA trails" in out["sign_convention"]


def _checkpoint_fixture():
    return {
        "I_E": np.asarray([10.0, 12.0, 14.0, 99.0]),
        "I_I": np.asarray([4.0, 6.0, 8.0, 99.0]),
        "I_E_rec": np.asarray([2.0, 4.0, 6.0, 99.0]),
        "slow.z": np.asarray([0.5, 0.75, 1.0, 1.0]),
        "slow.m": np.asarray([2.0, 3.0, 4.0, 0.0]),
        "slow.S_G": np.asarray(0.25),
        "slow.H": np.asarray(0.5),
        "V": np.asarray([17.5, 10.0, 16.0, 0.0]),
        "ref": np.asarray([0, 2, 0, 0], dtype=np.int32),
    }


def test_effective_snapshot_reconstructs_z_m_sg_identity_and_signs():
    state = _checkpoint_fixture()
    alpha_G, alpha_H, eta_m, beta = 4.0, 2.0, 0.1, 0.2
    load = alpha_G * 0.25 + alpha_H * 0.5
    removed = state["I_E_rec"][:3] * load / (1 + load)
    expected = (
        state["I_E"][:3]
        - state["slow.z"][:3] * state["I_I"][:3]
        - eta_m * state["slow.m"][:3]
        - removed
        - beta * 0.25
    )
    out = reconstruct_effective_snapshot(
        state,
        nE=3,
        alpha_G=alpha_G,
        alpha_H=alpha_H,
        eta_m=eta_m,
        beta_SG=beta,
        expected_net=expected,
    )
    assert out["evidence_label"] == "effective_snapshot"
    np.testing.assert_allclose(out["effective_net_drive_mV"], expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        out["effective_excitation_mV"] - out["effective_outward_total_mV"],
        expected,
        rtol=0,
        atol=1e-12,
    )
    assert np.all(out["recurrent_ampa_removed_by_SG_mV"] > 0)
    assert np.all(out["effective_inhibition_z_mV"] > 0)
    assert out["identity_max_abs_error_mV"] == pytest.approx(0.0)
    assert "transmembrane-current time series" in out["claim_boundary"]


def test_effective_snapshot_fails_closed_on_bad_expected_net_or_missing_state():
    state = _checkpoint_fixture()
    with pytest.raises(AssertionError):
        reconstruct_effective_snapshot(
            state, nE=3, alpha_G=4, eta_m=0.1, expected_net=np.zeros(3)
        )
    broken = dict(state)
    broken.pop("I_E_rec")
    with pytest.raises(KeyError):
        reconstruct_effective_snapshot(broken, nE=3, alpha_G=4, eta_m=0.1)


def test_free_e_threshold_margin_uses_only_nonrefractory_e_cells_and_core_split():
    state = _checkpoint_fixture()
    vth = np.asarray([18.0, 18.0, 17.0, 18.0])
    core = np.asarray([True, True, False])
    out = free_e_threshold_margin_snapshot(
        state, vth, nE=3, core_mask_E=core, quantiles=(50,)
    )
    # E0 margin=.5 and E2 margin=1; E1 is refractory and excluded.
    assert out["all_free_E"]["n"] == 2
    assert out["all_free_E"]["quantiles_mV"]["50.0"] == pytest.approx(0.75)
    assert out["core_free_E"]["n"] == 1
    assert out["core_free_E"]["quantiles_mV"]["50.0"] == pytest.approx(0.5)
    assert out["surround_free_E"]["quantiles_mV"]["50.0"] == pytest.approx(1.0)
    assert out["evidence_label"] == "effective_snapshot"
    assert out["snapshot_quantity"] == "free_E_Vth_minus_V"


def test_observation_labels_never_upgrade_raw_to_effective():
    rec = PhaseCCurrentRecorder(
        _InnerRecorder(), np.asarray([True, True, False, False]), stride_steps=1
    )
    rec.sample(np.ones(4), np.ones(4))
    tr = rec.traces(dt_ms=0.1)
    assert tr["evidence_label"] == "raw_synaptic"
    assert "effective" not in tr["evidence_label"]
    state = _checkpoint_fixture()
    snap = reconstruct_effective_snapshot(state, nE=3, alpha_G=4.0, eta_m=0.1)
    assert snap["evidence_label"] == "effective_snapshot"
