"""Observation-wrapper tests; no full SNN is constructed here."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy import sparse

from src.topic4_zm_fast_carrier_runtime import (
    build_dual_scale_i2e_gaba,
    build_pv_som_inhibitory_subtypes,
    DiagnosticSlowWrapper,
    FrozenAllNoStepWrapper,
    rescale_i2e_delay_bins,
)


class _CurrentInner:
    def __init__(self):
        self.nE = 2
        self.S_G = 0.5
        self.H = 0.0
        self.z = np.array([0.5, 1.0, 1.0])
        self.cfg = SimpleNamespace(
            use_SG=True,
            alpha_G=2.0,
            use_H=False,
            alpha_H=0.0,
            beta_SG=0.0,
            use_z=True,
            cond_tau_m_E=20.0,
        )

    def apply_currents(self, I_E, I_I, labels=None, I_E_rec=None):
        frac = 0.5
        out = np.asarray(I_E, float) - np.asarray(I_I, float)
        out[:2] = I_E[:2] - I_E_rec[:2] * frac - self.z[:2] * I_I[:2]
        return out


def test_current_diagnostic_delegates_and_separates_effective_charge():
    wrapped = DiagnosticSlowWrapper(_CurrentInner())
    I_E = np.array([6.0, 8.0, 4.0])
    I_I = np.array([2.0, 4.0, 1.0])
    I_rec = np.array([2.0, 2.0, 1.0])
    expected = wrapped.inner.apply_currents(I_E, I_I, None, I_rec)
    got = wrapped.apply_currents(I_E, I_I, None, I_rec)
    np.testing.assert_array_equal(got, expected)
    summary = wrapped.diagnostic_summary()
    assert summary["median_vinf_mv"] == np.median(expected[:2])
    expected_ratio = np.mean([1.0, 4.0]) / np.mean([5.0, 7.0])
    assert summary["effective_inhibitory_to_excitatory_charge_ratio"] == expected_ratio


def test_wrapper_forwards_state_writes_to_inner():
    wrapped = DiagnosticSlowWrapper(_CurrentInner())
    wrapped.S_G = 0.25
    assert wrapped.inner.S_G == 0.25


def test_zero_copy_freeze_all_skips_step_but_keeps_reads_live():
    inner = _CurrentInner()
    inner.cfg.use_phi = False
    calls = []
    inner.step = lambda *args: calls.append(args)
    diagnostic = DiagnosticSlowWrapper(inner)
    frozen = FrozenAllNoStepWrapper(diagnostic)
    frozen.step(np.array([True]), np.array([0]), 0.1)
    assert calls == []
    got = frozen.apply_currents(
        np.array([6.0, 8.0, 4.0]),
        np.array([2.0, 4.0, 1.0]),
        None,
        np.array([2.0, 2.0, 1.0]),
    )
    assert got.shape == (3,)


def test_zero_copy_freeze_refuses_dynamic_phi():
    inner = _CurrentInner()
    inner.cfg.use_phi = True
    with np.testing.assert_raises_regex(ValueError, "dynamic threshold"):
        FrozenAllNoStepWrapper(inner)


def test_dynamic_diagnostic_step_delegates_and_advances_counter():
    inner = _CurrentInner()
    inner.cfg.use_phi = False
    calls = []
    inner.step = lambda *args: calls.append(args)
    diagnostic = DiagnosticSlowWrapper(inner)
    diagnostic.step(np.array([True]), np.array([0]), 0.1)
    assert len(calls) == 1
    assert diagnostic._step_index == 1


def test_i2e_delay_rescaling_moves_only_e_targets_and_preserves_inflight_offsets():
    zero_a = sparse.csc_matrix((3, 2))
    zero_g = sparse.csc_matrix((3, 1))
    gaba_d1 = sparse.csc_matrix(([2.0, 5.0], ([0, 2], [0, 0])), shape=(3, 1))
    net = {
        "ampa_by_delay": [zero_a.copy() for _ in range(3)],
        "gaba_by_delay": [zero_g.copy(), gaba_d1, zero_g.copy()],
        "max_delay_steps": 2,
    }
    old_ring = np.arange(9.0).reshape(3, 3)
    state = {"t": np.asarray(1), "ring_sE": old_ring, "ring_sI": old_ring + 10}
    new_net, new_state, receipt = rescale_i2e_delay_bins(
        net, state, n_e=2, scale=3.0
    )
    assert new_net["max_delay_steps"] == 3
    assert new_net["gaba_by_delay"][1][2, 0] == 5.0  # I target unchanged
    assert new_net["gaba_by_delay"][1][0, 0] == 0.0
    assert new_net["gaba_by_delay"][3][0, 0] == 2.0  # E target delayed
    np.testing.assert_array_equal(new_state["ring_sE"][1], old_ring[1])
    np.testing.assert_array_equal(new_state["ring_sE"][2], old_ring[2])
    np.testing.assert_array_equal(new_state["ring_sE"][3], old_ring[0])
    assert receipt["edges_unchanged"]


def _delay_test_net():
    zero_a = sparse.csc_matrix((4, 3))
    zero_g = sparse.csc_matrix((4, 3))
    gaba_d2 = sparse.csc_matrix(
        (
            [1.0, 2.0, 3.0, 5.0],
            ([0, 1, 0, 3], [0, 1, 2, 2]),
        ),
        shape=(4, 3),
    )
    return {
        "ampa_by_delay": [zero_a.copy() for _ in range(4)],
        "gaba_by_delay": [zero_g.copy(), zero_g.copy(), gaba_d2, zero_g.copy()],
        "max_delay_steps": 3,
    }


def test_i2e_source_delay_dispersion_is_deterministic_and_weight_preserving():
    net = _delay_test_net()
    old_ring = np.arange(16.0).reshape(4, 4)
    state = {"t": np.asarray(2), "ring_sE": old_ring, "ring_sI": old_ring + 20}
    a_net, a_state, a_receipt = rescale_i2e_delay_bins(
        net, state, n_e=3, scale=3.0, source_delay_cv=0.5, source_delay_seed=7
    )
    b_net, b_state, b_receipt = rescale_i2e_delay_bins(
        net, state, n_e=3, scale=3.0, source_delay_cv=0.5, source_delay_seed=7
    )
    assert a_receipt == b_receipt
    assert len(a_receipt["occupied_i2e_delay_bins"]) >= 2
    for ampa_old, ampa_new in zip(net["ampa_by_delay"], a_net["ampa_by_delay"][:4]):
        np.testing.assert_array_equal(ampa_old.toarray(), ampa_new.toarray())
    old_gaba = sum(net["gaba_by_delay"])
    new_gaba = sum(a_net["gaba_by_delay"])
    np.testing.assert_allclose(old_gaba.toarray(), new_gaba.toarray())
    for x, y in zip(a_net["gaba_by_delay"], b_net["gaba_by_delay"]):
        np.testing.assert_array_equal(x.toarray(), y.toarray())
    np.testing.assert_array_equal(a_state["ring_sI"], b_state["ring_sI"])


def test_i2e_source_delay_cv_zero_retains_uniform_rescaling():
    net = _delay_test_net()
    state = {
        "t": np.asarray(0),
        "ring_sE": np.zeros((4, 4)),
        "ring_sI": np.zeros((4, 4)),
    }
    new_net, _, receipt = rescale_i2e_delay_bins(
        net, state, n_e=3, scale=3.0, source_delay_cv=0.0, source_delay_seed=999
    )
    assert receipt["occupied_i2e_delay_bins"] == [6]
    assert new_net["gaba_by_delay"][2][3, 2] == 5.0
    assert new_net["gaba_by_delay"][6][0, 0] == 1.0
    assert new_net["gaba_by_delay"][6][1, 1] == 2.0
    assert new_net["gaba_by_delay"][6][0, 2] == 3.0


def test_i2e_source_delay_dispersion_rejects_invalid_cv():
    net = _delay_test_net()
    state = {
        "t": np.asarray(0),
        "ring_sE": np.zeros((4, 4)),
        "ring_sI": np.zeros((4, 4)),
    }
    with np.testing.assert_raises_regex(ValueError, "source delay CV"):
        rescale_i2e_delay_bins(
            net, state, n_e=3, scale=3.0, source_delay_cv=-0.1
        )


def test_dual_scale_gaba_preserves_ee_i2i_and_integrated_i2e_budget():
    zero_a = sparse.csc_matrix((6, 4))
    zero_g = sparse.csc_matrix((6, 2))
    ampa_d1 = sparse.csc_matrix(([7.0], ([0], [1])), shape=(6, 4))
    gaba_d1 = sparse.csc_matrix(
        ([2.0, 4.0, 5.0], ([0, 2, 5], [0, 1, 1])), shape=(6, 2)
    )
    net = {
        "pos": np.array([
            [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0],
            [0.0, 1.0], [3.0, 1.0],
        ]),
        "ampa_by_delay": [zero_a.copy(), ampa_d1, zero_a.copy()],
        "gaba_by_delay": [zero_g.copy(), gaba_d1, zero_g.copy()],
        "max_delay_steps": 2,
    }
    ring = np.arange(18.0).reshape(3, 6)
    state = {"t": np.asarray(1), "ring_sE": ring, "ring_sI": ring + 20}
    new_net, new_state, receipt = build_dual_scale_i2e_gaba(
        net,
        state,
        n_e=4,
        slow_fraction=0.25,
        broad_sigma_mm=2.0,
        broad_in_degree=1,
        broad_candidate_count=2,
        seed=9,
        dt_ms=0.1,
        delay_dt_ms=0.1,
        tau0_ms=0.1,
        v_axon_mm_per_ms=1.0,
        tau_r_fast_ms=1.0,
        tau_r_slow_ms=4.0,
        tau_d_slow_ms=60.0,
    )
    old_a = sum(net["ampa_by_delay"])
    new_a = sum(new_net["ampa_by_delay"])
    np.testing.assert_array_equal(old_a.toarray(), new_a.toarray())
    old_g = sum(net["gaba_by_delay"])
    fast_g = sum(new_net["gaba_by_delay"])
    slow_g = sum(new_net["gaba_slow_by_delay"])
    assert fast_g[5, 1] == old_g[5, 1]  # I->I is exact
    old_budget = np.asarray(old_g[:4, :].sum(axis=1)).ravel()
    matched = (
        np.asarray(fast_g[:4, :].sum(axis=1)).ravel()
        + 4.0 * np.asarray(slow_g[:4, :].sum(axis=1)).ravel()
    )
    np.testing.assert_allclose(matched, old_budget)
    assert receipt["i2e_integrated_budget_max_relative_error"] < 1e-12
    assert receipt["ee_ampa_untouched"] and receipt["i2i_gaba_untouched"]
    # Ring padding may be required, but every pre-existing arrival offset is exact.
    t_abs = int(state["t"])
    for delta in range(3):
        np.testing.assert_array_equal(
            new_state["ring_sI"][(t_abs + delta) % new_state["ring_sI"].shape[0]],
            state["ring_sI"][(t_abs + delta) % 3],
        )


def test_dual_scale_gaba_rejects_invalid_budget_fraction():
    net = _delay_test_net()
    net["pos"] = np.arange(8.0).reshape(4, 2)
    state = {
        "t": np.asarray(0),
        "ring_sE": np.zeros((4, 4)),
        "ring_sI": np.zeros((4, 4)),
    }
    with np.testing.assert_raises_regex(ValueError, "slow_fraction"):
        build_dual_scale_i2e_gaba(
            net, state, n_e=3, slow_fraction=1.0, broad_sigma_mm=1.0,
            broad_in_degree=1, broad_candidate_count=1, seed=0, dt_ms=0.1,
            delay_dt_ms=0.1, tau0_ms=0.1, v_axon_mm_per_ms=1.0,
            tau_r_fast_ms=1.0, tau_r_slow_ms=4.0, tau_d_slow_ms=60.0,
        )


def test_pv_som_subtypes_preserve_ee_i2i_and_total_synaptic_budgets():
    n, ne, ni = 6, 4, 2
    zero_a = sparse.csc_matrix((n, ne))
    zero_g = sparse.csc_matrix((n, ni))
    ampa_d1 = sparse.csc_matrix(
        ([7.0, 3.0, 4.0], ([0, 4, 5], [1, 0, 1])), shape=(n, ne)
    )
    g_rows, g_cols, g_data = [], [], []
    for target in range(ne):
        for source in range(ni):
            g_rows.append(target); g_cols.append(source); g_data.append(2.0 + source)
    g_rows.extend([4, 5]); g_cols.extend([0, 1]); g_data.extend([5.0, 6.0])
    gaba_d1 = sparse.csc_matrix((g_data, (g_rows, g_cols)), shape=(n, ni))
    net = {
        "pos": np.array([
            [0., 0.], [1., 0.], [2., 0.], [3., 0.],
            [0., 1.], [3., 1.],
        ]),
        "ampa_by_delay": [zero_a.copy(), ampa_d1, zero_a.copy()],
        "gaba_by_delay": [zero_g.copy(), gaba_d1, zero_g.copy()],
        "max_delay_steps": 2,
    }
    ring = np.arange(18.0).reshape(3, 6)
    state = {"t": np.asarray(1), "ring_sE": ring, "ring_sI": ring + 30}
    out, new_state, receipt = build_pv_som_inhibitory_subtypes(
        net, state, n_e=ne, som_source_fraction=0.5,
        som_slow_budget_fraction=0.35, som_sigma_mm=2.0,
        som_in_degree=1, som_candidate_count=1, som_recruit_delay_scale=3.0,
        seed=4, dt_ms=0.1, delay_dt_ms=0.1, tau0_ms=0.1,
        v_axon_mm_per_ms=1.0, tau_r_fast_ms=1.0,
        tau_r_som_ms=4.0, tau_d_som_ms=60.0,
    )
    old_a = sum(net["ampa_by_delay"])
    new_a = sum(out["ampa_by_delay"])
    np.testing.assert_array_equal(old_a.toarray(), new_a.toarray())
    # The E->E edge remains in its original delay bin; one E->SOM target moves.
    assert out["ampa_by_delay"][1][0, 1] == 7.0
    assert sum(x[4:, :].nnz for x in out["ampa_by_delay"][2:]) > 0
    old_g = sum(net["gaba_by_delay"])
    fast_g = sum(out["gaba_by_delay"])
    slow_g = sum(out["gaba_slow_by_delay"])
    np.testing.assert_array_equal(old_g[4:, :].toarray(), fast_g[4:, :].toarray())
    old_budget = np.asarray(old_g[:ne, :].sum(axis=1)).ravel()
    matched = (
        np.asarray(fast_g[:ne, :].sum(axis=1)).ravel()
        + 4.0 * np.asarray(slow_g[:ne, :].sum(axis=1)).ravel()
    )
    np.testing.assert_allclose(matched, old_budget)
    assert receipt["i2e_integrated_budget_max_relative_error"] < 1e-12
    t_abs = int(state["t"])
    for delta in range(3):
        np.testing.assert_array_equal(
            new_state["ring_sE"][(t_abs + delta) % new_state["ring_sE"].shape[0]],
            state["ring_sE"][(t_abs + delta) % 3],
        )
