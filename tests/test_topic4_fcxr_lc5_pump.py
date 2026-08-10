"""FCXR-LC5 rev2 contracts for the rectified per-cell episode-load actuator."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from src.snn_engine.mz_slow_vars import MZSlowVars, MZSlowVarsConfig
from src.topic4_fcxr_lc5 import (
    AtomicStageBundle,
    ExactInputHasher,
    RecurrentDriveBlockObserver,
    SparseSpikeBinaryWriter,
    SparseSpikeStream,
    json_sanitize,
    load_sparse_spike_stream,
    lock_load_scales,
    replay_sparse_loads,
)
from src.topic4_mz_fcxr_pump import excess_pump_current, pump_activation


def _slow(*, mode="rectified_excess", p0=0.4, u0=0.2, imax=2.0):
    ne, n = 4, 6
    cfg = MZSlowVarsConfig(
        membrane_mode="full_conductance",
        E_E=58.0,
        c_E=1.0,
        v_match=18.0,
        e_gaba=0.0,
        e_k=0.0,
        ff_conductance=False,
        rec_conductance=True,
        rec_sat_g=21.6,
        max_total_conductance=99.0,
        use_pump=True,
        pump_a_load=0.5,
        pump_tau_ms=3000.0,
        pump_Imax=imax,
        pump_p0_E=np.full(ne, p0),
        pump_u_init_E=np.full(ne, u0),
        pump_excess_mode=mode,
    )
    return MZSlowVars(n, 18.0, cfg, NE=ne, core_mask_E=np.zeros(ne, bool))


def _inputs():
    n, ne = 6, 4
    i_e = np.linspace(1.0, 3.0, n)
    i_i = np.linspace(0.5, 1.5, n)
    return i_e, i_i, 0.6 * i_e, ne


def test_rectified_helper_is_zero_below_p0_and_positive_above():
    p0 = np.array([0.4, 0.4])
    u = np.array([0.2, 2.0])
    got = excess_pump_current(u, p0, Imax=3.0, mode="rectified_excess")
    assert got[0] == 0.0
    assert got[1] == pytest.approx(3.0 * (pump_activation(2.0) - 0.4))


def test_historical_signed_helper_remains_negative_below_p0():
    got = excess_pump_current(
        np.array([0.2]), np.array([0.4]), Imax=3.0, mode="signed_centered"
    )
    assert got[0] < 0.0
    assert got[0] == pytest.approx(3.0 * (pump_activation(0.2) - 0.4))


def test_rectified_engine_delivers_no_current_below_p0_and_is_e_only_above():
    i_e, i_i, i_rec, ne = _inputs()
    ref = _slow(mode="rectified_excess", imax=0.0)
    low = _slow(mode="rectified_excess", p0=0.4, u0=0.2)
    d_ref, _, _ = ref.membrane_terms(i_e, i_i, None, I_E_rec=i_rec)
    d_low, _, _ = low.membrane_terms(i_e, i_i, None, I_E_rec=i_rec)
    assert np.array_equal(d_low, d_ref)

    high = _slow(mode="rectified_excess", p0=0.4, u0=2.0)
    d_high, _, _ = high.membrane_terms(i_e, i_i, None, I_E_rec=i_rec)
    expected = 2.0 * (pump_activation(2.0) - 0.4)
    assert np.allclose(d_ref[:ne] - d_high[:ne], expected)
    assert np.array_equal(d_ref[ne:], d_high[ne:])


def test_signed_engine_keeps_historical_negative_compensation():
    i_e, i_i, i_rec, ne = _inputs()
    ref = _slow(mode="signed_centered", imax=0.0)
    signed = _slow(mode="signed_centered", p0=0.4, u0=0.2)
    d_ref, _, _ = ref.membrane_terms(i_e, i_i, None, I_E_rec=i_rec)
    d_signed, _, _ = signed.membrane_terms(i_e, i_i, None, I_E_rec=i_rec)
    assert np.all(d_signed[:ne] > d_ref[:ne])
    assert np.array_equal(d_signed[ne:], d_ref[ne:])


def test_invalid_excess_mode_fails_loudly_even_if_pump_is_off():
    with pytest.raises(ValueError, match="pump_excess_mode"):
        MZSlowVars(
            6,
            18.0,
            MZSlowVarsConfig(pump_excess_mode="not-a-mode"),
            NE=4,
            core_mask_E=np.zeros(4, bool),
        )


def test_sparse_replay_matches_dense_discrete_equation_at_landmarks():
    dense = np.zeros((9, 4), dtype=bool)
    dense[1, [0, 2]] = True
    dense[3, [1]] = True
    dense[6, [0, 3]] = True
    stream = SparseSpikeStream.from_dense(dense)
    replay = replay_sparse_loads(
        stream,
        candidates={"t3": {"a_load": 0.4, "tau_ms": 3000.0}},
        dt_ms=0.05,
        snapshot_steps={2: "early", 8: "late"},
        blocks={"baseline": (0, 5), "high": (5, 9)},
    )["t3"]

    u = np.zeros(4)
    expected = {}
    phi_sum = {"baseline": np.zeros(4), "high": np.zeros(4)}
    counts = {"baseline": 0, "high": 0}
    for step, spk in enumerate(dense):
        phi = pump_activation(u)
        block = "baseline" if step < 5 else "high"
        phi_sum[block] += phi
        counts[block] += 1
        u = np.maximum(u + 0.4 * spk - (0.05 / 3000.0) * phi, 0.0)
        if step in (2, 8):
            expected["early" if step == 2 else "late"] = u.copy()

    assert np.array_equal(replay["snapshots"]["early"], expected["early"])
    assert np.array_equal(replay["snapshots"]["late"], expected["late"])
    assert np.array_equal(replay["u_final"], u)
    for block in counts:
        assert np.allclose(replay["block_phi_mean"][block], phi_sum[block] / counts[block])


def test_sparse_active_fraction_matches_dense_any_within_bin():
    dense = np.zeros((11, 4), dtype=bool)
    dense[[0, 1, 4, 5, 9], [0, 0, 2, 3, 1]] = True
    stream = SparseSpikeStream.from_dense(dense)
    got, got_dt = stream.active_fraction(dt_ms=0.5, bin_ms=2.0)
    expected = dense[:8].reshape(2, 4, 4).any(axis=1).mean(axis=1)
    assert got_dt == 2.0
    assert np.array_equal(got, expected)


def test_sparse_binary_writer_roundtrip_and_rates(tmp_path):
    raw = tmp_path / "spikes.bin"
    out = tmp_path / "spikes.npz"
    writer = SparseSpikeBinaryWriter(raw, step_origin=10, n_steps=5, n_cells=3)
    writer(10, np.asarray([0, 2]))
    writer(12, np.asarray([1]))
    written = writer.finalize(out)
    loaded = load_sparse_spike_stream(out)
    assert loaded.sha256 == written.sha256
    assert np.array_equal(loaded.steps, [0, 0, 2])
    assert np.allclose(
        loaded.per_cell_rate_hz(lo_step=0, hi_step=5, dt_ms=1.0), [200, 200, 200]
    )


def test_recurrent_drive_observer_preserves_per_cell_block_support():
    obs = RecurrentDriveBlockObserver(2, sample_every=2, steps_per_block=4, force_scale=40.0)
    for step in range(8):
        obs.sample(np.array([step, step + 1.0]), np.array([0.5 * step, 1.0]), step)
    got = obs.arrays()
    assert np.array_equal(got["block_index"], [0, 1])
    assert np.allclose(got["raw_conductance_mean"], [[1, 2], [5, 6]])
    assert np.allclose(got["effective_force_mean"], [[20, 40], [100, 40]])


def test_exact_input_hasher_is_order_and_value_sensitive():
    a, b, c = ExactInputHasher(), ExactInputHasher(), ExactInputHasher()
    for h in (a, b):
        h(0, 0.25, np.array([0.0, 1.0]))
        h(1, 0.5, np.array([1.0, 0.0]))
    c(0, 0.25, np.array([0.0, 1.0]))
    c(1, 0.5, np.array([0.0, 1.0]))
    assert a.sha256 == b.sha256
    assert a.sha256 != c.sha256
    assert a.n_steps == 2


def test_load_scale_lock_is_one_common_rate_distribution_gate():
    rates = np.array([20.0, 40.0, 60.0, 80.0])
    lock = lock_load_scales(
        r_hi_ref_hz=50.0,
        per_cell_rate_hz=rates,
        tau_ms=(3000.0, 8000.0, 15000.0),
        target_activation=0.5,
    )
    assert lock["admissible"] is True
    assert np.allclose(lock["q_star"], 0.5 * rates / 50.0)
    assert lock["q_star_sha256"]
    for tau in (3000.0, 8000.0, 15000.0):
        assert lock["a_load_by_tau_ms"][str(tau)] == pytest.approx(
            0.5 / ((50.0 / 1000.0) * tau)
        )


def test_load_scale_lock_rejects_any_cell_without_finite_equilibrium():
    lock = lock_load_scales(
        r_hi_ref_hz=50.0,
        per_cell_rate_hz=np.array([10.0, 100.0]),
        tau_ms=(3000.0,),
        target_activation=0.5,
    )
    assert lock["admissible"] is False
    assert lock["divergent_fraction"] == pytest.approx(0.5)


def test_json_sanitize_handles_numpy_path_and_small_arrays(tmp_path):
    got = json_sanitize(
        {
            "ok": np.bool_(True),
            "x": np.float32(1.25),
            "n": np.int64(4),
            "path": tmp_path / "x",
            "a": np.array([1, 2]),
        }
    )
    assert got == {"ok": True, "x": 1.25, "n": 4, "path": str(tmp_path / "x"), "a": [1, 2]}


def test_atomic_stage_bundle_never_publishes_an_incomplete_bundle(tmp_path):
    final = tmp_path / "u1_bundle"
    with pytest.raises(FileNotFoundError):
        with AtomicStageBundle(final) as bundle:
            bundle.path("summary.json").write_text("{}")
            bundle.commit(required=("summary.json", "spikes.npz"))
    assert not final.exists()
    assert not list(tmp_path.glob("u1_bundle.tmp-*"))


def test_atomic_stage_bundle_publishes_only_after_all_required_files_exist(tmp_path):
    final = tmp_path / "u1_bundle"
    with AtomicStageBundle(final) as bundle:
        bundle.path("summary.json").write_text("{}")
        bundle.path("states/onset.pkl").write_bytes(b"state")
        bundle.commit(required=("summary.json", "states/onset.pkl"))
    assert (final / "summary.json").read_text() == "{}"
    assert (final / "states/onset.pkl").read_bytes() == b"state"
