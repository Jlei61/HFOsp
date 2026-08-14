import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_topic4_fcxr_lc6a_natural_trajectory.py"
SPEC = importlib.util.spec_from_file_location("lc6a_natural", SCRIPT)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_fresh_config_has_only_ZH_dynamic_and_freezes_X_at_one():
    summary = {"config_scalar": {
        "use_z": True, "use_h_lc2": True, "use_pump": True,
        "pump_Imax": 4.0, "use_m": True, "use_x": True,
    }}
    cfg = RUNNER._fresh_config(summary, 3)
    assert cfg["use_z"] is True and cfg["use_h_lc2"] is True
    assert cfg["use_pump"] is False and cfg["pump_Imax"] == 0.0
    assert cfg["use_m"] is False and cfg["use_x"] is True
    assert cfg["x_relay_frozen_E"].tolist() == [1.0, 1.0, 1.0]


def test_manifest_and_runner_lock_no_kick_event_aligned_contract():
    path, manifest, source = RUNNER._validate_manifest(
        ROOT / "config/topic4_fcxr_lc6a_patient_axis_surround.json", "Q2"
    )
    assert path.is_file() and source.is_file()
    assert manifest["model"]["kick"] is False
    assert manifest["observation"]["post_onset_ms"] == 12000.0
    assert manifest["observation"]["hard_cap_ms"] == 65000.0
    text = SCRIPT.read_text()
    assert "run_fcxr_loop" in text
    assert "simulate_kick" not in text
    assert "spatial_readouts.npz" in text
    assert "membrane_term_sink=current_observer.sample" in text
    assert '"current_decomposition"' in text


def test_trace_chunk_converts_z_to_D():
    class Slow:
        trace_z_mean = [1.0, .8]
        trace_h_lc2_mean = [0.0, .2]
        trace_gA_raw_lc2_mean = [0.0, .1]
        trace_gErec_mean = [0.0, .3]
        trace_conductance_clip_frac = [0.0, 0.0]
    starts = {name: 0 for name in RUNNER.TRACE_ATTRS.values()}
    trace = RUNNER._trace_chunk(Slow(), starts, 1)
    assert np.allclose(trace["D_mean"], [0.0, .2])


def test_non_c0_arm_requires_completed_c0_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(RUNNER, "OUT", tmp_path)
    try:
        RUNNER._load_c0_ied_reference("Q1")
    except RuntimeError as exc:
        assert "C0 natural trajectory" in str(exc)
    else:
        raise AssertionError("Q arms must not invent an IED exposure reference")
    path = tmp_path / "trajectories/C0"
    path.mkdir(parents=True)
    (path / "summary.json").write_text(json.dumps({"n_returning_pre_onset": 12}))
    assert RUNNER._load_c0_ied_reference("Q1") == 12


def test_mechanism_source_hashes_include_unblessed_slow_module():
    hashes = RUNNER._source_hashes()
    assert "src/snn_engine/mz_slow_vars.py" in hashes
    assert "src/topic4_fcxr_lc3.py" in hashes
    assert "src/topic4_fcxr_lc3_statefork.py" in hashes
    assert "src/topic4_fcxr_lc6_trajectory.py" in hashes
    assert all(len(value) == 64 for value in hashes.values())


def test_pinned_checkpoint_records_actual_timing(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(RUNNER.U2, "save_loop_state", lambda path, state: (Path(path).write_bytes(b"x"), calls.append(path)))
    monkeypatch.setattr(RUNNER, "state_hash", lambda state: "state-hash")
    record = RUNNER._pin_checkpoint(
        tmp_path, object(), name="onset_plus_2s", onset_ms=11250.0,
        target_ms=13250.0, actual_ms=14000.0,
    )
    assert record["timing_error_ms"] == 750.0
    assert record["state_hash"] == "state-hash"
    assert Path(calls[0]).name == "checkpoint_onset_plus_2s.npz"
    assert RUNNER._pin_checkpoint(
        tmp_path, object(), name="onset_plus_2s", onset_ms=11250.0,
        target_ms=13250.0, actual_ms=15000.0,
    ) is None


def test_recovery_checkpoint_keeps_current_and_previous(tmp_path, monkeypatch):
    counter = iter((b"first", b"second", b"third"))
    monkeypatch.setattr(
        RUNNER.U2, "save_loop_state",
        lambda path, state: Path(path).write_bytes(next(counter)),
    )
    RUNNER._rotate_recovery_checkpoint(tmp_path, object())
    RUNNER._rotate_recovery_checkpoint(tmp_path, object())
    RUNNER._rotate_recovery_checkpoint(tmp_path, object())
    assert (tmp_path / "rolling_checkpoint.current.npz").read_bytes() == b"third"
    assert (tmp_path / "rolling_checkpoint.previous.npz").read_bytes() == b"second"


def test_non_c0_does_not_claim_control_parity():
    assert RUNNER._c0_control_parity("Q2", object(), object()) == {"required": False}


def test_c0_control_parity_is_exact(monkeypatch):
    stream = RUNNER.SparseSpikeStream(
        np.array([0, 2], np.int64), np.array([0, 1], np.int32), 4, 2,
    )
    rate = np.array([1., 2., 3., 4.])
    monkeypatch.setattr(RUNNER.PREFIX, "RUN_MS", .2)
    monkeypatch.setattr(RUNNER.U2, "DT_MS", .05)
    monkeypatch.setattr(RUNNER.U2, "TRACE_DT_MS", .05)
    monkeypatch.setattr(
        RUNNER.PREFIX, "_reference_prefix",
        lambda end_ms: (stream, rate.astype(np.float32)),
    )
    result = RUNNER._c0_control_parity("C0", stream, rate)
    assert result["spike_exact"] is True
    assert result["rate_max_abs_diff_hz"] == 0.0
