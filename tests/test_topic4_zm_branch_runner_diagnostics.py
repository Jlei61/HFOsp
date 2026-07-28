"""Observation-only diagnostics added to the branch runner must preserve time and space axes."""
from __future__ import annotations

import importlib.util
import os

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH = os.path.join(_ROOT, "scripts", "run_topic4_zm_branch_decision.py")
_SPEC = importlib.util.spec_from_file_location("topic4_zm_branch_runner", _PATH)
R = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(R)


def _chunk(offset):
    n = 4
    base = np.arange(n, dtype=float) + offset
    return {
        "n_bins": n,
        "r_core": 100.0 + base,
        "r_surround": 80.0 + base,
        "r_all": 120.0 + 0.01 * base,
        "A_active": 0.4 + 0.0001 * base,
        "E_vSEEG": 10.0 + 0.01 * base,
        "H_spatial": 0.8 + 0.0001 * base,
        "n_grid_active": np.full(n, 128.0),
        "kymo_axial": np.full((24, n), 1.0 + offset),
        "_lfp_raw": np.ones((1000, 2), np.float32) * (1.0 + offset),
        "_slow": {
            "z_core": np.full(n, 0.5),
            "z_surround": np.full(n, 0.6),
            "m_core": np.full(n, 2.0),
            "S_G": np.full(n, 0.7),
        },
    }


def _locks():
    ref = {}
    for k, mu in {
        "r_core": 1.0,
        "r_surround": 1.0,
        "A_active": 0.01,
        "E_vSEEG": 0.1,
        "H_spatial": 0.1,
    }.items():
        ref[k] = {"mu": mu, "sd": max(0.01, 0.1 * mu)}
    ref["n_bins"] = 4
    return {"rest_reference": ref, "d_rest_thresh": 2.0, "rest_dwell_ms": 200.0}


def test_dump_preserves_axial_time_and_native_lfp_rate(tmp_path):
    run = {
        "chunks": [_chunk(0), _chunk(1)],
        "burn_in_ms": 0.0,
        "wall_s": 1.0,
        "end_reason": None,
        "runaway_ms": None,
    }
    path = tmp_path / "trace.npz"
    R.dump_continuation_traces(str(path), run, _locks(), dt_ms=0.1)
    z = np.load(path)
    assert z["kymo_axial"].shape == (24, 8)
    assert z["lfp"].shape == (400, 2)
    assert float(z["lfp_fs"]) == 2000.0


def test_summary_labels_nearly_flat_high_rate_as_tonic_like_fixed():
    run = {
        "chunks": [_chunk(0), _chunk(1)],
        "burn_in_ms": 0.0,
        "wall_s": 1.0,
        "end_reason": None,
        "runaway_ms": None,
    }
    out = R.summarize_continuation(run, _locks(), T_ms=200.0)
    assert out["morphology_label"] == "tonic_at_25ms"
    assert out["r_all_cv"] < 0.05
    assert out["spatial_extent_fraction"] == 0.5


def test_summary_preserves_streaming_basin_termination_over_rest_classifier():
    locks = _locks()
    chunks = [_chunk(0), _chunk(1)]
    for chunk in chunks:
        for key in R.MC.REST_KEYS:
            chunk[key] = np.full(
                chunk["n_bins"], locks["rest_reference"][key]["mu"]
            )
    run = {
        "chunks": chunks,
        "burn_in_ms": 0.0,
        "wall_s": 1.0,
        "end_reason": "dead_in_rest_basin",
        "runaway_ms": None,
    }

    out = R.summarize_continuation(run, locks, T_ms=200.0)

    assert out["run_end_reason"] == "dead_in_rest_basin"
    assert out["classifier_end_reason"] == "rest_return"
    assert out["end_reason"] == "dead_in_rest_basin"


def test_effective_rank_runner_records_unavailable_central_pair_contract():
    source = open(_PATH, encoding="utf-8").read()
    assert "except ER.CentralPairUnavailable as exc:" in source
    assert "central_pair_unavailable:" in source
    assert 'manifest["probe_matrix_complete"]' in source


def test_wait_only_finalizer_cannot_launch_an_snn_worker():
    path = os.path.join(
        _ROOT, "scripts", "finalize_topic4_zm_branch_when_ready.sh"
    )
    source = open(path, encoding="utf-8").read()
    assert "wait_phase effective_rank" in source
    assert "wait_phase entry_boundary" in source
    assert "wait_phase offset_boundary" in source
    assert "assert_all_workers" in source
    assert "python scripts/run_topic4_zm_branch_decision.py \\\n" not in source
    assert "P0:" in source


def test_finalizer_rechecks_current_shard_and_analyzer_gates_before_completion():
    path = os.path.join(
        _ROOT, "scripts", "finalize_topic4_zm_branch_when_ready.sh"
    )
    source = open(path, encoding="utf-8").read()
    required = {
        "tests/test_topic4_zm_entry_shards.py",
        "tests/test_topic4_zm_entry_shard_coordinator.py",
        "tests/test_topic4_zm_offset_shards.py",
        "tests/test_topic4_zm_offset_shard_coordinator.py",
    }
    assert required <= set(source.split())
    assert 'append_gate_test_glob "tests/test_topic4_zm_*boundar*.py"' in source
    assert 'append_gate_test_glob "tests/test_topic4_zm_*analy*.py"' in source
    assert "adjudicate_topic4_zm_branch_decision.py --verify-gates" in source

    fresh_gate = source.index("run_fresh_gate_tests")
    adjudicate = source.rindex(
        "python scripts/adjudicate_topic4_zm_branch_decision.py --verify-gates"
    )
    evidence_check = source.rindex(
        'json_true "$out/phase0/gate_evidence.json" passed'
    )
    plot = source.rindex("python scripts/plot_topic4_zm_branch_decision.py")
    complete = source.rindex('echo "[finalizer] $(date -Is) complete"')
    assert fresh_gate < adjudicate < evidence_check < plot < complete
