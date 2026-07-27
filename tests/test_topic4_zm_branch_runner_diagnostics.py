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
    assert out["morphology_label"] == "tonic_like_fixed"
    assert out["r_all_cv"] < 0.05
    assert out["spatial_extent_fraction"] == 0.5
