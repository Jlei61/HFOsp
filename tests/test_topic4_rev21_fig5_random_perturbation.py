import hashlib
import json
import sys

import numpy as np
import pytest

from scripts import aggregate_topic4_rev21_fig5_random_perturbation as aggregate


def _write_state(tmp_path, label, *, shift_sites=False, indices=None, suffix=""):
    indices = np.arange(16) if indices is None else np.asarray(indices)
    path = tmp_path / f"{label}{suffix}.npz"
    sites = np.column_stack((indices, indices)).astype(np.float32)
    if shift_sites:
        sites[0, 0] += 0.1
    np.savez(
        path,
        positions_E=np.array([[0.1, 0.2], [0.3, 0.4]], np.float32),
        contact_xy_mm=np.array([[0.2, 0.3]], np.float32),
        site_index=indices.astype(np.int16),
        site_xy_mm=sites,
        excess_per_neuron_early=np.full(
            (len(indices), 2), 1.0 if label == "low_activity" else 2.0,
            np.float32,
        ),
        excess_per_neuron_full=np.zeros((len(indices), 2), np.float32),
        excess_spikes_early=indices.astype(np.float32),
        e1_evaluable=np.ones(len(indices), bool),
    )
    meta = {
        "status": "REV21_FIG5_RANDOM_PERTURBATION_COMPLETE",
        "state_label": label,
        "candidate_id": "rev21_ts_tz3000_ta500",
        "substrate": "dualcore_s39 + Joint=1.25",
        "topology_seed": 2542,
        "dynamics_seed": 2642,
        "checkpoint": {"time_ms": 1000.0 if label == "low_activity" else 2615.4},
        "checkpoint_manifest": {"path": "/frozen/checkpoint_manifest.json",
                                "sha256": "frozen"},
        "site_contract": {"n_total": 16, "seed": 20260820},
        "dose_contract": "frozen baseline dose",
        "dose_cells": 16,
        "window_ms": 200.0,
        "response_window": "paired probe-minus-sham descendant spikes, 0-50 ms",
        "resumed_sham_exact": True,
        "rows": [{"site_index": int(index)} for index in indices],
        "npz": {"path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
    }
    path.with_suffix(".json").write_text(json.dumps(meta))
    return path


def test_state_contrast_retains_identical_sites_and_averages(tmp_path, monkeypatch):
    low = _write_state(tmp_path, "low_activity")
    early = _write_state(tmp_path, "early_ictal")
    out = tmp_path / "contrast.npz"
    monkeypatch.setattr(
        sys, "argv", ["aggregate", "--low", str(low),
                      "--early-ictal", str(early), "--out", str(out)])
    aggregate.main()
    with np.load(out, allow_pickle=False) as arrays:
        assert np.array_equal(arrays["site_index"], np.arange(16))
        assert np.allclose(arrays["low_response_early_mean"], 1.0)
        assert np.allclose(arrays["early_ictal_response_early_mean"], 2.0)
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["all_sites_retained"] is True
    assert meta["state_times_ms"] == {
        "low_activity": 1000.0, "early_ictal": 2615.4}


def test_state_contrast_rejects_different_random_sites(tmp_path, monkeypatch):
    low = _write_state(tmp_path, "low_activity")
    early = _write_state(tmp_path, "early_ictal", shift_sites=True)
    monkeypatch.setattr(
        sys, "argv", ["aggregate", "--low", str(low),
                      "--early-ictal", str(early),
                      "--out", str(tmp_path / "contrast.npz")])
    with pytest.raises(RuntimeError, match="mismatched site_xy_mm"):
        aggregate.main()


def test_state_contrast_joins_disjoint_chunks_in_site_order(tmp_path, monkeypatch):
    low = _write_state(tmp_path, "low_activity")
    early_hi = _write_state(
        tmp_path, "early_ictal", indices=np.arange(8, 16), suffix="_hi")
    early_lo = _write_state(
        tmp_path, "early_ictal", indices=np.arange(0, 8), suffix="_lo")
    out = tmp_path / "contrast.npz"
    monkeypatch.setattr(
        sys, "argv", ["aggregate", "--low", str(low),
                      "--early-ictal", str(early_hi), str(early_lo),
                      "--out", str(out)])
    aggregate.main()
    with np.load(out, allow_pickle=False) as arrays:
        assert np.array_equal(arrays["site_index"], np.arange(16))
        assert np.allclose(arrays["early_ictal_response_early_mean"], 2.0)
