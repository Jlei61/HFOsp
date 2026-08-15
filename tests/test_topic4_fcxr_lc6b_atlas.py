"""FCXR-LC6B atlas contracts: the four review locks, tested where they can be tested cheaply."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "src", "snn_engine")):
    if path not in sys.path:
        sys.path.insert(0, path)

import run_topic4_fcxr_lc6b_atlas as ATLAS  # noqa: E402


# ---------------------------------------------------------------- lock 2

def test_stream_hasher_ignores_the_step_index():
    """The three initialisations legitimately carry different step counters.

    ExactInputHasher folds the absolute step into its digest, so comparing ITS output across
    initialisations would report a violation of lock 2 whenever the draws were in fact identical.
    """
    rng = np.random.default_rng(7)
    xis = rng.standard_normal(50)
    draws = rng.poisson(0.3, size=(50, 16)).astype(float)

    early, late = ATLAS.StreamHasher(), ATLAS.StreamHasher()
    for index, (xi, ext) in enumerate(zip(xis, draws)):
        early(index, xi, ext)
        late(index + 123456, xi, ext)         # same content, different absolute step
    assert early.sha256 == late.sha256
    assert early.n_steps == late.n_steps == 50


def test_stream_hasher_still_sees_a_real_input_difference():
    """Guards the test above: the digest must not be insensitive to the content itself."""
    a, b = ATLAS.StreamHasher(), ATLAS.StreamHasher()
    ext = np.ones(8)
    a(0, 0.5, ext)
    b(0, 0.5, ext * 2)
    assert a.sha256 != b.sha256
    c, d = ATLAS.StreamHasher(), ATLAS.StreamHasher()
    c(0, 0.5, ext)
    d(0, 0.6, ext)
    assert c.sha256 != d.sha256


# ---------------------------------------------------------------- lock 4

def _fake_rows(labels_by_field):
    rows = {}
    for field, labels in labels_by_field.items():
        for init, label in labels.items():
            rows[(field, init)] = {
                "verdict": {"label": label, "per_second_mean_hz": [50.0] * 10},
                "median_active_area_mm2": 200.0,
                "future_input_content_sha256": "same",
            }
    return rows


def test_an_initialisation_split_is_only_ever_a_candidate(monkeypatch, tmp_path):
    """A low/high split must never be written out as demonstrated bistability."""
    fields = {
        "field_t97s": {"path_native": "BOUNDED_OSCILLATORY", "locked_low": "LOW_STATE",
                       "locked_high": "BOUNDED_OSCILLATORY"},
        "field_t98s": {"path_native": "BOUNDED_OSCILLATORY", "locked_low": "BOUNDED_OSCILLATORY",
                       "locked_high": "BOUNDED_OSCILLATORY"},
    }
    index = {"fields": {
        name: {"snapshot_time_ms": 97000.0 + 1000.0 * offset, "relative_to_onset_ms": 1000.0 * offset,
               "D_mean": .1, "H_mean": 1.0, "h_gate_mean": .5, "file": f"{name}.npz", "sha256": "x"}
        for offset, name in enumerate(fields)}}
    rows = _fake_rows(fields)

    monkeypatch.setattr(ATLAS, "_fields_index", lambda: index)
    monkeypatch.setattr(ATLAS, "OUT", tmp_path)
    monkeypatch.setattr(ATLAS.NAT, "_write_json", lambda path, payload: None)
    # Build the bundles on disk so finalize's own reader finds them.
    for (field, init), row in rows.items():
        bundle = tmp_path / f"{field}__{init}"
        bundle.mkdir(parents=True)
        (bundle / "summary.json").write_text(__import__("json").dumps(row))

    payload = ATLAS.finalize()
    assert payload["per_field"]["field_t97s"]["initialisation_split"] is True
    assert payload["per_field"]["field_t97s"]["verdict"] == (
        "BISTABILITY_CANDIDATE_PENDING_PERTURBATION_AND_SECOND_STREAM")
    assert payload["per_field"]["field_t98s"]["initialisation_split"] is False
    assert payload["per_field"]["field_t98s"]["verdict"] == (
        "SINGLE_OUTCOME_FROM_BOTH_LOCKED_INITIALISATIONS")
    assert payload["lock_4_split_handling"].startswith("reported as BISTABILITY_CANDIDATE")
    assert payload["perturbation_return_tested"] is False


def test_finalize_refuses_points_that_do_not_share_the_input_stream(monkeypatch, tmp_path):
    fields = {"field_t97s": {"path_native": "BOUNDED_OSCILLATORY",
                             "locked_low": "BOUNDED_OSCILLATORY",
                             "locked_high": "BOUNDED_OSCILLATORY"}}
    index = {"fields": {"field_t97s": {
        "snapshot_time_ms": 97000.0, "relative_to_onset_ms": 1000.0, "D_mean": .1,
        "H_mean": 1.0, "h_gate_mean": .5, "file": "f.npz", "sha256": "x"}}}
    rows = _fake_rows(fields)
    rows[("field_t97s", "locked_high")]["future_input_content_sha256"] = "different"
    monkeypatch.setattr(ATLAS, "_fields_index", lambda: index)
    monkeypatch.setattr(ATLAS, "OUT", tmp_path)
    for (field, init), row in rows.items():
        bundle = tmp_path / f"{field}__{init}"
        bundle.mkdir(parents=True)
        (bundle / "summary.json").write_text(__import__("json").dumps(row))
    with pytest.raises(RuntimeError, match="lock 2"):
        ATLAS.finalize()


def test_finalize_refuses_an_incomplete_atlas(monkeypatch, tmp_path):
    index = {"fields": {"field_t97s": {
        "snapshot_time_ms": 97000.0, "relative_to_onset_ms": 1000.0, "D_mean": .1,
        "H_mean": 1.0, "h_gate_mean": .5, "file": "f.npz", "sha256": "x"}}}
    monkeypatch.setattr(ATLAS, "_fields_index", lambda: index)
    monkeypatch.setattr(ATLAS, "OUT", tmp_path)
    with pytest.raises(RuntimeError, match="atlas incomplete"):
        ATLAS.finalize()


# ---------------------------------------------------------------- lock 1 / 3

def test_registered_observation_is_ten_seconds():
    assert ATLAS.OBSERVE_MS == 10000.0


def test_the_three_registered_initialisations_are_fixed():
    assert ATLAS.INITIALISATIONS == ("path_native", "locked_low", "locked_high")


# ---------------------------------------------------------------- round-1 cross-check

def test_finalize_refuses_a_path_native_point_that_contradicts_round_one(monkeypatch, tmp_path):
    """field_t13s__path_native is, by construction, round 1's S2_DH_CLAMP_EXT run again.

    If the regenerated path fields were not the trajectory round 1 forked from, this is where it
    shows up, so finalize must refuse rather than publish an atlas built on a different trajectory.
    """
    import json as _json
    index = {"fields": {"field_t13s": {
        "snapshot_time_ms": 13000.0, "relative_to_onset_ms": 2000.0, "D_mean": .1364,
        "H_mean": 2.1, "h_gate_mean": .556, "file": "f.npz", "sha256": "x"}}}
    rows = _fake_rows({"field_t13s": {init: "BOUNDED_OSCILLATORY" for init in ATLAS.INITIALISATIONS}})
    monkeypatch.setattr(ATLAS, "_fields_index", lambda: index)
    monkeypatch.setattr(ATLAS, "OUT", tmp_path)
    monkeypatch.setattr(ATLAS.NAT, "_write_json", lambda path, payload: None)
    for (field, init), row in rows.items():
        bundle = tmp_path / f"{field}__{init}"
        bundle.mkdir(parents=True)
        (bundle / "summary.json").write_text(_json.dumps(row))

    round1 = tmp_path / "round1"
    (round1 / "S2_DH_CLAMP_EXT").mkdir(parents=True)
    (round1 / "S2_DH_CLAMP_EXT" / "summary.json").write_text(_json.dumps(
        {"verdict": {"per_second_mean_hz": [39.3] * 10}}))       # the real arm, not 50.0
    monkeypatch.setattr(ATLAS.CF, "FORK_ROOT", round1)
    with pytest.raises(RuntimeError, match="does not reproduce round 1"):
        ATLAS.finalize()


def test_finalize_accepts_a_path_native_point_that_matches_round_one(monkeypatch, tmp_path):
    import json as _json
    index = {"fields": {"field_t13s": {
        "snapshot_time_ms": 13000.0, "relative_to_onset_ms": 2000.0, "D_mean": .1364,
        "H_mean": 2.1, "h_gate_mean": .556, "file": "f.npz", "sha256": "x"}}}
    rows = _fake_rows({"field_t13s": {init: "BOUNDED_OSCILLATORY" for init in ATLAS.INITIALISATIONS}})
    monkeypatch.setattr(ATLAS, "_fields_index", lambda: index)
    monkeypatch.setattr(ATLAS, "OUT", tmp_path)
    monkeypatch.setattr(ATLAS.NAT, "_write_json", lambda path, payload: None)
    for (field, init), row in rows.items():
        bundle = tmp_path / f"{field}__{init}"
        bundle.mkdir(parents=True)
        (bundle / "summary.json").write_text(_json.dumps(row))
    round1 = tmp_path / "round1"
    (round1 / "S2_DH_CLAMP_EXT").mkdir(parents=True)
    (round1 / "S2_DH_CLAMP_EXT" / "summary.json").write_text(_json.dumps(
        {"verdict": {"per_second_mean_hz": [50.0] * 10}}))       # matches the fake atlas rows
    monkeypatch.setattr(ATLAS.CF, "FORK_ROOT", round1)
    payload = ATLAS.finalize()
    assert payload["round1_cross_check"]["field_t13s"]["identical"] is True
    assert payload["round1_cross_check"]["field_t13s"]["n_compared_seconds"] == 10
