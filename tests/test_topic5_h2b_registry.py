"""B2 contract tests: reading Agent A's checkpoint registry.

The rule that matters is "缺失 producer 报 not_available，不得静默 fallback":
B reads every qualifying producer and reports what it could not get, and never
quietly substitutes something else for a producer it cannot load.
"""
from __future__ import annotations
import json
import numpy as np
import pytest
from src.topic5_h2b_transfer.registry import read_registry, resolve_subject_arms


def _reg(tmp_path, producers):
    p = tmp_path / "checkpoint_registry.json"
    p.write_text(json.dumps({"registry_version": "t", "producers": producers}))
    return p


def _anchor(tmp_path, name, n=10, d=3, t0=1000.0):
    p = tmp_path / f"{name}.npz"
    np.savez(p, state=np.arange(n * d, dtype=np.float32).reshape(n, d),
             t_anchor=np.arange(n, dtype=np.float64) * 300.0 + t0,
             split_index=np.zeros(n, dtype=np.int64),
             session_id=np.zeros(n, dtype=np.int64))
    return p


def test_every_registered_producer_is_reported_even_when_unloadable(tmp_path):
    reg = _reg(tmp_path, {
        "P_slow": {"producer_id": "P_slow", "status": "partial",
                   "subjects": {"s1": {"1": {"anchor_state": str(_anchor(tmp_path, "a"))}}}},
        "B_multiscale": {"producer_id": "B_multiscale", "status": "complete",
                         "subjects": {"s1": {"result": "/nowhere.json"}}},
    })
    arms = resolve_subject_arms(read_registry(reg), "s1")
    assert set(arms) == {"P_slow", "B_multiscale"}


def test_a_producer_without_an_anchor_state_is_marked_not_available(tmp_path):
    reg = _reg(tmp_path, {
        "B_multiscale": {"producer_id": "B_multiscale", "status": "complete",
                         "subjects": {"s1": {"result": "/nowhere.json"}}},
    })
    arm = resolve_subject_arms(read_registry(reg), "s1")["B_multiscale"]
    assert arm.status == "not_available"
    assert "anchor_state" in arm.reason
    assert arm.state is None


def test_a_subject_absent_from_a_producer_is_not_available_not_an_error(tmp_path):
    reg = _reg(tmp_path, {
        "P_slow": {"producer_id": "P_slow", "status": "partial",
                   "subjects": {"other": {"1": {"anchor_state": str(_anchor(tmp_path, "a"))}}}},
    })
    arm = resolve_subject_arms(read_registry(reg), "s1")["P_slow"]
    assert arm.status == "not_available"
    assert "subject" in arm.reason


def test_a_loadable_producer_returns_its_anchor_grid(tmp_path):
    reg = _reg(tmp_path, {
        "P_slow": {"producer_id": "P_slow", "status": "partial",
                   "subjects": {"s1": {"1": {"anchor_state": str(_anchor(tmp_path, "a"))}}}},
    })
    arm = resolve_subject_arms(read_registry(reg), "s1")["P_slow"]
    assert arm.status == "ok"
    assert arm.state.shape == (10, 3)
    assert arm.t_anchor[0] == 1000.0
    assert arm.seed == "1"


def test_a_missing_anchor_file_is_reported_rather_than_raising(tmp_path):
    reg = _reg(tmp_path, {
        "P_slow": {"producer_id": "P_slow", "status": "partial",
                   "subjects": {"s1": {"1": {"anchor_state": str(tmp_path / "gone.npz")}}}},
    })
    arm = resolve_subject_arms(read_registry(reg), "s1")["P_slow"]
    assert arm.status == "not_available"


def test_the_requested_seed_is_honoured(tmp_path):
    reg = _reg(tmp_path, {
        "P_slow": {"producer_id": "P_slow", "status": "partial",
                   "subjects": {"s1": {"1": {"anchor_state": str(_anchor(tmp_path, "a", t0=1000.0))},
                                       "2": {"anchor_state": str(_anchor(tmp_path, "b", t0=9000.0))}}}},
    })
    arm = resolve_subject_arms(read_registry(reg), "s1", seed="2")["P_slow"]
    assert arm.t_anchor[0] == 9000.0


def test_an_unavailable_requested_seed_is_refused_not_silently_substituted(tmp_path):
    """Falling back to seed 1 when seed 3 is asked for fabricates replication.

    Three "seeds" that are byte-identical are one fit, not three
    (v0.2 engineering invariants §2).
    """
    reg = _reg(tmp_path, {
        "P_slow": {"producer_id": "P_slow", "status": "partial",
                   "subjects": {"s1": {"1": {"anchor_state": str(_anchor(tmp_path, "a"))}}}},
    })
    arm = resolve_subject_arms(read_registry(reg), "s1", seed="3")["P_slow"]
    assert arm.status == "not_available"
    assert "seed" in arm.reason


def test_omitting_the_seed_still_takes_the_only_one_available(tmp_path):
    reg = _reg(tmp_path, {
        "P_slow": {"producer_id": "P_slow", "status": "partial",
                   "subjects": {"s1": {"1": {"anchor_state": str(_anchor(tmp_path, "a"))}}}},
    })
    assert resolve_subject_arms(read_registry(reg), "s1")["P_slow"].status == "ok"


def test_provenance_travels_with_every_arm(tmp_path):
    reg = _reg(tmp_path, {
        "P_slow": {"producer_id": "P_slow", "status": "partial",
                   "source_commit": "abc123", "config_hash": "cfg",
                   "subjects": {"s1": {"1": {"anchor_state": str(_anchor(tmp_path, "a"))}}}},
    })
    arm = resolve_subject_arms(read_registry(reg), "s1")["P_slow"]
    assert arm.source_commit == "abc123"
    assert arm.config_hash == "cfg"
