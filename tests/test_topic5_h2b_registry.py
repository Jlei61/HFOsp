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


# --- P0-2: "complete" is not provenance. Verify, fail closed. ----------------


def _anchor_with(tmp_path, name, **extra):
    p = _anchor(tmp_path, name)
    return p


def _cell(tmp_path, name, chunk=128, batch=8, commit="aaaa1111"):
    """An anchor_state plus the sibling result.json that records how it trained."""
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    ap = _anchor(d, "anchor")
    (d / "result.json").write_text(json.dumps(
        {"chunk_events": chunk, "batch_segments": batch, "commit": commit}))
    return {"anchor_state": str(ap)}


def test_a_cell_whose_training_config_was_degraded_by_oom_is_refused(tmp_path):
    """The substantive defect is chunk/batch actually differing, not a hash.

    config_hash turned out to hash things that are not the training
    configuration (57 distinct hashes over 18 distinct real configs), so
    refusing on it measured the wrong thing.
    """
    reg = _reg(tmp_path, {
        "P_slow": {"producer_id": "P_slow", "status": "complete",
                   "subjects": {"s1": {"1": _cell(tmp_path, "c1", chunk=32, batch=2)},
                                "s2": {"1": _cell(tmp_path, "c2")},
                                "s3": {"1": _cell(tmp_path, "c3")}}},
    })
    arm = resolve_subject_arms(read_registry(reg), "s1", verify=True)["P_slow"]
    assert arm.status == "not_available"
    assert "chunk" in arm.reason or "batch" in arm.reason


def test_a_cell_at_the_producers_usual_config_is_admissible(tmp_path):
    """Differing commits alone do not make a cell inadmissible; they are flagged."""
    reg = _reg(tmp_path, {
        "P_local": {"producer_id": "P_local", "status": "complete",
                    "subjects": {"s1": {"1": _cell(tmp_path, "d1", commit="1111")},
                                 "s2": {"1": _cell(tmp_path, "d2", commit="2222")}}},
    })
    arm = resolve_subject_arms(read_registry(reg), "s1", verify=True)["P_local"]
    assert arm.status == "ok"
    assert arm.commit == "1111"
    assert arm.provenance_flags and any("commit" in f for f in arm.provenance_flags)


def test_a_declared_checkpoint_hash_that_does_not_match_is_refused(tmp_path):
    ck = tmp_path / "ck.pt"
    ck.write_bytes(b"not the checkpoint you think")
    reg = _reg(tmp_path, {
        "P_slow": {"producer_id": "P_slow", "status": "complete",
                   "subjects": {"s1": {"1": {"anchor_state": str(_anchor(tmp_path, "a")),
                                            "checkpoint": str(ck),
                                            "checkpoint_sha256": "0" * 64}}}},
    })
    arm = resolve_subject_arms(read_registry(reg), "s1", verify=True)["P_slow"]
    assert arm.status == "not_available"
    assert "checkpoint_sha256" in arm.reason


def test_a_cell_that_verifies_is_loaded_and_says_so(tmp_path):
    reg = _reg(tmp_path, {
        "P_slow": {"producer_id": "P_slow", "status": "complete", "config_hash": "C",
                   "subjects": {"s1": {"1": {"anchor_state": str(_anchor(tmp_path, "a")),
                                            "config_hash": "C"}}}},
    })
    arm = resolve_subject_arms(read_registry(reg), "s1", verify=True)["P_slow"]
    assert arm.status == "ok"
    assert arm.verified is True


def test_verification_can_be_switched_off_for_diagnostics_but_is_recorded(tmp_path):
    reg = _reg(tmp_path, {
        "P_slow": {"producer_id": "P_slow", "status": "complete",
                   "subjects": {"s1": {"1": _cell(tmp_path, "e1", chunk=32, batch=2)},
                                "s2": {"1": _cell(tmp_path, "e2")},
                                "s3": {"1": _cell(tmp_path, "e3")}}},
    })
    arm = resolve_subject_arms(read_registry(reg), "s1", verify=False)["P_slow"]
    assert arm.status == "ok"
    assert arm.verified is False


def test_verification_is_on_by_default(tmp_path):
    """Fail closed: a caller who forgets the flag gets the strict behaviour."""
    reg = _reg(tmp_path, {
        "P_slow": {"producer_id": "P_slow", "status": "complete",
                   "subjects": {"s1": {"1": _cell(tmp_path, "f1", chunk=32, batch=2)},
                                "s2": {"1": _cell(tmp_path, "f2")},
                                "s3": {"1": _cell(tmp_path, "f3")}}},
    })
    assert resolve_subject_arms(read_registry(reg), "s1")["P_slow"].status == "not_available"
