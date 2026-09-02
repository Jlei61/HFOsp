"""Task 1: training request interface (design §2, clauses Q1-Q8)."""

from __future__ import annotations

import json
import hashlib

import src.topic5_group_event_state.v033_training_lab.paths as paths_mod
from src.topic5_group_event_state.v033_training_lab.paths import RELEASE_FILENAME, release_status
from src.topic5_group_event_state.v033_training_lab.request import (
    REQUIRED_FIELDS,
    JobStatus,
    hash_mismatch_verdict,
    is_human_view,
    job_key,
    parse_request,
    validate_request,
)

REGISTERED = ("count_profile",)


def _payload(**over):
    base = {
        "request_id": "req_toy_001",
        "schema_version": "v2", "sealed": False,
        "scientific_target": {"family": "S_N", "predictive_view": "S_N", "objective": "count_profile",
                              "bin_convention": "left_closed_right_open_[t+a,t+b)",
                              "bins_seconds": [[0, 300], [300, 900], [900, 1800]]},
        "input_view": {"kind": "toy", "seed": 0},
        "state_architecture": "fixed_leaky",
        "split_hash": "a" * 64,
        "baseline_H": {"name": "H_mark", "hash": "b" * 64, "source": "provisional_local"},
        "endpoint_and_reduction": {"selection_phase": "inner_val", "metric": "nb_nll",
                                   "reduction": "mean_per_anchor"},
        "search_budget": {"n_configs": 4, "max_steps": 60, "rung_steps": [20, 60], "eta": 2,
                          "seeds_low": 1, "seeds_mid": 3, "seeds_final": 5, "n_final": 2},
        "seed_policy": {"base_seed": 7},
        "resource_ceiling": {"max_workers": 1, "gpu_ids": [], "vram_gib": 0, "ram_gib": 8, "threads": 1},
        "science_code_commit": "c" * 40,
        "input_hash": "d" * 64,
        "requested_by": "agent_c",
    }
    base.update(over)
    return base


def _validate(payload, **over):
    kw = dict(registered_objectives=REGISTERED, release_present=False, head_commit="deadbeef")
    kw.update(over)
    return validate_request(payload, **kw)


def test_q1_missing_or_empty_fields_are_invalid_and_all_listed():
    p = _payload()
    del p["split_hash"]
    p["requested_by"] = ""
    request, verdict = parse_request(p)
    assert request is None
    assert verdict["status"] == JobStatus.INVALID_REQUEST.value
    assert set(verdict["missing_fields"]) == {"split_hash", "requested_by"}
    assert len(REQUIRED_FIELDS) == 13


def test_q2_unregistered_objective_is_invalid_not_guessed():
    p = _payload(scientific_target={"family": "S_G", "objective": "subset_identity"})
    verdict = _validate(p)
    assert verdict["status"] == "INVALID_REQUEST"
    assert any("objective" in reason for reason in verdict["reasons"])


def test_q3_gated_exploratory_requires_explicit_approval():
    p = _payload(state_architecture="gated_exploratory")
    assert _validate(p)["status"] == "INVALID_REQUEST"
    p["exploratory_approved"] = True
    assert _validate(p)["status"] == "PENDING"


def test_q4_human_view_is_held_without_release_while_toy_pends():
    human = _payload(input_view={"kind": "R1", "subject": "epilepsiae_1146", "data_registry_key": "dev"})
    assert is_human_view(human["input_view"]) is True
    assert is_human_view(_payload()["input_view"]) is False
    assert _validate(human, release_present=False)["status"] == "HELD_NO_RELEASE"
    assert _validate(human, release_present=True)["status"] == "PENDING"
    assert _validate(_payload(), release_present=False)["status"] == "PENDING"


def test_q5_science_commit_is_independent_of_trainer_head():
    verdict = _validate(_payload(), head_commit="0000")
    assert verdict["status"] == "PENDING"


def test_q6_job_key_tracks_identity_fields_and_ignores_dict_order():
    request, _ = parse_request(_payload())
    kw = {"trainer_code_commit": "e" * 40}
    key = job_key(request, subject="toy", seed=1, config_hash="c1", **kw)
    assert key != job_key(request, subject="toy", seed=2, config_hash="c1", **kw)
    assert key != job_key(request, subject="toy", seed=1, config_hash="c2", **kw)
    assert key != job_key(request, subject="other", seed=1, config_hash="c1", **kw)
    other_input, _ = parse_request(_payload(input_hash="f" * 64))
    assert key != job_key(other_input, subject="toy", seed=1, config_hash="c1", **kw)
    reordered, _ = parse_request(dict(reversed(list(_payload().items()))))
    assert job_key(reordered, subject="toy", seed=1, config_hash="c1", **kw) == key
    assert job_key(reordered, subject="toy", seed=1, config_hash="c1", trainer_code_commit="f" * 40) != key


def test_q7_release_status_reports_absent_and_validated(tmp_path, monkeypatch):
    monkeypatch.setattr(paths_mod, "repo_root", lambda: tmp_path)
    candidates = [tmp_path / RELEASE_FILENAME, tmp_path / "shared" / RELEASE_FILENAME]
    absent = release_status(candidates=candidates)
    assert absent["present"] is False and absent["path"] is None and absent["payload"] is None
    (tmp_path / "spec.md").write_text("spec")
    (tmp_path / "plan.md").write_text("plan")
    sha = lambda name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    (tmp_path / RELEASE_FILENAME).write_text(json.dumps({
        "format": "group_event_state_v0_3_3_execution_release", "status": "ACTIVE_DEVELOPMENT_ONLY",
        "user_approved": True, "sealed": False, "base_commit": "a" * 40,
        "scope": {"development_only": True, "sealed_partition_opened": False},
        "spec": {"path": "spec.md", "sha256": sha("spec.md")}, "spec_sha256": sha("spec.md"),
        "plan": {"path": "plan.md", "sha256": sha("plan.md")}, "plan_sha256": sha("plan.md"),
    }))
    present = release_status(candidates=candidates)
    assert present["present"] is True
    assert present["path"] == str(tmp_path / RELEASE_FILENAME)
    assert present["payload"]["base_commit"] == "a" * 40


def test_q8_split_or_input_hash_or_missing_h_bins_hold_the_request():
    request, _ = parse_request(_payload())
    ok = hash_mismatch_verdict(request, split_hash="a" * 64, input_hash="d" * 64, missing_h_bins=[])
    assert ok["status"] == "PENDING"
    bad_split = hash_mismatch_verdict(request, split_hash="z" * 64, input_hash="d" * 64, missing_h_bins=[])
    assert bad_split["status"] == "HELD_MISMATCH" and "split_hash" in " ".join(bad_split["reasons"])
    bad_h = hash_mismatch_verdict(request, split_hash="a" * 64, input_hash="d" * 64, missing_h_bins=[1, 2])
    assert bad_h["status"] == "HELD_MISMATCH" and "baseline_H" in " ".join(bad_h["reasons"])
