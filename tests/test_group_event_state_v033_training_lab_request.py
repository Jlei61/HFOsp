"""Task 1: training request interface (design §2, clauses Q1-Q8)."""

from __future__ import annotations

import json

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
        "scientific_target": {"family": "S_N", "objective": "count_profile",
                              "bins_seconds": [[0, 300], [300, 900], [900, 1800]]},
        "input_view": {"kind": "toy", "seed": 0},
        "state_family": "fixed_leaky",
        "split_hash": "abc",
        "baseline_H": {"source": "provisional_local"},
        "endpoint_and_reduction": {"selection_phase": "inner_val", "metric": "nb_nll",
                                   "reduction": "mean_per_anchor"},
        "search_budget": {"n_configs": 4, "max_steps": 60, "rung_steps": [20, 60], "eta": 2,
                          "seeds_low": 1, "seeds_mid": 3, "seeds_final": 5, "n_final": 2},
        "seed_policy": {"base_seed": 7},
        "resource_ceiling": {"max_workers": 1, "gpu_ids": [], "vram_gib": 0, "ram_gib": 8, "threads": 1},
        "code_commit": "deadbeef",
        "input_hash": "xyz",
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
    p = _payload(state_family="gated_exploratory")
    assert _validate(p)["status"] == "INVALID_REQUEST"
    p["exploratory_approved"] = True
    assert _validate(p)["status"] == "PENDING"


def test_q4_human_view_is_held_without_release_while_toy_pends():
    human = _payload(input_view={"kind": "R1", "subject": "epilepsiae_1146"})
    assert is_human_view(human["input_view"]) is True
    assert is_human_view(_payload()["input_view"]) is False
    assert _validate(human, release_present=False)["status"] == "HELD_NO_RELEASE"
    assert _validate(human, release_present=True)["status"] == "PENDING"
    assert _validate(_payload(), release_present=False)["status"] == "PENDING"


def test_q5_code_commit_mismatch_is_held_before_release_check():
    verdict = _validate(_payload(), head_commit="0000")
    assert verdict["status"] == "HELD_CODE_COMMIT_MISMATCH"
    human = _payload(input_view={"kind": "R0", "subject": "epilepsiae_1146"})
    assert _validate(human, head_commit="0000")["status"] == "HELD_CODE_COMMIT_MISMATCH"


def test_q6_job_key_tracks_identity_fields_and_ignores_dict_order():
    request, _ = parse_request(_payload())
    key = job_key(request, subject="toy", seed=1, config_hash="c1")
    assert key != job_key(request, subject="toy", seed=2, config_hash="c1")
    assert key != job_key(request, subject="toy", seed=1, config_hash="c2")
    assert key != job_key(request, subject="other", seed=1, config_hash="c1")
    other_input, _ = parse_request(_payload(input_hash="different"))
    assert key != job_key(other_input, subject="toy", seed=1, config_hash="c1")
    reordered, _ = parse_request(dict(reversed(list(_payload().items()))))
    assert job_key(reordered, subject="toy", seed=1, config_hash="c1") == key


def test_q7_release_status_reports_absent_and_present(tmp_path):
    candidates = [tmp_path / RELEASE_FILENAME, tmp_path / "shared" / RELEASE_FILENAME]
    absent = release_status(candidates=candidates)
    assert absent["present"] is False and absent["path"] is None and absent["payload"] is None
    (tmp_path / RELEASE_FILENAME).write_text(json.dumps(
        {"approved_by": "user", "sealed": False, "base_commit": "233f3ad1"}))
    present = release_status(candidates=candidates)
    assert present["present"] is True
    assert present["path"] == str(tmp_path / RELEASE_FILENAME)
    assert present["payload"]["base_commit"] == "233f3ad1"


def test_q8_split_or_input_hash_or_missing_h_bins_hold_the_request():
    request, _ = parse_request(_payload())
    ok = hash_mismatch_verdict(request, split_hash="abc", input_hash="xyz", missing_h_bins=[])
    assert ok["status"] == "PENDING"
    bad_split = hash_mismatch_verdict(request, split_hash="zzz", input_hash="xyz", missing_h_bins=[])
    assert bad_split["status"] == "HELD_MISMATCH" and "split_hash" in " ".join(bad_split["reasons"])
    bad_h = hash_mismatch_verdict(request, split_hash="abc", input_hash="xyz", missing_h_bins=[1, 2])
    assert bad_h["status"] == "HELD_MISMATCH" and "baseline_H" in " ".join(bad_h["reasons"])
