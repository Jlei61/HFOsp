from scripts.audit_topic4_rev10_d7_active_zm_continuous_field_canary import (
    summarize_runaway,
)


def _record(candidate, seed, runaway):
    return {
        "candidate": {"candidate_id": candidate},
        "seed": seed,
        "run": {"runaway_early_stop_ms": runaway},
    }


def test_d7_summary_requires_complete_candidate_seed_grid():
    records = [
        _record("a", 1, 6000.0), _record("a", 2, 7000.0),
        _record("b", 1, 8000.0), _record("b", 2, 9000.0),
    ]
    summary = summarize_runaway(records, ["a", "b"], [1, 2])
    assert summary["n_runaway_workers"] == 4
    assert summary["safe_candidate_ids"] == []
    assert summary["all_candidates_runaway_on_all_networks"] is True
    assert summary["runaway_time"]["minimum_ms"] == 6000.0
    assert summary["runaway_time"]["maximum_ms"] == 9000.0


def test_d7_summary_identifies_a_fully_safe_candidate():
    records = [
        _record("a", 1, None), _record("a", 2, None),
        _record("b", 1, 8000.0), _record("b", 2, 9000.0),
    ]
    summary = summarize_runaway(records, ["a", "b"], [1, 2])
    assert summary["safe_candidate_ids"] == ["a"]
    assert summary["all_candidates_runaway_on_all_networks"] is False
    assert "a" not in summary["candidate_mean_runaway_time_ms"]
