from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.run_topic4_mz_m_gated_reserve_coupled_canary import (
    EARLY_EXIT,
    PREMATURE_ENTRY,
    SUPPORTED,
    UNRESOLVED,
    _classify,
    _save_json,
    _validate_config,
    _validate_inputs,
    first_sustained_low,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/topic4_mz_m_gated_reserve_coupled_canary.yaml"
RESULT = ROOT / "results/topic4_sef_hfo/mz_m_gated_reserve_coupled_canary"


def _config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_center_config_locks_one_path_and_forbids_grid_or_ablation() -> None:
    cfg = _config()
    _validate_config(cfg)
    assert cfg["resource_contract"]["canary_count"] == 1
    assert not cfg["resource_contract"]["launch_remaining_arms"]
    assert not cfg["scope"]["grid"] and not cfg["scope"]["ablation"]
    cfg["center_canary"]["tau_m_up_ms"] = 250.0
    with pytest.raises(RuntimeError, match="tau_m_up"):
        _validate_config(cfg)


def test_locked_inputs_and_center_mapping_are_current() -> None:
    observed = _validate_inputs(_config())
    assert set(observed) == set(_config()["input_sha256"])
    assert observed["r3_summary_path"] == _config()["input_sha256"]["r3_summary_path"]


def test_sustained_low_requires_both_regions_for_entire_interval() -> None:
    time = np.arange(0.0, 1001.0, 1.0)
    rates = np.full((time.size, 4), 0.020)
    rates[300:, :] = 0.004
    assert first_sustained_low(time, rates, 0.005, 250.0, start_ms=100.0) == 300.0
    rates[500, 1] = 0.006
    assert first_sustained_low(time, rates, 0.005, 250.0, start_ms=100.0) == 501.0


def test_strict_summary_json_rejects_nan(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _save_json(tmp_path / "bad.json", {"bad": float("nan")})


def _synthetic_result(*, entry: bool, latch_set: bool) -> tuple[dict, dict]:
    cfg = _config()
    cfg["background_event_challenge"]["realized_onsets_ms"] = [
        100.0, 200.0, 300.0, 400.0, 500.0, 600.0,
    ]
    cfg["background_event_challenge"]["pulse_free_analysis_start_ms"] = 620.0
    time = np.arange(0.0, 1001.0, 1.0)
    q = np.full((time.size, 1, 3), 0.90)
    rate = np.full((time.size, 1, 3), 0.001)
    m = np.zeros_like(q)
    latch = np.zeros_like(q, dtype=np.uint8)
    if entry:
        q[600:, 0, :2] = 0.85
        rate[600:700, 0, :2] = 0.020
    if latch_set:
        m[650:, 0, :2] = 0.1
        latch[650:, 0, :2] = 1
    result = {
        "time_ms": time, "z": q, "m": m, "p": np.zeros_like(q),
        "rE": rate, "rE_fast": rate.copy(), "latch": latch,
        "return_times_ms": [[[], [], []]],
        "latch_set_times_ms": [[650.0] if latch_set else []],
        "first_support_failure_ms": np.asarray([np.nan]),
        "first_bound_failure_ms": np.asarray([np.nan]),
        "first_nonfinite_ms": np.asarray([np.nan]),
        "support_violation_count": np.zeros((1, 3), dtype=int),
        "state_bound_violation_count": np.zeros((1, 3), dtype=int),
        "finite": np.asarray([True]), "active_at_end": np.asarray([True]),
        "final_latch_state": latch[-1].astype(bool),
    }
    return result, cfg


def test_prevention_or_no_entry_cannot_be_mislabeled_as_early_m_exit() -> None:
    result, cfg = _synthetic_result(entry=False, latch_set=False)
    row = _classify(result, cfg)
    assert row["status"] == UNRESOLVED
    assert row["entry_event_index"] is None


def test_exact_early_exit_label_requires_event6_entry_and_real_m_latch() -> None:
    result, cfg = _synthetic_result(entry=True, latch_set=True)
    row = _classify(result, cfg)
    assert row["entry_event_index"] == 6
    assert row["status"] == EARLY_EXIT


def test_canonical_canary_stops_or_unlocks_exactly_as_registered() -> None:
    path = RESULT / "coupled_canary_summary.json"
    if not path.is_file():
        pytest.skip("canonical center canary has not run yet")

    def reject_constant(value: str) -> None:
        raise AssertionError(f"non-standard JSON constant {value}")

    summary = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    assert summary["status"] in {EARLY_EXIT, PREMATURE_ENTRY, SUPPORTED}
    assert summary["registered_paths_executed"] == 1
    assert summary["registered_paths_not_executed"] == 17
    assert summary["segments_executed"] == ["A_center_canary"]
    assert all(summary["resource_gates"].values())
    assert Path(ROOT / summary["artifacts"]["figure"]).is_file()
    assert (RESULT / "figures/README.md").is_file()
    if summary["status"] == EARLY_EXIT:
        assert summary["pulse_free_core_returns"] < 4
        assert summary["pulse_free_annulus_returns"] < 4
        assert summary["stop_rule_applied"]
        assert summary["decision"] == "stop_before_grid_and_reconsider_separated_termination_timescale"
    elif summary["status"] == PREMATURE_ENTRY:
        assert summary["entry_event_index"] < 6
        assert not summary["gates"]["events_1_to_5_do_not_cross_entry_fold"]
        assert summary["decision"] == "stop_before_grid_and_reassess_coupled_event_map"
    else:
        assert all(summary["gates"].values())
