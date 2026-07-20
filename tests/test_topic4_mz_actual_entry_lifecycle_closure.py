from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.run_topic4_mz_actual_entry_lifecycle_closure import (
    SUPPORTED,
    _analytic_active_latch,
    _analytic_released_latch,
    _classify_challenge,
    _early_gates,
    _hash_manifest,
    _late_gates,
    _pair_crossings,
    _save_json,
    _solve_released_duration_for_q,
    _validate_config,
    _validate_inputs,
    _verify_hash_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/topic4_mz_actual_entry_lifecycle_closure.yaml"
RESULT = ROOT / "results/topic4_sef_hfo/mz_actual_entry_lifecycle_closure"


def _config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_config_is_center_only_and_locks_exact_three_source_artifacts() -> None:
    cfg = _config()
    _validate_config(cfg)
    assert set(cfg["input_sha256"]) == {
        "source_summary_path", "source_trace_path", "source_config_path",
    }
    assert cfg["integration"]["dt_values_ms"] == [0.125, 0.0625]
    assert cfg["resource_contract"]["base_must_pass_before_half_dt"]
    drifted = json.loads(json.dumps(cfg))
    drifted["retrigger"]["onsets_ms"][4] = 7532.0
    with pytest.raises(RuntimeError, match="challenge"):
        _validate_config(drifted)


def test_locked_r3_artifacts_and_double_checkpoint_are_current() -> None:
    observed, _, arrays, _ = _validate_inputs(_config())
    assert observed == _config()["input_sha256"]
    assert arrays["final_state"].dtype == np.float64
    assert arrays["final_state"].shape == (32,)
    assert arrays["final_latch_state"].tolist() == [True, True, False]


def test_pairing_is_one_to_one_and_enforces_maximum_lag() -> None:
    pairs = _pair_crossings([10.0, 30.0, 50.0], [11.0, 29.0, 80.0], 2.0)
    assert pairs == [(10.0, 11.0), (30.0, 29.0)]
    assert _pair_crossings([10.0], [31.0], 20.0) == []


def test_active_latch_bridge_matches_closed_form_and_keeps_m_fixed() -> None:
    q, p, m = _analytic_active_latch(
        q=0.854, p=0.02, m=0.24, duration_ms=1000.0,
        q_rest=0.9, tau_p_ms=750.0, tau_slow_ms=90000.0,
        tau_fast_ms=20000.0,
    )
    rate = (1.0 - 0.24) / 90000.0 + 0.24 / 20000.0
    assert q == pytest.approx(0.9 - (0.9 - 0.854) * np.exp(-rate * 1000.0))
    assert p == pytest.approx(0.02 * np.exp(-1000.0 / 750.0))
    assert m == 0.24


def test_released_bridge_and_q_inverse_are_consistent() -> None:
    q1, p1, m1 = _analytic_released_latch(
        q=0.886, p=0.001, m=0.20, duration_ms=12000.0,
        q_rest=0.9, tau_p_ms=750.0, tau_m_ms=12000.0,
        tau_slow_ms=90000.0, tau_fast_ms=20000.0,
    )
    solved = _solve_released_duration_for_q(
        q=0.886, m=0.20, target_q=q1, q_rest=0.9,
        tau_m_ms=12000.0, tau_slow_ms=90000.0,
        tau_fast_ms=20000.0,
    )
    assert solved == pytest.approx(12000.0, abs=1.0e-5)
    assert m1 == pytest.approx(0.20 / np.e)
    assert p1 < 1.0e-9


def test_common_classifier_rejects_preexisting_entry() -> None:
    cfg = _config()
    time = np.arange(0.0, 20001.0, 1.0)
    q = np.full((time.size, 3), 0.85)
    rate = np.full((time.size, 3), 0.001)
    result = {
        "time_ms": time,
        "z": q[:, None, :],
        "rE": rate[:, None, :],
        "rE_fast": rate[:, None, :],
        "finite": np.asarray([True]),
        "active_at_end": np.asarray([True]),
        "first_support_failure_ms": np.asarray([np.nan]),
        "first_bound_failure_ms": np.asarray([np.nan]),
        "first_nonfinite_ms": np.asarray([np.nan]),
        "support_violation_count": np.zeros((1, 3), dtype=int),
        "state_bound_violation_count": np.zeros((1, 3), dtype=int),
    }
    row = _classify_challenge(result, cfg)
    assert row["entry_event_index"] == 0
    assert not row["actual_entry_lifecycle_candidate"]


def _synthetic_prebelow_result(*, four_returns: bool, high_tail: bool = False) -> dict:
    time = np.arange(0.0, 20001.0, 1.0)
    q = np.full((time.size, 3), 0.85)
    rate = np.full((time.size, 3), 0.001)
    if four_returns:
        for core_time in (8500, 9000, 9500, 10000):
            rate[core_time:core_time + 12, 0] = 0.025
            rate[core_time + 5:core_time + 17, 1] = 0.025
    if high_tail:
        rate[-1000:, :2] = 0.030
    return {
        "time_ms": time, "z": q[:, None, :], "rE": rate[:, None, :],
        "rE_fast": rate[:, None, :], "finite": np.asarray([True]),
        "active_at_end": np.asarray([True]),
        "first_support_failure_ms": np.asarray([np.nan]),
        "first_bound_failure_ms": np.asarray([np.nan]),
        "first_nonfinite_ms": np.asarray([np.nan]),
        "support_violation_count": np.zeros((1, 3), dtype=int),
        "state_bound_violation_count": np.zeros((1, 3), dtype=int),
    }


def test_prebelow_fold_four_return_train_is_still_a_lifecycle_candidate() -> None:
    row = _classify_challenge(
        _synthetic_prebelow_result(four_returns=True), _config(),
    )
    assert row["entry_event_index"] == 0
    assert row["candidate_trigger_event_index"] == 5
    assert row["paired_returns"] == 4
    assert row["actual_entry_lifecycle_candidate"]
    assert not _early_gates(row)["no_lifecycle_candidate"]


def test_prebelow_fold_sustained_high_tail_cannot_pass_early_suppression() -> None:
    row = _classify_challenge(
        _synthetic_prebelow_result(four_returns=False, high_tail=True), _config(),
    )
    gates = _early_gates(row)
    assert not row["final_sustained_low"]
    assert row["bounded_high_or_runaway_tail"]
    assert not gates["finite_supported_sustained_low_tail"]
    assert not gates["not_bounded_high_or_runaway"]


def test_late_gate_rejects_a_transient_exit_followed_by_high_tail() -> None:
    row = _classify_challenge(
        _synthetic_prebelow_result(four_returns=True, high_tail=True), _config(),
    )
    row["events_1_to_4_above_fold"] = True
    row["event_5_first_entry"] = True
    row["event6_no_section_crossing"] = True
    gates = _late_gates(row, _config())
    assert row["actual_entry_lifecycle_candidate"]
    assert gates["finite_low_exit_recurs"]
    assert not gates["final_sustained_low"]
    assert not gates["not_bounded_high_or_runaway_tail"]
    assert not all(gates.values())


def test_output_hash_manifest_fails_closed_on_tamper(tmp_path: Path) -> None:
    artifact = tmp_path / "trace.npz"
    artifact.write_bytes(b"canonical")
    manifest = _hash_manifest([artifact], root=tmp_path)
    _verify_hash_manifest(manifest, root=tmp_path)
    artifact.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        _verify_hash_manifest(manifest, root=tmp_path)


def test_strict_json_rejects_nonfinite_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _save_json(tmp_path / "bad.json", {"x": float("nan")})


def test_canonical_closure_is_strict_and_complete_if_present() -> None:
    path = RESULT / "actual_entry_lifecycle_closure_summary.json"
    if not path.is_file():
        pytest.skip("canonical R4 center closure has not run")

    def reject_constant(value: str) -> None:
        raise AssertionError(f"non-standard JSON constant {value}")

    summary = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    assert summary["status"] == SUPPORTED or summary["status"].startswith("R4_")
    assert summary["dt_runs"][0]["dt_ms"] == 0.125
    assert summary["resource_gates"]["trace_files_below_64_mib"]
    assert summary["resource_gates"]["peak_rss_below_1p5_gib"]
    assert set(summary["output_sha256"]) == {
        *(run["trace_path"] for run in summary["dt_runs"]),
        summary["artifacts"]["gate_csv"],
        summary["artifacts"]["endpoint_csv"],
    }
    _verify_hash_manifest(summary["output_sha256"])
    for run in summary["dt_runs"]:
        assert run["late_gates"]["final_sustained_low"]
        assert run["late_gates"]["not_bounded_high_or_runaway_tail"]
    assert (RESULT / "figures/README.md").is_file()
    assert (RESULT / "figures/mz_actual_entry_lifecycle_closure.png").is_file()
    assert (RESULT / "figures/mz_actual_entry_lifecycle_closure.pdf").is_file()
