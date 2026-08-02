import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "topic4_zm_lifecycle_unified_ledger",
    ROOT / "scripts/build_topic4_zm_lifecycle_unified_ledger.py",
)
U = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(U)


def test_successful_wave_overrides_earlier_not_run_row_with_same_config_id():
    early = {"rows": {"x": {
        "family": "m_response_panel", "adjudicated_status": "not_run_after_adaptive_stop",
    }}}
    later = {"rows": {"x": {
        "family": "m_response_panel", "adjudicated_status": "success",
        "artifact_path": "results/x/summary.json", "wall_s": 10.0,
    }}}
    rows = U.merge_adjudicated_ledgers([(Path("early.json"), early), (Path("later.json"), later)])
    assert rows["x"]["adjudicated_status"] == "success"
    assert rows["x"]["wall_s"] == 10.0
    assert rows["x"]["source_ledger_paths"] == ["early.json", "later.json"]


def test_merge_preserves_stage_and_unique_config_count(monkeypatch):
    monkeypatch.setattr(U, "_analysis_indexes", lambda: ({}, {}))
    monkeypatch.setattr(U, "native_long_run_rows", lambda: {})
    fast = {"rows": {"a": {"family": "depression_only_lhs", "adjudicated_status": "success"}}}
    control = {"rows": {"b": {"family": "finite_control_dose", "adjudicated_status": "worker_failed"}}}
    payload = U.build_payload([(Path("fast.json"), fast), (Path("control.json"), control)])
    assert payload["n_unique_configs"] == 2
    assert payload["stage_counts"] == {"fast_phase_map": 1, "control_dose": 1}
    assert payload["status_counts"] == {"success": 1, "worker_failed": 1}


def test_analysis_readout_keeps_exit_tail_recovery_and_paired_control_evidence():
    got = U._analysis_readout({
        "phenotype": "weak_or_fragmented",
        "causal_exit_candidate": True,
        "tail_state": {
            "label": "deep_gap_burst_tail", "core_mean_hz": 21.8,
            "all_E_mean_hz": 4.3, "deep_gap_fraction": 0.86,
            "common_mode_pc1_fraction": 0.62,
        },
        "slow_trace": {
            "z_core_at_offset": 0.53, "z_core_final": 0.65,
            "z_core_post_offset_recovery": 0.12, "m_at_offset": 28.0,
            "m_peak": 77.8,
        },
        "causal_M_effect": {"status": "offset_vs_censored_gM0"},
        "causal_control_effect": {"status": "offset_advanced"},
        "control_response": {
            "fractional_core_drop": 0.6, "precontrol_pair_identical": True,
            "calibration_target_met": True,
        },
    })
    assert got["tail_state_label"] == "deep_gap_burst_tail"
    assert got["tail_deep_gap_fraction"] == 0.86
    assert got["z_core_post_offset_recovery"] == 0.12
    assert got["M_effect_status"] == "offset_vs_censored_gM0"
    assert got["paired_control_drop_fraction"] == 0.6
    assert got["precontrol_pair_identical"] is True


def test_native_long_run_is_joined_to_its_terminal_receipt(tmp_path, monkeypatch):
    out = tmp_path / "results" / "lifecycle_sprint"
    receipts = out.parent / "worker_receipts"
    out.mkdir(parents=True)
    receipts.mkdir(parents=True)
    summary = "results/lifecycle_sprint/seed1/long/summary.json"
    (out / "native_long45_analysis.json").write_text(json.dumps({
        "summary_path": summary,
        "phenotype": "weak_or_fragmented",
        "episode": {"onset_ms": 500.0, "offset_ms": 5950.0},
        "recovery": {
            "single_event_candidate": True,
            "distribution_recovered": False,
            "n_post": 74,
            "matched_event_fraction": 0.405,
        },
    }))
    (receipts / "run.json").write_text(json.dumps({
        "artifact_path": summary,
        "config_hash": "abcdef1234567890",
        "status": "success",
        "terminal_time_utc": "2026-08-02T10:00:00+00:00",
    }))
    monkeypatch.setattr(U, "ROOT", tmp_path)
    monkeypatch.setattr(U, "OUT", out)

    rows = U.native_long_run_rows()
    assert rows["abcdef123456"]["stage"] == "native_long_run"
    assert rows["abcdef123456"]["scientific_readout"]["n_post_offset_event_candidates"] == 74
    assert rows["abcdef123456"]["scientific_readout"]["returning_distribution_recovered"] is False
