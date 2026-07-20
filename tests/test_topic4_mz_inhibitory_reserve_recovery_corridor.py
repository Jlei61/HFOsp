from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.run_topic4_mz_inhibitory_reserve_recovery_corridor import (
    _consecutive_components,
    _fixed_parameter_sensitivity_contract,
    _save_json,
    _schedule_contract,
    _status,
    classify_event_replay,
    exact_periodic_q_orbit,
    exact_q_trace,
    hybrid_handoff_predictor,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/topic4_mz_inhibitory_reserve_recovery_corridor.yaml"
RESULT = ROOT / "results/topic4_sef_hfo/mz_inhibitory_reserve_recovery_corridor"


def _config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_exact_q_trace_matches_constant_use_solution_and_subdivision() -> None:
    durations = np.asarray([11.0, 23.0, 7.0])
    use = np.full(3, 0.2)
    kwargs = {
        "q_initial": 0.87,
        "q_rest": 0.9,
        "q_reserve": 0.8,
        "tau_recovery_ms": 20_000.0,
        "tau_depletion_ms": 200.0,
    }
    base = exact_q_trace(durations, use, substeps=1, **kwargs)
    split = exact_q_trace(durations, use, substeps=7, **kwargs)

    decay = 1.0 / 20_000.0 + 0.2 / 200.0
    equilibrium = (0.9 / 20_000.0 + 0.8 * 0.2 / 200.0) / decay
    expected = equilibrium + (0.87 - equilibrium) * np.exp(-decay * durations.sum())
    assert base["q"][-1] == pytest.approx(expected)
    assert split["q"][-1] == pytest.approx(expected)
    assert split["integral"] == pytest.approx(base["integral"])
    assert split["map_alpha"] == pytest.approx(base["map_alpha"])


def test_exact_periodic_q_orbit_closes_at_constant_use_fixed_point() -> None:
    durations = np.full(8, 10.0)
    use = np.full(8, 0.2)
    result = exact_periodic_q_orbit(
        durations,
        use,
        q_rest=0.9,
        q_reserve=0.8,
        tau_recovery_ms=20_000.0,
        tau_depletion_ms=200.0,
        integrated_returns=8,
    )
    decay = 1.0 / 20_000.0 + 0.2 / 200.0
    fixed = (0.9 / 20_000.0 + 0.8 * 0.2 / 200.0) / decay
    assert result["q_min"] == pytest.approx(fixed, abs=1.0e-12)
    assert result["q_max"] == pytest.approx(fixed, abs=1.0e-12)
    assert result["q_mean"] == pytest.approx(fixed, abs=1.0e-12)
    assert result["per_return_multiplier"] == pytest.approx(np.exp(-decay * 10.0))
    assert result["closure_error"] <= 1.0e-14


def test_event_classifier_requires_last_event_crossing_and_prelast_margin() -> None:
    trace = {
        "time_ms": np.asarray([0.0, 10.0, 20.0, 29.0, 30.0, 31.0, 40.0]),
        "q": np.asarray([0.90, 0.88, 0.87, 0.861, 0.860, 0.849, 0.840]),
    }
    accepted = classify_event_replay(
        trace,
        [10.0, 20.0, 30.0],
        entry_fold_q=0.855,
        final_target_q=0.850,
        pre_last_margin_q=0.005,
    )
    assert accepted["outcome"] == "target_entry_event_3"
    assert accepted["entry_pass"] is True

    early = deepcopy(trace)
    early["q"] = np.asarray([0.90, 0.88, 0.854, 0.854, 0.853, 0.849, 0.840])
    rejected = classify_event_replay(
        early,
        [10.0, 20.0, 30.0],
        entry_fold_q=0.855,
        final_target_q=0.850,
        pre_last_margin_q=0.005,
    )
    assert rejected["outcome"] == "premature_entry_event_2"
    assert rejected["entry_pass"] is False


def test_schedule_contract_does_not_force_heldout_label_mixture() -> None:
    rows = []
    labels = {
        "isolated": (False, None, "no_entry"),
        "dense_1200ms": (True, 6, "entry_event_6"),
        "sparse_3400ms": (False, None, "no_entry"),
        "heldout_seed_a": (False, None, "no_entry"),
        "heldout_seed_b": (False, None, "no_entry"),
    }
    for schedule, (entered, index, outcome) in labels.items():
        for substeps in (1, 2):
            rows.append({
                "schedule": schedule,
                "substeps": substeps,
                "entered": entered,
                "entry_event_index": index,
                "outcome": outcome,
            })
    assert _schedule_contract(rows) is True
    assert _schedule_contract(rows[:-1]) is False
    for row in rows:
        if row["schedule"] == "heldout_seed_a":
            row["entered"] = True
            row["entry_event_index"] = 3
            row["outcome"] = "premature_entry_event_3"
    assert _schedule_contract(rows) is False


def test_fixed_parameter_sensitivity_contract_is_complete_and_fail_closed() -> None:
    cfg = _config()
    q_hold = float(cfg["model"]["preferred_q_hold"])
    events = [
        {
            "tau_recovery_s": tau,
            "q_hold": q_hold,
            "substeps": substeps,
            "outcome": "entry_event_6",
            "post_event_fold_margin_pass": True,
        }
        for tau in cfg["sensitivity"]["fixed_parameter_tau_recovery_s"]
        for substeps in cfg["integration"]["event_scalar_substeps"]
    ]
    periodic = [
        {
            "tau_recovery_s": tau,
            "q_hold": q_hold,
            "phase": phase,
            "source_dt_ms": dt,
            "sensitivity_periodic_pass": True,
        }
        for tau in cfg["sensitivity"]["fixed_parameter_tau_recovery_s"]
        for phase in cfg["periodic_gate"]["relative_phase_fractions"]
        for dt in cfg["periodic_gate"]["source_dt_ms"]
    ]
    assert _fixed_parameter_sensitivity_contract(events, periodic, cfg) is True
    assert _fixed_parameter_sensitivity_contract(events[:-1], periodic, cfg) is False
    assert _fixed_parameter_sensitivity_contract(events, periodic[:-1], cfg) is False

    failed = deepcopy(events)
    failed[0]["post_event_fold_margin_pass"] = False
    assert _fixed_parameter_sensitivity_contract(failed, periodic, cfg) is False


def test_hybrid_handoff_freezes_additive_until_state_defined_reset() -> None:
    cfg = _config()
    fold_q = np.asarray([0.83, float(cfg["model"]["entry_fold_q"])])
    fold_a = np.asarray([0.20, 0.0])
    result = hybrid_handoff_predictor(
        q_start=0.84235,
        q_rest=0.9,
        tau_recovery_ms=80_000.0,
        additive_start_mv=0.37,
        fold_q=fold_q,
        fold_a=fold_a,
        cfg=cfg,
    )
    t_poff = 750.0 * np.log(1.0 / 0.03)
    assert result["reset_time_ms"] == pytest.approx(
        max(result["time_to_qsafe_ms"], t_poff)
    )
    np.testing.assert_array_equal(result["trace_additive_mv"], 0.37)
    assert result["q_at_reset"] == pytest.approx(0.885)
    assert result["time_to_additive_release_ms"] == pytest.approx(
        result["reset_time_ms"] + 12_000.0 * np.log(0.37 / 0.020)
    )
    assert result["q_at_additive_release"] > result["q_at_reset"]
    assert result["handoff_pass"] is True

    too_slow = hybrid_handoff_predictor(
        q_start=0.84235,
        q_rest=0.9,
        tau_recovery_ms=160_000.0,
        additive_start_mv=0.37,
        fold_q=fold_q,
        fold_a=fold_a,
        cfg=cfg,
    )
    assert too_slow["reset_horizon_pass"] is False
    assert too_slow["handoff_pass"] is False


def test_status_and_component_contract_do_not_overclaim() -> None:
    assert _consecutive_components([False, True, True, False, True]) == [[1, 2], [4]]
    assert _status(True, True).endswith("SUPPORTED_SHORT_P3_STATE_FORK_ONLY")
    assert _status(False, True).endswith("CLEAN_NO_GO_REGISTERED_GATES")
    assert _status(False, False).endswith("NUMERICALLY_UNRESOLVED")


def test_strict_json_writer_rejects_nan(tmp_path: Path) -> None:
    path = tmp_path / "strict.json"
    _save_json(path, {"missing": None, "value": 1.0})
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "missing": None,
        "value": 1.0,
    }
    with pytest.raises(ValueError, match="Out of range float"):
        _save_json(path, {"invalid": np.nan})


def test_canonical_artifact_is_complete_strict_clean_no_go() -> None:
    text = (RESULT / "recovery_corridor_summary.json").read_text(encoding="utf-8")

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    summary = json.loads(text, parse_constant=reject_constant)
    assert summary["status"] == "R2_RECOVERY_TIMESCALE_CORRIDOR_CLEAN_NO_GO_REGISTERED_GATES"
    assert summary["decision"] == "do_not_run_coupled_R2_and_proceed_to_two_pool_resource_design"
    assert summary["registered_cell_count"] == 30
    assert summary["root_found_cell_count"] == 30
    assert summary["numeric_error_cell_count"] == 0
    assert summary["root_scan_row_count"] == 4800
    assert summary["expected_root_scan_row_count"] == 4800
    assert summary["passing_tau_nodes"] == [80.0]
    assert summary["preferred_component_tau_nodes"] == [80.0]
    assert summary["fixed_parameter_sensitivity_event_row_count"] == 4
    assert summary["expected_fixed_parameter_sensitivity_event_row_count"] == 4
    assert summary["fixed_parameter_sensitivity_periodic_row_count"] == 16
    assert summary["expected_fixed_parameter_sensitivity_periodic_row_count"] == 16
    assert summary["gates"]["fixed_parameter_72_88s_sensitivity_pass"] is True
    assert summary["gates"]["three_node_component_contains_preregistered_80s"] is False
    assert summary["plot_status"] == "complete"

    expected_rows = {
        "recovery_corridor_mapping.csv": 30,
        "recovery_corridor_root_scan.csv": 4800,
        "recovery_corridor_event_entry.csv": 60,
        "recovery_corridor_periodic_oracle.csv": 240,
        "recovery_corridor_hybrid_handoff.csv": 30,
        "recovery_corridor_schedule_probes.csv": 120,
        "recovery_corridor_sensitivity_event.csv": 4,
        "recovery_corridor_sensitivity_periodic.csv": 16,
        "recovery_corridor_tau_acceptance.csv": 10,
    }
    for name, count in expected_rows.items():
        with (RESULT / name).open(newline="", encoding="utf-8") as handle:
            assert len(list(csv.DictReader(handle))) == count
    assert (RESULT / "figures/mz_inhibitory_reserve_recovery_corridor.png").is_file()
    assert (RESULT / "figures/README.md").is_file()
