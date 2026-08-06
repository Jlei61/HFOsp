from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.run_topic4_mz_m_gated_reserve_recovery import (
    CLEAN_NO_GO,
    SUPPORTED,
    _first_sustained_low,
    _load_inputs,
    _r2_parity,
    _save_json,
    _validate_config,
    protected_handoff,
    q_nullcline,
    recovery_rate_per_ms,
    replay_q_with_sensor,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/topic4_mz_m_gated_reserve_recovery.yaml"
RESULT = ROOT / "results/topic4_sef_hfo/mz_m_gated_reserve_recovery"


def _config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_recovery_rate_uses_dimensionless_m_and_has_registered_endpoints() -> None:
    values = recovery_rate_per_ms(np.asarray([0.0, 0.5, 1.0]), 80.0, 20.0)
    assert np.isclose(values[0], 1.0 / 80000.0)
    assert np.isclose(values[-1], 1.0 / 20000.0)
    assert np.all(np.diff(values) > 0.0)
    with pytest.raises(ValueError, match="dimensionless"):
        recovery_rate_per_ms(1.2, 80.0, 20.0)
    with pytest.raises(ValueError):
        recovery_rate_per_ms(0.5, 20.0, 20.0)


def test_q_nullcline_rises_with_m_but_stays_unique_and_stable() -> None:
    m = np.linspace(0.0, 1.0, 101)
    qstar = q_nullcline(m, 0.18, 0.90, 0.8415, 280.0, 80.0, 20.0)
    assert np.all(np.diff(qstar) > 0.0)
    assert np.all((qstar > 0.8415) & (qstar < 0.90))
    eigenvalue = -(recovery_rate_per_ms(m, 80.0, 20.0) + 0.18 / 280.0)
    assert np.all(eigenvalue < 0.0)


def test_sensor_replay_matches_constant_affine_solution() -> None:
    time = np.arange(0.0, 101.0, 2.0)
    use = np.full_like(time, 0.25)
    m = np.full_like(time, 0.4)
    result = replay_q_with_sensor(
        time, use, m, q_initial=0.845, q_rest=0.90, q_reserve=0.841,
        tau_depletion_ms=275.0, tau_slow_s=80.0, tau_fast_s=20.0,
        stop_time_ms=100.0, reporting_dt_ms=1.0,
    )
    a = float(recovery_rate_per_ms(0.4, 80.0, 20.0))
    b = 0.25 / 275.0
    fixed = (a * 0.90 + b * 0.841) / (a + b)
    expected = fixed + (0.845 - fixed) * np.exp(-(a + b) * result["time_ms"])
    assert np.allclose(result["q"], expected, atol=1.0e-14)


def test_sustained_low_requires_both_regions_for_full_window() -> None:
    time = np.arange(0.0, 142.0, 2.0)
    rates = np.full((time.size, 2), 0.02)
    rates[10:, :] = 0.004
    assert _first_sustained_low(time, rates, 0.005, 50.0) == 10
    rates[30, 1] = 0.006
    assert _first_sustained_low(time, rates, 0.005, 50.0) == 31


def test_protected_handoff_freezes_m_until_reset_then_releases() -> None:
    cfg = _config()
    fold_q = np.asarray([0.84, cfg["model"]["entry_fold_q"]])
    fold_a = np.asarray([0.20, 0.0])
    result = protected_handoff(
        0.841, 0.20, tau_slow_s=80.0, tau_fast_s=20.0,
        q_rest=0.90, fold_q=fold_q, fold_a=fold_a, cfg=cfg,
    )
    assert result["reset_time_ms"] >= result["persistence_off_bound_ms"]
    assert result["q_at_reset"] >= cfg["handoff"]["q_reset_safe"] - 1.0e-12
    assert np.all(np.diff(result["release_m"]) <= 0.0)
    assert np.all(np.diff(result["release_q"]) >= -1.0e-12)
    assert np.isclose(
        cfg["model"]["additive_max_mv"] * result["release_m"][-1],
        cfg["handoff"]["additive_release_threshold_mv"], atol=1.0e-10,
    )


def test_config_is_fail_closed_and_forbids_scope_drift() -> None:
    cfg = _config()
    _validate_config(cfg)
    cfg["model"]["tau_slow_s_axis"] = [80.0, 90.0, 100.0]
    with pytest.raises(RuntimeError, match="tau_slow"):
        _validate_config(cfg)
    cfg = _config()
    cfg["scope"]["ee_weight_change"] = True
    with pytest.raises(RuntimeError, match="forbidden"):
        _validate_config(cfg)


def test_strict_json_rejects_nan(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _save_json(tmp_path / "bad.json", {"bad": float("nan")})


def test_existing_r2_schedule_parity_is_recomputed_and_fails_closed() -> None:
    cfg = _config()
    _, r2, _, _, tables, _, _ = _load_inputs(cfg)
    canonical_rows, canonical_schedule, _ = _r2_parity(cfg, r2, tables)
    assert all(row["r2_schedule_label_match"] for row in canonical_rows)
    assert canonical_schedule == {
        70.0: True, 80.0: True, 90.0: True,
        100.0: True, 120.0: False, 160.0: False,
    }

    tampered = deepcopy(tables)
    dense_base = next(
        row for row in tampered["schedule"]
        if float(row["tau_recovery_s"]) == 80.0
        and row["schedule"] == "dense_1200ms"
        and int(row["substeps"]) == 1
    )
    dense_base["entered"] = False
    tampered_rows, _, _ = _r2_parity(cfg, r2, tampered)
    assert not all(
        row["r2_schedule_label_match"]
        for row in tampered_rows if row["tau_slow_s"] == 80.0
    )

    missing = deepcopy(tables)
    missing["schedule"] = [
        row for row in missing["schedule"]
        if not (
            float(row["tau_recovery_s"]) == 120.0
            and row["schedule"] == "heldout_seed_20260723"
            and int(row["substeps"]) == 2
        )
    ]
    missing_rows, _, _ = _r2_parity(cfg, r2, missing)
    assert not all(
        row["r2_schedule_label_match"]
        for row in missing_rows if row["tau_slow_s"] == 120.0
    )

    missing_tau = deepcopy(tables)
    missing_tau["schedule"] = [
        row for row in missing_tau["schedule"]
        if float(row["tau_recovery_s"]) != 100.0
    ]
    with pytest.raises(RuntimeError, match="missing for a registered tau"):
        _r2_parity(cfg, r2, missing_tau)


def test_canonical_artifact_is_complete_and_fail_closed() -> None:
    path = RESULT / "m_gated_reserve_recovery_summary.json"
    if not path.is_file():
        pytest.skip("canonical producer has not run yet")

    def reject_constant(value: str) -> None:
        raise AssertionError(f"non-standard JSON constant {value}")

    summary = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    assert summary["status"] in {SUPPORTED, CLEAN_NO_GO}
    assert summary["registered_sensor_cell_count"] == 24
    assert summary["path_row_count"] == summary["expected_path_row_count"] == 432
    assert summary["plot_status"] == "complete"
    assert summary["peak_rss_kib"] < 1.5 * 1024 * 1024
    assert Path(ROOT / summary["artifacts"]["figure"]).is_file()
    assert (RESULT / "figures/README.md").is_file()
    if summary["status"] == SUPPORTED:
        assert all(summary["gates"].values())
        assert summary["accepted_tau_slow_by_tau_fast"] == {
            "20.0": [80.0, 90.0, 100.0],
            "15.0": [80.0, 90.0, 100.0],
            "25.0": [80.0, 90.0, 100.0],
        }
