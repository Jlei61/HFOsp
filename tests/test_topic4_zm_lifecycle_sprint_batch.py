import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_topic4_zm_lifecycle_sprint_batch.py"
SPEC = importlib.util.spec_from_file_location("topic4_zm_lifecycle_sprint_batch", SCRIPT)
BATCH = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BATCH)


def test_batch1_manifest_is_deterministic_unique_and_in_registered_ranges():
    a, b = BATCH.build_manifest(), BATCH.build_manifest()
    assert a["rows"] == b["rows"]
    assert a["n_configs"] == 36
    assert len({row["config_id"] for row in a["rows"]}) == 36
    assert sum(row["family"] == "depression_only_lhs" for row in a["rows"]) == 16
    assert sum(row["family"] == "combined_lhs" for row in a["rows"]) == 16
    for row in a["rows"]:
        assert 300 <= row["tau_D_ms"] <= 850
        assert 0.55 <= row["d_star"] <= 0.85
        assert row["g_M"] == 1 and row["T_ms"] == 12000
        if row["arm"] == "combined":
            assert 60 <= row["tau_aI_ms"] <= 350
            assert 0 <= row["f_aI"] <= 0.12


def test_worker_command_uses_full_dynamic_sprint_and_single_thread_contract():
    row = BATCH.build_manifest()["rows"][20]
    cmd = BATCH._command(row)
    assert "sprint-cell" in cmd
    assert "--confirm-run" in cmd
    assert cmd[cmd.index("--T-ms") + 1] == "12000.0"


def test_existing_control_artifact_reuse_requires_matching_scientific_coordinates(tmp_path):
    row = {
        "arm": "i2e", "tau_D_ms": 300.7, "d_star": 0.7281,
        "strength_scale": 1.0, "g_M": 1.0, "tau_M_ms": 2000.0,
        "g_Z": 1.0, "T_ms": 20000.0, "burn_ms": 1000.0,
        "control_clock": "relative_to_pre_entry_checkpoint_v2",
        "control_target": "all_E", "control_uplift_mV": 4.0,
        "control_t0_ms": 2520.0, "control_duration_ms": 50.0,
    }
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({
        "T_ms": 20000.0, "equilibration_ms": 1000.0,
        "mechanism": {
            "arm": "i2e", "strength_scale": 1.0,
            "i2e_depression": {"tau_D_ms": 300.7, "d_star_nominal": 0.7281},
            "dynamic_slow_flow": {"g_M": 1.0, "tau_M_ms": 2000.0, "g_Z": 1.0},
        },
        "finite_control": {
            "clock": "relative_to_pre_entry_checkpoint_v2", "target": "all_E",
            "uplift_mV": 4.0, "t0_ms": 2520.0, "duration_ms": 50.0,
        },
    }))
    assert BATCH._existing_artifact_matches(row, summary)
    row["control_duration_ms"] = 200.0
    assert not BATCH._existing_artifact_matches(row, summary)


def test_expected_control_artifact_uses_clock_versioned_stem(monkeypatch, tmp_path):
    monkeypatch.setattr(BATCH, "OUT", tmp_path)
    row = {
        "arm": "i2e", "tau_D_ms": 300.7, "d_star": 0.7281,
        "strength_scale": 1.0, "g_M": 1.0, "tau_M_ms": 2000.0,
        "g_Z": 1.0, "T_ms": 20000.0, "burn_ms": 1000.0,
        "control_target": "all_E", "control_uplift_mV": 4.0,
        "control_t0_ms": 2520.0, "control_duration_ms": 50.0,
    }
    assert BATCH._expected_artifact_path(row).name == "summary.json"
    assert "__ctlall_E__u4__t2520__dur50__clkrel2__T20s__gM1__tauM2000" in str(
        BATCH._expected_artifact_path(row)
    )
