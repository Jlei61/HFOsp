import importlib.util
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
