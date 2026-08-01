import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_topic4_zm_lifecycle_m_panel.py"
SPEC = importlib.util.spec_from_file_location("topic4_zm_lifecycle_m_panel", SCRIPT)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


def _selected(index):
    return {
        "config_id": f"c{index}", "arm": "combined" if index % 2 else "i2e",
        "tau_D_ms": 400.0 + index, "d_star": 0.6,
        "tau_aI_ms": 100.0, "f_aI": 0.05, "strength_scale": 1.0,
    }


def test_m_panel_has_four_by_nine_distinct_coordinates():
    manifest = M.build_manifest({"rows": [_selected(i) for i in range(4)]})
    assert manifest["n_configs"] == 36
    assert len({row["config_id"] for row in manifest["rows"]}) == 36
    for rank in range(4):
        rows = [row for row in manifest["rows"] if row["selection_rank"] == rank]
        assert len(rows) == 9
        assert len({(row["g_M"], row["tau_M_ms"]) for row in rows}) == 9
        assert sum(row["g_M"] == 0 for row in rows) == 1


def test_m_panel_command_is_a_twenty_second_dynamic_sprint():
    manifest = M.build_manifest({"rows": [_selected(i) for i in range(4)]})
    cmd = M.B._command(manifest["rows"][0])
    assert "sprint-cell" in cmd
    assert cmd[cmd.index("--T-ms") + 1] == "20000.0"
