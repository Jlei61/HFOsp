import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "lc4_lifecycle_runner", ROOT / "scripts" / "run_topic4_fcxr_lc4_lifecycle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_import_has_no_simulation_side_effect(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mod = _load()
    assert mod.NOMINAL_MS == 70000.0
    assert not list(tmp_path.iterdir())


def test_candidate_fails_closed_without_gates(tmp_path, monkeypatch):
    mod = _load()
    monkeypatch.setattr(mod, "OUT", str(tmp_path))
    with pytest.raises(SystemExit, match="complete F0 and F1"):
        mod._candidate()


def test_cfg_uses_candidate_without_tuning(monkeypatch):
    mod = _load()
    monkeypatch.setattr(mod.GEO, "_point", lambda _x: {})
    monkeypatch.setattr(mod.E01, "_dynamic_cfg", lambda _p: {"use_z": True, "use_x": True})
    c = dict(tau_adp_ms=1000.0, K=45.0, n=6, tau_a_on_ms=100.0,
             tau_a_off_ms=10000.0, g_m_max=49.0)
    cfg = mod._cfg(c)
    assert cfg["use_m"] is True
    assert cfg["eta_m"] == 0.0
    assert cfg["m_hill_n"] == 6.0
    assert cfg["g_m_max"] == 49.0


def test_cfg_optionally_overrides_h_entry_threshold(monkeypatch):
    mod = _load()
    monkeypatch.setattr(mod.GEO, "_point", lambda _x: {})
    monkeypatch.setattr(mod.E01, "_dynamic_cfg", lambda _p: {"theta_h_lc2": 1.0})
    c = dict(tau_adp_ms=1000.0, K=45.0, n=4, tau_a_on_ms=100.0,
             tau_a_off_ms=10000.0, g_m_max=49.0, theta_h_lc2=1.7317)
    assert mod._cfg(c)["theta_h_lc2"] == pytest.approx(1.7317)
