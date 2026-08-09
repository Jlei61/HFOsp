from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "lc4d_runner", ROOT / "scripts" / "run_topic4_fcxr_lc4d.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_import_has_no_simulation_side_effect(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mod = _load()
    assert mod.SCREEN_MS == 18000.0
    assert not list(tmp_path.iterdir())


def test_candidate_fails_closed_without_passing_lock(tmp_path, monkeypatch):
    mod = _load()
    lock = tmp_path / "candidate_lock.json"
    lock.write_text(json.dumps({"status": "NO", "verdict": "NO"}))
    monkeypatch.setattr(mod, "LOCK", lock)
    with pytest.raises(SystemExit, match="passing L0"):
        mod._candidate()


def test_nominal_requires_passing_screen(tmp_path, monkeypatch):
    mod = _load()
    monkeypatch.setattr(mod, "OUT", tmp_path)
    with pytest.raises(SystemExit, match="L1 latency screen"):
        mod._require_l1()


def test_runner_delegates_candidate_to_unchanged_lifecycle_module():
    mod = _load()
    assert mod.LC4._candidate is mod._candidate
    assert mod.LC4.OUT.endswith("lc4d_offset_latency_alignment")
