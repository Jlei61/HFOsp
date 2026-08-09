from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "src", "snn_engine")):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_topic4_fcxr_lc4_gate as L4  # noqa: E402
import run_topic4_fcxr_lc4_lifecycle as LIFE  # noqa: E402
import run_topic4_fcxr_lc4b_deadzone as DZ  # noqa: E402


def test_deadzone_candidate_reaches_the_engine_config():
    c = DZ._candidate()
    cfg = L4._cfg(np.zeros(8), c)
    assert cfg["m_hill_deadzone"] == c["deadzone"]
    assert cfg["m_hill_K"] == c["K"]
    assert cfg["m_hill_n"] == 4.0


def test_old_hill_candidate_does_not_silently_gain_a_deadzone():
    old = L4._candidates()[0]
    cfg = L4._cfg(np.zeros(8), old)
    assert "m_hill_deadzone" not in cfg


def test_lifecycle_config_uses_the_same_locked_deadzone():
    c = DZ._candidate()
    cfg = LIFE._cfg(c)
    assert cfg["m_hill_deadzone"] == c["deadzone"]
    assert cfg["m_hill_K"] == c["K"]


def test_control_and_positive_control_artifacts_are_hash_locked():
    for path, expected in DZ.EXPECTED.items():
        assert path.is_file()
        assert DZ.sha256_file(path) == expected


def test_new_results_are_isolated_from_the_closed_lc4_root():
    assert DZ.OUT.endswith("lc4b_deadzone_lifecycle")
    assert str(DZ.OLD) != DZ.OUT


def test_positive_control_really_departs():
    row = DZ._positive_control()
    assert row["role"] == "positive_control"
    assert row["d_label"] == "D10"
    assert row["departed"] is True
    assert row["departure_ms"] == 7000.0
