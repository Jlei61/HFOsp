from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "src", "snn_engine")):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_topic4_fcxr_lc4_gate as R  # noqa: E402


def test_locked_candidate_family_is_only_n6_and_n8_and_is_force_matched():
    rows = R._candidates()
    assert [r["n"] for r in rows] == [6.0, 8.0]
    delivered = [r["g_m_max"] * r["ictal_activation"] for r in rows]
    assert delivered[0] == pytest.approx(delivered[1])


def test_control_has_no_cooperative_mechanism_and_candidate_does():
    import numpy as np
    d = np.zeros(8)
    control = R._cfg(d, None)
    candidate = R._cfg(d, R._candidates()[0])
    assert not control["use_m"]
    assert control["use_z"] is False and control["z_frozen_E"].shape == (8,)
    assert candidate["use_m"] and candidate["m_hill_n"] == 6.0
    assert candidate["tau_a_off"] == 10000.0


def test_long_rows_are_single_worker_by_construction():
    # There is no workers CLI and the stage loops synchronously.  This constant-level test catches
    # accidental shortening of the registered observation windows.
    assert R.BASELINE_MS == 12000.0 and R.ONSET_MS == 12000.0
