import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "paper_figures" / "plot_fig5_mz_divisive_current_stage.py"
SPEC = importlib.util.spec_from_file_location("plot_fig5_mz_divisive_current_stage", SCRIPT)
PLOT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PLOT)


def test_causal_recruited_episode_uses_trailing_samples_and_one_second_hold():
    dt_ms = 10.0
    rate = np.zeros(300, float)
    rate[100:250] = 30.0

    state, envelope = PLOT._causal_recruited_episode(rate, dt_ms)

    assert state["status"] == "recruited_macrostate"
    # A 25-sample trailing window first contains 17 active samples at index 116:
    # 17 * 30 / 25 = 20.4 Hz.  The old centered detector crossed at 1040 ms.
    assert state["onset_ms"] == 1160.0
    assert state["confirmed_ms"] == 2160.0
    assert state["envelope_support"] == "trailing_only"
    assert envelope[115] < 20.0 <= envelope[116]


def test_causal_recruited_episode_rejects_short_crossing():
    rate = np.zeros(300, float)
    rate[100:180] = 30.0
    state, _ = PLOT._causal_recruited_episode(rate, 10.0)
    assert state["status"] == "no_recruited_macrostate"
    assert state["onset_ms"] is None


def test_trailing_envelope_has_no_future_leakage():
    original = np.zeros(20, float)
    changed_future = original.copy()
    changed_future[10:] = 100.0
    a = PLOT._causal_trailing_mean_1d(original, 5)
    b = PLOT._causal_trailing_mean_1d(changed_future, 5)
    np.testing.assert_array_equal(a[:10], b[:10])

