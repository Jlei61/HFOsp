"""Contract tests for the per-seed core_only spontaneous-ignition audit
(scripts/audit_m3_core_only_seed_confounds.py).

Reviewer reframe (2026-06-23): the substrate-level majority-vote core_only_quiet
gate is CORRECT and must NOT be rewritten. The real problem is that core_only was
summarized by the MEAN — one spontaneously-igniting seed (w18.0: co_ds=1282 vs the
other 7 seeds ~19-51) dragged the mean to 188.8 while the median (32) ~= the bare
sheet. So the fix lives in the ANALYSIS layer: a per-seed spontaneous-ignition flag
+ a robust (median) summary + seed-level contamination tracking, so a single
igniting seed can be excluded as a sensitivity check (and never pollute W_event).

These tests pin the science contract of the flag and the robust summary on
synthetic per-seed core_only values (no SNN).
"""
import importlib.util
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPT = os.path.join(_HERE, "..", "scripts", "audit_m3_core_only_seed_confounds.py")
_spec = importlib.util.spec_from_file_location("m3_confounds", _SCRIPT)
m3 = importlib.util.module_from_spec(_spec)
sys.modules["m3_confounds"] = m3
_spec.loader.exec_module(m3)


# --------------------------------------------------------------------------- #
# per-seed spontaneous-ignition flag                                          #
# --------------------------------------------------------------------------- #
def test_large_spontaneous_event_is_flagged():
    # The w18.0 contaminated seed: co_only downstream 1282 with bare background ~30.
    assert m3.spontaneous_ignition_flag(1282.0, 30.0) is True


def test_normal_seed_not_flagged():
    # The largest NON-igniting w18.0 seed (51) must NOT be flagged.
    assert m3.spontaneous_ignition_flag(51.0, 30.0) is False
    # a bare-like seed at the bare-sheet background is obviously not igniting
    assert m3.spontaneous_ignition_flag(30.0, 30.0) is False
    # even the loudest bare background slice (48) stays unflagged
    assert m3.spontaneous_ignition_flag(48.0, 30.0) is False


def test_relative_ratio_protects_high_background():
    # If the bare sheet itself runs hot (bg=80), a moderately elevated core_only
    # (120 = 1.5x bg, below the 3x ratio AND the test would falsely fire on a bare
    # absolute floor alone) must NOT be flagged — the ratio dominates the floor.
    assert m3.spontaneous_ignition_flag(120.0, 80.0) is False
    # but a genuine ignition well above 3x the hot background IS flagged
    assert m3.spontaneous_ignition_flag(400.0, 80.0) is True


# --------------------------------------------------------------------------- #
# robust per-substrate summary: median (not mean) + seed-level contamination  #
# --------------------------------------------------------------------------- #
def test_summary_uses_median_not_mean_and_counts_one_igniter():
    # Exactly the real w18.0 per-seed core_only values (kick-independent).
    co = [19.0, 24.0, 29.0, 32.0, 33.0, 40.0, 51.0, 1282.0]
    s = m3.summarize_core_only(co, bare_bg_med=30.0)
    assert s["n_seeds"] == 8
    assert s["n_spontaneous"] == 1                  # only the 1282 seed
    assert s["median"] == pytest.approx(32.5)       # robust center ~= bare
    assert s["mean"] == pytest.approx(188.75)       # the misleading mean
    # the robust center must be close to the bare background, unlike the mean
    assert abs(s["median"] - 30.0) < 10.0
    assert s["mean"] > 5 * s["median"]              # mean is distorted by the outlier


def test_clean_substrate_has_zero_igniters():
    # A genuinely quiet narrow core (all seeds ~ bare background): 0 igniters.
    co = [19.0, 24.0, 28.0, 29.0, 32.0, 38.0, 43.0, 48.0]
    s = m3.summarize_core_only(co, bare_bg_med=30.0)
    assert s["n_spontaneous"] == 0
    assert s["median"] == pytest.approx(30.5)
