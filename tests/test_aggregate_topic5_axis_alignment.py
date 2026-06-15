"""TDD for the patient-level effect-size / bootstrap-CI / rank-biserial helper
added to scripts/aggregate_topic5_axis_alignment.py.

These tests pin only the pure statistical helper `_effect_stats(diffs, seed, n_boot)`,
which takes a list of per-subject (real - null) diffs and returns
{effect_size, ci_lo, ci_hi, rank_biserial}.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from aggregate_topic5_axis_alignment import _effect_stats  # noqa: E402


def test_constant_diffs_ci_does_not_collapse_and_equals_value():
    diffs = [0.1] * 18
    out = _effect_stats(diffs, seed=0, n_boot=2000)
    assert out["effect_size"] == 0.1
    # every bootstrap resample of a constant vector has median 0.1
    assert abs(out["ci_lo"] - 0.1) < 1e-12
    assert abs(out["ci_hi"] - 0.1) < 1e-12
    assert -1.0 <= out["rank_biserial"] <= 1.0


def test_all_positive_diffs_effect_and_ci_positive_rb_near_plus_one():
    diffs = [0.2, 0.3, 0.25, 0.18, 0.22, 0.27, 0.21, 0.24, 0.19, 0.26]
    out = _effect_stats(diffs, seed=1, n_boot=5000)
    assert out["effect_size"] > 0
    assert out["ci_lo"] > 0
    assert out["ci_hi"] > 0
    # all diffs positive => W- = 0 => rank-biserial = +1
    assert abs(out["rank_biserial"] - 1.0) < 1e-9


def test_all_negative_diffs_rank_biserial_near_minus_one():
    diffs = [-0.2, -0.3, -0.25, -0.18, -0.22]
    out = _effect_stats(diffs, seed=2, n_boot=2000)
    assert out["effect_size"] < 0
    # all diffs negative => W+ = 0 => rank-biserial = -1
    assert abs(out["rank_biserial"] - (-1.0)) < 1e-9


def test_rank_biserial_in_range_and_direction_for_mixed():
    # mostly positive with two small negatives -> rb positive but < 1
    diffs = [0.3, 0.4, 0.2, 0.35, 0.1, -0.05, -0.02, 0.25]
    out = _effect_stats(diffs, seed=3, n_boot=2000)
    assert -1.0 <= out["rank_biserial"] <= 1.0
    assert out["rank_biserial"] > 0
    assert out["rank_biserial"] < 1.0

    # mirror: mostly negative -> rb negative
    diffs_neg = [-d for d in diffs]
    out_neg = _effect_stats(diffs_neg, seed=3, n_boot=2000)
    assert -1.0 <= out_neg["rank_biserial"] <= 1.0
    assert out_neg["rank_biserial"] < 0
    assert abs(out_neg["rank_biserial"] + out["rank_biserial"]) < 1e-9


def test_fixed_seed_is_deterministic():
    diffs = [0.2, -0.1, 0.3, 0.05, -0.02, 0.15, 0.22, -0.03, 0.11, 0.18]
    a = _effect_stats(diffs, seed=42, n_boot=4000)
    b = _effect_stats(diffs, seed=42, n_boot=4000)
    assert a["ci_lo"] == b["ci_lo"]
    assert a["ci_hi"] == b["ci_hi"]
    assert a["effect_size"] == b["effect_size"]
    assert a["rank_biserial"] == b["rank_biserial"]
