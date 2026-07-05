"""Contract tests for the ceiling B2 seed-level statistics
(scripts/analyze_m3_ceiling_b2_stats.py).

Reviewer P1 (2026-06-23): the K_min 1.1 vs 1.6 number is the COHORT P_EA>=0.7
crossing of the proportion curve — it is NOT a seed-level threshold that dropped
0.5. The seed-level reality is a paired sign test (most seeds advance ~one kick
grid step, none delayed) with right-censoring (some bare seeds never cross within
the scan). These tests pin the seed-level primitives on synthetic data (no SNN).
"""
import importlib.util
import math
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPT = os.path.join(_HERE, "..", "scripts", "analyze_m3_ceiling_b2_stats.py")
_spec = importlib.util.spec_from_file_location("m3_b2stats", _SCRIPT)
m3 = importlib.util.module_from_spec(_spec)
sys.modules["m3_b2stats"] = m3
_spec.loader.exec_module(m3)


# --------------------------------------------------------------------------- #
# first_crossing_kick: first kick at which the seed becomes EA-local-returned #
# --------------------------------------------------------------------------- #
def test_first_crossing_returns_first_local_kick():
    kicks = [1.0, 1.1, 1.2, 1.3]
    assert m3.first_crossing_kick({1.0: 0, 1.1: 1, 1.2: 1, 1.3: 1}, kicks) == 1.1


def test_first_crossing_censored_when_never_local():
    kicks = [1.0, 1.1, 1.2]
    assert math.isinf(m3.first_crossing_kick({1.0: 0, 1.1: 0, 1.2: 0}, kicks))


def test_first_crossing_takes_FIRST_even_if_non_monotonic():
    # a seed local at 1.1, not at 1.2, local again at 1.3 -> first crossing is 1.1
    kicks = [1.0, 1.1, 1.2, 1.3]
    assert m3.first_crossing_kick({1.0: 0, 1.1: 1, 1.2: 0, 1.3: 1}, kicks) == 1.1


# --------------------------------------------------------------------------- #
# paired_sign_counts: (core earlier, same, core later), censoring-aware        #
# --------------------------------------------------------------------------- #
def test_paired_sign_counts_basic():
    core = {0: 1.0, 1: 1.1}
    bare = {0: 1.1, 1: 1.1}
    n_earlier, n_same, n_later = m3.paired_sign_counts(core, bare)
    assert (n_earlier, n_same, n_later) == (1, 1, 0)   # seed0 core earlier, seed1 same


def test_paired_sign_counts_censored_bare_counts_as_core_earlier():
    # bare never crosses (inf), core crosses at 1.4 -> core earlier
    core = {0: 1.4}
    bare = {0: math.inf}
    assert m3.paired_sign_counts(core, bare) == (1, 0, 0)


def test_paired_sign_counts_both_censored_is_same():
    core = {0: math.inf}
    bare = {0: math.inf}
    assert m3.paired_sign_counts(core, bare) == (0, 1, 0)


# --------------------------------------------------------------------------- #
# binomial_sign_p: exact two-sided sign test on DISCORDANT pairs only          #
# --------------------------------------------------------------------------- #
def test_binomial_sign_p_seven_zero():
    # 7 earlier, 0 later -> two-sided exact = 2 * 0.5**7
    assert m3.binomial_sign_p(7, 0) == pytest.approx(2 * 0.5 ** 7)


def test_binomial_sign_p_no_discordant_is_one():
    assert m3.binomial_sign_p(0, 0) == 1.0


def test_binomial_sign_p_symmetric_is_one():
    assert m3.binomial_sign_p(3, 3) == pytest.approx(1.0)
