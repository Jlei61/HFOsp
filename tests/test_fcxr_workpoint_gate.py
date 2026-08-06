"""Regression: the FCXR workpoint candidate gate must require workpoint.all_bands=True.

Reviewer P1 (2026-07-20): the old filter accepted any settled-safe cell with returning events even if
its event profile was far off the reference bands, so an over-active c_E could be picked as a "workpoint".
"""
import importlib.util
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

_spec = importlib.util.spec_from_file_location(
    "run_topic4_mz_fcxr", os.path.join(ROOT, "scripts", "run_topic4_mz_fcxr.py"))
fcxr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fcxr)


def _cell(c_E, settled_safe, all_bands, n_ret, pheno, score):
    return dict(cfg=dict(c_E=c_E), numerical=dict(settled_safe=settled_safe),
                workpoint=dict(all_bands=all_bands, baseline_distance_score=score),
                event_profile=dict(n_returning=n_ret), phenotype=pheno)


def test_settled_safe_but_bands_false_is_not_a_candidate():
    """The exact bug: numerically safe + has events, but event profile off-band -> must NOT be picked."""
    rows = [_cell(1.0, settled_safe=True, all_bands=False, n_ret=58, pheno="interictal_like", score=3.7)]
    ranked, pick = fcxr._workpoint_candidates(rows)
    assert pick is None and ranked == []


def test_over_active_and_suppressed_both_rejected():
    rows = [_cell(1.0, False, False, 58, "interictal_like", 3.7),          # over-active + unsafe
            _cell(1.15, False, False, 45, "expanded_bounded", 10.9),       # over-active + unsafe
            _cell(0.85, True, False, 0, "suppress", math.nan)]             # safe but no events/bands
    ranked, pick = fcxr._workpoint_candidates(rows)
    assert pick is None


def test_all_bands_safe_cell_is_picked_by_min_score():
    rows = [_cell(0.90, True, True, 20, "interictal_like", 0.50),
            _cell(0.95, True, True, 22, "interictal_like", 0.30),          # lower score -> picked
            _cell(1.00, True, False, 58, "interictal_like", 3.70)]         # off-band -> excluded
    ranked, pick = fcxr._workpoint_candidates(rows)
    assert pick == 0.95
    assert [r["cfg"]["c_E"] for r in ranked] == [0.95, 0.90]


def test_runaway_even_if_bands_true_is_excluded():
    rows = [_cell(1.0, True, True, 20, "runaway", 0.4)]
    ranked, pick = fcxr._workpoint_candidates(rows)
    assert pick is None
