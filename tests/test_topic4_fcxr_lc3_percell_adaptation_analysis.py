"""Tests for the per-cell adaptation adjudication and its two slow-variable readouts.

The verdict function guards the mistake the brake adjudication actually made: a bout whose end is
the end of the record read as a termination, so every arm -- including the one that never stops --
reported as terminated and silenced.  The mode decomposition guards the mirror-image mistake:
reading a lobe count into a change field that has no structure.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "_adapt_analysis",
    Path(__file__).resolve().parents[1]
    / "scripts/analyze_topic4_fcxr_lc3_percell_adaptation.py")
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules["_adapt_analysis"] = _MOD
_SPEC.loader.exec_module(_MOD)

RUN_MS = 100000.0


def _rec(**kw):
    base = dict(run_ms=RUN_MS, onset_ms=5000.0, offset_ms=RUN_MS, terminated=False,
                return_check=None)
    base.update(kw)
    return base


def test_a_bout_that_runs_to_the_end_of_the_record_is_not_a_termination():
    v, why = _MOD._verdict(_rec(offset_ms=RUN_MS, terminated=False))
    assert v == "never stopped"
    assert "end of the record" in why


def test_no_bout_is_reported_as_never_entered_not_as_a_success():
    v, _ = _MOD._verdict(_rec(onset_ms=None, offset_ms=None))
    assert v == "never entered"


def test_terminating_without_recovery_is_kept_apart_from_closing_the_loop():
    v, why = _MOD._verdict(_rec(offset_ms=40000.0, terminated=True,
                                return_check=dict(returned=False, reason="rate below band")))
    assert v == "stopped but did not recover"
    assert "rate below band" in why


def test_closing_the_loop_requires_both_termination_and_return():
    v, _ = _MOD._verdict(_rec(offset_ms=40000.0, terminated=True,
                              return_check=dict(returned=True)))
    assert v == "closed the loop"


def test_a_terminated_bout_with_no_return_check_is_not_called_a_success():
    v, _ = _MOD._verdict(_rec(offset_ms=40000.0, terminated=True, return_check=None))
    assert v == "stopped but did not recover"


def test_entry_is_reported_broken_when_the_pre_entry_train_collapses():
    """An arm that terminates by never letting the tissue build up has not solved the problem."""
    assert _MOD._entry_intact(_rec(n_returning_before_onset=12))["looks_intact"] is True
    assert _MOD._entry_intact(_rec(n_returning_before_onset=1))["looks_intact"] is False


def _z(m_grid, t_ms):
    return {"snapshot_t_ms": np.asarray(t_ms, float),
            "m_grid": np.asarray(m_grid, float),
            "z_grid": np.ones_like(np.asarray(m_grid, float)),
            "m_mean": np.asarray([g.mean() for g in np.asarray(m_grid, float)])}


def _field(grid=16, lobes=0, amp=1.0, seed=0):
    y, x = np.mgrid[0:grid, 0:grid]
    if lobes == 0:
        return np.random.default_rng(seed).normal(0, 0.05, (grid, grid))
    f = np.zeros((grid, grid))
    centres = [(grid * 0.3, grid * 0.5)] if lobes == 1 else [(grid * 0.25, grid * 0.5),
                                                             (grid * 0.75, grid * 0.5)]
    for cx, cy in centres:
        f += amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * (grid * 0.09) ** 2))
    return f


def test_modes_recover_a_structured_change_and_report_its_share():
    t = np.arange(0, 12000, 250.0)
    flat, two = _field(lobes=0), _field(lobes=2, amp=3.0)
    grids = np.asarray([flat if tt < 5000 else flat + two for tt in t])
    md = _MOD._modes(_z(grids, t), dict(onset_ms=6000.0))
    assert md is not None
    assert md["share1"] > 0.7, "a clean two-lobe change should be carried by one mode"
    assert np.corrcoef(md["mode1"].ravel(), two.ravel())[0, 1] > 0.9


def test_modes_do_not_manufacture_structure_from_an_unstructured_change():
    """The failure this guards: reading a lobe count into noise."""
    rng = np.random.default_rng(3)
    t = np.arange(0, 12000, 250.0)
    grids = np.asarray([rng.normal(0, 1.0, (16, 16)) for _ in t])
    md = _MOD._modes(_z(grids, t), dict(onset_ms=6000.0))
    assert md["share1"] < 0.5, "unstructured change must not read as one dominant mode"


def test_modes_return_none_when_there_was_no_entry_to_look_around():
    t = np.arange(0, 12000, 250.0)
    grids = np.asarray([_field(lobes=1) for _ in t])
    assert _MOD._modes(_z(grids, t), dict(onset_ms=None)) is None


def test_modes_return_none_when_the_record_ends_before_the_pre_entry_window():
    t = np.arange(0, 800, 250.0)                      # shorter than the 1-4 s baseline window
    grids = np.asarray([_field(lobes=1) for _ in t])
    assert _MOD._modes(_z(grids, t), dict(onset_ms=600.0)) is None
