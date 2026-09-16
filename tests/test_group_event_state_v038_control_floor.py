"""Null-arm and provenance contracts for the v0.3.8 review-repair summary.

Three defects survived the repair round, all of the same family the earlier
rounds already fixed elsewhere:

1.  A control arm may score *worse* than the strong baseline it is nested
    above.  When it does, "the state beats its own constant / its matched
    random component" measures the control's harm, not the state's skill.  In
    this cohort the random component is 0.34 log-score worse than the baseline
    for the retained short-scale candidate, so two thirds of its headline
    contrast is control harm.  The reportable contrast floors every control at
    the strong baseline.
2.  The v0.3.8 finaliser dropped the code-provenance audit that v0.3.7 gained,
    so a summary can silently pool cards written by two different versions of
    the training code.
3.  The innermost rate baseline can select its checkpoint at the origin -- the
    whole nested chain then sits on a readout that contributes nothing -- and
    the stage audit still marks the unit ``qualified``.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FINALISER = REPO_ROOT / "scripts/finalize_group_event_state_v038.py"


def _finaliser_source() -> str:
    return FINALISER.read_text(encoding="utf-8")


# ------------------------------------------------------------------ 1. floored nulls


def test_control_contrasts_are_floored_at_the_strong_baseline() -> None:
    source = _finaliser_source()
    assert "dynamic_over_constant_floored" in source
    assert "gain_over_random_floored" in source
    assert "control_worse_than_strong_baseline" in source, (
        "a control that scores worse than the baseline is harmful, not null, and the "
        "summary must say so next to the contrast it inflates"
    )


def test_floor_helper_takes_the_better_of_control_and_baseline() -> None:
    module = _load_finaliser()
    floor = module._floored_control
    # control worse than the baseline: the null collapses onto the baseline, so
    # the reportable gain is the clean nested gain.
    assert floor(control_gain=0.52, baseline_gain=0.176) == pytest.approx(0.176)
    # control better than the baseline: nothing to floor, the contrast stands.
    assert floor(control_gain=0.043, baseline_gain=0.489) == pytest.approx(0.043)
    assert floor(control_gain=None, baseline_gain=0.4) is None
    assert floor(control_gain=0.4, baseline_gain=None) is None


def test_floor_never_reports_more_than_the_raw_contrast() -> None:
    module = _load_finaliser()
    floor = module._floored_control
    for control, baseline in ((0.2, 0.1), (0.1, 0.2), (-0.3, 0.1), (0.0, 0.0)):
        assert floor(control_gain=control, baseline_gain=baseline) <= control + 1e-12


# ------------------------------------------------------------------ 2. provenance


def test_finaliser_audits_the_code_version_of_every_card() -> None:
    source = _finaliser_source()
    assert "_code_provenance_audit" in source, (
        "v0.3.7 gained this audit and v0.3.8 dropped it; a summary that pools two "
        "versions of the training code cannot be checked afterwards"
    )
    assert "mixed_source_versions" in source
    assert "cards_without_code_provenance" in source


def test_provenance_audit_flags_two_versions_of_one_file() -> None:
    module = _load_finaliser()
    audit = module._code_provenance_audit([
        {"code_provenance": {"source_file": "h2a.py", "source_sha256": "a" * 64}},
        {"code_provenance": {"source_file": "h2a.py", "source_sha256": "b" * 64}},
        {"code_provenance": {"source_file": "h1_train.py", "source_sha256": "c" * 64}},
        {"status": "NOT_ESTIMABLE"},
    ])
    assert audit["cards_without_code_provenance"] == 1
    assert audit["single_version_per_source_file"] is False
    assert "h2a.py" in audit["mixed_source_versions"]
    assert "h1_train.py" not in audit["mixed_source_versions"]


def test_provenance_audit_passes_a_single_version_run() -> None:
    module = _load_finaliser()
    audit = module._code_provenance_audit([
        {"code_provenance": {"source_file": "h1_train.py", "source_sha256": "c" * 64}},
        {"code_provenance": {"source_file": "h1_train.py", "source_sha256": "c" * 64}},
    ])
    assert audit["single_version_per_source_file"] is True
    assert audit["mixed_source_versions"] == {}
    assert audit["cards_without_code_provenance"] == 0


# ------------------------------------------------------------------ 3. foundation stage


def test_stage_audit_marks_a_rate_baseline_that_never_left_the_origin() -> None:
    source = _finaliser_source()
    assert "foundation_stage_at_origin" in source, (
        "the rate stage is the innermost nested control; selecting it at the origin "
        "leaves every arm above it standing on a readout that contributes nothing"
    )


def _load_finaliser():
    import importlib.util

    spec = importlib.util.spec_from_file_location("v038_finaliser", FINALISER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_finaliser_module_still_imports() -> None:
    module = _load_finaliser()
    assert inspect.isfunction(module._code_provenance_audit)
    assert inspect.isfunction(module._floored_control)
