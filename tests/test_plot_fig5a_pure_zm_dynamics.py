import pytest

from scripts.paper_figures.plot_fig5a_pure_zm_dynamics import (
    select_pure_zm_candidate,
)


def _row(candidate_id, *, primary=True, comparator=False, duty=1.0,
         frequency_shift=-2.0, clauses=None):
    if clauses is None:
        clauses = {
            "complete_one_second_recruitment": True,
            "joint_broad_recruitment_duty": True,
            "population_rate_ratio": True,
            "contact_frequency_increased": False,
            "numerically_safe": True,
        }
    return {
        "candidate_id": candidate_id,
        "primary_zm_only": primary,
        "edge_dose_comparator": comparator,
        "model_ictal_qualification": {
            "joint_duty": duty,
            "contact_centroid_shift_hz": frequency_shift,
            "clauses": clauses,
        },
    }


def test_selection_uses_full_edge_duty_not_comparator_or_frequency():
    rows = [
        _row("full_high_frequency", duty=0.75, frequency_shift=31.0),
        _row("full_sustained", duty=1.0, frequency_shift=-2.0),
        _row("pretty_comparator", primary=False, comparator=True,
             duty=1.0, frequency_shift=40.0),
    ]
    selected = select_pure_zm_candidate(rows)
    assert selected["candidate_id"] == "full_sustained"


def test_selection_fails_closed_without_complete_high_recruitment_candidate():
    clauses = {
        "complete_one_second_recruitment": False,
        "joint_broad_recruitment_duty": True,
        "population_rate_ratio": True,
        "numerically_safe": True,
    }
    with pytest.raises(RuntimeError, match="no numerically safe full-edge"):
        select_pure_zm_candidate([_row("incomplete", clauses=clauses)])
