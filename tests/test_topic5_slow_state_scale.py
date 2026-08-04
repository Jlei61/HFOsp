import numpy as np
import pytest

from src.topic5_slow_state_scale import (
    scale_states,
    select_scales,
    window_agreements,
    window_state,
)

B, R, C, F = "BELOW_CHANCE", "RELIABLE", "CHRONOLOGY_BREAK", "UNRESOLVED_TOO_FEW_WINDOWS"
FAMILIES = ("participation", "mean_rank", "precedence")


# ---------------------------------------------------------------------------
# select_scales — given verbatim in task-6-brief.md Step 1 (transcribed exactly;
# this is the piece an earlier draft got wrong, so these tests are not rewritten).
# ---------------------------------------------------------------------------


def test_n_obs_is_the_smallest_reliable_scale_not_the_largest():
    out = select_scales({50: B, 100: R, 200: R, 500: C, 1000: C})
    assert out["n_obs"] == 100
    assert out["n_break"] == 500
    assert out["status"] == "SCALE_RESOLVED"


def test_a_leading_run_of_below_chance_windows_does_not_force_unresolved():
    # rev1's contiguous-prefix rule wrongly returned unresolved here
    out = select_scales({50: B, 100: B, 200: R, 500: C})
    assert out["status"] == "SCALE_RESOLVED"
    assert out["n_obs"] == 200


def test_the_dwell_is_an_interval_between_the_last_reliable_and_the_break():
    out = select_scales({50: B, 100: R, 200: R, 500: C})
    assert out["n_last_reliable"] == 200
    assert out["dwell_interval"] == (200, 500)


def test_a_dwell_interval_is_open_ended_when_no_break_is_reached():
    out = select_scales({50: B, 100: R, 200: R, 500: R})
    assert out["n_break"] is None
    assert out["dwell_interval"] == (500, None)


def test_reliability_returning_after_a_break_is_reported_not_coerced():
    out = select_scales({50: R, 100: C, 200: R})
    assert out["status"] == "UNRESOLVED_NONMONOTONE"
    assert out["n_obs"] is None


def test_no_reliable_scale_anywhere_is_unresolved_scale():
    out = select_scales({50: B, 100: B, 200: B})
    assert out["status"] == "UNRESOLVED_SCALE"
    assert out["n_obs"] is None


def test_scales_with_too_few_windows_are_skipped_not_counted_as_failures():
    out = select_scales({50: F, 100: R, 200: R, 500: C})
    assert out["status"] == "SCALE_RESOLVED"
    assert out["n_obs"] == 100


def test_a_scale_gap_from_too_few_windows_does_not_break_monotonicity():
    out = select_scales({50: B, 100: R, 200: F, 500: R, 1000: C})
    assert out["status"] == "SCALE_RESOLVED"
    assert out["n_obs"] == 100
    assert out["n_break"] == 1000


# ---------------------------------------------------------------------------
# window_agreements — not given as code in the brief; tests below pin the
# contract stated in the Interfaces block plus the task's explicit numeric
# requirements ("BEYOND THE BRIEF" items 1-2).
# ---------------------------------------------------------------------------


def _consistent_window():
    """60 events, 5 contacts. Rank and recruitment-group order are identical
    for every single event, so any split of this window reproduces the same
    repertoire descriptors exactly. The only way two halves can disagree is
    if contact identity itself is scrambled — which is exactly what
    `contact_null` is supposed to do, and a (deliberately broken) null that
    permutes event order instead would not do.

    Participation is above the floor (5) for every contact but deliberately
    varies across contacts (60, 54, 48, 42, 36 events out of 60, spread
    evenly via linspace so no half-split starves a contact below floor) so
    that `participation_rate` itself has cross-contact variance -- a
    perfectly uniform 100%-for-everyone participation vector has zero
    variance and Spearman agreement on it is undefined (None) by
    `family_agreement`'s own zero-variance guard, which would make this
    fixture unable to exercise the "participation" family at all.
    """
    n_events, n_contacts = 60, 5
    rank = np.tile(np.array([0.1, 0.3, 0.5, 0.7, 0.9]), (n_events, 1))
    groups = np.array([0, 0, 1, 2, 3], dtype=np.int16)  # contacts 0 and 1 tied
    group_ids = np.tile(groups, (n_events, 1))
    participation = np.ones((n_events, n_contacts), dtype=np.uint8)
    drop_fraction = [0.0, 0.1, 0.2, 0.3, 0.4]
    for contact, frac in enumerate(drop_fraction):
        n_drop = int(round(n_events * frac))
        if n_drop:
            drop_idx = np.linspace(0, n_events - 1, n_drop, dtype=int)
            participation[drop_idx, contact] = 0
    return rank, participation, group_ids


_FLOORS = {"min_participation_count": 5, "min_pair_count": 5}


def test_window_agreements_returns_one_entry_per_family():
    rank, participation, group_ids = _consistent_window()
    out = window_agreements(
        rank,
        participation,
        group_ids,
        random_half_draws=8,
        null_draws=6,
        seed=0,
        floors=_FLOORS,
    )
    assert set(out["random_half"]) == set(FAMILIES)
    assert set(out["chronological"]) == set(FAMILIES)
    assert set(out["contact_null"]) == set(FAMILIES)
    for family in FAMILIES:
        assert len(out["random_half"][family]) == 8
        assert len(out["contact_null"][family]) == 6


def test_the_contact_null_permutes_contact_identity_not_event_order():
    rank, participation, group_ids = _consistent_window()
    out = window_agreements(
        rank,
        participation,
        group_ids,
        random_half_draws=40,
        null_draws=40,
        seed=1,
        floors=_FLOORS,
    )
    assert np.median(out["contact_null"]["mean_rank"]) < np.median(
        out["random_half"]["mean_rank"]
    )


# ---------------------------------------------------------------------------
# window_state — synthetic agreements dicts handed directly, per the task's
# instruction not to route these through real data.
# ---------------------------------------------------------------------------


def test_window_state_is_below_chance_when_random_half_does_not_beat_the_null():
    agreements = {
        "random_half": {family: [0.1] * 20 for family in FAMILIES},
        "chronological": {family: None for family in FAMILIES},
        "contact_null": {family: [0.5] * 20 for family in FAMILIES},
    }
    assert window_state(agreements, alpha=0.05, min_resolved_families=2) == "BELOW_CHANCE"


def test_window_state_is_chronology_break_when_the_chronological_value_sits_low():
    agreements = {
        "random_half": {family: [0.9] * 20 for family in FAMILIES},
        "chronological": {family: 0.0 for family in FAMILIES},
        "contact_null": {family: [0.1] * 20 for family in FAMILIES},
    }
    assert (
        window_state(agreements, alpha=0.05, min_resolved_families=2)
        == "CHRONOLOGY_BREAK"
    )


def test_window_state_is_unresolved_when_too_few_families_resolve():
    agreements = {
        "random_half": {
            "participation": [],
            "mean_rank": [],
            "precedence": [0.9] * 10,
        },
        "chronological": {
            "participation": None,
            "mean_rank": None,
            "precedence": 0.85,
        },
        "contact_null": {
            "participation": [],
            "mean_rank": [],
            "precedence": [0.1] * 10,
        },
    }
    assert (
        window_state(agreements, alpha=0.05, min_resolved_families=2)
        == "UNRESOLVED_FAMILIES"
    )


# ---------------------------------------------------------------------------
# scale_states
# ---------------------------------------------------------------------------


def test_scale_states_returns_too_few_windows_below_the_minimum():
    four_windows = [R, R, R, B]
    assert scale_states(four_windows, min_windows=5) == "UNRESOLVED_TOO_FEW_WINDOWS"

    five_windows = [R, R, R, R, B]
    assert scale_states(five_windows, min_windows=5) == R
