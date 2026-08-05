import numpy as np
import pytest

from src.topic5_slow_state_scale import (
    scale_states,
    select_scales,
    window_agreements,
    window_state,
)

B, R, C, F = "BELOW_CHANCE", "RELIABLE", "CHRONOLOGY_BREAK", "UNRESOLVED_TOO_FEW_WINDOWS"
UF = "UNRESOLVED_FAMILIES"
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


def test_an_unresolved_families_scale_is_dropped_not_counted_as_a_failure():
    # fix round 1, C2: UNRESOLVED_FAMILIES must be dropped before pattern matching, the
    # same as UNRESOLVED_TOO_FEW_WINDOWS -- otherwise a single unresolved scale (most
    # likely at the smallest grid point, where the support floors bite hardest) throws
    # the whole patient to UNRESOLVED_NONMONOTONE, exactly the failure mode the
    # leading-below-chance rule exists to prevent.
    out = select_scales({50: UF, 100: R, 200: R, 500: C})
    assert out["status"] == "SCALE_RESOLVED"
    assert out["n_obs"] == 100
    assert out["n_break"] == 500


def test_a_mixed_windows_scale_is_dropped_not_counted_as_a_failure():
    # coordinator follow-up: the previous round's C2 defect recurring under a new
    # label. UNRESOLVED_MIXED_WINDOWS (rev3 R3-B) is a scale that could not be decided,
    # not a scale that failed -- it must be dropped before pattern matching exactly like
    # UNRESOLVED_TOO_FEW_WINDOWS and UNRESOLVED_FAMILIES already are. Before this fix,
    # NOT_EVALUATED did not include it, so this exact input produced
    # UNRESOLVED_NONMONOTONE with n_obs=None -- one undecidable mid-grid scale throwing
    # the whole patient away, exactly the failure mode the leading-below-chance rule and
    # the original C2 fix exist to prevent.
    out = select_scales(
        {50: B, 100: R, 200: "UNRESOLVED_MIXED_WINDOWS", 500: R, 1000: C}
    )
    assert out["status"] == "SCALE_RESOLVED"
    assert out["n_obs"] == 100
    assert out["n_break"] == 1000


def test_dropping_a_mixed_scale_cannot_fabricate_a_reliable_run_that_was_never_observed():
    # the dropped scale must not silently extend the reliable run it sits inside: 200's
    # true label is unknown (it was undecidable, not observed to be RELIABLE), so
    # n_last_reliable must stay at the last CONFIRMED reliable grid point (100) and the
    # dwell interval (100, 500) must carry that, not silently widen to (200, 500) or
    # otherwise imply 200 was known to be reliable.
    out = select_scales({50: B, 100: R, 200: "UNRESOLVED_MIXED_WINDOWS", 500: C})
    assert out["status"] == "SCALE_RESOLVED"
    assert out["n_obs"] == 100
    assert out["n_last_reliable"] == 100
    assert out["n_break"] == 500


def test_family_discordance_reaching_select_scales_is_dropped_not_a_failure():
    # coordinator fix-round-4, ITEM 1: UNRESOLVED_FAMILY_DISCORDANCE cannot currently be
    # produced by scale_states (it is a window_state-level verdict only), so it cannot
    # reach select_scales through the normal window_state -> scale_states -> select_scales
    # pipeline today. But R3-C/R3-F will add the first real callers that may construct a
    # `states` mapping some other way, and there was no test pinning the behaviour if it
    # ever does arrive here -- this closes that gap directly, independent of how it gets
    # there. Must be dropped exactly like UNRESOLVED_MIXED_WINDOWS already is, not treated
    # as an observed, pattern-breaking state.
    out = select_scales(
        {50: B, 100: R, 200: "UNRESOLVED_FAMILY_DISCORDANCE", 500: R, 1000: C}
    )
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


def test_window_agreements_accepts_the_raw_config_floor_spelling():
    # fix round 1, I6: config/topic5_slow_state_v4_0.yaml spells the pair floor
    # min_pair_coparticipation_count; local_repertoire's parameter is min_pair_count.
    # Previously this was only checked by an ad-hoc `python -c` snippet, not a committed
    # test. Same seed, same window -> the two spellings must produce byte-identical
    # output, not just "no TypeError".
    rank, participation, group_ids = _consistent_window()
    raw_config_floors = {
        "min_participation_count": 5,
        "min_pair_coparticipation_count": 5,
    }
    renamed_floors = {"min_participation_count": 5, "min_pair_count": 5}

    out_raw = window_agreements(
        rank,
        participation,
        group_ids,
        random_half_draws=8,
        null_draws=6,
        seed=7,
        floors=raw_config_floors,
    )
    out_renamed = window_agreements(
        rank,
        participation,
        group_ids,
        random_half_draws=8,
        null_draws=6,
        seed=7,
        floors=renamed_floors,
    )
    assert out_raw == out_renamed


# ---------------------------------------------------------------------------
# window_state — synthetic agreements dicts handed directly, per the task's
# instruction not to route these through real data.
# ---------------------------------------------------------------------------


def test_window_state_is_below_chance_when_random_half_does_not_beat_the_null():
    # fix round 1, I5: constant lists make every percentile of a list coincide, so this
    # fixture could not tell a correct q95 threshold from a mutant reading q50 (both
    # trivially equal the one repeated value). Rewritten with SPREAD distributions:
    # contact_null's q50=0.45 and q95=0.855 genuinely differ (verified: np.percentile on
    # list(np.linspace(0.0, 0.9, 200))), and random_half's median (0.60) sits strictly
    # between them -- above chance under q50 (0.60 > 0.45), NOT above chance under the
    # correct q95 (0.60 < 0.855). Only the correct threshold gives BELOW_CHANCE.
    agreements = {
        "random_half": {family: list(np.linspace(0.55, 0.65, 200)) for family in FAMILIES},
        "chronological": {family: 0.6 for family in FAMILIES},
        "contact_null": {family: list(np.linspace(0.0, 0.9, 200)) for family in FAMILIES},
    }
    assert window_state(agreements, alpha=0.05, min_resolved_families=2) == "BELOW_CHANCE"


def test_window_state_is_chronology_break_when_the_chronological_value_sits_low():
    # fix round 1, I5: rewritten with a SPREAD random_half and a non-default alpha=0.30
    # so the test is sensitive to an "alpha ignored" mutant (e.g. one that hardcodes
    # 100*alpha as 5 regardless of what is passed). random_half's true alpha=0.30
    # quantile is 0.65 and its alpha=0.05 quantile is 0.525 (verified via
    # np.percentile(list(np.linspace(0.5, 1.0, 200)), ...)); chronological=0.59 sits
    # strictly between them, so only the CORRECT (passed) alpha=0.30 calls it a break.
    agreements = {
        "random_half": {family: list(np.linspace(0.5, 1.0, 200)) for family in FAMILIES},
        "chronological": {family: 0.59 for family in FAMILIES},
        "contact_null": {family: list(np.linspace(0.0, 0.1, 200)) for family in FAMILIES},
    }
    assert (
        window_state(agreements, alpha=0.30, min_resolved_families=2)
        == "CHRONOLOGY_BREAK"
    )


def test_window_state_is_unresolved_when_too_few_families_resolve():
    agreements = {
        "random_half": {
            "participation": [],
            "mean_rank": [],
            "precedence": [0.9] * 20,
        },
        "chronological": {
            "participation": None,
            "mean_rank": None,
            "precedence": 0.85,
        },
        "contact_null": {
            "participation": [],
            "mean_rank": [],
            "precedence": [0.1] * 20,
        },
    }
    assert (
        window_state(agreements, alpha=0.05, min_resolved_families=2)
        == "UNRESOLVED_FAMILIES"
    )


def test_a_tied_window_is_not_called_reliable():
    # fix round 1, C1: exactly two resolved families ("precedence" is left entirely
    # empty so only "participation" and "mean_rank" resolve). "participation" is
    # BELOW_CHANCE on its own (median 0.1 <= null q95 0.5) and "mean_rank" would be
    # CHRONOLOGY_BREAK on its own (above chance, 0.9 > 0.1, and chronological 0.0 sits
    # below any alpha-quantile of a constant-0.9 random_half). That is a 1-of-2 tie on
    # "above chance", which the old fallthrough silently called RELIABLE. The correct
    # precedence (BELOW_CHANCE unless a strict majority are above chance) must call this
    # BELOW_CHANCE, never RELIABLE.
    agreements = {
        "random_half": {
            "participation": [0.1] * 20,
            "mean_rank": [0.9] * 20,
            "precedence": [],
        },
        "chronological": {
            "participation": 0.3,
            "mean_rank": 0.0,
            "precedence": None,
        },
        "contact_null": {
            "participation": [0.5] * 20,
            "mean_rank": [0.1] * 20,
            "precedence": [],
        },
    }
    result = window_state(agreements, alpha=0.05, min_resolved_families=2)
    assert result != "RELIABLE"
    assert result == "BELOW_CHANCE"


def test_a_family_with_too_few_finite_draws_does_not_count_as_resolved():
    # fix round 1, I3: "precedence" has a full contact_null (20) and a full
    # chronological value, but only 5 finite random_half draws -- below
    # MIN_FINITE_DRAWS_FOR_RESOLUTION (20) -- so it must not count as resolved even
    # though it is not literally empty. With min_resolved_families=3 (all three
    # families required), only "participation" and "mean_rank" actually resolve, so
    # the window must be UNRESOLVED_FAMILIES.
    agreements = {
        "random_half": {
            "participation": [0.9] * 20,
            "mean_rank": [0.9] * 20,
            "precedence": [0.9] * 5,
        },
        "chronological": {
            "participation": 0.0,
            "mean_rank": 0.0,
            "precedence": 0.85,
        },
        "contact_null": {
            "participation": [0.1] * 20,
            "mean_rank": [0.1] * 20,
            "precedence": [0.1] * 20,
        },
    }
    assert (
        window_state(agreements, alpha=0.05, min_resolved_families=3)
        == "UNRESOLVED_FAMILIES"
    )


def test_a_below_chance_family_does_not_vote_on_the_chronology_break():
    # fix round 2, Item A: the below-chance family ("participation") would itself flag
    # a "break" if it were naively evaluated (its chronological value 0.05 sits below
    # its own random-half alpha=0.05-quantile of 0.1), but it is below chance and must
    # not be allowed to cast that vote at all. The two above-chance families
    # ("mean_rank", "precedence") do NOT show a break. Result must be RELIABLE.
    #
    # NOTE (see report task-6-report.md, fix round 2 section, for the full derivation):
    # with exactly 3 total resolved families and only 1 of them (the below-chance one)
    # able to cast a "True" break vote, this composition cannot mathematically cross
    # the OLD shared-denominator majority threshold either (strict majority of 3 needs
    # >= 2 True votes; only 1 is achievable here) -- so restoring the shared
    # denominator on THIS specific fixture does not, in fact, flip the result. That is
    # reported honestly below rather than staged to "look like" a failing mutant.
    # `test_a_below_chance_familys_true_flag_no_longer_tips_a_genuine_break_majority`
    # right after this one uses a 3rd, genuinely above-chance-and-breaking family (the
    # composition the reviewer's own bug illustration used) and DOES flip under the
    # deliberate break; that is the test the round-2 break/revert ritual was run
    # against.
    agreements = {
        "random_half": {
            "participation": [0.1] * 20,
            "mean_rank": [0.9] * 20,
            "precedence": [0.8] * 20,
        },
        "chronological": {
            "participation": 0.05,  # < 0.1 -> would flag "break" if it were voted
            "mean_rank": 0.95,  # >= 0.9 -> not a break
            "precedence": 0.85,  # >= 0.8 -> not a break
        },
        "contact_null": {
            "participation": [0.5] * 20,  # q95=0.5, median 0.1 <= 0.5 -> below chance
            "mean_rank": [0.1] * 20,  # q95=0.1, median 0.9 > 0.1 -> above chance
            "precedence": [0.05] * 20,  # q95=0.05, median 0.8 > 0.05 -> above chance
        },
    }
    assert (
        window_state(agreements, alpha=0.05, min_resolved_families=2) == "RELIABLE"
    )


def test_a_below_chance_familys_true_flag_no_longer_tips_a_genuine_break_majority():
    # fix round 2, Item A, supplementary test (added beyond the literal request; see
    # task-6-report.md for why): mirrors the reviewer's own illustrative bug
    # description -- one BELOW_CHANCE family whose own naive flag is True, one
    # above-chance family that is NOT a break ("mean_rank"), one above-chance family
    # that IS a break ("precedence"). Under the pre-Item-A shared denominator (n=3
    # resolved families), the below-chance family's True flag combines with
    # "precedence"'s True flag to reach 2-of-3, a majority, giving the wrong answer
    # CHRONOLOGY_BREAK. Restricting the vote to the 2 above-chance families
    # ("mean_rank"=False, "precedence"=True) gives 1-of-2, not a majority for BREAK.
    # This is the fixture the round-2 deliberate-break-and-revert ritual was run on,
    # because it is the one that can actually distinguish the two denominators.
    #
    # rev3 R3-A update: 1-of-2 is also not a majority for "no break" -- it is an exact
    # tie on the above-chance denominator. Round 2 called this RELIABLE (a bare
    # fallthrough once BREAK was ruled out); that fallthrough is exactly the bug R3-A
    # fixes. The correct verdict is UNRESOLVED_FAMILY_DISCORDANCE. The name of this test
    # ("...no_longer_tips_a_genuine_break_majority") is still accurate -- the
    # below-chance family's flag still cannot tip a BREAK verdict -- so only the
    # asserted outcome changes, not what the test demonstrates.
    agreements = {
        "random_half": {
            "participation": [0.1] * 20,
            "mean_rank": [0.9] * 20,
            "precedence": [0.8] * 20,
        },
        "chronological": {
            "participation": 0.05,  # below-chance family's own naive flag: True
            "mean_rank": 0.95,  # above-chance, not a break
            "precedence": 0.1,  # above-chance, IS a break
        },
        "contact_null": {
            "participation": [0.5] * 20,
            "mean_rank": [0.1] * 20,
            "precedence": [0.05] * 20,
        },
    }
    result = window_state(agreements, alpha=0.05, min_resolved_families=2)
    assert result != "CHRONOLOGY_BREAK"
    assert result == "UNRESOLVED_FAMILY_DISCORDANCE"


def test_a_family_with_no_chronological_value_is_not_resolved_even_with_full_draws():
    # fix round 2, Item B: "participation" has FULL random_half (20) and contact_null
    # (20) -- both clear MIN_FINITE_DRAWS_FOR_RESOLUTION -- but chronological is None.
    # Every fixture before this round that set chronological=None also starved that
    # family's draw lists, so the `chrono is None` clause in the resolution gate was
    # never exercised in isolation; a mutant deleting that clause passed all 18 prior
    # tests. Only "mean_rank" is an ordinary fully-resolved family, so with
    # min_resolved_families=2, only 1 family actually resolves -> UNRESOLVED_FAMILIES.
    agreements = {
        "random_half": {
            "participation": [0.9] * 20,
            "mean_rank": [0.9] * 20,
            "precedence": [],
        },
        "chronological": {
            "participation": None,
            "mean_rank": 0.0,
            "precedence": None,
        },
        "contact_null": {
            "participation": [0.1] * 20,
            "mean_rank": [0.1] * 20,
            "precedence": [],
        },
    }
    assert (
        window_state(agreements, alpha=0.05, min_resolved_families=2)
        == "UNRESOLVED_FAMILIES"
    )


def test_a_family_level_tie_is_discordance_not_reliable():
    # rev3 R3-A: exactly two above-chance families ("mean_rank", "precedence";
    # "participation" is left entirely unresolved so the above-chance denominator is
    # exactly 2, not 3). "mean_rank" breaks (chronological 0.1 sits below its constant
    # random_half's own alpha=0.05 quantile of 0.9); "precedence" does not (chronological
    # 0.9 sits at/above its constant random_half's alpha=0.05 quantile of 0.8). That is a
    # 1-of-2 split on the break vote -- neither a strict break-majority nor a strict
    # no-break-majority -- so the old fallthrough-to-RELIABLE behaviour is wrong on this
    # exact fixture: a family-level tie is discordance among the families, not evidence
    # of reliability.
    agreements = {
        "random_half": {
            "participation": [],
            "mean_rank": [0.9] * 20,
            "precedence": [0.8] * 20,
        },
        "chronological": {
            "participation": None,
            "mean_rank": 0.1,  # < 0.9 alpha-quantile -> break
            "precedence": 0.9,  # >= 0.8 alpha-quantile -> not a break
        },
        "contact_null": {
            "participation": [],
            "mean_rank": [0.1] * 20,
            "precedence": [0.05] * 20,
        },
    }
    result = window_state(agreements, alpha=0.05, min_resolved_families=2)
    assert result == "UNRESOLVED_FAMILY_DISCORDANCE"
    assert result != "RELIABLE"


# ---------------------------------------------------------------------------
# scale_states
# ---------------------------------------------------------------------------


def test_scale_states_returns_too_few_windows_below_the_minimum():
    four_windows = [R, R, R, B]
    assert scale_states(four_windows, min_windows=5) == "UNRESOLVED_TOO_FEW_WINDOWS"

    five_windows = [R, R, R, R, B]
    assert scale_states(five_windows, min_windows=5) == R


def test_scale_states_rejects_a_label_outside_the_known_alphabet():
    # fix round 2, Item C: the ValueError in scale_states was made reachable in round 1
    # (validated up front instead of an unreachable post-vote raise) but no test ever
    # triggered it. fix-round-4 ITEM 4 correction: validation runs unconditionally as
    # the FIRST thing scale_states does, before the evaluable-count/min_windows gate --
    # not after it, as this comment previously (incorrectly) said. min_windows=5 with
    # exactly 5 states here is incidental, not load-bearing: the raise fires regardless
    # of min_windows because nothing gates validation.
    with pytest.raises(ValueError, match="BOGUS_LABEL"):
        scale_states([R, R, R, "BOGUS_LABEL", R], min_windows=5)


def test_two_of_five_windows_cannot_name_a_scale():
    # rev3 R3-B: one UNRESOLVED_FAMILIES window is dropped before counting, leaving 4
    # evaluable windows (still >= min_windows=4). 2-of-4 RELIABLE and 2-of-4
    # CHRONOLOGY_BREAK: neither reaches a strict majority of the evaluable count, so the
    # old mode-with-tiebreak rule (which would have picked one of them by fixed order)
    # must not name a scale state at all.
    windows = [R, R, C, C, UF]
    assert scale_states(windows, min_windows=4) == "UNRESOLVED_MIXED_WINDOWS"


def test_a_three_way_split_is_mixed_not_the_modal_label():
    # rev3 R3-B: drop the 1 UNRESOLVED_FAMILIES window, leaving 4 evaluable: R, R, B, C.
    # RELIABLE is the mode (2 of 4) but 2-of-4 is not a strict majority -- the old
    # mode-with-tiebreak rule would have returned RELIABLE anyway because it was the
    # most frequent label, which is exactly the "reduction to the mode" this fix removes.
    windows = [R, R, B, C, UF]
    assert scale_states(windows, min_windows=4) == "UNRESOLVED_MIXED_WINDOWS"


def test_dropping_unevaluable_windows_can_take_a_scale_below_the_minimum():
    # rev3 R3-B: original length is 5 (>= min_windows=4), so a check against the
    # ORIGINAL count would proceed to vote and find 3-of-3 RELIABLE unanimous. But 2 of
    # the 5 are UNRESOLVED_FAMILIES and must be dropped before the minimum is checked,
    # leaving only 3 evaluable windows -- below min_windows=4 -- so the correct answer is
    # UNRESOLVED_TOO_FEW_WINDOWS, not a unanimous RELIABLE.
    windows = [R, R, R, UF, UF]
    assert scale_states(windows, min_windows=4) == "UNRESOLVED_TOO_FEW_WINDOWS"


# ---------------------------------------------------------------------------
# meta-test — coordinator fix-round-4, ITEM 1. Four review rounds in a row found a
# variant of the same defect: a new "unresolved"/"could not be decided" label was added
# to the module's alphabet but not reliably added to every filter that needs to treat it
# as non-evaluable. This test does not pin one input/output pair; it pins the STRUCTURAL
# invariant that would have caught all four -- every such constant is classified into
# exactly one of the two disjoint sets that consume the alphabet.
# ---------------------------------------------------------------------------


def test_every_unresolved_constant_is_classified_exactly_once():
    import src.topic5_slow_state_scale as mod

    candidates = {
        value
        for name, value in vars(mod).items()
        if isinstance(value, str)
        and (value.startswith("UNRESOLVED_") or value == mod.TOO_FEW)
    }
    assert candidates, "no UNRESOLVED_*/TOO_FEW string constants found -- fixture is stale"

    classified = mod.NOT_EVALUABLE | mod.OUTPUT_ONLY_STATUSES
    unclassified = candidates - classified
    assert not unclassified, f"unclassified unresolved constant(s): {sorted(unclassified)}"

    assert mod.NOT_EVALUABLE.isdisjoint(mod.OUTPUT_ONLY_STATUSES)
