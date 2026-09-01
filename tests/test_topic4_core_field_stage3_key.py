"""Task 2 -- recruitment enters the candidate key per direction, not as a union.

Stage 2's failure mode was a field that had plenty to say in one direction and
almost nothing in the other. A union count rewards exactly that, so the
load-bearing quantity is the smaller of the two directions.
"""
import numpy as np
import pytest

from src.topic4_core_field_scoring import (candidate_key3, coverage_tier,
                                           recruited_contacts,
                                           recruited_per_direction)

SUPPORT = [f"C{i}" for i in range(1, 16)]
PART_MIN = 5


def _event(names, sign, n_part=None):
    return dict(sign=float(sign), n_part=int(n_part if n_part is not None else len(names)),
                ranks={n: float(i) for i, n in enumerate(names)})


# ------------------------------------------------------------ per direction
def test_counts_each_direction_separately():
    evs = [_event(SUPPORT[:12], +1), _event(SUPPORT[:4], -1, n_part=5)]
    assert recruited_per_direction(evs, SUPPORT, PART_MIN) == (12, 4)


def test_load_bearing_quantity_is_the_smaller_direction():
    # the union would say 15 and reward a field that only works one way
    evs = [_event(SUPPORT, +1), _event(SUPPORT[:3], -1, n_part=5)]
    assert recruited_per_direction(evs, SUPPORT, PART_MIN) == (15, 3)
    assert recruited_contacts(evs, SUPPORT, PART_MIN) == 3


def test_a_lopsided_field_scores_below_a_balanced_one():
    lopsided = [_event(SUPPORT, +1), _event(SUPPORT[:3], -1, n_part=5)]
    balanced = [_event(SUPPORT[:9], +1), _event(SUPPORT[:9], -1)]
    assert (coverage_tier(recruited_contacts(lopsided, SUPPORT, PART_MIN))
            < coverage_tier(recruited_contacts(balanced, SUPPORT, PART_MIN)))


def test_single_direction_run_recruits_nothing():
    evs = [_event(SUPPORT, +1), _event(SUPPORT, +1)]
    assert recruited_per_direction(evs, SUPPORT, PART_MIN) == (15, 0)
    assert recruited_contacts(evs, SUPPORT, PART_MIN) == 0
    assert coverage_tier(recruited_contacts(evs, SUPPORT, PART_MIN)) == 0


def test_contacts_outside_the_frozen_support_are_not_counted():
    evs = [_event(SUPPORT[:5] + ["ZZ1", "ZZ2"], +1),
           _event(SUPPORT[:5] + ["ZZ3"], -1)]
    assert recruited_per_direction(evs, SUPPORT, PART_MIN) == (5, 5)


def test_events_below_the_participation_floor_are_ignored():
    evs = [_event(SUPPORT[:12], +1, n_part=4), _event(SUPPORT[:6], -1)]
    assert recruited_per_direction(evs, SUPPORT, PART_MIN) == (0, 6)


def test_unsigned_events_are_ignored():
    evs = [dict(sign=None, n_part=9, ranks={n: 0.0 for n in SUPPORT}),
           _event(SUPPORT[:7], +1), _event(SUPPORT[:7], -1)]
    assert recruited_per_direction(evs, SUPPORT, PART_MIN) == (7, 7)


def test_none_ranks_do_not_count_as_participation():
    e = _event(SUPPORT[:8], +1)
    e["ranks"][SUPPORT[8]] = None
    assert recruited_per_direction([e, _event(SUPPORT[:8], -1)],
                                   SUPPORT, PART_MIN) == (8, 8)


# ------------------------------------------------------------------- tiers
@pytest.mark.parametrize("n,tier", [(0, 0), (2, 0), (3, 1), (5, 1),
                                    (6, 2), (9, 3), (12, 4), (14, 4), (15, 5)])
def test_tier_boundaries_are_pinned(n, tier):
    assert coverage_tier(n) == tier


# --------------------------------------------------------------------- key
def test_recruitment_outranks_template_match():
    assert candidate_key3(2, 15, 0.1) > candidate_key3(2, 11, 0.9)


def test_direction_count_still_outranks_recruitment():
    assert candidate_key3(2, 3, -0.9) > candidate_key3(1, 15, 0.99)


def test_missing_score_sorts_last_within_its_tier():
    assert candidate_key3(2, 9, float("nan")) < candidate_key3(2, 9, -0.99)
    assert candidate_key3(2, 9, float("nan"))[2] == -np.inf


def test_key_is_totally_ordered_with_nan_present():
    keys = [candidate_key3(2, 9, float("nan")), candidate_key3(2, 9, 0.5),
            candidate_key3(1, 15, 0.9), candidate_key3(2, 12, -0.2)]
    assert sorted(keys) == [keys[2], keys[0], keys[1], keys[3]]


# ------------------------------------------------ regression lock on real runs
import glob
import json
import os

_POOL = ("results/topic4_sef_hfo/field_swap_subject_snn/"
         "readout_epilepsiae_1146_learned_core_field_pool_s*.json")
_CFG = "results/topic4_sef_hfo/data_driven_core_field/config/stage_config.json"


@pytest.mark.integration
@pytest.mark.skipif(not os.path.exists(_CFG) or not glob.glob(_POOL),
                    reason="Stage 2 learned-field pool not on disk")
def test_the_metric_still_discriminates_on_the_real_stage2_runs():
    """The key is applied per candidate run, so that is where it must bite.

    Pooling all 120 runs saturates both directions at 15 and would make this
    lock vacuous -- which is exactly why the assertion is written against the
    single-run granularity the optimiser actually sees.
    """
    cfg = json.load(open(_CFG))
    support, part_min = cfg["support"], cfg["part_min"]
    pairs, pooled = [], []
    for path in glob.glob(_POOL):
        events = json.load(open(path))["events"]
        pairs.append(recruited_per_direction(events, support, part_min))
        pooled.extend(events)

    assert len(pairs) >= 100
    mins = [min(a, b) for a, b in pairs]
    tiers = {coverage_tier(m) for m in mins}

    # single-direction runs must land in tier 0, and there are plenty of them
    assert 0 in tiers and 0.15 < np.mean([m == 0 for m in mins]) < 0.60
    # the metric is nowhere near its ceiling at run granularity
    assert max(mins) < 15 and len(tiers) >= 3
    # forward is the weaker direction in this field: the union would hide that
    assert np.median([a for a, _ in pairs]) < np.median([b for _, b in pairs])

    # pooled over every run both directions saturate -- the union is useless here
    assert recruited_per_direction(pooled, support, part_min) == (15, 15)
