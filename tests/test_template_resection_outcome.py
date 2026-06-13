"""TDD for Track E1 template-resection metric functions.

Contract clauses (spec docs/superpowers/specs/2026-06-13-yuquan-template-resection-outcome-design.md):
  - coverage: NA (None) vs 0.0 are distinct (empty target = NA, not 0).
  - shared_endpoint_core: intersection across templates; <2 templates -> None.
  - hfo_rate_core: bipolar->monopolar first-contact bridge (reuse build_rate_lookup),
    top-k by event_rate with k=min(soz_size, n_available), min_ch_events gate, tie-break by name.
  - discordant_candidate: rule-driven, NA-safe (None metric -> clause unevaluable, not crash).
"""
from src.template_resection_outcome import (
    coverage,
    shared_endpoint_core,
    hfo_rate_core,
    discordant_candidate,
)


# ---- coverage: NA vs 0 (user P1) ----
def test_coverage_basic_fraction():
    assert coverage({"A1", "A2", "A3", "A4"}, {"A1", "A2"}) == 0.5


def test_coverage_zero_when_no_overlap_nonempty_target():
    # denominator non-empty, 0 hits -> 0.0 (a real "nothing covered"), NOT None
    assert coverage({"A1", "A2"}, {"B1"}) == 0.0


def test_coverage_none_when_target_empty():
    # empty/absent target -> NA (None), distinct from 0.0
    assert coverage(set(), {"A1"}) is None


def test_coverage_full():
    assert coverage({"A1", "A2"}, {"A1", "A2", "A3"}) == 1.0


# ---- shared_endpoint_core: intersection across the two stable templates ----
def test_shared_endpoint_core_intersection():
    assert shared_endpoint_core([{"K5", "K6", "K7"}, {"K6", "K7", "K8"}]) == {"K6", "K7"}


def test_shared_endpoint_core_none_when_fewer_than_two_templates():
    # gate is the caller's (stable_k>=2); still guard <2 templates -> None
    assert shared_endpoint_core([{"K5", "K6"}]) is None


def test_shared_endpoint_core_empty_intersection_is_set_not_none():
    # two templates, no shared endpoint -> empty set (covered=0), NOT None
    assert shared_endpoint_core([{"K1", "K2"}, {"K3", "K4"}]) == set()


# ---- hfo_rate_core: bipolar bridge + top-k SOZ-size (frozen rule) ----
def _cm(name, rate, n=100):
    return {"ch_name": name, "event_rate": rate, "n_events": n}


def test_hfo_rate_core_bipolar_topk_sozsize():
    cms = [_cm("E1-E2", 300), _cm("E3-E4", 200), _cm("E5-E6", 100), _cm("E7-E8", 50)]
    core, n_avail = hfo_rate_core(cms, soz_size=2, min_ch_events=30)
    assert n_avail == 4
    assert core == {"E1", "E3"}  # top-2 by rate, bridged to first contacts


def test_hfo_rate_core_excludes_low_event_channels():
    cms = [_cm("E1-E2", 300, n=100), _cm("E3-E4", 999, n=10)]  # E3-E4 below 30-event gate
    core, n_avail = hfo_rate_core(cms, soz_size=5, min_ch_events=30)
    assert n_avail == 1
    assert core == {"E1"}


def test_hfo_rate_core_k_capped_by_available():
    cms = [_cm("E1-E2", 300), _cm("E3-E4", 200)]
    core, n_avail = hfo_rate_core(cms, soz_size=10, min_ch_events=30)
    assert core == {"E1", "E3"}  # k=min(10,2)=2 -> all


def test_hfo_rate_core_tie_break_by_name():
    cms = [_cm("E5-E6", 100), _cm("E1-E2", 100), _cm("E3-E4", 100)]
    core, _ = hfo_rate_core(cms, soz_size=2, min_ch_events=30)
    assert core == {"E1", "E3"}  # equal rate -> E1 < E3 < E5 by name


def test_hfo_rate_core_empty_when_no_rate():
    core, n_avail = hfo_rate_core([], soz_size=5, min_ch_events=30)
    assert core == set()
    assert n_avail == 0


# ---- discordant_candidate: rule-driven, NA-safe (user P1) ----
def test_discordant_true_on_zero_early_end():
    assert discordant_candidate(0.0, 0.9, 0.9, {"A1"}, {"A1"}, False) is True


def test_discordant_true_on_low_template_endpoint():
    assert discordant_candidate(0.7, 0.4, 0.9, {"A1"}, {"A1"}, False) is True


def test_discordant_true_on_low_network():
    assert discordant_candidate(0.7, 0.9, 0.49, {"A1"}, {"A1"}, False) is True


def test_discordant_true_on_origin_disjoint():
    assert discordant_candidate(0.7, 0.9, 0.9, {"A1"}, {"B1"}, False) is True


def test_discordant_true_on_multi_session():
    assert discordant_candidate(0.9, 0.9, 0.9, {"A1"}, {"A1"}, True) is True


def test_discordant_false_when_all_good():
    assert discordant_candidate(0.7, 0.9, 0.9, {"A1"}, {"A1"}, False) is False


def test_discordant_na_safe_metrics_dont_crash():
    # None metrics (NA) -> those clauses unevaluable (False), no crash
    assert discordant_candidate(None, None, None, set(), set(), False) is False


def test_discordant_origin_clause_needs_both_nonempty():
    # template_source empty (NA) -> origin-disjoint clause must NOT fire
    assert discordant_candidate(0.7, 0.9, 0.9, set(), {"B1"}, False) is False
