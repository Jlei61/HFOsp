import numpy as np
import pytest
from src.topic4_core_field_scoring import (
    adversarial_gain, assignment_invariant_S, axis_only_templates,
    balanced_pair_score, model_templates, sim_matrix,
)

SUPPORT = ["c1", "c2", "c3", "c4", "c5", "c6"]
TARGET = {"t_a": {n: float(i) for i, n in enumerate(SUPPORT)},
          "t_b": {n: float(-i) for i, n in enumerate(SUPPORT)}}


def _ev(sign, ranks):
    return {"sign": sign, "n_part": len(ranks), "ranks": ranks}


def _both(names_fwd, names_rev):
    return [_ev(1.0, {n: float(i) for i, n in enumerate(names_fwd)}),
            _ev(-1.0, {n: float(-i) for i, n in enumerate(names_rev)})]


def test_model_templates_never_widen_beyond_the_frozen_support():
    m = model_templates([_ev(1.0, {"c1": 0.0, "c2": 1.0, "c3": 2.0, "c4": 3.0,
                                   "c5": 4.0, "zzz": 9.0})], SUPPORT, part_min=5)
    assert set(m["forward"]) <= set(SUPPORT)


def test_events_below_the_participation_floor_are_dropped():
    m = model_templates([_ev(1.0, {"c1": 0.0, "c2": 1.0})], SUPPORT, part_min=5)
    assert m["n_dir"] == 0


def test_coverage_is_reported_per_direction_not_only_as_a_union():
    """P0-4: a union hides 'each direction covers a different small patch'."""
    m = model_templates(_both(SUPPORT[:5], SUPPORT[1:6]), SUPPORT, part_min=5)
    assert m["coverage_forward"] == pytest.approx(5 / 6)
    assert m["coverage_reverse"] == pytest.approx(5 / 6)
    assert m["coverage_union"] == pytest.approx(1.0)
    assert m["coverage_union"] > min(m["coverage_forward"], m["coverage_reverse"])


def test_score_is_invariant_to_swapping_the_two_patient_templates():
    m = model_templates(_both(SUPPORT, SUPPORT), SUPPORT, part_min=5)
    s1 = assignment_invariant_S(sim_matrix(m, TARGET, SUPPORT, "mean_rank"))
    s2 = assignment_invariant_S(
        sim_matrix(m, {"t_a": TARGET["t_b"], "t_b": TARGET["t_a"]}, SUPPORT, "mean_rank"))
    assert s1 == pytest.approx(s2)


def test_a_single_direction_model_yields_nan_not_a_best_cell():
    """spec 5.3: S_rank must never be a number that can be differenced against a
    two-direction score. One direction -> undefined."""
    m = model_templates([_ev(1.0, {n: float(i) for i, n in enumerate(SUPPORT)})],
                        SUPPORT, part_min=5)
    assert m["n_dir"] == 1
    assert np.isnan(assignment_invariant_S(sim_matrix(m, TARGET, SUPPORT, "mean_rank")))


def test_balanced_pair_score_is_assignment_invariant_and_bidirectional():
    m = model_templates(_both(SUPPORT, SUPPORT), SUPPORT, part_min=5)
    s1 = balanced_pair_score(m, TARGET, SUPPORT)
    s2 = balanced_pair_score(m, {"t_a": TARGET["t_b"], "t_b": TARGET["t_a"]}, SUPPORT)
    assert s1 == pytest.approx(s2)
    one = model_templates([_ev(1.0, {n: float(i) for i, n in enumerate(SUPPORT)})],
                          SUPPORT, part_min=5)
    assert np.isnan(balanced_pair_score(one, TARGET, SUPPORT))


def test_balanced_pair_score_uses_a_fixed_denominator():
    full = model_templates(_both(SUPPORT, SUPPORT), SUPPORT, part_min=5)
    half = model_templates(_both(SUPPORT[:5], SUPPORT[:5]), SUPPORT, part_min=5)
    assert balanced_pair_score(half, TARGET, SUPPORT) < \
           balanced_pair_score(full, TARGET, SUPPORT)


def test_axis_only_templates_are_exact_mirrors():
    names = ["c1", "c2", "c3"]
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    t = axis_only_templates(names, coords, np.array([1.0, 0.0]), np.array([1.0, 0.0]))
    assert np.allclose([t["forward"][n] for n in names], [-t["reverse"][n] for n in names])


def test_adversarial_gain_reports_how_much_dropping_one_contact_can_help():
    """Diagnostic, not an assertion: under mean-rank filling a badly ranked
    contact CAN be worth dropping, and we need to see by how much."""
    ranks = {n: float(i) for i, n in enumerate(SUPPORT)}
    ranks["c3"] = 99.0
    m = model_templates([_ev(1.0, ranks), _ev(-1.0, {n: -v for n, v in ranks.items()})],
                        SUPPORT, part_min=5)
    g = adversarial_gain(m, TARGET, SUPPORT, "mean_rank")
    assert g["worst_contact"] == "c3"
    assert g["gain"] > 0


import glob, json, sys
from pathlib import Path

RUN = Path("results/topic4_sef_hfo/field_swap_subject_snn")


@pytest.mark.integration
def test_common_only_mode_reproduces_the_published_sim_matrix():
    """Our reimplementation must agree with the file carrying the published
    Figure 4 numbers, in the mode that file implements."""
    sys.path.insert(0, ".")
    from scripts.paper_figures.plot_fig_subject_snn_realvsmodel import (
        _model_templates, _real_templates, _sim_matrix)
    tags = sorted(glob.glob(str(RUN / "readout_epilepsiae_1146_paired_tsrc_highn_s*_20260721.json")))
    if not tags:
        pytest.skip("paired_tsrc_highn artifacts not present")
    tag = Path(tags[0]).stem.removeprefix("readout_")
    real = _real_templates("epilepsiae_1146", "narrow")
    ref_M, _ = _sim_matrix(_model_templates(tag), real, B=1, seed=0)
    ro = json.load(open(RUN / f"readout_{tag}.json"))
    support = sorted(set(real["t_a"]) | set(real["t_b"]))
    # The published file admits n_part >= 2*k_dir; match it here so the regression
    # compares SCORING, not the participation floor.
    ours = model_templates(ro["events"], support, part_min=2 * int(ro.get("k_dir", 2)))
    our_M = sim_matrix(ours, {"t_a": real["t_a"], "t_b": real["t_b"]}, support, "common_only")
    assert np.allclose(ref_M, our_M, atol=1e-9, equal_nan=True)


@pytest.mark.integration
def test_no_signed_event_falls_below_the_participation_floor():
    """Justifies deleting gate=4: endpoint_centroid_axis returns None below
    2*k_dir+1, so a post-hoc gate of 4 admits nothing."""
    below = 0
    for path in glob.glob(str(RUN / "readout_epilepsiae_1146_*.json")):
        d = json.load(open(path))
        if d.get("k_dir") != 2:
            continue
        below += sum(1 for e in d.get("events", [])
                     if e.get("sign") is not None and e.get("n_part", 0) < 5)
    assert below == 0


from src.topic4_core_field_scoring import candidate_key


def test_two_directions_always_outrank_one_even_when_S_is_lower():
    """The counterexample that killed scalar grading: one direction matching a
    template perfectly scores 0.5; two directions whose best assignment is +1 and
    -1 score 0."""
    assert candidate_key(2, 0.0) > candidate_key(1, 0.5)


def test_within_a_tier_the_better_match_ranks_higher():
    assert candidate_key(2, 0.8) > candidate_key(2, 0.3)


def test_no_direction_ranks_last_and_tolerates_nan():
    assert candidate_key(0, float("nan")) < candidate_key(1, -0.9)
    assert candidate_key(0, float("nan")) == candidate_key(0, float("nan"))
