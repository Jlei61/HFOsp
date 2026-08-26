from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.topic5_continuous_marked_state_r1.coverage import (
    build_coverage,
    clip_intervals,
    merge_labeled_intervals,
    merge_intervals,
    recorded_duration_between,
)
from src.topic5_continuous_marked_state_r1.mark_likelihood import (
    conditional_k_subset_log_prob,
    log_elementary_symmetric,
    tied_group_mark_log_prob,
)
from src.topic5_continuous_marked_state_r1.survival import (
    point_process_log_likelihood,
    rescaled_interevent_integrals,
)


def test_coverage_union_clip_and_duration_exclude_gaps() -> None:
    start, stop = merge_intervals(
        np.asarray([0.0, 9.5, 30.0, 41.0]),
        np.asarray([10.0, 20.0, 40.0, 50.0]),
    )
    np.testing.assert_allclose(start, [0.0, 30.0, 41.0])
    np.testing.assert_allclose(stop, [20.0, 40.0, 50.0])
    left, right = clip_intervals(start, stop, 5.0, 45.0)
    np.testing.assert_allclose(left, [5.0, 30.0, 41.0])
    np.testing.assert_allclose(right, [20.0, 40.0, 45.0])
    duration = recorded_duration_between(
        start, stop, np.asarray([5.0, 18.0]), np.asarray([45.0, 32.0])
    )
    np.testing.assert_allclose(duration, [29.0, 4.0])


def test_labeled_coverage_preserves_abutting_session_reset() -> None:
    start, stop, label = merge_labeled_intervals(
        np.asarray([0.0, 10.0, 20.0]),
        np.asarray([10.0, 20.0, 30.0]),
        np.asarray([0, 0, 1]),
    )
    np.testing.assert_allclose(start, [0.0, 20.0])
    np.testing.assert_allclose(stop, [20.0, 30.0])
    np.testing.assert_array_equal(label, [0, 1])


def test_real_pilot_coverage_has_exact_duration_parity() -> None:
    coverage, manifest = build_coverage("yuquan_huanghanwen")
    coverage.validate()
    assert manifest["all_events_inside_coverage"] is True
    assert manifest["duration_parity"]["max_abs_seconds"] == 0.0


def test_constant_poisson_matches_closed_form_with_gap() -> None:
    dtype = torch.float64
    rate = 0.2
    event = torch.tensor([1.0, 4.0, 12.0], dtype=dtype)
    start = torch.tensor([0.0, 10.0], dtype=dtype)
    stop = torch.tensor([5.0, 15.0], dtype=dtype)
    terms = point_process_log_likelihood(
        event, start, stop,
        lambda t: torch.full_like(t, math.log(rate)),
        quadrature_order=4,
    )
    expected = len(event) * math.log(rate) - rate * 10.0
    assert terms.log_likelihood.item() == pytest.approx(expected, abs=1e-12)
    assert terms.survival_integral.item() == pytest.approx(2.0, abs=1e-12)


def test_rescaled_integral_uses_only_recorded_portion() -> None:
    event = torch.tensor([1.0, 12.0], dtype=torch.float64)
    start = torch.tensor([0.0, 10.0], dtype=torch.float64)
    stop = torch.tensor([5.0, 15.0], dtype=torch.float64)
    value = rescaled_interevent_integrals(
        event, start, stop, lambda t: torch.zeros_like(t), quadrature_order=4
    )
    # [1,5) plus [10,12) = six recorded seconds, not eleven wall seconds.
    assert value.tolist() == pytest.approx([6.0], abs=1e-12)


@pytest.mark.parametrize("n,k", [(4, 0), (4, 1), (4, 2), (5, 3)])
def test_log_esp_and_subset_probability_match_enumeration(n: int, k: int) -> None:
    logits = torch.linspace(-0.7, 0.9, n, dtype=torch.float64)
    candidate = torch.ones(n, dtype=torch.bool)
    got = log_elementary_symmetric(logits, candidate, k)
    terms = [sum(float(logits[i]) for i in subset)
             for subset in itertools.combinations(range(n), k)]
    expected = torch.logsumexp(torch.tensor(terms, dtype=torch.float64), dim=0)
    assert got.item() == pytest.approx(expected.item(), abs=1e-12)

    probability = 0.0
    for subset in itertools.combinations(range(n), k):
        target = torch.zeros(n, dtype=torch.bool)
        target[list(subset)] = True
        probability += math.exp(float(
            conditional_k_subset_log_prob(logits, target, candidate)
        ))
    assert probability == pytest.approx(1.0, abs=1e-12)


def test_subset_law_is_tie_permutation_invariant_and_has_gradients() -> None:
    logits = torch.tensor([0.2, -0.5, 1.1, 0.4], requires_grad=True)
    candidate = torch.tensor([True, True, True, False])
    target = torch.tensor([True, False, True, False])
    first = conditional_k_subset_log_prob(logits, target, candidate)
    permutation = torch.tensor([2, 0, 1, 3])
    second = conditional_k_subset_log_prob(
        logits[permutation], target[permutation], candidate[permutation]
    )
    assert first.item() == pytest.approx(second.item(), abs=1e-7)
    (-first).backward()
    assert torch.isfinite(logits.grad).all()


def test_complete_tied_event_matches_manual_size_and_subset_terms() -> None:
    # event 0: tied {0,2}, then {1}, then STOP; event 1: {1}, then STOP
    group_ids = torch.tensor([[0, 1, 0, -1], [-1, 0, -1, -1]])
    group_count = torch.tensor([2, 1])
    steps = 3
    size_logits = torch.zeros(2, steps, 5, dtype=torch.float64)
    contact_logits = torch.tensor(
        [
            [[0.2, -0.1, 0.7, 0.0], [0.1, 0.5, -0.2, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [[0.3, 0.8, -0.4, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        ], dtype=torch.float64,
    )
    node_mask = torch.tensor([[True, True, True, False], [True, True, True, False]])
    terms = tied_group_mark_log_prob(
        group_ids, group_count, size_logits, contact_logits, node_mask
    )
    assert terms.event_log_prob.shape == (2,)
    assert terms.active_step.tolist() == [[True, True, True], [True, True, False]]
    assert terms.select_step.tolist() == [[True, True, False], [True, False, False]]
    assert torch.isfinite(terms.event_log_prob).all()


def test_complete_tied_event_rejects_non_dense_groups() -> None:
    with pytest.raises(ValueError, match="dense"):
        tied_group_mark_log_prob(
            torch.tensor([[0, 2, -1]]), torch.tensor([2]),
            torch.zeros(1, 3, 4), torch.zeros(1, 3, 3),
        )
