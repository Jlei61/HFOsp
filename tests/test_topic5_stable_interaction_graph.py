from __future__ import annotations

import numpy as np
import torch

import pytest

from src.topic5_shared_propagation_field import (
    generate_from_step_logits,
    suffix_log_likelihood,
)
from src.topic5_stable_interaction_graph import (
    StableInteractionGraph,
    MatchedPhaseMarkovMixtureModel,
    cardinality_schedule,
    frozen_synthetic_graph,
    independent_synthetic_graph,
    fit_synthetic_sig,
    precedence_distance,
    precedence_matrix,
    simulate_synthetic_events,
    top_positive_overlap,
    uniform_provenance,
)


def _fitted_shaped_model(seed: int = 11) -> StableInteractionGraph:
    """A SIG with non-trivial graph and phase head, as scored in the ladder."""
    truth = frozen_synthetic_graph()
    model = StableInteractionGraph(
        12,
        static_bias=np.asarray(truth["static_bias"]),
        learn_graph=True,
        initial_leak=float(truth["leak"]),
    )
    generator = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        model.raw_weight.copy_(
            torch.randn(12, 12, generator=generator) * 0.4
        )
        model.phase_loading.copy_(
            torch.randn(12, 3, generator=generator) * 0.3
        )
    model.eval()
    return model


def _shared_loop_logits(model: StableInteractionGraph, counts: torch.Tensor):
    """Drive the shared ladder loop with the SIG state update."""
    state = model.static_bias.new_zeros((counts.shape[0], model.n_contacts))

    def logit_fn(step, previous, active):
        nonlocal state
        state = model.transition(state, previous)
        phi = step / (counts.to(state.dtype) - 1.0).clamp_min(1.0)
        return model.emission_logits(state, phi)

    return logit_fn


def test_weight_orientation_is_target_by_source_and_diagonal_is_zero():
    model = StableInteractionGraph(4, learn_graph=True, max_weight=3.0)
    with torch.no_grad():
        model.raw_weight.zero_()
        # W[target=2, source=1] is positive.
        model.raw_weight[2, 1] = torch.atanh(torch.tensor(2.5 / 3.0))
        model.raw_weight[1, 1] = 10.0
    weight = model.effective_weight()
    assert torch.isclose(weight[2, 1], torch.tensor(2.5), atol=1e-5)
    assert torch.allclose(torch.diag(weight), torch.zeros(4))
    state = torch.zeros((1, 4))
    previous = torch.tensor([[False, True, False, False]])
    next_state = model.transition(state, previous)
    assert next_state[0, 2] > 0.9
    assert next_state[0, 0] == 0.0


def test_no_graph_has_no_contact_feedback_but_keeps_phase_head():
    model = StableInteractionGraph(4, learn_graph=False)
    state = torch.zeros((2, 4))
    previous = torch.tensor(
        [[True, False, False, False], [False, True, False, False]]
    )
    next_state = model.transition(state, previous)
    assert torch.allclose(next_state, torch.zeros_like(next_state))
    assert model.phase_loading.shape == (4, 3)


def test_markov_control_uses_exact_same_phase_nuisance_basis_as_sig():
    sig = StableInteractionGraph(4, learn_graph=False)
    markov = MatchedPhaseMarkovMixtureModel(
        4, np.zeros(4, dtype=np.float32), n_components=1
    )
    loading = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 20.0
    with torch.no_grad():
        sig.phase_loading.copy_(loading)
        markov.phase_basis[0].copy_(loading.T)
        markov.transition.zero_()
    counts = torch.tensor([5, 7])
    previous = torch.tensor(
        [[True, False, False, False], [False, True, False, False]]
    )
    phi = 2 / (counts.float() - 1)
    sig_logits = sig.emission_logits(torch.zeros((2, 4)), phi)
    markov_logits = markov.component_logits(
        0, previous, step=2, group_count=counts
    )
    assert torch.allclose(sig_logits, markov_logits)


def test_one_step_intervention_uses_matched_candidate_mask():
    model = StableInteractionGraph(4, learn_graph=True, max_weight=3.0)
    with torch.no_grad():
        model.raw_weight.zero_()
        model.phase_loading.zero_()
        model.raw_weight[2, 1] = torch.atanh(torch.tensor(2.4 / 3.0))
    influence = model.one_step_intervention_matrix()
    assert influence.shape == (4, 4)
    assert influence[2, 1] > 0.2
    # Source j is excluded in both arms; it cannot acquire a trivial self-effect.
    assert torch.isclose(influence[1, 1], torch.tensor(0.0))


def test_free_rollout_returns_exact_schedule_without_recruitment_repeats():
    model = StableInteractionGraph(5, learn_graph=False)
    first = torch.tensor(
        [[True, False, False, False, False], [False, True, False, False, False]]
    )
    counts = torch.tensor([4, 3])
    schedule = torch.tensor([[1, 1, 1], [1, 1, 0]])
    generated = model.rollout(
        first,
        counts,
        schedule,
        generator=torch.Generator().manual_seed(3),
    )
    assert torch.equal((generated >= 0).sum(1), counts)
    for row, count in zip(generated, counts):
        assert torch.equal(
            torch.sort(row[row >= 0]).values,
            torch.arange(int(count)),
        )


def test_synthetic_graph_has_branching_and_unseen_start_is_seen_internally():
    truth = frozen_synthetic_graph()
    weight = np.asarray(truth["weight"])
    assert weight.shape == (12, 12)
    assert np.allclose(np.diag(weight), 0.0)
    assert np.sum(weight[:, 0] > 1.0) >= 2
    train = simulate_synthetic_events(
        800, starts=[0, 3, 6], seed=7
    )
    assert train.group_ids.shape == (800, 12)
    # Contact 9 is the held-out start and must have support as a middle node.
    assert np.sum(train.group_ids[:, 9] > 0) >= 20
    assert np.all(train.start_contact != 9)


def test_empirical_intervention_operator_uses_observed_prefix_contexts():
    data = simulate_synthetic_events(80, starts=[0, 3, 6], seed=17)
    truth = frozen_synthetic_graph()
    model = StableInteractionGraph(
        12,
        static_bias=np.asarray(truth["static_bias"]),
        learn_graph=True,
        initial_leak=float(truth["leak"]),
    )
    with torch.no_grad():
        model.raw_weight.copy_(
            torch.atanh(
                torch.as_tensor(np.asarray(truth["weight"]) / 3.0).clamp(
                    -0.999, 0.999
                )
            )
        )
        model.phase_loading.copy_(
            torch.as_tensor(np.asarray(truth["phase_loading"]))
        )
    groups, counts = data.torch()
    influence = model.empirical_one_step_intervention_matrix(groups, counts)
    assert influence.shape == (12, 12)
    assert torch.all(torch.isfinite(influence[~torch.eye(12, dtype=torch.bool)]))
    assert influence[1, 0] > 0.0


def test_marginal_intervention_matches_singleton_operator():
    data = simulate_synthetic_events(80, starts=[0, 3, 6], seed=117)
    model = _fitted_shaped_model(seed=19)
    groups, counts = data.torch()
    old = model.empirical_one_step_intervention_matrix(groups, counts)
    marginal, support = model.empirical_marginal_intervention_matrix(
        groups, counts, return_support=True
    )
    assert torch.allclose(old, marginal, equal_nan=True, atol=1e-7)
    assert torch.all(support[torch.isfinite(marginal)] > 0)


def test_marginal_intervention_handles_tied_rank_without_full_attribution():
    model = StableInteractionGraph(5, learn_graph=True, max_weight=3.0)
    with torch.no_grad():
        model.raw_weight.zero_()
        model.phase_loading.zero_()
        model.raw_weight[2, 0] = torch.atanh(torch.tensor(2.4 / 3.0))
    groups = torch.tensor(
        [
            [0, 0, 1, 2, -1],
            [0, 0, 1, 2, -1],
        ]
    )
    counts = torch.tensor([3, 3])
    with pytest.raises(ValueError, match="singleton"):
        model.empirical_one_step_intervention_matrix(groups, counts)
    influence = model.empirical_marginal_intervention_matrix(groups, counts)
    assert influence[2, 0] > 0.0
    assert torch.isclose(influence[2, 1], torch.tensor(0.0), atol=1e-7)


def test_precedence_metrics_and_top_overlap_are_well_defined():
    data = simulate_synthetic_events(100, starts=[0, 3], seed=9)
    matrix = precedence_matrix(data.group_ids)
    assert matrix.shape == (12, 12)
    assert precedence_distance(matrix, matrix) == 0.0
    truth = np.asarray(frozen_synthetic_graph()["weight"])
    assert top_positive_overlap(truth, truth) == 1.0
    schedule = cardinality_schedule(data.group_ids, data.group_count)
    assert np.all(schedule.sum(1) == data.group_count - 1)


def test_independent_confirmation_graph_is_deterministic_and_branching():
    left = independent_synthetic_graph(20260801)
    right = independent_synthetic_graph(20260801)
    assert np.array_equal(left["weight"], right["weight"])
    weight = np.asarray(left["weight"])
    assert np.allclose(np.diag(weight), 0.0)
    assert np.all(np.sum(weight > 1.0, axis=0) == 2)


def test_sig_likelihood_uses_the_shared_ladder_scoring_contract():
    """SIG re-implements the suffix loop; it must stay bit-comparable.

    Every SIG-versus-baseline number divides by a per-decision denominator.
    If SIG counted decisions or masked candidates differently from the loop
    that scores M1/M2/M3, the reported gap would be an accounting artifact.
    """
    data = simulate_synthetic_events(64, starts=[0, 3, 6], seed=31)
    groups, counts = data.torch()
    model = _fitted_shaped_model()
    shared = suffix_log_likelihood(
        _shared_loop_logits(model, counts), groups, counts
    )
    own = model.suffix_log_likelihood(groups, counts)
    assert torch.allclose(
        shared["event_log_probability"],
        own["event_log_probability"],
        atol=1e-6,
    )
    assert torch.equal(shared["decision_count"], own["decision_count"])
    # M1/M2/M3 divide by (T-1) summed over events; SIG must use the same one.
    assert torch.equal(own["decision_count"], (counts - 1).clamp_min(0))
    assert torch.allclose(
        shared["nll_per_decision"],
        model.nll_per_decision(groups, counts),
        atol=1e-6,
    )


def test_nograph_sig_and_matched_markov_agree_in_the_static_limit():
    """Cross-implementation equality of the two scoring paths."""
    data = simulate_synthetic_events(64, starts=[0, 3, 6], seed=41)
    groups, counts = data.torch()
    bias = np.linspace(-0.4, 0.4, 12).astype(np.float32)
    sig = StableInteractionGraph(12, static_bias=bias, learn_graph=False)
    markov = MatchedPhaseMarkovMixtureModel(12, bias, n_components=1)
    with torch.no_grad():
        sig.phase_loading.zero_()
        markov.transition.zero_()
        markov.bias_offset.zero_()
        markov.phase_basis.zero_()
    sig_nll = float(sig.nll_per_decision(groups, counts))
    markov_nll = float(
        markov.conditional_nll(groups, counts)["nll_per_decision"]
    )
    assert abs(sig_nll - markov_nll) < 1e-6


def test_sig_free_rollout_matches_the_shared_generation_contract():
    """The rollout may read the first rank, the length and the cardinalities."""
    data = simulate_synthetic_events(48, starts=[0, 3, 6], seed=53)
    groups, counts = data.torch()
    model = _fitted_shaped_model(seed=17)
    schedule = torch.as_tensor(
        cardinality_schedule(data.group_ids, data.group_count),
        dtype=torch.long,
    )
    own = model.rollout(
        groups == 0,
        counts,
        schedule,
        generator=torch.Generator().manual_seed(5),
    )
    shared = generate_from_step_logits(
        _shared_loop_logits(model, counts),
        groups,
        counts,
        generator=torch.Generator().manual_seed(5),
    )
    assert torch.equal(own, shared)


def test_aggregated_runs_must_share_one_source_and_config():
    left = {"config_sha256": "c", "source_sha256": {"runner": "r"}}
    right = {"config_sha256": "c", "source_sha256": {"runner": "r"}}
    assert uniform_provenance(
        [left, right],
        ("config_sha256", "source_sha256"),
        current_source_sha256={"runner": "r"},
    )["config_sha256"] == "c"
    with pytest.raises(RuntimeError, match="mix"):
        uniform_provenance(
            [left, {**right, "config_sha256": "other"}], ("config_sha256",)
        )
    with pytest.raises(RuntimeError, match="missing"):
        uniform_provenance([{"config_sha256": "c"}], ("source_sha256",))
    with pytest.raises(RuntimeError, match="fit-time source"):
        uniform_provenance(
            [left, right],
            ("source_sha256",),
            current_source_sha256={"runner": "edited"},
        )


def test_fit_records_preupdate_validation_and_fail_closed_adequacy():
    train = simulate_synthetic_events(80, starts=[0, 3, 6], seed=101)
    validation = simulate_synthetic_events(40, starts=[0, 3, 6], seed=102)
    fitted = fit_synthetic_sig(
        train,
        validation,
        seed=103,
        learn_graph=False,
        max_epochs=60,
        patience=3,
        learning_rate=0.0,
        batch_size=40,
        minimum_relative_improvement=0.0,
        minimum_training_epochs=5,
        minimum_best_epoch=0,
        maximum_recovery_depth=0,
    )
    assert fitted.history[0]["epoch"] == 0.0
    assert fitted.history[0]["train_nll_per_decision"] is None
    assert fitted.adequacy["converged"]
    assert fitted.best_optimizer_state
