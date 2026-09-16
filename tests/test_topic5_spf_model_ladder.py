"""Contract tests for the SPF-RNN comparison ladder M0-M4.

The G1 verdict compares complete-suffix likelihoods across models. All models
share the exact conditional ``k``-subset observation likelihood and the same
inner-train events; non-latent models maximize it directly and latent models
optimize its ELBO. These tests pin that parity, M3-vs-M4 isolation, and the
training-adequacy verdict that decides whether a run may enter G1 at all.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.topic5_shared_propagation_field import (  # noqa: E402
    LatentTemplateModel,
    MarkovMixtureModel,
    PhaseConditionedPropagationFieldRNN,
    SharedPropagationFieldRNN,
    baseline_conditioned_log_likelihood,
    estimate_static_participation_bias,
    fit_static_scaffold_ml,
    generate_from_step_logits,
    log_elementary_symmetric,
    suffix_log_likelihood,
    training_adequacy_verdict,
)


def _events() -> tuple[np.ndarray, np.ndarray]:
    """A small patient with reproducible, non-degenerate propagation order."""
    rng = np.random.default_rng(11)
    groups = []
    counts = []
    for _ in range(160):
        n_participants = int(rng.integers(3, 6))
        order = rng.permutation(6)[:n_participants]
        event = np.full(6, -1, dtype=np.int16)
        # Bias the order so that a first-order transition carries real signal.
        order = order[np.argsort(order + rng.normal(0.0, 0.6, size=n_participants))]
        for rank, contact in enumerate(order):
            event[contact] = rank
        groups.append(event)
        counts.append(n_participants)
    return np.asarray(groups, dtype=np.int16), np.asarray(counts, dtype=np.int16)


def _tensors(groups: np.ndarray, counts: np.ndarray):
    return (
        torch.as_tensor(groups, dtype=torch.long),
        torch.as_tensor(counts, dtype=torch.long),
    )


def test_shared_step_loop_reproduces_the_existing_baseline_likelihood() -> None:
    """The refactored step loop must not change M0/M1 numbers."""
    groups, counts = _events()
    tensor_groups, tensor_counts = _tensors(groups, counts)
    bias = torch.as_tensor(
        estimate_static_participation_bias(groups, np.arange(120)),
        dtype=torch.float32,
    )
    residual = torch.randn(6, 6, generator=torch.Generator().manual_seed(3)) * 0.3

    for transition in (None, residual):
        reference = baseline_conditioned_log_likelihood(
            bias, tensor_groups, tensor_counts, transition_residual=transition
        )

        def logit_fn(step, previous, active, transition=transition):
            logits = bias[None, :].expand(tensor_groups.shape[0], -1)
            if transition is None:
                return logits
            weight = previous.to(bias.dtype)
            return logits + weight @ transition / weight.sum(
                1, keepdim=True
            ).clamp_min(1.0)

        observed = suffix_log_likelihood(logit_fn, tensor_groups, tensor_counts)
        torch.testing.assert_close(
            observed["event_log_probability"],
            reference["event_log_probability"],
        )
        assert observed["step_log_probability"].shape[0] == len(tensor_groups)
        assert observed["step_active"].dtype == torch.bool
        torch.testing.assert_close(
            observed["step_log_probability"].sum(1),
            observed["event_log_probability"],
        )


def test_ml_scaffold_beats_the_moment_estimate_on_its_own_fitting_events() -> None:
    """M0 must be the maximum-likelihood static model, not a moment estimate.

    Otherwise an M4-vs-M0/M1 gap conflates mechanism with fitting procedure.
    """
    groups, counts = _events()
    indices = np.arange(120)
    tensor_groups, tensor_counts = _tensors(groups[indices], counts[indices])
    moment = torch.as_tensor(
        estimate_static_participation_bias(groups, indices), dtype=torch.float32
    )
    maximum_likelihood = torch.as_tensor(
        fit_static_scaffold_ml(groups, counts, indices, steps=400, seed=5),
        dtype=torch.float32,
    )
    moment_nll = baseline_conditioned_log_likelihood(
        moment, tensor_groups, tensor_counts
    )["nll_per_event"]
    ml_nll = baseline_conditioned_log_likelihood(
        maximum_likelihood, tensor_groups, tensor_counts
    )["nll_per_event"]
    assert float(ml_nll) <= float(moment_nll) + 1e-4


def test_markov_family_reduces_to_the_frozen_static_scaffold() -> None:
    """K=1 without a transition term must equal M0 exactly."""
    groups, counts = _events()
    tensor_groups, tensor_counts = _tensors(groups, counts)
    bias = np.zeros(6, dtype=np.float32)
    model = MarkovMixtureModel(6, bias, n_components=1, use_transition=False)
    observed = model.event_log_probability(tensor_groups, tensor_counts)
    reference = baseline_conditioned_log_likelihood(
        torch.as_tensor(bias), tensor_groups, tensor_counts
    )["event_log_probability"]
    torch.testing.assert_close(observed, reference)


def test_markov_mixture_of_identical_components_equals_one_component() -> None:
    groups, counts = _events()
    tensor_groups, tensor_counts = _tensors(groups, counts)
    bias = np.zeros(6, dtype=np.float32)
    single = MarkovMixtureModel(6, bias, n_components=1)
    mixture = MarkovMixtureModel(6, bias, n_components=3)
    with torch.no_grad():
        for component in range(3):
            mixture.bias_offset[component].copy_(single.bias_offset[0])
            mixture.transition[component].copy_(single.transition[0])
        mixture.mixture_logit.zero_()
    torch.testing.assert_close(
        mixture.event_log_probability(tensor_groups, tensor_counts),
        single.event_log_probability(tensor_groups, tensor_counts),
    )


def test_cardinality_one_fast_path_equals_the_dynamic_program() -> None:
    """The k<=1 shortcut must be exact, not an approximation.

    It carries nearly all of the training cost because every subject in the
    frozen cohort has median rank-set size 1.
    """
    generator = torch.Generator().manual_seed(23)
    logits = torch.randn(64, 9, dtype=torch.float64, generator=generator)
    logits.requires_grad_(True)
    candidate = (
        torch.rand(64, 9, generator=generator, dtype=torch.float64) > 0.3
    )
    candidate[:, 0] = True  # guarantee a non-empty candidate set
    cardinality = torch.ones(64, dtype=torch.long)
    fast = log_elementary_symmetric(logits, candidate, cardinality)

    # Reference: brute-force e_1 over the candidate set of each row.
    reference = torch.stack(
        [
            torch.logsumexp(logits[row][candidate[row]], dim=0)
            for row in range(64)
        ]
    )
    torch.testing.assert_close(fast, reference)
    fast.sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.all(logits.grad[~candidate] == 0)


def test_mixture_components_are_not_initialized_identically() -> None:
    """Identical components share a gradient and can never separate.

    A K>1 model that silently stays at K=1 would be reported as "a small
    repertoire of discrete routes did not help" when it was never fitted.
    """
    bias = np.zeros(6, dtype=np.float32)
    mixture = MarkovMixtureModel(6, bias, n_components=3)
    assert not torch.allclose(mixture.transition[0], mixture.transition[1])
    assert not torch.allclose(mixture.bias_offset[0], mixture.bias_offset[1])
    single = MarkovMixtureModel(6, bias, n_components=1)
    torch.testing.assert_close(single.transition, torch.zeros_like(single.transition))


def test_phase_matched_markov_reads_progress_but_stationary_markov_does_not() -> None:
    bias = np.zeros(6, dtype=np.float32)
    previous = torch.tensor(
        [[True, False, False, False, False, False]], dtype=torch.bool
    )
    counts = torch.tensor([6], dtype=torch.long)
    stationary = MarkovMixtureModel(6, bias, n_components=1)
    clocked = MarkovMixtureModel(6, bias, n_components=1, phase_order=2)
    with torch.no_grad():
        stationary.transition.zero_()
        clocked.transition.zero_()
        clocked.phase_basis[0, 0] = torch.arange(6, dtype=torch.float32)
    torch.testing.assert_close(
        stationary.component_logits(
            0, previous, step=1, group_count=counts
        ),
        stationary.component_logits(
            0, previous, step=4, group_count=counts
        ),
    )
    assert not torch.allclose(
        clocked.component_logits(0, previous, step=1, group_count=counts),
        clocked.component_logits(0, previous, step=4, group_count=counts),
    )


def test_latent_template_has_no_autonomous_recurrence() -> None:
    """M3's state at step t is a function of (z0, t/T) alone.

    If M3 could carry state forward it would stop being the control that
    separates "low-dimensional time template" from "autonomous dynamics".
    """
    groups, counts = _events()
    _, tensor_counts = _tensors(groups[:4], counts[:4])
    model = LatentTemplateModel(6, np.zeros(6, dtype=np.float32), latent_dim=3)
    initial = torch.randn(4, 3, generator=torch.Generator().manual_seed(7))
    forward = model.state_factory(initial, tensor_counts)
    active = torch.ones(4, dtype=torch.bool)
    ordered = [forward(step, active) for step in range(1, 4)]
    # Re-entering at an arbitrary step must give the identical state.
    reentry = model.state_factory(initial, tensor_counts)
    assert torch.allclose(reentry(3, active), ordered[2])
    assert not hasattr(model, "field_weight")


def test_template_and_field_share_decoder_and_encoder_shapes() -> None:
    """Only the trajectory generator may differ between M3 and M4."""
    bias = np.zeros(6, dtype=np.float32)
    template = LatentTemplateModel(6, bias, latent_dim=4, encoder_hidden=16)
    field = SharedPropagationFieldRNN(6, bias, latent_dim=4, encoder_hidden=16)
    assert template.contact_loading.shape == field.contact_loading.shape
    for left, right in zip(
        template.prior_head.parameters(), field.prior_head.parameters()
    ):
        assert left.shape == right.shape
    for left, right in zip(
        template.posterior_head.parameters(), field.posterior_head.parameters()
    ):
        assert left.shape == right.shape


def test_clocked_field_is_explicitly_nonautonomous() -> None:
    bias = np.zeros(6, dtype=np.float32)
    initial = torch.zeros(2, 3)
    counts = torch.tensor([4, 8], dtype=torch.long)
    active = torch.ones(2, dtype=torch.bool)
    autonomous = SharedPropagationFieldRNN(6, bias, latent_dim=3)
    clocked = PhaseConditionedPropagationFieldRNN(6, bias, latent_dim=3)
    with torch.no_grad():
        autonomous.field_weight.zero_()
        autonomous.field_bias.zero_()
        clocked.field_weight.zero_()
        clocked.field_bias.zero_()
        clocked.phase_drive.fill_(1.0)
    autonomous_state = autonomous.state_factory(initial, counts)(1, active)
    clocked_state = clocked.state_factory(initial, counts)(1, active)
    torch.testing.assert_close(autonomous_state[0], autonomous_state[1])
    assert not torch.allclose(clocked_state[0], clocked_state[1])


@pytest.mark.parametrize("model_name", ["template", "field", "markov"])
def test_every_model_generates_under_the_identical_fixed_schedule(
    model_name: str,
) -> None:
    groups, counts = _events()
    tensor_groups, tensor_counts = _tensors(groups[:32], counts[:32])
    bias = np.zeros(6, dtype=np.float32)
    if model_name == "template":
        generated = LatentTemplateModel(6, bias, latent_dim=3).generate_conditioned(
            tensor_groups, tensor_counts, seed=13
        )
    elif model_name == "field":
        generated = SharedPropagationFieldRNN(
            6, bias, latent_dim=3
        ).generate_conditioned(tensor_groups, tensor_counts, seed=13)
    else:
        generated = MarkovMixtureModel(6, bias, n_components=2).generate_conditioned(
            tensor_groups, tensor_counts, seed=13
        )
    assert torch.equal(generated == 0, tensor_groups == 0)
    for observed, prediction, count in zip(tensor_groups, generated, tensor_counts):
        for step in range(int(count)):
            assert int((observed == step).sum()) == int((prediction == step).sum())
        assert int((prediction >= 0).sum()) == int((observed >= 0).sum())


def test_free_running_generation_feeds_back_only_generated_sets() -> None:
    """Generation must never read the observed suffix identities."""
    conditioning = torch.tensor(
        [[0, 1, 2, 3, -1, -1], [0, 3, 2, 1, -1, -1]], dtype=torch.long
    )
    counts = torch.tensor([4, 4], dtype=torch.long)
    seen: list[torch.Tensor] = []

    def logit_fn(step, previous, active):
        seen.append(previous.clone())
        return torch.zeros(
            conditioning.shape[0], conditioning.shape[1], dtype=torch.float32
        )

    generator = torch.Generator().manual_seed(17)
    generated = generate_from_step_logits(
        logit_fn, conditioning, counts, generator=generator
    )
    # Both events share the same first set, so a history-driven model sees the
    # identical feedback for both rows despite different observed suffixes.
    for previous in seen[1:]:
        assert torch.equal(previous[0], previous[1]) or not torch.equal(
            generated[0], generated[1]
        )
    assert torch.equal(generated == 0, conditioning == 0)


def test_training_adequacy_verdict_separates_plateau_from_still_improving() -> None:
    plateaued = [9.0, 8.0, 7.5, 7.40, 7.39, 7.385, 7.384, 7.384]
    still_improving = [9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5]
    converged = training_adequacy_verdict(plateaued, patience=3, tolerance=0.002)
    running = training_adequacy_verdict(still_improving, patience=3, tolerance=0.002)
    assert converged["converged"] is True
    assert running["converged"] is False
    assert running["verdict"] == "NOT_CONVERGED"
    assert converged["best_epoch"] < len(plateaued)
    with pytest.raises(ValueError):
        training_adequacy_verdict([], patience=3, tolerance=0.002)


def test_training_adequacy_rejects_early_stop_without_learning_progress() -> None:
    flat = [7.0, 7.01, 7.02, 7.03, 7.04, 7.05]
    verdict = training_adequacy_verdict(
        flat,
        patience=3,
        tolerance=0.002,
        minimum_epochs=5,
        stopped_by_patience=True,
    )
    assert verdict["converged"] is False
    assert verdict["verdict"] == "NO_LEARNING_PROGRESS"


def test_training_adequacy_requires_later_optimum() -> None:
    overshoot = [7.0, 6.5, 6.7, 6.8, 6.9, 7.0, 7.1]
    verdict = training_adequacy_verdict(
        overshoot,
        patience=3,
        tolerance=0.002,
        minimum_epochs=5,
        minimum_best_epoch=3,
        stopped_by_patience=True,
    )
    assert verdict["converged"] is False
    assert verdict["verdict"] == "EARLY_OPTIMUM_UNVERIFIED"
    assert verdict["early_optimum"] is True
