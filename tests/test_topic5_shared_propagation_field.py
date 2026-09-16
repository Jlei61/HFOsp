from __future__ import annotations

import itertools

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.topic5_shared_propagation_field import (
    SharedPropagationFieldRNN,
    SubjectRankEvents,
    audit_legacy_snn_lagpat,
    baseline_conditioned_log_likelihood,
    conditional_k_subset_log_prob,
    estimate_static_participation_bias,
    first_rank_mask,
    generate_first_order_markov_conditioned,
    generate_static_conditioned,
    log_elementary_symmetric,
    rank_cardinality_schedule,
    sample_conditional_k_subset,
    validate_rank_event_arrays,
)


def _events() -> tuple[np.ndarray, np.ndarray]:
    groups = np.asarray(
        [
            [0, 1, 2, -1, -1],
            [0, 2, 1, 3, -1],
            [1, 0, 2, -1, -1],
            [2, 0, 1, 3, -1],
            [0, 1, 1, 2, -1],
            [1, 0, 0, 2, -1],
        ],
        dtype=np.int16,
    )
    counts = np.asarray([3, 4, 3, 4, 3, 3], dtype=np.int16)
    return groups, counts


def test_rank_event_contract_requires_contiguous_nonempty_groups() -> None:
    groups, counts = _events()
    validate_rank_event_arrays(
        groups,
        counts,
        np.asarray([0, 0, 0, 1, 1, 1], dtype=np.uint8),
        np.arange(len(groups), dtype=float),
    )
    broken = groups.copy()
    broken[0, broken[0] == 1] = 3
    with pytest.raises(ValueError, match="non-contiguous"):
        validate_rank_event_arrays(broken, counts)
    with pytest.raises(ValueError, match="non-empty train and heldout"):
        validate_rank_event_arrays(
            groups,
            counts,
            np.zeros(len(groups), dtype=np.uint8),
            np.arange(len(groups), dtype=float),
        )


def test_development_split_keeps_monitor_and_test_disjoint_from_outer_holdout(
    tmp_path,
) -> None:
    groups, counts = _events()
    groups = np.concatenate([groups, groups], axis=0)
    counts = np.concatenate([counts, counts], axis=0)
    split = np.asarray([0] * 9 + [1] * 3, dtype=np.uint8)
    record = SubjectRankEvents(
        subject="toy",
        dataset="toy",
        path=tmp_path / "toy.npz",
        contact_names=np.asarray(["A", "B", "C", "D", "E"]),
        group_ids=groups,
        group_count=counts,
        event_split=split,
        event_abs_time=np.arange(12, dtype=float),
        event_source_index=np.arange(12),
        input_sha256="toy",
        target_values_read=False,
    )
    train, monitor, test = record.development_split(0.2, 0.2)
    assert max(train) < min(monitor) < min(test)
    assert not np.intersect1d(np.r_[train, monitor, test], record.old_heldout20_indices).size
    assert set(np.r_[train, monitor, test]) == set(record.train80_indices)


def test_rank_schedule_preserves_tied_set_cardinality() -> None:
    groups, counts = _events()
    schedule = rank_cardinality_schedule(groups, counts)
    assert schedule.shape == (6, 3)
    assert schedule[4].tolist() == [2, 1, 0]
    assert np.all(first_rank_mask(groups).sum(1) >= 1)


def test_exact_k_subset_normalizer_matches_brute_force() -> None:
    logits = torch.tensor(
        [[0.2, -0.4, 1.1, 0.7]], dtype=torch.float64, requires_grad=True
    )
    candidate = torch.tensor([[True, False, True, True]])
    cardinality = torch.tensor([2])
    observed = log_elementary_symmetric(logits, candidate, cardinality)
    eligible = [0, 2, 3]
    brute = torch.logsumexp(
        torch.stack(
            [logits[0, list(pair)].sum() for pair in itertools.combinations(eligible, 2)]
        ),
        dim=0,
    )
    torch.testing.assert_close(observed[0], brute)
    observed.sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 1] == 0


def test_exact_k_subset_likelihood_is_unordered_and_finite() -> None:
    logits = torch.tensor([[0.5, 0.1, -0.2, 0.7]], requires_grad=True)
    candidate = torch.ones_like(logits, dtype=torch.bool)
    target = torch.tensor([[True, False, True, False]])
    first = conditional_k_subset_log_prob(logits, target, candidate)
    permuted_logits = logits[:, [2, 1, 0, 3]]
    permuted_target = target[:, [2, 1, 0, 3]]
    permuted_candidate = candidate[:, [2, 1, 0, 3]]
    second = conditional_k_subset_log_prob(
        permuted_logits, permuted_target, permuted_candidate
    )
    torch.testing.assert_close(first, second)
    (-first.mean()).backward()
    assert torch.isfinite(logits.grad).all()


def test_exact_subset_sampler_matches_small_enumerated_distribution() -> None:
    logits = torch.tensor([[0.0, 0.7, -0.4]], dtype=torch.float64).expand(8000, -1)
    candidate = torch.ones_like(logits, dtype=torch.bool)
    cardinality = torch.ones(8000, dtype=torch.long)
    generator = torch.Generator().manual_seed(19)
    sample = sample_conditional_k_subset(
        logits, candidate, cardinality, generator=generator
    )
    observed = sample.to(torch.float64).mean(0)
    expected = torch.softmax(logits[0], dim=0)
    torch.testing.assert_close(observed, expected, atol=0.02, rtol=0.0)
    assert torch.all(sample.sum(1) == 1)


def test_static_bias_uses_only_requested_training_events() -> None:
    groups, _ = _events()
    first = estimate_static_participation_bias(groups, [0, 1])
    second = estimate_static_participation_bias(groups, [4, 5])
    assert not np.allclose(first, second)
    assert np.isfinite(first).all()


def test_spf_training_loss_is_finite_and_static_bias_is_frozen() -> None:
    groups, counts = _events()
    static = estimate_static_participation_bias(groups, np.arange(4))
    model = SharedPropagationFieldRNN(
        groups.shape[1],
        static,
        latent_dim=3,
        encoder_hidden=12,
    )
    tensor_groups = torch.as_tensor(groups, dtype=torch.long)
    tensor_counts = torch.as_tensor(counts, dtype=torch.long)
    loss = model.elbo_loss(
        tensor_groups,
        tensor_counts,
        beta=0.2,
        free_bits=0.01,
        jacobian_weight=0.001,
        weight_decay=0.0001,
    )
    assert torch.isfinite(loss["loss"])
    loss["loss"].backward()
    assert model.static_bias.grad is None
    assert model.field_weight.grad is not None
    assert model.contact_loading.grad is not None


def test_prior_is_future_blind_but_posterior_can_read_full_event() -> None:
    groups, counts = _events()
    static = estimate_static_participation_bias(groups, np.arange(4))
    model = SharedPropagationFieldRNN(groups.shape[1], static, latent_dim=2)
    # Same first set and same length/cardinality, different future order.
    paired = torch.tensor(
        [[0, 1, 2, 3, -1], [0, 3, 2, 1, -1]], dtype=torch.long
    )
    paired_count = torch.tensor([4, 4], dtype=torch.long)
    p_mean, p_logvar = model.prior_parameters(paired == 0)
    torch.testing.assert_close(p_mean[0], p_mean[1])
    torch.testing.assert_close(p_logvar[0], p_logvar[1])
    q_mean, _ = model.posterior_parameters(paired, paired_count)
    # This is a capacity check, not a requirement that random initialization
    # separate the pair strongly.
    assert q_mean.shape == (2, 2)


def test_conditioned_generation_never_reads_future_identity() -> None:
    groups = torch.tensor(
        [[0, 1, 2, 3, -1], [0, 3, 2, 1, -1]], dtype=torch.long
    )
    counts = torch.tensor([4, 4], dtype=torch.long)
    static = np.zeros(5, dtype=np.float32)
    model = SharedPropagationFieldRNN(5, static, latent_dim=2)
    first = model.generate_conditioned(groups[:1], counts[:1], seed=29)
    second = model.generate_conditioned(groups[1:], counts[1:], seed=29)
    assert torch.equal(first, second)
    assert torch.equal(first == 0, groups[:1] == 0)
    assert torch.equal(
        torch.sort(first[first >= 0]).values,
        torch.arange(4, dtype=torch.long),
    )


@pytest.mark.parametrize("baseline", ["static", "markov"])
def test_fixed_schedule_baselines_preserve_first_set_and_cardinalities(
    baseline: str,
) -> None:
    groups, counts = _events()
    tensor_groups = torch.as_tensor(groups, dtype=torch.long)
    tensor_counts = torch.as_tensor(counts, dtype=torch.long)
    static = torch.as_tensor(
        estimate_static_participation_bias(groups, np.arange(4))
    )
    if baseline == "static":
        generated = generate_static_conditioned(
            static, tensor_groups, tensor_counts, seed=37
        )
    else:
        residual = torch.zeros((groups.shape[1], groups.shape[1]))
        generated = generate_first_order_markov_conditioned(
            static,
            residual,
            tensor_groups,
            tensor_counts,
            seed=37,
        )
    assert torch.equal(generated == 0, tensor_groups == 0)
    for observed, prediction, count in zip(
        tensor_groups, generated, tensor_counts
    ):
        for step in range(int(count)):
            assert int(torch.sum(observed == step)) == int(
                torch.sum(prediction == step)
            )
        assert int(torch.sum(prediction >= 0)) == int(
            torch.sum(observed >= 0)
        )


def test_baseline_likelihood_is_exact_and_markov_interface_is_separate() -> None:
    groups, counts = _events()
    tensor_groups = torch.as_tensor(groups, dtype=torch.long)
    tensor_counts = torch.as_tensor(counts, dtype=torch.long)
    static = torch.zeros(groups.shape[1])
    residual = torch.zeros((groups.shape[1], groups.shape[1]))
    m0 = baseline_conditioned_log_likelihood(
        static, tensor_groups, tensor_counts
    )
    m1 = baseline_conditioned_log_likelihood(
        static,
        tensor_groups,
        tensor_counts,
        transition_residual=residual,
    )
    torch.testing.assert_close(m0["event_log_probability"], m1["event_log_probability"])
    assert torch.isfinite(m0["nll_per_decision"])


def test_legacy_snn_audit_does_not_unpickle_object_channel_names(
    tmp_path,
) -> None:
    path = tmp_path / "trusted_local_lagPat_withFreqCent.npz"
    np.savez(
        path,
        lagPatRank=np.asarray([[0, 1], [1, 0]], dtype=np.int16),
        eventsBool=np.ones((2, 2), dtype=bool),
        chnNames=np.asarray(["A1", "A2"], dtype=object),
    )
    result = audit_legacy_snn_lagpat(path)
    assert result["n_events"] == 2
    assert result["channel_name_key_present"]
    assert result["channel_name_count_status"] == "NOT_READ_OBJECT_ARRAY"
