import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.topic5_minimal_sequence_kernel import (
    ResidualFIRH3SequenceModel,
    block_hankel_from_lag_kernels,
    decomposed_next_set_stop_loss,
    hankel_singular_summary,
    linear_state_contact_lag_kernels,
    linear_state_lag_ablation_outputs,
    merge_frozen_groups_by_lag_tolerance,
)
from src.topic5_rank_distribution import (
    LinearStateSequenceRNN,
    StaticSequenceContactQuery,
    next_set_stop_loss,
)


def _kwargs():
    return {
        "hidden_size": 7,
        "contact_embedding_dim": 6,
        "contact_encoder_hidden": 5,
        "local_offset_dim": 3,
    }


def _batch():
    features = torch.randn(3, 6, 8)
    mask = torch.ones(3, 6, dtype=torch.bool)
    groups = torch.tensor(
        [
            [0, 1, 2, 3, -1, -1],
            [1, 0, 2, -1, 3, -1],
            [0, 0, 1, 2, -1, -1],
        ]
    )
    counts = torch.tensor([4, 4, 3])
    offset = torch.zeros(6, 3)
    return features, mask, groups, counts, offset


def test_joint_likelihood_decomposition_exactly_reconstructs_frozen_loss():
    model = LinearStateSequenceRNN(8, **_kwargs())
    features, mask, groups, counts, offset = _batch()
    outputs = model(features, mask, groups, counts, offset)
    original = next_set_stop_loss(outputs, groups, counts)
    split = decomposed_next_set_stop_loss(outputs, groups, counts)
    torch.testing.assert_close(original["event_nll"], split["event_total_nll"])
    torch.testing.assert_close(original["total"], split["total"])
    torch.testing.assert_close(
        split["event_total_nll"],
        split["event_stop_contribution_nll"]
        + split["event_contact_contribution_nll"],
    )


def test_stop_probability_uses_joint_softmax_not_raw_logit_sigmoid():
    groups = torch.tensor([[0, -1]])
    counts = torch.tensor([1])
    outputs = {
        "contact_logits": torch.tensor([[[0.0, -1.0e9], [-1.0e9, 0.0]]]),
        "stop_logits": torch.tensor([[0.0, 0.0]]),
        "candidate_mask": torch.tensor(
            [[[True, False], [False, True]]]
        ),
    }
    split = decomposed_next_set_stop_loss(outputs, groups, counts)
    # A raw sigmoid would be 0.5. With one equally likely candidate, the
    # joint-softmax STOP probability is also 0.5 here; add a second eligible
    # contact to make the distinction explicit.
    outputs["contact_logits"][0, 0, 1] = 0.0
    outputs["candidate_mask"][0, 0, 1] = True
    split = decomposed_next_set_stop_loss(outputs, groups, counts)
    assert split["stop_probability"][0, 0].item() == pytest.approx(1.0 / 3.0)


def test_zero_initialized_fir_matches_copied_unordered_baseline():
    baseline = StaticSequenceContactQuery(8, mode="unordered", **_kwargs())
    fir = ResidualFIRH3SequenceModel(8, **_kwargs())
    missing, unexpected = fir.load_state_dict(
        baseline.state_dict(), strict=False
    )
    assert all(key.startswith("lag_projections.") for key in missing)
    assert not unexpected
    features, mask, groups, counts, offset = _batch()
    expected = baseline(features, mask, groups, counts, offset)
    observed = fir(features, mask, groups, counts, offset)
    torch.testing.assert_close(
        expected["contact_logits"], observed["contact_logits"]
    )
    torch.testing.assert_close(expected["stop_logits"], observed["stop_logits"])
    assert torch.equal(expected["candidate_mask"], observed["candidate_mask"])


def test_fir_freeze_leaves_only_three_ordered_projections_trainable():
    model = ResidualFIRH3SequenceModel(8, **_kwargs())
    model.freeze_unordered_baseline()
    trainable = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable == {
        f"lag_projections.{lag}.weight" for lag in range(3)
    }


def test_fir_has_exact_three_rank_identity_horizon():
    model = ResidualFIRH3SequenceModel(8, **_kwargs())
    with torch.no_grad():
        for lag, projection in enumerate(model.lag_projections):
            projection.weight.fill_(0.01 * (lag + 1))
    features = torch.randn(1, 7, 8).expand(2, -1, -1).clone()
    mask = torch.ones(2, 7, dtype=torch.bool)
    # At prediction step 5, the unordered prefix and last three ranks are
    # identical. Only ranks 0 and 1 are swapped, so FIR-H3 must agree.
    groups = torch.tensor(
        [
            [0, 1, 2, 3, 4, -1, -1],
            [1, 0, 2, 3, 4, -1, -1],
        ]
    )
    counts = torch.tensor([5, 5])
    outputs = model(features, mask, groups, counts, torch.zeros(7, 3))
    torch.testing.assert_close(
        outputs["contact_logits"][0, 5],
        outputs["contact_logits"][1, 5],
    )
    torch.testing.assert_close(
        outputs["stop_logits"][0, 5],
        outputs["stop_logits"][1, 5],
    )


def test_linear_no_ablation_replay_matches_forward_and_kernels_have_shapes():
    model = LinearStateSequenceRNN(8, **_kwargs())
    features, mask, groups, counts, offset = _batch()
    direct = model(features, mask, groups, counts, offset)
    replay = linear_state_lag_ablation_outputs(
        model,
        features,
        mask,
        groups,
        counts,
        offset,
    )
    torch.testing.assert_close(direct["contact_logits"], replay["contact_logits"])
    torch.testing.assert_close(direct["stop_logits"], replay["stop_logits"])
    assert torch.equal(direct["candidate_mask"], replay["candidate_mask"])

    kernels = linear_state_contact_lag_kernels(
        model,
        features[:1],
        offset,
        max_lag=4,
    )
    assert kernels["contact"].shape == (5, 6, 6)
    assert kernels["stop"].shape == (5, 1, 6)
    assert torch.isfinite(kernels["contact"]).all()


def test_block_hankel_and_singular_summary():
    kernels = np.arange(5 * 2 * 3, dtype=float).reshape(5, 2, 3)
    hankel = block_hankel_from_lag_kernels(kernels)
    assert hankel.shape == (6, 9)
    np.testing.assert_array_equal(hankel[:2, :3], kernels[0])
    np.testing.assert_array_equal(hankel[4:6, 6:9], kernels[4])
    summary = hankel_singular_summary(hankel)
    assert summary["rank90"] >= 1
    assert summary["rank95"] >= summary["rank90"]
    assert np.all(np.diff(summary["cumulative_energy"]) >= -1.0e-12)


def test_tolerance_merge_preserves_frozen_zero_and_only_merges_adjacent_groups():
    groups = np.asarray([[0, 1, 2, -1], [0, 0, 1, 2]], dtype=np.int16)
    counts = np.asarray([3, 3], dtype=np.int16)
    # The first two saved float32 lags are equal despite distinct frozen
    # groups. Zero tolerance must therefore keep the frozen encoding.
    lag = np.asarray(
        [[0.1, 0.1, 0.104, np.nan], [0.0, 0.0, 0.002, 0.020]],
        dtype=np.float32,
    )
    zero_groups, zero_counts = merge_frozen_groups_by_lag_tolerance(
        groups, counts, lag, tolerance_seconds=0.0
    )
    np.testing.assert_array_equal(zero_groups, groups)
    np.testing.assert_array_equal(zero_counts, counts)

    merged, merged_counts = merge_frozen_groups_by_lag_tolerance(
        groups, counts, lag, tolerance_seconds=0.005
    )
    np.testing.assert_array_equal(merged[0], [0, 0, 0, -1])
    np.testing.assert_array_equal(merged[1], [0, 0, 0, 1])
    np.testing.assert_array_equal(merged_counts, [1, 2])
