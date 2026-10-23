import numpy as np

from src.interictal_rank_holdout import (
    METHODS, closest_template_error, normalized_order,
    prepare_reconstruction, shuffle_participating_orders,
)


def test_exact_order_and_phantom_rank_invariance():
    ranks = np.array([[7., 3.], [99., -100.], [2., 2.], [5., 1.]])
    mask = np.array([[1, 1], [0, 0], [1, 1], [1, 1]], bool)
    templates = {m: [[7, 0, 2, 5], [3, 123, 2, 1]] for m in METHODS}
    truth, used, predicted, eligible, coverage = prepare_reconstruction(ranks, mask, templates)
    result = closest_template_error(truth, used, predicted)
    np.testing.assert_array_equal(result['mae'], np.zeros((2, 2)))
    np.testing.assert_array_equal(result['assignments'], [[0, 1], [0, 1]])
    assert eligible.all() and coverage['n_participating_contacts'] == 6


def test_hand_computed_normalized_absolute_rank_error():
    mask = np.ones((1, 4), bool)
    truth = normalized_order([[1, 2, 3, 4]], mask)
    prediction = normalized_order([[1, 3, 2, 4]], mask)
    result = closest_template_error(truth, mask, np.tile(prediction[None, :, None, :], (2, 1, 2, 1)))
    np.testing.assert_allclose(result['mae'], 1/6)
    np.testing.assert_allclose(result['mse'], 1/18)


def test_joint_support_excludes_whole_sequence_and_preserves_missingness():
    ranks = np.array([[0, 0], [1, 1], [2, 2]], float)
    mask = np.array([[1, 1], [1, 1], [1, 0]], bool)
    templates = {METHODS[0]: [[0, 1, np.nan], [2, 1, 0]],
                 METHODS[1]: [[0, 1, 2], [2, 1, 0]]}
    _, _, _, eligible, coverage = prepare_reconstruction(ranks, mask, templates)
    np.testing.assert_array_equal(eligible, [False, True])
    assert coverage['n_unsupported_sequence'] == 1


def test_shuffle_keeps_participation_and_rank_multiset_with_ties():
    mask = np.array([[1, 0, 1, 0, 1], [0, 1, 1, 1, 0]], bool)
    truth = normalized_order([[3, 9, 3, 9, 6], [9, 1, 2, 3, 9]], mask)
    rng = np.random.default_rng(123)
    changed = False
    for _ in range(40):
        shuffled = shuffle_participating_orders(truth, mask, rng)
        assert np.all(shuffled[~mask] == 0)
        for i in range(len(mask)):
            np.testing.assert_array_equal(np.sort(shuffled[i, mask[i]]), np.sort(truth[i, mask[i]]))
        changed |= bool(np.any(shuffled != truth))
    assert changed


def test_channel_permutation_equivariance():
    rng = np.random.default_rng(3)
    ranks = rng.normal(size=(7, 25))
    mask = rng.random((7, 25)) > .2
    mask[:2] = True
    templates = {m: rng.normal(size=(2, 7)) for m in METHODS}
    prepared = prepare_reconstruction(ranks, mask, templates)
    expected = closest_template_error(*prepared[:3])['mae']
    order = rng.permutation(7)
    permuted = prepare_reconstruction(ranks[order], mask[order], {m: t[:, order] for m,t in templates.items()})
    np.testing.assert_allclose(closest_template_error(*permuted[:3])['mae'], expected)
