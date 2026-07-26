import numpy as np
import pytest

from src.topic5_state_conditioned_rnn import (
    LREICTRNN,
    derive_prefix_axis,
    robust_rebaseline_activation,
    signed_axis_label,
    swap_ab_features,
)


def _toy_prefix(seed=1, n_event=120):
    rng = np.random.default_rng(seed)
    n_ch = 8
    bools = rng.random((n_ch, n_event)) < 0.85
    bools[:3, :] = True
    ranks = np.zeros((n_ch, n_event), dtype=float)
    order_a = np.arange(n_ch)
    order_b = order_a[::-1]
    for e in range(n_event):
        order = order_a if e < n_event // 2 else order_b
        noisy = order + rng.normal(0, 0.05, n_ch)
        ranks[:, e] = np.argsort(np.argsort(noisy))
        # Deliberately poisonous phantom values must be ignored.
        ranks[~bools[:, e], e] = 1000 + rng.integers(0, 100, np.sum(~bools[:, e]))
    return ranks, bools


def test_prefix_axis_ignores_phantom_ranks_and_is_stable():
    ranks, bools = _toy_prefix()
    axis = derive_prefix_axis(ranks, bools, np.arange(ranks.shape[1]))
    assert axis["seed_ami"] > 0.95
    assert min(axis["cluster_fractions"]) > 0.4
    q = np.asarray(axis["support_q"])
    b = np.asarray(axis["direction_basis"])
    assert np.isclose(np.sum((q / q.sum()) * b * b), 1.0)


def test_prefix_axis_is_invariant_to_post_prefix_events():
    ranks, bools = _toy_prefix(n_event=160)
    prefix = np.arange(100)
    first = derive_prefix_axis(ranks, bools, prefix)
    changed = ranks.copy()
    changed[:, 100:] = changed[::-1, 100:] + 5000
    second = derive_prefix_axis(changed, bools, prefix)
    np.testing.assert_allclose(first["direction_basis"], second["direction_basis"], equal_nan=True)
    np.testing.assert_allclose(first["support_q"], second["support_q"], equal_nan=True)


def test_fixed_baseline_and_eeg_relative_target_window():
    rel = np.arange(-130, 20.1, 0.1)
    z = np.zeros((2, rel.size))
    z[:, (rel >= -120) & (rel <= -90)] = np.linspace(
        -1, 1, np.sum((rel >= -120) & (rel <= -90))
    )
    z[0, (rel >= 0) & (rel <= 10)] = 3.0
    z[1, (rel >= 0) & (rel <= 10)] = -3.0
    activation = robust_rebaseline_activation(z, rel, onset_rel=0.0)
    assert activation[0] > 0
    assert activation[1] < 0


def test_ab_swap_changes_input_and_label_sign():
    x = np.array([[2.0, 0.5, -0.25, 0.1, 100.0, 0.02, 1.2]])
    swapped = swap_ab_features(x)
    assert swapped[0, 0] == -2.0
    assert swapped[0, 2] == 0.25
    np.testing.assert_allclose(swapped[0, [1, 3, 4, 5, 6]], x[0, [1, 3, 4, 5, 6]])

    y = np.array([2.0, -2.0, 0.0, 0.0])
    names = ["a", "b", "c", "d"]
    b = np.array([1.0, -1.0, 0.0, 0.0])
    q = np.ones(4)
    c = signed_axis_label(y, names, names, b, q)["coefficient"]
    c_swap = signed_axis_label(y, names, names, -b, q)["coefficient"]
    assert c == pytest.approx(-c_swap)


def test_dale_low_rank_has_presynaptic_column_sign():
    torch = pytest.importorskip("torch")
    model = LREICTRNN(7, 2, dale=True)
    w = model.low_rank_matrix().detach().cpu().numpy()
    sign = model.unit_sign.detach().cpu().numpy()
    assert np.all(w[:, sign > 0] >= -1e-8)
    assert np.all(w[:, sign < 0] <= 1e-8)


def test_event_driven_core_shapes_and_gradients():
    torch = pytest.importorskip("torch")
    model = LREICTRNN(7, 1)
    events = torch.randn(3, 9, 7)
    dt = torch.rand(3, 9) * 60
    mask = torch.ones(3, 9, dtype=torch.bool)
    final, sequence = model(events, dt, mask)
    assert final.shape == (3, 61)
    assert sequence.shape == (3, 9, 61)
    final.square().mean().backward()
    assert any(p.grad is not None for p in model.parameters())
