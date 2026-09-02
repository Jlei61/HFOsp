import numpy as np
import torch

from src.topic5_shared_functional_necessity_v0_1 import (
    centered_normalized_operator,
    dose_auc,
    drop_heldout_outcome_fields,
    equal_norm_toward_center,
    holm_adjust,
    leave_one_topology_component,
    orthonormal_row_basis,
    projection_erasure,
    rank_set_nll,
    subspace_projection_erasure,
)


def test_heldout_suffix_derived_fields_are_removed_before_lesion_geometry():
    archive = {
        "hidden": np.ones((2, 3)),
        "event_u": np.asarray([99.0, -99.0]),
        "conditional_center": np.full((2, 3), 123.0),
        "phase_target": np.asarray([0.25, 0.75]),
    }
    cleaned = drop_heldout_outcome_fields(archive)
    assert set(cleaned) == {"hidden", "phase_target"}
    np.testing.assert_array_equal(cleaned["hidden"], archive["hidden"])


def test_centered_operator_removes_common_contact_shift_and_normalizes():
    operator = np.arange(20.0).reshape(4, 5) + np.arange(5.0)[None, :] * 10
    value = centered_normalized_operator(operator)
    np.testing.assert_allclose(value.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(value), 1.0, atol=1e-12)


def test_leave_one_topology_never_reads_heldout_operator():
    rng = np.random.default_rng(2)
    operators = {arm: rng.normal(size=(6, 8)) for arm in ("L0", "L1", "L2m", "L3")}
    basis = np.eye(8)
    first = leave_one_topology_component(operators, "L3", basis)
    operators["L3"] = rng.normal(size=(6, 8)) * 1e6
    second = leave_one_topology_component(operators, "L3", basis)
    np.testing.assert_allclose(first["hidden_components"], second["hidden_components"])
    assert set(first["source_arms"].tolist()) == {"L0", "L1", "L2m"}


def test_projection_erasure_is_sign_invariant_and_removes_projection():
    hidden = np.asarray([[2.0, 3.0], [-1.0, 2.0]])
    center = np.zeros_like(hidden)
    positive, delta_positive, _ = projection_erasure(hidden, center, np.asarray([1.0, 0.0]), 1.0)
    negative, delta_negative, _ = projection_erasure(hidden, center, np.asarray([-1.0, 0.0]), 1.0)
    np.testing.assert_allclose(positive, negative)
    np.testing.assert_allclose(delta_positive, delta_negative)
    np.testing.assert_allclose(positive[:, 0], 0.0, atol=1e-12)


def test_control_displacement_has_exact_target_norm():
    hidden = np.asarray([[2.0, 1.0], [-2.0, 1.0]])
    center = np.zeros_like(hidden)
    moved, delta = equal_norm_toward_center(
        hidden, center, np.asarray([0.0, 1.0]), np.asarray([0.3, 0.7])
    )
    np.testing.assert_allclose(np.linalg.norm(delta, axis=1), [0.3, 0.7])
    assert np.isfinite(moved).all()


def test_rank_set_nll_masks_recruited_contacts_and_averages_ties():
    logits = torch.tensor([[9.0, 2.0, 2.0, 0.0]])
    target = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
    available = torch.tensor([[False, True, True, True]])
    observed = rank_set_nll(logits, target, available)
    logp = torch.log_softmax(torch.tensor([[2.0, 2.0, 0.0]]), dim=-1)
    expected = -(logp[0, 0] + logp[0, 1]) / 2
    torch.testing.assert_close(observed[0], expected)


def test_auc_and_holm_are_deterministic():
    assert np.isclose(dose_auc([0.25, 0.5, 1.0], [1.0, 2.0, 4.0]), 2.0)
    np.testing.assert_allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_cumulative_subspace_erasure_removes_first_two_coordinates():
    basis = orthonormal_row_basis(np.asarray([[2.0, 0.0, 0.0], [1.0, 3.0, 0.0]]))
    np.testing.assert_allclose(basis @ basis.T, np.eye(2), atol=1e-12)
    hidden = np.asarray([[2.0, -4.0, 5.0]])
    moved, delta = subspace_projection_erasure(hidden, np.zeros_like(hidden), basis, 1.0)
    np.testing.assert_allclose(moved, [[0.0, 0.0, 5.0]], atol=1e-12)
    np.testing.assert_allclose(hidden + delta, moved)
