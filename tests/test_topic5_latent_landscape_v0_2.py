from pathlib import Path

import numpy as np
import torch

from src.topic5_latent_pass1_v0_2 import (
    event_first_phase_balanced_weights,
    leaky_rnn_jvp,
    spline_basis,
    spline_derivative,
    weighted_r2,
    weighted_ridge,
)

from src.topic5_latent_landscape_v0_2 import (
    DecoderState,
    PUBLIC_ARMS,
    arrays_sha256,
    canonical_json_sha256,
    classify_future_field_axis,
    enumerate_cell_keys,
    parse_bool,
    parameter_state_sha256,
    rank_matrix_to_event_fields,
    response_blind_event_sample,
    resolve_unit_dir,
    stable_event_hash,
)


def test_enumerate_complete_checkpoint_matrix() -> None:
    keys = enumerate_cell_keys([f"fit_{index}" for index in range(42)])
    assert len(keys) == 630
    assert len(set(keys)) == 630
    assert {arm for _, arm, _ in keys} == set(PUBLIC_ARMS)


def test_exact_reuse_applies_only_to_l0_l1_l3() -> None:
    parent = Path("/parent")
    old = Path("/old")
    reused = {"fit_a"}
    for arm in ("L0", "L1", "L3"):
        path, source = resolve_unit_dir(parent, old, "fit_a", arm, 2, reused)
        assert source == "V0_3_EXACT_REUSE"
        assert str(path).startswith("/old/per_fit/")
    for arm in ("L2m", "C-suffix"):
        path, source = resolve_unit_dir(parent, old, "fit_a", arm, 2, reused)
        assert source == "V0_5_FORMAL_UNIT"
        assert str(path).startswith("/parent/formal_units/")


def test_array_hash_includes_name_shape_and_dtype() -> None:
    value = np.arange(6, dtype=np.int16).reshape(2, 3)
    base = arrays_sha256({"x": value})
    assert base == arrays_sha256({"x": value.copy()})
    assert base != arrays_sha256({"y": value})
    assert base != arrays_sha256({"x": value.astype(np.int32)})
    assert base != arrays_sha256({"x": value.reshape(3, 2)})


def test_canonical_json_hash_is_order_invariant() -> None:
    assert canonical_json_sha256({"a": 1, "b": [2, 3]}) == canonical_json_sha256(
        {"b": [2, 3], "a": 1}
    )


def test_parse_bool_is_not_python_string_truthiness() -> None:
    assert parse_bool("True") is True
    assert parse_bool("False") is False
    assert parse_bool(0) is False
    assert parse_bool(1) is True


def test_complete_decoder_state_clone_has_no_tensor_alias() -> None:
    state = DecoderState(
        h=torch.arange(4, dtype=torch.float32).reshape(1, 4),
        recruited=torch.tensor([[1.0, 0.0, 0.0]]),
        rank_index=2,
    )
    cloned = state.clone()
    cloned.h[0, 0] = -1
    cloned.recruited[0, 1] = 1
    assert state.h[0, 0].item() == 0
    assert state.recruited[0, 1].item() == 0
    assert cloned.rank_index == state.rank_index


def test_parameter_hash_detects_state_mutation() -> None:
    module = torch.nn.Linear(3, 2)
    before = parameter_state_sha256(module)
    with torch.no_grad():
        module.weight[0, 0] += 1
    assert parameter_state_sha256(module) != before


def test_future_field_axis_tiers_do_not_relabel_own_fits_as_ab() -> None:
    shared = classify_future_field_axis("shared", "B", "A")
    assert shared["tier"] == "CANONICAL_AB_SHARED"
    assert shared["positive_mode"] == 1
    assert shared["negative_mode"] == 0
    own = classify_future_field_axis("own_a", "A", "A")
    assert own["tier"] == "WITHIN_FIT_MODE_ONLY"
    assert own["canonical_ab"] is False
    assert own["positive_label"] == "mode1"
    invalid = classify_future_field_axis("shared", "A", "A")
    assert invalid["tier"] == "FIELD_AXIS_NOT_IDENTIFIABLE"


def test_rank_fields_match_parent_start_removed_contract() -> None:
    ranks = np.asarray([[0, 1, 2, -1], [0, 0, 1, 2]], dtype=np.int16)
    full, recurrence = rank_matrix_to_event_fields(ranks)
    np.testing.assert_allclose(full[0], [1.0, 0.5, 0.0, 0.0])
    assert np.isnan(recurrence[0, 0])
    np.testing.assert_allclose(recurrence[0, 1:], [1.0, 0.0, 0.0])
    assert np.isnan(recurrence[1, 0]) and np.isnan(recurrence[1, 1])
    np.testing.assert_allclose(recurrence[1, 2:], [1.0, 0.0])


def test_response_blind_sample_is_deterministic_and_phase_filtered() -> None:
    split = np.asarray([0, 0, 0, 1, 1, 2], dtype=np.int8)
    source = np.arange(6)
    dataset = np.arange(10, 16)
    phase_defined = np.asarray([True, False, True, True, True, True])
    kwargs = dict(
        patient="p",
        split=split,
        event_source_index=source,
        event_dataset_index=dataset,
        phase_defined=phase_defined,
        caps={0: 1, 1: 2, 2: 1},
    )
    first = response_blind_event_sample(**kwargs)
    second = response_blind_event_sample(**kwargs)
    assert first.equals(second)
    assert 1 not in set(first["event_array_index"])
    assert len(first) == 4
    assert first["identity_sha256"].str.len().eq(64).all()
    assert stable_event_hash("p", 0, 0, 10) == stable_event_hash("p", 0, 0, 10)


def test_phase_weights_balance_bins_and_events() -> None:
    event = np.asarray([0, 0, 1, 1, 1, 2, 2])
    split = np.asarray([0, 0, 0, 0, 0, 1, 1])
    bins = np.asarray([0, 0, 0, 1, 1, 0, 1])
    weights = event_first_phase_balanced_weights(event, split, bins, n_bins=2)
    assert np.isclose(weights[split == 0].sum(), 1.0)
    assert np.isclose(weights[split == 1].sum(), 1.0)
    assert np.isclose(weights[(split == 0) & (bins == 0)].sum(), 0.5)
    assert np.isclose(weights[(split == 0) & (bins == 1)].sum(), 0.5)


def test_spline_derivative_and_weighted_ridge_recover_linear_signal() -> None:
    s = np.linspace(0.0, 1.0, 21)
    basis = spline_basis(s, ())
    derivative = spline_derivative(s, ())
    np.testing.assert_allclose(derivative[:, 1], 1.0)
    y = (2.0 + 3.0 * s)[:, None]
    weights = np.full(len(s), 1.0 / len(s))
    coefficient = weighted_ridge(basis[:, :2], y, weights, 1e-10)
    prediction = basis[:, :2] @ coefficient
    assert weighted_r2(y, prediction, weights) > 0.999999


def test_analytic_leaky_rnn_jvp_matches_autograd() -> None:
    class Tiny(torch.nn.Module):
        cell = "rnn"
        state_dim = 1

        def __init__(self) -> None:
            super().__init__()
            self.n_nodes = 3
            self.n_contacts = 2
            self.recurrent = torch.nn.Parameter(torch.tensor([[
                [0.2, -0.1, 0.0], [0.3, 0.1, -0.2], [0.0, 0.4, 0.1]
            ]]))
            self.bias = torch.nn.Parameter(torch.tensor([[0.1, -0.2, 0.05]]))
            self.kappa_logit = torch.nn.Parameter(torch.tensor(0.3))
            self.input_weight = torch.nn.Parameter(torch.tensor([
                [0.2, 0.0], [0.1, -0.3], [0.0, 0.4]
            ]))

        def masked_recurrent(self):
            return self.recurrent

        def _inject(self, x):
            return (x @ self.input_weight.T)[:, None, None, :]

        def _step(self, h, x):
            u = self._inject(x).reshape(h.shape[0], -1, 3)[:, 0]
            pre = u + h @ self.recurrent[0].T + self.bias[0]
            k = torch.sigmoid(self.kappa_logit)
            return (1-k)*h + k*torch.tanh(pre)

    model = Tiny()
    h = torch.tensor([[0.2, -0.1, 0.4]], requires_grad=True)
    x = torch.tensor([[1.0, 0.0]])
    v = torch.tensor([[0.3, -0.2, 0.5]])
    _, autodiff = torch.autograd.functional.jvp(lambda value: model._step(value, x), h, v)
    analytic = leaky_rnn_jvp(model, h.detach(), x, v)
    torch.testing.assert_close(analytic, autodiff, rtol=1e-6, atol=1e-7)
