import numpy as np
import pytest
import torch

from src.topic5_symmetric_axis_propagation_state_v2_2 import (
    SymmetricAxisPropagationStateRNN,
    fixed_local_scale,
    symmetric_axis_operator,
    validate_normalization_contract,
)


def _coords() -> torch.Tensor:
    return torch.tensor(
        [[-2.0, 0.0, 0.0], [-1.0, 0.1, 0.0], [0.0, 0.0, 0.0],
         [1.0, -0.1, 0.0], [2.0, 0.0, 0.0]],
        dtype=torch.float64,
    )


def test_operator_is_symmetric_and_axis_sign_invariant():
    coords = _coords()
    kwargs = dict(anisotropy_ratio=3.0, gamma=0.8, gain=1.2)
    positive = symmetric_axis_operator(
        coords, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64), **kwargs
    )
    negative = symmetric_axis_operator(
        coords, torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float64), **kwargs
    )
    assert torch.allclose(positive["W"], positive["W"].T, atol=1e-12)
    assert torch.allclose(positive["W"], negative["W"], atol=1e-12)
    assert torch.all(torch.diag(positive["W"]) == 0)


def test_fixed_local_scale_is_median_nearest_neighbour():
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    assert np.isclose(fixed_local_scale(coords), 1.0)


def test_model_has_only_allowed_patient_specific_parameters_and_no_dense_bypass():
    model = SymmetricAxisPropagationStateRNN(
        coords=_coords().numpy(), node_bias=np.zeros(5)
    )
    names = {name for name, _ in model.named_parameters()}
    assert names == {
        "axis_raw", "gamma_raw", "gain_raw", "raw_anisotropy",
        "raw_rho", "c0", "raw_c_p", "raw_c_n",
    }
    for _, parameter in model.named_parameters():
        assert parameter.ndim <= 1


def test_row_normalization_is_rejected():
    validate_normalization_contract("symmetric_degree")
    try:
        validate_normalization_contract("row")
    except ValueError:
        pass
    else:
        raise AssertionError("row normalization must be rejected")


def test_same_symmetric_graph_from_opposite_ends_has_opposite_displacement():
    coords = _coords()
    operator = symmetric_axis_operator(
        coords,
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        anisotropy_ratio=3.0,
        gamma=1.0,
        gain=1.0,
    )["W"]
    position = coords[:, 0]
    left = operator @ torch.tensor([1, 0, 0, 0, 0], dtype=torch.float64)
    right = operator @ torch.tensor([0, 0, 0, 0, 1], dtype=torch.float64)
    left_displacement = torch.sum(left * (position - position[0])) / left.sum()
    right_displacement = torch.sum(right * (position - position[-1])) / right.sum()
    assert left_displacement > 0
    assert right_displacement < 0


def test_geometry_incomplete_cannot_build_axis_model_and_isotropic_is_exact():
    coords = _coords().numpy()
    incomplete = coords.copy()
    incomplete[0] = np.nan
    try:
        SymmetricAxisPropagationStateRNN(
            coords=incomplete, node_bias=np.zeros(len(incomplete))
        )
    except ValueError:
        pass
    else:
        raise AssertionError("incomplete geometry entered the physical-axis model")
    isotropic = SymmetricAxisPropagationStateRNN(
        coords=coords, node_bias=np.zeros(len(coords)), isotropic=True
    )
    assert float(isotropic.gamma) == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_gpu_operator_agree_within_float_tolerance():
    coords = _coords().float()
    axis = torch.tensor([0.7, 0.2, -0.4], dtype=torch.float32)
    kwargs = dict(anisotropy_ratio=2.5, gamma=0.6, gain=0.8)
    cpu = symmetric_axis_operator(coords, axis, **kwargs)["W"]
    gpu = symmetric_axis_operator(coords.cuda(), axis.cuda(), **kwargs)["W"].cpu()
    assert torch.allclose(cpu, gpu, atol=2e-6, rtol=2e-6)
