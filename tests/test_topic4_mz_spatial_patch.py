from __future__ import annotations

import numpy as np
import pytest

from src.topic4_mz_entry_exit_nullclines import additive_rhs
from src.topic4_mz_spatial_patch import (
    LOCAL_FIELDS,
    PatchKernels,
    PatchParameters,
    pack_patch_state,
    patch_rhs,
    patch_rhs_to_stage0c,
    patch_to_stage0c_state,
    stage0c_to_patch_state,
    state_size,
    uniform_patch_state,
    unpack_patch_state,
)
from src.topic4_spatial_slowfast_stage0c import PoolParameters, equilibrium_state, recruitment_sensor


class DummyTransfer:
    """Smooth deterministic transfer shared by both parity implementations."""

    @staticmethod
    def rate(mu, sigma, pop):
        ceiling = 0.5 if pop == "E" else 1.0
        value = 0.02 + 0.002 * np.asarray(mu) + 0.001 * np.asarray(sigma)
        return np.clip(value, 0.0, ceiling)


def _off_manifold_stage_state() -> np.ndarray:
    return np.asarray([0.080, 0.150, 0.003, 0.008, 0.011, 0.019, 0.071, 0.42, 0.73])


def test_p1_state_roundtrip_preserves_stage0c_and_has_10p_plus_2_shape():
    params = PatchParameters(additive_max_mv=1.6)
    stage = _off_manifold_stage_state()
    patch = stage0c_to_patch_state(stage, z=0.85, additive_mv=0.3, parameters=params)
    assert patch.shape == (state_size(1),) == (12,)
    np.testing.assert_array_equal(patch_to_stage0c_state(patch), stage)
    local, mu_g, s_g = unpack_patch_state(patch, 1)
    assert set(local) == set(LOCAL_FIELDS)
    assert local["m"][0] == pytest.approx(0.3 / 1.6)
    assert local["z"][0] == 0.85
    assert (mu_g, s_g) == (0.42, 0.73)


@pytest.mark.parametrize("additive_mv", [0.0, 0.2, 0.316])
def test_p1_rhs_matches_locked_stage0c_additive_rhs_off_manifold(additive_mv):
    transfer = DummyTransfer()
    params = PatchParameters(alpha_g=15.0, additive_max_mv=1.6)
    stage = _off_manifold_stage_state()
    patch = stage0c_to_patch_state(
        stage, z=0.85, additive_mv=additive_mv, parameters=params
    )
    observed = patch_rhs_to_stage0c(
        patch_rhs(patch, PatchKernels.identity(1), params, transfer)
    )
    expected = additive_rhs(
        stage, PoolParameters(0.85, 15.0, 1.1, 1.0), transfer, additive_mv
    )
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=2e-18)


def test_uniform_patches_reduce_to_same_stage0c_rhs_with_shared_pool():
    transfer = DummyTransfer()
    params = PatchParameters(alpha_g=15.0, additive_max_mv=1.6)
    stage = equilibrium_state((0.02, 0.04))
    state = uniform_patch_state(
        stage, n_patches=4, z=0.85, additive_mv=0.2, parameters=params
    )
    k_ee = np.asarray([
        [0.4, 0.3, 0.0, 0.3],
        [0.3, 0.4, 0.3, 0.0],
        [0.0, 0.3, 0.4, 0.3],
        [0.3, 0.0, 0.3, 0.4],
    ])
    k_i = np.full((4, 4), 0.25)
    rhs = patch_rhs(state, PatchKernels(k_ee, k_i), params, transfer)
    local, dmu, ds = unpack_patch_state(rhs, 4)
    expected = additive_rhs(
        stage, PoolParameters(0.85, 15.0, 1.1, 1.0), transfer, 0.2
    )
    for index, name in enumerate(LOCAL_FIELDS[:7]):
        np.testing.assert_allclose(local[name], expected[index], rtol=0.0, atol=2e-18)
    assert dmu == pytest.approx(expected[7], abs=2e-18)
    assert ds == pytest.approx(expected[8], abs=2e-18)
    np.testing.assert_array_equal(local["z"], 0.0)
    np.testing.assert_array_equal(local["p"], 0.0)
    np.testing.assert_array_equal(local["m"], 0.0)


def test_pool_is_one_shared_scalar_and_averages_patch_recruitment():
    params = PatchParameters(pool_p=1.0)
    local = {name: np.zeros(2) for name in LOCAL_FIELDS}
    local["rE_fast"] = np.asarray([0.005, 0.020])
    local["z"].fill(0.85)
    state = pack_patch_state(local, mu_g=0.0, s_g=0.0)
    rhs = patch_rhs(state, PatchKernels.identity(2), params, DummyTransfer())
    _, dmu, _ = unpack_patch_state(rhs, 2)
    expected_area = np.mean(recruitment_sensor(local["rE_fast"]))
    assert dmu == pytest.approx(expected_area / 30.0)
    assert rhs.shape == (22,)


def test_shared_pool_uses_explicit_patch_area_weights():
    params = PatchParameters(pool_p=1.0)
    local = {name: np.zeros(2) for name in LOCAL_FIELDS}
    local["rE_fast"] = np.asarray([0.005, 0.020])
    local["z"].fill(0.85)
    state = pack_patch_state(local, mu_g=0.0, s_g=0.0)
    kernels = PatchKernels(np.eye(2), np.eye(2), np.asarray([0.75, 0.25]))
    rhs = patch_rhs(state, kernels, params, DummyTransfer())
    _, dmu, _ = unpack_patch_state(rhs, 2)
    expected_area = float(np.sum(kernels.patch_weights * recruitment_sensor(local["rE_fast"])))
    assert dmu == pytest.approx(expected_area / 30.0)


def test_kernel_contract_rejects_batch_as_space_nonconstant_operator():
    with pytest.raises(ValueError, match="preserve a constant"):
        PatchKernels(np.eye(2) * 0.5, np.eye(2)).validate()


def test_pack_rejects_missing_slow_field_instead_of_implicitly_broadcasting():
    local = {name: np.zeros(2) for name in LOCAL_FIELDS if name != "z"}
    with pytest.raises(ValueError, match="exactly"):
        pack_patch_state(local, mu_g=0.0, s_g=0.0)
