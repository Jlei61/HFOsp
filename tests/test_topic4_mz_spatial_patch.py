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
    patch_rhs_fast,
    patch_rhs_fast_and_moments,
    patch_rhs_and_moments,
    patch_rhs_to_stage0c,
    patch_to_stage0c_state,
    stage0c_to_patch_state,
    state_size,
    prepare_patch_rhs,
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


def test_kernel_contract_rejects_area_weights_inconsistent_with_operator():
    matrix = np.asarray([[0.8, 0.2], [0.1, 0.9]])
    with pytest.raises(ValueError, match="stationary"):
        PatchKernels(matrix, matrix, np.asarray([0.5, 0.5])).validate()


def test_fast_rhs_matches_validated_oracle_for_off_manifold_batch():
    transfer = DummyTransfer()
    params = PatchParameters(alpha_g=15.0, additive_max_mv=1.6, pool_p=1.0)
    weights = np.asarray([1.0 / 3.0, 2.0 / 3.0])
    kernels = PatchKernels(
        np.asarray([[0.8, 0.2], [0.1, 0.9]]),
        np.asarray([[0.9, 0.1], [0.05, 0.95]]),
        weights,
    ).validate()
    rng = np.random.default_rng(8127)
    states = []
    for _ in range(12):
        local = {
            "rE": rng.uniform(0.001, 0.08, 2),
            "rI": rng.uniform(0.004, 0.18, 2),
            "sEE": rng.uniform(0.001, 0.07, 2),
            "sEI": rng.uniform(0.004, 0.15, 2),
            "sIE": rng.uniform(0.001, 0.07, 2),
            "sII": rng.uniform(0.004, 0.15, 2),
            "rE_fast": rng.uniform(0.001, 0.07, 2),
            "z": rng.uniform(0.82, 1.0, 2),
            "p": rng.uniform(0.0, 1.0, 2),
            "m": rng.uniform(0.0, 0.25, 2),
        }
        states.append(pack_patch_state(local, mu_g=rng.uniform(0.0, 0.7), s_g=rng.uniform(0.0, 0.5)))
    batch = np.asarray(states)
    expected = np.asarray([patch_rhs(state, kernels, params, transfer) for state in batch])
    prepared = prepare_patch_rhs(kernels, params)
    observed = patch_rhs_fast(batch, prepared, transfer)
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=4e-18)
    np.testing.assert_allclose(
        patch_rhs_fast(batch[0], prepared, transfer), expected[0], rtol=0.0, atol=4e-18
    )

    expected_rhs, expected_moments = patch_rhs_and_moments(batch[0], kernels, params, transfer)
    observed_rhs, observed_moments = patch_rhs_fast_and_moments(batch[0], prepared, transfer)
    np.testing.assert_allclose(observed_rhs, expected_rhs, rtol=0.0, atol=4e-18)
    for actual, reference in zip(observed_moments, expected_moments):
        np.testing.assert_array_equal(actual, reference)


def test_pack_rejects_missing_slow_field_instead_of_implicitly_broadcasting():
    local = {name: np.zeros(2) for name in LOCAL_FIELDS if name != "z"}
    with pytest.raises(ValueError, match="exactly"):
        pack_patch_state(local, mu_g=0.0, s_g=0.0)


def test_optional_external_e_drive_is_zero_default_and_patch_local():
    transfer = DummyTransfer()
    params = PatchParameters(additive_max_mv=1.6)
    kernels = PatchKernels.identity(2)
    prepared = prepare_patch_rhs(kernels, params)
    state = uniform_patch_state(
        equilibrium_state((0.02, 0.04)),
        n_patches=2, z=0.9, additive_mv=0.0, parameters=params,
    )
    default = patch_rhs_fast(state, prepared, transfer)
    explicit_zero = patch_rhs_fast(state, prepared, transfer, external_e_mv=[0.0, 0.0])
    np.testing.assert_array_equal(default, explicit_zero)
    driven = patch_rhs_fast(state, prepared, transfer, external_e_mv=[100.0, 0.0])
    assert driven[0] > default[0]
    assert driven[1] == default[1]
