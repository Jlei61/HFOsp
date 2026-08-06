from pathlib import Path

import numpy as np
import pytest

from src.topic4_mz_spatial_entry_exit import (
    regional_equilibrium_residual,
    regional_equilibrium_state,
    regional_fast_jacobian,
    solve_regional_additive_fold,
    solve_regional_fold,
)
from src.topic4_mz_spatial_patch import PatchParameters, prepare_patch_rhs
from src.topic4_mz_spatial_reduction import canonical_m3b_core_annulus_bath
from src.topic4_spatial_slowfast_stage0c_transfer import ExtendedSiegertTransfer
from src.topic4_spatial_slowfast_stage0f import SmoothDomain
from src.topic4_spatial_slowfast_stage0f_v1_1 import SmoothSiegertTransferV11


ROOT = Path(__file__).resolve().parents[1]
TRANSFER_PATH = (
    ROOT / "results/topic4_sef_hfo/spatial_slowfast_topology/"
    "stage0c_transfer_support_audit_v1_1/extended_transfer_extra_fine.npz"
)


def _objects():
    with np.load(TRANSFER_PATH, allow_pickle=False) as payload:
        exact = ExtendedSiegertTransfer(
            payload["mu_axis"], payload["sigma_axis"], payload["log_integral_table"],
            name="extra_fine",
        )
    transfer = SmoothSiegertTransferV11.from_extended(
        exact, domain=SmoothDomain(-160.0, 80.0, 3.0, 20.0)
    )
    reduction = canonical_m3b_core_annulus_bath()
    parameters = PatchParameters(
        alpha_g=15.0, w_ee_mult=1.1, ratio=1.0,
        additive_max_mv=1.6, pool_p=1.0,
    )
    return transfer, parameters, prepare_patch_rhs(reduction.kernels, parameters)


@pytest.mark.skipif(not TRANSFER_PATH.is_file(), reason="requires locked Stage0C transfer")
def test_regional_fold_reproduces_localized_real_entry_boundary():
    transfer, parameters, prepared = _objects()
    fold = solve_regional_fold(prepared, parameters, transfer)
    assert abs(fold.z_regional - 0.8558315843) < 2.0e-7
    np.testing.assert_allclose(
        1000.0 * fold.rates_khz[:3], [3.265159, 2.229004, 0.867350], atol=2.0e-3
    )
    assert fold.residual_inf < 2.0e-8
    assert fold.rate_sigma_min < 1.0e-8
    assert abs(fold.fast_leading_real_per_ms) < 1.0e-6
    assert fold.fast_leading_imag_per_ms < 1.0e-8
    assert fold.support_all
    assert abs(fold.left_fz) > 1.0e-3
    assert abs(fold.left_d2f_vv) > 1.0


@pytest.mark.skipif(not TRANSFER_PATH.is_file(), reason="requires locked Stage0C transfer")
def test_regional_equilibrium_lift_and_fast_index_contract():
    transfer, parameters, prepared = _objects()
    fold = solve_regional_fold(prepared, parameters, transfer)
    state = regional_equilibrium_state(
        fold.rates_khz, prepared, parameters, z_regional=fold.z_regional
    )
    residual = regional_equilibrium_residual(
        fold.rates_khz, prepared, parameters, transfer, z_regional=fold.z_regional
    )
    jacobian, indices = regional_fast_jacobian(state, prepared, transfer)
    assert np.max(np.abs(residual)) < 1.0e-8
    assert jacobian.shape == (23, 23)
    np.testing.assert_array_equal(indices, np.r_[np.arange(21), [30, 31]])


@pytest.mark.skipif(not TRANSFER_PATH.is_file(), reason="requires locked Stage0C transfer")
def test_regional_additive_fold_moves_with_depletion_depth():
    transfer, parameters, prepared = _objects()
    shallow = solve_regional_additive_fold(
        0.855, prepared, parameters, transfer, initial_additive_mv=0.015
    )
    deep = solve_regional_additive_fold(
        0.850, prepared, parameters, transfer, initial_additive_mv=0.09
    )
    assert abs(shallow.additive_mv - 0.01277665) < 2.0e-6
    assert abs(deep.additive_mv - 0.08864422) < 2.0e-6
    assert deep.additive_mv > shallow.additive_mv
    assert shallow.rate_sigma_min < 1.0e-8
    assert deep.rate_sigma_min < 1.0e-8
