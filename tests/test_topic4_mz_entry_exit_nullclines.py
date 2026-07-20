from pathlib import Path

import numpy as np
import pytest

from src.topic4_mz_entry_exit_nullclines import (
    additive_rhs,
    fit_inverse_sqrt_period,
    macro_recovery_flow,
    solve_fold,
)
from src.topic4_spatial_slowfast_stage0c import PoolParameters, equilibrium_state
from src.topic4_spatial_slowfast_stage0c_transfer import (
    ExtendedSiegertTransfer,
    _rhs_and_moments,
    prepare_pool_parameters,
)
from src.topic4_spatial_slowfast_stage0f import SmoothDomain
from src.topic4_spatial_slowfast_stage0f_v1_1 import SmoothSiegertTransferV11


ROOT = Path(__file__).resolve().parents[1]
TRANSFER_PATH = (
    ROOT
    / "results/topic4_sef_hfo/spatial_slowfast_topology/"
    "stage0c_transfer_support_audit_v1_1/extended_transfer_extra_fine.npz"
)


def _transfers():
    with np.load(TRANSFER_PATH, allow_pickle=False) as payload:
        extended = ExtendedSiegertTransfer(
            payload["mu_axis"], payload["sigma_axis"], payload["log_integral_table"],
            name="extra_fine",
        )
    smooth = SmoothSiegertTransferV11.from_extended(
        extended, domain=SmoothDomain(-160.0, 80.0, 3.0, 20.0)
    )
    return extended, smooth


@pytest.mark.skipif(not TRANSFER_PATH.is_file(), reason="requires locked Stage0C transfer artifact")
def test_zero_additive_current_is_exact_stage0c_rhs_parity():
    transfer, _ = _transfers()
    params = PoolParameters(0.85, 15.0, 1.1, 1.0)
    state = equilibrium_state((0.006, 0.012))
    expected, _ = _rhs_and_moments(
        state[None, :], prepare_pool_parameters([params]), transfer,
        mechanism="dynamic", clamp_s=None, subtractive_beta_mv=None,
    )
    observed = additive_rhs(state, params, transfer, 0.0)
    np.testing.assert_array_equal(observed, expected[0])


@pytest.mark.skipif(not TRANSFER_PATH.is_file(), reason="requires locked Stage0C transfer artifact")
def test_smooth_fold_reproduces_locked_entry_boundary():
    _, transfer = _transfers()
    fold = solve_fold(0.0, transfer)
    assert abs(fold.z - 0.87447467) < 2e-6
    assert abs(1000.0 * fold.r_e_khz - 2.0264) < 2e-3
    assert abs(fold.leading_real_per_ms) < 1e-7


def test_inverse_sqrt_period_fit_recovers_synthetic_scaling():
    z_fold = 0.875
    z = np.asarray([0.84, 0.85, 0.86, 0.87])
    period = 500.0 + 25.0 / np.sqrt(z_fold - z)
    fit = fit_inverse_sqrt_period(z, period, z_fold)
    assert abs(fit["intercept_ms"] - 500.0) < 1e-8
    assert abs(fit["coefficient_ms_sqrt_z"] - 25.0) < 1e-8
    assert fit["r_squared"] > 0.999999999


def test_macro_recovery_flow_builds_only_with_drive_and_decays_without_it():
    assert macro_recovery_flow(0.2, 1.0, k_up_per_s=1.0, k_down_per_s=0.5) > 0.0
    assert macro_recovery_flow(0.2, 0.0, k_up_per_s=1.0, k_down_per_s=0.5) < 0.0
    assert macro_recovery_flow(
        0.2, 0.0, k_up_per_s=1.0, k_down_per_s=0.5, decay_guard=0.0
    ) == 0.0
