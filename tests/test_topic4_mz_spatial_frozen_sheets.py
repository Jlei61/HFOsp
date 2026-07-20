from __future__ import annotations

import numpy as np

from src.topic4_mz_spatial_frozen_sheets import (
    lift_product_history,
    sheet_label,
    summarize_local_state,
)
from src.topic4_mz_spatial_patch import PatchKernels, PatchParameters, unpack_patch_state


def _template(r_e: float, r_i: float, mu_g: float, s_g: float) -> np.ndarray:
    return np.asarray([r_e, r_i, 0.8 * r_e, 0.8 * r_i, 0.7 * r_e, 0.7 * r_i,
                       0.9 * r_e, mu_g, s_g])


def test_product_history_lift_mixes_synapses_and_has_one_area_weighted_pool():
    weights = np.asarray([1.0 / 3.0, 2.0 / 3.0])
    k_ee = np.asarray([[0.8, 0.2], [0.1, 0.9]])
    k_i = np.asarray([[0.9, 0.1], [0.05, 0.95]])
    kernels = PatchKernels(k_ee, k_i, weights).validate()
    templates = np.vstack([_template(0.08, 0.16, 0.5, 0.3), _template(0.002, 0.007, 0.01, 0.02)])
    state = lift_product_history(
        templates,
        kernels,
        z=[0.85, 0.9],
        parameters=PatchParameters(pool_p=1.0),
    )
    local, mu_g, s_g = unpack_patch_state(state, 2)
    np.testing.assert_allclose(local["sEE"], k_ee @ templates[:, 2])
    np.testing.assert_allclose(local["sEI"], k_i @ templates[:, 3])
    np.testing.assert_allclose(local["sIE"], k_i @ templates[:, 4])
    np.testing.assert_allclose(local["sII"], k_i @ templates[:, 5])
    assert mu_g == weights @ templates[:, 7]
    assert s_g == weights @ templates[:, 8]


def test_local_summary_resolves_clean_cycle_and_low_without_forcing_other():
    time = np.arange(0.0, 6000.0 + 2.0, 2.0)
    rate = 0.030 + 0.020 * np.sin(2.0 * np.pi * time / 600.0)
    fast = 0.025 + 0.012 * np.sin(2.0 * np.pi * (time - 30.0) / 600.0)
    returns = np.arange(120.0, 6000.0, 600.0)
    cycle = summarize_local_state(
        time, rate, fast, returns,
        support_violation_count=0, state_bound_violation_count=0, finite=True,
    )
    assert cycle["status"] == "C"
    assert cycle["recent_period_ms"] == 600.0

    low = summarize_local_state(
        time, np.full(time.size, 0.002), np.full(time.size, 0.002), [],
        support_violation_count=0, state_bound_violation_count=0, finite=True,
    )
    assert low["status"] == "L"
    assert sheet_label(cycle["status"], low["status"]) == "CL"
    assert sheet_label("tonic_plateau", "L") == "O_unresolved"
