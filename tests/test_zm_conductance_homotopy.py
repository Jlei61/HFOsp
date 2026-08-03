from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "src/snn_engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from slow_field import SpatialSlowFieldConfig  # noqa: E402
from zm_conductance import (  # noqa: E402
    ZMConductanceConfig,
    conductance_membrane_step,
    state_dependent_homotopy_step,
)


def _vectors(z_value):
    v = np.array([13.0, 16.0, 14.0])
    i_e = np.array([8.0, 9.0, 3.0])
    i_i = np.array([2.0, 4.0, 1.0])
    z = np.array([z_value, z_value, 1.0])
    m = np.array([10.0, 20.0, 0.0])
    decay = np.array([0.995, 0.995, 0.990])
    is_e = np.array([True, True, False])
    native = i_e - i_i
    native[:2] = i_e[:2] - z[:2] * i_i[:2] - 0.001 * m[:2]
    return v, i_e, i_i, native, z, m, decay, is_e


def test_high_z_is_literal_native_and_low_z_is_literal_conductance():
    cfg = ZMConductanceConfig()
    high = _vectors(0.8)
    got_high = state_dependent_homotopy_step(
        *high, cfg, z_native=0.6, z_conductance=0.4
    )
    native_next = high[3] + (high[0] - high[3]) * high[6]
    np.testing.assert_array_equal(got_high["V_next"], native_next)
    np.testing.assert_array_equal(got_high["lambda"], np.zeros(3))

    low = _vectors(0.2)
    got_low = state_dependent_homotopy_step(
        *low, cfg, z_native=0.6, z_conductance=0.4
    )
    pure = conductance_membrane_step(
        low[0], low[1], low[2], low[4], low[5], low[6], low[7], cfg
    )
    np.testing.assert_array_equal(got_low["V_next"][:2], pure["V_next"][:2])
    np.testing.assert_array_equal(got_low["lambda"], np.array([1.0, 1.0, 0.0]))


def test_mid_z_is_continuous_and_bounded_between_endpoint_updates():
    cfg = ZMConductanceConfig()
    mid = _vectors(0.5)
    got = state_dependent_homotopy_step(
        *mid, cfg, z_native=0.6, z_conductance=0.4
    )
    assert np.allclose(got["lambda"][:2], 0.5)
    high = state_dependent_homotopy_step(
        *_vectors(0.6), cfg, z_native=0.6, z_conductance=0.4
    )["V_next"]
    low = state_dependent_homotopy_step(
        *_vectors(0.4), cfg, z_native=0.6, z_conductance=0.4
    )["V_next"]
    assert np.all(got["V_next"][:2] >= np.minimum(high[:2], low[:2]))
    assert np.all(got["V_next"][:2] <= np.maximum(high[:2], low[:2]))


def test_homotopy_contract_is_off_by_default_and_validated():
    SpatialSlowFieldConfig().validate()
    with pytest.raises(ValueError, match="requires use_z"):
        SpatialSlowFieldConfig(
            use_qI=False,
            use_gK=False,
            use_zm_conductance_homotopy=True,
        ).validate()


def test_homotopy_allows_state_selective_recurrent_H_but_not_global_M_divisor():
    SpatialSlowFieldConfig(
        use_qI=False,
        use_gK=False,
        use_z=True,
        use_m=True,
        use_SG=True,
        alpha_G=16.0,
        use_mode_H=True,
        rho_mode_H=2.0,
        use_zm_conductance_homotopy=True,
    ).validate()
    with pytest.raises(ValueError, match="use_mode_M_divisive"):
        SpatialSlowFieldConfig(
            use_qI=False,
            use_gK=False,
            use_z=True,
            use_m=True,
            use_SG=True,
            alpha_G=16.0,
            use_mode_M_divisive=True,
            kappa_mode_M=2.0,
            use_zm_conductance_homotopy=True,
        ).validate()
    with pytest.raises(ValueError, match="thresholds"):
        SpatialSlowFieldConfig(
            use_qI=False,
            use_gK=False,
            use_z=True,
            use_m=True,
            use_zm_conductance_homotopy=True,
            cond_homotopy_z_native=0.4,
            cond_homotopy_z_conductance=0.6,
        ).validate()
