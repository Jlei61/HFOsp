# tests/test_m43a_p0_calibration.py
import numpy as np
import pytest
from src.sef_hfo_m4_load_shunt import LoadShuntParams
from scripts.run_m43a_p0_calibration import run_a_trace, calibrate_regimes


def _p():
    # rho_n=0 so sustained ictal keeps accumulating (R_A>1); k_n small so a single IED
    # bumps a measurably WITHOUT saturating (keeps the gate booleans meaningful).
    return LoadShuntParams(tau_n=20000.0, k_n=0.0008, rho_n=0.0, n_base=0.0, n50=0.4,
                           hill_h=2.0, a_max=1.0, u_n0=0.05, n_min=0.0, n_max=10.0)


def test_run_a_trace_shapes_and_quiet_stays_low():
    dt = 1.0
    u_quiet = np.full(3000, 0.05)                   # == u_n0 -> rectified to ~0 drive
    a = run_a_trace(u_quiet, dt, _p())
    assert a.shape == u_quiet.shape
    assert a.max() < 1e-2                            # quiet baseline: a low & stable


def test_calibrate_regimes_table_directions():
    dt = 1.0
    quiet = np.full(6000, 0.05)
    ied = quiet.copy(); ied[3000:3050] = 1.5        # one brief high-duty spike
    ictal = np.concatenate([np.full(1000, 0.05), np.full(5000, 1.2)])  # sustained
    post = np.concatenate([np.full(3000, 1.2), np.full(3000, 0.05)])   # drops to quiet
    out = calibrate_regimes(
        {"quiet": quiet, "isolated_ied": ied, "bounded_ictal": ictal, "post_offset": post},
        dt, _p(), event_idx={"isolated_ied": [3000]})
    assert out["table"]["quiet"]["a_max"] < out["table"]["bounded_ictal"]["a_max"]
    assert out["table"]["isolated_ied"]["delta_a_ied"] > 0        # IED nudges a
    assert out["metrics"]["R_A"] > 1.0                            # sustained >> single IED
    assert out["table"]["post_offset"]["a_end"] <= out["table"]["post_offset"]["a_mid"]  # decays (never rises)
    assert out["gate"]["soft_gate_fail"] is False                # IED did move a


def test_soft_ictal_gate_hard_fails_even_if_R_A_inf():
    # P1-3: an IED that does NOT move a (delta<=0) must HARD-fail, not pass via R_A==inf.
    dt = 1.0
    quiet = np.full(6000, 0.05)
    ied = quiet.copy()                                            # identical to quiet -> no a bump at all
    ictal = np.concatenate([np.full(1000, 0.05), np.full(5000, 1.2)])
    out = calibrate_regimes({"quiet": quiet, "isolated_ied": ied, "bounded_ictal": ictal},
                            dt, _p(), event_idx={"isolated_ied": [3000]})
    assert out["gate"]["delta_a_ied"] <= 0.0
    assert out["gate"]["soft_gate_fail"] is True
    assert out["gate"]["R_A_pass"] is False                       # inf must NOT sneak through
    assert out["gate"]["sensor_free_pass"] is False


def test_interictal_block_pass_requires_a_block():
    dt = 1.0
    quiet = np.full(6000, 0.05); ied = quiet.copy(); ied[3000:3050] = 1.5
    ictal = np.concatenate([np.full(1000, 0.05), np.full(5000, 1.2)])
    out = calibrate_regimes({"quiet": quiet, "isolated_ied": ied, "bounded_ictal": ictal},
                            dt, _p(), event_idx={"isolated_ied": [3000]}, a_block=None)
    assert out["gate"]["interictal_block_pass"] is None           # can't certify without network a_block
    assert out["gate"]["sensor_free_pass"] is False               # P0a proxy cannot fully pass
