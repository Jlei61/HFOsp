"""This round's recorders, added by subclass so the hash-locked engine file
(src/snn_engine/mz_slow_vars.py, pinned by the frozen Z/M baseline) stays
byte-identical."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "snn_engine"))
sys.path.insert(0, str(ROOT))

from src.snn_engine.mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402
from src.topic4_zm_slow_vars import ZMTracedSlowVars  # noqa: E402


def _module(weights=None, **kwargs):
    core = np.zeros(8, dtype=bool)
    core[:2] = True
    return ZMTracedSlowVars(10, 18.0, MZSlowVarsConfig(**kwargs), NE=8,
                            core_mask_E=core, trace_weights_E=weights)


def test_the_locked_engine_file_is_untouched():
    """The frozen Z/M baseline pins this file by sha256; editing it in place
    would break that lock and silently redefine the mechanism reference."""
    import hashlib, json
    locked = json.loads(
        (ROOT / "config/topic4_data_driven_snn_baseline_zm_v1.json").read_text()
    )["inputs"]["mz_engine"]
    digest = hashlib.sha256((ROOT / locked["path"]).read_bytes()).hexdigest()
    assert digest == locked["sha256"]


def test_subclass_inherits_the_equations_unchanged():
    core = np.zeros(8, dtype=bool); core[:2] = True      # same mask as _module
    base = MZSlowVars(10, 18.0, MZSlowVarsConfig(use_z=True, use_m=True,
                                                 I_th_EI=1.0, eta_m=0.5),
                      NE=8, core_mask_E=core)
    sub = _module(use_z=True, use_m=True, I_th_EI=1.0, eta_m=0.5)
    excitatory = np.arange(10, dtype=float)
    inhibitory = np.arange(10, dtype=float) * 0.3
    assert np.array_equal(base.apply_currents(excitatory, inhibitory),
                          sub.apply_currents(excitatory, inhibitory))
    spikes = np.zeros(10, bool); spikes[:3] = True
    base.step(spikes, None, 0.1); sub.step(spikes, None, 0.1)
    assert np.array_equal(base.z, sub.z) and np.array_equal(base.m, sub.m)
    for key in MZSlowVars.TRACE_NAMES:
        assert np.array_equal(base.trace_arrays()[key], sub.trace_arrays()[key]), key
def test_field_accumulator_is_off_by_default_and_exact_when_on():
    slow = _module(use_z=True, use_m=True, I_th_EI=1.0, eta_m=0.5)
    assert slow.field_accumulator_result() is None

    slow.enable_field_accumulator(3)
    excitatory = np.zeros(10)
    d_expected = np.zeros(8)
    a_expected = np.zeros(8)
    for k in range(3):
        inhibitory = np.full(10, float(k) + 1.0)
        slow.z[:8] = 0.25 * (k + 1)
        slow.m[:8] = float(k)
        d_expected += (1.0 - slow.z[:8]) * inhibitory[:8]
        a_expected += 0.5 * slow.m[:8]
        slow.apply_currents(excitatory, inhibitory)
    out = slow.field_accumulator_result()
    assert out["n_steps"] == 3
    assert np.allclose(out["disinhibition_D"], d_expected / 3.0, rtol=0, atol=1e-12)
    assert np.allclose(out["adaptation_A"], a_expected / 3.0, rtol=0, atol=1e-12)
    assert np.allclose(out["net_slow_current"],
                       out["disinhibition_D"] - out["adaptation_A"])


def test_field_accumulator_stops_after_n_steps():
    slow = _module(use_z=True, use_m=True, I_th_EI=1.0, eta_m=0.5)
    slow.enable_field_accumulator(2)
    for _ in range(5):
        slow.apply_currents(np.zeros(10), np.ones(10))
    assert slow.field_accumulator_result()["n_steps"] == 2


def test_field_accumulator_does_not_change_returned_current():
    a = _module(use_z=True, use_m=True, I_th_EI=1.0, eta_m=0.5)
    b = _module(use_z=True, use_m=True, I_th_EI=1.0, eta_m=0.5)
    b.enable_field_accumulator(4)
    excitatory, inhibitory = np.arange(10, dtype=float), np.arange(10, dtype=float) * 0.3
    assert np.array_equal(a.apply_currents(excitatory, inhibitory),
                          b.apply_currents(excitatory, inhibitory))


def test_field_accumulator_is_a_product_average_not_a_product_of_averages():
    """The whole reason this lives inside apply_currents: mean_t[(1-z)*I_I] and
    (1-mean_t[z])*mean_t[I_I] differ whenever z and I_I co-vary, and they do."""
    slow = _module(use_z=True, use_m=True, I_th_EI=1.0, eta_m=0.0)
    slow.enable_field_accumulator(2)
    for z_val, i_val in ((0.0, 10.0), (1.0, 0.0)):
        slow.z[:8] = z_val
        slow.apply_currents(np.zeros(10), np.full(10, i_val))
    out = slow.field_accumulator_result()
    assert np.allclose(out["disinhibition_D"], 5.0)      # mean of (1*10, 0*0)
    product_of_averages = (1.0 - 0.5) * 5.0             # would be 2.5
    assert not np.allclose(out["disinhibition_D"], product_of_averages)


def _weighted(weights, **kwargs):
    return _module(weights=weights, **kwargs)


def test_weighted_trace_is_absent_by_default_and_leaves_the_old_trace_intact():
    plain = _module(use_z=True, use_m=True, I_th_EI=1.0, eta_m=0.5,
                    trace_stride_steps=1)
    assert plain.weighted_trace_arrays() is None
    weighted = _weighted(np.ones(8), use_z=True, use_m=True, I_th_EI=1.0,
                         eta_m=0.5, trace_stride_steps=1)
    for module in (plain, weighted):
        module.z[:8] = 0.4
        module.m[:8] = 2.0
        module.apply_currents(np.zeros(10), np.full(10, 3.0))
        module.step(np.zeros(10, bool), None, 0.1)
    a, b = plain.trace_arrays(), weighted.trace_arrays()
    for key in MZSlowVars.TRACE_NAMES:
        assert np.array_equal(a[key], b[key]), key


def test_weighted_trace_concentrates_on_the_field():
    """Uniform weights reproduce the population mean; a field-shaped weight
    reports the field, which is the whole point -- the core is 3.5 % of E."""
    weights = np.zeros(8); weights[:2] = 1.0          # all mass on 2 of 8 cells
    module = _weighted(weights, use_z=True, use_m=True, I_th_EI=1.0, eta_m=0.5,
                       trace_stride_steps=1)
    module.z[:8] = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    module.m[:8] = np.array([4.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    module.apply_currents(np.zeros(10), np.full(10, 2.0))
    module.step(np.zeros(10, bool), None, 0.1)
    out = module.weighted_trace_arrays()
    # values are recorded POST-update, matching what the next step consumes:
    # m has already decayed by dt/tau_adp = 0.1/2000.
    m_post = 4.0 * (1.0 - 0.1 / 2000.0)
    assert np.isclose(out["z_weighted_mean"][0], 0.0)           # not the 0.75 popn mean
    assert np.isclose(out["m_weighted_mean"][0], m_post)        # not the 1.0 popn mean
    assert np.isclose(out["disinhibition_weighted_mean"][0], 2.0)
    assert np.isclose(out["adaptation_weighted_mean"][0], 0.5 * m_post)
    assert np.isclose(out["net_slow_current_weighted_mean"][0], 2.0 - 0.5 * m_post)


def test_weighted_trace_rejects_degenerate_weights():
    for bad in (np.zeros(8), np.full(8, np.nan), np.ones(7)):
        with pytest.raises(ValueError):
            _weighted(bad, use_z=True, use_m=True, I_th_EI=1.0, eta_m=0.5)
