import numpy as np

from scripts.prepare_topic4_zm_fig5_state_contrast import (
    _apply_workpoint, _continuation_slice, _select_dose,
)


def test_continuation_slice_includes_requested_checkpoint_step():
    block = _continuation_slice(4699.6, 620.1, 0.1)
    assert block.start == 46996
    assert block.stop == 53197


def test_workpoint_parameters_override_round_defaults():
    config = {"zm": {"I_th_EI": 95.0, "tau_z": 1, "tau_adp": 2, "eta_m": 3}}
    workpoint = {"I_th_EI": 76.0, "tau_z": 5000, "tau_adp": 500, "eta_m": 0.007}
    updated = _apply_workpoint(config, workpoint)
    assert updated["zm"] == workpoint
    assert config["zm"]["I_th_EI"] == 95.0


def test_selects_largest_baseline_subevent_dose():
    rows = [
        {"dose_cells": 8, "e1_evaluable": True, "excess_spikes_early": -13},
        {"dose_cells": 16, "e1_evaluable": True, "excess_spikes_early": -19},
        {"dose_cells": 32, "e1_evaluable": True, "excess_spikes_early": 17712},
        {"dose_cells": 64, "e1_evaluable": False, "excess_spikes_early": 10},
    ]
    assert _select_dose(rows) == 16


def test_refuses_scan_without_baseline_subevent_regime():
    with np.testing.assert_raises_regex(RuntimeError, "no near-zero baseline subevent"):
        _select_dose([{"dose_cells": 8, "e1_evaluable": False,
                       "excess_spikes_early": 0}])
