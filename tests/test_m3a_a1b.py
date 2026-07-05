"""TDD for the A1b local-loop + global-restraint weight lesion on the Stage-3 two-focus core.

Spec (user 2026-06-25): global_restraint scales EVERY E-target's GABA input by global_ei_scale;
the core E targets get an ADDITIONAL local factor core_ei_scale (E-target GABA scale =
global_ei_scale; core E-target GABA scale = global_ei_scale * core_ei_scale). The local recurrent
E loop is core_ee_gain (both-in-core E->E). local_global_ratio = (core_ee_gain/core_ei_scale)/global_ei_scale
is a model-topography coordinate, NOT a physiological quantity.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.sef_hfo_a1b import a1b_weight_lesion, local_global_ratio


def test_global_restraint_scales_all_E_targets_core_extra():
    NE, NI = 10, 4
    core = np.zeros(NE, bool); core[:3] = True
    ls, gain = a1b_weight_lesion(NE, NI, core, core_ei_scale=0.7, core_ee_gain=1.3, global_ei_scale=1.6)
    assert np.allclose(ls[3:NE], 1.6)            # surround E targets: global only
    assert np.allclose(ls[:3], 1.6 * 0.7)        # core E targets: global * core
    assert gain == 1.3                            # core recurrent E loop gain
    assert ls.shape == (NE + NI,)


def test_defaults_are_identity_bit_parity():
    NE, NI = 10, 4
    core = np.zeros(NE, bool); core[:3] = True
    ls, gain = a1b_weight_lesion(NE, NI, core, 1.0, 1.0, 1.0)
    assert np.all(ls == 1.0) and gain == 1.0     # all-ones + gain 1 => no weight change (bit-parity)


def test_local_global_ratio_is_topography_coordinate():
    # (core_ee_gain / core_ei_scale) / global_ei_scale
    assert local_global_ratio(1.3, 0.7, 1.6) == (1.3 / 0.7) / 1.6
    assert local_global_ratio(1.0, 1.0, 1.0) == 1.0     # baseline corner
