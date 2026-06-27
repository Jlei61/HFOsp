"""A1b state-topography knobs on the Stage-3 two-focus core (user spec 2026-06-25).

A1b asks: with the local lesion E/I loop strength and the global feedback-inhibition (restraint)
held fixed, WHERE does the local:global ratio land — protective/silent, interictal-like axial
self-limited propagation, seizure-like large synchronized recruitment, or runaway? It does NOT ask
about z dynamics (that is A2). Both knobs are static weight modifications fed to
build_connectivity_rot (local_scale_EI + w_EE_gain_core) — NO engine change.

  local_loop_strength (paired knob): core_ei_scale (local I->E onto core E) + core_ee_gain (core
    recurrent E->E). Weaker local inhibition + stronger local recurrent E = local E/I loop more
    ignitable.
  global_restraint: global_ei_scale scales the GABA input to EVERY E target uniformly; the core E
    targets get the ADDITIONAL local factor:
        E-target GABA scale      = global_ei_scale
        core E-target GABA scale = global_ei_scale * core_ei_scale
  This v1 is a STATIC restraint, not a real dynamic global feedback (A1c: I_global(t) =
    feedback_gain * filtered_global_E_rate(t), an engine hook, later).
"""
from __future__ import annotations

import numpy as np


def a1b_weight_lesion(NE, NI, core_mask_E, core_ei_scale, core_ee_gain, global_ei_scale):
    """Return (local_scale_EI[N], w_EE_gain_core) for build_connectivity_rot.

    local_scale_EI[i] multiplies E-target i's GABA input: surround E -> global_ei_scale, core E ->
    global_ei_scale * core_ei_scale (the global restraint with the extra local factor in the core).
    w_EE_gain_core scales both-in-core E->E (the local recurrent loop). Defaults (1,1,1) -> all-ones
    + gain 1.0 -> bit-parity with the no-lesion path.
    """
    ls = np.full(NE + NI, float(global_ei_scale))
    core = np.asarray(core_mask_E, bool)
    ls[:NE][core] *= float(core_ei_scale)
    return ls, float(core_ee_gain)


def local_global_ratio(core_ee_gain, core_ei_scale, global_ei_scale):
    """Model-topography coordinate (NOT a physiological quantity):
        local_loop_drive = core_ee_gain / core_ei_scale ;  global_restraint = global_ei_scale .
    """
    return (core_ee_gain / core_ei_scale) / global_ei_scale
