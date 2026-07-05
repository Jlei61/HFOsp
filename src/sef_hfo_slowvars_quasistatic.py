"""Quasi-static (frozen) slow-variable layer for M3A-A1.

A1 freezes ONE slow variable at a constant value and runs a no-kick spontaneous sim, asking:
if the tissue were ALREADY at this slow state, would the spontaneous-event phenotype change?
This is the quasi-static counterpart to the dynamic ``snn_engine.slow_vars.SlowVars`` — the slow
state does NOT evolve (``step()`` is a no-op; that clamp IS what "quasi-static" means). Letting
the state evolve with activity history is A2 (the dynamic plan), out of A1 scope.

Engine path (verified, src/snn_engine/kick_probe.py::simulate_kick):
  - z / phi-offset / g_K ride the ``slow=`` path: ``simulate_kick(slow=FrozenSlowVars(...))``. That
    path uses ``slow.apply_currents`` / ``slow.threshold(p.V_th)`` and BYPASSES ``membrane_step`` —
    so ``shunt_gaba`` / ``e_gaba`` are inert under slow!=None, and the threshold is the uniform
    ``p.V_th`` (V_th_per_neuron is also bypassed).
  - static e_GABA does NOT live here. It uses the membrane shunt path: ``slow=None,
    shunt_gaba=True, e_gaba=...`` (so the runner must never set slow AND shunt together — that is
    the "do not combine z and e_GABA" trap).

No engine-file edit: ``FrozenSlowVars`` subclasses the git-tracked ``SlowVars`` and only overrides
``step()`` to a no-op, so ``slow=None`` bit-parity is untouched and engine_versions need no re-bless.
"""
from __future__ import annotations

import os
import sys

import numpy as np

# slow_vars lives in the git-tracked engine dir; add it to sys.path the same way
# src/topic4_propagation_operator.py reaches the engine (self-contained import).
_ENG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snn_engine")
if _ENG not in sys.path:
    sys.path.insert(0, _ENG)
from slow_vars import SlowVars, SlowVarsConfig  # noqa: E402

__all__ = ["FrozenSlowVars", "build_frozen_slowvars"]


class FrozenSlowVars(SlowVars):
    """A ``SlowVars`` whose state never evolves: ``step()`` is a no-op (quasi-static clamp).

    ``apply_currents`` and ``threshold`` are inherited unchanged, so the frozen z / phi / g_K
    enter the engine exactly as the dynamic layer would — they simply never update.
    """

    def step(self, spk, labels, dt):  # noqa: D401 — quasi-static: clamped, no dynamics
        return


def build_frozen_slowvars(N, V_th0, *, z=None, phi_offset=None, gK=None, vth_field=None):
    """Build a ``FrozenSlowVars`` with EXACTLY ONE slow variable active (A1 single-variable rule).

    Parameters (pass exactly one; the others stay None / off):
      z          : fixed disinhibition in [0, 1]. ``I_net = I_E - z*I_I``; z=1 is full inhibition
                   (= baseline), z<1 weakens inhibition (more excitable).
      phi_offset : fixed adaptive-threshold offset (mV). Per-neuron threshold = base + phi_offset;
                   positive raises the threshold (more inhibitory).
      gK         : fixed sAHP outward current (mV-equivalent). ``I_net -= gK``; larger suppresses
                   excitability (more inhibitory).
      vth_field  : optional per-neuron base threshold field (length N) for phi — when given, the
                   frozen phi rides this Stage-3 core field (phi = vth_field + phi_offset) instead of
                   a uniform V_th0. Ignored for z/gK (they ride the core via the simulate_kick
                   V_th_per_neuron hook). e_GABA is absent — it is a membrane-shunt parameter.
    """
    active = [name for name, v in (("z", z), ("phi_offset", phi_offset), ("gK", gK)) if v is not None]
    if len(active) != 1:
        raise ValueError(
            f"build_frozen_slowvars needs EXACTLY ONE of z/phi_offset/gK active (A1 single-variable "
            f"rule); got {active or 'none'}")
    cfg = SlowVarsConfig(use_z=(z is not None), use_phi=(phi_offset is not None),
                         use_gK=(gK is not None))
    sv = FrozenSlowVars(N, V_th0, cfg)
    if z is not None:
        sv.z = np.full(N, float(z))
    if phi_offset is not None:
        base = np.full(N, float(V_th0)) if vth_field is None else np.asarray(vth_field, float).copy()
        sv.phi = base + float(phi_offset)
    if gK is not None:
        sv.gK = np.full(N, float(gK))
    return sv
