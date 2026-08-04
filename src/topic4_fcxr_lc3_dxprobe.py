"""Freeze a live FCXR-LC3 trajectory state so D and X can be interrogated separately.

The 102-row frozen map covered mean wear ``D`` in ``[0, 0.097]`` -- the levels a 24 s
interictal record ever reached.  The first no-kick trajectory drove ``D`` to ``0.663``
within 45 s and did not terminate, with every cell already below the map's ``a_X =
0.65`` return boundary.  The map therefore cannot say whether the seizure persists
because ``X`` cannot brake hard enough or because ``D`` has left the region where
braking works at all.

Separating those needs the *actual* late-bout state, not a fresh preparation: the
membrane voltages, refractory counters, synaptic filters, delay rings and RNG all
matter.  That state exists as a byte-parity-verified landmark checkpoint, so the probe
seeds from it and freezes the two slow fields at chosen values.

``replace_frozen_fields`` answers a different question -- it swaps the fields of a state
that is *already* frozen, and leaves ``cfg.z_frozen_E`` / ``cfg.x_relay_frozen_E``
untouched when they are ``None``, which is exactly the case for a dynamic state.  A
dynamic state needs the config flipped as well, and the engine requires ``z_frozen_E``
to travel with ``use_z=False``.
"""
from __future__ import annotations

import numpy as np

from src.topic4_fcxr_lc3 import clone_loop_state

DXPROBE_SCHEMA = "fcxr-lc3-dx-probe-1.0"


def _validated(field, ne, name):
    a = np.asarray(field, dtype=float)
    if a.ndim == 0:
        a = np.full(int(ne), float(a))
    if a.shape != (int(ne),):
        raise ValueError(f"{name} must be a scalar or a field of shape ({ne},), got {a.shape}")
    if not np.all(np.isfinite(a)) or np.any((a < 0.0) | (a > 1.0)):
        raise ValueError(f"{name} must be finite and within [0,1]")
    return a


def freeze_dynamic_state(state, *, d_field=None, x_field=None):
    """Clone a dynamic state and hold its D and X fields fixed from here on.

    ``d_field`` / ``x_field`` may be a scalar or a per-E field; ``None`` freezes that
    variable at whatever the state currently carries, which is what the control arm
    needs.  ``ee_relay_send`` is synchronised with X so the first resumed spike cannot
    scatter through a stale pre-freeze availability.
    """

    child = clone_loop_state(state)
    slow = child.slow
    ne = int(slow.NE)

    z = (np.asarray(slow.z[:ne], dtype=float).copy() if d_field is None
         else 1.0 - _validated(d_field, ne, "d_field"))
    if not np.all(np.isfinite(z)) or np.any((z < 0.0) | (z > 1.0)):
        raise ValueError("frozen z must be finite and within [0,1]")
    slow.z[:ne] = z
    slow.cfg.z_frozen_E = z.copy()
    slow.cfg.use_z = False          # the engine rejects a frozen field that still evolves

    x = (np.asarray(slow.x_relay, dtype=float).copy() if x_field is None
         else _validated(x_field, ne, "x_field"))
    slow.x_relay[:] = x
    slow.ee_relay_send[:] = x
    slow.cfg.x_relay_frozen_E = x.copy()
    slow.cfg.use_x = True           # the frozen branch lives inside the use_x block
    return child


def probe_summary(*, arm_id, d_field, x_field, classification, total_ms, extended) -> dict:
    """One arm's record, carrying the fields it was actually frozen at."""
    d = np.asarray(d_field, dtype=float)
    x = np.asarray(x_field, dtype=float)
    return dict(
        schema=DXPROBE_SCHEMA, arm_id=str(arm_id),
        D_mean=float(d.mean()), D_min=float(d.min()), D_max=float(d.max()),
        X_mean=float(x.mean()), X_min=float(x.min()), X_max=float(x.max()),
        resolved_label=classification["label"],
        workpoint_label=classification["workpoint_label"],
        refractory_ceiling_fraction=classification["refractory_ceiling_fraction"],
        h_mean=classification["h_mean"], h_slope_per_s=classification["h_slope_per_s"],
        numerical_unsafe=classification["numerical_unsafe"],
        total_ms=float(total_ms), extended=bool(extended),
    )
