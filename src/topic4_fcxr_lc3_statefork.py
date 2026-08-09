"""Persist a complete loop state, and reload it for a frozen-slow-state fork.

The frozen-state map needs to start many short runs from the *same* two moments of one real
trajectory -- one interictal, one inside the discharge -- with only the slow fields changed
between grid points.  Copying the slow variables and resetting the membrane would not be a fork
of that trajectory; it would be a fresh run wearing its slow fields.  So everything the engine
carries forward is written: membrane, refractory counters, all four synaptic states, the recurrent
pair, both delay rings, the OU variable, the generator's own state, and the slow fields.

`state_hash` is recorded on write and re-checked on read, so a state that was truncated or
silently altered between the two fails instead of quietly seeding a grid.
"""
from __future__ import annotations

import json
import os

import numpy as np

from src.topic4_fcxr_lc3 import FCXRLoopState, clone_loop_state, state_hash

FAST_ARRAYS = ("V", "ref", "s_E", "I_E", "s_I", "I_I", "s_E_rec", "I_E_rec",
               "ring_sE", "ring_sI")
SLOW_ARRAYS = ("z", "m", "phi", "x_relay", "y", "ee_relay_send", "h_lc2_E",
               "_h_source_lc2_E", "_z_sensor_last_E",
               "a")   # FCXR-LC4 channel open fraction; absent from files written before it existed,
                      # which load_into already tolerates by skipping missing keys


def save_loop_state(path, state: FCXRLoopState):
    """Write one state plus its hash; the hash is what makes the reload checkable."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arrays = {f"fast__{k}": np.asarray(getattr(state, k)) for k in FAST_ARRAYS}
    slow = state.slow
    for k in SLOW_ARRAYS:
        v = getattr(slow, k, None)
        if v is not None:
            arrays[f"slow__{k}"] = np.asarray(v)
    meta = dict(t=int(state.t), xi=float(state.xi),
                slow_step_i=int(getattr(slow, "_step_i", -1)),
                rng_state=_jsonable(state.rng_state),
                state_hash=state_hash(state))
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, meta=np.asarray([json.dumps(meta)]), **arrays)
    os.replace(tmp, path)
    return meta["state_hash"]


def load_into(path, template: FCXRLoopState) -> FCXRLoopState:
    """Rebuild a saved state onto a live template (which supplies the config objects).

    The template is cloned first, so the caller's own state is never written through.
    """
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"][0]))
    child = clone_loop_state(template)
    for k in FAST_ARRAYS:
        arr = z[f"fast__{k}"]
        cur = getattr(child, k)
        if np.asarray(cur).shape != arr.shape:
            raise ValueError(f"{k}: saved shape {arr.shape} != template {np.asarray(cur).shape}")
        np.asarray(cur)[...] = arr
    for k in SLOW_ARRAYS:
        key = f"slow__{k}"
        cur = getattr(child.slow, k, None)
        if cur is None:
            continue
        if key not in z.files:
            if k == "a":
                # A file written before this variable existed is a state whose channel was shut,
                # not a state whose channel happened to hold whatever the template was carrying.
                # Leaving the template's value here is the exact failure this module exists to
                # prevent: a fork that restores most of the state and silently keeps the rest.
                np.asarray(cur)[...] = 0.0
            continue
        np.asarray(cur)[...] = z[key]
    child.t = int(meta["t"])
    child.xi = float(meta["xi"])
    child.rng_state = _unjsonable(meta["rng_state"])
    if hasattr(child.slow, "_step_i"):
        child.slow._step_i = int(meta["slow_step_i"])
    got = state_hash(child)
    if got != meta["state_hash"]:
        raise ValueError(f"state hash mismatch for {path}: {got} != {meta['state_hash']}")
    return child


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return dict(__ndarray__=value.tolist(), dtype=str(value.dtype))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _unjsonable(value):
    if isinstance(value, dict) and "__ndarray__" in value:
        return np.asarray(value["__ndarray__"], dtype=np.dtype(value["dtype"]))
    if isinstance(value, dict):
        return {k: _unjsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unjsonable(v) for v in value]
    return value


def scaled_fields(d_star, x_star, alpha_d, alpha_x):
    """Scale the amplitude of two real spatial fields, keeping their shape.

    The pathology here is spatially heterogeneous, so a grid point that replaced every cell's slow
    variable with one number would be asking a question about a different substrate.  Scaling a
    field taken from a real trajectory keeps the two cores, the patient's axis and the recruitment
    history, and moves only how far along that field the tissue sits.
    """
    d = np.clip(np.asarray(d_star, float) * float(alpha_d), 0.0, 1.0)
    x_load = np.clip((1.0 - np.asarray(x_star, float)) * float(alpha_x), 0.0, 1.0)
    return d, 1.0 - x_load
