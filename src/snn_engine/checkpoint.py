"""Single enumeration point for LIF simulator state.

Every mutable quantity the integration loop reads is captured here. Adding a new
mutable engine variable without adding it to REQUIRED_KEYS is the failure mode
this module exists to prevent: a checkpoint that silently omits one delay-ring
slot or one RNG stream produces a resumed trajectory that looks plausible and is
wrong. Gate C (perturbed reload == full rerun) is what catches an omission.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile

import numpy as np

CHECKPOINT_SCHEMA = "topic4_snn_checkpoint_v1"

REQUIRED_KEYS = (
    "schema", "step", "absolute_time_ms", "V", "ref", "s_E", "I_E", "s_I", "I_I",
    "ring_sE", "ring_sI", "xi", "rng_state", "ras_keep", "es_ema", "es_run",
    "track_rec", "s_E_rec", "I_E_rec", "slow", "external_drive",
)

_SLOW_BASE_ARRAYS = ("z", "m", "I_I_last", "acc_D", "acc_A")
_SLOW_SPATIAL_ARRAYS = (
    # SpatialZMQIGKSlowVars state. These keys are absent for the historical
    # lumped Z/M object and therefore do not alter its checkpoint bytes.
    "q_I", "qdriver_rE", "qdriver_rI", "field_count_E", "field_count_I",
    "last_m_drive_E", "last_q_drive",
)
_SLOW_ARRAYS = _SLOW_BASE_ARRAYS + _SLOW_SPATIAL_ARRAYS
_DRIVE_ARRAYS = ("field_state", "cached")


def capture(*, step, absolute_time_ms, V, ref, s_E, I_E, s_I, I_I, ring_sE,
            ring_sI, xi, rng, ras_keep, es_ema, es_run, track_rec,
            s_E_rec, I_E_rec, slow, external_drive):
    state = {
        "schema": CHECKPOINT_SCHEMA,
        "step": int(step),
        "absolute_time_ms": float(absolute_time_ms),
        "V": np.array(V, copy=True),
        "ref": np.array(ref, copy=True),
        "s_E": np.array(s_E, copy=True),
        "I_E": np.array(I_E, copy=True),
        "s_I": np.array(s_I, copy=True),
        "I_I": np.array(I_I, copy=True),
        "ring_sE": np.array(ring_sE, copy=True),
        "ring_sI": np.array(ring_sI, copy=True),
        "xi": float(xi),
        "rng_state": rng.bit_generator.state,
        "ras_keep": np.array(ras_keep, copy=True),
        "es_ema": float(es_ema),
        "es_run": int(es_run),
        "track_rec": bool(track_rec),
        "s_E_rec": None if s_E_rec is None else np.array(s_E_rec, copy=True),
        "I_E_rec": None if I_E_rec is None else np.array(I_E_rec, copy=True),
        "slow": None,
        "external_drive": None,
    }
    if slow is not None:
        acc_d = getattr(slow, "_acc_D", None)
        acc_a = getattr(slow, "_acc_A", None)
        state["slow"] = {
            "kind": type(slow).__name__,
            "z": np.array(slow.z, copy=True),
            "m": np.array(slow.m, copy=True),
            "I_I_last": np.array(slow._I_I_last, copy=True),
            "step_index": int(slow._step_index),
            "acc_n": int(getattr(slow, "_acc_n", 0)),
            "acc_seen": int(getattr(slow, "_acc_seen", 0)),
            "acc_D": None if acc_d is None else np.array(acc_d, copy=True),
            "acc_A": None if acc_a is None else np.array(acc_a, copy=True),
        }
        if hasattr(slow, "q_I"):
            state["slow"].update({
                "q_I": np.array(slow.q_I, copy=True),
                "qdriver_rE": np.array(slow._qdriver.rE, copy=True),
                "qdriver_rI": np.array(slow._qdriver.rI, copy=True),
                "field_count_E": np.array(slow._field_count_E, copy=True),
                "field_count_I": np.array(slow._field_count_I, copy=True),
                "last_m_drive_E": np.array(slow._last_m_drive_E, copy=True),
                "last_q_drive": np.array(slow._last_q_drive, copy=True),
                "field_steps_seen": int(slow._field_steps_seen),
                "field_steps_per_update": (
                    None if slow._field_steps_per_update is None
                    else int(slow._field_steps_per_update)
                ),
            })
    if external_drive is not None:
        state["external_drive"] = {
            "field_state": np.array(external_drive._state, copy=True),
            "cached": np.array(external_drive._cached, copy=True),
            "next_step": int(external_drive._next_step),
            "last_step": int(external_drive._last_step),
            "rng_state": external_drive._rng.bit_generator.state,
        }
    return state


def restore_slow(state, slow):
    payload = state["slow"]
    if payload is None or slow is None:
        if (payload is None) != (slow is None):
            raise ValueError("checkpoint slow payload and slow object disagree")
        return
    if payload["kind"] != type(slow).__name__:
        raise ValueError("checkpoint slow protocol differs from the live object")
    slow.z[:] = payload["z"]
    slow.m[:] = payload["m"]
    slow._I_I_last = np.array(payload["I_I_last"], copy=True)
    slow._step_index = int(payload["step_index"])
    slow._acc_n = int(payload.get("acc_n", 0))
    slow._acc_seen = int(payload.get("acc_seen", 0))
    acc_d, acc_a = payload.get("acc_D"), payload.get("acc_A")
    slow._acc_D = None if acc_d is None else np.array(acc_d, copy=True)
    slow._acc_A = None if acc_a is None else np.array(acc_a, copy=True)
    if hasattr(slow, "q_I"):
        required = (
            "q_I", "qdriver_rE", "qdriver_rI", "field_count_E",
            "field_count_I", "last_m_drive_E", "last_q_drive",
            "field_steps_seen", "field_steps_per_update",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(
                "spatial Z/M checkpoint is incomplete: " + ", ".join(missing))
        slow.q_I[:] = payload["q_I"]
        slow._qdriver.rE[:] = payload["qdriver_rE"]
        slow._qdriver.rI[:] = payload["qdriver_rI"]
        slow._field_count_E[:] = payload["field_count_E"]
        slow._field_count_I[:] = payload["field_count_I"]
        slow._last_m_drive_E[:] = payload["last_m_drive_E"]
        slow._last_q_drive[:] = payload["last_q_drive"]
        slow._field_steps_seen = int(payload["field_steps_seen"])
        steps = payload["field_steps_per_update"]
        slow._field_steps_per_update = None if steps is None else int(steps)


def restore_external_drive(state, drive):
    payload = state["external_drive"]
    if payload is None or drive is None:
        if (payload is None) != (drive is None):
            raise ValueError("checkpoint drive payload and drive object disagree")
        return
    drive._state = np.array(payload["field_state"], copy=True)
    drive._cached = np.array(payload["cached"], copy=True)
    drive._next_step = int(payload["next_step"])
    drive._last_step = int(payload["last_step"])
    drive._rng.bit_generator.state = payload["rng_state"]


def _flatten(state):
    arrays, scalars = {}, {}
    for key, value in state.items():
        if key in ("slow", "external_drive"):
            continue
        if isinstance(value, np.ndarray):
            arrays[key] = value
        else:
            scalars[key] = value
    for prefix, payload, array_names in (("slow", state["slow"], _SLOW_ARRAYS),
                                         ("external_drive", state["external_drive"],
                                          _DRIVE_ARRAYS)):
        scalars[f"{prefix}__present"] = payload is not None
        if payload is None:
            continue
        for key, value in payload.items():
            if key in array_names and value is not None:
                arrays[f"{prefix}__{key}"] = np.asarray(value)
            else:
                scalars[f"{prefix}__{key}"] = value
    return arrays, scalars


def save(state, path):
    """Atomic write; returns the file's sha256 so a caller can record it."""
    path = str(path)
    arrays, scalars = _flatten(state)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    os.close(handle)
    np.savez(tmp, __meta__=np.array(json.dumps(scalars, sort_keys=True)), **arrays)
    os.replace(tmp + ".npz", path)
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as handle:
        scalars = json.loads(str(handle["__meta__"]))
        arrays = {key: handle[key] for key in handle.files if key != "__meta__"}
    state = {}
    slow, drive = {}, {}
    for key, value in scalars.items():
        if key.startswith("slow__"):
            slow[key[len("slow__"):]] = value
        elif key.startswith("external_drive__"):
            drive[key[len("external_drive__"):]] = value
        else:
            state[key] = value
    for key, value in arrays.items():
        if key.startswith("slow__"):
            slow[key[len("slow__"):]] = value
        elif key.startswith("external_drive__"):
            drive[key[len("external_drive__"):]] = value
        else:
            state[key] = value
    for name, payload, array_names in (("slow", slow, _SLOW_BASE_ARRAYS),
                                       ("external_drive", drive, _DRIVE_ARRAYS)):
        present = payload.pop("present", False)
        if not present:
            state[name] = None
            continue
        for array_name in array_names:
            payload.setdefault(array_name, None)
        state[name] = payload
    for key in ("s_E_rec", "I_E_rec"):
        state.setdefault(key, None)
    return state


def _digest_update(hasher, value, label):
    hasher.update(label.encode())
    if value is None:
        hasher.update(b"None")
    elif isinstance(value, np.ndarray):
        hasher.update(np.ascontiguousarray(value).tobytes())
        hasher.update(str(value.dtype).encode())
        hasher.update(str(value.shape).encode())
    else:
        hasher.update(json.dumps(value, sort_keys=True, default=str).encode())


def digest(state):
    hasher = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if isinstance(value, dict):
            for sub in sorted(value):
                _digest_update(hasher, value[sub], f"{key}.{sub}")
        else:
            _digest_update(hasher, value, key)
    return hasher.hexdigest()
