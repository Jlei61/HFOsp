"""Exact-state checkpoint controller for the canonical Z/M simulator (spec rev3.1 §3.1, plan Task 2/3).

`simulate_kick` keeps ONE integration loop. All checkpoint machinery -- state packing, restore, the
snapshot schedule, the mean-input arm and the external-drive audit recorders -- lives here and is
reached through a single `zm_ckpt=` argument. `zm_ckpt=None` takes no branch in the engine, so the
default numerical path is byte-identical to the pre-edit engine.

Freeze semantics live in `topic4_zm_fork_state.FreezeWrapper`, not here: freezing is a property of
the slow layer, not of the checkpoint.

Serialization is explicit and pickle-free: every array is stored under its own npz key, scalars as
0-d arrays, the bit-generator state and the manifest as UTF-8 byte arrays.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

STATE_SCHEMA = "zm_sim_state_v1"

#: engine arrays that always exist
_ENGINE_ARRAYS = ("V", "ref", "s_E", "I_E", "s_I", "I_I", "ring_sE", "ring_sI")
#: engine arrays that exist only when the matching feature is on
_ENGINE_OPT_ARRAYS = ("s_E_rec", "I_E_rec", "x_dep")
#: engine scalars
_ENGINE_SCALARS = ("xi", "t", "_es_ema", "_es_run", "r_ema")
#: slow-layer arrays / scalars (names are the SpatialSlowField attributes)
_SLOW_ARRAYS = (
    "z", "m", "phi_increment", "_I_I_last", "rE", "rI", "rE_fast",
    "q_I", "g_K", "p", "n_load", "a_shunt",
)
_SLOW_SCALARS = ("mu_G", "S_G", "h_G", "H", "_t")


def _inner(slow):
    """Unwrap a FreezeWrapper so scalar writes land on the real SpatialSlowField."""
    return getattr(slow, "inner", slow)


# ================================================================= capture / restore
def capture_slow(slow):
    if slow is None:
        return {}
    s = _inner(slow)
    out = {}
    for k in _SLOW_ARRAYS:
        v = getattr(s, k, None)
        if v is not None:
            out[f"slow.{k}"] = np.array(v, copy=True)
    for k in _SLOW_SCALARS:
        v = getattr(s, k, None)
        if v is not None:
            out[f"slow.{k}"] = np.asarray(float(v))
    return out


def restore_slow(slow, state):
    if slow is None:
        return
    s = _inner(slow)
    for k in _SLOW_ARRAYS:
        key = f"slow.{k}"
        if key not in state:
            continue
        cur = getattr(s, k, None)
        if cur is None:
            setattr(s, k, np.array(state[key], copy=True))
        else:
            np.asarray(cur)[...] = state[key]
    for k in _SLOW_SCALARS:
        key = f"slow.{k}"
        if key in state:
            setattr(s, k, float(np.asarray(state[key])))


def state_hash(state):
    """Deterministic SHA256 over every array in the state (sorted key order, dtype+shape+bytes)."""
    h = hashlib.sha256()
    for k in sorted(state):
        if k == "rng_state":
            h.update(k.encode())
            h.update(json.dumps(state[k], sort_keys=True, default=str).encode())
            continue
        a = np.asarray(state[k])
        h.update(f"{k}|{a.dtype.str}|{a.shape}|".encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


# ================================================================= controller
class ZMCheckpoint:
    """The single object `simulate_kick(zm_ckpt=...)` talks to.

    initial_state      restore-and-continue from an exact snapshot (its 't' sets the absolute step)
    snapshot_steps     absolute steps at which to capture (capture happens at END of step k-1, i.e.
                       the captured state is the one a continuation starting at step k must have)
    return_final_state capture after the loop; refused when a runaway early-stop truncated the run
                       (that state is mid-step: the delay scatter for the break frame never ran)
    ext_mean_only      replace the Poisson external drive by its MEAN and hold the OU term at 0
                       (spec §3.2 'matched external mean with stochastic fluctuations removed')
    dump_ext           record per-step (nu_now, ext.sum()) so paired-noise matching across arms is
                       auditable rather than assumed
    """

    def __init__(self, initial_state=None, snapshot_steps=None, return_final_state=False,
                 ext_mean_only=False, dump_ext=False, rng_state=None):
        self.initial_state = initial_state
        self.snapshot_steps = set(int(s) for s in snapshot_steps) if snapshot_steps else None
        self.return_final_state = bool(return_final_state)
        self.ext_mean_only = bool(ext_mean_only)
        self.dump_ext = bool(dump_ext)
        self.rng_state = rng_state          # overrides the snapshot's stream (noise_resample_*)
        self.snapshots = {}
        self.final_state = None
        self.final_truncated = False
        self.ext_nu = None
        self.ext_sum = None

    # ---------------- called once, before the loop ----------------
    def begin(self, *, nsteps, rng, slow=None, **_ignored):
        if self.dump_ext:
            self.ext_nu = np.zeros(nsteps)
            self.ext_sum = np.zeros(nsteps)
        st = self.initial_state
        if st is not None:
            restore_slow(slow, st)
            if "rng_state" in st:
                rng.bit_generator.state = _decode_rng(st["rng_state"])
        if self.rng_state is not None:      # paired-noise replicate: a different external stream
            rng.bit_generator.state = _decode_rng(self.rng_state)
        return st

    # ---------------- called at snapshot steps and after the loop ----------------
    def capture(self, t_abs, *, rng, slow=None, **arrays):
        """Exact state a continuation starting at absolute step `t_abs` must be given."""
        state = {"t": np.asarray(int(t_abs))}
        for k, v in arrays.items():
            if v is None:
                continue
            state[k] = np.asarray(v).copy() if np.ndim(v) else np.asarray(float(v))
        state["rng_state"] = _encode_rng(rng.bit_generator.state)
        state.update(capture_slow(slow))
        return state

    def take(self, t_abs, *, store, rng, slow=None, **arrays):
        state = self.capture(t_abs, rng=rng, slow=slow, **arrays)
        if store:
            self.snapshots[int(t_abs)] = state
        else:
            self.final_state = state
        return state


def _encode_rng(state):
    """PCG64 state dict -> plain JSON-safe dict (ints become str to survive npz round-trip)."""
    return json.loads(json.dumps(state, default=str))


def _decode_rng(state):
    out = json.loads(json.dumps(state)) if not isinstance(state, str) else json.loads(state)
    st = dict(out)
    if isinstance(st.get("state"), dict):
        inner = dict(st["state"])
        for k in ("state", "inc"):
            if k in inner:
                inner[k] = int(inner[k])
        st["state"] = inner
    for k in ("has_uint32", "uinteger"):
        if k in st:
            st[k] = int(st[k])
    return st


# ================================================================= npz serialization
def save_state_npz(state, manifest, path):
    """Explicit schema, no pickle, no object arrays."""
    payload = {}
    for k, v in state.items():
        if k == "rng_state":
            payload["__rng_state__"] = np.frombuffer(
                json.dumps(v, sort_keys=True, default=str).encode(), dtype=np.uint8)
            continue
        payload[k] = np.asarray(v)
    man = dict(manifest or {})
    man.update(schema=STATE_SCHEMA, state_hash=state_hash(state),
               keys=sorted(k for k in state if k != "rng_state"))
    payload["__manifest__"] = np.frombuffer(
        json.dumps(man, sort_keys=True, default=str).encode(), dtype=np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.npz"
    np.savez_compressed(tmp, **payload)   # lossless -> exactness preserved; delay rings are sparse
    os.replace(tmp, path)
    return man


def read_manifest(path):
    with np.load(path, allow_pickle=False) as z:
        return json.loads(bytes(z["__manifest__"]).decode())


def load_state_npz(path, expected_config_sha=None, expected_engine_sha=None, expected_dt=None):
    """Fail closed on schema / config / engine / dt / state-hash mismatch (spec §14.1)."""
    with np.load(path, allow_pickle=False) as z:
        man = json.loads(bytes(z["__manifest__"]).decode())
        state = {}
        for k in z.files:
            if k in ("__manifest__", "__rng_state__"):
                continue
            state[k] = np.array(z[k])
        if "__rng_state__" in z.files:
            state["rng_state"] = json.loads(bytes(z["__rng_state__"]).decode())
    if man.get("schema") != STATE_SCHEMA:
        raise ValueError(f"{path}: schema {man.get('schema')!r} != {STATE_SCHEMA!r}")
    if expected_config_sha is not None and man.get("config_sha") != expected_config_sha:
        raise ValueError(f"{path}: config_sha mismatch "
                         f"({man.get('config_sha')} != {expected_config_sha})")
    if expected_engine_sha is not None and man.get("engine_sha") != expected_engine_sha:
        raise ValueError(f"{path}: engine_sha mismatch")
    if expected_dt is not None and float(man.get("dt", -1)) != float(expected_dt):
        raise ValueError(f"{path}: dt mismatch ({man.get('dt')} != {expected_dt})")
    got = state_hash(state)
    if man.get("state_hash") not in (None, got):
        raise ValueError(f"{path}: state_hash mismatch (file corrupted or hand-edited)")
    return state, man
