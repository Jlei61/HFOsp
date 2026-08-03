"""FCXR-LC3 exact-state continuation for the registered RC1 + H + Z + X path.

This module is deliberately outside the six blessed engine files.  It is a small,
resumable transcription of the *specific* no-kick ``simulate_kick`` path used by
LC3: full-conductance split excitation, recurrent smooth saturation, local H,
dynamic/frozen Z and the presynaptic X relay.  Unsupported engine branches fail
closed instead of silently producing a scientifically different trajectory.

The scientific requirement is stronger than the historical LC2 "fork": a fork
contains the fast membrane/synaptic/delay state, all slow fields and the network
RNG.  Tests compare this loop against the blessed simulator and compare a split
continuation against an uninterrupted run byte for byte.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from dataclasses import dataclass

import numpy as np


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENG = os.path.join(_ROOT, "snn_engine")
if _ENG not in sys.path:
    sys.path.insert(0, _ENG)

from kick_probe import _flatten_by_source, ee_std_apply  # noqa: E402
from params import compute_nu_theta  # noqa: E402


SCHEMA_VERSION = "fcxr-lc3-exact-state-1.0"


@dataclass
class FCXRLoopState:
    """Complete mutable state required for an exact LC3 continuation."""

    t: int
    V: np.ndarray
    ref: np.ndarray
    s_E: np.ndarray
    I_E: np.ndarray
    s_I: np.ndarray
    I_I: np.ndarray
    s_E_rec: np.ndarray
    I_E_rec: np.ndarray
    ring_sE: np.ndarray
    ring_sI: np.ndarray
    xi: float
    rng_state: dict
    slow: object


def clone_loop_state(state: FCXRLoopState) -> FCXRLoopState:
    """Deep clone a state; child forks must not alias any mutable array."""

    if not isinstance(state, FCXRLoopState):
        raise TypeError("state must be FCXRLoopState")
    return copy.deepcopy(state)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _hash_array(h, name, value):
    a = np.asarray(value)
    h.update(name.encode("utf-8"))
    h.update(a.dtype.str.encode("ascii"))
    h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(a).tobytes())


def state_hash(state: FCXRLoopState) -> str:
    """Hash all dynamical state used by the registered continuation path."""

    h = hashlib.sha256()
    h.update(SCHEMA_VERSION.encode("ascii"))
    h.update(np.asarray([state.t], dtype=np.int64).tobytes())
    h.update(np.asarray([state.xi], dtype=np.float64).tobytes())
    for name in (
        "V", "ref", "s_E", "I_E", "s_I", "I_I", "s_E_rec", "I_E_rec",
        "ring_sE", "ring_sI",
    ):
        _hash_array(h, name, getattr(state, name))
    h.update(json.dumps(_jsonable(state.rng_state), sort_keys=True,
                        separators=(",", ":")).encode("utf-8"))
    slow = state.slow
    for name in (
        "z", "m", "phi", "x_relay", "y", "ee_relay_send", "h_lc2_E",
        "_h_source_lc2_E", "_z_sensor_last_E",
    ):
        if hasattr(slow, name):
            _hash_array(h, f"slow.{name}", getattr(slow, name))
    h.update(np.asarray([int(getattr(slow, "_step_i", -1))], dtype=np.int64).tobytes())
    return h.hexdigest()


def validate_loop_state(state: FCXRLoopState, *, n, ne, max_delay_steps):
    """Fail closed on incomplete, aliased-by-shape or non-finite checkpoints."""

    if not isinstance(state, FCXRLoopState):
        raise TypeError("state must be FCXRLoopState")
    if int(state.t) < 0:
        raise ValueError("state.t must be non-negative")
    vector_float = ("V", "s_E", "I_E", "s_I", "I_I", "s_E_rec", "I_E_rec")
    for name in vector_float:
        a = np.asarray(getattr(state, name))
        if a.shape != (n,) or not np.all(np.isfinite(a)):
            raise ValueError(f"{name} must be finite shape ({n},)")
    ref = np.asarray(state.ref)
    if ref.shape != (n,) or not np.issubdtype(ref.dtype, np.integer) or np.any(ref < 0):
        raise ValueError(f"ref must be nonnegative integer shape ({n},)")
    m = int(max_delay_steps) + 1
    for name in ("ring_sE", "ring_sI"):
        a = np.asarray(getattr(state, name))
        if a.shape != (m, n) or not np.all(np.isfinite(a)):
            raise ValueError(f"{name} must be finite shape ({m},{n})")
    if not np.isfinite(state.xi):
        raise ValueError("xi must be finite")
    if not isinstance(state.rng_state, dict):
        raise ValueError("rng_state must be a bit-generator state dict")
    slow = state.slow
    if slow is None or int(getattr(slow, "N", -1)) != n or int(getattr(slow, "NE", -1)) != ne:
        raise ValueError("slow object does not match N/NE")
    if int(getattr(slow, "_step_i", -1)) != int(state.t):
        raise ValueError("slow step counter does not match state.t")
    z = np.asarray(slow.z[:ne])
    x = np.asarray(slow.x_relay)
    h = np.asarray(slow.h_lc2_E)
    if z.shape != (ne,) or not np.all(np.isfinite(z)) or np.any((z < 0.0) | (z > 1.0)):
        raise ValueError("slow z field is invalid")
    if x.shape != (ne,) or not np.all(np.isfinite(x)) or np.any((x < 0.0) | (x > 1.0)):
        raise ValueError("slow X field is invalid")
    if h.shape != (ne,) or not np.all(np.isfinite(h)) or np.any(h < 0.0):
        raise ValueError("slow H field is invalid")
    return True


def replace_frozen_fields(state: FCXRLoopState, *, d_field=None, x_field=None) -> FCXRLoopState:
    """Clone and replace only registered D and/or presynaptic-X fields.

    The duplicated config arrays are bookkeeping copies of the same frozen field,
    not independent dynamical coordinates.  ``ee_relay_send`` is synchronized with
    X so the first resumed spike cannot use a stale pre-fork availability.
    """

    child = clone_loop_state(state)
    slow = child.slow
    ne = int(slow.NE)
    if d_field is not None:
        d = np.asarray(d_field, dtype=float)
        if d.shape != (ne,) or not np.all(np.isfinite(d)) or np.any((d < 0.0) | (d > 1.0)):
            raise ValueError(f"d_field must be finite in [0,1] and shape ({ne},)")
        z = 1.0 - d
        slow.z[:ne] = z
        if getattr(slow.cfg, "z_frozen_E", None) is not None:
            slow.cfg.z_frozen_E = z.copy()
    if x_field is not None:
        x = np.asarray(x_field, dtype=float)
        if x.shape != (ne,) or not np.all(np.isfinite(x)) or np.any((x < 0.0) | (x > 1.0)):
            raise ValueError(f"x_field must be finite in [0,1] and shape ({ne},)")
        slow.x_relay[:] = x
        slow.ee_relay_send[:] = x
        if getattr(slow.cfg, "x_relay_frozen_E", None) is not None:
            slow.cfg.x_relay_frozen_E = x.copy()
    return child


def _constants(p, net):
    ne, ni = int(net["NE"]), int(net["NI"])
    n = ne + ni
    labels = np.asarray(net["labels"])
    ampa, gaba = net["ampa_by_delay"], net["gaba_by_delay"]
    m = int(net["max_delay_steps"]) + 1
    dt = float(p.dt)
    ampa_bins = [d for d in range(m) if ampa[d].nnz > 0]
    gaba_bins = [d for d in range(m) if gaba[d].nnz > 0]
    if "ampa_flat" not in net:
        net["ampa_flat"] = _flatten_by_source(ampa, ampa_bins, ne)
        net["gaba_flat"] = _flatten_by_source(gaba, gaba_bins, ni)
    tau_m = np.where(labels == 0, p.tau_m_E, p.tau_m_I).astype(np.float64)
    nu_theta, _, _ = compute_nu_theta(p)
    sigma_xi = p.sigma_n * 1e-3 * np.sqrt(p.tau_n / 2.0)
    ou_a = np.exp(-dt / p.tau_n)
    return dict(
        ne=ne, ni=ni, n=n, labels=labels, is_e=(labels == 0), m=m, dt=dt,
        decay_s_e=np.exp(-dt / p.tau_r_AMPA), decay_i_e=np.exp(-dt / p.tau_d_AMPA),
        decay_s_i=np.exp(-dt / p.tau_r_GABA), decay_i_i=np.exp(-dt / p.tau_d_GABA),
        decay_v=np.exp(-dt / tau_m),
        ref_steps=np.where(labels == 0, int(round(p.tau_ref_E / dt)),
                           int(round(p.tau_ref_I / dt))).astype(np.int32),
        ext_incr=(tau_m / p.tau_r_AMPA) * np.where(labels == 0, p.J_ext_E, p.J_ext_I),
        nu_theta=nu_theta, nu_sig_const=p.nu_ext_ratio * nu_theta,
        ou_a=ou_a, ou_b=sigma_xi * np.sqrt(1.0 - ou_a * ou_a),
    )


def _assert_registered_path(slow):
    if slow is None:
        raise ValueError("LC3 exact loop requires the registered MZ slow object")
    if not slow.uses_conductance_membrane() or not slow.uses_split_excitation():
        raise ValueError("LC3 exact loop requires full-conductance split excitation")
    if not getattr(slow.cfg, "rec_conductance", False):
        raise ValueError("LC3 exact loop requires recurrent conductance")
    if getattr(slow.cfg, "use_SG", False):
        raise ValueError("shared-inhibition branch is outside LC3")
    if not getattr(slow.cfg, "use_h_lc2", False):
        raise ValueError("LC3 exact loop requires local H")
    if not slow.uses_ee_relay():
        raise ValueError("LC3 exact loop requires presynaptic X relay")


def run_fcxr_loop(p, net, *, slow=None, start=None, n_steps, capture_final=False,
                  store_spikes=True, v_th_per_neuron=None):
    """Run or resume the registered no-kick FCXR path with exact engine arithmetic."""

    if (slow is None) == (start is None):
        raise ValueError("provide exactly one of fresh slow or start state")
    c = _constants(p, net)
    ne, ni, n, m, dt = c["ne"], c["ni"], c["n"], c["m"], c["dt"]
    labels, is_e = c["labels"], c["is_e"]
    rng = net["rng"]
    a_indptr, a_dst, a_dly, a_w = net["ampa_flat"]
    g_indptr, g_dst, g_dly, g_w = net["gaba_flat"]
    base_vth = p.V_th if v_th_per_neuron is None else np.asarray(v_th_per_neuron, dtype=float)

    if start is None:
        _assert_registered_path(slow)
        t0 = 0
        V = np.full(n, p.V_reset, dtype=np.float64)
        ref = np.zeros(n, dtype=np.int32)
        s_E = np.zeros(n); I_E = np.zeros(n); s_I = np.zeros(n); I_I = np.zeros(n)
        s_E_rec = np.zeros(n); I_E_rec = np.zeros(n)
        ring_sE = np.zeros((m, n)); ring_sI = np.zeros((m, n))
        xi = 0.0
        # simulate_kick consumes these recorder-sampling draws before the loop.
        _ = rng.choice(ne, size=min(80, ne), replace=False)
        _ = ne + rng.choice(ni, size=min(20, ni), replace=False)
    else:
        validate_loop_state(start, n=n, ne=ne, max_delay_steps=m - 1)
        slow = start.slow
        _assert_registered_path(slow)
        t0 = int(start.t)
        V = start.V.copy(); ref = start.ref.copy()
        s_E = start.s_E.copy(); I_E = start.I_E.copy()
        s_I = start.s_I.copy(); I_I = start.I_I.copy()
        s_E_rec = start.s_E_rec.copy(); I_E_rec = start.I_E_rec.copy()
        ring_sE = start.ring_sE.copy(); ring_sI = start.ring_sI.copy()
        xi = float(start.xi)
        rng.bit_generator.state = copy.deepcopy(start.rng_state)

    rate_e = np.zeros(int(n_steps), dtype=float)
    rate_i = np.zeros(int(n_steps), dtype=float)
    e_spikes = np.zeros((int(n_steps), ne), dtype=bool) if store_spikes else None

    for k in range(int(n_steps)):
        t = t0 + k
        xi = c["ou_a"] * xi + c["ou_b"] * rng.standard_normal()
        nu_now = c["nu_sig_const"] + xi
        if nu_now < 0.0:
            nu_now = 0.0

        s_E *= c["decay_s_e"]
        s_I *= c["decay_s_i"]
        slot = t % m
        s_E_rec *= c["decay_s_e"]
        s_E_rec += ring_sE[slot]
        s_E += ring_sE[slot]; ring_sE[slot] = 0.0
        s_I += ring_sI[slot]; ring_sI[slot] = 0.0
        nu_vec = np.full(n, max(nu_now, 0.0))
        ext = rng.poisson(nu_vec * dt, size=n).astype(np.float64)
        s_E += ext * c["ext_incr"]

        I_E = s_E + (I_E - s_E) * c["decay_i_e"]
        I_I = s_I + (I_I - s_I) * c["decay_i_i"]
        I_E_rec = s_E_rec + (I_E_rec - s_E_rec) * c["decay_i_e"]

        drive, g_rel, g_rev = slow.membrane_terms(I_E, I_I, labels, I_E_rec=I_E_rec)
        vth = slow.threshold(base_vth)
        ref -= 1
        np.maximum(ref, 0, out=ref)
        free = ref == 0
        denom = 1.0 + g_rel
        v_inf = (drive + g_rev) / denom
        vtmp = v_inf + (V - v_inf) * c["decay_v"] ** denom
        vtmp[~is_e] = drive[~is_e] + (V[~is_e] - drive[~is_e]) * c["decay_v"][~is_e]
        V = np.where(free, vtmp, p.V_reset)
        spk = free & (V >= (vth if np.isscalar(vth) else vth))
        V[spk] = p.V_reset
        ref[spk] = c["ref_steps"][spk]

        slow.step(spk, labels, dt)
        rate_e[k] = spk[:ne].sum()
        rate_i[k] = spk[ne:].sum()
        if store_spikes:
            e_spikes[k] = spk[:ne]

        if spk.any():
            sp_e = np.where(spk[:ne])[0]
            sp_i = np.where(spk[ne:])[0]
            if sp_e.size:
                st = a_indptr[sp_e]
                cnt = a_indptr[sp_e + 1] - st
                total = int(cnt.sum())
                if total:
                    idx = (np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt)
                           + np.repeat(st, cnt))
                    x_per_edge = np.repeat(slow.ee_relay_send[sp_e], cnt)
                    w_eff = ee_std_apply(a_w[idx], a_dst[idx], x_per_edge, ne)
                    np.add.at(ring_sE, ((t + a_dly[idx]) % m, a_dst[idx]), w_eff)
            if sp_i.size:
                st = g_indptr[sp_i]
                cnt = g_indptr[sp_i + 1] - st
                total = int(cnt.sum())
                if total:
                    idx = (np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt)
                           + np.repeat(st, cnt))
                    np.add.at(ring_sI, ((t + g_dly[idx]) % m, g_dst[idx]), g_w[idx])

    checkpoint = None
    if capture_final:
        checkpoint = FCXRLoopState(
            t=t0 + int(n_steps), V=V.copy(), ref=ref.copy(), s_E=s_E.copy(), I_E=I_E.copy(),
            s_I=s_I.copy(), I_I=I_I.copy(), s_E_rec=s_E_rec.copy(), I_E_rec=I_E_rec.copy(),
            ring_sE=ring_sE.copy(), ring_sI=ring_sI.copy(), xi=float(xi),
            rng_state=copy.deepcopy(rng.bit_generator.state), slow=copy.deepcopy(slow),
        )
        validate_loop_state(checkpoint, n=n, ne=ne, max_delay_steps=m - 1)

    return dict(
        rate_E=rate_e / ne / dt * 1e3,
        rate_I=rate_i / ni / dt * 1e3,
        E_spk_bool=e_spikes,
        n_steps=int(n_steps),
        t0=t0,
        checkpoint=checkpoint,
    )
