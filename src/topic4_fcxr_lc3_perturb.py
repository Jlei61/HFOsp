"""Exact-state LC3 continuation with one registered additive E-current pulse.

This module is intentionally separate from :mod:`topic4_fcxr_lc3`: the active
field/geometry jobs hash that implementation.  A zero-amplitude pulse follows
the exact same arithmetic as the registered continuation; nonzero current is
added only after slow-variable membrane terms are evaluated and before the
membrane update.  No state variable, parameter or connectivity object is reset.
"""
from __future__ import annotations

import copy

import numpy as np

from src.topic4_fcxr_lc3 import (
    FCXRLoopState,
    _assert_registered_path,
    _constants,
    validate_loop_state,
)
from kick_probe import ee_std_apply


SCHEMA_VERSION = "fcxr-lc3-perturb-1.0"


def validate_current_pattern(pattern, ne: int) -> np.ndarray:
    pattern = np.asarray(pattern, dtype=float)
    if pattern.shape != (int(ne),) or not np.all(np.isfinite(pattern)):
        raise ValueError(f"current pattern must be finite shape ({int(ne)},)")
    if not np.any(pattern):
        raise ValueError("current pattern must contain at least one nonzero entry")
    return pattern


def current_accounting(pattern, *, amplitude: float, duration_ms: float) -> dict:
    """Return explicit cell-count, positive charge and RMS contracts."""

    pattern = np.asarray(pattern, dtype=float)
    current = float(amplitude) * pattern
    if not np.isfinite(amplitude) or not np.isfinite(duration_ms) or duration_ms < 0:
        raise ValueError("amplitude/duration must be finite and duration non-negative")
    return dict(
        active_cell_count=int(np.count_nonzero(current)),
        positive_charge=float(np.maximum(current, 0.0).sum() * duration_ms),
        negative_charge_magnitude=float((-np.minimum(current, 0.0)).sum() * duration_ms),
        rms_current=float(np.sqrt(np.mean(current * current))),
        l2_current=float(np.linalg.norm(current)),
        duration_ms=float(duration_ms),
    )


def run_fcxr_perturbation(
    p,
    net,
    *,
    start: FCXRLoopState,
    n_steps: int,
    current_pattern,
    amplitude: float,
    pulse_steps: int,
    pulse_start_step: int = 0,
    capture_final: bool = False,
    store_spikes: bool = True,
    v_th_per_neuron=None,
):
    """Resume an exact checkpoint and apply one local-time additive pulse.

    ``amplitude=0`` is the sham and must remain byte-identical to
    ``run_fcxr_loop``.  Positive/negative arms started from clones of the same
    checkpoint use the checkpoint RNG state, providing common random numbers.
    """

    c = _constants(p, net)
    ne, ni, n, m, dt = c["ne"], c["ni"], c["n"], c["m"], c["dt"]
    labels, is_e = c["labels"], c["is_e"]
    validate_loop_state(start, n=n, ne=ne, max_delay_steps=m - 1)
    slow = start.slow
    _assert_registered_path(slow)
    pattern = validate_current_pattern(current_pattern, ne)
    amplitude = float(amplitude)
    if not np.isfinite(amplitude):
        raise ValueError("amplitude must be finite")
    if not isinstance(pulse_steps, (int, np.integer)) or int(pulse_steps) < 0:
        raise ValueError("pulse_steps must be a non-negative integer")
    if not isinstance(pulse_start_step, (int, np.integer)) or int(pulse_start_step) < 0:
        raise ValueError("pulse_start_step must be a non-negative integer")
    pulse_steps = int(pulse_steps)
    pulse_start_step = int(pulse_start_step)

    rng = net["rng"]
    rng.bit_generator.state = copy.deepcopy(start.rng_state)
    a_indptr, a_dst, a_dly, a_w = net["ampa_flat"]
    g_indptr, g_dst, g_dly, g_w = net["gaba_flat"]
    base_vth = p.V_th if v_th_per_neuron is None else np.asarray(v_th_per_neuron, dtype=float)
    t0 = int(start.t)
    V = start.V.copy(); ref = start.ref.copy()
    s_E = start.s_E.copy(); I_E = start.I_E.copy()
    s_I = start.s_I.copy(); I_I = start.I_I.copy()
    s_E_rec = start.s_E_rec.copy(); I_E_rec = start.I_E_rec.copy()
    ring_sE = start.ring_sE.copy(); ring_sI = start.ring_sI.copy()
    xi = float(start.xi)

    rate_e = np.zeros(int(n_steps), dtype=float)
    rate_i = np.zeros(int(n_steps), dtype=float)
    e_spikes = np.zeros((int(n_steps), ne), dtype=bool) if store_spikes else None

    for k in range(int(n_steps)):
        t = t0 + k
        xi = c["ou_a"] * xi + c["ou_b"] * rng.standard_normal()
        nu_now = max(c["nu_sig_const"] + xi, 0.0)

        s_E *= c["decay_s_e"]
        s_I *= c["decay_s_i"]
        slot = t % m
        s_E_rec *= c["decay_s_e"]
        s_E_rec += ring_sE[slot]
        s_E += ring_sE[slot]; ring_sE[slot] = 0.0
        s_I += ring_sI[slot]; ring_sI[slot] = 0.0
        ext = rng.poisson(np.full(n, nu_now) * dt, size=n).astype(np.float64)
        s_E += ext * c["ext_incr"]

        I_E = s_E + (I_E - s_E) * c["decay_i_e"]
        I_I = s_I + (I_I - s_I) * c["decay_i_i"]
        I_E_rec = s_E_rec + (I_E_rec - s_E_rec) * c["decay_i_e"]

        drive, g_rel, g_rev = slow.membrane_terms(I_E, I_I, labels, I_E_rec=I_E_rec)
        if amplitude != 0.0 and pulse_start_step <= k < pulse_start_step + pulse_steps:
            drive = np.asarray(drive).copy()
            drive[:ne] += amplitude * pattern
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
        n_steps=int(n_steps), t0=t0, checkpoint=checkpoint,
        pulse_accounting=current_accounting(
            pattern, amplitude=amplitude, duration_ms=pulse_steps * dt),
    )
