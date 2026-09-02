"""Untrained control states evaluated at the same causal query times as ``S``.

* ``times_only``: leaky integrators of unit impulses at event times (no marks);
* ``linear_marked_ema``: linear EWMA of the standardised event mark vector;
* ``random_reservoir``: fixed random leaky bank driven by the marks, no training.

All three evolve in real seconds, reset at segment start, and never read the
event at the query time itself.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .history import HistoryInputs, _decay_to_query, _ewma_after_each_event, _last_event_before

TIMES_ONLY_TAUS = (300.0, 900.0, 1800.0, 3600.0, 7200.0, 21600.0)


def _leaky_impulse_bank(inputs: HistoryInputs, taus: Sequence[float]):
    """First- and second-order leaky counts after each event, per tau."""

    n = inputs.event_times.size
    first = np.zeros((n, len(taus)))
    second = np.zeros((n, len(taus)))
    acc1 = np.zeros(len(taus))
    acc2 = np.zeros(len(taus))
    prev_t = None
    prev_seg = -1
    tau_arr = np.asarray(taus, dtype=np.float64)
    for i in range(n):
        seg = int(inputs.event_segment[i])
        t = float(inputs.event_times[i])
        if seg != prev_seg or prev_t is None:
            acc1 = np.zeros(len(taus))
            acc2 = np.zeros(len(taus))
        else:
            dt = t - prev_t
            decay = np.exp(-dt / tau_arr)
            # exact solution of a two-stage leaky cascade between events
            acc2 = decay * (acc2 + acc1 * dt / tau_arr)
            acc1 = acc1 * decay
        acc1 = acc1 + 1.0
        first[i] = acc1
        second[i] = acc2
        prev_t = t
        prev_seg = seg
    return first, second


def times_only_state(inputs: HistoryInputs, query_times: np.ndarray, query_segment: np.ndarray,
                     taus: Sequence[float] = TIMES_ONLY_TAUS) -> tuple[np.ndarray, list[str]]:
    q = np.asarray(query_times, dtype=np.float64)
    seg = np.asarray(query_segment, dtype=np.int64)
    first, second = _leaky_impulse_bank(inputs, taus)
    last = _last_event_before(inputs, q, seg)
    t = np.asarray(inputs.event_times, dtype=np.float64)
    cols = []
    names = []
    tau_arr = np.asarray(taus, dtype=np.float64)
    has = last >= 0
    dt = np.zeros(q.size)
    dt[has] = q[has] - t[last[has]]
    for j, tau in enumerate(tau_arr):
        decay = np.exp(-dt / tau)
        f1 = np.where(has, first[np.clip(last, 0, None), j] * decay, 0.0)
        f2 = np.where(has, decay * (second[np.clip(last, 0, None), j] + first[np.clip(last, 0, None), j] * dt / tau), 0.0)
        cols += [np.log1p(f1), np.log1p(f2)]
        names += [f"times_only_first_tau{int(tau)}", f"times_only_second_tau{int(tau)}"]
    return np.column_stack(cols), names


def linear_marked_ema(inputs: HistoryInputs, query_times: np.ndarray, query_segment: np.ndarray,
                      taus: Sequence[float]) -> tuple[np.ndarray, list[str]]:
    q = np.asarray(query_times, dtype=np.float64)
    seg = np.asarray(query_segment, dtype=np.int64)
    last = _last_event_before(inputs, q, seg)
    t = np.asarray(inputs.event_times, dtype=np.float64)
    cols = []
    names = []
    for tau in taus:
        num, den = _ewma_after_each_event(inputs, inputs.mark_continuous, float(tau))
        num_q = _decay_to_query(num, t, last, q, float(tau))
        den_q = _decay_to_query(den[:, None], t, last, q, float(tau))[:, 0]
        cols.append(num_q / (den_q + 1e-3)[:, None])
        names += [f"linear_ema:{n}_tau{int(tau)}" for n in inputs.mark_names]
    return np.concatenate(cols, axis=1), names


def random_reservoir_state(
    inputs: HistoryInputs,
    query_times: np.ndarray,
    query_segment: np.ndarray,
    *,
    dim: int = 12,
    taus: Sequence[float] = (300.0, 1800.0, 7200.0),
    seed: int = 20260902,
    update_fraction: float = 0.2,
) -> tuple[np.ndarray, list[str]]:
    """Fixed random leaky bank driven by event marks; weights never trained."""

    rng = np.random.default_rng(seed)
    d_in = int(inputs.mark_continuous.shape[1])
    w_in = rng.normal(scale=1.0 / np.sqrt(max(d_in, 1)), size=(dim, d_in))
    w_rec = rng.normal(scale=1.0 / np.sqrt(dim), size=(dim, dim))
    bias = rng.normal(scale=0.1, size=dim)
    tau_arr = np.repeat(np.asarray(taus, dtype=np.float64), int(np.ceil(dim / len(taus))))[:dim]
    n = inputs.event_times.size
    post = np.zeros((n, dim))
    state = np.zeros(dim)
    prev_t = None
    prev_seg = -1
    marks = np.nan_to_num(np.asarray(inputs.mark_continuous, dtype=np.float64))
    for i in range(n):
        seg = int(inputs.event_segment[i])
        t = float(inputs.event_times[i])
        if seg != prev_seg or prev_t is None:
            state = np.zeros(dim)
        else:
            state = state * np.exp(-(t - prev_t) / tau_arr)
        candidate = np.tanh(w_in @ marks[i] + w_rec @ state + bias)
        state = state + update_fraction * (candidate - state)
        post[i] = state
        prev_t = t
        prev_seg = seg
    q = np.asarray(query_times, dtype=np.float64)
    seg_q = np.asarray(query_segment, dtype=np.int64)
    last = _last_event_before(inputs, q, seg_q)
    out = np.zeros((q.size, dim))
    has = last >= 0
    dt = q[has] - np.asarray(inputs.event_times)[last[has]]
    out[has] = post[last[has]] * np.exp(-dt[:, None] / tau_arr[None, :])
    names = [f"reservoir[{j}]_tau{int(tau_arr[j])}" for j in range(dim)]
    return out, names
