"""M3 degree-normalized threshold transform (spec §5.2 surgery 2, plan Task 3).

Homeostatic / degree-normalized threshold theta_i = theta0 + alpha * g_deg(i), applied
to EVERY E cell as a pure pre-transform on the per-neuron threshold vector — ZERO
``simulate_kick`` change (it rides the existing ``V_th_per_neuron``). Hub cells, carrying
the extra long-range broadcast edges, have the highest degree and so receive the largest
threshold bump (mechanism-driven, not hand-picked) — this is what makes the hub the
high-threshold gate.

THREE pre-registered schemes (审阅 P1 — no single primary; chosen at plan Task 8):
  out_strength : g_deg(i) = total E->E weight cell i sends as a SOURCE (column sum of the
                 E->E matrix). "A broadcaster is hard to ignite" — protects the hub source.
  in_strength  : g_deg(i) = total E->E weight cell i receives as a TARGET (row sum). The
                 homeostatic / input-normalized reading (closer to the cited critical-init
                 paper's mean-subtraction); also protects high-input global-region targets.
  hybrid       : elementwise max of the two (each median-normalized first).
All schemes are median-normalized (g_deg ~ 1 for a typical cell, > 1 for hubs), so alpha
is the mV bump for a median-strength cell and is comparable across schemes. The constant
~alpha global component on all E cells is absorbed by the operating-point re-tune (plan
Task 10); the hub-vs-corridor DIFFERENTIAL is the gating quantity.
"""
from __future__ import annotations
import numpy as np


def _ee_matrix(net, NE):
    """Summed E->E AMPA submatrix [target, source] from net['ampa_by_delay']."""
    A = sum(m.tocsr() for m in net["ampa_by_delay"])
    return A[:NE, :NE]


def _median_norm(x):
    pos = x[x > 0]
    m = np.median(pos) if pos.size else 1.0
    return x / max(float(m), 1e-9)


def ee_degree(net, NE, scheme="out_strength"):
    """Per-E-cell degree measure (length NE), median-normalized (~1 typical, >1 hubs)."""
    W = _ee_matrix(net, NE)
    out_s = np.asarray(W.sum(axis=0)).ravel()   # column sums: cell as SOURCE
    in_s = np.asarray(W.sum(axis=1)).ravel()    # row sums:    cell as TARGET
    if scheme == "out_strength":
        return _median_norm(out_s)
    if scheme == "in_strength":
        return _median_norm(in_s)
    if scheme == "hybrid":
        return np.maximum(_median_norm(out_s), _median_norm(in_s))
    raise ValueError(f"unknown degnorm scheme {scheme!r} (out_strength|in_strength|hybrid)")


def degnorm_vth_delta(net, NE, NI, alpha, scheme="out_strength"):
    """Length-(NE+NI) threshold-delta vector: E cells get alpha*ee_degree, I cells get 0.
    Add to V_th_per_neuron BEFORE simulate_kick. alpha=0 -> all-zero (no-op)."""
    delta = np.zeros(int(NE) + int(NI))
    if alpha != 0.0:
        delta[:NE] = float(alpha) * ee_degree(net, NE, scheme)
    return delta
