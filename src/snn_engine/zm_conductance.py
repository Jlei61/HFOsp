"""Unit-safe Phase-D conductance membrane for the per-neuron Z/M SNN.

This module is deliberately pure: it does not know about spike scattering,
random streams, checkpoints, or the slow-field implementation.  The guarded
SNN loop may call it only after its math/sign contracts pass.

Model-coordinate units use ``g_L=1`` and ``C=tau_m_E*g_L``.  ``I_E`` and
``I_I`` are the existing non-negative mV-equivalent synaptic drives; kappa
converts them to dimensionless conductances.  They are never inserted directly
into a conductance denominator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ZMConductanceConfig:
    """Locked conductance parameters for one Phase-D arm."""

    kappa_E: float = 0.1
    kappa_I: float = 0.25
    g_M: float = 0.001 / 15.0
    gamma: float = 0.0
    z_spares_global: bool = False
    g_L: float = 1.0
    E_L: float = 0.0
    E_E: float = 25.0
    E_I: float = 11.0
    E_K: float = 0.0
    tau_m_E: float = 20.0

    def validate(self):
        for name in ("kappa_E", "kappa_I", "g_L", "tau_m_E"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0, got {value}")
        if not np.isfinite(self.g_M) or self.g_M < 0.0:
            raise ValueError(f"g_M must be finite and >= 0, got {self.g_M}")
        if not np.isfinite(self.gamma) or not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"gamma must be in [0,1], got {self.gamma}")
        for name in ("E_L", "E_E", "E_I", "E_K"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not self.E_E > self.E_I >= self.E_K:
            raise ValueError(
                "reversals must satisfy E_E > E_I >= E_K, got "
                f"{self.E_E}, {self.E_I}, {self.E_K}"
            )
        return self


def analytic_anchor(
    *,
    V_ref,
    V_th_median,
    V_reset,
    eta_m,
    gamma=0.0,
    z_spares_global=False,
    scale_E=1.0,
    scale_I=1.0,
    scale_M=1.0,
    E_L=0.0,
    tau_m_E=20.0,
):
    """Return the baseline-only current-tangent conductance anchor.

    At ``V_ref`` the E-cell conductance RHS equals the current-based RHS
    ``-V + I_E - z*I_I - eta_m*m`` when gamma is evaluated on a spatially
    uniform state.
    """
    vals = {
        "V_ref": V_ref,
        "V_th_median": V_th_median,
        "V_reset": V_reset,
        "eta_m": eta_m,
        "gamma": gamma,
        "scale_E": scale_E,
        "scale_I": scale_I,
        "scale_M": scale_M,
        "E_L": E_L,
        "tau_m_E": tau_m_E,
    }
    for name, value in vals.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    for name, value in {
        "scale_E": scale_E,
        "scale_I": scale_I,
        "scale_M": scale_M,
    }.items():
        if not 0.8 <= float(value) <= 1.2:
            raise ValueError(f"{name} must be in [0.8,1.2], got {value}")
    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError(f"gamma must be in [0,1], got {gamma}")
    if float(eta_m) < 0.0:
        raise ValueError(f"eta_m must be >= 0, got {eta_m}")

    E_E = 2.0 * float(V_th_median) - float(V_reset)
    E_I = float(V_reset)
    E_K = float(E_L)
    V_ref = float(V_ref)
    if not E_I < V_ref < E_E:
        raise ValueError(
            f"V_ref must satisfy E_I < V_ref < E_E, got "
            f"{E_I} < {V_ref} < {E_E}"
        )
    if V_ref <= E_K:
        raise ValueError(f"V_ref must be > E_K, got {V_ref} <= {E_K}")

    cfg = ZMConductanceConfig(
        kappa_E=float(scale_E) / (E_E - V_ref),
        kappa_I=float(scale_I) / (V_ref - E_I),
        g_M=float(scale_M) * float(eta_m) / (V_ref - E_K),
        gamma=float(gamma),
        z_spares_global=bool(z_spares_global),
        g_L=1.0,
        E_L=float(E_L),
        E_E=E_E,
        E_I=E_I,
        E_K=E_K,
        tau_m_E=float(tau_m_E),
    )
    return cfg.validate()


def distribution_magnitude_anchor(
    *,
    V_free,
    V_th_median,
    V_reset,
    eta_m,
    gamma=0.0,
    z_spares_global=False,
    scale_E=1.0,
    scale_I=1.0,
    scale_M=1.0,
    E_L=0.0,
    tau_m_E=20.0,
):
    """Positive baseline anchor when the voltage distribution crosses reversals.

    The historic current model can drive free cells below ``E_I`` and ``E_K``.
    A signed point-tangent there would require a negative conductance.  This
    anchor instead matches the median *magnitude* of each reversal driving
    force on the locked baseline distribution.  It does not claim pointwise
    sign equivalence; the returned diagnostics make that limitation explicit
    and the empirical baseline-preservation gate remains decisive.
    """
    voltage = np.asarray(V_free, dtype=float)
    if voltage.ndim != 1 or voltage.size < 2:
        raise ValueError("V_free must be a one-dimensional baseline sample")
    if not np.all(np.isfinite(voltage)):
        raise ValueError("V_free must be finite")
    for name, value in {
        "V_th_median": V_th_median,
        "V_reset": V_reset,
        "eta_m": eta_m,
        "gamma": gamma,
        "scale_E": scale_E,
        "scale_I": scale_I,
        "scale_M": scale_M,
        "E_L": E_L,
        "tau_m_E": tau_m_E,
    }.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    for name, value in {
        "scale_E": scale_E,
        "scale_I": scale_I,
        "scale_M": scale_M,
    }.items():
        if not 0.8 <= float(value) <= 1.2:
            raise ValueError(f"{name} must be in [0.8,1.2], got {value}")
    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError(f"gamma must be in [0,1], got {gamma}")
    if float(eta_m) < 0.0:
        raise ValueError(f"eta_m must be >= 0, got {eta_m}")

    E_E = 2.0 * float(V_th_median) - float(V_reset)
    E_I = float(V_reset)
    E_K = float(E_L)
    d_E = float(np.median(E_E - voltage))
    d_I = float(np.median(np.abs(voltage - E_I)))
    d_M = float(np.median(np.abs(voltage - E_K)))
    if min(d_E, d_I, d_M) <= 0.0:
        raise ValueError(
            f"baseline driving-force magnitudes must be >0, got {d_E}, {d_I}, {d_M}"
        )
    if np.any(voltage >= E_E):
        raise ValueError("baseline V_free reaches/exceeds the excitatory reversal")

    cfg = ZMConductanceConfig(
        kappa_E=float(scale_E) / d_E,
        kappa_I=float(scale_I) / d_I,
        g_M=float(scale_M) * float(eta_m) / d_M,
        gamma=float(gamma),
        z_spares_global=bool(z_spares_global),
        g_L=1.0,
        E_L=float(E_L),
        E_E=E_E,
        E_I=E_I,
        E_K=E_K,
        tau_m_E=float(tau_m_E),
    ).validate()
    diagnostics = {
        "definition": "median_baseline_reversal_driving_force_magnitude",
        "n_free_e": int(voltage.size),
        "V_free_percentiles_mv": {
            str(q): float(np.percentile(voltage, q))
            for q in (5, 25, 50, 75, 95)
        },
        "driving_force_median_mv": {"E": d_E, "I": d_I, "M": d_M},
        "fraction_V_above_EI": float(np.mean(voltage > E_I)),
        "fraction_V_above_EK": float(np.mean(voltage > E_K)),
        "signed_point_tangent_feasible_at_median": bool(
            float(np.median(voltage)) > E_I
        ),
        "pointwise_sign_equivalence_claimed": False,
    }
    return cfg, diagnostics


def _vectors(*values):
    arrays = [np.asarray(value) for value in values]
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError(
            f"all vectors must share a shape, got {[a.shape for a in arrays]}"
        )
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("conductance inputs must be one-dimensional")
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("conductance inputs must be finite")
    return arrays


def decompose_conductances(I_E, I_I, z, m, cfg, is_E):
    """Map raw filtered drives and Z/M coordinates to model conductances."""
    cfg.validate()
    I_E, I_I, z, m, is_E = _vectors(I_E, I_I, z, m, is_E)
    is_E = is_E.astype(bool, copy=False)
    if not np.any(is_E):
        raise ValueError("at least one E cell is required")
    if np.any((z[is_E] < 0.0) | (z[is_E] > 1.0)):
        raise ValueError("E-cell z must lie in [0,1]")
    if np.any(m[is_E] < 0.0):
        raise ValueError("E-cell m must be >= 0")

    g_E = np.zeros(I_E.shape, dtype=float)
    g_I_local = np.zeros(I_I.shape, dtype=float)
    g_Mm = np.zeros(m.shape, dtype=float)
    g_E[is_E] = cfg.kappa_E * np.maximum(I_E[is_E], 0.0)
    g_I_local[is_E] = cfg.kappa_I * np.maximum(I_I[is_E], 0.0)
    g_Mm[is_E] = cfg.g_M * m[is_E]
    g_I_global = float(np.mean(g_I_local[is_E]))

    g_I_eff = np.zeros(I_I.shape, dtype=float)
    if cfg.z_spares_global:
        g_I_eff[is_E] = (
            (1.0 - cfg.gamma) * z[is_E] * g_I_local[is_E]
            + cfg.gamma * g_I_global
        )
    else:
        g_I_eff[is_E] = z[is_E] * (
            (1.0 - cfg.gamma) * g_I_local[is_E]
            + cfg.gamma * g_I_global
        )
    return {
        "g_E": g_E,
        "g_I_local": g_I_local,
        "g_I_global": g_I_global,
        "g_I_eff": g_I_eff,
        "g_Mm": g_Mm,
    }


def conductance_currents(V, g_E, g_I_eff, g_Mm, cfg, is_E):
    """Return separated E-cell synaptic and sAHP currents."""
    cfg.validate()
    V, g_E, g_I_eff, g_Mm, is_E = _vectors(
        V, g_E, g_I_eff, g_Mm, is_E
    )
    is_E = is_E.astype(bool, copy=False)
    if np.any(g_E < 0.0) or np.any(g_I_eff < 0.0) or np.any(g_Mm < 0.0):
        raise ValueError("conductances must be non-negative")
    I_exc = np.zeros(V.shape, dtype=float)
    I_inh = np.zeros(V.shape, dtype=float)
    I_sahp = np.zeros(V.shape, dtype=float)
    I_exc[is_E] = g_E[is_E] * (cfg.E_E - V[is_E])
    I_inh[is_E] = g_I_eff[is_E] * (cfg.E_I - V[is_E])
    I_sahp[is_E] = g_Mm[is_E] * (cfg.E_K - V[is_E])
    return {"I_exc": I_exc, "I_inh": I_inh, "I_sahp": I_sahp}


def conductance_membrane_step(
    V,
    I_E,
    I_I,
    z,
    m,
    decay_V,
    is_E,
    cfg,
):
    """Exact E-cell conductance update plus literal current-based I update."""
    cfg.validate()
    V, I_E, I_I, z, m, decay_V, is_E = _vectors(
        V, I_E, I_I, z, m, decay_V, is_E
    )
    is_E = is_E.astype(bool, copy=False)
    if np.any((decay_V <= 0.0) | (decay_V > 1.0)):
        raise ValueError("decay_V must lie in (0,1]")

    dec = decompose_conductances(I_E, I_I, z, m, cfg, is_E)
    cur = conductance_currents(
        V, dec["g_E"], dec["g_I_eff"], dec["g_Mm"], cfg, is_E
    )
    g_sigma = np.ones(V.shape, dtype=float)
    g_sigma[is_E] = (
        cfg.g_L
        + dec["g_E"][is_E]
        + dec["g_I_eff"][is_E]
        + dec["g_Mm"][is_E]
    )
    V_inf = np.empty(V.shape, dtype=float)
    V_inf[is_E] = (
        cfg.g_L * cfg.E_L
        + dec["g_E"][is_E] * cfg.E_E
        + dec["g_I_eff"][is_E] * cfg.E_I
        + dec["g_Mm"][is_E] * cfg.E_K
    ) / g_sigma[is_E]

    current_target = I_E - I_I
    V_inf[~is_E] = current_target[~is_E]
    V_next = current_target + (V - current_target) * decay_V
    V_next[is_E] = V_inf[is_E] + (
        V[is_E] - V_inf[is_E]
    ) * np.power(decay_V[is_E], g_sigma[is_E])

    tau_eff_ms = np.full(V.shape, np.nan, dtype=float)
    tau_eff_ms[is_E] = cfg.tau_m_E / g_sigma[is_E]
    return {
        "V_next": V_next,
        "V_inf": V_inf,
        "tau_eff_ms": tau_eff_ms,
        "g_sigma": g_sigma,
        **dec,
        **cur,
    }


__all__ = [
    "ZMConductanceConfig",
    "analytic_anchor",
    "decompose_conductances",
    "conductance_currents",
    "conductance_membrane_step",
]
