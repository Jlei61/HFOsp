"""Pure contracts for the FCXR-LC4b exact-dead-zone cooperative actuator."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


TAU_ADP_MS = 1000.0
HILL_N = 4.0
TAU_A_ON_MS = 100.0
TAU_A_OFF_MS = 10000.0
TARGET_CURRENT = 44.8619393917937


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def deadzone_activation(load, *, deadzone: float, excess_scale: float,
                        n: float = HILL_N) -> np.ndarray:
    """Exact-zero below ``deadzone`` and Hill activation of the positive excess above it."""
    a = np.asarray(load, dtype=float)
    if not (np.all(np.isfinite(a)) and np.isfinite(deadzone) and deadzone >= 0.0):
        raise ValueError("load and deadzone must be finite; deadzone must be non-negative")
    if not (np.isfinite(excess_scale) and excess_scale > 0.0):
        raise ValueError("excess_scale must be finite and positive")
    if not (np.isfinite(n) and n > 0.0):
        raise ValueError("n must be finite and positive")
    u = np.maximum(a - float(deadzone), 0.0)
    x = (u / float(excess_scale)) ** float(n)
    return x / (1.0 + x)


def build_locked_candidate(interictal_peak, ictal_settled,
                           *, target_current: float = TARGET_CURRENT) -> dict:
    """Apply the pre-registered midpoint/median rules; there is no searched parameter here."""
    quiet = np.asarray(interictal_peak, dtype=float)
    ictal = np.asarray(ictal_settled, dtype=float)
    if quiet.ndim != 1 or ictal.ndim != 1 or not quiet.size or not ictal.size:
        raise ValueError("interictal_peak and ictal_settled must be non-empty vectors")
    if not (np.all(np.isfinite(quiet)) and np.all(np.isfinite(ictal))):
        raise ValueError("load arrays must be finite")
    quiet_max = float(np.max(quiet))
    ictal_min = float(np.min(ictal))
    ictal_median = float(np.median(ictal))
    if not quiet_max < ictal_min:
        raise ValueError("DEADZONE_NOT_IDENTIFIABLE: interictal/ictal load extremes overlap")
    deadzone = 0.5 * (quiet_max + ictal_min)
    excess_scale = ictal_median - deadzone
    if excess_scale <= 0.0:
        raise ValueError("DEADZONE_NOT_IDENTIFIABLE: ictal median is not above the dead zone")
    a_quiet = deadzone_activation(quiet, deadzone=deadzone, excess_scale=excess_scale)
    a_ictal = deadzone_activation(ictal, deadzone=deadzone, excess_scale=excess_scale)
    mean_ictal = float(np.mean(a_ictal))
    if not (np.all(a_quiet == 0.0) and mean_ictal >= 0.20):
        raise ValueError("DEADZONE_NOT_IDENTIFIABLE: exact-zero/ictal-activation gate failed")
    g_max = float(target_current) / mean_ictal
    if not np.isfinite(g_max):
        raise ValueError("DEADZONE_NOT_IDENTIFIABLE: non-finite force match")
    return {
        "name": "hill_n4_exact_deadzone_slow",
        "n": HILL_N,
        "tau_adp_ms": TAU_ADP_MS,
        "tau_a_on_ms": TAU_A_ON_MS,
        "tau_a_off_ms": TAU_A_OFF_MS,
        "deadzone": deadzone,
        "K": excess_scale,
        "g_m_max": g_max,
        "matched_ictal_current": float(target_current),
        "calibration": {
            "interictal_peak_max": quiet_max,
            "ictal_settled_min": ictal_min,
            "ictal_settled_median": ictal_median,
            "extreme_gap": ictal_min - quiet_max,
            "deadzone_margin_each_side": ictal_min - deadzone,
            "interictal_activation_max": float(np.max(a_quiet)),
            "interictal_activation_mean": float(np.mean(a_quiet)),
            "ictal_activation_min": float(np.min(a_ictal)),
            "ictal_activation_mean": mean_ictal,
            "ictal_fraction_half_on": float(np.mean(a_ictal >= 0.5)),
        },
    }
