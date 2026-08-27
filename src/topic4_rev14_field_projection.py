"""Project an observation-free rev14 Fourier field onto the frozen Node map.

The Fourier field is the complete latent support field, not a residual around a
historical contact-fitted field.  Only the signed per-neuron depth, total support
mass and engine constants are inherited from the frozen Node contract.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence

import numpy as np

from src.topic4_core_field import EPS, TAU_H, project_to_budget
from src.topic4_rev14_fourier_field import (
    DEFAULT_SHEET_LENGTH_MM,
    array_sha256,
    evaluate,
    uniform_quadrature,
    weighted_centered_surface_rms,
)


_LOG_MAX = float(np.log(np.finfo(np.float64).max))


def _finite_float(value: float, *, name: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive_float(value: float, *, name: str) -> float:
    number = _finite_float(value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _validated_positions(positions: np.ndarray, *, L: float) -> np.ndarray:
    xy = np.asarray(positions, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) == 0:
        raise ValueError("positions must have shape (n_E, 2) with n_E > 0")
    if not np.isfinite(xy).all():
        raise ValueError("positions must be finite")
    if np.any((xy < 0.0) | (xy > L)):
        raise ValueError("positions must lie inside the physical sheet [0, L]^2")
    return xy


def _validated_signed_depth(values: np.ndarray, *, n_e: int) -> np.ndarray:
    depth = np.asarray(values, dtype=np.float64)
    if depth.shape != (n_e,) or not np.isfinite(depth).all():
        raise ValueError("frozen_signed_depth must contain one finite value per E neuron")
    return depth


def _project_latent(
    latent: np.ndarray,
    *,
    target_count: float,
    tau_h: float,
    eps: float,
) -> tuple[np.ndarray, float]:
    """Use the established q-to-h operator without changing the latent gauge."""
    values = np.asarray(latent, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("latent field must be a nonempty finite vector")
    shifted = values - float(np.min(values)) + float(np.log(2.0 * eps))
    if float(np.max(shifted)) >= _LOG_MAX:
        raise ValueError("latent field dynamic range exceeds finite projection range")
    q = np.exp(shifted) - eps
    if not np.isfinite(q).all() or np.any(q <= 0.0):
        raise ValueError("latent field cannot be represented by a finite positive q")
    h, level = project_to_budget(
        q, target_count=target_count, tau_h=tau_h, eps=eps,
    )
    h = np.asarray(h, dtype=np.float64)
    if not np.isfinite(h).all() or np.any((h <= 0.0) | (h >= 1.0)):
        raise RuntimeError("mass projection returned an invalid h field")
    if not np.isclose(float(np.sum(h)), target_count, rtol=0.0, atol=1e-8):
        raise RuntimeError("mass projection did not preserve the frozen h budget")
    return h, float(level)


def _rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("RMS input must be finite")
    scale = float(np.max(np.abs(array), initial=0.0))
    if scale == 0.0:
        return 0.0
    return float(scale * np.sqrt(np.mean((array / scale) ** 2)))


def _contract_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def project_fourier_to_frozen_node(
    coefficients: np.ndarray,
    positions: np.ndarray,
    modes: Iterable[Sequence[int]],
    *,
    frozen_signed_depth: np.ndarray,
    n_total: int,
    target_count: float,
    v_base: float,
    tau_h: float = TAU_H,
    eps: float = EPS,
    L: float = DEFAULT_SHEET_LENGTH_MM,
    quadrature_n: int = 128,
) -> dict:
    """Map the complete Fourier latent field to ``h`` and ``Vtheta``.

    Zero coefficients produce a spatially uniform support field at the frozen
    mass.  The historical ``exact_off`` field is deliberately absent from this
    API and must be run as a separate nonselectable benchmark.
    """
    length = _positive_float(L, name="L")
    tau = _positive_float(tau_h, name="tau_h")
    epsilon = _positive_float(eps, name="eps")
    base = _finite_float(v_base, name="v_base")
    xy = _validated_positions(positions, L=length)
    depth = _validated_signed_depth(frozen_signed_depth, n_e=len(xy))
    total = int(n_total)
    if total < len(xy):
        raise ValueError("n_total cannot be smaller than n_E")
    target = _positive_float(target_count, name="target_count")
    if not target < len(xy):
        raise ValueError("target_count must lie in (0, n_E)")
    try:
        frozen_modes = tuple(tuple(mode) for mode in modes)
    except TypeError as exc:
        raise TypeError("modes must be an iterable of integer pairs") from exc

    coeff = np.asarray(coefficients, dtype=np.float64)
    latent_at_neurons = np.asarray(
        evaluate(coeff, xy, frozen_modes, L=length), dtype=np.float64,
    )
    sheet_positions, sheet_weights = uniform_quadrature(quadrature_n, L=length)
    latent_on_sheet = np.asarray(
        evaluate(coeff, sheet_positions, frozen_modes, L=length), dtype=np.float64,
    )
    if not np.isfinite(latent_at_neurons).all() or not np.isfinite(latent_on_sheet).all():
        raise ValueError("Fourier evaluation returned non-finite values")
    sheet_mean = float(np.dot(sheet_weights, latent_on_sheet) / np.sum(sheet_weights))
    centered_latent = latent_at_neurons - sheet_mean
    centered_latent_rms = weighted_centered_surface_rms(
        latent_on_sheet, sheet_weights,
    )

    h, projection_level = _project_latent(
        centered_latent, target_count=target, tau_h=tau, eps=epsilon,
    )
    vtheta = np.full(total, base, dtype=np.float64)
    delta_vtheta = -h * depth
    vtheta[:len(xy)] += delta_vtheta
    if not np.isfinite(vtheta).all():
        raise RuntimeError("projected Vtheta contains non-finite values")

    uniform_h = np.full(len(xy), target / len(xy), dtype=np.float64)
    zero_coefficients = bool(np.count_nonzero(coeff) == 0)
    zero_uniform_error = float(np.max(np.abs(h - uniform_h), initial=0.0))
    hashes = {
        "coefficients_sha256": array_sha256(coeff),
        "positions_sha256": array_sha256(xy),
        "frozen_signed_depth_sha256": array_sha256(depth),
        "latent_at_neurons_sha256": array_sha256(centered_latent),
        "h_sha256": array_sha256(h),
        "delta_vtheta_sha256": array_sha256(delta_vtheta),
        "vtheta_sha256": array_sha256(vtheta),
    }
    contract = {
        "schema_id": "topic4_rev14_fourier_frozen_node_projection_v2",
        "mapping": "observation_free_complete_fourier_field_times_frozen_signed_depth",
        "historical_field_inherited": False,
        "zero_coefficients_semantics": "uniform_node_support",
        "modes": [list(map(int, mode)) for mode in frozen_modes],
        "L_mm": length,
        "tau_h": tau,
        "eps": epsilon,
        "target_h_mass": target,
        "n_total": total,
        "quadrature_n": int(quadrature_n),
        "hashes": hashes,
    }
    hashes["projection_sha256"] = _contract_sha256(contract)

    return {
        "h": h,
        "vtheta": vtheta,
        "delta_vtheta": delta_vtheta,
        "signed_depth": depth,
        "latent_surface_at_neurons": centered_latent,
        "audit": {
            "mapping_type": contract["mapping"],
            "historical_field_inherited": False,
            "zero_coefficients": zero_coefficients,
            "zero_uniform_max_abs_error": zero_uniform_error,
            "mass_projection_level": projection_level,
            "target_h_mass": target,
            "observed_h_mass": float(np.sum(h)),
            "h_mass_error": float(np.sum(h) - target),
            "centered_latent_surface_rms": float(centered_latent_rms),
            "latent_physical_sheet_mean": sheet_mean,
            "h_rms": _rms(h),
            "h_centered_rms": _rms(h - float(np.mean(h))),
            "h_delta_from_uniform_rms": _rms(h - uniform_h),
            "threshold_modulation_rms_mV": _rms(delta_vtheta),
            "vtheta_e_rms_mV": _rms(vtheta[:len(xy)]),
            "vtheta_e_centered_rms_mV": _rms(
                vtheta[:len(xy)] - float(np.mean(vtheta[:len(xy)]))
            ),
        },
        "hashes": hashes,
    }


__all__ = ["project_fourier_to_frozen_node"]
