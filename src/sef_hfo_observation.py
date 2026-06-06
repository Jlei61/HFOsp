# src/sef_hfo_observation.py
"""Virtual-SEEG observation layer (Increment 1, spec 2026-06-06).

Small pure functions: montage -> envelope sampler -> lag extractor -> direction
estimators -> legacy-key artifact writer/validator. Model adapters live next to
the models, NOT here. No model dynamics in this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VirtualMontage:
    contacts: np.ndarray   # (n_contact, 2) coords in model frame, mm
    names: list            # length n_contact
    provenance: str

    def spans_2d(self, tol: float = 1e-6) -> bool:
        c = self.contacts - self.contacts.mean(axis=0, keepdims=True)
        return int(np.linalg.matrix_rank(c, tol=tol)) >= 2


def build_shaft(angle_rad, pitch, n_contacts, origin=(0.0, 0.0),
                name_prefix="A") -> VirtualMontage:
    """One linear shaft: n_contacts evenly spaced (pitch mm) along angle_rad,
    centered on origin. Contacts named <prefix>0..<prefix>(n-1)."""
    d = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    offs = (np.arange(n_contacts) - (n_contacts - 1) / 2.0) * pitch
    contacts = np.asarray(origin, float)[None, :] + offs[:, None] * d[None, :]
    names = [f"{name_prefix}{i}" for i in range(n_contacts)]
    return VirtualMontage(contacts, names, provenance="parametric_shaft")


def merge_montages(montages) -> VirtualMontage:
    """Combine shafts into one montage (the ≥2-non-parallel-shaft 2D read-out)."""
    contacts = np.vstack([m.contacts for m in montages])
    names = [nm for m in montages for nm in m.names]
    return VirtualMontage(contacts, names, provenance="parametric_multi_shaft")


def from_real_geometry(*args, **kwargs):
    """Layer-2 stub (spec §7): 3D real SEEG coords -> 2D model frame. Loud-fail
    until the per-patient heterogeneity round builds it."""
    raise NotImplementedError(
        "real-geometry montage (3D->2D registration) is layer 2; see spec §7")


def grid_coords(n: int, L: float) -> np.ndarray:
    """Flattened (n*n, 2) mm coords matching src.sef_hfo_field._grid (ij-indexed,
    centered, spacing L/n). Row-major .ravel() order aligns with a field reshaped
    via field.reshape(n_time, -1)."""
    x = (np.arange(n) - n // 2) * (L / n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel()])


def sample_envelopes(source_frames, grid_xy, montage, kernel_width,
                     Rr=None) -> np.ndarray:
    """Per-contact activity envelope = distance-weighted average of source over
    nearby grid pixels (generalizes engine/lfp.py to arbitrary contact coords).

    source_frames : (n_time, n_pix) — a field time series flattened per frame.
    grid_xy       : (n_pix, 2) mm coords (from grid_coords).
    Gaussian footprint exp(-d^2 / 2 kernel_width^2), normalized per contact.
    Optional Rr (mm) hard cutoff; None = all pixels.
    Returns (n_contact, n_time).
    """
    frames = np.asarray(source_frames, float)
    out = np.empty((len(montage.contacts), frames.shape[0]))
    for ci, c in enumerate(montage.contacts):
        d = np.linalg.norm(grid_xy - c[None, :], axis=1)
        mask = np.ones(d.shape, bool) if Rr is None else (d <= Rr)
        dd = d[mask]
        w = np.exp(-(dd ** 2) / (2.0 * kernel_width ** 2))
        w = w / max(w.sum(), 1e-12)
        out[ci] = frames[:, mask] @ w
    return out
