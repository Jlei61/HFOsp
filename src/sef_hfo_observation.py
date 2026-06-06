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
