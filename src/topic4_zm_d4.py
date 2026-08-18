"""Covariant square-symmetry transforms of the data-driven substrate.

Rotating the node field alone is NOT a rigid transform of this substrate: the
local connectivity mapper's last two features are signed and linear in the
source-minus-target displacement (src/topic4_local_connectivity.py:50-62), so a
field-only rotation reverses the correspondence between field structure and the
flow it drives. Rotating the two flow coefficients by the same matrix restores
it, and because the group elements only swap and negate components, the frozen
coefficient bounds survive element-wise with no re-clipping.

Scope: this makes the field-and-flow RULE a rigid image of the original. It does
NOT make the substrate an isometric copy -- the realized random graph, its
patient-derived anisotropic topology and the contacts stay fixed. The construct
is a matched spatial re-registration control, not an isometry.
"""
from __future__ import annotations

import numpy as np

D4_ELEMENTS = ("r90", "r180", "r270", "mx", "my", "md1", "md2")

CONTROL_NAME = "matched spatial re-registration control"

_MATRICES = {
    "r90": np.array([[0.0, -1.0], [1.0, 0.0]]),
    "r180": np.array([[-1.0, 0.0], [0.0, -1.0]]),
    "r270": np.array([[0.0, 1.0], [-1.0, 0.0]]),
    "mx": np.array([[1.0, 0.0], [0.0, -1.0]]),
    "my": np.array([[-1.0, 0.0], [0.0, 1.0]]),
    "md1": np.array([[0.0, 1.0], [1.0, 0.0]]),
    "md2": np.array([[0.0, -1.0], [-1.0, 0.0]]),
}


def d4_matrix(element):
    if element not in _MATRICES:
        raise ValueError(f"unknown D4 element {element!r}")
    return _MATRICES[element].copy()


def inverse_query_positions(positions, element, *, L):
    """Where to evaluate the ORIGINAL spline so the field appears transformed."""
    positions = np.asarray(positions, float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (n, 2)")
    centre = float(L) / 2.0
    inverse = np.linalg.inv(d4_matrix(element))
    return (inverse @ (positions - centre).T).T + centre


def transform_flow_coefficients(coefficients, element):
    """Rotate only the two directed-flow components, by the SAME matrix."""
    coefficients = np.asarray(coefficients, float)
    if coefficients.ndim != 2 or coefficients.shape[1] != 6:
        raise ValueError("coefficients must have shape (pathways, 6)")
    matrix = d4_matrix(element)
    out = coefficients.copy()
    out[:, 4:] = (matrix @ coefficients[:, 4:].T).T
    return out


def transform_report(element, coefficients, *, axis_unit):
    """Provenance row for the manifest.

    ``preserves_undirected_axis`` and ``preserves_directed_axis`` are reported
    separately because r180 keeps the field aligned with the patient's
    undirected propagation axis while reversing its directed sense -- the
    field's source end lands on the patient's sink end. That makes r180 a
    substantive probe, not a filler control, and it must not be described as an
    electrode-alignment-only comparison.
    """
    matrix = d4_matrix(element)
    axis_unit = np.asarray(axis_unit, float)
    mapped = matrix @ axis_unit
    aligned = float(np.dot(mapped, axis_unit))
    return {
        "element": element,
        "name": CONTROL_NAME,
        "matrix": matrix.tolist(),
        "preserves_undirected_axis": bool(np.isclose(abs(aligned), 1.0, atol=1e-12)),
        "preserves_directed_axis": bool(np.isclose(aligned, 1.0, atol=1e-12)),
        "axis_alignment_cosine": aligned,
        "coefficients_before": np.asarray(coefficients, float).tolist(),
        "coefficients_after": transform_flow_coefficients(coefficients, element).tolist(),
        "boundary": ("field-and-flow rule transformed as a rigid unit; realized graph, "
                     "patient-derived anisotropy and contacts held fixed. Not an "
                     "isometric copy of the substrate."),
    }
