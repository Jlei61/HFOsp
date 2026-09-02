"""Numerical core for the frozen-RNN shared-computation necessity test.

The module has no filesystem or target-table reader.  It only implements the
pre-registered operator normalisation, leave-one-topology-out component,
projection erasure, matched displacement, rank-set NLL and patient-first
summary helpers.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch


REAL_ARMS = ("L0", "L1", "L2m", "L3")
CONTROL_FAMILIES = ("SHARED", "ORTHOGONAL", "PCA", "C_SUFFIX")
LESION_DOSES = np.asarray([0.25, 0.50, 1.00], dtype=np.float64)
HELDOUT_OUTCOME_FIELDS = frozenset({"event_u", "conditional_center"})


def drop_heldout_outcome_fields(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Return a reference-state archive with completed-suffix fields removed."""
    return {
        str(name): value
        for name, value in arrays.items()
        if str(name) not in HELDOUT_OUTCOME_FIELDS
    }


def unit_vector(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("direction is not finite and non-zero")
    return vector / norm


def sign_stabilize(value: np.ndarray) -> np.ndarray:
    """Choose a deterministic sign without using any response endpoint."""
    vector = unit_vector(value)
    pivot = int(np.argmax(np.abs(vector)))
    return vector if vector[pivot] >= 0 else -vector


def centered_normalized_operator(operator: np.ndarray) -> np.ndarray:
    """Remove unidentifiable common-logit shifts and equalise arm amplitude."""
    value = np.asarray(operator, dtype=np.float64).copy()
    if value.ndim != 2:
        raise ValueError("operator must be contact x patch")
    finite_count = np.isfinite(value).sum(axis=0, keepdims=True)
    column_mean = np.divide(
        np.nansum(value, axis=0, keepdims=True),
        finite_count,
        out=np.full((1, value.shape[1]), np.nan, dtype=np.float64),
        where=finite_count > 0,
    )
    value -= column_mean
    value[~np.isfinite(value)] = 0.0
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("operator has no finite centred variation")
    return value / norm


def leave_one_topology_component(
    operators: Mapping[str, np.ndarray],
    heldout_arm: str,
    patch_basis: np.ndarray,
    *,
    rank: int = 3,
) -> dict[str, np.ndarray]:
    """Build patch and hidden components without reading the held-out arm."""
    if heldout_arm not in REAL_ARMS:
        raise ValueError(f"unknown held-out arm: {heldout_arm}")
    sources = [arm for arm in REAL_ARMS if arm != heldout_arm]
    missing = [arm for arm in sources if arm not in operators]
    if missing:
        raise ValueError(f"missing source operators: {missing}")
    source = np.stack([centered_normalized_operator(operators[arm]) for arm in sources])
    consensus = np.median(source, axis=0)
    _, singular_values, vh = np.linalg.svd(consensus, full_matrices=False)
    keep = min(int(rank), vh.shape[0])
    patch_components = np.stack([sign_stabilize(vh[index]) for index in range(keep)])
    basis = np.asarray(patch_basis, dtype=np.float64)
    if basis.ndim != 2 or basis.shape[0] != consensus.shape[1]:
        raise ValueError("patch basis must be patch x hidden and align to the operator")
    hidden_components = np.stack([
        sign_stabilize(component @ basis) for component in patch_components
    ])
    energy = singular_values**2
    explained = energy / max(float(energy.sum()), 1e-12)
    return {
        "consensus_operator": consensus,
        "patch_components": patch_components,
        "hidden_components": hidden_components,
        "singular_values": singular_values,
        "explained_fraction": explained,
        "source_arms": np.asarray(sources),
    }


def single_operator_component(
    operator: np.ndarray, patch_basis: np.ndarray, *, rank: int = 3
) -> dict[str, np.ndarray]:
    """SVD component for the shuffled-ending control network."""
    value = centered_normalized_operator(operator)
    _, singular_values, vh = np.linalg.svd(value, full_matrices=False)
    keep = min(int(rank), vh.shape[0])
    patch_components = np.stack([sign_stabilize(vh[index]) for index in range(keep)])
    basis = np.asarray(patch_basis, dtype=np.float64)
    hidden_components = np.stack([
        sign_stabilize(component @ basis) for component in patch_components
    ])
    energy = singular_values**2
    return {
        "operator": value,
        "patch_components": patch_components,
        "hidden_components": hidden_components,
        "singular_values": singular_values,
        "explained_fraction": energy / max(float(energy.sum()), 1e-12),
    }


def projection_erasure(
    hidden: np.ndarray,
    center: np.ndarray,
    direction: np.ndarray,
    dose: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Delete a fraction of a state's component relative to its conditional centre."""
    h = np.asarray(hidden, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)
    u = unit_vector(direction)
    projection = (h - c) @ u
    delta = -float(dose) * projection[:, None] * u[None, :]
    return h + delta, delta, projection


def equal_norm_toward_center(
    hidden: np.ndarray,
    center: np.ndarray,
    directions: np.ndarray,
    displacement_norm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Move along controls by the target lesion norm, oriented toward the centre."""
    h = np.asarray(hidden, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)
    v = np.asarray(directions, dtype=np.float64)
    if v.ndim == 1:
        v = np.broadcast_to(unit_vector(v), h.shape)
    else:
        v = np.stack([unit_vector(row) for row in v])
    residual_projection = np.einsum("ij,ij->i", h - c, v)
    sign = np.where(residual_projection >= 0, -1.0, 1.0)
    delta = sign[:, None] * np.asarray(displacement_norm, float)[:, None] * v
    return h + delta, delta


def orthogonalize_rows(candidates: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Project state-specific candidate rows away from a fixed target direction."""
    values = np.asarray(candidates, dtype=np.float64)
    u = unit_vector(target)
    values = values - (values @ u)[:, None] * u[None, :]
    output = np.full_like(values, np.nan)
    for index, row in enumerate(values):
        norm = float(np.linalg.norm(row))
        if np.isfinite(norm) and norm > 1e-10:
            output[index] = row / norm
    return output


def orthonormal_row_basis(vectors: np.ndarray) -> np.ndarray:
    """Deterministic Gram-Schmidt basis for cumulative component sensitivities."""
    basis: list[np.ndarray] = []
    for candidate in np.asarray(vectors, dtype=np.float64):
        vector = candidate.copy()
        for previous in basis:
            vector -= previous * float(np.dot(previous, vector))
        norm = float(np.linalg.norm(vector))
        if np.isfinite(norm) and norm > 1e-10:
            vector /= norm
            pivot = int(np.argmax(np.abs(vector)))
            basis.append(vector if vector[pivot] >= 0 else -vector)
    if not basis:
        raise ValueError("no finite independent vector in subspace")
    return np.stack(basis)


def subspace_projection_erasure(
    hidden: np.ndarray,
    center: np.ndarray,
    basis: np.ndarray,
    dose: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Delete a fraction of the residual projected into an orthonormal subspace."""
    h = np.asarray(hidden, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)
    rows = orthonormal_row_basis(basis)
    projected = ((h - c) @ rows.T) @ rows
    delta = -float(dose) * projected
    return h + delta, delta


def equal_norm_subspace_toward_center(
    hidden: np.ndarray,
    center: np.ndarray,
    basis: np.ndarray,
    displacement_norm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Use a control subspace but exactly match the target hidden displacement."""
    h = np.asarray(hidden, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)
    rows = orthonormal_row_basis(basis)
    projected = ((h - c) @ rows.T) @ rows
    norm = np.linalg.norm(projected, axis=1)
    unit = np.divide(
        projected,
        norm[:, None],
        out=np.full_like(projected, np.nan),
        where=norm[:, None] > 1e-10,
    )
    delta = -np.asarray(displacement_norm, float)[:, None] * unit
    return h + delta, delta


def rank_set_nll(
    logits: torch.Tensor,
    target: torch.Tensor,
    available: torch.Tensor,
) -> torch.Tensor:
    """Per-state mean NLL of the observed next-rank set."""
    if logits.ndim != 2 or target.shape != logits.shape or available.shape != logits.shape:
        raise ValueError("logits, target and available must align as state x contact")
    masked = logits.masked_fill(~available.bool(), -1e9)
    log_probability = torch.log_softmax(masked, dim=-1)
    chosen = (log_probability * target.float()).sum(-1)
    denominator = target.float().sum(-1)
    result = -chosen / denominator.clamp_min(1.0)
    return torch.where(denominator > 0, result, torch.full_like(result, torch.nan))


def dose_auc(doses: Sequence[float], effects: Sequence[float]) -> float:
    """Mean loss over dose, anchored at zero dose and normalised to [0, 1]."""
    x = np.r_[0.0, np.asarray(doses, dtype=np.float64)]
    y = np.r_[0.0, np.asarray(effects, dtype=np.float64)]
    use = np.isfinite(x) & np.isfinite(y)
    if int(use.sum()) < 2:
        return float("nan")
    order = np.argsort(x[use])
    return float(np.trapz(y[use][order], x[use][order]) / (x[use][order][-1] - x[use][order][0]))


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm family-wise adjusted p values in original order."""
    p = np.asarray(p_values, dtype=np.float64)
    output = np.full_like(p, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    if not len(finite):
        return output
    order = finite[np.argsort(p[finite])]
    running = 0.0
    total = len(order)
    for rank_index, index in enumerate(order):
        running = max(running, (total - rank_index) * float(p[index]))
        output[index] = min(1.0, running)
    return output
