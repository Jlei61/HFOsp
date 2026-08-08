"""Patient plane, latent node placement and the virtual-SEEG observation operator.

The scientific point of this module is that a contact is an observation port, not
a node.  State lives on latent rate units placed in the patient's own tissue
plane; a contact sees a distance-weighted average of the units near it, and the
observed rank set is injected back through the transpose of that same map.

Kernel and normalisation are the ones already used by the SNN virtual-SEEG
readout (``src.sef_hfo_observation.sample_envelopes``): a Gaussian footprint
``exp(-d^2 / 2 sigma^2)`` normalised per contact.  ``sigma`` follows the existing
``0.5 * pitch`` rule rather than being fitted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Support cut-off in units of sigma.  Beyond this the Gaussian weight is below
# 1.2e-2 of the peak and the operator would stop being local.
SUPPORT_SIGMA = 3.0
# Floor on the kernel width in mm, so a densely sampled shaft cannot collapse
# the operator onto a single node.
MIN_SIGMA_MM = 2.0


@dataclass(frozen=True)
class PatientPlane:
    """Contacts of one patient in their own propagation plane."""

    subject: str
    contact_names: tuple[str, ...]
    xy_mm: np.ndarray          # (C, 2) along-axis, signed-transverse
    xyz_mm: np.ndarray | None  # (C, 3) raw coordinates, for the 3D sensitivity arm
    sigma_mm: float

    def __post_init__(self) -> None:
        if self.xy_mm.shape != (len(self.contact_names), 2):
            raise ValueError("xy_mm must be (n_contacts, 2) aligned to contact_names")
        if self.xyz_mm is not None and self.xyz_mm.shape[0] != len(self.contact_names):
            raise ValueError("xyz_mm must be aligned to contact_names")


def kernel_sigma_mm(xy: np.ndarray) -> float:
    """Half the median nearest-neighbour contact spacing, floored at 2 mm.

    This is the ``0.5 * pitch`` rule already used for the virtual montage, read
    off the real geometry instead of a synthetic shaft pitch.
    """
    points = np.asarray(xy, float)
    if points.shape[0] < 2:
        raise ValueError("a plane needs at least two contacts to define a pitch")
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    pitch = float(np.median(d.min(axis=1)))
    return max(MIN_SIGMA_MM, 0.5 * pitch)


def sample_latent_nodes(
    xy: np.ndarray,
    n_nodes: int,
    sigma_mm: float,
    seed: int,
    *,
    grid_step_factor: float = 0.25,
) -> np.ndarray:
    """Farthest-point sample ``n_nodes`` positions in the dilated contact region.

    The valid domain is every candidate grid point within ``SUPPORT_SIGMA *
    sigma`` of at least one contact.  That keeps nodes in tissue the montage
    actually observes, and guarantees each contact has support.  Farthest-point
    sampling is used rather than a regular grid so node positions cannot line up
    with the contact lattice by construction.
    """
    points = np.asarray(xy, float)
    reach = SUPPORT_SIGMA * float(sigma_mm)
    step = max(grid_step_factor * float(sigma_mm), 1e-3)
    lo = points.min(axis=0) - reach
    hi = points.max(axis=0) + reach
    gx = np.arange(lo[0], hi[0] + step, step)
    gy = np.arange(lo[1], hi[1] + step, step)
    grid = np.stack(np.meshgrid(gx, gy, indexing="ij"), axis=-1).reshape(-1, 2)
    d_to_contact = np.linalg.norm(grid[:, None, :] - points[None, :, :], axis=-1)
    candidates = grid[d_to_contact.min(axis=1) <= reach]
    if len(candidates) < n_nodes:
        raise ValueError(
            f"valid domain holds {len(candidates)} candidates, fewer than "
            f"{n_nodes} requested nodes; lower grid_step_factor"
        )

    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(candidates)))]
    dist = np.linalg.norm(candidates - candidates[chosen[0]], axis=1)
    for _ in range(n_nodes - 1):
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(candidates - candidates[nxt], axis=1))
    return candidates[np.array(chosen)]


def build_observation_operator(
    contacts_xy: np.ndarray,
    nodes_xy: np.ndarray,
    sigma_mm: float,
) -> np.ndarray:
    """Row-normalised local Gaussian read-out ``H`` of shape (n_contacts, n_nodes).

    Entries beyond ``SUPPORT_SIGMA * sigma`` are exactly zero, so the operator is
    local by construction rather than by decay.  Rows sum to one, so a contact
    reports a weighted average of nearby tissue and not a sum that grows with
    node density.
    """
    contacts = np.asarray(contacts_xy, float)
    nodes = np.asarray(nodes_xy, float)
    d = np.linalg.norm(contacts[:, None, :] - nodes[None, :, :], axis=-1)
    weights = np.exp(-(d ** 2) / (2.0 * float(sigma_mm) ** 2))
    weights[d > SUPPORT_SIGMA * float(sigma_mm)] = 0.0
    totals = weights.sum(axis=1, keepdims=True)
    if not np.all(totals > 0):
        empty = np.flatnonzero(totals[:, 0] <= 0)
        raise ValueError(f"contacts {empty.tolist()} observe no latent node")
    return weights / totals


def node_count(n_contacts: int) -> int:
    """``M_p = min(64, max(24, 4 C_p))`` from the frozen spec."""
    return int(min(64, max(24, 4 * int(n_contacts))))


# A contact that reads a single latent node is not observing a field; the
# readout has collapsed into a per-contact parameter, which is exactly the thing
# the latent parameterisation exists to remove.  Three is the smallest count for
# which the contact still averages a neighbourhood.
MIN_NODES_PER_CONTACT = 3
MAX_NODES = 192


def resolve_node_count(
    xy: np.ndarray,
    sigma_mm: float,
    seed: int,
) -> tuple[int, np.ndarray, np.ndarray, int]:
    """Grow the node count until every contact observes a real neighbourhood.

    The spec formula is a lower bound, not the answer: it was written for the
    median montage of ~15 contacts, and at 38 or 52 contacts the 64-node cap
    leaves nodes sparser than contacts, so some contacts fall back to reading one
    node.  Returns ``(n_nodes, nodes_xy, H, nominal_n_nodes)``.
    """
    nominal = node_count(len(xy))
    n_nodes = nominal
    while True:
        nodes = sample_latent_nodes(xy, n_nodes, sigma_mm, seed=seed)
        try:
            H = build_observation_operator(xy, nodes, sigma_mm)
        except ValueError:
            # A contact that sees nothing at all is the same problem as a contact
            # that sees too few nodes, only worse: at a narrow kernel the sampler
            # can miss a whole disc.  Growing the node count is the same cure, so
            # this path must grow rather than abort -- otherwise the operator is
            # simply unusable below about 2 mm on the wide montages.
            if n_nodes >= MAX_NODES:
                raise ValueError(
                    f"{len(xy)} contacts still leave a contact with no node at "
                    f"sigma={sigma_mm} mm and the {MAX_NODES}-node ceiling"
                ) from None
            n_nodes = min(MAX_NODES, int(np.ceil(n_nodes * 1.25)))
            continue
        if int((H > 0).sum(axis=1).min()) >= MIN_NODES_PER_CONTACT:
            return n_nodes, nodes, H, nominal
        if n_nodes >= MAX_NODES:
            raise ValueError(
                f"{len(xy)} contacts still leave a contact below "
                f"{MIN_NODES_PER_CONTACT} nodes at the {MAX_NODES}-node ceiling"
            )
        n_nodes = min(MAX_NODES, int(np.ceil(n_nodes * 1.25)))


def knn_edge_mask(nodes_xy: np.ndarray, k: int) -> np.ndarray:
    """Directed k-nearest-neighbour mask used by the fixed-local control arm."""
    nodes = np.asarray(nodes_xy, float)
    d = np.linalg.norm(nodes[:, None, :] - nodes[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    mask = np.zeros(d.shape, bool)
    for i, row in enumerate(d):
        mask[i, np.argsort(row)[:k]] = True
    return mask


def normalised_distance(nodes_xy: np.ndarray) -> np.ndarray:
    """Pairwise distance divided by the median off-diagonal distance."""
    nodes = np.asarray(nodes_xy, float)
    d = np.linalg.norm(nodes[:, None, :] - nodes[None, :, :], axis=-1)
    off = d[~np.eye(len(nodes), dtype=bool)]
    scale = float(np.median(off))
    if not scale > 0:
        raise ValueError("degenerate node layout: median inter-node distance is zero")
    return d / scale


def hop_reachability(
    edge_mask: np.ndarray,
    contact_node: Sequence[int],
    transitions: Sequence[tuple[int, int]],
    k_hops: int,
) -> float:
    """Fraction of observed contact transitions reachable within ``k_hops``.

    Spec §4.4: microsteps and the wiring cost are not independent knobs.  A
    configuration whose reachability is low is hop-limited, and a negative result
    from it cannot be attributed to the wiring economy.
    """
    adjacency = np.asarray(edge_mask, bool)
    reach = adjacency | np.eye(len(adjacency), dtype=bool)
    power = reach.copy()
    for _ in range(int(k_hops) - 1):
        power = power @ reach
    anchors = np.asarray(contact_node, int)
    if not len(transitions):
        return float("nan")
    hits = [bool(power[anchors[a], anchors[b]]) for a, b in transitions]
    return float(np.mean(hits))


def nearest_node(contacts_xy: np.ndarray, nodes_xy: np.ndarray) -> np.ndarray:
    """Index of the latent node closest to each contact."""
    contacts = np.asarray(contacts_xy, float)
    nodes = np.asarray(nodes_xy, float)
    d = np.linalg.norm(contacts[:, None, :] - nodes[None, :, :], axis=-1)
    return np.argmin(d, axis=1)
