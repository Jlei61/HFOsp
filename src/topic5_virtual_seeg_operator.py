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
from scipy.spatial import ConvexHull, QhullError

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


def kernel_sigma_mm(xy: np.ndarray, floor_mm: float = MIN_SIGMA_MM) -> float:
    """Half the median nearest-neighbour contact spacing, floored at ``floor_mm``.

    This is the ``0.5 * pitch`` rule already used for the virtual montage, read
    off the real geometry instead of a synthetic shaft pitch.

    The floor is a parameter because it is the only part of the rule that is not
    adaptive: on a dense montage it pushes the kernel far above half the pitch --
    up to 7.5x on the densest patient here -- and the contact-similarity ladder
    shows the evidence in a smoothed read-out collapses once the scale doubles.
    The default is left where v0.1 and v0.2 set it so their numbers are unchanged.
    """
    points = np.asarray(xy, float)
    if points.shape[0] < 2:
        raise ValueError("a plane needs at least two contacts to define a pitch")
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    pitch = float(np.median(d.min(axis=1)))
    return max(float(floor_mm), 0.5 * pitch)


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


# Full-tissue v0.3 uses a larger but still bounded latent mesh.  The historical
# 192-node ceiling remains untouched above so the old contact-dilated results
# stay reproducible.
FULL_TISSUE_MAX_NODES = 384
FULL_TISSUE_MIN_BACKGROUND_NODES = 64
FULL_TISSUE_MIN_ZERO_H_NODES = 16
FULL_TISSUE_MIN_ZERO_H_FRACTION = 0.10


@dataclass(frozen=True)
class FullTissueLayout:
    """Versioned latent interpolation domain for LBSS v0.3.

    ``nodes_xy`` contains both quasi-uniform background tissue nodes and any
    extra local nodes needed for a non-degenerate SEEG readout.  ``H`` is still
    exactly local, so columns whose sum is zero are genuine latent tissue state:
    they can only be driven and observed through recurrent propagation.
    """

    nodes_xy: np.ndarray
    H: np.ndarray
    domain_area_mm2: float
    domain_margin_mm: float
    background_spacing_mm: float
    candidate_step_mm: float
    n_background_nodes: int
    n_support_nodes_added: int
    n_zero_h_nodes: int
    zero_h_fraction: float
    contact_pitch_mm: float
    envelope_kind: str


def _median_contact_pitch(xy: np.ndarray) -> float:
    points = np.asarray(xy, float)
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    value = float(np.median(d.min(axis=1)))
    if not np.isfinite(value) or value <= 0:
        raise ValueError("contact geometry has no positive nearest-neighbour pitch")
    return value


def _full_tissue_candidates(
    xy: np.ndarray,
    sigma_mm: float,
) -> tuple[np.ndarray, float, float, float, str]:
    """Grid the offset contact-cloud envelope, independent of readout support.

    A non-degenerate cloud uses an outward-shifted convex hull.  A nearly
    collinear cloud uses a PCA-aligned rectangle.  Neither path clips candidates
    by distance to a contact; this is the key v0.3 change.
    """
    points = np.asarray(xy, float)
    pitch = _median_contact_pitch(points)
    margin = max(SUPPORT_SIGMA * float(sigma_mm), pitch)
    background_spacing = max(2.0, pitch)
    step = max(0.25, min(1.0, 0.5 * min(background_spacing, float(sigma_mm))))
    lo = points.min(axis=0) - margin
    hi = points.max(axis=0) + margin
    gx = np.arange(lo[0], hi[0] + step, step)
    gy = np.arange(lo[1], hi[1] + step, step)
    grid = np.stack(np.meshgrid(gx, gy, indexing="ij"), axis=-1).reshape(-1, 2)

    centred = points - points.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centred, compute_uv=False)
    near_collinear = bool(len(singular) < 2 or singular[1] <= 1e-6 * max(singular[0], 1e-12))
    if not near_collinear:
        try:
            hull = ConvexHull(points)
            normals = hull.equations[:, :2]
            offsets = hull.equations[:, 2]
            norm = np.linalg.norm(normals, axis=1)
            inside = np.all(
                grid @ normals.T + offsets <= margin * norm + 1e-9,
                axis=1,
            )
            kind = "OFFSET_CONVEX_HULL"
        except QhullError:
            near_collinear = True
    if near_collinear:
        centre = points.mean(axis=0)
        _, _, vh = np.linalg.svd(points - centre, full_matrices=False)
        basis = vh if vh.shape == (2, 2) else np.eye(2)
        contact_uv = (points - centre) @ basis.T
        grid_uv = (grid - centre) @ basis.T
        lower = contact_uv.min(axis=0) - margin
        upper = contact_uv.max(axis=0) + margin
        inside = np.all((grid_uv >= lower) & (grid_uv <= upper), axis=1)
        kind = "PCA_RECTANGLE_FALLBACK"

    candidates = grid[inside]
    if len(candidates) < FULL_TISSUE_MIN_BACKGROUND_NODES:
        raise ValueError("full-tissue envelope contains too few grid candidates")
    area = float(len(candidates) * step * step)
    return candidates, area, margin, background_spacing, kind


def _farthest_points(
    candidates: np.ndarray,
    n_points: int,
    seed: int,
) -> np.ndarray:
    if n_points > len(candidates):
        raise ValueError("requested more background nodes than domain candidates")
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(candidates)))]
    distance = np.linalg.norm(candidates - candidates[chosen[0]], axis=1)
    for _ in range(int(n_points) - 1):
        nxt = int(np.argmax(distance))
        chosen.append(nxt)
        distance = np.minimum(
            distance,
            np.linalg.norm(candidates - candidates[nxt], axis=1),
        )
    return candidates[np.asarray(chosen, dtype=int)]


def _farthest_points_from_existing(
    candidates: np.ndarray,
    existing: np.ndarray,
    n_points: int,
) -> np.ndarray:
    """Deterministically fill holes while staying far from the existing mesh."""
    pool = np.asarray(candidates, float)
    fixed = np.asarray(existing, float)
    if int(n_points) <= 0:
        return np.empty((0, 2), dtype=float)
    distance = np.linalg.norm(pool[:, None, :] - fixed[None, :, :], axis=-1).min(axis=1)
    chosen: list[int] = []
    for _ in range(int(n_points)):
        nxt = int(np.argmax(distance))
        if not np.isfinite(distance[nxt]) or distance[nxt] <= 1e-7:
            raise ValueError("not enough distinct candidates to fill the latent mesh")
        chosen.append(nxt)
        distance = np.minimum(distance, np.linalg.norm(pool - pool[nxt], axis=1))
        distance[nxt] = -np.inf
    return pool[np.asarray(chosen, dtype=int)]


def _add_contact_support_nodes(
    contacts_xy: np.ndarray,
    nodes_xy: np.ndarray,
    sigma_mm: float,
) -> tuple[np.ndarray, int]:
    """Add only the local latent nodes needed to keep every H row a neighbourhood."""
    contacts = np.asarray(contacts_xy, float)
    nodes = [row.copy() for row in np.asarray(nodes_xy, float)]
    radius = 0.5 * float(sigma_mm)
    base_angles = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
    added = 0
    for contact_index, contact in enumerate(contacts):
        array = np.asarray(nodes, float)
        count = int(np.sum(np.linalg.norm(array - contact, axis=1) <= SUPPORT_SIGMA * sigma_mm))
        angle_offset = (contact_index % 7) * (np.pi / 31.0)
        for angle in base_angles + angle_offset:
            if count >= MIN_NODES_PER_CONTACT:
                break
            candidate = contact + radius * np.array([np.cos(angle), np.sin(angle)])
            array = np.asarray(nodes, float)
            if np.min(np.linalg.norm(array - candidate, axis=1)) <= 1e-7:
                continue
            nodes.append(candidate)
            count += 1
            added += 1
    return np.asarray(nodes, float), int(added)


def resolve_full_tissue_layout(
    xy: np.ndarray,
    sigma_mm: float,
    seed: int,
    *,
    max_nodes: int = FULL_TISSUE_MAX_NODES,
) -> FullTissueLayout:
    """Build the v0.3 full-tissue mesh and local virtual-SEEG operator.

    The background node budget is area based rather than contact-count based.
    Up to three support nodes per contact are reserved before applying the hard
    ceiling, so narrow readout kernels cannot make a contact disappear.
    """
    points = np.asarray(xy, float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("xy must be an (n_contacts, 2) array with at least two contacts")
    candidates, area, margin, spacing, envelope_kind = _full_tissue_candidates(
        points, sigma_mm
    )
    pitch = _median_contact_pitch(points)
    reserve = MIN_NODES_PER_CONTACT * len(points)
    if reserve + FULL_TISSUE_MIN_BACKGROUND_NODES > int(max_nodes):
        raise ValueError("max_nodes cannot hold the minimum background and contact support nodes")
    target_background = int(np.ceil(area / (spacing * spacing)))
    target_background = max(FULL_TISSUE_MIN_BACKGROUND_NODES, target_background)
    target_background = min(int(max_nodes) - reserve, target_background)
    background = _farthest_points(candidates, target_background, seed)
    nodes, n_support_added = _add_contact_support_nodes(points, background, sigma_mm)
    if len(nodes) > int(max_nodes):
        raise RuntimeError("full-tissue node ceiling exceeded after support-node placement")
    H = build_observation_operator(points, nodes, sigma_mm)
    zero = np.asarray(H.sum(axis=0) <= 1e-12)
    minimum_zero = max(
        FULL_TISSUE_MIN_ZERO_H_NODES,
        int(np.ceil(FULL_TISSUE_MIN_ZERO_H_FRACTION * len(nodes))),
    )
    if int(zero.sum()) < minimum_zero:
        needed = minimum_zero - int(zero.sum())
        if len(nodes) + needed > int(max_nodes):
            raise ValueError(
                "node ceiling leaves too little room for explicit zero-H tissue nodes"
            )
        distance_to_contact = np.linalg.norm(
            candidates[:, None, :] - points[None, :, :], axis=-1
        ).min(axis=1)
        zero_candidates = candidates[
            distance_to_contact > SUPPORT_SIGMA * float(sigma_mm) + 1e-9
        ]
        fill = _farthest_points_from_existing(zero_candidates, nodes, needed)
        nodes = np.concatenate([nodes, fill], axis=0)
        H = build_observation_operator(points, nodes, sigma_mm)
        zero = np.asarray(H.sum(axis=0) <= 1e-12)
        if int(zero.sum()) < minimum_zero:
            raise RuntimeError("zero-H fill did not satisfy the frozen coverage contract")
    return FullTissueLayout(
        nodes_xy=nodes,
        H=H,
        domain_area_mm2=area,
        domain_margin_mm=margin,
        background_spacing_mm=spacing,
        candidate_step_mm=max(0.25, min(1.0, 0.5 * min(spacing, float(sigma_mm)))),
        n_background_nodes=int(len(background)),
        n_support_nodes_added=n_support_added,
        n_zero_h_nodes=int(zero.sum()),
        zero_h_fraction=float(zero.mean()),
        contact_pitch_mm=pitch,
        envelope_kind=envelope_kind,
    )


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
