"""Topic 5.2D v0.2 — frozen spatial bases, matched nulls and basis ceilings.

Every basis is a ``C x r`` column-orthonormal matrix that constrains the encoder
*and* the readout of the ordered model, so an arm cannot compensate a bad basis
with a free contact readout.  Truncations are nested by construction
(``Q_r = U[:, :r]``), which is what makes the capacity curve interpretable.

What each basis is allowed to see
---------------------------------
``GEOMETRY_LAYOUT``   3-D contact positions only, through an isotropic local kernel.
``SHAFT_GRADIENT``    within-shaft linear coordinate, shaft identity, shaft adjacency.
``PATIENT_ALIGNED``   the above plus a split-0-only late-field displacement axis.
``ANGLE_ROTATED_AXIS``  the aligned recipe with the axis rotated by a frozen angle.
``IDENTITY_PERMUTED``   the aligned basis with contact rows permuted inside
                        (shaft, radial-distance, degree) bins.
``LOCALITY_REWIRED``    the aligned recipe on a degree/length-matched rewired graph.

None of them reads a held-out split, a mode label or a seizure target.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

BASIS_KINDS = (
    "GEOMETRY_LAYOUT",
    "SHAFT_GRADIENT",
    "PATIENT_ALIGNED",
    "ANGLE_ROTATED_AXIS",
    "IDENTITY_PERMUTED",
    "LOCALITY_REWIRED",
    "TRAIN_ONLY_FREE_PCA",
)
RANKS: tuple[int, ...] = (1, 2, 4, 8)
N_ANGLE_NULLS = 8
N_IDENTITY_NULLS = 4
N_REWIRE_NULLS = 4
ANGLE_SEED = 20260817
IDENTITY_SEED = 20260818
REWIRE_SEED = 20260819
ANGLE_NULL_INELIGIBLE = "ANGLE_NULL_INELIGIBLE"

# Frozen rotation grid: eight evenly spaced angles inside (0, pi).  The axis is
# undirected, so pi is the aligned axis again and is excluded.
ANGLE_GRID_RAD: tuple[float, ...] = tuple(float(k * np.pi / 9.0) for k in range(1, N_ANGLE_NULLS + 1))
# Pre-frozen angle subsets used by the peripheral blocks so that capacity and
# learning curves do not repeat the whole null family at every combination.
ANGLE_SUBSET_4: tuple[int, ...] = (1, 3, 5, 7)
ANGLE_SUBSET_2: tuple[int, ...] = (1, 5)


# ---------------------------------------------------------------------------
# geometry primitives
# ---------------------------------------------------------------------------
def local_kernel_sigma(coords_3d: np.ndarray) -> float:
    """Median nearest-neighbour distance — a geometry-only length scale."""
    distance = np.linalg.norm(coords_3d[:, None, :] - coords_3d[None, :, :], axis=2)
    np.fill_diagonal(distance, np.inf)
    return float(np.median(distance.min(axis=1)))


LOCAL_NEIGHBOURS = 3


def local_graph(coords_3d: np.ndarray, k: int = LOCAL_NEIGHBOURS) -> np.ndarray:
    """Symmetrised k-nearest-neighbour support, union a minimum spanning tree.

    Every basis in the aligned family (aligned, angle-rotated, identity-permuted,
    locality-rewired) and the geometry basis use this same support, so the
    rewired null differs from the observed one *only* in which pairs are
    connected — not in how dense the kernel is.  ``k`` is small on purpose: a
    near-complete support would leave the locality-rewired null nothing to
    scramble, and the spanning-tree union keeps the support connected without
    reading any event.
    """
    n_contacts = coords_3d.shape[0]
    if n_contacts < 2:
        return np.zeros((n_contacts, n_contacts), dtype=bool)
    neighbours = min(max(1, k), n_contacts - 1)
    distance = np.linalg.norm(coords_3d[:, None, :] - coords_3d[None, :, :], axis=2)
    masked = distance.copy()
    np.fill_diagonal(masked, np.inf)
    graph = np.zeros((n_contacts, n_contacts), dtype=bool)
    for index in range(n_contacts):
        graph[index, np.argsort(masked[index])[:neighbours]] = True
    graph = graph | graph.T
    return graph | _minimum_spanning_tree(distance)


def _minimum_spanning_tree(distance: np.ndarray) -> np.ndarray:
    n_contacts = distance.shape[0]
    inside = np.zeros(n_contacts, dtype=bool)
    inside[0] = True
    tree = np.zeros((n_contacts, n_contacts), dtype=bool)
    for _ in range(n_contacts - 1):
        block = distance[np.ix_(inside, ~inside)]
        source, target = np.unravel_index(int(np.argmin(block)), block.shape)
        left = np.flatnonzero(inside)[source]
        right = np.flatnonzero(~inside)[target]
        tree[left, right] = tree[right, left] = True
        inside[right] = True
    return tree


def isotropic_kernel(coords_3d: np.ndarray, sigma: float, support: np.ndarray | None = None) -> np.ndarray:
    distance = np.linalg.norm(coords_3d[:, None, :] - coords_3d[None, :, :], axis=2)
    kernel = np.exp(-0.5 * (distance / max(sigma, 1e-6)) ** 2)
    if support is not None:
        kernel = kernel * support
    np.fill_diagonal(kernel, 1.0)
    return kernel


def directional_kernels(
    kernel: np.ndarray,
    coords_3d: np.ndarray,
    coords_2d: np.ndarray,
    axis_2d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward/backward anisotropic kernels along ``axis_2d``.

    The anisotropy is a cosine gate on the in-plane displacement, so rotating
    ``axis_2d`` changes the direction and nothing else — that is what makes the
    angle-rotated null strength-matched (same ``K_0``, same anisotropy strength,
    same rank, same parameter count, same contact identity).
    """
    delta_2d = coords_2d[None, :, :] - coords_2d[:, None, :]
    delta_3d = coords_3d[None, :, :] - coords_3d[:, None, :]
    norm = np.linalg.norm(delta_3d, axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = (delta_2d @ axis_2d) / np.where(norm > 1e-9, norm, np.inf)
    cosine = np.nan_to_num(cosine)
    offdiag = kernel.copy()
    np.fill_diagonal(offdiag, 0.0)
    return offdiag * np.clip(cosine, 0.0, None), offdiag * np.clip(-cosine, 0.0, None)


def shaft_indicator_matrix(shafts: list[str]) -> np.ndarray:
    names = sorted(set(shafts))
    matrix = np.zeros((len(shafts), len(names)))
    for index, shaft in enumerate(shafts):
        matrix[index, names.index(shaft)] = 1.0
    return matrix


def project_out(dictionary: np.ndarray, nuisance: np.ndarray) -> np.ndarray:
    """Remove the span of ``nuisance`` (constant field + shaft indicators)."""
    basis, _ = np.linalg.qr(nuisance)
    return dictionary - basis @ (basis.T @ dictionary)


def orthonormal_truncation(dictionary: np.ndarray, max_rank: int) -> tuple[np.ndarray, np.ndarray]:
    left, singular, _ = np.linalg.svd(dictionary, full_matrices=False)
    keep = min(max_rank, left.shape[1])
    return np.ascontiguousarray(left[:, :keep]), singular[:keep]


# ---------------------------------------------------------------------------
# per-basis construction
# ---------------------------------------------------------------------------
def geometry_basis(kernel: np.ndarray, max_rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Low-frequency dictionary ``[K_0, K_0^2]`` — 3-D positions only."""
    dictionary = np.concatenate([kernel, kernel @ kernel], axis=1)
    return orthonormal_truncation(dictionary, max_rank)


def shaft_basis(shafts: list[str], coords_3d: np.ndarray, max_rank: int
                ) -> tuple[np.ndarray, np.ndarray]:
    """Within-shaft linear coordinate, shaft identity and shaft adjacency only."""
    n_contacts = len(shafts)
    columns = []
    adjacency = np.zeros((n_contacts, n_contacts))
    for name in sorted(set(shafts)):
        members = np.flatnonzero(np.asarray(shafts) == name)
        indicator = np.zeros(n_contacts)
        indicator[members] = 1.0
        columns.append(indicator)
        if members.size >= 2:
            block = coords_3d[members] - coords_3d[members].mean(axis=0)
            direction = np.linalg.svd(block, full_matrices=False)[2][0]
            coordinate = block @ direction
            scale = np.abs(coordinate).max()
            coordinate = coordinate / (scale if scale > 1e-9 else 1.0)
            linear = np.zeros(n_contacts)
            linear[members] = coordinate
            quadratic = np.zeros(n_contacts)
            quadratic[members] = coordinate ** 2 - float((coordinate ** 2).mean())
            columns.extend([linear, quadratic])
            order = members[np.argsort(coordinate)]
            for left, right in zip(order[:-1], order[1:]):
                adjacency[left, right] = adjacency[right, left] = 1.0
    dictionary = np.column_stack(columns)
    smoothing = np.eye(n_contacts) + adjacency
    return orthonormal_truncation(np.concatenate([dictionary, smoothing @ dictionary], axis=1), max_rank)


def estimate_axis_2d(displacements: np.ndarray) -> tuple[np.ndarray, dict]:
    """Principal eigenvector of the displacement outer-product (undirected).

    The sign is fixed by a pure geometric convention; it cannot change the span
    of the basis because the forward and backward kernels simply swap roles.
    """
    outer = displacements.T @ displacements
    values, vectors = np.linalg.eigh(outer)
    order = np.argsort(values)[::-1]
    axis = vectors[:, order[0]]
    if axis[0] < 0 or (abs(axis[0]) < 1e-12 and axis[1] < 0):
        axis = -axis
    total = float(values.sum())
    return axis, {
        "eigenvalues": [float(values[i]) for i in order],
        "anisotropy_ratio": float(values[order[0]] / total) if total > 0 else float("nan"),
        "n_displacements": int(displacements.shape[0]),
    }


def aligned_dictionary(
    kernel: np.ndarray,
    coords_3d: np.ndarray,
    coords_2d: np.ndarray,
    axis_2d: np.ndarray,
    shafts: list[str],
) -> np.ndarray:
    """``[K_0, K_+, K_-, K_+^2, K_-^2]`` with the constant field and the frozen
    shaft indicators projected out (spec §4.5 step 6)."""
    forward, backward = directional_kernels(kernel, coords_3d, coords_2d, axis_2d)
    dictionary = np.concatenate(
        [kernel, forward, backward, forward @ forward, backward @ backward], axis=1
    )
    nuisance = np.column_stack([np.ones(len(shafts)), shaft_indicator_matrix(shafts)])
    return project_out(dictionary, nuisance)


def rotate_axis(axis_2d: np.ndarray, angle: float) -> np.ndarray:
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    return rotation @ axis_2d


def identity_permutation(
    coords_3d: np.ndarray,
    shafts: list[str],
    support: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Permute contacts inside (shaft, radial-distance bin, degree bin) cells.

    A row permutation of a column-orthonormal basis stays column-orthonormal, so
    rank, singular values, column norms and parameter count are preserved by
    construction; only which contact carries which basis coordinate changes.
    """
    rng = np.random.default_rng(seed)
    centre = coords_3d.mean(axis=0)
    radial = np.linalg.norm(coords_3d - centre, axis=1)
    radial_bin = np.digitize(radial, np.quantile(radial, [0.25, 0.5, 0.75]))
    degree = support.sum(axis=1)
    degree_bin = np.digitize(degree, np.quantile(degree, [0.5]))
    permutation = np.arange(len(shafts))
    keys = np.asarray([f"{shafts[i]}|{radial_bin[i]}|{degree_bin[i]}" for i in range(len(shafts))])
    for key in sorted(set(keys.tolist())):
        members = np.flatnonzero(keys == key)
        permutation[members] = rng.permutation(members)
    return permutation


def rewire_support(
    coords_3d: np.ndarray,
    shafts: list[str],
    support: np.ndarray,
    seed: int,
    n_attempts: int = 40000,
    length_tolerance: float = 0.5,
    mean_length_tolerance: float = 0.1,
) -> tuple[np.ndarray, dict]:
    """Degree-preserving double-edge swaps that keep the within/cross-shaft mix
    and the edge-length distribution.

    Quantities that could not be matched are reported rather than silently
    accepted; a complete graph (small contact sets) cannot be rewired at all and
    is flagged ``REWIRE_DEGENERATE``.
    """
    rng = np.random.default_rng(seed)
    original = np.asarray(support, dtype=bool).copy()
    np.fill_diagonal(original, False)
    graph = original.copy()
    distance = np.linalg.norm(coords_3d[:, None, :] - coords_3d[None, :, :], axis=2)
    same_shaft = np.asarray(shafts)[:, None] == np.asarray(shafts)[None, :]
    n_contacts = graph.shape[0]
    edges = [(i, j) for i in range(n_contacts) for j in range(i + 1, n_contacts) if graph[i, j]]
    complete = len(edges) == n_contacts * (n_contacts - 1) // 2
    accepted = 0
    target_total = float(sum(distance[i, j] for i, j in edges))
    total = target_total
    budget = mean_length_tolerance * max(target_total, 1e-9)
    if len(edges) >= 2 and not complete:
        for _ in range(n_attempts):
            first, second = (int(v) for v in rng.integers(0, len(edges), size=2))
            if first == second:
                continue
            (a, b), (c, d) = edges[first], edges[second]
            if len({a, b, c, d}) < 4 or graph[a, d] or graph[c, b]:
                continue
            if same_shaft[a, b] != same_shaft[a, d] or same_shaft[c, d] != same_shaft[c, b]:
                continue
            before = distance[a, b] + distance[c, d]
            after = distance[a, d] + distance[c, b]
            if abs(after - before) > length_tolerance * max(before, 1e-9):
                continue
            if abs(total - before + after - target_total) > budget:
                continue
            total = total - before + after
            graph[a, b] = graph[b, a] = False
            graph[c, d] = graph[d, c] = False
            graph[a, d] = graph[d, a] = True
            graph[c, b] = graph[b, c] = True
            edges[first] = (min(a, d), max(a, d))
            edges[second] = (min(c, b), max(c, b))
            accepted += 1
    report = _rewire_report(original, graph, distance, same_shaft)
    report["accepted_swaps"] = accepted
    report["complete_graph"] = bool(complete)
    report["contact_identity_overlap"] = float(
        (original & graph).sum() / max(1, original.sum())
    )
    if complete or accepted == 0:
        report["unmatched"] = sorted(set(report["unmatched"] + ["REWIRE_DEGENERATE"]))
    return graph, report


def _rewire_report(original: np.ndarray, rewired: np.ndarray, distance: np.ndarray,
                   same_shaft: np.ndarray) -> dict:
    def summary(graph: np.ndarray) -> dict:
        edge = graph.astype(bool)
        integer = edge.astype(np.int32)
        two_step = ((integer @ integer) > 0) | edge
        off = edge.size - graph.shape[0]
        return {
            "n_edges": int(edge.sum() // 2),
            "degree_min": int(integer.sum(axis=1).min()),
            "degree_max": int(integer.sum(axis=1).max()),
            "mean_edge_length_mm": float(distance[edge].mean()) if edge.any() else float("nan"),
            "within_shaft_fraction": float((edge & same_shaft).sum() / max(1, edge.sum())),
            "one_step_reachable_fraction": float(edge.sum() / max(1, off)),
            "two_step_reachable_fraction": float((two_step.sum() - graph.shape[0]) / max(1, off)),
            "connected": bool(_is_connected(edge)),
        }

    before, after = summary(original), summary(rewired)
    unmatched = [
        key for key in before
        if key.endswith("fraction") and abs(before[key] - after[key]) > 0.05
    ]
    if abs(before["mean_edge_length_mm"] - after["mean_edge_length_mm"]) > 0.1 * max(
        before["mean_edge_length_mm"], 1e-6
    ):
        unmatched.append("mean_edge_length_mm")
    if before["degree_min"] != after["degree_min"] or before["degree_max"] != after["degree_max"]:
        unmatched.append("degree_sequence")
    if before["connected"] != after["connected"]:
        unmatched.append("connectedness")
    return {"observed": before, "rewired": after, "unmatched": sorted(set(unmatched))}


def _is_connected(edge: np.ndarray) -> bool:
    n_contacts = edge.shape[0]
    seen = np.zeros(n_contacts, dtype=bool)
    stack = [0]
    seen[0] = True
    while stack:
        node = stack.pop()
        for neighbour in np.flatnonzero(edge[node] & ~seen):
            seen[neighbour] = True
            stack.append(int(neighbour))
    return bool(seen.all())


# ---------------------------------------------------------------------------
# basis ceiling
# ---------------------------------------------------------------------------
def masked_projection_residual(field: np.ndarray, mask: np.ndarray, basis: np.ndarray) -> tuple[float, float]:
    """Best per-field projection error under a per-sample contact mask.

    The coefficients are re-optimised for every field, which is exactly what
    makes this a representation ceiling rather than a predictor.  Solved in one
    batched pseudo-inverse so the ceiling is affordable on the large montages.
    """
    keep = np.asarray(mask, dtype=bool)
    target = np.where(keep, field, 0.0).astype(np.float64)
    energies = (target ** 2).sum(axis=1)
    usable = (keep.sum(axis=1) >= 2) & (energies > 1e-12)
    if not usable.any():
        return 0.0, 0.0
    target = target[usable]
    design = basis[None, :, :] * keep[usable][:, :, None]
    gram = np.einsum("sci,scj->sij", design, design)
    projection = np.einsum("sci,sc->si", design, target)
    coefficients = np.einsum("sij,sj->si", np.linalg.pinv(gram), projection)
    explained = np.einsum("si,si->s", projection, coefficients)
    energy = energies[usable]
    return float(np.clip(energy - explained, 0.0, None).sum()), float(energy.sum())


def principal_angles(first: np.ndarray, second: np.ndarray) -> list[float]:
    q_first = np.linalg.qr(first)[0]
    q_second = np.linalg.qr(second)[0]
    singular = np.linalg.svd(q_first.T @ q_second, compute_uv=False)
    return [float(np.arccos(np.clip(value, -1.0, 1.0))) for value in singular]


def effective_rank(matrix: np.ndarray) -> float:
    singular = np.linalg.svd(matrix, compute_uv=False)
    energy = singular ** 2
    total = energy.sum()
    if total <= 0:
        return float("nan")
    weights = energy / total
    return float(np.exp(-(weights * np.log(np.clip(weights, 1e-300, None))).sum()))


# ---------------------------------------------------------------------------
# manifest helpers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BasisRecord:
    patient: str
    kind: str
    null_id: str
    rank: int
    basis: np.ndarray
    singular_values: np.ndarray
    metadata: dict

    def orthogonality_error(self) -> float:
        gram = self.basis.T @ self.basis
        return float(np.abs(gram - np.eye(gram.shape[0])).max())

    def shaft_projection(self, shafts: list[str]) -> float:
        indicators = shaft_indicator_matrix(shafts)
        indicators = indicators / np.linalg.norm(indicators, axis=0, keepdims=True)
        return float(np.linalg.norm(indicators.T @ self.basis))

    def hash(self) -> str:
        return hashlib.sha256(np.ascontiguousarray(self.basis.astype(np.float64)).tobytes()).hexdigest()


def save_basis_bundle(records: list[BasisRecord], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    index = []
    for record in records:
        key = f"{record.kind}|{record.null_id}|r{record.rank}"
        payload[key] = record.basis.astype(np.float32)
        payload[key + "|sv"] = record.singular_values.astype(np.float64)
        index.append(
            {
                "key": key,
                "patient": record.patient,
                "kind": record.kind,
                "null_id": record.null_id,
                "rank": record.rank,
                "orthogonality_error": record.orthogonality_error(),
                "sha256": record.hash(),
                **record.metadata,
            }
        )
    payload["index"] = np.asarray(json.dumps(index))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_basis_bundle(path: Path) -> tuple[dict[str, np.ndarray], list[dict]]:
    payload = np.load(path, allow_pickle=False)
    index = json.loads(str(payload["index"]))
    bases = {entry["key"]: payload[entry["key"]] for entry in index}
    return bases, index
