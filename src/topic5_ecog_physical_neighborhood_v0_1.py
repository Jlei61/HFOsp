"""Physical-grid graph and RNN contracts for Topic 5.2 ECoG v0.1."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor

from src.topic5_lbss_rnn_v0_2 import LBSSConfig, LBSSModel


GRID_ROWS = "ABCDEFGH"


def contact_xy(contact: str) -> tuple[int, int]:
    name = str(contact).upper()
    if len(name) < 3 or name[0] != "G" or name[1] not in GRID_ROWS:
        raise ValueError(f"not an 8x8 grid contact: {contact!r}")
    column = int(name[2:]) - 1
    if not 0 <= column < 8:
        raise ValueError(f"grid column outside 1..8: {contact!r}")
    return GRID_ROWS.index(name[1]), column


def coordinate_array(channel_names: Iterable[str]) -> np.ndarray:
    return np.asarray([contact_xy(name) for name in channel_names], dtype=float)


def distance_matrix(channel_names: Iterable[str]) -> np.ndarray:
    xy = coordinate_array(channel_names)
    return np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)


def true_grid_mask(channel_names: Iterable[str], diagonal: bool = False) -> np.ndarray:
    names = [str(name) for name in channel_names]
    xy = coordinate_array(names)
    delta = np.abs(xy[:, None, :] - xy[None, :, :])
    if diagonal:
        adjacent = (delta.max(axis=-1) == 1) & (delta.sum(axis=-1) >= 1)
    else:
        adjacent = delta.sum(axis=-1) == 1
    np.fill_diagonal(adjacent, False)
    return adjacent.astype(np.uint8)


def is_connected(mask: np.ndarray) -> bool:
    graph = np.asarray(mask, dtype=bool)
    if graph.shape[0] == 0:
        return False
    seen = {0}
    queue: deque[int] = deque([0])
    while queue:
        source = queue.popleft()
        for target in np.flatnonzero(graph[source] | graph[:, source]):
            target = int(target)
            if target not in seen:
                seen.add(target)
                queue.append(target)
    return len(seen) == graph.shape[0]


def degree_class_permutation(mask: np.ndarray, seed: int) -> np.ndarray:
    graph = np.asarray(mask, dtype=np.uint8)
    degree = graph.sum(axis=0).astype(int)
    classes: dict[int, list[int]] = defaultdict(list)
    for node, value in enumerate(degree):
        classes[int(value)].append(node)
    rng = np.random.default_rng(seed)
    permutation = np.arange(graph.shape[0])
    for nodes in classes.values():
        shuffled = np.asarray(nodes, dtype=int).copy()
        rng.shuffle(shuffled)
        permutation[np.asarray(nodes, dtype=int)] = shuffled
    wrong = graph[np.ix_(permutation, permutation)]
    if not np.array_equal(wrong.sum(axis=0), graph.sum(axis=0)):
        raise RuntimeError("degree-class permutation changed node degree")
    return wrong.astype(np.uint8)


def _undirected_edges(mask: np.ndarray) -> set[tuple[int, int]]:
    graph = np.asarray(mask, dtype=bool)
    return {(i, j) for i in range(graph.shape[0]) for j in range(i + 1, graph.shape[0]) if graph[i, j] or graph[j, i]}


def degree_preserving_random_mask(mask: np.ndarray, seed: int, swaps_per_edge: int = 30) -> np.ndarray:
    base = np.asarray(mask, dtype=np.uint8)
    if not np.array_equal(base, base.T):
        raise ValueError("degree-preserving randomizer expects a symmetric grid mask")
    rng = np.random.default_rng(seed)
    original_degree = base.sum(axis=0)
    original_edges = _undirected_edges(base)
    n_target = max(1, int(swaps_per_edge) * len(original_edges))
    for restart in range(20):
        edges = set(original_edges)
        accepted = 0
        attempts = 0
        local_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)) + restart)
        while accepted < n_target and attempts < n_target * 100:
            attempts += 1
            edge_list = tuple(edges)
            first, second = local_rng.choice(len(edge_list), size=2, replace=False)
            a, b = edge_list[int(first)]
            c, d = edge_list[int(second)]
            if len({a, b, c, d}) < 4:
                continue
            if bool(local_rng.integers(0, 2)):
                proposed = {(min(a, d), max(a, d)), (min(c, b), max(c, b))}
            else:
                proposed = {(min(a, c), max(a, c)), (min(b, d), max(b, d))}
            if len(proposed) < 2 or any(i == j for i, j in proposed) or any(edge in edges for edge in proposed):
                continue
            edges.remove((a, b))
            edges.remove((c, d))
            edges.update(proposed)
            accepted += 1
        random_mask = np.zeros_like(base)
        for a, b in edges:
            random_mask[a, b] = 1
            random_mask[b, a] = 1
        if np.array_equal(random_mask.sum(axis=0), original_degree) and is_connected(random_mask):
            return random_mask
    raise RuntimeError("could not generate connected degree-preserving random graph")


def graph_audit(mask: np.ndarray, true_mask: np.ndarray) -> dict[str, object]:
    graph = np.asarray(mask, dtype=np.uint8)
    truth = np.asarray(true_mask, dtype=np.uint8)
    degree = graph.sum(axis=0).astype(int)
    true_edges = truth.astype(bool)
    overlap = int(np.sum(graph.astype(bool) & true_edges))
    union = int(np.sum(graph.astype(bool) | true_edges))
    return {
        "n_nodes": int(graph.shape[0]),
        "n_directed_edges": int(graph.sum()),
        "symmetric": bool(np.array_equal(graph, graph.T)),
        "connected": is_connected(graph),
        "degree": degree.tolist(),
        "minimum_degree": int(degree.min()),
        "maximum_degree": int(degree.max()),
        "true_edge_overlap_fraction": float(overlap / max(1, int(true_edges.sum()))),
        "true_edge_jaccard": float(overlap / max(1, union)),
        "eigenvalues": np.linalg.eigvalsh(graph.astype(float)).round(10).tolist(),
    }


class ECoGGridRNN(LBSSModel):
    """Fixed-mask contact-as-node RNN with within-rank recurrent relaxation."""

    def __init__(self, config: LBSSConfig, microsteps: int = 2):
        super().__init__(config)
        if int(microsteps) < 1:
            raise ValueError("microsteps must be positive")
        self.microsteps = int(microsteps)

    def forward(self, x: Tensor, recruited: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        batch, steps, _ = x.shape
        device = x.device
        hidden = torch.zeros(batch, self.n_nodes * self.state_dim, device=device)
        zero_input = torch.zeros(batch, self.n_contacts, device=device)
        logits: list[Tensor] = []
        stops: list[Tensor] = []
        denominator = max(1, self.n_contacts - 1)
        for step in range(steps):
            hidden = self._step(hidden, x[:, step])
            for _ in range(self.microsteps - 1):
                hidden = self._step(hidden, zero_input)
            logits.append(self._readout(hidden))
            t_norm = torch.full((batch,), step / denominator, device=device)
            stops.append(self._stop(hidden, t_norm, recruited[:, step].mean(-1)))
        return torch.stack(logits, 1), torch.stack(stops, 1)


def build_fixed_grid_model(
    channel_names: Iterable[str],
    mask: np.ndarray,
    seed: int,
    *,
    state_dim: int = 1,
    microsteps: int = 2,
) -> ECoGGridRNN:
    names = [str(name) for name in channel_names]
    graph = np.asarray(mask, dtype=np.uint8)
    n = len(names)
    if graph.shape != (n, n):
        raise ValueError("mask does not align to channel_names")
    zeros = np.zeros_like(graph, dtype=np.uint8)
    config = LBSSConfig(
        arm="L0_LOCAL_ONLY",
        n_contacts=n,
        n_nodes=n,
        observation_operator=np.eye(n, dtype=np.float32),
        node_distance_mm=distance_matrix(names).astype(np.float32),
        local_mask=graph,
        extra_local_pool=zeros,
        nonlocal_pool=zeros,
        k_added=1,
        seed=int(seed),
        state_dim=int(state_dim),
    )
    return ECoGGridRNN(config, microsteps=microsteps)


@dataclass(frozen=True)
class PatchEdgeContract:
    patch_nodes: tuple[int, ...]
    edge_mask: np.ndarray


def patch_edge_mask(recurrent_mask: np.ndarray, patch_nodes: Iterable[int]) -> PatchEdgeContract:
    graph = np.asarray(recurrent_mask, dtype=bool)
    nodes = tuple(sorted({int(node) for node in patch_nodes}))
    involved = np.zeros(graph.shape[0], dtype=bool)
    involved[list(nodes)] = True
    edge = graph & (involved[:, None] | involved[None, :])
    return PatchEdgeContract(nodes, edge.astype(np.uint8))


def enumerate_square_patches(
    channel_names: Sequence[str], side: int,
) -> list[tuple[str, tuple[int, ...]]]:
    """Enumerate complete side-by-side grid patches without filling bad contacts."""
    if int(side) < 2 or int(side) > 8:
        raise ValueError("side must lie in 2..8")
    coordinates = {contact_xy(name): index for index, name in enumerate(channel_names)}
    patches: list[tuple[str, tuple[int, ...]]] = []
    for row in range(9 - int(side)):
        for column in range(9 - int(side)):
            cells = [
                (row + dr, column + dc)
                for dr in range(int(side)) for dc in range(int(side))
            ]
            if not all(cell in coordinates for cell in cells):
                continue
            nodes = tuple(coordinates[cell] for cell in cells)
            patch_id = f"{GRID_ROWS[row]}{column + 1}_{GRID_ROWS[row + side - 1]}{column + side}"
            patches.append((patch_id, nodes))
    return patches


def undirected_edges(mask: np.ndarray) -> list[tuple[int, int]]:
    graph = np.asarray(mask, dtype=bool)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("mask must be square")
    return [
        (source, target)
        for source in range(graph.shape[0])
        for target in range(source + 1, graph.shape[0])
        if graph[source, target] or graph[target, source]
    ]


def edge_set_mask(n_nodes: int, edges: Iterable[tuple[int, int]]) -> np.ndarray:
    output = np.zeros((int(n_nodes), int(n_nodes)), dtype=np.uint8)
    for first, second in edges:
        output[int(first), int(second)] = 1
        output[int(second), int(first)] = 1
    return output


def largest_edge_component(edges: Iterable[tuple[int, int]]) -> int:
    adjacency: dict[int, set[int]] = defaultdict(set)
    for first, second in edges:
        adjacency[int(first)].add(int(second))
        adjacency[int(second)].add(int(first))
    largest = 0
    remaining = set(adjacency)
    while remaining:
        start = remaining.pop()
        seen = {start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbour in adjacency[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    remaining.discard(neighbour)
                    queue.append(neighbour)
        largest = max(largest, len(seen))
    return largest


def matched_dispersed_edge_sets(
    recurrent_mask: np.ndarray,
    recurrent_weight: np.ndarray,
    lesion_edge_mask: np.ndarray,
    *,
    n_controls: int,
    seed: int,
    max_component_nodes: int = 4,
    candidates_per_control: int = 512,
) -> tuple[list[np.ndarray], list[dict[str, float | int]]]:
    """Match local lesion edges while dispersing them across the grid.

    Matching is exact for undirected edge count and endpoint degree classes.
    Among eligible draws, the function minimizes the discrepancy in learned
    pre-lesion weight quantiles.  Both directions of every selected local edge
    are attenuated together.
    """
    graph = np.asarray(recurrent_mask, dtype=bool)
    lesion = np.asarray(lesion_edge_mask, dtype=bool)
    weight = np.asarray(recurrent_weight, dtype=float)
    if graph.shape != lesion.shape or graph.shape != weight.shape:
        raise ValueError("graph, lesion and recurrent_weight must align")
    all_edges = undirected_edges(graph)
    lesion_edges = [edge for edge in all_edges if lesion[edge] or lesion[edge[::-1]]]
    candidates = [edge for edge in all_edges if not (lesion[edge] or lesion[edge[::-1]])]
    if not lesion_edges:
        raise ValueError("lesion contains no recurrent edge")
    degree = graph.sum(axis=0).astype(int)
    strength = {
        edge: 0.5 * (abs(float(weight[edge])) + abs(float(weight[edge[::-1]])))
        for edge in all_edges
    }
    strength_order = sorted(all_edges, key=lambda edge: (strength[edge], edge))
    quantile = {edge: rank / max(1, len(strength_order) - 1) for rank, edge in enumerate(strength_order)}

    def edge_class(edge: tuple[int, int]) -> tuple[int, int]:
        return tuple(sorted((int(degree[edge[0]]), int(degree[edge[1]]))))

    target_by_class: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    pool_by_class: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for edge in lesion_edges:
        target_by_class[edge_class(edge)].append(edge)
    for edge in candidates:
        pool_by_class[edge_class(edge)].append(edge)
    for key, target in target_by_class.items():
        if len(pool_by_class[key]) < len(target):
            raise RuntimeError(f"not enough dispersed edges in degree class {key}")

    rng = np.random.default_rng(int(seed))
    accepted: dict[tuple[tuple[int, int], ...], tuple[float, list[tuple[int, int]]]] = {}
    draws = max(int(n_controls) * int(candidates_per_control), int(n_controls))
    for _ in range(draws):
        chosen: list[tuple[int, int]] = []
        score = 0.0
        for key, target_edges in sorted(target_by_class.items()):
            pool = pool_by_class[key]
            available = list(pool)
            target_order = list(target_edges)
            rng.shuffle(target_order)
            for target_edge in target_order:
                distances = np.asarray([
                    abs(quantile[candidate] - quantile[target_edge]) for candidate in available
                ])
                # Randomize among near matches so the 32 controls are not clones.
                scale = max(0.03, float(np.quantile(distances, 0.25)))
                probability = np.exp(-distances / scale)
                probability /= probability.sum()
                picked_index = int(rng.choice(len(available), p=probability))
                picked = available.pop(picked_index)
                chosen.append(picked)
                score += abs(quantile[picked] - quantile[target_edge])
        chosen_key = tuple(sorted(chosen))
        if largest_edge_component(chosen_key) > int(max_component_nodes):
            continue
        score /= len(lesion_edges)
        prior = accepted.get(chosen_key)
        if prior is None or score < prior[0]:
            accepted[chosen_key] = (score, chosen)
    if len(accepted) < int(n_controls):
        raise RuntimeError(
            f"only {len(accepted)} dispersed controls passed; need {int(n_controls)}"
        )
    selected = sorted(accepted.values(), key=lambda item: item[0])[: int(n_controls)]
    masks: list[np.ndarray] = []
    audits: list[dict[str, float | int]] = []
    target_quantiles = sorted(quantile[edge] for edge in lesion_edges)
    for score, edges in selected:
        masks.append(edge_set_mask(graph.shape[0], edges))
        control_quantiles = sorted(quantile[edge] for edge in edges)
        audits.append({
            "n_undirected_edges": len(edges),
            "n_directed_edges": 2 * len(edges),
            "largest_component_nodes": largest_edge_component(edges),
            "weight_quantile_mean_absolute_error": float(np.mean(np.abs(
                np.asarray(target_quantiles) - np.asarray(control_quantiles)
            ))),
            "matching_score": float(score),
        })
    return masks, audits


def matched_dispersed_directed_edge_sets(
    recurrent_mask: np.ndarray,
    recurrent_weight: np.ndarray,
    lesion_edge_mask: np.ndarray,
    *,
    forbidden_nodes: Iterable[int],
    n_controls: int,
    seed: int,
    max_component_nodes: int = 4,
    candidates_per_control: int = 512,
) -> tuple[list[np.ndarray], list[dict[str, float | int]]]:
    """Match a directed lesion under the recurrent ``[target, source]`` convention."""
    graph = np.asarray(recurrent_mask, dtype=bool)
    lesion = np.asarray(lesion_edge_mask, dtype=bool)
    weight = np.asarray(recurrent_weight, dtype=float)
    if graph.shape != lesion.shape or graph.shape != weight.shape:
        raise ValueError("graph, lesion and recurrent_weight must align")
    forbidden = np.zeros(graph.shape[0], dtype=bool)
    forbidden[list(sorted({int(node) for node in forbidden_nodes}))] = True
    lesion_edges = [tuple(map(int, edge)) for edge in np.argwhere(graph & lesion)]
    candidates = [
        (int(target), int(source)) for target, source in np.argwhere(graph & ~lesion)
        if not forbidden[int(target)] and not forbidden[int(source)]
    ]
    if not lesion_edges:
        raise ValueError("directed lesion contains no recurrent edge")
    target_degree = graph.sum(axis=1).astype(int)
    source_degree = graph.sum(axis=0).astype(int)
    active_edges = [(int(target), int(source)) for target, source in np.argwhere(graph)]
    ordered = sorted(active_edges, key=lambda edge: (abs(float(weight[edge])), edge))
    quantile = {edge: index / max(1, len(ordered) - 1) for index, edge in enumerate(ordered)}

    def edge_class(edge: tuple[int, int]) -> tuple[int, int]:
        target, source = edge
        return int(source_degree[source]), int(target_degree[target])

    target_by_class: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    pool_by_class: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for edge in lesion_edges:
        target_by_class[edge_class(edge)].append(edge)
    for edge in candidates:
        pool_by_class[edge_class(edge)].append(edge)
    for key, target in target_by_class.items():
        if len(pool_by_class[key]) < len(target):
            raise RuntimeError(f"not enough directed controls in degree class {key}")

    rng = np.random.default_rng(int(seed))
    accepted: dict[tuple[tuple[int, int], ...], tuple[float, list[tuple[int, int]]]] = {}
    for _ in range(max(int(n_controls), int(n_controls) * int(candidates_per_control))):
        chosen: list[tuple[int, int]] = []
        score = 0.0
        for key, target_edges in sorted(target_by_class.items()):
            available = list(pool_by_class[key])
            target_order = list(target_edges)
            rng.shuffle(target_order)
            for target_edge in target_order:
                distances = np.asarray([
                    abs(quantile[candidate] - quantile[target_edge]) for candidate in available
                ])
                scale = max(0.03, float(np.quantile(distances, 0.25)))
                probability = np.exp(-distances / scale)
                probability /= probability.sum()
                picked_index = int(rng.choice(len(available), p=probability))
                picked = available.pop(picked_index)
                chosen.append(picked)
                score += abs(quantile[picked] - quantile[target_edge])
        key = tuple(sorted(chosen))
        undirected = [(min(target, source), max(target, source)) for target, source in key]
        if largest_edge_component(undirected) > int(max_component_nodes):
            continue
        score /= len(lesion_edges)
        prior = accepted.get(key)
        if prior is None or score < prior[0]:
            accepted[key] = (score, chosen)
    if len(accepted) < int(n_controls):
        raise RuntimeError(f"only {len(accepted)} directed controls passed; need {int(n_controls)}")
    selected = sorted(accepted.values(), key=lambda item: item[0])[: int(n_controls)]
    target_quantiles = sorted(quantile[edge] for edge in lesion_edges)
    masks: list[np.ndarray] = []
    audits: list[dict[str, float | int]] = []
    for score, edges in selected:
        mask = np.zeros_like(graph, dtype=np.uint8)
        for target, source in edges:
            mask[target, source] = 1
        masks.append(mask)
        control_quantiles = sorted(quantile[edge] for edge in edges)
        audits.append({
            "n_directed_edges": len(edges),
            "largest_component_nodes": largest_edge_component([
                (min(target, source), max(target, source)) for target, source in edges
            ]),
            "weight_quantile_mean_absolute_error": float(np.mean(np.abs(
                np.asarray(target_quantiles) - np.asarray(control_quantiles)
            ))),
            "matching_score": float(score),
        })
    return masks, audits
