"""Target-free graph and sequence contracts for Topic 5.1 LBSS-RNN v0.2.

The module deliberately contains no early-ictal reader.  It defines the
patient-specific local backbone, the extra-local/nonlocal candidate pools,
balanced proposal sampling, the rank-set derangement control, and the spatial
distance estimands used by every later trainer and scorer.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import torch
from torch import Tensor
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from src.topic5_wiring_economy_rnn import WEConfig, WEModel, active_edge_count


LBSS_ARMS = (
    "L0_LOCAL_ONLY",
    "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
    "L2_LOCAL_PLUS_RANDOM_LR",
    "L3_LOCAL_PLUS_LEARNED_LR",
    "C_L3_ORDER_SHUFFLED",
)


@dataclass(frozen=True)
class PoolContract:
    local_mask: np.ndarray
    extra_local_pool: np.ndarray
    nonlocal_pool: np.ndarray
    k_neighbors: int
    k_added: int
    r_local_mm: float
    target_local_edges: int


def _validate_distance(distance_mm: np.ndarray) -> np.ndarray:
    distance = np.asarray(distance_mm, dtype=float)
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("distance_mm must be a square matrix")
    if distance.shape[0] < 3:
        raise ValueError("LBSS needs at least three tissue nodes")
    if not np.isfinite(distance).all() or np.any(distance < 0):
        raise ValueError("distance_mm must be finite and non-negative")
    if not np.allclose(distance, distance.T, atol=1e-6, rtol=0):
        raise ValueError("distance_mm must be symmetric")
    if not np.allclose(np.diag(distance), 0.0, atol=1e-6, rtol=0):
        raise ValueError("distance_mm diagonal must be zero")
    return distance


def strong_component_audit(mask: np.ndarray, supported: np.ndarray | None = None) -> dict:
    """Audit directed reachability; symmetric masks should give one component."""
    directed = np.asarray(mask, dtype=bool)
    if directed.ndim != 2 or directed.shape[0] != directed.shape[1]:
        raise ValueError("mask must be square")
    n_components, labels = connected_components(
        csr_matrix(directed.astype(np.uint8)), directed=True, connection="strong"
    )
    support = (np.ones(directed.shape[0], dtype=bool) if supported is None
               else np.asarray(supported, dtype=bool))
    if support.shape != (directed.shape[0],):
        raise ValueError("supported must align to mask nodes")
    supported_labels = np.unique(labels[support]) if support.any() else np.array([], dtype=int)
    supported_one_component = bool(supported_labels.size == 1)
    return {
        "n_strong_components": int(n_components),
        "all_nodes_one_strong_component": bool(n_components == 1),
        "all_supported_nodes_one_strong_component": supported_one_component,
        "contact_supported_pairwise_reachability": float(supported_one_component),
        "minimum_in_degree": int(directed.sum(axis=0).min()),
        "minimum_out_degree": int(directed.sum(axis=1).min()),
    }


def symmetrized_knn_mask(distance_mm: np.ndarray, density: float = 0.10) -> tuple[np.ndarray, int]:
    """Choose the strongly-connected symmetrized-kNN mask nearest the edge budget.

    If either node regards the other as a k-nearest neighbour, both directed
    edges enter the mask.  Direction is therefore learned in the weights rather
    than imposed by the topology.
    """
    distance = _validate_distance(distance_mm)
    n = distance.shape[0]
    target = active_edge_count(n, density)
    candidates: list[tuple[int, int, np.ndarray]] = []
    stable_order = []
    for source in range(n):
        row = distance[source].copy()
        row[source] = np.inf
        stable_order.append(np.argsort(row, kind="stable"))
    for k in range(1, n):
        mask = np.zeros((n, n), dtype=bool)
        for source in range(n):
            for target_node in stable_order[source][:k]:
                mask[source, target_node] = True
                mask[target_node, source] = True
        np.fill_diagonal(mask, False)
        audit = strong_component_audit(mask)
        if audit["all_nodes_one_strong_component"]:
            candidates.append((abs(int(mask.sum()) - target), k, mask))
    if not candidates:
        raise RuntimeError("no strongly connected symmetrized-kNN mask exists")
    _, k, selected = min(candidates, key=lambda item: (item[0], item[1]))
    return selected.astype(np.uint8), int(k)


def build_pool_contract(
    distance_mm: np.ndarray,
    density: float = 0.10,
    added_fraction: float = 0.10,
) -> PoolContract:
    distance = _validate_distance(distance_mm)
    local, k_neighbors = symmetrized_knn_mask(distance, density=density)
    local_bool = local.astype(bool)
    local_lengths = distance[local_bool]
    if local_lengths.size == 0:
        raise RuntimeError("local backbone contains no edges")
    r_local = 2.0 * float(np.median(local_lengths))
    off = ~np.eye(distance.shape[0], dtype=bool)
    available = off & ~local_bool
    extra = available & (distance <= r_local)
    nonlocal_pool = available & (distance > r_local)
    k_added = max(1, int(round(float(added_fraction) * int(local_bool.sum()))))
    return PoolContract(
        local_mask=local.astype(np.uint8),
        extra_local_pool=extra.astype(np.uint8),
        nonlocal_pool=nonlocal_pool.astype(np.uint8),
        k_neighbors=k_neighbors,
        k_added=k_added,
        r_local_mm=r_local,
        target_local_edges=active_edge_count(distance.shape[0], density),
    )


def source_balanced_sample(pool_mask: np.ndarray, n_edges: int, seed: int) -> np.ndarray:
    """Sample directed candidates source-first without replacement.

    Recurrent matrices follow PyTorch's ``weight[target, source]`` convention,
    so a source is a column and a target is a row throughout LBSS.
    """
    pool = np.asarray(pool_mask, dtype=bool)
    if pool.ndim != 2 or pool.shape[0] != pool.shape[1]:
        raise ValueError("pool_mask must be square")
    if n_edges < 0 or n_edges > int(pool.sum()):
        raise ValueError("requested edge count exceeds candidate pool")
    rng = np.random.default_rng(seed)
    available = [list(np.flatnonzero(pool[:, source])) for source in range(pool.shape[1])]
    selected = np.zeros_like(pool)
    for _ in range(int(n_edges)):
        sources = np.array([source for source, targets in enumerate(available) if targets], dtype=int)
        if sources.size == 0:
            raise RuntimeError("candidate pool exhausted before reaching edge budget")
        source = int(rng.choice(sources))
        target_index = int(rng.integers(len(available[source])))
        target = int(available[source].pop(target_index))
        selected[target, source] = True
    return selected.astype(np.uint8)


@dataclass(frozen=True)
class LBSSConfig:
    arm: str
    n_contacts: int
    n_nodes: int
    observation_operator: np.ndarray
    node_distance_mm: np.ndarray
    local_mask: np.ndarray
    extra_local_pool: np.ndarray
    nonlocal_pool: np.ndarray
    k_added: int
    seed: int = 0
    state_dim: int = 1
    stop_hidden: int = 16
    new_edge_grace_intervals: int = 1

    def __post_init__(self) -> None:
        if self.arm not in LBSS_ARMS:
            raise ValueError(f"unknown LBSS arm {self.arm!r}")
        if int(self.k_added) < 1:
            raise ValueError("k_added must be positive")


class LBSSModel(WEModel):
    """Leaky tissue RNN with a fixed local mask and arm-specific added edges."""

    REWIRING_ARMS = {
        "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
        "L3_LOCAL_PLUS_LEARNED_LR",
        "C_L3_ORDER_SHUFFLED",
    }

    def __init__(self, config: LBSSConfig):
        # DENSE_TISSUE gives every arm the same parameter initialization before
        # we replace the mask.  No dense path survives this constructor.
        base = WEConfig(
            arm="DENSE_TISSUE",
            cell="rnn",
            n_contacts=int(config.n_contacts),
            n_nodes=int(config.n_nodes),
            state_dim=int(config.state_dim),
            density=0.10,
            eta=0.0,
            stop_hidden=int(config.stop_hidden),
            seed=int(config.seed),
            observation_operator=np.asarray(config.observation_operator, dtype=np.float32),
            node_distance_mm=np.asarray(config.node_distance_mm, dtype=np.float32),
        )
        super().__init__(base)
        self.lbss_config = config
        self.arm = config.arm
        self.config.arm = config.arm
        local = np.asarray(config.local_mask, dtype=np.uint8)
        extra = np.asarray(config.extra_local_pool, dtype=np.uint8)
        nonlocal_pool = np.asarray(config.nonlocal_pool, dtype=np.uint8)
        expected = (self.n_nodes, self.n_nodes)
        if local.shape != expected or extra.shape != expected or nonlocal_pool.shape != expected:
            raise ValueError("all LBSS graph arrays must align to n_nodes")
        if np.any(local & extra) or np.any(local & nonlocal_pool) or np.any(extra & nonlocal_pool):
            raise ValueError("local, extra-local and nonlocal masks must be disjoint")
        if config.arm == "L0_LOCAL_ONLY":
            added = np.zeros_like(local)
            pool = np.zeros_like(local)
        elif config.arm == "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL":
            pool = extra
            added = source_balanced_sample(pool, int(config.k_added), int(config.seed) + 1701)
        else:
            pool = nonlocal_pool
            # L2, L3 and order-shuffle deliberately share this exact mask.
            added = source_balanced_sample(pool, int(config.k_added), int(config.seed) + 2903)
        self.node_mask.copy_(torch.as_tensor((local | added).astype(np.float32)))
        self.initial_node_mask.copy_(self.node_mask)
        self.register_buffer("local_mask", torch.as_tensor(local.astype(np.float32)))
        self.register_buffer("added_mask", torch.as_tensor(added.astype(np.float32)))
        self.register_buffer("initial_added_mask", torch.as_tensor(added.astype(np.float32)))
        self.register_buffer("candidate_pool", torch.as_tensor(pool.astype(np.float32)))
        self.register_buffer("edge_age", torch.full(expected, -1, dtype=torch.int16))
        self.edge_age[self.added_mask > 0] = int(config.new_edge_grace_intervals)
        self.register_buffer("proposal_count", torch.zeros(expected, dtype=torch.int32))
        self.register_buffer("exposure_count", torch.zeros(expected, dtype=torch.int32))
        self.register_buffer("rewire_counter", torch.zeros((), dtype=torch.int64))
        self.mask_frozen = False

    def _refresh_node_mask(self) -> None:
        self.node_mask.copy_(((self.local_mask > 0) | (self.added_mask > 0)).float())

    @torch.no_grad()
    def rewire_added(self, zeta: float, rng: np.random.Generator) -> dict:
        """Rewire only arm-owned additions and preserve a one-interval grace."""
        if self.arm not in self.REWIRING_ARMS or self.mask_frozen or zeta <= 0:
            return {"n_drop": 0, "touched": torch.zeros_like(self.node_mask, dtype=torch.bool)}
        active = self.added_mask.detach().cpu().numpy().astype(bool)
        age = self.edge_age.detach().cpu().numpy()
        eligible = active & (age >= int(self.lbss_config.new_edge_grace_intervals))
        desired = int(round(float(zeta) * int(active.sum())))
        n_drop = min(desired, int(eligible.sum()))
        if n_drop < 1:
            self.edge_age[self.added_mask > 0] += 1
            self.rewire_counter += 1
            return {"n_drop": 0, "touched": torch.zeros_like(self.node_mask, dtype=torch.bool)}

        strength = self.edge_strength().detach().cpu().numpy()
        eligible_index = np.argwhere(eligible)
        order = np.argsort(strength[eligible], kind="stable")[:n_drop]
        dropped = eligible_index[order]
        for target, source in dropped:
            active[target, source] = False

        pool = self.candidate_pool.detach().cpu().numpy().astype(bool)
        available = pool & ~active
        grown: list[tuple[int, int]] = []
        for _ in range(n_drop):
            sources = np.flatnonzero(available.any(axis=0))
            if sources.size == 0:
                raise RuntimeError("LBSS candidate pool exhausted during rewiring")
            source = int(rng.choice(sources))
            targets = np.flatnonzero(available[:, source])
            # Exposure is an opportunity under the source-first proposal null.
            self.exposure_count[torch.as_tensor(targets), source] += 1
            target = int(rng.choice(targets))
            self.proposal_count[target, source] += 1
            available[target, source] = False
            active[target, source] = True
            grown.append((target, source))

        touched = torch.zeros_like(self.node_mask, dtype=torch.bool)
        for target, source in dropped:
            touched[target, source] = True
        for target, source in grown:
            touched[target, source] = True
        self.added_mask.copy_(torch.as_tensor(active, device=self.added_mask.device))
        self._refresh_node_mask()
        self.edge_age[self.edge_age >= 0] += 1
        for target, source in dropped:
            self.edge_age[target, source] = -1
        for target, source in grown:
            self.edge_age[target, source] = 0
        self.recurrent[:, touched] = 0.0
        self.rewire_counter += 1
        return {"n_drop": n_drop, "touched": touched}

    def freeze_mask(self) -> None:
        self.mask_frozen = True

    def graph_snapshot(self) -> dict:
        snapshot = super().graph_snapshot()
        snapshot.update({
            "local_mask": self.local_mask.detach().cpu().numpy().astype(np.uint8),
            "added_mask": self.added_mask.detach().cpu().numpy().astype(np.uint8),
            "initial_added_mask": self.initial_added_mask.detach().cpu().numpy().astype(np.uint8),
            "candidate_pool": self.candidate_pool.detach().cpu().numpy().astype(np.uint8),
            "edge_age": self.edge_age.detach().cpu().numpy().astype(np.int16),
            "proposal_count": self.proposal_count.detach().cpu().numpy().astype(np.int32),
            "exposure_count": self.exposure_count.detach().cpu().numpy().astype(np.int32),
            "rewire_counter": np.asarray(int(self.rewire_counter), dtype=np.int64),
            "mask_frozen": np.asarray(bool(self.mask_frozen)),
        })
        return snapshot

    def runtime_state(self) -> dict:
        return {
            "model": self.state_dict(),
            "mask_frozen": bool(self.mask_frozen),
        }

    def restore_runtime_state(self, state: dict) -> None:
        self.load_state_dict(state["model"])
        self.mask_frozen = bool(state["mask_frozen"])


def clear_recurrent_optimizer_state(
    model: LBSSModel,
    optimizer: torch.optim.Optimizer,
    touched_node_edges: Tensor,
) -> None:
    """Clear Adam moments when an edge is dropped or newly activated."""
    if not bool(touched_node_edges.any()):
        return
    expanded = touched_node_edges
    if model.state_dim > 1:
        expanded = torch.kron(
            expanded.float(),
            torch.ones(model.state_dim, model.state_dim, device=expanded.device),
        ).bool()
    state = optimizer.state.get(model.recurrent, {})
    for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        if name in state:
            state[name][:, expanded] = 0.0


def derange_rank_sets(ranks: np.ndarray, seed: int) -> tuple[np.ndarray, dict]:
    """Keep rank 0 and derange every event with at least two later rank sets."""
    source = np.asarray(ranks)
    if source.ndim != 2:
        raise ValueError("ranks must be event by contact")
    rng = np.random.default_rng(seed)
    output = source.copy()
    effectively_shuffled = 0
    unchanged_length_two = 0
    eligible_events = 0
    kendall_distances: list[float] = []
    for event, row in enumerate(source):
        later = np.unique(row[row > 0])
        if later.size < 2:
            if later.size == 1:
                unchanged_length_two += 1
            continue
        eligible_events += 1
        # A cyclic shift by a random non-zero offset is a uniform-enough
        # derangement for this null and guarantees no fixed rank position.
        shift = int(rng.integers(1, later.size))
        permuted = np.roll(later, shift)
        mapping = dict(zip(later.tolist(), permuted.tolist()))
        for old, new in mapping.items():
            output[event, row == old] = new
        effectively_shuffled += 1
        old_order = later.tolist()
        new_position = {int(label): index for index, label in enumerate(permuted.tolist())}
        discordant = 0
        pairs = 0
        for i in range(len(old_order)):
            for j in range(i + 1, len(old_order)):
                pairs += 1
                discordant += int(new_position[old_order[i]] > new_position[old_order[j]])
        kendall_distances.append(discordant / max(1, pairs))
    n_events = int(source.shape[0])
    return output, {
        "n_events": n_events,
        "n_eligible_derangements": eligible_events,
        "n_effectively_shuffled": effectively_shuffled,
        "n_unchanged_due_to_length_2": unchanged_length_two,
        "fraction_events_effectively_shuffled": effectively_shuffled / max(1, n_events),
        "fraction_events_unchanged_due_to_length_2": unchanged_length_two / max(1, n_events),
        "mean_kendall_distance_from_true_order": (
            float(np.mean(kendall_distances)) if kendall_distances else 0.0
        ),
    }


def transition_frontier_distance(
    current_contacts: Iterable[int],
    recruited_contacts: Iterable[int],
    next_contacts: Iterable[int],
    contact_xy_mm: np.ndarray,
) -> float:
    """Median distance from newly recruited contacts to the current frontier."""
    current = np.unique(np.asarray(list(current_contacts), dtype=int))
    recruited = set(np.asarray(list(recruited_contacts), dtype=int).tolist())
    next_set = np.unique(np.asarray(list(next_contacts), dtype=int))
    novel = np.asarray([contact for contact in next_set if int(contact) not in recruited], dtype=int)
    if current.size == 0 or novel.size == 0:
        return float("nan")
    xy = np.asarray(contact_xy_mm, dtype=float)
    distances = np.linalg.norm(xy[novel, None, :] - xy[current][None, :, :], axis=-1)
    return float(np.median(distances.min(axis=1)))


def semantic_snapshot_epochs(warmup_epochs: int, rewire_epochs: int) -> dict[str, int]:
    if warmup_epochs < 1 or rewire_epochs < 3:
        raise ValueError("snapshot contract needs warmup>=1 and rewire_epochs>=3")
    start = int(warmup_epochs)
    return {
        "SNAPSHOT_INIT": -1,
        "SNAPSHOT_AFTER_WARMUP": start - 1,
        "SNAPSHOT_REWIRE_1_3": start + int(np.ceil(rewire_epochs / 3)) - 1,
        "SNAPSHOT_REWIRE_2_3": start + int(np.ceil(2 * rewire_epochs / 3)) - 1,
        "SNAPSHOT_MASK_FREEZE": start + rewire_epochs - 1,
    }


def checkpoint_is_eligible(epoch: int, structural_phase_epoch: int) -> bool:
    return int(epoch) >= int(structural_phase_epoch)
