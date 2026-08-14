"""Graph-only utilities for the FCXR-LC6A patient-axis E->I surround probe.

The repository uses target-first matrices.  Biological E->I edges therefore
live in the inhibitory-target rows of ``net['ampa_by_delay']``.  This module
changes only those source identities and their physical delays; E->E, I->E,
I->I, target in-degree, and each target's incoming weight multiset remain
unchanged.

No runtime/noise RNG is accepted here.  Every stochastic function constructs a
private graph RNG from an explicit seed so graph generation cannot consume the
SNN external-input stream.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

import numpy as np
from scipy import sparse, stats


@dataclass(frozen=True)
class EToIGraph:
    """Canonical target-major biological E->I edge table."""

    sources: np.ndarray       # (NI, C), E-local source ids, sorted per target
    weights: np.ndarray       # (NI, C), target-first AMPA jump weights
    delay_steps: np.ndarray   # (NI, C), engine dt-step delays

    def __post_init__(self) -> None:
        shape = np.asarray(self.sources).shape
        if len(shape) != 2 or np.asarray(self.weights).shape != shape:
            raise ValueError("sources and weights must share a two-dimensional shape")
        if np.asarray(self.delay_steps).shape != shape:
            raise ValueError("sources and delay_steps must share a two-dimensional shape")

    @property
    def n_targets(self) -> int:
        return int(self.sources.shape[0])

    @property
    def in_degree(self) -> int:
        return int(self.sources.shape[1])


def _canonical_target_table(rows, cols, values, delay_steps, *, ne, ni) -> EToIGraph:
    rows = np.asarray(rows, dtype=np.int64) - int(ne)
    cols = np.asarray(cols, dtype=np.int64)
    values = np.asarray(values)
    delay_steps = np.asarray(delay_steps, dtype=np.int64)
    if rows.size == 0 or np.any(rows < 0) or np.any(rows >= ni):
        raise ValueError("edge table does not contain a legal E-to-I population direction")
    counts = np.bincount(rows, minlength=ni)
    if counts.size != ni or not np.all(counts == counts[0]):
        raise ValueError("E-to-I target in-degree is not constant")
    degree = int(counts[0])
    order = np.lexsort((cols, rows))
    sources = cols[order].reshape(ni, degree)
    weights = values[order].reshape(ni, degree)
    delays = delay_steps[order].reshape(ni, degree)
    if np.any(np.diff(sources, axis=1) == 0):
        raise ValueError("duplicate E-to-I source within an inhibitory target")
    return EToIGraph(sources.astype(np.int32), weights.copy(), delays.astype(np.int32))


def extract_e_to_i(net, ne: int, ni: int) -> EToIGraph:
    """Extract biological E->I edges from target-first AMPA delay matrices."""

    rows, cols, values, steps = [], [], [], []
    for delay_step, matrix in enumerate(net["ampa_by_delay"]):
        coo = matrix[ne:ne + ni, :ne].tocoo(copy=False)
        if coo.nnz == 0:
            continue
        rows.append(coo.row.astype(np.int64) + ne)
        cols.append(coo.col.astype(np.int64))
        values.append(np.asarray(coo.data))
        steps.append(np.full(coo.nnz, delay_step, dtype=np.int64))
    if not rows:
        raise ValueError("network contains no biological E-to-I edges")
    return _canonical_target_table(
        np.concatenate(rows), np.concatenate(cols), np.concatenate(values),
        np.concatenate(steps), ne=ne, ni=ni,
    )


def _extract_target_major(matrices, *, row_start: int, n_target: int, n_source: int) -> EToIGraph:
    rows, cols, values, steps = [], [], [], []
    for delay_step, matrix in enumerate(matrices):
        coo = matrix[row_start:row_start + n_target, :n_source].tocoo(copy=False)
        if coo.nnz == 0:
            continue
        rows.append(coo.row.astype(np.int64) + row_start)
        cols.append(coo.col.astype(np.int64))
        values.append(np.asarray(coo.data))
        steps.append(np.full(coo.nnz, delay_step, dtype=np.int64))
    if not rows:
        raise ValueError("requested population direction has no edges")
    return _canonical_target_table(
        np.concatenate(rows), np.concatenate(cols), np.concatenate(values),
        np.concatenate(steps), ne=row_start, ni=n_target,
    )


def extract_e_to_e(net, ne: int) -> EToIGraph:
    """Extract E-source/E-target edges for the frozen patient-axis reference."""

    return _extract_target_major(
        net["ampa_by_delay"], row_start=0, n_target=ne, n_source=ne,
    )


def extract_i_to_e(net, ne: int, ni: int) -> EToIGraph:
    """Extract biological I->E edges from target-first GABA matrices."""

    return _extract_target_major(
        net["gaba_by_delay"], row_start=0, n_target=ne, n_source=ni,
    )


def graph_sha256(graph: EToIGraph) -> str:
    digest = hashlib.sha256()
    for name, array in (
        ("sources", graph.sources), ("weights", graph.weights),
        ("delay_steps", graph.delay_steps),
    ):
        a = np.ascontiguousarray(array)
        digest.update(name.encode("ascii"))
        digest.update(str(a.dtype).encode("ascii"))
        digest.update(np.asarray(a.shape, dtype=np.int64).tobytes())
        digest.update(a.tobytes())
    return digest.hexdigest()


def axis_unit(axis: Iterable[float]) -> np.ndarray:
    unit = np.asarray(axis, dtype=float)
    if unit.shape != (2,) or not np.all(np.isfinite(unit)):
        raise ValueError("patient axis must be a finite two-vector")
    norm = float(np.linalg.norm(unit))
    if norm <= 0.0:
        raise ValueError("patient axis has zero length")
    return unit / norm


def projected_offsets(pos_source, pos_target, axis) -> tuple[np.ndarray, np.ndarray]:
    unit = axis_unit(axis)
    perpendicular = np.asarray([-unit[1], unit[0]])
    delta = np.asarray(pos_source, float) - np.asarray(pos_target, float)
    return delta @ unit, delta @ perpendicular


def empirical_edge_widths(
    sources: np.ndarray, source_positions: np.ndarray,
    target_positions: np.ndarray, axis, *, chunk_targets: int = 256,
) -> dict:
    """Unweighted displacement covariance widths without a giant edge array."""

    sources = np.asarray(sources)
    if sources.ndim != 2 or sources.shape[0] != len(target_positions):
        raise ValueError("sources must be target-major and align to target_positions")
    count = 0
    sum_parallel = sum_perpendicular = 0.0
    sumsq_parallel = sumsq_perpendicular = 0.0
    unit = axis_unit(axis)
    perpendicular = np.asarray([-unit[1], unit[0]])
    for start in range(0, sources.shape[0], int(chunk_targets)):
        stop = min(sources.shape[0], start + int(chunk_targets))
        src = source_positions[sources[start:stop]]
        tgt = target_positions[start:stop, None, :]
        delta = src - tgt
        parallel = delta @ unit
        across = delta @ perpendicular
        count += parallel.size
        sum_parallel += float(parallel.sum())
        sum_perpendicular += float(across.sum())
        sumsq_parallel += float(np.square(parallel).sum())
        sumsq_perpendicular += float(np.square(across).sum())
    mean_parallel = sum_parallel / count
    mean_perpendicular = sum_perpendicular / count
    var_parallel = max(0.0, sumsq_parallel / count - mean_parallel ** 2)
    var_perpendicular = max(0.0, sumsq_perpendicular / count - mean_perpendicular ** 2)
    return {
        "n_edges": int(count),
        "mean_parallel_mm": mean_parallel,
        "mean_perpendicular_mm": mean_perpendicular,
        "sigma_parallel_mm": float(np.sqrt(var_parallel)),
        "sigma_perpendicular_mm": float(np.sqrt(var_perpendicular)),
    }


def construction_q(e_to_i_width: dict, i_to_e_width: dict, e_to_e_width: dict) -> float:
    denominator = float(e_to_e_width["sigma_parallel_mm"])
    if denominator <= 0.0:
        raise ValueError("E-to-E patient-axis width must be positive")
    numerator = np.hypot(
        float(e_to_i_width["sigma_parallel_mm"]),
        float(i_to_e_width["sigma_parallel_mm"]),
    )
    return float(numerator / denominator)


def elliptical_log_weight(pos_e, pos_i, axis, l_parallel, l_perpendicular) -> np.ndarray:
    if l_parallel <= 0.0 or l_perpendicular <= 0.0:
        raise ValueError("elliptical kernel widths must be positive")
    d_parallel, d_perpendicular = projected_offsets(pos_e, pos_i, axis)
    return -np.sqrt(
        np.square(d_parallel / float(l_parallel))
        + np.square(d_perpendicular / float(l_perpendicular))
    )


def metropolis_hastings_acceptance(
    log_target_old: float, log_target_new: float,
    *, log_q_reverse: float = 0.0, log_q_forward: float = 0.0,
) -> float:
    """Acceptance including Hastings correction; symmetric proposals use q=1."""

    log_ratio = (
        float(log_target_new) - float(log_target_old)
        + float(log_q_reverse) - float(log_q_forward)
    )
    return float(min(1.0, np.exp(min(0.0, log_ratio))))


def _uniform_nonmember(rng, selected, n_source: int) -> int:
    # Rejection from the complete population is exactly uniform over nonmembers.
    while True:
        candidate = int(rng.integers(n_source))
        if not selected[candidate]:
            return candidate


def _uniform_nonmember_block(rng, selected, n_source: int, size: int) -> np.ndarray:
    chosen = []
    chosen_set = set()
    while len(chosen) < int(size):
        candidate = _uniform_nonmember(rng, selected, n_source)
        if candidate not in chosen_set:
            chosen.append(candidate)
            chosen_set.add(candidate)
    return np.asarray(chosen, dtype=np.int32)


def _weighted_nonmember_from_cdf(rng, pool, cdf, selected) -> int:
    if cdf.size == 0 or not np.isfinite(cdf[-1]) or cdf[-1] <= 0.0:
        raise RuntimeError("proposal bin has no positive-weight source")
    while True:
        index = int(np.searchsorted(cdf, rng.random() * cdf[-1], side="right"))
        index = min(index, len(pool) - 1)
        candidate = int(pool[index])
        if not selected[candidate]:
            return candidate


def rewire_e_to_i_targetwise(
    base_sources: np.ndarray,
    pos_e: np.ndarray,
    pos_i: np.ndarray,
    axis,
    *,
    l_parallel: float,
    l_perpendicular: float,
    graph_seed: int,
    n_sweeps: int,
    proposal_block_size: int = 1,
    proposal_perpendicular_bin_mm: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Target-wise source-replacement MCMC from the same C0 graph.

    The default proposal is symmetric.  The optional perpendicular-stratified
    proposal preserves each target's coarse perpendicular edge counts, draws a
    nonmember from the desired elliptical kernel, and applies the exact
    forward/reverse nonmember-normalization Hastings correction.
    """

    base = np.asarray(base_sources, dtype=np.int32)
    if base.ndim != 2 or base.shape[0] != len(pos_i):
        raise ValueError("base_sources must align to inhibitory targets")
    if n_sweeps < 1:
        raise ValueError("n_sweeps must be positive")
    if proposal_block_size < 1 or proposal_block_size > base.shape[1]:
        raise ValueError("proposal_block_size must lie within [1, in_degree]")
    if proposal_perpendicular_bin_mm is not None:
        if proposal_perpendicular_bin_mm <= 0.0:
            raise ValueError("proposal_perpendicular_bin_mm must be positive")
        if proposal_block_size != 1:
            raise ValueError("stratified asymmetric proposal currently requires block size one")
    ne = int(len(pos_e))
    if base.shape[1] >= ne:
        raise ValueError("source replacement needs at least one nonmember")
    rng = np.random.default_rng(int(graph_seed))
    result = base.copy()
    accepted_blocks = proposed_blocks = 0
    accepted_edges = proposed_edges = 0
    selected = np.zeros(ne, dtype=bool)
    base_selected = np.zeros(ne, dtype=bool)
    accepted_blocks_by_sweep = np.zeros(int(n_sweeps), dtype=np.int64)
    proposed_blocks_by_sweep = np.zeros(int(n_sweeps), dtype=np.int64)
    changed_by_sweep = np.zeros(int(n_sweeps), dtype=np.int64)
    for target in range(result.shape[0]):
        current = result[target]
        selected.fill(False)
        selected[current] = True
        base_selected.fill(False)
        base_selected[base[target]] = True
        # The target geometry is fixed across sweeps; compute its complete
        # candidate energy once rather than once per sweep.
        logw = elliptical_log_weight(
            pos_e, pos_i[target], axis, l_parallel, l_perpendicular,
        )
        target_weight = np.exp(logw - float(np.max(logw)))
        pools = pool_cdfs = pool_totals = source_bins = None
        selected_weight_by_bin = None
        selected_count_by_bin = None
        if proposal_perpendicular_bin_mm is not None:
            _, d_perpendicular = projected_offsets(pos_e, pos_i[target], axis)
            source_bins = np.floor(
                d_perpendicular / float(proposal_perpendicular_bin_mm)
            ).astype(np.int32)
            pools, pool_cdfs, pool_totals = {}, {}, {}
            # Sort once and slice contiguous bin segments.  Re-scanning all E
            # sources once per bin would multiply the 40k build cost by ~80.
            bin_order = np.argsort(source_bins, kind="stable")
            sorted_bins = source_bins[bin_order]
            unique_bins, starts = np.unique(sorted_bins, return_index=True)
            stops = np.concatenate((starts[1:], [len(bin_order)]))
            for bin_id, start, stop in zip(unique_bins, starts, stops):
                pool = bin_order[start:stop].astype(np.int32, copy=False)
                pools[int(bin_id)] = pool
                cdf = np.cumsum(target_weight[pool])
                pool_cdfs[int(bin_id)] = cdf
                pool_totals[int(bin_id)] = float(cdf[-1])
            selected_weight_by_bin = {
                int(bin_id): float(target_weight[current[source_bins[current] == bin_id]].sum())
                for bin_id in np.unique(source_bins[current])
            }
            selected_count_by_bin = {
                int(bin_id): int(np.count_nonzero(source_bins[current] == bin_id))
                for bin_id in np.unique(source_bins[current])
            }
        for sweep in range(int(n_sweeps)):
            permutation = rng.permutation(current.size)
            for start in range(0, current.size, int(proposal_block_size)):
                slots = permutation[start:start + int(proposal_block_size)]
                old = current[slots].copy()
                if proposal_perpendicular_bin_mm is None:
                    new = _uniform_nonmember_block(rng, selected, ne, slots.size)
                    log_q_forward = log_q_reverse = 0.0
                else:
                    bin_id = int(source_bins[int(old[0])])
                    if selected_count_by_bin[bin_id] >= len(pools[bin_id]):
                        # This coarse perpendicular stratum is fully occupied;
                        # leaving the edge unchanged is the only legal move.
                        continue
                    candidate = _weighted_nonmember_from_cdf(
                        rng, pools[bin_id], pool_cdfs[bin_id], selected,
                    )
                    new = np.asarray([candidate], dtype=np.int32)
                    z_forward = pool_totals[bin_id] - selected_weight_by_bin[bin_id]
                    z_reverse = z_forward - target_weight[candidate] + target_weight[int(old[0])]
                    if min(z_forward, z_reverse) <= 0.0:
                        raise RuntimeError("invalid nonmember proposal normalization")
                    log_q_forward = float(logw[candidate] - np.log(z_forward))
                    log_q_reverse = float(logw[int(old[0])] - np.log(z_reverse))
                proposed_blocks += 1
                proposed_edges += slots.size
                proposed_blocks_by_sweep[sweep] += 1
                acceptance = metropolis_hastings_acceptance(
                    float(logw[old].sum()), float(logw[new].sum()),
                    log_q_reverse=log_q_reverse, log_q_forward=log_q_forward,
                )
                if rng.random() < acceptance:
                    selected[old] = False
                    selected[new] = True
                    current[slots] = new
                    if proposal_perpendicular_bin_mm is not None:
                        selected_weight_by_bin[bin_id] += (
                            target_weight[candidate] - target_weight[int(old[0])]
                        )
                    accepted_blocks += 1
                    accepted_edges += slots.size
                    accepted_blocks_by_sweep[sweep] += 1
            changed_by_sweep[sweep] += int(current.size - np.count_nonzero(base_selected[current]))
    acceptance_by_sweep = (
        accepted_blocks_by_sweep / np.maximum(1, proposed_blocks_by_sweep)
    ).astype(float).tolist()
    hamming_by_sweep = (
        changed_by_sweep / float(result.shape[0] * result.shape[1])
    ).astype(float).tolist()
    result.sort(axis=1)
    if np.any(np.diff(result, axis=1) == 0):
        raise RuntimeError("MCMC produced a duplicate E source")
    return result, {
        "proposal": (
            "perpendicular_stratified_target_weighted_nonmember_asymmetric"
            if proposal_perpendicular_bin_mm is not None
            else "uniform_slot_block_uniform_nonmember_block_symmetric"
        ),
        "hastings_correction": (
            "full_target_and_forward_reverse_nonmember_normalizers"
            if proposal_perpendicular_bin_mm is not None
            else "included_general_form_symmetric_terms_zero"
        ),
        "graph_seed": int(graph_seed),
        "n_sweeps": int(n_sweeps),
        "proposal_block_size": int(proposal_block_size),
        "proposal_perpendicular_bin_mm": proposal_perpendicular_bin_mm,
        "n_proposed_blocks": int(proposed_blocks),
        "n_accepted_blocks": int(accepted_blocks),
        "n_proposed_edges": int(proposed_edges),
        "n_accepted_edges": int(accepted_edges),
        "acceptance_by_sweep": acceptance_by_sweep,
        "hamming_by_sweep": hamming_by_sweep,
    }


def assign_frozen_target_weights(
    base_weights: np.ndarray, new_sources: np.ndarray, *, graph_seed: int,
) -> np.ndarray:
    """Preserve each target's incoming weight multiset without distance fitting."""

    base = np.asarray(base_weights)
    sources = np.asarray(new_sources)
    if base.shape != sources.shape:
        raise ValueError("weight and source tables must have the same shape")
    out = np.empty_like(base)
    for target in range(base.shape[0]):
        if np.all(base[target] == base[target, 0]):
            out[target].fill(base[target, 0])
            continue
        seed = np.random.SeedSequence([int(graph_seed), int(target), 0x6C36])
        permutation = np.random.default_rng(seed).permutation(base.shape[1])
        out[target] = base[target, permutation]
    return out


def recompute_physical_delays(
    new_sources: np.ndarray, pos_e: np.ndarray, pos_i: np.ndarray,
    *, tau0_ms: float, v_axon_mm_per_ms: float,
    delay_dt_ms: float, engine_dt_ms: float,
) -> np.ndarray:
    """Apply the frozen tau0+d/v rule and the engine's delay quantization."""

    if min(v_axon_mm_per_ms, delay_dt_ms, engine_dt_ms) <= 0.0:
        raise ValueError("delay scales must be positive")
    sources = np.asarray(new_sources)
    distance = np.linalg.norm(pos_e[sources] - pos_i[:, None, :], axis=2)
    physical = float(tau0_ms) + distance / float(v_axon_mm_per_ms)
    stride = max(1, int(round(float(delay_dt_ms) / float(engine_dt_ms))))
    return (
        np.maximum(1, np.round(physical / float(delay_dt_ms)).astype(np.int64)) * stride
    ).astype(np.int32)


def make_rewired_graph(
    base: EToIGraph, new_sources: np.ndarray, pos_e, pos_i, *, graph_seed: int,
    tau0_ms: float, v_axon_mm_per_ms: float, delay_dt_ms: float, engine_dt_ms: float,
) -> EToIGraph:
    weights = assign_frozen_target_weights(base.weights, new_sources, graph_seed=graph_seed)
    delay_steps = recompute_physical_delays(
        new_sources, pos_e, pos_i, tau0_ms=tau0_ms,
        v_axon_mm_per_ms=v_axon_mm_per_ms, delay_dt_ms=delay_dt_ms,
        engine_dt_ms=engine_dt_ms,
    )
    return EToIGraph(np.asarray(new_sources, np.int32), weights, delay_steps)


def source_outdegree_audit(sources, pos_e, axis, *, sheet_size_mm: float, edge_margin_mm: float) -> dict:
    sources = np.asarray(sources)
    outdegree = np.bincount(sources.ravel(), minlength=len(pos_e)).astype(float)
    unit = axis_unit(axis)
    perpendicular = np.asarray([-unit[1], unit[0]])
    centered = np.asarray(pos_e, float) - float(sheet_size_mm) / 2.0
    along = centered @ unit
    across = centered @ perpendicular
    distance_to_edge = np.min(
        np.column_stack((pos_e, float(sheet_size_mm) - np.asarray(pos_e))), axis=1,
    )
    interior = distance_to_edge >= float(edge_margin_mm)
    edge = ~interior
    if not np.any(interior) or not np.any(edge):
        raise ValueError("edge margin does not produce both interior and edge cells")
    mean = float(outdegree.mean())
    return {
        "mean": mean,
        "cv": float(outdegree.std() / mean),
        "q95": float(np.quantile(outdegree, .95)),
        "q99": float(np.quantile(outdegree, .99)),
        "interior_mean": float(outdegree[interior].mean()),
        "edge_mean": float(outdegree[edge].mean()),
        "interior_edge_ratio": float(outdegree[interior].mean() / outdegree[edge].mean()),
        "spearman_parallel": float(stats.spearmanr(outdegree, along).statistic),
        "spearman_perpendicular": float(stats.spearmanr(outdegree, across).statistic),
    }


def compare_outdegree_to_c0(candidate: dict, c0: dict, *, relative_tolerance: float = .10) -> dict:
    relative = {}
    for key in ("cv", "q95", "q99", "interior_edge_ratio"):
        denominator = abs(float(c0[key]))
        relative[key] = abs(float(candidate[key]) - float(c0[key])) / max(denominator, 1e-12)
    correlation_delta = {
        key: abs(float(candidate[key]) - float(c0[key]))
        for key in ("spearman_parallel", "spearman_perpendicular")
    }
    passed = (
        all(value <= float(relative_tolerance) for value in relative.values())
        and all(value <= .10 for value in correlation_delta.values())
    )
    return {
        "relative_difference": relative,
        "spearman_absolute_difference": correlation_delta,
        "within_contract": bool(passed),
    }


def replace_e_to_i_in_net(net, graph: EToIGraph, *, ne: int, ni: int) -> dict:
    """Return a shallow net copy with only biological E->I AMPA edges replaced."""

    if graph.n_targets != ni:
        raise ValueError("candidate graph target count does not match network NI")
    ee_rows, ee_cols, ee_data, ee_steps = [], [], [], []
    for delay_step, matrix in enumerate(net["ampa_by_delay"]):
        coo = matrix[:ne, :ne].tocoo(copy=False)
        if coo.nnz:
            ee_rows.append(coo.row.astype(np.int64))
            ee_cols.append(coo.col.astype(np.int64))
            ee_data.append(np.asarray(coo.data))
            ee_steps.append(np.full(coo.nnz, delay_step, dtype=np.int64))
    target = np.repeat(np.arange(ni, dtype=np.int64) + ne, graph.in_degree)
    source = graph.sources.ravel().astype(np.int64)
    weight = graph.weights.ravel()
    delay = graph.delay_steps.ravel().astype(np.int64)
    rows = np.concatenate([*ee_rows, target])
    cols = np.concatenate([*ee_cols, source])
    data = np.concatenate([*ee_data, weight])
    steps = np.concatenate([*ee_steps, delay])
    max_step = int(max(steps.max(), len(net["gaba_by_delay"]) - 1))
    order = np.argsort(steps, kind="stable")
    rows, cols, data, steps = rows[order], cols[order], data[order], steps[order]
    unique, starts = np.unique(steps, return_index=True)
    boundaries = list(starts) + [len(steps)]
    matrices = [sparse.csc_matrix((ne + ni, ne)) for _ in range(max_step + 1)]
    for index, delay_step in enumerate(unique):
        sl = slice(boundaries[index], boundaries[index + 1])
        matrices[int(delay_step)] = sparse.csc_matrix(
            (data[sl], (rows[sl], cols[sl])), shape=(ne + ni, ne),
        )
    updated = dict(net)
    updated["ampa_by_delay"] = matrices
    if len(updated["gaba_by_delay"]) < max_step + 1:
        empty = sparse.csc_matrix((ne + ni, ni))
        updated["gaba_by_delay"] = list(updated["gaba_by_delay"]) + [
            empty for _ in range(max_step + 1 - len(updated["gaba_by_delay"]))
        ]
    updated["max_delay_steps"] = max_step
    return updated


def validate_q_target(q_observed: float, q_target: float, tolerance: float) -> None:
    if not np.isfinite(q_observed):
        raise RuntimeError("non-finite construction q")
    if abs(float(q_observed) - float(q_target)) > float(tolerance):
        raise RuntimeError(
            f"construction q target unreachable: observed={q_observed:.6f}, "
            f"target={q_target:.6f}, tolerance={tolerance:.6f}"
        )


def audit_basic_legality(base: EToIGraph, candidate: EToIGraph, *, ne: int) -> dict:
    if candidate.sources.shape != base.sources.shape:
        raise RuntimeError("target in-degree or target count changed")
    if np.any(candidate.sources < 0) or np.any(candidate.sources >= int(ne)):
        raise RuntimeError("candidate contains a non-E source")
    if np.any(np.diff(candidate.sources, axis=1) == 0):
        raise RuntimeError("candidate contains duplicate E-to-I edges")
    weight_multiset_exact = all(
        np.array_equal(np.sort(base.weights[i]), np.sort(candidate.weights[i]))
        for i in range(base.n_targets)
    )
    if not weight_multiset_exact:
        raise RuntimeError("per-target incoming weight multiset changed")
    return {
        "population_direction": "E_to_I_code_IE",
        "n_targets": candidate.n_targets,
        "target_in_degree": candidate.in_degree,
        "n_edges": int(candidate.sources.size),
        "duplicates": 0,
        "per_target_weight_multiset_exact": True,
        "graph_sha256": graph_sha256(candidate),
    }
