"""Runtime helpers shared by Phase-D parity, calibration and carrier forks."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree


def git_sha(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def rescale_i2e_delay_bins(
    net: dict,
    state: dict,
    *,
    n_e: int,
    scale: float,
    source_delay_cv: float = 0.0,
    source_delay_seed: int = 0,
):
    """Delay only GABA arrivals onto E cells and preserve in-flight arrivals.

    E->E weights and all graph edges remain unchanged.  Existing ring contents
    keep their original future arrival offset; only emissions after the branch
    point use the rescaled I->E delay bins.
    """
    scale = float(scale)
    source_delay_cv = float(source_delay_cv)
    if not np.isfinite(scale) or scale < 1.0:
        raise ValueError("i2e delay scale must be finite and >=1")
    if not np.isfinite(source_delay_cv) or source_delay_cv < 0.0:
        raise ValueError("i2e source delay CV must be finite and >=0")
    if np.isclose(scale, 1.0) and np.isclose(source_delay_cv, 0.0):
        return net, state, {"scale": 1.0, "changed": False}
    old_max = int(net["max_delay_steps"])
    old_m = old_max + 1
    gaba = net["gaba_by_delay"]
    ampa = net["ampa_by_delay"]
    n, n_i = gaba[0].shape
    if np.isclose(source_delay_cv, 0.0):
        source_factors = np.ones(n_i, dtype=float)
    else:
        sigma = np.sqrt(np.log1p(source_delay_cv ** 2))
        rng = np.random.default_rng(int(source_delay_seed))
        source_factors = np.exp(
            sigma * rng.standard_normal(n_i) - 0.5 * sigma ** 2
        )
        source_factors /= np.mean(source_factors)
    row_e = np.zeros(n, dtype=float); row_e[: int(n_e)] = 1.0
    row_i = 1.0 - row_e
    prospective_delays = [old_max]
    for d, matrix in enumerate(gaba):
        if not matrix.nnz:
            continue
        coo = matrix.tocoo()
        to_e = coo.row < int(n_e)
        if np.any(to_e):
            prospective_delays.extend(
                np.maximum(
                    1, np.rint(d * scale * source_factors[coo.col[to_e]]).astype(int)
                ).tolist()
            )
    new_max = max(prospective_delays)
    zero_g = sparse.csc_matrix((n, n_i), dtype=float)
    g_new = [zero_g.copy() for _ in range(new_max + 1)]
    for d, matrix in enumerate(gaba):
        if not matrix.nnz:
            continue
        to_e = matrix.multiply(row_e[:, None]).tocsc()
        to_i = matrix.multiply(row_i[:, None]).tocsc()
        if to_i.nnz:
            g_new[d] = (g_new[d] + to_i).tocsc()
        if to_e.nnz:
            coo = to_e.tocoo()
            new_delays = np.maximum(
                1, np.rint(d * scale * source_factors[coo.col]).astype(int)
            )
            for new_d in np.unique(new_delays):
                keep = new_delays == new_d
                moved = sparse.csc_matrix(
                    (coo.data[keep], (coo.row[keep], coo.col[keep])),
                    shape=coo.shape,
                )
                g_new[int(new_d)] = (g_new[int(new_d)] + moved).tocsc()
    zero_a = sparse.csc_matrix(ampa[0].shape, dtype=float)
    a_new = list(ampa) + [zero_a.copy() for _ in range(new_max + 1 - len(ampa))]
    net_new = dict(net)
    net_new.update(
        ampa_by_delay=a_new,
        gaba_by_delay=g_new,
        max_delay_steps=new_max,
    )
    state_new = dict(state)
    t_abs = int(np.asarray(state["t"]))
    for key in ("ring_sE", "ring_sI"):
        old_ring = np.asarray(state[key])
        if old_ring.shape[0] != old_m:
            raise ValueError(f"{key} has {old_ring.shape[0]} bins, expected {old_m}")
        new_ring = np.zeros((new_max + 1, old_ring.shape[1]), dtype=old_ring.dtype)
        for delta in range(old_m):
            new_ring[(t_abs + delta) % (new_max + 1)] += old_ring[(t_abs + delta) % old_m]
        state_new[key] = new_ring
    factor_percentiles = np.percentile(source_factors, [5, 25, 50, 75, 95])
    occupied_i2e_bins = [
        d for d, matrix in enumerate(g_new)
        if matrix[: int(n_e), :].nnz
    ]
    return net_new, state_new, {
        "scale": scale,
        "source_delay_cv_requested": source_delay_cv,
        "source_delay_cv_realized": float(np.std(source_factors) / np.mean(source_factors)),
        "source_delay_seed": int(source_delay_seed),
        "source_factor_percentiles_5_25_50_75_95": factor_percentiles.tolist(),
        "occupied_i2e_delay_bins": occupied_i2e_bins,
        "changed": True,
        "old_max_delay_steps": old_max,
        "new_max_delay_steps": new_max,
        "edges_unchanged": True,
        "existing_arrival_offsets_preserved": True,
    }


def build_dual_scale_i2e_gaba(
    net: dict,
    state: dict,
    *,
    n_e: int,
    slow_fraction: float,
    broad_sigma_mm: float,
    broad_in_degree: int,
    broad_candidate_count: int,
    seed: int,
    dt_ms: float,
    delay_dt_ms: float,
    tau0_ms: float,
    v_axon_mm_per_ms: float,
    tau_r_fast_ms: float,
    tau_r_slow_ms: float,
    tau_d_slow_ms: float,
):
    """Split the I->E inhibition budget into local-fast and broad-slow paths.

    Existing E->E/AMPA and I->I edges are untouched.  For each E target, the
    local I->E jump budget is multiplied by ``1-slow_fraction`` and a new broad
    set of I sources receives the complementary *integrated* budget.  The slow
    jump is rescaled by ``tau_r_fast/tau_r_slow`` because connectivity weights
    encode the per-spike jump onto the synaptic rise variable.

    Broad partners are sampled deterministically from a spatial candidate set
    with an exponential kernel.  No simulator RNG state is consumed.
    """
    frac = float(slow_fraction)
    sigma = float(broad_sigma_mm)
    c_in = int(broad_in_degree)
    c_cand = int(broad_candidate_count)
    scalars = (
        dt_ms, delay_dt_ms, tau0_ms, v_axon_mm_per_ms,
        tau_r_fast_ms, tau_r_slow_ms, tau_d_slow_ms,
    )
    if not (0.0 < frac < 1.0):
        raise ValueError("dual GABA slow_fraction must lie strictly between 0 and 1")
    if not np.all(np.isfinite(scalars)) or min(scalars) <= 0.0:
        raise ValueError("dual GABA time and velocity parameters must be finite and positive")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("dual GABA broad_sigma_mm must be finite and positive")
    if c_in < 1 or c_cand < c_in:
        raise ValueError("dual GABA requires broad_candidate_count >= broad_in_degree >= 1")
    if "gaba_slow_by_delay" in net:
        raise ValueError("network already has a dual GABA slow channel")

    gaba = net["gaba_by_delay"]
    ampa = net["ampa_by_delay"]
    n, n_i = gaba[0].shape
    if int(n_e) <= 0 or int(n_e) >= n or n_i != n - int(n_e):
        raise ValueError("dual GABA population dimensions are inconsistent")
    pos = np.asarray(net["pos"], dtype=float)
    pos_e, pos_i = pos[: int(n_e)], pos[int(n_e) :]

    # Original per-target jump budget, before the local path is scaled.
    original = sparse.csc_matrix((n, n_i), dtype=float)
    for matrix in gaba:
        original = (original + matrix).tocsc()
    target_budget = np.asarray(original[: int(n_e), :].sum(axis=1)).ravel()

    row_e = np.zeros(n, dtype=float)
    row_e[: int(n_e)] = 1.0
    row_i = 1.0 - row_e
    local = []
    for matrix in gaba:
        to_e = matrix.multiply(row_e[:, None]) * (1.0 - frac)
        to_i = matrix.multiply(row_i[:, None])
        local.append((to_e + to_i).tocsc())

    rng = np.random.default_rng(int(seed))
    tree = cKDTree(pos_i)
    rows, cols, weights, delays, broad_distances = [], [], [], [], []
    k = min(c_cand, n_i)
    jump_scale = frac * float(tau_r_fast_ms) / float(tau_r_slow_ms)
    chunk = 512
    for start in range(0, int(n_e), chunk):
        stop = min(int(n_e), start + chunk)
        dist, cand = tree.query(pos_e[start:stop], k=k)
        dist = np.atleast_2d(dist)
        cand = np.atleast_2d(cand)
        for local_i, target in enumerate(range(start, stop)):
            drow = np.asarray(dist[local_i], dtype=float)
            crow = np.asarray(cand[local_i], dtype=np.int64)
            kernel = np.exp(-drow / sigma)
            keys = rng.standard_exponential(k) / np.maximum(kernel, np.finfo(float).tiny)
            take = np.argpartition(keys, c_in - 1)[:c_in]
            selected_d = drow[take]
            selected_c = crow[take]
            per_edge = target_budget[target] * jump_scale / c_in
            rows.append(np.full(c_in, target, dtype=np.int64))
            cols.append(selected_c)
            weights.append(np.full(c_in, per_edge, dtype=float))
            broad_distances.append(selected_d)
            delay_ms = float(tau0_ms) + selected_d / float(v_axon_mm_per_ms)
            step = max(1, int(round(float(delay_dt_ms) / float(dt_ms))))
            delay_steps = (
                np.maximum(1, np.rint(delay_ms / float(delay_dt_ms)).astype(int)) * step
            )
            delays.append(delay_steps)

    rows_a = np.concatenate(rows)
    cols_a = np.concatenate(cols)
    weights_a = np.concatenate(weights)
    delays_a = np.concatenate(delays)
    new_max = max(int(net["max_delay_steps"]), int(np.max(delays_a)))
    zero_g = sparse.csc_matrix((n, n_i), dtype=float)
    slow = [zero_g.copy() for _ in range(new_max + 1)]
    for delay in np.unique(delays_a):
        keep = delays_a == delay
        slow[int(delay)] = sparse.csc_matrix(
            (weights_a[keep], (rows_a[keep], cols_a[keep])), shape=(n, n_i)
        )
    local.extend(zero_g.copy() for _ in range(new_max + 1 - len(local)))
    zero_a = sparse.csc_matrix(ampa[0].shape, dtype=float)
    ampa_new = list(ampa) + [zero_a.copy() for _ in range(new_max + 1 - len(ampa))]

    net_new = dict(net)
    for cache_key in ("gaba_flat", "gaba_slow_flat"):
        net_new.pop(cache_key, None)
    net_new.update(
        ampa_by_delay=ampa_new,
        gaba_by_delay=local,
        gaba_slow_by_delay=slow,
        gaba_slow_tau_r_ms=float(tau_r_slow_ms),
        gaba_slow_tau_d_ms=float(tau_d_slow_ms),
        max_delay_steps=new_max,
    )

    state_new = dict(state)
    old_m = int(net["max_delay_steps"]) + 1
    t_abs = int(np.asarray(state["t"]))
    for key in ("ring_sE", "ring_sI"):
        old_ring = np.asarray(state[key])
        if old_ring.shape[0] != old_m:
            raise ValueError(f"{key} has {old_ring.shape[0]} bins, expected {old_m}")
        if new_max + 1 == old_m:
            continue
        new_ring = np.zeros((new_max + 1, old_ring.shape[1]), dtype=old_ring.dtype)
        for delta in range(old_m):
            new_ring[(t_abs + delta) % (new_max + 1)] += old_ring[(t_abs + delta) % old_m]
        state_new[key] = new_ring

    fast_total = sparse.csc_matrix((n, n_i), dtype=float)
    slow_total = sparse.csc_matrix((n, n_i), dtype=float)
    for matrix in local:
        fast_total = (fast_total + matrix).tocsc()
    for matrix in slow:
        slow_total = (slow_total + matrix).tocsc()
    matched = (
        np.asarray(fast_total[: int(n_e), :].sum(axis=1)).ravel()
        + np.asarray(slow_total[: int(n_e), :].sum(axis=1)).ravel()
        * float(tau_r_slow_ms) / float(tau_r_fast_ms)
    )
    relative_error = np.max(
        np.abs(matched - target_budget) / np.maximum(np.abs(target_budget), 1e-12)
    )
    return net_new, state_new, {
        "changed": True,
        "slow_fraction_integrated_budget": frac,
        "broad_sigma_mm": sigma,
        "broad_in_degree": c_in,
        "broad_candidate_count": c_cand,
        "seed": int(seed),
        "tau_r_fast_ms": float(tau_r_fast_ms),
        "tau_r_slow_ms": float(tau_r_slow_ms),
        "tau_d_slow_ms": float(tau_d_slow_ms),
        "broad_distance_percentiles_mm_5_25_50_75_95": np.percentile(
            np.concatenate(broad_distances), [5, 25, 50, 75, 95]
        ).tolist(),
        "broad_edge_count": int(weights_a.size),
        "i2e_integrated_budget_max_relative_error": float(relative_error),
        "ee_ampa_untouched": True,
        "i2i_gaba_untouched": True,
        "existing_arrival_offsets_preserved": True,
        "old_max_delay_steps": int(net["max_delay_steps"]),
        "new_max_delay_steps": new_max,
    }


def build_pv_som_inhibitory_subtypes(
    net: dict,
    state: dict,
    *,
    n_e: int,
    som_source_fraction: float,
    som_slow_budget_fraction: float,
    som_sigma_mm: float,
    som_in_degree: int,
    som_candidate_count: int,
    som_recruit_delay_scale: float,
    seed: int,
    dt_ms: float,
    delay_dt_ms: float,
    tau0_ms: float,
    v_axon_mm_per_ms: float,
    tau_r_fast_ms: float,
    tau_r_som_ms: float,
    tau_d_som_ms: float,
):
    """Partition existing I cells into local-fast PV and delayed broad-slow SOM.

    Unlike :func:`build_dual_scale_i2e_gaba`, the two output filters are driven
    by disjoint inhibitory neurons.  E->SOM arrivals are delayed, PV keeps the
    original local I->E path, and SOM gets a broad slow I->E path.  E->E and
    I->I matrices are exact; I->E integrated budget is matched per E target.
    """
    f_som = float(som_source_fraction)
    f_budget = float(som_slow_budget_fraction)
    sigma = float(som_sigma_mm)
    c_in, c_cand = int(som_in_degree), int(som_candidate_count)
    recruit_scale = float(som_recruit_delay_scale)
    positive = (
        sigma, recruit_scale, dt_ms, delay_dt_ms, tau0_ms,
        v_axon_mm_per_ms, tau_r_fast_ms, tau_r_som_ms, tau_d_som_ms,
    )
    if not (0.0 < f_som < 1.0 and 0.0 < f_budget < 1.0):
        raise ValueError("PV/SOM fractions must lie strictly between 0 and 1")
    if not np.all(np.isfinite(positive)) or min(positive) <= 0.0:
        raise ValueError("PV/SOM spatial and temporal parameters must be positive")
    if recruit_scale < 1.0:
        raise ValueError("SOM recruit delay scale must be >=1")
    if c_in < 1 or c_cand < c_in:
        raise ValueError("SOM candidate count must be >= in-degree >=1")
    if "gaba_slow_by_delay" in net:
        raise ValueError("network already has a slow inhibitory channel")

    gaba, ampa = net["gaba_by_delay"], net["ampa_by_delay"]
    n, n_i = gaba[0].shape
    if n_i != n - int(n_e):
        raise ValueError("PV/SOM population dimensions are inconsistent")
    rng = np.random.default_rng(int(seed))
    n_som = max(1, min(n_i - 1, int(round(f_som * n_i))))
    som_sources = np.sort(rng.choice(n_i, size=n_som, replace=False))
    som_mask = np.zeros(n_i, dtype=bool)
    som_mask[som_sources] = True
    pv_mask = ~som_mask

    original_g = sparse.csc_matrix((n, n_i), dtype=float)
    for matrix in gaba:
        original_g = (original_g + matrix).tocsc()
    target_budget = np.asarray(original_g[: int(n_e), :].sum(axis=1)).ravel()
    pv_budget = np.asarray(
        original_g[: int(n_e), :].multiply(pv_mask[None, :]).sum(axis=1)
    ).ravel()
    if np.any((target_budget > 0.0) & (pv_budget <= 0.0)):
        raise ValueError("an E target has no PV input to carry the local-fast budget")
    pv_row_scale = np.divide(
        (1.0 - f_budget) * target_budget,
        pv_budget,
        out=np.zeros_like(target_budget),
        where=pv_budget > 0.0,
    )

    row_e = np.zeros(n, dtype=float)
    row_e[: int(n_e)] = 1.0
    row_i = 1.0 - row_e
    local_g = []
    for matrix in gaba:
        pv_to_e = matrix.multiply(row_e[:, None]).multiply(pv_mask[None, :])
        pv_to_e = pv_to_e.multiply(
            np.concatenate([pv_row_scale, np.zeros(n - int(n_e))])[:, None]
        )
        to_i = matrix.multiply(row_i[:, None])
        local_g.append((pv_to_e + to_i).tocsc())

    pos = np.asarray(net["pos"], dtype=float)
    pos_e, pos_i = pos[: int(n_e)], pos[int(n_e) :]
    pos_som = pos_i[som_sources]
    tree = cKDTree(pos_som)
    k = min(c_cand, n_som)
    if c_in > k:
        raise ValueError("SOM in-degree exceeds the available SOM candidate population")
    step = max(1, int(round(float(delay_dt_ms) / float(dt_ms))))
    jump_scale = f_budget * float(tau_r_fast_ms) / float(tau_r_som_ms)
    broad_rows, broad_cols, broad_w, broad_dly, broad_dist = [], [], [], [], []
    chunk = 512
    for start in range(0, int(n_e), chunk):
        stop = min(int(n_e), start + chunk)
        dist, cand_local = tree.query(pos_e[start:stop], k=k)
        if k == 1:
            dist = np.asarray(dist)[:, None]
            cand_local = np.asarray(cand_local)[:, None]
        for ii, target in enumerate(range(start, stop)):
            drow = np.asarray(dist[ii], dtype=float)
            crow_local = np.asarray(cand_local[ii], dtype=np.int64)
            kernel = np.exp(-drow / sigma)
            keys = rng.standard_exponential(k) / np.maximum(kernel, np.finfo(float).tiny)
            take = np.argpartition(keys, c_in - 1)[:c_in]
            selected_d = drow[take]
            selected_src = som_sources[crow_local[take]]
            broad_rows.append(np.full(c_in, target, dtype=np.int64))
            broad_cols.append(selected_src)
            broad_w.append(np.full(
                c_in, target_budget[target] * jump_scale / c_in, dtype=float
            ))
            broad_dist.append(selected_d)
            delay_ms = float(tau0_ms) + selected_d / float(v_axon_mm_per_ms)
            broad_dly.append(
                np.maximum(1, np.rint(delay_ms / float(delay_dt_ms)).astype(int)) * step
            )
    br = np.concatenate(broad_rows)
    bc = np.concatenate(broad_cols)
    bw = np.concatenate(broad_w)
    bd = np.concatenate(broad_dly)

    # Delay only E inputs arriving onto SOM cells.  E->E rows are never touched.
    som_target_global = int(n_e) + som_sources
    som_target_mask = np.zeros(n, dtype=bool)
    som_target_mask[som_target_global] = True
    moved_ampa = []
    max_moved = int(net["max_delay_steps"])
    for d, matrix in enumerate(ampa):
        if not matrix.nnz:
            continue
        coo = matrix.tocoo()
        move = som_target_mask[coo.row]
        if np.any(move):
            new_d = max(1, int(round(d * recruit_scale)))
            max_moved = max(max_moved, new_d)
            moved_ampa.append((new_d, coo.row[move], coo.col[move], coo.data[move]))
    new_max = max(int(net["max_delay_steps"]), int(np.max(bd)), max_moved)
    zero_a = sparse.csc_matrix(ampa[0].shape, dtype=float)
    ampa_new = [zero_a.copy() for _ in range(new_max + 1)]
    for d, matrix in enumerate(ampa):
        if not matrix.nnz:
            continue
        unchanged = matrix.multiply((~som_target_mask)[:, None]).tocsc()
        ampa_new[d] = (ampa_new[d] + unchanged).tocsc()
    for new_d, rr, cc, ww in moved_ampa:
        moved = sparse.csc_matrix((ww, (rr, cc)), shape=ampa[0].shape)
        ampa_new[new_d] = (ampa_new[new_d] + moved).tocsc()

    zero_g = sparse.csc_matrix((n, n_i), dtype=float)
    local_g.extend(zero_g.copy() for _ in range(new_max + 1 - len(local_g)))
    slow_g = [zero_g.copy() for _ in range(new_max + 1)]
    for delay in np.unique(bd):
        keep = bd == delay
        slow_g[int(delay)] = sparse.csc_matrix(
            (bw[keep], (br[keep], bc[keep])), shape=(n, n_i)
        )

    net_new = dict(net)
    for cache_key in ("ampa_flat", "gaba_flat", "gaba_slow_flat"):
        net_new.pop(cache_key, None)
    net_new.update(
        ampa_by_delay=ampa_new,
        gaba_by_delay=local_g,
        gaba_slow_by_delay=slow_g,
        gaba_slow_tau_r_ms=float(tau_r_som_ms),
        gaba_slow_tau_d_ms=float(tau_d_som_ms),
        max_delay_steps=new_max,
    )

    state_new = dict(state)
    old_m = int(net["max_delay_steps"]) + 1
    t_abs = int(np.asarray(state["t"]))
    for key in ("ring_sE", "ring_sI"):
        old_ring = np.asarray(state[key])
        if old_ring.shape[0] != old_m:
            raise ValueError(f"{key} has {old_ring.shape[0]} bins, expected {old_m}")
        if new_max + 1 == old_m:
            continue
        new_ring = np.zeros((new_max + 1, old_ring.shape[1]), dtype=old_ring.dtype)
        for delta in range(old_m):
            new_ring[(t_abs + delta) % (new_max + 1)] += old_ring[(t_abs + delta) % old_m]
        state_new[key] = new_ring

    fast_total = sum(local_g, sparse.csc_matrix((n, n_i), dtype=float))
    slow_total = sum(slow_g, sparse.csc_matrix((n, n_i), dtype=float))
    matched = (
        np.asarray(fast_total[: int(n_e), :].sum(axis=1)).ravel()
        + np.asarray(slow_total[: int(n_e), :].sum(axis=1)).ravel()
        * float(tau_r_som_ms) / float(tau_r_fast_ms)
    )
    relative_error = np.max(
        np.abs(matched - target_budget) / np.maximum(np.abs(target_budget), 1e-12)
    )
    old_ampa = sum(ampa, sparse.csc_matrix(ampa[0].shape, dtype=float))
    new_ampa = sum(ampa_new, sparse.csc_matrix(ampa[0].shape, dtype=float))
    if (old_ampa != new_ampa).nnz:
        raise RuntimeError("PV/SOM transformation changed AMPA edges or weights")
    return net_new, state_new, {
        "changed": True,
        "som_source_fraction_requested": f_som,
        "som_source_fraction_realized": n_som / n_i,
        "som_slow_integrated_budget_fraction": f_budget,
        "som_sigma_mm": sigma,
        "som_in_degree": c_in,
        "som_candidate_count": c_cand,
        "som_recruit_delay_scale": recruit_scale,
        "tau_r_som_ms": float(tau_r_som_ms),
        "tau_d_som_ms": float(tau_d_som_ms),
        "seed": int(seed),
        "som_membership_sha256": hashlib.sha256(
            np.ascontiguousarray(som_sources).tobytes()
        ).hexdigest(),
        "som_broad_distance_percentiles_mm_5_25_50_75_95": np.percentile(
            np.concatenate(broad_dist), [5, 25, 50, 75, 95]
        ).tolist(),
        "som_broad_edge_count": int(bw.size),
        "pv_row_scale_percentiles_5_25_50_75_95": np.percentile(
            pv_row_scale, [5, 25, 50, 75, 95]
        ).tolist(),
        "i2e_integrated_budget_max_relative_error": float(relative_error),
        "ee_ampa_untouched": True,
        "ampa_edges_and_weights_untouched": True,
        "i2i_gaba_untouched": True,
        "existing_arrival_offsets_preserved": True,
        "old_max_delay_steps": int(net["max_delay_steps"]),
        "new_max_delay_steps": new_max,
    }


def build_source_locked_context(root: Path, manifest: dict, runner: Any) -> dict:
    """Rebuild static substrate from its old lock without rewriting the lock."""
    source = manifest["source"]
    lock = json.loads((root / source["canonical_config_path"]).read_text())
    seed = int(source["seed"])
    parent = lock["seeds"][str(seed)]
    if parent["config_sha"] != source["canonical_config_sha"]:
        raise RuntimeError("source canonical seed lock drift")
    dt = float(source["dt_ms"])
    static = runner.PP.build_substrate(seed=seed, dt=dt)
    static["seed"] = seed
    static["I_th_EI"] = float(parent["config"]["I_th_EI"])
    montage = static["reg"]["montage_sheet"]
    recorder = runner.LFPRecorder(
        static["p"],
        static["net"]["pos"],
        static["net"]["labels"],
        sites=np.asarray(montage.contacts, float),
    )
    core = runner.ZM._core_mask_E(static)
    along, _ = runner.CG.axis_transverse_coords(
        static["posE"], static["src_xy"], static["axis_unit"]
    )
    return {
        "S": static,
        "rec": recorder,
        "core": core,
        "axis": along,
        "contacts": list(montage.names),
        "cfg_locked": parent["config"],
        "cfg_sha": parent["config_sha"],
        "smoke": False,
        "resolution": "dt",
        "dt": dt,
        "anchor_root": "anchors",
        "runtime_git_sha": git_sha(root),
        "runtime_started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


class DiagnosticSlowWrapper:
    """Observation-only wrapper; every numerical method delegates literally."""

    def __init__(self, inner, *, diagnostic_stride_steps: int = 100):
        object.__setattr__(self, "inner", inner)
        if int(diagnostic_stride_steps) < 1:
            raise ValueError("diagnostic_stride_steps must be >=1")
        object.__setattr__(self, "diagnostic_stride_steps", int(diagnostic_stride_steps))
        object.__setattr__(self, "_step_index", 0)
        object.__setattr__(self, "trace_vinf_median", [])
        object.__setattr__(self, "trace_tau_eff_median", [])
        object.__setattr__(self, "trace_exc_charge_mean", [])
        object.__setattr__(self, "trace_inh_charge_mean", [])
        object.__setattr__(self, "trace_vinf_above_EI", [])
        object.__setattr__(self, "trace_v_above_EI", [])

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __setattr__(self, name, value):
        if name == "inner" or name.startswith("trace_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self.inner, name, value)

    def apply_currents(self, I_E, I_I, labels=None, I_E_rec=None):
        out = self.inner.apply_currents(I_E, I_I, labels, I_E_rec)
        if self._step_index % self.diagnostic_stride_steps:
            return out
        n_e = int(self.inner.nE)
        cfg = self.inner.cfg
        exc = np.asarray(I_E, float)[:n_e].copy()
        if cfg.use_SG and I_E_rec is not None:
            a_s = cfg.alpha_G * self.inner.S_G
            a_h = cfg.alpha_H * self.inner.H if cfg.use_H else 0.0
            a_m = (
                cfg.kappa_mode_M * (
                    self.inner.mode_M_memory
                    if getattr(cfg, "use_mode_M_memory", False)
                    else self.inner.mode_M_pool()
                )
                if getattr(cfg, "use_mode_M_divisive", False) else 0.0
            )
            denom = 1.0 + a_s + a_h + a_m
            frac = (a_s + a_h + a_m) / denom
            exc -= np.asarray(I_E_rec, float)[:n_e] * frac
            if getattr(cfg, "use_mode_H", False):
                exc += (
                    np.asarray(I_E_rec, float)[:n_e]
                    * self.inner.mode_H_gain_at_E() / denom
                )
            exc -= cfg.beta_SG * self.inner.S_G
        inh = np.asarray(I_I, float)[:n_e]
        if cfg.use_z:
            inh = self.inner.z[:n_e] * inh
        self.trace_vinf_median.append(float(np.median(out[:n_e])))
        self.trace_tau_eff_median.append(float(self.inner.cfg.cond_tau_m_E))
        self.trace_exc_charge_mean.append(float(np.mean(np.abs(exc))))
        self.trace_inh_charge_mean.append(float(np.mean(np.abs(inh))))
        self.trace_vinf_above_EI.append(float("nan"))
        self.trace_v_above_EI.append(float("nan"))
        return out

    def zm_conductance_step(self, V, I_E, I_I, decay_V):
        out = self.inner.zm_conductance_step(V, I_E, I_I, decay_V)
        if self._step_index % self.diagnostic_stride_steps:
            return out
        e = self.inner.is_E
        cfg = self.inner.zm_conductance_config()
        self.trace_vinf_median.append(float(np.median(out["V_inf"][e])))
        self.trace_tau_eff_median.append(float(np.median(out["tau_eff_ms"][e])))
        self.trace_exc_charge_mean.append(float(np.mean(np.abs(out["I_exc"][e]))))
        self.trace_inh_charge_mean.append(float(np.mean(np.abs(out["I_inh"][e]))))
        self.trace_vinf_above_EI.append(float(np.mean(out["V_inf"][e] > cfg.E_I)))
        self.trace_v_above_EI.append(float(np.mean(np.asarray(V)[e] > cfg.E_I)))
        return out

    def advance_step(self):
        object.__setattr__(self, "_step_index", self._step_index + 1)

    def step(self, spk, labels, dt):
        self.inner.step(spk, labels, dt)
        self.advance_step()

    def diagnostic_summary(self) -> dict:
        def median(name):
            values = np.asarray(getattr(self, name), float)
            finite = values[np.isfinite(values)]
            return None if not finite.size else float(np.median(finite))

        exc = np.asarray(self.trace_exc_charge_mean, float)
        inh = np.asarray(self.trace_inh_charge_mean, float)
        ratio = float(np.sum(inh) / np.sum(exc)) if np.sum(exc) > 0 else None
        return {
            "median_vinf_mv": median("trace_vinf_median"),
            "median_tau_eff_ms": median("trace_tau_eff_median"),
            "effective_inhibitory_to_excitatory_charge_ratio": ratio,
            "median_fraction_vinf_above_EI": median("trace_vinf_above_EI"),
            "median_fraction_v_above_EI": median("trace_v_above_EI"),
            "n_steps": len(self.trace_vinf_median),
            "diagnostic_stride_steps": self.diagnostic_stride_steps,
            "n_simulation_steps_seen": self._step_index,
        }


class FrozenAllNoStepWrapper:
    """Zero-copy freeze-all semantics for z/m/S_G with phi disabled.

    ``FreezeWrapper`` must execute and undo ``step`` for partially frozen
    scientific arms.  Calibration freezes every evolving slow coordinate, so
    running that update only to copy back 40k-element fields is equivalent but
    prohibitively expensive.  This wrapper keeps current/threshold reads live
    and makes only the slow update a no-op.
    """

    def __init__(self, inner):
        self.inner = inner
        if bool(getattr(inner.cfg, "use_phi", False)):
            raise ValueError("zero-copy freeze-all cannot freeze an enabled dynamic threshold")

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def apply_currents(self, I_E, I_I, labels=None, I_E_rec=None):
        return self.inner.apply_currents(I_E, I_I, labels, I_E_rec)

    def threshold(self, V_th_base):
        return self.inner.threshold(V_th_base)

    def zm_conductance_step(self, V, I_E, I_I, decay_V):
        return self.inner.zm_conductance_step(V, I_E, I_I, decay_V)

    def step(self, spk, labels, dt):
        if hasattr(self.inner, "advance_step"):
            self.inner.advance_step()
        return None


def make_frozen_diagnostic_slow(
    ctx: dict,
    runner: Any,
    *,
    conductance_config: dict | None,
):
    """Build the exact Arm-A or conductance slow layer, frozen at checkpoint."""
    if conductance_config is None:
        cfg = runner.ZM._zm_cfg(ctx["S"]["I_th_EI"], **runner.ARM_KWARGS)
    else:
        cfg = runner.ZM._zm_cfg(ctx["S"]["I_th_EI"], use_SG=False, alpha_G=0.0)
        cfg = dataclasses.replace(
            cfg,
            use_zm_conductance=True,
            cond_kappa_E=float(conductance_config["kappa_E"]),
            cond_kappa_I=float(conductance_config["kappa_I"]),
            cond_g_M=float(conductance_config["g_M"]),
            cond_gamma=float(conductance_config["gamma"]),
            cond_z_spares_global=bool(conductance_config["z_spares_global"]),
            cond_g_L=float(conductance_config["g_L"]),
            cond_E_L=float(conductance_config["E_L"]),
            cond_E_E=float(conductance_config["E_E"]),
            cond_E_I=float(conductance_config["E_I"]),
            cond_E_K=float(conductance_config["E_K"]),
            cond_tau_m_E=float(conductance_config["tau_m_E"]),
        )
    base = runner.SpatialSlowField(
        ctx["S"]["N"],
        18.0,
        ctx["S"]["posE"],
        ctx["S"]["posI"],
        ctx["S"]["L"],
        core_mask_E=ctx["core"],
        cfg=cfg,
    )
    diagnostic = DiagnosticSlowWrapper(base)
    frozen = FrozenAllNoStepWrapper(diagnostic)
    return frozen, diagnostic


def make_dynamic_diagnostic_slow(
    ctx: dict,
    runner: Any,
    *,
    conductance_config: dict | None,
):
    """Build a dynamic Z/M baseline or conductance arm from canonical t=0."""
    if conductance_config is None:
        cfg = runner.ZM._zm_cfg(ctx["S"]["I_th_EI"], **runner.ARM_KWARGS)
    else:
        cfg = runner.ZM._zm_cfg(ctx["S"]["I_th_EI"], use_SG=False, alpha_G=0.0)
        cfg = dataclasses.replace(
            cfg,
            use_zm_conductance=True,
            cond_kappa_E=float(conductance_config["kappa_E"]),
            cond_kappa_I=float(conductance_config["kappa_I"]),
            cond_g_M=float(conductance_config["g_M"]),
            cond_gamma=float(conductance_config["gamma"]),
            cond_z_spares_global=bool(conductance_config["z_spares_global"]),
            cond_g_L=float(conductance_config["g_L"]),
            cond_E_L=float(conductance_config["E_L"]),
            cond_E_E=float(conductance_config["E_E"]),
            cond_E_I=float(conductance_config["E_I"]),
            cond_E_K=float(conductance_config["E_K"]),
            cond_tau_m_E=float(conductance_config["tau_m_E"]),
        )
    base = runner.SpatialSlowField(
        ctx["S"]["N"],
        18.0,
        ctx["S"]["posE"],
        ctx["S"]["posI"],
        ctx["S"]["L"],
        core_mask_E=ctx["core"],
        cfg=cfg,
    )
    diagnostic = DiagnosticSlowWrapper(base)
    return diagnostic, diagnostic


__all__ = [
    "DiagnosticSlowWrapper",
    "FrozenAllNoStepWrapper",
    "build_source_locked_context",
    "git_sha",
    "make_dynamic_diagnostic_slow",
    "make_frozen_diagnostic_slow",
]
