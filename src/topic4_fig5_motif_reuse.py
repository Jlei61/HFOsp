"""Model-internal motif reuse and its matched nulls.

The question is narrow and entirely inside the model: does the recruitment order
of the early model-ictal state look like the recruitment order of the
trajectory's own interictal events, more than a null that keeps everything about
those events except which contact is which?

Reuse is measured on contacts and on edges. Only the contact family is
computable from the archived Z/M artifacts: no producer in this round ever wrote
a per-window recurrent-E edge flow, so the edge-flow cosine is reported as
missing evidence rather than approximated. The edge permutation and its
structural audit are still implemented, because the plan requires the null to be
proved structure-preserving before it is ever used.

Every contact timing here comes from the SAME frozen readout that produced the
interictal event onsets (``_contact_onsets`` with the run's own envelope floor
and margin). A rank comparison between two differently-defined timings would be
measuring the definitions, not the model.
"""
from __future__ import annotations

import numpy as np

NOT_EVALUABLE = "NOT_EVALUABLE_FROM_EXISTING_ARTIFACTS"


def spearman_with_coverage(a, b):
    """Spearman over the contacts recruited in BOTH vectors, plus that count."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    both = np.isfinite(a) & np.isfinite(b)
    n = int(both.sum())
    if n < 3:
        return {"rho": float("nan"), "n_common": n, "status": NOT_EVALUABLE}
    x, y = a[both], b[both]
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if np.std(rx) == 0.0 or np.std(ry) == 0.0:
        return {"rho": float("nan"), "n_common": n, "status": "DEGENERATE"}
    return {"rho": float(np.corrcoef(rx, ry)[0, 1]), "n_common": n,
            "status": "OK",
            "coverage": float(n / max(int(np.isfinite(b).sum()), 1))}


def precedence_matrix(values):
    """sign(t_j - t_i) for every pair recruited in the same vector; NaN elsewhere."""
    values = np.asarray(values, float)
    n = len(values)
    matrix = np.full((n, n), np.nan)
    finite = np.isfinite(values)
    index = np.flatnonzero(finite)
    if len(index) < 2:
        return matrix
    sub = values[index]
    sign = np.sign(sub[None, :] - sub[:, None])
    np.fill_diagonal(sign, 0.0)
    # only recruited contacts get a diagonal zero; an absent contact keeps a
    # fully NaN row and column so it can never be read as "tied with itself"
    matrix[np.ix_(index, index)] = sign
    return matrix


def mode_precedence_matrix(event_values):
    """Mean precedence over a mode's events.

    The magnitude is how consistently that pair is ordered inside the mode, and
    is used as the weight below: a pair the mode itself never orders the same way
    twice should not dominate the agreement score.
    """
    stack = [precedence_matrix(row) for row in np.asarray(event_values, float)]
    if not stack:
        return None, None
    cube = np.stack(stack)
    support = np.sum(np.isfinite(cube), axis=0)
    totals = np.nansum(cube, axis=0)
    mean = np.full(support.shape, np.nan)
    good = support > 0
    mean[good] = totals[good] / support[good]
    return mean, support


def precedence_agreement(reference_matrix, target_values, *, weights=None,
                         target_matrix=None):
    """Weighted fraction of comparable pairs ordered the same way.

    ``weights`` defaults to ``abs(reference_matrix)``, which is 1 for a single
    event (a hard +/-1 precedence) and the within-mode consistency for a mode
    matrix, so the same function serves both levels of the spec's phrase
    "event/frozen-mode contact-pair precedence matrix".

    ``target_matrix`` lets a caller pass the already-built target precedence so a
    permutation loop does not rebuild the same matrix once per event per draw.
    It is an optimisation only; the returned values are identical.
    """
    reference = np.asarray(reference_matrix, float)
    target = (precedence_matrix(target_values) if target_matrix is None
              else np.asarray(target_matrix, float))
    upper = np.triu(np.ones_like(reference, dtype=bool), k=1)
    comparable = upper & np.isfinite(reference) & np.isfinite(target)
    comparable &= (reference != 0.0) & (target != 0.0)
    n = int(comparable.sum())
    if n == 0:
        return {"agreement": float("nan"), "n_pairs": 0, "status": NOT_EVALUABLE}
    weight = (np.abs(reference) if weights is None
              else np.asarray(weights, float))[comparable]
    same = (np.sign(reference[comparable]) == np.sign(target[comparable]))
    total = float(weight.sum())
    if total <= 0.0:
        return {"agreement": float("nan"), "n_pairs": n, "status": "DEGENERATE"}
    return {"agreement": float((weight * same).sum() / total),
            "n_pairs": n, "weight_sum": total, "status": "OK"}


def within_shaft_label_permutation(values, shaft_ids, rng):
    """Null 1: permute which contact of a shaft carries which time.

    The number of recruited contacts per shaft and the multiset of their times
    are preserved exactly; only the contact identity is destroyed.
    """
    values = np.asarray(values, float)
    permuted = np.empty_like(values)
    for shaft in np.unique(np.asarray(shaft_ids)):
        index = np.flatnonzero(np.asarray(shaft_ids) == shaft)
        permuted[index] = values[rng.permutation(index)]
    return permuted


def circular_shift(values, shift):
    """Null 2: re-pair events with times-to-transition by a circular shift.

    Event count, event rate and the transition time are untouched; only the
    correspondence between an event and its distance from the transition moves.
    """
    return np.roll(np.asarray(values, float), int(shift))


def rank_reuse(event_ranks, early_ranks, shaft_ids, *, n_draws, seed):
    """Observed rank agreement and its within-shaft permutation null."""
    observed = spearman_with_coverage(event_ranks, early_ranks)
    if observed["status"] != "OK":
        return {"observed": observed, "null": {"status": NOT_EVALUABLE}}
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(n_draws), float)
    for index in range(int(n_draws)):
        shuffled = within_shaft_label_permutation(early_ranks, shaft_ids, rng)
        draws[index] = spearman_with_coverage(event_ranks, shuffled)["rho"]
    finite = draws[np.isfinite(draws)]
    return {"observed": observed,
            "null": _null_summary(observed["rho"], finite, n_draws)}


def precedence_reuse(reference_matrix, early_values, shaft_ids, *, n_draws, seed,
                     weights=None):
    observed = precedence_agreement(reference_matrix, early_values,
                                    weights=weights)
    if observed["status"] != "OK":
        return {"observed": observed, "null": {"status": NOT_EVALUABLE}}
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(n_draws), float)
    for index in range(int(n_draws)):
        shuffled = within_shaft_label_permutation(early_values, shaft_ids, rng)
        draws[index] = precedence_agreement(
            reference_matrix, shuffled, weights=weights)["agreement"]
    del index
    finite = draws[np.isfinite(draws)]
    return {"observed": observed,
            "null": _null_summary(observed["agreement"], finite, n_draws)}


def _null_summary(observed, draws, n_draws):
    """Null position of an observed statistic, with the sign kept visible.

    The within-shaft null can sit well below zero -- with two shafts, relabelling
    inside each shaft leaves the between-shaft ordering intact and can force a
    strongly negative agreement. An observed value can then clear the null's q95
    while still being a NEGATIVE agreement, which is the opposite of reuse.
    ``exceeds_q95`` is the spec's literal gate and is reported as such;
    ``reuse_supported`` additionally requires the observed agreement to be
    positive, and is the flag a reuse claim may rest on.
    """
    if not len(draws):
        return {"status": NOT_EVALUABLE, "n_draws": int(n_draws)}
    q95 = float(np.quantile(draws, 0.95))
    exceeds = bool(observed > q95)
    return {
        "status": "OK",
        "n_draws": int(n_draws),
        "n_finite_draws": int(len(draws)),
        "median": float(np.median(draws)),
        "q95": q95,
        "observed": float(observed),
        "observed_is_positive": bool(observed > 0.0),
        "exceeds_q95": exceeds,
        "reuse_supported": bool(exceeds and observed > 0.0),
        "exceedance_probability": float(np.mean(draws >= observed)),
        "excess_over_null_median": float(observed - float(np.median(draws))),
    }


def reuse_trajectory(per_event_scores, time_to_transition_ms, *, n_draws, seed):
    """Reuse versus time-to-transition, against the circular-shift null."""
    scores = np.asarray(per_event_scores, float)
    delays = np.asarray(time_to_transition_ms, float)
    usable = np.isfinite(scores) & np.isfinite(delays)
    if int(usable.sum()) < 4:
        return {"status": NOT_EVALUABLE, "n_events": int(usable.sum())}
    x, y = scores[usable], delays[usable]
    rho = spearman_with_coverage(x, y)
    rng = np.random.default_rng(int(seed))
    shifts = rng.integers(1, len(x), size=int(n_draws))
    draws = np.asarray([
        spearman_with_coverage(x, circular_shift(y, shift))["rho"]
        for shift in shifts], float)
    finite = draws[np.isfinite(draws)]
    return {"status": "OK", "n_events": int(usable.sum()),
            "spearman_reuse_vs_time_to_transition": rho["rho"],
            "null": _null_summary(abs(rho["rho"]), np.abs(finite), n_draws)}


def network_level_aggregate(per_network_values, *, draws, seed):
    """Every network carries equal weight; the bootstrap is over networks."""
    values = np.asarray([v for v in per_network_values if np.isfinite(v)], float)
    if not len(values):
        return {"status": NOT_EVALUABLE, "n_networks": 0}
    rng = np.random.default_rng(int(seed))
    index = rng.integers(0, len(values), size=(int(draws), len(values)))
    means = values[index].mean(axis=1)
    return {"status": "OK", "n_networks": int(len(values)),
            "mean": float(values.mean()), "median": float(np.median(values)),
            "bootstrap_q05": float(np.quantile(means, 0.05)),
            "bootstrap_q95": float(np.quantile(means, 0.95))}


# --------------------------------------------------------------------------
# Edge family: permutation and structural audit only.
# --------------------------------------------------------------------------

def edge_distance_bins(n_bins=8):
    return int(n_bins)


def permute_edge_weights(ampa_by_delay, n_e, positions_all, *, rng, n_distance_bins=8):
    """Null 3: permute weights inside pathway x delay x distance strata.

    Topology, delay assignment and edge distances cannot move because only the
    DATA of each sparse matrix is reordered, never its indices. The target-wise
    incoming budget is restored exactly afterwards, which is the one thing a
    plain permutation does break.
    """
    import scipy.sparse as sp

    positions_all = np.asarray(positions_all, float)
    permuted = []
    for matrix in ampa_by_delay:
        coo = matrix.tocoo(copy=False)
        rows = np.asarray(coo.row, np.int64)
        cols = np.asarray(coo.col, np.int64)
        data = np.asarray(coo.data, float).copy()
        if not len(data):
            permuted.append(matrix.copy())
            continue
        distance = np.linalg.norm(
            positions_all[rows] - positions_all[cols], axis=1)
        pathway = (rows >= n_e).astype(np.int64)
        for pathway_id in (0, 1):
            mask = pathway == pathway_id
            if not np.any(mask):
                continue
            local = np.flatnonzero(mask)
            edges = np.quantile(distance[local],
                                np.linspace(0.0, 1.0, n_distance_bins + 1))
            edges[0] -= 1e-9
            edges[-1] += 1e-9
            which = np.clip(np.digitize(distance[local], edges[1:-1]), 0,
                            n_distance_bins - 1)
            for stratum in range(n_distance_bins):
                members = local[which == stratum]
                if len(members) > 1:
                    data[members] = data[rng.permutation(members)]
        new = sp.coo_matrix((data, (rows, cols)), shape=matrix.shape).tocsr()
        permuted.append(new)

    # exact target-wise budget renormalization, separately per pathway
    from src.topic4_local_connectivity import _incoming_by_pathway
    n_i = ampa_by_delay[0].shape[0] - n_e
    for pathway in ("E_to_E", "E_to_I"):
        before = _incoming_by_pathway(ampa_by_delay, n_e, n_i, pathway)
        after = _incoming_by_pathway(permuted, n_e, n_i, pathway)
        scale = np.divide(before, after, out=np.ones_like(before),
                          where=after > 0.0)
        for index, matrix in enumerate(permuted):
            coo = matrix.tocoo(copy=False)
            rows = np.asarray(coo.row, np.int64)
            mask = rows < n_e if pathway == "E_to_E" else rows >= n_e
            if not np.any(mask):
                continue
            local_rows = rows[mask] if pathway == "E_to_E" else rows[mask] - n_e
            data = np.asarray(coo.data, float).copy()
            data[mask] *= scale[local_rows]
            permuted[index] = sp.coo_matrix(
                (data, (rows, np.asarray(coo.col, np.int64))),
                shape=matrix.shape).tocsr()
    return permuted


def _quantile_deviation(before, after, quantiles=(0.05, 0.5, 0.95)):
    before, after = np.asarray(before, float), np.asarray(after, float)
    if not len(before) or not len(after):
        return {"quantiles": list(quantiles), "before": [], "after": [],
                "max_relative_deviation": float("nan")}
    qb = [float(np.quantile(before, q)) for q in quantiles]
    qa = [float(np.quantile(after, q)) for q in quantiles]
    deviation = [abs(a - b) / max(abs(b), 1e-12) for a, b in zip(qb, qa)]
    return {"quantiles": list(quantiles), "before": qb, "after": qa,
            "relative_deviation": deviation,
            "max_relative_deviation": float(max(deviation))}


def audit_edge_permutation(original, permuted, n_e, positions_all, *,
                           n_distance_bins=8, tolerance=1e-9,
                           weight_quantile_relative_tolerance=0.02):
    """Everything the null must preserve, checked as values not as intent.

    The weight distribution is part of the contract, not a diagnostic. A plain
    within-stratum permutation preserves it exactly, but the exact target-wise
    budget renormalisation that follows does not: where incoming budgets are
    heterogeneous, rescaling per target moves the marginal weight distribution.
    Reporting those quantiles without gating on them let a permutation whose
    median weight moved by 42% still be called structure-preserving, so the
    deviation is now stratified by pathway and distance bin and enters the
    conjunction.
    """
    from src.topic4_core_connectivity import _hash_sparse_bins
    from src.topic4_local_connectivity import _incoming_by_pathway

    n_i = original[0].shape[0] - n_e
    positions_all = np.asarray(positions_all, float)
    report = {
        "topology_unchanged": (_hash_sparse_bins(original, include_data=False)
                               == _hash_sparse_bins(permuted, include_data=False)),
        "n_delay_bins_unchanged": len(original) == len(permuted),
        "delay_assignment_unchanged": all(
            a.nnz == b.nnz for a, b in zip(original, permuted)),
        "data_changed": _hash_sparse_bins(original) != _hash_sparse_bins(permuted),
    }
    for pathway in ("E_to_E", "E_to_I"):
        before = _incoming_by_pathway(original, n_e, n_i, pathway)
        after = _incoming_by_pathway(permuted, n_e, n_i, pathway)
        report[f"{pathway}_max_abs_incoming_error"] = float(
            np.max(np.abs(before - after))) if len(before) else 0.0
        report[f"{pathway}_incoming_budget_preserved"] = bool(
            np.allclose(before, after, rtol=0.0, atol=tolerance))

    def _edge_table(bins):
        """Canonical CSR order, matching ``_hash_sparse_bins``.

        A rebuilt matrix can enumerate the same edges in a different order, so an
        order-sensitive comparison reports a difference where there is none. The
        frozen 32 M-edge graph does exactly that.
        """
        rows, cols, data = [], [], []
        for matrix in bins:
            csr = matrix.tocsr(copy=True)
            csr.sort_indices()
            coo = csr.tocoo(copy=False)
            rows.append(np.asarray(coo.row, np.int64))
            cols.append(np.asarray(coo.col, np.int64))
            data.append(np.asarray(coo.data, float))
        return (np.concatenate(rows), np.concatenate(cols),
                np.concatenate(data))

    rows_a, cols_a, data_a = _edge_table(original)
    rows_b, cols_b, data_b = _edge_table(permuted)
    report["edge_index_sets_identical"] = bool(
        np.array_equal(rows_a, rows_b) and np.array_equal(cols_a, cols_b))
    distance = np.linalg.norm(positions_all[rows_a] - positions_all[cols_a], axis=1)
    report["edge_distance_distribution"] = {
        "quantiles": [0.05, 0.5, 0.95],
        "original": [float(np.quantile(distance, q)) for q in (0.05, 0.5, 0.95)],
        "identical_by_construction": True,
    }
    report["source_degree_identical"] = bool(np.array_equal(
        np.bincount(cols_a, minlength=n_e), np.bincount(cols_b, minlength=n_e)))
    report["target_degree_identical"] = bool(np.array_equal(
        np.bincount(rows_a, minlength=n_e + n_i),
        np.bincount(rows_b, minlength=n_e + n_i)))
    report["weight_distribution"] = {
        "quantiles": [0.05, 0.5, 0.95],
        "original": [float(np.quantile(data_a, q)) for q in (0.05, 0.5, 0.95)],
        "permuted": [float(np.quantile(data_b, q)) for q in (0.05, 0.5, 0.95)],
        **_quantile_deviation(data_a, data_b),
    }

    # stratified exactly as the permutation is: pathway x distance bin
    pathway_a = (rows_a >= n_e).astype(np.int64)
    strata = {}
    worst = 0.0
    for pathway_id, name in ((0, "E_to_E"), (1, "E_to_I")):
        mask = pathway_a == pathway_id
        if not np.any(mask):
            continue
        local_distance = distance[mask]
        edges = np.quantile(local_distance,
                            np.linspace(0.0, 1.0, n_distance_bins + 1))
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        which = np.clip(np.digitize(local_distance, edges[1:-1]), 0,
                        n_distance_bins - 1)
        for stratum in range(n_distance_bins):
            members = np.flatnonzero(mask)[which == stratum]
            if len(members) < 2:
                continue
            row = _quantile_deviation(data_a[members], data_b[members])
            strata[f"{name}_d{stratum}"] = row
            worst = max(worst, float(row["max_relative_deviation"]))
    report["weight_quantiles_by_pathway_and_distance_bin"] = strata
    report["weight_quantile_relative_tolerance"] = float(
        weight_quantile_relative_tolerance)
    report["max_weight_quantile_relative_deviation"] = float(max(
        worst, float(report["weight_distribution"]["max_relative_deviation"])))
    report["weight_distribution_preserved"] = bool(
        report["max_weight_quantile_relative_deviation"]
        <= float(weight_quantile_relative_tolerance))
    report["budget_and_degree_joint_contract"] = bool(
        report["E_to_E_incoming_budget_preserved"]
        and report["E_to_I_incoming_budget_preserved"]
        and report["source_degree_identical"]
        and report["target_degree_identical"])

    report["all_structural_clauses_pass"] = bool(
        report["topology_unchanged"] and report["n_delay_bins_unchanged"]
        and report["delay_assignment_unchanged"]
        and report["edge_index_sets_identical"]
        and report["budget_and_degree_joint_contract"]
        and report["weight_distribution_preserved"]
        and report["data_changed"])
    return report


def matched_off_motif_node_sets(*args, **kwargs):
    """Null 4. Not implemented on purpose.

    It needs a node-level motif, which needs the recurrent-E edge flow that no
    archived Z/M artifact contains, and a per-node baseline rate that is likewise
    not stored. A "best effort" version would produce a plausible number for a
    comparison that cannot be made, so this raises instead.
    """
    raise NotImplementedError(
        "matched off-motif node sets require a per-window recurrent-E edge flow "
        "and per-node baseline rates; neither exists in the archived artifacts "
        "(spec section 7.3, plan Task 4)")


def _median_event_rank_agreement(event_ranks, early_ranks):
    rows = [spearman_with_coverage(row, early_ranks) for row in event_ranks]
    values = np.asarray([row["rho"] for row in rows], float)
    finite = values[np.isfinite(values)]
    coverage = [row["n_common"] for row in rows]
    return (float(np.median(finite)) if len(finite) else float("nan")), rows, coverage


NULL_MATCHES = ["shaft identity", "per-shaft recruitment count",
                "the multiset of times inside each shaft"]
NULL_DOES_NOT_MATCH = [
    "distance from the contact to the high-excitability core",
    "the contact's local baseline excitability h",
    "any monotone spatial gradient along a shaft",
    "local connection strength at the contact",
]
CONTACT_CLAIM_BOUNDARY = (
    "This is contact-ORDER similarity between a trajectory's interictal events "
    "and its own early high state, above a shaft-matched relabelling. Because "
    "the null does not match distance to core, local excitability or a monotone "
    "along-shaft gradient, it cannot separate 'the early dynamics reuse the "
    "interictal propagation motif' from 'both orders are dictated by the same "
    "fixed spatial gradient'. It may not be reported as motif reuse.")


def contact_geometry_diagnostic(contact_xy, positions_e, h_e, event_ranks,
                                early_ranks, *, local_radius_mm=1.0):
    """How much of the order is explained by a fixed geometry alone.

    Not a null and not a control: a null that matched these covariates has not
    been built. This reports whether the confound the shaft-matched null leaves
    open is actually present in this substrate, so the size of the gap is a
    number rather than a caveat.
    """
    contact_xy = np.asarray(contact_xy, float)
    positions = np.asarray(positions_e, float)
    h_e = np.asarray(h_e, float)
    weight = h_e / max(float(h_e.sum()), 1e-12)
    core_xy = np.asarray([float(np.dot(weight, positions[:, 0])),
                          float(np.dot(weight, positions[:, 1]))])
    distance = np.linalg.norm(contact_xy - core_xy[None, :], axis=1)
    local_h = np.asarray([
        float(np.mean(h_e[np.linalg.norm(positions - point[None, :], axis=1)
                          <= float(local_radius_mm)]))
        if np.any(np.linalg.norm(positions - point[None, :], axis=1)
                  <= float(local_radius_mm)) else float("nan")
        for point in contact_xy])
    event_ranks = np.atleast_2d(np.asarray(event_ranks, float))
    per_event = [spearman_with_coverage(row, distance)["rho"]
                 for row in event_ranks]
    finite = np.asarray([v for v in per_event if np.isfinite(v)], float)
    return {
        "core_centroid_xy_mm": core_xy.tolist(),
        "local_radius_mm": float(local_radius_mm),
        "contact_distance_to_core_mm": distance.tolist(),
        "contact_local_h": local_h.tolist(),
        "median_event_rank_vs_distance_spearman": (
            float(np.median(finite)) if len(finite) else float("nan")),
        "early_rank_vs_distance_spearman": spearman_with_coverage(
            early_ranks, distance)["rho"],
        "early_rank_vs_local_h_spearman": spearman_with_coverage(
            early_ranks, local_h)["rho"],
        "interpretation": (
            "if both the event ranks and the early ranks track distance to core, "
            "the shaft-matched agreement is at least partly a fixed-geometry "
            "effect and the contact-order result cannot be attributed to a "
            "reused propagation motif"),
    }


def network_rank_reuse(event_ranks, early_ranks, shaft_ids, *, n_draws, seed):
    """Network statistic: median over the network's events, one null per draw.

    The permutation is applied ONCE per draw to the early-ictal vector and the
    whole network statistic is recomputed, so the null keeps the events' own
    structure and asks only whether the early-ictal order matches the particular
    contacts it matched.
    """
    event_ranks = np.atleast_2d(np.asarray(event_ranks, float))
    early_ranks = np.asarray(early_ranks, float)
    if not len(event_ranks) or not np.isfinite(early_ranks).any():
        return {"status": NOT_EVALUABLE, "n_events": int(len(event_ranks))}
    observed, rows, coverage = _median_event_rank_agreement(event_ranks, early_ranks)
    if not np.isfinite(observed):
        return {"status": NOT_EVALUABLE, "n_events": int(len(event_ranks)),
                "reason": "no event shares three contacts with the early state"}
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(n_draws), float)
    for index in range(int(n_draws)):
        shuffled = within_shaft_label_permutation(early_ranks, shaft_ids, rng)
        draws[index] = _median_event_rank_agreement(event_ranks, shuffled)[0]
    finite = draws[np.isfinite(draws)]
    return {
        "status": "OK",
        "n_events": int(len(event_ranks)),
        "median_event_spearman": observed,
        "per_event_spearman": [row["rho"] for row in rows],
        "median_common_contacts": float(np.median(coverage)),
        "min_common_contacts": int(np.min(coverage)),
        "null": _null_summary(observed, finite, n_draws),
        "null_matches": list(NULL_MATCHES),
        "null_does_not_match": list(NULL_DOES_NOT_MATCH),
        "claim_boundary": CONTACT_CLAIM_BOUNDARY,
    }


def _median_event_precedence(event_values, early_values, *,
                             event_matrices=None):
    target = precedence_matrix(early_values)
    matrices = (event_matrices if event_matrices is not None
                else [precedence_matrix(row) for row in event_values])
    rows = [precedence_agreement(matrix, early_values, target_matrix=target)
            for matrix in matrices]
    values = np.asarray([row["agreement"] for row in rows], float)
    finite = values[np.isfinite(values)]
    return (float(np.median(finite)) if len(finite) else float("nan")), rows


def network_precedence_reuse(event_values, early_values, shaft_ids, *, n_draws,
                             seed, mode_labels=None):
    """Primary: median per-event precedence agreement. Secondary: per frozen mode."""
    event_values = np.atleast_2d(np.asarray(event_values, float))
    early_values = np.asarray(early_values, float)
    # the event precedence matrices do not depend on the draw; building them
    # once instead of once per event per draw is the whole cost of this loop
    event_matrices = [precedence_matrix(row) for row in event_values]
    observed, rows = _median_event_precedence(event_values, early_values,
                                              event_matrices=event_matrices)
    if not np.isfinite(observed):
        return {"status": NOT_EVALUABLE, "n_events": int(len(event_values))}
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(n_draws), float)
    for index in range(int(n_draws)):
        shuffled = within_shaft_label_permutation(early_values, shaft_ids, rng)
        draws[index] = _median_event_precedence(
            event_values, shuffled, event_matrices=event_matrices)[0]
    finite = draws[np.isfinite(draws)]
    output = {
        "status": "OK",
        "n_events": int(len(event_values)),
        "median_event_agreement": observed,
        "per_event_agreement": [row["agreement"] for row in rows],
        "median_pairs_per_event": float(np.median(
            [row["n_pairs"] for row in rows])),
        "null": _null_summary(observed, finite, n_draws),
        "null_matches": list(NULL_MATCHES),
        "null_does_not_match": list(NULL_DOES_NOT_MATCH),
        "claim_boundary": CONTACT_CLAIM_BOUNDARY,
    }
    if mode_labels is not None:
        modes = {}
        labels = np.asarray(mode_labels)
        for name in np.unique(labels[labels != None]):  # noqa: E711
            members = event_values[labels == name]
            if len(members) < 2:
                modes[str(name)] = {"status": NOT_EVALUABLE,
                                    "n_events": int(len(members))}
                continue
            matrix, _ = mode_precedence_matrix(members)
            modes[str(name)] = precedence_reuse(
                matrix, early_values, shaft_ids, n_draws=n_draws,
                seed=int(seed) + 1)
            modes[str(name)]["n_events"] = int(len(members))
        output["per_frozen_mode_weighted"] = modes
    return output
