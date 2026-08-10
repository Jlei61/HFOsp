"""Finite-library per-network and shared-oracle summaries for rev9-L."""
from __future__ import annotations

import numpy as np


def summarize_network_oracles(objective_by_candidate_network, *, tolerance=1e-12):
    values = {
        str(candidate): {int(seed): float(value) for seed, value in rows.items()}
        for candidate, rows in objective_by_candidate_network.items()
    }
    if not values:
        raise ValueError("oracle summary requires candidates")
    candidates = sorted(values)
    seeds = sorted(next(iter(values.values())))
    if any(sorted(rows) != seeds for rows in values.values()):
        raise ValueError("all oracle candidates must share network seeds")
    if any(not np.isfinite(value) for rows in values.values() for value in rows.values()):
        raise ValueError("oracle objectives must be finite")

    per_network = []
    for seed in seeds:
        minimum = min(values[candidate][seed] for candidate in candidates)
        tied = [
            candidate for candidate in candidates
            if abs(values[candidate][seed] - minimum) <= tolerance
        ]
        per_network.append({
            "network_seed": seed,
            "minimum_objective": float(minimum),
            "tied_candidate_ids": tied,
            "representative_candidate_id": tied[0],
        })
    medians = {
        candidate: float(np.median(list(rows.values())))
        for candidate, rows in values.items()
    }
    means = {
        candidate: float(np.mean(list(rows.values())))
        for candidate, rows in values.items()
    }
    shared_minimum = min(medians.values())
    shared_ties = [
        candidate for candidate in candidates
        if abs(medians[candidate] - shared_minimum) <= tolerance
    ]
    shared_candidate = min(
        shared_ties, key=lambda candidate: (means[candidate], candidate))
    c_per_net = float(np.median([
        row["minimum_objective"] for row in per_network
    ]))
    return {
        "per_network": per_network,
        "C_per_net": c_per_net,
        "candidate_median_objective": medians,
        "candidate_mean_objective": means,
        "shared": {
            "C_shared": float(shared_minimum),
            "tied_candidate_ids": shared_ties,
            "n_tied_candidates": len(shared_ties),
            "tie_break_rule": "lowest mean objective, then candidate id",
            "selected_candidate_id": shared_candidate,
        },
        "Delta_network": float(shared_minimum - c_per_net),
    }


__all__ = ["summarize_network_oracles"]
