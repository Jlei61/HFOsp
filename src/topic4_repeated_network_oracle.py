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


def review_repeated_capacity(payload, floors, *, baseline_id="sobol_000",
                             tolerance=1e-12):
    """Interpret the finite-library oracle without turning it into a pass gate."""
    seeds = [int(seed) for seed in payload["network_seeds"]]
    oracle = payload["oracle"]
    shared_id = oracle["shared"]["selected_candidate_id"]
    values = payload["objective_by_candidate_network"]
    if baseline_id not in values or shared_id not in values:
        raise ValueError("repeated-capacity review lacks baseline or shared candidate")
    rows = {
        (row["candidate_id"], int(row["network_seed"])): row
        for row in payload["candidate_network_rows"]
    }
    baseline = np.asarray([values[baseline_id][str(seed)] for seed in seeds], float)
    shared = np.asarray([values[shared_id][str(seed)] for seed in seeds], float)
    shared_gain = baseline - shared

    per_network = []
    oracle_gains = []
    n_mode_a_within_q95 = 0
    for oracle_row in oracle["per_network"]:
        seed = int(oracle_row["network_seed"])
        candidate_id = oracle_row["representative_candidate_id"]
        row = rows[(candidate_id, seed)]
        score = row["score"]
        count = int(score["matched_floor_event_count_by_mode"]["A"])
        if count not in floors:
            raise ValueError(f"missing mode-A floor for n={count}")
        calibration = floors[count]["floor"]["modes"]["A"]
        descriptors = score["standardized_descriptors"]["A"]
        comparisons = {
            name: {
                "raw": float(record["raw"]),
                "patient_training_q95": float(calibration[name]["q95"]),
                "raw_over_q95": float(record["raw"] / calibration[name]["q95"]),
                "above_q95": bool(record["raw"] > calibration[name]["q95"]),
            }
            for name, record in descriptors.items()
        }
        n_above = int(sum(value["above_q95"] for value in comparisons.values()))
        n_mode_a_within_q95 += int(n_above == 0)
        gain = float(values[baseline_id][str(seed)]
                     - oracle_row["minimum_objective"])
        oracle_gains.append(gain)
        per_network.append({
            "network_seed": seed,
            "candidate_id": candidate_id,
            "objective_gain_vs_scalar": gain,
            "mode_A_score": float(score["mode_scores"]["A"]),
            "mode_B_score": float(score["mode_scores"]["B"]),
            "mode_A_event_count": count,
            "mode_A_descriptors": comparisons,
            "n_mode_A_descriptors_above_patient_q95": n_above,
        })

    shared_improved = int(np.sum(shared_gain > float(tolerance)))
    shared_mean_gain = float(np.mean(shared_gain))
    mode_a_capacity_observed = n_mode_a_within_q95 > len(seeds) / 2.0
    shared_capacity_supported = bool(
        shared_mean_gain > float(tolerance)
        and shared_improved > len(seeds) / 2.0
        and mode_a_capacity_observed)
    if shared_capacity_supported:
        status = "FINITE_LIBRARY_SHARED_FORCED_CAPACITY_SUPPORTED"
    elif n_mode_a_within_q95 == 0:
        status = "FINITE_LIBRARY_MODE_A_CAPACITY_NOT_OBSERVED"
    else:
        status = "FINITE_LIBRARY_SHARED_CAPACITY_NOT_OBSERVED"
    return {
        "status": status,
        "baseline_candidate_id": baseline_id,
        "shared_candidate_id": shared_id,
        "baseline_median_objective": float(np.median(baseline)),
        "baseline_mean_objective": float(np.mean(baseline)),
        "shared_median_objective": float(np.median(shared)),
        "shared_mean_objective": float(np.mean(shared)),
        "shared_gain_vs_scalar_by_network": {
            str(seed): float(gain) for seed, gain in zip(seeds, shared_gain)},
        "shared_n_networks_improved": shared_improved,
        "shared_n_networks_total": len(seeds),
        "shared_mean_gain_vs_scalar": shared_mean_gain,
        "shared_median_paired_gain_vs_scalar": float(np.median(shared_gain)),
        "per_network_oracle_gain_vs_scalar": per_network,
        "per_network_oracle_median_gain": float(np.median(oracle_gains)),
        "per_network_oracle_mean_gain": float(np.mean(oracle_gains)),
        "per_network_oracle_improved_all_networks": bool(
            np.all(np.asarray(oracle_gains) > float(tolerance))),
        "n_networks_with_mode_A_all_descriptors_within_patient_q95": (
            n_mode_a_within_q95),
        "mode_A_capacity_observed": bool(mode_a_capacity_observed),
        "shared_forced_capacity_supported": shared_capacity_supported,
    }
__all__ = ["review_repeated_capacity", "summarize_network_oracles"]
