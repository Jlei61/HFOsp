#!/usr/bin/env python3
"""Prove the motif nulls preserve what they claim, before any of them is used.

Two kinds of check:

* synthetic graph with a planted, distance-orthogonal source motif -- the
  permutation must destroy the motif while leaving topology, delays, degrees and
  target-wise budgets untouched;
* the frozen 32 M-edge substrate -- the same structural clauses, on the graph the
  null would actually be applied to.

The edge null has no observed statistic to be compared against in this round,
because no archived Z/M artifact stores a per-window recurrent-E edge flow. The
audit is still run so the null is proved before it could be used, exactly as the
plan requires, and so the missing evidence is recorded rather than assumed away.

No SNN is launched here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "src" / "snn_engine")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from scripts.freeze_topic4_zm_discovery_boundary import (  # noqa: E402
    load_audit_config, sha256_file)
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_fig5_motif_reuse import (  # noqa: E402
    NOT_EVALUABLE, audit_edge_permutation, permute_edge_weights,
    within_shaft_label_permutation)


def synthetic_edge_audit(seed=0, n_e=60, n_i=15, n_bins=3, n_distance_bins=4):
    import scipy.sparse as sp
    rng = np.random.default_rng(seed)
    n_total = n_e + n_i
    positions = rng.uniform(0.0, 20.0, size=(n_total, 2))
    motif = np.zeros(n_e, bool)
    motif[rng.choice(n_e, size=n_e // 2, replace=False)] = True
    bins = []
    for _ in range(n_bins):
        rows, cols = [], []
        for target in range(n_total):
            sources = rng.choice(n_e, size=8, replace=False)
            rows.extend([target] * len(sources))
            cols.extend(sources.tolist())
        rows, cols = np.asarray(rows), np.asarray(cols)
        data = np.where(motif[cols], 3.0, 1.0) * rng.uniform(0.9, 1.1, len(cols))
        bins.append(sp.coo_matrix((data, (rows, cols)),
                                  shape=(n_total, n_e)).tocsr())

    def alignment(source_bins):
        values, flags = [], []
        for matrix in source_bins:
            coo = matrix.tocoo(copy=False)
            values.append(np.asarray(coo.data, float))
            flags.append(motif[np.asarray(coo.col, np.int64)].astype(float))
        return float(np.corrcoef(np.concatenate(values),
                                 np.concatenate(flags))[0, 1])

    permuted = permute_edge_weights(bins, n_e, positions,
                                    rng=np.random.default_rng(seed + 1),
                                    n_distance_bins=n_distance_bins)
    report = audit_edge_permutation(bins, permuted, n_e, positions,
                                    n_distance_bins=n_distance_bins)
    report["planted_motif_alignment_before"] = alignment(bins)
    report["planted_motif_alignment_after"] = alignment(permuted)
    report["motif_identity_destroyed"] = bool(
        abs(report["planted_motif_alignment_after"]) < 0.25
        and report["planted_motif_alignment_before"] > 0.8)
    return report


def contact_null_audit(config, draws=512, seed=0):
    """The within-shaft permutation must keep per-shaft recruitment counts."""
    contract = json.loads(
        (ROOT / config["immutable_inputs"]["contact_contract"]["path"]).read_text())
    shafts = np.asarray([row["shaft_id"] for row in contract["contacts"]])
    rng = np.random.default_rng(seed)
    values = np.arange(float(len(shafts)))
    values[[1, 4, 11]] = np.nan
    preserved, moved = [], []
    for _ in range(int(draws)):
        permuted = within_shaft_label_permutation(values, shafts, rng)
        ok = True
        for shaft in np.unique(shafts):
            index = shafts == shaft
            ok &= int(np.isfinite(permuted[index]).sum()) == int(
                np.isfinite(values[index]).sum())
            ok &= sorted(permuted[index][np.isfinite(permuted[index])].tolist()) \
                == sorted(values[index][np.isfinite(values[index])].tolist())
        preserved.append(bool(ok))
        moved.append(not np.array_equal(np.nan_to_num(permuted, nan=-1.0),
                                        np.nan_to_num(values, nan=-1.0)))
    return {
        "draws": int(draws),
        "per_shaft_recruitment_count_and_values_preserved": bool(all(preserved)),
        "fraction_of_draws_that_move_at_least_one_label": float(np.mean(moved)),
        "shaft_sizes": {shaft: int((shafts == shaft).sum())
                        for shaft in np.unique(shafts).tolist()},
    }


def frozen_graph_audit(config, seed, n_distance_bins):
    from src.topic4_zm_ictal_transition import build_substrate
    round_config = json.loads(
        (ROOT / config["immutable_inputs"]["round_config"]["path"]).read_text())
    cache = ROOT / round_config["output_root"] / "network_cache"
    substrate = build_substrate(round_config, "joint_04_control", int(seed),
                                cache_dir=str(cache))
    positions = np.concatenate([substrate.positions_e, substrate.positions_i])
    original = substrate.net["ampa_by_delay"]
    permuted = permute_edge_weights(
        original, substrate.n_e, positions,
        rng=np.random.default_rng(int(seed)), n_distance_bins=n_distance_bins)
    report = audit_edge_permutation(original, permuted, substrate.n_e, positions,
                                    n_distance_bins=n_distance_bins)
    report["seed"] = int(seed)
    report["n_edges"] = int(sum(matrix.nnz for matrix in original))
    report["n_distance_bins"] = int(n_distance_bins)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/topic4_data_driven_zm_discovery_audit_v1.json")
    parser.add_argument("--seed", type=int, default=1801)
    parser.add_argument("--distance-bins", type=int, default=8)
    parser.add_argument("--skip-frozen-graph", action="store_true")
    args = parser.parse_args()

    config = load_audit_config(args.config)
    frozen = config["motif_reuse"]["frozen_permutations"]
    report = {
        "status": config["status"],
        "frozen_permutation_counts": frozen,
        "declared_before_scores": config["motif_reuse"]["declared_before_scores"],
        "within_shaft_contact_permutation": contact_null_audit(
            config, draws=512, seed=int(frozen["within_shaft_contact_permutation"]["seed"])),
        "onset_circular_shift": {
            "preserves": ["event count", "event rate", "transition time"],
            "destroys": ["which event sits at which time-to-transition"],
            "applies_to": "the reuse-versus-time-to-transition trajectory only",
        },
        "learned_edge_gain_permutation": {
            "synthetic": synthetic_edge_audit(
                seed=int(frozen["edge_gain_permutation"]["seed"]),
                n_distance_bins=args.distance_bins),
        },
        "matched_off_motif_node_sets": {
            "status": NOT_EVALUABLE,
            "reason": ("needs a node-level motif from a per-window recurrent-E "
                       "edge flow and per-node baseline rates; no archived Z/M "
                       "artifact stores either"),
        },
        "edge_flow_reuse_statistic": {
            "status": NOT_EVALUABLE,
            "reason": ("no producer in this round wrote a per-window recurrent-E "
                       "edge flow, so the edge null has no observed statistic to "
                       "be compared against; the null is audited, not run"),
        },
        "simulation_launched": False,
    }
    if not args.skip_frozen_graph:
        report["learned_edge_gain_permutation"]["frozen_graph"] = frozen_graph_audit(
            config, args.seed, args.distance_bins)

    output_root = ROOT / config["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    report["audit_config_sha256"] = sha256_file(ROOT / args.config)
    atomic_write_json(report, str(output_root / "motif_null_audit.json"))
    synthetic = report["learned_edge_gain_permutation"]["synthetic"]
    print(json.dumps({
        "synthetic_structural_pass": synthetic["all_structural_clauses_pass"],
        "synthetic_motif_destroyed": synthetic["motif_identity_destroyed"],
        "frozen_graph_structural_pass": report["learned_edge_gain_permutation"]
        .get("frozen_graph", {}).get("all_structural_clauses_pass"),
        "contact_null_preserved": report["within_shaft_contact_permutation"][
            "per_shaft_recruitment_count_and_values_preserved"],
    }, indent=1))


if __name__ == "__main__":
    main()
