"""Select D5.3 by canonical masked-rank KMeans and patient geometry."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

from scripts.paper_figures.plot_fig4_spatial_edge_flow_validation import (
    _canonical_rank_kmeans,
    _column_stats,
    _load_bundle,
    _patient_profiles,
    normalize_event_ranks,
)
from scripts.run_topic4_rev9l_forced_source_worker import _sha256
from src.lagpat_rank_audit import build_masked_kmeans_features


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_d5_3_spatial_ou_kmeans_grid.json"


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        Path(temporary).write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def direction_purity(labels, direction):
    contingency = np.zeros((2, 2), int)
    for cluster, mode in zip(np.asarray(labels, int), np.asarray(direction, int)):
        contingency[int(cluster), int(mode)] += 1
    identity = contingency[0, 0] + contingency[1, 1]
    swapped = contingency[0, 1] + contingency[1, 0]
    return float(max(identity, swapped) / max(1, contingency.sum())), contingency


def _rank_kmeans(ranks, direction, *, random_state):
    rank_matrix = np.asarray(ranks, float).T
    participation = np.isfinite(rank_matrix)
    valid = participation.sum(axis=0) >= 3
    rank_matrix, participation = rank_matrix[:, valid], participation[:, valid]
    direction = np.asarray(direction, int)[valid]
    if len(direction) < 4 or len(np.unique(direction)) != 2:
        raise RuntimeError("balanced KMeans draw has insufficient direction support")
    features = build_masked_kmeans_features(
        rank_matrix, participation, impute="event_median",
    )
    labels = KMeans(
        n_clusters=2, n_init=50, random_state=int(random_state),
    ).fit_predict(features)
    purity, contingency = direction_purity(labels, direction)
    return purity, contingency


def balanced_network_mode_bootstrap(bundle, *, events_per_mode, draws, seed):
    valid_rank = np.sum(np.isfinite(bundle["ranks"]), axis=1) >= 3
    eligible = []
    by_seed_mode = {}
    record_seed = np.asarray([row["seed"] for row in bundle["records"]], int)
    for network_seed in bundle["config"]["search"][bundle["network_seed_key"]]:
        by_mode = []
        for mode in (0, 1):
            by_mode.append(np.flatnonzero(
                bundle["clean"] & valid_rank & (bundle["labels"] == mode)
                & (record_seed == int(network_seed))
            ))
        by_seed_mode[int(network_seed)] = by_mode
        if min(map(len, by_mode)) >= 3:
            eligible.append(int(network_seed))
    if len(eligible) < 2:
        return {
            "status": "INSUFFICIENT_NETWORK_MODE_SUPPORT",
            "eligible_network_seeds": eligible,
            "n_eligible_networks": len(eligible),
            "draws": [],
        }

    rng = np.random.default_rng(int(seed))
    values, contingencies = [], []
    for repeat in range(int(draws)):
        selected = []
        for network_seed in eligible:
            for mode in (0, 1):
                available = by_seed_mode[network_seed][mode]
                selected.extend(rng.choice(
                    available, size=int(events_per_mode),
                    replace=len(available) < int(events_per_mode),
                ).tolist())
        selected = np.asarray(selected, int)
        value, contingency = _rank_kmeans(
            bundle["ranks"][selected], bundle["labels"][selected],
            random_state=repeat,
        )
        values.append(value)
        contingencies.append(contingency.tolist())
    values = np.asarray(values, float)
    return {
        "status": "OK",
        "sampling": (
            "equal eligible-network and supervised-mode weight; six draws per "
            "network-mode with replacement only when fewer than six are available"
        ),
        "eligible_network_seeds": eligible,
        "n_eligible_networks": len(eligible),
        "events_per_mode_per_network": int(events_per_mode),
        "n_draws": int(len(values)),
        "purity_q05": float(np.quantile(values, 0.05)),
        "purity_median": float(np.median(values)),
        "purity_q95": float(np.quantile(values, 0.95)),
        "purity_draws": values.tolist(),
        "contingency_draws": contingencies,
    }


def supervised_patient_matrix(bundle, canonical_index):
    index = np.asarray(canonical_index, int)
    ranks = normalize_event_ranks(bundle["ranks"][index])
    direction = np.asarray(bundle["labels"][index], int)
    model = np.asarray([
        _column_stats(ranks[direction == mode])[0] for mode in (0, 1)
    ])
    patient = _patient_profiles(bundle)[0]
    matrix = np.full((2, 2), np.nan)
    for row in range(2):
        for column in range(2):
            finite = np.isfinite(model[row]) & np.isfinite(patient[column])
            if np.sum(finite) >= 3:
                matrix[row, column] = spearmanr(
                    model[row, finite], patient[column, finite]
                ).statistic
    signed = np.asarray([
        matrix[0, 0], matrix[1, 1], -matrix[0, 1], -matrix[1, 0],
    ], float)
    margin = float(np.nanmin(signed)) if np.any(np.isfinite(signed)) else -1.0
    return matrix, float(np.clip(margin, -1.0, 1.0))


def continuous_selection_score(*, purity, signed_margin, ood, occupancy):
    return float(
        (1.0 - float(purity))
        + 0.125 * (1.0 - float(np.clip(signed_margin, -1.0, 1.0)))
        + 0.10 * float(ood)
        + 0.05 * float(occupancy)
    )


def audit_candidate(config_path, root, candidate, aggregate_row, selection):
    bundle = _load_bundle(
        config_path, root, candidate["candidate_id"],
        allow_exploratory_candidate=True,
    )
    bootstrap = balanced_network_mode_bootstrap(
        bundle,
        events_per_mode=selection["balanced_events_per_mode_per_network"],
        draws=selection["bootstrap_draws"], seed=selection["bootstrap_seed"],
    )
    evaluable = bool(
        aggregate_row["n_runaway_networks"] == 0
        and bootstrap["n_eligible_networks"] >= selection[
            "minimum_networks_with_three_clean_events_per_mode"
        ]
        and bootstrap["status"] == "OK"
    )
    if evaluable:
        canonical = _canonical_rank_kmeans(bundle)
        matrix, margin = supervised_patient_matrix(
            bundle, canonical["clean_global_index"],
        )
        pooled = {
            "n_events": int(len(canonical["labels"])),
            "cluster_counts": canonical["cluster_counts"].tolist(),
            "direction_contingency": canonical[
                "direction_contingency"
            ].tolist(),
            "direction_purity": float(canonical["direction_purity"]),
            "stability_ami_median": canonical["stability_ami_median"],
            "silhouette_median": canonical["silhouette_median"],
        }
    else:
        matrix, margin = np.full((2, 2), np.nan), -1.0
        pooled = {
            "status": "NOT_EVALUABLE_SUPPORT",
            "n_formal_clean_events": int(np.sum(bundle["clean"])),
            "formal_clean_mode_counts": np.bincount(
                bundle["labels"][bundle["clean"]], minlength=2,
            ).tolist(),
        }
    score = None
    if evaluable:
        score = continuous_selection_score(
            purity=bootstrap["purity_median"], signed_margin=margin,
            ood=aggregate_row["mean_network_ood_fraction"],
            occupancy=aggregate_row["mean_network_fraction_time_above_detector"],
        )
    return {
        "candidate_id": candidate["candidate_id"],
        "spatial_ou": candidate["spatial_ou"],
        "evaluable": evaluable,
        "selection_score": score,
        "balanced_kmeans": bootstrap,
        "pooled_canonical_kmeans": pooled,
        "supervised_direction_vs_patient_spearman": matrix.tolist(),
        "signed_patient_margin": margin,
        "activity": {
            "mean_network_ood_fraction": float(
                aggregate_row["mean_network_ood_fraction"]
            ),
            "mean_network_fraction_time_above_detector": float(
                aggregate_row["mean_network_fraction_time_above_detector"]
            ),
            "mean_network_returned_events": float(
                aggregate_row["mean_network_returned_events_scored"]
            ),
            "networks_with_both_clean_modes": int(
                aggregate_row["networks_with_both_clean_modes"]
            ),
            "n_runaway_networks": int(aggregate_row["n_runaway_networks"]),
        },
    }


def adjudicate(rows, *, anchor_purity, patient_q05):
    eligible = [row for row in rows if row["evaluable"]]
    selected = min(eligible, key=lambda row: row["selection_score"]) if eligible else None
    if selected is None:
        status = "REV10D5_3_NO_KMEANS_EVALUABLE_CONTINUOUS_OU_CANDIDATE"
    elif (selected["balanced_kmeans"]["purity_median"] > anchor_purity
          and selected["signed_patient_margin"] > 0):
        status = "REV10D5_3_KMEANS_CANDIDATE_SELECTED_FOR_FRESH_CONFIRMATION"
    else:
        status = "REV10D5_3_KMEANS_GRID_DID_NOT_IMPROVE_FROZEN_ANCHOR"
    return {
        "status": status,
        "selected_candidate_id": (
            None if selected is None else selected["candidate_id"]
        ),
        "selected_row": selected,
        "d5_2_anchor_direction_purity": float(anchor_purity),
        "patient_matched_direction_purity_q05": float(patient_q05),
        "selection_is_exploratory": True,
        "candidate_rows": sorted(
            rows,
            key=lambda row: (
                row["selection_score"] is None,
                np.inf if row["selection_score"] is None else row["selection_score"],
            ),
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    root = ROOT / config["output_root"]
    manifest_path = root / "candidate_manifest.json"
    summary_path = root / "canary_summary_returned_only.json"
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())
    if manifest.get("status") != "REV10D5_3_SPATIAL_OU_KMEANS_GRID_FROZEN":
        raise RuntimeError("D5.3 manifest is not frozen")
    if summary.get("status") != "REV10D5_3_RETURNED_ONLY_KMEANS_GRID_COMPLETE":
        raise RuntimeError("D5.3 aggregate is incomplete")
    aggregate = {row["candidate_id"]: row for row in summary["candidate_rows"]}
    selection = config["search"]["kmeans_selection"]
    rows = [
        audit_candidate(config_path, root, candidate,
                        aggregate[candidate["candidate_id"]], selection)
        for candidate in manifest["candidate_set"]["candidates"]
        if candidate["spatial_ou"]["mode"] == "local"
    ]
    payload = adjudicate(
        rows,
        anchor_purity=manifest["d5_2_anchor"]["canonical_direction_purity"],
        patient_q05=manifest["d5_2_anchor"]["patient_matched_q05"],
    )
    payload.update({
        "selection_contract": selection,
        "claim_boundary": (
            "new development networks; canonical KMeans-guided OU dose/time "
            "grid only; not final Fig4 confirmation or patient generalization"
        ),
        "inputs": {
            "config": {"path": str(config_path.relative_to(ROOT)),
                       "sha256": _sha256(config_path)},
            "manifest": {"path": str(manifest_path.relative_to(ROOT)),
                         "sha256": _sha256(manifest_path)},
            "summary": {"path": str(summary_path.relative_to(ROOT)),
                        "sha256": _sha256(summary_path)},
        },
    })
    output = root / "canary_verdict.json"
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"],
        "selected_candidate_id": payload["selected_candidate_id"],
        "selected_score": (
            None if payload["selected_row"] is None
            else payload["selected_row"]["selection_score"]
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
