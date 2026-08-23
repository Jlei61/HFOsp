#!/usr/bin/env python3
"""Aggregate cascade-event fields with matched sampling and KMeans alignment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.aggregate_topic4_rev12_node_canary import (  # noqa: E402
    _source_bundle,
    _strip_natural_arrays,
)
from scripts.rescore_topic4_rev12_node_historical import (  # noqa: E402
    DEFAULT_ARTIFACT_ROOT,
    _classifier_contract,
    _load_network_worker,
    _old_to_patient_label_map,
    _patient_data,
    _reorder_patient_contract,
    score_candidate,
)
from src.topic4_d6_natural_kmeans import natural_kmeans  # noqa: E402
from src.topic4_node_dualmode import (  # noqa: E402
    calibrate_component_scales,
    fixed_projection_matrix,
    matched_sample_dual_mode_objective,
    topology_network_reproducibility,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(artifact_root: Path, relative: str) -> Path:
    local = ROOT / relative
    return local if local.exists() else artifact_root / relative


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(
            json.dumps(_jsonable(payload), indent=2, allow_nan=False) + "\n"
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def cascade_selection_objective(*, patient_loss: float,
                                kmeans_balanced_alignment: float,
                                ood_fraction: float,
                                compound_fraction: float,
                                k2_support: float | None = None,
                                k2_support_weight: float = 0.0) -> dict:
    values = np.asarray([
        patient_loss, kmeans_balanced_alignment, ood_fraction, compound_fraction,
    ], float)
    if not np.all(np.isfinite(values)) or not 0.0 <= kmeans_balanced_alignment <= 1.0:
        raise ValueError("cascade objective inputs must be finite and bounded")
    if (k2_support is not None and not 0.0 <= float(k2_support) <= 1.0) \
            or float(k2_support_weight) < 0.0:
        raise ValueError("K2 support and weight must be bounded")
    kmeans_loss = 1.0 - float(kmeans_balanced_alignment)
    k2_loss = 0.0 if k2_support is None else 1.0 - float(k2_support)
    return {
        "matched_patient_loss": float(patient_loss),
        "kmeans_direction_loss": kmeans_loss,
        "ood_fraction": float(ood_fraction),
        "compound_fraction": float(compound_fraction),
        "k2_support": None if k2_support is None else float(k2_support),
        "k2_support_loss": float(k2_loss),
        "objective": float(
            patient_loss + 0.5 * kmeans_loss
            + 0.25 * ood_fraction + 0.25 * compound_fraction
            + float(k2_support_weight) * k2_loss
        ),
        "weights": {
            "matched_patient_loss": 1.0,
            "kmeans_direction_loss": 0.5,
            "ood_fraction": 0.25,
            "compound_fraction": 0.25,
            "k2_support_loss": float(k2_support_weight),
        },
    }


def validate_contact_readout(payload: dict, expected: dict) -> None:
    """Fail closed when a worker used another event-to-contact mapping."""
    source = expected.get("source")
    if source is None:
        return
    observed = payload.get("contact_readout", {})
    if observed.get("source") != source:
        raise RuntimeError("aggregate received the wrong contact readout source")
    if source == "lineage_restricted_sheet_activity":
        required = float(expected["minimum_full_trace_pearson"])
        minimum = observed.get("full_trace_pearson_minimum")
        if (observed.get("parity_status") != "PASS" or minimum is None
                or float(minimum) < required):
            raise RuntimeError("worker contact sampler parity is not acceptable")


def equal_network_natural_kmeans(workers: list[dict], *, seed: int) -> dict:
    """Run pooled KMeans after giving every network the same event count."""
    if not workers:
        raise ValueError("KMeans requires at least one network")
    n_per_network = min(len(worker["ranks"]) for worker in workers)
    if n_per_network < 2:
        return {
            "status": "INSUFFICIENT_EQUAL_NETWORK_EVENTS",
            "n_per_network": int(n_per_network),
            "direction_balanced_alignment": 0.0,
            "direction_purity": 0.0,
        }
    rng = np.random.default_rng(int(seed))
    ranks, labels = [], []
    for worker in workers:
        selected = rng.choice(len(worker["ranks"]), size=n_per_network, replace=False)
        ranks.append(np.asarray(worker["ranks"])[selected])
        labels.append(np.asarray(worker["labels"])[selected])
    result = _strip_natural_arrays(natural_kmeans(
        np.concatenate(ranks), np.concatenate(labels), random_state=int(seed),
    ))
    result["n_per_network"] = int(n_per_network)
    result["network_weighting"] = "equal event count per network"
    return result


def k2_support_score(delta_loglik_per_event: float | None, *,
                     scale: float = 0.5) -> float:
    """Map held-out K2-vs-K1 evidence to a bounded continuous score."""
    if delta_loglik_per_event is None or not np.isfinite(delta_loglik_per_event):
        return 0.0
    if float(scale) <= 0.0:
        raise ValueError("K2 support scale must be positive")
    return float(expit(float(delta_loglik_per_event) / float(scale)))


def per_network_natural_kmeans(workers: list[dict], *, seed: int,
                               support_scale: float = 0.5) -> dict:
    """Keep network seed as the unit for natural two-cluster evidence."""
    rows = []
    for index, worker in enumerate(workers):
        result = _strip_natural_arrays(natural_kmeans(
            worker["ranks"], worker["labels"],
            random_state=int(seed) + index,
        ))
        result["seed"] = int(worker["seed"])
        result["k2_support_score"] = k2_support_score(
            result.get("heldout_gmm_k2_minus_k1_loglik_per_event"),
            scale=float(support_scale),
        )
        rows.append(result)
    evaluable = [row for row in rows if row.get("status") == "OK"]
    return {
        "rows": rows,
        "n_networks": len(rows),
        "n_evaluable": len(evaluable),
        "equal_network_mean_balanced_alignment": float(np.mean([
            row["direction_balanced_alignment"] for row in evaluable
        ])) if evaluable else 0.0,
        "equal_network_mean_k2_support": float(np.mean([
            row["k2_support_score"] for row in rows
        ])) if rows else 0.0,
        "networks_with_positive_k2_delta": int(np.sum([
            row.get("heldout_gmm_k2_minus_k1_loglik_per_event") is not None
            and row["heldout_gmm_k2_minus_k1_loglik_per_event"] > 0.0
            for row in rows
        ])),
        "support_scale_loglik_per_event": float(support_scale),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument(
        "--seed-pool", choices=("canary", "fit", "selection", "confirmation"),
        default="fit",
    )
    args = parser.parse_args()
    artifact_root = args.artifact_root.resolve()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    for record in config["inputs"].values():
        path = _resolve(artifact_root, record["path"])
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")
    manifest_path = artifact_root / config["candidate_manifest"]
    manifest = json.loads(manifest_path.read_text())
    if manifest["config_sha256"] != _sha256(config_path):
        raise RuntimeError("cascade candidate manifest is stale")
    cohort = json.loads(_resolve(
        artifact_root, config["inputs"]["cohort_config"]["path"],
    ).read_text())
    classifier_config = json.loads(_resolve(
        artifact_root, config["inputs"]["classifier_config"]["path"],
    ).read_text())
    patient = _patient_data(cohort, artifact_root)
    classifier = _classifier_contract(classifier_config, artifact_root)
    semantics = _old_to_patient_label_map(patient, classifier)
    patient = _reorder_patient_contract(patient, classifier["names"])
    projections = fixed_projection_matrix(
        2 * len(patient["contact_names"]), n_directions=64, seed=20260824,
    )
    objective_config = config["cascade_objective"]
    expected_event_unit = config["event_unit"]["name"]
    calibration = calibrate_component_scales(
        patient["train_ranks"], patient["train_labels"], patient["train_blocks"],
        patient["contact_names"], projections,
        sample_size=int(objective_config["sample_size_per_side"]),
        draws=int(objective_config["calibration_draws"]),
        seed=int(objective_config["seed"]),
    )
    seeds = [
        int(seed) for seed in config["search"][f"{args.seed_pool}_network_seeds"]
    ]
    output_root = artifact_root / config["output_root"]
    rows = []
    for candidate in manifest["candidates"]:
        workers, worker_payloads, source_maps, source_labels = [], [], [], []
        for seed in seeds:
            stem = f"{candidate['candidate_id']}_seed_{seed}"
            npz_path = output_root / "workers" / f"{stem}.npz"
            json_path = npz_path.with_suffix(".json")
            if not npz_path.exists() or not json_path.exists():
                continue
            payload = json.loads(json_path.read_text())
            if payload["event_unit"].get("name") != expected_event_unit:
                raise RuntimeError("aggregate received the wrong frozen event unit")
            validate_contact_readout(
                payload, config["search"].get("contact_readout", {}),
            )
            worker = _load_network_worker(
                npz_path, patient["contact_names"], classifier,
                semantics["raw_to_patient"],
            )
            maps, labels = _source_bundle(npz_path, worker)
            workers.append(worker)
            worker_payloads.append(payload)
            source_maps.append(maps)
            source_labels.append(labels)
        if not workers:
            continue
        if len(workers) != len(seeds):
            raise RuntimeError(
                f"candidate {candidate['candidate_id']} is missing a frozen network"
            )
        matched_scores = [
            matched_sample_dual_mode_objective(
                worker["ranks"], worker["labels"],
                patient["train_ranks"], patient["train_labels"],
                patient["train_blocks"], patient["contact_names"],
                projections=projections, calibration=calibration,
                sample_size=int(objective_config["sample_size_per_side"]),
                draws=int(objective_config["score_draws"]),
                seed=int(objective_config["seed"]) + int(worker["seed"]),
            )
            for worker in workers
        ]
        natural = equal_network_natural_kmeans(
            workers, seed=int(objective_config["seed"]),
        )
        network_natural = per_network_natural_kmeans(
            workers, seed=int(objective_config["seed"]),
            support_scale=float(objective_config.get("k2_support_scale", 0.5)),
        )
        ood_fraction = float(np.mean([
            float(np.mean(worker["ood"])) if len(worker["ood"]) else 1.0
            for worker in workers
        ]))
        compound_fraction = float(np.mean([
            payload["event_unit"]["compound_detector_fragment_fraction"]
            for payload in worker_payloads
        ]))
        matched_loss = float(np.mean([score["objective"] for score in matched_scores]))
        alignment_source = objective_config.get(
            "kmeans_alignment_source", "pooled_equal_event_count",
        )
        if alignment_source == "pooled_equal_event_count":
            balanced_alignment = float(natural["direction_balanced_alignment"])
        elif alignment_source == "equal_network_mean":
            balanced_alignment = float(
                network_natural["equal_network_mean_balanced_alignment"]
            )
        else:
            raise RuntimeError(f"unknown KMeans alignment source: {alignment_source}")
        selection = cascade_selection_objective(
            patient_loss=matched_loss,
            kmeans_balanced_alignment=balanced_alignment,
            ood_fraction=ood_fraction,
            compound_fraction=compound_fraction,
            k2_support=float(network_natural["equal_network_mean_k2_support"]),
            k2_support_weight=float(objective_config.get("k2_support_weight", 0.0)),
        )
        historical_diagnostics = score_candidate(
            candidate, workers, patient, projections, calibration,
        )
        topology = topology_network_reproducibility(source_maps, source_labels)
        rows.append({
            "candidate_id": candidate["candidate_id"],
            "field_sha256": candidate["node_field"]["field_sha256"],
            "n_networks": len(workers),
            "selection_objective": selection,
            "matched_network_scores": matched_scores,
            "equal_network_natural_kmeans": natural,
            "per_network_natural_kmeans": network_natural,
            "historical_unmatched_diagnostics": historical_diagnostics,
            "source_topology": topology,
            "per_seed": [{
                "seed": worker["seed"],
                "n_cascade_events": worker["n_returned"],
                "patient_mode_counts": np.bincount(worker["labels"], minlength=2),
                "ood_fraction": (
                    float(np.mean(worker["ood"])) if len(worker["ood"]) else 1.0
                ),
                "compound_fraction": payload["event_unit"][
                    "compound_detector_fragment_fraction"
                ],
                "runaway_early_stop_ms": payload["simulation"][
                    "runaway_early_stop_ms"
                ],
            } for worker, payload in zip(workers, worker_payloads)],
        })
    if not rows:
        raise RuntimeError(f"no completed {args.seed_pool} cascade workers")
    rows.sort(key=lambda row: (
        row["selection_objective"]["objective"], row["candidate_id"],
    ))
    payload = {
        "schema_id": "topic4_rev12_cascade_fit_aggregate_v1",
        "status": "REV12ND_CASCADE_FIT_AGGREGATE_COMPLETE",
        "scientific_role": "development_only_node_dualmode_refit",
        "seed_pool": args.seed_pool,
        "requested_seeds": seeds,
        "rows": rows,
        "component_calibration": calibration,
        "selection_contract": {
            "patient_sampling": "matched 6 model vs 6 one-block patient events",
            "normalization": "raw distance / patient block floor q95; no clipping",
            "patient_data": "training only",
            "kmeans_alignment_source": objective_config.get(
                "kmeans_alignment_source", "pooled_equal_event_count",
            ),
            "k2_support": (
                "per-network held-out diagonal-GMM K2-vs-K1 log likelihood, "
                "mapped continuously; no hard blocker"
            ),
            "heldout_r2_used_for_selection": False,
            "hard_scientific_gates": [],
        },
        "inputs": {
            "config": str(config_path), "config_sha256": _sha256(config_path),
            "candidate_manifest": str(manifest_path),
            "candidate_manifest_sha256": _sha256(manifest_path),
        },
        "claim_boundary": (
            "Exploratory cascade-event field ranking. Patient held-out R2, topology "
            "and figures are diagnostics and do not select this fit library."
        ),
    }
    aggregate = output_root / "aggregate"
    _atomic_json(aggregate / f"{args.seed_pool}_cascade_summary.json", payload)
    aggregate.mkdir(parents=True, exist_ok=True)
    with (aggregate / f"{args.seed_pool}_cascade_summary.csv").open(
            "w", newline="") as handle:
        fields = [
            "candidate_id", "objective", "matched_patient_loss",
            "kmeans_balanced_alignment", "k2_support", "ood_fraction",
            "compound_fraction", "heldout_r2_diagnostic",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            selection = row["selection_objective"]
            writer.writerow({
                "candidate_id": row["candidate_id"],
                "objective": selection["objective"],
                "matched_patient_loss": selection["matched_patient_loss"],
                "kmeans_balanced_alignment": (
                    1.0 - selection["kmeans_direction_loss"]
                ),
                "k2_support": selection["k2_support"],
                "ood_fraction": selection["ood_fraction"],
                "compound_fraction": selection["compound_fraction"],
                "heldout_r2_diagnostic": row[
                    "historical_unmatched_diagnostics"
                ]["model_prototype_r2_on_heldout"],
            })
    print(json.dumps({
        "status": payload["status"], "n_candidates": len(rows),
        "output": str(aggregate),
    }, indent=2))


if __name__ == "__main__":
    main()
