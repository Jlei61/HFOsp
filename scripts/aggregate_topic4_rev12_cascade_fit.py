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
    causal_direction_alignment,
    fixed_projection_matrix,
    matched_sample_dual_mode_objective,
    topology_network_reproducibility,
)
from src.topic4_shaft_aware_direction import assign_direction_modes  # noqa: E402


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
                                k2_support_weight: float = 0.0,
                                causal_direction_score: float | None = None,
                                causal_direction_weight: float = 0.0) -> dict:
    values = np.asarray([
        patient_loss, kmeans_balanced_alignment, ood_fraction, compound_fraction,
    ], float)
    if not np.all(np.isfinite(values)) or not 0.0 <= kmeans_balanced_alignment <= 1.0:
        raise ValueError("cascade objective inputs must be finite and bounded")
    if ((k2_support is not None and not 0.0 <= float(k2_support) <= 1.0)
            or float(k2_support_weight) < 0.0
            or (causal_direction_score is not None
                and not 0.0 <= float(causal_direction_score) <= 1.0)
            or float(causal_direction_weight) < 0.0):
        raise ValueError("auxiliary scores and weights must be bounded")
    kmeans_loss = 1.0 - float(kmeans_balanced_alignment)
    k2_loss = 0.0 if k2_support is None else 1.0 - float(k2_support)
    direction_loss = (
        0.0 if causal_direction_score is None
        else 1.0 - float(causal_direction_score)
    )
    return {
        "matched_patient_loss": float(patient_loss),
        "kmeans_direction_loss": kmeans_loss,
        "ood_fraction": float(ood_fraction),
        "compound_fraction": float(compound_fraction),
        "k2_support": None if k2_support is None else float(k2_support),
        "k2_support_loss": float(k2_loss),
        "causal_direction_score": (
            None if causal_direction_score is None
            else float(causal_direction_score)
        ),
        "causal_direction_loss": float(direction_loss),
        "objective": float(
            patient_loss + 0.5 * kmeans_loss
            + 0.25 * ood_fraction + 0.25 * compound_fraction
            + float(k2_support_weight) * k2_loss
            + float(causal_direction_weight) * direction_loss
        ),
        "weights": {
            "matched_patient_loss": 1.0,
            "kmeans_direction_loss": 0.5,
            "ood_fraction": 0.25,
            "compound_fraction": 0.25,
            "k2_support_loss": float(k2_support_weight),
            "causal_direction_loss": float(causal_direction_weight),
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
    elif source == "lineage_restricted_neuron_activity":
        if (observed.get("parity_status") != "EXACT_SHARED_PER_NEURON_KERNEL"
                or observed.get("spatial_sampler") != (
                    "exact_normalized_per_neuron_gaussian")):
            raise RuntimeError("worker did not use the exact neuron contact sampler")


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


def patient_direction_contract(patient_ranks: np.ndarray,
                               patient_labels: np.ndarray,
                               contact_xy: np.ndarray) -> dict:
    """Freeze a training-only spatial axis and each mode's expected sign."""
    ranks = np.asarray(patient_ranks, float)
    labels = np.asarray(patient_labels, int)
    coordinates = np.asarray(contact_xy, float)
    if ranks.ndim != 2 or labels.shape != (len(ranks),) \
            or coordinates.shape != (ranks.shape[1], 2):
        raise ValueError("patient direction inputs do not align")
    prototypes = np.asarray([
        np.nanmean(ranks[labels == mode], axis=0) for mode in (0, 1)
    ])
    contrast = prototypes[0] - prototypes[1]
    valid = np.isfinite(contrast) & np.all(np.isfinite(coordinates), axis=1)
    if int(np.sum(valid)) < 6:
        raise RuntimeError("patient direction axis has insufficient contacts")
    centered = coordinates[valid] - np.mean(coordinates[valid], axis=0)
    target = contrast[valid] - float(np.mean(contrast[valid]))
    beta, *_ = np.linalg.lstsq(centered, target, rcond=0.05)
    norm = float(np.linalg.norm(beta))
    if not np.isfinite(norm) or norm <= 1e-9:
        raise RuntimeError("patient direction axis is degenerate")
    axis = beta / norm
    along = (coordinates - np.mean(coordinates[valid], axis=0)) @ axis
    expected_signs = []
    mode_slopes = []
    for mode in (0, 1):
        selected = np.isfinite(prototypes[mode]) & np.isfinite(along)
        slope = float(np.dot(
            along[selected] - np.mean(along[selected]),
            prototypes[mode, selected] - np.mean(prototypes[mode, selected]),
        ))
        mode_slopes.append(slope)
        expected_signs.append(float(np.sign(slope)))
    if set(expected_signs) != {-1.0, 1.0}:
        raise RuntimeError("patient training modes are not oppositely directed")
    fitted = centered @ beta
    denominator = float(np.sum(target ** 2))
    r2 = float(1.0 - np.sum((target - fitted) ** 2) / denominator)
    return {
        "axis_unit_xy": axis.tolist(),
        "expected_mode_signs": expected_signs,
        "mode_rank_slopes": mode_slopes,
        "contrast_gradient_r2": r2,
        "n_contacts": int(np.sum(valid)),
        "source": "patient-training TA/TB rank-contrast gradient",
    }


def equal_network_causal_direction(onset_maps_by_network: list[np.ndarray],
                                   labels_by_network: list[np.ndarray], *,
                                   contract: dict, bin_mm: float,
                                   tail_fraction: float) -> dict:
    """Give every network equal weight in the causal direction endpoint."""
    if len(onset_maps_by_network) != len(labels_by_network):
        raise ValueError("network causal direction bundles do not align")
    rows = []
    for maps, labels in zip(onset_maps_by_network, labels_by_network):
        rows.append(causal_direction_alignment(
            maps, labels,
            axis_unit=np.asarray(contract["axis_unit_xy"], float),
            expected_mode_signs=np.asarray(contract["expected_mode_signs"], float),
            bin_mm=float(bin_mm), tail_fraction=float(tail_fraction),
        ))
    return {
        "score": float(np.mean([row["score"] for row in rows])) if rows else 0.0,
        "per_network": rows,
        "n_networks": int(len(rows)),
        "network_weighting": "equal network weight; weakest mode within network",
    }


def robust_sensitivity_envelope(rows: list[dict]) -> dict:
    """Combine one-axis event-definition sensitivities conservatively."""
    if not rows:
        raise ValueError("sensitivity envelope requires at least one variant")
    fields = {
        "matched_patient_loss": max,
        "kmeans_balanced_alignment": min,
        "k2_support": min,
        "ood_fraction": max,
        "compound_fraction": max,
    }
    output = {}
    for field, reducer in fields.items():
        values = [float(row[field]) for row in rows]
        if not np.all(np.isfinite(values)):
            raise ValueError(f"non-finite event sensitivity field: {field}")
        output[field] = float(reducer(values))
    output["n_variants"] = int(len(rows))
    output["componentwise_worst_case"] = True
    return output


def load_event_sensitivity_workers(npz_path: Path, target_names: np.ndarray,
                                   classifier_contract: dict,
                                   label_map: np.ndarray) -> list[dict]:
    """Load all frozen event-definition variants from one simulation."""
    payload = json.loads(npz_path.with_suffix(".json").read_text())
    with np.load(npz_path, allow_pickle=False) as loaded:
        required = {
            "lineage_sensitivity_values",
            "lineage_sensitivity_minimum_dominances",
            "lineage_sensitivity_primary",
            "lineage_sensitivity_event_counts",
            "lineage_sensitivity_onsets",
            "lineage_sensitivity_ranks",
            "lineage_sensitivity_returned",
            "lineage_sensitivity_fragment_partition",
            "contact_names",
        }
        if not required.issubset(loaded.files):
            raise RuntimeError("causal-root worker lacks event sensitivity arrays")
        names = np.asarray(loaded["contact_names"]).astype(str)
        order = []
        for name in np.asarray(target_names).astype(str):
            matches = np.flatnonzero(names == name)
            if len(matches) != 1:
                raise RuntimeError("event sensitivity contact order is ambiguous")
            order.append(int(matches[0]))
        values = np.asarray(loaded["lineage_sensitivity_values"], float)
        dominances = np.asarray(
            loaded["lineage_sensitivity_minimum_dominances"], float,
        )
        primary = np.asarray(loaded["lineage_sensitivity_primary"], bool)
        counts = np.asarray(loaded["lineage_sensitivity_event_counts"], int)
        onsets = np.asarray(loaded["lineage_sensitivity_onsets"], float)[..., order]
        ranks = np.asarray(loaded["lineage_sensitivity_ranks"], float)[..., order]
        returned = np.asarray(loaded["lineage_sensitivity_returned"], bool)
        partitions = np.asarray(
            loaded["lineage_sensitivity_fragment_partition"], int,
        )
    if not (len(values) == len(dominances) == len(primary) == len(counts)
            == len(onsets) == len(ranks) == len(returned) == len(partitions)):
        raise RuntimeError("event sensitivity variant arrays do not align")
    label_map = np.asarray(label_map, int)
    seed = int(payload["seed"])
    rows = []
    for index in range(len(values)):
        n_events = int(counts[index])
        current_onsets = onsets[index, :n_events]
        current_ranks = ranks[index, :n_events]
        current_returned = returned[index, :n_events]
        assigned = assign_direction_modes(
            current_onsets,
            groups=classifier_contract["groups"],
            embedding=classifier_contract["embedding"],
            classifier=classifier_contract["classifier"],
        )
        rows.append({
            "variant": {
                "sensitivity_value": float(values[index]),
                "minimum_dominance": float(dominances[index]),
                "primary": bool(primary[index]),
            },
            "worker": {
                "seed": seed,
                "ranks": current_ranks[current_returned],
                "onsets": current_onsets[current_returned],
                "labels": label_map[np.asarray(assigned["labels"], int)][
                    current_returned
                ],
                "ood": np.asarray(assigned["ood"], bool)[current_returned],
                "n_detected": n_events,
                "n_returned": int(np.sum(current_returned)),
            },
            "compound_fraction": float(np.mean(partitions[index] < 0)),
        })
    if int(np.sum(primary)) != 1:
        raise RuntimeError("event sensitivity arrays lack one primary variant")
    return rows


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
    first_npz = output_root / "workers" / (
        f"{manifest['candidates'][0]['candidate_id']}_seed_{seeds[0]}.npz"
    )
    if not first_npz.exists():
        raise RuntimeError("first completed worker is absent for direction contract")
    with np.load(first_npz, allow_pickle=False) as loaded:
        names = np.asarray(loaded["contact_names"]).astype(str)
        contact_xy = np.asarray(loaded["contact_xy_mm"], float)
    order = []
    for name in patient["contact_names"]:
        matches = np.flatnonzero(names == name)
        if len(matches) != 1:
            raise RuntimeError("worker contact geometry does not match patient target")
        order.append(int(matches[0]))
    direction_contract = patient_direction_contract(
        patient["train_ranks"], patient["train_labels"], contact_xy[order],
    )
    rows = []
    for candidate in manifest["candidates"]:
        workers, worker_payloads, source_maps, source_labels = [], [], [], []
        sensitivity_by_network = []
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
            sensitivity_by_network.append(load_event_sensitivity_workers(
                npz_path, patient["contact_names"], classifier,
                semantics["raw_to_patient"],
            ))
        if not workers:
            continue
        if len(workers) != len(seeds):
            raise RuntimeError(
                f"candidate {candidate['candidate_id']} is missing a frozen network"
            )
        alignment_source = objective_config.get(
            "kmeans_alignment_source", "pooled_equal_event_count",
        )
        variant_rows = []
        for variant_index, reference in enumerate(sensitivity_by_network[0]):
            reference_variant = reference["variant"]
            current = []
            compound_values = []
            for network_variants in sensitivity_by_network:
                row = network_variants[variant_index]
                if row["variant"] != reference_variant:
                    raise RuntimeError("event sensitivity ordering differs across networks")
                current.append(row["worker"])
                compound_values.append(row["compound_fraction"])
            current_scores = [
                matched_sample_dual_mode_objective(
                    worker["ranks"], worker["labels"],
                    patient["train_ranks"], patient["train_labels"],
                    patient["train_blocks"], patient["contact_names"],
                    projections=projections, calibration=calibration,
                    sample_size=int(objective_config["sample_size_per_side"]),
                    draws=int(objective_config["score_draws"]),
                    seed=int(objective_config["seed"]) + int(worker["seed"]),
                )
                for worker in current
            ]
            current_natural = equal_network_natural_kmeans(
                current, seed=int(objective_config["seed"]),
            )
            current_network_natural = per_network_natural_kmeans(
                current, seed=int(objective_config["seed"]),
                support_scale=float(objective_config.get("k2_support_scale", 0.5)),
            )
            if alignment_source == "pooled_equal_event_count":
                current_alignment = float(
                    current_natural["direction_balanced_alignment"]
                )
            elif alignment_source == "equal_network_mean":
                current_alignment = float(
                    current_network_natural[
                        "equal_network_mean_balanced_alignment"
                    ]
                )
            else:
                raise RuntimeError(
                    f"unknown KMeans alignment source: {alignment_source}"
                )
            variant_rows.append({
                **reference_variant,
                "matched_patient_loss": float(np.mean([
                    score["objective"] for score in current_scores
                ])),
                "kmeans_balanced_alignment": current_alignment,
                "k2_support": float(
                    current_network_natural["equal_network_mean_k2_support"]
                ),
                "ood_fraction": float(np.mean([
                    float(np.mean(worker["ood"])) if len(worker["ood"]) else 1.0
                    for worker in current
                ])),
                "compound_fraction": float(np.mean(compound_values)),
                "matched_network_scores": current_scores,
                "equal_network_natural_kmeans": current_natural,
                "per_network_natural_kmeans": current_network_natural,
            })
        primary_rows = [row for row in variant_rows if row["primary"]]
        if len(primary_rows) != 1:
            raise RuntimeError("candidate does not have one primary event variant")
        for worker, variants in zip(workers, sensitivity_by_network):
            primary_workers = [
                row["worker"] for row in variants if row["variant"]["primary"]
            ]
            if len(primary_workers) != 1 or not np.array_equal(
                    worker["ranks"], primary_workers[0]["ranks"], equal_nan=True):
                raise RuntimeError("primary event sensitivity arrays drifted from worker output")
        primary = primary_rows[0]
        matched_scores = primary["matched_network_scores"]
        natural = primary["equal_network_natural_kmeans"]
        network_natural = primary["per_network_natural_kmeans"]
        ood_fraction = primary["ood_fraction"]
        compound_fraction = primary["compound_fraction"]
        robust = robust_sensitivity_envelope(variant_rows)
        direction = equal_network_causal_direction(
            source_maps, source_labels, contract=direction_contract,
            bin_mm=float(config["source_topology"]["bin_mm"]),
            tail_fraction=float(objective_config.get(
                "causal_direction_tail_fraction", 0.2,
            )),
        )
        selection = cascade_selection_objective(
            patient_loss=robust["matched_patient_loss"],
            kmeans_balanced_alignment=robust["kmeans_balanced_alignment"],
            ood_fraction=robust["ood_fraction"],
            compound_fraction=robust["compound_fraction"],
            k2_support=robust["k2_support"],
            k2_support_weight=float(objective_config.get("k2_support_weight", 0.0)),
            causal_direction_score=float(direction["score"]),
            causal_direction_weight=float(objective_config.get(
                "causal_direction_weight", 0.0,
            )),
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
            "event_sensitivity_envelope": robust,
            "event_sensitivity_rows": variant_rows,
            "matched_network_scores": matched_scores,
            "equal_network_natural_kmeans": natural,
            "per_network_natural_kmeans": network_natural,
            "historical_unmatched_diagnostics": historical_diagnostics,
            "source_topology": topology,
            "causal_direction_alignment": direction,
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
        "patient_direction_contract": direction_contract,
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
            "event_sensitivity": (
                "componentwise worst case across frozen one-axis memory and "
                "root-dominance variants"
            ),
            "causal_direction": (
                "patient-training rank-contrast axis; equal network weight; "
                "weakest patient-labelled causal-root mode protected"
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
            "Exploratory causal-root field ranking. Patient held-out R2 and "
            "unconditioned topology figures are diagnostics. Training-only patient "
            "event distributions and the frozen patient direction axis select the fit."
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
            "compound_fraction", "causal_direction_score",
            "heldout_r2_diagnostic",
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
                "causal_direction_score": selection["causal_direction_score"],
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
