#!/usr/bin/env python3
"""Held-out patient event-variance audit for the frozen E1146 SNN substrate.

The audit deliberately separates prototype similarity from distribution-level
fit.  It reconstructs the frozen recording-block split, builds a fixed event
representation from recruitment and normalized contact ranks, and asks how
much held-out patient variance is explained by patient or model mode means.
"""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_topic4_data_driven_snn_cohort_targets import (  # noqa: E402
    _target_config,
)
from scripts.paper_figures.plot_fig4_spatial_edge_flow_validation import (  # noqa: E402
    _load_bundle,
)
from src.interictal_propagation import load_subject_propagation_events  # noqa: E402
from src.lagpat_rank_audit import mask_phantom_ranks  # noqa: E402
from src.topic4_data_driven_cohort import (  # noqa: E402
    build_crossfit_patient_target,
    canonical_pair_contract,
    subject_raw_root,
    subset_pair_contract,
)


DEFAULT_MODEL_CONFIG = ROOT / "config/topic4_rev11_nlc_frozen_substrate_confirmation.json"
DEFAULT_COHORT_CONFIG = ROOT / "config/topic4_data_driven_snn_cohort_v1.json"
DEFAULT_OUTPUT = ROOT / (
    "results/topic4_sef_hfo/data_driven_local_connectivity_rev11_nlc/"
    "interictal_variance_capture"
)
ARMS = (
    "node_baseline",
    "joint_04_ee_only",
    "joint_04_etoi_only",
    "joint_04_control",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        Path(temporary).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def event_features(normalized_ranks: np.ndarray) -> np.ndarray:
    """Return [recruitment mask, masked normalized rank] event features."""
    ranks = np.asarray(normalized_ranks, float)
    if ranks.ndim != 2:
        raise ValueError("normalized_ranks must have shape (event, contact)")
    mask = np.isfinite(ranks).astype(float)
    return np.concatenate([mask, np.nan_to_num(ranks, nan=0.0)], axis=1)


def shaft_balanced_weights(contact_names: np.ndarray) -> np.ndarray:
    """Weight recruitment/rank equally and ICL/SCL equally within each."""
    names = np.asarray(contact_names).astype(str)
    shafts = np.asarray([
        "".join(character for character in name if not character.isdigit())
        for name in names
    ])
    weights = np.zeros(2 * len(names), float)
    for offset in (0, len(names)):
        for shaft in ("ICL", "SCL"):
            indices = np.flatnonzero(shafts == shaft)
            if len(indices) == 0:
                raise ValueError(f"contact contract misses shaft {shaft}")
            weights[offset + indices] = 0.25 / len(indices)
    if not np.isclose(np.sum(weights), 1.0):
        raise RuntimeError("event feature weights do not sum to one")
    return weights


def weighted_r2(events: np.ndarray, labels: np.ndarray, prototypes: np.ndarray,
                global_mean: np.ndarray, weights: np.ndarray) -> dict:
    events = np.asarray(events, float)
    labels = np.asarray(labels, int)
    prototypes = np.asarray(prototypes, float)
    global_mean = np.asarray(global_mean, float)
    weights = np.asarray(weights, float)
    if np.any((labels < 0) | (labels >= len(prototypes))):
        raise ValueError("labels do not index the supplied prototypes")
    sst = float(np.sum((events - global_mean[None, :]) ** 2 * weights[None, :]))
    sse = float(np.sum((events - prototypes[labels]) ** 2 * weights[None, :]))
    return {
        "n_events": int(len(events)),
        "sst": sst,
        "sse": sse,
        "r2": None if sst <= 0.0 else float(1.0 - sse / sst),
    }


def _patient_contract(cohort_config: dict) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    subject_id = "epilepsiae_1146"
    inputs = cohort_config["inputs"]
    rank_path = ROOT / inputs["rank_displacement_root"] / f"{subject_id}.json"
    geometry_path = ROOT / inputs["gradient_geometry_root"] / f"{subject_id}.json"
    pair = canonical_pair_contract(json.loads(rank_path.read_text()))
    geometry = json.loads(geometry_path.read_text())
    field = geometry.get("interictal_field") or {}
    if field.get("contact_order"):
        pair = subset_pair_contract(pair, [str(value) for value in field["contact_order"]])
    raw_root = subject_raw_root(
        subject_id,
        epilepsiae_root=inputs["epilepsiae_raw_root"],
        yuquan_root=inputs["yuquan_raw_root"],
    )
    raw = load_subject_propagation_events(raw_root)
    target = build_crossfit_patient_target(
        raw, pair, config=_target_config(cohort_config),
    )
    lookup = {str(name): index for index, name in enumerate(raw["channel_names"])}
    rows = np.asarray([lookup[str(name)] for name in target["contact_order"]], int)
    normalized = mask_phantom_ranks(
        np.asarray(raw["ranks"], float)[rows],
        np.asarray(raw["bools"], bool)[rows], normalize=True,
    ).T
    blocks = np.asarray(raw["block_ids"])
    return target, raw, normalized, blocks


def _patient_arrays(target: dict, normalized: np.ndarray, blocks: np.ndarray) -> dict:
    train_index = np.asarray(target["train_event_indices"], int)
    heldout_index = np.asarray(target["heldout_event_indices"], int)
    train = event_features(normalized[train_index])
    heldout = event_features(normalized[heldout_index])
    train_labels = np.asarray(target["train_labels"], int)
    heldout_labels = np.asarray(target["heldout_labels"], int)
    prototypes = np.asarray([
        np.mean(train[train_labels == mode], axis=0) for mode in (0, 1)
    ])
    heldout_prototypes = np.asarray([
        np.mean(heldout[heldout_labels == mode], axis=0) for mode in (0, 1)
    ])
    return {
        "contact_names": np.asarray(target["contact_order"]).astype(str),
        "train": train,
        "train_labels": train_labels,
        "heldout": heldout,
        "heldout_labels": heldout_labels,
        "heldout_blocks": blocks[heldout_index],
        "global_mean": np.mean(train, axis=0),
        "patient_prototypes": prototypes,
        "patient_heldout_prototypes": heldout_prototypes,
    }


def _model_prototypes(bundle: dict, patient_names: np.ndarray) -> dict[int, np.ndarray]:
    model_names = np.asarray(bundle["static"]["contact_names"]).astype(str)
    if set(model_names) != set(patient_names):
        raise RuntimeError("model and patient contact sets differ")
    order = np.asarray([int(np.flatnonzero(model_names == name)[0]) for name in patient_names])
    record_seeds = np.asarray([record["seed"] for record in bundle["records"]], int)
    output = {}
    for seed in bundle["config"]["search"][bundle["network_seed_key"]]:
        selected = np.flatnonzero(bundle["clean"] & (record_seeds == int(seed)))
        labels = np.asarray(bundle["labels"])[selected]
        if any(not np.any(labels == mode) for mode in (0, 1)):
            continue
        ranks = np.asarray(bundle["ranks"])[selected][:, order]
        # Worker ranks are already normalized, but normalizing again is exact
        # for finite ordinal rows and protects the public feature contract.
        normalized = np.full_like(ranks, np.nan, dtype=float)
        for row_index, row in enumerate(ranks):
            finite = np.isfinite(row)
            if not np.any(finite):
                continue
            values = row[finite]
            span = float(np.max(values) - np.min(values))
            normalized[row_index, finite] = (
                (values - np.min(values)) / span if span > 0.0 else 0.0
            )
        features = event_features(normalized)
        output[int(seed)] = np.asarray([
            np.mean(features[labels == mode], axis=0) for mode in (0, 1)
        ])
    return output


def _block_indices(blocks: np.ndarray) -> dict[int, np.ndarray]:
    return {
        int(block): np.flatnonzero(blocks == block)
        for block in np.unique(blocks)
    }


def align_model_modes(model_prototypes: dict[int, np.ndarray],
                      patient_train_prototypes: np.ndarray,
                      weights: np.ndarray) -> tuple[dict[int, np.ndarray], dict]:
    """Map raw model labels to patient TA/TB using training prototypes only."""
    mean_raw = np.mean(list(model_prototypes.values()), axis=0)
    patient = np.asarray(patient_train_prototypes, float)
    weights = np.asarray(weights, float)
    costs = {}
    for order in ((0, 1), (1, 0)):
        costs[order] = float(np.sum(
            (mean_raw[np.asarray(order)] - patient) ** 2 * weights[None, :]
        ))
    selected = min(costs, key=lambda order: (costs[order], order))
    aligned = {
        int(seed): np.asarray(prototypes)[np.asarray(selected)]
        for seed, prototypes in model_prototypes.items()
    }
    return aligned, {
        "raw_model_order_for_patient_TA_TB": list(selected),
        "identity_cost": costs[(0, 1)],
        "swapped_cost": costs[(1, 0)],
        "selection_data": "patient training prototypes only",
    }


def contrast_metrics(model_prototypes: np.ndarray,
                     patient_train_prototypes: np.ndarray,
                     patient_heldout_prototypes: np.ndarray,
                     weights: np.ndarray) -> dict:
    """Evaluate the patient TA-TB contrast, with scale fit on train only."""
    model_delta = np.asarray(model_prototypes[0] - model_prototypes[1], float)
    train_delta = np.asarray(
        patient_train_prototypes[0] - patient_train_prototypes[1], float,
    )
    heldout_delta = np.asarray(
        patient_heldout_prototypes[0] - patient_heldout_prototypes[1], float,
    )
    weights = np.asarray(weights, float)
    model_energy = float(np.sum(weights * model_delta ** 2))
    scale = (
        0.0 if model_energy <= 0.0 else max(
            0.0, float(np.sum(weights * train_delta * model_delta) / model_energy),
        )
    )
    denominator = float(np.sum(weights * heldout_delta ** 2))
    raw_sse = float(np.sum(weights * (heldout_delta - model_delta) ** 2))
    scaled_sse = float(np.sum(weights * (heldout_delta - scale * model_delta) ** 2))
    cosine_denominator = float(np.sqrt(
        np.sum(weights * heldout_delta ** 2)
        * np.sum(weights * model_delta ** 2)
    ))
    return {
        "train_fitted_nonnegative_scale": scale,
        "heldout_raw_contrast_r2": (
            None if denominator <= 0.0 else float(1.0 - raw_sse / denominator)
        ),
        "heldout_scale_calibrated_contrast_r2": (
            None if denominator <= 0.0 else float(1.0 - scaled_sse / denominator)
        ),
        "heldout_weighted_cosine": (
            None if cosine_denominator <= 0.0
            else float(np.sum(weights * heldout_delta * model_delta) / cosine_denominator)
        ),
    }


def _bootstrap(patient: dict, model_prototypes: dict[int, np.ndarray],
               weights: np.ndarray, *, draws: int, seed: int) -> dict:
    rng = np.random.default_rng(int(seed))
    block_rows = _block_indices(patient["heldout_blocks"])
    block_ids = np.asarray(sorted(block_rows), int)
    network_ids = np.asarray(sorted(model_prototypes), int)
    if len(network_ids) == 0:
        raise RuntimeError("no network contains both frozen model modes")
    r2_values, patient_r2_values, captured = [], [], []
    raw_contrast, scaled_contrast = [], []
    for _ in range(int(draws)):
        sampled_blocks = rng.choice(block_ids, size=len(block_ids), replace=True)
        event_index = np.concatenate([block_rows[int(block)] for block in sampled_blocks])
        sampled_networks = rng.choice(network_ids, size=len(network_ids), replace=True)
        prototype = np.mean([model_prototypes[int(item)] for item in sampled_networks], axis=0)
        metric = weighted_r2(
            patient["heldout"][event_index], patient["heldout_labels"][event_index],
            prototype, patient["global_mean"], weights,
        )
        patient_metric = weighted_r2(
            patient["heldout"][event_index], patient["heldout_labels"][event_index],
            patient["patient_prototypes"], patient["global_mean"], weights,
        )
        r2 = float(metric["r2"])
        denominator = float(patient_metric["r2"])
        r2_values.append(r2)
        patient_r2_values.append(denominator)
        captured.append(r2 / denominator if denominator != 0.0 else np.nan)
        heldout_prototypes = []
        for mode in (0, 1):
            mode_rows = patient["heldout_labels"][event_index] == mode
            heldout_prototypes.append(np.mean(
                patient["heldout"][event_index][mode_rows], axis=0,
            ))
        contrast = contrast_metrics(
            prototype, patient["patient_prototypes"],
            np.asarray(heldout_prototypes), weights,
        )
        raw_contrast.append(contrast["heldout_raw_contrast_r2"])
        scaled_contrast.append(
            contrast["heldout_scale_calibrated_contrast_r2"]
        )
    def summary(values):
        values = np.asarray(values, float)
        values = values[np.isfinite(values)]
        return {
            "median": float(np.median(values)),
            "q05": float(np.quantile(values, 0.05)),
            "q95": float(np.quantile(values, 0.95)),
        }
    return {
        "draws": int(draws),
        "recording_blocks_resampled": int(len(block_ids)),
        "network_seeds_resampled": int(len(network_ids)),
        "patient_train_k2_r2_on_resampled_heldout": summary(patient_r2_values),
        "r2": summary(r2_values),
        "fraction_of_patient_k2_r2": summary(captured),
        "between_mode_contrast": {
            "heldout_raw_contrast_r2": summary(raw_contrast),
            "heldout_scale_calibrated_contrast_r2": summary(scaled_contrast),
        },
    }


def _component_metrics(patient: dict, prototypes: np.ndarray,
                       patient_prototypes: np.ndarray, weights: np.ndarray) -> dict:
    n_contacts = len(patient["contact_names"])
    shafts = np.asarray([
        "".join(character for character in name if not character.isdigit())
        for name in patient["contact_names"]
    ])
    components = {
        "all": np.arange(2 * n_contacts),
        "recruitment": np.arange(n_contacts),
        "rank": np.arange(n_contacts, 2 * n_contacts),
        "ICL": np.concatenate([
            np.flatnonzero(shafts == "ICL"),
            np.flatnonzero(shafts == "ICL") + n_contacts,
        ]),
        "SCL": np.concatenate([
            np.flatnonzero(shafts == "SCL"),
            np.flatnonzero(shafts == "SCL") + n_contacts,
        ]),
    }
    output = {}
    for name, index in components.items():
        component_weights = weights[index]
        component_weights = component_weights / np.sum(component_weights)
        model = weighted_r2(
            patient["heldout"][:, index], patient["heldout_labels"],
            prototypes[:, index], patient["global_mean"][index], component_weights,
        )
        reference = weighted_r2(
            patient["heldout"][:, index], patient["heldout_labels"],
            patient_prototypes[:, index], patient["global_mean"][index],
            component_weights,
        )
        output[name] = {
            "patient_train_k2_r2_on_heldout": reference["r2"],
            "model_r2_on_patient_heldout": model["r2"],
            "fraction_of_patient_k2_r2": (
                None if reference["r2"] == 0.0
                else float(model["r2"] / reference["r2"])
            ),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--cohort-config", type=Path, default=DEFAULT_COHORT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260821)
    args = parser.parse_args()

    model_config_path = args.model_config.resolve()
    cohort_config_path = args.cohort_config.resolve()
    model_config = json.loads(model_config_path.read_text())
    cohort_config = json.loads(cohort_config_path.read_text())
    target, _, normalized, blocks = _patient_contract(cohort_config)
    patient = _patient_arrays(target, normalized, blocks)
    weights = shaft_balanced_weights(patient["contact_names"])
    patient_metric = weighted_r2(
        patient["heldout"], patient["heldout_labels"],
        patient["patient_prototypes"], patient["global_mean"], weights,
    )
    patient_r2 = float(patient_metric["r2"])

    output_root = ROOT / model_config["output_root"]
    arms = {}
    csv_rows = []
    for arm in ARMS:
        bundle = _load_bundle(
            model_config_path, output_root, arm,
            allow_exploratory_candidate=True,
        )
        prototypes_by_network = _model_prototypes(bundle, patient["contact_names"])
        prototypes_by_network, mode_alignment = align_model_modes(
            prototypes_by_network, patient["patient_prototypes"], weights,
        )
        mean_prototypes = np.mean(list(prototypes_by_network.values()), axis=0)
        components = _component_metrics(
            patient, mean_prototypes, patient["patient_prototypes"], weights,
        )
        bootstrap = _bootstrap(
            patient, prototypes_by_network, weights,
            draws=args.bootstrap_draws, seed=args.bootstrap_seed,
        )
        arms[arm] = {
            "n_networks": int(len(prototypes_by_network)),
            "network_seeds": sorted(prototypes_by_network),
            "mode_alignment": mode_alignment,
            "components": components,
            "between_mode_contrast": contrast_metrics(
                mean_prototypes, patient["patient_prototypes"],
                patient["patient_heldout_prototypes"], weights,
            ),
            "hierarchical_bootstrap": bootstrap,
        }
        csv_rows.append({
            "arm": arm,
            "n_networks": len(prototypes_by_network),
            "patient_k2_r2": components["all"]["patient_train_k2_r2_on_heldout"],
            "model_r2": components["all"]["model_r2_on_patient_heldout"],
            "fraction_patient_k2_r2": components["all"]["fraction_of_patient_k2_r2"],
            "heldout_raw_contrast_r2": arms[arm]["between_mode_contrast"][
                "heldout_raw_contrast_r2"
            ],
            "heldout_scale_calibrated_contrast_r2": arms[arm][
                "between_mode_contrast"
            ]["heldout_scale_calibrated_contrast_r2"],
            "bootstrap_r2_q05": bootstrap["r2"]["q05"],
            "bootstrap_r2_q95": bootstrap["r2"]["q95"],
        })

    payload = {
        "schema_id": "topic4_interictal_variance_capture_v1",
        "status": "DEVELOPMENT_ONLY_HELDOUT_EVENT_VARIANCE_AUDIT",
        "scientific_question": (
            "How much held-out E1146 event-level recruitment and rank variance is "
            "captured by frozen model mode prototypes?"
        ),
        "feature_contract": {
            "representation": "[recruitment mask, masked per-event normalized contact rank]",
            "feature_range": "both blocks lie in [0,1]",
            "weights": (
                "recruitment and rank each receive 1/2 total weight; within each, "
                "ICL and SCL each receive 1/2"
            ),
            "reference_mean": "patient training events pooled over frozen TA/TB labels",
            "patient_reference": "patient training TA/TB means evaluated on held-out recording blocks",
            "model_reference": "equal-network mean model TA/TB prototypes evaluated on the same held-out patient events",
            "negative_r2_allowed": True,
            "between_mode_contrast": (
                "TA minus TB model contrast; one nonnegative amplitude is fitted to "
                "patient training contrast and evaluated on held-out contrast"
            ),
        },
        "patient": {
            "subject_id": "epilepsiae_1146",
            "n_train_events": int(len(patient["train"])),
            "n_heldout_events": int(len(patient["heldout"])),
            "n_heldout_blocks": int(len(np.unique(patient["heldout_blocks"]))),
            "patient_train_k2_r2_on_heldout": patient_r2,
        },
        "arms": arms,
        "interpretation_boundary": (
            "This is event-cloud variance in a frozen contact-level representation, not "
            "electrophysiological voltage variance, neuron-level mechanism evidence, or a "
            "license to reinterpret Spearman rho squared as explained variance."
        ),
        "inputs": {
            "model_config": str(model_config_path),
            "model_config_sha256": _sha256(model_config_path),
            "cohort_config": str(cohort_config_path),
            "cohort_config_sha256": _sha256(cohort_config_path),
        },
    }
    output = args.output.resolve()
    _atomic_json(output / "variance_capture.json", payload)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "variance_capture.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
