#!/usr/bin/env python3
"""Aggregate the rev21 coarse Z/M screen with network cells as units."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_rev20_dual_core_endpoint import (  # noqa: E402
    embedding_from_training_arrays, matched_patient_floor,
    score_complete_distribution, score_validation_endpoints,
)
from src.topic4_shaft_aware import contract_groups  # noqa: E402


MATCHED_RETENTION_DRAWS = 128
MATCHED_RETENTION_SEED = 21051


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(artifact_root: Path, path: str) -> Path:
    local = ROOT / path
    return local if local.exists() else artifact_root / path


def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as handle:
        return {key: np.asarray(handle[key]).copy() for key in handle.files}


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(json.dumps(
            payload, indent=2, sort_keys=True, allow_nan=False,
        ) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _finite(values):
    return np.asarray([
        float(value) for value in values
        if value is not None and np.isfinite(float(value))
    ], float)


def qualification_shortfall(state: dict) -> float:
    """Normalized weakest-clause deficit for timescale seeding only.

    A zero is a formal pass. A finite positive value does not relax the state
    definition; it only prevents the predeclared timescale experiment from
    being blocked when amplitude and timescale jointly determine one clause.
    """
    if state.get("status") == "MODEL_ICTAL_ELIGIBLE_REV21":
        return 0.0
    if not state.get("clauses", {}).get("operational_detector_reached"):
        return float("inf")
    thresholds = state.get("thresholds", {})
    recruitment = state.get("recruitment", {})
    rate = state.get("population_rate", {})
    frequency = state.get("contact_frequency", {})
    values = (
        (recruitment.get("joint_duty"), thresholds.get("duty")),
        (rate.get("ratio_early_over_base"),
         thresholds.get("population_rate_ratio_min")),
        (frequency.get("primary_shift_hz"),
         thresholds.get("contact_centroid_shift_min_hz")),
        (frequency.get("primary_ratio"),
         thresholds.get("contact_centroid_ratio_min")),
    )
    deficits = []
    for observed, target in values:
        if observed is None or target is None or not np.isfinite(float(observed)):
            return float("inf")
        scale = max(abs(float(target)), 1e-12)
        deficits.append(max(0.0, (float(target) - float(observed)) / scale))
    if not state.get("clauses", {}).get("transition_after_minimum_dwell"):
        deficits.append(1.0)
    if not state.get("clauses", {}).get("numerically_safe"):
        return float("inf")
    return float(max(deficits))


def model_ictal_or_control(worker: dict) -> dict:
    if "model_ictal_rev21" in worker:
        return worker["model_ictal_rev21"]
    if worker.get("candidate_id") in {"rev21_zm_off", "rev21_confirm_zm_off"}:
        return {
            "schema_id": "model_ictal_control_not_scored_v1",
            "status": "MODEL_ICTAL_CONTROL_NOT_SCORED",
            "eligible": False,
            "clauses": {},
            "boundary": (
                "Z/M-off is a paired interictal reference and exact historical "
                "parity path; it is not ranked as an active transition candidate"
            ),
        }
    raise RuntimeError(
        f"active candidate {worker.get('candidate_id')} lacks model-ictal evidence"
    )


def reference_support(endpoint_matrices: dict) -> dict:
    mapping = {
        "complete_distribution": "training_complete_distribution",
        "two_template_alignment": "two_template_alignment",
        "ood": "ood_all_returned",
    }
    output = {}
    for target, source in mapping.items():
        values = _finite(np.asarray(endpoint_matrices[source], object).ravel())
        if len(values) < 4:
            raise RuntimeError(f"off reference {source} has fewer than four values")
        output[target] = {
            "n": int(len(values)),
            "q05": float(np.quantile(values, 0.05)),
            "q50": float(np.quantile(values, 0.50)),
            "q95": float(np.quantile(values, 0.95)),
        }
    return output


def _quantiles(values) -> dict:
    finite = _finite(values)
    return {
        "n": int(len(finite)),
        "q05": float(np.quantile(finite, 0.05)) if len(finite) else None,
        "q50": float(np.quantile(finite, 0.50)) if len(finite) else None,
        "q95": float(np.quantile(finite, 0.95)) if len(finite) else None,
    }


def _returned_tables(arrays: dict) -> tuple[np.ndarray, np.ndarray]:
    returned = np.asarray(arrays["event_returned"], bool)
    return (
        np.asarray(arrays["onsets"], float)[returned],
        np.asarray(arrays["ranks"], float)[returned],
    )


def matched_interictal_retention(
        candidate_by_cell: dict, off_by_cell: dict, *, contract: dict,
        training_arrays: dict, classifier: dict, kmeans_seed: int,
        draws: int = MATCHED_RETENTION_DRAWS,
        seed: int = MATCHED_RETENTION_SEED) -> dict:
    """Calibrate short pre-transition samples without an event-count gate."""
    if set(candidate_by_cell) != set(off_by_cell):
        raise RuntimeError("candidate and Z/M-off seed cells do not match")
    keys = sorted(candidate_by_cell)
    candidate_onsets = []
    candidate_ranks = []
    counts = {}
    off_tables = {}
    for key in keys:
        onsets, ranks = _returned_tables(candidate_by_cell[key])
        off_onsets, off_ranks = _returned_tables(off_by_cell[key])
        if len(onsets) > len(off_onsets):
            raise RuntimeError("candidate has more events than its matched off cell")
        counts[f"{key[0]}:{key[1]}"] = int(len(onsets))
        candidate_onsets.append(onsets)
        candidate_ranks.append(ranks)
        off_tables[key] = (off_onsets, off_ranks)
    pooled_onsets = np.concatenate(candidate_onsets)
    pooled_ranks = np.concatenate(candidate_ranks)
    returned = np.ones(len(pooled_onsets), bool)
    observed_selection = score_complete_distribution(
        pooled_onsets, returned, contract=contract,
        training_arrays=training_arrays,
    )
    observed_validation = score_validation_endpoints(
        pooled_onsets, pooled_ranks, returned, contract=contract,
        training_arrays=training_arrays, classifier=classifier,
        kmeans_seed=int(kmeans_seed),
    )
    groups = contract_groups(contract)
    embedding = embedding_from_training_arrays(training_arrays)
    patient_floor = matched_patient_floor(
        training_arrays["patient_train_onsets"], len(pooled_onsets),
        groups=groups, embedding=embedding, draws=int(draws), seed=int(seed),
    )

    rng = np.random.default_rng(int(seed))
    null_complete = []
    null_ood = []
    null_alignment = []
    null_mode1_fraction = []
    null_two_direction = []
    for _ in range(int(draws)):
        sampled_onsets = []
        sampled_ranks = []
        for key in keys:
            n = counts[f"{key[0]}:{key[1]}"]
            off_onsets, off_ranks = off_tables[key]
            if n == 0:
                continue
            chosen = rng.choice(len(off_onsets), n, replace=False)
            sampled_onsets.append(off_onsets[chosen])
            sampled_ranks.append(off_ranks[chosen])
        if not sampled_onsets:
            continue
        onsets = np.concatenate(sampled_onsets)
        ranks = np.concatenate(sampled_ranks)
        selection = score_complete_distribution(
            onsets, np.ones(len(onsets), bool), contract=contract,
            training_arrays=training_arrays,
        )
        validation = score_validation_endpoints(
            onsets, ranks, np.ones(len(onsets), bool), contract=contract,
            training_arrays=training_arrays, classifier=classifier,
            kmeans_seed=int(kmeans_seed),
        )
        null_complete.append(selection[
            "complete_distribution_distance_training"
        ])
        null_ood.append(validation["ood_all_returned"])
        null_two_direction.append(bool(
            validation["frozen_two_directions_present"]
        ))
        direction_counts = validation["frozen_direction_counts"]
        if sum(direction_counts):
            null_mode1_fraction.append(direction_counts[1] / sum(direction_counts))
        alignment = validation.get("direction_balanced_alignment")
        if alignment is not None:
            null_alignment.append(alignment)

    observed_counts = observed_validation["frozen_direction_counts"]
    observed_mode1 = (
        observed_counts[1] / sum(observed_counts) if sum(observed_counts) else None
    )
    support = {
        "complete_distribution_patient_floor": patient_floor,
        "complete_distribution_matched_off": _quantiles(null_complete),
        "ood_matched_off": _quantiles(null_ood),
        "two_template_alignment_matched_off": _quantiles(null_alignment),
        "frozen_mode1_fraction_matched_off": _quantiles(null_mode1_fraction),
        "matched_off_two_direction_probability": (
            float(np.mean(null_two_direction)) if null_two_direction else None
        ),
    }
    complete = observed_selection[
        "complete_distribution_distance_training"
    ]
    alignment = observed_validation.get("direction_balanced_alignment")
    clauses = {
        "minimum_eight_pooled_events_for_natural_kmeans": bool(
            len(pooled_onsets) >= 8
        ),
        "complete_distribution_retained_vs_matched_off": bool(
            complete is not None
            and support["complete_distribution_matched_off"]["q95"] is not None
            and complete
            <= support["complete_distribution_matched_off"]["q95"]
        ),
        "ood_within_matched_off_support": bool(
            support["ood_matched_off"]["q95"] is not None
            and observed_validation["ood_all_returned"]
            <= support["ood_matched_off"]["q95"]
        ),
        "natural_two_clusters_present": bool(
            observed_validation.get("two_clusters_present", False)
        ),
        "both_frozen_patient_directions_present": bool(
            observed_validation["frozen_two_directions_present"]
        ),
        "kmeans_alignment_within_matched_off_support": bool(
            alignment is not None
            and support["two_template_alignment_matched_off"]["q05"] is not None
            and alignment
            >= support["two_template_alignment_matched_off"]["q05"]
        ),
    }
    deterioration = {}
    for name, value, limits, reverse in (
        ("complete_distribution", complete,
         support["complete_distribution_matched_off"], False),
        ("ood", observed_validation["ood_all_returned"],
         support["ood_matched_off"], False),
        ("two_template_alignment", alignment,
         support["two_template_alignment_matched_off"], True),
    ):
        if value is None or limits.get("q05") is None or limits.get("q95") is None:
            deterioration[name] = None
            continue
        width = max(float(limits["q95"]) - float(limits["q05"]), 1e-12)
        raw = (
            float(limits["q50"]) - float(value) if reverse
            else float(value) - float(limits["q50"])
        )
        deterioration[name] = float(raw / width)
    finite_deterioration = [
        value for value in deterioration.values() if value is not None
    ]
    return {
        "schema_id": "topic4_rev21_event_count_matched_retention_v2",
        "draws": int(draws), "seed": int(seed),
        "event_counts_by_cell": counts,
        "pooled_event_count": int(len(pooled_onsets)),
        "observed": {
            "complete_distribution": complete,
            "ood": observed_validation["ood_all_returned"],
            "natural_kmeans_status": observed_validation["natural_kmeans_status"],
            "cluster_counts": observed_validation.get("cluster_counts"),
            "two_template_alignment": alignment,
            "frozen_direction_counts": observed_counts,
            "frozen_mode1_fraction": observed_mode1,
            "absolute_patient_distribution_within_floor": bool(
                complete is not None and patient_floor.get("q95") is not None
                and complete <= patient_floor["q95"]
            ),
        },
        "support": support, "clauses": clauses,
        "retained": bool(all(clauses.values())),
        "standardized_deterioration": deterioration,
        "worst_standardized_deterioration": (
            max(finite_deterioration) if finite_deterioration else None
        ),
        "boundary": (
            "pooled development-screen evidence with per-cell event-count-matched "
            "off resampling; not same-network two-mode confirmation"
        ),
    }


def replace_retention_with_matched(summary: dict, matched: dict) -> None:
    """Preserve the old audit while making matched retention authoritative."""
    summary["legacy_full_length_reference_retention"] = {
        "retained": summary["interictal_substrate_retained"],
        "support": summary["off_reference_support"],
        "role": "superseded audit; confounded by event-count mismatch",
    }
    summary["event_count_matched_retention"] = matched
    summary["interictal_substrate_retained"] = matched["retained"]
    summary["standardized_deterioration"] = matched[
        "standardized_deterioration"
    ]
    summary["worst_standardized_deterioration"] = matched[
        "worst_standardized_deterioration"
    ]


def summarize_candidate(candidate_id: str, cells: list[dict], off_lookup: dict,
                        support: dict, pooled: dict, level) -> dict:
    complete = _finite([
        row["selection"].get("complete_distribution_distance_training")
        for row in cells
    ])
    alignment = _finite([
        row["validation"].get("direction_balanced_alignment")
        for row in cells
    ])
    ood = _finite([
        row["validation"].get("ood_all_returned") for row in cells
    ])
    eligible = np.asarray([
        row["model_ictal"].get("status") == "MODEL_ICTAL_ELIGIBLE_REV21"
        for row in cells
    ], bool)
    onsets = _finite([row.get("operational_onset_ms") for row in cells])
    metrics = {
        "complete_distribution": (
            float(np.median(complete)) if len(complete) else None),
        "two_template_alignment": (
            float(np.median(alignment)) if len(alignment) else None),
        "ood": float(np.mean(ood)) if len(ood) else None,
    }
    estimable_cells = {
        "complete_distribution": int(len(complete)),
        "two_template_alignment": int(len(alignment)),
        "ood": int(len(ood)),
    }
    paired = {name: [] for name in metrics}
    for row in cells:
        key = (row["topology_seed"], row["dynamics_seed"])
        off = off_lookup.get(key)
        if off is None:
            continue
        candidates = {
            "complete_distribution": (
                row["selection"].get("complete_distribution_distance_training"),
                off["selection"].get("complete_distribution_distance_training")),
            "two_template_alignment": (
                row["validation"].get("direction_balanced_alignment"),
                off["validation"].get("direction_balanced_alignment")),
            "ood": (row["validation"].get("ood_all_returned"),
                    off["validation"].get("ood_all_returned")),
        }
        for name, (value, reference) in candidates.items():
            if value is not None and reference is not None:
                paired[name].append(float(value) - float(reference))
    paired_summary = {
        name: {
            "n": len(values),
            "median_candidate_minus_off": (
                float(np.median(values)) if values else None),
            "values": values,
        }
        for name, values in paired.items()
    }
    inside = {
        "complete_distribution": (
            metrics["complete_distribution"] is not None
            and metrics["complete_distribution"]
            <= support["complete_distribution"]["q95"]),
        "two_template_alignment": (
            metrics["two_template_alignment"] is not None
            and metrics["two_template_alignment"]
            >= support["two_template_alignment"]["q05"]),
        "ood": (
            metrics["ood"] is not None
            and metrics["ood"] <= support["ood"]["q95"]),
        "pooled_two_clusters_present": bool(
            pooled["validation"].get("two_clusters_present", False)),
    }
    retained = bool(all(inside.values()))
    deterioration = {}
    for name, value in metrics.items():
        if value is None:
            deterioration[name] = None
            continue
        width = max(support[name]["q95"] - support[name]["q05"], 1e-12)
        if name == "two_template_alignment":
            raw = support[name]["q50"] - value
        else:
            raw = value - support[name]["q50"]
        deterioration[name] = float(raw / width)
    finite_deterioration = [value for value in deterioration.values()
                            if value is not None]
    if isinstance(level, dict):
        distance = math.hypot(
            math.log(float(level["I_th_EI_scale"])),
            math.log(float(level["integrated_M_scale"])),
        )
    else:
        distance = None
    counts = [int(row["selection"]["n_returned_families"]) for row in cells]
    shortfalls = _finite([
        qualification_shortfall(row["model_ictal"]) for row in cells
    ])
    operational = np.asarray([
        bool(row["model_ictal"].get("clauses", {}).get(
            "operational_detector_reached")) for row in cells
    ], bool)
    return {
        "candidate_id": candidate_id,
        "level": level,
        "cell_count": len(cells),
        "model_ictal_eligible_cells": int(np.sum(eligible)),
        "model_ictal_eligible_fraction": float(np.mean(eligible)),
        "model_ictal_qualification_shortfall": {
            "n_finite": int(len(shortfalls)),
            "median": (float(np.median(shortfalls)) if len(shortfalls) else None),
            "q75": (float(np.quantile(shortfalls, 0.75))
                    if len(shortfalls) else None),
            "operational_transition_fraction": float(np.mean(operational)),
            "role": "timescale-seed ranking only; not an eligibility relaxation",
        },
        "operational_onset_ms": {
            "n": int(len(onsets)),
            "median": float(np.median(onsets)) if len(onsets) else None,
            "range": ([float(np.min(onsets)), float(np.max(onsets))]
                      if len(onsets) else None),
        },
        "equal_network_metrics": metrics,
        "estimable_cells": estimable_cells,
        "paired_candidate_minus_off": paired_summary,
        "off_reference_support": inside,
        "interictal_substrate_retained": retained,
        "standardized_deterioration": deterioration,
        "worst_standardized_deterioration": (
            float(max(finite_deterioration))
            if len(finite_deterioration) == 3 else None),
        "pooled_diagnostic": pooled,
        "returned_family_counts_by_cell": counts,
        "largest_cell_event_share": (
            float(max(counts) / sum(counts)) if sum(counts) else None),
        "log_distance_from_reference": distance,
    }


def _neighbor_scores(summaries: list[dict]) -> None:
    active = [row for row in summaries if isinstance(row["level"], dict)]
    s_i_levels = sorted({float(row["level"]["I_th_EI_scale"])
                         for row in active})
    s_m_levels = sorted({float(row["level"]["integrated_M_scale"])
                         for row in active})
    for row in active:
        level = row["level"]
        i_index = s_i_levels.index(float(level["I_th_EI_scale"]))
        m_index = s_m_levels.index(float(level["integrated_M_scale"]))
        nearest = [other for other in active if other is not row and (
            abs(s_i_levels.index(float(other["level"]["I_th_EI_scale"]))
                - i_index)
            + abs(s_m_levels.index(float(other["level"]["integrated_M_scale"]))
                  - m_index)
        ) == 1]
        row["neighbor_eligible_fraction"] = (
            float(np.mean([other["model_ictal_eligible_fraction"]
                           for other in nearest])) if nearest else 0.0
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path,
                        default=Path("/home/honglab/leijiaxin/HFOsp"))
    args = parser.parse_args()
    artifact_root = args.artifact_root.resolve()
    config = json.loads(args.config.read_text())
    output_root = artifact_root / config["output_root"]
    controller = json.loads((output_root / "coarse/status/controller.json").read_text())
    if controller.get("status") != "COMPLETE":
        raise RuntimeError("coarse controller is not complete")
    seed_audit = json.loads((
        output_root / "seed_audit/seed_factorization_audit.json"
    ).read_text())
    support = reference_support(seed_audit["endpoint_matrices"])

    training = _load_npz(_resolve(
        artifact_root, config["inputs"]["patient_training_target"]["path"],
    ))
    support_config = json.loads(_resolve(
        artifact_root, config["inputs"]["patient_support_config"]["path"],
    ).read_text())
    contract_path = _resolve(
        artifact_root, support_config["inputs"]["contact_contract"]["path"],
    )
    classifier_path = _resolve(
        artifact_root,
        support_config["inputs"]["old_ab_train_only_classifier"]["path"],
    )
    if (_sha256(contract_path) != support_config["inputs"]["contact_contract"]["sha256"]
            or _sha256(classifier_path) != support_config["inputs"][
                "old_ab_train_only_classifier"]["sha256"]):
        raise RuntimeError("training contact/classifier contract changed")
    contract = json.loads(contract_path.read_text())
    classifier = json.loads(classifier_path.read_text())["direction_classifier"]
    manifest = json.loads((artifact_root / config["candidate_manifest"]).read_text())
    levels = {row["candidate_id"]: row["level"] for row in manifest["candidates"]}

    cells = []
    arrays_by_candidate = {}
    arrays_by_candidate_cell = {}
    for path in sorted((output_root / "coarse/workers").glob("*.json")):
        worker = json.loads(path.read_text())
        arrays = _load_npz(Path(worker["arrays"]["path"]))
        selection = score_complete_distribution(
            arrays["onsets"], arrays["event_returned"],
            contract=contract, training_arrays=training,
        )
        validation = score_validation_endpoints(
            arrays["onsets"], arrays["ranks"], arrays["event_returned"],
            contract=contract, training_arrays=training,
            classifier=classifier,
            kmeans_seed=int(config["validation"]["natural_kmeans_seed"]),
        )
        row = {
            "candidate_id": worker["candidate_id"],
            "topology_seed": worker["topology_seed"],
            "dynamics_seed": worker["dynamics_seed"],
            "operational_onset_ms": worker["simulation"]["runaway_early_stop_ms"],
            "model_ictal": model_ictal_or_control(worker),
            "selection": selection, "validation": validation,
            "worker_json": str(path),
        }
        cells.append(row)
        arrays_by_candidate.setdefault(worker["candidate_id"], []).append(arrays)
        arrays_by_candidate_cell.setdefault(worker["candidate_id"], {})[
            (int(worker["topology_seed"]), int(worker["dynamics_seed"]))
        ] = arrays
    grouped = {}
    for row in cells:
        grouped.setdefault(row["candidate_id"], []).append(row)
    off_lookup = {
        (row["topology_seed"], row["dynamics_seed"]): row
        for row in grouped["rev21_zm_off"]
    }
    summaries = []
    for candidate_id, rows in grouped.items():
        group_arrays = arrays_by_candidate[candidate_id]
        pooled_onsets = np.concatenate([array["onsets"] for array in group_arrays])
        pooled_ranks = np.concatenate([array["ranks"] for array in group_arrays])
        pooled_returned = np.concatenate([
            array["event_returned"] for array in group_arrays
        ])
        pooled = {
            "selection": score_complete_distribution(
                pooled_onsets, pooled_returned, contract=contract,
                training_arrays=training,
            ),
            "validation": score_validation_endpoints(
                pooled_onsets, pooled_ranks, pooled_returned,
                contract=contract, training_arrays=training,
                classifier=classifier,
                kmeans_seed=int(config["validation"]["natural_kmeans_seed"]),
            ),
        }
        summary = summarize_candidate(
            candidate_id, rows, off_lookup, support, pooled,
            levels[candidate_id],
        )
        if isinstance(levels[candidate_id], dict):
            matched = matched_interictal_retention(
                arrays_by_candidate_cell[candidate_id],
                arrays_by_candidate_cell["rev21_zm_off"],
                contract=contract, training_arrays=training,
                classifier=classifier,
                kmeans_seed=int(config["validation"]["natural_kmeans_seed"]),
            )
            replace_retention_with_matched(summary, matched)
        summaries.append(summary)
    _neighbor_scores(summaries)
    active = [row for row in summaries if isinstance(row["level"], dict)]
    ranked = sorted(active, key=lambda row: (
        -row["model_ictal_eligible_fraction"],
        -int(row["interictal_substrate_retained"]),
        (float("inf") if row["worst_standardized_deterioration"] is None
         else row["worst_standardized_deterioration"]),
        -row.get("neighbor_eligible_fraction", 0.0),
        row["log_distance_from_reference"],
        row["candidate_id"],
    ))
    eligible_retained = [row for row in ranked
                         if row["model_ictal_eligible_cells"] > 0
                         and row["interictal_substrate_retained"]]
    near_retained = sorted([
        row for row in ranked
        if row["interictal_substrate_retained"]
        and row["model_ictal_qualification_shortfall"]["median"] is not None
    ], key=lambda row: (
        row["model_ictal_qualification_shortfall"]["median"],
        -row["model_ictal_qualification_shortfall"][
            "operational_transition_fraction"],
        (float("inf") if row["worst_standardized_deterioration"] is None
         else row["worst_standardized_deterioration"]),
        row["log_distance_from_reference"], row["candidate_id"],
    ))
    timescale_seed = (
        eligible_retained[0] if eligible_retained
        else (near_retained[0] if near_retained else None)
    )
    if eligible_retained:
        status = "REV21_COARSE_HAS_CROSS_STATE_CANDIDATE"
        seed_role = "FORMALLY_ELIGIBLE_AND_INTERICTAL_RETAINED"
    elif timescale_seed is not None:
        status = "REV21_COARSE_HAS_NEAR_STATE_TIMESCALE_SEED"
        seed_role = "INTERICTAL_RETAINED_NEAREST_FORMAL_STATE_SHORTFALL"
    else:
        status = "NO_TIMESCALE_SEED_IN_FROZEN_ZM_AMPLITUDE_GRID"
        seed_role = None
    payload = {
        "schema_id": "topic4_rev21_zm_coarse_aggregate_v2",
        "status": status,
        "reference_support": support,
        "per_cell": cells,
        "candidate_summaries": summaries,
        "ranked_active_candidate_ids": [row["candidate_id"] for row in ranked],
        "coarse_candidate_for_timescale_refinement": (
            None if timescale_seed is None else timescale_seed["candidate_id"]),
        "timescale_seed_role": seed_role,
        "formal_cross_state_candidate_present": bool(eligible_retained),
        "patient_heldout_opened": False,
        "patient_ictal_inputs_read": False,
        "selection_unit": "topology_by_dynamics_seed_cell",
        "pooled_role": (
            "event-count-matched development retention screen; not same-network "
            "two-mode confirmation"
        ),
        "retention_calibration": (
            "patient matched-N complete-distribution floor plus per-cell "
            "event-count-matched paired Z/M-off KMeans/OOD null"
        ),
    }
    output = output_root / "coarse/aggregate.json"
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"],
        "candidate": payload["coarse_candidate_for_timescale_refinement"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
