#!/usr/bin/env python3
"""Review-requested closeout audits for the frozen Topic 5 RNN v0.4 run.

This script never trains or selects a model.  It only re-expresses already
frozen outputs with explicit cohort attrition, field-component contrasts,
wiring-resource definitions, early-ictal target reliability, and rollout
diagnostics that are less quantised than the registered Spearman endpoint.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, wilcoxon


PRIMARY_MODEL = "M6_SPATIAL_MID"
NO_REC_MODEL = "M0_NO_REC"
ORDER_CONTROL = "C_ORDER_SHUFFLED"
NO_COST_MODEL = "M4_SPATIAL_GROWTH"
DENSE_MODEL = "M1_DENSE"
EMPIRICAL_REFERENCE = "EMPIRICAL_REFERENCE"
SUPPORTIVE_SUBJECT = "epilepsiae_1146"
TOL = 1e-9
# Pre-flight smoke units share the per_subject tree with the formal run but are
# not part of the frozen 1,426-unit cohort; every glob below must drop them.
SMOKE_PREFIX = "SMOKE_"
# The frozen analysis aggregates seeds inside a fit and then fits inside a
# patient, so a shared fit and a per-side fit carry equal weight.  Collapsing
# both stages at once silently re-weights patients that have two fits; the
# parity assertion in wiring_decomposition() is what keeps this honest.
CWIRING_PARITY_TOL = 1e-5


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty table: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    def clean(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): clean(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(child) for child in item]
        if isinstance(item, np.ndarray):
            return clean(item.tolist())
        if isinstance(item, (np.integer, np.bool_)):
            return item.item()
        if isinstance(item, np.floating):
            item = float(item)
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def paired_summary(values: Iterable[float], seed: int = 20260810,
                   draws: int = 10000) -> dict[str, Any]:
    values = np.asarray(list(values), float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "median": np.nan, "bootstrap_95ci": [np.nan, np.nan],
                "positive": 0, "negative": 0, "tied": 0, "wilcoxon_p": np.nan}
    nonzero = values[np.abs(values) > TOL]
    p = float(wilcoxon(nonzero, method="auto").pvalue) if nonzero.size else 1.0
    rng = np.random.default_rng(seed)
    boot = np.median(rng.choice(values, size=(draws, len(values)), replace=True), axis=1)
    return {
        "n": int(len(values)), "median": float(np.median(values)),
        "bootstrap_95ci": np.quantile(boot, [0.025, 0.975]).tolist(),
        "positive": int(np.sum(values > TOL)), "negative": int(np.sum(values < -TOL)),
        "tied": int(np.sum(np.abs(values) <= TOL)), "wilcoxon_p": p,
    }


def median_by_fit_then_patient(rows: list[dict[str, Any]], metrics: Iterable[str]
                               ) -> list[dict[str, Any]]:
    """Reproduce the frozen two-stage aggregation: seeds inside a fit, fits inside a patient."""
    metrics = list(metrics)
    by_fit: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_fit[(row["subject"], row["fit_id"], row["model"], row["cell"])].append(row)
    by_patient: dict[tuple[str, str, str], list[dict[str, float]]] = defaultdict(list)
    for (subject, _, model, cell), selected in by_fit.items():
        by_patient[(subject, model, cell)].append(
            {key: float(np.nanmedian([float(row[key]) for row in selected])) for key in metrics}
        )
    return [
        {"subject": subject, "model": model, "cell": cell,
         **{key: float(np.nanmedian([fit[key] for fit in fits])) for key in metrics}}
        for (subject, model, cell), fits in sorted(by_patient.items())
    ]


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    finite = sorted(((key, float(value)) for key, value in pvalues.items()
                     if np.isfinite(value)), key=lambda item: item[1])
    result: dict[str, float] = {key: np.nan for key in pvalues}
    running = 0.0
    for index, (key, value) in enumerate(finite):
        running = max(running, min(1.0, (len(finite) - index) * value))
        result[key] = running
    return result


def build_attrition(out_root: Path, source_manifest: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inventory = json.loads((out_root / "EARLY_ICTAL_METADATA_INVENTORY.json").read_text())
    source = json.loads(source_manifest.read_text())
    source_by_subject = {row["subject"]: row for row in source["subjects"]}
    target_root = Path(inventory["target_cache_root"])
    strict = sorted(path.name.removeprefix("outer_") for path in target_root.glob("outer_epilepsiae_*")
                    if list(path.glob("*.npz")))
    expected_primary = [subject for subject in strict if subject != SUPPORTIVE_SUBJECT]
    actual = set(inventory["actual_primary_join"])
    rows = []
    for subject in strict:
        source_row = source_by_subject.get(subject, {})
        geometry = source_row.get("geometry", {}).get(source.get("primary_geometry_tree", "narrow"), {})
        if subject == SUPPORTIVE_SUBJECT:
            status = "SUPPORTIVE_ENGINEERING_ONLY"
            reason = "preassigned development/supportive patient; excluded from primary inference"
        elif subject in actual:
            status = "PRIMARY_INCLUDED"
            reason = "strict target and frozen physical-coordinate model both available"
        else:
            status = "PRIMARY_EXCLUDED_MODEL_INPUT"
            reason = (
                f"frozen model requires >= {source['min_joint_contacts']} exact event/geometry contacts; "
                f"available={geometry.get('n_joint_contacts', 0)}"
            )
        rows.append({
            "subject": subject, "status": status, "reason": reason,
            "strict_target_available": True,
            "n_target_seizures": len(list((target_root / f"outer_{subject}").glob(f"{subject}__*.npz"))),
            "n_event_contacts": source_row.get("n_event_contacts"),
            "geometry_tree": source.get("primary_geometry_tree", "narrow"),
            "geometry_status": geometry.get("status"),
            "n_joint_contacts": geometry.get("n_joint_contacts"),
            "event_contacts_without_geometry": ";".join(geometry.get("event_contacts_without_geometry", [])),
            "minimum_joint_contacts": source.get("min_joint_contacts"),
            "coordinate_available": geometry.get("coord_mm_available"),
            "geometry_eligible": geometry.get("eligible"),
        })
    summary = {
        "contract": "topic5_rnn_motif_review_attrition_v0_4",
        "strict_target_patients": len(strict),
        "strict_target_subjects": strict,
        "supportive_removed_before_primary": SUPPORTIVE_SUBJECT,
        "expected_primary_patients": len(expected_primary),
        "expected_primary_subjects": expected_primary,
        "actual_primary_patients": len(actual),
        "actual_primary_subjects": sorted(actual),
        "excluded_primary_patients": sorted(set(expected_primary) - actual),
        "exclusion_is_posthoc_target_filter": False,
        "exclusion_contract": (
            "pre-existing physical-coordinate model eligibility: at least eight exact "
            "event/geometry contacts"
        ),
        "can_restore_without_changing_frozen_model_cohort": False,
        "interpretation": (
            "The five missing strict-target patients are not silent scoring joins. They were absent "
            "from the frozen physical-coordinate RNN cohort because each had fewer than eight exact "
            "event/geometry contacts. Adding them now would change the model-cohort contract."
        ),
    }
    return rows, summary


def target_cache_dependency(out_root: Path) -> dict[str, Any]:
    """Pin the early-ictal target cache, which still lives outside this result tree.

    The geometry manifest was copied in and is therefore durable, but the frozen
    target arrays are not; if that producing worktree is removed the attrition
    and leave-one-seizure-out sections cannot be re-derived.  Record exactly what
    is depended on so the loss is loud rather than silent.
    """
    inventory = json.loads((out_root / "EARLY_ICTAL_METADATA_INVENTORY.json").read_text())
    target_root = Path(inventory["target_cache_root"])
    files = sorted(path for path in target_root.glob("outer_epilepsiae_*/*.npz"))
    return {
        "contract": "topic5_early_ictal_target_cache_dependency_v0_4",
        "root": str(target_root),
        "inside_this_result_tree": False,
        "n_files": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "files": [{"relative_path": str(path.relative_to(target_root)),
                   "sha256": file_sha256(path)} for path in files],
        "consequence_if_removed": (
            "cohort attrition and leave-one-seizure-out target reliability become "
            "non-re-derivable; the archived JSON/CSV values remain valid but unverifiable"
        ),
    }


def target_leave_one_seizure_out(files: list[Path]) -> list[dict[str, Any]]:
    seizures = []
    for path in sorted(files):
        with np.load(path, allow_pickle=False) as data:
            names = np.asarray(data["contact_names"]).astype(str)
            values = np.asarray(data["target_1_150"], float)
        seizures.append((path.stem.split("__", 1)[-1], names, values))
    rows = []
    for index, (seizure_id, names, observed) in enumerate(seizures):
        others = seizures[:index] + seizures[index + 1:]
        if not others:
            continue
        observed_aligned, predicted = [], []
        for name, value in zip(names, observed):
            other_values = []
            for _, other_names, other_target in others:
                lookup = {str(contact): float(target) for contact, target in zip(other_names, other_target)}
                if name in lookup and np.isfinite(lookup[name]):
                    other_values.append(lookup[name])
            if other_values and np.isfinite(value):
                observed_aligned.append(float(value))
                predicted.append(float(np.mean(other_values)))
        if len(observed_aligned) < 3:
            continue
        rho = spearmanr(observed_aligned, predicted).statistic
        rows.append({
            "seizure_id": seizure_id, "n_contacts": len(observed_aligned),
            "loo_spearman": float(rho) if np.isfinite(rho) else np.nan,
        })
    return rows


def target_reliability(out_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inventory = json.loads((out_root / "EARLY_ICTAL_METADATA_INVENTORY.json").read_text())
    target_root = Path(inventory["target_cache_root"])
    seizure_rows, patient_rows = [], []
    for subject in inventory["actual_primary_join"]:
        rows = target_leave_one_seizure_out(
            list((target_root / f"outer_{subject}").glob(f"{subject}__*.npz"))
        )
        for row in rows:
            seizure_rows.append({"subject": subject, **row})
        values = np.asarray([row["loo_spearman"] for row in rows], float)
        patient_rows.append({
            "subject": subject, "n_seizures": len(list(
                (target_root / f"outer_{subject}").glob(f"{subject}__*.npz")
            )),
            "n_loo_seizures": int(np.isfinite(values).sum()),
            "loo_spearman_median": float(np.nanmedian(values)) if np.isfinite(values).any() else np.nan,
            "loo_abs_spearman_median": float(np.nanmedian(np.abs(values))) if np.isfinite(values).any() else np.nan,
        })
    eligible = [row["loo_spearman_median"] for row in patient_rows
                if np.isfinite(row["loo_spearman_median"])]
    summary = {
        "contract": "topic5_early_ictal_target_loo_reliability_v0_4",
        "target": "clinical-onset 0-10 s, 1-150 Hz baseline-normalized broadband energy",
        "n_primary_patients": len(patient_rows),
        "n_patients_with_at_least_two_seizures": sum(row["n_seizures"] >= 2 for row in patient_rows),
        "n_patients_with_estimable_reliability": len(eligible),
        "n_patients_unestimable_single_seizure": sum(
            not np.isfinite(row["loo_spearman_median"]) for row in patient_rows
        ),
        "patient_level_signed_reliability": paired_summary(eligible),
        "patient_median_signed_rho": float(np.median(eligible)) if eligible else np.nan,
        "interpretation": (
            "Leave-one-seizure-out contact-field reliability is estimable only in patients with at "
            "least two seizures. It measures repeatability of the early-ictal target, not model accuracy."
        ),
    }
    return seizure_rows + patient_rows, summary


def field_decomposition(patient_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    primary = [row for row in patient_rows if row["primary"] == "True"]
    lookup = {(row["subject"], row["model"], row["cell"], row["endpoint"]):
              float(row["all_contact_margin"]) for row in primary}
    subjects = sorted({row["subject"] for row in primary})
    contrasts = {
        "source_contribution": lambda subject: (
            lookup[(subject, PRIMARY_MODEL, "rnn", "canonical_full")]
            - lookup[(subject, PRIMARY_MODEL, "rnn", "seed_removed")]
        ),
        "recurrence_increment": lambda subject: (
            lookup[(subject, PRIMARY_MODEL, "rnn", "canonical_full")]
            - lookup[(subject, NO_REC_MODEL, "rnn", "canonical_full")]
        ),
        "order_specific_increment": lambda subject: (
            lookup[(subject, PRIMARY_MODEL, "rnn", "canonical_full")]
            - lookup[(subject, ORDER_CONTROL, "rnn", "canonical_full")]
        ),
        "wiring_cost_increment": lambda subject: (
            lookup[(subject, PRIMARY_MODEL, "rnn", "canonical_full")]
            - lookup[(subject, NO_COST_MODEL, "rnn", "canonical_full")]
        ),
    }
    rows = []
    for subject in subjects:
        row = {
            "subject": subject,
            "empirical_reference_margin": lookup[(subject, EMPIRICAL_REFERENCE, "reference", "canonical_full")],
            "static_no_recurrence_margin": lookup[(subject, NO_REC_MODEL, "rnn", "canonical_full")],
            "true_order_recurrent_margin": lookup[(subject, PRIMARY_MODEL, "rnn", "canonical_full")],
            "order_shuffled_recurrent_margin": lookup[(subject, ORDER_CONTROL, "rnn", "canonical_full")],
        }
        row.update({name: function(subject) for name, function in contrasts.items()})
        rows.append(row)
    summaries = {name: paired_summary(row[name] for row in rows) for name in contrasts}
    adjusted = holm({name: summary["wilcoxon_p"] for name, summary in summaries.items()})
    for name, value in adjusted.items():
        summaries[name]["holm_q_review_family"] = value
    absolute = {
        "empirical_reference_margin": paired_summary(row["empirical_reference_margin"] for row in rows),
        "static_no_recurrence_margin": paired_summary(row["static_no_recurrence_margin"] for row in rows),
        "true_order_recurrent_margin": paired_summary(row["true_order_recurrent_margin"] for row in rows),
        "order_shuffled_recurrent_margin": paired_summary(row["order_shuffled_recurrent_margin"] for row in rows),
    }
    summary = {
        "contract": "topic5_source_scaffold_recurrence_decomposition_v0_4",
        "status": "POSTHOC_REVIEW_REQUESTED_DECOMPOSITION_OF_FROZEN_FIELDS",
        "primary_endpoint_unchanged": "FIELD_CANONICAL_FULL",
        "key_secondary_unchanged": "FIELD_SEED_REMOVED",
        "n_patients": len(rows), "absolute_margins": absolute, "paired_increments": summaries,
        "definitions": {
            "source_contribution": "M6 canonical-full minus M6 seed-removed",
            "static_no_recurrence_margin": "M0 canonical-full minus synchronized all-contact null",
            "recurrence_increment": "M6 canonical-full minus M0 canonical-full",
            "order_specific_increment": "M6 canonical-full minus order-shuffled M6 canonical-full",
            "wiring_cost_increment": "M6 spatial+cost minus M4 spatial-no-cost canonical-full",
        },
    }
    return rows, summary


def graph_wiring_metrics(graph: dict[str, np.ndarray], d0_mm: float) -> dict[str, float]:
    mask = np.asarray(graph["mask"], bool)
    strength = np.asarray(graph["strength"], float)
    distance = np.asarray(graph["D_mm"], float)
    edge_strength = strength[mask]
    edge_distance = distance[mask]
    edge_count = int(mask.sum())
    total_strength = float(np.sum(edge_strength))
    total_weighted_mm = float(np.sum(edge_strength * edge_distance))
    return {
        "edge_count": edge_count,
        "mean_edge_length_mm": float(np.mean(edge_distance)) if edge_count else np.nan,
        "total_geometric_length_mm": float(np.sum(edge_distance)),
        "total_edge_strength": total_strength,
        "total_strength_weighted_length_mm": total_weighted_mm,
        "mean_edge_strength_weighted_length_over_d0": (
            total_weighted_mm / (edge_count * d0_mm) if edge_count else np.nan
        ),
        "strength_normalized_mean_length_mm": (
            total_weighted_mm / total_strength if total_strength > 0 else np.nan
        ),
    }


def registered_cwiring_parity(out_root: Path, patient_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Guard the aggregation rule against the frozen per-patient table it must match."""
    registered = {(row["subject"], row["model"], row["cell"]): float(row["c_wiring"])
                  for row in read_csv(out_root / "interictal_per_patient.csv")}
    unknown = sorted({(row["subject"], row["model"], row["cell"]) for row in patient_rows}
                     - set(registered))
    if unknown:
        raise RuntimeError(f"wiring rows outside the frozen patient cohort: {unknown}")
    differences = {
        f"{row['subject']}|{row['model']}|{row['cell']}":
            abs(row["mean_edge_strength_weighted_length_over_d0"]
                - registered[(row["subject"], row["model"], row["cell"])])
        for row in patient_rows
    }
    worst = max(differences, key=differences.get)
    return {
        "reference": "interictal_per_patient.csv::c_wiring",
        "n_compared": len(differences),
        "maximum_absolute_difference": differences[worst],
        "maximum_absolute_difference_row": worst,
        "tolerance": CWIRING_PARITY_TOL,
    }


def wiring_decomposition(out_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_rows = []
    n_smoke_skipped = 0
    for path in sorted((out_root / "per_subject").glob("*/*__*/seed*/graph.npz")):
        model = path.parents[1].name.rsplit("__", 1)[0]
        if model.startswith(SMOKE_PREFIX):
            n_smoke_skipped += 1
            continue
        metrics_path = path.parent / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        with np.load(path, allow_pickle=False) as data:
            graph = {key: np.asarray(data[key]) for key in data.files}
        values = graph_wiring_metrics(graph, float(metrics["config"]["d0_mm"]))
        run_rows.append({
            "subject": metrics["subject"], "fit_id": metrics["fit_id"],
            "scope": metrics["fit_scope"], "model": model,
            "cell": metrics["cell"], "seed": metrics["seed"],
            "d0_mm": metrics["config"]["d0_mm"], **values,
            "reported_c_wiring": metrics["c_wiring"],
            "c_wiring_absolute_error": abs(values["mean_edge_strength_weighted_length_over_d0"]
                                            - float(metrics["c_wiring"])),
        })
    metrics = (
        "edge_count", "mean_edge_length_mm", "total_geometric_length_mm",
        "total_edge_strength", "total_strength_weighted_length_mm",
        "mean_edge_strength_weighted_length_over_d0", "strength_normalized_mean_length_mm",
    )
    patient_rows = median_by_fit_then_patient(run_rows, metrics)
    maximum_error = max(row["c_wiring_absolute_error"] for row in run_rows)
    parity = registered_cwiring_parity(out_root, patient_rows)
    if parity["maximum_absolute_difference"] > CWIRING_PARITY_TOL:
        raise RuntimeError(
            "patient-level wiring cost does not reproduce the frozen interictal table: "
            f"max|Δ|={parity['maximum_absolute_difference']:.6g} over {parity['n_compared']} rows"
        )
    summary = {
        "contract": "topic5_wiring_resource_decomposition_v0_4",
        "n_graph_runs": len(run_rows), "n_patient_model_rows": len(patient_rows),
        "n_smoke_graph_runs_excluded": n_smoke_skipped,
        "patient_aggregation": "median over seeds within a fit, then median over fits within a patient",
        "registered_c_wiring_parity": parity,
        "reported_c_wiring_definition": (
            "mean across active edges of edge_strength * distance_mm / d0_mm"
        ),
        "reported_c_wiring_is_total_wiring": False,
        "maximum_recomputed_c_wiring_absolute_error": maximum_error,
        "metrics": {
            "total_geometric_length_mm": "sum of physical lengths over active directed edges",
            "total_strength_weighted_length_mm": "sum of edge_strength times physical length",
            "mean_edge_strength_weighted_length_over_d0": (
                "training wiring cost; average edge_strength times normalized distance"
            ),
            "strength_normalized_mean_length_mm": (
                "edge-strength-weighted mean physical edge length"
            ),
        },
    }
    return patient_rows, summary


def sequence_metrics(observed: np.ndarray, generated: list[list[int]]) -> dict[str, float]:
    observed = np.asarray(observed, int)
    observed_order = {int(contact): int(observed[contact] - 1)
                      for contact in np.flatnonzero(observed > 0)}
    generated_order = {int(contact): rank for rank, rank_set in enumerate(generated[1:])
                       for contact in rank_set}
    observed_contacts, generated_contacts = set(observed_order), set(generated_order)
    union = observed_contacts | generated_contacts
    shared = sorted(observed_contacts & generated_contacts)
    jaccard = len(shared) / len(union) if union else np.nan
    if len(shared) < 3:
        return {"kendall_tau_b": np.nan, "normalized_rank_mae": np.nan,
                "participation_jaccard": jaccard}
    left = np.asarray([observed_order[contact] for contact in shared], float)
    right = np.asarray([generated_order[contact] for contact in shared], float)
    # A small direct tau-b implementation is much faster here than repeatedly
    # invoking SciPy's decorated hypothesis-test wrapper for tens of thousands
    # of short contact vectors.
    upper = np.triu_indices(len(shared), 1)
    sign_left = np.sign(left[:, None] - left[None, :])[upper]
    sign_right = np.sign(right[:, None] - right[None, :])[upper]
    product = sign_left * sign_right
    concordant = int(np.sum(product > 0))
    discordant = int(np.sum(product < 0))
    tied_left_only = int(np.sum((sign_left == 0) & (sign_right != 0)))
    tied_right_only = int(np.sum((sign_right == 0) & (sign_left != 0)))
    denominator = np.sqrt(
        (concordant + discordant + tied_left_only)
        * (concordant + discordant + tied_right_only)
    )
    tau = ((concordant - discordant) / denominator) if denominator > 0 else np.nan
    scale = max(float(np.max(left)), float(np.max(right)), 1.0)
    return {"kendall_tau_b": float(tau) if np.isfinite(tau) else np.nan,
            "normalized_rank_mae": float(np.mean(np.abs(left - right)) / scale),
            "participation_jaccard": float(jaccard)}


def rollout_diagnostics(out_root: Path, max_events_per_fit_seed: int = 128
                        ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    fit_seed = []
    n_available = 0
    n_evaluated = 0
    keys = ("kendall_tau_b", "normalized_rank_mae", "participation_jaccard")
    n_smoke_skipped = 0
    for path in sorted((out_root / "per_subject").glob("*/*__*/seed*/heldout_rollouts.json.gz")):
        model = path.parents[1].name.rsplit("__", 1)[0]
        if model.startswith(SMOKE_PREFIX):
            n_smoke_skipped += 1
            continue
        metrics = json.loads((path.parent / "metrics.json").read_text())
        fit_id = metrics["fit_id"]
        if fit_id not in cache:
            with np.load(out_root / "cache" / fit_id / "events.npz", allow_pickle=False) as data:
                keep = np.asarray(data["split"]) >= 0
                cache[fit_id] = (np.asarray(data["ranks"])[keep], np.asarray(data["split"])[keep])
        ranks, split = cache[fit_id]
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            records = json.load(handle)
        available_count = len(records)
        n_available += available_count
        if len(records) > max_events_per_fit_seed:
            indices = np.linspace(0, len(records) - 1, max_events_per_fit_seed).round().astype(int)
            records = [records[index] for index in np.unique(indices)]
        n_evaluated += len(records)
        selected = []
        for record in records:
            index = int(record["kept_event_index"])
            if split[index] != 2:
                raise RuntimeError(f"non-heldout rollout record: {path}: {index}")
            selected.append(sequence_metrics(ranks[index], record["generated_rank_sets"]))
        fit_seed.append({
            "subject": metrics["subject"], "fit_id": fit_id, "model": model,
            "cell": metrics["cell"], "seed": metrics["seed"],
            "n_rollout_records_available": available_count,
            "n_rollout_records_evaluated": len(selected),
            **{key: float(np.nanmedian([row[key] for row in selected])) for key in keys},
        })
    patients = median_by_fit_then_patient(fit_seed, keys)
    model_summaries = {}
    for model, cell in sorted({(row["model"], row["cell"]) for row in patients}):
        selected = [row for row in patients if row["model"] == model and row["cell"] == cell]
        model_summaries[f"{model}|{cell}"] = {
            key: paired_summary(row[key] for row in selected) for key in keys
        }
    patient_lookup = {(row["subject"], row["model"], row["cell"]): row for row in patients}
    contrasts = {}
    for comparator in (NO_REC_MODEL, ORDER_CONTROL):
        subjects = sorted({row["subject"] for row in patients
                           if (row["subject"], PRIMARY_MODEL, "rnn") in patient_lookup
                           and (row["subject"], comparator, "rnn") in patient_lookup})
        contrasts[f"{PRIMARY_MODEL}_vs_{comparator}"] = {
            "kendall_tau_b_gain": paired_summary(
                patient_lookup[(subject, PRIMARY_MODEL, "rnn")]["kendall_tau_b"]
                - patient_lookup[(subject, comparator, "rnn")]["kendall_tau_b"]
                for subject in subjects
            ),
            "normalized_rank_mae_reduction": paired_summary(
                patient_lookup[(subject, comparator, "rnn")]["normalized_rank_mae"]
                - patient_lookup[(subject, PRIMARY_MODEL, "rnn")]["normalized_rank_mae"]
                for subject in subjects
            ),
            "participation_jaccard_gain": paired_summary(
                patient_lookup[(subject, PRIMARY_MODEL, "rnn")]["participation_jaccard"]
                - patient_lookup[(subject, comparator, "rnn")]["participation_jaccard"]
                for subject in subjects
            ),
        }
    summary = {
        "contract": "topic5_rollout_rank_diagnostics_v0_4",
        "n_rollout_records_available": n_available,
        "n_rollout_records_evaluated": n_evaluated,
        "maximum_events_per_fit_seed": max_events_per_fit_seed,
        "sampling_rule": "deterministic evenly spaced heldout events within each fit/seed",
        "n_smoke_units_excluded": n_smoke_skipped,
        "patient_aggregation": "median over seeds within a fit, then median over fits within a patient",
        "n_patient_model_rows": len(patients),
        "model_summaries": model_summaries,
        "primary_model_contrasts": contrasts,
        "registered_primary_unchanged": "seed-removed Spearman rank correlation",
        "diagnostics": {
            "kendall_tau_b": "tie-aware concordance on contacts present in observed and generated post-seed sequence",
            "normalized_rank_mae": "mean shared-contact rank error divided by maximum post-seed rank",
            "participation_jaccard": "post-seed observed/generated contact-set overlap",
        },
    }
    return patients, summary


def wiring_ratio_to_dense(wiring_rows: list[dict[str, Any]], keys: list[str]
                          ) -> list[dict[str, Any]]:
    lookup = {(row["subject"], row["model"]): row for row in wiring_rows if row["cell"] == "rnn"}
    subjects = sorted({subject for subject, model in lookup if model == PRIMARY_MODEL}
                      & {subject for subject, model in lookup if model == DENSE_MODEL})
    return [
        {"subject": subject,
         **{key: (float(lookup[(subject, PRIMARY_MODEL)][key])
                  / float(lookup[(subject, DENSE_MODEL)][key])
                  if float(lookup[(subject, DENSE_MODEL)][key]) else np.nan) for key in keys}}
        for subject in subjects
    ]


FIGURE_README_MARKER = "<!-- topic5-rnn-motif-v0.4-review-closeout-figure -->"
FIGURE_README_SECTION = """### topic5_rnn_motif_review_closeout.png / .pdf

审阅后收口诊断六联图。A 是 strict early-ictal target 16 人 → primary 期望 15 人 →
与冻结物理坐标模型精确相交 10 人的队列漏斗；B 是 early-ictal target 自身的
留一次发作可靠性（只有 ≥2 次发作的患者可估计）；C 把经验间期场、静态场、
顺序打乱场和真实顺序 RNN 场放在同一 null-relative 尺度上比较；D 是同一批患者上
起点 / recurrence / 真实顺序 / wiring cost 四项配对增量；E 是 Spatial + cost 相对
Dense 的四种连接资源占比（对数轴，100% 为 dense 参照）；F 是自由推演的
Kendall τb，用来检查登记的 Spearman 端点 0.5 平台是否掩盖模型差异。

**关注点**：每个面板都标出了自己的患者分母（10 / 8 / 10 / 10 / 21 / 21），不同面板
分母不同是设计使然，不可跨面板合读；C 与 D 是同一个 10 人交集，E 与 F 是 21 人
间期队列。
"""


def write_figure_readme(out_root: Path) -> None:
    """Insert this figure's section ahead of the plotting script's own marker.

    plot_topic5_rnn_motif_figures_v0_4.py rewrites everything from its marker
    onwards, so anything appended after it would be lost on the next figure run.
    """
    readme = out_root / "figures" / "README.md"
    existing = readme.read_text() if readme.exists() else "# Topic 5 RNN connectivity-motif figures\n"
    head, separator, tail = existing.partition("<!-- topic5-rnn-motif-v0.4-stage-and-final-figures -->")
    head = head.split(FIGURE_README_MARKER)[0].rstrip()
    block = f"{FIGURE_README_MARKER}\n\n{FIGURE_README_SECTION.strip()}\n"
    readme.write_text(f"{head}\n\n{block}\n{separator}{tail}" if separator
                      else f"{head}\n\n{block}")


def plot_closeout(out_root: Path, attrition: dict[str, Any], target_rows: list[dict[str, Any]],
                  target_summary: dict[str, Any], decomposition: list[dict[str, Any]],
                  wiring_rows: list[dict[str, Any]], rollout_rows: list[dict[str, Any]]) -> None:
    plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "axes.linewidth": 0.7})
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.1), constrained_layout=True)

    ax = axes[0, 0]
    counts = [attrition["strict_target_patients"], attrition["expected_primary_patients"],
              attrition["actual_primary_patients"]]
    ax.plot(range(3), counts, color="#333333", lw=1.4, marker="o", ms=5)
    ax.fill_between(range(3), counts, color="#d9d9d9", alpha=0.35)
    ax.set_xticks(range(3), ["Strict\ntarget", "Primary\nexpected", "Exact model–\ntarget join"])
    ax.set_ylabel("Patients"); ax.set_ylim(0, max(counts) + 2)
    for x, value in enumerate(counts): ax.text(x, value + 0.45, str(value), ha="center", fontsize=8)

    ax = axes[0, 1]
    reliability = [row for row in target_rows if "subject" in row and "loo_spearman_median" in row
                   and np.isfinite(row["loo_spearman_median"])]
    values = np.asarray([row["loo_spearman_median"] for row in reliability], float)
    ax.scatter(np.zeros(len(values)), values, s=22, color="#2d6ca2", alpha=0.75)
    if len(values): ax.plot([-0.18, 0.18], [np.median(values)] * 2, color="#111111", lw=1.5)
    ax.axhline(0, color="#999999", lw=0.7); ax.set_xlim(-0.35, 0.35)
    ax.set_xticks([0], [f"n={len(values)} of {target_summary['n_primary_patients']}\n(≥2 seizures)"])
    ax.set_ylabel("Early-ictal field\nLOO Spearman ρ")

    ax = axes[0, 2]
    absolute_keys = ["empirical_reference_margin", "static_no_recurrence_margin",
                     "order_shuffled_recurrent_margin", "true_order_recurrent_margin"]
    labels = ["Empirical", "Static", "Order\nshuffle", "True-order\nRNN"]
    colors = ["#cf3d3d", "#9b9b9b", "#6c65b8", "#d64c4c"]
    for index, (key, color) in enumerate(zip(absolute_keys, colors)):
        values = np.asarray([row[key] for row in decomposition], float)
        ax.scatter(index + np.linspace(-0.10, 0.10, len(values)), values, s=15, color=color, alpha=0.65)
        ax.plot([index - 0.2, index + 0.2], [np.median(values)] * 2, color="#111111", lw=1.2)
    ax.axhline(0, color="#999999", lw=0.7); ax.set_xticks(range(4), labels)
    ax.set_ylabel("Early-ictal null-relative margin")
    ax.set_title(f"n={len(decomposition)} patients", fontsize=8, pad=3)

    ax = axes[1, 0]
    increment_keys = ["source_contribution", "recurrence_increment",
                      "order_specific_increment", "wiring_cost_increment"]
    labels = ["Start", "Recurrence", "True order", "Wiring cost"]
    for index, key in enumerate(increment_keys):
        values = np.asarray([row[key] for row in decomposition], float)
        ax.scatter(index + np.linspace(-0.10, 0.10, len(values)), values, s=15,
                   color="#3b75af", alpha=0.65)
        ax.plot([index - 0.2, index + 0.2], [np.median(values)] * 2, color="#111111", lw=1.2)
    ax.axhline(0, color="#999999", lw=0.7); ax.set_xticks(range(4), labels, rotation=18, ha="right")
    ax.set_ylabel("Paired change in margin")
    ax.set_title(f"n={len(decomposition)} patients", fontsize=8, pad=3)

    ax = axes[1, 1]
    # The claim in the closeout text is the resource ratio against dense, so plot
    # that directly; total length against |w|-weighted length is a near-collinear
    # redraw of one quantity and cannot carry it.
    resources = [("edge_count", "Active\nedges"),
                 ("total_geometric_length_mm", "Total\nlength"),
                 ("total_strength_weighted_length_mm", "Total |w|×\nlength"),
                 ("mean_edge_strength_weighted_length_over_d0", "Mean edge\ncost")]
    ratios = wiring_ratio_to_dense(wiring_rows, [key for key, _ in resources])
    for index, (key, _) in enumerate(resources):
        values = np.asarray([row[key] for row in ratios], float) * 100.0
        values = values[np.isfinite(values)]
        ax.scatter(index + np.linspace(-0.11, 0.11, len(values)), values, s=15,
                   color="#d64c4c", alpha=0.6)
        if len(values): ax.plot([index - 0.2, index + 0.2], [np.median(values)] * 2,
                                color="#111111", lw=1.3)
    ax.axhline(100, color="#999999", lw=0.7)
    ax.set_yscale("log"); ax.set_ylim(1, 200)
    ax.set_xticks(range(len(resources)), [label for _, label in resources])
    ax.set_ylabel("Spatial + cost as %\nof dense (log)")
    ax.set_title(f"n={len(ratios)} patients", fontsize=8, pad=3)

    ax = axes[1, 2]
    models = ["M0_NO_REC", "M1_DENSE", "M3_FIXED_LOCAL", "M6_SPATIAL_MID", "C_ORDER_SHUFFLED"]
    colors = ["#9b9b9b", "#252525", "#3b75af", "#d64c4c", "#6c65b8"]
    labels = ["No rec.", "Dense", "Local", "Sp. + cost", "Shuffle"]
    for index, (model, color) in enumerate(zip(models, colors)):
        values = np.asarray([row["kendall_tau_b"] for row in rollout_rows
                             if row["cell"] == "rnn" and row["model"] == model], float)
        values = values[np.isfinite(values)]
        ax.scatter(index + np.linspace(-0.10, 0.10, len(values)), values, s=14, color=color, alpha=0.6)
        if len(values): ax.plot([index - 0.2, index + 0.2], [np.median(values)] * 2, color="#111111", lw=1.2)
    ax.axhline(0, color="#999999", lw=0.7); ax.set_xticks(range(len(models)), labels, rotation=24, ha="right")
    ax.set_ylabel("Free-rollout Kendall τb")
    n_rollout_patients = len({row["subject"] for row in rollout_rows if row["cell"] == "rnn"})
    ax.set_title(f"n={n_rollout_patients} patients", fontsize=8, pad=3)

    for label, ax in zip("ABCDEF", axes.ravel()):
        ax.text(-0.17, 1.04, label, transform=ax.transAxes, fontsize=11, fontweight="bold")
    figure_dir = out_root / "figures"; figure_dir.mkdir(exist_ok=True)
    stem = figure_dir / "topic5_rnn_motif_review_closeout"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    write_figure_readme(out_root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--source-manifest", type=Path,
        help=("Geometry-source manifest. On the first run this is copied into the "
              "closeout result root; later runs may omit this argument."),
    )
    args = parser.parse_args()
    out_root = args.out_root.resolve()

    durable_source = out_root / "REVIEW_SOURCE_GEOMETRY_MANIFEST.json"
    source_manifest = args.source_manifest.resolve() if args.source_manifest else durable_source
    if not source_manifest.exists():
        parser.error(
            "--source-manifest is required on the first run because the durable review copy "
            f"does not yet exist: {durable_source}"
        )
    input_manifest_sha256 = file_sha256(source_manifest)
    source_payload = json.loads(source_manifest.read_text())
    write_json(durable_source, source_payload)
    # Hash the durable copy, not the input: reruns that pass --source-manifest
    # would otherwise record a digest that does not verify the file it names.
    source_sha256 = file_sha256(durable_source)

    target_dependency = target_cache_dependency(out_root)
    write_json(out_root / "REVIEW_EARLY_ICTAL_TARGET_DEPENDENCY.json", target_dependency)

    attrition_rows, attrition_summary = build_attrition(out_root, durable_source)
    attrition_summary["source_manifest"] = str(durable_source)
    attrition_summary["source_manifest_sha256"] = source_sha256
    attrition_summary["input_manifest"] = str(source_manifest)
    attrition_summary["input_manifest_sha256"] = input_manifest_sha256
    attrition_summary["early_ictal_target_cache_root"] = target_dependency["root"]
    attrition_summary["early_ictal_target_cache_inside_result_tree"] = False
    write_csv(out_root / "review_attrition_audit.csv", attrition_rows)
    write_json(out_root / "REVIEW_ATTRITION_AUDIT.json", attrition_summary)

    target_rows, target_summary = target_reliability(out_root)
    # Separate row types remain identifiable by their available columns in JSON;
    # the compact CSV is patient-level only.
    target_patient_rows = [row for row in target_rows if "loo_spearman_median" in row]
    write_csv(out_root / "early_ictal_target_reliability_patient.csv", target_patient_rows)
    write_json(out_root / "EARLY_ICTAL_TARGET_RELIABILITY.json", {
        **target_summary,
        "per_seizure": [row for row in target_rows if "seizure_id" in row],
        "per_patient": target_patient_rows,
    })

    decomposition_rows, decomposition_summary = field_decomposition(
        read_csv(out_root / "early_ictal_per_patient_model.csv")
    )
    write_csv(out_root / "early_ictal_field_decomposition_patient.csv", decomposition_rows)
    write_json(out_root / "EARLY_ICTAL_FIELD_DECOMPOSITION.json", decomposition_summary)

    wiring_rows, wiring_summary = wiring_decomposition(out_root)
    write_csv(out_root / "wiring_resource_decomposition_patient.csv", wiring_rows)
    write_json(out_root / "WIRING_RESOURCE_DECOMPOSITION.json", wiring_summary)

    rollout_rows, rollout_summary = rollout_diagnostics(out_root)
    write_csv(out_root / "rollout_rank_diagnostics_patient.csv", rollout_rows)
    write_json(out_root / "ROLLOUT_RANK_DIAGNOSTICS.json", rollout_summary)

    plot_closeout(out_root, attrition_summary, target_patient_rows, target_summary,
                  decomposition_rows, wiring_rows, rollout_rows)
    write_json(out_root / "REVIEW_CLOSEOUT_AUDIT_COMPLETE.json", {
        "status": "COMPLETE", "training_rerun": False,
        "reason_no_training_rerun": (
            "All review items addressed here are denominator, endpoint decomposition, resource-definition, "
            "target-reliability, or diagnostic-readout issues. Frozen model parameters are unchanged."
        ),
        "registered_primary_endpoint": "FIELD_CANONICAL_FULL",
        "registered_key_secondary_endpoint": "FIELD_SEED_REMOVED",
        "review_source_geometry_manifest": str(durable_source),
        "review_source_geometry_manifest_sha256": source_sha256,
        "early_ictal_target_cache_dependency": str(
            out_root / "REVIEW_EARLY_ICTAL_TARGET_DEPENDENCY.json"
        ),
        "early_ictal_target_cache_root": target_dependency["root"],
        "early_ictal_target_cache_inside_result_tree": False,
        "patient_aggregation_matches_frozen_interictal_table": True,
        "smoke_units_excluded": True,
        "target_values_read": True,
        "posthoc_review_decomposition": True,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
