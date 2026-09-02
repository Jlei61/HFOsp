#!/usr/bin/env python3
"""Repair the v0.5 non-oracle train-prevalence mixture before target unseal.

The oracle A/B candidate contract is intentionally unchanged:

* shared subjects use the two train-only modes of one fit;
* non-collinear subjects use the all-event ``own_a`` and ``own_b`` geometry
  fits as the two oracle candidate fields.

For a non-oracle mixture, however, an all-event field from each geometry must
not be weighted by mode prevalence.  This target-free repair derives an A-mode
component from ``own_a`` and a B-mode component from ``own_b``, using only
train-fitted mode labels and existing held-out rollouts, and replaces only the
two ``*_train_prevalence_mixture`` vectors.  A/B oracle vectors and every model
checkpoint remain byte-for-byte scientifically unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_topic5_multiscale_fields_v0_5 import (
    ARMS,
    ab_prevalence,
    sha256_file,
    train_mode_to_ab,
    unit_metrics_path,
    vector_sha256,
)
from build_topic5_train_only_modes_suffix_null_v0_5 import features
from build_topic5_rnn_motif_fields_v0_4 import aggregate_records


DEFAULT_OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
DEFAULT_OLD = ROOT / "results/topic5_lbss_full_tissue_rnn_v0_3"
FIELD_ROOT = Path(
    "/home/honglab/leijiaxin/HFOsp/results/"
    "interictal_propagation_masked/template_gradient_fields/per_subject"
)
MIXTURE_ENDPOINTS = ("canonical_full", "seed_removed")


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def atomic_savez(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.stem + ".", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def target_label(scope: str) -> str:
    if scope == "own_a":
        return "A"
    if scope == "own_b":
        return "B"
    raise ValueError(f"mode component requires own_a/own_b scope, got {scope}")


def weighted_mixture(a: np.ndarray, b: np.ndarray, prevalence: dict[str, float]) -> np.ndarray:
    """A/B-label invariant train-prevalence mixture of two mode fields."""
    return prevalence["A"] * np.asarray(a, float) + prevalence["B"] * np.asarray(b, float)


def full_event_train_modes(cache: Path) -> np.ndarray:
    """Assign every event to a frozen train-only mode using its full rank field.

    The K=2 centers were fitted on outer-training events only.  Applying those
    fixed centers to validation/test events is an evaluation-time grouping; it
    neither refits the modes nor exposes any early-ictal target.  This is
    deliberately distinct from ``events['mode']``, which is a prefix-only hard
    classifier and can collapse to one class even when both train modes exist.
    """
    with np.load(cache / "events.npz", allow_pickle=False) as events:
        ranks = np.asarray(events["ranks"], dtype=np.int16)
    with np.load(cache / "train_only_modes.npz", allow_pickle=False) as modes:
        centers = np.asarray(modes["centers"], dtype=float)
    event_features = features(ranks)
    if centers.shape != (2, event_features.shape[1]):
        raise RuntimeError(
            f"train-center/event-feature shape mismatch: {cache}: "
            f"{centers.shape} vs {event_features.shape}"
        )
    squared_distance = np.mean(
        (event_features[:, None, :] - centers[None, :, :]) ** 2,
        axis=2,
    )
    if not np.isfinite(squared_distance).all():
        raise RuntimeError(f"non-finite full-event train-mode distances: {cache}")
    return np.argmin(squared_distance, axis=1).astype(np.int8)


def remap_record_full_train_modes(records: list[dict], cache: Path) -> list[dict]:
    """Attach frozen-center full-event mode labels to held-out rollouts."""
    with np.load(cache / "events.npz", allow_pickle=False) as events:
        source_indices = np.asarray(events["event_source_index"], dtype=np.int64)
    labels = full_event_train_modes(cache)
    source_to_mode: dict[int, int] = {}
    for source, mode in zip(source_indices, labels):
        source, mode = int(source), int(mode)
        if source in source_to_mode and source_to_mode[source] != mode:
            raise RuntimeError(f"event_source_index {source} maps to two full-event modes")
        source_to_mode[source] = mode
    output = []
    for record in records:
        source = int(record["event_source_index"])
        if source not in source_to_mode:
            raise RuntimeError(f"rollout event_source_index absent from v0.5 cache: {source}")
        row = dict(record)
        row["mode"] = source_to_mode[source]
        output.append(row)
    return output


def aligned_event_labels(cache: Path, subject: str, contacts: np.ndarray) -> tuple[np.ndarray, dict[int, str]]:
    mapping = train_mode_to_ab(cache, subject, contacts, FIELD_ROOT)
    labels = np.asarray(
        [mapping[int(mode)] for mode in full_event_train_modes(cache)], dtype="U1"
    )
    return labels, mapping


def mode_component_for_seed(
    out: Path,
    old: Path,
    reused: set[str],
    fit_id: str,
    arm: str,
    seed: int,
    cache: Path,
    subject: str,
    contacts: np.ndarray,
    label: str,
) -> tuple[dict[str, np.ndarray], int]:
    metrics_path = unit_metrics_path(out, old, fit_id, arm, seed, reused)
    metrics = json.loads(metrics_path.read_text())
    if metrics.get("target_values_read") is not False:
        raise RuntimeError(f"target flag is not false: {metrics_path}")
    with gzip.open(metrics_path.parent / "heldout_rollouts.json.gz", "rt", encoding="utf-8") as stream:
        records = remap_record_full_train_modes(json.load(stream), cache)
    mapping = train_mode_to_ab(cache, subject, contacts, FIELD_ROOT)
    selected = [row for row in records if mapping[int(row["mode"])] == label]
    if not selected:
        raise RuntimeError(f"no heldout {label}-mode records: {fit_id} {arm} seed{seed}")
    return aggregate_records(selected, len(contacts)), len(selected)


def subject_mode_components(
    out: Path,
    old: Path,
    reused: set[str],
    subject: str,
    arm: str,
    fits: pd.DataFrame,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, float], dict]:
    by_scope = {str(row.scope): row for row in fits.itertuples()}
    if set(by_scope) != {"own_a", "own_b"}:
        raise RuntimeError(f"non-collinear subject lacks exact own_a/own_b pair: {subject}")
    components: dict[str, dict[str, np.ndarray]] = {}
    evidence: dict = {"subject": subject, "arm": arm}
    reference_labels = None
    prevalence_by_scope = {}
    for scope in ("own_a", "own_b"):
        fit = by_scope[scope]
        cache = out / "cache" / str(fit.fit_id)
        provenance = json.loads((cache / "provenance.json").read_text())
        contacts = np.asarray(provenance["joint_contacts"], dtype="U64")
        labels, mapping = aligned_event_labels(cache, subject, contacts)
        if reference_labels is None:
            reference_labels = labels
        elif not np.array_equal(reference_labels, labels):
            raise RuntimeError(f"own_a/own_b train-mode labels disagree: {subject}")
        modes = np.load(cache / "train_only_modes.npz", allow_pickle=False)
        prevalence_by_scope[scope] = ab_prevalence(modes["train_counts"], mapping)
        label = target_label(scope)
        seed_payloads, counts = [], []
        for seed in range(3):
            payload, count = mode_component_for_seed(
                out, old, reused, str(fit.fit_id), arm, seed, cache,
                subject, contacts, label,
            )
            seed_payloads.append(payload)
            counts.append(count)
        component = {"contacts": contacts}
        for endpoint in MIXTURE_ENDPOINTS:
            component[endpoint] = np.nanmedian(
                np.stack([payload[endpoint] for payload in seed_payloads]), axis=0
            )
        components[label] = component
        evidence[f"producer_{label}"] = str(fit.fit_id)
        evidence[f"heldout_{label}_events_per_seed"] = counts
        evidence[f"mode_mapping_{scope}"] = {str(key): value for key, value in mapping.items()}
    if not np.array_equal(components["A"]["contacts"], components["B"]["contacts"]):
        raise RuntimeError(f"own_a/own_b contacts disagree: {subject}")
    for key in ("A", "B"):
        values = [prevalence_by_scope[scope][key] for scope in ("own_a", "own_b")]
        if not np.isclose(values[0], values[1], rtol=0, atol=1e-12):
            raise RuntimeError(f"own_a/own_b train prevalence disagrees: {subject} {key}: {values}")
    prevalence = prevalence_by_scope["own_a"]
    evidence["train_prevalence_A"] = prevalence["A"]
    evidence["train_prevalence_B"] = prevalence["B"]
    evidence["component_event_label_contract"] = (
        "FULL_EVENT_ASSIGNED_TO_FROZEN_TRAIN_ONLY_K2_CENTERS"
    )
    return components, prevalence, evidence


def update_manifest(out: Path) -> None:
    manifest_path = out / "MODEL_FIELD_MANIFEST.csv"
    manifest = pd.read_csv(manifest_path)
    for path_string, group in manifest.groupby("path", sort=False):
        path = Path(path_string)
        with np.load(path, allow_pickle=False) as data:
            payload = {name: np.asarray(data[name]) for name in data.files}
        digest = sha256_file(path)
        manifest.loc[group.index, "file_sha256"] = digest
        for index in group.index:
            endpoint = str(manifest.at[index, "endpoint"])
            manifest.at[index, "vector_sha256"] = vector_sha256(payload[endpoint])
    manifest.to_csv(manifest_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD)
    args = parser.parse_args()
    out, old = args.out_root.resolve(), args.old_root.resolve()
    if os.environ.get("TOPIC5_V0_5_TARGET_SEALED") != "1":
        raise RuntimeError("mixture repair must run inside the physical target embargo")
    if not (out / "STAGE_F_TARGET_FREE_COMPLETE.json").exists():
        raise RuntimeError("Stage F must finish before mixture repair")
    if (out / "TARGET_UNSEAL_AUTHORIZATION.json").exists():
        raise RuntimeError("mixture repair is forbidden after target authorization")

    census = pd.read_csv(out / "FULL_PARENT_FIT_CENSUS.csv")
    reuse = pd.read_csv(out / "V0_3_CHECKPOINT_REUSE_AUDIT.csv")
    reused = set(reuse.loc[reuse.checkpoint_reuse_eligible.astype(bool), "fit_id"].astype(str))
    patient_metrics = pd.read_csv(out / "MODEL_FIELD_PATIENT_METRICS.csv")
    rows = []
    changed_subjects = []
    for subject, fits in census.groupby("subject", sort=False):
        if set(fits.scope.astype(str)) == {"shared"}:
            continue
        changed_subjects.append(str(subject))
        for arm in ARMS:
            components, prevalence, evidence = subject_mode_components(
                out, old, reused, str(subject), arm, fits
            )
            destination = out / "model_fields/intact/per_patient" / str(subject) / f"{arm}.npz"
            with np.load(destination, allow_pickle=False) as data:
                payload = {name: np.asarray(data[name]) for name in data.files}
            oracle_hashes_before = {
                name: vector_sha256(payload[name])
                for name in ("A_canonical_full", "B_canonical_full", "A_seed_removed", "B_seed_removed")
            }
            for endpoint in MIXTURE_ENDPOINTS:
                left = np.asarray(components["A"][endpoint], float)
                right = np.asarray(components["B"][endpoint], float)
                if endpoint == "seed_removed":
                    left = np.nan_to_num(left, nan=0.0)
                    right = np.nan_to_num(right, nan=0.0)
                payload[f"{endpoint}_train_prevalence_mixture"] = weighted_mixture(
                    left, right, prevalence
                ).astype(np.float32)
            component_path = (
                out / "model_fields/train_mode_mixture_components/per_patient" /
                str(subject) / f"{arm}.npz"
            )
            atomic_savez(component_path, {
                "contacts": components["A"]["contacts"],
                "A_canonical_full": components["A"]["canonical_full"],
                "B_canonical_full": components["B"]["canonical_full"],
                "A_seed_removed": components["A"]["seed_removed"],
                "B_seed_removed": components["B"]["seed_removed"],
                "train_prevalence_A": np.asarray(prevalence["A"], dtype=np.float32),
                "train_prevalence_B": np.asarray(prevalence["B"], dtype=np.float32),
            })
            atomic_savez(destination, payload)
            oracle_hashes_after = {
                name: vector_sha256(payload[name]) for name in oracle_hashes_before
            }
            if oracle_hashes_before != oracle_hashes_after:
                raise RuntimeError(f"oracle A/B vectors changed during mixture-only repair: {subject} {arm}")
            rows.append({
                **evidence,
                "component_path": str(component_path),
                "component_sha256": sha256_file(component_path),
                "patient_field_path": str(destination),
                "patient_field_sha256": sha256_file(destination),
                "oracle_vectors_unchanged": True,
                "target_values_read": False,
            })
            patient_metrics.loc[
                (patient_metrics.subject.astype(str) == str(subject)) &
                (patient_metrics.arm.astype(str) == arm),
                "field_sha256",
            ] = sha256_file(destination)

    patient_metrics.to_csv(out / "MODEL_FIELD_PATIENT_METRICS.csv", index=False)
    update_manifest(out)
    repair_table = out / "TRAIN_PREVALENCE_MIXTURE_REPAIR.csv"
    pd.DataFrame(rows).to_csv(repair_table, index=False)
    if len(rows) != 14 * len(ARMS):
        raise RuntimeError(f"expected 70 non-collinear patient-arm repairs, found {len(rows)}")

    marker_path = out / "MODEL_FIELDS_FROZEN.json"
    marker = json.loads(marker_path.read_text())
    marker.update({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_manifest_sha256": sha256_file(out / "MODEL_FIELD_MANIFEST.csv"),
        "train_prevalence_mixture_contract": (
            "shared: train-mode A/B; non-collinear: full-event fields assigned to "
            "frozen train-only K2 centers, then A-aligned own_a plus B-aligned "
            "own_b, weighted by train-only prevalence"
        ),
        "train_prevalence_mixture_repair_sha256": sha256_file(repair_table),
        "oracle_ab_vectors_changed": False,
        "target_values_read": False,
    })
    write_json(marker_path, marker)

    stage_f_path = out / "STAGE_F_TARGET_FREE_COMPLETE.json"
    stage_f = json.loads(stage_f_path.read_text())
    stage_f.setdefault("artifact_hashes", {})["MODEL_FIELDS_FROZEN.json"] = sha256_file(marker_path)
    stage_f["post_stage_target_free_repairs"] = {
        "TRAIN_PREVALENCE_MIXTURE_REPAIR.csv": sha256_file(repair_table),
        "scientific_scope": "NONORACLE_MIXTURE_ONLY_ORACLE_AB_UNCHANGED",
        "target_values_read": False,
    }
    write_json(stage_f_path, stage_f)

    complete = {
        "contract": "topic5_train_prevalence_mixture_repair_v0_5",
        "status": "PASS_TARGET_FREE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "changed_subjects": len(changed_subjects),
        "changed_patient_arm_fields": len(rows),
        "oracle_ab_vectors_changed": False,
        "repair_table": str(repair_table),
        "repair_table_sha256": sha256_file(repair_table),
        "model_field_manifest_sha256": sha256_file(out / "MODEL_FIELD_MANIFEST.csv"),
        "model_fields_marker_sha256": sha256_file(marker_path),
        "producer_script": str(Path(__file__).resolve()),
        "producer_script_sha256": sha256_file(Path(__file__).resolve()),
        "target_values_read": False,
    }
    write_json(out / "TRAIN_PREVALENCE_MIXTURE_REPAIR_COMPLETE.json", complete)
    print(json.dumps(complete, indent=2))


if __name__ == "__main__":
    main()
