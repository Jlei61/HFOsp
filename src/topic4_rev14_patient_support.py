"""Training-only patient-support acceptance for Topic 4 rev14.

The module has no simulator or filesystem I/O.  It freezes patient-training
cross-block floors and scores every model ``contact_primary`` family without
silently deleting missing, out-of-distribution, or single-shaft events.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, TypedDict

import numpy as np


SCHEMA_ID = "topic4_rev14_patient_support_v2"
PRIMARY_LABEL_KEY = "patient_train_old_labels"
CLASSIFIER_LABEL_KEY = "patient_train_classifier_labels"
SHAFTS = ("ICL", "SCL")
MODE_ENDPOINTS = (
    "recruitment.ICL",
    "recruitment.SCL",
    "precedence.ICL-SCL",
    "joint_shaft_fraction",
)
OOD_ENDPOINT = "OOD"
ALL_ENDPOINTS = tuple(
    [f"mode_{mode}.{key}" for mode in (0, 1) for key in MODE_ENDPOINTS]
    + [OOD_ENDPOINT]
)


class PatientSupportSchemaError(ValueError):
    """An input violates the frozen patient-support schema."""


class PatientSupportHashError(PatientSupportSchemaError):
    """An input no longer matches the hashes frozen with the calibration."""


class SupportEndpointRecord(TypedDict):
    raw_distance: float
    floor_q95: float
    ratio: float


@dataclass(frozen=True)
class PatientTrainingData:
    contact_names: tuple[str, ...]
    shaft_ids: tuple[str, ...]
    onsets: np.ndarray
    old_labels: np.ndarray
    classifier_labels: np.ndarray
    block_ids: np.ndarray
    ood: np.ndarray
    data_sha256: str
    contact_sha256: str
    declared_source_sha256: str | None = None
    shaft_aware_k2_labels: np.ndarray | None = None
    primary_label_key: str = PRIMARY_LABEL_KEY


@dataclass(frozen=True)
class ModelContactPrimaryData:
    contact_names: tuple[str, ...]
    onsets: np.ndarray
    classifier_labels: np.ndarray
    ood: np.ndarray


@dataclass(frozen=True)
class PatientSupportCalibration:
    schema_id: str
    primary_label_key: str
    patient_data_sha256: str
    contact_sha256: str
    declared_source_sha256: str | None
    sample_size: int
    floor_draws: int
    seed: int
    floor_q95: Mapping[str, float]
    floor_distributions: Mapping[str, np.ndarray]
    eligible_block_counts: Mapping[str, int]
    joint_draws: int
    joint_inner_draws: int
    joint_threshold_q95: float
    joint_distribution: np.ndarray
    joint_endpoint_distributions: Mapping[str, np.ndarray]
    joint_pseudo_blocks: np.ndarray
    joint_reference_blocks: np.ndarray
    joint_distribution_sha256: str
    joint_block_audit_sha256: str
    calibration_sha256: str


@dataclass(frozen=True)
class PatientSupportResult:
    status: str
    score: float | None
    endpoints: Mapping[str, SupportEndpointRecord]
    support: Mapping[str, Any]
    sampling: Mapping[str, Any]
    invalid_reason: str | None = None


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _update_array_hash(digest: "hashlib._Hash", values: np.ndarray) -> None:
    array = np.ascontiguousarray(np.asarray(values))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def _array_sha256(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    _update_array_hash(digest, values)
    return digest.hexdigest()


def _named_arrays_sha256(values: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(values):
        digest.update(key.encode("utf-8"))
        _update_array_hash(digest, np.asarray(values[key]))
    return digest.hexdigest()


def _block_audit_sha256(pseudo_blocks: np.ndarray,
                        reference_blocks: np.ndarray) -> str:
    digest = hashlib.sha256()
    _update_array_hash(digest, np.asarray(pseudo_blocks, dtype=str))
    _update_array_hash(digest, np.asarray(reference_blocks, dtype=str))
    return digest.hexdigest()


def _validate_sha256(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    text = str(value).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise PatientSupportSchemaError(f"{name} is not a SHA-256 digest")
    return text


def _contact_sha256(contact_names: tuple[str, ...],
                    shaft_ids: tuple[str, ...]) -> str:
    return _canonical_json_sha256(sorted(zip(contact_names, shaft_ids)))


def _patient_data_sha256(contact_names: tuple[str, ...],
                         shaft_ids: tuple[str, ...], onsets: np.ndarray,
                         old_labels: np.ndarray, classifier_labels: np.ndarray,
                         block_ids: np.ndarray,
                         ood: np.ndarray) -> str:
    """Hash patient arrays after canonical contact reordering."""
    order = np.argsort(np.asarray(contact_names), kind="stable")
    digest = hashlib.sha256()
    digest.update(_contact_sha256(contact_names, shaft_ids).encode("ascii"))
    _update_array_hash(digest, np.asarray(onsets)[:, order])
    _update_array_hash(digest, old_labels)
    _update_array_hash(digest, classifier_labels)
    _update_array_hash(digest, block_ids)
    _update_array_hash(digest, ood)
    return digest.hexdigest()


def _reject_heldout_keys(value: Mapping[str, Any]) -> None:
    contaminated: list[str] = []

    def visit(current: Any, prefix: str) -> None:
        if isinstance(current, Mapping):
            for key, item in current.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                if "heldout" in str(key).lower():
                    contaminated.append(path)
                visit(item, path)
        elif isinstance(current, (list, tuple)):
            for index, item in enumerate(current):
                visit(item, f"{prefix}[{index}]")

    visit(value, "")
    if contaminated:
        raise PatientSupportSchemaError(
            f"held-out fields are forbidden in training support: {sorted(contaminated)}"
        )


def patient_training_from_mapping(
        value: Mapping[str, Any], *,
        expected_source_sha256: str | None = None) -> PatientTrainingData:
    """Load only the allowlisted training arrays and reject held-out poisoning."""
    _reject_heldout_keys(value)
    required = {
        "contact_names", "shaft_ids", "patient_train_onsets",
        PRIMARY_LABEL_KEY, CLASSIFIER_LABEL_KEY, "patient_train_block_ids",
        "patient_train_ood",
    }
    missing = sorted(required.difference(value))
    if missing:
        raise PatientSupportSchemaError(f"patient training input lacks {missing}")
    primary = str(value.get("primary_label_key", PRIMARY_LABEL_KEY))
    if primary != PRIMARY_LABEL_KEY:
        raise PatientSupportSchemaError(
            f"primary labels must be {PRIMARY_LABEL_KEY}, not {primary}"
        )

    contact_names = tuple(str(item) for item in np.asarray(value["contact_names"]).tolist())
    shaft_ids = tuple(str(item).upper() for item in np.asarray(value["shaft_ids"]).tolist())
    onsets = np.asarray(value["patient_train_onsets"], dtype=np.float64).copy()
    old_labels = np.asarray(value[PRIMARY_LABEL_KEY], dtype=np.int8).copy()
    classifier_labels = np.asarray(value[CLASSIFIER_LABEL_KEY], dtype=np.int8).copy()
    block_ids = np.asarray(value["patient_train_block_ids"]).copy()
    ood = np.asarray(value["patient_train_ood"], dtype=bool).copy()
    diagnostic = value.get("patient_train_shaft_aware_k2_labels")
    diagnostic_labels = (
        None if diagnostic is None else np.asarray(diagnostic, dtype=np.int8).copy()
    )
    _validate_training_arrays(
        contact_names, shaft_ids, onsets, old_labels, classifier_labels,
        block_ids, ood,
        diagnostic_labels,
    )
    # A contact-primary event with no observable contact is necessarily OOD.
    ood |= ~np.isfinite(onsets).any(axis=1)
    declared = _validate_sha256(
        value.get("source_sha256"), "source_sha256",
    )
    expected = _validate_sha256(expected_source_sha256, "expected_source_sha256")
    if expected is not None and declared != expected:
        raise PatientSupportHashError("patient source SHA-256 differs from contract")
    return PatientTrainingData(
        contact_names=contact_names,
        shaft_ids=shaft_ids,
        onsets=onsets,
        old_labels=old_labels,
        classifier_labels=classifier_labels,
        block_ids=block_ids,
        ood=ood,
        data_sha256=_patient_data_sha256(
            contact_names, shaft_ids, onsets, old_labels, classifier_labels,
            block_ids, ood,
        ),
        contact_sha256=_contact_sha256(contact_names, shaft_ids),
        declared_source_sha256=declared,
        shaft_aware_k2_labels=diagnostic_labels,
    )


def model_contact_primary_from_mapping(
        value: Mapping[str, Any]) -> ModelContactPrimaryData:
    """Build the all-family model table; no OOD or missing row is filtered."""
    _reject_heldout_keys(value)
    required = {"contact_names", "onsets", "classifier_labels", "ood"}
    missing = sorted(required.difference(value))
    if missing:
        raise PatientSupportSchemaError(f"model contact-primary input lacks {missing}")
    names = tuple(str(item) for item in np.asarray(value["contact_names"]).tolist())
    if len(names) != len(set(names)):
        raise PatientSupportSchemaError("model contact names must be unique")
    onsets = np.asarray(value["onsets"], dtype=np.float64).copy()
    labels = np.asarray(value["classifier_labels"], dtype=np.int8).copy()
    ood = np.asarray(value["ood"], dtype=bool).copy()
    if onsets.ndim != 2 or onsets.shape[1] != len(names):
        raise PatientSupportSchemaError("model onsets must be events x contacts")
    if labels.shape != (len(onsets),) or ood.shape != (len(onsets),):
        raise PatientSupportSchemaError("model labels and OOD must align with events")
    if np.any(~np.isin(labels, (0, 1))):
        raise PatientSupportSchemaError("model classifier labels must be binary")
    ood |= ~np.isfinite(onsets).any(axis=1)
    return ModelContactPrimaryData(names, onsets, labels, ood)


def _validate_training_arrays(
        contact_names: tuple[str, ...], shaft_ids: tuple[str, ...],
        onsets: np.ndarray, old_labels: np.ndarray,
        classifier_labels: np.ndarray, block_ids: np.ndarray,
        ood: np.ndarray, diagnostic_labels: np.ndarray | None) -> None:
    n_contacts = len(contact_names)
    if n_contacts == 0 or len(set(contact_names)) != n_contacts:
        raise PatientSupportSchemaError("patient contact names must be nonempty and unique")
    if len(shaft_ids) != n_contacts or set(shaft_ids) != set(SHAFTS):
        raise PatientSupportSchemaError("patient contacts must contain ICL and SCL shafts")
    if onsets.ndim != 2 or onsets.shape[1] != n_contacts:
        raise PatientSupportSchemaError("patient onsets must be events x contacts")
    n_events = len(onsets)
    if (old_labels.shape != (n_events,)
            or classifier_labels.shape != (n_events,)
            or block_ids.shape != (n_events,)
            or ood.shape != (n_events,)):
        raise PatientSupportSchemaError("patient labels, blocks and OOD must align")
    if np.any(~np.isin(old_labels, (0, 1))):
        raise PatientSupportSchemaError("patient old labels must be binary")
    if np.any(~np.isin(classifier_labels, (0, 1))):
        raise PatientSupportSchemaError("patient classifier labels must be binary")
    if diagnostic_labels is not None and diagnostic_labels.shape != (n_events,):
        raise PatientSupportSchemaError("shaft-aware K2 diagnostic does not align")


def sample_cross_block_pairs(
        labels: np.ndarray, block_ids: np.ndarray, *, mode: int | None,
        sample_size: int, draws: int,
        rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw two sides from disjoint blocks, using one event from every block."""
    labels = np.asarray(labels)
    blocks = np.asarray(block_ids)
    sample_size, draws = int(sample_size), int(draws)
    if labels.shape != blocks.shape or labels.ndim != 1:
        raise PatientSupportSchemaError("labels and block ids must be aligned vectors")
    if sample_size <= 0 or draws <= 0:
        raise PatientSupportSchemaError("sample size and draws must be positive")
    eligible = np.ones(len(labels), dtype=bool) if mode is None else labels == int(mode)
    eligible_blocks = np.unique(blocks[eligible])
    required = 2 * sample_size
    if len(eligible_blocks) < required:
        descriptor = "all events" if mode is None else f"old mode {int(mode)}"
        raise PatientSupportSchemaError(
            f"{descriptor} has {len(eligible_blocks)} blocks; {required} required"
        )
    left = np.empty((draws, sample_size), dtype=np.int64)
    right = np.empty_like(left)
    left_blocks = np.empty((draws, sample_size), dtype=blocks.dtype)
    right_blocks = np.empty_like(left_blocks)
    for draw in range(draws):
        chosen_blocks = rng.choice(eligible_blocks, size=required, replace=False)
        for selected_blocks, indices, audit in (
                (chosen_blocks[:sample_size], left, left_blocks),
                (chosen_blocks[sample_size:], right, right_blocks)):
            audit[draw] = selected_blocks
            for column, block in enumerate(selected_blocks):
                candidates = np.flatnonzero(eligible & (blocks == block))
                indices[draw, column] = int(rng.choice(candidates))
    return left, right, left_blocks, right_blocks


def _shaft_indices(shaft_ids: tuple[str, ...]) -> dict[str, np.ndarray]:
    shafts = np.asarray(shaft_ids)
    return {shaft: np.flatnonzero(shafts == shaft) for shaft in SHAFTS}


def _cross_pairs(groups: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.asarray([
        (left, right)
        for left in np.asarray(groups["ICL"], dtype=int)
        for right in np.asarray(groups["SCL"], dtype=int)
    ], dtype=np.int64)


def _four_state_precedence(onsets: np.ndarray, pairs: np.ndarray,
                           tie_tolerance: float) -> np.ndarray:
    values = np.asarray(onsets, dtype=np.float64)
    output = np.empty((len(pairs), 4), dtype=np.float64)
    for row, (left, right) in enumerate(pairs):
        left_values, right_values = values[:, left], values[:, right]
        jointly = np.isfinite(left_values) & np.isfinite(right_values)
        difference = left_values - right_values
        output[row] = (
            np.mean(jointly & (difference < -tie_tolerance)),
            np.mean(jointly & (difference > tie_tolerance)),
            np.mean(jointly & (np.abs(difference) <= tie_tolerance)),
            np.mean(~jointly),
        )
    return output


def _row_js(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    midpoint = 0.5 * (left + right)
    with np.errstate(divide="ignore", invalid="ignore"):
        term_left = np.where(left > 0.0, left * np.log2(left / midpoint), 0.0)
        term_right = np.where(right > 0.0, right * np.log2(right / midpoint), 0.0)
    return 0.5 * (term_left.sum(axis=1) + term_right.sum(axis=1))


def mode_endpoint_distances(
        left: np.ndarray, right: np.ndarray, shaft_ids: tuple[str, ...], *,
        tie_tolerance: float = 1e-12) -> dict[str, float]:
    """Four shaft-balanced distances for two matched event tables."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1:] != right.shape[1:]:
        raise PatientSupportSchemaError("matched event tables have different contacts")
    if len(left) == 0 or len(right) == 0:
        raise PatientSupportSchemaError("matched event tables must be nonempty")
    groups = _shaft_indices(shaft_ids)
    left_mask, right_mask = np.isfinite(left), np.isfinite(right)
    output: dict[str, float] = {}
    for shaft in SHAFTS:
        indices = groups[shaft]
        left_recruitment = left_mask[:, indices].mean(axis=0)
        right_recruitment = right_mask[:, indices].mean(axis=0)
        output[f"recruitment.{shaft}"] = float(
            np.mean(np.abs(left_recruitment - right_recruitment))
        )
    pairs = _cross_pairs(groups)
    left_precedence = _four_state_precedence(left, pairs, float(tie_tolerance))
    right_precedence = _four_state_precedence(right, pairs, float(tie_tolerance))
    output["precedence.ICL-SCL"] = float(
        np.mean(_row_js(left_precedence, right_precedence))
    )
    left_joint = (
        left_mask[:, groups["ICL"]].any(axis=1)
        & left_mask[:, groups["SCL"]].any(axis=1)
    )
    right_joint = (
        right_mask[:, groups["ICL"]].any(axis=1)
        & right_mask[:, groups["SCL"]].any(axis=1)
    )
    output["joint_shaft_fraction"] = float(abs(left_joint.mean() - right_joint.mean()))
    return output


def _draw_outer_block_split(
        pseudo_labels: np.ndarray, reference_labels: np.ndarray,
        block_ids: np.ndarray, *, sample_size: int,
        rng: np.random.Generator, max_attempts: int = 10000,
        ) -> tuple[np.ndarray, np.ndarray]:
    """Split blocks into model/reference pools that can support every endpoint."""
    pseudo_labels = np.asarray(pseudo_labels)
    reference_labels = np.asarray(reference_labels)
    blocks = np.asarray(block_ids)
    if (pseudo_labels.shape != blocks.shape
            or reference_labels.shape != blocks.shape):
        raise PatientSupportSchemaError("joint calibration labels do not align")
    unique_blocks = np.unique(blocks)
    minimum_side = 2 * int(sample_size)
    if len(unique_blocks) < 2 * minimum_side:
        raise PatientSupportSchemaError(
            f"joint calibration needs at least {2 * minimum_side} recording blocks"
        )
    pseudo_size = len(unique_blocks) // 2
    for _ in range(int(max_attempts)):
        shuffled = rng.permutation(unique_blocks)
        pseudo, reference = shuffled[:pseudo_size], shuffled[pseudo_size:]
        if len(pseudo) < minimum_side or len(reference) < minimum_side:
            continue
        valid = True
        for side, labels in (
                (pseudo, pseudo_labels), (reference, reference_labels)):
            side_mask = np.isin(blocks, side)
            for mode in (0, 1):
                mode_blocks = np.unique(blocks[side_mask & (labels == mode)])
                valid &= len(mode_blocks) >= int(sample_size)
        if valid:
            return np.asarray(pseudo), np.asarray(reference)
    raise PatientSupportSchemaError(
        "could not construct a joint block-disjoint pseudo/reference split"
    )


def _joint_null_calibration(
        patient: PatientTrainingData, floors: Mapping[str, float], *,
        sample_size: int, joint_draws: int, joint_inner_draws: int,
        rng: np.random.Generator, tie_tolerance: float,
        ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Generate synchronized nine-endpoint null rows under block separation."""
    endpoint_rows = {
        key: np.empty(int(joint_draws), dtype=np.float64)
        for key in ALL_ENDPOINTS
    }
    unique_blocks = np.unique(patient.block_ids)
    pseudo_size = len(unique_blocks) // 2
    reference_size = len(unique_blocks) - pseudo_size
    pseudo_audit = np.empty((int(joint_draws), pseudo_size), dtype=unique_blocks.dtype)
    reference_audit = np.empty(
        (int(joint_draws), reference_size), dtype=unique_blocks.dtype,
    )

    for outer in range(int(joint_draws)):
        pseudo_blocks, reference_blocks = _draw_outer_block_split(
            patient.classifier_labels, patient.old_labels, patient.block_ids,
            sample_size=sample_size, rng=rng,
        )
        pseudo_audit[outer] = pseudo_blocks
        reference_audit[outer] = reference_blocks
        pseudo_mask = np.isin(patient.block_ids, pseudo_blocks)
        endpoint_sums = {key: 0.0 for key in ALL_ENDPOINTS}

        for _ in range(int(joint_inner_draws)):
            for mode in (0, 1):
                eligible = np.flatnonzero(
                    pseudo_mask & (patient.classifier_labels == mode)
                )
                pseudo_sample, missing = _draw_model_mode_with_missing(
                    patient.onsets, eligible, size=sample_size, rng=rng,
                )
                if missing:
                    raise PatientSupportSchemaError(
                        "joint pseudo-model unexpectedly required mode padding"
                    )
                reference_indices = _draw_patient_reference(
                    patient, mode=mode, size=sample_size, rng=rng,
                    allowed_blocks=reference_blocks,
                )
                distances = mode_endpoint_distances(
                    pseudo_sample, patient.onsets[reference_indices],
                    patient.shaft_ids, tie_tolerance=tie_tolerance,
                )
                for endpoint in MODE_ENDPOINTS:
                    endpoint_sums[f"mode_{mode}.{endpoint}"] += distances[endpoint]

            pseudo_indices = np.flatnonzero(pseudo_mask)
            pseudo_ood, missing = _draw_model_ood_with_missing(
                patient.ood[pseudo_indices], size=2 * sample_size, rng=rng,
            )
            if missing:
                raise PatientSupportSchemaError(
                    "joint pseudo-model unexpectedly required OOD padding"
                )
            reference_indices = _draw_patient_reference(
                patient, mode=None, size=2 * sample_size, rng=rng,
                allowed_blocks=reference_blocks,
            )
            endpoint_sums[OOD_ENDPOINT] += abs(
                float(pseudo_ood.mean())
                - float(patient.ood[reference_indices].mean())
            )

        for key in ALL_ENDPOINTS:
            endpoint_rows[key][outer] = _safe_ratio(
                endpoint_sums[key] / float(joint_inner_draws), floors[key],
            )

    normalized = np.stack([endpoint_rows[key] for key in ALL_ENDPOINTS], axis=1)
    joint_distribution = np.max(normalized, axis=1)
    if not np.all(np.isfinite(joint_distribution)):
        raise PatientSupportSchemaError(
            "joint calibration is non-finite; a marginal q95 cannot normalize its endpoint"
        )
    return endpoint_rows, joint_distribution, pseudo_audit, reference_audit


def _calibration_payload(
        calibration: PatientSupportCalibration | None = None, **values: Any,
        ) -> dict[str, Any]:
    if calibration is not None:
        values = {
            "schema_id": calibration.schema_id,
            "primary_label_key": calibration.primary_label_key,
            "patient_data_sha256": calibration.patient_data_sha256,
            "contact_sha256": calibration.contact_sha256,
            "declared_source_sha256": calibration.declared_source_sha256,
            "sample_size": int(calibration.sample_size),
            "floor_draws": int(calibration.floor_draws),
            "seed": int(calibration.seed),
            "floor_q95": dict(calibration.floor_q95),
            "eligible_block_counts": dict(calibration.eligible_block_counts),
            "joint_draws": int(calibration.joint_draws),
            "joint_inner_draws": int(calibration.joint_inner_draws),
            "joint_threshold_q95": float(calibration.joint_threshold_q95),
            "joint_distribution_sha256": calibration.joint_distribution_sha256,
            "joint_endpoint_distributions_sha256": _named_arrays_sha256(
                calibration.joint_endpoint_distributions
            ),
            "joint_block_audit_sha256": calibration.joint_block_audit_sha256,
        }
    return values


def build_patient_support_calibration(
        patient: PatientTrainingData, *, sample_size: int = 6,
        floor_draws: int = 4096, joint_draws: int = 512,
        joint_inner_draws: int = 512, seed: int = 20260827,
        tie_tolerance: float = 1e-12) -> PatientSupportCalibration:
    """Freeze marginal scales and a joint block-disjoint patient-support null."""
    if patient.primary_label_key != PRIMARY_LABEL_KEY:
        raise PatientSupportSchemaError("shaft-aware K2 cannot replace old A/B labels")
    if min(int(sample_size), int(floor_draws), int(joint_draws),
           int(joint_inner_draws)) <= 0:
        raise PatientSupportSchemaError("all calibration draw counts must be positive")
    rng = np.random.default_rng(int(seed))
    distributions: dict[str, np.ndarray] = {}
    eligible_counts: dict[str, int] = {}
    for mode in (0, 1):
        left, right, _, _ = sample_cross_block_pairs(
            patient.old_labels, patient.block_ids, mode=mode,
            sample_size=sample_size, draws=floor_draws, rng=rng,
        )
        eligible_counts[str(mode)] = int(len(np.unique(
            patient.block_ids[patient.old_labels == mode]
        )))
        rows = {key: np.empty(int(floor_draws), dtype=np.float64)
                for key in MODE_ENDPOINTS}
        for draw in range(int(floor_draws)):
            distances = mode_endpoint_distances(
                patient.onsets[left[draw]], patient.onsets[right[draw]],
                patient.shaft_ids, tie_tolerance=tie_tolerance,
            )
            for key in MODE_ENDPOINTS:
                rows[key][draw] = distances[key]
        for key, values in rows.items():
            distributions[f"mode_{mode}.{key}"] = values

    left, right, _, _ = sample_cross_block_pairs(
        patient.old_labels, patient.block_ids, mode=None,
        sample_size=sample_size, draws=floor_draws, rng=rng,
    )
    distributions[OOD_ENDPOINT] = np.abs(
        patient.ood[left].mean(axis=1) - patient.ood[right].mean(axis=1)
    )
    floors = {
        key: float(np.quantile(values, 0.95, method="linear"))
        for key, values in distributions.items()
    }
    if any(not np.isfinite(value) or value < 0.0 for value in floors.values()):
        raise PatientSupportSchemaError(
            "every marginal q95 must be finite and nonnegative"
        )
    eligible_counts["all"] = int(len(np.unique(patient.block_ids)))
    joint_endpoints, joint_distribution, pseudo_blocks, reference_blocks = (
        _joint_null_calibration(
            patient, floors, sample_size=int(sample_size),
            joint_draws=int(joint_draws), joint_inner_draws=int(joint_inner_draws),
            rng=rng, tie_tolerance=float(tie_tolerance),
        )
    )
    joint_threshold = float(np.quantile(
        joint_distribution, 0.95, method="linear",
    ))
    joint_distribution_sha = _array_sha256(joint_distribution)
    joint_block_sha = _block_audit_sha256(pseudo_blocks, reference_blocks)
    payload = _calibration_payload(
        schema_id=SCHEMA_ID,
        primary_label_key=PRIMARY_LABEL_KEY,
        patient_data_sha256=patient.data_sha256,
        contact_sha256=patient.contact_sha256,
        declared_source_sha256=patient.declared_source_sha256,
        sample_size=int(sample_size),
        floor_draws=int(floor_draws),
        seed=int(seed),
        floor_q95=floors,
        eligible_block_counts=eligible_counts,
        joint_draws=int(joint_draws),
        joint_inner_draws=int(joint_inner_draws),
        joint_threshold_q95=joint_threshold,
        joint_distribution_sha256=joint_distribution_sha,
        joint_endpoint_distributions_sha256=_named_arrays_sha256(joint_endpoints),
        joint_block_audit_sha256=joint_block_sha,
    )
    return PatientSupportCalibration(
        schema_id=SCHEMA_ID,
        primary_label_key=PRIMARY_LABEL_KEY,
        patient_data_sha256=patient.data_sha256,
        contact_sha256=patient.contact_sha256,
        declared_source_sha256=patient.declared_source_sha256,
        sample_size=int(sample_size),
        floor_draws=int(floor_draws),
        seed=int(seed),
        floor_q95=floors,
        floor_distributions=distributions,
        eligible_block_counts=eligible_counts,
        joint_draws=int(joint_draws),
        joint_inner_draws=int(joint_inner_draws),
        joint_threshold_q95=joint_threshold,
        joint_distribution=joint_distribution,
        joint_endpoint_distributions=joint_endpoints,
        joint_pseudo_blocks=pseudo_blocks,
        joint_reference_blocks=reference_blocks,
        joint_distribution_sha256=joint_distribution_sha,
        joint_block_audit_sha256=joint_block_sha,
        calibration_sha256=_canonical_json_sha256(payload),
    )


def _verify_calibration(calibration: PatientSupportCalibration) -> None:
    expected_keys = set(ALL_ENDPOINTS)
    if set(calibration.floor_q95) != expected_keys:
        raise PatientSupportHashError("calibration endpoint set changed")
    if set(calibration.floor_distributions) != expected_keys:
        raise PatientSupportHashError("calibration floor distributions changed")
    if set(calibration.joint_endpoint_distributions) != expected_keys:
        raise PatientSupportHashError("joint endpoint set changed")
    for key in expected_keys:
        values = np.asarray(calibration.floor_distributions[key], dtype=np.float64)
        if values.shape != (int(calibration.floor_draws),) or not np.all(np.isfinite(values)):
            raise PatientSupportHashError(f"calibration distribution is invalid: {key}")
        observed = float(np.quantile(values, 0.95, method="linear"))
        if observed != float(calibration.floor_q95[key]):
            raise PatientSupportHashError(f"calibration q95 changed: {key}")
        joint_values = np.asarray(
            calibration.joint_endpoint_distributions[key], dtype=np.float64,
        )
        if joint_values.shape != (int(calibration.joint_draws),):
            raise PatientSupportHashError(f"joint endpoint distribution changed: {key}")
    normalized = np.stack([
        calibration.joint_endpoint_distributions[key] for key in ALL_ENDPOINTS
    ], axis=1)
    observed_joint = np.max(normalized, axis=1)
    if not np.array_equal(observed_joint, calibration.joint_distribution):
        raise PatientSupportHashError("joint null is not synchronized across endpoints")
    if _array_sha256(calibration.joint_distribution) != calibration.joint_distribution_sha256:
        raise PatientSupportHashError("joint distribution SHA-256 changed")
    threshold = float(np.quantile(observed_joint, 0.95, method="linear"))
    if threshold != float(calibration.joint_threshold_q95):
        raise PatientSupportHashError("joint q95 threshold changed")
    pseudo = np.asarray(calibration.joint_pseudo_blocks)
    reference = np.asarray(calibration.joint_reference_blocks)
    if pseudo.shape[0] != int(calibration.joint_draws) or reference.shape[0] != int(calibration.joint_draws):
        raise PatientSupportHashError("joint block audit draw count changed")
    for draw in range(int(calibration.joint_draws)):
        if set(pseudo[draw].tolist()).intersection(reference[draw].tolist()):
            raise PatientSupportHashError("joint pseudo/reference blocks overlap")
    if _block_audit_sha256(pseudo, reference) != calibration.joint_block_audit_sha256:
        raise PatientSupportHashError("joint block audit SHA-256 changed")
    if _canonical_json_sha256(
            _calibration_payload(calibration)) != calibration.calibration_sha256:
        raise PatientSupportHashError("calibration content SHA-256 changed")


def _draw_patient_reference(
        patient: PatientTrainingData, *, mode: int | None, size: int,
        rng: np.random.Generator,
        allowed_blocks: np.ndarray | None = None) -> np.ndarray:
    eligible = np.ones(len(patient.old_labels), dtype=bool)
    if mode is not None:
        eligible &= patient.old_labels == int(mode)
    if allowed_blocks is not None:
        eligible &= np.isin(patient.block_ids, np.asarray(allowed_blocks))
    blocks = np.unique(patient.block_ids[eligible])
    if len(blocks) < int(size):
        raise PatientSupportSchemaError("patient reference has too few recording blocks")
    chosen = rng.choice(blocks, size=int(size), replace=False)
    return np.asarray([
        int(rng.choice(np.flatnonzero(eligible & (patient.block_ids == block))))
        for block in chosen
    ], dtype=np.int64)


def _draw_model_mode_with_missing(
        onsets: np.ndarray, eligible_indices: np.ndarray, *, size: int,
        rng: np.random.Generator) -> tuple[np.ndarray, int]:
    output = np.full((int(size), onsets.shape[1]), np.nan, dtype=np.float64)
    n_take = min(int(size), len(eligible_indices))
    if n_take:
        chosen = rng.choice(eligible_indices, size=n_take, replace=False)
        output[:n_take] = onsets[chosen]
    return output, int(size) - n_take


def _draw_model_ood_with_missing(
        ood: np.ndarray, *, size: int,
        rng: np.random.Generator) -> tuple[np.ndarray, int]:
    output = np.ones(int(size), dtype=bool)
    n_take = min(int(size), len(ood))
    if n_take:
        chosen = rng.choice(len(ood), size=n_take, replace=False)
        output[:n_take] = ood[chosen]
    return output, int(size) - n_take


def _safe_ratio(distance: float, floor_q95: float) -> float:
    if floor_q95 < 0.0 or not np.isfinite(floor_q95):
        raise PatientSupportHashError("calibration contains an invalid q95 floor")
    if floor_q95 == 0.0:
        return 0.0 if distance == 0.0 else float("inf")
    return float(distance / floor_q95)


def _reorder_model_contacts(model: ModelContactPrimaryData,
                            patient: PatientTrainingData) -> np.ndarray:
    if set(model.contact_names) != set(patient.contact_names):
        raise PatientSupportSchemaError("model and patient contact sets differ")
    source = {name: index for index, name in enumerate(model.contact_names)}
    return model.onsets[:, [source[name] for name in patient.contact_names]]


def _validate_score_inputs(patient: PatientTrainingData,
                           model: ModelContactPrimaryData) -> None:
    _validate_training_arrays(
        patient.contact_names, patient.shaft_ids, np.asarray(patient.onsets),
        np.asarray(patient.old_labels), np.asarray(patient.classifier_labels),
        np.asarray(patient.block_ids),
        np.asarray(patient.ood), patient.shaft_aware_k2_labels,
    )
    observed_hash = _patient_data_sha256(
        patient.contact_names, patient.shaft_ids, patient.onsets,
        patient.old_labels, patient.classifier_labels, patient.block_ids,
        patient.ood,
    )
    if observed_hash != patient.data_sha256:
        raise PatientSupportHashError("patient training data SHA-256 changed")
    if len(model.contact_names) != len(set(model.contact_names)):
        raise PatientSupportSchemaError("model contact names must be unique")
    if np.asarray(model.onsets).ndim != 2:
        raise PatientSupportSchemaError("model onsets must be events x contacts")
    if np.asarray(model.onsets).shape[1] != len(model.contact_names):
        raise PatientSupportSchemaError("model contact columns do not align")
    n_events = len(model.onsets)
    if (np.asarray(model.classifier_labels).shape != (n_events,)
            or np.asarray(model.ood).shape != (n_events,)):
        raise PatientSupportSchemaError("model labels and OOD do not align")
    if np.any(~np.isin(model.classifier_labels, (0, 1))):
        raise PatientSupportSchemaError("model classifier labels must be binary")


def score_patient_support(
        patient: PatientTrainingData, calibration: PatientSupportCalibration,
        model: ModelContactPrimaryData, *, score_draws: int | None = None,
        seed: int = 20260828,
        tie_tolerance: float = 1e-12) -> PatientSupportResult:
    """Score one model network and return the single joint support condition."""
    _verify_calibration(calibration)
    _validate_score_inputs(patient, model)
    if calibration.schema_id != SCHEMA_ID or calibration.primary_label_key != PRIMARY_LABEL_KEY:
        raise PatientSupportHashError("patient-support calibration schema changed")
    if patient.primary_label_key != PRIMARY_LABEL_KEY:
        raise PatientSupportSchemaError("primary patient labels are not old A/B")
    if patient.data_sha256 != calibration.patient_data_sha256:
        raise PatientSupportHashError("patient training arrays differ from calibration")
    if patient.contact_sha256 != calibration.contact_sha256:
        raise PatientSupportHashError("patient contact contract differs from calibration")
    if patient.declared_source_sha256 != calibration.declared_source_sha256:
        raise PatientSupportHashError("patient source hash differs from calibration")
    score_draws = (
        int(calibration.joint_inner_draws) if score_draws is None else int(score_draws)
    )
    if score_draws <= 0:
        raise PatientSupportSchemaError("score_draws must be positive")
    if score_draws != int(calibration.joint_inner_draws):
        raise PatientSupportSchemaError(
            "score_draws must equal the joint calibration inner draw count"
        )

    onsets = _reorder_model_contacts(model, patient)
    forced_ood = np.asarray(model.ood, dtype=bool) | ~np.isfinite(onsets).any(axis=1)
    rng = np.random.default_rng(int(seed))
    raw_draws: dict[str, np.ndarray] = {
        f"mode_{mode}.{key}": np.empty(int(score_draws), dtype=np.float64)
        for mode in (0, 1) for key in MODE_ENDPOINTS
    }
    raw_draws[OOD_ENDPOINT] = np.empty(int(score_draws), dtype=np.float64)
    padding = {"mode_0": 0, "mode_1": 0, OOD_ENDPOINT: 0}
    for draw in range(int(score_draws)):
        for mode in (0, 1):
            # OOD is a support diagnosis, not permission to delete contact loss.
            eligible = np.flatnonzero(model.classifier_labels == mode)
            model_sample, missing = _draw_model_mode_with_missing(
                onsets, eligible, size=calibration.sample_size, rng=rng,
            )
            padding[f"mode_{mode}"] += missing
            patient_indices = _draw_patient_reference(
                patient, mode=mode, size=calibration.sample_size, rng=rng,
            )
            distances = mode_endpoint_distances(
                model_sample, patient.onsets[patient_indices], patient.shaft_ids,
                tie_tolerance=tie_tolerance,
            )
            for key in MODE_ENDPOINTS:
                raw_draws[f"mode_{mode}.{key}"][draw] = distances[key]

        model_ood, missing = _draw_model_ood_with_missing(
            forced_ood, size=2 * calibration.sample_size, rng=rng,
        )
        padding[OOD_ENDPOINT] += missing
        patient_indices = _draw_patient_reference(
            patient, mode=None, size=2 * calibration.sample_size, rng=rng,
        )
        raw_draws[OOD_ENDPOINT][draw] = abs(
            float(model_ood.mean()) - float(patient.ood[patient_indices].mean())
        )

    endpoints: dict[str, SupportEndpointRecord] = {}
    for key, values in raw_draws.items():
        raw = float(np.mean(values))
        floor = float(calibration.floor_q95[key])
        endpoints[key] = {
            "raw_distance": raw,
            "floor_q95": floor,
            "ratio": _safe_ratio(raw, floor),
        }
    score = float(max(record["ratio"] for record in endpoints.values()))
    dominant = max(endpoints, key=lambda key: endpoints[key]["ratio"])
    return PatientSupportResult(
        status=(
            "PASS" if score <= float(calibration.joint_threshold_q95) else "FAIL"
        ),
        score=score,
        endpoints=endpoints,
        support={
            "n_contact_primary": int(len(onsets)),
            "n_ood_including_all_missing": int(np.sum(forced_ood)),
            "n_all_missing": int(np.sum(~np.isfinite(onsets).any(axis=1))),
            "n_classifier_A": int(np.sum(model.classifier_labels == 0)),
            "n_classifier_B": int(np.sum(model.classifier_labels == 1)),
            "n_in_support_classifier_A": int(np.sum(
                (model.classifier_labels == 0) & ~forced_ood
            )),
            "n_in_support_classifier_B": int(np.sum(
                (model.classifier_labels == 1) & ~forced_ood
            )),
            "padding_rows_across_draws": padding,
            "all_contact_primary_rows_retained": True,
            "dominant_endpoint": dominant,
            "joint_statistic": score,
            "joint_threshold_q95": float(calibration.joint_threshold_q95),
            "joint_margin": float(calibration.joint_threshold_q95) - score,
        },
        sampling={
            "score_draws": int(score_draws),
            "seed": int(seed),
            "mode_sample_size": int(calibration.sample_size),
            "ood_sample_size": int(2 * calibration.sample_size),
            "model_sampling": "unique_without_replacement_then_all-missing_padding",
            "patient_sampling": "one_event_per_distinct_training_recording_block",
            "primary_patient_labels": PRIMARY_LABEL_KEY,
            "shaft_aware_k2_role": "diagnostic_only",
            "acceptance_rule": "joint_statistic <= joint_threshold_q95",
            "endpoint_ratios_role": "descriptive_components_not_independent_gates",
        },
    )


def evaluate_patient_support(
        patient: PatientTrainingData, calibration: PatientSupportCalibration,
        model: ModelContactPrimaryData, **kwargs: Any) -> PatientSupportResult:
    """Fail closed only for schema/hash errors; ordinary weak support is FAIL."""
    try:
        return score_patient_support(patient, calibration, model, **kwargs)
    except PatientSupportSchemaError as error:
        return PatientSupportResult(
            status="INVALID",
            score=None,
            endpoints={},
            support={},
            sampling={},
            invalid_reason=str(error),
        )
