#!/usr/bin/env python3
"""Freeze the rev14 interictal patient held-out endpoint without scoring a model.

The freezer reconstructs the rev10 recording-block split from raw interictal
artifacts, applies the train-only rev11 old-A/B direction classifier, and writes
one sealed held-out NPZ plus its manifest.  It never runs the SNN and it does not
read ictal data.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.interictal_propagation import load_subject_propagation_events  # noqa: E402
from src.topic4_core_field_profile import (  # noqa: E402
    MIN_PARTICIPANTS,
    normalized_rank_curve,
    profile_grid,
    split_by_block,
)
from src.topic4_node_dualmode import (  # noqa: E402
    calibrate_component_scales,
    fixed_projection_matrix,
)
from src.topic4_shaft_aware import contract_groups  # noqa: E402
from src.topic4_shaft_aware_direction import assign_direction_modes  # noqa: E402
from src.topic4_shaft_aware_direction import fit_direction_classifier  # noqa: E402


DEFAULT_ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_SPLIT_CONFIG = ROOT / "config/topic4_rev10_sa_shaft_aware.json"
DEFAULT_TRAINING_TARGET = Path(
    "results/topic4_sef_hfo/data_driven_core_field_rev10_sa/shaft_aware_target/"
    "shaft_aware_patient_training_target.npz"
)
DEFAULT_CONTACT_CONTRACT = Path(
    "results/topic4_sef_hfo/data_driven_core_field_rev10_sa/shaft_aware_target/"
    "contact_shaft_contract.json"
)
DEFAULT_CLASSIFIER_MANIFEST = Path(
    "results/topic4_sef_hfo/data_driven_local_connectivity_rev11_nlc/"
    "frozen_substrate_confirmation/candidate_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev14/"
    "patient_heldout_endpoint"
)
MANIFEST_NAME = "patient_heldout_endpoint_manifest_v1.json"
NPZ_NAME = "patient_heldout_endpoint_v1.npz"


CANONICAL_EXPECTED_SHA256 = {
    "split_config": "88ea711352088e32dd1d4f361dced9e8d1bd4153fc08df584c4ef02ca6a0fb57",
    "training_target": "7d11100c0f431479f8decb17b6de97575a91dcb889887ce9e2cedbbdcd54140b",
    "contact_contract": "548643b6e89b61408b8bdf4cbc1566133937204dc36e8d512049cb25762fde91",
    "classifier_manifest": "545b029d2d7947de5a27979e7166f6bc55ea2b01c64a67a9b736977ec960fcbb",
    "old_rank_curve_reference": "987e6b4bdfdc9b3d31485852e2eb71cd8df3ee7d92ff6eb0456c8fa04925adc1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    header = _canonical_json_bytes({
        "dtype": array.dtype.str,
        "shape": list(array.shape),
    })
    return hashlib.sha256(header + array.view(np.uint8).tobytes()).hexdigest()


def _resolve(path: Path | str, artifact_root: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (artifact_root / value).resolve()


def _file_record(path: Path, *, role: str, root: Path | None = None) -> dict:
    path = path.resolve()
    display = str(path)
    if root is not None:
        try:
            display = str(path.relative_to(root.resolve()))
        except ValueError:
            pass
    return {
        "role": str(role),
        "path": display,
        "absolute_path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _git_output(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _runtime_provenance(paths: list[Path]) -> dict:
    relative = [str(path.resolve().relative_to(ROOT)) for path in paths]
    dirty = _git_output("status", "--porcelain", "--", *relative)
    return {
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_branch": _git_output("branch", "--show-current"),
        "runtime_paths_dirty": bool(dirty),
        "runtime_dirty_porcelain": dirty.splitlines(),
        "runtime_path_sha256": {
            str(path.resolve()): _sha256(path.resolve()) for path in paths
        },
    }


def _raw_source_paths(patient_root: Path) -> list[tuple[str, Path]]:
    patient_root = patient_root.resolve()
    lagpat = sorted(patient_root.glob("*_lagPat_withFreqCent.npz"))
    suffix = "_lagPat_withFreqCent.npz"
    packed_suffix = "_packedTimes_withFreqCent.npy"
    if not lagpat:
        lagpat = sorted(patient_root.glob("*_lagPat.npz"))
        suffix = "_lagPat.npz"
        packed_suffix = "_packedTimes.npy"
    if not lagpat:
        raise FileNotFoundError(f"no interictal lagPat files in {patient_root}")
    output: list[tuple[str, Path]] = []
    for lag_path in lagpat:
        record = lag_path.name[: -len(suffix)]
        packed = lag_path.with_name(f"{record}{packed_suffix}")
        if not packed.exists():
            raise RuntimeError(f"held-out source lacks packed event times: {packed}")
        output.extend((
            ("interictal_lagpat", lag_path.resolve()),
            ("interictal_packed_times", packed.resolve()),
        ))
    return output


def _source_snapshot(
        *, split_config: Path, training_target: Path, contact_contract: Path,
        classifier_manifest: Path, old_reference: Path,
        patient_root: Path) -> dict:
    fixed = [
        _file_record(split_config, role="split_config"),
        _file_record(training_target, role="patient_training_target"),
        _file_record(contact_contract, role="contact_contract"),
        _file_record(classifier_manifest, role="train_only_direction_classifier"),
        _file_record(old_reference, role="old_rank_curve_reference"),
    ]
    raw = [
        _file_record(path, role=role, root=patient_root)
        for role, path in _raw_source_paths(patient_root)
    ]
    records = fixed + raw
    return {
        "files": records,
        "source_set_sha256": _canonical_sha256(records),
        "n_files": len(records),
    }


def _verify_expected_hashes(snapshot: Mapping, expected: Mapping[str, str]) -> None:
    by_role = {row["role"]: row for row in snapshot["files"]}
    role_map = {
        "split_config": "split_config",
        "training_target": "patient_training_target",
        "contact_contract": "contact_contract",
        "classifier_manifest": "train_only_direction_classifier",
        "old_rank_curve_reference": "old_rank_curve_reference",
    }
    for key, expected_hash in expected.items():
        if key not in role_map:
            raise ValueError(f"unknown expected input hash key: {key}")
        observed = by_role[role_map[key]]["sha256"]
        if observed != str(expected_hash):
            raise RuntimeError(
                f"frozen {key} hash changed: expected {expected_hash}, got {observed}"
            )


def _assert_existing_sources_unchanged(manifest_path: Path,
                                       current_snapshot: Mapping) -> None:
    if not manifest_path.exists():
        return
    existing = json.loads(manifest_path.read_text())
    frozen = existing.get("source_snapshot")
    if frozen is None:
        raise RuntimeError("existing held-out manifest lacks source_snapshot")
    if frozen != current_snapshot:
        raise RuntimeError("held-out source drift detected; frozen endpoint not overwritten")


def _load_training_target(path: Path) -> dict:
    required = (
        "contact_names", "shaft_ids", "sheet_xy_mm",
        "shared_axis_coordinate_mm", "patient_train_event_indices",
        "patient_train_block_ids", "patient_train_old_labels",
        "patient_train_onsets",
        "feature_center", "feature_scale", "pca_components",
    )
    with np.load(path, allow_pickle=False) as loaded:
        missing = set(required).difference(loaded.files)
        if missing:
            raise RuntimeError(f"training target lacks keys: {sorted(missing)}")
        return {key: np.asarray(loaded[key]) for key in required}


def _load_classifier(path: Path) -> dict:
    payload = json.loads(path.read_text())
    classifier = payload.get("direction_classifier")
    if not isinstance(classifier, dict):
        raise RuntimeError("classifier manifest lacks direction_classifier")
    required = (
        "coef", "intercept", "class_centers", "class_precisions",
        "ood_distance_thresholds", "ood_quantile", "regularization_c", "n_train",
        "folds",
    )
    missing = set(required).difference(classifier)
    if missing:
        raise RuntimeError(f"direction classifier lacks keys: {sorted(missing)}")
    output = {key: classifier[key] for key in required}
    for key in ("coef", "class_centers", "class_precisions", "ood_distance_thresholds"):
        output[key] = np.asarray(output[key], dtype=np.float64)
    return output


def _verify_train_only_classifier(training: Mapping, contract: Mapping,
                                  classifier: Mapping) -> None:
    embedding = {
        "center": np.asarray(training["feature_center"], dtype=np.float64),
        "scale": np.asarray(training["feature_scale"], dtype=np.float64),
        "components": np.asarray(training["pca_components"], dtype=np.float64),
    }
    rebuilt = fit_direction_classifier(
        np.asarray(training["patient_train_onsets"], dtype=np.float64),
        np.asarray(training["patient_train_old_labels"], dtype=np.int64),
        np.asarray(training["patient_train_block_ids"]),
        groups=contract_groups(contract), embedding=embedding,
        n_splits=len(classifier["folds"]),
        regularization_c=float(classifier["regularization_c"]),
        ood_quantile=float(classifier["ood_quantile"]),
    )
    for key in ("coef", "class_centers", "class_precisions",
                "ood_distance_thresholds"):
        if not np.allclose(
                np.asarray(classifier[key], dtype=np.float64),
                np.asarray(rebuilt[key], dtype=np.float64),
                rtol=0.0, atol=1e-12):
            raise RuntimeError(
                f"direction classifier is not reproducible from frozen training blocks: {key}"
            )
    if not np.isclose(
            float(classifier["intercept"]), float(rebuilt["intercept"]),
            rtol=0.0, atol=1e-12):
        raise RuntimeError(
            "direction classifier is not reproducible from frozen training blocks: intercept"
        )


def _heldout_table(raw: Mapping, contract: Mapping, *,
                   split_fraction: float, split_seed: int,
                   training: Mapping, classifier: Mapping) -> dict:
    names = np.asarray([row["contact_name"] for row in contract["contacts"]]).astype(str)
    train_names = np.asarray(training["contact_names"]).astype(str)
    if not np.array_equal(names, train_names):
        raise RuntimeError("training target and contact contract order differ")
    raw_names = np.asarray(raw["channel_names"]).astype(str)
    if set(raw_names) != set(names):
        raise RuntimeError("raw interictal contacts differ from frozen contact contract")
    lookup = {name: index for index, name in enumerate(raw_names)}
    reorder = np.asarray([lookup[name] for name in names], dtype=int)

    block_ids = np.asarray(raw["block_ids"])
    train_all, heldout_all = split_by_block(
        block_ids, float(split_fraction), int(split_seed),
    )
    train_blocks = np.unique(block_ids[train_all])
    heldout_blocks = np.unique(block_ids[heldout_all])
    if np.intersect1d(train_blocks, heldout_blocks).size:
        raise RuntimeError("recording-block split leaked")
    frozen_training_blocks = np.unique(np.asarray(training["patient_train_block_ids"]))
    if not np.array_equal(np.sort(frozen_training_blocks), np.sort(train_blocks)):
        raise RuntimeError("reconstructed train blocks differ from frozen training target")
    frozen_training_events = np.asarray(
        training["patient_train_event_indices"], dtype=np.int64,
    )
    frozen_training_event_blocks = np.asarray(training["patient_train_block_ids"])
    if frozen_training_event_blocks.shape != frozen_training_events.shape:
        raise RuntimeError("frozen training event and block arrays differ in length")

    axial = {
        row["contact_name"]: float(row["shared_axis_coordinate_mm"])
        for row in contract["contacts"]
    }
    grid = profile_grid(axial)
    raw_ranks = np.asarray(raw["ranks"], dtype=np.float64)
    raw_onsets = np.asarray(raw["lag_raw"], dtype=np.float64)
    raw_mask = np.asarray(raw["bools"], dtype=bool)
    old_label_part_min = int(MIN_PARTICIPANTS)
    readout_contract_part_min = int(
        contract.get("readout_parameters", {}).get("readability_part_min", 5)
    )
    old_curve_readable = np.zeros(len(block_ids), dtype=bool)
    readout_contract_readable = np.zeros(len(block_ids), dtype=bool)
    for event_index in range(len(block_ids)):
        participating = np.flatnonzero(raw_mask[:, event_index])
        rank_dict = {
            raw_names[index]: float(raw_ranks[index, event_index])
            for index in participating
        }
        old_curve_readable[event_index] = normalized_rank_curve(
            rank_dict, axial, grid=grid, part_min=old_label_part_min,
        ) is not None
        readout_contract_readable[event_index] = normalized_rank_curve(
            rank_dict, axial, grid=grid, part_min=readout_contract_part_min,
        ) is not None
    reconstructed_training_events = train_all[old_curve_readable[train_all]]
    if not np.array_equal(frozen_training_events, reconstructed_training_events):
        raise RuntimeError(
            "frozen training events differ from reconstructed readable training events"
        )
    if not np.array_equal(
            frozen_training_event_blocks, block_ids[frozen_training_events]):
        raise RuntimeError(
            "frozen training event block IDs differ from reconstructed recording blocks"
        )

    event_indices: list[int] = []
    onsets: list[np.ndarray] = []
    ranks: list[np.ndarray] = []
    for event_index in heldout_all:
        onset = raw_onsets[:, event_index].copy()
        onset[~raw_mask[:, event_index]] = np.nan
        rank = raw_ranks[:, event_index].copy()
        rank[~raw_mask[:, event_index]] = np.nan
        event_indices.append(int(event_index))
        onsets.append(onset[reorder])
        ranks.append(rank[reorder])
    if not event_indices:
        raise RuntimeError("held-out split contains no events")
    event_indices_array = np.asarray(event_indices, dtype=np.int64)
    if np.intersect1d(event_indices_array, frozen_training_events).size:
        raise RuntimeError("held-out output contains frozen training events")
    heldout_event_blocks = block_ids[event_indices_array]
    if np.intersect1d(heldout_event_blocks, frozen_training_blocks).size:
        raise RuntimeError("held-out output contains a training recording block")

    onsets_array = np.asarray(onsets, dtype=np.float64)
    ranks_array = np.asarray(ranks, dtype=np.float64)
    groups = contract_groups(contract)
    embedding = {
        "center": np.asarray(training["feature_center"], dtype=np.float64),
        "scale": np.asarray(training["feature_scale"], dtype=np.float64),
        "components": np.asarray(training["pca_components"], dtype=np.float64),
    }
    assigned = assign_direction_modes(
        onsets_array, groups=groups, embedding=embedding, classifier=classifier,
    )
    labels = np.asarray(assigned["labels"], dtype=np.int8)
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise RuntimeError("frozen train-only classifier assigns only one held-out mode")
    boundaries = {int(row["block_id"]): str(row["record_name"])
                  for row in raw["block_boundaries"]}
    record_names = np.asarray([
        boundaries[int(block)] for block in heldout_event_blocks
    ])
    return {
        "contact_names": names,
        "event_indices": event_indices_array,
        "block_ids": np.asarray(heldout_event_blocks),
        "record_names": record_names,
        "onsets": onsets_array,
        "ranks": ranks_array,
        "labels": labels,
        "probability_b": np.asarray(assigned["probability_B"], dtype=np.float64),
        "ood_distance": np.asarray(assigned["ood_distance"], dtype=np.float64),
        "ood": np.asarray(assigned["ood"], dtype=bool),
        "old_curve_readable": old_curve_readable[event_indices_array],
        "readout_contract_readable": readout_contract_readable[event_indices_array],
        "event_abs_times": np.asarray(raw["event_abs_times"])[event_indices_array],
        "event_abs_end_times": np.asarray(raw["event_abs_end_times"])[event_indices_array],
        "train_blocks": np.asarray(train_blocks),
        "heldout_blocks": np.asarray(heldout_blocks),
        "old_label_part_min": old_label_part_min,
        "readout_contract_part_min": readout_contract_part_min,
    }


def _evaluation_contract(*, sample_size: int, draws: int, projection_count: int,
                         projection_seed: int, calibration_draws: int,
                         calibration_seed: int) -> dict:
    return {
        "schema_id": "topic4_rev14_patient_heldout_evaluation_v1",
        "patient_endpoint": "interictal_recording_block_heldout_only",
        "patient_mode_identity": (
            "frozen old A/B direction labels assigned by the training-only "
            "FULL_TIMING shaft-aware logistic classifier; no held-out refit"
        ),
        "four_layer_mode_distance": {
            "components": ["recruitment", "precedence", "profile", "cloud"],
            "component_mean": "unweighted arithmetic mean after floor normalization",
            "recruitment": "ICL/SCL shaft-balanced recruitment-probability MAE",
            "precedence": (
                "mean pair-class-balanced Jensen-Shannon divergence over "
                "ICL-ICL, SCL-SCL and ICL-SCL, including not-jointly-recruited"
            ),
            "profile": "shaft-balanced RMS error of recruitment plus normalized rank",
            "cloud": "fixed-projection sliced Wasserstein on the same event features",
            "normalization": "raw distance divided by held-out cross-block floor q95",
            "weakest_mode": "LSE_tau_0.25(mode_A_mean, mode_B_mean)",
        },
        "sampling": {
            "sample_size_per_side": int(sample_size),
            "draws_per_network": int(draws),
            "patient_sampling": (
                "without replacement within one held-out recording block and mode"
            ),
            "model_sampling": (
                "confidence-weighted without replacement, then explicit all-contact "
                "missing rows"
            ),
            "projection_count": int(projection_count),
            "projection_seed": int(projection_seed),
            "calibration_draws": int(calibration_draws),
            "calibration_seed": int(calibration_seed),
        },
        "paired_comparison": {
            "unit": "network seed with common random numbers",
            "reference": "exact_off",
            "weakest_mode_rule": "candidate_weakest_mode_lse < exact_off_weakest_mode_lse",
            "other_mode_index": "argmin(exact_off_mode_A_mean, exact_off_mode_B_mean)",
            "non_worsening_rule": (
                "candidate_other_mode_mean <= 1.10 * exact_off_other_mode_mean"
            ),
            "candidate_replacement_after_failure": False,
        },
        "selection_boundary": (
            "opened once after model-internal winner freeze; no score may return to fitting"
        ),
        "implementation": {
            "objective_module": "src/topic4_rev14_static_node_objective.py",
            "distance_module": "src/topic4_node_dualmode.py",
        },
    }


def _deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
            output, "w", compression=zipfile.ZIP_DEFLATED,
            compresslevel=9) as archive:
        for key in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer, np.asarray(arrays[key]), allow_pickle=False,
            )
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED,
                             compresslevel=9)
    return output.getvalue()


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz.tmp")
    os.close(handle)
    try:
        Path(temporary).write_bytes(_deterministic_npz_bytes(arrays))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(handle)
    try:
        Path(temporary).write_bytes(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
            + b"\n"
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def freeze_endpoint(
        *, artifact_root: Path, split_config_path: Path,
        training_target_path: Path, contact_contract_path: Path,
        classifier_manifest_path: Path, output_dir: Path,
        expected_hashes: Mapping[str, str] | None = CANONICAL_EXPECTED_SHA256,
        evaluation_overrides: Mapping[str, int] | None = None) -> dict:
    artifact_root = artifact_root.resolve()
    split_config_path = _resolve(split_config_path, artifact_root)
    training_target_path = _resolve(training_target_path, artifact_root)
    contact_contract_path = _resolve(contact_contract_path, artifact_root)
    classifier_manifest_path = _resolve(classifier_manifest_path, artifact_root)
    output_dir = _resolve(output_dir, artifact_root)
    split_config = json.loads(split_config_path.read_text())
    if split_config.get("subject") != "epilepsiae_1146":
        raise RuntimeError("rev14 held-out freezer is locked to epilepsiae_1146")
    split = split_config.get("patient_split", {})
    if split.get("unit") != "recording_block":
        raise RuntimeError("held-out split unit must remain recording_block")
    patient_root = _resolve(split_config["inputs"]["patient_root"], artifact_root)
    old_reference = _resolve(
        split_config["inputs"]["old_rank_curve_reference"], artifact_root,
    )
    snapshot = _source_snapshot(
        split_config=split_config_path, training_target=training_target_path,
        contact_contract=contact_contract_path,
        classifier_manifest=classifier_manifest_path,
        old_reference=old_reference, patient_root=patient_root,
    )
    if expected_hashes is not None:
        _verify_expected_hashes(snapshot, expected_hashes)

    manifest_path = output_dir / MANIFEST_NAME
    npz_path = output_dir / NPZ_NAME
    if manifest_path.exists() != npz_path.exists():
        raise RuntimeError("held-out freeze is partial: manifest/NPZ presence differs")
    _assert_existing_sources_unchanged(manifest_path, snapshot)

    training = _load_training_target(training_target_path)
    contract = json.loads(contact_contract_path.read_text())
    classifier = _load_classifier(classifier_manifest_path)
    if int(classifier["n_train"]) != len(training["patient_train_event_indices"]):
        raise RuntimeError("direction classifier training count differs from training target")
    with np.load(old_reference, allow_pickle=False) as reference:
        old_grid = np.asarray(reference["grid"], dtype=np.float64)
    axial = {
        row["contact_name"]: float(row["shared_axis_coordinate_mm"])
        for row in contract["contacts"]
    }
    if not np.allclose(profile_grid(axial), old_grid, rtol=0.0, atol=1e-12):
        raise RuntimeError("old A/B rank grid differs from the contact contract")

    raw = load_subject_propagation_events(patient_root)
    _verify_train_only_classifier(training, contract, classifier)
    heldout = _heldout_table(
        raw, contract, split_fraction=float(split["heldout_fraction"]),
        split_seed=int(split["seed"]), training=training, classifier=classifier,
    )

    evaluation_values = {
        "sample_size": 6,
        "draws": 64,
        "projection_count": 64,
        "projection_seed": 20260825,
        "calibration_draws": 256,
        "calibration_seed": 20260828,
    }
    if evaluation_overrides:
        unknown = set(evaluation_overrides).difference(evaluation_values)
        if unknown:
            raise ValueError(f"unknown evaluation override: {sorted(unknown)}")
        evaluation_values.update({key: int(value) for key, value in evaluation_overrides.items()})
    evaluation = _evaluation_contract(**evaluation_values)
    evaluation_hash = _canonical_sha256(evaluation)
    projections = fixed_projection_matrix(
        2 * len(heldout["contact_names"]),
        n_directions=evaluation_values["projection_count"],
        seed=evaluation_values["projection_seed"],
    )
    calibration = calibrate_component_scales(
        heldout["ranks"], heldout["labels"], heldout["block_ids"],
        heldout["contact_names"], projections,
        sample_size=evaluation_values["sample_size"],
        draws=evaluation_values["calibration_draws"],
        seed=evaluation_values["calibration_seed"],
    )
    calibration["global_null_resampling"] = (
        "pooled patient held-out events across frozen old A/B modes"
    )
    calibration["endpoint_split"] = "patient_interictal_heldout_recording_blocks"

    arrays = {
        "contact_names": np.asarray(heldout["contact_names"]),
        "shaft_ids": np.asarray(training["shaft_ids"]),
        "sheet_xy_mm": np.asarray(training["sheet_xy_mm"], dtype=np.float64),
        "shared_axis_coordinate_mm": np.asarray(
            training["shared_axis_coordinate_mm"], dtype=np.float64,
        ),
        "heldout_event_indices": heldout["event_indices"],
        "heldout_block_ids": heldout["block_ids"],
        "heldout_record_names": heldout["record_names"],
        "heldout_event_abs_times": np.asarray(
            heldout["event_abs_times"], dtype=np.float64,
        ),
        "heldout_event_abs_end_times": np.asarray(
            heldout["event_abs_end_times"], dtype=np.float64,
        ),
        "heldout_onsets": np.asarray(heldout["onsets"], dtype=np.float32),
        "heldout_ranks": np.asarray(heldout["ranks"], dtype=np.float32),
        "heldout_old_labels": heldout["labels"],
        "heldout_probability_B": np.asarray(
            heldout["probability_b"], dtype=np.float32,
        ),
        "heldout_ood_distance": np.asarray(
            heldout["ood_distance"], dtype=np.float32,
        ),
        "heldout_ood": heldout["ood"],
        "heldout_old_curve_readable": heldout["old_curve_readable"],
        "heldout_readout_contract_readable": heldout[
            "readout_contract_readable"
        ],
        "heldout_finite_contact_count": np.sum(
            np.isfinite(heldout["ranks"]), axis=1,
        ).astype(np.int16),
    }
    npz_bytes = _deterministic_npz_bytes(arrays)
    npz_sha = hashlib.sha256(npz_bytes).hexdigest()
    array_hashes = {key: _array_sha256(value) for key, value in arrays.items()}

    contact_names = heldout["contact_names"].tolist()
    heldout_blocks = np.sort(np.unique(heldout["heldout_blocks"]))
    train_blocks = np.sort(np.unique(heldout["train_blocks"]))
    runtime_paths = [
        Path(__file__).resolve(),
        ROOT / "src/interictal_propagation.py",
        ROOT / "src/topic4_core_field_profile.py",
        ROOT / "src/topic4_node_dualmode.py",
        ROOT / "src/topic4_shaft_aware.py",
        ROOT / "src/topic4_shaft_aware_direction.py",
        ROOT / "src/topic4_rev14_static_node_objective.py",
    ]
    manifest = {
        "schema_id": "topic4_rev14_patient_heldout_endpoint_manifest_v1",
        "status": "PATIENT_INTERICTAL_HELDOUT_ENDPOINT_FROZEN",
        "subject": split_config["subject"],
        "claim_boundary": (
            "Development-only interictal held-out endpoint. No patient ictal data, "
            "SNN simulation, candidate score or candidate selection is present."
        ),
        "training_runner_dependency": {
            "required": False,
            "manifest_name": MANIFEST_NAME,
            "npz_name": NPZ_NAME,
            "rule": "training and selection runners must not accept either path",
        },
        "source_snapshot": snapshot,
        "split_contract": {
            "unit": "recording_block",
            "heldout_fraction": float(split["heldout_fraction"]),
            "seed": int(split["seed"]),
            "implementation": "src.topic4_core_field_profile.split_by_block",
            "train_block_ids": train_blocks.tolist(),
            "heldout_block_ids": heldout_blocks.tolist(),
            "train_block_ids_sha256": _array_sha256(train_blocks),
            "heldout_block_ids_sha256": _array_sha256(heldout_blocks),
            "blocks_disjoint": bool(not np.intersect1d(train_blocks, heldout_blocks).size),
        },
        "contact_contract": {
            "contact_order": contact_names,
            "contact_order_sha256": _canonical_sha256(contact_names),
            "n_contacts": len(contact_names),
            "contract_hashes": contract.get("hashes", {}),
        },
        "old_A_B_label_definition": {
            "semantics": "original_Fig4_A_B_propagation_direction",
            "feature_semantics": "FULL_TIMING_shaft_aware",
            "classifier_fit_data": "patient training recording blocks only",
            "heldout_classifier_refit": False,
            "threshold": "probability_B >= 0.5",
            "readability": (
                "original producer normalized_rank_curve diagnostic only; every "
                "held-out event remains in the endpoint"
            ),
            "readability_part_min": int(heldout["old_label_part_min"]),
            "readout_contract_part_min": int(
                heldout["readout_contract_part_min"]
            ),
            "classifier_parameter_hashes": {
                key: _array_sha256(classifier[key])
                for key in (
                    "coef", "class_centers", "class_precisions",
                    "ood_distance_thresholds",
                )
            },
        },
        "evaluation_contract": evaluation,
        "evaluation_schema_sha256": evaluation_hash,
        "heldout_calibration": calibration,
        "endpoint_counts": {
            "n_events": int(len(heldout["event_indices"])),
            "n_old_curve_readable": int(np.sum(heldout["old_curve_readable"])),
            "n_old_curve_unreadable": int(np.sum(~heldout["old_curve_readable"])),
            "n_readout_contract_readable": int(np.sum(
                heldout["readout_contract_readable"]
            )),
            "mode_counts": np.bincount(heldout["labels"], minlength=2).tolist(),
            "n_blocks": int(len(heldout_blocks)),
            "ood_fraction": float(np.mean(heldout["ood"])),
        },
        "disjointness_audit": {
            "training_event_overlap": 0,
            "training_block_overlap": 0,
            "output_contains_training_event_arrays": False,
        },
        "artifact": {
            "npz_name": NPZ_NAME,
            "npz_sha256": npz_sha,
            "array_sha256": array_hashes,
            "array_keys": sorted(arrays),
        },
        "runtime_sources": [
            _file_record(path, role="runtime_source", root=ROOT)
            for path in runtime_paths
        ],
        "provenance": _runtime_provenance(runtime_paths),
    }

    current_snapshot = _source_snapshot(
        split_config=split_config_path, training_target=training_target_path,
        contact_contract=contact_contract_path,
        classifier_manifest=classifier_manifest_path,
        old_reference=old_reference, patient_root=patient_root,
    )
    if current_snapshot != snapshot:
        raise RuntimeError("held-out source drift detected during endpoint construction")

    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing != manifest or _sha256(npz_path) != npz_sha:
            raise RuntimeError("deterministic held-out rebuild differs from frozen artifacts")
        return existing

    if output_dir.exists():
        raise RuntimeError("new held-out output directory is not empty/frozen")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(
        dir=output_dir.parent, prefix=f".{output_dir.name}.staging.",
    ))
    try:
        staged_npz = staging_dir / NPZ_NAME
        staged_manifest = staging_dir / MANIFEST_NAME
        staged_npz.write_bytes(npz_bytes)
        _atomic_json(staged_manifest, manifest)
        if _sha256(staged_npz) != manifest["artifact"]["npz_sha256"]:
            raise RuntimeError("staged held-out NPZ hash differs from manifest")
        final_snapshot = _source_snapshot(
            split_config=split_config_path, training_target=training_target_path,
            contact_contract=contact_contract_path,
            classifier_manifest=classifier_manifest_path,
            old_reference=old_reference, patient_root=patient_root,
        )
        if final_snapshot != snapshot:
            raise RuntimeError(
                "held-out source drift detected before atomic endpoint publication"
            )
        os.replace(staging_dir, output_dir)
    finally:
        if staging_dir.exists():
            for child in staging_dir.iterdir():
                child.unlink()
            staging_dir.rmdir()
    if _sha256(npz_path) != manifest["artifact"]["npz_sha256"]:
        raise RuntimeError("written held-out NPZ hash differs from manifest")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG)
    parser.add_argument("--training-target", type=Path, default=DEFAULT_TRAINING_TARGET)
    parser.add_argument("--contact-contract", type=Path, default=DEFAULT_CONTACT_CONTRACT)
    parser.add_argument(
        "--classifier-manifest", type=Path, default=DEFAULT_CLASSIFIER_MANIFEST,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    manifest = freeze_endpoint(
        artifact_root=args.artifact_root,
        split_config_path=args.split_config,
        training_target_path=args.training_target,
        contact_contract_path=args.contact_contract,
        classifier_manifest_path=args.classifier_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps({
        "status": manifest["status"],
        "n_events": manifest["endpoint_counts"]["n_events"],
        "mode_counts": manifest["endpoint_counts"]["mode_counts"],
        "manifest": str(_resolve(args.output_dir, args.artifact_root) / MANIFEST_NAME),
        "npz": str(_resolve(args.output_dir, args.artifact_root) / NPZ_NAME),
    }, indent=2))


if __name__ == "__main__":
    main()
