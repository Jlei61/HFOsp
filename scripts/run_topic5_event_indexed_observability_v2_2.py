#!/usr/bin/env python3
"""Run the frozen v2.2 non-parametric observability audit on the six pilots."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_event_indexed_evolving_rank_field import (  # noqa: E402
    EventBlock,
    block_matrix,
    block_reliability,
    center_within_source,
    dynamic_observability_audit,
    global_rank_prior,
    make_equal_event_blocks,
    pca_reconstruction_gain,
    permute_block_events_within_source,
    stratified_uniform_subsample_blocks,
    stratified_uniform_subsample_blocks_with_adjacency,
    subspace_similarity,
    within_source_pairs,
)


DEFAULT_CONFIG = ROOT / "config/topic5_event_indexed_evolving_rank_field_v2_2.yaml"
DEFAULT_OUTPUT = (
    ROOT
    / "results/topic5_event_indexed_evolving_rank_field/development/phase0_observability"
)
INPUT_AUDIT = (
    ROOT
    / "results/topic5_event_indexed_evolving_rank_field/development/input_audit"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_subject(subject: str, dataset_root: Path) -> dict[str, Any]:
    dataset_path = dataset_root / "per_subject" / f"{subject}.npz"
    mapping_path = INPUT_AUDIT / "per_subject" / f"{subject}.npz"
    audit_path = INPUT_AUDIT / "per_subject" / f"{subject}.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise RuntimeError(f"{subject}: input audit did not pass")
    if audit.get("source_dataset_npz_sha256") != sha256(dataset_path):
        raise RuntimeError(f"{subject}: input audit dataset hash drift")
    with np.load(dataset_path, allow_pickle=False) as data:
        values = {
            "local_rank": np.asarray(data["event_local_rank"], np.float32),
            "group_ids": np.asarray(data["event_group_ids"], np.int16),
            "participation": np.asarray(data["event_participation"], bool),
            "time": np.asarray(data["event_abs_time"], np.float64),
            "split": np.asarray(data["event_split"], np.uint8),
            "contact_names": [str(value) for value in data["contact_names"]],
        }
    with np.load(mapping_path, allow_pickle=False) as mapping:
        if str(mapping["source_dataset_npz_sha256"]) != sha256(dataset_path):
            raise RuntimeError(f"{subject}: block mapping source hash drift")
        values["source_block"] = np.asarray(
            mapping["event_source_block_id"], np.int32
        )
    values.update(
        {
            "dataset_path": dataset_path,
            "dataset_sha256": sha256(dataset_path),
            "mapping_path": mapping_path,
            "mapping_sha256": sha256(mapping_path),
            "audit_path": audit_path,
            "audit_sha256": sha256(audit_path),
        }
    )
    return values


def _split_train80(split: np.ndarray, calibration_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    train = np.flatnonzero(np.asarray(split) == 0)
    cut = int(np.floor(float(calibration_fraction) * len(train)))
    if cut < 1 or cut >= len(train):
        raise ValueError("train80 calibration split is empty")
    return train[:cut], train[cut:]


def _candidate_reliability(
    values: dict[str, Any],
    calibration: np.ndarray,
    confirmation: np.ndarray,
    config: dict[str, Any],
    rank_prior: np.ndarray,
    seed: int,
) -> tuple[list[dict[str, Any]], int | None]:
    block_config = config["blocks"]
    rows = []
    selected = None
    for block_size in map(int, block_config["candidate_event_counts"]):
        calibration_blocks = make_equal_event_blocks(
            calibration, values["source_block"], block_size
        )
        confirmation_blocks = make_equal_event_blocks(
            confirmation, values["source_block"], block_size
        )
        reliability_blocks = stratified_uniform_subsample_blocks(
            calibration_blocks,
            int(block_config["max_calibration_blocks_for_reliability"]),
        )
        reliability = block_reliability(
            values["local_rank"],
            values["participation"],
            values["group_ids"],
            reliability_blocks,
            rank_prior=rank_prior,
            shrinkage_prior_events=float(block_config["shrinkage_prior_events"]),
            beta_prior=float(block_config["beta_prior"]),
            repeats=int(block_config["reliability_repeats"]),
            seed=seed + block_size,
        )
        adjacent = len(within_source_pairs(confirmation_blocks, adjacent_only=True))
        pass_reliability = bool(
            len(calibration_blocks) >= int(block_config["min_calibration_blocks"])
            and len(confirmation_blocks) >= int(block_config["min_confirmation_blocks"])
            and adjacent >= int(block_config["min_confirmation_adjacent_pairs"])
            and reliability["rank_spearman_median"]
            >= float(block_config["rank_reliability_threshold"])
            and reliability["participation_spearman_median"]
            >= float(block_config["participation_reliability_threshold"])
        )
        row = {
            "block_size": block_size,
            "n_calibration_blocks": len(calibration_blocks),
            "n_calibration_blocks_used_for_reliability": len(reliability_blocks),
            "n_confirmation_blocks": len(confirmation_blocks),
            "n_confirmation_adjacent_pairs": adjacent,
            **reliability,
            "reliability_pass": pass_reliability,
        }
        rows.append(row)
        if selected is None and pass_reliability:
            selected = block_size
    return rows, selected


def _chronological_calibration_split(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cut = int(np.floor(0.8 * len(matrix)))
    cut = min(max(cut, 2), len(matrix) - 2)
    return matrix[:cut], matrix[cut:]


def _low_rank_audit(
    values: dict[str, Any],
    calibration_blocks: list[EventBlock],
    confirmation_blocks: list[EventBlock],
    block_size: int,
    rank_prior: np.ndarray,
    config: dict[str, Any],
    seed: int,
    *,
    contact_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    blocks_cfg = config["blocks"]
    low_cfg = config["low_rank"]
    calibration_matrix, calibration_source = block_matrix(
        values["local_rank"],
        values["participation"],
        values["group_ids"],
        calibration_blocks,
        rank_prior=rank_prior,
        shrinkage_prior_events=float(blocks_cfg["shrinkage_prior_events"]),
        beta_prior=float(blocks_cfg["beta_prior"]),
        contact_mask=contact_mask,
    )
    confirmation_matrix, confirmation_source = block_matrix(
        values["local_rank"],
        values["participation"],
        values["group_ids"],
        confirmation_blocks,
        rank_prior=rank_prior,
        shrinkage_prior_events=float(blocks_cfg["shrinkage_prior_events"]),
        beta_prior=float(blocks_cfg["beta_prior"]),
        contact_mask=contact_mask,
    )
    calibration_residual = center_within_source(
        calibration_matrix, calibration_source
    )
    confirmation_residual = center_within_source(
        confirmation_matrix, confirmation_source
    )
    selection_train, selection_validation = _chronological_calibration_split(
        calibration_residual
    )
    candidate = {
        int(dimension): pca_reconstruction_gain(
            selection_train, selection_validation, int(dimension)
        )
        for dimension in low_cfg["dimensions"]
    }
    selected_dimension = min(
        (
            dimension
            for dimension, gain in candidate.items()
            if gain >= max(candidate.values()) - 0.01
        ),
        default=min(candidate, key=candidate.get),
    )
    heldout_gain = pca_reconstruction_gain(
        calibration_residual, confirmation_residual, selected_dimension
    )
    half = len(calibration_residual) // 2
    basis_stability = subspace_similarity(
        calibration_residual[:half],
        calibration_residual[half:],
        selected_dimension,
    )
    rng = np.random.default_rng(int(seed))
    null_gain = []
    null_stability = []
    for _ in range(int(low_cfg["permutation_draws"])):
        shuffled_calibration_blocks = permute_block_events_within_source(
            calibration_blocks, rng
        )
        shuffled_confirmation_blocks = permute_block_events_within_source(
            confirmation_blocks, rng
        )
        cal_matrix, cal_source = block_matrix(
            values["local_rank"], values["participation"], values["group_ids"],
            shuffled_calibration_blocks,
            rank_prior=rank_prior,
            shrinkage_prior_events=float(blocks_cfg["shrinkage_prior_events"]),
            beta_prior=float(blocks_cfg["beta_prior"]),
            contact_mask=contact_mask,
        )
        con_matrix, con_source = block_matrix(
            values["local_rank"], values["participation"], values["group_ids"],
            shuffled_confirmation_blocks,
            rank_prior=rank_prior,
            shrinkage_prior_events=float(blocks_cfg["shrinkage_prior_events"]),
            beta_prior=float(blocks_cfg["beta_prior"]),
            contact_mask=contact_mask,
        )
        cal_residual = center_within_source(cal_matrix, cal_source)
        con_residual = center_within_source(con_matrix, con_source)
        null_gain.append(
            pca_reconstruction_gain(cal_residual, con_residual, selected_dimension)
        )
        null_half = len(cal_residual) // 2
        null_stability.append(
            subspace_similarity(
                cal_residual[:null_half],
                cal_residual[null_half:],
                selected_dimension,
            )
        )
    null_gain_array = np.asarray(null_gain, float)
    null_stability_array = np.asarray(null_stability, float)
    gain_p = float(
        (1 + np.sum(null_gain_array >= heldout_gain)) / (1 + len(null_gain_array))
    )
    stability_p = float(
        (1 + np.sum(null_stability_array >= basis_stability))
        / (1 + len(null_stability_array))
    )
    return {
        "candidate_calibration_cv_gain": candidate,
        "selected_dimension": selected_dimension,
        "heldout_reconstruction_gain": heldout_gain,
        "heldout_gain_null_median": float(np.median(null_gain_array)),
        "heldout_gain_permutation_p": gain_p,
        "basis_split_stability": basis_stability,
        "basis_stability_null_median": float(np.median(null_stability_array)),
        "basis_stability_permutation_p": stability_p,
        "low_rank_gain_pass": bool(
            heldout_gain >= float(low_cfg["minimum_heldout_gain"])
            and gain_p <= 1.0 - float(low_cfg["null_quantile"])
        ),
        "basis_stability_pass": bool(
            stability_p <= 1.0 - float(low_cfg["null_quantile"])
        ),
    }


def run_subject(
    subject: str,
    dataset_root: Path,
    config: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    values = _load_subject(subject, dataset_root)
    calibration, confirmation = _split_train80(
        values["split"], float(config["data"]["train80_calibration_fraction"])
    )
    rank_prior = global_rank_prior(
        values["local_rank"], values["participation"], calibration
    )
    seed = int(config["g0"]["random_seed"]) + sum(map(ord, subject))
    reliability, selected_size = _candidate_reliability(
        values, calibration, confirmation, config, rank_prior, seed
    )
    result: dict[str, Any] = {
        "contract": "topic5_event_indexed_evolving_rank_field_v2_2_phase0",
        "subject": subject,
        "dataset": subject.split("_", 1)[0],
        "n_events": len(values["time"]),
        "n_train80": int(np.sum(values["split"] == 0)),
        "n_calibration_events": len(calibration),
        "n_confirmation_events": len(confirmation),
        "n_old_heldout20": int(np.sum(values["split"] == 1)),
        "candidate_reliability": reliability,
        "selected_block_size": selected_size,
        "block_reliability_status": (
            "PASS" if selected_size is not None else "BLOCK_FIELD_UNRELIABLE"
        ),
        "g0": None,
        "g0_middle_only": None,
        "low_rank": None,
        "low_rank_middle_only": None,
        "elr_model_authorized": False,
        "old_heldout20_entered_into_analysis": False,
        "snn_inputs_read": False,
        "forbidden_labels_read": False,
        "dependencies": {
            "dataset": {"path": str(values["dataset_path"]), "sha256": values["dataset_sha256"]},
            "block_mapping": {"path": str(values["mapping_path"]), "sha256": values["mapping_sha256"]},
            "input_audit": {"path": str(values["audit_path"]), "sha256": values["audit_sha256"]},
        },
    }
    if selected_size is not None:
        g0_config = config["g0"]
        calibration_blocks = make_equal_event_blocks(
            calibration, values["source_block"], selected_size
        )
        calibration_blocks = stratified_uniform_subsample_blocks(
            calibration_blocks,
            int(config["blocks"]["max_calibration_blocks_for_low_rank"]),
        )
        confirmation_blocks = make_equal_event_blocks(
            confirmation, values["source_block"], selected_size
        )
        confirmation_blocks = stratified_uniform_subsample_blocks_with_adjacency(
            confirmation_blocks,
            int(config["blocks"]["max_confirmation_blocks_for_g0"]),
        )
        confirmation_audit = np.concatenate(
            [block.indices for block in confirmation_blocks]
        )
        result["g0"] = dynamic_observability_audit(
            values["local_rank"],
            values["participation"],
            values["group_ids"],
            confirmation_audit,
            values["source_block"],
            selected_size,
            rank_prior=rank_prior,
            shrinkage_prior_events=float(config["blocks"]["shrinkage_prior_events"]),
            beta_prior=float(config["blocks"]["beta_prior"]),
            split_repeats=int(g0_config["split_repeats"]),
            permutation_draws=int(g0_config["permutation_draws"]),
            max_between_pairs=int(g0_config["max_between_pairs_per_repeat"]),
            distance_ratio_threshold=float(g0_config["distance_ratio_threshold"]),
            p_threshold=float(g0_config["permutation_p_threshold"]),
            seed=seed + 1000,
            prebuilt_blocks=confirmation_blocks,
            min_adjacent_pairs=int(
                config["blocks"]["min_confirmation_adjacent_pairs"]
            ),
        )
        low_cfg = config["low_rank"]
        middle_mask = (
            (rank_prior > float(low_cfg["middle_rank_lower"]))
            & (rank_prior < float(low_cfg["middle_rank_upper"]))
        )
        if int(np.sum(middle_mask)) >= 3:
            result["g0_middle_only"] = dynamic_observability_audit(
                values["local_rank"],
                values["participation"],
                values["group_ids"],
                confirmation_audit,
                values["source_block"],
                selected_size,
                rank_prior=rank_prior,
                shrinkage_prior_events=float(config["blocks"]["shrinkage_prior_events"]),
                beta_prior=float(config["blocks"]["beta_prior"]),
                split_repeats=int(g0_config["split_repeats"]),
                permutation_draws=int(g0_config["permutation_draws"]),
                max_between_pairs=int(g0_config["max_between_pairs_per_repeat"]),
                distance_ratio_threshold=float(g0_config["distance_ratio_threshold"]),
                p_threshold=float(g0_config["permutation_p_threshold"]),
                seed=seed + 2000,
                contact_mask=middle_mask,
                prebuilt_blocks=confirmation_blocks,
                min_adjacent_pairs=int(
                    config["blocks"]["min_confirmation_adjacent_pairs"]
                ),
            )
        if bool(result["g0"]["g0_pass"]):
            result["low_rank"] = _low_rank_audit(
                values,
                calibration_blocks,
                confirmation_blocks,
                selected_size,
                rank_prior,
                config,
                seed + 3000,
            )
            if int(np.sum(middle_mask)) >= 3:
                result["low_rank_middle_only"] = _low_rank_audit(
                    values,
                    calibration_blocks,
                    confirmation_blocks,
                    selected_size,
                    rank_prior,
                    config,
                    seed + 4000,
                    contact_mask=middle_mask,
                )
            middle_support = bool(
                result["g0_middle_only"] is not None
                and result["g0_middle_only"]["field_variation_pass"]
                and result["low_rank_middle_only"] is not None
                and result["low_rank_middle_only"]["low_rank_gain_pass"]
                and result["low_rank_middle_only"]["basis_stability_pass"]
            )
            result["elr_model_authorized"] = bool(
                result["low_rank"]["low_rank_gain_pass"]
                and result["low_rank"]["basis_stability_pass"]
                and middle_support
            )
    result["status"] = (
        "ELR_ELIGIBLE"
        if result["elr_model_authorized"]
        else "G0_PASS_LOW_RANK_NOT_ESTABLISHED"
        if result["g0"] is not None and result["g0"]["g0_pass"]
        else "G0_NOT_PASSED"
        if result["g0"] is not None
        else "BLOCK_FIELD_UNRELIABLE"
    )
    subject_path = output / "per_subject" / f"{subject}.json"
    atomic_json(subject_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    prior_state = output / "EERF_V2_2_PHASE0_STATE.json"
    adjacency_archive = output / "EERF_V2_2_PHASE0_STATE_PRE_ADJACENCY_REPAIR.json"
    if prior_state.exists() and not adjacency_archive.exists():
        output.mkdir(parents=True, exist_ok=True)
        shutil.copy2(prior_state, adjacency_archive)
    dataset_root = ROOT / config["data"]["dataset_dir"]
    rows = []
    for subject in map(str, config["pilot"]["subjects"]):
        result = run_subject(subject, dataset_root, config, output)
        rows.append(result)
        print(
            subject,
            result["status"],
            "N=", result["selected_block_size"],
            "ratio=", None if result["g0"] is None else result["g0"]["observed"]["field_ratio"],
        )
    table_rows = []
    for result in rows:
        g0 = result["g0"] or {}
        observed = g0.get("observed") or {}
        p_values = g0.get("permutation_p") or {}
        low = result["low_rank"] or {}
        table_rows.append(
            {
                "dataset": result["dataset"],
                "subject": result["subject"],
                "status": result["status"],
                "selected_block_size": result["selected_block_size"],
                "field_ratio": observed.get("field_ratio"),
                "field_permutation_p": p_values.get("field_ratio"),
                "precedence_ratio": observed.get("precedence_ratio"),
                "precedence_permutation_p": p_values.get("precedence_ratio"),
                "distance_lag_spearman": observed.get("field_distance_lag_spearman"),
                "distance_lag_permutation_p": p_values.get("field_distance_lag_spearman"),
                "neighbor_gain": observed.get("neighbor_gain"),
                "neighbor_gain_permutation_p": p_values.get("neighbor_gain"),
                "g0_pass": g0.get("g0_pass", False),
                "middle_field_pass": bool(
                    result["g0_middle_only"]
                    and result["g0_middle_only"]["field_variation_pass"]
                ),
                "selected_low_rank_dimension": low.get("selected_dimension"),
                "heldout_low_rank_gain": low.get("heldout_reconstruction_gain"),
                "low_rank_gain_p": low.get("heldout_gain_permutation_p"),
                "basis_stability": low.get("basis_split_stability"),
                "basis_stability_p": low.get("basis_stability_permutation_p"),
                "elr_model_authorized": result["elr_model_authorized"],
            }
        )
    table = pd.DataFrame(table_rows)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "phase0_patient_summary.csv", index=False)
    payload = {
        "contract": "topic5_event_indexed_evolving_rank_field_v2_2_phase0",
        "status": "COMPLETE_PHASE0_DEVELOPMENT",
        "n_subjects": len(rows),
        "n_block_reliable": sum(row["selected_block_size"] is not None for row in rows),
        "n_g0_pass": sum(bool(row["g0"] and row["g0"]["g0_pass"]) for row in rows),
        "n_temporal_structure_supportive": sum(
            bool(row["g0"] and row["g0"]["temporal_structure_supportive"])
            for row in rows
        ),
        "n_elr_authorized": sum(row["elr_model_authorized"] for row in rows),
        "patients": rows,
        "decision": (
            "OPEN_MINIMAL_ELR_CONTRACT"
            if any(row["elr_model_authorized"] for row in rows)
            else "DO_NOT_IMPLEMENT_ELR_RNN"
        ),
        "claim_boundary": (
            "Phase 0 measures observable block variation, low-dimensionality, "
            "and temporal continuity. It does not identify event-driven shaping."
        ),
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__)),
        "module_sha256": sha256(
            ROOT / "src/topic5_event_indexed_evolving_rank_field.py"
        ),
        "old_heldout20_entered_into_analysis": False,
        "snn_inputs_read": False,
        "forbidden_labels_read": False,
    }
    atomic_json(output / "EERF_V2_2_PHASE0_STATE.json", payload)
    print(
        json.dumps(
            {
                "n_block_reliable": payload["n_block_reliable"],
                "n_g0_pass": payload["n_g0_pass"],
                "n_elr_authorized": payload["n_elr_authorized"],
                "decision": payload["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
