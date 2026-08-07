#!/usr/bin/env python3
"""Run the frozen EERF v2.2 Phase-1 event-history increment pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_event_history_increment import (  # noqa: E402
    build_block_observations,
    circular_shift_delta_within_source,
    evaluate_event_history_mse,
    evaluate_model_ladder,
    make_pair_dataset,
    permute_delta_within_source,
    replace_delta,
)
from src.topic5_event_indexed_evolving_rank_field import (  # noqa: E402
    global_rank_prior,
    make_equal_event_blocks,
)


DEFAULT_CONFIG = ROOT / "config/topic5_event_indexed_evolving_rank_field_v2_2_phase1.yaml"


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


def load_subject(subject: str, config: dict[str, Any]) -> dict[str, Any]:
    dataset_path = ROOT / config["dataset_dir"] / "per_subject" / f"{subject}.npz"
    audit_root = ROOT / config["input_audit_dir"] / "per_subject"
    mapping_path = audit_root / f"{subject}.npz"
    audit_path = audit_root / f"{subject}.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise RuntimeError(f"{subject}: event-indexed input audit is not PASS")
    if audit.get("source_dataset_npz_sha256") != sha256(dataset_path):
        raise RuntimeError(f"{subject}: source dataset drifted after input audit")
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
            raise RuntimeError(f"{subject}: source-block mapping drifted")
        values["source_block"] = np.asarray(
            mapping["event_source_block_id"], np.int32
        )
    values["dependencies"] = {
        "dataset": {"path": str(dataset_path), "sha256": sha256(dataset_path)},
        "mapping": {"path": str(mapping_path), "sha256": sha256(mapping_path)},
        "input_audit": {"path": str(audit_path), "sha256": sha256(audit_path)},
    }
    return values


def split_train80(split: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    indices = np.flatnonzero(np.asarray(split) == 0)
    cut = int(np.floor(float(fraction) * len(indices)))
    if cut < 1 or cut >= len(indices):
        raise ValueError("invalid train80 Phase-1 split")
    return indices[:cut], indices[cut:]


def _null_p(observed: float, values: list[float]) -> float:
    null = np.asarray(values, float)
    return float((1 + np.sum(null >= observed)) / (1 + len(null)))


def run_variant(
    values: dict[str, Any],
    calibration_blocks,
    confirmation_blocks,
    rank_prior: np.ndarray,
    dimension: int,
    config: dict[str, Any],
    seed: int,
    *,
    contact_mask: np.ndarray | None,
) -> dict[str, Any]:
    estimation = config["estimation"]
    common = {
        "rank_prior": rank_prior,
        "shrinkage_prior_events": float(estimation["shrinkage_prior_events"]),
        "beta_prior": float(estimation["beta_prior"]),
        "contact_mask": contact_mask,
    }
    calibration_observations = build_block_observations(
        values["local_rank"], values["participation"], values["group_ids"],
        values["time"], calibration_blocks, **common
    )
    confirmation_observations = build_block_observations(
        values["local_rank"], values["participation"], values["group_ids"],
        values["time"], confirmation_blocks, **common
    )
    train = make_pair_dataset(calibration_observations)
    test = make_pair_dataset(confirmation_observations)
    ladder = evaluate_model_ladder(
        train,
        test,
        dimension=int(dimension),
        alpha_grid=estimation["ridge_alpha_grid"],
        switching_state_grid=estimation["switching_state_grid"],
        validation_fraction=float(estimation["inner_validation_fraction"]),
        seed=int(seed),
    )
    best_mse = float(ladder["mse"][ladder["best_baseline"]])
    observed_increment = float(ladder["event_increment_over_best"])
    rng = np.random.default_rng(int(seed))
    nulls = {"order_shuffle": [], "block_permutation": [], "circular_shift": []}
    for _ in range(int(estimation["null_draws"])):
        shuffled_train_observations = build_block_observations(
            values["local_rank"], values["participation"], values["group_ids"],
            values["time"], calibration_blocks, rng=rng, **common
        )
        shuffled_test_observations = build_block_observations(
            values["local_rank"], values["participation"], values["group_ids"],
            values["time"], confirmation_blocks, rng=rng, **common
        )
        shuffled_train = make_pair_dataset(shuffled_train_observations)
        shuffled_test = make_pair_dataset(shuffled_test_observations)
        order_mse = evaluate_event_history_mse(
            shuffled_train,
            shuffled_test,
            dimension=int(dimension),
            alpha_grid=estimation["ridge_alpha_grid"],
            validation_fraction=float(estimation["inner_validation_fraction"]),
        )
        nulls["order_shuffle"].append(best_mse - order_mse)

        permuted_train = replace_delta(
            train, permute_delta_within_source(train.delta, train.source, rng)
        )
        permuted_test = replace_delta(
            test, permute_delta_within_source(test.delta, test.source, rng)
        )
        permutation_mse = evaluate_event_history_mse(
            permuted_train,
            permuted_test,
            dimension=int(dimension),
            alpha_grid=estimation["ridge_alpha_grid"],
            validation_fraction=float(estimation["inner_validation_fraction"]),
        )
        nulls["block_permutation"].append(best_mse - permutation_mse)

        shifted_train = replace_delta(
            train, circular_shift_delta_within_source(train.delta, train.source, rng)
        )
        shifted_test = replace_delta(
            test, circular_shift_delta_within_source(test.delta, test.source, rng)
        )
        shift_mse = evaluate_event_history_mse(
            shifted_train,
            shifted_test,
            dimension=int(dimension),
            alpha_grid=estimation["ridge_alpha_grid"],
            validation_fraction=float(estimation["inner_validation_fraction"]),
        )
        nulls["circular_shift"].append(best_mse - shift_mse)
    return {
        "ladder": ladder,
        "null_p": {
            name: _null_p(observed_increment, values)
            for name, values in nulls.items()
        },
        "null_median_increment": {
            name: float(np.median(values)) for name, values in nulls.items()
        },
        "n_calibration_blocks": len(calibration_blocks),
        "n_confirmation_blocks": len(confirmation_blocks),
        "n_calibration_true_adjacent_pairs": len(train.target),
        "n_confirmation_true_adjacent_pairs": len(test.target),
    }


def run_subject(
    phase0_patient: dict[str, Any],
    config: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    subject = str(phase0_patient["subject"])
    values = load_subject(subject, config)
    calibration, confirmation = split_train80(
        values["split"], float(config["data"]["train80_calibration_fraction"])
    )
    block_size = int(phase0_patient["selected_block_size"])
    dimension = int(phase0_patient["low_rank"]["selected_dimension"])
    rank_prior = global_rank_prior(
        values["local_rank"], values["participation"], calibration
    )
    calibration_blocks = make_equal_event_blocks(
        calibration, values["source_block"], block_size
    )
    confirmation_blocks = make_equal_event_blocks(
        confirmation, values["source_block"], block_size
    )
    seed = int(config["estimation"]["random_seed"]) + sum(map(ord, subject))
    full = run_variant(
        values,
        calibration_blocks,
        confirmation_blocks,
        rank_prior,
        dimension,
        config,
        seed,
        contact_mask=None,
    )
    lower = float(config["estimation"]["middle_rank_lower"])
    upper = float(config["estimation"]["middle_rank_upper"])
    middle_mask = (rank_prior > lower) & (rank_prior < upper)
    if int(np.sum(middle_mask)) < 3:
        raise RuntimeError(f"{subject}: fewer than three middle contacts")
    middle = run_variant(
        values,
        calibration_blocks,
        confirmation_blocks,
        rank_prior,
        dimension,
        config,
        seed + 10000,
        contact_mask=middle_mask,
    )
    threshold = float(config["gate"]["permutation_p_threshold"])
    full_pass = bool(
        full["ladder"]["event_increment_over_best"] > 0
        and all(
            full["null_p"][name] <= threshold
            for name in config["gate"]["full_required_nulls"]
        )
    )
    middle_pass = bool(
        middle["ladder"]["event_increment_over_best"] > 0
        and all(
            middle["null_p"][name] <= threshold
            for name in config["gate"]["middle_required_nulls"]
        )
    )
    authorized = bool(full_pass and middle_pass)
    result = {
        "contract": config["contract"]["name"],
        "subject": subject,
        "dataset": phase0_patient["dataset"],
        "phase0_block_size": block_size,
        "phase0_dimension": dimension,
        "n_contacts": len(values["contact_names"]),
        "n_middle_contacts": int(np.sum(middle_mask)),
        "full": full,
        "middle_only": middle,
        "full_gate_pass": full_pass,
        "middle_gate_pass": middle_pass,
        "state_space_elr_authorized": authorized,
        "status": "OPEN_LINEAR_STATE_SPACE_ELR" if authorized else "EVENT_HISTORY_INCREMENT_NOT_ESTABLISHED",
        "dependencies": values["dependencies"],
        "old_heldout20_entered_into_analysis": False,
        "snn_inputs_read": False,
        "forbidden_labels_read": False,
        "claim_boundary": (
            "Development-only association of within-block event-history direction "
            "with the next block field; not causal plasticity."
        ),
    }
    atomic_json(output / "per_subject" / f"{subject}.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    phase0_path = ROOT / config["phase0_state"]
    phase0 = json.loads(phase0_path.read_text(encoding="utf-8"))
    if phase0.get("decision") != "OPEN_MINIMAL_ELR_CONTRACT":
        raise RuntimeError("Phase 0 did not authorize the Phase-1 contract")
    eligible = [
        patient for patient in phase0["patients"]
        if patient.get("elr_model_authorized")
    ]
    output = ROOT / config["output_dir"]
    results = []
    for patient in eligible:
        result = run_subject(patient, config, output)
        results.append(result)
        print(
            result["subject"], result["status"],
            "increment=", result["full"]["ladder"]["event_increment_over_best"],
            "p=", result["full"]["null_p"],
        )
    rows = []
    for result in results:
        full = result["full"]
        middle = result["middle_only"]
        rows.append(
            {
                "dataset": result["dataset"],
                "subject": result["subject"],
                "status": result["status"],
                "block_size": result["phase0_block_size"],
                "dimension": result["phase0_dimension"],
                "best_baseline": full["ladder"]["best_baseline"],
                "event_increment": full["ladder"]["event_increment_over_best"],
                "event_relative_gain": full["ladder"]["event_relative_gain_over_best"],
                "order_shuffle_p": full["null_p"]["order_shuffle"],
                "block_permutation_p": full["null_p"]["block_permutation"],
                "circular_shift_p": full["null_p"]["circular_shift"],
                "middle_event_increment": middle["ladder"]["event_increment_over_best"],
                "middle_order_shuffle_p": middle["null_p"]["order_shuffle"],
                "middle_circular_shift_p": middle["null_p"]["circular_shift"],
                "state_space_elr_authorized": result["state_space_elr_authorized"],
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "phase1_patient_summary.csv", index=False)
    state = {
        "contract": config["contract"]["name"],
        "status": "COMPLETE_DEVELOPMENT_PHASE1",
        "n_phase0_eligible": len(eligible),
        "n_phase1_pass": sum(result["state_space_elr_authorized"] for result in results),
        "decision": (
            "OPEN_LINEAR_STATE_SPACE_ELR_CONTRACT"
            if any(result["state_space_elr_authorized"] for result in results)
            else "STOP_EVENT_DRIVEN_ELR"
        ),
        "patients": results,
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__)),
        "module_sha256": sha256(ROOT / "src/topic5_event_history_increment.py"),
        "phase0_state_path": str(phase0_path),
        "phase0_state_sha256": sha256(phase0_path),
        "old_heldout20_entered_into_analysis": False,
        "snn_inputs_read": False,
        "forbidden_labels_read": False,
        "claim_boundary": (
            "This development test cannot establish activity-dependent shaping "
            "or independent replication."
        ),
    }
    atomic_json(output / "EERF_V2_2_PHASE1_STATE.json", state)
    print(json.dumps({"n_phase1_pass": state["n_phase1_pass"], "decision": state["decision"]}, indent=2))


if __name__ == "__main__":
    main()
