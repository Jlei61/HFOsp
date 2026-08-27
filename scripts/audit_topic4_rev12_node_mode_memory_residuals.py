#!/usr/bin/env python3
"""Zero-simulation discriminator for the next rev12 Node mechanism."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rescore_topic4_rev12_node_historical import _patient_data  # noqa: E402


DEFAULT_ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_CONFIG = ROOT / "config/topic4_rev12_nd_node_mode_memory_residual_audit.json"
COMPONENTS = ("recruitment", "precedence", "profile", "cloud")
PRECEDENCE_CLASSES = ("ICL-ICL", "SCL-SCL", "ICL-SCL")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def ordered_block_records(labels: np.ndarray, blocks: np.ndarray,
                          times: np.ndarray, *,
                          minimum_events: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = np.asarray(labels, int)
    blocks = np.asarray(blocks)
    times = np.asarray(times, float)
    if labels.shape != blocks.shape or labels.shape != times.shape:
        raise ValueError("patient labels, blocks and event times must align")
    output = []
    for block in np.unique(blocks):
        index = np.flatnonzero((blocks == block) & np.isfinite(times))
        if len(index) < int(minimum_events):
            continue
        order = index[np.argsort(times[index], kind="stable")]
        output.append((labels[order], times[order]))
    return output


def ordered_blocks(labels: np.ndarray, blocks: np.ndarray,
                   times: np.ndarray, *, minimum_events: int = 3) -> list[np.ndarray]:
    return [labels for labels, _ in ordered_block_records(
        labels, blocks, times, minimum_events=minimum_events,
    )]


def _mutual_information(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, int)
    right = np.asarray(right, int)
    if len(left) == 0:
        return float("nan")
    joint = np.zeros((2, 2), float)
    np.add.at(joint, (left, right), 1.0)
    joint /= joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    expected = px @ py
    mask = joint > 0.0
    return float(np.sum(joint[mask] * np.log(joint[mask] / expected[mask])))


def sequence_statistics(sequences: list[np.ndarray], *, maximum_lag: int = 5) -> dict:
    transitions = np.zeros((2, 2), int)
    run_lengths = []
    lag_rows = {}
    for sequence in sequences:
        sequence = np.asarray(sequence, int)
        if len(sequence) >= 2:
            np.add.at(transitions, (sequence[:-1], sequence[1:]), 1)
        if len(sequence):
            starts = np.r_[0, 1 + np.flatnonzero(sequence[1:] != sequence[:-1])]
            stops = np.r_[starts[1:], len(sequence)]
            run_lengths.extend((stops - starts).tolist())
    total = int(transitions.sum())
    same = float(np.trace(transitions) / total) if total else float("nan")
    for lag in range(1, int(maximum_lag) + 1):
        left, right = [], []
        for sequence in sequences:
            if len(sequence) > lag:
                left.append(sequence[:-lag])
                right.append(sequence[lag:])
        left_array = np.concatenate(left) if left else np.zeros(0, int)
        right_array = np.concatenate(right) if right else np.zeros(0, int)
        lag_rows[str(lag)] = {
            "n_pairs": int(len(left_array)),
            "same_fraction": (
                float(np.mean(left_array == right_array))
                if len(left_array) else float("nan")
            ),
            "mutual_information_nats": _mutual_information(left_array, right_array),
        }
    row_sum = transitions.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        transitions, row_sum, out=np.full((2, 2), np.nan), where=row_sum > 0,
    )
    return {
        "n_blocks": int(len(sequences)),
        "n_events": int(sum(len(row) for row in sequences)),
        "transition_counts": transitions,
        "transition_probabilities": probabilities,
        "same_mode_fraction": same,
        "lag": lag_rows,
        "run_length_median": float(np.median(run_lengths)) if run_lengths else None,
        "run_length_q95": float(np.quantile(run_lengths, 0.95)) if run_lengths else None,
        "run_length_max": int(max(run_lengths)) if run_lengths else None,
    }


def _gap_same_fractions(sequences: list[np.ndarray],
                        time_sequences: list[np.ndarray],
                        thresholds: list[float]) -> dict[str, dict]:
    output = {}
    for threshold in thresholds:
        same, total = 0, 0
        for labels, times in zip(sequences, time_sequences):
            eligible = np.diff(times) >= float(threshold)
            same += int(np.sum((labels[:-1] == labels[1:]) & eligible))
            total += int(np.sum(eligible))
        output[str(float(threshold))] = {
            "n_pairs": total,
            "same_mode_fraction": float(same / total) if total else float("nan"),
        }
    return output


def block_permutation_audit(sequences: list[np.ndarray], *, draws: int,
                            seed: int, maximum_lag: int = 5,
                            time_sequences: list[np.ndarray] | None = None,
                            gap_thresholds: list[float] | None = None) -> dict:
    observed = sequence_statistics(sequences, maximum_lag=maximum_lag)
    rng = np.random.default_rng(int(seed))
    same_draws = np.empty(int(draws), float)
    mi_draws = np.empty(int(draws), float)
    thresholds = list(gap_thresholds or [])
    gap_observed = (
        _gap_same_fractions(sequences, time_sequences, thresholds)
        if time_sequences is not None and thresholds else {}
    )
    gap_draws = {
        key: np.empty(int(draws), float) for key in gap_observed
    }
    for draw in range(int(draws)):
        shuffled = [rng.permutation(row) for row in sequences]
        stats = sequence_statistics(shuffled, maximum_lag=1)
        same_draws[draw] = stats["same_mode_fraction"]
        mi_draws[draw] = stats["lag"]["1"]["mutual_information_nats"]
        if gap_draws:
            gap_stats = _gap_same_fractions(
                shuffled, time_sequences, thresholds,
            )
            for key in gap_draws:
                gap_draws[key][draw] = gap_stats[key]["same_mode_fraction"]
    observed_same = float(observed["same_mode_fraction"])
    observed_mi = float(observed["lag"]["1"]["mutual_information_nats"])
    return {
        "observed": observed,
        "same_mode_null": {
            "q05_q50_q95": np.quantile(same_draws, [0.05, 0.5, 0.95]),
            "excess_over_null_median": observed_same - float(np.median(same_draws)),
            "upper_tail_p": float((1 + np.sum(same_draws >= observed_same)) / (draws + 1)),
        },
        "lag1_mi_null": {
            "q05_q50_q95": np.quantile(mi_draws, [0.05, 0.5, 0.95]),
            "excess_over_null_median": observed_mi - float(np.median(mi_draws)),
            "upper_tail_p": float((1 + np.sum(mi_draws >= observed_mi)) / (draws + 1)),
        },
        "draws": int(draws),
        "null": "within-recording-block occupancy-preserving label permutation",
        "gap_sensitivity": {
            key: {
                **gap_observed[key],
                "null_q05_q50_q95": np.quantile(values, [0.05, 0.5, 0.95]),
                "excess_over_null_median": (
                    gap_observed[key]["same_mode_fraction"] - float(np.median(values))
                ),
                "upper_tail_p": float((
                    1 + np.sum(values >= gap_observed[key]["same_mode_fraction"])
                ) / (draws + 1)),
            } for key, values in gap_draws.items()
        },
    }


def classify_mode_memory(splits: dict[str, dict], *, threshold: float,
                         quantile: float, short_gap_seconds: float,
                         persistent_gap_seconds: float) -> dict:
    def supported(row: dict) -> bool:
        return bool(
            row["excess_over_null_median"] >= threshold
            and row["upper_tail_p"] <= 1.0 - quantile
        )

    overall = all(supported(split["same_mode_null"]) for split in splits.values())
    short_key = str(float(short_gap_seconds))
    persistent_key = str(float(persistent_gap_seconds))
    short = all(
        supported(split["gap_sensitivity"][short_key]) for split in splits.values()
    )
    persistent = all(
        supported(split["gap_sensitivity"][persistent_key])
        for split in splits.values()
    )
    if overall and short and not persistent:
        status = "PATIENT_MODE_MEMORY_CONFINED_TO_SUBSECOND_EVENT_HISTORY"
    elif overall and persistent:
        status = "PATIENT_MODE_MEMORY_PERSISTS_BEYOND_ONE_SECOND"
    elif overall:
        status = "PATIENT_MODE_MEMORY_NOT_ROBUST_TO_GAP_CONTROL"
    else:
        status = "PATIENT_MODE_MEMORY_NOT_REPLICATED"
    return {
        "status": status,
        "overall_memory_replicated": overall,
        "subsecond_memory_replicated": short,
        "persistent_ge_1s_memory_replicated": persistent,
        "short_lived_activity_dependent_recovery_authorized": bool(overall and short),
        "persistent_state_authorized": bool(overall and persistent),
        "short_memory_gap_seconds": float(short_gap_seconds),
        "persistent_memory_gap_seconds": float(persistent_gap_seconds),
    }


def summarize_candidate(row: dict) -> dict:
    output = {
        "candidate_id": row["candidate_id"],
        "n_networks": int(row["n_networks"]),
        "mean_events": float(row["mean_events"]),
        "soft_objective": float(row["mean_soft_objective"]),
        "causal_direction": float(row["mean_soft_causal_direction"]),
        "causal_monotonicity": float(row["mean_soft_causal_monotonicity"]),
        "modes": {},
    }
    for mode in (0, 1):
        mode_rows = [item["soft_objective"]["modes"][str(mode)]
                     for item in row["per_network"]]
        direction_rows = [item["soft_causal_direction"]["modes"][str(mode)]
                          for item in row["per_network"]]
        output["modes"][str(mode)] = {
            **{
                key: float(np.mean([item[key] for item in mode_rows]))
                for key in COMPONENTS
            },
            "precedence_classes_raw": {
                key: float(np.mean([
                    item["raw"]["precedence_classes"][key] for item in mode_rows
                ])) for key in PRECEDENCE_CLASSES
            },
            "direction_alignment": float(np.mean([
                item["alignment_score"] for item in direction_rows
            ])),
            "effective_events": float(np.mean([
                item["effective_events"] for item in mode_rows
            ])),
        }
    return output


def residual_audit(libraries: dict[str, dict]) -> dict:
    rows = []
    for library_id, payload in libraries.items():
        for row in payload["rows"]:
            summary = summarize_candidate(row)
            summary["library_id"] = library_id
            rows.append(summary)
    local = [row for row in rows if row["library_id"] == "stage_al"]
    anchor = next(row for row in local
                  if row["candidate_id"] == "stage_al_historical_anchor")
    for row in rows:
        row["relative_to_historical_anchor"] = {
            "soft_objective_utility": anchor["soft_objective"] - row["soft_objective"],
            "modes": {
                str(mode): {
                    key + "_utility": (
                        anchor["modes"][str(mode)][key]
                        - row["modes"][str(mode)][key]
                    ) for key in COMPONENTS
                } for mode in (0, 1)
            },
        }
    best_fit = min(local, key=lambda row: row["soft_objective"])
    return {
        "rows": rows,
        "historical_anchor": anchor["candidate_id"],
        "best_stage_al_fit": best_fit["candidate_id"],
        "best_stage_al_fit_bottleneck": {
            str(mode): max(
                COMPONENTS, key=lambda key: best_fit["modes"][str(mode)][key]
            ) for mode in (0, 1)
        },
        "interpretation_boundary": (
            "Component losses are normalized by patient recording-block q95. "
            "Values above one remain outside that component's patient floor."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    artifact_root = args.artifact_root.resolve()
    config = json.loads(config_path.read_text())
    resolved = {}
    for name, contract in config["inputs"].items():
        path = _resolve(artifact_root, contract["path"])
        if _sha256(path) != contract["sha256"]:
            raise RuntimeError(f"input hash mismatch: {name}")
        resolved[name] = path

    cohort = json.loads(resolved["cohort_config"].read_text())
    patient = _patient_data(cohort, artifact_root)
    memory_config = config["patient_mode_memory"]
    splits = {}
    for split_index, split in enumerate(("train", "heldout")):
        records = ordered_block_records(
            patient[f"{split}_labels"], patient[f"{split}_blocks"],
            patient[f"{split}_event_abs_times"],
            minimum_events=int(memory_config["minimum_events_per_block"]),
        )
        sequences = [labels for labels, _ in records]
        time_sequences = [times for _, times in records]
        splits[split] = block_permutation_audit(
            sequences, draws=int(memory_config["permutation_draws"]),
            seed=int(memory_config["seed"]) + split_index,
            maximum_lag=int(memory_config["maximum_lag"]),
            time_sequences=time_sequences,
            gap_thresholds=[
                float(value) for value in memory_config["gap_thresholds_seconds"]
            ],
        )
    threshold = float(memory_config["minimum_absolute_same_mode_excess"])
    quantile = float(memory_config["required_null_quantile"])
    decision = classify_mode_memory(
        splits,
        threshold=threshold,
        quantile=quantile,
        short_gap_seconds=float(memory_config["short_memory_gap_seconds"]),
        persistent_gap_seconds=float(memory_config["persistent_memory_gap_seconds"]),
    )
    libraries = {
        key.removesuffix("_summary"): json.loads(path.read_text())
        for key, path in resolved.items() if key.endswith("_summary")
    }
    residuals = residual_audit(libraries)
    status = decision["status"]
    output = {
        "schema_id": config["schema_id"],
        "status": status,
        "patient_mode_memory": {
            "status": status,
            "splits": splits,
            "decision": {
                "minimum_absolute_same_mode_excess": threshold,
                "required_null_quantile": quantile,
                **decision,
            },
        },
        "node_residuals": residuals,
        "next_mechanism_constraint": (
            "test only a short-lived activity-dependent Node recovery canary; "
            "do not add a persistent autonomous mode state and do not use patient labels at runtime"
            if decision["short_lived_activity_dependent_recovery_authorized"]
            and not decision["persistent_state_authorized"] else
            "patient event history does not justify the proposed short-lived Node recovery canary"
        ),
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in resolved.items()
        },
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "claim_boundary": config["claim_boundary"],
    }
    output_root = artifact_root / config["output_root"]
    _atomic_json(output_root / "mode_memory_residual_audit.json", output)
    print(json.dumps({
        "status": status,
        "train_same_mode_excess": splits["train"]["same_mode_null"]["excess_over_null_median"],
        "heldout_same_mode_excess": splits["heldout"]["same_mode_null"]["excess_over_null_median"],
        "best_stage_al_fit": residuals["best_stage_al_fit"],
        "output": str(output_root / "mode_memory_residual_audit.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
