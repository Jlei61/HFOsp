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
LOCAL_OCCUPANCY_WINDOWS_SECONDS = (10.0, 30.0, 60.0, 300.0)


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


def ordered_covariate_records(
    labels: np.ndarray,
    blocks: np.ndarray,
    start_times: np.ndarray,
    end_times: np.ndarray,
    event_sizes: np.ndarray,
    *,
    minimum_events: int = 3,
) -> list[dict]:
    arrays = [
        np.asarray(labels, int),
        np.asarray(blocks),
        np.asarray(start_times, float),
        np.asarray(end_times, float),
        np.asarray(event_sizes, int),
    ]
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError("labels, blocks, start/end times and event sizes must align")
    output = []
    for block in np.unique(arrays[1]):
        index = np.flatnonzero(
            (arrays[1] == block)
            & np.isfinite(arrays[2])
            & np.isfinite(arrays[3])
        )
        if len(index) < int(minimum_events):
            continue
        order = index[np.argsort(arrays[2][index], kind="stable")]
        output.append({
            "block": str(block),
            "labels": arrays[0][order],
            "start_times": arrays[2][order],
            "end_times": arrays[3][order],
            "event_sizes": arrays[4][order],
        })
    return output


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
    run_array = np.asarray(run_lengths, float)
    return {
        "n_blocks": int(len(sequences)),
        "n_events": int(sum(len(row) for row in sequences)),
        "transition_counts": transitions,
        "transition_probabilities": probabilities,
        "same_mode_fraction": same,
        "lag": lag_rows,
        "run_length_median": float(np.median(run_lengths)) if run_lengths else None,
        "run_length_mean": float(np.mean(run_array)) if run_lengths else None,
        "run_length_q95": float(np.quantile(run_lengths, 0.95)) if run_lengths else None,
        "run_length_max": int(max(run_lengths)) if run_lengths else None,
        "run_fraction_ge_3": float(np.mean(run_array >= 3)) if run_lengths else None,
        "run_fraction_ge_5": float(np.mean(run_array >= 5)) if run_lengths else None,
    }


def _separated_same_fractions(
    sequences: list[np.ndarray],
    start_sequences: list[np.ndarray],
    thresholds: list[float],
    *,
    end_sequences: list[np.ndarray] | None = None,
) -> dict[str, dict]:
    if end_sequences is not None and len(end_sequences) != len(start_sequences):
        raise ValueError("start and end sequence lists must align")
    output = {}
    for threshold in thresholds:
        same, total = 0, 0
        for block_index, (labels, starts) in enumerate(zip(sequences, start_sequences)):
            separation = (
                starts[1:] - np.asarray(end_sequences[block_index], float)[:-1]
                if end_sequences is not None else np.diff(starts)
            )
            eligible = separation >= float(threshold)
            same += int(np.sum((labels[:-1] == labels[1:]) & eligible))
            total += int(np.sum(eligible))
        output[str(float(threshold))] = {
            "n_pairs": total,
            "same_mode_fraction": float(same / total) if total else float("nan"),
        }
    return output


def _permutation_summary(observed: float, draws: np.ndarray) -> dict:
    draws = np.asarray(draws, float)
    finite = draws[np.isfinite(draws)]
    if not np.isfinite(observed) or not len(finite):
        return {
            "q05_q50_q95": np.full(3, np.nan),
            "q025_q50_q975": np.full(3, np.nan),
            "excess_over_null_median": float("nan"),
            "upper_tail_p": float("nan"),
        }
    return {
        "q05_q50_q95": np.quantile(finite, [0.05, 0.5, 0.95]),
        "q025_q50_q975": np.quantile(finite, [0.025, 0.5, 0.975]),
        "excess_over_null_median": float(observed - np.median(finite)),
        "upper_tail_p": float((1 + np.sum(finite >= observed)) / (len(finite) + 1)),
    }


def _sequence_endpoints(stats: dict) -> dict[str, float]:
    lag2 = stats["lag"].get("2", {})
    return {
        "same_mode_fraction": float(stats["same_mode_fraction"]),
        "lag2_same_fraction": float(lag2.get("same_fraction", np.nan)),
        "run_length_mean": float(stats["run_length_mean"]),
        "run_fraction_ge_3": float(stats["run_fraction_ge_3"]),
        "run_fraction_ge_5": float(stats["run_fraction_ge_5"]),
    }


def _endpoint_null_summaries(observed: dict, draws: dict[str, np.ndarray]) -> dict:
    endpoints = _sequence_endpoints(observed)
    return {
        key: {
            "observed": value,
            **_permutation_summary(value, draws[key]),
        }
        for key, value in endpoints.items()
    }


def block_permutation_audit(sequences: list[np.ndarray], *, draws: int,
                            seed: int, maximum_lag: int = 5,
                            time_sequences: list[np.ndarray] | None = None,
                            end_time_sequences: list[np.ndarray] | None = None,
                            gap_thresholds: list[float] | None = None) -> dict:
    observed = sequence_statistics(sequences, maximum_lag=maximum_lag)
    rng = np.random.default_rng(int(seed))
    endpoint_draws = {
        key: np.empty(int(draws), float) for key in _sequence_endpoints(observed)
    }
    mi_draws = np.empty(int(draws), float)
    thresholds = list(gap_thresholds or [])
    post_end_gap_observed = (
        _separated_same_fractions(
            sequences, time_sequences, thresholds,
            end_sequences=end_time_sequences,
        )
        if time_sequences is not None and end_time_sequences is not None and thresholds
        else {}
    )
    start_interval_observed = (
        _separated_same_fractions(sequences, time_sequences, thresholds)
        if time_sequences is not None and thresholds else {}
    )
    post_end_gap_draws = {
        key: np.empty(int(draws), float) for key in post_end_gap_observed
    }
    start_interval_draws = {
        key: np.empty(int(draws), float) for key in start_interval_observed
    }
    for draw in range(int(draws)):
        shuffled = [rng.permutation(row) for row in sequences]
        stats = sequence_statistics(shuffled, maximum_lag=max(2, maximum_lag))
        for key, value in _sequence_endpoints(stats).items():
            endpoint_draws[key][draw] = value
        mi_draws[draw] = stats["lag"]["1"]["mutual_information_nats"]
        if post_end_gap_draws:
            gap_stats = _separated_same_fractions(
                shuffled, time_sequences, thresholds,
                end_sequences=end_time_sequences,
            )
            for key in post_end_gap_draws:
                post_end_gap_draws[key][draw] = gap_stats[key]["same_mode_fraction"]
        if start_interval_draws:
            interval_stats = _separated_same_fractions(
                shuffled, time_sequences, thresholds,
            )
            for key in start_interval_draws:
                start_interval_draws[key][draw] = interval_stats[key]["same_mode_fraction"]
    observed_mi = float(observed["lag"]["1"]["mutual_information_nats"])
    endpoint_nulls = _endpoint_null_summaries(observed, endpoint_draws)
    post_end_gap_sensitivity = {
        key: {
            **post_end_gap_observed[key],
            **_permutation_summary(
                post_end_gap_observed[key]["same_mode_fraction"], values,
            ),
        } for key, values in post_end_gap_draws.items()
    }
    return {
        "observed": observed,
        "same_mode_null": endpoint_nulls["same_mode_fraction"],
        "lag1_mi_null": {
            "observed": observed_mi,
            **_permutation_summary(observed_mi, mi_draws),
        },
        "permutation_endpoints": endpoint_nulls,
        "draws": int(draws),
        "null": "within-recording-block occupancy-preserving label permutation",
        "gap_definition": "next_event_start_seconds - previous_event_end_seconds",
        "post_end_gap_sensitivity": post_end_gap_sensitivity,
        "gap_sensitivity": post_end_gap_sensitivity,
        "start_to_start_interval_sensitivity": {
            key: {
                **start_interval_observed[key],
                **_permutation_summary(
                    start_interval_observed[key]["same_mode_fraction"], values,
                ),
            } for key, values in start_interval_draws.items()
        },
    }


def _audit_stratified_permutation(
    sequences: list[np.ndarray],
    strata: list[np.ndarray],
    *,
    draws: int,
    seed: int,
    maximum_lag: int,
    null_description: str,
) -> dict:
    if len(sequences) != len(strata):
        raise ValueError("sequence and stratum lists must align")
    if any(len(labels) != len(keys) for labels, keys in zip(sequences, strata)):
        raise ValueError("each sequence must align with its strata")
    observed = sequence_statistics(sequences, maximum_lag=maximum_lag)
    endpoint_draws = {
        key: np.empty(int(draws), float) for key in _sequence_endpoints(observed)
    }
    rng = np.random.default_rng(int(seed))
    group_indices = []
    n_strata = 0
    variable_strata = 0
    movable_events = 0
    for labels, keys in zip(sequences, strata):
        block_groups = []
        for key in np.unique(keys):
            index = np.flatnonzero(keys == key)
            block_groups.append(index)
            n_strata += 1
            if len(index) > 1:
                movable_events += len(index)
                if len(np.unique(labels[index])) > 1:
                    variable_strata += 1
        group_indices.append(block_groups)
    for draw in range(int(draws)):
        shuffled = []
        for labels, groups in zip(sequences, group_indices):
            row = np.asarray(labels, int).copy()
            for index in groups:
                row[index] = rng.permutation(row[index])
            shuffled.append(row)
        stats = sequence_statistics(shuffled, maximum_lag=max(2, maximum_lag))
        for key, value in _sequence_endpoints(stats).items():
            endpoint_draws[key][draw] = value
    return {
        "observed": observed,
        "permutation_endpoints": _endpoint_null_summaries(observed, endpoint_draws),
        "same_mode_null": _endpoint_null_summaries(
            observed, endpoint_draws,
        )["same_mode_fraction"],
        "draws": int(draws),
        "null": null_description,
        "n_strata": int(n_strata),
        "n_variable_label_strata": int(variable_strata),
        "movable_events": int(movable_events),
        "movable_event_fraction": float(
            movable_events / max(1, sum(len(row) for row in sequences))
        ),
    }


def conditional_size_iei_permutation_audit(
    records: list[dict], *, draws: int, seed: int,
    maximum_lag: int = 5, n_iei_quantiles: int = 5,
) -> dict:
    if int(n_iei_quantiles) < 2:
        raise ValueError("n_iei_quantiles must be at least two")
    log_iei_rows = []
    for record in records:
        starts = np.asarray(record["start_times"], float)
        if len(starts) > 1:
            iei = np.diff(starts)
            log_iei_rows.append(np.log(iei[np.isfinite(iei) & (iei > 0)]))
    pooled = np.concatenate(log_iei_rows) if log_iei_rows else np.zeros(0, float)
    if not len(pooled):
        raise ValueError("no positive within-block start-to-start IEIs")
    cutpoints = np.quantile(
        pooled, np.linspace(0.0, 1.0, int(n_iei_quantiles) + 1),
    )
    internal_cuts = cutpoints[1:-1]
    sequences = []
    strata = []
    for block_index, record in enumerate(records):
        labels = np.asarray(record["labels"], int)
        starts = np.asarray(record["start_times"], float)
        sizes = np.asarray(record["event_sizes"], int)
        iei_bins = np.full(len(labels), -1, int)
        if len(labels) > 1:
            iei = np.diff(starts)
            valid = np.isfinite(iei) & (iei > 0)
            iei_bins[1:][valid] = np.searchsorted(
                internal_cuts, np.log(iei[valid]), side="right",
            )
        keys = np.asarray([
            f"{block_index}|{int(size)}|{int(iei_bin)}"
            for size, iei_bin in zip(sizes, iei_bins)
        ])
        sequences.append(labels)
        strata.append(keys)
    audit = _audit_stratified_permutation(
        sequences, strata, draws=draws, seed=seed, maximum_lag=maximum_lag,
        null_description=(
            "recording-block x exact-event-size x within-split "
            "start-to-start-log-IEI-quintile conditional label permutation"
        ),
    )
    audit.update({
        "iei_definition": "current_event_start_seconds - previous_event_start_seconds",
        "iei_role": "conditioning_diagnostic_not_a_post_end_gap",
        "n_iei_quantiles": int(n_iei_quantiles),
        "log_iei_cutpoints": cutpoints,
    })
    return audit


def local_occupancy_permutation_audit(
    records: list[dict], *, window_seconds: float, draws: int, seed: int,
    maximum_lag: int = 5,
) -> dict:
    if float(window_seconds) <= 0:
        raise ValueError("window_seconds must be positive")
    sequences = []
    strata = []
    within_window_pairs = 0
    for block_index, record in enumerate(records):
        labels = np.asarray(record["labels"], int)
        starts = np.asarray(record["start_times"], float)
        bins = np.floor((starts - starts[0]) / float(window_seconds)).astype(int)
        keys = np.asarray([f"{block_index}|{int(value)}" for value in bins])
        sequences.append(labels)
        strata.append(keys)
        within_window_pairs += int(np.sum(bins[:-1] == bins[1:]))
    audit = _audit_stratified_permutation(
        sequences, strata, draws=draws, seed=seed, maximum_lag=maximum_lag,
        null_description=(
            f"within-recording-block {float(window_seconds):g}-second "
            "local-occupancy-preserving label permutation"
        ),
    )
    audit.update({
        "window_seconds": float(window_seconds),
        "window_anchor": "first eligible event start within each recording block",
        "n_adjacent_pairs_within_same_window": int(within_window_pairs),
    })
    return audit


def recording_block_bootstrap(
    sequences: list[np.ndarray], *, draws: int, seed: int,
) -> dict:
    rows = []
    for sequence in sequences:
        labels = np.asarray(sequence, int)
        if len(labels) < 2:
            continue
        observed = float(np.mean(labels[:-1] == labels[1:]))
        counts = np.bincount(labels, minlength=2).astype(float)
        expected = float(np.sum(counts * (counts - 1)) / (len(labels) * (len(labels) - 1)))
        rows.append({
            "n_events": int(len(labels)),
            "n_pairs": int(len(labels) - 1),
            "observed_same_mode_fraction": observed,
            "occupancy_expected_same_mode_fraction": expected,
            "same_mode_excess": observed - expected,
        })
    if not rows:
        raise ValueError("recording-block bootstrap requires at least one block")

    excess = np.asarray([row["same_mode_excess"] for row in rows], float)
    weights = np.asarray([row["n_pairs"] for row in rows], float)

    def estimates(index: np.ndarray) -> tuple[float, float]:
        selected = excess[index]
        selected_weights = weights[index]
        return (
            float(np.average(selected, weights=selected_weights)),
            float(np.mean(selected)),
        )

    observed_event_weighted, observed_equal_block = estimates(np.arange(len(rows)))
    rng = np.random.default_rng(int(seed))
    event_weighted_draws = np.empty(int(draws), float)
    equal_block_draws = np.empty(int(draws), float)
    for draw in range(int(draws)):
        index = rng.integers(0, len(rows), size=len(rows))
        event_weighted_draws[draw], equal_block_draws[draw] = estimates(index)
    return {
        "n_blocks": int(len(rows)),
        "draws": int(draws),
        "block_rows": rows,
        "event_weighted_same_mode_excess": {
            "estimate": observed_event_weighted,
            "ci95": np.quantile(event_weighted_draws, [0.025, 0.975]),
        },
        "equal_block_same_mode_excess": {
            "estimate": observed_equal_block,
            "ci95": np.quantile(equal_block_draws, [0.025, 0.975]),
        },
        "bootstrap_unit": "recording_block",
        "null_expectation": "exact occupancy-preserving random-order expectation per block",
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
        supported(split["post_end_gap_sensitivity"][short_key])
        for split in splits.values()
    )
    persistent = all(
        supported(split["post_end_gap_sensitivity"][persistent_key])
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
        "event_burst_autocorrelation_not_excluded": bool(not persistent),
        "short_lived_activity_dependent_recovery_authorized": bool(overall and short),
        "persistent_state_authorized": bool(overall and persistent),
        "short_memory_gap_seconds": float(short_gap_seconds),
        "persistent_memory_gap_seconds": float(persistent_gap_seconds),
        "gap_definition": "next_event_start_seconds - previous_event_end_seconds",
        "decision_boundary": (
            "conditional and local-occupancy controls can confirm short-range "
            "persistence only; they cannot authorize a persistent state"
        ),
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
        records = ordered_covariate_records(
            patient[f"{split}_labels"], patient[f"{split}_blocks"],
            patient[f"{split}_event_abs_times"],
            patient[f"{split}_event_abs_end_times"],
            np.sum(np.isfinite(patient[f"{split}_ranks"]), axis=1),
            minimum_events=int(memory_config["minimum_events_per_block"]),
        )
        sequences = [record["labels"] for record in records]
        start_sequences = [record["start_times"] for record in records]
        end_sequences = [record["end_times"] for record in records]
        split_audit = block_permutation_audit(
            sequences, draws=int(memory_config["permutation_draws"]),
            seed=int(memory_config["seed"]) + split_index,
            maximum_lag=int(memory_config["maximum_lag"]),
            time_sequences=start_sequences,
            end_time_sequences=end_sequences,
            gap_thresholds=[
                float(value) for value in memory_config["gap_thresholds_seconds"]
            ],
        )
        split_audit["conditional_size_log_iei_quintile"] = (
            conditional_size_iei_permutation_audit(
                records,
                draws=int(memory_config["permutation_draws"]),
                seed=int(memory_config["seed"]) + 100 + split_index,
                maximum_lag=int(memory_config["maximum_lag"]),
            )
        )
        split_audit["local_occupancy_sensitivity"] = {
            str(float(window)): local_occupancy_permutation_audit(
                records,
                window_seconds=window,
                draws=int(memory_config["permutation_draws"]),
                seed=int(memory_config["seed"]) + 200 + 10 * split_index + index,
                maximum_lag=int(memory_config["maximum_lag"]),
            )
            for index, window in enumerate(LOCAL_OCCUPANCY_WINDOWS_SECONDS)
        }
        split_audit["recording_block_bootstrap"] = recording_block_bootstrap(
            sequences,
            draws=int(memory_config["permutation_draws"]),
            seed=int(memory_config["seed"]) + 300 + split_index,
        )
        splits[split] = split_audit
    threshold = float(memory_config["minimum_absolute_same_mode_excess"])
    quantile = float(memory_config["required_null_quantile"])
    decision = classify_mode_memory(
        splits,
        threshold=threshold,
        quantile=quantile,
        short_gap_seconds=float(memory_config["short_memory_gap_seconds"]),
        persistent_gap_seconds=float(memory_config["persistent_memory_gap_seconds"]),
    )
    short_range_controls = {}
    for split, split_audit in splits.items():
        conditional = split_audit["conditional_size_log_iei_quintile"]["same_mode_null"]
        local = split_audit["local_occupancy_sensitivity"]["10.0"]["same_mode_null"]
        bootstrap = split_audit["recording_block_bootstrap"]
        short_range_controls[split] = {
            "conditional_size_log_iei_supported": bool(
                conditional["excess_over_null_median"] >= threshold
                and conditional["upper_tail_p"] <= 1.0 - quantile
            ),
            "local_10s_occupancy_supported": bool(
                local["excess_over_null_median"] >= threshold
                and local["upper_tail_p"] <= 1.0 - quantile
            ),
            "event_weighted_block_bootstrap_ci_excludes_zero": bool(
                bootstrap["event_weighted_same_mode_excess"]["ci95"][0] > 0
            ),
            "equal_block_bootstrap_ci_excludes_zero": bool(
                bootstrap["equal_block_same_mode_excess"]["ci95"][0] > 0
            ),
        }
        short_range_controls[split]["all_confirmed"] = bool(
            all(short_range_controls[split].values())
        )
    decision["short_range_controls"] = short_range_controls
    decision["short_range_persistence_confirmed"] = bool(
        all(row["all_confirmed"] for row in short_range_controls.values())
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
            "burst_separated_analysis": {
                "status": (
                    "POST_END_GE_1S_PERSISTENCE_REPLICATED"
                    if decision["persistent_ge_1s_memory_replicated"] else
                    "EVENT_BURST_AUTOCORRELATION_NOT_EXCLUDED"
                ),
                "gap_definition": (
                    "next_event_start_seconds - previous_event_end_seconds"
                ),
                "start_to_start_role": "diagnostic_interval_only_not_gap",
                "event_burst_autocorrelation_not_excluded": decision[
                    "event_burst_autocorrelation_not_excluded"
                ],
                "persistent_state_authorized": decision[
                    "persistent_state_authorized"
                ],
            },
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
