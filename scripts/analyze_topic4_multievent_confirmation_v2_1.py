#!/usr/bin/env python3
"""G4 statistics and diagnostic figures for the frozen G3 confirmation set."""
from __future__ import annotations

import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_topic4_xy_research as base
from scripts.run_topic4_multievent_distribution_v2_1 import OUT, repaired_observation
from src.topic4_joint_xy_kernel import event_kernel_features


EVALUATOR = ROOT / "results/topic4_sef_hfo/multidimensional_interictal_observation_repair_v2/evaluator.pkl"
PACKET = OUT / "patient_time_packet/packet_manifest.json"
G3 = OUT / "execution/confirmation_24s"
FIGURES = OUT / "figures"
GROUPS = {
    "upper_SCL": ["SCL6", "SCL7", "SCL8", "SCL9"],
    "right_ICL": ["ICL1", "ICL2"],
    "middle_ICL": ["ICL5", "ICL6", "ICL7", "ICL8"],
    "left_ICL": ["ICL9", "ICL10", "ICL11"],
}
COLORS = {0: "#2166ac", 1: "#b2182b", -1: "#888888"}
EVENT_COLUMNS = [
    "candidate_id", "topology_seed", "dynamics_seed",
    "primary_local_index", "detected_event_index", "window_start_ms",
    "window_stop_ms", "mode", "support_state",
    "assigned_patient_distance", "distance_to_mode0", "distance_to_mode1",
    "relative_distance_mode0_minus_mode1",
    "upper_SCL_n", "upper_SCL_ms", "right_ICL_n", "right_ICL_ms",
    "middle_ICL_n", "middle_ICL_ms", "left_ICL_n", "left_ICL_ms",
    "middle_before_both_ICL_ends", "right_before_upper_and_left",
    "upper_and_left_before_right", "t10_t50_t90_ms",
    "left_10ms_mass_fraction", "right_10ms_mass_fraction",
    "native_mapping_status", "native_overlapping_worker_events",
    "native_worker_event_index", "native_source_evaluable",
    "native_arrival_finite_bins", "native_arrival_span_p10_p90_ms",
    "native_arrival_total_span_ms",
    "native_early_quartile_components_4_neighbor",
    "native_early_quartile_extent_mm",
]


def group_times(row, names):
    result = {}
    for key, contacts in GROUPS.items():
        values = row[[names.index(name) for name in contacts]]
        valid = values[np.isfinite(values)]
        result[f"{key}_n"] = int(len(valid))
        result[f"{key}_ms"] = float(np.median(valid)) if len(valid) else None
    upper, right, middle, left = [result[f"{key}_ms"] for key in GROUPS]
    result["middle_before_both_ICL_ends"] = (
        None if any(value is None for value in (right, middle, left))
        else bool(middle + 2.0 < min(right, left))
    )
    result["right_before_upper_and_left"] = (
        None if any(value is None for value in (right, upper, left))
        else bool(right + 2.0 < min(upper, left))
    )
    result["upper_and_left_before_right"] = (
        None if any(value is None for value in (right, upper, left))
        else bool(max(upper, left) + 2.0 < right)
    )
    return result


def _array_digest(values):
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _worker(candidate_id, topology, dynamics):
    stem = f"{candidate_id}_topo_{topology}_dyn_{dynamics}"
    path = G3 / "workers" / f"{stem}.json"
    record = base.read(path)
    if base.sha(record["arrays"]["path"]) != record["arrays"]["sha256"]:
        raise RuntimeError(f"G3 worker arrays changed: {stem}")
    observation_json, observation_npz, observation = repaired_observation(path)
    with np.load(observation_npz) as arrays:
        primary = np.asarray(arrays["primary_event_indices"], int)
        centroids = np.asarray(arrays["centroid_ms"], float)
        windows = np.asarray(arrays["windows_ms"], float)
    return stem, path, record, observation_json, observation, centroids, primary, windows


def _patient_block_reference(evaluator, n, rng):
    if n <= 0:
        return {"status": "NOT_ESTIMABLE_NO_MODEL_EVENTS", "actual_N": int(n)}
    probe_index = np.asarray(evaluator.index["PROBE"], int)
    table = evaluator.patient[probe_index]
    blocks = evaluator.blocks[probe_index]
    eligible = [block for block in np.unique(blocks) if np.sum(blocks == block) >= n]
    if not eligible:
        return {"status": "NOT_ESTIMABLE_NO_PATIENT_BLOCK_WITH_MATCHED_N"}
    counts, supported = [], []
    for _ in range(128):
        block = int(rng.choice(eligible))
        available = np.flatnonzero(blocks == block)
        selected = rng.choice(available, n, replace=False)
        labels, state, _ = evaluator.classify(table[selected])
        counts.append(np.bincount(labels, minlength=evaluator.k))
        supported.append([np.sum((labels == mode) & (state == 1))
                          for mode in range(evaluator.k)])
    counts = np.asarray(counts)
    supported = np.asarray(supported)
    return {
        "status": "ESTIMABLE_REUSED_DEVELOPMENT_PROBE_BLOCK_REFERENCE",
        "actual_N": n, "resamples": 128, "eligible_blocks": len(eligible),
        "mode_count_q025_median_q975": [
            np.quantile(counts[:, mode], [0.025, 0.5, 0.975]).tolist()
            for mode in range(evaluator.k)
        ],
        "supported_count_q025_median_q975": [
            np.quantile(supported[:, mode], [0.025, 0.5, 0.975]).tolist()
            for mode in range(evaluator.k)
        ],
        "one_patient_recording_block_per_draw": True,
        "independent_clinical_holdout": False,
    }


def _segments(observation, primary, labels, support, actual_duration_ms):
    primary_lookup = {int(event_index): local for local, event_index in enumerate(primary)}
    rows = []
    primary_midpoints = []
    for event_index, event in enumerate(observation["events"]):
        interval = event.get("qualifying_interval_ms", event["window_ms"])
        midpoint = float(np.mean(interval))
        local = primary_lookup.get(event_index)
        label = int(labels[local]) if local is not None else -1
        state = int(support[local]) if local is not None else 0
        rows.append({
            "detected_event_index": event_index,
            "qualifying_midpoint_ms": midpoint,
            "primary": local is not None, "mode": label,
            "support_state": state,
        })
        if local is not None:
            primary_midpoints.append((midpoint, local))
    segment_rows = []
    for segment, (start, stop) in enumerate(zip((0, 6000, 12000, 18000),
                                                 (6000, 12000, 18000, 24000))):
        available_stop = min(float(stop), actual_duration_ms)
        observed_start = max(float(start), 500.0)
        effective = max(0.0, available_stop - observed_start)
        local = [row for row in rows if start <= row["qualifying_midpoint_ms"]
                 < stop + (1 if stop == 24000 else 0)]
        primary_local = [row for row in local if row["primary"]]
        segment_rows.append({
            "segment": segment, "bounds_ms": [start, stop],
            "effective_post_burnin_ms": effective,
            "detected_count": len(local), "primary_count": len(primary_local),
            "primary_mode_counts": [sum(row["mode"] == mode for row in primary_local)
                                    for mode in (0, 1)],
        })
    intervals = []
    for (previous, _), (current, local) in zip(primary_midpoints[:-1], primary_midpoints[1:]):
        previous_segment = min(int(previous // 6000), 3)
        current_segment = min(int(current // 6000), 3)
        intervals.append({
            "interval_ms": current - previous,
            "assigned_to_later_segment": current_segment,
            "cross_segment": previous_segment != current_segment,
            "later_primary_local_index": local,
        })
    return {"events": rows, "segments": segment_rows, "inter_event_intervals": intervals,
            "burnin_ms": 500.0, "whole_record_actual_duration_ms": actual_duration_ms}


def _model_envelope_quantiles(record, observation, centroids, primary):
    with np.load(record["arrays"]["path"]) as arrays:
        envelope = np.asarray(arrays["contact_envelope"], float)
        dt = float(arrays["contact_envelope_dt_ms"])
    output = []
    for local, event_index in enumerate(primary):
        event = observation["events"][int(event_index)]
        start, stop = event["window_ms"]
        lo, hi = int(round(start / dt)), int(round(stop / dt))
        baseline = np.asarray(event["local_baseline"], float)
        positive = np.maximum(envelope[:, lo:hi] - baseline[:, None], 0.0)
        participating = np.isfinite(centroids[int(event_index)])
        mass = positive[participating].sum(axis=0)
        first = float(np.nanmin(centroids[int(event_index)]))
        time = (np.arange(lo, hi) + 0.5) * dt - first
        if mass.sum() <= 0:
            quantiles = [None, None, None]
        else:
            cumulative = np.cumsum(mass) / mass.sum()
            quantiles = [float(time[min(np.searchsorted(cumulative, q), len(time) - 1)])
                         for q in (0.1, 0.5, 0.9)]
        edge_n = max(1, int(np.ceil(10.0 / dt)))
        output.append({
            "primary_local_index": local, "detected_event_index": int(event_index),
            "t10_t50_t90_ms": quantiles,
            "left_10ms_mass_fraction": float(mass[:edge_n].sum() / mass.sum()) if mass.sum() else None,
            "right_10ms_mass_fraction": float(mass[-edge_n:].sum() / mass.sum()) if mass.sum() else None,
        })
    return output


def _components4(mask):
    mask = np.asarray(mask, bool)
    unseen = set(map(tuple, np.argwhere(mask)))
    components = 0
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            row, column = stack.pop()
            for neighbor in ((row - 1, column), (row + 1, column),
                             (row, column - 1), (row, column + 1)):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
    return components


def _native_source_summaries(record, observation):
    """Link repaired-contact windows to native physical events without forcing a match."""
    with np.load(record["arrays"]["path"]) as arrays:
        maps = np.asarray(arrays["source_onset_maps_ms"], float)
        evaluable = np.asarray(arrays["source_onset_evaluable"], bool)
        event_on = np.asarray(arrays["event_t_on_ms"], float)
        event_off = np.asarray(arrays["event_t_off_ms"], float)
        bin_mm = float(arrays["source_bin_mm"])
    summaries = []
    for event in observation["events"]:
        start, stop = map(float, event["window_ms"])
        overlap = np.maximum(0.0, np.minimum(stop, event_off) - np.maximum(start, event_on))
        candidates = np.flatnonzero(overlap > 0)
        empty = {
            "native_worker_event_index": None,
            "native_source_evaluable": False,
            "native_arrival_finite_bins": 0,
            "native_arrival_span_p10_p90_ms": None,
            "native_arrival_total_span_ms": None,
            "native_early_quartile_components_4_neighbor": None,
            "native_early_quartile_extent_mm": None,
        }
        if not len(candidates):
            summaries.append({
                "native_mapping_status": "NOT_ESTIMABLE_NO_OVERLAPPING_PHYSICAL_EVENT",
                "native_overlapping_worker_events": 0, **empty,
            })
            continue
        chosen = int(candidates[np.argmax(overlap[candidates])])
        status = ("UNIQUE_OVERLAPPING_PHYSICAL_EVENT" if len(candidates) == 1
                  else "AMBIGUOUS_OVERLAP_CHOSEN_MAXIMUM_FOR_DESCRIPTION")
        if chosen >= len(evaluable) or not evaluable[chosen]:
            summaries.append({
                "native_mapping_status": f"{status}_SOURCE_NOT_EVALUABLE",
                "native_overlapping_worker_events": int(len(candidates)),
                **{**empty, "native_worker_event_index": chosen},
            })
            continue
        source_map = maps[chosen]
        finite = np.isfinite(source_map)
        values = source_map[finite]
        if not len(values):
            summaries.append({
                "native_mapping_status": f"{status}_EMPTY_SOURCE_MAP",
                "native_overlapping_worker_events": int(len(candidates)),
                **{**empty, "native_worker_event_index": chosen},
            })
            continue
        q10, q25, q90 = np.quantile(values, (0.1, 0.25, 0.9))
        early = finite & (source_map <= q25)
        coords = np.argwhere(early).astype(float) * bin_mm
        if len(coords) > 1:
            delta = coords[:, None, :] - coords[None, :, :]
            extent = float(np.sqrt(np.sum(delta * delta, axis=2)).max())
        else:
            extent = 0.0
        summaries.append({
            "native_mapping_status": status,
            "native_overlapping_worker_events": int(len(candidates)),
            "native_worker_event_index": chosen,
            "native_source_evaluable": True,
            "native_arrival_finite_bins": int(len(values)),
            "native_arrival_span_p10_p90_ms": float(q90 - q10),
            "native_arrival_total_span_ms": float(np.max(values) - np.min(values)),
            "native_early_quartile_components_4_neighbor": int(_components4(early)),
            "native_early_quartile_extent_mm": extent,
        })
    return summaries


def analyze():
    qualification = base.read(EVALUATOR.parent / "qualification.json")
    if base.sha(EVALUATOR) != qualification["evaluator_sha256"]:
        raise RuntimeError("frozen evaluator changed")
    with open(EVALUATOR, "rb") as stream:
        evaluator = pickle.load(stream)
    names = base.read(EVALUATOR.parent / "observation_contract.json")["contact_names"]
    candidates = base.read(OUT / "g3_candidates.json")["candidates"]
    rng = np.random.default_rng(2026090704)
    runs, event_rows, segment_records, patient_references = [], [], [], []
    static = {}
    dynamic_identity = {}
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        for topology in (6101, 6102):
            for dynamics in (7101, 7102):
                (stem, worker_path, record, observation_json, observation,
                 centroids, primary, windows) = _worker(candidate_id, topology, dynamics)
                table = centroids[primary]
                labels, support, assigned_distance = evaluator.classify(table)
                joint = event_kernel_features(
                    table, evaluator.xy, evaluator.groups, evaluator.scale,
                )["joint"] if len(table) else np.empty((0, evaluator.fit_joint.shape[1]))
                distance_to_modes = np.column_stack([
                    evaluator.trees[mode].query(joint, k=5)[0].mean(axis=1)
                    for mode in range(evaluator.k)
                ]) if len(table) else np.empty((0, evaluator.k))
                model_quantiles = _model_envelope_quantiles(
                    record, observation, centroids, primary,
                )
                native_summaries = _native_source_summaries(record, observation)
                for local, event_index in enumerate(primary):
                    row = {
                        "candidate_id": candidate_id, "topology_seed": topology,
                        "dynamics_seed": dynamics, "primary_local_index": local,
                        "detected_event_index": int(event_index),
                        "window_start_ms": float(windows[event_index, 0]),
                        "window_stop_ms": float(windows[event_index, 1]),
                        "mode": int(labels[local]), "support_state": int(support[local]),
                        "assigned_patient_distance": float(assigned_distance[local]),
                        "distance_to_mode0": float(distance_to_modes[local, 0]),
                        "distance_to_mode1": float(distance_to_modes[local, 1]),
                        "relative_distance_mode0_minus_mode1": float(
                            distance_to_modes[local, 0] - distance_to_modes[local, 1]
                        ),
                        **group_times(table[local], names),
                        **model_quantiles[local],
                        **native_summaries[int(event_index)],
                    }
                    event_rows.append(row)
                metrics = evaluator.metrics(table, np.zeros(len(table), int), detail=True)
                segments = _segments(
                    observation, primary, labels, support,
                    record["simulation"]["actual_duration_ms"],
                )
                segment_records.append({"candidate_id": candidate_id,
                                        "topology_seed": topology,
                                        "dynamics_seed": dynamics, **segments})
                patient_reference = _patient_block_reference(evaluator, len(table), rng)
                patient_references.append({"candidate_id": candidate_id,
                                           "topology_seed": topology,
                                           "dynamics_seed": dynamics,
                                           **patient_reference})
                runs.append({
                    "candidate_id": candidate_id, "topology_seed": topology,
                    "dynamics_seed": dynamics,
                    "execution_status": record["execution_status"],
                    "physical_status": record["physical_status"],
                    "actual_duration_ms": record["simulation"]["actual_duration_ms"],
                    "n_detected": observation["n_detected_windows"],
                    "n_primary": len(table),
                    "mode_counts": np.bincount(labels, minlength=evaluator.k).astype(int).tolist(),
                    "supported_counts": [int(np.sum((labels == mode) & (support == 1)))
                                         for mode in range(evaluator.k)],
                    "mode_metrics": metrics["modes"],
                    "worker_path": str(worker_path), "worker_sha256": base.sha(worker_path),
                    "observation_path": str(observation_json),
                    "observation_sha256": base.sha(observation_json),
                })
                static[(candidate_id, topology, dynamics)] = record["static_array_identity"]
                with np.load(record["arrays"]["path"]) as worker_arrays:
                    dynamic_identity[(candidate_id, topology, dynamics)] = {
                        "active_fraction_sha256": _array_digest(
                            worker_arrays["active_fraction"]
                        ),
                        "contact_envelope_sha256": _array_digest(
                            worker_arrays["contact_envelope"]
                        ),
                        "stored_topology_seed": int(worker_arrays["topology_seed"]),
                        "stored_dynamics_seed": int(worker_arrays["dynamics_seed"]),
                    }
    replay_checks = []
    for candidate in candidates:
        for topology in (6101, 6102):
            left = static[(candidate["candidate_id"], topology, 7101)]
            right = static[(candidate["candidate_id"], topology, 7102)]
            left_dynamic = dynamic_identity[(candidate["candidate_id"], topology, 7101)]
            right_dynamic = dynamic_identity[(candidate["candidate_id"], topology, 7102)]
            keys = [key for key in left if key.endswith("sha256")]
            replay_checks.append({
                "candidate_id": candidate["candidate_id"], "topology_seed": topology,
                "all_static_array_hashes_identical": all(left[key] == right[key] for key in keys),
                "hash_comparison": {key: left[key] == right[key] for key in keys},
                "dynamics_seeds_distinct_and_stored": (
                    left_dynamic["stored_topology_seed"] == topology
                    and right_dynamic["stored_topology_seed"] == topology
                    and left_dynamic["stored_dynamics_seed"] == 7101
                    and right_dynamic["stored_dynamics_seed"] == 7102
                ),
                "dynamic_outputs_distinct": (
                    left_dynamic["active_fraction_sha256"]
                    != right_dynamic["active_fraction_sha256"]
                    or left_dynamic["contact_envelope_sha256"]
                    != right_dynamic["contact_envelope_sha256"]
                ),
                "dynamic_output_hashes": {
                    "7101": left_dynamic, "7102": right_dynamic,
                },
            })
    with open(OUT / "g3_all_primary_events.csv", "w", newline="") as stream:
        fieldnames = list(event_rows[0]) if event_rows else EVENT_COLUMNS
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(event_rows)
    base.write(OUT / "g3_confirmation_analysis.json", {
        "status": "G3_CONFIRMATION_ANALYZED",
        "runs": runs,
        "same_graph_replay_static_checks": replay_checks,
        "all_same_graph_static_checks_pass": all(
            row["all_static_array_hashes_identical"] for row in replay_checks
        ),
        "all_dynamics_seed_and_distinct_output_checks_pass": all(
            row["dynamics_seeds_distinct_and_stored"]
            and row["dynamic_outputs_distinct"] for row in replay_checks
        ),
        "cross_candidate_randomness_term": (
            "shared integer seed only; common random innovations are not claimed "
            "without a verified identical random-call trace"
        ),
        "exact_full_unit_repeat": (
            "NOT_RUN: no extra 24 s evaluation was added outside the fixed 156-run budget; "
            "resume verifies frozen inputs and complete output hashes but is not an independent repeat"
        ),
        "mode_absence_interpretation": "NOT_ESTIMABLE for conditional propagation; not evidence of biological absence",
        "four_replays_random_effect_limit": "insufficient to decompose all topology and dynamics variance",
        "native_mapping_contract": (
            "repaired-contact windows are linked only to temporally overlapping physical events; "
            "multiple overlaps remain explicitly ambiguous and native descriptors are descriptive"
        ),
    })
    base.write(OUT / "g3_six_second_segments.json", {
        "status": "SIX_SECOND_DIAGNOSTIC_COMPLETE",
        "assignment": "qualifying interval midpoint; interval assigned to later event",
        "feeds_training": False, "runs": segment_records,
    })
    base.write(OUT / "g3_patient_block_matched_reference.json", {
        "status": "ACTUAL_N_BLOCK_MATCHED_REFERENCE_COMPLETE",
        "runs": patient_references,
        "balanced_packet_used_for_frequency": False,
    })
    return candidates, runs, event_rows, segment_records


def _fraction(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return {"numerator": int(sum(values)), "denominator": len(values),
            "fraction": float(np.mean(values)) if values else None}


def patient_model_comparison(candidates, runs, event_rows):
    packet = base.read(PACKET)
    selection = base.read(PACKET.parent / "selection.json")
    natural_counts = selection["natural_mode_counts_FIT"]
    observation_names = base.read(
        EVALUATOR.parent / "observation_contract.json"
    )["contact_names"]
    patient_rows = []
    for row in packet["readable_events"]:
        with np.load(row["arrays_path"]) as arrays:
            route = group_times(np.asarray(arrays["stored_lag_ms"], float),
                                arrays["contact_names"].astype(str).tolist())
        patient_rows.append({**row, **route})
    route_keys = ["middle_before_both_ICL_ends", "right_before_upper_and_left",
                  "upper_and_left_before_right"]
    comparison = {"patient_packet": {}, "model": {}}
    for mode in (0, 1):
        rows = [row for row in patient_rows if row["mode"] == mode]
        comparison["patient_packet"][str(mode)] = {
            "n": len(rows), "no_SCL": int(sum(row["upper_SCL_n"] == 0 for row in rows)),
            "routes": {key: _fraction(rows, key) for key in route_keys},
            "t10_t50_t90_ms": {
                f"q{index}": np.quantile(
                    [row["t10_t50_t90_ms"][column] for row in rows],
                    [0.25, 0.5, 0.75],
                ).tolist()
                for index, column in ((10, 0), (50, 1), (90, 2))
            },
        }
    for candidate in candidates:
        cid = candidate["candidate_id"]
        comparison["model"][cid] = {}
        for mode in (0, 1):
            rows = [row for row in event_rows
                    if row["candidate_id"] == cid and row["mode"] == mode]
            comparison["model"][cid][str(mode)] = {
                "n": len(rows), "no_SCL": int(sum(row["upper_SCL_n"] == 0 for row in rows)),
                "routes": {key: _fraction(rows, key) for key in route_keys},
                "t10_t50_t90_ms": {
                    f"q{index}": (np.quantile(
                        [row["t10_t50_t90_ms"][column] for row in rows
                         if row["t10_t50_t90_ms"][column] is not None],
                        [0.25, 0.5, 0.75],
                    ).tolist() if any(
                        row["t10_t50_t90_ms"][column] is not None for row in rows
                    ) else None)
                    for index, column in ((10, 0), (50, 1), (90, 2))
                },
            }
    payload = {
        "status": "PATIENT_PACKET_AND_G3_MODEL_COMPARISON_COMPLETE",
        "comparison": comparison,
        "patient_packet_sha256": base.sha(PACKET),
        "patient_packet_balanced_not_frequency_estimator": True,
        "natural_FIT_mode_counts": natural_counts,
        "time_measurement_warning": (
            "patient raw HFO envelope and model firing-density envelope have distinct amplitudes; "
            "t10/t50/t90 are window-relative mass summaries, not neuron ignition times"
        ),
        "route_warning": (
            "sparse-contact group-centroid descriptions are not continuous wave paths or causal routes"
        ),
        "contact_names": observation_names,
    }
    base.write(OUT / "g4_patient_model_comparison.json", payload)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(candidates)); width = 0.34
    for mode, shift in ((0, -width / 2), (1, width / 2)):
        values = []
        for candidate in candidates:
            rows = [row for row in runs if row["candidate_id"] == candidate["candidate_id"]]
            values.append(np.mean([
                row["mode_counts"][mode] / row["n_primary"] if row["n_primary"] else np.nan
                for row in rows
            ]))
        axes[0].bar(x + shift, values, width, color=COLORS[mode], label=f"M{mode}")
    patient_fraction = np.asarray(natural_counts) / np.sum(natural_counts)
    axes[0].axhline(patient_fraction[0], color=COLORS[0], ls="--", lw=1)
    axes[0].axhline(patient_fraction[1], color=COLORS[1], ls="--", lw=1)
    axes[0].set(ylabel="Mean mode fraction over four runs", ylim=(0, 1),
                xticks=x, xticklabels=[row["candidate_id"] for row in candidates])
    axes[0].tick_params(axis="x", rotation=65, labelsize=7)
    axes[0].legend(frameon=False)
    route_key = {0: "right_before_upper_and_left", 1: "upper_and_left_before_right"}
    for mode, shift in ((0, -width / 2), (1, width / 2)):
        values = [comparison["model"][row["candidate_id"]][str(mode)]["routes"][
            route_key[mode]]["fraction"] for row in candidates]
        axes[1].bar(x + shift, [np.nan if value is None else value for value in values],
                    width, color=COLORS[mode], label=f"M{mode}")
        patient_value = comparison["patient_packet"][str(mode)]["routes"][
            route_key[mode]]["fraction"]
        axes[1].axhline(patient_value, color=COLORS[mode], ls="--", lw=1)
    axes[1].set(ylabel="Mode-conditional coarse route fraction", ylim=(0, 1),
                xticks=x, xticklabels=[row["candidate_id"] for row in candidates])
    axes[1].tick_params(axis="x", rotation=65, labelsize=7)
    axes[1].legend(frameon=False)
    fig.suptitle("G3 model distributions vs fixed patient references\nDashed: patient FIT frequency or balanced-packet route fraction")
    fig.tight_layout()
    png = FIGURES / "patient_model_distribution_summary.png"
    pdf = png.with_suffix(".pdf")
    fig.savefig(png, dpi=200); fig.savefig(pdf); plt.close(fig)
    payload["figure"] = {"png": str(png), "png_sha256": base.sha(png),
                         "pdf": str(pdf), "pdf_sha256": base.sha(pdf)}
    base.write(OUT / "g4_patient_model_comparison.json", payload)
    return payload


def _overview(candidate, run, segment, evaluator):
    candidate_id = candidate["candidate_id"]
    topology, dynamics = run["topology_seed"], run["dynamics_seed"]
    stem, _, record, _, observation, centroids, primary, _ = _worker(
        candidate_id, topology, dynamics,
    )
    with np.load(record["arrays"]["path"]) as arrays:
        active = np.asarray(arrays["active_fraction"], float)
        active_dt = float(arrays["active_fraction_bin_ms"])
    time = (np.arange(len(active)) + 0.5) * active_dt / 1000.0
    labels, support, _ = evaluator.classify(centroids[primary])
    primary_lookup = {int(event): local for local, event in enumerate(primary)}
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.8), gridspec_kw={"height_ratios": [2.2, 1]})
    axes[0].plot(time, active, color=".2", lw=0.55)
    for event_index, event in enumerate(observation["events"]):
        local = primary_lookup.get(event_index)
        mode = int(labels[local]) if local is not None else -1
        axes[0].axvspan(event["window_ms"][0] / 1000,
                        event["window_ms"][1] / 1000,
                        color=COLORS[mode], alpha=0.20)
    for boundary in (6, 12, 18): axes[0].axvline(boundary, color=".65", ls="--", lw=.7)
    axes[0].axvspan(0, .5, color=".7", alpha=.2, label="burn-in")
    actual_seconds = float(run["actual_duration_ms"]) / 1000.0
    if actual_seconds < 24.0:
        axes[0].axvspan(
            actual_seconds, 24.0, color="white", edgecolor=".65",
            hatch="///", alpha=.75, label="not simulated after early stop",
        )
    axes[0].set(xlim=(0, 24), ylabel="Active E fraction",
                title=(f"{candidate_id} | topo {topology}, dyn {dynamics} | "
                       f"{run['physical_status']} | actual {actual_seconds:.3f} s"))
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    counts = np.asarray([row["primary_mode_counts"] for row in segment["segments"]])
    x = np.arange(4)
    axes[1].bar(x, counts[:, 0], color=COLORS[0], label="M0")
    axes[1].bar(x, counts[:, 1], bottom=counts[:, 0], color=COLORS[1], label="M1")
    axes[1].set(xticks=x, xticklabels=["0–6", "6–12", "12–18", "18–24"],
                xlabel="Simulation segment (s)", ylabel="Primary events")
    axes[1].legend(frameon=False, ncol=2)
    fig.tight_layout()
    png = FIGURES / f"overview_{stem}.png"; pdf = png.with_suffix(".pdf")
    fig.savefig(png, dpi=180); fig.savefig(pdf); plt.close(fig)
    return {"png": str(png), "png_sha256": base.sha(png),
            "pdf": str(pdf), "pdf_sha256": base.sha(pdf)}


def _event_pages_and_gif(candidate, topology, dynamics, evaluator, *, make_gif):
    candidate_id = candidate["candidate_id"]
    stem, _, record, _, observation, centroids, primary, _ = _worker(
        candidate_id, topology, dynamics,
    )
    with np.load(record["arrays"]["path"]) as arrays:
        names = arrays["contact_names"].astype(str).tolist()
        xy = np.asarray(arrays["contact_xy_mm"], float)
        movie = np.asarray(arrays["sheet_activity_counts"])
        frame_ms = float(arrays["sheet_activity_frame_ms"])
    labels = np.full(len(centroids), -1, int)
    if len(primary): labels[primary] = evaluator.classify(centroids[primary])[0]
    pages = []
    offsets = np.asarray([-16, 0, 16, 32, 64], float)
    vmax = max(1.0, float(np.quantile(movie, 0.9995)))
    for page_index, start in enumerate(range(0, len(centroids), 6)):
        count = min(6, len(centroids) - start)
        fig, axes = plt.subplots(count, 6, figsize=(12, 1.85 * count), squeeze=False)
        for row, event_index in enumerate(range(start, start + count)):
            event = observation["events"][event_index]
            center = (float(np.nanmin(centroids[event_index]))
                      if np.isfinite(centroids[event_index]).any()
                      else float(np.mean(event["window_ms"])))
            for column, offset in enumerate(offsets):
                frame = int(np.clip(round((center + offset) / frame_ms - 0.5), 0, len(movie) - 1))
                ax = axes[row, column]
                ax.imshow(movie[frame], origin="lower", extent=(0, 20, 0, 20),
                          cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
                ax.set(xticks=[], yticks=[], title=f"{offset:+.0f} ms")
            ax = axes[row, -1]
            times = centroids[event_index]
            valid = np.isfinite(times)
            if valid.any():
                ax.scatter(xy[valid, 0], xy[valid, 1], c=times[valid] - np.nanmin(times),
                           cmap="viridis", vmin=0, vmax=100, s=24)
            ax.scatter(xy[~valid, 0], xy[~valid, 1], facecolors="none", edgecolors=".6", s=20)
            ax.set(xticks=[], yticks=[], aspect="equal", title=f"event {event_index + 1} | M{labels[event_index]}")
        fig.suptitle(f"{candidate_id} | topo {topology}, dyn {dynamics} | all windows chronological")
        fig.tight_layout(rect=(0, 0, 1, .97))
        png = FIGURES / f"native_{stem}_page_{page_index + 1:02d}.png"
        pdf = png.with_suffix(".pdf")
        fig.savefig(png, dpi=140); fig.savefig(pdf); plt.close(fig)
        pages.append({
            "png": str(png), "png_sha256": base.sha(png),
            "pdf": str(pdf), "pdf_sha256": base.sha(pdf),
        })
    gif_record = None
    if make_gif and len(centroids):
        frames = []
        for event_index in range(len(centroids)):
            event = observation["events"][event_index]
            center = (float(np.nanmin(centroids[event_index]))
                      if np.isfinite(centroids[event_index]).any()
                      else float(np.mean(event["window_ms"])))
            for offset in offsets:
                frame = int(np.clip(round((center + offset) / frame_ms - .5), 0, len(movie) - 1))
                fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=90)
                axes[0].imshow(movie[frame], origin="lower", extent=(0, 20, 0, 20),
                               cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
                axes[0].set(title=f"native field {offset:+.0f} ms", xlim=(0, 20), ylim=(0, 20))
                times = centroids[event_index]; valid = np.isfinite(times)
                if valid.any():
                    axes[1].scatter(xy[valid, 0], xy[valid, 1],
                                    c=times[valid] - np.nanmin(times), cmap="viridis",
                                    vmin=0, vmax=100, s=38)
                axes[1].scatter(xy[~valid, 0], xy[~valid, 1], facecolors="none",
                                edgecolors=".6", s=30)
                axes[1].set(title="frozen-contact centroid readout", aspect="equal")
                fig.suptitle(f"{candidate_id} | topo {topology}, dyn {dynamics} | event {event_index + 1}/{len(centroids)} | M{labels[event_index]}")
                fig.tight_layout()
                fig.canvas.draw()
                frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3]).convert("P", palette=Image.Palette.ADAPTIVE))
                plt.close(fig)
        gif = FIGURES / f"chronological_{stem}.gif"
        frames[0].save(gif, save_all=True, append_images=frames[1:], duration=160,
                       loop=0, optimize=False, disposal=2)
        with Image.open(gif) as image:
            decoded = image.n_frames
            for index in range(image.n_frames): image.seek(index); image.load()
        gif_record = {"path": str(gif), "sha256": base.sha(gif),
                      "frames": decoded, "expected_frames": len(frames),
                      "all_frames_decoded": decoded == len(frames),
                      "events_covered": len(centroids),
                      "offsets_ms": offsets.tolist()}
    return {"pages": pages, "gif": gif_record}


def _patient_packet_pages():
    """Render every frozen readable patient event without example selection."""
    packet = base.read(PACKET)
    outputs = []
    for mode in (0, 1):
        rows = [row for row in packet["readable_events"] if row["mode"] == mode]
        for page_index, start in enumerate(range(0, len(rows), 8)):
            page_rows = rows[start:start + 8]
            fig, axes = plt.subplots(
                len(page_rows), 2, figsize=(11, 1.8 * len(page_rows)), squeeze=False,
                gridspec_kw={"width_ratios": [4.5, 1.4]},
            )
            for axis_row, record in enumerate(page_rows):
                with np.load(record["arrays_path"]) as arrays:
                    names = arrays["contact_names"].astype(str)
                    time_ms = np.asarray(arrays["time_ms"], float)
                    envelope = np.asarray(arrays["envelope_background_robust_z"], float)
                    packed = np.asarray(arrays["packed_window_mask"], bool)
                    centroid = np.asarray(arrays["stored_lag_ms"], float)
                    participating = np.asarray(arrays["participation_mask"], bool)
                shown = envelope[:, packed]
                shown_time = time_ms[packed]
                ax = axes[axis_row, 0]
                vmax = max(1.0, float(np.nanquantile(shown, 0.99)))
                ax.imshow(
                    shown, aspect="auto", origin="lower", cmap="magma",
                    extent=(shown_time[0], shown_time[-1], -0.5, len(names) - 0.5),
                    vmin=0, vmax=vmax, interpolation="nearest",
                )
                ax.set_yticks(np.arange(len(names)), names, fontsize=6)
                ax.set_ylabel(f"raw {record['raw_global_event_index']}", fontsize=7)
                if axis_row == len(page_rows) - 1:
                    ax.set_xlabel("Patient packed-window time (ms)")
                bx = axes[axis_row, 1]
                y = np.arange(len(names))
                bx.scatter(centroid[participating], y[participating], s=18,
                           color=COLORS[mode])
                bx.scatter(centroid[~participating], y[~participating], s=14,
                           facecolors="none", edgecolors=".7")
                bx.axvline(0, color=".65", lw=.6)
                bx.set_yticks([])
                bx.set_title(
                    f"{record['packet_role']} | block {record['block_id']}", fontsize=7,
                )
                if axis_row == len(page_rows) - 1:
                    bx.set_xlabel("Stored centroid lag (ms)")
            fig.suptitle(
                f"Frozen patient packet mode {mode}: all readable events "
                f"{start + 1}-{start + len(page_rows)} / {len(rows)}"
            )
            fig.tight_layout(rect=(0, 0, 1, .98))
            png = FIGURES / f"patient_packet_mode{mode}_page_{page_index + 1:02d}.png"
            pdf = png.with_suffix(".pdf")
            fig.savefig(png, dpi=160); fig.savefig(pdf); plt.close(fig)
            outputs.append({
                "mode": mode, "page": page_index + 1,
                "event_ids": [row["raw_global_event_index"] for row in page_rows],
                "png": str(png), "png_sha256": base.sha(png),
                "pdf": str(pdf), "pdf_sha256": base.sha(pdf),
            })
    return outputs


def render(candidates, runs, segments):
    FIGURES.mkdir(parents=True, exist_ok=True)
    with open(EVALUATOR, "rb") as stream: evaluator = pickle.load(stream)
    segment_lookup = {(row["candidate_id"], row["topology_seed"], row["dynamics_seed"]): row
                      for row in segments}
    outputs = {"overviews": [], "chronological": [], "patient_packet_pages": []}
    for candidate in candidates:
        cid = candidate["candidate_id"]
        for run in [row for row in runs if row["candidate_id"] == cid]:
            outputs["overviews"].append(_overview(
                candidate, run,
                segment_lookup[(cid, run["topology_seed"], run["dynamics_seed"])],
                evaluator,
            ))
        # Prespecified representative pair plus same-topology dynamics replay.
        for dynamics in (7101, 7102):
            outputs["chronological"].append({
                "candidate_id": cid, "topology_seed": 6101,
                "dynamics_seed": dynamics,
                **_event_pages_and_gif(candidate, 6101, dynamics, evaluator, make_gif=True),
            })
        # Other topology combinations retain complete per-event snapshot pages.
        for dynamics in (7101, 7102):
            outputs["chronological"].append({
                "candidate_id": cid, "topology_seed": 6102,
                "dynamics_seed": dynamics,
                **_event_pages_and_gif(candidate, 6102, dynamics, evaluator, make_gif=False),
            })
    outputs["patient_packet_pages"] = _patient_packet_pages()
    base.write(OUT / "g4_visual_manifest.json", {
        "status": "G4_DIAGNOSTIC_VISUALS_GENERATED_PENDING_AGENT_AND_HUMAN_REVIEW",
        "representative_seed_rule": "numerically smallest topology 6101 and dynamics 7101",
        "same_topology_second_dynamics_included": True,
        "outputs": outputs,
    })
    readme = """### overview_*.png / .pdf
每个候选、每个 topology/dynamics 组合的完整 24 秒活动概览；按冻结修复观测标出所有检测窗，并分四个 6 秒段报告两类主要事件数量。灰区为既有 0–0.5 秒 burn-in，图中不把它解释成无事件。
**关注点**：两类是否只集中在启动或后期，以及低事件/runaway 是否完整保留。

### chronological_*_topo_6101_dyn_7101.gif
预先规定的数值最小 topology/dynamics 组合，按时间顺序覆盖全部检测窗；每个事件固定显示 −16、0、+16、+32、+64 ms 的原生场和冻结接触质心。没有按患者相似度挑漂亮事件，模式标签只用于组织观察。
**关注点**：原生传播是否出现患者允许的分支，接触质心偏早究竟对应起始、持续短还是长尾。

### chronological_*_topo_6101_dyn_7102.gif
同一 topology 的第二段动力学噪声重演，展示规则与代表组合完全相同。
**关注点**：固定图换噪声后，多事件能力与偏差是否保留；四次重演不足以分解全部随机效应。

### native_*_page_*.png / .pdf
所有四个 seed 组合的逐事件原生场固定抽帧页，最后一列为参与接触的质心时序；空心接触表示该事件不参与。
**关注点**：少数模式未观察时相应条件传播为 NOT_ESTIMABLE，不能把空白当作机制不存在。

### patient_model_distribution_summary.png / .pdf
左侧比较四次 G3 重演的平均模式比例与完整患者 FIT 表频率，右侧比较模式条件下的粗区域顺序与预先冻结的患者 32+32 包。虚线患者参照来自两种不同统计对象，图题与元数据明确区分。
**关注点**：平衡包不估计自然频率；区域中位质心顺序不是连续组织传播路径，也不是起源核证据。

### patient_packet_mode*_page_*.png / .pdf
按冻结顺序展示患者小包全部 32+32 个可读事件；左侧是既定 packed window 内的逐接触背景稳健 z 包络，右侧是原 producer 追溯出的参与接触质心。没有根据模型结果、路线或视觉相似度筛例。
**关注点**：窗口边界质量、包络持续/长尾与质心时序是否一致；患者接触帧不能当作连续组织原生场。
"""
    (FIGURES / "README.md").write_text(readme)
    return outputs


def main():
    candidates, runs, events, segments = analyze()
    patient_model_comparison(candidates, runs, events)
    render(candidates, runs, segments)
    print(json.dumps({"status": "G4_ANALYSIS_AND_VISUALS_READY_FOR_REVIEW",
                      "runs": len(runs), "primary_events": len(events)}, indent=2))


if __name__ == "__main__": main()
