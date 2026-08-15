#!/usr/bin/env python3
"""Audit the frozen rev11-NLC pathway-mechanism confirmation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_topic4_rev11_nlc_joint_node_connectivity_selection import (  # noqa: E402
    _network_rows,
)
from scripts.rescore_topic4_rev10_d6_natural_kmeans import (  # noqa: E402
    _candidate_metrics,
    _jsonable,
)
from scripts.rescore_topic4_rev10_sa_historical_artifacts import (  # noqa: E402
    load_scoring_contract,
)
from scripts.run_topic4_rev9l_forced_source_worker import _sha256  # noqa: E402
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_nlc_pathway_mechanism import (  # noqa: E402
    ARM_IDS,
    MODE_NAMES,
    bootstrap_mean,
    event_aligned_pathway_readout,
    factorial_bootstrap,
    formal_mode_assignments,
    network_mode_endpoints,
    paired_bootstrap,
)
from src.topic4_shaft_aware import contract_groups  # noqa: E402


DEFAULT_CONFIG = (
    ROOT / "config/topic4_rev11_nlc_pathway_mechanism_confirmation.json"
)
EXPECTED_ROLE = (
    "development_only_data_driven_node_local_connectivity_mechanism_confirmation"
)
EXPECTED_MANIFEST_STATUS = (
    "REV11NLC_PATHWAY_MECHANISM_CONFIRMATION_LIBRARY_FROZEN"
)
TRACE_NAMES = (
    "population_rate_E_hz",
    "population_rate_I_hz",
    "recurrent_E_to_E_mean",
    "recurrent_E_to_I_mean",
    "GABA_to_E_mean",
)
PRIMARY_ENDPOINTS = (
    "TA_like_rate_hz",
    "TB_like_rate_hz",
    "TB_like_fraction",
    "ood_fraction_returned",
    "natural_alignment",
    "crossfit_patient_margin",
    "shape_A",
    "shape_B",
)


def _classifier(manifest):
    classifier = dict(manifest["direction_classifier"])
    for key in (
        "coef", "class_centers", "class_precisions", "ood_distance_thresholds",
    ):
        classifier[key] = np.asarray(classifier[key], float)
    return classifier


def _trace_arrays(loaded):
    return {
        "population_rate_E_hz": np.asarray(
            loaded["mechanism_population_rate_E_hz"], float,
        ),
        "population_rate_I_hz": np.asarray(
            loaded["mechanism_population_rate_I_hz"], float,
        ),
        "recurrent_E_to_E_mean": np.asarray(
            loaded["mechanism_recurrent_E_to_E_mean"], float,
        ),
        "recurrent_E_to_I_mean": np.asarray(
            loaded["mechanism_recurrent_E_to_I_mean"], float,
        ),
        "GABA_to_E_mean": np.asarray(
            loaded["mechanism_GABA_to_E_mean"], float,
        ),
    }


def _window_values(aligned):
    output = {}
    for mode_name, mode in aligned["modes"].items():
        for trace_name, trace in mode["traces"].items():
            for window_name, value in trace["windows"].items():
                output[f"{mode_name}|{trace_name}|{window_name}"] = value
    return output


def _aggregate_curves(curves_by_arm):
    output = {}
    for arm_id, by_mode in curves_by_arm.items():
        output[arm_id] = {}
        for mode_name, networks in by_mode.items():
            output[arm_id][mode_name] = {}
            for trace_name in TRACE_NAMES:
                rows = [
                    np.asarray(row["traces"][trace_name]["mean"], float)
                    for row in networks if row["n_events"] > 0
                ]
                if not rows:
                    output[arm_id][mode_name][trace_name] = {
                        "n_networks": 0, "mean": None,
                        "network_q05": None, "network_q95": None,
                    }
                    continue
                values = np.asarray(rows, float)
                output[arm_id][mode_name][trace_name] = {
                    "n_networks": int(len(values)),
                    "mean": np.nanmean(values, axis=0).tolist(),
                    "network_q05": np.nanquantile(values, 0.05, axis=0).tolist(),
                    "network_q95": np.nanquantile(values, 0.95, axis=0).tolist(),
                }
    return output


def audit(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if config.get("scientific_role") != EXPECTED_ROLE:
        raise RuntimeError("pathway mechanism scientific role changed")
    root = ROOT / config["output_root"]
    manifest_path = root / "candidate_manifest.json"
    summary_path = root / "confirmation_summary_returned_only.json"
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())
    if manifest.get("status") != EXPECTED_MANIFEST_STATUS:
        raise RuntimeError("pathway mechanism manifest is not frozen")
    if summary.get("scientific_role") != EXPECTED_ROLE:
        raise RuntimeError("pathway mechanism aggregate has another role")

    contract = json.loads(
        (ROOT / config["inputs"]["contact_contract"]["path"]).read_text()
    )
    groups = contract_groups(contract)
    names, embedding, _, _ = load_scoring_contract(
        config["inputs"]["shaft_aware_target_npz"]["path"],
        config["inputs"]["shaft_aware_floors"]["path"],
        "FULL_TIMING", fixed_events_per_mode=6,
    )
    classifier = _classifier(manifest)
    candidates = {
        row["candidate_id"]: row
        for row in manifest["candidate_set"]["candidates"]
    }
    if tuple(candidates) != ARM_IDS:
        raise RuntimeError("pathway mechanism arm order changed")
    seeds = list(map(int, config["search"]["confirmation_network_seeds"]))
    duration_ms = float(config["search"]["simulation"]["duration_ms"])
    readout = config["mechanism_readout"]
    bootstrap = config["search"]["paired_network_bootstrap"]
    draws, bootstrap_seed = int(bootstrap["draws"]), int(bootstrap["seed"])
    summary_details = summary["candidate_details"]

    rows_by_arm, curves_by_arm, worker_inputs, invalid_workers = {}, {}, [], []
    for arm_index, arm_id in enumerate(ARM_IDS):
        candidate = candidates[arm_id]
        metrics = _candidate_metrics(
            config_path, root, candidate, contract,
            config["search"]["kmeans_selection"],
        )
        geometry = _network_rows(metrics, summary_details[arm_id])
        arm_rows = {}
        curves_by_arm[arm_id] = {name: [] for name in MODE_NAMES.values()}
        for seed in seeds:
            stem = root / "workers" / f"{arm_id}_seed_{seed}"
            json_path, npz_path = stem.with_suffix(".json"), stem.with_suffix(".npz")
            payload = json.loads(json_path.read_text())
            problems = []
            if payload.get("status") != "REV10R_EDGE_FLOW_WORKER_COMPLETE":
                problems.append("worker_status")
            if payload.get("run", {}).get("runaway_early_stop_ms") is not None:
                problems.append("late_runaway")
            if payload.get("mechanism_readout", {}).get("trace_samples", 0) <= 0:
                problems.append("missing_pathway_trace")
            with np.load(npz_path, allow_pickle=False) as loaded:
                worker_names = np.asarray(loaded["contact_names"]).astype(str)
                onsets = np.asarray(loaded["onsets"], float)
                returned = np.asarray(loaded["event_returned"], bool)
                event_onsets = np.asarray(loaded["event_t_on_ms"], float)
                time_ms = np.asarray(loaded["mechanism_time_ms"], float)
                traces = _trace_arrays(loaded)
            if not np.array_equal(worker_names, names.astype(str)):
                problems.append("contact_order")
            expected_samples = int(round(duration_ms / readout["trace_dt_ms"]))
            if len(time_ms) != expected_samples:
                problems.append("pathway_trace_length")
            if any(len(values) != len(time_ms) or not np.isfinite(values).all()
                   for values in traces.values()):
                problems.append("pathway_trace_nonfinite")
            if problems:
                invalid_workers.append({
                    "candidate_id": arm_id, "seed": seed, "problems": problems,
                })

            assignments = formal_mode_assignments(
                onsets, returned, groups=groups, embedding=embedding,
                classifier=classifier,
                minimum_recruited_contacts=readout[
                    "minimum_recruited_contacts"
                ],
            )
            assignments["returned"] = returned
            endpoints = network_mode_endpoints(assignments, duration_ms)
            aligned = event_aligned_pathway_readout(
                time_ms, traces, event_onsets,
                assignments["labels"], assignments["clean"],
                event_window_ms=readout["event_window_ms"],
                baseline_window_ms=readout["baseline_window_ms"],
                summary_windows_ms=readout["summary_windows_ms"],
                trace_dt_ms=readout["trace_dt_ms"],
            )
            for mode_name in MODE_NAMES.values():
                curves_by_arm[arm_id][mode_name].append(
                    aligned["modes"][mode_name]
                )
            geo = geometry.get(str(seed), {})
            detail = summary_details[arm_id]["by_seed"][str(seed)]
            arm_rows[str(seed)] = {
                **endpoints,
                "natural_alignment": geo.get("natural"),
                "crossfit_patient_margin": geo.get("crossfit"),
                "shape_A": detail["shape_by_mode"]["A"],
                "shape_B": detail["shape_by_mode"]["B"],
                "pathway_windows": _window_values(aligned),
            }
            worker_inputs.append({
                "candidate_id": arm_id,
                "seed": seed,
                "json": str(json_path.relative_to(ROOT)),
                "json_sha256": _sha256(json_path),
                "npz": str(npz_path.relative_to(ROOT)),
                "npz_sha256": _sha256(npz_path),
            })
        rows_by_arm[arm_id] = arm_rows

    endpoint_names = list(PRIMARY_ENDPOINTS)
    window_names = sorted({
        key for arm in rows_by_arm.values()
        for row in arm.values() for key in row["pathway_windows"]
    })
    summaries, comparisons, interactions = {}, {}, {}
    for arm_index, arm_id in enumerate(ARM_IDS):
        summaries[arm_id] = {}
        for endpoint_index, endpoint in enumerate(endpoint_names):
            values = [rows_by_arm[arm_id][str(seed)][endpoint] for seed in seeds]
            summaries[arm_id][endpoint] = bootstrap_mean(
                values, draws=draws,
                seed=bootstrap_seed + 100 * arm_index + endpoint_index,
            )
        for window_index, window in enumerate(window_names):
            values = [
                rows_by_arm[arm_id][str(seed)]["pathway_windows"].get(window)
                for seed in seeds
            ]
            summaries[arm_id][window] = bootstrap_mean(
                values, draws=draws,
                seed=bootstrap_seed + 1000 + 100 * arm_index + window_index,
            )

    baseline_id = "node_baseline"
    for arm_index, arm_id in enumerate(ARM_IDS[1:], start=1):
        comparisons[arm_id] = {}
        for endpoint_index, endpoint in enumerate(endpoint_names + window_names):
            left = [
                rows_by_arm[arm_id][str(seed)].get(
                    endpoint,
                    rows_by_arm[arm_id][str(seed)]["pathway_windows"].get(endpoint),
                ) for seed in seeds
            ]
            right = [
                rows_by_arm[baseline_id][str(seed)].get(
                    endpoint,
                    rows_by_arm[baseline_id][str(seed)]["pathway_windows"].get(endpoint),
                ) for seed in seeds
            ]
            comparisons[arm_id][endpoint] = paired_bootstrap(
                left, right, draws=draws,
                seed=bootstrap_seed + 2000 + 100 * arm_index + endpoint_index,
            )

    for endpoint_index, endpoint in enumerate(endpoint_names + window_names):
        values = {}
        for arm_id in ARM_IDS:
            values[arm_id] = [
                rows_by_arm[arm_id][str(seed)].get(
                    endpoint,
                    rows_by_arm[arm_id][str(seed)]["pathway_windows"].get(endpoint),
                ) for seed in seeds
            ]
        interactions[endpoint] = factorial_bootstrap(
            values[ARM_IDS[0]], values[ARM_IDS[1]],
            values[ARM_IDS[2]], values[ARM_IDS[3]],
            draws=draws, seed=bootstrap_seed + 4000 + endpoint_index,
        )

    payload = {
        "status": "REV11NLC_PATHWAY_MECHANISM_CONFIRMATION_COMPLETE",
        "scientific_state": (
            "PATHWAY_EFFECT_PATTERN_ESTIMATED"
            if not invalid_workers else "ENGINEERING_INVALID"
        ),
        "figure_eligible": not invalid_workers,
        "network_seed_is_the_independent_unit": True,
        "mode_label_contract": {
            "classifier_0": "TA-like",
            "classifier_1": "TB-like",
            "natural_clusters": ["C1", "C2"],
            "patient_classifier_refit": False,
        },
        "event_filter": readout["formal_event_filter"],
        "network_seeds": seeds,
        "per_network": rows_by_arm,
        "equal_network_summaries": summaries,
        "paired_differences_vs_node": comparisons,
        "factorial_interactions": interactions,
        "event_aligned_curves": {
            "relative_time_ms": np.arange(
                readout["event_window_ms"][0],
                readout["event_window_ms"][1] + 0.5 * readout["trace_dt_ms"],
                readout["trace_dt_ms"],
            ).tolist(),
            "arms": _aggregate_curves(curves_by_arm),
        },
        "engineering_audit": {
            "n_expected_workers": len(ARM_IDS) * len(seeds),
            "n_invalid_workers": len(invalid_workers),
            "invalid_workers": invalid_workers,
        },
        "inputs": {
            "config": {"path": str(config_path.relative_to(ROOT)), "sha256": _sha256(config_path)},
            "manifest": {"path": str(manifest_path.relative_to(ROOT)), "sha256": _sha256(manifest_path)},
            "summary": {"path": str(summary_path.relative_to(ROOT)), "sha256": _sha256(summary_path)},
            "workers": worker_inputs,
        },
        "claim_boundary": config["claim_boundary"],
    }
    return _jsonable(payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    payload = audit(args.config)
    config = json.loads(Path(args.config).read_text())
    output = ROOT / config["output_root"] / "mechanism_verdict.json"
    atomic_write_json(payload, output)
    print(json.dumps({
        "status": payload["status"],
        "scientific_state": payload["scientific_state"],
        "figure_eligible": payload["figure_eligible"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
