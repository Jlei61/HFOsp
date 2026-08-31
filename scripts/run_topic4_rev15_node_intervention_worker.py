#!/usr/bin/env python3
"""Run one network of the crossed rev15 same-checkpoint intervention."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT, ROOT / "src" / "snn_engine"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts.rescore_topic4_rev12_node_historical import (  # noqa: E402
    _classifier_contract,
    _load_network_worker,
    _old_to_patient_label_map,
    _patient_data,
    _reorder_patient_contract,
)
from scripts.run_topic4_rev12_node_intervention import (  # noqa: E402
    _atomic_json,
    _atomic_npz,
    _branch_readout,
    _event_contract,
    _rank_distance,
    _source_bundle,
)
from scripts.topic4_rev15_node_substrate_adapter import (  # noqa: E402
    build_projected_node_substrate,
    verify_projection_against_worker,
)
from src.topic4_node_intervention import (  # noqa: E402
    grid_covariates,
    intervention_footprint_covariates,
    network_balanced_early_support,
    select_hotspot_triplet,
)
from src.topic4_node_dualmode import (  # noqa: E402
    cosine_similarity,
    source_topology_features,
)
from src.topic4_zm_ictal_transition import make_external_drive  # noqa: E402
from kick_probe import simulate_kick  # noqa: E402


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
EXPECTED_CONFIG_SCHEMA = "topic4_rev15_node_crossed_intervention_v1"
OUTPUT_SCHEMA = "topic4_rev15_node_crossed_intervention_worker_v1"
WORKER_STATUS = "REV15_NODE_CROSSED_INTERVENTION_WORKER_COMPLETE"
FINAL_AUDIT_ADVANCE_STATUS = "NODE_FINAL_SCIENCE_ADVANCES_TO_INTERVENTION"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(record: dict[str, str], artifact_root: Path) -> Path:
    for root in (artifact_root, ROOT):
        path = root / str(record["path"])
        if path.is_file() and _sha256(path) == str(record["sha256"]):
            return path.resolve()
    raise RuntimeError(f"intervention input changed: {record['path']}")


def _provenance(config_path: Path, expected_commit: str) -> dict[str, Any]:
    expected = subprocess.check_output(
        ["git", "rev-parse", expected_commit], cwd=ROOT, text=True,
    ).strip()
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT, text=True,
    ).splitlines()
    config_relative = config_path.resolve().relative_to(ROOT)
    committed = subprocess.check_output(
        ["git", "show", f"{expected}:{config_relative}"], cwd=ROOT,
    )
    config_matches = hashlib.sha256(committed).hexdigest() == _sha256(config_path)
    output = {
        "expected_commit": expected,
        "runtime_commit": head,
        "worktree_status": status,
        "config_matches_expected_commit": config_matches,
        "formal_ready": bool(head == expected and not status and config_matches),
    }
    if not output["formal_ready"]:
        raise RuntimeError("intervention worker runtime is not frozen")
    return output


def _context(config: dict[str, Any], paths: dict[str, Path],
             artifact_root: Path) -> dict[str, Any]:
    cohort = json.loads(paths["cohort_config"].read_text())
    classifier_config = json.loads(paths["classifier_config"].read_text())
    patient = _patient_data(cohort, artifact_root)
    classifier = _classifier_contract(classifier_config, artifact_root)
    semantics = _old_to_patient_label_map(patient, classifier)
    patient = _reorder_patient_contract(patient, classifier["names"])
    robust_config = json.loads(paths["robust_config"].read_text())
    return {
        "patient": patient,
        "classifier": classifier,
        "label_map": semantics["raw_to_patient"],
        "robust_config": robust_config,
    }


def _worker_path(robust_config: dict[str, Any], candidate_id: str,
                 seed: int, artifact_root: Path) -> Path:
    return artifact_root / robust_config["output_root"] / "workers" / (
        f"{candidate_id}_seed_{int(seed)}.npz"
    )


def _load_all_networks(
    config: dict[str, Any], context: dict[str, Any], artifact_root: Path,
) -> tuple[dict[int, tuple[Path, dict]], list[np.ndarray], list[np.ndarray]]:
    workers, maps_by_network, labels_by_network = {}, [], []
    candidate_id = str(config["candidate_id"])
    for seed in config["network_seeds"]:
        path = _worker_path(
            context["robust_config"], candidate_id, int(seed), artifact_root,
        )
        worker = _load_network_worker(
            path, context["patient"]["contact_names"], context["classifier"],
            context["label_map"],
        )
        maps, labels = _source_bundle(path, worker)
        if set(np.unique(labels).tolist()) != {0, 1}:
            raise RuntimeError(f"intervention seed {seed} lacks one source-evaluable mode")
        workers[int(seed)] = (path, worker)
        maps_by_network.append(maps)
        labels_by_network.append(labels)
    return workers, maps_by_network, labels_by_network


def _final_worker_hashes(final_audit: dict[str, Any]) -> dict[int, str]:
    return {
        int(row["seed"]): str(row["npz"]["sha256"])
        for row in final_audit["inputs"]["candidate_workers"]
    }


def run_worker(
    *, config_path: Path, seed: int, expected_commit: str,
    artifact_root: Path = ARTIFACT_ROOT,
    out_json: Path | None = None, out_npz: Path | None = None,
) -> dict[str, Any]:
    started = time.time()
    config_path = config_path.resolve()
    artifact_root = artifact_root.resolve()
    config = json.loads(config_path.read_text())
    if config.get("schema_id") != EXPECTED_CONFIG_SCHEMA:
        raise RuntimeError("intervention config schema changed")
    if int(seed) not in {int(value) for value in config["network_seeds"]}:
        raise RuntimeError("intervention seed is outside the frozen pool")
    if config["mechanism_freeze"] != {
        "EE": "off", "E_to_I": "off", "Z_M": "off",
    }:
        raise RuntimeError("intervention config activated another mechanism")
    provenance = _provenance(config_path, expected_commit)
    paths = {
        key: _resolve(record, artifact_root)
        for key, record in config["inputs"].items()
    }
    final_audit = json.loads(paths["final_science_audit"].read_text())
    if final_audit.get("status") != FINAL_AUDIT_ADVANCE_STATUS:
        raise RuntimeError("final science audit no longer advances")
    if final_audit.get("candidate_id") != config["candidate_id"]:
        raise RuntimeError("intervention candidate changed")
    context = _context(config, paths, artifact_root)
    workers, maps_by_network, labels_by_network = _load_all_networks(
        config, context, artifact_root,
    )
    npz_path, worker = workers[int(seed)]
    expected_hashes = _final_worker_hashes(final_audit)
    if _sha256(npz_path) != expected_hashes[int(seed)]:
        raise RuntimeError("intervention worker differs from final-science input")
    event_contracts = {
        mode: _event_contract(npz_path, worker, mode) for mode in (0, 1)
    }
    substrate, projection, transition = build_projected_node_substrate(
        robust_config_path=paths["robust_config"],
        candidate_id=str(config["candidate_id"]), seed=int(seed),
        artifact_root=artifact_root,
    )
    projection_parity = verify_projection_against_worker(projection, npz_path)
    substrate.params.T = float(context["robust_config"]["search"]["simulation"]["duration_ms"])
    substrate.net["rng"] = np.random.default_rng(int(seed))
    dt_ms = float(substrate.engine["dt"])
    protocol = config["intervention"]
    checkpoint_lead = float(protocol["checkpoint_lead_ms"])
    checkpoint_steps = {
        int(round((row["event_t_on_ms"] - checkpoint_lead) / dt_ms))
        for row in event_contracts.values()
    }
    if len(checkpoint_steps) != 2 or min(checkpoint_steps) <= 0:
        raise RuntimeError("intervention events do not define two valid checkpoints")
    checkpoints: dict[int, dict] = {}
    baseline = simulate_kick(
        substrate.params, substrate.net, KICK_BOOST=0.0, t_kick=1e9,
        V_th_per_neuron=substrate.vtheta, slow=None,
        early_stop_runaway=False,
        external_e_rate_drive=make_external_drive(
            substrate, transition["spatial_ou"], int(seed),
        ),
        checkpoint_steps=checkpoint_steps,
        checkpoint_sink=lambda step, state: checkpoints.setdefault(step, state),
    )
    if set(checkpoints) != checkpoint_steps:
        raise RuntimeError("intervention baseline replay missed a checkpoint")
    full_spikes = np.asarray(baseline["E_spk_bool"], dtype=bool)
    active, _ = substrate.extras["cmrun"].active_fraction(
        full_spikes, dt_ms, substrate.extras["cmrun"].BIN_MS,
    )
    with np.load(npz_path, allow_pickle=False) as loaded:
        stored_active = np.asarray(loaded["active_fraction"], dtype=np.float32)
    if not np.array_equal(np.asarray(active, dtype=np.float32), stored_active):
        raise RuntimeError("intervention baseline replay differs from robust worker")

    baseline_window = protocol["baseline_activity_window_ms"]
    low = int(round(float(baseline_window[0]) / dt_ms))
    high = int(round(float(baseline_window[1]) / dt_ms))
    hotspot_contract = config["hotspot_construction"]
    if hotspot_contract.get("leave_one_network_out", False):
        peer_indices = [
            index for index, peer_seed in enumerate(config["network_seeds"])
            if int(peer_seed) != int(seed)
        ]
        if len(peer_indices) < 2:
            raise RuntimeError("leave-one-network-out hotspot needs two peer networks")
    else:
        peer_indices = list(range(len(maps_by_network)))
    templates = {
        mode: network_balanced_early_support(
            [maps_by_network[index] for index in peer_indices],
            [labels_by_network[index] for index in peer_indices], mode,
            fraction=float(hotspot_contract.get("early_support_fraction", 0.10)),
        ) for mode in (0, 1)
    }

    if hotspot_contract.get("covariate_footprint") == "pulse_target_disk":
        covariates = intervention_footprint_covariates(
            substrate.positions_e, substrate.h_e, full_spikes[low:high],
            dt_ms=dt_ms, sheet_mm=float(substrate.engine["L"]), bin_mm=1.0,
            target_radius_mm=float(protocol["target_radius_mm"]),
            additional_node_covariates=(
                {"delta_vtheta_mean": np.asarray(substrate.delta_vtheta, float)}
                if "delta_vtheta_mean" in hotspot_contract.get(
                    "matching_covariate_keys", []
                ) else None
            ),
        )
    else:
        covariates = grid_covariates(
            substrate.positions_e, substrate.h_e, full_spikes[low:high],
            dt_ms=dt_ms, sheet_mm=float(substrate.engine["L"]), bin_mm=1.0,
        )
    discriminative = bool(hotspot_contract.get("mode_discriminative_contrast", False))
    targets = {
        mode: select_hotspot_triplet(
            templates[mode], covariates, bin_mm=1.0,
            minimum_separation_mm=float(
                hotspot_contract["minimum_hotspot_separation_mm"]
            ),
            competing_probability=templates[1 - mode] if discriminative else None,
            require_positive_contrast=discriminative,
            maximum_standardized_l1=hotspot_contract.get(
                "maximum_control_standardized_l1"
            ),
            maximum_standardized_component=hotspot_contract.get(
                "maximum_control_standardized_component"
            ),
            covariate_keys=hotspot_contract.get("matching_covariate_keys"),
        ) for mode in (0, 1)
    }
    hotspot_distance = float(np.linalg.norm(
        np.asarray(targets[0]["dominant"]["xy_mm"], float)
        - np.asarray(targets[1]["dominant"]["xy_mm"], float)
    ))
    hotspots_distinct = bool(
        hotspot_distance >= float(hotspot_contract["minimum_hotspot_separation_mm"])
    )
    arm_targets = {
        "sham": None,
        "mode0_hotspot": targets[0]["dominant"],
        "mode0_matched_off_template": targets[0]["matched_off_template"],
        "mode1_hotspot": targets[1]["dominant"],
        "mode1_matched_off_template": targets[1]["matched_off_template"],
    }
    native_modes, output_arrays = {}, {}
    for native_mode in (0, 1):
        contract = event_contracts[native_mode]
        step = int(round((contract["event_t_on_ms"] - checkpoint_lead) / dt_ms))
        checkpoint = checkpoints[step]
        branch_records, branch_spikes = {}, {}
        for arm, target in arm_targets.items():
            record, arrays, spikes = _branch_readout(
                substrate, transition, checkpoint, full_spikes, contract, target,
                duration_ms=float(protocol["continuation_ms"]),
                pulse_delay_ms=float(protocol["pulse_delay_from_checkpoint_ms"]),
                pulse_duration_ms=float(protocol["pulse_duration_ms"]),
                pulse_delta_vtheta=float(protocol["pulse_delta_vtheta_mv"]),
                target_radius_mm=float(protocol["target_radius_mm"]),
                classifier=context["classifier"], label_map=context["label_map"],
                contact_names=context["patient"]["contact_names"],
                maximum_event_shift_ms=float(protocol["maximum_event_shift_ms"]),
            )
            branch_records[arm] = record
            branch_spikes[arm] = spikes
            for key, value in arrays.items():
                output_arrays[f"mode{native_mode}_{arm}_{key}"] = value
        for target_mode in (0, 1):
            branch_records[f"mode{target_mode}_matched_off_template"][
                "control_match_acceptable"
            ] = bool(targets[target_mode]["match_quality"]["acceptable"])
        sham = branch_records["sham"]
        if not sham["event_occurred"]:
            raise RuntimeError("same-checkpoint sham did not reproduce the native event")
        continuation_steps = int(round(float(protocol["continuation_ms"]) / dt_ms))
        native = full_spikes[step:step + continuation_steps]
        if not np.array_equal(branch_spikes["sham"], native):
            raise RuntimeError("same-checkpoint sham differs from native continuation")
        pulse_step = int(round(
            float(protocol["pulse_delay_from_checkpoint_ms"]) / dt_ms,
        ))
        for arm in arm_targets:
            if arm == "sham":
                continue
            parity = bool(np.array_equal(
                branch_spikes[arm][:pulse_step], branch_spikes["sham"][:pulse_step],
            ))
            branch_records[arm]["pre_intervention_spike_parity"] = parity
            if not parity:
                raise RuntimeError(f"{arm} diverged before the threshold pulse")
            if branch_records[arm]["event_occurred"]:
                branch_records[arm]["rank_distance_from_sham"] = _rank_distance(
                    output_arrays[f"mode{native_mode}_{arm}_rank"],
                    output_arrays[f"mode{native_mode}_sham_rank"],
                    context["patient"]["contact_names"],
                )
                branch_records[arm]["source_topology_cosine_to_sham"] = cosine_similarity(
                    source_topology_features(output_arrays[
                        f"mode{native_mode}_{arm}_source_onset_map_ms"
                    ][None, ...])[0],
                    source_topology_features(output_arrays[
                        f"mode{native_mode}_sham_source_onset_map_ms"
                    ][None, ...])[0],
                )
        native_modes[str(native_mode)] = {
            "event_contract": contract,
            "cross_mode_hotspots_distinct": hotspots_distinct,
            "cross_mode_hotspot_distance_mm": hotspot_distance,
            **branch_records,
        }

    if out_json is None or out_npz is None:
        output_root = artifact_root / config["output_root"] / "workers"
        out_json = output_root / f"intervention_seed_{int(seed)}.json"
        out_npz = output_root / f"intervention_seed_{int(seed)}.npz"
    output_arrays.update({
        "mode0_early_probability": templates[0].astype(np.float32),
        "mode1_early_probability": templates[1].astype(np.float32),
        "positions_E": np.asarray(substrate.positions_e, dtype=np.float32),
        "h": np.asarray(substrate.h_e, dtype=np.float64),
        "vtheta": np.asarray(substrate.vtheta, dtype=np.float64),
        "delta_vtheta": np.asarray(substrate.delta_vtheta, dtype=np.float64),
    })
    _atomic_npz(out_npz, **output_arrays)
    payload = {
        "schema_id": OUTPUT_SCHEMA,
        "status": WORKER_STATUS,
        "candidate_id": config["candidate_id"],
        "network_seed": int(seed),
        "native_modes": native_modes,
        "targets": targets,
        "target_template_network_seeds": [
            int(config["network_seeds"][index]) for index in peer_indices
        ],
        "target_template_leave_one_network_out": bool(
            hotspot_contract.get("leave_one_network_out", False)
        ),
        "projection_parity": projection_parity,
        "arrays": {"path": str(out_npz), "sha256": _sha256(out_npz)},
        "inputs": {
            "config": {"path": str(config_path), "sha256": _sha256(config_path)},
            "robust_worker": {"path": str(npz_path), "sha256": _sha256(npz_path)},
            "hashed_inputs": config["inputs"],
        },
        "provenance": provenance,
        "mechanism_freeze": config["mechanism_freeze"],
        "wall_seconds": float(time.time() - started),
        "claim_boundary": config["claim_boundary"],
    }
    _atomic_json(out_json, payload)
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-npz", type=Path)
    args = parser.parse_args(argv)
    payload = run_worker(
        config_path=args.config, seed=args.seed,
        expected_commit=args.expected_commit,
        artifact_root=args.artifact_root,
        out_json=args.out_json, out_npz=args.out_npz,
    )
    print(json.dumps({
        "status": payload["status"],
        "network_seed": payload["network_seed"],
        "candidate_id": payload["candidate_id"],
    }, indent=2))


if __name__ == "__main__":
    main()
