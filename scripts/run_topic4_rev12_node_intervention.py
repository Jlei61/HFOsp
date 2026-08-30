#!/usr/bin/env python3
"""Replay selected rev12 Node events under same-checkpoint local suppression."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src" / "snn_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.rescore_topic4_rev12_node_historical import (  # noqa: E402
    _classifier_contract,
    _load_network_worker,
    _old_to_patient_label_map,
    _patient_data,
    _reorder_patient_contract,
)
from scripts import rescore_topic4_rev14_static_node_historical_libraries as historical  # noqa: E402
from scripts.run_topic4_rev10_sa_spectral_field_worker import _contact_onsets  # noqa: E402
from scripts.run_topic4_rev9l_forced_source_worker import _runtime_provenance  # noqa: E402
from src.topic4_node_dualmode import (  # noqa: E402
    cosine_similarity,
    event_features,
    event_source_onset_maps,
    normalize_event_ranks,
    shaft_balanced_feature_weights,
    source_topology_features,
)
from src.topic4_node_intervention import (  # noqa: E402
    grid_covariates,
    network_balanced_early_support,
    representative_event_index,
    select_hotspot_triplet,
    select_representative_seed,
)
from src.topic4_shaft_aware_direction import assign_direction_modes  # noqa: E402
from src.topic4_zm_ictal_transition import (  # noqa: E402
    build_substrate,
    load_round_config,
    make_external_drive,
)
from kick_probe import simulate_kick  # noqa: E402
from src.sef_hfo_events import detect_events  # noqa: E402
from src.sef_hfo_snn_adapter import snn_event_envelope  # noqa: E402


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(artifact_root: Path, relative: str) -> Path:
    local = ROOT / relative
    return local if local.exists() else artifact_root / relative


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(
            json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n"
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(handle)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _formal_clean_source_rows(
    arrays: dict[str, np.ndarray], worker: dict, detected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Align source-evaluable families to returned, patient-supported events."""
    required = ("labels", "ranks", "formal_clean")
    if any(key not in worker for key in required):
        raise RuntimeError("source worker lacks the formal-clean event contract")
    returned_indices = np.flatnonzero(np.asarray(arrays["event_returned"], bool))
    detected = np.asarray(detected, dtype=int)
    positions = np.searchsorted(returned_indices, detected)
    labels = np.asarray(worker.get("labels"), dtype=int)
    ranks = np.asarray(worker.get("ranks"), dtype=float)
    formal_clean = np.asarray(worker.get("formal_clean"), dtype=bool)
    if (
        np.any(positions >= len(returned_indices))
        or not np.array_equal(returned_indices[positions], detected)
        or labels.shape != (len(returned_indices),)
        or ranks.ndim != 2
        or ranks.shape[0] != len(returned_indices)
        or formal_clean.shape != (len(returned_indices),)
    ):
        raise RuntimeError(
            "source families, returned labels and formal-clean support differ"
        )
    keep = formal_clean[positions]
    return detected[keep], positions[keep]


def _source_bundle(npz_path: Path, worker: dict) -> tuple[np.ndarray, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as loaded:
        keys = (
            "event_returned", "source_onset_evaluable", "event_t_on_ms",
            "event_trigger_t_on_ms", "event_t_off_ms", "event_fragment_count",
            "event_directed_root_id", "event_root_count", "onsets", "ranks",
            "source_onset_maps_ms", "positions_E", "delta_vtheta", "source_bin_mm",
        )
        arrays = {key: np.asarray(loaded[key]) for key in keys}
    selection = historical.three_layer_event_selection(
        arrays, minimum_readable_contacts=3,
    )
    detected, positions = _formal_clean_source_rows(
        arrays, worker,
        np.asarray(selection["topology_primary_indices"], dtype=int),
    )
    return (
        np.asarray(arrays["source_onset_maps_ms"], float)[detected],
        np.asarray(worker["labels"], int)[positions],
    )


def _event_contract(npz_path: Path, worker: dict, mode: int) -> dict:
    with np.load(npz_path, allow_pickle=False) as loaded:
        keys = (
            "event_returned", "source_onset_evaluable", "event_t_on_ms",
            "event_trigger_t_on_ms", "event_t_off_ms", "event_fragment_count",
            "event_directed_root_id", "event_root_count", "onsets", "ranks",
            "source_onset_maps_ms", "positions_E", "delta_vtheta", "source_bin_mm",
        )
        arrays = {key: np.asarray(loaded[key]) for key in keys}
        selection = historical.three_layer_event_selection(
            arrays, minimum_readable_contacts=3,
        )
        detected_indices, returned_positions = _formal_clean_source_rows(
            arrays, worker,
            np.asarray(selection["topology_primary_indices"], dtype=int),
        )
        maps = np.asarray(arrays["source_onset_maps_ms"], float)[detected_indices]
        ranks = np.asarray(worker["ranks"], float)[returned_positions]
        labels = np.asarray(worker["labels"], int)[returned_positions]
        local = representative_event_index(maps, ranks, labels, int(mode))
        returned_position = int(returned_positions[local])
        detected_index = int(detected_indices[local])
        return {
            "mode": int(mode),
            "detected_event_index": detected_index,
            "returned_event_position": returned_position,
            "event_t_on_ms": float(arrays["event_t_on_ms"][detected_index]),
            "event_t_off_ms": float(arrays["event_t_off_ms"][detected_index]),
            "reference_rank": np.asarray(worker["ranks"][returned_position], float),
            "reference_source_map": np.asarray(
                arrays["source_onset_maps_ms"][detected_index], float,
            ),
        }


def _select_branch_event(events: list[dict], expected_relative_ms: float,
                         maximum_shift_ms: float) -> dict | None:
    returned = [event for event in events if bool(event["returned"])]
    if not returned:
        return None
    selected = min(returned, key=lambda event: abs(event["t_on"] - expected_relative_ms))
    return (
        selected if abs(float(selected["t_on"]) - float(expected_relative_ms))
        <= float(maximum_shift_ms) else None
    )


def _rank_distance(left: np.ndarray, right: np.ndarray,
                   contact_names: np.ndarray) -> float:
    features = event_features(normalize_event_ranks(np.asarray([left, right], float)))
    weights = shaft_balanced_feature_weights(contact_names)
    return float(np.sqrt(np.sum(weights * (features[0] - features[1]) ** 2)))


def _classify_onsets(onsets: np.ndarray, classifier: dict,
                     label_map: np.ndarray) -> tuple[int, bool]:
    assigned = assign_direction_modes(
        np.asarray(onsets, float)[None, :], groups=classifier["groups"],
        embedding=classifier["embedding"], classifier=classifier["classifier"],
    )
    raw = int(np.asarray(assigned["labels"], int)[0])
    return int(np.asarray(label_map, int)[raw]), bool(np.asarray(assigned["ood"], bool)[0])


def _continue(substrate, transition: dict, checkpoint: dict, *, duration_ms: float,
              perturb: dict | None) -> dict:
    params = copy.deepcopy(substrate.params)
    params.T = float(duration_ms)
    drive = make_external_drive(substrate, transition["spatial_ou"], int(params.seed))
    return simulate_kick(
        params, substrate.net, KICK_BOOST=0.0, t_kick=1e9,
        V_th_per_neuron=substrate.vtheta, slow=None,
        external_e_rate_drive=drive,
        resume_state=copy.deepcopy(checkpoint),
        time_offset_ms=float(checkpoint["absolute_time_ms"]),
        perturb=perturb,
    )


def _branch_readout(substrate, transition: dict, checkpoint: dict,
                    full_spikes: np.ndarray, event_contract: dict,
                    target: dict | None, *, duration_ms: float,
                    pulse_delay_ms: float, pulse_duration_ms: float,
                    pulse_delta_vtheta: float, target_radius_mm: float,
                    classifier: dict, label_map: np.ndarray,
                    contact_names: np.ndarray,
                    maximum_event_shift_ms: float = 100.0,
                    ) -> tuple[dict, dict, np.ndarray]:
    offset = float(checkpoint["absolute_time_ms"])
    target_mask = None
    perturb = None
    if target is not None:
        center = np.asarray(target["xy_mm"], float)
        all_positions = np.asarray(substrate.net["pos"], float)
        target_mask = np.zeros(len(all_positions), bool)
        target_mask[:substrate.n_e] = np.linalg.norm(
            all_positions[:substrate.n_e] - center[None, :], axis=1,
        ) <= float(target_radius_mm)
        if not np.any(target_mask):
            raise RuntimeError("intervention target contains no E neuron")
        perturb = {
            "kind": "inhibitory_pulse",
            "t0": offset + float(pulse_delay_ms),
            "t1": offset + float(pulse_delay_ms) + float(pulse_duration_ms),
            "val": float(pulse_delta_vtheta),
            "target_mask": target_mask,
        }
    result = _continue(
        substrate, transition, checkpoint, duration_ms=duration_ms,
        perturb=perturb,
    )
    spikes = np.asarray(result["E_spk_bool"], bool)
    active, active_dt = substrate.extras["cmrun"].active_fraction(
        spikes, float(substrate.engine["dt"]), substrate.extras["cmrun"].BIN_MS,
    )
    events = detect_events(active, active_dt, event_on_frac=substrate.detector_threshold)
    event = _select_branch_event(
        events,
        event_contract["event_t_on_ms"] - offset,
        maximum_shift_ms=float(maximum_event_shift_ms),
    )
    arrays = {"active_fraction": np.asarray(active, np.float32)}
    record = {
        "event_occurred": event is not None,
        "target_neuron_count": 0 if target_mask is None else int(np.sum(target_mask)),
        "event_t_on_ms": None,
        "event_t_off_ms": None,
        "latency_from_checkpoint_ms": None,
        "patient_mode": None,
        "ood": None,
        "rank_distance_from_sham": None,
        "source_topology_cosine_to_sham": None,
    }
    if event is None:
        return record, arrays, spikes

    envelope, envelope_dt, _ = snn_event_envelope(
        spikes, substrate.positions_e, substrate.montage,
        float(substrate.engine["dt"]),
    )
    onset, rank = _contact_onsets(
        envelope, envelope_dt, substrate.montage, substrate.valid_contacts,
        (float(event["t_on"]), float(event["t_off"])),
        0.1, 0.5,
    )
    order = np.asarray([
        int(np.flatnonzero(np.asarray(substrate.contact_names).astype(str) == name)[0])
        for name in np.asarray(contact_names).astype(str)
    ])
    onset, rank = np.asarray(onset, float)[order], np.asarray(rank, float)[order]
    patient_mode, ood = _classify_onsets(onset, classifier, label_map)
    absolute_onset = offset + float(event["t_on"])
    checkpoint_step = int(round(offset / float(substrate.engine["dt"])))
    spliced = np.concatenate([full_spikes[:checkpoint_step], spikes], axis=0)
    source = event_source_onset_maps(
        spliced, substrate.positions_e, np.asarray([absolute_onset]),
        np.asarray([True]), dt_ms=float(substrate.engine["dt"]),
        sheet_mm=float(substrate.engine["L"]), bin_mm=1.0,
    )
    source_map = np.asarray(source["onset_maps_ms"][0], float)
    record.update({
        "event_t_on_ms": absolute_onset,
        "event_t_off_ms": offset + float(event["t_off"]),
        "latency_from_checkpoint_ms": float(event["t_on"]),
        "patient_mode": patient_mode,
        "ood": ood,
        "rank": rank,
        "onset": onset,
        "source_map_evaluable": bool(source["evaluable"][0]),
    })
    arrays.update({"rank": rank.astype(np.float32),
                   "onset": onset.astype(np.float32),
                   "source_onset_map_ms": source_map.astype(np.float32)})
    return record, arrays, spikes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()

    started = time.time()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    artifact_root = args.artifact_root.resolve()
    inputs = {}
    for key, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"intervention input changed: {record['path']}")
        inputs[key] = path
    expected = subprocess.check_output(
        ["git", "rev-parse", args.expected_commit], cwd=ROOT, text=True,
    ).strip()
    provenance = _runtime_provenance(expected)
    if provenance["runtime_modules_dirty"] or not provenance[
            "runtime_modules_match_expected_commit"]:
        raise RuntimeError("intervention runtime is not frozen")
    config_content = subprocess.check_output(
        ["git", "show", f"{expected}:{config_path.relative_to(ROOT)}"], cwd=ROOT,
    )
    if hashlib.sha256(config_content).hexdigest() != _sha256(config_path):
        raise RuntimeError("intervention config differs from expected commit")

    confirmation_config = json.loads(inputs["confirmation_config"].read_text())
    aggregate = json.loads(inputs["confirmation_summary"].read_text())
    manifest = json.loads(inputs["confirmation_manifest"].read_text())
    candidate_id = str(config["candidate_id"])
    candidate_rows = [row for row in manifest["candidates"]
                      if row["candidate_id"] == candidate_id]
    aggregate_rows = [row for row in aggregate["rows"]
                      if row["candidate_id"] == candidate_id]
    if len(candidate_rows) != 1 or len(aggregate_rows) != 1:
        raise RuntimeError("selected intervention candidate is not unique")
    candidate, aggregate_row = candidate_rows[0], aggregate_rows[0]

    cohort = json.loads(inputs["cohort_config"].read_text())
    classifier_config = json.loads(inputs["classifier_config"].read_text())
    patient = _patient_data(cohort, artifact_root)
    classifier = _classifier_contract(classifier_config, artifact_root)
    semantics = _old_to_patient_label_map(patient, classifier)
    patient = _reorder_patient_contract(patient, classifier["names"])
    label_map = semantics["raw_to_patient"]
    output_root = artifact_root / confirmation_config["output_root"]

    maps_by_network, labels_by_network, workers = [], [], {}
    source_counts = {}
    for seed in aggregate["requested_seeds"]:
        npz_path = output_root / "workers" / f"{candidate_id}_seed_{seed}.npz"
        worker = _load_network_worker(
            npz_path, patient["contact_names"], classifier, label_map,
        )
        maps, labels = _source_bundle(npz_path, worker)
        workers[int(seed)] = (npz_path, worker)
        maps_by_network.append(maps)
        labels_by_network.append(labels)
        source_counts[int(seed)] = int(len(maps))
    seed = select_representative_seed(
        aggregate_row["score"]["network_scores"], source_counts,
    )
    npz_path, worker = workers[seed]
    event_contracts = {
        mode: _event_contract(npz_path, worker, mode) for mode in (0, 1)
    }
    templates = {
        mode: network_balanced_early_support(
            maps_by_network, labels_by_network, mode,
        ) for mode in (0, 1)
    }

    transition = load_round_config(inputs["transition_config"])
    substrate = build_substrate(
        transition, "node_baseline", seed,
        cache_dir=str(artifact_root / confirmation_config["network_cache"]),
        ee_dose=0.0, etoi_dose=0.0,
        node_candidate_override=candidate["node_field"],
        artifact_root=artifact_root,
    )
    substrate.params.T = float(confirmation_config["search"]["simulation"]["duration_ms"])
    substrate.net["rng"] = np.random.default_rng(seed)
    dt_ms = float(substrate.engine["dt"])
    checkpoint_lead = float(config["intervention"]["checkpoint_lead_ms"])
    checkpoint_steps = {
        int(round((contract["event_t_on_ms"] - checkpoint_lead) / dt_ms))
        for contract in event_contracts.values()
    }
    if len(checkpoint_steps) != 2 or min(checkpoint_steps) <= 0:
        raise RuntimeError("selected events do not define two valid checkpoints")
    checkpoints = {}
    baseline = simulate_kick(
        substrate.params, substrate.net, KICK_BOOST=0.0, t_kick=1e9,
        V_th_per_neuron=substrate.vtheta, slow=None,
        early_stop_runaway=False,
        external_e_rate_drive=make_external_drive(
            substrate, transition["spatial_ou"], seed,
        ),
        checkpoint_steps=checkpoint_steps,
        checkpoint_sink=lambda step, state: checkpoints.setdefault(step, state),
    )
    if set(checkpoints) != checkpoint_steps:
        raise RuntimeError("baseline replay missed a requested checkpoint")
    full_spikes = np.asarray(baseline["E_spk_bool"], bool)
    active, active_dt = substrate.extras["cmrun"].active_fraction(
        full_spikes, dt_ms, substrate.extras["cmrun"].BIN_MS,
    )
    with np.load(npz_path, allow_pickle=False) as stored:
        stored_active = np.asarray(stored["active_fraction"], float)
    if not np.array_equal(np.asarray(active, np.float32), stored_active):
        raise RuntimeError("baseline checkpoint replay differs from confirmation worker")

    baseline_window = config["intervention"]["baseline_activity_window_ms"]
    low = int(round(float(baseline_window[0]) / dt_ms))
    high = int(round(float(baseline_window[1]) / dt_ms))
    covariates = grid_covariates(
        substrate.positions_e, substrate.h_e, full_spikes[low:high],
        dt_ms=dt_ms, sheet_mm=float(substrate.engine["L"]), bin_mm=1.0,
    )
    targets = {
        mode: select_hotspot_triplet(
            templates[mode], covariates, bin_mm=1.0,
            minimum_separation_mm=float(
                config["intervention"]["minimum_hotspot_separation_mm"]
            ),
        ) for mode in (0, 1)
    }

    intervention = config["intervention"]
    records, arrays = [], {}
    for mode in (0, 1):
        contract = event_contracts[mode]
        step = int(round(
            (contract["event_t_on_ms"] - checkpoint_lead) / dt_ms,
        ))
        checkpoint = checkpoints[step]
        branch_records, branch_arrays, branch_spikes = {}, {}, {}
        for arm, target_key in (
            ("sham", None),
            ("dominant", "dominant"),
            ("secondary", "secondary"),
            ("matched_off_template", "matched_off_template"),
        ):
            target = None if target_key is None else targets[mode][target_key]
            record, output_arrays, output_spikes = _branch_readout(
                substrate, transition, checkpoint, full_spikes, contract, target,
                duration_ms=float(intervention["continuation_ms"]),
                pulse_delay_ms=float(intervention["pulse_delay_from_checkpoint_ms"]),
                pulse_duration_ms=float(intervention["pulse_duration_ms"]),
                pulse_delta_vtheta=float(intervention["pulse_delta_vtheta_mv"]),
                target_radius_mm=float(intervention["target_radius_mm"]),
                classifier=classifier, label_map=label_map,
                contact_names=patient["contact_names"],
            )
            branch_records[arm] = record
            branch_spikes[arm] = output_spikes
            for key, value in output_arrays.items():
                branch_arrays[f"mode{mode}_{arm}_{key}"] = value
        sham = branch_records["sham"]
        if not sham["event_occurred"]:
            raise RuntimeError("same-checkpoint sham did not reproduce the selected event")
        expected_step = int(round(checkpoint_lead / dt_ms))
        native = full_spikes[step:step + int(round(
            float(intervention["continuation_ms"]) / dt_ms,
        ))]
        if not np.array_equal(branch_spikes["sham"], native):
            raise RuntimeError("same-checkpoint sham does not match native continuation")
        pulse_step = int(round(
            float(intervention["pulse_delay_from_checkpoint_ms"]) / dt_ms,
        ))
        for arm in ("dominant", "secondary", "matched_off_template"):
            # Exact spike parity is required before the threshold pulse begins.
            parity = bool(
                np.array_equal(
                    branch_spikes[arm][:pulse_step],
                    branch_spikes["sham"][:pulse_step],
                )
            )
            branch_records[arm]["pre_intervention_spike_parity"] = parity
            if not parity:
                raise RuntimeError(
                    f"mode {mode} {arm} diverged before the intervention pulse"
                )
            if branch_records[arm]["event_occurred"]:
                branch_records[arm]["rank_distance_from_sham"] = _rank_distance(
                    branch_arrays[f"mode{mode}_{arm}_rank"],
                    branch_arrays[f"mode{mode}_sham_rank"],
                    patient["contact_names"],
                )
                branch_records[arm]["source_topology_cosine_to_sham"] = cosine_similarity(
                    source_topology_features(branch_arrays[
                        f"mode{mode}_{arm}_source_onset_map_ms"
                    ][None, ...])[0],
                    source_topology_features(branch_arrays[
                        f"mode{mode}_sham_source_onset_map_ms"
                    ][None, ...])[0],
                )
        records.append({
            "mode": mode,
            "event_contract": contract,
            "targets": targets[mode],
            "branches": branch_records,
            "expected_native_onset_from_checkpoint_ms": checkpoint_lead,
            "pulse_step": pulse_step,
            "native_event_step": expected_step,
        })
        arrays.update(branch_arrays)

    output_root = artifact_root / config["output_root"]
    out_npz = output_root / "intervention_arrays.npz"
    out_json = output_root / "intervention_result.json"
    arrays.update({
        "mode0_early_probability": templates[0].astype(np.float32),
        "mode1_early_probability": templates[1].astype(np.float32),
        "positions_E": np.asarray(substrate.positions_e, np.float32),
        "h": np.asarray(substrate.h_e, np.float32),
        "contact_names": np.asarray(substrate.contact_names, dtype="U16"),
        "contact_xy_mm": np.asarray(substrate.contact_xy, np.float32),
    })
    _atomic_npz(out_npz, **arrays)
    payload = {
        "schema_id": "topic4_rev12_nd_node_intervention_result_v1",
        "status": "REV12ND_NODE_INTERVENTION_COMPLETE",
        "scientific_role": "model_internal_same_checkpoint_necessity_canary",
        "candidate_id": candidate_id,
        "candidate_field_sha256": candidate["node_field"]["field_sha256"],
        "selected_seed": seed,
        "seed_selection": "dual-mode source-evaluable network nearest median objective",
        "records": records,
        "arrays": {"path": str(out_npz), "sha256": _sha256(out_npz)},
        "inputs": {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in inputs.items()
        },
        "provenance": provenance,
        "wall_seconds": float(time.time() - started),
        "claim_boundary": (
            "The pulse is an idealized local E-threshold suppression assay. It tests "
            "model-internal regional necessity, not patient causality or biological therapy."
        ),
    }
    _atomic_json(out_json, payload)
    print(json.dumps({
        "status": payload["status"], "selected_seed": seed,
        "output": str(out_json),
    }, indent=2))


if __name__ == "__main__":
    main()
