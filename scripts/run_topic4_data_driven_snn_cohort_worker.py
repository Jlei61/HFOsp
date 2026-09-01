#!/usr/bin/env python3
"""Run one continuous substrate candidate with several patient montages."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "snn_engine"))

from scripts.run_topic4_core_field_stage3_fit import _load_cmrun  # noqa: E402
from scripts.run_topic4_rev10_sa_spectral_field_worker import (  # noqa: E402
    _candidate_node,
    _contact_onsets,
)
from scripts.run_topic4_rev9_node_kick_canary import _load_network  # noqa: E402
from scripts.run_topic4_rev9l_forced_source_worker import (  # noqa: E402
    _atomic_npz,
    _load_json_input,
    _runtime_provenance,
    _sha256,
)
from src.sef_hfo_observation import VirtualMontage  # noqa: E402
from src.topic4_continuous_field import continuous_field_h_with_queries  # noqa: E402
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_graph_edge_flow import array_sha256  # noqa: E402
from src.topic4_local_connectivity import continuous_local_e_source_flow  # noqa: E402
from src.topic4_spatial_ou_drive import SpatialOUConfig, SpatialOUDrive  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic4_data_driven_snn_cohort_canary_v1.json"
EXPECTED_ROLE = (
    "development_only_multisubject_pretrained_continuous_substrate_canary"
)


def _config_dirty(path: Path) -> bool:
    return bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip())


def _load_montages(config: dict) -> tuple[list[dict], VirtualMontage]:
    records = []
    combined_xy = []
    combined_names = []
    start = 0
    for subject_index, subject_id in enumerate(config["subjects"]):
        root = ROOT / config["target_root"]
        metadata_path = root / f"{subject_id}.json"
        target_path = root / f"{subject_id}_target.npz"
        metadata = json.loads(metadata_path.read_text())
        if not metadata.get("snn_eligible"):
            raise RuntimeError(f"cohort subject is not SNN eligible: {subject_id}")
        if _sha256(target_path) != _sha256(ROOT / metadata["target_npz"]):
            raise RuntimeError(f"target path mismatch for {subject_id}")
        with np.load(target_path, allow_pickle=False) as loaded:
            names = [str(value) for value in loaded["contact_order"]]
            coords = np.asarray(loaded["geometry_coords_sheet"], float)
        if coords.shape != (len(names), 2):
            raise RuntimeError(f"target geometry shape mismatch for {subject_id}")
        prefixed = [f"s{subject_index:02d}:{name}" for name in names]
        stop = start + len(names)
        records.append({
            "subject_id": subject_id,
            "subject_index": subject_index,
            "contact_names": names,
            "coords_sheet": coords,
            "slice": (start, stop),
            "target_json": str(metadata_path.relative_to(ROOT)),
            "target_json_sha256": _sha256(metadata_path),
            "target_npz": str(target_path.relative_to(ROOT)),
            "target_npz_sha256": _sha256(target_path),
        })
        combined_xy.extend(coords.tolist())
        combined_names.extend(prefixed)
        start = stop
    montage = VirtualMontage(
        np.asarray(combined_xy, float), combined_names,
        provenance="cohort_geometry_only_pca_combined_readout",
    )
    return records, montage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-npz", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    if config.get("scientific_role") != EXPECTED_ROLE:
        raise RuntimeError("cohort worker scientific role changed")
    if config["search"]["beta"] != "closed" or config["search"]["Z_M"] != "off":
        raise RuntimeError("cohort canary must keep beta and Z/M closed")
    all_seeds = {
        int(seed) for key in (
            "fit_network_seeds", "selection_network_seeds", "confirmation_network_seeds"
        ) for seed in config["search"][key]
    }
    if args.seed not in all_seeds:
        parser.error("worker seed is outside the frozen cohort pools")
    for record in config["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")

    output_root = ROOT / config["output_root"]
    manifest_path = output_root / "candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("status") != "TOPIC4_DATA_DRIVEN_SNN_COHORT_CANARY_LIBRARY_FROZEN"
            or manifest.get("config", {}).get("sha256") != _sha256(config_path)):
        raise RuntimeError("cohort candidate manifest is stale")
    matches = [
        row for row in manifest["candidate_set"]["candidates"]
        if row["candidate_id"] == args.candidate_id
    ]
    if len(matches) != 1:
        parser.error("candidate is outside the frozen cohort library")
    candidate = matches[0]
    provenance = _runtime_provenance(args.expected_commit)
    provenance["systemd_unit"] = os.environ.get("TOPIC4_COHORT_SYSTEMD_UNIT")
    if (_config_dirty(config_path) or provenance["runtime_modules_dirty"]
            or not provenance["runtime_modules_match_expected_commit"]):
        raise RuntimeError("cohort worker runtime or config is not frozen")

    output_json = args.out_json or (
        output_root / "workers" / f"{args.candidate_id}_seed_{args.seed}.json"
    )
    output_npz = args.out_npz or (
        output_root / "workers" / f"{args.candidate_id}_seed_{args.seed}.npz"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = str(args.cache_dir or (
        ROOT / "results/topic4_sef_hfo/data_driven_snn_cohort_v1/network_cache"
    ))
    started = time.time()

    base = _load_json_input(config["inputs"]["rev9_base_config"])
    stage = _load_json_input(config["inputs"]["stage_config"])
    anchor_config = _load_json_input(config["inputs"]["node_anchor_config"])
    detector_audit = _load_json_input(config["inputs"]["common_detector_audit"])
    detector = float(config["search"]["detector"]["population_active_fraction_threshold"])
    if detector != float(detector_audit["common_detector"]["central_threshold"]):
        raise RuntimeError("cohort common detector changed")
    subject_records, combined_montage = _load_montages(config)

    engine = stage["engine"]
    cmrun = _load_cmrun()
    cmrun.DT = float(engine["dt"])
    cmrun.KDIR = int(engine["k_dir"])
    cmrun.PART_MIN = 2 * int(engine["k_dir"]) + 1
    params_cls = __import__("params").Params
    simulate_kick = __import__("kick_probe").simulate_kick
    detect_events = __import__(
        "src.sef_hfo_events", fromlist=["detect_events"]
    ).detect_events
    snn_event_envelope = __import__(
        "src.sef_hfo_snn_adapter", fromlist=["snn_event_envelope"]
    ).snn_event_envelope
    params = params_cls(
        g=engine["g"], L=engine["L"], density=engine["density"],
        T=float(config["search"]["simulation"]["duration_ms"]),
        dt=engine["dt"], nu_ext_ratio=cmrun.DRIVE, seed=int(args.seed),
    )
    net, n_e, n_i, cache_hit, cache_source = _load_network(
        params, stage, {"theta_deg": 0.0}, int(args.seed), base, cache_dir,
    )
    positions_e = np.asarray(net["pos"][:n_e], float)
    positions_i = np.asarray(net["pos"][n_e:], float)
    node = _candidate_node(
        candidate["node_field"], positions_e, n_total=n_e + n_i,
        stage=stage, config=anchor_config,
    )
    h_e, h_i, field_query_audit = continuous_field_h_with_queries(
        candidate["node_field"]["coefficients"], positions_e, positions_i,
        n_basis=candidate["node_field"]["n_basis"],
        degree=candidate["node_field"]["degree"],
        target_count=stage["N_core_manual"], L=engine["L"],
    )
    if not np.array_equal(h_e, node["h"]):
        raise RuntimeError("cohort E/I field query changed the E-node field")
    coefficients = np.asarray(candidate["coefficients"], float)
    if array_sha256(coefficients) != candidate["coefficients_sha256"]:
        raise RuntimeError("cohort edge coefficient hash changed")
    local = config["local_connectivity_basis"]
    mapped_net, edge_audit = continuous_local_e_source_flow(
        net, np.asarray(net["pos"], float), np.concatenate([h_e, h_i]),
        coefficients,
        l_ee=float(local["E_to_E_length_scale_mm"]),
        l_e_to_i=float(local["E_to_I_length_scale_mm"]),
        raw_logit_clip=float(candidate["raw_logit_clip"]),
    )
    valid = cmrun.valid_mask(combined_montage, positions_e, engine["L"], params.Rr)
    if not np.all(valid):
        raise RuntimeError("cohort geometry-only contacts are not all readable")

    spatial_ou = candidate["spatial_ou"]
    external_drive = None
    if spatial_ou["mode"] != "off":
        external_drive = SpatialOUDrive(
            positions_e, float(engine["L"]), float(engine["dt"]),
            SpatialOUConfig(
                mode=spatial_ou["mode"],
                sigma_rate_per_ms=float(spatial_ou["sigma_rate_per_ms"]),
                tau_ms=float(spatial_ou["tau_ms"]),
                ell_mm=float(spatial_ou["ell_mm"]),
                update_interval_ms=float(spatial_ou["update_interval_ms"]),
                grid_spacing_mm=float(spatial_ou["grid_spacing_mm"]),
                seed=int(args.seed) + int(spatial_ou["seed_offset"]),
            ),
        )
    mapped_net["rng"] = np.random.default_rng(int(args.seed))
    result = simulate_kick(
        params, mapped_net, KICK_BOOST=0.0, t_kick=1e9,
        V_th_per_neuron=node["vtheta"],
        early_stop_runaway=bool(config["search"]["simulation"]["early_stop_runaway"]),
        external_e_rate_drive=external_drive,
    )
    spikes = np.asarray(result["E_spk_bool"], bool)
    active, active_dt = cmrun.active_fraction(spikes, engine["dt"], cmrun.BIN_MS)
    detected = detect_events(active, active_dt, event_on_frac=detector)
    envelope, envelope_dt, _ = snn_event_envelope(
        spikes, positions_e, combined_montage, engine["dt"],
    )

    arrays = {
        "subject_ids": np.asarray(config["subjects"], dtype="U64"),
        "combined_contact_names": np.asarray(combined_montage.names, dtype="U96"),
        "combined_contact_xy_mm": np.asarray(combined_montage.contacts, np.float32),
        "active_fraction": np.asarray(active, np.float32),
        "active_fraction_bin_ms": np.asarray(active_dt, float),
        "event_t_on_ms": np.asarray([row["t_on"] for row in detected], np.float32),
        "event_t_off_ms": np.asarray([row["t_off"] for row in detected], np.float32),
        "event_returned": np.asarray([row["returned"] for row in detected], bool),
        "positions_E": np.asarray(positions_e, np.float32),
        "h_E": np.asarray(h_e, np.float32),
        "edge_coefficients": coefficients,
    }
    readout = config["search"]["contact_readout"]
    per_subject_summary = []
    for record in subject_records:
        start, stop = record["slice"]
        sub_envelope = envelope[start:stop]
        sub_montage = VirtualMontage(
            record["coords_sheet"], record["contact_names"],
            provenance="cohort_geometry_only_pca_subject_readout",
        )
        sub_valid = valid[start:stop]
        onset_rows = []
        rank_rows = []
        for event in detected:
            onset, rank = _contact_onsets(
                sub_envelope, envelope_dt, sub_montage, sub_valid,
                (event["t_on"], event["t_off"]),
                readout["participation_margin_fraction"],
                readout["timing_fraction"],
            )
            onset_rows.append(onset)
            rank_rows.append(rank)
        key = f"subject_{record['subject_index']:02d}"
        onsets = np.asarray(onset_rows, float).reshape((-1, len(record["contact_names"])))
        ranks = np.asarray(rank_rows, float).reshape((-1, len(record["contact_names"])))
        arrays[f"{key}_contact_names"] = np.asarray(record["contact_names"], dtype="U64")
        arrays[f"{key}_contact_xy_mm"] = np.asarray(record["coords_sheet"], np.float32)
        arrays[f"{key}_onsets"] = np.asarray(onsets, np.float32)
        arrays[f"{key}_ranks"] = np.asarray(ranks, np.float32)
        arrays[f"{key}_contact_envelope"] = np.asarray(sub_envelope, np.float32)
        per_subject_summary.append({
            **{name: value for name, value in record.items() if name not in {"coords_sheet"}},
            "n_detected_events": int(len(detected)),
            "n_events_with_minimum_contacts": int(np.sum(
                np.isfinite(ranks).sum(axis=1) >= 3
            )),
            "median_recruited_contacts": (
                None if not len(ranks) else float(np.median(np.isfinite(ranks).sum(axis=1)))
            ),
        })
    arrays["contact_envelope_dt_ms"] = np.asarray(envelope_dt, float)
    _atomic_npz(output_npz, **arrays)

    runaway_stop_ms = result.get("runaway_early_stop_ms")
    runaway = runaway_stop_ms is not None
    payload = {
        "status": "INVALID_RUNAWAY" if runaway else "COMPLETE",
        "scientific_role": EXPECTED_ROLE,
        "candidate_id": args.candidate_id,
        "arm": candidate["arm"],
        "rotation_deg": candidate["node_field"]["transform"]["rotation_deg"],
        "seed": int(args.seed),
        "network_axis_deg": 0.0,
        "n_detected_events": int(len(detected)),
        "n_returned_events": int(sum(bool(row["returned"]) for row in detected)),
        "runaway": runaway,
        "runaway_early_stop_ms": runaway_stop_ms,
        "simulated_until_ms": float(len(spikes) * engine["dt"]),
        "wall_seconds": float(time.time() - started),
        "cache_hit": bool(cache_hit),
        "cache_source": cache_source,
        "field": {
            "field_sha256": candidate["node_field"]["field_sha256"],
            "source_field_sha256": candidate["node_field"]["source_field_sha256"],
            "query_audit": field_query_audit,
        },
        "edge_audit": edge_audit,
        "subjects": per_subject_summary,
        "output_npz": str(output_npz.relative_to(ROOT)),
        "output_npz_sha256": _sha256(output_npz),
        "provenance": provenance,
    }
    atomic_write_json(payload, output_json)
    print(json.dumps({
        "status": payload["status"],
        "candidate": args.candidate_id,
        "seed": args.seed,
        "events": len(detected),
        "subjects": len(subject_records),
        "wall_seconds": payload["wall_seconds"],
    }))


if __name__ == "__main__":
    main()
