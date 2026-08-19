#!/usr/bin/env python3
"""Run one patient-specific field/connectivity candidate on one network seed."""
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
    _runtime_provenance,
)
from src.sef_hfo_observation import VirtualMontage  # noqa: E402
from src.snn_engine.mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402
from src.topic4_cohort_fast_readout import batched_snn_event_envelope  # noqa: E402
from src.topic4_cohort_formal_scoring import confirm_subject, score_readout  # noqa: E402
from src.topic4_continuous_field import continuous_field_h_with_queries  # noqa: E402
from src.topic4_data_driven_snn_baseline import load_data_driven_snn_baseline  # noqa: E402
from src.topic4_local_connectivity import continuous_local_e_source_flow  # noqa: E402
from src.topic4_patient_specific_field_cohort import (  # noqa: E402
    atomic_json,
    json_ready,
    load_config,
    load_subject_contract,
    objective_from_score,
    patient_target_arrays,
    resolve_network_source_artifact,
    sha256,
    source_path,
    verify_inputs,
)
from src.topic4_spatial_ou_drive import SpatialOUConfig, SpatialOUDrive  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic4_patient_specific_field_connectivity_cohort_v2.json"


def _load_json(path: Path, expected_sha256: str | None = None) -> dict:
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        raise RuntimeError(f"frozen JSON changed: {path}")
    return json.loads(path.read_text())


def _config_dirty(path: Path) -> bool:
    return bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip())


def _compact_score(score: dict) -> dict:
    keep = (
        "status", "n_readable_events", "n_in_distribution_events",
        "supervised_mode_counts", "ood_fraction", "mode_details",
        "mode_losses", "weakest_mode_loss", "selection_score",
        "supervised_profile_matrix", "natural_kmeans",
    )
    return json_ready({key: score[key] for key in keep if key in score})


def _slow_state(baseline: dict, runtime_mode: str, *, n_total: int, n_e: int,
                v_th0: float, h_e: np.ndarray):
    if runtime_mode == "paired_slow_off":
        return None, baseline["paired_slow_off"]
    if runtime_mode != "active_z_plus_m":
        raise ValueError(f"unknown runtime mode: {runtime_mode}")
    mz = baseline["active_slow_state"]
    slow = MZSlowVars(
        n_total, v_th0,
        MZSlowVarsConfig(
            use_z=True,
            use_m=True,
            I_th_EI=float(mz["I_th_EI"]),
            tau_z=float(mz["tau_z"]),
            tau_adp=float(mz["tau_adp"]),
            eta_m=float(mz["eta_m"]),
            trace_stride_steps=int(mz["trace_stride_steps"]),
        ),
        NE=n_e,
        core_mask_E=np.asarray(h_e >= 0.5, bool),
    )
    return slow, mz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--phase", choices=("canary", "fit", "selection", "confirmation", "mechanism"), required=True)
    parser.add_argument("--runtime-mode", choices=("active_z_plus_m", "paired_slow_off"), default="active_z_plus_m")
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-npz", type=Path, required=True)
    parser.add_argument("--store-envelope", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    verify_inputs(config, code_root=ROOT)
    if _config_dirty(config_path):
        raise RuntimeError("patient-specific worker config is dirty")
    provenance = _runtime_provenance(args.expected_commit)
    if provenance["runtime_modules_dirty"] or not provenance["runtime_modules_match_expected_commit"]:
        raise RuntimeError("patient-specific worker modules differ from expected commit")
    provenance["systemd_unit"] = os.environ.get("TOPIC4_PATIENT_SPECIFIC_SYSTEMD_UNIT")

    candidate_path = args.candidate_json.resolve()
    candidate = _load_json(candidate_path)
    if candidate.get("subject_id") != args.subject_id:
        raise RuntimeError("candidate belongs to another subject")
    if candidate.get("basis_sha256") is None:
        raise RuntimeError("candidate lacks frozen whole-sheet basis identity")

    contract = load_subject_contract(config, args.subject_id)
    if contract["real_coords_sheet"] is None:
        raise RuntimeError(f"{args.subject_id} has no real geometry")
    contact_names = contract["contact_order"]
    montage = VirtualMontage(
        np.asarray(contract["real_coords_sheet"], float), contact_names,
        provenance=f"patient_specific_real_geometry_{args.subject_id}",
    )

    base_record = config["inputs"]["rev9_base_config"]
    base = resolve_network_source_artifact(
        config, _load_json(ROOT / base_record["path"], base_record["sha256"]),
    )
    stage_record = config["inputs"]["stage_config"]
    stage = _load_json(source_path(config, stage_record["path"]), stage_record["sha256"])
    anchor_record = config["inputs"]["node_anchor_config"]
    anchor = _load_json(ROOT / anchor_record["path"], anchor_record["sha256"])
    detector_record = config["inputs"]["common_detector_audit"]
    detector_audit = _load_json(
        source_path(config, detector_record["path"]), detector_record["sha256"],
    )
    detector = float(config["runtime"]["detector"]["population_active_fraction_threshold"])
    if detector != float(detector_audit["common_detector"]["central_threshold"]):
        raise RuntimeError("common absolute detector changed")
    baseline_record = config["inputs"]["data_driven_snn_baseline"]
    baseline = load_data_driven_snn_baseline(
        ROOT / baseline_record["path"], root=Path(config["source_workspace"]),
    )
    if baseline["baseline_id"] != baseline_record["baseline_id"]:
        raise RuntimeError("data-driven SNN baseline identity changed")
    if float(config["runtime"]["simulation_duration_ms"]) < float(
        baseline["consumer_contract"]["minimum_simulation_duration_ms"]
    ):
        raise RuntimeError("simulation is too short to audit delayed runaway")

    started = time.time()
    engine = stage["engine"]
    cmrun = _load_cmrun()
    cmrun.DT = float(engine["dt"])
    cmrun.KDIR = int(engine["k_dir"])
    cmrun.PART_MIN = 2 * int(engine["k_dir"]) + 1
    params_cls = __import__("params").Params
    simulate_kick = __import__("kick_probe").simulate_kick
    detect_events = __import__("src.sef_hfo_events", fromlist=["detect_events"]).detect_events
    params = params_cls(
        g=engine["g"], L=engine["L"], density=engine["density"],
        T=float(config["runtime"]["simulation_duration_ms"]),
        dt=engine["dt"], nu_ext_ratio=cmrun.DRIVE, seed=int(args.seed),
    )
    cache_dir = Path(config["source_workspace"]) / (
        "results/topic4_sef_hfo/data_driven_snn_cohort_v1/network_cache"
    )
    net, n_e, n_i, cache_hit, cache_source = _load_network(
        params, stage, {"theta_deg": 0.0}, int(args.seed), base, str(cache_dir),
    )
    positions_e = np.asarray(net["pos"][:n_e], float)
    node = _candidate_node(
        candidate["node_field"], positions_e, n_total=n_e + n_i,
        stage=stage, config=anchor,
    )
    h_e, h_i, field_query = continuous_field_h_with_queries(
        candidate["node_field"]["coefficients"],
        positions_e, np.asarray(net["pos"][n_e:], float),
        n_basis=int(candidate["node_field"]["n_basis"]),
        degree=int(candidate["node_field"]["degree"]),
        target_count=stage["N_core_manual"], L=engine["L"],
    )
    if not np.array_equal(h_e, node["h"]):
        raise RuntimeError("E/I field query changed the node field")
    local = config["local_connectivity"]
    mapped_net, edge_audit = continuous_local_e_source_flow(
        net, np.asarray(net["pos"], float), np.concatenate([h_e, h_i]),
        np.asarray(candidate["edge_coefficients"], float),
        l_ee=float(local["E_to_E_length_scale_mm"]),
        l_e_to_i=float(local["E_to_I_length_scale_mm"]),
        raw_logit_clip=float(local["raw_logit_clip_abs"]),
    )
    if not edge_audit["topology_unchanged"] or not edge_audit["delay_assignment_unchanged"] or not edge_audit["gaba_unchanged"]:
        raise RuntimeError("local connectivity changed a frozen structural object")
    for pathway in edge_audit["pathway_audit"].values():
        if float(pathway["max_abs_incoming_error"]) > 1e-9:
            raise RuntimeError("local connectivity did not conserve incoming budget")

    valid = cmrun.valid_mask(montage, positions_e, engine["L"], params.Rr)
    if not np.all(valid):
        raise RuntimeError("some real-geometry contacts are unreadable")
    spatial = baseline["spatial_ou"]
    external_drive = SpatialOUDrive(
        positions_e, float(engine["L"]), float(engine["dt"]),
        SpatialOUConfig(
            mode=spatial["mode"],
            sigma_rate_per_ms=float(spatial["sigma_rate_per_ms"]),
            tau_ms=float(spatial["tau_ms"]), ell_mm=float(spatial["ell_mm"]),
            update_interval_ms=float(spatial["update_interval_ms"]),
            grid_spacing_mm=float(spatial["grid_spacing_mm"]),
            seed=int(args.seed) + int(spatial["seed_offset"]),
        ),
    )
    slow, mz_contract = _slow_state(
        baseline, args.runtime_mode, n_total=n_e + n_i, n_e=n_e,
        v_th0=params.V_th, h_e=h_e,
    )
    mapped_net["rng"] = np.random.default_rng(int(args.seed))
    result = simulate_kick(
        params, mapped_net, KICK_BOOST=0.0, t_kick=1e9,
        V_th_per_neuron=node["vtheta"], slow=slow,
        early_stop_runaway=bool(config["runtime"]["early_stop_runaway"]),
        external_e_rate_drive=external_drive,
    )
    spikes = np.asarray(result["E_spk_bool"], bool)
    active, active_dt = cmrun.active_fraction(spikes, engine["dt"], cmrun.BIN_MS)
    detected = detect_events(active, active_dt, event_on_frac=detector)
    envelope, envelope_dt, _ = batched_snn_event_envelope(
        spikes, positions_e, montage, engine["dt"], contact_chunk=128,
    )
    onset_rows, rank_rows, event_rows = [], [], []
    readout = config["runtime"]["contact_readout"]
    for event in detected:
        if not bool(event.get("returned", False)):
            continue
        onset, rank = _contact_onsets(
            envelope, envelope_dt, montage, valid,
            (event["t_on"], event["t_off"]),
            readout["participation_margin_fraction"], readout["timing_fraction"],
        )
        onset_rows.append(onset)
        rank_rows.append(rank)
        event_rows.append({
            "t_on_ms": float(event["t_on"]), "t_off_ms": float(event["t_off"]),
            "duration_ms": float(event["dur_ms"]),
            "peak_active_fraction": float(event["peak_ext"]),
            "n_recruited_contacts": int(np.isfinite(rank).sum()),
        })
    width = len(contact_names)
    onsets = np.asarray(onset_rows, float).reshape((-1, width))
    ranks = np.asarray(rank_rows, float).reshape((-1, width))
    runaway_ms = result.get("runaway_early_stop_ms")

    split = "heldout" if args.phase in {"confirmation", "mechanism"} else "train"
    target_bundle = patient_target_arrays(contract["target_npz_path"], split)
    score_kwargs = {
        "target": target_bundle["target"],
        "contact_names": contact_names,
        "patient_centers": target_bundle["patient_centers"],
        "ood_threshold": float(contract["target_json"]["target"]["train_distance_q95"]),
        "minimum_contacts": int(config["runtime"]["minimum_contacts_per_event"]),
        "minimum_events_per_mode": int(config["runtime"]["minimum_events_per_mode"]),
        "kmeans_seed": int(config["runtime"]["kmeans_seed"]),
        "kmeans_n_init": int(config["runtime"]["kmeans_n_init"]),
    }
    if runaway_ms is None:
        score = score_readout(ranks, **score_kwargs)
        objective = objective_from_score(score, candidate, config)
        confirmation = (
            confirm_subject(
                ranks, permutations=contract["null_permutations"],
                minimum_events=int(config["runtime"]["minimum_events_per_mode"]),
                minimum_seed_ami=0.5, **score_kwargs,
            ) if args.phase == "confirmation" else None
        )
    else:
        score = {"status": "INVALID_RUNAWAY", "n_readable_events": 0}
        objective = objective_from_score(
            score, candidate, config, runaway_ms=float(runaway_ms),
        )
        confirmation = None

    mz_trace = (
        slow.trace_arrays() if slow is not None else {
            key: np.empty(0, np.float32) for key in MZSlowVars.TRACE_NAMES
        }
    )
    arrays = {
        "contact_names": np.asarray(contact_names, dtype="U64"),
        "contact_xy_mm": np.asarray(montage.contacts, np.float32),
        "onsets": np.asarray(onsets, np.float32),
        "ranks": np.asarray(ranks, np.float32),
        "active_fraction": np.asarray(active, np.float32),
        "active_fraction_dt_ms": np.asarray(active_dt, float),
        "h_E": np.asarray(h_e, np.float32),
        "positions_E": np.asarray(positions_e, np.float32),
        "mz_time_ms": np.asarray(mz_trace["time_ms"], np.float32),
        "mz_z_mean": np.asarray(mz_trace["z_mean"], np.float32),
        "mz_m_mean": np.asarray(mz_trace["m_mean"], np.float32),
    }
    if args.store_envelope:
        arrays["contact_envelope"] = np.asarray(envelope, np.float32)
        arrays["contact_envelope_dt_ms"] = np.asarray(envelope_dt, float)
    _atomic_npz(args.out_npz, **arrays)

    payload = {
        "status": "INVALID_RUNAWAY" if runaway_ms is not None else "COMPLETE",
        "scientific_role": config["scientific_role"],
        "subject_id": args.subject_id,
        "candidate_id": candidate["candidate_id"],
        "candidate_json": str(candidate_path),
        "candidate_sha256": sha256(candidate_path),
        "seed": int(args.seed), "phase": args.phase, "target_split": split,
        "runtime_mode": args.runtime_mode,
        "n_detected_events": int(len(detected)),
        "n_returned_events": int(len(ranks)),
        "runaway": runaway_ms is not None,
        "runaway_early_stop_ms": runaway_ms,
        "simulated_until_ms": float(len(spikes) * engine["dt"]),
        "wall_seconds": float(time.time() - started),
        "cache_hit": bool(cache_hit), "cache_source": cache_source,
        "field": {"query_audit": field_query, **candidate["node_field"]},
        "edge_audit": edge_audit,
        "mz_contract": mz_contract,
        "mz_summary": None if slow is None else slow.summary(),
        "events": event_rows,
        "score": _compact_score(score),
        "objective": objective,
        "confirmation": confirmation,
        "subject_input_hashes": contract["hashes"],
        "output_npz": str(args.out_npz),
        "output_npz_sha256": sha256(args.out_npz),
        "provenance": provenance,
    }
    atomic_json(payload, args.out_json)
    print(json.dumps({
        "status": payload["status"], "subject": args.subject_id,
        "candidate": candidate["candidate_id"], "seed": args.seed,
        "objective": objective["objective"], "events": len(ranks),
        "wall_seconds": payload["wall_seconds"],
    }), flush=True)


if __name__ == "__main__":
    main()
