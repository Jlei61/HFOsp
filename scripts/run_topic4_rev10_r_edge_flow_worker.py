"""Run one frozen rev10-R graph edge field on one spontaneous SNN network."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join("src", "snn_engine"))
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
from src.topic4_core_field_runner import _placement, atomic_write_json  # noqa: E402
from src.topic4_dynamic_accessibility import (  # noqa: E402
    AdaptationConfig,
    SpikeTriggeredAdaptation,
)
from src.topic4_graph_edge_flow import (  # noqa: E402
    array_sha256,
    graph_spectral_ee_flow,
)
from src.topic4_spatial_edge_flow import spatial_vector_ee_flow  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_r_graph_edge_flow.json"


def active_network_seeds(config):
    phase = config.get("search", {}).get("phase", "fit")
    key = {
        "fit": "fit_network_seeds",
        "selection": "selection_network_seeds",
        "confirmation": "confirmation_network_seeds",
    }.get(phase)
    if key is None:
        raise ValueError(f"unknown rev10-R search phase: {phase}")
    return list(map(int, config["search"][key]))


def _load_basis(npz_path, record, seed):
    if _sha256(npz_path) != record["npz_sha256"]:
        raise RuntimeError(f"graph basis NPZ changed: seed {seed}")
    with np.load(npz_path, allow_pickle=False) as loaded:
        basis = {
            "u": np.asarray(loaded["u"], float),
            "v": np.asarray(loaded["v"], float),
            "singular_values": np.asarray(loaded["singular_values"], float),
            "n_e": int(loaded["n_e"]),
            "rank": int(loaded["rank"]),
            "graph_weight_sha256": str(loaded["graph_weight_sha256"].item()),
        }
    return basis


def _config_dirty(path):
    return bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--out-json")
    parser.add_argument("--out-npz")
    parser.add_argument("--cache-dir")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    allowed_roles = {
        "development_only_contact_density_invariant_route_capacity",
        "development_only_observation_invariant_spatial_route_capacity",
        "development_only_observation_invariant_spatial_route_selection",
        "development_only_observation_invariant_spatial_route_confirmation",
        "development_only_dynamic_accessibility_canary",
    }
    if config["scientific_role"] not in allowed_roles:
        raise RuntimeError("rev10-R scientific role changed")
    if args.seed not in set(active_network_seeds(config)):
        parser.error("worker seed is outside the active frozen network set")
    if config["search"]["beta"] != "closed":
        raise RuntimeError("beta must remain closed")
    for record in config["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")

    output_root = ROOT / config["output_root"]
    manifest_path = output_root / "candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    allowed_manifests = {
        "REV10R_GRAPH_SPECTRAL_LIBRARY_FROZEN",
        "REV10R2_SPATIAL_EDGE_LIBRARY_FROZEN",
        "REV10R2_SPATIAL_EDGE_SELECTION_LIBRARY_FROZEN",
        "REV10R2_SPATIAL_EDGE_CONFIRMATION_LIBRARY_FROZEN",
        "REV10D_LOCAL_ADAPTATION_LIBRARY_FROZEN",
    }
    if (manifest.get("status") not in allowed_manifests
            or manifest.get("config", {}).get("sha256") != _sha256(config_path)):
        raise RuntimeError("rev10-R candidate manifest is stale")
    matches = [
        row for row in manifest["candidate_set"]["candidates"]
        if row["candidate_id"] == args.candidate_id
    ]
    if len(matches) != 1:
        parser.error("candidate is outside the frozen rev10-R library")
    candidate = matches[0]
    basis_record = None
    basis_npz = None
    if manifest["status"] == "REV10R_GRAPH_SPECTRAL_LIBRARY_FROZEN":
        basis_records = {
            int(row["seed"]): row for row in manifest["graph_bases"]
        }
        basis_record = basis_records[args.seed]
        basis_npz = ROOT / basis_record["npz"]

    provenance = _runtime_provenance(args.expected_commit)
    provenance["systemd_unit"] = os.environ.get("REV10R_SYSTEMD_UNIT")
    if (provenance["runtime_modules_dirty"] or _config_dirty(config_path)
            or not provenance["runtime_modules_match_expected_commit"]):
        raise RuntimeError("edge worker runtime modules or config are not frozen")

    stem = f"{candidate['candidate_id']}_seed_{args.seed}"
    output_json = Path(args.out_json or output_root / "workers" / f"{stem}.json")
    output_npz = Path(args.out_npz or output_root / "workers" / f"{stem}.npz")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = str(Path(
        args.cache_dir or ROOT /
        "results/topic4_sef_hfo/data_driven_core_field_rev9/network_cache"
    ))
    started = time.time()

    inputs = config["inputs"]
    base = _load_json_input(inputs["rev9_base_config"])
    stage = _load_json_input(inputs["stage_config"])
    contract = _load_json_input(inputs["contact_contract"])
    detector_audit = _load_json_input(inputs["common_detector_audit"])
    anchor_config = _load_json_input(inputs["node_anchor_config"])
    anchor_manifest = _load_json_input(inputs["node_anchor_manifest"])
    anchor_matches = [
        row for row in anchor_manifest["candidate_set"]["candidates"]
        if row["candidate_id"] == config["node_anchor"]["candidate_id"]
    ]
    if (len(anchor_matches) != 1 or anchor_matches[0]["field_sha256"]
            != config["node_anchor"]["field_sha256"]):
        raise RuntimeError("frozen Node anchor cannot be reconstructed")
    anchor = anchor_matches[0]
    detector = float(config["search"]["detector"][
        "population_active_fraction_threshold"
    ])
    if detector != float(detector_audit["common_detector"]["central_threshold"]):
        raise RuntimeError("common detector changed")

    engine = stage["engine"]
    simulation = config["search"]["simulation"]
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
        T=float(simulation["duration_ms"]), dt=engine["dt"],
        nu_ext_ratio=cmrun.DRIVE, seed=int(args.seed),
    )
    net, n_e, n_i, cache_hit, cache_source = _load_network(
        params, stage, _placement(stage), int(args.seed), base, cache_dir,
    )
    positions = np.asarray(net["pos"][:n_e], float)
    node = _candidate_node(
        anchor, positions, n_total=n_e + n_i,
        stage=stage, config=anchor_config,
    )
    if not np.isclose(node["h"].sum(), float(stage["N_core_manual"]), atol=1e-8):
        raise RuntimeError("Node anchor field budget changed")
    coefficients = np.asarray(candidate["coefficients"], float)
    if array_sha256(coefficients) != candidate["coefficients_sha256"]:
        raise RuntimeError("edge coefficient hash changed")
    net["rng"] = np.random.default_rng(int(args.seed))
    if manifest["status"] == "REV10R_GRAPH_SPECTRAL_LIBRARY_FROZEN":
        basis = _load_basis(basis_npz, basis_record, args.seed)
        mapped_net, edge_audit = graph_spectral_ee_flow(
            net, basis, coefficients,
        )
        edge_basis = {
            "family": "graph_spectral_chebyshev",
            "seed": int(args.seed), "npz": str(basis_npz),
            "npz_sha256": basis_record["npz_sha256"],
            "graph_weight_sha256": basis["graph_weight_sha256"],
            "rank": basis["rank"],
        }
    else:
        spatial = config["spatial_edge_basis"]
        mapped_net, edge_audit = spatial_vector_ee_flow(
            net, positions, coefficients,
            L=float(spatial["sheet_L_mm"]),
            length_scale=float(spatial["displacement_length_scale_mm"]),
            raw_logit_clip=candidate.get("raw_logit_clip"),
        )
        edge_basis = {
            "family": "continuous_quadratic_midpoint_vector_flow",
            "feature_names": edge_audit["feature_names"],
            "sheet_L_mm": edge_audit["sheet_L_mm"],
            "displacement_length_scale_mm": edge_audit[
                "displacement_length_scale_mm"
            ],
        }

    adaptation = candidate.get("adaptation", {"mode": "off"})
    if adaptation["mode"] == "off":
        slow = None
    else:
        slow = SpikeTriggeredAdaptation(
            n_total=n_e + n_i,
            n_e=n_e,
            dt_ms=float(engine["dt"]),
            cfg=AdaptationConfig(
                mode=adaptation["mode"],
                tau_ms=float(adaptation["tau_ms"]),
                increment_mV=float(adaptation["increment_mV"]),
                trace_dt_ms=float(adaptation["trace_dt_ms"]),
            ),
        )
    if manifest["status"] == "REV10D_LOCAL_ADAPTATION_LIBRARY_FROZEN":
        if not np.all(coefficients == 0.0):
            raise RuntimeError("rev10-D requires exact no-op edge coefficients")
        if not (
            edge_audit["edge_ratio"]["min"] == 1.0
            and edge_audit["edge_ratio"]["max"] == 1.0
        ):
            raise RuntimeError("rev10-D edge mapper is not an exact no-op")

    contacts = contract["contacts"]
    contact_names = [row["contact_name"] for row in contacts]
    contact_xy = np.asarray([row["sheet_xy_mm"] for row in contacts], float)
    shaft_ids = np.asarray([row["shaft_id"] for row in contacts], dtype="U8")
    montage = VirtualMontage(
        contact_xy, contact_names,
        provenance="rev10_r_observation_only_contact_contract",
    )
    valid = cmrun.valid_mask(montage, positions, engine["L"], params.Rr)
    if not np.all(valid):
        raise RuntimeError("all frozen contacts must be locally readable")
    result = simulate_kick(
        params, mapped_net, KICK_BOOST=0.0, t_kick=1e9,
        V_th_per_neuron=node["vtheta"], slow=slow,
        early_stop_runaway=bool(simulation["early_stop_runaway"]),
    )
    spikes = np.asarray(result["E_spk_bool"], bool)
    active, active_dt = cmrun.active_fraction(spikes, engine["dt"], cmrun.BIN_MS)
    detected = detect_events(active, active_dt, event_on_frac=detector)
    envelope, envelope_dt, _ = snn_event_envelope(
        spikes, positions, montage, engine["dt"],
    )
    readout = config["search"]["contact_readout"]
    onset_rows, rank_rows, event_rows = [], [], []
    for event_index, event in enumerate(detected):
        onset, rank = _contact_onsets(
            envelope, envelope_dt, montage, valid,
            (event["t_on"], event["t_off"]),
            readout["participation_margin_fraction"],
            readout["timing_fraction"],
        )
        onset_rows.append(onset)
        rank_rows.append(rank)
        event_rows.append({
            "event_index": int(event_index),
            "t_on_ms": float(event["t_on"]),
            "t_off_ms": float(event["t_off"]),
            "duration_ms": float(event["dur_ms"]),
            "peak_active_fraction": float(event["peak_ext"]),
            "returned": bool(event["returned"]),
            "n_recruited_contacts": int(np.isfinite(onset).sum()),
            "ICL_recruited_fraction": float(np.isfinite(
                onset[shaft_ids == "ICL"]
            ).mean()),
            "SCL_recruited_fraction": float(np.isfinite(
                onset[shaft_ids == "SCL"]
            ).mean()),
        })
    onsets = np.asarray(onset_rows, float).reshape((-1, len(contact_names)))
    ranks = np.asarray(rank_rows, float).reshape((-1, len(contact_names)))
    adaptation_trace = (
        slow.trace_arrays() if slow is not None else {
            key: np.empty(0, dtype=np.float32)
            for key in ("time_ms", "mean_mV", "sd_mV", "max_mV")
        }
    )
    _atomic_npz(
        output_npz,
        contact_names=np.asarray(contact_names, dtype="U16"),
        shaft_ids=shaft_ids,
        contact_xy_mm=contact_xy,
        onsets=onsets.astype(np.float32),
        ranks=ranks.astype(np.float32),
        event_t_on_ms=np.asarray([row["t_on_ms"] for row in event_rows], np.float32),
        event_t_off_ms=np.asarray([row["t_off_ms"] for row in event_rows], np.float32),
        event_returned=np.asarray([row["returned"] for row in event_rows], bool),
        active_fraction=np.asarray(active, np.float32),
        active_fraction_bin_ms=np.asarray(active_dt, float),
        contact_envelope=np.asarray(envelope, np.float32),
        contact_envelope_dt_ms=np.asarray(envelope_dt, float),
        positions_E=np.asarray(positions, np.float32),
        h=np.asarray(node["h"], np.float32),
        delta_vtheta=np.asarray(node["delta_vtheta"], np.float32),
        edge_coefficients=coefficients.astype(np.float64),
        edge_response=np.asarray(edge_audit.get("spectral_response", []), np.float64),
        adaptation_time_ms=adaptation_trace["time_ms"],
        adaptation_mean_mV=adaptation_trace["mean_mV"],
        adaptation_sd_mV=adaptation_trace["sd_mV"],
        adaptation_max_mV=adaptation_trace["max_mV"],
    )
    payload = {
        "status": "REV10R_EDGE_FLOW_WORKER_COMPLETE",
        "scientific_role": config["scientific_role"],
        "candidate": candidate,
        "seed": int(args.seed),
        "events": event_rows,
        "run": {
            "n_common_detector_events": int(len(detected)),
            "n_returned_events": int(sum(row["returned"] for row in event_rows)),
            "runaway_early_stop_ms": result["runaway_early_stop_ms"],
            "peak_active_fraction": float(np.max(active, initial=0.0)),
            "fraction_time_above_common_detector": float(np.mean(active > detector)),
        },
        "node_anchor": {
            "candidate_id": anchor["candidate_id"],
            "field_sha256": anchor["field_sha256"],
            "sum_h": float(node["h"].sum()),
            "node_hashes": node["hashes"],
        },
        "edge_audit": edge_audit,
        "edge_basis": edge_basis,
        "dynamic_accessibility": {
            **adaptation,
            "trace_samples": int(len(adaptation_trace["time_ms"])),
            "peak_mean_mV": float(np.max(
                adaptation_trace["mean_mV"], initial=0.0,
            )),
            "peak_spatial_sd_mV": float(np.max(
                adaptation_trace["sd_mV"], initial=0.0,
            )),
            "peak_neuron_mV": float(np.max(
                adaptation_trace["max_mV"], initial=0.0,
            )),
        },
        "network": {
            "n_E": int(n_e), "n_I": int(n_i),
            "cache_hit": bool(cache_hit), "cache_source": cache_source,
        },
        "simulation": {
            **simulation, "common_detector_threshold": detector,
            "wall_seconds": float(time.time() - started),
        },
        "arrays": {"path": str(output_npz), "sha256": _sha256(output_npz)},
        "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "config": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": _sha256(config_path),
        },
        "provenance": provenance,
    }
    atomic_write_json(payload, output_json)
    print(json.dumps({
        "status": payload["status"], "candidate": args.candidate_id,
        "seed": args.seed, "n_events": len(detected),
        "runaway": result["runaway_early_stop_ms"],
        "edge_ratio": edge_audit["edge_ratio"],
        "wall_seconds": payload["simulation"]["wall_seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
