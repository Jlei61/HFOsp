"""Run one SA6 field candidate on one paired network seed."""
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
from scripts.freeze_topic4_rev10_sa_dual_shaft_candidates import (  # noqa: E402
    build_manifest as build_k3_manifest,
)
from scripts.run_topic4_core_field_stage3_fit import _load_cmrun  # noqa: E402
from scripts.run_topic4_rev9_node_kick_canary import _load_network  # noqa: E402
from scripts.run_topic4_rev9l_forced_source_worker import (  # noqa: E402
    _atomic_npz,
    _load_json_input,
    _pad_rows,
    _runtime_provenance,
    _sha256,
)
from src.sef_hfo_observation import VirtualMontage, extract_lagpat  # noqa: E402
from src.topic4_continuous_field import continuous_field_h  # noqa: E402
from src.topic4_core_field_rev9 import (  # noqa: E402
    reconstruct_frozen_node,
    reconstruct_node_from_h,
)
from src.topic4_core_field_runner import _placement, atomic_write_json  # noqa: E402
from src.topic4_forced_source_capacity import (  # noqa: E402
    exclude_injected_packet_frame,
    select_triggered_event,
)
from src.topic4_rev10_sa_canary import matched_contact_packets  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_sa_dual_shaft_canary.json"


def _contact_onsets(envelope, envelope_dt, montage, valid, window,
                    margin_fraction, timing_fraction):
    selected = np.asarray(envelope, float)[np.asarray(valid, bool)]
    valid_names = np.asarray(montage.names)[np.asarray(valid, bool)]
    output_onsets = np.full(len(montage.names), np.nan)
    output_ranks = np.full(len(montage.names), np.nan)
    if not len(selected):
        return output_onsets, output_ranks
    floor = float(selected.min())
    margin = float(margin_fraction) * (float(selected.max()) - floor)
    artifact = extract_lagpat(
        selected, float(envelope_dt), [tuple(window)], floor, margin,
        timing_frac=float(timing_fraction), tie_tol=float(envelope_dt),
    )
    lookup = {name: index for index, name in enumerate(montage.names)}
    for local_index, name in enumerate(valid_names):
        target = lookup[str(name)]
        output_onsets[target] = artifact.lag_raw[local_index, 0]
        output_ranks[target] = artifact.ranks[local_index, 0]
    return output_onsets, output_ranks


def _event_rows(active, bin_width, threshold):
    detect_events = __import__(
        "src.sef_hfo_events", fromlist=["detect_events"]
    ).detect_events
    return detect_events(active, bin_width, event_on_frac=float(threshold))


def _candidate_node(candidate, positions, *, n_total, stage):
    engine = stage["engine"]
    if candidate.get("field_type") == "continuous_bspline":
        h, _ = continuous_field_h(
            candidate["coefficients"], positions,
            n_basis=int(candidate["n_basis"]), degree=int(candidate["degree"]),
            target_count=stage["N_core_manual"], L=engine["L"],
        )
        return reconstruct_node_from_h(
            h, n_total=n_total, quantile_seed=stage["quantile_seed"],
            core_mean=engine["core_mean"], core_std=engine["core_std"],
            v_base=engine["v_base"],
        )
    return reconstruct_frozen_node(
        candidate["theta"], positions, n_total=n_total,
        target_count=stage["N_core_manual"],
        quantile_seed=stage["quantile_seed"],
        core_mean=engine["core_mean"], core_std=engine["core_std"],
        v_base=engine["v_base"], K=3, L=engine["L"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--expected-commit")
    parser.add_argument("--out-json")
    parser.add_argument("--out-npz")
    parser.add_argument("--cache-dir")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    continuous = "sa6f_continuous_field" in config
    if continuous:
        assay = config["sa6f_continuous_field"]
        if assay["status"] != "DESIGN_FROZEN_AFTER_K3_REPRESENTATION_AUDIT":
            raise RuntimeError("SA6F continuous-field design is not frozen")
        from scripts.freeze_topic4_rev10_sa_continuous_field_candidates import (  # noqa: E402
            build_manifest as build_continuous_manifest,
        )
        manifest_builder = build_continuous_manifest
        output_subdir = "continuous_field_capacity"
        worker_status = "SA6F_CONTINUOUS_FIELD_WORKER_COMPLETE"
    else:
        assay = config["sa6_dual_shaft_field"]
        if assay["status"] != "DESIGN_FROZEN_SA5_READOUT_CLEARED":
            raise RuntimeError("SA6 design is not cleared by SA5")
        manifest_builder = build_k3_manifest
        output_subdir = "dual_shaft_capacity"
        worker_status = "SA6_DUAL_SHAFT_WORKER_COMPLETE"
    if assay["arm"] != "Node_only" or assay["edge"] != "off" or assay["beta"] != "closed":
        raise RuntimeError("SA6 must remain Node-only with edge and beta closed")
    if args.seed not in {int(value) for value in assay["network_seeds"]}:
        parser.error("--seed is outside the frozen SA6 seed set")
    inputs = config["inputs"]
    base = _load_json_input(inputs["rev9_base_config"])
    stage = _load_json_input(inputs["stage_config"])
    contract = _load_json_input(inputs["contact_contract"])
    sa5 = _load_json_input(inputs["sa5_summary"])
    if sa5["status"] != "SCL_READOUT_NOT_PRIMARY_LIMIT":
        raise RuntimeError("SA5 status changed")
    manifest = manifest_builder(config_path, args.expected_commit)
    matches = [row for row in manifest["candidate_set"]["candidates"]
               if row["candidate_id"] == args.candidate_id]
    if len(matches) != 1:
        parser.error("--candidate-id is outside the frozen SA6 manifest")
    candidate = matches[0]

    provenance = _runtime_provenance(args.expected_commit)
    provenance["systemd_unit"] = os.environ.get("REV10SA_SYSTEMD_UNIT")
    config_dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(config_path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip())
    if provenance["runtime_modules_dirty"] or config_dirty:
        raise RuntimeError("SA6 runtime modules or config are dirty")
    if (args.expected_commit is not None
            and not provenance["runtime_modules_match_expected_commit"]):
        raise RuntimeError("SA6 runtime modules differ from launcher commit")

    output_root = ROOT / config["output_root"] / output_subdir
    stem = f"{args.candidate_id}_seed_{args.seed}"
    output_json = Path(args.out_json or output_root / "workers" / f"{stem}.json")
    output_npz = Path(args.out_npz or output_root / "workers" / f"{stem}.npz")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = str(Path(args.cache_dir or ROOT /
                         "results/topic4_sef_hfo/data_driven_core_field_rev9/network_cache"))

    started = time.time()
    engine = stage["engine"]
    simulation = assay["simulation"]
    detector = float(assay["detector"]["population_active_fraction_threshold"])
    reg = _placement(stage)
    cmrun = _load_cmrun()
    cmrun.DT = float(engine["dt"])
    cmrun.KDIR = int(engine["k_dir"])
    cmrun.PART_MIN = 2 * int(engine["k_dir"]) + 1
    params_cls = __import__("params").Params
    simulate_kick = __import__("kick_probe").simulate_kick
    snn_event_envelope = __import__(
        "src.sef_hfo_snn_adapter", fromlist=["snn_event_envelope"]
    ).snn_event_envelope
    params = params_cls(
        g=engine["g"], L=engine["L"], density=engine["density"],
        T=float(simulation["duration_ms"]), dt=engine["dt"],
        nu_ext_ratio=cmrun.DRIVE, seed=int(args.seed),
    )
    net, n_e, n_i, cache_hit, cache_source = _load_network(
        params, stage, reg, int(args.seed), base, cache_dir,
    )
    positions = np.asarray(net["pos"][:n_e], float)
    node = _candidate_node(
        candidate, positions, n_total=n_e + n_i, stage=stage,
    )
    if not np.isclose(node["h"].sum(), float(stage["N_core_manual"]), atol=1e-8):
        raise RuntimeError("SA6 field budget projection failed")

    contacts = contract["contacts"]
    contact_names = [row["contact_name"] for row in contacts]
    contact_xy = np.asarray([row["sheet_xy_mm"] for row in contacts], float)
    shaft_ids = np.asarray([row["shaft_id"] for row in contacts], dtype="U8")
    montage = VirtualMontage(
        contact_xy, contact_names, provenance="rev10_sa_frozen_contact_contract",
    )
    valid = cmrun.valid_mask(montage, positions, engine["L"], params.Rr)
    if not np.all(valid):
        raise RuntimeError("SA6 requires all 15 frozen contacts to be locally readable")

    sources = manifest["forced_sources"]
    source_xy = np.asarray([row["xy_mm"] for row in sources], float)
    packet = assay["forced_packet"]
    requested_count = max(1, int(round(float(packet["fraction_of_E"]) * n_e)))
    packets = matched_contact_packets(
        positions, source_xy, radius_mm=float(packet["radius_mm"]),
        requested_count=requested_count,
        minimum_count=int(packet["minimum_common_count"]),
    )

    dynamics_seed = int(args.seed)
    net["rng"] = np.random.default_rng(dynamics_seed)
    sham = simulate_kick(
        params, net, KICK_BOOST=0.0, t_kick=1e9,
        V_th_per_neuron=node["vtheta"],
        early_stop_runaway=bool(simulation["early_stop_runaway"]),
    )
    sham_spikes = np.asarray(sham["E_spk_bool"], bool)
    sham_active, active_dt = cmrun.active_fraction(
        sham_spikes, engine["dt"], cmrun.BIN_MS,
    )
    sham_events = _event_rows(sham_active, active_dt, detector)
    sham_envelope, envelope_dt, _ = snn_event_envelope(
        sham_spikes, positions, montage, engine["dt"],
    )
    spontaneous_onsets, spontaneous_ranks = [], []
    for event in sham_events:
        onset, rank = _contact_onsets(
            sham_envelope, envelope_dt, montage, valid,
            (event["t_on"], event["t_off"]),
            assay["contact_readout"]["participation_margin_fraction"],
            assay["contact_readout"]["timing_fraction"],
        )
        spontaneous_onsets.append(onset)
        spontaneous_ranks.append(rank)

    trigger_step = int(round(float(simulation["forced_spike_ms"]) / engine["dt"]))
    forced_onsets, forced_ranks, forced_active = [], [], []
    run_rows = []
    for source, packet_e in zip(sources, packets["masks"]):
        packet_all = np.zeros(n_e + n_i, bool)
        packet_all[:n_e] = packet_e
        net["rng"] = np.random.default_rng(dynamics_seed)
        forced = simulate_kick(
            params, net, KICK_BOOST=0.0, t_kick=1e9,
            V_th_per_neuron=node["vtheta"],
            forced_spike_mask=packet_all,
            forced_spike_ms=float(simulation["forced_spike_ms"]),
            early_stop_runaway=bool(simulation["early_stop_runaway"]),
        )
        spikes = np.asarray(forced["E_spk_bool"], bool)
        if not np.array_equal(spikes[:trigger_step], sham_spikes[:trigger_step]):
            raise RuntimeError(f"paired randomness mismatch before {source['id']}")
        response_spikes = exclude_injected_packet_frame(
            spikes, sham_spikes, packet_e, trigger_step=trigger_step,
        )
        response_envelope, response_dt, _ = snn_event_envelope(
            response_spikes, positions, montage, engine["dt"],
        )
        if response_dt != envelope_dt:
            raise RuntimeError("forced/sham envelope dt changed")
        n_frame = min(response_envelope.shape[1], sham_envelope.shape[1])
        excess = np.clip(
            response_envelope[:, :n_frame] - sham_envelope[:, :n_frame],
            0.0, None,
        )
        onset, rank = _contact_onsets(
            excess, envelope_dt, montage, valid,
            simulation["response_window_ms"],
            assay["contact_readout"]["participation_margin_fraction"],
            assay["contact_readout"]["timing_fraction"],
        )
        active, forced_active_dt = cmrun.active_fraction(
            spikes, engine["dt"], cmrun.BIN_MS,
        )
        if forced_active_dt != active_dt:
            raise RuntimeError("active-fraction bin changed")
        events = _event_rows(active, active_dt, detector)
        triggered = select_triggered_event(
            events, trigger_ms=float(simulation["forced_spike_ms"]),
            max_latency_ms=40.0,
        )
        scl_recruitment = float(np.isfinite(onset[shaft_ids == "SCL"]).mean())
        icl_recruitment = float(np.isfinite(onset[shaft_ids == "ICL"]).mean())
        run_rows.append({
            "source_id": source["id"],
            "patient_mode": source["patient_mode"],
            "source_xy_mm": source["xy_mm"],
            "packet_n_E": int(packets["common_count"]),
            "packet_max_distance_mm": float(np.max(np.linalg.norm(
                positions[packet_e] - np.asarray(source["xy_mm"]), axis=1,
            ))),
            "pretrigger_spikes_bit_identical": True,
            "forced_spike_collision_count": int(
                forced["forced_spike_collision_count"]
            ),
            "ICL_recruited_contact_fraction": icl_recruitment,
            "SCL_recruited_contact_fraction": scl_recruitment,
            "multishaft": bool(icl_recruitment > 0.0 and scl_recruitment > 0.0),
            "n_common_detector_events": int(len(events)),
            "triggered_returned_event": triggered,
            "runaway_early_stop_ms": forced["runaway_early_stop_ms"],
            "peak_active_fraction": float(np.max(active, initial=0.0)),
            "fraction_time_above_common_detector": float(np.mean(active > detector)),
        })
        forced_onsets.append(onset)
        forced_ranks.append(rank)
        forced_active.append(active)
        print(json.dumps({
            "progress": "forced_source_complete",
            "candidate": args.candidate_id,
            "seed": args.seed,
            "source": source["id"],
            "scl_recruitment": scl_recruitment,
            "runaway": forced["runaway_early_stop_ms"],
        }), flush=True)

    spontaneous_onsets = np.asarray(spontaneous_onsets, float).reshape(
        (-1, len(contact_names))
    )
    spontaneous_ranks = np.asarray(spontaneous_ranks, float).reshape(
        (-1, len(contact_names))
    )
    _atomic_npz(
        output_npz,
        contact_names=np.asarray(contact_names, dtype="U16"),
        shaft_ids=shaft_ids,
        contact_xy_mm=contact_xy,
        source_ids=np.asarray([row["id"] for row in sources], dtype="U16"),
        source_xy_mm=source_xy,
        packet_masks_E=np.asarray(packets["masks"], bool),
        forced_onsets=np.asarray(forced_onsets, np.float32),
        forced_ranks=np.asarray(forced_ranks, np.float32),
        spontaneous_onsets=spontaneous_onsets.astype(np.float32),
        spontaneous_ranks=spontaneous_ranks.astype(np.float32),
        spontaneous_active_fraction=np.asarray(sham_active, np.float32),
        forced_active_fraction=_pad_rows(forced_active, dtype=np.float32),
        active_fraction_bin_ms=np.asarray(active_dt, float),
        positions_E=np.asarray(positions, np.float32),
        h=np.asarray(node["h"], np.float32),
        delta_vtheta=np.asarray(node["delta_vtheta"], np.float32),
    )
    payload = {
        "status": worker_status,
        "scientific_role": (
            "exploratory non-component continuous Node-field capacity"
            if continuous else "exploratory fixed-K3 component-relocation canary"
        ),
        "candidate": candidate,
        "seed": int(args.seed),
        "runs": run_rows,
        "spontaneous": {
            "n_common_detector_events": int(len(sham_events)),
            "n_returned_events": int(sum(bool(row.get("returned")) for row in sham_events)),
            "runaway_early_stop_ms": sham["runaway_early_stop_ms"],
            "peak_active_fraction": float(np.max(sham_active, initial=0.0)),
            "fraction_time_above_common_detector": float(np.mean(sham_active > detector)),
        },
        "field": {
            "sum_h": float(node["h"].sum()),
            "max_h": float(node["h"].max(initial=0.0)),
            "n_h_ge_0p5": int(np.sum(node["h"] >= 0.5)),
            "n_h_ge_0p9": int(np.sum(node["h"] >= 0.9)),
            "node_hashes": node["hashes"],
        },
        "packet": {
            "common_count": int(packets["common_count"]),
            "available_counts": packets["available_counts"].astype(int).tolist(),
            "radius_mm": float(packets["radius_mm"]),
        },
        "network": {
            "n_E": int(n_e), "n_I": int(n_i),
            "cache_hit": bool(cache_hit), "cache_source": cache_source,
        },
        "simulation": {
            **simulation,
            "common_detector_threshold": detector,
            "wall_seconds": float(time.time() - started),
        },
        "arrays": {"path": str(output_npz), "sha256": _sha256(output_npz)},
        "config": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": _sha256(config_path),
        },
        "provenance": provenance,
    }
    atomic_write_json(payload, output_json)
    print(json.dumps({
        "status": payload["status"],
        "candidate": args.candidate_id,
        "seed": args.seed,
        "n_spontaneous_events": len(sham_events),
        "n_runaway": int(sum(row["runaway_early_stop_ms"] is not None
                             for row in run_rows)),
        "elapsed_seconds": payload["simulation"]["wall_seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
