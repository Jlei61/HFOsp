"""Run one fresh network's paired D4.1 A/B packet-dose confirmation."""
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
from scripts.run_topic4_rev10_d4_uniform_source_worker import (  # noqa: E402
    _classifier,
    _embedding,
    _load_record,
)
from scripts.run_topic4_rev10_sa_spectral_field_worker import (  # noqa: E402
    _candidate_node,
    _contact_onsets,
)
from scripts.run_topic4_rev9_node_kick_canary import _load_network  # noqa: E402
from scripts.run_topic4_rev9l_forced_source_worker import (  # noqa: E402
    _atomic_npz,
    _runtime_provenance,
    _sha256,
)
from src.sef_hfo_observation import VirtualMontage  # noqa: E402
from src.topic4_core_field_runner import _placement, atomic_write_json  # noqa: E402
from src.topic4_forced_source_capacity import (  # noqa: E402
    exclude_injected_packet_frame,
    paired_excess_geometry,
    select_source_indices,
    select_triggered_event,
)
from src.topic4_shaft_aware import contract_groups  # noqa: E402
from src.topic4_shaft_aware_direction import assign_direction_modes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_d4_1_packet_dose_confirmation.json"


def _full_event_diagnostics(spikes, *, cmrun, detect_events, detector,
                            trigger_ms, latency_ms):
    active, bin_ms = cmrun.active_fraction(spikes, cmrun.DT, cmrun.BIN_MS)
    events = detect_events(active, bin_ms, event_on_frac=float(detector))
    triggered = select_triggered_event(
        events, trigger_ms=trigger_ms, max_latency_ms=latency_ms,
    )
    rows = [{
        "t_on_ms": float(event["t_on"]),
        "t_off_ms": float(event["t_off"]),
        "returned": bool(event.get("returned", False)),
    } for event in events]
    selected = None if triggered is None else {
        "t_on_ms": float(triggered["t_on"]),
        "t_off_ms": float(triggered["t_off"]),
        "returned": bool(triggered.get("returned", False)),
    }
    return active, selected, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--out-json")
    parser.add_argument("--out-npz")
    parser.add_argument("--cache-dir")
    parser.add_argument("--only-packet-fraction", type=float)
    parser.add_argument("--dump-active", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    if config["scientific_role"] != (
        "development_only_fresh_network_forced_route_dose_confirmation"
    ):
        raise RuntimeError("rev10-D4.1 scientific role changed")
    if args.seed not in set(map(int, config["network_seeds"])):
        parser.error("seed is outside the frozen D4.1 network set")
    for record in config["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")
    output_root = ROOT / config["output_root"]
    manifest_path = output_root / "packet_dose_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("status") != "REV10D4_1_PACKET_DOSE_CONFIRMATION_FROZEN"
        or manifest.get("config", {}).get("sha256") != _sha256(config_path)
    ):
        raise RuntimeError("D4.1 manifest is stale")

    provenance = _runtime_provenance(args.expected_commit)
    provenance["systemd_unit"] = os.environ.get("REV10D4_1_SYSTEMD_UNIT")
    config_dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(config_path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip()
    if (
        provenance["runtime_modules_dirty"] or config_dirty
        or not provenance["runtime_modules_match_expected_commit"]
    ):
        raise RuntimeError("D4.1 worker modules or config are not frozen")

    stem = f"packet_dose_seed_{args.seed}"
    output_json = Path(args.out_json or output_root / "workers" / f"{stem}.json")
    output_npz = Path(args.out_npz or output_root / "workers" / f"{stem}.npz")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()

    inputs = config["inputs"]
    base = _load_record(inputs["rev9_base_config"])
    stage = _load_record(inputs["stage_config"])
    contract = _load_record(inputs["contact_contract"])
    detector_audit = _load_record(inputs["common_detector_audit"])
    anchor_config = _load_record(inputs["node_anchor_config"])
    anchor_manifest = _load_record(inputs["node_anchor_manifest"])
    anchor_matches = [
        row for row in anchor_manifest["candidate_set"]["candidates"]
        if row["candidate_id"] == config["node_anchor"]["candidate_id"]
    ]
    if (
        len(anchor_matches) != 1
        or anchor_matches[0]["field_sha256"]
        != config["node_anchor"]["field_sha256"]
    ):
        raise RuntimeError("D4.1 Node anchor changed")
    anchor = anchor_matches[0]
    detector = float(config["detector"]["population_active_fraction_threshold"])
    if detector != float(detector_audit["common_detector"]["central_threshold"]):
        raise RuntimeError("D4.1 common detector changed")

    engine = stage["engine"]
    simulation = config["simulation"]
    cmrun = _load_cmrun()
    cmrun.DT = float(engine["dt"])
    cmrun.KDIR = int(engine["k_dir"])
    cmrun.PART_MIN = 2 * int(engine["k_dir"]) + 1
    params_cls = __import__("params").Params
    simulate_kick = __import__("kick_probe").simulate_kick
    detect_events = __import__(
        "src.sef_hfo_events", fromlist=["detect_events"],
    ).detect_events
    snn_event_envelope = __import__(
        "src.sef_hfo_snn_adapter", fromlist=["snn_event_envelope"],
    ).snn_event_envelope
    params = params_cls(
        g=engine["g"], L=engine["L"], density=engine["density"],
        T=float(simulation["duration_ms"]), dt=engine["dt"],
        nu_ext_ratio=cmrun.DRIVE, seed=int(args.seed),
    )
    cache_dir = str(Path(
        args.cache_dir or ROOT
        / "results/topic4_sef_hfo/data_driven_core_field_rev9/network_cache"
    ))
    net, n_e, n_i, cache_hit, cache_source = _load_network(
        params, stage, _placement(stage), int(args.seed), base, cache_dir,
    )
    positions = np.asarray(net["pos"][:n_e], float)
    node = _candidate_node(
        anchor, positions, n_total=n_e + n_i, stage=stage, config=anchor_config,
    )
    if not np.isclose(node["h"].sum(), float(stage["N_core_manual"]), atol=1e-8):
        raise RuntimeError("D4.1 Node field budget changed")

    contacts = contract["contacts"]
    contact_names = np.asarray([row["contact_name"] for row in contacts]).astype(str)
    contact_xy = np.asarray([row["sheet_xy_mm"] for row in contacts], float)
    shaft_ids = np.asarray([row["shaft_id"] for row in contacts]).astype(str)
    montage = VirtualMontage(
        contact_xy, contact_names.tolist(), provenance="rev10_d4_1_readout_only",
    )
    valid = cmrun.valid_mask(montage, positions, engine["L"], params.Rr)
    if not np.all(valid):
        raise RuntimeError("D4.1 requires all frozen contacts to be readable")
    target_names, embedding = _embedding(ROOT / inputs["shaft_aware_target_npz"]["path"])
    if not np.array_equal(target_names, contact_names):
        raise RuntimeError("D4.1 target and contact order differ")
    groups = contract_groups(contract)
    classifier = _classifier(manifest)

    net["rng"] = np.random.default_rng(int(args.seed))
    sham = simulate_kick(
        params, net, KICK_BOOST=0.0, t_kick=1e9,
        V_th_per_neuron=node["vtheta"],
        early_stop_runaway=bool(simulation["early_stop_runaway"]),
    )
    sham_spikes = np.asarray(sham["E_spk_bool"], bool)
    sham_envelope, envelope_dt, _ = snn_event_envelope(
        sham_spikes, positions, montage, engine["dt"],
    )
    sham_active, sham_triggered, sham_events = _full_event_diagnostics(
        sham_spikes, cmrun=cmrun, detect_events=detect_events,
        detector=detector, trigger_ms=simulation["forced_spike_ms"],
        latency_ms=simulation["trigger_max_latency_ms"],
    )

    trigger_step = int(round(simulation["forced_spike_ms"] / engine["dt"]))
    fractions = list(map(float, manifest["packet_fractions_of_E"]))
    diagnostic_subset = args.only_packet_fraction is not None
    if diagnostic_subset:
        matches = [
            value for value in fractions
            if np.isclose(value, float(args.only_packet_fraction))
        ]
        if len(matches) != 1:
            parser.error("only-packet-fraction is outside the frozen dose ladder")
        fractions = matches
    rows, onsets_rows, ranks_rows, active_rows = [], [], [], []
    for fraction in fractions:
        packet_n = max(1, int(round(float(fraction) * n_e)))
        for source in manifest["sources"]:
            indices = select_source_indices(
                positions, {"kind": "matched_off_field", "xy_mm": source["xy_mm"]},
                n_cells=packet_n,
            )
            packet_e = np.zeros(n_e, bool)
            packet_e[indices] = True
            packet_all = np.zeros(n_e + n_i, bool)
            packet_all[:n_e] = packet_e
            actual_center = positions[indices].mean(axis=0)
            radius = np.linalg.norm(positions[indices] - actual_center[None, :], axis=1)

            net["rng"] = np.random.default_rng(int(args.seed))
            forced = simulate_kick(
                params, net, KICK_BOOST=0.0, t_kick=1e9,
                V_th_per_neuron=node["vtheta"], forced_spike_mask=packet_all,
                forced_spike_ms=float(simulation["forced_spike_ms"]),
                early_stop_runaway=bool(simulation["early_stop_runaway"]),
            )
            forced_spikes = np.asarray(forced["E_spk_bool"], bool)
            pretrigger_identical = bool(np.array_equal(
                forced_spikes[:trigger_step], sham_spikes[:trigger_step],
            ))
            response_spikes = exclude_injected_packet_frame(
                forced_spikes, sham_spikes, packet_e, trigger_step=trigger_step,
            )
            response_envelope, response_dt, _ = snn_event_envelope(
                response_spikes, positions, montage, engine["dt"],
            )
            if response_dt != envelope_dt:
                raise RuntimeError("D4.1 forced/sham envelope dt changed")
            n_frame = min(response_envelope.shape[1], sham_envelope.shape[1])
            excess_envelope = np.clip(
                response_envelope[:, :n_frame] - sham_envelope[:, :n_frame],
                0.0, None,
            )
            onset, rank = _contact_onsets(
                excess_envelope, envelope_dt, montage, valid,
                (simulation["forced_spike_ms"], simulation["paired_response_end_ms"]),
                config["contact_readout"]["participation_margin_fraction"],
                config["contact_readout"]["timing_fraction"],
            )
            assigned = assign_direction_modes(
                onset[None, :], groups=groups, embedding=embedding,
                classifier=classifier,
            )
            label = "A" if int(assigned["labels"][0]) == 0 else "B"
            ood = bool(assigned["ood"][0])
            joint = bool(
                np.isfinite(onset[shaft_ids == "ICL"]).any()
                and np.isfinite(onset[shaft_ids == "SCL"]).any()
            )
            active, triggered, detected_events = _full_event_diagnostics(
                forced_spikes, cmrun=cmrun, detect_events=detect_events,
                detector=detector, trigger_ms=simulation["forced_spike_ms"],
                latency_ms=simulation["trigger_max_latency_ms"],
            )
            geometry = paired_excess_geometry(
                forced_spikes, sham_spikes, positions, packet_e,
                dt_ms=engine["dt"], start_ms=simulation["forced_spike_ms"],
                end_ms=simulation["paired_response_end_ms"],
                source_center=actual_center,
            )
            returned = bool(triggered is not None and triggered["returned"])
            expected = label == source["expected_mode"]
            clean = bool(
                pretrigger_identical and returned and joint and not ood and expected
                and forced["runaway_early_stop_ms"] is None
            )
            rows.append({
                "source_id": source["source_id"],
                "source_xy_mm": list(map(float, source["xy_mm"])),
                "expected_mode": source["expected_mode"],
                "packet_fraction_of_E": float(fraction),
                "packet_n_E": int(packet_n),
                "actual_center_xy_mm": actual_center.astype(float).tolist(),
                "packet_radius_p95_mm": float(np.quantile(radius, 0.95)),
                "pretrigger_spikes_bit_identical": pretrigger_identical,
                "forced_spike_collision_count": int(forced["forced_spike_collision_count"]),
                "triggered_event": triggered,
                "detected_events": detected_events,
                "runaway_early_stop_ms": forced["runaway_early_stop_ms"],
                "joint_shaft": joint,
                "assigned_mode": label,
                "expected_mode_match": expected,
                "probability_B": float(assigned["probability_B"][0]),
                "ood_distance": float(assigned["ood_distance"][0]),
                "ood": ood,
                "clean_expected_response": clean,
                "n_recruited_contacts": int(np.isfinite(onset).sum()),
                "mean_source_h": float(np.mean(node["h"][indices])),
                "downstream_positive_spike_mass": float(
                    geometry["downstream_positive_spike_mass"]
                ),
                "downstream_positive_neurons": int(
                    geometry["downstream_positive_neurons"]
                ),
                "r90_mm": geometry["r90_mm"],
                "peak_active_fraction": float(np.max(active, initial=0.0)),
            })
            onsets_rows.append(onset)
            ranks_rows.append(rank)
            if args.dump_active:
                active_rows.append(active)

    arrays = dict(
        contact_names=contact_names,
        shaft_ids=shaft_ids,
        source_ids=np.asarray([row["source_id"] for row in rows]),
        packet_fraction_of_E=np.asarray([
            row["packet_fraction_of_E"] for row in rows
        ], np.float32),
        onsets=np.asarray(onsets_rows, np.float32),
        ranks=np.asarray(ranks_rows, np.float32),
        labels=np.asarray([0 if row["assigned_mode"] == "A" else 1 for row in rows]),
        ood=np.asarray([row["ood"] for row in rows], bool),
        joint=np.asarray([row["joint_shaft"] for row in rows], bool),
        clean=np.asarray([row["clean_expected_response"] for row in rows], bool),
        sham_active_fraction=np.asarray(sham_active, np.float32),
    )
    if args.dump_active:
        arrays["forced_active_fraction"] = np.asarray(active_rows, np.float32)
    _atomic_npz(output_npz, **arrays)
    payload = {
        "status": (
            "REV10D4_1_PACKET_DOSE_TIMING_AUDIT_COMPLETE"
            if diagnostic_subset else
            "REV10D4_1_PACKET_DOSE_WORKER_COMPLETE"
        ),
        "scientific_role": config["scientific_role"],
        "seed": int(args.seed),
        "response_rows": rows,
        "sham": {
            "triggered_event": sham_triggered,
            "detected_events": sham_events,
            "runaway_early_stop_ms": sham["runaway_early_stop_ms"],
            "peak_active_fraction": float(np.max(sham_active, initial=0.0)),
        },
        "node_anchor": {
            "candidate_id": anchor["candidate_id"],
            "field_sha256": anchor["field_sha256"],
            "node_hashes": node["hashes"],
        },
        "network": {
            "n_E": int(n_e), "n_I": int(n_i),
            "cache_hit": bool(cache_hit), "cache_source": cache_source,
        },
        "arrays": {"path": str(output_npz), "sha256": _sha256(output_npz)},
        "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "config": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": _sha256(config_path),
        },
        "wall_seconds": float(time.time() - started),
        "diagnostic_subset": {
            "enabled": diagnostic_subset,
            "only_packet_fraction": (
                None if not diagnostic_subset else float(fractions[0])
            ),
            "active_fraction_dumped": bool(args.dump_active),
            "formal_verdict_unchanged": bool(diagnostic_subset),
        },
        "provenance": provenance,
    }
    atomic_write_json(payload, output_json)
    print(json.dumps({
        "status": payload["status"], "seed": args.seed,
        "clean_expected": int(sum(row["clean_expected_response"] for row in rows)),
        "wall_seconds": payload["wall_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
