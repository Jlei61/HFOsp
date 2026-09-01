"""Run one paired-network SA5 virtual-contact detectability worker."""
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
from scripts.run_topic4_rev9_node_kick_canary import _load_network  # noqa: E402
from scripts.run_topic4_rev9l_forced_source_worker import (  # noqa: E402
    _atomic_npz,
    _load_json_input,
    _runtime_provenance,
    _sha256,
)
from src.sef_hfo_observation import VirtualMontage  # noqa: E402
from src.topic4_core_field_runner import _placement, atomic_write_json  # noqa: E402
from src.topic4_rev10_sa_canary import (  # noqa: E402
    contact_response_metrics,
    lfp_kernel_audit,
    matched_contact_packets,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_sa_dual_shaft_canary.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--expected-commit")
    parser.add_argument("--out-json")
    parser.add_argument("--out-npz")
    parser.add_argument("--cache-dir")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    if config["scientific_role"] != (
        "development_only_contact_detectability_then_fixed_budget_dual_shaft_capacity"
    ):
        raise RuntimeError("SA5 scientific role changed")
    assay = config["sa5_contact_detectability"]
    if args.seed not in {int(seed) for seed in assay["network_seeds"]}:
        parser.error("--seed is outside the frozen SA5 seed set")
    if assay["substrate"] != "uniform_vthreshold_null":
        raise RuntimeError("SA5 must isolate the observation layer on Null thresholds")

    inputs = config["inputs"]
    base = _load_json_input(inputs["rev9_base_config"])
    stage = _load_json_input(inputs["stage_config"])
    contract = _load_json_input(inputs["contact_contract"])
    sa4 = _load_json_input(inputs["sa4_summary"])
    if sa4["status"] != "SA4_HISTORICAL_RESCORING_COMPLETE":
        raise RuntimeError("SA4 is not complete")
    contacts = contract["contacts"]
    contact_names = [row["contact_name"] for row in contacts]
    contact_xy = np.asarray([row["sheet_xy_mm"] for row in contacts], float)
    shaft_ids = np.asarray([row["shaft_id"] for row in contacts], dtype="U8")
    if len(contacts) != 15 or set(shaft_ids) != {"ICL", "SCL"}:
        raise RuntimeError("SA0 contact contract changed")

    output_root = ROOT / config["output_root"] / "contact_detectability"
    output_json = Path(args.out_json or output_root / "workers" / f"seed_{args.seed}.json")
    output_npz = Path(args.out_npz or output_root / "workers" / f"seed_{args.seed}.npz")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = str(Path(args.cache_dir or
                         ROOT / "results/topic4_sef_hfo/data_driven_core_field_rev9/network_cache"))

    provenance = _runtime_provenance(args.expected_commit)
    provenance["systemd_unit"] = os.environ.get("REV10SA_SYSTEMD_UNIT")
    config_dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(config_path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip())
    if provenance["runtime_modules_dirty"] or config_dirty:
        raise RuntimeError("SA5 runtime modules or config are dirty")
    if (args.expected_commit is not None
            and not provenance["runtime_modules_match_expected_commit"]):
        raise RuntimeError("SA5 runtime modules differ from launcher commit")

    started = time.time()
    engine = stage["engine"]
    reg = _placement(stage)
    cmrun = _load_cmrun()
    cmrun.DT = float(engine["dt"])
    params_cls = __import__("params").Params
    simulate_kick = __import__("kick_probe").simulate_kick
    lfp_recorder_cls = __import__("lfp").LFPRecorder
    params = params_cls(
        g=engine["g"], L=engine["L"], density=engine["density"],
        T=float(assay["duration_ms"]), dt=engine["dt"],
        nu_ext_ratio=cmrun.DRIVE, seed=int(args.seed),
    )
    net, n_e, n_i, cache_hit, cache_source = _load_network(
        params, stage, reg, int(args.seed), base, cache_dir,
    )
    positions = np.asarray(net["pos"][:n_e], float)
    requested_count = max(
        1, int(round(float(assay["requested_packet_fraction_of_E"]) * n_e)),
    )
    packets = matched_contact_packets(
        positions, contact_xy,
        radius_mm=float(assay["packet_radius_mm"]),
        requested_count=requested_count,
        minimum_count=int(assay["minimum_common_packet_count"]),
    )
    kernel = lfp_kernel_audit(
        positions, contact_xy, cutoff_mm=params.Rr, rx_mm=params.rx,
    )
    montage = VirtualMontage(
        contact_xy, contact_names, provenance="rev10_sa_frozen_contact_contract",
    )
    recorder = lfp_recorder_cls(
        params, net["pos"], net["labels"], sites=montage.contacts,
    )
    vtheta = np.full(n_e + n_i, float(engine["v_base"]))
    dynamics_seed = int(args.seed)
    net["rng"] = np.random.default_rng(dynamics_seed)
    sham = simulate_kick(
        params, net, KICK_BOOST=0.0, t_kick=1e9,
        V_th_per_neuron=vtheta, lfp_recorder=recorder,
        early_stop_runaway=True,
    )
    if sham["runaway_early_stop_ms"] is not None:
        raise RuntimeError("SA5 sham entered runaway")
    trigger_step = int(round(float(assay["forced_spike_ms"]) / engine["dt"]))
    rows = []
    forced_lfp = []
    packet_spike_traces = []
    local_spike_traces = []
    for contact_index, (name, shaft, xy, packet_e) in enumerate(zip(
        contact_names, shaft_ids, contact_xy, packets["masks"],
    )):
        packet_all = np.zeros(n_e + n_i, bool)
        packet_all[:n_e] = packet_e
        net["rng"] = np.random.default_rng(dynamics_seed)
        forced = simulate_kick(
            params, net, KICK_BOOST=0.0, t_kick=1e9,
            V_th_per_neuron=vtheta, lfp_recorder=recorder,
            forced_spike_mask=packet_all,
            forced_spike_ms=float(assay["forced_spike_ms"]),
            early_stop_runaway=True,
        )
        if forced["runaway_early_stop_ms"] is not None:
            raise RuntimeError(f"SA5 forced packet at {name} entered runaway")
        pretrigger_spikes = np.array_equal(
            forced["E_spk_bool"][:trigger_step], sham["E_spk_bool"][:trigger_step],
        )
        pretrigger_lfp = np.array_equal(
            forced["lfp_trace"][:trigger_step], sham["lfp_trace"][:trigger_step],
        )
        if not pretrigger_spikes or not pretrigger_lfp:
            raise RuntimeError(f"paired randomness mismatch before {name} packet")
        metrics = contact_response_metrics(
            forced["lfp_trace"][:, contact_index],
            sham["lfp_trace"][:, contact_index],
            forced["E_spk_bool"], sham["E_spk_bool"], positions, xy, packet_e,
            dt_ms=float(engine["dt"]),
            forced_spike_ms=float(assay["forced_spike_ms"]),
            response_stop_ms=float(assay["response_stop_ms"]),
            baseline_window_ms=assay["baseline_window_ms"],
            local_radius_mm=float(assay["local_neural_radius_mm"]),
        )
        local_mask = np.linalg.norm(positions - xy, axis=1) <= float(
            assay["local_neural_radius_mm"])
        rows.append({
            "contact_name": name,
            "contact_index": int(contact_index),
            "shaft_id": str(shaft),
            "sheet_xy_mm": xy.tolist(),
            "available_neurons_in_fixed_radius": int(
                packets["available_counts"][contact_index]
            ),
            "packet_n_E": int(packets["common_count"]),
            "packet_radius_mm": float(assay["packet_radius_mm"]),
            "packet_max_distance_mm": float(np.linalg.norm(
                positions[packet_e] - xy, axis=1,
            ).max()),
            "pretrigger_spikes_bit_identical": pretrigger_spikes,
            "pretrigger_lfp_bit_identical": pretrigger_lfp,
            "forced_spike_collision_count": int(
                forced["forced_spike_collision_count"]
            ),
            "lfp_kernel": kernel[contact_index],
            **metrics,
        })
        forced_lfp.append(np.asarray(forced["lfp_trace"], np.float32))
        packet_spike_traces.append(np.asarray(
            forced["E_spk_bool"][:, packet_e].sum(axis=1), np.int16,
        ))
        local_spike_traces.append(np.asarray(
            forced["E_spk_bool"][:, local_mask].sum(axis=1), np.int16,
        ))
        print(json.dumps({
            "progress": "contact_complete", "seed": args.seed,
            "contact": name, "shaft": str(shaft),
            "lfp_gain": metrics["peak_lfp_excess_per_packet_cell"],
            "local_neural_gain": metrics["local_positive_spike_excess_per_cell"],
        }), flush=True)

    _atomic_npz(
        output_npz,
        contact_names=np.asarray(contact_names, dtype="U16"),
        shaft_ids=shaft_ids,
        contact_xy_mm=contact_xy,
        packet_masks_E=np.asarray(packets["masks"], bool),
        packet_available_counts=np.asarray(packets["available_counts"], np.int64),
        packet_common_count=np.asarray(packets["common_count"], np.int64),
        times_ms=np.asarray(sham["times"], np.float32),
        sham_lfp=np.asarray(sham["lfp_trace"], np.float32),
        forced_lfp=np.asarray(forced_lfp, np.float32),
        packet_spike_count=np.asarray(packet_spike_traces, np.int16),
        local_spike_count=np.asarray(local_spike_traces, np.int16),
    )
    payload = {
        "status": "SA5_CONTACT_DETECTABILITY_WORKER_COMPLETE",
        "scientific_role": (
            "paired uniform-threshold contact observation audit; no patient score "
            "and no dual-shaft field capacity conclusion"
        ),
        "seed": int(args.seed),
        "substrate": assay["substrate"],
        "packet": {
            "requested_count": requested_count,
            "common_count": packets["common_count"],
            "fixed_radius_mm": packets["radius_mm"],
            "minimum_available_count": int(packets["available_counts"].min()),
            "maximum_available_count": int(packets["available_counts"].max()),
        },
        "contacts": rows,
        "network": {
            "n_E": int(n_e), "n_I": int(n_i),
            "cache_hit": bool(cache_hit), "cache_source": cache_source,
        },
        "simulation": {
            "duration_ms": float(assay["duration_ms"]),
            "forced_spike_ms": float(assay["forced_spike_ms"]),
            "response_stop_ms": float(assay["response_stop_ms"]),
            "wall_seconds": float(time.time() - started),
        },
        "arrays": {"path": str(output_npz), "sha256": _sha256(output_npz)},
        "config": {"path": str(config_path.relative_to(ROOT)),
                   "sha256": _sha256(config_path)},
        "provenance": provenance,
    }
    atomic_write_json(payload, output_json)
    print(json.dumps({
        "status": payload["status"], "seed": args.seed,
        "packet_common_count": packets["common_count"],
        "wall_seconds": payload["simulation"]["wall_seconds"],
        "arrays_sha256": payload["arrays"]["sha256"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
