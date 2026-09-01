"""Run one rev9 spontaneous arm on one frozen network seed."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join("src", "snn_engine"))
from scripts.run_topic4_core_field_stage3_fit import _load_cmrun  # noqa: E402
from scripts.run_topic4_core_field_stage3_profile_round1 import axial_map  # noqa: E402
from scripts.run_topic4_rev9_node_kick_canary import (  # noqa: E402
    _atomic_npz,
    _candidate,
    _git,
    _load_network,
    _sha256,
)
from src.topic4_core_connectivity import field_normalized_ee_pair  # noqa: E402
from src.topic4_core_field_profile import normalized_rank_curve  # noqa: E402
from src.topic4_core_field_rev9 import reconstruct_frozen_node  # noqa: E402
from src.topic4_core_field_runner import (  # noqa: E402
    _placement,
    atomic_write_json,
    provenance,
)
from src.topic4_rev9_factorial import arm_contract  # noqa: E402


DEFAULT_CONFIG = "config/topic4_rev9_factorial.json"


def _load_contract(path):
    factorial = json.loads(Path(path).read_text())
    base_path = factorial["base_config"]["path"]
    if _sha256(base_path) != factorial["base_config"]["sha256"]:
        raise RuntimeError("rev9 factorial base-config hash mismatch")
    base = json.loads(Path(base_path).read_text())
    frozen = factorial["frozen_readouts"]
    if _sha256(frozen["json"]) != frozen["json_sha256"]:
        raise RuntimeError("rev9 frozen-readout JSON hash mismatch")
    frozen_summary = json.loads(Path(frozen["json"]).read_text())
    if _sha256(frozen["npz"]) != frozen_summary["arrays"]["sha256"]:
        raise RuntimeError("rev9 frozen-readout array hash mismatch")
    alpha = factorial["alpha_reference"]
    if _sha256(alpha["selection_summary"]) != alpha["selection_summary_sha256"]:
        raise RuntimeError("rev9 alpha-selection summary hash mismatch")
    alpha_summary = json.loads(Path(alpha["selection_summary"]).read_text())
    if not np.isclose(alpha_summary["selection"]["alpha_star"], alpha["value"]):
        raise RuntimeError("configured alpha does not reproduce alpha_star")
    return factorial, base, frozen_summary, alpha_summary


def _event_histogram(points, *, sheet_length, n_bins):
    edges = np.linspace(0.0, float(sheet_length), int(n_bins) + 1)
    if not len(points):
        return np.zeros((int(n_bins), int(n_bins)), np.float32), edges
    histogram, _, _ = np.histogram2d(
        points[:, 1], points[:, 0], bins=(edges, edges))
    if histogram.sum() > 0:
        histogram /= histogram.sum()
    return np.asarray(histogram, np.float32), edges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--arm", required=True,
                        choices=("Null", "Node", "Edge", "Node+Edge"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--out-json")
    parser.add_argument("--out-npz")
    parser.add_argument("--cache-dir")
    parser.add_argument("--capture-lfp", action="store_true")
    args = parser.parse_args()

    factorial, base, frozen_summary, alpha_summary = _load_contract(args.config)
    if int(args.seed) not in {int(value) for value in factorial["seeds"]}:
        parser.error("--seed is outside the frozen factorial seed set")
    switches = arm_contract(args.arm)
    alpha = float(factorial["alpha_reference"]["value"])
    stage = json.loads(Path(base["inputs"]["stage_config"]).read_text())
    candidate = _candidate(base)
    engine = stage["engine"]
    reg = _placement(stage)
    slug = args.arm.lower().replace("+", "_")
    output_root = Path(factorial["output_root"])
    suffix = "_capture" if args.capture_lfp else ""
    output_json = Path(args.out_json or output_root / "workers" /
                       f"{slug}_seed{args.seed}{suffix}.json")
    output_npz = Path(args.out_npz or output_root / "workers" /
                      f"{slug}_seed{args.seed}{suffix}.npz")
    cache_dir = str(Path(args.cache_dir or Path(base["output_root"]) /
                         "network_cache"))

    execution_provenance = dict(
        **provenance(),
        git_status_porcelain=_git("status", "--porcelain"),
        producer_sha256=_sha256(__file__),
        factorial_config_sha256=_sha256(args.config),
        python_executable=sys.executable,
        python_version=platform.python_version(),
        systemd_unit=os.environ.get("REV9_SYSTEMD_UNIT"),
        network_seed=int(args.seed), ou_seed=int(args.seed),
        poisson_seed=int(args.seed), readout_seed=None,
    )
    started = time.time()
    cmrun = _load_cmrun()
    cmrun.DT = float(engine["dt"])
    cmrun.KDIR = int(engine["k_dir"])
    cmrun.PART_MIN = 2 * int(engine["k_dir"]) + 1
    params_cls = __import__("params").Params
    simulate_kick = __import__("kick_probe").simulate_kick
    lfp_recorder_cls = __import__("lfp").LFPRecorder
    detect_events = __import__(
        "src.sef_hfo_events", fromlist=["detect_events"]).detect_events
    snn_event_envelope = __import__(
        "src.sef_hfo_snn_adapter", fromlist=["snn_event_envelope"]
    ).snn_event_envelope

    params = params_cls(
        g=engine["g"], L=engine["L"], density=engine["density"],
        T=float(factorial["simulation_duration_ms"]), dt=engine["dt"],
        nu_ext_ratio=cmrun.DRIVE, seed=int(args.seed))
    net, n_e, n_i, cache_hit, cache_source = _load_network(
        params, stage, reg, int(args.seed), base, cache_dir)
    pos_e = np.asarray(net["pos"][:n_e], float)
    node = reconstruct_frozen_node(
        candidate["theta"], pos_e, n_total=n_e + n_i,
        target_count=stage["N_core_manual"],
        quantile_seed=stage["quantile_seed"],
        core_mean=engine["core_mean"], core_std=engine["core_std"],
        v_base=engine["v_base"], K=candidate["K"], L=engine["L"])
    edge_diagnostics = None
    if switches["edge"]:
        net, edge_diagnostics = field_normalized_ee_pair(
            net, node["h"], alpha, beta=0.0,
            active_vth_shift=node["delta_vtheta"])
    vtheta = (node["vtheta"] if switches["node"] else
              np.full(n_e + n_i, float(engine["v_base"])))

    montage = reg["montage_sheet"]
    valid_contacts = cmrun.valid_mask(
        montage, pos_e, engine["L"], params.Rr)
    recorder = (lfp_recorder_cls(
        params, net["pos"], net["labels"], sites=montage.contacts)
        if args.capture_lfp else None)
    net["rng"] = np.random.default_rng(int(args.seed))
    result = simulate_kick(
        params, net, KICK_BOOST=0.0, kick_center=list(reg["center"]),
        r_kick=engine["core_r"], t_kick=1e9,
        V_th_per_neuron=vtheta, lfp_recorder=recorder,
        early_stop_runaway=True)
    spikes = np.asarray(result["E_spk_bool"], bool)
    active_fraction, bin_width = cmrun.active_fraction(
        spikes, engine["dt"], cmrun.BIN_MS)
    if not np.isfinite(active_fraction).all():
        raise RuntimeError("nonfinite active-fraction trajectory")
    baseline_start = int(cmrun.BASELINE_MS[0] / bin_width)
    baseline_stop = min(len(active_fraction),
                        int(cmrun.BASELINE_MS[1] / bin_width))
    if baseline_stop <= baseline_start:
        floor = float(np.min(active_fraction, initial=0.0))
    else:
        floor = float(np.percentile(
            active_fraction[baseline_start:baseline_stop], 95))
    peak = float(np.max(active_fraction, initial=0.0))
    detector_threshold = floor + cmrun.CAL_FRAC * (peak - floor)
    detected = detect_events(
        active_fraction, bin_width, event_on_frac=detector_threshold)
    envelope, envelope_dt, _ = snn_event_envelope(
        spikes, pos_e, montage, engine["dt"])
    axial = axial_map()
    profile_arrays = np.load(base["inputs"]["profiles"])
    grid = np.asarray(profile_arrays["grid"], float)
    contact_names = sorted(axial, key=axial.get)
    n_bins = int(factorial["onset_density_bins_per_axis"])
    early_quantile = float(factorial["early_onset_quantile"])

    event_rows = []
    curves = np.full((len(detected), len(grid)), np.nan, np.float32)
    ranks = np.full((len(detected), len(contact_names)), np.nan, np.float32)
    early_density = np.zeros((len(detected), n_bins, n_bins), np.float32)
    early_centroids = np.full((len(detected), 2), np.nan, np.float32)
    density_edges = np.linspace(0.0, float(engine["L"]), n_bins + 1)
    for local_index, event in enumerate(detected):
        readout = cmrun.read_event(
            envelope, envelope_dt, montage, valid_contacts,
            (event["t_on"], event["t_off"]), reg["axis_unit_vec"],
            k_dir=int(engine["k_dir"]),
            part_min=2 * int(engine["k_dir"]) + 1)
        rank_dict = readout.get("ranks") or {}
        ranks[local_index] = [
            np.nan if rank_dict.get(name) is None else float(rank_dict[name])
            for name in contact_names]
        curve = normalized_rank_curve(rank_dict, axial, grid=grid)
        if curve is not None:
            curves[local_index] = np.asarray(curve, np.float32)
        onset = np.asarray(cmrun.per_neuron_onset(
            spikes, event["t_on"], event["t_off"], engine["dt"]), float)
        finite = np.isfinite(onset)
        early_points = np.empty((0, 2), float)
        if finite.any():
            relative = onset[finite] - float(np.min(onset[finite]))
            threshold = float(np.quantile(relative, early_quantile))
            early_points = pos_e[finite][relative <= threshold]
            if len(early_points):
                early_centroids[local_index] = early_points.mean(axis=0)
        histogram, density_edges = _event_histogram(
            early_points, sheet_length=engine["L"], n_bins=n_bins)
        early_density[local_index] = histogram
        event_rows.append(dict(
            local_event_index=int(local_index),
            t_on_ms=float(event["t_on"]), t_off_ms=float(event["t_off"]),
            duration_ms=float(event["t_off"] - event["t_on"]),
            returned=bool(event.get("returned", False)),
            n_part=int(readout.get("n_part", 0)),
            n_active_neurons=int(finite.sum()),
            curve_usable=bool(curve is not None),
            early_centroid=(None if not len(early_points) else
                            early_points.mean(axis=0).tolist()),
        ))

    arrays = dict(
        grid=np.asarray(grid, np.float32),
        contact_names=np.asarray(contact_names, dtype="U32"),
        valid_contacts=np.asarray(valid_contacts, bool),
        event_local_indices=np.arange(len(detected), dtype=np.int64),
        event_t_on_ms=np.asarray([row["t_on_ms"] for row in event_rows], float),
        event_t_off_ms=np.asarray([row["t_off_ms"] for row in event_rows], float),
        event_returned=np.asarray([row["returned"] for row in event_rows], bool),
        event_n_part=np.asarray([row["n_part"] for row in event_rows], np.int64),
        event_n_active_neurons=np.asarray(
            [row["n_active_neurons"] for row in event_rows], np.int64),
        event_curves=curves,
        event_ranks=ranks,
        event_early_density=early_density,
        event_early_centroids=early_centroids,
        density_edges=np.asarray(density_edges, np.float32),
        active_fraction=np.asarray(active_fraction, np.float32),
        bin_width_ms=np.asarray(bin_width, float),
        h=np.asarray(node["h"], np.float32),
        delta_vtheta=np.asarray(
            node["delta_vtheta"] if switches["node"] else
            np.zeros(n_e), np.float32),
    )
    if args.capture_lfp:
        arrays.update(
            lfp_trace=np.asarray(result["lfp_trace"], np.float32),
            times=np.asarray(result["times"], np.float32),
            contacts=np.asarray(montage.contacts, np.float32),
            pos_e=np.asarray(pos_e, np.float32),
            vtheta_e=np.asarray(vtheta[:n_e], np.float32),
        )
    _atomic_npz(output_npz, **arrays)

    runaway = result.get("runaway_early_stop_ms")
    status = ("REV9_FACTORIAL_RUNAWAY_EARLY_STOP" if runaway is not None
              else "REV9_FACTORIAL_WORKER_COMPLETE")
    payload = dict(
        status=status,
        scientific_role=(
            "one exploratory spontaneous arm-seed measurement; no patient "
            "held-out access and no arm-level inference"),
        arm=args.arm, switches=switches, seed=int(args.seed), alpha=alpha,
        alpha_role=factorial["alpha_reference"]["role"],
        simulation=dict(
            requested_duration_ms=float(factorial["simulation_duration_ms"]),
            simulated_until_ms=float(len(spikes) * engine["dt"]),
            wall_seconds=float(result["wall_s"]),
            runaway_early_stop_ms=(None if runaway is None else float(runaway)),
            active_fraction_floor=floor, active_fraction_peak=peak,
            detector_threshold=float(detector_threshold),
            n_detected=int(len(detected)),
            n_usable=int(np.isfinite(curves).all(axis=1).sum()),
        ),
        events=event_rows,
        network=dict(
            cache_hit=bool(cache_hit), cache_source=cache_source,
            n_E=int(n_e), n_I=int(n_i), node_hashes=node["hashes"],
            edge_diagnostics=edge_diagnostics),
        capture_lfp=bool(args.capture_lfp),
        arrays=dict(path=str(output_npz), sha256=_sha256(output_npz)),
        inputs=dict(
            factorial_config=dict(path=args.config, sha256=_sha256(args.config)),
            base_config=factorial["base_config"],
            frozen_readouts=factorial["frozen_readouts"],
            alpha_selection=dict(
                path=factorial["alpha_reference"]["selection_summary"],
                sha256=factorial["alpha_reference"]["selection_summary_sha256"],
                status=alpha_summary["status"]),
            candidate_id=candidate["candidate_id"],
            theta_sha256=candidate["theta_sha256"],
            frozen_readout_status=frozen_summary["status"],
        ),
        elapsed_seconds=float(time.time() - started),
        provenance=execution_provenance,
    )
    atomic_write_json(payload, output_json)
    print(json.dumps(dict(
        status=status, arm=args.arm, seed=int(args.seed),
        n_detected=len(detected), n_usable=payload["simulation"]["n_usable"],
        runaway_early_stop_ms=payload["simulation"]["runaway_early_stop_ms"],
        wall_seconds=payload["simulation"]["wall_seconds"],
        arrays_sha256=payload["arrays"]["sha256"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
