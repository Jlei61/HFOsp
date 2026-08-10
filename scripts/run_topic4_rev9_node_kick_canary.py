"""Run paired rev9 Node or Edge small-kick response measurements."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join("src", "snn_engine"))
from scripts.run_topic4_core_field_stage3_fit import _load_cmrun  # noqa: E402
from src.topic4_core_connectivity import field_normalized_ee_pair  # noqa: E402
from src.topic4_core_field_rev9 import reconstruct_frozen_node  # noqa: E402
from src.topic4_core_field_runner import (  # noqa: E402
    CONNECTIVITY_FIELDS,
    _placement,
    atomic_write_json,
    canonical_checksum,
    get_network,
    provenance,
)
from src.topic4_rev9_local_response import (  # noqa: E402
    event_window_overlap,
    fit_response_slope,
    paired_spike_response,
)


DEFAULT_CONFIG = "config/topic4_rev9_exploratory.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args, default="unknown"):
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return default


def _atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _event_summary(result, cmrun, detect_events, *, interval_ms):
    spikes = np.asarray(result["E_spk_bool"], bool)
    active_fraction, bin_width = cmrun.active_fraction(spikes, cmrun.DT, cmrun.BIN_MS)
    baseline_start = int(cmrun.BASELINE_MS[0] / bin_width)
    baseline_stop = min(len(active_fraction), int(cmrun.BASELINE_MS[1] / bin_width))
    if baseline_stop <= baseline_start:
        floor = float(np.min(active_fraction, initial=0.0))
    else:
        floor = float(np.percentile(
            active_fraction[baseline_start:baseline_stop], 95))
    peak = float(np.max(active_fraction, initial=0.0))
    threshold = floor + cmrun.CAL_FRAC * (peak - floor)
    events = detect_events(active_fraction, bin_width, event_on_frac=threshold)
    overlapping = [event for event in events
                   if event["t_on"] < interval_ms[1]
                   and event["t_off"] > interval_ms[0]]
    return dict(
        active_fraction_floor=floor, active_fraction_peak=peak,
        detector_threshold=float(threshold), n_events=int(len(events)),
        n_events_in_response_interval=int(len(overlapping)),
        response_interval_event=bool(overlapping),
        events_in_response_interval=[dict(
            t_on=float(event["t_on"]), t_off=float(event["t_off"]))
            for event in overlapping],
        runaway_early_stop_ms=result["runaway_early_stop_ms"],
        simulated_until_ms=float(len(spikes) * cmrun.DT),
        wall_seconds=float(result["wall_s"]),
    )


def _candidate(config):
    confirmation = json.loads(Path(config["inputs"]["confirmation"]).read_text())
    rows = [row for row in confirmation["candidates"]
            if row["candidate_id"] == config["inputs"]["candidate_id"]]
    if len(rows) != 1 or rows[0]["theta_sha256"] != config["inputs"]["theta_sha256"]:
        raise RuntimeError("configured frozen candidate or theta hash does not reproduce")
    return rows[0]


def _load_network(params, stage, reg, seed, config, cache_dir):
    source_path = config["small_kick_instrument"].get(
        "network_cache_source_artifact")
    if source_path:
        source = json.loads(Path(source_path).read_text())
        source_commit = source["provenance"]["git_commit"]
        source_numpy = source["provenance"]["numpy_version"]
        cache_config = {
            field: getattr(params, field) for field in CONNECTIVITY_FIELDS
        }
        cache_config.update(
            theta_EE_deg=float(reg["theta_deg"]),
            AR=float(stage["engine"]["AR"]),
            numpy_version=str(source_numpy), rng_bit_generator="PCG64",
            git_commit=str(source_commit))
        key = canonical_checksum(cache_config, drop=())
        path = Path(cache_dir) / f"{key}.pkl"
        if not path.exists():
            raise RuntimeError(f"frozen canary network cache is missing: {path}")
        with open(path, "rb") as handle:
            cached = pickle.load(handle)
        return (
            cached["net"], int(cached["NE"]), int(cached["NI"]), True,
            dict(source_artifact=source_path, source_commit=source_commit,
                 cache_key=key, cache_path=str(path), cache_sha256=_sha256(path)))
    net, n_e, n_i, hit = get_network(
        params, reg["theta_deg"], stage["engine"]["AR"], cache_dir)
    return net, n_e, n_i, hit, dict(source_artifact=None)


def _without_maps(row):
    return {key: value for key, value in row.items()
            if key not in {"signed_map_per_cell", "positive_map_per_cell",
                           "spatial_edges"}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--out-json")
    parser.add_argument("--out-npz")
    parser.add_argument("--cache-dir")
    parser.add_argument("--arm", choices=("Node", "Edge"), default="Node")
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    if args.arm == "Node" and args.alpha != 0.0:
        parser.error("--alpha is only defined for the Edge arm")
    if args.alpha < 0.0 or not np.isfinite(args.alpha):
        parser.error("--alpha must be finite and non-negative")

    config = json.loads(Path(args.config).read_text())
    instrument = config["small_kick_instrument"]
    stage = json.loads(Path(config["inputs"]["stage_config"]).read_text())
    candidate = _candidate(config)
    output_root = Path(config["output_root"]) / "node_edge_calibration"
    output_json = Path(args.out_json or output_root / "node_kick_canary.json")
    output_npz = Path(args.out_npz or output_root / "node_kick_canary.npz")
    cache_dir = str(Path(args.cache_dir or Path(config["output_root"]) / "network_cache"))

    engine = stage["engine"]
    reg = _placement(stage)
    cmrun = _load_cmrun()
    cmrun.DT = float(engine["dt"])
    params_cls = __import__("params").Params
    compute_nu_theta = __import__("params").compute_nu_theta
    simulate_kick = __import__("kick_probe").simulate_kick
    detect_events = __import__(
        "src.sef_hfo_events", fromlist=["detect_events"]).detect_events

    seeds = [int(value) for value in (
        args.seeds if args.seeds is not None else instrument["canary_seeds"])]
    sites = list(instrument["origins"])
    multipliers = np.asarray(instrument["amplitude_multipliers"], float)
    windows = np.asarray(instrument["candidate_windows_after_pulse_end_ms"], float)
    pulse_onset = float(instrument["kick_onset_ms"])
    pulse_end = pulse_onset + float(instrument["kick_duration_ms"])
    response_interval = (pulse_onset, pulse_end + float(windows[:, 1].max()))
    n_seed, n_site, n_amp, n_window = (
        len(seeds), len(sites), len(multipliers), len(windows))
    n_spatial = int(instrument["spatial_bins_per_axis"])
    scalar_names = (
        "source_signed_per_cell", "downstream_signed_per_cell",
        "source_positive_per_cell", "downstream_positive_per_cell",
        "positive_mass", "r50_mm", "r90_mm", "axis_variance_ratio",
    )
    scalars = {
        key: np.full((n_seed, n_site, n_amp, n_window), np.nan)
        for key in scalar_names
    }
    signed_maps = np.full(
        (n_seed, n_site, n_amp, n_window, n_spatial, n_spatial), np.nan,
        dtype=np.float32)
    positive_maps = np.full_like(signed_maps, np.nan)
    response_event = np.zeros((n_seed, n_site, n_amp), bool)
    window_event_overlap = np.zeros(
        (n_seed, n_site, n_amp, n_window), bool)
    runaway_ms = np.full((n_seed, n_site, n_amp), np.nan)
    simulation_wall = np.full((n_seed, n_site, n_amp), np.nan)
    records, sham_records, network_records = [], [], []
    actual_amplitudes = None
    started = time.time()
    package_lock = "requirements.txt"
    execution_provenance = dict(
        **provenance(),
        git_status_porcelain=_git("status", "--porcelain"),
        producer_sha256=_sha256(__file__),
        config_sha256=_sha256(args.config),
        python_executable=sys.executable,
        python_version=platform.python_version(),
        package_lock=dict(path=package_lock, sha256=_sha256(package_lock)),
        systemd_unit=os.environ.get("REV9_SYSTEMD_UNIT"),
    )
    execution_inputs = dict(
        config=dict(path=args.config, sha256=_sha256(args.config)),
        stage_config=dict(
            path=config["inputs"]["stage_config"],
            sha256=_sha256(config["inputs"]["stage_config"])),
        confirmation=dict(
            path=config["inputs"]["confirmation"],
            sha256=_sha256(config["inputs"]["confirmation"])),
        candidate_id=candidate["candidate_id"],
        theta_sha256=candidate["theta_sha256"],
    )

    for seed_index, seed in enumerate(seeds):
        params = params_cls(
            g=engine["g"], L=engine["L"], density=engine["density"],
            T=instrument["simulation_duration_ms"], dt=engine["dt"],
            nu_ext_ratio=cmrun.DRIVE, seed=seed)
        nu_theta = float(compute_nu_theta(params)[0])
        amplitudes = multipliers * nu_theta
        if actual_amplitudes is None:
            actual_amplitudes = amplitudes
        elif not np.array_equal(actual_amplitudes, amplitudes):
            raise RuntimeError("nu_theta changed across paired network seeds")
        network_started = time.time()
        net, n_e, n_i, cache_hit, cache_source = _load_network(
            params, stage, reg, seed, config, cache_dir)
        node = reconstruct_frozen_node(
            candidate["theta"], net["pos"][:n_e], n_total=n_e + n_i,
            target_count=stage["N_core_manual"],
            quantile_seed=stage["quantile_seed"],
            core_mean=engine["core_mean"], core_std=engine["core_std"],
            v_base=engine["v_base"], K=candidate["K"], L=engine["L"])
        edge_diagnostics = None
        if args.arm == "Edge":
            net, edge_diagnostics = field_normalized_ee_pair(
                net, node["h"], args.alpha, beta=0.0,
                active_vth_shift=node["delta_vtheta"])
            vtheta = np.full(n_e + n_i, float(engine["v_base"]))
        else:
            vtheta = node["vtheta"]
        network_records.append(dict(
            seed=seed, cache_hit=bool(cache_hit), n_E=int(n_e), n_I=int(n_i),
            setup_seconds=float(time.time() - network_started),
            node_hashes=node["hashes"], cache_source=cache_source,
            edge_diagnostics=edge_diagnostics))

        net["rng"] = np.random.default_rng(seed)
        sham = simulate_kick(
            params, net, KICK_BOOST=0.0,
            kick_center=sites[0]["xy_mm"],
            r_kick=instrument["radius_mm"], t_kick=pulse_onset,
            V_th_per_neuron=vtheta, early_stop_runaway=True)
        sham_summary = _event_summary(
            sham, cmrun, detect_events, interval_ms=response_interval)
        sham_summary.update(seed=seed, arm=args.arm, alpha=float(args.alpha),
                            rng_seed=seed)
        sham_records.append(sham_summary)
        print(json.dumps(dict(
            progress="sham_complete", arm=args.arm, alpha=float(args.alpha),
            seed=seed,
            wall_seconds=sham_summary["wall_seconds"],
            response_event=sham_summary["response_interval_event"])), flush=True)

        for site_index, site in enumerate(sites):
            for amplitude_index, (multiplier, amplitude) in enumerate(
                    zip(multipliers, amplitudes)):
                net["rng"] = np.random.default_rng(seed)
                kick = simulate_kick(
                    params, net, KICK_BOOST=float(amplitude),
                    kick_center=site["xy_mm"],
                    r_kick=instrument["radius_mm"], t_kick=pulse_onset,
                    V_th_per_neuron=vtheta, early_stop_runaway=True)
                event = _event_summary(
                    kick, cmrun, detect_events, interval_ms=response_interval)
                measured = paired_spike_response(
                    kick["E_spk_bool"], sham["E_spk_bool"], net["pos"][:n_e],
                    site["xy_mm"], reg["axis_unit_vec"], dt=engine["dt"],
                    pulse_end_ms=pulse_end, windows_after_ms=windows,
                    source_radius_mm=instrument["radius_mm"], L=engine["L"],
                    spatial_bins_per_axis=n_spatial)
                response_event[seed_index, site_index, amplitude_index] = bool(
                    event["response_interval_event"]
                    or sham_summary["response_interval_event"])
                window_event_overlap[
                    seed_index, site_index, amplitude_index] = event_window_overlap(
                        event["events_in_response_interval"], pulse_end, windows)
                if sham_summary["response_interval_event"]:
                    window_event_overlap[
                        seed_index, site_index, amplitude_index] |= event_window_overlap(
                            sham_summary["events_in_response_interval"],
                            pulse_end, windows)
                if event["runaway_early_stop_ms"] is not None:
                    runaway_ms[seed_index, site_index, amplitude_index] = float(
                        event["runaway_early_stop_ms"])
                simulation_wall[seed_index, site_index, amplitude_index] = event[
                    "wall_seconds"]
                serialized_windows = []
                for window_index, row in enumerate(measured["windows"]):
                    serialized_windows.append(_without_maps(row))
                    if row["status"] != "ok":
                        continue
                    for key in scalar_names:
                        value = row[key]
                        if value is not None:
                            scalars[key][seed_index, site_index,
                                         amplitude_index, window_index] = value
                    signed_maps[seed_index, site_index, amplitude_index,
                                window_index] = row["signed_map_per_cell"]
                    positive_maps[seed_index, site_index, amplitude_index,
                                  window_index] = row["positive_map_per_cell"]
                records.append(dict(
                    seed=seed, site_id=site["id"], site_role=site["role"],
                    arm=args.arm, alpha=float(args.alpha),
                    site_xy_mm=site["xy_mm"], amplitude_multiplier=float(multiplier),
                    kick_boost_1_per_ms=float(amplitude), response=serialized_windows,
                    event=event, paired_sham_response_event=bool(
                        sham_summary["response_interval_event"]),
                    linear_eligible=bool(
                        measured["status"] == "ok"
                        and not response_event[seed_index, site_index, amplitude_index]
                        and event["runaway_early_stop_ms"] is None),
                ))
                print(json.dumps(dict(
                    progress="kick_complete", arm=args.arm,
                    alpha=float(args.alpha), seed=seed, site=site["id"],
                    amplitude_multiplier=float(multiplier),
                    wall_seconds=event["wall_seconds"],
                    response_event=bool(response_event[
                        seed_index, site_index, amplitude_index]),
                    runaway_ms=event["runaway_early_stop_ms"])), flush=True)
                del kick
        del sham, net, node

    selection_multiplier = float(instrument["window_selection_amplitude_multiplier"])
    amplitude_matches = np.flatnonzero(np.isclose(multipliers, selection_multiplier))
    if len(amplitude_matches) != 1:
        raise RuntimeError("window-selection amplitude is absent or ambiguous")
    selection_amplitude_index = int(amplitude_matches[0])
    site_seed_window_eligible = (
        ~window_event_overlap.any(axis=2)
        & ~np.isfinite(runaway_ms).any(axis=2)[:, :, None])
    primary_window = np.asarray(
        instrument["primary_window_after_pulse_end_ms"], float)
    primary_matches = np.flatnonzero(np.all(np.isclose(windows, primary_window), axis=1))
    if len(primary_matches) != 1:
        raise RuntimeError("primary response window is absent or ambiguous")
    selected_window_index = int(primary_matches[0])
    site_seed_linear_eligible = site_seed_window_eligible[
        :, :, selected_window_index]
    eligible_seed_mask = site_seed_linear_eligible.any(axis=1)
    n_eligible_seeds = int(eligible_seed_mask.sum())
    selection_values = scalars["downstream_positive_per_cell"][
        :, :, selection_amplitude_index, :]
    window_medians = np.asarray([
        np.nanmedian(np.where(
            site_seed_window_eligible[:, :, window_index],
            selection_values[:, :, window_index], np.nan))
        if site_seed_window_eligible[:, :, window_index].any() else np.nan
        for window_index in range(n_window)
    ])

    source_slopes = np.full((n_seed, n_site, n_window), np.nan)
    downstream_slopes = np.full_like(source_slopes, np.nan)
    slope_records = []
    for seed_index, seed in enumerate(seeds):
        for site_index, site in enumerate(sites):
            for window_index in range(n_window):
                eligible = bool(site_seed_window_eligible[
                    seed_index, site_index, window_index])
                source = fit_response_slope(
                    actual_amplitudes if eligible else [],
                    (scalars["source_signed_per_cell"][
                        seed_index, site_index, :, window_index]
                     if eligible else []))
                downstream = fit_response_slope(
                    actual_amplitudes if eligible else [],
                    (scalars["downstream_signed_per_cell"][
                        seed_index, site_index, :, window_index]
                     if eligible else []))
                if source["slope"] is not None:
                    source_slopes[seed_index, site_index, window_index] = source["slope"]
                if downstream["slope"] is not None:
                    downstream_slopes[seed_index, site_index, window_index] = downstream["slope"]
                slope_records.append(dict(
                    seed=seed, site_id=site["id"], window_index=window_index,
                    source=source, downstream=downstream,
                    all_amplitudes_linear_eligible=eligible,
                ))

    paired = scalars["downstream_signed_per_cell"]
    if site_seed_window_eligible.any():
        eligible_paired = np.where(
            site_seed_window_eligible[:, :, None, :], paired, np.nan)
        paired_median = np.nanmedian(eligible_paired, axis=0)
        paired_mad = np.nanmedian(
            np.abs(eligible_paired - paired_median[None, ...]), axis=0)
        downstream_snr = np.abs(paired_median) / (1.4826 * paired_mad + 1e-6)
    else:
        downstream_snr = np.full((n_site, n_amp, n_window), np.nan)
    _atomic_npz(
        output_npz,
        seeds=np.asarray(seeds, np.int64),
        site_ids=np.asarray([site["id"] for site in sites], dtype="U32"),
        site_roles=np.asarray([site["role"] for site in sites], dtype="U32"),
        site_xy_mm=np.asarray([site["xy_mm"] for site in sites], np.float64),
        amplitude_multipliers=np.asarray(multipliers, np.float64),
        kick_boost_1_per_ms=np.asarray(actual_amplitudes, np.float64),
        windows_after_pulse_end_ms=np.asarray(windows, np.float64),
        selected_window_index=np.asarray(
            -1 if selected_window_index is None else selected_window_index, np.int64),
        window_downstream_positive_median=np.asarray(window_medians, np.float64),
        response_interval_event=response_event,
        event_window_overlap=window_event_overlap,
        site_seed_linear_eligible=site_seed_linear_eligible,
        site_seed_window_linear_eligible=site_seed_window_eligible,
        runaway_early_stop_ms=runaway_ms,
        simulation_wall_seconds=simulation_wall,
        source_slopes=source_slopes,
        downstream_slopes=downstream_slopes,
        downstream_snr=downstream_snr,
        signed_maps_per_cell=signed_maps,
        positive_maps_per_cell=positive_maps,
        **{key: np.asarray(value, np.float64) for key, value in scalars.items()},
    )

    if n_eligible_seeds < 2:
        status = f"REV9_{args.arm.upper()}_KICK_CANARY_SPARSE_CROSS_NETWORK_SUPPORT"
    else:
        status = f"REV9_{args.arm.upper()}_KICK_CANARY_COMPLETE"
    payload = dict(
        status=status,
        scientific_role=(
            f"{args.arm} small-kick response measurement; exploratory response "
            "matching only, not equivalence evidence or patient validation"),
        arm=args.arm, alpha=float(args.alpha), seeds=seeds, sites=sites,
        amplitude_multipliers=multipliers.tolist(),
        kick_boost_1_per_ms=actual_amplitudes.tolist(),
        pulse=dict(onset_ms=pulse_onset, duration_ms=instrument["kick_duration_ms"],
                   end_ms=pulse_end, population="E", kernel="top_hat_disk",
                   radius_mm=instrument["radius_mm"]),
        windows_after_pulse_end_ms=windows.tolist(),
        window_selection=dict(
            amplitude_multiplier=selection_multiplier,
            downstream_positive_per_cell_median=window_medians.tolist(),
            selected_index=selected_window_index,
            selected_window=(None if selected_window_index is None
                             else windows[selected_window_index].tolist()),
            selection_rule="predefined_first_generation_window",
            eligible_site_seed_by_window=(
                site_seed_window_eligible.sum(axis=(0, 1)).astype(int).tolist()),
            n_eligible_seeds=n_eligible_seeds,
            eligible_seeds=np.asarray(seeds)[eligible_seed_mask].tolist(),
            cross_network_support=bool(n_eligible_seeds >= 2),
            support_role=(
                "diagnostic only; sparse support does not fail execution, but the "
                "selected window remains a canary candidate rather than a frozen "
                "cross-network instrument"),
            tie_break="earliest"),
        event_diagnostics=dict(
            n_kick_pairs=int(response_event.size),
            n_response_interval_event=int(response_event.sum()),
            n_event_overlap_by_window=(
                window_event_overlap.sum(axis=(0, 1, 2)).astype(int).tolist()),
            n_runaway=int(np.isfinite(runaway_ms).sum()),
            n_eligible_site_seed=int(site_seed_linear_eligible.sum()),
            n_total_site_seed=int(site_seed_linear_eligible.size),
            n_eligible_seeds=n_eligible_seeds,
            sham=sham_records),
        slopes=slope_records,
        runs=records,
        networks=network_records,
        snr=dict(
            formula=(
                "abs(median paired downstream signed response across canary seeds) / "
                "(1.4826*MAD + 1e-6 spikes_per_cell)"),
            role="diagnostic only; no acceptance threshold"),
        arrays=dict(path=str(output_npz), sha256=_sha256(output_npz)),
        wall_seconds=float(time.time() - started),
        inputs=execution_inputs,
        provenance=dict(
            **execution_provenance,
            network_seed=seeds, ou_seed=seeds, poisson_seed=seeds,
            readout_seed=None),
    )
    atomic_write_json(payload, str(output_json))
    print(json.dumps(dict(
        status=payload["status"], selected_window=payload["window_selection"],
        event_diagnostics=payload["event_diagnostics"],
        wall_seconds=payload["wall_seconds"], arrays_sha256=payload["arrays"]["sha256"]
    ), indent=2), flush=True)


if __name__ == "__main__":
    main()
