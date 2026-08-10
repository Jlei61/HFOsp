"""Find a globally quiet small-kick onset from three frozen Node sham runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join("src", "snn_engine"))
from scripts.run_topic4_core_field_stage3_fit import _load_cmrun  # noqa: E402
from src.topic4_core_field_rev9 import reconstruct_frozen_node  # noqa: E402
from src.topic4_core_field_runner import (  # noqa: E402
    CONNECTIVITY_FIELDS,
    _placement,
    atomic_write_json,
    canonical_checksum,
    provenance,
)


DEFAULT_CONFIG = "config/topic4_rev9_exploratory.json"
DEFAULT_CANARY = (
    "results/topic4_sef_hfo/data_driven_core_field_rev9/"
    "node_edge_calibration/node_kick_canary.json")


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


def _candidate(config):
    confirmation = json.loads(Path(config["inputs"]["confirmation"]).read_text())
    rows = [row for row in confirmation["candidates"]
            if row["candidate_id"] == config["inputs"]["candidate_id"]]
    if len(rows) != 1 or rows[0]["theta_sha256"] != config["inputs"]["theta_sha256"]:
        raise RuntimeError("configured frozen candidate or theta hash does not reproduce")
    return rows[0]


def _cache_path(stage, seed, commit, numpy_version, cache_dir):
    params_cls = __import__("params").Params
    engine = stage["engine"]
    params = params_cls(
        g=engine["g"], L=engine["L"], density=engine["density"],
        T=400.0, dt=engine["dt"], nu_ext_ratio=0.6, seed=int(seed))
    reg = _placement(stage)
    cache_config = {field: getattr(params, field) for field in CONNECTIVITY_FIELDS}
    cache_config.update(
        theta_EE_deg=float(reg["theta_deg"]), AR=float(engine["AR"]),
        numpy_version=str(numpy_version), rng_bit_generator="PCG64",
        git_commit=str(commit))
    key = canonical_checksum(cache_config, drop=())
    path = Path(cache_dir) / f"{key}.pkl"
    if not path.exists():
        raise RuntimeError(f"canary network cache is missing: {path}")
    return path, key, params, reg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--canary", default=DEFAULT_CANARY)
    parser.add_argument("--out-json")
    parser.add_argument("--out-npz")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    canary = json.loads(Path(args.canary).read_text())
    stage = json.loads(Path(config["inputs"]["stage_config"]).read_text())
    candidate = _candidate(config)
    instrument = config["small_kick_instrument"]
    output_root = Path(config["output_root"]) / "node_edge_calibration"
    output_json = Path(args.out_json or output_root / "sham_onset_scan.json")
    output_npz = Path(args.out_npz or output_root / "sham_onset_scan.npz")
    seeds = [int(value) for value in instrument["canary_seeds"]]
    onsets = np.asarray(instrument["onset_scan_candidates_ms"], float)
    interval_duration = float(instrument["kick_duration_ms"]) + max(
        float(row[1]) for row in instrument["candidate_windows_after_pulse_end_ms"])
    engine = stage["engine"]
    cmrun = _load_cmrun()
    cmrun.DT = float(engine["dt"])
    simulate_kick = __import__("kick_probe").simulate_kick
    detect_events = __import__(
        "src.sef_hfo_events", fromlist=["detect_events"]).detect_events
    cache_dir = config["inputs"]["upstream_network_cache"]
    # The canary used rev9's output cache, not the Stage-2 cache.
    cache_dir = str(Path(config["output_root"]) / "network_cache")
    source_commit = canary["provenance"]["git_commit"]
    source_numpy = canary["provenance"]["numpy_version"]
    active_rows, event_rows, network_rows = [], [], []
    started = time.time()

    for seed in seeds:
        path, key, params, reg = _cache_path(
            stage, seed, source_commit, source_numpy, cache_dir)
        with open(path, "rb") as handle:
            cached = pickle.load(handle)
        net, n_e, n_i = cached["net"], int(cached["NE"]), int(cached["NI"])
        node = reconstruct_frozen_node(
            candidate["theta"], net["pos"][:n_e], n_total=n_e + n_i,
            target_count=stage["N_core_manual"],
            quantile_seed=stage["quantile_seed"],
            core_mean=engine["core_mean"], core_std=engine["core_std"],
            v_base=engine["v_base"], K=candidate["K"], L=engine["L"])
        net["rng"] = np.random.default_rng(seed)
        result = simulate_kick(
            params, net, KICK_BOOST=0.0,
            kick_center=instrument["origins"][0]["xy_mm"],
            r_kick=instrument["radius_mm"], t_kick=1e9,
            V_th_per_neuron=node["vtheta"], early_stop_runaway=True)
        active, bin_width = cmrun.active_fraction(
            result["E_spk_bool"], engine["dt"], cmrun.BIN_MS)
        baseline_start = int(cmrun.BASELINE_MS[0] / bin_width)
        baseline_stop = int(cmrun.BASELINE_MS[1] / bin_width)
        floor = float(np.percentile(active[baseline_start:baseline_stop], 95))
        peak = float(active.max(initial=0.0))
        threshold = floor + cmrun.CAL_FRAC * (peak - floor)
        events = detect_events(active, bin_width, event_on_frac=threshold)
        event_rows.append([dict(t_on=float(row["t_on"]), t_off=float(row["t_off"]))
                           for row in events])
        active_rows.append(np.asarray(active, np.float32))
        network_rows.append(dict(
            seed=seed, cache_path=str(path), cache_key=key,
            cache_sha256=_sha256(path), n_E=n_e, n_I=n_i,
            active_fraction_floor=floor, active_fraction_peak=peak,
            detector_threshold=float(threshold), n_events=int(len(events)),
            runaway_early_stop_ms=result["runaway_early_stop_ms"],
            simulation_wall_seconds=float(result["wall_s"])))
        print(json.dumps(dict(
            progress="sham_complete", seed=seed, n_events=len(events),
            event_times=event_rows[-1], wall_seconds=result["wall_s"])), flush=True)
        del result, net, node, cached

    active_rows = np.asarray(active_rows, np.float32)
    event_free = np.zeros((len(seeds), len(onsets)), bool)
    interval_mean = np.full(event_free.shape, np.nan)
    interval_peak = np.full(event_free.shape, np.nan)
    for seed_index, events in enumerate(event_rows):
        for onset_index, onset in enumerate(onsets):
            stop = float(onset + interval_duration)
            overlapping = [row for row in events
                           if row["t_on"] < stop and row["t_off"] > onset]
            event_free[seed_index, onset_index] = not overlapping
            start_bin = int(round(onset / cmrun.BIN_MS))
            stop_bin = int(round(stop / cmrun.BIN_MS))
            values = active_rows[seed_index, start_bin:stop_bin]
            interval_mean[seed_index, onset_index] = float(values.mean())
            interval_peak[seed_index, onset_index] = float(values.max(initial=0.0))
    free_counts = event_free.sum(axis=0)
    median_peak = np.median(interval_peak, axis=0)
    all_seed_quiet = np.flatnonzero(free_counts == len(seeds))
    if len(all_seed_quiet):
        selected_index = int(sorted(
            all_seed_quiet,
            key=lambda index: (median_peak[index], onsets[index]))[0])
        selected_onset = float(onsets[selected_index])
        status = "REV9_SHAM_ONSET_SCAN_QUIET_ONSET_FOUND"
    else:
        selected_index = None
        selected_onset = None
        status = "REV9_SHAM_ONSET_SCAN_NO_GLOBAL_QUIET_ONSET"

    _atomic_npz(
        output_npz, seeds=np.asarray(seeds, np.int64),
        candidate_onsets_ms=onsets, active_fraction=active_rows,
        event_free=event_free, interval_active_fraction_mean=interval_mean,
        interval_active_fraction_peak=interval_peak,
        selected_onset_index=np.asarray(
            -1 if selected_index is None else selected_index, np.int64))
    payload = dict(
        status=status,
        scientific_role=(
            "instrument-timing development from Node sham trajectories only; "
            "does not select edge alpha or read patient data"),
        candidate_onsets_ms=onsets.tolist(),
        response_interval_duration_ms=interval_duration,
        event_free_matrix=event_free.tolist(),
        event_free_seed_counts=free_counts.astype(int).tolist(),
        interval_active_fraction_median_peak=median_peak.tolist(),
        selected_index=selected_index, selected_onset_ms=selected_onset,
        selection_rule=(
            "require all three seeds event-free; among eligible candidates choose "
            "lowest median interval active-fraction peak, tie earliest"),
        networks=network_rows, detector_events=event_rows,
        arrays=dict(path=str(output_npz), sha256=_sha256(output_npz)),
        wall_seconds=float(time.time() - started),
        inputs=dict(
            config=dict(path=args.config, sha256=_sha256(args.config)),
            canary=dict(path=args.canary, sha256=_sha256(args.canary)),
            source_network_commit=source_commit),
        provenance=dict(
            **provenance(), git_status_porcelain=_git("status", "--porcelain"),
            producer_sha256=_sha256(__file__), python_executable=sys.executable,
            python_version=platform.python_version(),
            systemd_unit=os.environ.get("REV9_SYSTEMD_UNIT")),
    )
    atomic_write_json(payload, str(output_json))
    print(json.dumps(dict(
        status=status, event_free_seed_counts=payload["event_free_seed_counts"],
        selected_onset_ms=selected_onset, wall_seconds=payload["wall_seconds"],
        arrays_sha256=payload["arrays"]["sha256"]), indent=2))


if __name__ == "__main__":
    main()
