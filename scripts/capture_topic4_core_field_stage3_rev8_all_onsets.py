"""Capture all final rev8.1 events' earliest-activation locations.

Every final unseen-network seed is rerun. Usable event curves must reproduce the
hashed confirmation pool before an event contributes to the event-equal onset
density or nearest-Gaussian-component table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join("src", "snn_engine"))
from scripts.run_topic4_core_field_stage3_fit import STAGE2, _load_cmrun  # noqa: E402
from scripts.run_topic4_core_field_stage3_joint_confirm import _atomic_npz  # noqa: E402
from scripts.run_topic4_core_field_stage3_profile_round1 import axial_map  # noqa: E402
from src.topic4_core_field import (  # noqa: E402
    build_vth,
    core_thresholds,
    sample_core_quantiles,
    signed_depth,
)
from src.topic4_core_field_profile import normalized_rank_curve  # noqa: E402
from src.topic4_core_field_runner import (  # noqa: E402
    _placement,
    atomic_write_json,
    get_network,
    provenance,
)
from src.topic4_core_field_stage3 import params_to_h, unpack  # noqa: E402


ROOT = "results/topic4_sef_hfo/data_driven_core_field_stage3"
CONFIRM = f"{ROOT}/joint_confirmation_rev8_1/final_confirmation.json"
PROFILES = f"{ROOT}/joint_confirmation_rev8_1/final_event_profiles.npz"
OUT_JSON = f"{ROOT}/joint_confirmation_rev8_1/all_event_onset_diagnostics.json"
OUT_NPZ = f"{ROOT}/joint_confirmation_rev8_1/all_event_onset_diagnostics.npz"
N_DENSITY_BINS = 40
EARLY_QUANTILE = 0.01


def _sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def event_equal_density(points_by_event, sheet_length, n_bins=N_DENSITY_BINS):
    """Average per-event spatial histograms so large events cannot dominate."""
    edges = np.linspace(0.0, float(sheet_length), int(n_bins) + 1)
    density = np.zeros((int(n_bins), int(n_bins)), dtype=float)
    valid = 0
    for points in points_by_event:
        points = np.asarray(points, float)
        if points.ndim != 2 or points.shape[1:] != (2,) or not len(points):
            continue
        hist, _, _ = np.histogram2d(points[:, 1], points[:, 0], bins=(edges, edges))
        if hist.sum() > 0:
            density += hist / hist.sum()
            valid += 1
    if valid:
        density /= valid
    return density, edges, valid


def _capture_seed(job):
    seed, theta, component_count, cfg, cache, expected = job
    try:
        cmrun = _load_cmrun()
        engine = cfg["engine"]
        cmrun.KDIR = int(engine["k_dir"])
        cmrun.PART_MIN = 2 * int(engine["k_dir"]) + 1
        reg = _placement(cfg)
        params_cls = __import__("params").Params
        simulate_kick = __import__("kick_probe").simulate_kick
        detect_events = __import__(
            "src.sef_hfo_events", fromlist=["detect_events"]).detect_events
        snn_event_envelope = __import__(
            "src.sef_hfo_snn_adapter", fromlist=["snn_event_envelope"]
        ).snn_event_envelope

        p = params_cls(
            g=engine["g"], L=engine["L"], density=engine["density"],
            T=cfg["duration_ms"], dt=engine["dt"],
            nu_ext_ratio=cmrun.DRIVE, seed=int(seed))
        net, n_e, n_i, cache_hit = get_network(
            p, reg["theta_deg"], engine["AR"], cache)
        pos_e = net["pos"][:n_e]
        h = params_to_h(
            np.asarray(theta, float), pos_e, int(component_count),
            float(engine["L"]), float(cfg["N_core_manual"]))
        depth = signed_depth(core_thresholds(
            sample_core_quantiles(n_e, cfg["quantile_seed"]),
            engine["core_mean"], engine["core_std"]), engine["v_base"])
        vth = build_vth(
            h, depth, n_total=n_e + n_i, n_E=n_e, v_base=engine["v_base"])

        montage = reg["montage_sheet"]
        valid = cmrun.valid_mask(montage, pos_e, engine["L"], p.Rr)
        net["rng"] = np.random.default_rng(int(seed))
        result = simulate_kick(
            p, net, KICK_BOOST=0.0, kick_center=list(reg["center"]),
            r_kick=engine["core_r"], t_kick=1e9,
            V_th_per_neuron=vth, lfp_recorder=None)
        spikes = result["E_spk_bool"]
        active_fraction, bin_width = cmrun.active_fraction(
            spikes, engine["dt"], cmrun.BIN_MS)
        baseline_start = int(cmrun.BASELINE_MS[0] / bin_width)
        baseline_stop = int(cmrun.BASELINE_MS[1] / bin_width)
        floor = float(np.percentile(
            active_fraction[baseline_start:baseline_stop], 95))
        bar = floor + cmrun.CAL_FRAC * (float(active_fraction.max()) - floor)
        detected = detect_events(active_fraction, bin_width, event_on_frac=bar)
        envelope, envelope_dt, _ = snn_event_envelope(
            spikes, pos_e, montage, engine["dt"])
        axial = axial_map()
        grid = np.asarray(expected["grid"], float)
        expected_events = expected["events"]

        records = []
        reproduced = set()
        for local_index, event in enumerate(detected):
            readout = cmrun.read_event(
                envelope, envelope_dt, montage, valid,
                (event["t_on"], event["t_off"]), reg["axis_unit_vec"],
                k_dir=int(engine["k_dir"]),
                part_min=2 * int(engine["k_dir"]) + 1)
            curve = normalized_rank_curve(readout["ranks"], axial, grid=grid)
            if curve is None:
                continue
            key = str(int(local_index))
            if key not in expected_events:
                raise RuntimeError(
                    f"seed {seed} produced unconfirmed usable event {local_index}")
            target = expected_events[key]
            if not np.allclose(curve, target["curve"], atol=2e-7, rtol=0.0):
                raise RuntimeError(f"curve drift for seed {seed} event {local_index}")
            reproduced.add(key)
            onset = np.asarray(cmrun.per_neuron_onset(
                spikes, event["t_on"], event["t_off"], engine["dt"]), float)
            finite = np.isfinite(onset)
            if finite.sum() < 2:
                raise RuntimeError(f"no finite neuron onsets for seed {seed} event {local_index}")
            relative = onset[finite] - onset[finite].min()
            threshold = float(np.quantile(relative, EARLY_QUANTILE))
            early_points = pos_e[finite][relative <= threshold]
            if not len(early_points):
                raise RuntimeError(f"empty earliest set for seed {seed} event {local_index}")
            records.append(dict(
                seed=int(seed), local_event_index=int(local_index),
                mode=int(target["mode"]), t_on=float(event["t_on"]),
                t_off=float(event["t_off"]), n_part=int(readout["n_part"]),
                n_active_neurons=int(finite.sum()), n_early_neurons=int(len(early_points)),
                early_threshold_ms=threshold,
                source_centroid=np.asarray(early_points.mean(axis=0), float),
                early_points=np.asarray(early_points, np.float32),
            ))
        if reproduced != set(expected_events):
            missing = sorted(set(expected_events) - reproduced, key=int)
            raise RuntimeError(f"seed {seed} did not reproduce confirmed events {missing}")
        return dict(
            seed=int(seed), cache_hit=bool(cache_hit), n_detected=int(len(detected)),
            records=records, pos_e=np.asarray(pos_e, np.float32))
    except Exception as exc:  # noqa: BLE001
        return dict(seed=int(seed), error=repr(exc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation", default=CONFIRM)
    parser.add_argument("--profiles", default=PROFILES)
    parser.add_argument("--out-json", default=OUT_JSON)
    parser.add_argument("--out-npz", default=OUT_NPZ)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    confirmation = json.load(open(args.confirmation))
    if confirmation["event_profiles"]["sha256"] != _sha256(args.profiles):
        raise RuntimeError("confirmation/event-profile hash mismatch")
    arrays = np.load(args.profiles)
    candidate = confirmation["candidates"][0]
    seed_ids = np.asarray(arrays["model_seed_ids"], int)
    local_indices = np.asarray(arrays["model_local_event_indices"], int)
    labels = np.asarray(arrays["model_labels"], int)
    curves = np.asarray(arrays["model_curves"], float)
    seeds = [int(value) for value in confirmation["confirm_network_seeds"]]
    expected_by_seed = {}
    for seed in seeds:
        selected = np.flatnonzero(seed_ids == seed)
        expected_by_seed[seed] = dict(
            grid=np.asarray(arrays["grid"], float),
            events={
                str(int(local_indices[index])): dict(
                    mode=int(labels[index]), curve=np.asarray(curves[index], float))
                for index in selected
            },
        )

    cfg = json.load(open(f"{STAGE2}/config/stage_config.json"))
    cache = os.path.join(STAGE2, "network_cache")
    jobs = [
        (seed, candidate["theta"], int(candidate["K"]), cfg, cache,
         expected_by_seed[seed])
        for seed in seeds
    ]
    with Pool(int(args.workers), maxtasksperchild=1) as pool:
        rows = pool.map(_capture_seed, jobs)
    errors = [dict(seed=row["seed"], error=row["error"])
              for row in rows if "error" in row]
    if errors:
        atomic_write_json(dict(
            status="FAIL_CLOSED_ONSET_CAPTURE_ERRORS", errors=errors,
            input_confirmation=dict(
                path=args.confirmation, sha256=_sha256(args.confirmation)),
            input_profiles=dict(path=args.profiles, sha256=_sha256(args.profiles)),
            provenance=provenance(),
        ), args.out_json)
        raise RuntimeError(f"all-event onset capture failed closed: {errors}")

    records = [record for row in rows for record in row["records"]]
    if len(records) != len(curves):
        raise RuntimeError(f"captured {len(records)} events, expected {len(curves)}")
    records.sort(key=lambda record: (record["seed"], record["local_event_index"]))
    sheet_length = float(cfg["engine"]["L"])
    densities, edges, density_counts = [], None, []
    for mode in (0, 1):
        density, edges, count = event_equal_density(
            [record["early_points"] for record in records if record["mode"] == mode],
            sheet_length)
        densities.append(density)
        density_counts.append(count)

    components = unpack(
        np.asarray(candidate["theta"], float), int(candidate["K"]), sheet_length)
    component_centers = np.asarray([row["center"] for row in components], float)
    centroids = np.asarray([record["source_centroid"] for record in records], float)
    assignments = np.argmin(
        np.linalg.norm(centroids[:, None, :] - component_centers[None, :, :], axis=2),
        axis=1)
    component_counts = np.zeros((2, len(components)), dtype=int)
    for record, assignment in zip(records, assignments):
        record["nearest_component"] = int(assignment)
        component_counts[record["mode"], assignment] += 1

    _atomic_npz(
        args.out_npz,
        density=np.asarray(densities, np.float32),
        density_edges=np.asarray(edges, np.float32),
        source_centroids=np.asarray(centroids, np.float32),
        event_modes=np.asarray([record["mode"] for record in records], np.int8),
        event_seed_ids=np.asarray([record["seed"] for record in records], np.int64),
        event_local_indices=np.asarray(
            [record["local_event_index"] for record in records], np.int64),
        nearest_components=np.asarray(assignments, np.int8),
        component_centers=np.asarray(component_centers, np.float32),
        component_counts=np.asarray(component_counts, np.int64),
    )
    serialized_records = [
        {key: (value.tolist() if isinstance(value, np.ndarray) else value)
         for key, value in record.items() if key != "early_points"}
        for record in records
    ]
    probabilities = component_counts / component_counts.sum(axis=1, keepdims=True)
    payload = dict(
        status="REV8_1_ALL_EVENT_ONSET_CAPTURED",
        scientific_role=(
            "post-freeze causal diagnostic of where each confirmed model mode starts; "
            "does not change candidate selection or the blind-confirmation verdict"),
        earliest_activation_contract=dict(
            definition="neurons at or below each event's 1st percentile relative onset",
            quantile=EARLY_QUANTILE,
            density="each event histogram normalized to unit mass before mode averaging",
            component_assignment="nearest Gaussian center to the event source centroid",
        ),
        n_events=int(len(records)), mode_counts=np.bincount(
            [record["mode"] for record in records], minlength=2).astype(int).tolist(),
        density_event_counts=density_counts,
        component_centers=component_centers.tolist(),
        component_counts_by_mode=component_counts.tolist(),
        probability_component_given_mode=probabilities.tolist(),
        events=serialized_records,
        exact_curve_reproduction=True,
        network_cache=dict(
            hits=int(sum(row["cache_hit"] for row in rows)),
            builds=int(sum(not row["cache_hit"] for row in rows))),
        input_confirmation=dict(path=args.confirmation, sha256=_sha256(args.confirmation)),
        input_profiles=dict(path=args.profiles, sha256=_sha256(args.profiles)),
        arrays=dict(path=args.out_npz, sha256=_sha256(args.out_npz)),
        provenance=provenance(),
    )
    atomic_write_json(payload, args.out_json)
    print(json.dumps({
        "status": payload["status"], "n_events": payload["n_events"],
        "mode_counts": payload["mode_counts"],
        "probability_component_given_mode": probabilities.tolist(),
        "arrays_sha256": payload["arrays"]["sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
