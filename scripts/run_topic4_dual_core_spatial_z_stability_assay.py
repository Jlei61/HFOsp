#!/usr/bin/env python3
"""Delay-aware and OU residence assay at the Fig.5C operating section."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "snn_engine"))

from scripts.run_topic4_dual_core_spatial_z_bifurcation import (  # noqa: E402
    atomic_json,
    atomic_npz,
    leading_dynamic_eigenvalues,
    load_z_map,
    sha256,
)
from src.topic4_dual_core_spatial_z import (  # noqa: E402
    path_state,
    regional_rates_hz,
    solve_spatial_z_fixed_point,
)
from src.topic4_dual_core_spatial_z_delay import (  # noqa: E402
    build_coarse_delay_operators,
    coarse_spatial_ou_trace,
    delayed_growth_rate,
    simulate_delayed_ou_trajectory,
)
from src.topic4_patient_zm_meanfield import load_patient_coarse_model  # noqa: E402
from src.topic4_legacy_dual_core_transition import build_substrate, load_round_config  # noqa: E402


def _substrate(config_path: Path, artifact_root: Path, topology_seed: int):
    config = json.loads(config_path.read_text())
    transition_path = (ROOT / config["inputs"]["transition_config"]["path"]).resolve()
    transition = load_round_config(transition_path)
    manifest_path = (
        artifact_root
        / "results/topic4_sef_hfo/data_driven_dual_core_zm_transition/"
          "coarse_candidate_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    candidates = [row for row in manifest["candidates"]
                  if row["candidate_id"] == "rev21_zm_off"]
    if len(candidates) != 1:
        raise RuntimeError("rev21_zm_off must occur exactly once")
    candidate = candidates[0]
    mechanism = candidate["mechanisms"]
    mapping = candidate["node_mapping"]
    substrate = build_substrate(
        transition, config["reference"]["base_substrate_candidate_id"],
        int(topology_seed), cache_dir=str(artifact_root / config["network_cache"]),
        ee_dose=float(mechanism["g_EE"]),
        etoi_dose=float(mechanism["g_EtoI"]),
        node_candidate_override=candidate["node_field"],
        node_depth_shrinkage=float(mapping["signed_depth_shrinkage"]),
        node_gain=float(mapping["node_gain"]),
        ee_ellipse_angle_deg=float(mechanism["ellipse_angle_deg"]),
        ee_ellipse_aspect_ratio=float(mechanism["ellipse_aspect_ratio"]),
        artifact_root=artifact_root)
    return substrate, transition_path, transition["spatial_ou"], manifest_path


def _distinct_roots(model, z_map, parameter, seeds):
    roots = []
    for seed in seeds:
        solution = solve_spatial_z_fixed_point(
            model, z_map, parameter=parameter,
            initial_rates=np.asarray(seed, float), maxfev=30000)
        if not solution.converged or not solution.physical:
            continue
        if any(np.sqrt(np.mean((solution.rates - root.rates) ** 2)) < 1e-7
               for root in roots):
            continue
        roots.append(solution)
    return sorted(roots, key=lambda root: root.mean_rate_e_hz)


def main() -> None:
    base = ("/data/hfosp_topic4_fig45_artifacts/fig5/"
            "data_driven_dual_core_spatial_z/bifurcation/"
            "dualcore_spatial_z_bifurcation.json")
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default=base)
    parser.add_argument(
        "--config", default="config/topic4_rev21_dual_core_zm_transition.json")
    parser.add_argument("--artifact-root", default="/home/honglab/leijiaxin/HFOsp")
    parser.add_argument("--topology-seed", type=int, default=2542)
    parser.add_argument("--growth-steps", type=int, default=3000)
    parser.add_argument("--growth-burn-in", type=int, default=1500)
    parser.add_argument("--ou-duration-ms", type=float, default=300.0)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()
    started = time.time()

    result_path = Path(args.result).resolve()
    payload = json.loads(result_path.read_text())
    arrays_path = result_path.with_suffix(".npz")
    archive = np.load(arrays_path, allow_pickle=False)
    model = load_patient_coarse_model(payload["substrate"]["model"]["path"])
    z_map = load_z_map(Path(payload["substrate"]["z_map"]["path"]))
    parameter = float(payload["empirical_ou_on_projection"]["s_from_core_median"])
    z_a, z_b, z_surround, field = path_state(z_map, parameter)
    second = z_map.z_second_moment_field(
        z_a=z_a, z_b=z_b, z_surround=z_surround)

    low_seed = np.r_[np.full(model.n_cells, 1e-4),
                     np.full(model.n_cells, 5e-4)]
    outer_index = int(np.argmin(np.abs(archive["outer_high__s"] - parameter)))
    high_seed = np.asarray(archive["outer_high__rates"][outer_index], float)
    low = solve_spatial_z_fixed_point(
        model, z_map, parameter=parameter, initial_rates=low_seed, maxfev=30000)
    high = solve_spatial_z_fixed_point(
        model, z_map, parameter=parameter, initial_rates=high_seed, maxfev=30000)
    seeds = [low.rates, high.rates]
    branch_s = np.asarray(archive["entry_saddle__s"], float)
    branch_rates = np.asarray(archive["entry_saddle__rates"], float)
    crossings = np.flatnonzero(
        (branch_s[:-1] - parameter) * (branch_s[1:] - parameter) <= 0.0)
    seeds.extend(branch_rates[index] for index in crossings)
    for alpha in np.linspace(0.0, 1.0, 17):
        seeds.append((1.0 - alpha) * low.rates + alpha * high.rates)
    fa = np.asarray(z_map.core_a_fraction_e, float)
    fb = np.asarray(z_map.core_b_fraction_e, float)
    fs = np.asarray(z_map.surround_fraction_e, float)
    for amplitude in (0.05, 0.15, 0.30, 0.42):
        for mask in (fa, fb, np.maximum(fa, fb), fs, np.ones_like(fa)):
            rate_e = low.rate_e + (amplitude - low.rate_e) * mask
            rate_i = low.rate_i + (min(amplitude + 0.03, 0.8) - low.rate_i) * mask
            seeds.append(np.r_[rate_e, rate_i])
    roots = _distinct_roots(model, z_map, parameter, seeds)
    if len(roots) < 3:
        raise RuntimeError("operating section did not recover at least three roots")

    config_path = (ROOT / args.config).resolve()
    artifact_root = Path(args.artifact_root).resolve()
    substrate, transition_path, ou_config, manifest_path = _substrate(
        config_path, artifact_root, int(args.topology_seed))
    operators = build_coarse_delay_operators(substrate, model)
    root_records = []
    for index, root in enumerate(roots):
        zero_delay = leading_dynamic_eigenvalues(model, z_map, root, k=4)
        delayed = delayed_growth_rate(
            model, operators, root.rates, z_field=field,
            z_second_moment=second, n_steps=int(args.growth_steps),
            burn_in_steps=int(args.growth_burn_in),
            seeds=(1701, 1702, 1703))
        root_records.append({
            "root_index": int(index),
            "mean_e_rate_hz": float(root.mean_rate_e_hz),
            "regional_e_rate_hz": regional_rates_hz(model, z_map, root.rate_e),
            "zero_delay_leading_eigenvalues_per_ms": zero_delay,
            "delay_aware_native_dt": delayed,
        })

    n_steps = int(round(float(args.ou_duration_ms) / operators.dt_ms))
    ou_seeds = (6101, 6102, 6103)
    ou_arrays = {}
    ou_records = []
    reference = np.asarray([root.rates for root in roots], float)
    for seed in ou_seeds:
        ou = coarse_spatial_ou_trace(
            substrate, model, ou_config, seed=int(seed), n_steps=n_steps)
        ou_arrays[f"ou_seed_{seed}__coarse_rate_per_ms"] = ou
        for index, root in enumerate(roots):
            trajectory = simulate_delayed_ou_trajectory(
                model, operators, root.rates, z_field=field,
                z_second_moment=second, ou_rate_e=ou,
                tail_steps=min(500, n_steps))
            distances = 1000.0 * np.sqrt(np.mean(
                (reference - trajectory["tail_mean_rates"][None, :]) ** 2,
                axis=1))
            nearest = int(np.argmin(distances))
            ou_arrays[f"root_{index}_seed_{seed}__mean_e_rate_hz"] = (
                trajectory["mean_e_rate_hz"])
            ou_arrays[f"root_{index}_seed_{seed}__mean_i_rate_hz"] = (
                trajectory["mean_i_rate_hz"])
            ou_arrays[f"root_{index}_seed_{seed}__rms_deviation_hz"] = (
                trajectory["rms_rate_deviation_from_initial_hz"])
            tail_trace = np.asarray(
                trajectory["mean_e_rate_hz"][-min(2000, n_steps):], float)
            centered = tail_trace - np.mean(tail_trace)
            frequencies = np.fft.rfftfreq(
                tail_trace.size, d=operators.dt_ms / 1000.0)
            power = np.abs(np.fft.rfft(centered)) ** 2
            eligible = (frequencies >= 5.0) & (frequencies <= 150.0)
            peak_frequency = float(
                frequencies[np.flatnonzero(eligible)[np.argmax(power[eligible])]])
            q05, q95 = np.quantile(tail_trace, [0.05, 0.95])
            tail_mean = float(np.mean(tail_trace))
            ou_records.append({
                "initial_root_index": int(index),
                "ou_seed": int(seed),
                "nearest_tail_root_index": nearest,
                "retained_initial_root": bool(nearest == index),
                "tail_rms_distance_to_each_root_hz": distances.tolist(),
                "tail_mean_e_rate_hz": tail_mean,
                "tail_q05_q95_e_rate_hz": [float(q05), float(q95)],
                "tail_modulation_half_range_over_mean": float(
                    (q95 - q05) / max(2.0 * tail_mean, 1e-12)),
                "tail_peak_frequency_5_150_hz": peak_frequency,
            })

    out_prefix = Path(args.out_prefix).resolve()
    atomic_npz(out_prefix.with_suffix(".npz"), **ou_arrays)
    result = {
        "status": "DUAL_CORE_SPATIAL_Z_DELAY_OU_ASSAY_COMPLETE",
        "operating_parameter": {
            "s": parameter, "z_a": z_a, "z_b": z_b,
            "z_surround": z_surround,
        },
        "root_catalog": root_records,
        "root_catalog_n": int(len(roots)),
        "delay_contract": {
            "dt_ms": float(operators.dt_ms),
            "maximum_delay_steps": int(operators.max_delay_steps),
            "maximum_delay_ms": float(
                operators.dt_ms * operators.max_delay_steps),
            "all_realized_delay_bins_retained": True,
            "growth_assay": (
                "native-dt delayed tangent power iteration with operating "
                "mean gain and instantaneous diffusion-variance gain both "
                "included in the linearization"),
        },
        "ou_contract": {
            **ou_config,
            "coarse_projection": (
                "original neuron-level SpatialOUDrive averaged onto the same grid"),
            "duration_ms": float(args.ou_duration_ms),
            "seeds": list(ou_seeds),
            "variance": (
                "recurrent and clipped external variance recomputed each step"),
        },
        "ou_residence": ou_records,
        "source": {
            "bifurcation_json": {"path": str(result_path),
                                 "sha256": sha256(result_path)},
            "bifurcation_npz": {"path": str(arrays_path),
                                "sha256": sha256(arrays_path)},
            "config": {"path": str(config_path), "sha256": sha256(config_path)},
            "transition_config": {"path": str(transition_path),
                                  "sha256": sha256(transition_path)},
            "candidate_manifest": {"path": str(manifest_path),
                                   "sha256": sha256(manifest_path)},
        },
        "claim_boundary": (
            "delay-aware and OU-aware assay of the matched 2-mm coarse model; "
            "not a full 40,000-neuron stochastic bifurcation theorem"),
        "wall_seconds": float(time.time() - started),
    }
    atomic_json(result, out_prefix.with_suffix(".json"))
    print(json.dumps({
        "status": result["status"], "root_catalog_n": len(roots),
        "delay_classifications": [
            row["delay_aware_native_dt"]["classification"]
            for row in root_records],
        "ou_retained": int(sum(row["retained_initial_root"]
                               for row in ou_records)),
        "ou_total": len(ou_records),
        "output": str(out_prefix.with_suffix(".json")),
        "wall_seconds": result["wall_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
