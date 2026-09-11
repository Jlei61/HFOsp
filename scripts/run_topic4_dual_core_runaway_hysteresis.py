#!/usr/bin/env python3
"""Continuous-state up/down sweep across the dual-core runaway boundary."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "src" / "snn_engine"):
    sys.path.insert(0, str(item))

from scripts.run_topic4_dual_core_oscillatory_phase_map import (  # noqa: E402
    _perturbed,
    _terminal_low_seed,
)
from scripts.run_topic4_dual_core_spatial_z_bifurcation import (  # noqa: E402
    atomic_json,
    atomic_npz,
    load_z_map,
    sha256,
)
from scripts.run_topic4_dual_core_spatial_z_stability_assay import _substrate  # noqa: E402
from src.topic4_dual_core_oscillation_phase import coarsen_delay_operators  # noqa: E402
from src.topic4_dual_core_spatial_z import path_state, regional_rates_hz  # noqa: E402
from src.topic4_dual_core_spatial_z_delay import (  # noqa: E402
    build_coarse_delay_operators,
    simulate_delayed_ou_trajectory,
)
from src.topic4_patient_zm_meanfield import load_patient_coarse_model  # noqa: E402


DEFAULT_ROOT = Path(
    "/data/hfosp_topic4_fig45_artifacts/fig5/"
    "data_driven_dual_core_spatial_z")


def _run_segment(model, z_map, operators, rates, state, *, s, eta_m, tau_m,
                 duration_ms):
    z_a, z_b, z_surround, field = path_state(z_map, float(s))
    second = z_map.z_second_moment_field(
        z_a=z_a, z_b=z_b, z_surround=z_surround)
    steps = int(round(float(duration_ms) / operators.dt_ms))
    kwargs = {}
    if state is not None:
        kwargs = {
            "initial_m": state["final_adaptation_state"],
            "initial_synapses": state["final_synapses"],
            "initial_history_e": state["final_history_e"],
            "initial_history_i": state["final_history_i"],
        }
    trajectory = simulate_delayed_ou_trajectory(
        model, operators, rates, z_field=field, z_second_moment=second,
        ou_rate_e=np.zeros((steps, model.n_cells), np.float32),
        tail_steps=max(1, steps // 2), eta_m=eta_m,
        tau_m_slow_ms=tau_m, **kwargs)
    regional = regional_rates_hz(
        model, z_map, trajectory["tail_mean_rates"][:model.n_cells])
    trace = np.asarray(trajectory["mean_e_rate_hz"], float)
    first, second_half = np.array_split(trace[-steps // 2:], 2)
    return trajectory, {
        "s": float(s),
        "population_tail_mean_hz": float(np.mean(trace[-steps // 2:])),
        "half_tail_drift_hz": float(
            np.mean(second_half) - np.mean(first)),
        "regional_tail_rate_hz": regional,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--regime-config", type=Path,
        default=ROOT / "config/topic4_dual_core_zm_regime_v1.json")
    parser.add_argument(
        "--boundary-config", type=Path,
        default=ROOT / "config/topic4_dual_core_runaway_boundary_v1.json")
    parser.add_argument(
        "--rev21-config", type=Path,
        default=ROOT / "config/topic4_rev21_dual_core_zm_transition.json")
    parser.add_argument("--coarsen-factor", type=int, default=5)
    parser.add_argument("--settle-ms", type=float, default=5000.0)
    parser.add_argument("--step-ms", type=float, default=1000.0)
    parser.add_argument("--s-min", type=float, default=0.30)
    parser.add_argument("--s-start", type=float, default=0.40)
    parser.add_argument("--s-max", type=float, default=0.60)
    parser.add_argument("--s-step", type=float, default=0.005)
    parser.add_argument(
        "--artifact-root", type=Path,
        default=Path("/home/honglab/leijiaxin/HFOsp"))
    parser.add_argument(
        "--out-prefix", type=Path,
        default=DEFAULT_ROOT / "runaway_boundary/continuous_state_hysteresis")
    args = parser.parse_args()

    coarse_root = args.coarse_root.resolve()
    regime_config = json.loads(
        args.regime_config.resolve().read_text(encoding="utf-8"))
    boundary_config = json.loads(
        args.boundary_config.resolve().read_text(encoding="utf-8"))
    gate = boundary_config["predeclared_tonic_runaway_gate"]
    eta_m = float(regime_config["dynamics"]["base_eta_m"])
    tau_m = float(regime_config["dynamics"]["tau_m_ms"])
    model_path = coarse_root / (
        "deterministic_meanfield/dualcore_topology_2542_ngrid10.npz")
    z_map_path = model_path.with_suffix(".zmap.npz")
    branch_path = coarse_root / "bifurcation/dualcore_spatial_z_bifurcation.npz"
    model = replace(
        load_patient_coarse_model(model_path),
        tau_gaba_ms=float(regime_config["dynamics"]["tau_d_GABA_ms"]))
    z_map = load_z_map(z_map_path)
    with np.load(branch_path, allow_pickle=False) as branch:
        branch_arrays = {name: np.asarray(branch[name]) for name in branch.files}
    substrate, transition_path, _ou, manifest_path = _substrate(
        args.rev21_config.resolve(), args.artifact_root.resolve(), 2542)
    native = build_coarse_delay_operators(substrate, model)
    operators = coarsen_delay_operators(native, factor=args.coarsen_factor)

    initial = _perturbed(
        _terminal_low_seed(branch_arrays), n_cells=model.n_cells,
        fraction=float(regime_config["dynamics"]["perturbation_fraction"]),
        floor=float(regime_config["dynamics"]["perturbation_floor_per_ms"]))
    state, _ = _run_segment(
        model, z_map, operators, initial, None, s=args.s_start,
        eta_m=eta_m, tau_m=tau_m, duration_ms=args.settle_ms)

    up_s = np.arange(args.s_start, args.s_max + args.s_step / 2, args.s_step)
    down_s = np.arange(args.s_max, args.s_min - args.s_step / 2, -args.s_step)
    records = []
    traces = {}
    for direction, values in (("up", up_s), ("down", down_s)):
        if direction == "down":
            # Continue from the terminal high-s state without any reset.
            pass
        for index, s in enumerate(values):
            state, record = _run_segment(
                model, z_map, operators, state["final_rates"], state,
                s=float(s), eta_m=eta_m, tau_m=tau_m,
                duration_ms=args.step_ms)
            regions = np.asarray(list(record["regional_tail_rate_hz"].values()))
            record["direction"] = direction
            record["runaway"] = bool(
                record["population_tail_mean_hz"] >= float(
                    gate["minimum_population_mean_rate_hz"])
                and np.min(regions) >= float(
                    gate["minimum_each_regional_rate_hz"])
                and abs(record["half_tail_drift_hz"]) <= float(
                    gate["maximum_absolute_half_tail_drift_hz"]))
            records.append(record)
            traces[f"{direction}_{index:03d}"] = np.asarray(
                state["mean_e_rate_hz"], np.float32)

    def transition(direction):
        subset = [record for record in records if record["direction"] == direction]
        changes = []
        for left, right in zip(subset[:-1], subset[1:]):
            if left["runaway"] != right["runaway"]:
                changes.append([left["s"], right["s"]])
        return changes

    out_prefix = args.out_prefix.resolve()
    atomic_npz(
        out_prefix.with_suffix(".npz"),
        up_s=up_s, down_s=down_s,
        segment_time_ms=np.arange(
            int(round(args.step_ms / operators.dt_ms))) * operators.dt_ms,
        **traces)
    payload = {
        "status": "CONTINUOUS_STATE_HYSTERESIS_DISCOVERY_COMPLETE",
        "substrate": boundary_config["substrate"],
        "coarsening": {
            "factor": int(args.coarsen_factor),
            "dt_ms": float(operators.dt_ms),
            "role": "discovery_only; selected boundaries require native-dt validation",
        },
        "sweep": {
            "s_start": float(args.s_start), "s_min": float(args.s_min),
            "s_max": float(args.s_max), "s_step": float(args.s_step),
            "settle_ms": float(args.settle_ms),
            "per_step_ms": float(args.step_ms),
            "hidden_state_continuity": (
                "rates, synaptic currents, delay histories, and M are all carried "
                "between adjacent parameter values"),
        },
        "tonic_runaway_gate": gate,
        "up_sweep_transition_brackets": transition("up"),
        "down_sweep_transition_brackets": transition("down"),
        "records": records,
        "interpretation": (
            "Different up/down transition locations establish hysteresis and "
            "exclude a unique one-dimensional stable/runaway threshold. They do "
            "not identify the periodic-orbit bifurcation type."),
        "sources": {
            "model": {"path": str(model_path), "sha256": sha256(model_path)},
            "z_map": {"path": str(z_map_path), "sha256": sha256(z_map_path)},
            "branch": {"path": str(branch_path), "sha256": sha256(branch_path)},
            "regime_config": {"path": str(args.regime_config.resolve()),
                              "sha256": sha256(args.regime_config.resolve())},
            "boundary_config": {"path": str(args.boundary_config.resolve()),
                                "sha256": sha256(args.boundary_config.resolve())},
            "transition_config": {"path": str(transition_path),
                                  "sha256": sha256(transition_path)},
            "candidate_manifest": {"path": str(manifest_path),
                                   "sha256": sha256(manifest_path)},
        },
    }
    atomic_json(payload, out_prefix.with_suffix(".json"))
    print(json.dumps({
        "status": payload["status"],
        "up": payload["up_sweep_transition_brackets"],
        "down": payload["down_sweep_transition_brackets"],
        "output": str(out_prefix.with_suffix('.json')),
    }, indent=2))


if __name__ == "__main__":
    main()
