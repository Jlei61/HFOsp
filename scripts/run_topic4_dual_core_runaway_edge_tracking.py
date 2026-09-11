#!/usr/bin/env python3
"""Native-dt carried-state test across the fixed-initial runaway bracket."""
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
from src.topic4_dual_core_spatial_z import path_state, regional_rates_hz  # noqa: E402
from src.topic4_dual_core_spatial_z_delay import (  # noqa: E402
    build_coarse_delay_operators,
    simulate_delayed_ou_trajectory,
)
from src.topic4_patient_zm_meanfield import load_patient_coarse_model  # noqa: E402


DEFAULT_ROOT = Path(
    "/data/hfosp_topic4_fig45_artifacts/fig5/"
    "data_driven_dual_core_spatial_z")


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
    parser.add_argument("--source-s", type=float, default=0.428)
    parser.add_argument("--target-s", type=float, default=0.429)
    parser.add_argument("--duration-ms", type=float, default=10000.0)
    parser.add_argument(
        "--artifact-root", type=Path,
        default=Path("/home/honglab/leijiaxin/HFOsp"))
    parser.add_argument(
        "--out-prefix", type=Path,
        default=DEFAULT_ROOT / "runaway_boundary/native_edge_tracking_0p428_to_0p429")
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
    operators = build_coarse_delay_operators(substrate, model)
    steps = int(round(args.duration_ms / operators.dt_ms))
    initial = _perturbed(
        _terminal_low_seed(branch_arrays), n_cells=model.n_cells,
        fraction=float(regime_config["dynamics"]["perturbation_fraction"]),
        floor=float(regime_config["dynamics"]["perturbation_floor_per_ms"]))

    trajectories = []
    records = []
    state = None
    rates = initial
    for index, s in enumerate((args.source_s, args.target_s)):
        z_a, z_b, z_surround, field = path_state(z_map, float(s))
        second = z_map.z_second_moment_field(
            z_a=z_a, z_b=z_b, z_surround=z_surround)
        resume = {} if state is None else {
            "initial_m": state["final_adaptation_state"],
            "initial_synapses": state["final_synapses"],
            "initial_history_e": state["final_history_e"],
            "initial_history_i": state["final_history_i"],
        }
        state = simulate_delayed_ou_trajectory(
            model, operators, rates, z_field=field, z_second_moment=second,
            ou_rate_e=np.zeros((steps, model.n_cells), np.float32),
            tail_steps=int(round(
                float(gate["tail_duration_ms"]) / operators.dt_ms)),
            eta_m=eta_m, tau_m_slow_ms=tau_m, **resume)
        rates = state["final_rates"]
        trace = np.asarray(state["mean_e_rate_hz"], float)
        tail_steps = int(round(
            float(gate["tail_duration_ms"]) / operators.dt_ms))
        tail = trace[-tail_steps:]
        first, second_half = np.array_split(tail, 2)
        regional = regional_rates_hz(
            model, z_map, state["tail_mean_rates"][:model.n_cells])
        runaway = bool(
            np.mean(tail) >= float(gate["minimum_population_mean_rate_hz"])
            and min(regional.values()) >= float(
                gate["minimum_each_regional_rate_hz"])
            and abs(np.mean(second_half) - np.mean(first)) <= float(
                gate["maximum_absolute_half_tail_drift_hz"]))
        records.append({
            "stage": "source_settle" if index == 0 else "carried_state_target",
            "s": float(s), "population_tail_mean_hz": float(np.mean(tail)),
            "half_tail_drift_hz": float(
                np.mean(second_half) - np.mean(first)),
            "regional_tail_rate_hz": regional,
            "runaway": runaway,
        })
        trajectories.append(trace.astype(np.float32))

    out_prefix = args.out_prefix.resolve()
    atomic_npz(
        out_prefix.with_suffix(".npz"),
        time_ms=np.arange(steps) * operators.dt_ms,
        source_trace_hz=trajectories[0], target_trace_hz=trajectories[1])
    outcome = (
        "BASIN_BOUNDARY_CONFIRMED"
        if not records[0]["runaway"] and not records[1]["runaway"]
        else "ATTRACTOR_LOSS_OR_SLOW_ESCAPE_NOT_EXCLUDED")
    payload = {
        "status": outcome,
        "substrate": boundary_config["substrate"],
        "native_dt_ms": float(operators.dt_ms),
        "hidden_state_contract": (
            "rates, synaptic currents, every delay-history bin, and M are "
            "carried exactly from source_s to target_s"),
        "tonic_runaway_gate": gate,
        "records": records,
        "interpretation": (
            "If the carried localized state remains bounded at target_s while "
            "the standardized low/pre-fold preparation runs away there, the "
            "fixed-initial bracket is a basin-separatrix crossing rather than "
            "loss of the localized attractor."),
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
    print(json.dumps({"status": outcome, "records": records,
                      "output": str(out_prefix.with_suffix('.json'))}, indent=2))


if __name__ == "__main__":
    main()
