#!/usr/bin/env python3
"""Map low/tonic/oscillatory states in spatial-Z x fast-inhibition space."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "src" / "snn_engine"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from scripts.run_topic4_dual_core_spatial_z_bifurcation import load_z_map  # noqa: E402
from scripts.run_topic4_dual_core_spatial_z_stability_assay import _substrate  # noqa: E402
from src.topic4_dual_core_oscillation_phase import (  # noqa: E402
    classify_coarse_trajectory,
    coarsen_delay_operators,
)
from src.topic4_dual_core_spatial_z import (  # noqa: E402
    path_state,
    regional_rates_hz,
    solve_spatial_z_fixed_point,
)
from src.topic4_dual_core_spatial_z_delay import (  # noqa: E402
    build_coarse_delay_operators,
    simulate_delayed_ou_trajectory,
)
from src.topic4_patient_zm_meanfield import load_patient_coarse_model  # noqa: E402


DEFAULT_ARTIFACT_ROOT = Path(
    "/data/hfosp_topic4_fig45_artifacts/fig5/"
    "data_driven_dual_core_spatial_z")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(descriptor)
    try:
        Path(temporary).write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=path.stem + ".", suffix=".npz")
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _values(text: str | None, fallback) -> np.ndarray:
    if text is None:
        return np.asarray(fallback, float)
    values = np.asarray([float(value) for value in text.split(",")], float)
    if values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError("parameter override must contain finite comma-separated values")
    return values


def _perturbed(rates, *, n_cells: int, fraction: float, floor: float) -> np.ndarray:
    result = np.asarray(rates, float).copy()
    phase = np.linspace(0.0, 2.0 * np.pi, n_cells, endpoint=False)
    pattern = np.sin(phase) + 0.5 * np.cos(3.0 * phase)
    pattern /= max(float(np.max(np.abs(pattern))), 1e-12)
    result[:n_cells] += np.maximum(
        np.abs(result[:n_cells]) * float(fraction), float(floor)) * pattern
    result[n_cells:] -= np.maximum(
        np.abs(result[n_cells:]) * float(fraction), float(floor)) * pattern
    np.maximum(result, 0.0, out=result)
    return result


def _branch_seed(archive, *, prefix: str, parameter: float) -> np.ndarray:
    """Return the nearest audited continuation state for a target path value."""
    parameters = np.asarray(archive[f"{prefix}__s"], float)
    rates = np.asarray(archive[f"{prefix}__rates"], float)
    if rates.shape[0] != parameters.size:
        raise ValueError(f"continuation arrays for {prefix} do not align")
    index = int(np.argmin(np.abs(parameters - float(parameter))))
    return rates[index]


def _terminal_low_seed(archive) -> np.ndarray:
    """Return the last low-rate continuation state immediately before its fold."""
    rates_hz = np.asarray(archive["entry_saddle__mean_e_hz"], float)
    parameters = np.asarray(archive["entry_saddle__s"], float)
    candidates = np.flatnonzero(rates_hz < 1.0)
    if candidates.size == 0:
        raise ValueError("entry continuation contains no low-rate segment")
    index = int(candidates[np.argmax(parameters[candidates])])
    return np.asarray(archive["entry_saddle__rates"][index], float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "config/topic4_dual_core_oscillatory_phase_v1.json")
    parser.add_argument(
        "--rev21-config", type=Path,
        default=ROOT / "config/topic4_rev21_dual_core_zm_transition.json")
    parser.add_argument(
        "--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument(
        "--source-artifact-root", type=Path,
        default=Path("/home/honglab/leijiaxin/HFOsp"))
    parser.add_argument("--s-values")
    parser.add_argument("--tau-gaba-values")
    parser.add_argument("--duration-ms", type=float)
    parser.add_argument("--delay-coarsening-factor", type=int)
    parser.add_argument(
        "--initial-branches", choices=("both", "low", "high"), default="both")
    parser.add_argument("--out-prefix", type=Path)
    args = parser.parse_args()
    started = time.time()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    artifact_root = args.artifact_root.resolve()
    model_path = (
        artifact_root / "deterministic_meanfield/dualcore_topology_2542_ngrid10.npz")
    z_map_path = model_path.with_suffix(".zmap.npz")
    model = load_patient_coarse_model(model_path)
    z_map = load_z_map(z_map_path)
    branch_path = artifact_root / "bifurcation/dualcore_spatial_z_bifurcation.npz"
    branch_archive = np.load(branch_path)
    if config["substrate"] != "dualcore_s39 + Joint=1.25":
        raise RuntimeError("phase map substrate identity drifted")

    s_values = _values(args.s_values, config["axes"]["core_disinhibition_s"])
    tau_values = _values(
        args.tau_gaba_values, config["axes"]["tau_d_GABA_ms"])
    discovery = config["discovery"]
    duration_ms = float(
        discovery["duration_ms"] if args.duration_ms is None
        else args.duration_ms)
    tail_ms = float(discovery["tail_ms"])
    if duration_ms < tail_ms:
        raise ValueError("duration must be at least as long as the scored tail")

    substrate, transition_path, _ou, manifest_path = _substrate(
        args.rev21_config.resolve(), args.source_artifact_root.resolve(),
        int(config["topology_seed"]))
    native_operators = build_coarse_delay_operators(substrate, model)
    factor = int(
        discovery["delay_dt_coarsening_factor"]
        if args.delay_coarsening_factor is None
        else args.delay_coarsening_factor)
    operators = coarsen_delay_operators(native_operators, factor=factor)
    n_steps = int(round(duration_ms / operators.dt_ms))
    zero_drive = np.zeros((n_steps, model.n_cells), np.float32)

    shape = (len(s_values), len(tau_values))
    branch_names = {
        "both": ("low_initial", "high_initial"),
        "low": ("low_initial",),
        "high": ("high_initial",),
    }[args.initial_branches]
    state_code = {
        branch: np.full(shape, -1, np.int8) for branch in branch_names}
    dominant_hz = {
        branch: np.full(shape, np.nan, np.float32) for branch in branch_names}
    modulation = {
        branch: np.full(shape, np.nan, np.float32) for branch in branch_names}
    mean_rate = {
        branch: np.full(shape, np.nan, np.float32) for branch in branch_names}
    traces = {
        branch: np.full((len(s_values), len(tau_values), n_steps), np.nan,
                        np.float32)
        for branch in branch_names
    }
    records = []
    for row, parameter in enumerate(s_values):
        row_started = time.time()
        low_seed = np.r_[
            np.full(model.n_cells, 1e-4), np.full(model.n_cells, 5e-4)]
        terminal_low_seed = _terminal_low_seed(branch_archive)
        high_seed = _branch_seed(
            branch_archive, prefix="outer_high", parameter=float(parameter))
        low = solve_spatial_z_fixed_point(
            model, z_map, parameter=float(parameter), initial_rates=low_seed,
            maxfev=30000)
        high = solve_spatial_z_fixed_point(
            model, z_map, parameter=float(parameter), initial_rates=high_seed,
            maxfev=30000)
        z_a, z_b, z_surround, field = path_state(z_map, float(parameter))
        second = z_map.z_second_moment_field(
            z_a=z_a, z_b=z_b, z_surround=z_surround)
        initial_states = {
            # Above the low-branch fold, retain the terminal low state as a
            # standardized pre-fold initial condition.  Its ensuing runaway is
            # an outcome, not a fixed point at the target parameter.
            "low_initial": {
                "rates": (low.rates if low.converged and low.physical
                          else terminal_low_seed),
                "reference": (low.rates if low.converged and low.physical
                              else terminal_low_seed),
                "source": ("target_fixed_point" if low.converged and low.physical
                           else "nearest_pre_fold_continuation_state"),
                "fixed_point_mean": (
                    float(low.mean_rate_e_hz)
                    if low.converged and low.physical else None),
            },
            "high_initial": {
                "rates": high.rates if high.converged and high.physical else high_seed,
                "reference": high.rates if high.converged and high.physical else high_seed,
                "source": ("target_fixed_point" if high.converged and high.physical
                           else "nearest_outer_high_continuation_state"),
                "fixed_point_mean": (
                    float(high.mean_rate_e_hz)
                    if high.converged and high.physical else None),
            },
        }
        initial_states = {
            name: initial_states[name] for name in branch_names
        }
        for column, tau_gaba in enumerate(tau_values):
            dynamic_model = replace(model, tau_gaba_ms=float(tau_gaba))
            for branch, initial_state in initial_states.items():
                initial = _perturbed(
                    initial_state["rates"], n_cells=model.n_cells,
                    fraction=float(discovery["perturbation_fraction"]),
                    floor=float(discovery["perturbation_floor_per_ms"]))
                trajectory = simulate_delayed_ou_trajectory(
                    dynamic_model, operators, initial, z_field=field,
                    z_second_moment=second, ou_rate_e=zero_drive,
                    tail_steps=int(round(tail_ms / operators.dt_ms)),
                    reference_rates=initial_state["reference"])
                regional = regional_rates_hz(
                    model, z_map,
                    np.asarray(trajectory["tail_mean_rates"][:model.n_cells]))
                metrics = classify_coarse_trajectory(
                    trajectory["mean_e_rate_hz"], dt_ms=operators.dt_ms,
                    regional_tail_rate_hz=[
                        regional["core_a"], regional["core_b"],
                        regional["surround"]],
                    tail_ms=tail_ms,
                    window_ms=float(discovery["window_ms"]),
                    minimum_high_rate_hz=float(
                        config["state_gate"]["minimum_high_rate_hz"]),
                    minimum_regional_rate_hz=float(
                        config["state_gate"]["minimum_regional_rate_hz"]),
                    minimum_modulation_depth=float(
                        config["state_gate"][
                            "minimum_whole_tail_modulation_depth"]),
                )
                state_code[branch][row, column] = int(metrics["state_code"])
                dominant_hz[branch][row, column] = float(
                    metrics["whole_tail"]["dominant_hz"])
                modulation[branch][row, column] = float(
                    metrics["whole_tail"]["modulation_depth"])
                mean_rate[branch][row, column] = float(
                    metrics["whole_tail"]["mean_rate_hz"])
                traces[branch][row, column] = trajectory["mean_e_rate_hz"]
                records.append({
                    "s": float(parameter),
                    "Z_A": float(z_a), "Z_B": float(z_b),
                    "Z_surround": float(z_surround),
                    "tau_d_GABA_ms": float(tau_gaba),
                    "initial_branch": branch,
                    "initial_state_source": initial_state["source"],
                    "fixed_point_mean_E_rate_hz": initial_state["fixed_point_mean"],
                    **metrics,
                })
        row_counts = {
            branch: {
                label: int(np.sum(state_code[branch][row] == code))
                for label, code in (
                    ("low", 0), ("intermediate", 1),
                    ("tonic", 2), ("oscillatory", 3),
                    ("unresolved", -1))
            }
            for branch in branch_names
        }
        print(json.dumps({
            "row": row + 1,
            "rows": len(s_values),
            "s": float(parameter),
            "counts": row_counts,
            "row_seconds": round(time.time() - row_started, 2),
            "elapsed_seconds": round(time.time() - started, 2),
        }), flush=True)

    out_prefix = (args.out_prefix.resolve() if args.out_prefix else
                  artifact_root / "oscillatory_phase_map/"
                  "dualcore_spatial_z_tau_gaba_phase_v1")
    arrays = {
        "core_disinhibition_s": s_values,
        "tau_d_GABA_ms": tau_values,
        "time_ms": np.arange(n_steps, dtype=np.float32) * operators.dt_ms,
    }
    for branch in branch_names:
        arrays[f"{branch}__state_code"] = state_code[branch]
        arrays[f"{branch}__dominant_hz"] = dominant_hz[branch]
        arrays[f"{branch}__modulation_depth"] = modulation[branch]
        arrays[f"{branch}__mean_rate_hz"] = mean_rate[branch]
        arrays[f"{branch}__mean_e_trace_hz"] = traces[branch]
    _atomic_npz(out_prefix.with_suffix(".npz"), **arrays)
    payload = {
        "status": "DUAL_CORE_OSCILLATORY_PHASE_DISCOVERY_COMPLETE",
        "schema_id": config["schema_id"],
        "substrate": config["substrate"],
        "topology_seed": int(config["topology_seed"]),
        "axes": {
            "core_disinhibition_s": s_values.tolist(),
            "tau_d_GABA_ms": tau_values.tolist(),
        },
        "spatial_z_path": config["spatial_z_path"],
        "integration": {
            "native_dt_ms": float(native_operators.dt_ms),
            "discovery_dt_ms": float(operators.dt_ms),
            "delay_coarsening_factor": factor,
            "all_pathway_weights_conserved": True,
            "maximum_delay_ms": float(
                operators.dt_ms * operators.max_delay_steps),
            "duration_ms": duration_ms,
            "tail_ms": tail_ms,
            "deterministic_zero_OU_discovery": True,
        },
        "state_gate": config["state_gate"],
        "records": records,
        "counts": {
            branch: {
                state: int(np.sum(state_code[branch] == code))
                for state, code in {
                    "low": 0, "intermediate": 1,
                    "tonic_recruited": 2,
                    "oscillatory_recruited": 3,
                    "root_unresolved": -1,
                }.items()
            }
            for branch in branch_names
        },
        "source": {
            "config": {"path": str(config_path), "sha256": _sha256(config_path)},
            "rev21_config": {
                "path": str(args.rev21_config.resolve()),
                "sha256": _sha256(args.rev21_config.resolve())},
            "coarse_model": {"path": str(model_path), "sha256": _sha256(model_path)},
            "spatial_z_map": {"path": str(z_map_path), "sha256": _sha256(z_map_path)},
            "continuation_branches": {
                "path": str(branch_path), "sha256": _sha256(branch_path)},
            "transition_config": {
                "path": str(transition_path), "sha256": _sha256(transition_path)},
            "candidate_manifest": {
                "path": str(manifest_path), "sha256": _sha256(manifest_path)},
        },
        "required_followup": config["required_followup"],
        "claim_boundary": config["claim_boundary"],
        "wall_seconds": float(time.time() - started),
    }
    _atomic_json(out_prefix.with_suffix(".json"), payload)
    print(json.dumps({
        "status": payload["status"], "counts": payload["counts"],
        "output": str(out_prefix.with_suffix(".json")),
        "wall_seconds": payload["wall_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
