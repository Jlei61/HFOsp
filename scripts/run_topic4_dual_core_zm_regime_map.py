#!/usr/bin/env python3
"""Native-dt dual-core spatial-Z x dynamic-M basin/regime map."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import time

for _thread_env in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_env] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "src" / "snn_engine"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from scripts.run_topic4_dual_core_oscillatory_phase_map import (  # noqa: E402
    _branch_seed,
    _perturbed,
    _terminal_low_seed,
)
from scripts.run_topic4_dual_core_spatial_z_bifurcation import load_z_map  # noqa: E402
from scripts.run_topic4_dual_core_spatial_z_stability_assay import _substrate  # noqa: E402
from src.topic4_dual_core_oscillation_phase import classify_coarse_trajectory  # noqa: E402
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


DEFAULT_ROOT = Path(
    "/data/hfosp_topic4_fig45_artifacts/fig5/"
    "data_driven_dual_core_spatial_z/zm_regime_map")
_WORK = {}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(descriptor)
    try:
        Path(temporary).write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
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
    result = np.asarray([float(value) for value in text.split(",")], float)
    if result.size < 1 or np.any(~np.isfinite(result)):
        raise ValueError("axis override must contain finite comma-separated values")
    return result


def _cell(task):
    row, column, s, gain_scale = task
    model = _WORK["model"]
    z_map = _WORK["z_map"]
    archive = _WORK["archive"]
    config = _WORK["config"]
    operators = _WORK["operators"]
    dynamics = config["dynamics"]
    eta_m = float(dynamics["base_eta_m"]) * float(gain_scale)
    tau_m = float(dynamics["tau_m_ms"])
    dynamic_model = replace(model, tau_gaba_ms=float(dynamics["tau_d_GABA_ms"]))
    z_a, z_b, z_surround, field = path_state(z_map, float(s))
    second = z_map.z_second_moment_field(
        z_a=z_a, z_b=z_b, z_surround=z_surround)

    low_seed = np.r_[
        np.full(model.n_cells, 1e-4), np.full(model.n_cells, 5e-4)]
    terminal_low = _terminal_low_seed(archive)
    high_seed = _branch_seed(archive, prefix="outer_high", parameter=float(s))
    low = solve_spatial_z_fixed_point(
        dynamic_model, z_map, parameter=float(s), initial_rates=low_seed,
        eta_m=eta_m, tau_m_slow_ms=tau_m, maxfev=30000)
    high = solve_spatial_z_fixed_point(
        dynamic_model, z_map, parameter=float(s), initial_rates=high_seed,
        eta_m=eta_m, tau_m_slow_ms=tau_m, maxfev=30000)
    starts = {
        "low_initial": {
            "rates": (low.rates if low.converged and low.physical else terminal_low),
            "source": ("target_fixed_point" if low.converged and low.physical
                       else "nearest_pre_fold_continuation_state"),
            "fixed_point_mean_E_rate_hz": (
                float(low.mean_rate_e_hz)
                if low.converged and low.physical else None),
        },
        "high_initial": {
            "rates": (high.rates if high.converged and high.physical else high_seed),
            "source": ("target_fixed_point" if high.converged and high.physical
                       else "nearest_eta0_outer_high_continuation_state"),
            "fixed_point_mean_E_rate_hz": (
                float(high.mean_rate_e_hz)
                if high.converged and high.physical else None),
        },
    }
    starts = {name: value for name, value in starts.items()
              if name in _WORK["initial_branches"]}
    steps = int(round(float(dynamics["duration_ms"]) / operators.dt_ms))
    zero_ou = np.zeros((steps, model.n_cells), np.float32)
    records = {}
    traces = {}
    for branch, start in starts.items():
        initial = _perturbed(
            start["rates"], n_cells=model.n_cells,
            fraction=float(dynamics["perturbation_fraction"]),
            floor=float(dynamics["perturbation_floor_per_ms"]))
        trajectory = simulate_delayed_ou_trajectory(
            dynamic_model, operators, initial, z_field=field,
            z_second_moment=second, ou_rate_e=zero_ou,
            tail_steps=int(round(float(dynamics["tail_ms"]) / operators.dt_ms)),
            reference_rates=start["rates"], eta_m=eta_m,
            tau_m_slow_ms=tau_m)
        regional = regional_rates_hz(
            model, z_map,
            np.asarray(trajectory["tail_mean_rates"][:model.n_cells]))
        state = classify_coarse_trajectory(
            trajectory["mean_e_rate_hz"], dt_ms=operators.dt_ms,
            regional_tail_rate_hz=[
                regional["core_a"], regional["core_b"], regional["surround"]],
            tail_ms=float(dynamics["tail_ms"]),
            window_ms=float(dynamics["window_ms"]))
        records[branch] = {
            "initial_state_source": start["source"],
            "fixed_point_mean_E_rate_hz": start["fixed_point_mean_E_rate_hz"],
            "final_mean_adaptation_state": float(np.mean(
                trajectory["final_adaptation_state"])),
            **state,
        }
        traces[branch] = np.asarray(trajectory["mean_e_rate_hz"], np.float32)
    return {
        "row": int(row), "column": int(column), "s": float(s),
        "Z_A": float(z_a), "Z_B": float(z_b),
        "Z_surround": float(z_surround),
        "adaptation_gain_scale": float(gain_scale), "eta_m": eta_m,
        "records": records, "traces": traces,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "config/topic4_dual_core_zm_regime_v1.json")
    parser.add_argument(
        "--rev21-config", type=Path,
        default=ROOT / "config/topic4_rev21_dual_core_zm_transition.json")
    parser.add_argument(
        "--coarse-root", type=Path,
        default=Path("/data/hfosp_topic4_fig45_artifacts/fig5/"
                     "data_driven_dual_core_spatial_z"))
    parser.add_argument(
        "--source-artifact-root", type=Path,
        default=Path("/home/honglab/leijiaxin/HFOsp"))
    parser.add_argument("--s-values")
    parser.add_argument("--adaptation-gain-scales")
    parser.add_argument("--duration-ms", type=float)
    parser.add_argument(
        "--initial-branches", default="low_initial,high_initial",
        help="comma-separated subset of low_initial,high_initial")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-prefix", type=Path)
    args = parser.parse_args()
    started = time.time()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.duration_ms is not None:
        if not np.isfinite(args.duration_ms) or args.duration_ms < float(
                config["dynamics"]["tail_ms"]):
            raise ValueError("duration override must be finite and at least tail_ms")
        config["dynamics"]["duration_ms"] = float(args.duration_ms)
    if config["substrate"] != "dualcore_s39 + Joint=1.25":
        raise RuntimeError("Z/M regime substrate identity drifted")
    s_values = _values(args.s_values, config["axes"]["core_disinhibition_s"])
    gain_values = _values(
        args.adaptation_gain_scales,
        config["axes"]["adaptation_gain_scale"])
    initial_branches = tuple(
        value.strip() for value in args.initial_branches.split(",")
        if value.strip())
    allowed_branches = {"low_initial", "high_initial"}
    if (not initial_branches
            or len(set(initial_branches)) != len(initial_branches)
            or not set(initial_branches) <= allowed_branches):
        raise ValueError(
            "initial-branches must be a unique non-empty subset of "
            "low_initial,high_initial")
    if np.any(s_values < 0.0) or np.any(s_values > 1.0):
        raise ValueError("s values must lie in [0, 1]")
    if np.any(gain_values < 0.0):
        raise ValueError("adaptation gain scales must be non-negative")

    coarse_root = args.coarse_root.resolve()
    model_path = coarse_root / "deterministic_meanfield/dualcore_topology_2542_ngrid10.npz"
    z_map_path = model_path.with_suffix(".zmap.npz")
    branch_path = coarse_root / "bifurcation/dualcore_spatial_z_bifurcation.npz"
    model = load_patient_coarse_model(model_path)
    z_map = load_z_map(z_map_path)
    # A lazy NpzFile owns one shared compressed-file descriptor.  Forked
    # workers would race its seek/decompress state, so materialize every array
    # before the process pool is created.
    with np.load(branch_path) as branch_archive:
        archive = {name: np.asarray(branch_archive[name])
                   for name in branch_archive.files}
    substrate, transition_path, _ou, manifest_path = _substrate(
        args.rev21_config.resolve(), args.source_artifact_root.resolve(),
        int(config["topology_seed"]))
    operators = build_coarse_delay_operators(substrate, model)
    expected_dt = float(config["dynamics"]["integration_dt_ms"])
    if not np.isclose(operators.dt_ms, expected_dt):
        raise RuntimeError(
            f"native delay dt is {operators.dt_ms}, expected {expected_dt}")

    global _WORK
    _WORK = {
        "model": model, "z_map": z_map, "archive": archive,
        "config": config, "operators": operators,
        "initial_branches": initial_branches,
    }
    tasks = [
        (row, column, float(s), float(gain))
        for row, s in enumerate(s_values)
        for column, gain in enumerate(gain_values)
    ]
    cells = []
    workers = max(1, int(args.workers))
    if workers == 1:
        for index, task in enumerate(tasks, 1):
            cells.append(_cell(task))
            print(json.dumps({"completed": index, "total": len(tasks),
                              "s": task[2], "gain_scale": task[3]}), flush=True)
    else:
        with ProcessPoolExecutor(
                max_workers=workers, mp_context=mp.get_context("fork")) as pool:
            futures = {pool.submit(_cell, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), 1):
                task = futures[future]
                cells.append(future.result())
                print(json.dumps({"completed": index, "total": len(tasks),
                                  "s": task[2], "gain_scale": task[3],
                                  "elapsed_seconds": round(time.time() - started, 1)}),
                      flush=True)

    shape = (len(s_values), len(gain_values))
    steps = int(round(
        float(config["dynamics"]["duration_ms"]) / operators.dt_ms))
    arrays = {
        "core_disinhibition_s": s_values,
        "adaptation_gain_scale": gain_values,
        "eta_m": float(config["dynamics"]["base_eta_m"]) * gain_values,
        "time_ms": np.arange(steps, dtype=np.float32) * operators.dt_ms,
    }
    records = []
    for branch in initial_branches:
        code = np.full(shape, -1, np.int8)
        frequency = np.full(shape, np.nan, np.float32)
        depth = np.full(shape, np.nan, np.float32)
        mean = np.full(shape, np.nan, np.float32)
        traces = np.full(shape + (steps,), np.nan, np.float32)
        for cell in cells:
            row, column = cell["row"], cell["column"]
            state = cell["records"][branch]
            code[row, column] = int(state["state_code"])
            frequency[row, column] = float(state["whole_tail"]["dominant_hz"])
            depth[row, column] = float(state["whole_tail"]["modulation_depth"])
            mean[row, column] = float(state["whole_tail"]["mean_rate_hz"])
            traces[row, column] = cell["traces"][branch]
            records.append({
                key: value for key, value in cell.items()
                if key not in ("records", "traces")
            } | {"initial_branch": branch, **state})
        arrays[f"{branch}__state_code"] = code
        arrays[f"{branch}__dominant_hz"] = frequency
        arrays[f"{branch}__modulation_depth"] = depth
        arrays[f"{branch}__mean_rate_hz"] = mean
        arrays[f"{branch}__mean_e_trace_hz"] = traces

    out_prefix = (
        args.out_prefix.resolve() if args.out_prefix else
        DEFAULT_ROOT / "native_spatial_z_dynamic_m_tauGABA9")
    _atomic_npz(out_prefix.with_suffix(".npz"), **arrays)
    counts = {
        branch: {
            label: int(np.sum(arrays[f"{branch}__state_code"] == code))
            for label, code in (
                ("low", 0), ("intermediate", 1),
                ("tonic_recruited", 2), ("oscillatory_recruited", 3))
        }
        for branch in initial_branches
    }
    payload = {
        "status": "DUAL_CORE_NATIVE_SPATIAL_Z_DYNAMIC_M_REGIME_COMPLETE",
        "schema_id": config["schema_id"],
        "substrate": config["substrate"],
        "topology_seed": int(config["topology_seed"]),
        "axes": {
            "core_disinhibition_s": s_values.tolist(),
            "adaptation_gain_scale": gain_values.tolist(),
            "eta_m": arrays["eta_m"].tolist(),
        },
        "dynamics": config["dynamics"],
        "spatial_z_path": config["spatial_z_path"],
        "state_gate": config["state_gate"],
        "initial_basins": list(initial_branches),
        "counts": counts,
        "records": records,
        "sources": {
            "config": {"path": str(config_path), "sha256": _sha256(config_path)},
            "rev21_config": {"path": str(args.rev21_config.resolve()),
                             "sha256": _sha256(args.rev21_config.resolve())},
            "coarse_model": {"path": str(model_path), "sha256": _sha256(model_path)},
            "spatial_z_map": {"path": str(z_map_path), "sha256": _sha256(z_map_path)},
            "continuation_branches": {"path": str(branch_path),
                                      "sha256": _sha256(branch_path)},
            "transition_config": {"path": str(transition_path),
                                  "sha256": _sha256(transition_path)},
            "candidate_manifest": {"path": str(manifest_path),
                                   "sha256": _sha256(manifest_path)},
        },
        "claim_boundary": config["claim_boundary"],
        "wall_seconds": float(time.time() - started),
    }
    _atomic_json(out_prefix.with_suffix(".json"), payload)
    print(json.dumps({"status": payload["status"], "counts": counts,
                      "output": str(out_prefix.with_suffix(".json")),
                      "wall_seconds": payload["wall_seconds"]}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
