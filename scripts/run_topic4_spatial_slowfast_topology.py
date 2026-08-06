#!/usr/bin/env python3
"""Run the cheap Stage 0A canonical topology oracle.

The output is an analysis-chain sanity check only.  It must not be interpreted as a
mechanistic result for HFOsp, and it never starts the Stage 0B E--I or spatial models.
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import datetime as dt_datetime  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic4_spatial_slowfast_topology import (  # noqa: E402
    NormalFormParameters,
    OrbitClassifierThresholds,
    SlowLoopParameters,
    analyze_closed_slow_loop,
    bracket_contains,
    classify_orbit,
    dataclass_dict,
    detect_entry_exit_boundaries,
    run_state_fork_map,
    simulate_closed_slow_loop,
)


DEFAULT_CONFIG = ROOT / "config" / "topic4_spatial_slowfast_topology.yaml"
SOURCE_PATH = ROOT / "src" / "topic4_spatial_slowfast_topology.py"
CLAIM_BOUNDARY = "analysis_chain_oracle_only_no_hfosp_mechanism_claim"


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(command: list[str]) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(ROOT), *command], capture_output=True, text=True, check=False
    )
    value = proc.stdout.strip()
    return value or None


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(payload), handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _atomic_npz(path: pathlib.Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".tmp_", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _load_config(path: pathlib.Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return config


def _classifier_controls(thresholds: OrbitClassifierThresholds) -> dict[str, Any]:
    time_s = np.arange(0.0, 80.0 + 0.05, 0.05)
    phase = 2.0 * np.pi * time_s
    true_cycle = np.column_stack((np.cos(phase), np.sin(phase)))
    # This control deliberately oscillates at the expected frequency while pinned to
    # the ceiling.  A frequency-only classifier would falsely call it a limit cycle.
    ceiling = np.column_stack(
        (
            thresholds.ceiling_radius * np.cos(phase),
            thresholds.ceiling_radius * np.sin(phase),
        )
    )
    drifting_radius = np.linspace(1.0, 0.30, time_s.size)
    long_transient = np.column_stack(
        (drifting_radius * np.cos(phase), drifting_radius * np.sin(phase))
    )
    actual = {
        "known_cycle": classify_orbit(time_s, true_cycle, thresholds),
        "ceiling_control": classify_orbit(time_s, ceiling, thresholds),
        "long_transient_control": classify_orbit(time_s, long_transient, thresholds),
    }
    expected = {
        "known_cycle": "finite_limit_cycle",
        "ceiling_control": "saturation_or_ceiling",
        "long_transient_control": "indeterminate_long_transient",
    }
    checks = {
        name: actual[name]["classification"] == expected_label
        for name, expected_label in expected.items()
    }
    return {"pass": bool(all(checks.values())), "checks": checks, "expected": expected, "actual": actual}


def _find_row(rows: list[dict[str, Any]], mu: float, radius: float) -> dict[str, Any]:
    for row in rows:
        if np.isclose(row["mu"], mu, rtol=0, atol=1e-12) and np.isclose(
            row["initial_radius"], radius, rtol=0, atol=1e-12
        ):
            return row
    raise ValueError(f"state-fork map lacks required cell mu={mu}, radius={radius}")


def _representative_index(values: np.ndarray, target: float) -> int:
    indices = np.flatnonzero(np.isclose(values, target, rtol=0, atol=1e-12))
    if not indices.size:
        raise ValueError(f"required representative value absent: {target}")
    return int(indices[0])


def run(config_path: pathlib.Path, result_root: pathlib.Path) -> tuple[dict[str, Any], pathlib.Path, pathlib.Path]:
    config = _load_config(config_path)
    normal = NormalFormParameters(**config["normal_form"]).validate()
    classifier = OrbitClassifierThresholds(**config["classifier"]).validate()
    frozen = config["frozen_scan"]
    slow_config = config["slow_loop"]
    slow = SlowLoopParameters(**slow_config["parameters"]).validate()
    resource_contract = config["resource_contract"]
    if int(resource_contract["max_cpus"]) > 4:
        raise ValueError("Stage 0A config may not request more than 4 CPUs")
    if float(resource_contract["max_memory_gib"]) > 4.0:
        raise ValueError("Stage 0A config may not request more than 4 GiB")
    if int(resource_contract["blas_threads"]) != 1:
        raise ValueError("Stage 0A BLAS thread contract must remain 1")

    rows, map_arrays = run_state_fork_map(
        frozen["mu_values"],
        frozen["initial_radii"],
        params=normal,
        thresholds=classifier,
        dt_s=float(frozen["dt_s"]),
        duration_s=float(frozen["duration_s"]),
        save_stride=int(frozen["save_stride"]),
    )
    boundaries = detect_entry_exit_boundaries(
        rows,
        low_initial_radius=float(frozen["low_initial_radius"]),
        high_initial_radius=float(frozen["high_initial_radius"]),
    )

    low_cell = _find_row(rows, -0.10, float(frozen["low_initial_radius"]))
    high_cell = _find_row(rows, -0.10, float(frozen["high_initial_radius"]))
    monostable_cycle = _find_row(rows, 0.10, float(frozen["low_initial_radius"]))
    topology_gates = {
        "low_fixed_point_found": low_cell["classification"] == "low_fixed_point",
        "finite_high_cycle_found": monostable_cycle["classification"] == "finite_limit_cycle",
        "bistable_state_fork_found": low_cell["classification"] == "low_fixed_point"
        and high_cell["classification"] == "finite_limit_cycle",
        "entry_boundary_recovered": bracket_contains(
            normal.entry_mu,
            boundaries["entry_bracket_mu"],
            float(frozen["boundary_tolerance"]),
        ),
        "exit_boundary_recovered": bracket_contains(
            normal.exit_mu,
            boundaries["exit_bracket_mu"],
            float(frozen["boundary_tolerance"]),
        ),
        "finite_cycle_below_ceiling": monostable_cycle["ceiling_occupancy"]
        <= classifier.max_ceiling_occupancy,
        "cycle_frequency_matches_oracle": abs(
            monostable_cycle["dominant_frequency_hz"] - normal.omega_hz
        )
        <= 0.03,
    }
    classifier_controls = _classifier_controls(classifier)

    slow_traces = simulate_closed_slow_loop(
        normal=normal,
        slow=slow,
        dt_s=float(slow_config["dt_s"]),
        duration_s=float(slow_config["duration_s"]),
        save_stride=int(slow_config["save_stride"]),
    )
    slow_analysis = analyze_closed_slow_loop(
        slow_traces,
        normal=normal,
        slow=slow,
        classifier=classifier,
        retrigger_duration_s=float(slow_config["retrigger_duration_s"]),
        retrigger_dt_s=float(slow_config["retrigger_dt_s"]),
    )

    mu_values = map_arrays["mu_values"]
    radii = map_arrays["initial_radii"]
    low_mu_index = _representative_index(mu_values, -0.10)
    cycle_mu_index = _representative_index(mu_values, 0.10)
    low_radius_index = _representative_index(radii, float(frozen["low_initial_radius"]))
    high_radius_index = _representative_index(radii, float(frozen["high_initial_radius"]))
    n_radius = radii.size
    low_flat = low_mu_index * n_radius + low_radius_index
    bistable_high_flat = low_mu_index * n_radius + high_radius_index
    cycle_flat = cycle_mu_index * n_radius + low_radius_index

    trace_path = result_root / "stage0a_oracle_traces.npz"
    _atomic_npz(
        trace_path,
        frozen_time_s=map_arrays["time_s"],
        representative_low_xy=map_arrays["states"][:, low_flat, :],
        representative_bistable_high_xy=map_arrays["states"][:, bistable_high_flat, :],
        representative_monostable_cycle_xy=map_arrays["states"][:, cycle_flat, :],
        state_map_mu=mu_values,
        state_map_initial_radius=radii,
        state_map_classification=map_arrays["classification"],
        state_map_final_radius=map_arrays["final_radius"],
        slow_time_s=slow_traces["time_s"],
        slow_x=slow_traces["x"],
        slow_y=slow_traces["y"],
        slow_radius=slow_traces["radius"],
        slow_permissivity=slow_traces["permissivity"],
        slow_recovery=slow_traces["recovery"],
        slow_mu=slow_traces["mu"],
        slow_recovery_gate=slow_traces["recovery_gate"],
    )

    all_gates = {
        **{f"topology::{name}": bool(value) for name, value in topology_gates.items()},
        "classifier_controls": bool(classifier_controls["pass"]),
        "closed_slow_loop": bool(slow_analysis["pass"]),
    }
    verdict = "PASS" if all(all_gates.values()) else "FAIL"
    max_rss_gib = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0**2)
    provenance = {
        "utc": dt_datetime.datetime.now(dt_datetime.timezone.utc).isoformat(),
        "git_sha": _git(["rev-parse", "HEAD"]),
        "git_status_short": (_git(["status", "--short"]) or "").splitlines(),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "argv": sys.argv,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "source_sha256": _sha256(SOURCE_PATH),
        "runner_sha256": _sha256(pathlib.Path(__file__).resolve()),
        "trace_sha256": _sha256(trace_path),
        "resource_contract": resource_contract,
        "observed_max_rss_gib": max_rss_gib,
        "blas_threads": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
    }
    summary = {
        "stage": "Stage 0A topology oracle",
        "status": verdict,
        "scientific_gate": "analysis_chain_sanity",
        "claim_boundary": CLAIM_BOUNDARY,
        "downstream_authorization": "Stage 0B may start only after independent review; Stage 1/2/3 remain closed",
        "oracle": {
            "name": "quintic_subcritical_hopf_normal_form",
            "initial_condition_map_note": "radius-only forks are complete here because the oracle is rotationally symmetric; all initial phases are equivalent",
            "equations": {
                "dx_dt": "(mu + beta*rho^2 - rho^4)*x - omega*y",
                "dy_dt": "omega*x + (mu + beta*rho^2 - rho^4)*y",
            },
            "normal_form_parameters": dataclass_dict(normal),
            "analytic_entry_mu": normal.entry_mu,
            "analytic_exit_mu": normal.exit_mu,
            "analytic_bistable_interval_mu": [normal.exit_mu, normal.entry_mu],
        },
        "gates": all_gates,
        "topology": {
            "gates": topology_gates,
            "detected_boundaries": boundaries,
            "representative_low": low_cell,
            "representative_bistable_high": high_cell,
            "representative_monostable_cycle": monostable_cycle,
            "state_fork_rows": rows,
        },
        "classifier_controls": classifier_controls,
        "closed_slow_loop": slow_analysis,
        "closed_slow_loop_claim_boundary": {
            "purpose": "verifies entry/exit/retrigger analysis-chain logic on a constructed canonical toy",
            "does_not_support": "the actual HFOsp inhibitory-resource Z or recovery mechanism",
            "toy_permissivity_semantics": "permissivity rises to increase mu; it is intentionally not named or interpreted as project Z",
        },
        "config": {
            "classifier": dataclass_dict(classifier),
            "slow_loop": dataclass_dict(slow),
            "frozen_scan": frozen,
        },
        "artifacts": {
            "summary_json": str(result_root / "stage0a_oracle_summary.json"),
            "traces_npz": str(trace_path),
        },
        "provenance": provenance,
    }
    summary_path = result_root / "stage0a_oracle_summary.json"
    _atomic_json(summary_path, summary)
    return summary, summary_path, trace_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--result-root",
        type=pathlib.Path,
        default=None,
        help="Optional output override; relative paths are resolved from the repository root.",
    )
    parser.add_argument(
        "--confirm-run",
        action="store_true",
        help="Required guard even though this oracle is cheap and single-process.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.confirm_run:
        print("refusing to run: pass --confirm-run", file=sys.stderr)
        return 2
    config_path = args.config.resolve()
    config = _load_config(config_path)
    if args.result_root is None:
        result_root = ROOT / str(config["result_root"])
    else:
        result_root = args.result_root
        if not result_root.is_absolute():
            result_root = ROOT / result_root
    result_root = result_root.resolve()
    summary, summary_path, trace_path = run(config_path, result_root)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "claim_boundary": summary["claim_boundary"],
                "summary": str(summary_path),
                "traces": str(trace_path),
                "observed_max_rss_gib": summary["provenance"]["observed_max_rss_gib"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
