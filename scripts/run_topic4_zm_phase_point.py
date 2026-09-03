#!/usr/bin/env python3
"""Run one paired-contract endpoint of the spatial Z/M SNN phase screen.

The controlled coordinate is the actual frozen inhibitory resource ``q``:
``q_init == q_min == q_clamp`` and ``freeze_q == True``.  The low- and
high-start arms retain their checkpointed fast state and per-neuron M state,
but receive identical future Poisson and spatial-OU random streams at a given
``(q_clamp, eta_m, noise_seed)``.  This is a finite stochastic branch screen;
it does not assign a mathematical bifurcation type.
"""
from __future__ import annotations

import argparse
import copy
import contextlib
import hashlib
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "snn_engine"))

from scripts.run_topic4_spatial_zm_qigk_canary import (  # noqa: E402
    _frozen_endpoint_contact_centers,
    _frozen_gk_support_centers,
)
from src.snn_engine.checkpoint import load as load_checkpoint  # noqa: E402
from src.topic4_spatial_zm_qigk import (  # noqa: E402
    SpatialZMQIGKConfig,
    SpatialZMQIGKSlowVars,
)
from src.topic4_zm_ictal_transition import (  # noqa: E402
    build_substrate,
    load_round_config,
    make_external_drive,
)
from src.topic4_zm_phase_diagram import (  # noqa: E402
    BranchThresholds,
    classify_stationary_branch,
    scientific_contract_digest,
    stationary_metrics,
)


def validate_classifier_config(config):
    declared = config["classification"]
    limits = BranchThresholds()
    expected = {
        "contract_version": "event_tolerant_low_v2",
        "low": {
            "median_rate_hz_max": limits.low_median_rate_hz,
            "median_active_e_fraction_max_exclusive": limits.low_active_fraction,
            "high_rate_threshold_hz": limits.high_rate_threshold_hz,
            "maximum_contiguous_high_rate_ms_exclusive": (
                limits.low_max_sustained_high_run_ms),
        },
        "tonic_high": {
            "median_rate_hz_min": limits.high_median_rate_hz,
            "median_active_e_fraction_min": limits.high_active_fraction,
            "median_recruited_sheet_fraction_min": limits.high_sheet_fraction,
            "joint_global_recruitment_duty_min": limits.high_global_duty,
        },
    }
    observed = {
        "contract_version": declared["contract_version"],
        "low": declared["low"],
        "tonic_high": declared["tonic_high"],
    }
    if observed != expected:
        raise RuntimeError("phase config and code classifier thresholds disagree")
    return expected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _atomic_json(payload, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_npz(path: Path, **arrays) -> None:
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


def _resolved_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def validate_sources(config: dict) -> dict:
    """Fail before building the network if any frozen input has drifted."""
    source = config["source"]
    artifact_root = Path(source["repository_artifact_root"]).resolve()
    if not artifact_root.is_dir():
        raise FileNotFoundError(
            f"missing repository artifact root: {artifact_root}")
    specifications = {
        "round_config": source["round_config_sha256"],
        "trajectory_json": source["trajectory_json_sha256"],
        "low_checkpoint": source["low_checkpoint_sha256"],
        "high_checkpoint": source["high_checkpoint_sha256"],
    }
    audit = {}
    for name, expected in specifications.items():
        path = _resolved_path(source[name])
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen phase source: {path}")
        observed = _sha256(path)
        audit[name] = {
            "path": str(path),
            "expected_sha256": str(expected),
            "observed_sha256": observed,
            "match": observed == str(expected),
        }
        if observed != str(expected):
            raise RuntimeError(f"frozen phase source hash changed: {name}")
    cache_dir = _resolved_path(source["network_cache_dir"])
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"missing frozen network cache: {cache_dir}")
    audit["network_cache_dir"] = {"path": str(cache_dir), "exists": True}
    audit["repository_artifact_root"] = {
        "path": str(artifact_root), "exists": True,
        "role": "read-only resolver for gitignored frozen result artifacts",
    }
    return audit


def absolutize_round_inputs(round_config: dict, artifact_root: Path) -> dict:
    """Resolve ignored result artifacts without changing their frozen hashes."""
    resolved = copy.deepcopy(round_config)
    for record in resolved["inputs"].values():
        path = Path(record["path"])
        if not path.is_absolute():
            artifact_candidate = (Path(artifact_root) / path).resolve()
            code_candidate = (ROOT / path).resolve()
            if artifact_candidate.exists():
                record["path"] = str(artifact_candidate)
            elif code_candidate.exists():
                record["path"] = str(code_candidate)
            else:
                # Preserve a deterministic failure target for the producer's
                # own missing-input error.
                record["path"] = str(artifact_candidate)
    return resolved


@contextlib.contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _digest_item(hasher, label, value) -> None:
    hasher.update(str(label).encode())
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        hasher.update(array.tobytes())
        hasher.update(str(array.dtype).encode())
        hasher.update(str(array.shape).encode())
    else:
        hasher.update(json.dumps(value, sort_keys=True, default=str).encode())


def future_noise_digest(state: dict, *, update_steps: int) -> str:
    """Hash only the relative future stochastic process, not checkpoint time."""
    hasher = hashlib.sha256()
    _digest_item(hasher, "network_rng", state["rng_state"])
    _digest_item(hasher, "homogeneous_ou_xi", state["xi"])
    drive = state["external_drive"]
    for key in ("field_state", "cached", "rng_state"):
        _digest_item(hasher, f"spatial_ou.{key}", drive[key])
    _digest_item(hasher, "spatial_ou.relative_next_step", int(update_steps))
    _digest_item(hasher, "spatial_ou.relative_last_step", -1)
    return hasher.hexdigest()


def prepare_matched_state(checkpoint: dict, *, q_clamp: float,
                          noise_seed: int, fresh_drive) -> dict:
    """Make a checkpoint continuation with matched future stochastic input.

    Fast membrane/synaptic/delay state and M are retained.  Histories used only
    to evolve q are reset because q is frozen, preventing an irrelevant low-vs-
    high checkpoint history from contaminating the paired-noise contract.
    """
    q_clamp = float(q_clamp)
    if not 0.0 <= q_clamp <= 1.0:
        raise ValueError("q_clamp must lie in [0, 1]")
    state = copy.deepcopy(checkpoint)
    slow = state.get("slow")
    if slow is None or slow.get("kind") != "SpatialZMQIGKSlowVars":
        raise ValueError("phase continuation requires a spatial Z/M checkpoint")
    required = (
        "q_I", "qdriver_rE", "qdriver_rI", "field_count_E",
        "field_count_I", "last_m_drive_E", "last_q_drive", "z", "m",
    )
    missing = [name for name in required if slow.get(name) is None]
    if missing:
        raise ValueError("incomplete spatial Z/M checkpoint: " + ", ".join(missing))

    slow["q_I"] = np.full_like(slow["q_I"], q_clamp, dtype=float)
    for name in (
            "qdriver_rE", "qdriver_rI", "field_count_E", "field_count_I",
            "last_m_drive_E", "last_q_drive"):
        slow[name] = np.zeros_like(slow[name], dtype=float)
    slow["field_steps_seen"] = 0
    slow["field_steps_per_update"] = None
    slow["z"] = np.array(slow["z"], copy=True)
    n_e = int(np.asarray(slow["last_m_drive_E"]).size)
    slow["z"][:n_e] = q_clamp

    # The two checkpoint clocks differ, but their stochastic processes begin in
    # the same relative state and consume the same subsequent random numbers.
    state["rng_state"] = copy.deepcopy(
        np.random.default_rng(int(noise_seed)).bit_generator.state)
    state["xi"] = 0.0
    state["es_ema"] = 0.0
    state["es_run"] = 0
    absolute_step = int(state["step"])
    state["external_drive"] = {
        "field_state": np.array(fresh_drive._state, copy=True),
        "cached": np.array(fresh_drive._cached, copy=True),
        "next_step": absolute_step + int(fresh_drive.update_steps),
        "last_step": absolute_step - 1,
        "rng_state": copy.deepcopy(fresh_drive._rng.bit_generator.state),
    }
    return state


def _build_slow(substrate, source_payload, *, q_clamp, eta_m):
    endpoint_names, endpoint_xy = _frozen_endpoint_contact_centers(
        substrate, side="union")
    source_names, source_xy = _frozen_endpoint_contact_centers(
        substrate, side="source")
    sink_names, sink_xy = _frozen_endpoint_contact_centers(
        substrate, side="sink")
    gk_names, gk_xy = _frozen_gk_support_centers(
        substrate, support="downstream")
    values = dict(source_payload["hybrid_config"])
    values.update({
        "q_init": float(q_clamp),
        "q_min": float(q_clamp),
        "freeze_q": True,
        "eta_m": float(eta_m),
    })
    config = SpatialZMQIGKConfig(**values)
    slow = SpatialZMQIGKSlowVars(
        substrate.n_e + substrate.n_i, substrate.params.V_th,
        substrate.positions_e, substrate.positions_i,
        float(substrate.engine["L"]), substrate.h_e,
        core_mask_E=np.asarray(substrate.h_e >= 0.5, bool),
        endpoint_centers_xy=endpoint_xy,
        source_centers_xy=source_xy,
        sink_centers_xy=sink_xy,
        gk_centers_xy=gk_xy,
        cfg=config,
    )
    geometry = {
        "endpoint_names": endpoint_names,
        "source_names": source_names,
        "sink_names": sink_names,
        "gk_names": gk_names,
    }
    return slow, config, geometry


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="config/topic4_spatial_zm_phase_diagram_v1.json")
    parser.add_argument("--initial-state", choices=("low", "high"), required=True)
    parser.add_argument("--q-clamp", type=float, required=True)
    parser.add_argument("--eta-m", type=float, required=True)
    parser.add_argument("--noise-seed", type=int, required=True)
    parser.add_argument("--duration-ms", type=float, required=True)
    parser.add_argument("--burn-in-ms", type=float, required=True)
    parser.add_argument("--stage", default="manual")
    parser.add_argument("--out", required=True, help="Output JSON path; NPZ shares its stem")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.duration_ms <= 0.0:
        raise SystemExit("--duration-ms must be positive")
    if not 0.0 <= args.burn_in_ms < args.duration_ms:
        raise SystemExit("--burn-in-ms must lie in [0, duration-ms)")
    if args.eta_m < 0.0:
        raise SystemExit("--eta-m must be non-negative")

    config_path = _resolved_path(args.config)
    config = json.loads(config_path.read_text())
    classifier_contract = validate_classifier_config(config)
    source_audit = validate_sources(config)
    source = config["source"]
    trajectory_path = _resolved_path(source["trajectory_json"])
    source_payload = json.loads(trajectory_path.read_text())
    round_path = _resolved_path(source["round_config"])
    round_config = load_round_config(str(round_path))
    artifact_root = Path(source["repository_artifact_root"]).resolve()
    round_config = absolutize_round_inputs(round_config, artifact_root)
    round_config["simulation"] = dict(round_config["simulation"])
    round_config["simulation"]["duration_ms"] = float(args.duration_ms)

    started = time.time()
    substrate_seed = int(source_payload["seed"])
    with _working_directory(artifact_root):
        substrate = build_substrate(
            round_config, source_payload["candidate_id"], substrate_seed,
            cache_dir=source["network_cache_dir"], ee_dose=1.0, etoi_dose=1.0)
    slow, hybrid_config, geometry = _build_slow(
        substrate, source_payload, q_clamp=args.q_clamp, eta_m=args.eta_m)
    drive = make_external_drive(
        substrate, source_payload["applied_spatial_ou"], int(args.noise_seed))
    if drive is None:
        raise RuntimeError("phase screen requires stationary spatial OU drive")

    checkpoint_path = _resolved_path(source[f"{args.initial_state}_checkpoint"])
    checkpoint = load_checkpoint(checkpoint_path)
    state = prepare_matched_state(
        checkpoint, q_clamp=args.q_clamp, noise_seed=args.noise_seed,
        fresh_drive=drive)
    noise_sha = future_noise_digest(state, update_steps=drive.update_steps)
    initial_state_summary = {
        "absolute_time_ms": float(state["absolute_time_ms"]),
        "step": int(state["step"]),
        "mean_m_E": float(np.mean(state["slow"]["m"][:substrate.n_e])),
        "max_m_E": float(np.max(state["slow"]["m"][:substrate.n_e])),
        "mean_q_grid_after_clamp": float(np.mean(state["slow"]["q_I"])),
        "mean_V_E": float(np.mean(state["V"][:substrate.n_e])),
        "mean_I_E_on_E": float(np.mean(state["I_E"][:substrate.n_e])),
        "mean_I_I_on_E": float(np.mean(state["I_I"][:substrate.n_e])),
    }

    # Resume overwrites this seed state, but assigning it explicitly makes the
    # pre-resume setup deterministic too.
    substrate.net["rng"] = np.random.default_rng(int(args.noise_seed))
    from kick_probe import simulate_kick
    result = simulate_kick(
        substrate.params, substrate.net, KICK_BOOST=0.0, t_kick=1e9,
        V_th_per_neuron=substrate.vtheta, slow=slow,
        early_stop_runaway=False, external_e_rate_drive=drive,
        resume_state=state, time_offset_ms=float(state["absolute_time_ms"]),
        verbose=not args.quiet,
    )
    dt_ms = float(substrate.engine["dt"])
    metrics = stationary_metrics(
        result["rate_E"], result["E_spk_bool"], substrate.positions_e,
        dt_ms=dt_ms, sheet_l_mm=float(substrate.engine["L"]),
        burn_in_ms=float(args.burn_in_ms),
    )
    ceiling_hz = 1000.0 / float(substrate.params.tau_ref_E)
    q_grid = np.asarray(slow.q_I, float)
    stable_checks = {
        "rate_E_finite": bool(np.all(np.isfinite(result["rate_E"]))),
        "rate_I_finite": bool(np.all(np.isfinite(result["rate_I"]))),
        "slow_state_finite": bool(
            np.all(np.isfinite(q_grid)) and np.all(np.isfinite(slow.m))),
        "rate_below_refractory_ceiling": bool(
            float(np.max(result["rate_E"])) <= ceiling_hz + 1e-6),
        "q_remained_exactly_clamped": bool(
            np.allclose(q_grid, float(args.q_clamp), atol=1e-12, rtol=0.0)),
    }
    numerically_stable = bool(all(stable_checks.values()))
    classification = classify_stationary_branch(
        metrics, numerically_stable=numerically_stable)

    burn_steps = int(round(float(args.burn_in_ms) / dt_ms))
    spike_count_e = np.sum(
        np.asarray(result["E_spk_bool"], bool)[burn_steps:], axis=0,
        dtype=np.int64)
    output_path = Path(args.out).resolve()
    npz_path = output_path.with_suffix(".npz")
    slow_trace = slow.trace_arrays()
    _atomic_npz(
        npz_path,
        time_ms=np.asarray(result["times"], np.float32),
        rate_E_hz=np.asarray(result["rate_E"], np.float32),
        rate_I_hz=np.asarray(result["rate_I"], np.float32),
        stationary_spike_count_E=np.asarray(spike_count_e, np.int32),
        **{f"slow__{name}": np.asarray(values)
           for name, values in slow_trace.items()},
    )
    payload = {
        "status": "SPATIAL_ZM_PHASE_POINT_COMPLETE",
        "schema_version": config["schema_version"],
        "scientific_role": config["scientific_role"],
        "classifier_contract": classifier_contract,
        "stage": str(args.stage),
        "coordinates": {
            "q_clamp": float(args.q_clamp),
            "eta_m": float(args.eta_m),
            "noise_seed": int(args.noise_seed),
            "initial_state": args.initial_state,
        },
        "controlled_q_contract": {
            "q_init": float(hybrid_config.q_init),
            "q_min": float(hybrid_config.q_min),
            "freeze_q": bool(hybrid_config.freeze_q),
            "q_is_coordinate_not_a_floor": True,
        },
        "paired_noise_contract": {
            "future_noise_sha256": noise_sha,
            "same_sha_required_for_low_and_high": True,
            "homogeneous_ou_xi_reset": 0.0,
            "spatial_ou_relative_next_step": int(drive.update_steps),
            "spatial_ou_relative_last_step": -1,
        },
        "source_audit": source_audit,
        "phase_config": {
            "path": str(config_path),
            "sha256": _sha256(config_path),
        },
        "source_trajectory": {
            "path": str(trajectory_path),
            "substrate_seed": substrate_seed,
            "candidate_id": source_payload["candidate_id"],
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": source_audit[f"{args.initial_state}_checkpoint"][
                "observed_sha256"],
            "initial_state_summary": initial_state_summary,
        },
        "network_cache": substrate.network_cache,
        "full_edge_contract": {
            "E_to_E_dose": 1.0,
            "E_to_I_dose": 1.0,
            "learned_edges_modified": False,
        },
        "hybrid_config": asdict(hybrid_config),
        "applied_spatial_ou": source_payload["applied_spatial_ou"],
        "geometry": geometry,
        "simulation": {
            "duration_ms": float(args.duration_ms),
            "burn_in_ms": float(args.burn_in_ms),
            "dt_ms": dt_ms,
            "wall_s": float(result["wall_s"]),
            "total_wall_s": float(time.time() - started),
            "external_rate_clipping": result["external_e_rate_drive"],
        },
        "numerical_stability": {
            "all_checks_pass": numerically_stable,
            "checks": stable_checks,
            "max_rate_E_hz": float(np.max(result["rate_E"])),
            "refractory_ceiling_hz": ceiling_hz,
            "final_q_range": [float(np.min(q_grid)), float(np.max(q_grid))],
        },
        "stationary_metrics": metrics,
        "classification": classification,
        "slow_summary": slow.summary(),
        "trajectory_npz": {
            "path": str(npz_path),
            "sha256": _sha256(npz_path),
        },
        "claim_boundary": config["claim_boundary"],
    }
    contract_sha, contract_payload = scientific_contract_digest(payload)
    payload["scientific_contract_sha256"] = contract_sha
    payload["scientific_contract"] = contract_payload
    _atomic_json(payload, output_path)
    print(json.dumps({
        "status": payload["status"],
        "output": str(output_path),
        "label": classification["label"],
        "median_rate_hz": metrics["median_rate_hz"],
        "future_noise_sha256": noise_sha,
        "wall_s": payload["simulation"]["total_wall_s"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
