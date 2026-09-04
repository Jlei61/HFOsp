#!/usr/bin/env python3
"""Rebuild one frozen rev21 trajectory and save exact Fig.5 state checkpoints."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src" / "snn_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.run_topic4_rev9l_forced_source_worker import _atomic_npz  # noqa: E402
from src.snn_engine import checkpoint as ckpt  # noqa: E402
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_zm_ictal_transition import (  # noqa: E402
    build_substrate,
    load_round_config,
    make_external_drive,
    make_slow,
)
import src.topic4_manual_dual_core  # noqa: E402,F401
import src.topic4_rev20_dual_core_mechanism  # noqa: E402,F401


RUNTIME_MODULES = (
    "scripts/run_topic4_rev12_node_worker.py",
    "src/snn_engine/kick_probe.py",
    "src/topic4_zm_ictal_transition.py",
    "src/topic4_rev21_zm_transition.py",
    "src/topic4_manual_dual_core.py",
    "src/topic4_rev20_dual_core_mechanism.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(artifact_root: Path, relative: str) -> Path:
    local = ROOT / relative
    return local if local.exists() else artifact_root / relative


def load_selected_contract(config_path: Path, manifest_path: Path,
                           source_json: Path, artifact_root: Path):
    config = json.loads(config_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    source = json.loads(source_json.read_text())
    candidate_id = str(source["candidate_id"])
    matches = [row for row in manifest["candidates"]
               if row["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise RuntimeError("source candidate is missing or duplicated in manifest")
    candidate = matches[0]
    expected_npz = Path(source["arrays"]["path"])
    if not expected_npz.is_absolute():
        expected_npz = artifact_root / expected_npz
    if sha256(expected_npz) != source["arrays"]["sha256"]:
        raise RuntimeError("source worker array hash changed")
    if source.get("status") != "REV12ND_NODE_WORKER_COMPLETE":
        raise RuntimeError("source rev21 worker is incomplete")
    if source["model_ictal_rev21"]["status"] != "MODEL_ICTAL_ELIGIBLE_REV21":
        raise RuntimeError("selected trajectory is not the frozen eligible rev21 example")
    recorded = source["provenance"]["runtime_module_sha256"]
    mismatch = {}
    for relative in RUNTIME_MODULES:
        live = sha256(ROOT / relative)
        if recorded.get(relative) != live:
            mismatch[relative] = {"recorded": recorded.get(relative), "live": live}
    if mismatch:
        raise RuntimeError(f"runtime drift prevents exact checkpoint replay: {mismatch}")
    return config, candidate, source, expected_npz


def transition_config(config: dict, candidate: dict, artifact_root: Path) -> dict:
    path = resolve(artifact_root, config["inputs"]["transition_config"]["path"])
    transition = load_round_config(path)
    transition = copy.deepcopy(transition)
    transition["zm"] = copy.deepcopy(candidate["slow_variables"])
    transition["simulation"] = copy.deepcopy(config["search"]["simulation"])
    return transition


def build_selected_substrate(config: dict, candidate: dict, *, topology_seed: int,
                             dynamics_seed: int, artifact_root: Path):
    transition = transition_config(config, candidate, artifact_root)
    mapping = candidate.get("node_mapping", {})
    mechanisms = candidate.get("mechanisms", {})
    base_id = str(config["reference"]["base_substrate_candidate_id"])
    cache_dir = artifact_root / config["network_cache"]
    substrate = build_substrate(
        transition, base_id, int(topology_seed), cache_dir=str(cache_dir),
        ee_dose=float(mechanisms["g_EE"]),
        etoi_dose=float(mechanisms["g_EtoI"]),
        node_candidate_override=candidate["node_field"],
        node_depth_shrinkage=float(mapping.get("signed_depth_shrinkage", 1.0)),
        node_gain=float(mapping.get("node_gain", 1.0)),
        node_dispersion_candidate_override=candidate.get("node_dispersion_field"),
        ee_ellipse_angle_deg=float(mechanisms["ellipse_angle_deg"]),
        ee_ellipse_aspect_ratio=float(mechanisms["ellipse_aspect_ratio"]),
        artifact_root=artifact_root,
    )
    substrate.net["rng"] = np.random.default_rng(int(dynamics_seed))
    return transition, substrate


def main() -> None:
    parser = argparse.ArgumentParser()
    base = Path("/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/"
                "data_driven_dual_core_zm_transition")
    output = Path("/data/hfosp_topic4_fig45_artifacts/fig5/"
                  "data_driven_dual_core_spatial_z/perturbation")
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "config/topic4_rev21_dual_core_zm_transition.json")
    parser.add_argument(
        "--candidate-manifest", type=Path,
        default=base / "timescale/candidate_manifest.json")
    parser.add_argument(
        "--source-json", type=Path,
        default=base / "timescale/workers/"
        "rev21_ts_tz3000_ta500_topology_2542_dynamics_2642.json")
    parser.add_argument("--artifact-root", type=Path,
                        default=Path("/home/honglab/leijiaxin/HFOsp"))
    parser.add_argument("--out-dir", type=Path, default=output / "checkpoints")
    args = parser.parse_args()
    started = time.time()
    artifact_root = args.artifact_root.resolve()
    config, candidate, source, source_npz = load_selected_contract(
        args.config.resolve(), args.candidate_manifest.resolve(),
        args.source_json.resolve(), artifact_root)
    topology_seed = int(source["topology_seed"])
    dynamics_seed = int(source["dynamics_seed"])
    transition, substrate = build_selected_substrate(
        config, candidate, topology_seed=topology_seed,
        dynamics_seed=dynamics_seed, artifact_root=artifact_root)
    landmarks = source["model_ictal_rev21"]["landmarks"]
    low_ms = float(landmarks["t_base_ms"][1])
    early_ms = float(landmarks["w_early_ms"][0])
    dt_ms = float(substrate.engine["dt"])
    checkpoint_steps = [int(round(value / dt_ms))
                        for value in (low_ms, early_ms)]
    if not all(np.isclose(step * dt_ms, value, atol=1e-9)
               for step, value in zip(checkpoint_steps, (low_ms, early_ms))):
        raise RuntimeError("state times do not lie on the integration grid")

    from kick_probe import simulate_kick
    from lfp import LFPRecorder

    substrate.params.T = early_ms + dt_ms
    slow = make_slow(substrate, candidate["slow_variables"],
                     trace_weights_E=substrate.h_e)
    slow.enable_field_frames(max(1, int(round(20.0 / dt_ms))))
    drive = make_external_drive(substrate, transition["spatial_ou"], dynamics_seed)
    recorder = LFPRecorder(
        substrate.params, substrate.net["pos"], substrate.net["labels"],
        sites=substrate.contact_xy,
    )
    captured = {}
    simulation = config["search"]["simulation"]
    result = simulate_kick(
        substrate.params, substrate.net, KICK_BOOST=0.0, t_kick=1e9,
        V_th_per_neuron=substrate.vtheta, slow=slow, lfp_recorder=recorder,
        early_stop_runaway=bool(simulation["early_stop_runaway"]),
        es_thresh_hz=float(simulation["es_thresh_hz"]),
        es_dur_ms=float(simulation["es_dur_ms"]),
        post_runaway_record_ms=float(simulation["post_runaway_record_ms"]),
        external_e_rate_drive=drive,
        checkpoint_steps=checkpoint_steps,
        checkpoint_sink=lambda step, state: captured.setdefault(int(step), state),
    )
    if set(captured) != set(checkpoint_steps):
        raise RuntimeError("trajectory did not capture both declared checkpoints")
    with np.load(source_npz, allow_pickle=False) as archive:
        reference_rate = np.asarray(archive["transition_rate_E_hz_raw"], np.float32)
    replay_rate = np.asarray(result["rate_E"], np.float32)
    exact = bool(np.array_equal(replay_rate, reference_rate[:len(replay_rate)]))
    if not exact:
        mismatch = int(np.flatnonzero(replay_rate != reference_rate[:len(replay_rate)])[0])
        raise RuntimeError(f"checkpoint replay diverged from source at step {mismatch}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels = ("low_activity", "early_ictal")
    checkpoints = {}
    for label, step in zip(labels, checkpoint_steps):
        path = args.out_dir / f"dualcore_rev21_{label}.npz"
        checkpoints[label] = {
            "path": str(path.resolve()),
            "sha256": ckpt.save(captured[step], path),
            "time_ms": float(captured[step]["absolute_time_ms"]),
        }
    replay_path = args.out_dir / "dualcore_rev21_checkpoint_replay.npz"
    _atomic_npz(
        replay_path,
        rate_E_hz=replay_rate,
        reference_rate_E_hz=reference_rate[:len(replay_rate)],
    )
    payload = {
        "status": "REV21_FIG5_EXACT_STATE_CHECKPOINTS_COMPLETE",
        "candidate_id": source["candidate_id"],
        "substrate": "dualcore_s39 + Joint=1.25",
        "topology_seed": topology_seed,
        "dynamics_seed": dynamics_seed,
        "source_worker": {
            "json": str(args.source_json.resolve()),
            "json_sha256": sha256(args.source_json.resolve()),
            "npz": str(source_npz),
            "npz_sha256": sha256(source_npz),
        },
        "state_contract": {
            "low_activity": "end of the frozen 500-1000 ms baseline window",
            "early_ictal": (
                "start of the frozen early-ictal window; equals operational "
                "detector time and scientific onset +100 ms"
            ),
        },
        "checkpoints": checkpoints,
        "replay_prefix_exact": exact,
        "replay_steps": int(len(replay_rate)),
        "replay_npz": {"path": str(replay_path.resolve()),
                       "sha256": sha256(replay_path)},
        "runtime_module_sha256": {
            relative: sha256(ROOT / relative) for relative in RUNTIME_MODULES},
        "wall_seconds": float(time.time() - started),
    }
    atomic_write_json(payload, str(args.out_dir / "checkpoint_manifest.json"))
    print(json.dumps({
        "status": payload["status"], "checkpoints": checkpoints,
        "replay_prefix_exact": exact, "wall_seconds": payload["wall_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
