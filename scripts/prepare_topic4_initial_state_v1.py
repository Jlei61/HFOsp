#!/usr/bin/env python3
"""Freeze the initial-state v1 round: sources, membership, arms and job manifest.

Runs no simulation. Writes results/.../initial_state_conditioned_propagation_v1/
{frozen_manifest.json, initial_state_arrays/, jobs.json, seed_namespace_check.json}.
"""
from __future__ import annotations

import argparse
import dataclasses
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import topic4_initial_state_runtime as rt  # noqa: E402
from src.topic4_initial_state import (  # noqa: E402
    ARMS, PERTURBED_CORE, array_sha256, build_initial_voltage, check_arm_symmetry,
    core_membership, initial_voltage_summary, validate_initial_voltage,
)

SOURCE_MODULES = [
    "src/snn_engine/kick_probe.py", "src/snn_engine/params.py", "src/snn_engine/model.py",
    "src/snn_engine/connectivity.py", "src/snn_engine/connectivity_rot.py",
    "src/snn_engine/checkpoint.py",
    "src/topic4_initial_state.py", "src/topic4_initial_state_runtime.py",
    "src/topic4_zm_ictal_transition.py", "src/topic4_multidimensional_parameters.py",
    "src/topic4_manual_dual_core.py", "src/topic4_core_field_rev9.py",
    "src/topic4_core_field_runner.py", "src/topic4_rev20_dual_core_mechanism.py",
    "src/topic4_local_connectivity.py", "src/topic4_spatial_ou_drive.py",
    "src/topic4_xy_search.py", "src/topic4_observation_repaired.py",
    "src/topic4_interictal_repaired_evaluation.py", "src/topic4_joint_xy_kernel.py",
    "src/topic4_multievent_distribution_objective_v2_1.py",
    "src/topic4_multievent_distribution_objective.py",
    "src/topic4_node_dualmode.py", "src/sef_hfo_snn_adapter.py", "src/sef_hfo_observation.py",
    "src/sef_hfo_events.py", "src/lagpat_rank_audit.py",
    "scripts/run_topic4_initial_state_worker.py", "scripts/prepare_topic4_initial_state_v1.py",
    "scripts/run_sef_hfo_snn_cm_spontaneous_readout.py",
    "scripts/run_topic4_rev10_sa_spectral_field_worker.py",
    "scripts/run_topic4_core_field_stage3_fit.py",
    "scripts/run_topic4_rev9_node_kick_canary.py",
    "config/topic4_initial_state_conditioned_propagation_v1.json",
    "config/topic4_rev22_dci_transition_execution.json",
]


def seed_namespace_check(design):
    """X4: named seed fields in Topic 4 configs and current manifests must not reuse ours."""
    seeds = set(design["stages"]["screen"]["dynamics_seeds"]) | set(
        design["stages"]["replication"]["dynamics_seeds"]) | {
        design["stages"]["replication"]["topology_seed"]}
    pattern = re.compile(r'"[A-Za-z_]*seeds?"\s*:\s*(\[[^\]]*\]|\d+)')
    roots = [
        ROOT / "config", rt.ARTIFACT_ROOT / "config",
        Path(design["source_root"]) / "config",
        Path(design["source_root"]) / "results/topic4_sef_hfo",
        rt.ARTIFACT_ROOT / "results/topic4_sef_hfo",
    ]
    collisions, n_files = [], 0
    own = Path(design["source_root"]) / "config/topic4_initial_state_conditioned_propagation_v1.json"
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            if path.name == own.name or "initial_state_conditioned_propagation" in str(path):
                continue
            try:
                if path.stat().st_size > 50 * 1024 * 1024:
                    continue
                text = path.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            n_files += 1
            for match in pattern.finditer(text):
                values = {int(v) for v in re.findall(r"\d+", match.group(1))}
                hit = sorted(values & seeds)
                if hit:
                    collisions.append({"path": str(path), "seeds": hit, "field": match.group(0)[:80]})
    return {"seeds_checked": sorted(seeds), "n_files_scanned": n_files,
            "collisions": collisions, "scope": [str(r) for r in roots],
            "rule": "named seed fields only; a collision would require one whole-round reassignment before any result"}


def frozen_membership(design):
    """M1-M6 from the full-precision corrected graph and the frozen node mapping."""
    from src.topic4_core_field_rev9 import reconstruct_node_from_h
    from src.topic4_manual_dual_core import budget_matched_dual_core_h
    candidate = rt.candidate_record(design)
    record = rt.network_record(design, design["stages"]["screen"]["topology_seed"])
    with open(record["path"], "rb") as stream:
        payload = pickle.load(stream)
    n_e, n_i = int(payload["NE"]), int(payload["NI"])
    positions = np.asarray(payload["net"]["pos"][:n_e], np.float64)
    del payload
    centers = np.asarray(candidate["node_field"]["centers_mm"], np.float64)
    h, field_audit = budget_matched_dual_core_h(
        positions, centers, target_count=int(candidate["node_field"]["target_count"]))
    if field_audit["h_sha256"] != candidate["geometry"]["h_sha256"]:
        raise RuntimeError("actual h field differs from the inherited candidate geometry (M5)")
    transition, _ = rt.transition_config(design)
    stage = rt.read(rt.ARTIFACT_ROOT / transition["inputs"]["stage_config"]["path"])
    engine = stage["engine"]
    node = reconstruct_node_from_h(
        h, n_total=n_e + n_i, quantile_seed=stage["quantile_seed"],
        core_mean=engine["core_mean"], core_std=engine["core_std"], v_base=engine["v_base"])
    membership = core_membership(positions, h, centers)
    return positions, h, node["vtheta"], membership, field_audit, n_e, n_i


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, default=rt.DESIGN_PATH)
    args = parser.parse_args()
    design = rt.load_design(args.design)
    out = rt.output_root(design)
    out.mkdir(parents=True, exist_ok=True)
    arrays_dir = out / "initial_state_arrays"
    arrays_dir.mkdir(exist_ok=True)

    source_audit = rt.verify_design_sources(design)
    stale = [k for k, v in source_audit.items() if not v["match"]]
    engine_patched = stale == ["engine"] or stale == []
    if not engine_patched:
        raise RuntimeError(f"frozen design sources changed: {stale}")

    positions, h, vtheta, membership, field_audit, n_e, n_i = frozen_membership(design)
    init = design["initialization"]
    v_reset = float(init["default_V_mV"])
    from params import Params
    if Params().V_reset != v_reset:
        raise RuntimeError("design V_reset differs from the engine default")
    increment = float(init["increment_mV"])
    margin = float(init["required_threshold_margin_mV"])
    arms = {}
    for arm in ARMS:
        voltage = build_initial_voltage(n_e + n_i, v_reset, membership, increment, arm)
        audit = validate_initial_voltage(voltage, vtheta, v_reset, margin)
        summary = initial_voltage_summary(voltage, v_reset)
        np.save(arrays_dir / f"initial_voltage_{arm}.npy", voltage)
        arms[arm] = {
            "initial_state_id": f"{arm}_" + ("reset" if PERTURBED_CORE[arm] is None
                                             else f"core{PERTURBED_CORE[arm]}_plus{increment:g}mV"),
            "perturbed_core_index": PERTURBED_CORE[arm],
            "file": str(arrays_dir / f"initial_voltage_{arm}.npy"),
            "file_sha256": rt.sha(arrays_dir / f"initial_voltage_{arm}.npy"),
            "threshold_margin": audit, "summary": summary,
        }
    check_arm_symmetry(arms["B1"]["summary"], arms["B2"]["summary"])
    rt.atomic_npz(arrays_dir / "membership.npz",
                  S_A=membership["S_A"], S_B=membership["S_B"],
                  distance_A_mm=membership["distance_A_mm"],
                  distance_B_mm=membership["distance_B_mm"],
                  h=h, vtheta=vtheta, positions_E=positions,
                  centers_mm=np.asarray(membership["centers_mm"]))
    membership_summary = {k: v for k, v in membership.items()
                          if not isinstance(v, np.ndarray)}
    membership_summary.update({
        "membership_file": str(arrays_dir / "membership.npz"),
        "membership_file_sha256": rt.sha(arrays_dir / "membership.npz"),
        "h_sha256_float64": field_audit["h_sha256"],
        "vtheta_sha256_float64": array_sha256(np.asarray(vtheta, np.float64)),
        "vtheta_sha256_float32": array_sha256(np.asarray(vtheta, np.float32)),
        "positions_sha256_float64": array_sha256(np.asarray(positions, np.float64)),
        "field_audit": field_audit, "n_E": n_e, "n_I": n_i,
        "precision": "float64 runtime arrays from the corrected graph pickle",
    })

    # ---- job manifest (X2, X3, X5) ----
    jobs = []
    stages = design["stages"]
    for stage_name in ("screen", "replication"):
        stage = stages[stage_name]
        for dynamics_seed in stage["dynamics_seeds"]:
            for arm in ARMS:
                stem = (f"{design['candidate_id']}_topo_{stage['topology_seed']}"
                        f"_dyn_{dynamics_seed}_{arm}")
                jobs.append({
                    "stage": stage_name, "candidate_id": design["candidate_id"],
                    "topology_seed": int(stage["topology_seed"]),
                    "dynamics_seed": int(dynamics_seed), "arm": arm,
                    "initial_state_id": arms[arm]["initial_state_id"],
                    "initial_voltage_sha256": arms[arm]["summary"]["sha256"],
                    "membership_sha256": [membership["S_A_sha256"], membership["S_B_sha256"]],
                    "duration_ms": float(design["simulation"]["duration_ms"]),
                    "observer_contract_sha256": design["sources"]["observation_contract"]["sha256"],
                    "evaluator_sha256": design["sources"]["evaluator"]["sha256"],
                    "stem": stem,
                    "output_json": str(out / stage_name / "workers" / f"{stem}.json"),
                    "output_npz": str(out / stage_name / "workers" / f"{stem}.npz"),
                    "formal_canary": bool(stage_name == "screen"
                                          and dynamics_seed == stage["dynamics_seeds"][0]),
                    "gated_on": None if stage_name == "screen" else "screen_primary_persistent_effect_pass",
                })
    seed_check = seed_namespace_check(design)
    if seed_check["collisions"]:
        raise RuntimeError(f"seed collisions found: {seed_check['collisions']}")
    rt.write(out / "seed_namespace_check.json", seed_check)

    # import the worker so its dependency closure is part of the source snapshot
    import importlib
    importlib.import_module("scripts.run_topic4_initial_state_worker")
    loaded = rt.loaded_source_hashes()
    source_hashes = {p: rt.sha(ROOT / p) for p in SOURCE_MODULES if (ROOT / p).exists()}
    source_hashes.update(loaded)
    engine_diff = subprocess.check_output(
        ["git", "diff", "HEAD", "--", "src/snn_engine/kick_probe.py"], cwd=ROOT, text=True)
    manifest = {
        "status": "FROZEN_BEFORE_SIMULATION",
        "created_unix": time.time(),
        "design_path": str(args.design), "design_sha256": rt.sha(args.design),
        "output_root": str(out),
        "snapshot_worktree": str(ROOT), "git_commit": rt.git_commit(),
        "git_dirty_paths": rt.git_dirty_paths(),
        "source_root_of_design": design["source_root"],
        "source_hashes": source_hashes,
        "design_source_audit": source_audit,
        "engine_modified_after_design": {
            "changed_sources": stale,
            "note": ("kick_probe.py gained the off-by-default initial_voltage / "
                     "step_observer interface after the design was written; the "
                     "None path is byte-identical (tests/test_topic4_initial_state.py, "
                     "tests/test_kick_probe_zm_itx_parity.py)"),
            "diff_sha256": rt.sha_bytes(engine_diff.encode()),
        },
        "candidate": rt.candidate_record(design),
        "candidate_canonical_json_sha256": design["candidate_canonical_json_sha256"],
        "screen_network": rt.network_record(design, stages["screen"]["topology_seed"]),
        "replication_networks": {},
        "membership": membership_summary,
        "arms": arms,
        "initialization": init,
        "simulation": design["simulation"],
        "statistics": design["statistics"],
        "python": rt.PYTHON, "numpy_version": np.__version__,
        "jobs_path": str(out / "jobs.json"),
        "n_jobs_screen": sum(j["stage"] == "screen" for j in jobs),
        "n_jobs_replication": sum(j["stage"] == "replication" for j in jobs),
    }
    (out / "engine_patch.diff").write_text(engine_diff)
    rt.write(out / "jobs.json", {"jobs": jobs, "created_unix": time.time()})
    rt.write(out / "frozen_manifest.json", manifest)
    print({"status": manifest["status"], "K": membership["K"],
           "n_jobs": len(jobs), "seed_collisions": len(seed_check["collisions"]),
           "margin_min_perturbed_mV": arms["B1"]["threshold_margin"]["minimum_margin_perturbed_mV"]})


if __name__ == "__main__":
    main()
