"""Audit one real network's observation-free spatial edge feature capacity."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join("src", "snn_engine"))
from scripts.run_topic4_core_field_stage3_fit import _load_cmrun  # noqa: E402
from scripts.run_topic4_rev9_node_kick_canary import _load_network  # noqa: E402
from scripts.run_topic4_rev9l_forced_source_worker import (  # noqa: E402
    _atomic_npz,
    _load_json_input,
    _runtime_provenance,
    _sha256,
)
from src.topic4_core_field_runner import _placement, atomic_write_json  # noqa: E402
from src.topic4_spatial_edge_flow import (  # noqa: E402
    FEATURE_NAMES,
    array_sha256,
    sample_spatial_edge_features,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_r2_spatial_edge_flow.json"


def _config_dirty(path):
    return bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--out-json")
    parser.add_argument("--out-npz")
    parser.add_argument("--cache-dir")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    if config["scientific_role"] != (
            "development_only_observation_invariant_spatial_route_capacity"):
        raise RuntimeError("rev10-R2 scientific role changed")
    if args.seed not in set(map(int, config["search"]["fit_network_seeds"])):
        parser.error("audit seed is outside the frozen fit-network set")
    for record in config["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")
    provenance = _runtime_provenance(args.expected_commit)
    provenance["systemd_unit"] = os.environ.get("REV10R2_SYSTEMD_UNIT")
    if (provenance["runtime_modules_dirty"] or _config_dirty(config_path)
            or not provenance["runtime_modules_match_expected_commit"]):
        raise RuntimeError("spatial audit runtime is not frozen")

    output_root = ROOT / config["output_root"] / "feature_audit"
    output_json = Path(args.out_json or output_root / f"seed_{args.seed}.json")
    output_npz = Path(args.out_npz or output_root / f"seed_{args.seed}.npz")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = str(Path(
        args.cache_dir or ROOT /
        "results/topic4_sef_hfo/data_driven_core_field_rev9/network_cache"
    ))
    started = time.time()
    base = _load_json_input(config["inputs"]["rev9_base_config"])
    stage = _load_json_input(config["inputs"]["stage_config"])
    engine = stage["engine"]
    cmrun = _load_cmrun()
    cmrun.DT = float(engine["dt"])
    cmrun.KDIR = int(engine["k_dir"])
    cmrun.PART_MIN = 2 * int(engine["k_dir"]) + 1
    params_cls = __import__("params").Params
    params = params_cls(
        g=engine["g"], L=engine["L"], density=engine["density"],
        T=1.0, dt=engine["dt"], nu_ext_ratio=cmrun.DRIVE,
        seed=int(args.seed),
    )
    net, n_e, n_i, cache_hit, cache_source = _load_network(
        params, stage, _placement(stage), int(args.seed), base, cache_dir,
    )
    positions = np.asarray(net["pos"][:n_e], float)
    basis = config["spatial_edge_basis"]
    sampled = sample_spatial_edge_features(
        net["ampa_by_delay"], positions,
        L=float(basis["sheet_L_mm"]),
        length_scale=float(basis["displacement_length_scale_mm"]),
        sample_limit=int(basis["sample_limit"]),
    )
    n = int(sampled["n_ee_delay_entries"])
    mean = sampled["feature_sum"] / n
    covariance = sampled["feature_gram"] / n - np.outer(mean, mean)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    relative = eigenvalues / eigenvalues[0]
    effective_rank = int(np.sum(
        relative >= float(basis["minimum_effective_rank_relative_eigenvalue"])
    ))
    positive = eigenvalues[eigenvalues > eigenvalues[0] * 1e-12]
    condition = float(positive[0] / positive[-1])
    status = (
        "REV10R2_SPATIAL_FEATURE_CAPACITY_PASS"
        if effective_rank >= int(basis["minimum_effective_rank"])
        and condition <= float(basis["maximum_covariance_condition_number"])
        else "REV10R2_SPATIAL_FEATURE_CAPACITY_FAIL"
    )
    features = np.asarray(sampled["features"], np.float32)
    _atomic_npz(
        output_npz,
        feature_sample=features,
        feature_abs_max=np.asarray(sampled["feature_abs_max"], np.float64),
        feature_sum=np.asarray(sampled["feature_sum"], np.float64),
        feature_gram=np.asarray(sampled["feature_gram"], np.float64),
        covariance=covariance.astype(np.float64),
        covariance_eigenvalues=eigenvalues.astype(np.float64),
        feature_names=np.asarray(FEATURE_NAMES),
        n_ee_delay_entries=np.asarray(n, np.int64),
    )
    payload = {
        "status": status,
        "worker_status": "REV10R2_SPATIAL_EDGE_AUDIT_COMPLETE",
        "scientific_role": config["scientific_role"],
        "seed": int(args.seed),
        "feature_audit": {
            "feature_names": FEATURE_NAMES,
            "n_features": len(FEATURE_NAMES),
            "n_ee_delay_entries": n,
            "sample_size": int(len(features)),
            "sample_stride": int(sampled["sample_stride"]),
            "feature_abs_max": sampled["feature_abs_max"].tolist(),
            "covariance_eigenvalues": eigenvalues.tolist(),
            "relative_covariance_eigenvalues": relative.tolist(),
            "effective_rank": effective_rank,
            "covariance_condition_number": condition,
            "sample_sha256": array_sha256(features),
        },
        "network": {
            "n_E": int(n_e), "n_I": int(n_i),
            "cache_hit": bool(cache_hit), "cache_source": cache_source,
        },
        "observation_exclusion": {
            "builder_inputs": [
                "frozen E-to-E sparse matrices", "neuron sheet positions",
                "fixed sheet size", "fixed E-to-E length scale",
            ],
            "forbidden_inputs": basis["forbidden_inputs"],
            "forbidden_input_loaded_by_builder": False,
        },
        "arrays": {"path": str(output_npz), "sha256": _sha256(output_npz)},
        "config": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": _sha256(config_path),
        },
        "wall_seconds": float(time.time() - started),
        "provenance": provenance,
    }
    atomic_write_json(payload, output_json)
    print(json.dumps({
        "status": status, "seed": args.seed,
        "effective_rank": effective_rank,
        "condition_number": condition,
        "wall_seconds": payload["wall_seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
