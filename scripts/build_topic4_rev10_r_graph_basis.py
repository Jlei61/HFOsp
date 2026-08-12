"""Build one observation-free directed graph basis for Topic 4 rev10-R."""
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
from src.topic4_graph_edge_flow import (  # noqa: E402
    array_sha256,
    build_directed_spectral_basis,
    sample_spectral_edge_features,
    spectral_response_design,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_r_graph_edge_flow.json"


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


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
            "development_only_contact_density_invariant_route_capacity"):
        raise RuntimeError("rev10-R scientific role changed")
    allowed = set(map(int, config["search"]["fit_network_seeds"]))
    if args.seed not in allowed:
        parser.error("basis seed is outside the frozen fit-network set")
    if config["search"]["beta"] != "closed":
        raise RuntimeError("beta must remain closed")
    for record in config["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")

    provenance = _runtime_provenance(args.expected_commit)
    provenance["systemd_unit"] = os.environ.get("REV10R_SYSTEMD_UNIT")
    if (provenance["runtime_modules_dirty"] or _config_dirty(config_path)
            or not provenance["runtime_modules_match_expected_commit"]):
        raise RuntimeError("basis runtime modules or config are not frozen")

    output_root = ROOT / config["output_root"] / "graph_basis"
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
    basis_config = config["graph_basis"]
    basis = build_directed_spectral_basis(
        net["ampa_by_delay"], n_e,
        rank=int(basis_config["rank"]),
        extra_modes=int(basis_config["extra_modes"]),
        random_state=int(basis_config["random_state"]),
        tolerance=float(basis_config["svd_tolerance"]),
    )

    # Float32 is the frozen consumer representation; diagnostics are recomputed
    # from those exact arrays so the later mapper and the sidecar cannot drift.
    frozen_basis = {
        **basis,
        "u": np.asarray(basis["u"], np.float32),
        "v": np.asarray(basis["v"], np.float32),
        "singular_values": np.asarray(basis["singular_values"], np.float64),
    }
    frozen_basis["u_sha256"] = array_sha256(frozen_basis["u"])
    frozen_basis["v_sha256"] = array_sha256(frozen_basis["v"])
    frozen_basis["singular_values_sha256"] = array_sha256(
        frozen_basis["singular_values"]
    )
    sampled = sample_spectral_edge_features(
        net["ampa_by_delay"], frozen_basis,
        sample_limit=int(basis_config["sample_limit"]),
    )
    features = np.asarray(sampled["features"], np.float32)
    feature_abs_max = np.asarray(sampled["feature_abs_max"], np.float64)
    _atomic_npz(
        output_npz,
        u=frozen_basis["u"],
        v=frozen_basis["v"],
        singular_values=frozen_basis["singular_values"],
        feature_sample=features,
        feature_abs_max=feature_abs_max,
        n_e=np.asarray(n_e, np.int64),
        rank=np.asarray(basis_config["rank"], np.int64),
        graph_weight_sha256=np.asarray(frozen_basis["graph_weight_sha256"]),
    )
    payload = {
        "status": "REV10R_GRAPH_BASIS_COMPLETE",
        "scientific_role": config["scientific_role"],
        "seed": int(args.seed),
        "basis": {
            key: _jsonable(value) for key, value in frozen_basis.items()
            if key not in {"u", "v"}
        },
        "feature_audit": {
            "sample_size": int(len(features)),
            "sample_stride": int(sampled["sample_stride"]),
            "feature_abs_max": feature_abs_max.tolist(),
            "sample_abs_p99": np.percentile(
                np.abs(features), 99, axis=0,
            ).tolist(),
            "sample_sha256": array_sha256(features),
            "spectral_response_design_condition_number": float(np.linalg.cond(
                spectral_response_design(
                    frozen_basis["singular_values"],
                    int(frozen_basis["rank"]),
                )
            )),
        },
        "network": {
            "n_E": int(n_e), "n_I": int(n_i),
            "cache_hit": bool(cache_hit), "cache_source": cache_source,
        },
        "observation_exclusion": {
            "builder_inputs": [
                "frozen E-to-E sparse matrices", "n_E", "rank",
                "SVD numerical controls",
            ],
            "forbidden_inputs": basis_config["forbidden_inputs"],
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
        "status": payload["status"], "seed": args.seed,
        "singular_values": payload["basis"]["singular_values"],
        "truncation_gap": payload["basis"][
            "truncation_boundary_relative_gap"
        ],
        "wall_seconds": payload["wall_seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
