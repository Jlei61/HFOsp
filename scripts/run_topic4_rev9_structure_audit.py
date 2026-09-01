"""Audit the exploratory rev9 edge transform on full-size frozen networks."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join("src", "snn_engine"))
from scripts.run_topic4_core_field_stage3_fit import _load_cmrun  # noqa: E402
from src.topic4_core_connectivity import field_normalized_ee_pair  # noqa: E402
from src.topic4_core_field_rev9 import reconstruct_frozen_node  # noqa: E402
from src.topic4_core_field_runner import (  # noqa: E402
    _placement,
    atomic_write_json,
    get_network,
    provenance,
)


DEFAULT_CONFIG = "config/topic4_rev9_exploratory.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args, default="unknown"):
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return default


def _candidate(config):
    confirmation = json.loads(Path(config["inputs"]["confirmation"]).read_text())
    matches = [row for row in confirmation["candidates"]
               if row["candidate_id"] == config["inputs"]["candidate_id"]]
    if len(matches) != 1:
        raise RuntimeError("configured frozen candidate is missing or ambiguous")
    row = matches[0]
    if row["theta_sha256"] != config["inputs"]["theta_sha256"]:
        raise RuntimeError("configured theta hash differs from confirmation")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--out")
    parser.add_argument("--cache-dir")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    stage = json.loads(Path(config["inputs"]["stage_config"]).read_text())
    candidate = _candidate(config)
    output_root = Path(config["output_root"])
    output_path = Path(args.out or output_root / "edge_structure_audit.json")
    cache_dir = str(Path(args.cache_dir or output_root / "network_cache"))
    engine = stage["engine"]
    reg = _placement(stage)
    cmrun = _load_cmrun()
    params_cls = __import__("params").Params
    alpha_grid = [float(value) for value in config["edge"]["alpha_grid"]]
    rows = []
    started = time.time()

    for seed in config["edge"]["structure_audit_seeds"]:
        p = params_cls(
            g=engine["g"], L=engine["L"], density=engine["density"],
            T=400.0, dt=engine["dt"], nu_ext_ratio=cmrun.DRIVE,
            seed=int(seed))
        network_started = time.time()
        net, n_e, n_i, cache_hit = get_network(
            p, reg["theta_deg"], engine["AR"], cache_dir)
        node = reconstruct_frozen_node(
            candidate["theta"], net["pos"][:n_e], n_total=n_e + n_i,
            target_count=stage["N_core_manual"],
            quantile_seed=stage["quantile_seed"],
            core_mean=engine["core_mean"], core_std=engine["core_std"],
            v_base=engine["v_base"], K=candidate["K"], L=engine["L"])
        seed_row = dict(
            seed=int(seed), cache_hit=bool(cache_hit), n_E=int(n_e), n_I=int(n_i),
            network_seconds=float(time.time() - network_started),
            node_hashes=node["hashes"], alpha=[])
        for alpha in alpha_grid:
            transform_started = time.time()
            mapped, diagnostics = field_normalized_ee_pair(
                net, node["h"], alpha, beta=0.0,
                active_vth_shift=node["delta_vtheta"])
            diagnostics["transform_seconds"] = float(time.time() - transform_started)
            seed_row["alpha"].append(diagnostics)
            del mapped
            gc.collect()
        rows.append(seed_row)
        del net, node
        gc.collect()

    package_lock = "requirements.txt"
    payload = dict(
        status="REV9_EDGE_STRUCTURE_AUDIT_COMPLETE",
        scientific_role=(
            "full-network zero-integration audit of the exploratory edge family; "
            "diagnostic reference bands are not acceptance gates"),
        mechanism=config["edge"]["mechanism"],
        alpha_grid=alpha_grid,
        beta=0.0,
        hard_contract=dict(
            incoming_E_absolute_tolerance=float(
                config["edge"]["incoming_E_absolute_tolerance"]),
            topology_delay_E_to_I_GABA_unchanged=True,
            finite_positive_nonzero_edges=True),
        diagnostic_reference_bands=config["edge"]["diagnostic_reference_bands"],
        networks=rows,
        wall_seconds=float(time.time() - started),
        inputs=dict(
            config=dict(path=args.config, sha256=_sha256(args.config)),
            stage_config=dict(
                path=config["inputs"]["stage_config"],
                sha256=_sha256(config["inputs"]["stage_config"])),
            confirmation=dict(
                path=config["inputs"]["confirmation"],
                sha256=_sha256(config["inputs"]["confirmation"])),
            candidate_id=candidate["candidate_id"],
            theta_sha256=candidate["theta_sha256"]),
        network_cache=cache_dir,
        provenance=dict(
            **provenance(),
            git_status_porcelain=_git("status", "--porcelain"),
            producer_sha256=_sha256(__file__),
            config_sha256=_sha256(args.config),
            python_executable=sys.executable,
            python_version=platform.python_version(),
            package_lock=dict(path=package_lock, sha256=_sha256(package_lock)),
            systemd_unit=os.environ.get("REV9_SYSTEMD_UNIT"),
        ),
    )
    atomic_write_json(payload, str(output_path))
    print(json.dumps(dict(
        status=payload["status"], seeds=[row["seed"] for row in rows],
        alpha_grid=alpha_grid, wall_seconds=payload["wall_seconds"],
        out=str(output_path)), indent=2), flush=True)


if __name__ == "__main__":
    main()
