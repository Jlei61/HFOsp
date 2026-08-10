"""One-seed zero-integration structural audit of the frozen rev9 edge map."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join("src", "snn_engine"))
from scripts.run_topic4_core_field_stage3_fit import _load_cmrun  # noqa: E402
from src.topic4_core_connectivity import field_normalized_ee_pair  # noqa: E402
from src.topic4_core_field_rev9 import (  # noqa: E402
    component_contributions,
    reconstruct_frozen_node,
)
from src.topic4_core_field_runner import (  # noqa: E402
    _placement,
    atomic_write_json,
    cache_key,
    connectivity_config,
)
from src.topic4_rev9_edge_structure import (  # noqa: E402
    field_background_membership,
    summarize_edge_redistribution,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev9_exploratory.json"
DEFAULT_FACTORIAL = ROOT / "config/topic4_rev9_factorial.json"
DEFAULT_OUTPUT = ROOT / (
    "results/topic4_sef_hfo/data_driven_core_field_rev9/edge_structure_detail/workers")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args):
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True,
        stderr=subprocess.DEVNULL).strip()


def _candidate(config):
    confirmation_path = ROOT / config["inputs"]["confirmation"]
    confirmation = json.loads(confirmation_path.read_text())
    matches = [row for row in confirmation["candidates"]
               if row["candidate_id"] == config["inputs"]["candidate_id"]]
    if len(matches) != 1:
        raise RuntimeError("frozen candidate is missing or ambiguous")
    candidate = matches[0]
    if candidate["theta_sha256"] != config["inputs"]["theta_sha256"]:
        raise RuntimeError("frozen candidate theta hash changed")
    return candidate


def _json_ready(summary):
    output = {}
    for key, value in summary.items():
        if isinstance(value, np.ndarray):
            output[key] = value.tolist()
        elif isinstance(value, np.generic):
            output[key] = value.item()
        else:
            output[key] = value
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--factorial-config", default=str(DEFAULT_FACTORIAL))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--out")
    args = parser.parse_args()

    started = time.time()
    commit_at_start = _git("rev-parse", "HEAD")
    producer_sha_at_start = _sha256(__file__)
    relevant_paths = [
        "scripts/run_topic4_rev9_edge_structure_detail_worker.py",
        "src/topic4_rev9_edge_structure.py", "src/topic4_core_connectivity.py",
        "src/topic4_core_field_rev9.py", "src/topic4_core_field_runner.py",
    ]
    relevant_dirty = bool(_git("status", "--porcelain", "--", *relevant_paths))
    if relevant_dirty:
        raise RuntimeError("edge structure worker numeric dependencies are dirty")

    config_path = Path(args.config).resolve()
    factorial_path = Path(args.factorial_config).resolve()
    config = json.loads(config_path.read_text())
    factorial = json.loads(factorial_path.read_text())
    if int(args.seed) not in [int(value) for value in factorial["seeds"]]:
        raise ValueError("seed is outside the frozen factorial set")
    stage_path = ROOT / config["inputs"]["stage_config"]
    stage = json.loads(stage_path.read_text())
    candidate = _candidate(config)
    worker_path = ROOT / factorial["output_root"] / "workers" / f"node_seed{args.seed}.json"
    factorial_worker = json.loads(worker_path.read_text())
    source_commit = factorial_worker["provenance"]["git_commit"]
    if not np.isclose(float(factorial_worker["alpha"]), float(args.alpha)):
        raise RuntimeError("requested alpha differs from frozen factorial alpha")

    engine = stage["engine"]
    reg = _placement(stage)
    cmrun = _load_cmrun()
    params_cls = __import__("params").Params
    p = params_cls(
        g=engine["g"], L=engine["L"], density=engine["density"],
        T=400.0, dt=engine["dt"], nu_ext_ratio=cmrun.DRIVE,
        seed=int(args.seed))
    cache_config = connectivity_config(p, reg["theta_deg"], engine["AR"])
    cache_config["git_commit"] = source_commit
    cache_path = ROOT / config["output_root"] / "network_cache" / (
        cache_key(cache_config) + ".pkl")
    if not cache_path.exists():
        raise FileNotFoundError(f"frozen factorial network cache is missing: {cache_path}")
    cache_sha256 = _sha256(cache_path)
    with cache_path.open("rb") as stream:
        wrapper = pickle.load(stream)
    if wrapper["config"] != cache_config:
        raise RuntimeError("network cache config differs from reconstructed factorial config")
    net, n_e, n_i = wrapper["net"], int(wrapper["NE"]), int(wrapper["NI"])
    node = reconstruct_frozen_node(
        candidate["theta"], net["pos"][:n_e], n_total=n_e + n_i,
        target_count=stage["N_core_manual"], quantile_seed=stage["quantile_seed"],
        core_mean=engine["core_mean"], core_std=engine["core_std"],
        v_base=engine["v_base"], K=candidate["K"], L=engine["L"])
    expected_h = factorial_worker["network"]["node_hashes"]["h_vector_sha256"]
    if node["hashes"]["h_vector_sha256"] != expected_h:
        raise RuntimeError("reconstructed h differs from the frozen factorial worker")
    contributions = component_contributions(
        candidate["theta"], net["pos"][:n_e], K=candidate["K"], L=engine["L"])
    partition = field_background_membership(node["h"], contributions)
    mapped, mapper = field_normalized_ee_pair(
        net, node["h"], float(args.alpha), beta=0.0,
        active_vth_shift=node["delta_vtheta"])
    summary = summarize_edge_redistribution(
        net["ampa_by_delay"], mapped["ampa_by_delay"], partition["membership"],
        delay_dt_ms=float(cache_config["delay_dt"]), h=node["h"])
    if summary["incoming_max_abs_error"] > 1e-9:
        raise RuntimeError("detailed structure audit violates incoming-E conservation")

    out_path = Path(args.out or DEFAULT_OUTPUT / f"seed{args.seed}.json")
    payload = dict(
        status="REV9_EDGE_STRUCTURE_DETAIL_WORKER_COMPLETE",
        scientific_role=(
            "zero-integration structural interpretation of the frozen alpha; "
            "row=target and column=source"),
        seed=int(args.seed), alpha=float(args.alpha), beta=0.0,
        labels=partition["labels"], direction_contract="row_target_column_source",
        membership_contract=(
            "component_c=h*raw_component_c/sum_raw_components; background=1-h"),
        summary=_json_ready(summary), mapper_diagnostics=mapper,
        network=dict(
            n_E=n_e, n_I=n_i, cache_path=str(cache_path.relative_to(ROOT)),
            cache_sha256=cache_sha256, source_commit=source_commit,
            h_sha256=node["hashes"]["h_vector_sha256"]),
        inputs=dict(
            config={"path": str(config_path), "sha256": _sha256(config_path)},
            factorial_config={"path": str(factorial_path), "sha256": _sha256(factorial_path)},
            factorial_worker={"path": str(worker_path), "sha256": _sha256(worker_path)},
            stage_config={"path": str(stage_path), "sha256": _sha256(stage_path)}),
        provenance=dict(
            git_commit_at_start=commit_at_start,
            producer_sha256_at_start=producer_sha_at_start,
            module_sha256_at_start={path: _sha256(ROOT / path) for path in relevant_paths},
            relevant_modules_dirty_at_start=relevant_dirty,
            systemd_unit=os.environ.get("REV9_SYSTEMD_UNIT"),
            wall_seconds=float(time.time() - started)),
    )
    atomic_write_json(payload, str(out_path))
    print(json.dumps({
        "status": payload["status"], "seed": args.seed,
        "out": str(out_path), "wall_seconds": payload["provenance"]["wall_seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
