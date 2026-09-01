"""Freeze paired local/global E->E short-term-depression candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.topic4_graph_edge_flow import array_sha256  # noqa: E402

DEFAULT_CONFIG = ROOT / "config/topic4_rev10_d3_dynamic_ee_std_canary.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(fd)
    try:
        Path(temporary).write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def candidate_library(config):
    zero = np.zeros(int(config["spatial_edge_basis"]["coefficient_count"]), float)
    base = {
        "coefficients": zero.tolist(),
        "coefficients_sha256": array_sha256(zero),
    }
    rows = [{
        "candidate_id": "edge_noop",
        **base,
        "ee_std": {"mode": "off", "u": 0.0, "tau_ms": 0.0},
    }]
    library = config["ee_std_library"]
    for mode in library["modes"]:
        for u in library["u"]:
            for tau_ms in library["tau_ms"]:
                rows.append({
                    "candidate_id": (
                        f"eestd_{mode}_u{int(round(1000 * u)):03d}"
                        f"_tau{int(round(tau_ms)):04d}"
                    ),
                    **base,
                    "ee_std": {
                        "mode": mode,
                        "u": float(u),
                        "tau_ms": float(tau_ms),
                    },
                })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    if config["scientific_role"] != "development_only_dynamic_ee_std_accessibility_canary":
        raise RuntimeError("rev10-D3 scientific role changed")
    for record in config["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")
    source_record = config["inputs"]["frozen_direction_classifier_manifest"]
    source = json.loads((ROOT / source_record["path"]).read_text())
    candidates = candidate_library(config)
    if len(candidates) != 9 or any(np.any(row["coefficients"]) for row in candidates):
        raise RuntimeError("rev10-D3 candidate library changed")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    payload = {
        "status": "REV10D3_DYNAMIC_EE_STD_LIBRARY_FROZEN",
        "scientific_role": config["scientific_role"],
        "config": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": _sha256(config_path),
        },
        "candidate_set": {
            "n_candidates": len(candidates),
            "candidates": candidates,
            "frozen_before_network_seeds": config["search"]["fit_network_seeds"],
        },
        "direction_classifier": source["direction_classifier"],
        "direction_classifier_source": {
            "path": source_record["path"],
            "sha256": source_record["sha256"],
            "copied_without_refit": True,
        },
        "static_edge_contract": "all 12 coefficients are exact zero",
        "dynamic_control_contract": (
            "local and global retain the same latent per-source resource; "
            "global applies only its instantaneous mean"
        ),
        "git_commit_at_freeze": commit,
    }
    output = ROOT / config["output_root"] / "candidate_manifest.json"
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"],
        "n_candidates": len(candidates),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
