#!/usr/bin/env python3
"""Prove zero-residual dual Node reconstruction against frozen worker arrays."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_zm_ictal_transition import build_substrate, load_round_config  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(root: Path, relative: str) -> Path:
    local = ROOT / relative
    return local if local.exists() else root / relative


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def audit(config_path: Path, *, artifact_root: Path) -> dict:
    config = json.loads(config_path.read_text())
    manifest_path = artifact_root / config["candidate_manifest"]
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "REV17_DUAL_FIELD_RESIDUAL_ATLAS_FROZEN":
        raise RuntimeError("rev17 dual-field manifest is not frozen")
    if manifest.get("config_sha256") != _sha256(config_path):
        raise RuntimeError("rev17 manifest config hash changed")
    anchor_rows = [
        row for row in manifest["candidates"]
        if row["candidate_id"] == "exact_dual_anchor"
    ]
    if len(anchor_rows) != 1:
        raise RuntimeError("rev17 exact dual anchor is absent or duplicated")
    anchor = anchor_rows[0]
    transition_record = config["inputs"]["transition_config"]
    transition_path = _resolve(artifact_root, transition_record["path"])
    if _sha256(transition_path) != transition_record["sha256"]:
        raise RuntimeError("rev17 transition config changed")
    transition = load_round_config(transition_path)
    rows = []
    for seed in config["parity"]["reference_seeds"]:
        seed = int(seed)
        substrate = build_substrate(
            transition, "node_baseline", seed,
            cache_dir=str(artifact_root / config["network_cache"]),
            ee_dose=0.0, etoi_dose=0.0,
            node_candidate_override=anchor["node_field"],
            node_dispersion_candidate_override=anchor["node_dispersion_field"],
            artifact_root=artifact_root,
        )
        record = config["inputs"][f"exact_off_seed_{seed}"]
        reference_path = _resolve(artifact_root, record["path"])
        if _sha256(reference_path) != record["sha256"]:
            raise RuntimeError(f"rev17 exact-off reference changed for seed {seed}")
        with np.load(reference_path, allow_pickle=False) as loaded:
            reference_h = np.asarray(loaded["h"])
            reference_delta = np.asarray(loaded["delta_vtheta"])
        observed_h = np.asarray(substrate.h_e, dtype=reference_h.dtype)
        # The archived rev14 exact-off worker stored its bypass value as
        # ``vtheta - v_base``.  Compare through that same arithmetic path: the
        # SNN consumes vtheta, while the dual-field decomposition is retained
        # below as a separate internal-consistency diagnostic.
        v_base = float(substrate.engine["v_base"])
        vtheta_delta = np.asarray(substrate.vtheta[:substrate.n_e], float) - v_base
        observed_delta = np.asarray(vtheta_delta, dtype=reference_delta.dtype)
        decomposition_error = float(np.max(np.abs(
            np.asarray(substrate.delta_vtheta, float) - vtheta_delta
        )))
        row = {
            "seed": seed,
            "h_exact": bool(np.array_equal(observed_h, reference_h)),
            "delta_vtheta_exact": bool(np.array_equal(observed_delta, reference_delta)),
            "h_max_abs_error": float(np.max(np.abs(observed_h - reference_h))),
            "delta_vtheta_max_abs_error_mV": float(np.max(
                np.abs(observed_delta - reference_delta)
            )),
            "internal_decomposition_max_abs_error_mV": decomposition_error,
            "edge_coefficients_all_zero": bool(np.array_equal(
                np.asarray(substrate.edge_coefficients),
                np.zeros_like(np.asarray(substrate.edge_coefficients)),
            )),
            "mapping_type": substrate.extras["node_mapping_audit"]["mapping_type"],
        }
        row["pass"] = bool(
            row["h_exact"] and row["delta_vtheta_exact"]
            and row["edge_coefficients_all_zero"]
            and row["internal_decomposition_max_abs_error_mV"] <= 1e-14
            and row["mapping_type"] == "dual_continuous_mean_dispersion"
        )
        rows.append(row)
    passed = all(row["pass"] for row in rows)
    return {
        "schema_id": "topic4_rev17_dual_field_zero_residual_parity_v1",
        "status": "REV17_DUAL_FIELD_ZERO_RESIDUAL_PARITY_PASS" if passed else
                  "REV17_DUAL_FIELD_ZERO_RESIDUAL_PARITY_FAIL",
        "config_sha256": _sha256(config_path),
        "manifest_sha256": _sha256(manifest_path),
        "per_network": rows,
        "all_networks_pass": passed,
        "SNN_simulation_run": False,
        "claim_boundary": (
            "This audit proves reconstruction parity only; it does not establish "
            "two-mode patient fit or authorize EE, E-to-I, or Z/M."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    payload = audit(config_path, artifact_root=args.artifact_root.resolve())
    config = json.loads(config_path.read_text())
    output = args.artifact_root.resolve() / config["parity"]["output"]
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"], "output": str(output),
        "SNN_simulation_run": False,
    }, indent=2))
    if not payload["all_networks_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
