"""Rebuild one frozen rev17 dual-continuous Node substrate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.topic4_zm_ictal_transition import build_substrate, load_round_config


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _resolve(relative: str, artifact_root: Path) -> Path:
    for root in (ROOT, artifact_root):
        path = root / str(relative)
        if path.is_file():
            return path.resolve()
    raise RuntimeError(f"rev17 intervention input is missing: {relative}")


def _load_candidate(
    config_path: Path, config: dict[str, Any], candidate_id: str,
    artifact_root: Path,
) -> tuple[dict[str, Any], Path]:
    if config.get("schema_id") != "topic4_rev17_node_confirmation_v1":
        raise RuntimeError("rev17 intervention confirmation schema changed")
    manifest_path = artifact_root / str(config["candidate_manifest"])
    if not manifest_path.is_file():
        raise RuntimeError("rev17 intervention confirmation manifest is missing")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("status") != "REV17_NODE_CONFIRMATION_CANDIDATES_FROZEN"
        or manifest.get("config_sha256") != _sha256(config_path)
        or manifest.get("provenance", {}).get("formal_ready") is not True
    ):
        raise RuntimeError("rev17 intervention confirmation manifest is not frozen")
    candidates = [
        row for row in manifest["candidates"]
        if str(row["candidate_id"]) == str(candidate_id)
    ]
    if len(candidates) != 1:
        raise RuntimeError("rev17 intervention candidate is not unique")
    candidate = candidates[0]
    expected = config.get("selected_candidate", {})
    if (
        str(expected.get("candidate_id")) != str(candidate_id)
        or candidate.get("node_mapping", {}).get("mapping_sha256")
        != expected.get("mapping_sha256")
    ):
        raise RuntimeError("rev17 intervention selected mapping changed")
    return candidate, manifest_path


def build_projected_node_substrate(
    *, robust_config_path: Path, candidate_id: str, seed: int,
    artifact_root: Path = ARTIFACT_ROOT,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Recreate the exact dual-field Node substrate without stepping the SNN."""
    robust_config_path = robust_config_path.resolve()
    artifact_root = artifact_root.resolve()
    config = json.loads(robust_config_path.read_text())
    candidate, manifest_path = _load_candidate(
        robust_config_path, config, candidate_id, artifact_root,
    )
    seeds = {
        int(value) for value in config["search"]["confirmation_network_seeds"]
    }
    if int(seed) not in seeds:
        raise RuntimeError("rev17 intervention seed is outside confirmation pool")
    transition_record = config["inputs"]["transition_config"]
    transition_path = _resolve(transition_record["path"], artifact_root)
    if _sha256(transition_path) != transition_record["sha256"]:
        raise RuntimeError("rev17 intervention transition config changed")
    transition = load_round_config(transition_path)
    mapping = candidate.get("node_mapping", {})
    substrate = build_substrate(
        transition, "node_baseline", int(seed),
        cache_dir=str(artifact_root / config["network_cache"]),
        ee_dose=0.0, etoi_dose=0.0,
        node_candidate_override=candidate["node_field"],
        node_depth_shrinkage=float(mapping.get("signed_depth_shrinkage", 1.0)),
        node_gain=float(mapping.get("node_gain", 1.0)),
        node_dispersion_candidate_override=candidate["node_dispersion_field"],
        artifact_root=artifact_root,
    )
    if mapping.get("mapping_type") != "dual_continuous_mean_dispersion":
        raise RuntimeError("rev17 intervention mapping is not dual-continuous")
    if not np.array_equal(
        np.asarray(substrate.edge_coefficients),
        np.zeros_like(np.asarray(substrate.edge_coefficients)),
    ):
        raise RuntimeError("rev17 intervention activated edge coefficients")
    projection = {
        "h": np.asarray(substrate.h_e, dtype=np.float64),
        "vtheta": np.asarray(substrate.vtheta, dtype=np.float64),
        "delta_vtheta": np.asarray(substrate.delta_vtheta, dtype=np.float64),
        "mapping_sha256": mapping["mapping_sha256"],
        "audit": {
            **substrate.extras["node_mapping_audit"],
            "mapping_sha256": mapping["mapping_sha256"],
            "confirmation_config_sha256": _sha256(robust_config_path),
            "confirmation_manifest_sha256": _sha256(manifest_path),
        },
    }
    projection["hashes"] = {
        key: _array_sha256(projection[key])
        for key in ("h", "vtheta", "delta_vtheta")
    }
    substrate.extras["rev17_dual_projection"] = projection["audit"]
    return substrate, projection, transition


def verify_projection_against_worker(
    projection: dict[str, Any], worker_npz: Path,
) -> dict[str, Any]:
    """Require exact stored h/delta-Vtheta parity with a confirmation worker."""
    with np.load(worker_npz, allow_pickle=False) as loaded:
        observed = {
            key: np.asarray(loaded[key]) for key in ("h", "delta_vtheta")
        }
    exact = {
        key: bool(np.array_equal(
            observed[key], np.asarray(projection[key], dtype=observed[key].dtype),
        ))
        for key in observed
    }
    if not all(exact.values()):
        raise RuntimeError("rev17 intervention substrate differs from confirmation worker")
    return {
        "exact_array_parity": exact,
        "worker_npz": str(worker_npz.resolve()),
        "worker_npz_sha256": _sha256(worker_npz),
        "mapping_sha256": projection["mapping_sha256"],
        "array_hashes": projection["hashes"],
        "vtheta_storage_contract": (
            "not_stored_separately; reconstructed from the frozen baseline "
            "threshold and exactly matched delta_vtheta"
        ),
    }
