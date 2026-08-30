"""Rebuild one frozen rev15 Fourier Node field for downstream intervention."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts import freeze_topic4_rev15_m3_robust_candidates as robust_freezer
from scripts import run_topic4_rev14_m3_canary_worker as m3_worker
from src.topic4_zm_ictal_transition import load_round_config


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(config_path: Path, config: dict[str, Any],
                   artifact_root: Path) -> tuple[dict[str, Any], Path]:
    robust_freezer._validate_config(config)
    path = artifact_root / str(config["candidate_manifest"])
    if not path.is_file():
        raise RuntimeError("rev15 robust-candidate manifest is missing")
    manifest = json.loads(path.read_text())
    if manifest.get("status") != robust_freezer.STATUS:
        raise RuntimeError("rev15 robust-candidate manifest is not frozen")
    if manifest.get("schema_id") != robust_freezer.MANIFEST_SCHEMA:
        raise RuntimeError("rev15 robust-candidate manifest schema changed")
    if manifest.get("config_sha256") != _sha256(config_path):
        raise RuntimeError("rev15 robust-candidate manifest/config mismatch")
    return manifest, path


def _candidate(manifest: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    matches = [
        row for row in manifest["candidates"]
        if str(row["candidate_id"]) == str(candidate_id)
    ]
    if len(matches) != 1:
        raise RuntimeError("rev15 intervention candidate is not unique")
    candidate = matches[0]
    if candidate_id != "exact_off" and not candidate.get("selection_eligible"):
        raise RuntimeError("rev15 intervention candidate is not selectable")
    return candidate


def build_projected_node_substrate(
    *, robust_config_path: Path, candidate_id: str, seed: int,
    artifact_root: Path = ARTIFACT_ROOT,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Recreate the exact Node-only substrate used by the robust worker.

    The return values are substrate, Fourier projection, and transition config.
    No simulation step is executed here.
    """
    robust_config_path = robust_config_path.resolve()
    artifact_root = artifact_root.resolve()
    config = json.loads(robust_config_path.read_text())
    manifest, manifest_path = _load_manifest(
        robust_config_path, config, artifact_root,
    )
    active_seeds = {int(value) for value in config["search"]["active_network_seeds"]}
    if int(seed) not in active_seeds:
        raise RuntimeError("rev15 intervention seed is outside the robust pool")
    candidate = _candidate(manifest, candidate_id)

    compatibility, compatibility_audit = m3_worker._compatibility_config(
        config, manifest, artifact_root=artifact_root,
    )
    compatibility_manifest = m3_worker._compatibility_manifest(
        manifest, candidate, _sha256(robust_config_path),
    )
    state = m3_worker._RunState(
        config=copy.deepcopy(config),
        candidate=copy.deepcopy(candidate),
        manifest=manifest,
        provenance={"formal_ready": True, "simulation_run": False},
        manifest_audit={
            "manifest_read": True,
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
        },
        compatibility_config=compatibility,
        compatibility_manifest=compatibility_manifest,
        compatibility_audit=compatibility_audit,
        source_config_text=robust_config_path.read_text(),
        source_manifest_text=manifest_path.read_text(),
    )
    substrate = m3_worker._reconstruct_static_node_substrate(
        state, seed=int(seed), artifact_root=artifact_root,
    )
    projection = m3_worker._project_candidate(candidate, substrate, config)
    substrate.h_e = np.asarray(projection["h"], dtype=np.float64)
    substrate.vtheta = np.asarray(projection["vtheta"], dtype=np.float64)
    substrate.delta_vtheta = np.asarray(
        projection["delta_vtheta"], dtype=np.float64,
    )
    substrate.extras["rev15_m3_projection"] = copy.deepcopy(projection["audit"])

    if not np.array_equal(
        np.asarray(substrate.edge_coefficients),
        np.zeros_like(np.asarray(substrate.edge_coefficients)),
    ):
        raise RuntimeError("rev15 intervention substrate activated edge coefficients")
    if projection["hashes"]["h_sha256"] != m3_worker.array_sha256(
        np.asarray(substrate.h_e, dtype=np.float64)
    ):
        raise RuntimeError("rev15 intervention h projection hash changed")
    if projection["hashes"]["vtheta_sha256"] != m3_worker.array_sha256(
        np.asarray(substrate.vtheta, dtype=np.float64)
    ):
        raise RuntimeError("rev15 intervention threshold projection hash changed")
    if projection["hashes"]["frozen_signed_depth_sha256"] != config[
        "node_mapping"
    ]["signed_depth_contract"]["sha256"]:
        raise RuntimeError("rev15 intervention signed-depth contract changed")

    transition_record = compatibility["inputs"]["transition_config"]
    transition_path = robust_freezer._resolve(
        artifact_root, str(transition_record["path"]),
    )
    if _sha256(transition_path) != str(transition_record["sha256"]):
        raise RuntimeError("rev15 intervention transition config changed")
    transition = load_round_config(transition_path)
    return substrate, projection, transition


def verify_projection_against_worker(
    projection: dict[str, Any], worker_npz: Path,
) -> dict[str, Any]:
    """Require the downstream reconstruction to equal a frozen worker."""
    with np.load(worker_npz, allow_pickle=False) as loaded:
        comparisons = {
            "h": np.asarray(loaded["h"], dtype=np.float64),
            "vtheta": np.asarray(loaded["vtheta"], dtype=np.float64),
            "delta_vtheta": np.asarray(loaded["delta_vtheta"], dtype=np.float64),
        }
    expected = {
        "h": np.asarray(projection["h"], dtype=np.float64),
        "vtheta": np.asarray(projection["vtheta"], dtype=np.float64),
        "delta_vtheta": np.asarray(projection["delta_vtheta"], dtype=np.float64),
    }
    exact = {key: bool(np.array_equal(expected[key], comparisons[key])) for key in expected}
    if not all(exact.values()):
        raise RuntimeError("rev15 intervention substrate differs from frozen worker")
    return {
        "exact_array_parity": exact,
        "worker_npz": str(worker_npz.resolve()),
        "worker_npz_sha256": _sha256(worker_npz),
        "projection_sha256": projection["hashes"]["projection_sha256"],
    }
