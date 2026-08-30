"""Rebuild one frozen rev16 joint-M3+M4 Node field downstream."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts import freeze_topic4_rev16_joint_m3_m4_candidates as freezer
from scripts import run_topic4_rev14_m3_canary_worker as shared_worker
from scripts import topic4_rev15_node_substrate_adapter as base


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def build_projected_node_substrate(
    *, robust_config_path: Path, candidate_id: str, seed: int,
    artifact_root: Path = ARTIFACT_ROOT,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Recreate the exact rev16 Node-only substrate without stepping the SNN."""
    previous = (
        base.robust_freezer, shared_worker.freezer,
        shared_worker.EXPECTED_PATHWAYS,
    )
    base.robust_freezer = freezer
    shared_worker.freezer = freezer
    shared_worker.EXPECTED_PATHWAYS = freezer.EXPECTED_PATHWAYS
    try:
        substrate, projection, transition = base.build_projected_node_substrate(
            robust_config_path=robust_config_path,
            candidate_id=candidate_id, seed=int(seed),
            artifact_root=artifact_root,
        )
    finally:
        (
            base.robust_freezer, shared_worker.freezer,
            shared_worker.EXPECTED_PATHWAYS,
        ) = previous
    substrate.extras["rev16_joint_m3_m4_projection"] = projection["audit"]
    return substrate, projection, transition


def verify_projection_against_worker(
    projection: dict[str, Any], worker_npz: Path,
) -> dict[str, Any]:
    """Require exact h/Vtheta/delta-Vtheta parity with the frozen worker."""
    return base.verify_projection_against_worker(projection, worker_npz)
