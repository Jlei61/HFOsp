"""Shared, fail-closed loaders for the R1.6 optimisation audit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from . import contract
from .bridge_e1 import make_paired_models
from .coverage import CoverageTable
from .optimizer_audit import transfer_prefix_core
from .r1_2 import (
    FrozenEmbeddingStateModel,
    _bridge_scaler,
    load_full_admissible_event_stream,
    load_full_design,
)
from .r1_3 import FullAnchorObservationLoader, FullTargetObserverStateModel


def load_explicit_target_model(*, subject: str, seed: int,
                               device: torch.device | str,
                               r1_2_root: Path,
                               observation_cache_root: Path,
                               output_root: Path,
                               prefix_config_id: str) -> dict:
    """Load one selection-safe explicit model without reading dev outcomes."""
    manifest_path = observation_cache_root / subject / "manifest.json"
    if not manifest_path.exists():
        candidates = (
            contract.RESULT_ROOT / "r1_5/cache" / subject / "manifest.json",
            contract.RESULT_ROOT / "r1_3_long_t1_triage/cache"
            / subject / "manifest.json",
            contract.RESULT_ROOT / "r1_3/cache" / subject / "manifest.json",
        )
        manifest_path = next(
            (candidate for candidate in candidates if candidate.exists()),
            manifest_path,
        )
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("status") != "COMPLETE"
            or manifest.get("sealed_opened") is not False):
        raise ValueError("R1.6 observation cache is not admissible")
    design_path = Path(manifest["design"])
    if contract.sha256_file(design_path) != manifest["design_sha256"]:
        raise ValueError("R1.6 observation design hash mismatch")
    design = load_full_design(design_path)

    baseline_path = r1_2_root / "baselines" / subject / "seed_0/models.pt"
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
    coverage = CoverageTable.load(r1_2_root / "coverage" / f"{subject}.npz")
    stream = load_full_admissible_event_stream(subject, coverage)
    bridge_result_path = r1_2_root / "bridge_e1" / subject / "seed_0/result.json"
    bridge_result = json.loads(bridge_result_path.read_text())
    _, _, sampled, _ = _bridge_scaler(
        subject, baseline_path, bridge_result, stream, coverage
    )
    _, raw_bridge = make_paired_models(
        baseline, sampled, stream.adjacency, seed=0, device=device
    )
    bridge_checkpoint = torch.load(
        r1_2_root / "bridge_e1" / subject / "seed_0/models.pt",
        map_location=device, weights_only=False,
    )
    raw_bridge.load_state_dict(bridge_checkpoint["explicit_raw"])

    prefix_result_path = (
        output_root / "prefix_initialisation" / prefix_config_id
        / subject / f"seed_{seed}/result.json"
    )
    prefix_result = json.loads(prefix_result_path.read_text())
    prefix_checkpoint = Path(prefix_result["checkpoint"])
    if contract.sha256_file(prefix_checkpoint) != prefix_result["checkpoint_sha256"]:
        raise ValueError("R1.6 prefix checkpoint hash mismatch")
    prefix_payload = torch.load(
        prefix_checkpoint, map_location=device, weights_only=False
    )
    prefix_core = FrozenEmbeddingStateModel(
        baseline, design.event_history.shape[1], stream.n_contacts,
        stream.adjacency, observation_dim=64, state_dim=8,
    ).to(device)
    prefix_core.load_state_dict(prefix_payload["model"], strict=True)
    model = FullTargetObserverStateModel(
        baseline, design.event_history.shape[1], stream.n_contacts,
        stream.adjacency, raw_bridge.observer, use_raw=False, state_dim=8,
    ).to(device)
    transfer_prefix_core(model, prefix_core)

    explicit_path = Path(manifest["explicit"])
    mask_path = Path(manifest["contact_mask"])
    if contract.sha256_file(explicit_path) != manifest["explicit_sha256"]:
        raise ValueError("R1.6 explicit observation hash mismatch")
    if contract.sha256_file(mask_path) != manifest["contact_mask_sha256"]:
        raise ValueError("R1.6 contact mask hash mismatch")
    loader = FullAnchorObservationLoader(
        subject, design, stream.event_time,
        sampled.explicit_mean, sampled.explicit_scale,
        cached_explicit=np.load(explicit_path, mmap_mode="r"),
        cached_contact_mask=np.load(mask_path, mmap_mode="r"),
    )
    return {
        "model": model, "design": design, "loader": loader,
        "manifest": manifest, "manifest_path": manifest_path,
        "prefix_result": prefix_result,
        "prefix_result_path": prefix_result_path,
    }
