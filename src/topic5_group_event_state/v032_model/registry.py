"""Export selected development checkpoints as immutable frozen trajectories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .data import load_subject_bundle
from .paths import (
    FROZEN_STATE_REGISTRY,
    MODEL_ROOT,
    atomic_write_json,
    atomic_write_npz,
    file_hash,
    source_commit,
    repo_root,
)
from .trainer import bundle_tensors, load_checkpoint_model

REGISTRY_FORMAT = "group_event_state_v0_3_2_frozen_state_registry"


@torch.no_grad()
def export_checkpoint_trajectory(
    *,
    subject: str,
    seed: int,
    architecture: str,
    checkpoint: Path,
    out_root: Path = MODEL_ROOT / "frozen_states",
    device: torch.device,
) -> dict[str, Any]:
    """Replay one checkpoint from every segment start and write open-loop states."""

    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("subject") != subject or int(payload.get("seed")) != int(seed):
        raise ValueError("checkpoint subject/seed disagree with requested registry entry")
    if payload.get("architecture") != architecture or payload.get("arm") != "learned":
        raise ValueError("only a learned checkpoint of the requested architecture may be exported")
    bundle = load_subject_bundle(subject, allow_provisional_h=False)
    model = load_checkpoint_model(checkpoint, in_dim=bundle.x_std.shape[1], device=device)
    tensors = bundle_tensors(bundle, device)
    pre, post = model.trajectory(tensors["x_std"], tensors["times"], tensors["segment"])
    anchor = model.anchor_states(post, tensors["times"], tensors["t_anchor"], tensors["last_event_pos"])
    anchor_std = model.standardize_state(anchor)
    if not torch.isfinite(anchor).all() or not torch.isfinite(pre).all() or not torch.isfinite(post).all():
        raise FloatingPointError("non-finite frozen trajectory")
    out_path = Path(out_root) / architecture / subject / f"seed_{int(seed)}.npz"
    atomic_write_npz(out_path, {
        "anchor_time": bundle.t_anchor.astype(np.float64),
        "anchor_state": anchor.cpu().numpy().astype(np.float32),
        "anchor_state_train_standardized": anchor_std.cpu().numpy().astype(np.float32),
        "anchor_segment": bundle.anchor_segment.astype(np.int64),
        "anchor_session": bundle.anchor_session.astype(np.int64),
        "anchor_phase": bundle.anchor_phase.astype(np.int64),
        "event_time": bundle.event_times.astype(np.float64),
        "event_pre_state": pre.cpu().numpy().astype(np.float32),
        "event_post_state": post.cpu().numpy().astype(np.float32),
        "event_segment": bundle.event_segment.astype(np.int64),
        "event_session": bundle.event_session.astype(np.int64),
        "event_phase": bundle.event_phase.astype(np.int64),
        "train_mean_state": model.train_mean_state.cpu().numpy().astype(np.float32),
        "train_state_scale": model.train_state_scale.cpu().numpy().astype(np.float32),
    })
    return {
        "status": "complete",
        "arrays_path": str(out_path),
        "arrays_sha256": file_hash(out_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_hash(checkpoint),
        "checkpoint_parameter_sha256": payload.get("parameter_sha256"),
        "config_hash": payload.get("config_hash"),
        "architecture": architecture,
        "state_dim": int(anchor.shape[1]),
        "selected_epoch": int(payload["selected_step"]),
        "selected_step": int(payload["selected_step"]),
        "selection_phase": "dev_val",
        "open_loop": True,
        "h_source": payload.get("h_source"),
        "source_fingerprint": payload.get("fingerprint"),
        "model_training_partition": {
            "calibration": [0.0, 0.2],
            "state_train": [0.2, 0.7],
            "dev_val": [0.7, 0.8],
            "dev_test": [0.8, 1.0],
        },
        "dev_test_used_for_selection": False,
        "sealed_partition_opened": False,
    }


def write_frozen_registry(
    entries: Iterable[tuple[str, int, dict[str, Any]]],
    *,
    path: Path = FROZEN_STATE_REGISTRY,
) -> dict[str, Any]:
    """Atomically merge complete entries into the evaluator's shared registry."""

    path = Path(path)
    if path.exists():
        payload = json.loads(path.read_text())
        if payload.get("format") != REGISTRY_FORMAT:
            raise ValueError(f"refusing to merge unknown registry format in {path}")
    else:
        payload = {
            "format": REGISTRY_FORMAT,
            "status": "partial",
            "source_commit": source_commit(repo_root()),
            "partition": {
                "boundary_fractions": [0.6, 0.7, 0.8],
                "meaning": "evaluation base_fit/inner_val/dev_val/dev_test boundaries",
            },
            "patients": {},
        }
    for subject, seed, entry in entries:
        patient = payload["patients"].setdefault(subject, {"status": "partial", "seeds": {}})
        patient["seeds"][str(int(seed))] = dict(entry)
        patient["status"] = "complete" if patient["seeds"] else "partial"
    complete = sum(
        int(seed.get("status") == "complete")
        for patient in payload["patients"].values()
        for seed in patient.get("seeds", {}).values()
    )
    payload["n_complete_entries"] = complete
    payload["status"] = "complete" if complete else "partial"
    payload["source_commit"] = source_commit(repo_root())
    atomic_write_json(path, payload)
    return payload

