"""Filesystem roots, provenance helpers and atomic writers for v0.3.2 (model side)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from src.topic5_group_event_state.v02.registry import (  # re-use, do not re-invent
    atomic_write_json,
    file_hash,
    payload_hash,
    source_commit,
)

DATASET_ROOT = Path("/data/hfosp_group_event_state_v0_1/dataset")
V032_ROOT = Path("/data/hfosp_group_event_state_v0_3_2")
MODEL_ROOT = V032_ROOT / "model"
SHARED_ROOT = V032_ROOT / "shared"
HISTORY_BASELINE_REGISTRY = SHARED_ROOT / "history_baseline_registry.json"
ENDPOINT_ELIGIBILITY = SHARED_ROOT / "endpoint_eligibility.json"
FROZEN_STATE_REGISTRY = SHARED_ROOT / "frozen_state_registry.json"
SESSION_INVENTORY = Path(
    "/home/honglab/leijiaxin/HFOsp/results/epi_prssm/group_event_state/v0_1"
    "/contiguous_session_inventory.csv"
)
PYTHON = "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python"

FROZEN_SUBJECTS = ("epilepsiae_1146", "yuquan_pengzihang", "yuquan_zhangkexuan")
TRIAGE_SUBJECTS = ("epilepsiae_1146", "yuquan_zhangkexuan")
SEEDS = (20260902, 20260903, 20260904)

__all__ = [
    "DATASET_ROOT",
    "ENDPOINT_ELIGIBILITY",
    "FROZEN_STATE_REGISTRY",
    "FROZEN_SUBJECTS",
    "HISTORY_BASELINE_REGISTRY",
    "MODEL_ROOT",
    "PYTHON",
    "SEEDS",
    "SESSION_INVENTORY",
    "SHARED_ROOT",
    "TRIAGE_SUBJECTS",
    "V032_ROOT",
    "atomic_write_json",
    "atomic_write_npz",
    "atomic_write_torch",
    "file_hash",
    "payload_hash",
    "read_json",
    "repo_root",
    "source_commit",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """``np.savez_compressed`` into a sibling temp file, fsync, rename."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".npz.tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **{k: np.asarray(v) for k, v in arrays.items()})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_write_torch(path: Path, payload: Mapping[str, Any]) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".pt.tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
