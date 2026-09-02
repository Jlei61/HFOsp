"""Filesystem roots, release lookup, provenance and atomic writers (design §9, A2)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any, Sequence

from src.topic5_group_event_state.v02.registry import (  # re-use, do not re-invent
    atomic_write_json,
    file_hash,
    payload_hash,
    source_commit,
)
from src.topic5_group_event_state.v032_model.paths import atomic_write_npz, atomic_write_torch

V033_ROOT = Path("/data/hfosp_group_event_state_v0_3_3")
AGENT_B_ROOT = V033_ROOT / "agent_b"
SHARED_ROOT = V033_ROOT / "shared"
JOB_REQUESTS = SHARED_ROOT / "job_requests"
JOB_STATUS = SHARED_ROOT / "job_status"
RESOURCE_LEASES = SHARED_ROOT / "resource_leases"
RELEASE_FILENAME = "V0_3_3_EXECUTION_RELEASE.json"
PYTHON = "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python"
MAIN_TREE = Path("/home/honglab/leijiaxin/HFOsp")
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")

__all__ = [
    "AGENT_B_ROOT", "JOB_REQUESTS", "JOB_STATUS", "MAIN_TREE", "PYTHON", "RELEASE_FILENAME",
    "RESOURCE_LEASES", "SHARED_ROOT", "V033_ROOT", "atomic_write_json", "atomic_write_npz",
    "atomic_write_torch", "current_commit", "file_hash", "payload_hash", "read_json",
    "release_candidates", "release_status", "repo_root", "results_index", "set_single_thread_env",
    "source_commit",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def results_index(root: Path | None = None) -> Path:
    return (root or repo_root()) / "results" / "group_event_state" / "v0_3_3" / "training_laboratory"


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def current_commit(repo: Path | None = None) -> str:
    return source_commit(repo or repo_root())


def set_single_thread_env() -> None:
    """Every worker computes with one BLAS thread (supervisor runbook §6)."""

    for key in THREAD_ENV:
        os.environ[key] = "1"


def release_candidates(repo: Path | None = None) -> list[Path]:
    """Where the (undefined-by-spec) release file is looked for, in order (A2)."""

    repo = repo or repo_root()
    return [
        V033_ROOT / RELEASE_FILENAME,
        SHARED_ROOT / RELEASE_FILENAME,
        MAIN_TREE / "results" / "group_event_state" / "v0_3_3" / RELEASE_FILENAME,
        repo / "results" / "group_event_state" / "v0_3_3" / RELEASE_FILENAME,
    ]


def release_status(candidates: Sequence[Path] | None = None) -> dict[str, Any]:
    """[Q7] Absent from every candidate path -> ``present=False``; never inferred."""

    for path in list(candidates) if candidates is not None else release_candidates():
        path = Path(path)
        if path.exists():
            try:
                payload = json.loads(path.read_text())
            except json.JSONDecodeError as exc:
                return {"present": False, "path": str(path), "payload": None,
                        "reason": f"unreadable release file: {exc}"}
            return {"present": True, "path": str(path), "payload": payload, "reason": "found"}
    return {"present": False, "path": None, "payload": None, "reason": "no release file"}


def git_head(repo: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True, timeout=30)
        return out.stdout.strip()
    except Exception as exc:  # pragma: no cover - provenance is never faked
        return f"unavailable:{type(exc).__name__}"
