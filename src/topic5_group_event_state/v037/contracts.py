"""Machine-checkable version, causality, and registry contracts for v0.3.7."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


ARCHITECTURE_VERSION = "0.3.7"
FORMAT_PREFIX = "group_event_state_v0_3_7"
DEFAULT_RESULT_ROOT = Path("/data/hfosp_group_event_state_v0_3_7")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class VisibilityStamp:
    """Latest source time used to construct a prediction tensor."""

    name: str
    maximum_visible_time: float
    sources: tuple[str, ...] = ()

    def validate(self) -> "VisibilityStamp":
        if not self.name:
            raise ValueError("visibility stamp needs a name")
        if not float("-inf") < float(self.maximum_visible_time) < float("inf"):
            raise ValueError("maximum_visible_time must be finite")
        return self


def assert_causal_visibility(
    stamps: Sequence[VisibilityStamp],
    anchor_time: float,
    *,
    allow_equal: bool = False,
) -> None:
    """Reject a tensor whose provenance sees the anchor or its future.

    Pre-event H2a uses strict ``< anchor``.  A grid feature measured exactly at
    the left edge may opt into ``<= anchor`` explicitly.
    """

    anchor = float(anchor_time)
    bad = []
    for stamp in stamps:
        stamp.validate()
        visible = float(stamp.maximum_visible_time)
        causal = visible <= anchor if allow_equal else visible < anchor
        if not causal:
            bad.append((stamp.name, visible))
    if bad:
        detail = ", ".join(f"{name}@{value:.6f}" for name, value in bad)
        relation = "<=" if allow_equal else "<"
        raise ValueError(f"causal visibility failure: expected maximum source time {relation} {anchor}; {detail}")


@dataclass(frozen=True)
class CheckpointEntry:
    key: str
    model_family: str
    state_semantics: str
    subject: str
    seed: int
    checkpoint_path: str
    maximum_training_time: float
    input_streams: tuple[str, ...]
    objectives: tuple[str, ...]
    code_commit: str
    normalization_provenance: str
    selection_partition: str = "INNER"
    development_targets_read: bool = False
    seizure_targets_read: bool = False
    sealed_partition_opened: bool = False

    def validate(self) -> "CheckpointEntry":
        if self.state_semantics not in {"observer", "generative_physiological", "baseline"}:
            raise ValueError(f"invalid state semantics: {self.state_semantics}")
        if self.state_semantics == "observer" and self.model_family.startswith("M") and self.model_family in {
            "M0_common_drive", "M1_count_feedback", "M2_mark_feedback"
        }:
            raise ValueError("H3 generative models cannot be registered as observer state")
        if self.selection_partition != "INNER":
            raise ValueError("v0.3.7 checkpoints must be selected on INNER only")
        if self.development_targets_read or self.seizure_targets_read or self.sealed_partition_opened:
            raise ValueError("interictal checkpoint registry cannot contain downstream/sealed reads")
        checkpoint = Path(self.checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        if not self.key or not self.subject or not self.code_commit:
            raise ValueError("checkpoint entry is missing an identity field")
        return self


def update_checkpoint_registry(path: Path, entry: CheckpointEntry) -> dict[str, Any]:
    """Atomically insert an immutable registry entry under an advisory lock."""

    entry.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            payload = {
                "format": f"{FORMAT_PREFIX}_checkpoint_registry_v1",
                "architecture_version": ARCHITECTURE_VERSION,
                "entries": {},
                "development_targets_read": False,
                "sealed_partition_opened": False,
            }
        if payload.get("architecture_version") != ARCHITECTURE_VERSION:
            raise ValueError("refusing to mix checkpoint registry versions")
        serialised = asdict(entry)
        serialised["input_streams"] = list(entry.input_streams)
        serialised["objectives"] = list(entry.objectives)
        serialised["checkpoint_sha256"] = sha256_file(Path(entry.checkpoint_path))
        previous = payload["entries"].get(entry.key)
        if previous is not None and previous != serialised:
            raise FileExistsError(f"immutable checkpoint key already has different payload: {entry.key}")
        payload["entries"][entry.key] = serialised
        atomic_json(path, payload)
        return payload
