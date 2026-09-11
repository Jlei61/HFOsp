"""Shared producer registry, atomic outputs and no-silent-fallback lookups.

Agent B (seizure transfer) and Agent C (event feedback) read this registry to
find the frozen state producers.  Two rules make that safe:

* **one file per producer.**  A single ``checkpoint_registry.json`` rewritten by
  three agents is last-writer-wins; here each producer owns
  ``producers/<producer_id>.json`` written by atomic rename, and the combined
  view is assembled on read.
* **a missing producer is ``not_available``, never a substitute.**  Silently
  falling back to another producer would let a B or C result be attributed to a
  model that was never trained.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterable, Mapping

REGISTRY_VERSION = "group_event_state_v0_2_registry_1"

REQUIRED_FIELDS = (
    "producer_id",
    "model_family",
    "uses_waveform",
    "uses_multiband",
    "uses_background",
    "event_update",
    "feedback_model",
    "physical_dt",
    "training_objective",
    "anchor_grid_minutes",
    "source_commit",
    "config_hash",
    "checkpoint_hash",
)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write to a temporary file, fsync, then rename (EI 4).

    A reader must never see a half-written registry entry, and a crash must
    leave the previous entry intact rather than a truncated one.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, default=float)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def file_hash(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def payload_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def source_commit(repo: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=30,
        )
        return out.stdout.strip()
    except Exception as exc:  # pragma: no cover - provenance must never be faked
        return f"unavailable:{type(exc).__name__}"


@dataclass
class ProducerEntry:
    """One registry row.  Field names are fixed by CC 10 and are not optional."""

    producer_id: str
    model_family: str
    uses_waveform: bool
    uses_multiband: bool
    uses_background: bool
    event_update: bool
    feedback_model: str
    physical_dt: bool
    training_objective: list[str]
    anchor_grid_minutes: float
    source_commit: str
    config_hash: str
    checkpoint_hash: str
    dataset_root: str = ""
    timeline_config_hash: str = ""
    target_builder_hash: str = ""
    selection_objective: str = ""
    status: str = "complete"
    subjects: dict[str, Any] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        missing = [f for f in REQUIRED_FIELDS if getattr(self, f, None) in (None, "")]
        if missing:
            raise ValueError(f"{self.producer_id}: registry entry missing {missing}")


class ProducerRegistry:
    """Directory-backed registry with per-producer atomic entries."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.producer_dir = self.root / "producers"

    def write(self, entry: ProducerEntry) -> Path:
        entry.validate()
        path = self.producer_dir / f"{entry.producer_id}.json"
        atomic_write_json(path, asdict(entry))
        self.refresh_combined_view()
        return path

    def get(self, producer_id: str) -> dict[str, Any]:
        path = self.producer_dir / f"{producer_id}.json"
        if not path.exists():
            return {"producer_id": producer_id, "status": "not_available"}
        return json.loads(path.read_text())

    def require(self, producer_id: str) -> dict[str, Any]:
        entry = self.get(producer_id)
        if entry.get("status") != "complete":
            raise LookupError(
                f"producer {producer_id!r} is {entry.get('status')!r}; "
                "downstream lines must report not_available rather than "
                "substituting another producer"
            )
        return entry

    def list_producers(self) -> list[str]:
        if not self.producer_dir.exists():
            return []
        return sorted(p.stem for p in self.producer_dir.glob("*.json"))

    def refresh_combined_view(self) -> Path:
        combined = {
            "registry_version": REGISTRY_VERSION,
            "producers": {p: self.get(p) for p in self.list_producers()},
        }
        path = self.root / "checkpoint_registry.json"
        atomic_write_json(path, combined)
        return path
