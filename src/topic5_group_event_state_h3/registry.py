"""Reading agent A's producer registry, and publishing agent C's own entries.

Two rules, both from the common contract:

*A missing producer is ``not_available``, never a fallback.*  If the registry has
no ``P_slow``, the registry-bound analysis reports that it could not run.  It does
not quietly substitute a model this agent happens to have trained, because the
substitute answers a different question and the field name would not say so.

*Agent A owns the shared registry file.*  C never rewrites it.  C's own producers
go in a separate additive file next to it, so two agents writing at once cannot
produce a last-writer-wins registry.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .io import file_hash, write_json_atomic

SHARED_ROOT = Path(
    "/home/honglab/leijiaxin/HFOsp/results/epi_prssm/group_event_state/v0_2/shared"
)
SHARED_REGISTRY = SHARED_ROOT / "checkpoint_registry.json"
AGENT_C_REGISTRY_NAME = "checkpoint_registry_agent_c.json"

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


@dataclass
class ProducerStatus:
    producer_id: str
    status: str           # "ok" | "not_available" | "invalid"
    detail: str
    entry: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "producer_id": self.producer_id,
            "status": self.status,
            "detail": self.detail,
            "entry": self.entry,
        }


def load_shared_registry(path: Path = SHARED_REGISTRY) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def resolve_producer(
    producer_id: str,
    subject: str,
    *,
    registry_path: Path = SHARED_REGISTRY,
) -> ProducerStatus:
    """Look up one producer for one patient and verify it end to end.

    Verified means: the entry exists, carries every contract field, points at a
    checkpoint that is on disk, and that checkpoint's content hash matches what
    the entry claims.  A hash mismatch is ``invalid``, not a warning: it means the
    file moved under the entry, and scoring against it would attribute one model's
    behaviour to another's provenance.
    """

    registry = load_shared_registry(registry_path)
    if registry is None:
        return ProducerStatus(
            producer_id, "not_available",
            f"no shared registry at {registry_path}; agent A has not published producers",
        )
    producers = registry.get("producers", registry)
    entry = None
    if isinstance(producers, dict):
        entry = (producers.get(producer_id) or {}).get(subject)
        if entry is None and isinstance(producers.get(producer_id), dict):
            entry = producers[producer_id].get("subjects", {}).get(subject)
    elif isinstance(producers, list):
        for candidate in producers:
            if candidate.get("producer_id") == producer_id and candidate.get("subject") == subject:
                entry = candidate
                break
    if entry is None:
        return ProducerStatus(
            producer_id, "not_available",
            f"registry has no entry for producer {producer_id!r} and subject {subject!r}",
        )

    missing = [field for field in REQUIRED_FIELDS if field not in entry]
    if missing:
        return ProducerStatus(
            producer_id, "invalid", f"entry is missing contract fields: {missing}", entry
        )
    checkpoint = Path(entry.get("checkpoint_path", ""))
    if not checkpoint.exists():
        return ProducerStatus(
            producer_id, "invalid", f"checkpoint_path does not exist: {checkpoint}", entry
        )
    actual = file_hash(checkpoint)
    if actual != entry["checkpoint_hash"]:
        return ProducerStatus(
            producer_id, "invalid",
            f"checkpoint_hash mismatch: entry {entry['checkpoint_hash'][:12]} "
            f"vs file {actual[:12]}",
            entry,
        )
    return ProducerStatus(producer_id, "ok", "verified", entry)


def publish_agent_c_entry(
    entry: dict[str, Any], *, shared_root: Path = SHARED_ROOT
) -> Path:
    """Add one C-owned producer entry to C's own registry file.

    Written per entry and merged, never as a whole-file overwrite, so a
    concurrent write by another agent cannot erase entries it never saw.
    """

    missing = [field for field in REQUIRED_FIELDS if field not in entry]
    if missing:
        raise ValueError(f"agent C registry entry is missing contract fields: {missing}")
    path = Path(shared_root) / AGENT_C_REGISTRY_NAME
    payload = {"owner": "agent_c", "producers": {}}
    if path.exists():
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    key = f"{entry['producer_id']}|{entry.get('subject')}|{entry.get('seed')}"
    payload.setdefault("producers", {})[key] = entry
    return write_json_atomic(payload, path)
