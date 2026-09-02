"""Read Agent A's checkpoint registry, without ever silently filling a gap.

B consumes producers; it does not choose among them. Every registered producer
becomes an arm, and an arm B cannot load carries ``status="not_available"`` with
the reason attached, so a missing producer can never be mistaken for a producer
that failed to help (v0.2 common contract §10).

Provenance travels with each arm -- source commit, config hash, checkpoint hash
-- so a later comparison cannot silently mix producer versions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

DEFAULT_REGISTRY = Path(
    "/home/honglab/leijiaxin/HFOsp/results/epi_prssm/group_event_state/v0_2/"
    "shared/checkpoint_registry.json"
)


@dataclass(frozen=True)
class Registry:
    path: Path
    version: str
    producers: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class Arm:
    producer_id: str
    status: str                 # "ok" | "not_available"
    reason: str = ""
    seed: str | None = None
    state: np.ndarray | None = None
    t_anchor: np.ndarray | None = None
    split_index: np.ndarray | None = None
    session_id: np.ndarray | None = None
    source_commit: str = ""
    config_hash: str = ""
    checkpoint_hash: str = ""
    anchor_path: str = ""
    verified: bool = False
    commit: str = ""
    chunk_events: int | None = None
    batch_segments: int | None = None
    provenance_flags: tuple[str, ...] = ()


def read_registry(path: Path | str = DEFAULT_REGISTRY) -> Registry:
    p = Path(path)
    payload = json.loads(p.read_text())
    return Registry(path=p, version=str(payload.get("registry_version", "")),
                    producers=payload.get("producers", {}))


def _pick_seed(seeds: Mapping[str, Any], seed: str | None) -> tuple[str | None, Any, str]:
    """Resolve a seed, refusing to substitute a different one.

    Silently returning seed 1 when seed 3 was asked for would fabricate
    replication: three byte-identical "seeds" are one fit, not three
    (v0.2 engineering invariants §2).
    """

    if not isinstance(seeds, Mapping):
        return None, None, "producer entry has no seed mapping"
    keys = [k for k in seeds if isinstance(seeds[k], Mapping) and "anchor_state" in seeds[k]]
    if not keys:
        return None, None, ("producer exposes no anchor_state for this subject "
                            "(results-only payload)")
    if seed is None:
        k = sorted(keys)[0]
        return k, seeds[k], ""
    if seed in keys:
        return seed, seeds[seed], ""
    return None, None, f"seed {seed!r} not available for this subject (have {sorted(keys)})"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _training_record(info: Mapping[str, Any]) -> dict:
    """The substantive training facts, read from the run's own result.json.

    ``config_hash`` proved to hash things that are not the training
    configuration -- 57 distinct hashes over 18 distinct real
    ``(chunk_events, batch_segments, commit)`` combinations -- so verifying on
    it refused every cell for a bookkeeping difference. What actually matters is
    what the run recorded about itself.
    """

    ap = info.get("anchor_state")
    if not ap:
        return {}
    rp = Path(str(ap)).parent / "result.json"
    if not rp.exists():
        return {}
    try:
        d = json.loads(rp.read_text())
    except Exception:
        return {}
    return {"chunk_events": d.get("chunk_events"),
            "batch_segments": d.get("batch_segments"),
            "commit": str(d.get("commit", ""))[:8]}


def _modal_training_config(entry: Mapping[str, Any]) -> tuple:
    counts: dict[tuple, int] = {}
    for v in (entry.get("subjects", {}) or {}).values():
        if not isinstance(v, Mapping):
            continue
        for i in v.values():
            if not (isinstance(i, Mapping) and "anchor_state" in i):
                continue
            r = _training_record(i)
            k = (r.get("chunk_events"), r.get("batch_segments"))
            if k != (None, None):
                counts[k] = counts.get(k, 0) + 1
    return max(counts, key=counts.get) if counts else ()


def _verify_cell(entry: Mapping[str, Any], info: Mapping[str, Any]) -> str:
    """Return "" when the cell is admissible, else why it is not.

    "complete" in a registry is a writer's assertion, not provenance. Two things
    are checked: whether the run's own recorded batch/chunk match what the rest
    of this producer used (an OOM-degraded cell is not the same arm), and
    whether a declared checkpoint hash matches the file. Differing commits are
    flagged, not refused -- they need the producer to say whether the code
    changed materially, which is not decidable from here.
    """

    modal = _modal_training_config(entry)
    rec = _training_record(info)
    got = (rec.get("chunk_events"), rec.get("batch_segments"))
    if modal and got != (None, None) and got != modal:
        return (f"this cell trained at chunk_events/batch_segments {got}, while the rest "
                f"of this producer used {modal}; an OOM-degraded cell is not the same arm")
    declared = str(info.get("checkpoint_sha256", ""))
    ck = info.get("checkpoint")
    if declared and ck:
        p = Path(str(ck))
        if not p.exists():
            return f"declared checkpoint_sha256 but checkpoint missing: {p}"
        if _sha256(p) != declared:
            return "checkpoint_sha256 does not match the checkpoint on disk"
    return ""


def resolve_subject_arms(
    registry: Registry,
    subject: str,
    seed: str | None = None,
    verify: bool = True,
) -> dict[str, Arm]:
    """One :class:`Arm` per registered producer -- loaded, or reported as missing.

    ``verify`` defaults to True so a caller who forgets it gets the strict
    behaviour; pass False only for explicitly-labelled diagnostics, and the
    resulting arms carry ``verified=False`` so the output says so.
    """

    out: dict[str, Arm] = {}
    for pid, entry in registry.producers.items():
        common = dict(
            producer_id=pid,
            source_commit=str(entry.get("source_commit", "")),
            config_hash=str(entry.get("config_hash", "")),
            checkpoint_hash=str(entry.get("checkpoint_hash", "")),
        )
        subjects = entry.get("subjects", {}) or {}
        if subject not in subjects:
            out[pid] = Arm(status="not_available",
                           reason=f"subject {subject!r} not registered for this producer",
                           **common)
            continue
        chosen, info, why = _pick_seed(subjects[subject], seed)
        if info is None:
            out[pid] = Arm(status="not_available", reason=why, **common)
            continue
        ap = Path(str(info["anchor_state"]))
        if not ap.exists():
            out[pid] = Arm(status="not_available",
                           reason=f"anchor_state file missing: {ap}",
                           seed=chosen, anchor_path=str(ap), **common)
            continue
        if verify:
            why = _verify_cell(entry, info)
            if why:
                out[pid] = Arm(status="not_available", reason=why, seed=chosen,
                               anchor_path=str(ap), **common)
                continue
        rec = _training_record(info)
        flags = []
        commits = set()
        for v in (entry.get("subjects", {}) or {}).values():
            if isinstance(v, Mapping):
                for i in v.values():
                    if isinstance(i, Mapping) and "anchor_state" in i:
                        c = _training_record(i).get("commit")
                        if c:
                            commits.add(c)
        if len(commits) > 1:
            flags.append(f"producer spans {len(commits)} training commits while the "
                         f"registry advertises one")
        try:
            z = np.load(ap)
            out[pid] = Arm(
                status="ok", seed=chosen, anchor_path=str(ap), verified=bool(verify),
                commit=rec.get("commit", ""), chunk_events=rec.get("chunk_events"),
                batch_segments=rec.get("batch_segments"), provenance_flags=tuple(flags),
                state=z["state"], t_anchor=z["t_anchor"],
                split_index=z["split_index"] if "split_index" in z else None,
                session_id=z["session_id"] if "session_id" in z else None,
                **common,
            )
        except Exception as exc:  # unreadable payload is still "not available"
            out[pid] = Arm(status="not_available",
                           reason=f"anchor_state unreadable: {type(exc).__name__}: {exc}",
                           seed=chosen, anchor_path=str(ap), **common)
    return out
