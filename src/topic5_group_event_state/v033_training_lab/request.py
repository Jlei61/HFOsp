"""Atomic training-request interface (design §2; handoff §2).

Contract clauses honoured here (plan Task 1):
  [Q1] any of the 13 required fields missing / empty -> INVALID_REQUEST, all listed;
  [Q2] objective not in the trainable registry -> INVALID_REQUEST (returned to Agent C, never guessed);
  [Q3] gated_exploratory without ``exploratory_approved`` -> INVALID_REQUEST;
  [Q4] human input view (R0/R1) without an execution release -> HELD_NO_RELEASE; toy/synthetic -> PENDING;
  [Q5] code_commit != HEAD -> HELD_CODE_COMMIT_MISMATCH (checked before the release gate);
  [Q6] job key = sha256 over target / input_view / state_family / subject / seed / split / config /
       code / input hash, insensitive to dict ordering;
  [Q8] split_hash / input_hash / baseline_H bins disagreeing with the data view -> HELD_MISMATCH.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from .paths import payload_hash

REQUIRED_FIELDS: tuple[str, ...] = (
    "request_id",
    "scientific_target",
    "input_view",
    "state_family",
    "split_hash",
    "baseline_H",
    "endpoint_and_reduction",
    "search_budget",
    "seed_policy",
    "resource_ceiling",
    "code_commit",
    "input_hash",
    "requested_by",
)
STATE_FAMILIES = ("fixed_leaky", "gated_exploratory")
HUMAN_VIEW_KINDS = ("R0", "R1")
VIEW_KINDS = ("toy", "synthetic") + HUMAN_VIEW_KINDS


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    OOM_RETRYABLE = "OOM_RETRYABLE"
    RESOURCE_UNRESOLVED = "RESOURCE_UNRESOLVED"
    NAN = "NAN"
    INVALID_REQUEST = "INVALID_REQUEST"
    HELD_NO_RELEASE = "HELD_NO_RELEASE"
    HELD_MISMATCH = "HELD_MISMATCH"
    HELD_CODE_COMMIT_MISMATCH = "HELD_CODE_COMMIT_MISMATCH"
    SKIPPED_EXISTING = "SKIPPED_EXISTING"
    STALE = "STALE"


@dataclass(frozen=True)
class JobRequest:
    request_id: str
    scientific_target: dict[str, Any]
    input_view: dict[str, Any]
    state_family: str
    split_hash: str
    baseline_H: dict[str, Any]
    endpoint_and_reduction: dict[str, Any]
    search_budget: dict[str, Any]
    seed_policy: dict[str, Any]
    resource_ceiling: dict[str, Any]
    code_commit: str
    input_hash: str
    requested_by: str
    exploratory_approved: bool = False
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def objective(self) -> str:
        return str(self.scientific_target.get("objective", ""))

    @property
    def subject(self) -> str | None:
        value = self.input_view.get("subject")
        return None if value is None else str(value)


def _empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (dict, list, tuple)):
        return len(value) == 0
    return False


def parse_request(payload: Mapping[str, Any]) -> tuple[JobRequest | None, dict[str, Any]]:
    """[Q1] Structural check only; returns ``(None, verdict)`` when fields are missing."""

    missing = [name for name in REQUIRED_FIELDS if _empty(payload.get(name))]
    if missing:
        return None, {"status": JobStatus.INVALID_REQUEST.value,
                      "reasons": [f"missing or empty required field {name!r}" for name in missing],
                      "missing_fields": missing}
    kwargs = {name: payload[name] for name in REQUIRED_FIELDS}
    request = JobRequest(**kwargs, exploratory_approved=bool(payload.get("exploratory_approved", False)),
                         raw=dict(payload))
    return request, {"status": JobStatus.PENDING.value, "reasons": [], "missing_fields": []}


def is_human_view(input_view: Mapping[str, Any]) -> bool:
    return str(input_view.get("kind", "")) in HUMAN_VIEW_KINDS


def validate_request(
    payload: Mapping[str, Any],
    *,
    registered_objectives: Sequence[str],
    release_present: bool,
    head_commit: str,
) -> dict[str, Any]:
    """Full verdict with fixed precedence: INVALID > HELD_CODE_COMMIT_MISMATCH > HELD_NO_RELEASE > PENDING."""

    request, verdict = parse_request(payload)
    if request is None:
        return verdict
    reasons: list[str] = []
    if request.objective not in tuple(registered_objectives):                        # [Q2]
        reasons.append(f"objective {request.objective!r} is not registered "
                       f"(registered: {sorted(registered_objectives)}); returned to Agent C")
    if request.state_family not in STATE_FAMILIES:
        reasons.append(f"state_family {request.state_family!r} not in {STATE_FAMILIES}")
    if request.state_family == "gated_exploratory" and not request.exploratory_approved:  # [Q3]
        reasons.append("gated_exploratory requires exploratory_approved=true in the request")
    kind = str(request.input_view.get("kind", ""))
    if kind not in VIEW_KINDS:
        reasons.append(f"input_view.kind {kind!r} not in {VIEW_KINDS}")
    if is_human_view(request.input_view) and _empty(request.input_view.get("subject")):
        reasons.append("human input_view needs a subject")
    if reasons:
        return {"status": JobStatus.INVALID_REQUEST.value, "reasons": reasons, "missing_fields": []}
    if str(request.code_commit) != str(head_commit):                                  # [Q5]
        return {"status": JobStatus.HELD_CODE_COMMIT_MISMATCH.value,
                "reasons": [f"request code_commit {request.code_commit} != HEAD {head_commit}"],
                "missing_fields": []}
    if is_human_view(request.input_view) and not release_present:                     # [Q4]
        return {"status": JobStatus.HELD_NO_RELEASE.value,
                "reasons": ["human input view but V0_3_3_EXECUTION_RELEASE.json is absent"],
                "missing_fields": []}
    return {"status": JobStatus.PENDING.value, "reasons": [], "missing_fields": []}


def hash_mismatch_verdict(
    request: JobRequest,
    *,
    split_hash: str,
    input_hash: str,
    missing_h_bins: Sequence[int],
) -> dict[str, Any]:
    """[Q8] Compare the request against the data view actually built for it."""

    reasons: list[str] = []
    if str(request.split_hash) != str(split_hash):
        reasons.append(f"split_hash {request.split_hash} != data view {split_hash}")
    if str(request.input_hash) != str(input_hash):
        reasons.append(f"input_hash {request.input_hash} != data view {input_hash}")
    if len(missing_h_bins) > 0:
        reasons.append(f"baseline_H has no log_mu_H for bins {list(missing_h_bins)}; returned to Agent A/C")
    if reasons:
        return {"status": JobStatus.HELD_MISMATCH.value, "reasons": reasons, "missing_fields": []}
    return {"status": JobStatus.PENDING.value, "reasons": [], "missing_fields": []}


def job_key(request: JobRequest, *, subject: str | None, seed: int, config_hash: str) -> str:
    """[Q6] Deterministic identity of one training unit (supervisor runbook §6)."""

    return payload_hash({
        "scientific_target": request.scientific_target,
        "input_view": request.input_view,
        "state_family": request.state_family,
        "subject": subject,
        "seed": int(seed),
        "split_hash": request.split_hash,
        "config_hash": config_hash,
        "code_commit": request.code_commit,
        "input_hash": request.input_hash,
    })
