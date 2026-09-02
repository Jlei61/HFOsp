"""Atomic training-request interface (design §2; handoff §2).

Contract clauses honoured here (plan Task 1):
  [Q1] any of the 13 required fields missing / empty -> INVALID_REQUEST, all listed;
  [Q2] objective not in the trainable registry -> INVALID_REQUEST (returned to Agent C, never guessed);
  [Q3] gated_exploratory without ``exploratory_approved`` -> INVALID_REQUEST;
  [Q4] human input view (R0/R1) without an execution release -> HELD_NO_RELEASE; toy/synthetic -> PENDING;
  [Q5] science_code_commit identifies Agent C's immutable target/input producer;
       trainer_code_commit is recorded independently by Agent B;
  [Q6] job key = sha256 over target / input_view / state architecture / subject / seed / split / config /
       science + trainer commits / input hash, insensitive to dict ordering;
  [Q8] split_hash / input_hash / baseline_H bins disagreeing with the data view -> HELD_MISMATCH.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Mapping, Sequence

from .paths import payload_hash

REQUIRED_FIELDS: tuple[str, ...] = (
    "request_id",
    "scientific_target",
    "input_view",
    "state_architecture",
    "split_hash",
    "baseline_H",
    "endpoint_and_reduction",
    "search_budget",
    "seed_policy",
    "resource_ceiling",
    "science_code_commit",
    "input_hash",
    "requested_by",
)
STATE_FAMILIES = ("fixed_leaky", "gated_exploratory")
PREDICTIVE_VIEWS = ("S_N", "S_G_PRIMARY", "S_G_COMPOSITE")
HUMAN_VIEW_KINDS = ("R0", "R1")
VIEW_KINDS = ("toy", "synthetic") + HUMAN_VIEW_KINDS
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")


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
    state_architecture: str
    split_hash: str
    baseline_H: dict[str, Any]
    endpoint_and_reduction: dict[str, Any]
    search_budget: dict[str, Any]
    seed_policy: dict[str, Any]
    resource_ceiling: dict[str, Any]
    science_code_commit: str
    input_hash: str
    requested_by: str
    exploratory_approved: bool = False
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def objective(self) -> str:
        return str(self.scientific_target.get("objective", "")) if isinstance(self.scientific_target, dict) else ""

    @property
    def subject(self) -> str | None:
        value = self.input_view.get("subject") if isinstance(self.input_view, dict) else None
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
    try:
        kwargs = {name: payload[name] for name in REQUIRED_FIELDS}
        request = JobRequest(**kwargs, exploratory_approved=bool(payload.get("exploratory_approved", False)),
                             raw=dict(payload))
    except (TypeError, ValueError) as exc:
        return None, {"status": JobStatus.INVALID_REQUEST.value,
                      "reasons": [f"malformed nested request fields: {type(exc).__name__}: {exc}"],
                      "missing_fields": []}
    return request, {"status": JobStatus.PENDING.value, "reasons": [], "missing_fields": []}


def is_human_view(input_view: Mapping[str, Any]) -> bool:
    return isinstance(input_view, Mapping) and str(input_view.get("kind", "")) in HUMAN_VIEW_KINDS


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
    if not isinstance(request.scientific_target, dict) or not isinstance(request.input_view, dict):
        reasons.append("scientific_target and input_view must be objects")
    predictive_view = str(request.scientific_target.get("predictive_view", ""))
    if predictive_view not in PREDICTIVE_VIEWS:
        reasons.append(f"scientific_target.predictive_view {predictive_view!r} not in {PREDICTIVE_VIEWS}")
    if request.state_architecture not in STATE_FAMILIES:
        reasons.append(f"state_architecture {request.state_architecture!r} not in {STATE_FAMILIES}")
    if request.state_architecture == "gated_exploratory" and not request.exploratory_approved:  # [Q3]
        reasons.append("gated_exploratory requires exploratory_approved=true in the request")
    kind = str(request.input_view.get("kind", "")) if isinstance(request.input_view, dict) else ""
    if kind not in VIEW_KINDS:
        reasons.append(f"input_view.kind {kind!r} not in {VIEW_KINDS}")
    if is_human_view(request.input_view) and _empty(request.input_view.get("subject")):
        reasons.append("human input_view needs a subject")
    if is_human_view(request.input_view) and _empty(request.input_view.get("data_registry_key")):
        reasons.append("human input_view needs a data_registry_key")
    if not isinstance(request.baseline_H, dict) or request.baseline_H.get("name") != "H_mark" \
            or not HEX64.fullmatch(str(request.baseline_H.get("hash", ""))):
        reasons.append("baseline_H must name H_mark and carry an immutable SHA256 hash")
    if not HEX40.fullmatch(str(request.science_code_commit)):
        reasons.append("science_code_commit must be a full git SHA")
    if str(request.split_hash) != "auto" and not HEX64.fullmatch(str(request.split_hash)):
        reasons.append("split_hash must be a SHA256 (or auto for local toy smoke only)")
    if str(request.input_hash) != "auto" and not HEX64.fullmatch(str(request.input_hash)):
        reasons.append("input_hash must be a SHA256 (or auto for local toy smoke only)")
    if str(request.requested_by) != "agent_c":
        reasons.append("requested_by must be agent_c")
    if request.raw.get("schema_version") != "v2":
        reasons.append("schema_version must be v2")
    if request.raw.get("sealed") is not False:
        reasons.append("request must explicitly declare sealed=false")
    if request.input_view.get("materialized_arrays_only") is True and \
            request.scientific_target.get("bin_convention") != "left_closed_right_open_[t+a,t+b)":
        reasons.append("materialized count targets must use canonical [t+a,t+b) bins")
    if reasons:
        return {"status": JobStatus.INVALID_REQUEST.value, "reasons": reasons, "missing_fields": []}
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


def job_key(
    request: JobRequest, *, subject: str | None, seed: int, config_hash: str,
    trainer_code_commit: str,
) -> str:
    """[Q6] Deterministic identity of one training unit (supervisor runbook §6)."""

    return payload_hash({
        "scientific_target": request.scientific_target,
        "input_view": request.input_view,
        "state_architecture": request.state_architecture,
        "subject": subject,
        "seed": int(seed),
        "split_hash": request.split_hash,
        "config_hash": config_hash,
        "science_code_commit": request.science_code_commit,
        "trainer_code_commit": str(trainer_code_commit),
        "input_hash": request.input_hash,
    })
