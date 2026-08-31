"""Fail-closed constants and provenance helpers for H2b cross-task transfer.

This module contains no seizure model and no state training.  It centralises the
paths and invariants that every Phase 0--3 producer must share.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping


H2B_REVISION = "continuous_marked_state_h2b_cross_task_v0_1"
H2B_V0_2_REVISION = "continuous_marked_state_h2b_cross_task_v0_2"
H2B_V0_3_REVISION = "continuous_marked_state_h2b_cross_task_v0_3"
H2B_V0_4_REVISION = "continuous_marked_state_h2b_cross_task_v0_4"
LEAD_MINUTES = (5, 15, 30, 60, 120)
PRIMARY_LEAD_MINUTES = 30
POSTICTAL_GUARD_MINUTES = 120
FORMAL_TEST_PARTITION_OPENED = False
SEALED_OPENED = False
DEVELOPMENT_ONLY = True

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = (
    REPO_ROOT
    / "results/epi_prssm/continuous_marked_state/h2b_cross_task/v0_1"
)
V0_2_RESULT_ROOT = (
    REPO_ROOT
    / "results/epi_prssm/continuous_marked_state/h2b_cross_task/v0_2"
)
V0_3_RESULT_ROOT = (
    REPO_ROOT
    / "results/epi_prssm/continuous_marked_state/h2b_cross_task/v0_3"
)
V0_4_RESULT_ROOT = (
    REPO_ROOT
    / "results/epi_prssm/continuous_marked_state/h2b_cross_task/v0_4"
)
CANONICAL_V0_2_RESULT_ROOT = Path(
    "/home/honglab/leijiaxin/HFOsp/results/epi_prssm/continuous_marked_state/"
    "h2b_cross_task/v0_2"
)
CANONICAL_V0_3_RESULT_ROOT = Path(
    "/home/honglab/leijiaxin/HFOsp/results/epi_prssm/continuous_marked_state/"
    "h2b_cross_task/v0_3"
)
CANONICAL_V0_4_RESULT_ROOT = Path(
    "/home/honglab/leijiaxin/HFOsp/results/epi_prssm/continuous_marked_state/"
    "h2b_cross_task/v0_4"
)
R1_6_ROOT = Path(
    "/home/honglab/leijiaxin/HFOsp/results/epi_prssm/continuous_marked_state/"
    "r1/optimizer_identifiability_r1_6"
)
R1_6_MACHINE_AUDIT = REPO_ROOT / (
    "docs/archive/topic5/"
    "continuous_marked_state_optimizer_identifiability_r1_6_machine_audit_"
    "2026-08-27.json"
)
R1_7_WORKTREE = Path("/tmp/hfosp_r17_20260827")
R1_7_MACHINE_AUDIT = R1_7_WORKTREE / (
    "results/epi_prssm/continuous_marked_state/r1/r1_7a/reports/"
    "machine_audit.json"
)

SUPPORT_TIERS = (
    "primary_chronological",
    "sensitivity_loso",
    "descriptive_case_series",
    "not_estimable",
)
EVIDENCE_LAYERS = ("checkpoint_available", "h1_stable")
PROBE_ARMS = (
    "B_history",
    "B_observation",
    "B_state",
    "memoryless",
    "wrong_time",
)


@dataclass(frozen=True)
class RunBoundary:
    revision: str = H2B_REVISION
    development_only: bool = DEVELOPMENT_ONLY
    formal_test_partition_opened: bool = FORMAL_TEST_PARTITION_OPENED
    sealed_opened: bool = SEALED_OPENED
    seizure_loss_updates_state: bool = False
    state_source_uses_seizure_labels: bool = False
    h3_or_t2_run: bool = False
    physical_clock_run: bool = False
    paper_ready_figures_modified: bool = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path | str) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_safe_output_path(path: Path | str) -> Path:
    """Reject writes outside an explicitly versioned H2b result root."""
    target = Path(path).resolve()
    roots = (
        RESULT_ROOT.resolve(), V0_2_RESULT_ROOT.resolve(),
        CANONICAL_V0_2_RESULT_ROOT.resolve(),
        V0_3_RESULT_ROOT.resolve(), CANONICAL_V0_3_RESULT_ROOT.resolve(),
        V0_4_RESULT_ROOT.resolve(), CANONICAL_V0_4_RESULT_ROOT.resolve(),
    )
    if not any(target == root or root in target.parents for root in roots):
        raise ValueError(f"H2b output escapes isolated result root: {target}")
    if R1_7_WORKTREE.resolve() == target or R1_7_WORKTREE.resolve() in target.parents:
        raise ValueError("H2b must never write into the R1.7 worktree")
    return target


def _atomic_text(path: Path, text: str) -> None:
    target = assert_safe_output_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path | str, payload: Mapping[str, Any] | list[Any]) -> None:
    _atomic_text(Path(path), json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_csv(
    path: Path | str,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Iterable[str] | None = None,
) -> None:
    materialised = [dict(row) for row in rows]
    names = list(fieldnames or (materialised[0].keys() if materialised else ()))
    target = assert_safe_output_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=names, extrasaction="raise")
            writer.writeheader()
            writer.writerows(materialised)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run_contract_payload() -> dict[str, Any]:
    return {
        "status": "FROZEN",
        "created_utc": utc_now(),
        "boundary": asdict(RunBoundary()),
        "lead_minutes": list(LEAD_MINUTES),
        "primary_lead_minutes": PRIMARY_LEAD_MINUTES,
        "postictal_guard_minutes": POSTICTAL_GUARD_MINUTES,
        "support_tiers": list(SUPPORT_TIERS),
        "evidence_layers": list(EVIDENCE_LAYERS),
        "probe_arms": list(PROBE_ARMS),
        "primary_estimand": (
            "held-out 30-min conditional log loss: B_state - B_observation; "
            "negative favours the frozen persistent state"
        ),
        "patient_first": True,
        "seed_is_patient_replicate": False,
        "r1_7_requires_complete_machine_audit": True,
    }


def support_tier(n_eligible_seizures: int) -> str:
    n = int(n_eligible_seizures)
    if n >= 10:
        return "primary_chronological"
    if n >= 5:
        return "sensitivity_loso"
    if n >= 2:
        return "descriptive_case_series"
    return "not_estimable"


def validate_lead_minutes(values: Iterable[int]) -> tuple[int, ...]:
    leads = tuple(int(value) for value in values)
    if leads != LEAD_MINUTES:
        raise ValueError(f"lead-time contract drift: expected {LEAD_MINUTES}, got {leads}")
    return leads
