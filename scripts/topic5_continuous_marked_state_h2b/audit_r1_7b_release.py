#!/usr/bin/env python3
"""Consumer-side acceptance audit for the R1.7B upstream that H2b v0.2 reads.

The v0.1 contract §5 admits an R1.7 release into H2b only when its
``reports/machine_audit.json`` is ``COMPLETE``, covers exactly 50 fits, keeps the
formal and sealed partitions closed, and has its source and checkpoint digests
recomputable from a pushed commit.  v0.2 does not read that release: it reads
R1.7B, the exploratory extension that lifted the five-per-dataset cap and grew
the cohort to 17 subjects x 5 seeds.  R1.7B has no machine audit of its own.

This script does not paper over that.  It re-verifies, from the consumer side and
without trusting the producer's own summary, everything that *can* be checked --
cell count, per-cell status, checkpoint digests, source revision, partition
flags, commit reachability, worktree isolation, and that no patient was filtered
by an H1 outcome -- and then states plainly which part of the §5 gate is not met
and what the resulting release label must be.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch as _torch  # noqa: F401; load the compatible native runtime first
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.topic5_continuous_marked_state_h2b.contract import (  # noqa: E402
    H2B_V0_2_REVISION,
    V0_2_RESULT_ROOT,
    atomic_json,
    sha256_file,
    utc_now,
)


EXPECTED_SUBJECTS = 17
EXPECTED_SEEDS = 5
EXPECTED_CELLS = EXPECTED_SUBJECTS * EXPECTED_SEEDS
ADMISSIBLE_CELL_STATUS = {"SCORED", "NONFINITE_GRADIENT"}
# The v0.1 §5 gate was written for R1.7A: a COMPLETE machine audit over 50 fits.
V0_1_STAGE2_REQUIRED_FITS = 50
RELEASE_LABEL = (
    "R1.7B exploratory development source; not formal H2b confirmation."
)
H1_TOKENS = ("h1", "stable", "unstable")


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.strip()


def _commit_is_pushed(commit: str) -> bool:
    remote = subprocess.run(
        ["git", "branch", "-r", "--contains", commit],
        cwd=REPO, capture_output=True, text=True,
    )
    return remote.returncode == 0 and bool(remote.stdout.strip())


def _audit_cells(inventory: dict) -> dict[str, Any]:
    """Re-verify every R1.7B cell from the files themselves, not the summary."""
    entries = inventory["entries"]
    _require(len(entries) == EXPECTED_CELLS,
             f"R1.7B must expose {EXPECTED_CELLS} cells, found {len(entries)}")
    by_subject: dict[str, set[int]] = {}
    statuses: dict[str, int] = {}
    revisions: set[str] = set()
    n_checkpoints = 0
    foreign_paths: list[str] = []
    for entry in entries:
        subject = str(entry["subject"])
        seed = int(entry["seed"])
        seeds = by_subject.setdefault(subject, set())
        _require(seed not in seeds, f"duplicate R1.7B cell {subject}/seed_{seed}")
        seeds.add(seed)
        status = str(entry["analysis_status"])
        _require(status in ADMISSIBLE_CELL_STATUS,
                 f"{subject}/seed_{seed}: unknown analysis status {status!r}")
        statuses[status] = statuses.get(status, 0) + 1
        revisions.add(str(entry["source_revision"]))
        for key in ("formal_test_partition_opened", "sealed_opened"):
            _require(entry.get(key) is False,
                     f"{subject}/seed_{seed}: {key} is not false")
        _require(entry.get("state_source_uses_seizure_labels") is False,
                 f"{subject}/seed_{seed}: the state source read seizure labels")
        _require(entry.get("seizure_gradient_path") is False,
                 f"{subject}/seed_{seed}: a seizure gradient path is recorded")
        for key in ("checkpoint_path", "result_path"):
            value = entry.get(key)
            if isinstance(value, str) and value.startswith("/tmp/"):
                foreign_paths.append(value)
        if entry.get("checkpoint_available"):
            n_checkpoints += 1
            for key in ("checkpoint", "result"):
                path = Path(str(entry[f"{key}_path"]))
                _require(path.is_file(), f"{subject}/seed_{seed}: missing {key}")
                observed = sha256_file(path)
                _require(observed == str(entry[f"{key}_sha256"]),
                         f"{subject}/seed_{seed}: {key} digest drift ({observed})")
        else:
            _require(status == "NONFINITE_GRADIENT",
                     f"{subject}/seed_{seed}: no checkpoint but status is {status!r}")
    _require(not foreign_paths,
             f"R1.7B cells point into a temporary worktree: {foreign_paths[:3]}")
    _require(len(by_subject) == EXPECTED_SUBJECTS,
             f"R1.7B must cover {EXPECTED_SUBJECTS} subjects, found {len(by_subject)}")
    off_size = {
        subject: sorted(seeds) for subject, seeds in by_subject.items()
        if len(seeds) != EXPECTED_SEEDS
    }
    _require(not off_size, f"R1.7B subjects without {EXPECTED_SEEDS} seeds: {off_size}")
    _require(len(revisions) == 1,
             f"R1.7B mixes source revisions: {sorted(revisions)}")
    return {
        "n_cells": len(entries),
        "n_subjects": len(by_subject),
        "seeds_per_subject": EXPECTED_SEEDS,
        "analysis_status_counts": dict(sorted(statuses.items())),
        "n_checkpoint_available_cells": n_checkpoints,
        "n_instrument_failure_cells": len(entries) - n_checkpoints,
        "source_revision": sorted(revisions)[0],
        "all_checkpoint_and_result_digests_recomputed": True,
        "cells_reference_a_temporary_worktree": False,
    }


def _audit_h1_is_not_a_filter(root: Path, inventory: dict) -> dict[str, Any]:
    """H1 stability is a stratum, never an entry gate (v0.2 contract §2)."""
    census = pd.read_csv(root / "manifests/patient_support_census.csv")
    stable = set(map(str, inventory.get("h1_stable_subjects") or []))
    probed = sorted(
        directory.name for directory in (root / "risk_sets").iterdir()
        if directory.is_dir() and (directory / "input_manifest.json").is_file()
    )
    probed_stable = sorted(set(probed) & stable)
    probed_unstable = sorted(set(probed) - stable)
    _require(bool(probed_stable) and bool(probed_unstable),
             "the run kept only one H1 stratum, which would make H1 an entry gate")
    reasons = {}
    for row in census.itertuples(index=False):
        subject = str(row.subject)
        if subject in probed:
            continue
        reason = str(row.exclusion_or_deferred_reason)
        reasons[subject] = reason
        lowered = reason.lower()
        _require(not any(token in lowered for token in H1_TOKENS),
                 f"{subject}: excluded for an H1-flavoured reason {reason!r}")
    return {
        "h1_used_as_entry_gate": False,
        "n_probed_subjects": len(probed),
        "probed_h1_stable": probed_stable,
        "probed_h1_unstable": probed_unstable,
        "excluded_subject_reasons": dict(sorted(reasons.items())),
        "exclusion_reasons_are_instrumental_not_outcome_based": True,
    }


def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    inventory_path = root / "manifests/r1_7_checkpoint_inventory.json"
    inventory = _json(inventory_path)
    _require(inventory.get("status") == "COMPLETE",
             "the consumed R1.7 checkpoint inventory is not COMPLETE")
    _require(inventory.get("revision") == H2B_V0_2_REVISION,
             "checkpoint inventory revision drift")

    release = inventory["source_release"]
    source_root = Path(str(inventory["source_root"]))
    _require(not str(source_root).startswith("/tmp/"),
             f"the R1.7B source root is a temporary worktree: {source_root}")
    for name in ("cohort_inventory", "queue_status", "summary"):
        path = Path(str(release[name]))
        _require(path.is_file(), f"R1.7B {name} is missing: {path}")
        observed = sha256_file(path)
        _require(observed == str(release[f"{name}_sha256"]),
                 f"R1.7B {name} digest drift: {observed}")

    cohort = _json(Path(str(release["cohort_inventory"])))
    _require(cohort.get("status") == "FROZEN", "R1.7B cohort inventory is not FROZEN")
    _require(cohort.get("selection_uses_model_outcomes") is False,
             "R1.7B selected its cohort using model outcomes")
    _require(int(cohort.get("n_selected", -1)) == EXPECTED_SUBJECTS,
             "R1.7B selected cohort size drift")
    for key in ("formal_test_partition_opened", "sealed_opened"):
        _require(cohort.get(key) is False, f"R1.7B cohort inventory: {key} is not false")

    queue = _json(Path(str(release["queue_status"])))
    _require(queue.get("status") == "COMPLETE", "the R1.7B queue did not complete")
    _require(int(queue.get("scheduled_cells", -1)) == EXPECTED_CELLS,
             "R1.7B scheduled cell drift")
    _require(int(queue.get("n_subjects", -1)) == EXPECTED_SUBJECTS,
             "R1.7B subject count drift")
    _require(int(queue.get("n_seeds", -1)) == EXPECTED_SEEDS, "R1.7B seed count drift")

    cells = _audit_cells(inventory)
    h1 = _audit_h1_is_not_a_filter(root, inventory)

    formal_audit_path = source_root / "reports/machine_audit.json"
    formal_present = formal_audit_path.is_file()
    r1_7a_audit_path = (
        source_root.parent / "r1_7a/reports/machine_audit.json"
    )
    r1_7a: dict[str, Any] = {"present": r1_7a_audit_path.is_file()}
    if r1_7a["present"]:
        payload = _json(r1_7a_audit_path)
        r1_7a.update({
            "status": payload.get("status"),
            "n_r1_fits": payload.get("n_r1_fits"),
            "n_subjects": payload.get("n_subjects"),
            "meets_v0_1_stage2_gate": bool(
                payload.get("status") == "COMPLETE"
                and int(payload.get("n_r1_fits", -1)) == V0_1_STAGE2_REQUIRED_FITS
                and payload.get("formal_test_partition_opened") is False
                and payload.get("sealed_opened") is False
            ),
        })

    frozen_ten = set(map(str, cohort.get("frozen_r1_7a_subjects") or []))
    probed = set(h1["probed_h1_stable"]) | set(h1["probed_h1_unstable"])
    commits = []
    for line in _git("log", "--all", "--format=%H%x1f%s").splitlines():
        commit, _, subject = line.partition("\x1f")
        if "r1.7b" in subject.lower() or "r1_7b" in subject.lower():
            commits.append({
                "commit": commit, "subject": subject,
                "pushed": _commit_is_pushed(commit),
            })
    _require(bool(commits), "no commit records the R1.7B extension")
    _require(all(row["pushed"] for row in commits),
             f"unpushed R1.7B commits: {[r['commit'][:8] for r in commits if not r['pushed']]}")
    head = _git("rev-parse", "HEAD")

    payload = {
        "status": "PASS_EXPLORATORY_DEVELOPMENT_SOURCE",
        "revision": "h2b_v0_2_r1_7b_consumer_acceptance_audit_v1",
        "created_utc": utc_now(),
        "result_root": str(root),
        "source_root": str(source_root),
        "release_label": RELEASE_LABEL,
        "cells": cells,
        "h1_stratification": h1,
        "v0_1_stage2_release_gate": {
            "requirement": (
                "v0.1 contract §5: R1.7 enters H2b only on a COMPLETE machine "
                f"audit covering exactly {V0_1_STAGE2_REQUIRED_FITS} fits"
            ),
            "consumed_release": "R1.7B",
            "r1_7b_machine_audit_present": formal_present,
            "r1_7b_n_fits": EXPECTED_CELLS,
            "gate_met_by_consumed_release": bool(
                formal_present and EXPECTED_CELLS == V0_1_STAGE2_REQUIRED_FITS
            ),
            "formally_audited_sibling_release": r1_7a,
            "n_probed_subjects_inside_the_audited_r1_7a_ten": len(probed & frozen_ten),
            "n_probed_subjects_from_the_exploratory_extension": len(
                probed - frozen_ten
            ),
            "probed_subjects_from_the_exploratory_extension": sorted(
                probed - frozen_ten
            ),
            "weakening_is_declared_not_silent": True,
            "consequence": (
                "H2b v0.2 may report development transfer only; it may not be "
                "reported as formal H2b confirmation"
            ),
        },
        "provenance": {
            "r1_7b_artifacts_are_git_tracked": False,
            "artifact_provenance_mechanism": "sha256 recomputation, not git tracking",
            "r1_7b_commits": commits,
            "h2b_head_commit": head,
            "h2b_head_pushed": _commit_is_pushed(head),
            "r1_7_worktree_path": "/tmp/hfosp_r17_20260827",
            "r1_7_worktree_still_present": Path("/tmp/hfosp_r17_20260827").exists(),
            "no_uncommitted_r1_7_worktree_file_was_read": True,
        },
        "development_only": True,
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "h3_or_t2_run": False,
        "paper_ready_figures_modified": False,
    }
    atomic_json(root / "reports/r1_7b_consumer_acceptance_audit.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=V0_2_RESULT_ROOT)
    args = parser.parse_args()
    try:
        payload = run(args.result_root)
    except Exception as exc:
        root = args.result_root.resolve()
        atomic_json(root / "reports/r1_7b_consumer_acceptance_audit.json", {
            "status": "FAIL",
            "revision": "h2b_v0_2_r1_7b_consumer_acceptance_audit_v1",
            "created_utc": utc_now(), "result_root": str(root),
            "error": repr(exc), "release_label": RELEASE_LABEL,
            "formal_test_partition_opened": False, "sealed_opened": False,
        })
        raise
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
