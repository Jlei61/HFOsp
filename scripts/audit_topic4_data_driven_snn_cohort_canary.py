#!/usr/bin/env python3
"""Audit the multisubject canary against its frozen acceptance gate.

The canary is a capacity preflight, not a cohort result.  Its job is to show
that the shared morphology can be simulated and read through montages of very
different contact counts without the detector quietly failing on the small
ones, and that the produced figures actually exist on disk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "config/topic4_data_driven_snn_cohort_canary_v1.json"
MINIMUM_MEDIAN_RECRUITED_CONTACTS = 2.0
MINIMUM_READABLE_EVENT_FRACTION = 0.25


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(config_path: Path, expected_commit: str) -> dict:
    config = json.loads(config_path.read_text())
    output_root = ROOT / config["output_root"]
    manifest = json.loads((output_root / "candidate_manifest.json").read_text())
    candidates = [row["candidate_id"] for row in manifest["candidate_set"]["candidates"]]
    seeds = [int(seed) for seed in config["search"]["fit_network_seeds"]]

    workers, provenance_failures, runaway = [], [], []
    per_subject: dict[str, list[dict]] = {}
    for candidate in candidates:
        for seed in seeds:
            stem = output_root / "workers" / f"{candidate}_seed_{seed}"
            json_path, npz_path = stem.with_suffix(".json"), stem.with_suffix(".npz")
            if not json_path.exists():
                provenance_failures.append({"worker": stem.name, "why": "missing"})
                continue
            payload = json.loads(json_path.read_text())
            record = payload.get("provenance", {})
            problems = []
            if record.get("expected_git_commit") != expected_commit:
                problems.append("expected_commit")
            if not record.get("runtime_modules_match_expected_commit"):
                problems.append("modules_do_not_match_commit")
            if record.get("runtime_modules_dirty"):
                problems.append("dirty_runtime")
            if payload.get("output_npz_sha256") != _sha256(npz_path):
                problems.append("npz_hash")
            if problems:
                provenance_failures.append({"worker": stem.name, "why": problems})
            if payload.get("runaway"):
                runaway.append({
                    "worker": stem.name,
                    "stopped_at_ms": payload.get("runaway_early_stop_ms"),
                })
            workers.append({
                "candidate_id": candidate, "seed": seed,
                "status": payload.get("status"),
                "n_detected_events": payload.get("n_detected_events"),
                "wall_seconds": payload.get("wall_seconds"),
            })
            for subject in payload.get("subjects", []):
                detected = int(subject["n_detected_events"])
                readable = int(subject["n_events_with_minimum_contacts"])
                per_subject.setdefault(subject["subject_id"], []).append({
                    "candidate_id": candidate, "seed": seed,
                    "n_contacts": len(subject["contact_names"]),
                    "n_detected_events": detected,
                    "n_readable_events": readable,
                    "readable_fraction": readable / detected if detected else 0.0,
                    "median_recruited_contacts": subject["median_recruited_contacts"],
                })

    support = []
    for subject_id, rows in sorted(per_subject.items()):
        medians = [
            row["median_recruited_contacts"] for row in rows
            if row["median_recruited_contacts"] is not None
        ]
        fractions = [row["readable_fraction"] for row in rows]
        support.append({
            "subject_id": subject_id,
            "n_contacts": rows[0]["n_contacts"],
            "median_recruited_contacts": float(np.median(medians)) if medians else None,
            "median_readable_event_fraction": float(np.median(fractions)),
            "detector_usable": bool(
                medians and float(np.median(medians)) >= MINIMUM_MEDIAN_RECRUITED_CONTACTS
                and float(np.median(fractions)) >= MINIMUM_READABLE_EVENT_FRACTION
            ),
        })
    contacts = np.asarray([row["n_contacts"] for row in support], float)
    usable = np.asarray([row["detector_usable"] for row in support])
    smallest = [row for row in support if row["n_contacts"] <= 8]

    selection_path = output_root / "fit_selection.json"
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else None
    figures = output_root / "figures"
    figure_files = {
        name: {
            "exists": (figures / name).exists(),
            "bytes": (figures / name).stat().st_size if (figures / name).exists() else 0,
            "sha256": _sha256(figures / name) if (figures / name).exists() else None,
        }
        for name in (
            "data_driven_snn_cohort_canary_fit.png",
            "data_driven_snn_cohort_canary_fit.pdf",
            "README.md",
        )
    }

    gates = {
        "all_workers_present_and_frozen": not provenance_failures,
        "no_runaway_workers": not runaway,
        "figures_rendered": all(row["exists"] and row["bytes"] > 0
                                for row in figure_files.values()),
        "detector_usable_on_the_smallest_montages": bool(
            smallest and all(row["detector_usable"] for row in smallest)
        ),
        "minimum_evaluable_subjects_met": (
            selection is not None and selection["status"] == "CANARY_FIT_EVALUABLE"
        ),
    }
    status = "CANARY_ACCEPTED" if all(gates.values()) else "CANARY_GATE_NOT_MET"
    return {
        "schema_version": "topic4_data_driven_snn_cohort_canary_audit_v1",
        "status": status,
        "gates": gates,
        "scientific_boundary": (
            "The canary only clears capacity, detector behaviour and memory. It "
            "shares five subjects with the formal cohort, so its alignment scores "
            "may not prune the formal candidate library and it is not a cohort "
            "result."
        ),
        "n_workers": len(workers),
        "provenance_failures": provenance_failures,
        "runaway_workers": runaway,
        "event_support_by_subject": support,
        "contact_count_vs_usability": {
            "n_subjects": len(support),
            "n_usable": int(usable.sum()),
            "smallest_montage_contacts": (
                float(contacts.min()) if len(contacts) else None
            ),
            "largest_montage_contacts": (
                float(contacts.max()) if len(contacts) else None
            ),
        },
        "selection_status": None if selection is None else selection["status"],
        "selection_denominators": None if selection is None else selection["denominators"],
        "figures": figure_files,
        "expected_commit": expected_commit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    payload = audit(args.config.resolve(), args.expected_commit)
    config = json.loads(args.config.read_text())
    output = ROOT / config["output_root"] / "canary_acceptance_audit.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "gates": payload["gates"],
        "selection_status": payload["selection_status"],
        "event_support_by_subject": payload["event_support_by_subject"],
    }, indent=2))


if __name__ == "__main__":
    main()
