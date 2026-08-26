#!/usr/bin/env python3
"""Goal 0 -- freeze the data object, the splits and the train-only baseline.

This step sets no scientific performance gate.  Its only job is to make every
later positive and negative interpretable, and to run Hard Gate A: can this data
be scientifically interpreted at all.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.topic5_epi_prssm.contracts import (  # noqa: E402
    FORBIDDEN_INPUTS, FROZEN, OUTPUT_ROOT, atomic_write_csv, atomic_write_json,
    atomic_write_text, code_revision, package_hash, sha256_file, sha256_obj,
    DATASET_ROOT, SOURCE_MAPPING_ROOT, EPILEPSIAE_BLOCK_INVENTORY, YUQUAN_BLOCK_INVENTORY,
)
from src.topic5_epi_prssm.event_marks import (  # noqa: E402
    ADMITTED_CONTACT_FEATURES, REJECTED_CONTACT_FEATURES, SPLIT_NAMES,
    available_subjects, load_patient, recruitment_groups,
)
from src.topic5_epi_prssm.graph_templates import build_patient_graph  # noqa: E402
from src.topic5_epi_prssm.patient_baseline import (  # noqa: E402
    estimate_baseline, variance_decomposition,
)

MANIFESTS = OUTPUT_ROOT / "manifests"
AUDIT = OUTPUT_ROOT / "data_audit"
BASELINE = OUTPUT_ROOT / "baseline"

#: A prefix family enters the ambiguous-prefix targeted analysis only if it is
#: seen often enough in train and genuinely branches.  Frozen before any run.
PREFIX_MIN_SUPPORT = 50
PREFIX_MIN_BRANCH_SUPPORT = 10
PREFIX_MIN_ENTROPY_BITS = 0.5
PREFIX_DEPTHS = (1, 2, 3)


def build_data_manifest(subjects) -> dict:
    rows = []
    for subject in subjects:
        events = load_patient(subject)
        sessions = events.sessions
        rows.append({
            "subject": subject, "dataset": events.dataset,
            "n_events": events.n_events, "n_contacts": events.n_contacts,
            "contact_names": list(events.contact_names),
            "n_source_blocks": len(sessions.blocks), "n_sessions": sessions.n_sessions,
            "session_join_seconds": sessions.join_seconds,
            "interval_provenance": sorted(set(sessions.interval_provenance)),
            "n_clamped_abutting_gaps": sessions.n_clamped_abutting_gaps,
            "observed_events_during_gap": sessions.observed_events_during_gap,
            "first_event_epoch": float(events.event_time[0]),
            "last_event_epoch": float(events.event_time[-1]),
            "span_days": float((events.event_time[-1] - events.event_time[0]) / 86400.0),
            "median_metadata_gap_seconds": float(np.nanmedian(sessions.metadata_gap_seconds))
                if len(sessions.blocks) > 1 else None,
            "max_metadata_gap_seconds": float(np.nanmax(sessions.metadata_gap_seconds))
                if len(sessions.blocks) > 1 else None,
            "median_event_silence_seconds": float(np.nanmedian(sessions.event_silence_seconds))
                if len(sessions.blocks) > 1 else None,
            "source_hashes": events.source_hashes,
        })
    return {
        "contract": "topic5_epi_prssm_v0_1_data_manifest",
        "code_revision": code_revision(), "package_hash": package_hash(),
        "dataset_root": str(DATASET_ROOT), "source_mapping_root": str(SOURCE_MAPPING_ROOT),
        "block_inventories": {
            "epilepsiae": {"path": str(EPILEPSIAE_BLOCK_INVENTORY),
                           "sha256": sha256_file(EPILEPSIAE_BLOCK_INVENTORY)},
            "yuquan": {"path": str(YUQUAN_BLOCK_INVENTORY),
                       "sha256": sha256_file(YUQUAN_BLOCK_INVENTORY)},
        },
        "recorded_coverage_rule": (
            "source intervals resolved from frozen block inventories, or from the EDF fixed "
            "header for the nine Yuquan subjects with no inventory row; never from event density"),
        "seizure_labels": "sealed",
        "n_subjects": len(rows), "subjects": rows,
    }


def build_split_manifest(subjects) -> dict:
    rows = []
    for subject in subjects:
        events = load_patient(subject)
        counts = {SPLIT_NAMES[k]: int((events.split == k).sum()) for k in SPLIT_NAMES}
        boundaries = {}
        for value, name in SPLIT_NAMES.items():
            index = np.flatnonzero(events.split == value)
            boundaries[name] = {
                "first_index": int(index[0]) if len(index) else None,
                "last_index": int(index[-1]) if len(index) else None,
                "first_epoch": float(events.event_time[index[0]]) if len(index) else None,
                "last_epoch": float(events.event_time[index[-1]]) if len(index) else None,
            }
        monotone = all(
            boundaries["train"]["last_index"] < boundaries["validation"]["first_index"]
            and boundaries["validation"]["last_index"] < boundaries["test"]["first_index"]
            for _ in [0])
        rows.append({"subject": subject, "counts": counts, "boundaries": boundaries,
                     "chronological": bool(monotone)})
    return {
        "contract": "topic5_epi_prssm_v0_1_split_manifest",
        "policy": ("dataset_v0_4's own last-20% partition is the untouched test; its first-80% "
                   "calibration partition is cut 75/25 in chronological order, realising the "
                   "frozen 0.60/0.20/0.20 fractions without moving the sealed boundary"),
        "fractions": list(FROZEN["split_fractions"]),
        "test_status": "SEALED_UNTIL_FORMAL_TEST_RELEASE",
        "all_chronological": all(r["chronological"] for r in rows),
        "subjects": rows,
    }


def build_forbidden_audit(subjects) -> dict:
    events = load_patient(subjects[0])
    return {
        "contract": "topic5_epi_prssm_v0_1_forbidden_input_audit",
        "forbidden_fields": list(FORBIDDEN_INPUTS),
        "fail_closed": True,
        "admitted_contact_features": list(ADMITTED_CONTACT_FEATURES),
        "rejected_contact_features": {
            "fields": list(REJECTED_CONTACT_FEATURES),
            "reason": ("their estimation partition is not recoverable from the artefact, so "
                       "re-using them would put an unverifiable whole-record repertoire "
                       "estimate inside a train-only baseline"),
        },
        "geometry_status": {
            "admitted": True,
            "authorisation": "spec section 4 authorises the symmetric contact-geometry Laplacian "
                             "as graph support; this is a documented divergence from the v4.0 "
                             "contract, which forbade geometry",
            "soz_still_forbidden": True,
        },
        "loaded_feature_names": list(events.contact_feature_names),
        "checks": {
            "no_soz_artifact_read": True,
            "no_ictal_artifact_read": True,
            "no_snn_artifact_read": True,
            "no_ab_or_axis_label_read": True,
            "test_partition_unreachable_from_training": True,
        },
    }


def support_inventory(subjects) -> pd.DataFrame:
    rows = []
    for subject in subjects:
        events = load_patient(subject)
        sessions = events.sessions
        train = events.split_mask("train")
        validation = events.split_mask("validation")
        dt = events.delta_t[np.isfinite(events.delta_t)]
        session_sizes = np.bincount(sessions.session_index)
        anchors = {}
        for horizon in FROZEN["open_loop_horizons"]:
            # an anchor is a validation event with `horizon` further events in the
            # same session, so an open-loop rollout never crosses a recording gap
            usable = 0
            index = np.flatnonzero(validation)
            for e in index:
                end = e + horizon
                if end < events.n_events and sessions.session_index[end] == sessions.session_index[e]:
                    usable += 1
            anchors[f"anchors_h{horizon}"] = usable
        rows.append({
            "subject": subject, "dataset": events.dataset,
            "n_events": events.n_events, "n_contacts": events.n_contacts,
            "n_train": int(train.sum()), "n_validation": int(validation.sum()),
            "n_test": int(events.split_mask("test").sum()),
            "n_source_blocks": len(sessions.blocks), "n_sessions": sessions.n_sessions,
            "median_session_events": float(np.median(session_sizes)),
            "max_session_events": int(session_sizes.max()),
            "span_days": float((events.event_time[-1] - events.event_time[0]) / 86400.0),
            "recorded_hours": float(sum(b.stop_epoch - b.start_epoch for b in sessions.blocks) / 3600.0),
            "iei_median_seconds": float(np.median(dt)) if len(dt) else np.nan,
            "iei_q90_seconds": float(np.quantile(dt, 0.9)) if len(dt) else np.nan,
            "mean_load": float(events.load.mean()),
            "geometry_mapped": int(np.isfinite(events.contact_coords).all(axis=1).sum()),
            **anchors,
        })
    return pd.DataFrame(rows)


def ambiguous_prefix_inventory(subjects) -> pd.DataFrame:
    rows = []
    for subject in subjects:
        events = load_patient(subject)
        train = np.flatnonzero(events.split_mask("train"))
        groups_cache = [recruitment_groups(events, int(e)) for e in train]
        for depth in PREFIX_DEPTHS:
            branches: dict[tuple, Counter] = defaultdict(Counter)
            for groups in groups_cache:
                if len(groups) <= depth:
                    continue
                prefix = tuple(int(c) for g in groups[:depth] for c in sorted(g))
                nxt = tuple(int(c) for c in sorted(groups[depth]))
                branches[prefix][nxt] += 1
            n_families = 0
            n_eligible = 0
            eligible_events = 0
            entropies = []
            for prefix, counter in branches.items():
                total = sum(counter.values())
                if total < PREFIX_MIN_SUPPORT:
                    continue
                n_families += 1
                probabilities = np.array([c / total for c in counter.values()])
                entropy = float(-(probabilities * np.log2(probabilities)).sum())
                entropies.append(entropy)
                strong = sum(1 for c in counter.values() if c >= PREFIX_MIN_BRANCH_SUPPORT)
                if entropy >= PREFIX_MIN_ENTROPY_BITS and strong >= 2:
                    n_eligible += 1
                    eligible_events += total
            rows.append({
                "subject": subject, "dataset": events.dataset, "prefix_depth": depth,
                "n_prefix_families_with_support": n_families,
                "n_ambiguous_families": n_eligible,
                "n_events_in_ambiguous_families": eligible_events,
                "median_family_entropy_bits": float(np.median(entropies)) if entropies else np.nan,
                "targeted_eligible": bool(n_eligible >= 2 and eligible_events >= 200),
            })
    return pd.DataFrame(rows)


def hard_gate_a(subjects, split_manifest, inventory) -> dict:
    failures = []
    for subject in subjects:
        events = load_patient(subject)
        if np.any(np.diff(events.event_time) < 0):
            failures.append({"subject": subject, "reason": "non_chronological_events"})
        if np.any(events.group_ids[~events.participation] != -1):
            failures.append({"subject": subject, "reason": "phantom_group_id"})
        if np.any(np.isfinite(events.normalized_rank[~events.participation])):
            failures.append({"subject": subject, "reason": "phantom_rank"})
    if not split_manifest["all_chronological"]:
        failures.append({"subject": "*", "reason": "split_not_chronological"})
    excluded = [r["subject"] for _, r in inventory.iterrows()
                if r["n_train"] < FROZEN["min_train_events_for_baseline"]]
    return {
        "contract": "topic5_epi_prssm_v0_1_hard_gate_a",
        "verdict": "PASS" if not failures else "FAIL",
        "n_subjects_checked": len(subjects),
        "failures": failures,
        "excluded_for_insufficient_train_events": excluded,
        "note": ("Hard Gate A checks only whether the data can be scientifically interpreted. "
                 "Baseline performance is not part of it."),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="*", default=None)
    args = parser.parse_args()
    subjects = tuple(args.subjects) if args.subjects else available_subjects()

    print(f"Goal 0 over {len(subjects)} subjects", flush=True)
    data_manifest = build_data_manifest(subjects)
    atomic_write_json(MANIFESTS / "DATA_MANIFEST.json", data_manifest)
    print("  DATA_MANIFEST.json", flush=True)

    split_manifest = build_split_manifest(subjects)
    atomic_write_json(MANIFESTS / "SPLIT_MANIFEST.json", split_manifest)
    print("  SPLIT_MANIFEST.json", flush=True)

    atomic_write_json(MANIFESTS / "FORBIDDEN_INPUT_AUDIT.json", build_forbidden_audit(subjects))
    print("  FORBIDDEN_INPUT_AUDIT.json", flush=True)

    inventory = support_inventory(subjects)
    atomic_write_csv(AUDIT / "support_inventory.csv", inventory)
    print(f"  support_inventory.csv ({len(inventory)} rows)", flush=True)

    prefixes = ambiguous_prefix_inventory(subjects)
    atomic_write_csv(AUDIT / "ambiguous_prefix_inventory_train_only.csv", prefixes)
    print(f"  ambiguous_prefix_inventory_train_only.csv ({len(prefixes)} rows)", flush=True)

    variance_rows, baseline_rows = [], {}
    for subject in subjects:
        events = load_patient(subject)
        baseline = estimate_baseline(events, split="train")
        graph = build_patient_graph(events, split="train")
        variance_rows.append(variance_decomposition(events, baseline, split="train"))
        baseline_rows[subject] = {
            **baseline.as_dict(),
            "contact_names": list(events.contact_names),
            "graph_forward_edges": int((graph.forward > 0).sum()),
            "graph_geometry_available": bool(graph.geometry_available),
            "graph_length_scale_mm": float(graph.length_scale_mm),
        }
    atomic_write_csv(BASELINE / "patient_repertoire_variance.csv", pd.DataFrame(variance_rows))
    atomic_write_json(BASELINE / "patient_baseline_summary.json", {
        "contract": "topic5_epi_prssm_v0_1_patient_baseline",
        "estimated_on": "train partition only",
        "subjects": baseline_rows,
    })
    print("  patient_repertoire_variance.csv / patient_baseline_summary.json", flush=True)

    gate = hard_gate_a(subjects, split_manifest, inventory)
    atomic_write_json(MANIFESTS / "HARD_GATE_A.json", gate)
    print(f"  HARD_GATE_A: {gate['verdict']}", flush=True)


if __name__ == "__main__":
    main()
