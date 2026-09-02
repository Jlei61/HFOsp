#!/usr/bin/env python3
"""Machine audit of nested-time and legacy-source semantics for v0.3.1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.dataset import SubjectSequence  # noqa: E402
from src.topic5_group_event_state.v02.subject import load_subject_timeline  # noqa: E402
from src.topic5_group_event_state.v03.partition import PHASE_NAMES  # noqa: E402
from src.topic5_group_event_state.v03.pilot import (  # noqa: E402
    DATASET_ROOT,
    PILOT_SUBJECTS,
    _legacy_paths,
    grammar_fit_seq_positions,
    nested_partition,
    seq_positions_for_phase,
    validate_contact_contract,
)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def audit_subject(subject: str) -> dict:
    seq = SubjectSequence(DATASET_ROOT / subject)
    timeline = load_subject_timeline(subject)
    partition = nested_partition(timeline)
    legacy_checkpoint, legacy_dataset = _legacy_paths(subject)
    validate_contact_contract(subject, seq, legacy_dataset)
    prefix_pos = grammar_fit_seq_positions(timeline, partition)
    later_pos = np.concatenate(
        [
            seq_positions_for_phase(timeline, partition, phase)
            for phase in ("state_train", "dev_val", "dev_test")
        ]
    )
    prefix_part = np.asarray(
        seq.gather_positions(prefix_pos)["participation"], dtype=bool
    )
    later_part = np.asarray(
        seq.gather_positions(later_pos)["participation"], dtype=bool
    )
    prefix_support = prefix_part.any(axis=0)
    later_support = later_part.any(axis=0)
    legacy = torch.load(legacy_checkpoint, map_location="cpu", weights_only=False)
    index = timeline.index
    return {
        "subject": subject,
        "nested_boundary_epochs": partition.boundary_epochs.tolist(),
        "grammar_fit_stop_epoch": float(partition.grammar_fit_stop_epoch),
        "recorded_seconds": partition.recorded_seconds,
        "events_by_phase": {
            phase: int(seq_positions_for_phase(timeline, partition, phase).size)
            for phase in PHASE_NAMES
        },
        "grammar_fit_events": int(prefix_pos.size),
        "contact_order_matches_legacy_bundle": True,
        "n_contacts": int(prefix_support.size),
        "n_contacts_seen_in_grammar_fit": int(prefix_support.sum()),
        "n_later_participating_contacts_unseen_in_grammar_fit": int(
            ((~prefix_support) & later_support).sum()
        ),
        "legacy_checkpoint": str(legacy_checkpoint),
        "legacy_checkpoint_keys": sorted(str(key) for key in legacy),
        "legacy_model_kwargs": legacy["model_kwargs"],
        "primary_legacy_payload_policy": {
            "read": ["model_kwargs"],
            "forbidden": ["model_state", "heldout_local_offset"],
            "learned_weights_loaded": False,
        },
        "normalization_contract": {
            "contact_support": "grammar_fit_0_to_16_percent_only",
            "event_encoder_stats": "grammar_fit_0_to_16_percent_only",
            "grammar_selection": "calibration_16_to_20_percent_only",
            "state_selection": "development_validation_70_to_80_percent_only",
            "development_test": "80_to_100_percent_score_once",
        },
        "upstream_measurement": {
            "detector_reference": index.get("detector_reference"),
            "montage_provenance": index.get("montage_provenance"),
            "contact_vocabulary_source": "legacy_lagpat_full_record_refine_packing_artifact",
            "detector_threshold_provenance_in_bundle": "not_sufficiently_recorded_for_nested_refit",
            "full_record_data_adaptive_contact_selection": True,
            "nested_clean": False,
            "scientific_status": "upstream_transductive_development_only",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/data/hfosp_group_event_state_v0_3/source_audit/"
            "nested_source_audit.json"
        ),
    )
    args = parser.parse_args()
    subjects = [audit_subject(subject) for subject in PILOT_SUBJECTS]
    payload = {
        "format": "group_event_state_v0_3_1_nested_source_audit",
        "status": "complete",
        "model_layer_nested_contract": True,
        "measurement_layer_nested_contract": False,
        "cohort_expansion_allowed": False,
        "reason": (
            "current lagPat contact vocabulary was selected by a full-record "
            "legacy refine/packing pipeline; pilot may diagnose the model but "
            "cannot support a fully nested confirmatory claim"
        ),
        "subjects": subjects,
    }
    atomic_json(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
