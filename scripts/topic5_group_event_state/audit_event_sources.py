#!/usr/bin/env python3
"""Audit complete group-event artifacts against native SEEG sources.

Emits the machine-readable source audit for Group-Event State v0.1: which
patients have every packed event still traceable to a native sample range,
where continuous coverage actually breaks, and which analysis bands the native
sampling rate supports.  Every file is written atomically.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.source_audit import (  # noqa: E402
    audit_cohort,
    discover_existing_34,
    discover_subjects,
    inventory_index,
    seizure_index,
    write_csv_atomic,
    write_json_atomic,
)

MAIN_TREE = Path("/home/honglab/leijiaxin/HFOsp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="*", help="Explicit subject IDs.")
    parser.add_argument(
        "--all-discoverable",
        action="store_true",
        help="Audit every patient owning a group-event artifact on disk.",
    )
    parser.add_argument(
        "--existing-dataset-root",
        type=Path,
        default=MAIN_TREE / "results/topic5_interictal_rank_distribution/dataset_v0_4",
        help="Used to recover the historical 34-patient cohort membership.",
    )
    parser.add_argument(
        "--epilepsiae-inventory",
        type=Path,
        default=MAIN_TREE / "results/epilepsiae_block_inventory.csv",
    )
    parser.add_argument(
        "--yuquan-inventory",
        type=Path,
        default=ROOT / "results/dataset_inventory/yuquan_block_inventory.csv",
    )
    parser.add_argument(
        "--epilepsiae-seizures",
        type=Path,
        default=MAIN_TREE / "results/epilepsiae_seizure_inventory.csv",
    )
    parser.add_argument(
        "--yuquan-seizures",
        type=Path,
        default=ROOT / "results/dataset_inventory/yuquan_seizure_inventory.csv",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--no-hash",
        action="store_true",
        help="Skip artifact SHA256 (fast re-runs; the audit records that it did).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    historical = set(discover_existing_34(args.existing_dataset_root))
    if args.subjects:
        subjects = list(args.subjects)
    elif args.all_discoverable:
        subjects = discover_subjects()
    else:
        subjects = sorted(historical)
    if not subjects:
        raise SystemExit("no subjects discovered")

    inventory = inventory_index(args.epilepsiae_inventory, args.yuquan_inventory)
    seizures = seizure_index(args.epilepsiae_seizures, args.yuquan_seizures)
    print(f"auditing {len(subjects)} subjects", flush=True)
    payload = audit_cohort(
        subjects, inventory, seizures, hash_artifacts=not args.no_hash, progress=True
    )
    payload["artifact_hashing"] = "sha256" if not args.no_hash else "skipped"
    payload["historical_34_cohort"] = sorted(historical)
    payload["elapsed_sec"] = round(time.time() - started, 1)

    out = Path(args.out_dir)
    write_json_atomic(payload, out / "source_audit.json")

    subject_rows = []
    block_rows = []
    session_rows = []
    band_rows = []
    pointer_audit = {
        "contract": "topic5_group_event_state_v0_1_event_pointer_audit",
        "raw_fingerprint_kind": "edge_sha256_1MiB (size+mtime+first/last 1 MiB); NOT a whole-file digest",
        "subjects": {},
    }
    for rec in payload["subjects"]:
        subject_rows.append(
            {
                "subject": rec["subject"],
                "dataset": rec["dataset"],
                "in_historical_34": rec["subject"] in historical,
                "lagpat_variant": rec["lagpat_variant"],
                "n_blocks": rec["n_blocks"],
                "n_blocks_pass": rec["n_blocks_pass"],
                "n_blocks_recorded_in_inventory": rec["n_blocks_recorded_in_inventory"],
                "block_fail_reasons": "|".join(rec["block_fail_reasons"]),
                "n_events": rec["n_events"],
                "n_events_interictal": rec["n_events_interictal"],
                "n_events_ictal": rec["n_events_ictal"],
                "n_events_preictal_1h": rec["n_events_preictal_1h"],
                "waveform_pointer_fraction": rec["waveform_pointer_fraction"],
                "n_contacts": "|".join(str(c) for c in rec["n_contacts"]),
                "native_rate_hz": "|".join(str(r) for r in rec["native_rate_hz"]),
                "detector_reference": "|".join(rec["detector_reference"]),
                "n_detector_channels": "|".join(str(c) for c in rec["n_detector_channels"]),
                "lag_frequency_available": rec["lag_frequency_available"],
                "n_contiguous_sessions": rec["n_contiguous_sessions"],
                "max_events_in_contiguous_session": rec["max_events_in_contiguous_session"],
                "median_events_in_contiguous_session": rec["median_events_in_contiguous_session"],
                "longest_session_hours": rec["longest_session_hours"],
                "sessions_ge_1k_events": rec["sessions_ge_1k_events"],
                "sessions_ge_5k_events": rec["sessions_ge_5k_events"],
                "sessions_ge_10k_events": rec["sessions_ge_10k_events"],
                "recording_span_hours": rec["recording_span_hours"],
                "median_inter_event_sec": rec["median_inter_event_sec"],
                "p95_inter_event_sec": rec["p95_inter_event_sec"],
                "n_seizures": rec["n_seizures"],
                "seizure_patterns": "|".join(rec["seizure_patterns"]),
            }
        )
        block_rows.extend(rec["blocks"])
        for session in rec["sessions"]:
            session_rows.append(
                {
                    "subject": rec["subject"],
                    "dataset": rec["dataset"],
                    "session_index": session["session_index"],
                    "n_blocks": session["n_blocks"],
                    "n_events": session["n_events"],
                    "start_epoch": session["start_epoch"],
                    "end_epoch": session["end_epoch"],
                    "duration_hours": session["duration_hours"],
                    "first_record": session["record_names"][0],
                    "last_record": session["record_names"][-1],
                }
            )
        for band, n_events in sorted(rec["events_by_supported_band"].items()):
            band_rows.append(
                {
                    "subject": rec["subject"],
                    "dataset": rec["dataset"],
                    "band": band,
                    "native_rate_hz": "|".join(str(r) for r in rec["native_rate_hz"]),
                    "supported": n_events > 0,
                    "n_events_supported": n_events,
                    "n_events_total": rec["n_events"],
                }
            )
        pointer_audit["subjects"][rec["subject"]] = {
            "n_events": rec["n_events"],
            "waveform_pointer_fraction": rec["waveform_pointer_fraction"],
            "detector_reference": rec["detector_reference"],
            "samples": rec["pointer_samples"],
        }

    write_csv_atomic(subject_rows, out / "subject_inventory.csv")
    write_csv_atomic(block_rows, out / "block_inventory.csv")
    write_csv_atomic(session_rows, out / "contiguous_session_inventory.csv")
    write_csv_atomic(band_rows, out / "band_availability.csv")
    write_json_atomic(pointer_audit, out / "event_pointer_audit.json")

    print(
        f"{payload['status']}: {payload['n_subjects_audited']}/"
        f"{payload['n_subjects_requested']} subjects, {payload['n_events']} events "
        f"({payload['n_events_interictal']} interictal), waveform pointers "
        f"{payload['waveform_pointer_fraction']:.4f}, {payload['elapsed_sec']}s"
    )
    for failure in payload["failures"]:
        print(f"FAIL {failure['subject']}: {failure['error']}")


if __name__ == "__main__":
    main()
