#!/usr/bin/env python3
"""Rebuild the complete event-level stream, including the pre-ictal events that the
definite-interictal block policy deletes.

Why this exists
---------------
``dataset_v0_4`` keeps only blocks that survive a fail-closed definite-interictal
policy: a block is dropped if it overlaps a seizure or its frozen 120 min
post-ictal guard, if it crosses a local day/night boundary, or if it sits next to
a recording discontinuity larger than 5400 s.  Those rules delete exactly the
observations an online system would have had in the minutes before a seizure, so
a seizure-link analysis built on that stream is forced to ask "can a state
inferred hours ago survive until onset" instead of "does the state move after the
pre-ictal IEDs are observed".

What this does and does not change
----------------------------------
Nothing about the *event encoding* changes.  The same producer functions are
called with the same constants, so a rebuilt event is bit-identical to the frozen
one where both exist -- this is asserted, not assumed.  The only difference is
which events are kept: here, all of them.

This script reads **no seizure labels**.  It rebuilds the stream and nothing more;
the peri-ictal annotation happens downstream, after the interictal model freeze.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from scripts.build_topic5_state_conditioned_dataset import _raw_subject_dir  # noqa: E402
from src.interictal_propagation import load_subject_propagation_events  # noqa: E402
from src.topic5_epi_prssm.cohort import cohort_subjects, load_tensors  # noqa: E402
from src.topic5_epi_prssm.contracts import (  # noqa: E402
    OUTPUT_ROOT, atomic_write_json, code_revision, package_hash, sha256_file,
)
from src.topic5_interictal_operator import encode_recruitment_matrix  # noqa: E402

DATASET_CONFIG = ROOT / "config/topic5_interictal_rank_distribution_v0_4.yaml"
OUT = OUTPUT_ROOT / "full_event_stream"


def build_subject(subject: str, cfg: dict) -> dict:
    frozen = load_tensors([subject])[0]
    events = load_subject_propagation_events(_raw_subject_dir(subject))
    times = np.asarray(events["event_abs_times"], float)
    ranks = np.asarray(events["ranks"], float)
    bools = np.asarray(events["bools"], bool)
    lag_raw = np.asarray(events["lag_raw"], float)
    names = [str(x) for x in events["channel_names"]]
    block_ids = np.asarray(events["block_ids"], int)
    starts = np.asarray(events["block_start_times"], float)
    record_names = np.asarray([str(x) for x in events["record_names"]])

    frozen_names = list(frozen.meta["contact_names"])
    if names != frozen_names:
        raise RuntimeError(
            f"{subject}: channel order differs from the frozen cohort "
            f"({names} vs {frozen_names}); the frozen per-contact parameters would "
            "map onto the wrong contacts")

    selected = np.flatnonzero(np.isfinite(times))
    selected = selected[np.argsort(times[selected], kind="stable")]
    n_before_gate = int(selected.size)
    selected = selected[np.sum(bools[:, selected], axis=0)
                        >= int(cfg["cohort"]["min_participants"])]
    local_rank, group_ids, group_counts = encode_recruitment_matrix(
        ranks[:, selected], bools[:, selected], lag_raw[:, selected],
        tie_tolerance_seconds=float(cfg["event_encoding"]["tie_tolerance_seconds"]))
    keep = group_counts >= int(cfg["cohort"]["min_recruitment_sets"])
    selected = selected[keep]
    local_rank, group_ids, group_counts = local_rank[keep], group_ids[keep], group_counts[keep]

    event_time = times[selected].astype(np.float64)
    participation = bools[:, selected].T.astype(np.uint8)
    # record_names and block_start_times are per BLOCK, not per event
    event_block = block_ids[selected].astype(np.int32)
    event_record = record_names[event_block]
    event_block_start = starts[event_block].astype(np.float64)

    # which of these events the frozen definite-interictal stream also holds
    frozen_time = np.asarray(frozen.event_time, dtype=np.float64)
    position = np.searchsorted(event_time, frozen_time)
    position = np.clip(position, 0, len(event_time) - 1)
    matched = np.isclose(event_time[position], frozen_time, atol=1e-6)
    in_definite = np.zeros(len(event_time), dtype=bool)
    in_definite[position[matched]] = True

    # bit-identical check on the overlap: the rebuilt encoding must equal the frozen one
    parity = {"n_frozen_events": int(len(frozen_time)),
              "n_frozen_events_found": int(matched.sum())}
    if matched.sum():
        idx = position[matched]
        frozen_part = frozen.participation.cpu().numpy()[matched]
        frozen_gid = frozen.group_ids.cpu().numpy()[matched]
        parity["participation_identical"] = bool(
            np.array_equal(participation[idx].astype(bool), frozen_part))
        parity["group_ids_identical"] = bool(
            np.array_equal(group_ids[idx].astype(np.int64), frozen_gid))
        # the frozen tensors carry the rank inside the mark stack (channel 1,
        # zero-filled where masked), so compare on that representation
        frozen_rank = frozen.marks.cpu().numpy()[matched][..., 1]
        rebuilt_rank = np.where(participation[idx].astype(bool),
                                np.nan_to_num(local_rank[idx]), 0.0)
        parity["rank_identical_on_participants"] = bool(
            np.allclose(rebuilt_rank, frozen_rank, atol=1e-6))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_subject").mkdir(parents=True, exist_ok=True)
    path = OUT / "per_subject" / f"{subject}.npz"
    np.savez_compressed(
        path,
        event_abs_time=event_time,
        event_participation=participation,
        event_group_ids=group_ids.astype(np.int16),
        event_group_count=group_counts.astype(np.int16),
        event_local_rank=local_rank.astype(np.float32),
        event_block_id=event_block,
        event_record_name=event_record,
        event_block_start=event_block_start,
        in_definite_interictal=in_definite,
        contact_names=np.asarray(names),
    )
    return {
        "subject": subject, "dataset": subject.split("_", 1)[0],
        "n_contacts": len(names),
        "n_events_before_participant_gate": n_before_gate,
        "n_events_full_stream": int(len(event_time)),
        "n_events_definite_interictal_frozen": int(frozen.n_events),
        "n_events_recovered_beyond_frozen": int(len(event_time) - int(in_definite.sum())),
        "expansion_factor": float(len(event_time) / max(frozen.n_events, 1)),
        "n_blocks": int(len(np.unique(event_block))),
        "first_event_epoch": float(event_time[0]), "last_event_epoch": float(event_time[-1]),
        "frozen_event_recovery_fraction": float(matched.mean()) if len(frozen_time) else 0.0,
        "channel_order_identical_to_frozen": True,
        "encoding_parity": parity,
        "path": str(path), "sha256": sha256_file(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="*", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(DATASET_CONFIG.open())
    subjects = args.subjects or list(cohort_subjects())
    rows, failures = [], []
    started = time.time()
    for subject in subjects:
        try:
            row = build_subject(subject, cfg)
            rows.append(row)
            print(f"  {subject:22s} full {row['n_events_full_stream']:7d}  "
                  f"frozen {row['n_events_definite_interictal_frozen']:7d}  "
                  f"x{row['expansion_factor']:.2f}  "
                  f"recovery {row['frozen_event_recovery_fraction']:.4f}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failures.append({"subject": subject, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  {subject:22s} FAILED {type(exc).__name__}: {exc}", flush=True)
    atomic_write_json(OUT / "FULL_STREAM_MANIFEST.json", {
        "contract": "topic5_epi_prssm_v0_1_full_event_stream",
        "why": "the definite-interictal block policy deletes the pre-ictal observations an "
               "online system would have had; this stream keeps every event so a seizure-link "
               "analysis can observe them",
        "seizure_labels_read": False,
        "encoding_source": "scripts/build_topic5_interictal_operator_dataset.py via "
                           "src.topic5_interictal_operator.encode_recruitment_matrix, same "
                           "constants, same channel order",
        "dataset_config": str(DATASET_CONFIG),
        "dataset_config_sha256": sha256_file(DATASET_CONFIG),
        "n_subjects": len(rows), "n_failed": len(failures), "failures": failures,
        "build_seconds": time.time() - started,
        "code_revision": code_revision(), "package_hash": package_hash(),
        "subjects": rows,
    })
    total_full = sum(r["n_events_full_stream"] for r in rows)
    total_frozen = sum(r["n_events_definite_interictal_frozen"] for r in rows)
    print(f"\n{len(rows)} subjects: {total_full:,} events in the full stream vs "
          f"{total_frozen:,} in the frozen definite-interictal stream "
          f"({total_full / max(total_frozen, 1):.2f}x)")


if __name__ == "__main__":
    main()
