#!/usr/bin/env python3
"""Per-event recorded-time bookkeeping for the arrival channel.

The survival term of an arrival likelihood is a statement about *absence*: no
discharge happened between these two, and that is evidence only over wall time the
recorder was actually on.  The cohort cache keeps a session index but drops the
block-level metadata, and a session is a run of blocks joined when their gap is
under a threshold -- so any gap shorter than the join is invisible downstream.  An
audit found 3,189 intervals of at least the join length being carried into the
integral as if they were continuous recording.

Writes, per event: the wall time since the previous event that was actually
recorded, and whether any unrecorded time falls inside that interval.
"""
from __future__ import annotations

import argparse

import numpy as np

from _common import OUTPUT_ROOT, atomic_write_json, code_revision, package_hash  # noqa: E402
from src.topic5_epi_prssm.event_marks import SOURCE_MAPPING_ROOT, load_patient  # noqa: E402
from src.topic5_epi_prssm.sessions import build_sessions  # noqa: E402

OUT = OUTPUT_ROOT / "recorded_intervals"


def per_subject(subject: str) -> dict:
    events = load_patient(subject)
    with np.load(SOURCE_MAPPING_ROOT / f"{subject}.npz", allow_pickle=True) as m:
        record_names = np.asarray(m["event_source_record_name"]).astype(str)
    times = np.asarray(events.event_time, dtype=np.float64)
    order = np.argsort(times)
    times, record_names = times[order], record_names[order]

    table = build_sessions(subject, times, record_names)
    block_of_event = np.asarray(table.block_index, dtype=int)
    gap = np.asarray(table.metadata_gap_seconds, dtype=np.float64)   # per block, nan first

    n = len(times)
    elapsed = np.diff(times, prepend=times[0])
    # unrecorded time inside interval e = the metadata gaps of every block boundary
    # strictly crossed between event e-1 and event e
    unrecorded = np.zeros(n)
    for e in range(1, n):
        lo, hi = block_of_event[e - 1], block_of_event[e]
        if hi > lo:
            crossed = gap[lo + 1:hi + 1]
            unrecorded[e] = float(np.nansum(np.maximum(crossed, 0.0)))
    recorded = np.maximum(elapsed - unrecorded, 0.0)
    spans_gap = unrecorded > 0.0
    spans_gap[0] = True                       # nothing precedes the first event

    return {
        "subject": subject, "n_events": int(n),
        "elapsed": elapsed, "recorded": recorded, "spans_gap": spans_gap,
        "n_blocks": int(len(table.blocks)), "n_sessions": int(table.n_sessions),
        "unrecorded_seconds_total": float(unrecorded.sum()),
        "n_intervals_spanning_a_gap": int(spans_gap.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="all34")
    args = parser.parse_args()
    from _common import resolve_cohort
    OUT.mkdir(parents=True, exist_ok=True)

    summary = []
    for subject in resolve_cohort(args.cohort):
        row = per_subject(subject)
        np.savez(OUT / f"{subject}.npz", elapsed=row["elapsed"],
                 recorded=row["recorded"], spans_gap=row["spans_gap"])
        summary.append({k: v for k, v in row.items()
                        if not isinstance(v, np.ndarray)})
        print(f"{subject:22s} blocks={row['n_blocks']:5d} sessions={row['n_sessions']:4d} "
              f"gap-spanning intervals={row['n_intervals_spanning_a_gap']:5d} "
              f"unrecorded={row['unrecorded_seconds_total']/3600:8.1f} h", flush=True)

    atomic_write_json(OUT / "RECORDED_INTERVAL_MANIFEST.json", {
        "contract": "topic5_epi_prssm_v0_2_recorded_intervals",
        "why": "the survival term is evidence of absence and is only valid over wall "
               "time the recorder was on; session_open misses any gap shorter than the "
               "session join threshold",
        "subjects": summary,
        "n_intervals_spanning_a_gap": sum(s["n_intervals_spanning_a_gap"] for s in summary),
        "unrecorded_hours_total": sum(s["unrecorded_seconds_total"] for s in summary) / 3600.0,
        "code_revision": code_revision(), "package_hash": package_hash(),
    })


if __name__ == "__main__":
    main()
