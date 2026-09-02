#!/usr/bin/env python3
"""Real-data verification of the boundary contract, on every patient.

The unit tests prove the *rules* are implemented.  This proves the rules actually
held on the recordings the results were computed from -- which is a different
claim, and the one a reader of the numbers cares about.

Four checks per disjoint block, brute-forced against the raw event times:

1. the stored block count equals a direct count of events in ``[anchor, anchor+H)``
2. no seizure onset lies inside ``[anchor-H, anchor+H)``
3. the whole exposure-plus-target span sits inside one split piece
4. the whole span sits inside one coverage segment, so it never crosses a gap
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state_h3.io import write_json_atomic  # noqa: E402
from src.topic5_group_event_state_h3.runtime import DATASET_ROOT, load_subject  # noqa: E402
from src.topic5_group_event_state_h3.support import (  # noqa: E402
    MAIN_HORIZONS_MINUTES,
    load_seizures,
    segment_bounds,
)

OUT = ROOT / "results/epi_prssm/group_event_state/v0_2/h3/support/block_verification.json"


def verify(subject: str, horizons: list[int]) -> dict:
    ctx = load_subject(subject, torch.device("cpu"), horizons=horizons)
    t = ctx.stream.t_abs
    seizures = load_seizures(DATASET_ROOT / subject / "index.json")
    bounds = segment_bounds(ctx.intervals)

    checked = bad_count = bad_seizure = bad_split = bad_segment = 0
    for horizon in horizons:
        span = float(horizon) * 60.0
        for seg, anchor_index in ctx.disjoint[horizon]:
            tl = ctx.tensors.timelines[seg]
            anchor = float(tl.anchor_time[anchor_index])
            checked += 1
            if int(((t >= anchor) & (t < anchor + span)).sum()) != int(
                ctx.tensors.targets[seg][horizon]["count"][anchor_index]
            ):
                bad_count += 1
            lo, hi = anchor - span, anchor + span
            if any(lo <= onset < hi for onset, _offset in seizures):
                bad_seizure += 1
            pieces = [
                i for i in ctx.intervals
                if i.segment_id == tl.segment_id and i.start - 1e-6 <= anchor < i.stop
            ]
            if not pieces or anchor + span > pieces[0].stop + 1e-6:
                bad_split += 1
            seg_lo, seg_hi = bounds[tl.segment_id]
            if lo < seg_lo - 1e-6 or hi > seg_hi + 1e-6:
                bad_segment += 1

    return {
        "subject": subject,
        "n_events": int(t.size),
        "n_seizures": len(seizures),
        "n_segments": len(bounds),
        "n_blocks_checked": checked,
        "count_mismatch_vs_brute_force": bad_count,
        "blocks_containing_a_seizure_onset": bad_seizure,
        "blocks_straddling_a_split": bad_split,
        "blocks_crossing_a_recording_gap": bad_segment,
        "clean": bool(bad_count == bad_seizure == bad_split == bad_segment == 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--horizons", nargs="*", type=int, default=list(MAIN_HORIZONS_MINUTES))
    args = parser.parse_args()

    subjects = args.subjects or sorted(
        p.name for p in DATASET_ROOT.iterdir() if (p / "index.json").exists()
    )
    rows = []
    for subject in subjects:
        row = verify(subject, args.horizons)
        rows.append(row)
        print(
            f"{row['subject']:26s} blocks={row['n_blocks_checked']:6d} "
            f"count_mismatch={row['count_mismatch_vs_brute_force']:3d} "
            f"seizure_in_block={row['blocks_containing_a_seizure_onset']:3d} "
            f"split_straddle={row['blocks_straddling_a_split']:3d} "
            f"gap_cross={row['blocks_crossing_a_recording_gap']:3d} "
            f"{'CLEAN' if row['clean'] else 'DIRTY'}",
            flush=True,
        )

    payload = {
        "n_subjects": len(rows),
        "n_blocks_checked": int(sum(r["n_blocks_checked"] for r in rows)),
        "n_subjects_clean": int(sum(r["clean"] for r in rows)),
        "totals": {
            key: int(sum(r[key] for r in rows))
            for key in (
                "count_mismatch_vs_brute_force",
                "blocks_containing_a_seizure_onset",
                "blocks_straddling_a_split",
                "blocks_crossing_a_recording_gap",
            )
        },
        "subjects": rows,
    }
    write_json_atomic(payload, OUT)
    print(f"\n{payload['n_subjects_clean']}/{payload['n_subjects']} patients clean; "
          f"{payload['n_blocks_checked']} blocks checked; totals {payload['totals']}")


if __name__ == "__main__":
    main()
