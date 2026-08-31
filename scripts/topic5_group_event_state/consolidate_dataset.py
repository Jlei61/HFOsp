#!/usr/bin/env python3
"""Turn a patient's block shards into one memory-mappable event sequence.

Also attaches static contact geometry when the coordinate loader can resolve it;
a patient whose coordinates cannot be resolved keeps a geometry-flag of 0 rather
than silently receiving zeros that look like real positions.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.dataset import consolidate_subject  # noqa: E402
from src.topic5_group_event_state.source_audit import (  # noqa: E402
    seizure_index,
    write_json_atomic,
)

MAIN_TREE = Path("/home/honglab/leijiaxin/HFOsp")
V0_1 = MAIN_TREE / "results/epi_prssm/group_event_state/v0_1"


def _coords_for(subject: str, contacts: list[dict]) -> tuple[np.ndarray | None, dict]:
    dataset, patient = subject.split("_", 1)
    requested = [c["anode"] for c in contacts]
    try:
        from src.seeg_coord_loader import load_subject_coords

        result = load_subject_coords(dataset, patient, requested)
        coords = np.asarray(result.coords_array_in_requested_order, dtype=np.float32)
        mapped = np.asarray(result.mapped_mask_in_requested_order, dtype=bool)
        info = {
            "coord_space": result.coord_space,
            "coord_units": result.coord_units,
            "n_mapped": int(mapped.sum()),
            "n_requested": int(mapped.size),
            "normalization_certainty": result.normalization_certainty,
        }
        if not mapped.all():
            info["status"] = "partial"
            return None, info
        info["status"] = "ok"
        return coords, info
    except Exception as exc:
        return None, {"status": f"unavailable:{type(exc).__name__}", "detail": str(exc)[:200]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("/data/hfosp_group_event_state_v0_1/cache"))
    parser.add_argument("--out-root", type=Path, default=Path("/data/hfosp_group_event_state_v0_1/dataset"))
    parser.add_argument("--sessions", type=Path, default=V0_1 / "contiguous_session_inventory.csv")
    parser.add_argument("--epilepsiae-seizures", type=Path, default=MAIN_TREE / "results/epilepsiae_seizure_inventory.csv")
    parser.add_argument("--yuquan-seizures", type=Path, default=ROOT / "results/dataset_inventory/yuquan_seizure_inventory.csv")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--out-summary", type=Path, default=V0_1 / "dataset_summary.json")
    args = parser.parse_args()

    sessions_all = list(csv.DictReader(args.sessions.open()))
    seizures = seizure_index(args.epilepsiae_seizures, args.yuquan_seizures)

    summaries, failures = [], []
    for subject in args.subjects:
        dataset, patient = subject.split("_", 1)
        rows = [r for r in sessions_all if r["subject"] == subject]
        try:
            out_dir = args.out_root / subject
            index = consolidate_subject(
                subject,
                args.cache_root / subject,
                out_dir,
                session_rows=rows,
                seizures=seizures.get((dataset, patient), []),
                overwrite=args.overwrite,
            )
            coords, coord_info = _coords_for(subject, index["contacts"])
            if coords is not None:
                np.save(out_dir / "coords.npy", coords)
            index["geometry"] = coord_info
            (out_dir / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True, default=float))
            summaries.append(
                {
                    "subject": subject,
                    "n_events": index["n_events"],
                    "n_events_interictal": index["n_events_interictal"],
                    "n_events_ictal": index["n_events_ictal"],
                    "n_contacts": index["n_contacts"],
                    "n_sessions": len(index["sessions"]),
                    "max_session_events": max((s["stop_index"] - s["start_index"] for s in index["sessions"]), default=0),
                    "split": index["split_bounds_on_interictal_index"],
                    "geometry": coord_info,
                    "views": index["views"],
                    "band_available": index["band_available"],
                    "native_rate_hz": index["native_rate_hz"],
                }
            )
            print(
                f"{subject}: {index['n_events_interictal']} interictal / {index['n_events']} events, "
                f"{index['n_contacts']} contacts, {len(index['sessions'])} sessions, "
                f"geometry={coord_info.get('status')}",
                flush=True,
            )
        except Exception as exc:
            failures.append({"subject": subject, "error": f"{type(exc).__name__}: {exc}",
                             "traceback": traceback.format_exc(limit=6)})
            print(f"FAILED {subject}: {exc}", flush=True)

    write_json_atomic({"subjects": summaries, "failures": failures}, args.out_summary)
    print(f"consolidated {len(summaries)}/{len(args.subjects)} subjects")


if __name__ == "__main__":
    main()
