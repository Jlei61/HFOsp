#!/usr/bin/env python3
"""B0.1 -- patient -> recording -> seizure crosswalk + per-onset audit.

Coverage truth is the consolidated dataset itself (``index.json::source_shards``),
not the block inventory: a block that exists on disk but never entered the
dataset is *not* coverage. Seizure truth is the two canonical inventories.

Writes:
  support/seizure_crosswalk.csv           one row per inventory seizure
  support/seizure_crosswalk_summary.json  dispositions + both symmetric differences
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_h2b_transfer.crosswalk import (  # noqa: E402
    Disposition,
    build_recording_index,
    crosswalk_seizures,
    recording_code_of_record_name,
)

MAIN_TREE = Path("/home/honglab/leijiaxin/HFOsp")
V0_1 = MAIN_TREE / "results/epi_prssm/group_event_state/v0_1"
DEFAULT_OUT = ROOT / "results/epi_prssm/group_event_state/v0_2/h2b"


def load_dataset_coverage(dataset_root: Path) -> dict[str, dict]:
    """subject -> {covered_records, n_events, sessions, seizures, geometry}."""

    out: dict[str, dict] = {}
    for sub in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        index_path = sub / "index.json"
        if not index_path.exists():
            continue
        idx = json.loads(index_path.read_text())
        records = sorted({Path(s).stem for s in idx.get("source_shards", [])})
        out[sub.name] = {
            "covered_records": records,
            "n_records": len(records),
            "n_events": idx.get("n_events"),
            "n_events_interictal": idx.get("n_events_interictal"),
            "n_sessions": len(idx.get("sessions", [])),
            "n_seizures_v0_1_index": idx.get("n_seizures"),
            "dataset": idx.get("dataset"),
            "n_contacts": idx.get("n_contacts"),
            "native_rate_hz": idx.get("native_rate_hz"),
            "geometry_status": (idx.get("geometry") or {}).get("status"),
            "coord_space": (idx.get("geometry") or {}).get("coord_space"),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", type=Path, default=Path("/data/hfosp_group_event_state_v0_1/dataset"))
    ap.add_argument("--block-inventory", type=Path, default=V0_1 / "block_inventory.csv")
    ap.add_argument("--epilepsiae-seizures", type=Path, default=MAIN_TREE / "results/epilepsiae_seizure_inventory.csv")
    ap.add_argument("--yuquan-seizures", type=Path, default=ROOT / "results/dataset_inventory/yuquan_seizure_inventory.csv")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    coverage = load_dataset_coverage(args.dataset_root)
    covered_by_subject = {s: set(v["covered_records"]) for s, v in coverage.items()}

    blocks = list(csv.DictReader(args.block_inventory.open()))
    # Coverage clause: keep only blocks that actually entered the dataset.
    kept, dropped_not_in_dataset = [], 0
    for row in blocks:
        subj = row["subject"]
        if subj in covered_by_subject and row["record_name"] in covered_by_subject[subj]:
            kept.append(row)
        else:
            dropped_not_in_dataset += 1
    index = build_recording_index(kept)

    results = {}
    all_entries = []
    for dataset, path in (("epilepsiae", args.epilepsiae_seizures), ("yuquan", args.yuquan_seizures)):
        subjects = {s for s in coverage if coverage[s]["dataset"] == dataset}
        rows = list(csv.DictReader(Path(path).open()))
        res = crosswalk_seizures(rows, index, dataset, subjects)
        # C3: reconciliation is asserted, not assumed.
        assert sum(res.disposition_counts.values()) == res.n_input_rows == len(res.entries)
        results[dataset] = res
        all_entries.extend(res.entries)

    out_root = args.out_root
    (out_root / "support").mkdir(parents=True, exist_ok=True)

    csv_path = out_root / "support/seizure_crosswalk.csv"
    tmp = csv_path.with_suffix(".csv.tmp")
    cols = [
        "dataset", "subject", "seizure_id", "recording_code", "disposition", "flags",
        "onset_epoch", "offset_epoch", "duration_sec", "block_record_name",
        "onset_offset_into_block_sec", "onset_gap_to_recording_sec",
        "containing_recording_codes",
    ]
    with tmp.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for e in all_entries:
            w.writerow([
                e.dataset, e.subject, e.seizure_id, e.recording_code, e.disposition.value,
                "|".join(e.flags), f"{e.onset_epoch:.6f}", f"{e.offset_epoch:.6f}",
                f"{e.duration_sec:.6f}", e.block_record_name or "",
                "" if e.onset_offset_into_block_sec is None else f"{e.onset_offset_into_block_sec:.6f}",
                "" if e.onset_gap_to_recording_sec is None else f"{e.onset_gap_to_recording_sec:.6f}",
                "|".join(e.containing_recording_codes),
            ])
    tmp.rename(csv_path)

    summary = {
        "generated_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "inputs": {
            "dataset_root": str(args.dataset_root),
            "block_inventory": str(args.block_inventory),
            "epilepsiae_seizures": str(args.epilepsiae_seizures),
            "yuquan_seizures": str(args.yuquan_seizures),
        },
        "coverage": {
            "n_dataset_subjects": len(coverage),
            "n_blocks_in_inventory": len(blocks),
            "n_blocks_kept_as_coverage": len(kept),
            "n_blocks_not_in_dataset": dropped_not_in_dataset,
            "n_recordings_indexed": len(index),
        },
        "per_dataset": {},
        "per_subject": {},
        "provenance_caveats": [
            "Yuquan seizures are pr1 spatial-extent DETECTIONS, not clinical annotations "
            "(v0.1 data contract §11); a subject with zero rows reads as 'not detected', "
            "never as 'no seizures'.",
        ],
    }
    for dataset, res in results.items():
        summary["per_dataset"][dataset] = {
            "n_input_rows": res.n_input_rows,
            "disposition_counts": dict(sorted(res.disposition_counts.items())),
            "dataset_subjects_without_seizure_rows": list(res.dataset_subjects_without_seizure_rows),
            "inventory_subjects_not_in_dataset": list(res.inventory_subjects_not_in_dataset),
        }
    for subject, cov in sorted(coverage.items()):
        dataset = cov["dataset"]
        res = results[dataset]
        summary["per_subject"][subject] = {
            **{k: cov[k] for k in ("dataset", "n_records", "n_events", "n_events_interictal",
                                   "n_sessions", "n_contacts", "native_rate_hz",
                                   "geometry_status", "coord_space", "n_seizures_v0_1_index")},
            "dispositions": dict(sorted(res.per_subject.get(subject, {}).items())),
            "n_matched": res.per_subject.get(subject, {}).get(Disposition.MATCHED.value, 0),
        }

    sj = out_root / "support/seizure_crosswalk_summary.json"
    tmp = sj.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2))
    tmp.rename(sj)

    print(f"wrote {csv_path}")
    print(f"wrote {sj}")
    for dataset, res in results.items():
        print(f"\n[{dataset}] rows={res.n_input_rows}")
        for k, v in sorted(res.disposition_counts.items()):
            print(f"    {k:34s} {v}")
        print(f"    dataset subjects w/o seizure rows : {list(res.dataset_subjects_without_seizure_rows)}")
        print(f"    inventory subjects not in dataset : {list(res.inventory_subjects_not_in_dataset)}")


if __name__ == "__main__":
    main()
