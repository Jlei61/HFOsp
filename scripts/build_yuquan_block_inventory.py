"""Build yuquan block inventory CSV from EDF metadata probes.

For each yuquan subject under /mnt/yuquan_data/yuquan_24h_edf/, scan all *.edf
files, probe their EDF headers for start_time/duration/sample_rate, and emit
results/dataset_inventory/yuquan_block_inventory.csv with one row per EDF.

Yuquan EDFs are per-recording (~2h each), so block_id == recording_id ==
edf filename stem. There is no separate .head file (unlike epilepsiae).

Also rebuilds yuquan_seizure_inventory.csv with record_start_epoch /
record_end_epoch columns filled by joining against the new block inventory.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import (  # noqa: E402
    _parse_edf_header_for_streaming,
    read_edf_start_time,
)


YUQUAN_DATA_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
PR1_SEIZURE_DIR = ROOT / "results" / "seizure_detection"
INVENTORY_OUT_DIR = ROOT / "results" / "dataset_inventory"


@dataclass(frozen=True)
class BlockProbeResult:
    subject: str
    recording_id: str
    block_id: str
    block_stem: str
    block_start_epoch: float
    block_end_epoch: float
    duration_sec: float
    sample_rate: float
    n_channels_total: int
    head_path: str
    data_path: str
    edf_path: str


def _read_edf_total_n_signals(edf_path: Path) -> int:
    """Read the total signal count from EDF fixed header (bytes 252:256).

    `_parse_edf_header_for_streaming` only returns SEEG-channel info; the
    block inventory's `n_channels_total` is the *total* EDF signal count,
    so we parse the header byte field directly.
    """
    with open(edf_path, "rb") as f:
        fixed = f.read(256)
    return int(float(fixed[252:256].decode("ascii", errors="ignore").strip()))


def _probe_edf_metadata(edf_path: Path) -> Dict[str, float]:
    """Return {'duration_sec', 'sample_rate', 'n_channels_total'} from EDF header."""
    h = _parse_edf_header_for_streaming(edf_path)
    duration_sec = float(h["n_records"] * h["record_duration"])
    sample_rate = float(h["sfreq"])
    n_channels_total = _read_edf_total_n_signals(edf_path)
    return {
        "duration_sec": duration_sec,
        "sample_rate": sample_rate,
        "n_channels_total": n_channels_total,
    }


def probe_one_edf(subject: str, edf_path: Path) -> BlockProbeResult:
    start = read_edf_start_time(edf_path)
    meta = _probe_edf_metadata(edf_path)
    return BlockProbeResult(
        subject=subject,
        recording_id=edf_path.stem,
        block_id=edf_path.stem,
        block_stem=edf_path.stem,
        block_start_epoch=float(start),
        block_end_epoch=float(start) + float(meta["duration_sec"]),
        duration_sec=float(meta["duration_sec"]),
        sample_rate=float(meta["sample_rate"]),
        n_channels_total=int(meta["n_channels_total"]),
        head_path="",
        data_path=str(edf_path),
        edf_path=str(edf_path),
    )


def write_block_inventory_csv(rows: Iterable[BlockProbeResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(BlockProbeResult.__dataclass_fields__.keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))


def collect_subject_edfs(subject: str, root: Path = YUQUAN_DATA_ROOT) -> List[Path]:
    sub_dir = root / subject
    if not sub_dir.is_dir():
        raise FileNotFoundError(f"yuquan subject dir not found: {sub_dir}")
    return sorted(sub_dir.glob("*.edf"))


def build_block_inventory(
    subjects: List[str], *, root: Path = YUQUAN_DATA_ROOT
) -> List[BlockProbeResult]:
    rows: List[BlockProbeResult] = []
    for sid in subjects:
        for edf in collect_subject_edfs(sid, root):
            rows.append(probe_one_edf(sid, edf))
    return rows


def rebuild_seizure_inventory_with_record_epochs(
    block_rows: List[BlockProbeResult],
    pr1_dir: Path = PR1_SEIZURE_DIR,
) -> List[Dict[str, object]]:
    """Re-emit yuquan_seizure_inventory.csv joining pr1_seizure_*.json against block epochs.

    Mirrors load_yuquan_seizure_inventory_rows in src/interictal_synchrony_aggregation.py
    but fills record_start_epoch / record_end_epoch from the block inventory we just built.
    """
    block_lookup: Dict[tuple, BlockProbeResult] = {
        (b.subject, b.recording_id): b for b in block_rows
    }
    subjects_in_scope = {b.subject for b in block_rows}
    out_rows: List[Dict[str, object]] = []
    for json_path in sorted(pr1_dir.glob("pr1_seizure_*.json")):
        if json_path.name in {
            "pr1_seizure_all_yuquan_summary.json",
            "pr1_seizure_all_yuquan_summary_normalized.json",
            "pr1_seizure_offset_audit.json",
        }:
            continue
        payload = json.loads(json_path.read_text())
        subject = payload.get("subject", json_path.stem.replace("pr1_seizure_", ""))
        if subject not in subjects_in_scope:
            continue
        seizure_idx = 0
        for file_row in payload.get("files", []):
            record = str(file_row.get("record", ""))
            blk = block_lookup.get((subject, record))
            for interval in file_row.get("seizure_intervals", []):
                seizure_idx += 1
                if blk is None:
                    raise KeyError(
                        f"seizure {subject}_sz_{seizure_idx:03d} on record {record!r} "
                        f"has no matching block inventory entry — yuquan_block_inventory.csv "
                        f"must be regenerated covering this subject's EDFs"
                    )
                out_rows.append({
                    "subject": subject,
                    "patient_code": subject,
                    "recording_id": record,
                    "record": record,
                    "seizure_id": f"{subject}_sz_{seizure_idx:03d}",
                    "eeg_onset_epoch": interval.get("onset_epoch"),
                    "eeg_offset_epoch": interval.get("offset_epoch"),
                    "eeg_duration_sec": interval.get("duration_sec"),
                    "has_complete_eeg_interval": True,
                    "timezone_name": "Asia/Shanghai",
                    "eeg_onset_local_hour": "",
                    "eeg_onset_day_night": "",
                    "record_start_epoch": blk.block_start_epoch,
                    "record_end_epoch": blk.block_end_epoch,
                })
    subjects_with_rows = {r["subject"] for r in out_rows}
    missing = sorted(subjects_in_scope - subjects_with_rows)
    if missing:
        print(
            f"WARNING: {len(missing)} target subject(s) had no matching pr1_seizure_*.json "
            f"and produced 0 seizure rows: {missing}",
            file=sys.stderr, flush=True,
        )
    return out_rows


def write_seizure_inventory_csv(rows: List[Dict[str, object]], out_path: Path) -> None:
    fieldnames = [
        "subject", "patient_code", "recording_id", "record", "seizure_id",
        "eeg_onset_epoch", "eeg_offset_epoch", "eeg_duration_sec",
        "has_complete_eeg_interval", "timezone_name",
        "eeg_onset_local_hour", "eeg_onset_day_night",
        "record_start_epoch", "record_end_epoch",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Yuquan subject ids to probe. Default: all 9 audit_eligible.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=INVENTORY_OUT_DIR,
        help="Output directory for inventory CSVs.",
    )
    args = parser.parse_args()

    subjects = args.subjects or [
        "gaolan", "huanghanwen", "litengsheng", "pengzihang", "sunyuanxin",
        "xuxinyi", "zhangjinhan", "zhangkexuan", "zhaojinrui",
    ]
    print(f"Probing {len(subjects)} yuquan subjects under {YUQUAN_DATA_ROOT}", flush=True)

    block_rows = build_block_inventory(subjects)
    block_csv = args.out_dir / "yuquan_block_inventory.csv"
    write_block_inventory_csv(block_rows, block_csv)
    print(f"Wrote {len(block_rows)} block rows -> {block_csv}", flush=True)

    seizure_rows = rebuild_seizure_inventory_with_record_epochs(block_rows)
    seizure_csv = args.out_dir / "yuquan_seizure_inventory.csv"
    write_seizure_inventory_csv(seizure_rows, seizure_csv)
    print(f"Wrote {len(seizure_rows)} seizure rows -> {seizure_csv}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
