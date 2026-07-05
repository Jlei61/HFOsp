#!/usr/bin/env python3
"""Import a reconstructed Yuquan electrode CSV into chnXyzDict.npy.

The Yuquan coordinate loader consumes:

    patients_elecs_reGen/<subject>/chnXyzDict.npy

where the file is a Python dict mapping shaft name -> (n_contacts, 3) ndarray
in subject-native RAS millimetres. This script converts an exported electrode
table with contact-level coordinates into that contract and writes a QC sidecar.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.seeg_coord_loader import YUQUAN_ELEC_ROOT, load_subject_coords  # noqa: E402


YUQUAN_EDF_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")

COORD_COLUMNS = {
    "tkrRAS": ("tkrRAS_X", "tkrRAS_Y", "tkrRAS_Z"),
    "MNI305": ("MNI305_X", "MNI305_Y", "MNI305_Z"),
    "MNI152": ("MNI152_X", "MNI152_Y", "MNI152_Z"),
}

REQUIRED_COLUMNS = ("Label", "ElectrodeName", "ContactNumber")
_CHANNEL_RE = re.compile(r"^([A-Za-z]+'?)(\d+)$")


class ImportErrorWithContext(ValueError):
    """Raised for CSV content that cannot be converted without ambiguity."""


@dataclass(frozen=True)
class ContactRow:
    label: str
    shaft: str
    contact_number: int
    coord: Tuple[float, float, float]
    status: str
    position_status: str
    location_type: str


def _parse_int(value: str, *, column: str, label: str) -> int:
    text = str(value).strip()
    try:
        as_float = float(text)
    except ValueError as exc:
        raise ImportErrorWithContext(
            f"{label}: column {column} must be an integer, got {value!r}"
        ) from exc
    if not as_float.is_integer():
        raise ImportErrorWithContext(
            f"{label}: column {column} must be an integer, got {value!r}"
        )
    return int(as_float)


def _parse_float(value: str, *, column: str, label: str) -> float:
    text = str(value).strip()
    if text == "":
        raise ImportErrorWithContext(f"{label}: missing coordinate column {column}")
    try:
        out = float(text)
    except ValueError as exc:
        raise ImportErrorWithContext(
            f"{label}: column {column} must be numeric, got {value!r}"
        ) from exc
    if not np.isfinite(out):
        raise ImportErrorWithContext(
            f"{label}: column {column} must be finite, got {value!r}"
        )
    return out


def _expected_label(shaft: str, contact_number: int) -> str:
    return f"{shaft}{contact_number}"


def _channel_sort_key(name: str) -> Tuple[str, int, str]:
    first = name.split("-", 1)[0]
    m = _CHANNEL_RE.match(first)
    if not m:
        return (first, 0, name)
    return (m.group(1), int(m.group(2)), name)


def _count_values(rows: Iterable[ContactRow], attr: str) -> Dict[str, int]:
    return dict(Counter(str(getattr(r, attr) or "") for r in rows))


def read_electrode_csv(
    csv_path: Path,
    *,
    coord_space: str = "tkrRAS",
) -> List[ContactRow]:
    """Read and validate contact-level rows from an exported electrode CSV."""

    if coord_space not in COORD_COLUMNS:
        raise ValueError(f"coord_space must be one of {sorted(COORD_COLUMNS)}")
    coord_cols = COORD_COLUMNS[coord_space]

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = [c for c in (*REQUIRED_COLUMNS, *coord_cols) if c not in fieldnames]
        if missing:
            raise ImportErrorWithContext(
                f"{csv_path} missing required columns: {missing}"
            )

        rows: List[ContactRow] = []
        seen_labels = set()
        seen_contacts = set()
        for raw in reader:
            label = str(raw.get("Label", "")).strip()
            shaft = str(raw.get("ElectrodeName", "")).strip()
            if not label:
                raise ImportErrorWithContext("blank Label is not allowed")
            if not shaft:
                raise ImportErrorWithContext(f"{label}: blank ElectrodeName")
            if label in seen_labels:
                raise ImportErrorWithContext(f"duplicate Label {label!r}")
            seen_labels.add(label)

            contact_number = _parse_int(
                raw.get("ContactNumber", ""),
                column="ContactNumber",
                label=label,
            )
            if contact_number < 1:
                raise ImportErrorWithContext(
                    f"{label}: ContactNumber must be >= 1, got {contact_number}"
                )
            expected = _expected_label(shaft, contact_number)
            if label != expected:
                raise ImportErrorWithContext(
                    f"{label}: Label does not match ElectrodeName+ContactNumber "
                    f"({expected!r})"
                )

            contact_key = (shaft, contact_number)
            if contact_key in seen_contacts:
                raise ImportErrorWithContext(
                    f"duplicate contact {shaft}{contact_number}"
                )
            seen_contacts.add(contact_key)

            coord = tuple(
                _parse_float(raw.get(c, ""), column=c, label=label)
                for c in coord_cols
            )
            rows.append(
                ContactRow(
                    label=label,
                    shaft=shaft,
                    contact_number=contact_number,
                    coord=coord,  # type: ignore[arg-type]
                    status=str(raw.get("Status", "")).strip(),
                    position_status=str(raw.get("PositionStatus", "")).strip(),
                    location_type=str(raw.get("LocationType", "")).strip(),
                )
            )

    if not rows:
        raise ImportErrorWithContext(f"{csv_path} contains no contact rows")
    return rows


def build_chn_xyz_dict(rows: Sequence[ContactRow]) -> Dict[str, np.ndarray]:
    """Build shaft -> contact coordinate array.

    The current Yuquan loader resolves contact N as row N-1. Therefore each
    shaft must contain a contiguous 1..max(ContactNumber) sequence.
    """

    by_shaft: Dict[str, List[ContactRow]] = {}
    for row in rows:
        by_shaft.setdefault(row.shaft, []).append(row)

    out: Dict[str, np.ndarray] = {}
    for shaft, items in sorted(by_shaft.items()):
        items = sorted(items, key=lambda r: r.contact_number)
        nums = [r.contact_number for r in items]
        expected = list(range(1, max(nums) + 1))
        if nums != expected:
            raise ImportErrorWithContext(
                f"shaft {shaft!r} contact numbers must be contiguous 1..N; "
                f"got {nums}, expected {expected}"
            )
        out[shaft] = np.asarray([r.coord for r in items], dtype=float)
        if out[shaft].shape != (len(items), 3):
            raise ImportErrorWithContext(
                f"shaft {shaft!r} produced invalid shape {out[shaft].shape}"
            )
    return out


def _spacing_qc(
    rows: Sequence[ContactRow],
    *,
    low_mm: float,
    high_mm: float,
) -> Dict[str, object]:
    by_shaft: Dict[str, List[ContactRow]] = {}
    for row in rows:
        by_shaft.setdefault(row.shaft, []).append(row)

    all_dists: List[float] = []
    per_shaft: Dict[str, Dict[str, object]] = {}
    outliers: List[Dict[str, object]] = []
    for shaft, items in sorted(by_shaft.items()):
        items = sorted(items, key=lambda r: r.contact_number)
        dists: List[float] = []
        for a, b in zip(items[:-1], items[1:]):
            dist = float(np.linalg.norm(np.asarray(b.coord) - np.asarray(a.coord)))
            dists.append(dist)
            all_dists.append(dist)
            if dist < low_mm or dist > high_mm:
                outliers.append(
                    {
                        "shaft": shaft,
                        "left": a.label,
                        "right": b.label,
                        "distance_mm": round(dist, 4),
                        "left_status": a.status,
                        "right_status": b.status,
                    }
                )
        if dists:
            arr = np.asarray(dists, dtype=float)
            per_shaft[shaft] = {
                "n_contacts": len(items),
                "n_adjacent_pairs": len(dists),
                "median_mm": round(float(np.median(arr)), 4),
                "min_mm": round(float(np.min(arr)), 4),
                "max_mm": round(float(np.max(arr)), 4),
            }
        else:
            per_shaft[shaft] = {
                "n_contacts": len(items),
                "n_adjacent_pairs": 0,
                "median_mm": None,
                "min_mm": None,
                "max_mm": None,
            }

    all_arr = np.asarray(all_dists, dtype=float)
    return {
        "adjacent_pair_count": int(all_arr.size),
        "median_mm": round(float(np.median(all_arr)), 4) if all_arr.size else None,
        "p05_mm": round(float(np.percentile(all_arr, 5)), 4) if all_arr.size else None,
        "p95_mm": round(float(np.percentile(all_arr, 95)), 4) if all_arr.size else None,
        "min_mm": round(float(np.min(all_arr)), 4) if all_arr.size else None,
        "max_mm": round(float(np.max(all_arr)), 4) if all_arr.size else None,
        "outlier_rule_mm": {"low": low_mm, "high": high_mm},
        "outliers": outliers,
        "per_shaft": per_shaft,
    }


def _collect_lagpat_channels(subject_edf_dir: Path) -> List[str]:
    channels = set()
    files = sorted(subject_edf_dir.glob("*_lagPat_withFreqCent.npz"))
    if not files:
        files = sorted(subject_edf_dir.glob("*_lagPat.npz"))
    for path in files:
        with np.load(path, allow_pickle=True) as z:
            if "chnNames" not in z.files:
                continue
            channels.update(str(x) for x in z["chnNames"])
    return sorted(channels, key=_channel_sort_key)


def _collect_gpu_channels(subject_edf_dir: Path) -> List[str]:
    channels = set()
    for path in sorted(subject_edf_dir.glob("*_gpu.npz")):
        with np.load(path, allow_pickle=True) as z:
            if "chns_names" not in z.files:
                continue
            channels.update(str(x) for x in z["chns_names"])
    return sorted(channels, key=_channel_sort_key)


def _validate_name_overlap(
    rows: Sequence[ContactRow],
    *,
    subject_edf_dir: Path,
) -> Dict[str, object]:
    labels = {r.label for r in rows}
    report: Dict[str, object] = {
        "subject_edf_dir": str(subject_edf_dir),
        "subject_edf_dir_exists": subject_edf_dir.exists(),
    }
    if not subject_edf_dir.exists():
        report["status"] = "skipped_missing_subject_edf_dir"
        return report

    lagpat_channels = _collect_lagpat_channels(subject_edf_dir)
    lagpat_missing = sorted(set(lagpat_channels) - labels, key=_channel_sort_key)

    gpu_channels = _collect_gpu_channels(subject_edf_dir)
    gpu_missing = []
    gpu_partial = []
    gpu_ok = 0
    for ch in gpu_channels:
        endpoints = ch.split("-", 1) if "-" in ch else [ch]
        found = [ep in labels for ep in endpoints]
        if all(found):
            gpu_ok += 1
        elif any(found):
            gpu_partial.append({"channel": ch, "endpoints": endpoints, "found": found})
        else:
            gpu_missing.append(ch)

    report.update(
        {
            "status": "ok" if not lagpat_missing and not gpu_missing and not gpu_partial else "mismatch",
            "lagpat": {
                "n_channels": len(lagpat_channels),
                "n_missing_in_csv": len(lagpat_missing),
                "missing_in_csv": lagpat_missing,
            },
            "gpu": {
                "n_channels": len(gpu_channels),
                "n_both_endpoints_found": gpu_ok,
                "n_missing_pairs": len(gpu_missing),
                "missing_pairs": gpu_missing,
                "n_partial_pairs": len(gpu_partial),
                "partial_pairs": gpu_partial,
            },
        }
    )
    return report


def _write_temp_chn_xyz(root: Path, subject: str, shaft_dict: Dict[str, np.ndarray]) -> None:
    subject_dir = root / subject
    subject_dir.mkdir(parents=True, exist_ok=True)
    with (subject_dir / "chnXyzDict.npy").open("wb") as f:
        np.save(f, shaft_dict, allow_pickle=True)


def _validate_loader_contract(
    *,
    subject: str,
    shaft_dict: Dict[str, np.ndarray],
    lagpat_channels: Sequence[str],
    gpu_channels: Sequence[str],
) -> Dict[str, object]:
    report: Dict[str, object] = {}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_temp_chn_xyz(root, subject, shaft_dict)

        if lagpat_channels:
            res = load_subject_coords("yuquan", subject, lagpat_channels, yuquan_root=root)
            report["lagpat_loader"] = {
                "n_requested": len(lagpat_channels),
                "n_mapped": int(res.mapped_mask_in_requested_order.sum()),
                "missing": [
                    {"channel": m.channel, "reason": m.reason}
                    for m in res.missing
                ],
            }

        if gpu_channels:
            res = load_subject_coords("yuquan", subject, gpu_channels, yuquan_root=root)
            report["gpu_loader"] = {
                "n_requested": len(gpu_channels),
                "n_mapped": int(res.mapped_mask_in_requested_order.sum()),
                "missing": [
                    {"channel": m.channel, "reason": m.reason}
                    for m in res.missing
                ],
                "n_bipolar_resolution": len(res.bipolar_resolution),
            }

    return report


def build_metadata(
    *,
    subject: str,
    csv_path: Path,
    output_npy: Path,
    coord_space: str,
    rows: Sequence[ContactRow],
    spacing: Dict[str, object],
    validation: Dict[str, object],
    source_csv_copy: Optional[Path],
) -> Dict[str, object]:
    shaft_counts = Counter(r.shaft for r in rows)
    metadata = {
        "schema_version": "yuquan_electrode_csv_import_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "subject": subject,
        "source_csv": str(csv_path.resolve()),
        "source_csv_copy": str(source_csv_copy.resolve()) if source_csv_copy else None,
        "output_npy": str(output_npy.resolve()),
        "coord_source_columns": COORD_COLUMNS[coord_space],
        "coord_source_space": coord_space,
        "pipeline_coord_space": (
            "fs_native_ras_mm"
            if coord_space == "tkrRAS"
            else f"nondefault_{coord_space}_mm"
        ),
        "coord_units": "mm",
        "n_contacts": len(rows),
        "n_shafts": len(shaft_counts),
        "shaft_counts": dict(sorted(shaft_counts.items())),
        "status_counts": _count_values(rows, "status"),
        "position_status_counts": _count_values(rows, "position_status"),
        "location_type_counts": _count_values(rows, "location_type"),
        "spacing_qc": spacing,
        "validation": validation,
        "notes": [
            "Yuquan loader resolves contact N as row N-1 in each shaft array.",
            "tkrRAS is the default because existing Yuquan loader contract is subject-native RAS mm.",
            "Interpolated/extrapolated contacts are retained, but their counts and spacing outliers are recorded here.",
        ],
    }
    return metadata


def _atomic_write_json(path: Path, payload: Dict[str, object]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_npy(path: Path, shaft_dict: Dict[str, np.ndarray]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, shaft_dict, allow_pickle=True)
    os.replace(tmp, path)


def import_electrode_csv(
    *,
    subject: str,
    csv_path: Path,
    output_root: Path = YUQUAN_ELEC_ROOT,
    coord_space: str = "tkrRAS",
    subject_edf_dir: Optional[Path] = None,
    overwrite: bool = False,
    dry_run: bool = False,
    copy_source: bool = True,
    allow_channel_mismatch: bool = False,
    spacing_low_mm: float = 2.0,
    spacing_high_mm: float = 5.0,
) -> Dict[str, object]:
    """Convert a CSV and optionally write pipeline-ready Yuquan coord files."""

    csv_path = csv_path.expanduser().resolve()
    rows = read_electrode_csv(csv_path, coord_space=coord_space)
    shaft_dict = build_chn_xyz_dict(rows)
    spacing = _spacing_qc(rows, low_mm=spacing_low_mm, high_mm=spacing_high_mm)

    if subject_edf_dir is None:
        subject_edf_dir = YUQUAN_EDF_ROOT / subject
    validation = _validate_name_overlap(rows, subject_edf_dir=subject_edf_dir)
    if (
        validation.get("status") == "mismatch"
        and not allow_channel_mismatch
    ):
        raise ImportErrorWithContext(
            "CSV labels do not fully cover existing lagPat/GPU channels; "
            "rerun with --allow-channel-mismatch only for non-production imports. "
            f"Validation: {validation}"
        )

    if subject_edf_dir.exists():
        lagpat_channels = _collect_lagpat_channels(subject_edf_dir)
        gpu_channels = _collect_gpu_channels(subject_edf_dir)
        validation["loader_contract"] = _validate_loader_contract(
            subject=subject,
            shaft_dict=shaft_dict,
            lagpat_channels=lagpat_channels,
            gpu_channels=gpu_channels,
        )

    subject_out_dir = output_root / subject
    output_npy = subject_out_dir / "chnXyzDict.npy"
    metadata_json = subject_out_dir / "chnXyzDict.import_metadata.json"
    source_csv_copy = subject_out_dir / f"{csv_path.stem}.source.csv" if copy_source else None
    metadata = build_metadata(
        subject=subject,
        csv_path=csv_path,
        output_npy=output_npy,
        coord_space=coord_space,
        rows=rows,
        spacing=spacing,
        validation=validation,
        source_csv_copy=source_csv_copy,
    )

    if dry_run:
        metadata["dry_run"] = True
        return metadata

    if output_npy.exists() and not overwrite:
        raise FileExistsError(
            f"{output_npy} already exists; pass --overwrite to replace it"
        )

    subject_out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_npy(output_npy, shaft_dict)
    if source_csv_copy is not None:
        shutil.copy2(csv_path, source_csv_copy)

    # Verify the actual output path after writing, not only the temp fixture.
    res = load_subject_coords(
        "yuquan",
        subject,
        [rows[0].label],
        yuquan_root=output_root,
    )
    if int(res.mapped_mask_in_requested_order.sum()) != 1:
        raise RuntimeError(f"post-write loader verification failed for {rows[0].label}")

    metadata["written"] = {
        "chnXyzDict": str(output_npy.resolve()),
        "metadata": str(metadata_json.resolve()),
        "source_csv_copy": str(source_csv_copy.resolve()) if source_csv_copy else None,
    }
    metadata["post_write_loader_check"] = {
        "channel": rows[0].label,
        "mapped": int(res.mapped_mask_in_requested_order.sum()),
        "coord_space": res.coord_space,
        "coord_units": res.coord_units,
    }
    _atomic_write_json(metadata_json, metadata)
    return metadata


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert a reconstructed Yuquan electrode CSV to chnXyzDict.npy"
    )
    p.add_argument("--subject", required=True, help="Yuquan subject id, e.g. litengsheng")
    p.add_argument("--csv", required=True, type=Path, help="Exported electrode CSV")
    p.add_argument(
        "--output-root",
        type=Path,
        default=YUQUAN_ELEC_ROOT,
        help=f"Yuquan coord root (default: {YUQUAN_ELEC_ROOT})",
    )
    p.add_argument(
        "--coord-space",
        choices=sorted(COORD_COLUMNS),
        default="tkrRAS",
        help="Coordinate columns to save; tkrRAS is the Yuquan pipeline default",
    )
    p.add_argument(
        "--subject-edf-dir",
        type=Path,
        default=None,
        help="Optional subject EDF/HFO artifact dir for channel-overlap validation",
    )
    p.add_argument("--overwrite", action="store_true", help="Replace existing chnXyzDict.npy")
    p.add_argument("--dry-run", action="store_true", help="Validate and print metadata without writing")
    p.add_argument(
        "--no-copy-source",
        action="store_true",
        help="Do not copy the input CSV into the subject coordinate directory",
    )
    p.add_argument(
        "--allow-channel-mismatch",
        action="store_true",
        help="Do not fail if existing lagPat/GPU channels are not fully covered",
    )
    p.add_argument("--spacing-low-mm", type=float, default=2.0)
    p.add_argument("--spacing-high-mm", type=float, default=5.0)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    metadata = import_electrode_csv(
        subject=args.subject,
        csv_path=args.csv,
        output_root=args.output_root,
        coord_space=args.coord_space,
        subject_edf_dir=args.subject_edf_dir,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        copy_source=not args.no_copy_source,
        allow_channel_mismatch=args.allow_channel_mismatch,
        spacing_low_mm=args.spacing_low_mm,
        spacing_high_mm=args.spacing_high_mm,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
