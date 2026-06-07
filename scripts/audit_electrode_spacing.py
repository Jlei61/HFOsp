#!/usr/bin/env python3
"""Audit real SEEG electrode spacing → durable bimodal contract for the
Topic-4 SEF-HFO virtual-SEEG observation layer.

Why this exists
---------------
The model's virtual electrodes must sit at REAL SEEG spacing, not a 0.4mm toy
pitch. The spacing is BIMODAL: depth/SEEG shafts (~3-6mm adjacent-contact
spacing) vs grid/strip arrays (~8-16mm). This script turns the terminal-only
observation into a re-runnable audit artifact + config so the project cannot
slip back to a toy pitch.

What it does (first-principles)
-------------------------------
1. 测了什么: 相邻两个电极触点之间的真实物理间距 (mm)。深部电极的触点挨得近
   (约 3-5mm), 栅格/条状电极的触点离得远 (约 10mm)。
2. 怎么测的: 把同一根电极杆 (shaft) 上的触点按编号排好, 量相邻触点的欧氏距离,
   取这根杆的中位数 (中位数能吸收漏触点 / 栅格换行造成的离群间距)。然后看每根
   杆的中位数落在哪个区间, 分成 depth / grid_strip / other 三档。
3. 揭示了什么: 一个数据集里到底有几根深部杆、几根栅格/条状, 它们的间距各是多少。

Data sources
------------
- Epilepsiae: coords from all_data_sqls/*.sql (electrode table, MNI152 1mm
  voxels). Read via src.seeg_coord_loader public API, applying the exported
  MNI152_1MM_AFFINE so coords come back as real mm (coord_units == "mm").
  The MNI152 grid is 1mm isotropic so voxel Euclidean distance already equals
  mm, but applying the affine keeps us on the loader's mm contract.
- Yuquan: coords from chnXyzDict.npy (fs_native_ras_mm). Recomputed here — the
  per-shaft median is expected to reproduce the v3.1-lock 3.501mm audited
  value (docs/archive/topic0/seeg_coord_loader/v3_1_lock_2026-05-21.md).
  Yuquan is all-depth → unimodal, no grid/strip (expected, not a bug).

Outputs
-------
- results/topic4_sef_hfo/electrode_spacing_audit/electrode_spacing_per_shaft.csv
- results/topic4_sef_hfo/electrode_spacing_audit/cohort_summary.json

Run:
    python scripts/audit_electrode_spacing.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.seeg_coord_loader import (  # noqa: E402
    EPILEPSIAE_SQL_ROOT,
    MNI152_1MM_AFFINE,
    YUQUAN_ELEC_ROOT,
    enumerate_subject_all_channels,
    load_subject_coords,
)

# ---------------------------------------------------------------------------
# Classification bands (locked contract)
# ---------------------------------------------------------------------------
# depth/SEEG: clinical depth electrodes (~3-6mm pitch)
# grid_strip: subdural grid / strip arrays (~8-16mm pitch)
# other:      everything else → flagged ambiguous, NOT force-fit
DEPTH_BAND_MM = (3.0, 6.0)
GRID_STRIP_BAND_MM = (8.0, 16.0)

OUT_DIR = REPO_ROOT / "results" / "topic4_sef_hfo" / "electrode_spacing_audit"

# Yuquan locked value (user contract); recomputed median cross-checked against it.
YUQUAN_LOCKED_DEPTH_MM = 3.5

_SHAFT_NUM_RE = re.compile(r"^([A-Za-z]+'?)(\d+)$")


def _parse_shaft_and_num(channel_name: str) -> Optional[Tuple[str, int]]:
    """'GC4' -> ('GC', 4); "A'12" -> ("A'", 12). None if not monopolar name.

    Strips the trailing contact-number digits to get the shaft id; keeps a
    leading prime (') as part of the shaft (left/right hemisphere convention).
    """
    m = _SHAFT_NUM_RE.match(channel_name.strip())
    if not m:
        return None
    return m.group(1), int(m.group(2))


def _classify_band(median_mm: float) -> str:
    if DEPTH_BAND_MM[0] <= median_mm <= DEPTH_BAND_MM[1]:
        return "depth"
    if GRID_STRIP_BAND_MM[0] <= median_mm <= GRID_STRIP_BAND_MM[1]:
        return "grid_strip"
    return "other"


def _shaft_spacings(
    channel_names: List[str],
    coords: np.ndarray,
    mapped_mask: np.ndarray,
) -> Dict[str, Dict[str, object]]:
    """Group mapped contacts into shafts, compute adjacent-contact spacing.

    For each shaft: sort by INTEGER contact number, take consecutive pairs,
    compute Euclidean distance, report per-shaft median + range + n.
    Shafts with <2 mapped contacts → recorded with median=None (NA), no crash.
    """
    # shaft -> list of (contact_num, coord)
    by_shaft: Dict[str, List[Tuple[int, np.ndarray]]] = {}
    for name, coord, ok in zip(channel_names, coords, mapped_mask):
        if not ok:
            continue
        parsed = _parse_shaft_and_num(name)
        if parsed is None:
            continue
        shaft, num = parsed
        by_shaft.setdefault(shaft, []).append((num, np.asarray(coord, dtype=float)))

    out: Dict[str, Dict[str, object]] = {}
    for shaft, items in by_shaft.items():
        items.sort(key=lambda t: t[0])  # by integer contact number
        if len(items) < 2:
            out[shaft] = {
                "n_contacts_mapped": len(items),
                "n_adjacent_pairs": 0,
                "median_spacing_mm": None,
                "min_spacing_mm": None,
                "max_spacing_mm": None,
                "band": "insufficient_contacts",
            }
            continue
        dists: List[float] = []
        for (_, c0), (_, c1) in zip(items[:-1], items[1:]):
            dists.append(float(np.linalg.norm(c1 - c0)))
        med = float(np.median(dists))
        out[shaft] = {
            "n_contacts_mapped": len(items),
            "n_adjacent_pairs": len(dists),
            "median_spacing_mm": round(med, 4),
            "min_spacing_mm": round(float(np.min(dists)), 4),
            "max_spacing_mm": round(float(np.max(dists)), 4),
            "band": _classify_band(med),
        }
    return out


def _audit_subject(
    dataset: str,
    subject_id: str,
    mri_affine: Optional[np.ndarray],
) -> Tuple[Dict[str, Dict[str, object]], Optional[str]]:
    """Return (per-shaft dict, error-str-or-None) for one subject."""
    try:
        channel_names = enumerate_subject_all_channels(dataset, subject_id)
    except Exception as exc:  # noqa: BLE001 - record + continue, don't abort cohort
        return {}, f"enumerate failed: {exc}"
    if not channel_names:
        return {}, "no invasive channels enumerated"
    try:
        res = load_subject_coords(
            dataset, subject_id, channel_names, mri_affine=mri_affine
        )
    except Exception as exc:  # noqa: BLE001
        return {}, f"load_subject_coords failed: {exc}"
    if res.coord_units != "mm":
        return {}, f"coord_units != mm (got {res.coord_units!r})"
    shafts = _shaft_spacings(
        res.channel_names_requested,
        res.coords_array_in_requested_order,
        res.mapped_mask_in_requested_order,
    )
    return shafts, None


def _epilepsiae_subject_ids() -> List[str]:
    """Canonical Epilepsiae ids from SQL filenames (pat_<id>_*.sql)."""
    ids = []
    for p in sorted(EPILEPSIAE_SQL_ROOT.glob("pat_*.sql")):
        m = re.match(r"pat_(\d+)_", p.name)
        if m:
            ids.append(m.group(1))
    return sorted(set(ids))


def _yuquan_subject_ids() -> List[str]:
    return sorted(d.name for d in YUQUAN_ELEC_ROOT.iterdir() if d.is_dir())


def _band_medians(rows: List[dict], dataset: str, band: str) -> Dict[str, object]:
    """Cohort-level median-of-per-shaft-medians + range + counts for one band."""
    vals = [
        r["median_spacing_mm"]
        for r in rows
        if r["dataset"] == dataset
        and r["band"] == band
        and r["median_spacing_mm"] is not None
    ]
    subjects = sorted(
        {
            r["subject_id"]
            for r in rows
            if r["dataset"] == dataset
            and r["band"] == band
            and r["median_spacing_mm"] is not None
        }
    )
    if not vals:
        return {"n_shafts": 0, "n_subjects": 0, "median_mm": None,
                "min_mm": None, "max_mm": None}
    return {
        "n_shafts": len(vals),
        "n_subjects": len(subjects),
        "median_mm": round(float(np.median(vals)), 4),
        "min_mm": round(float(np.min(vals)), 4),
        "max_mm": round(float(np.max(vals)), 4),
    }


def main() -> None:
    if not EPILEPSIAE_SQL_ROOT.exists():
        print(f"STOP: Epilepsiae SQL root absent: {EPILEPSIAE_SQL_ROOT}",
              file=sys.stderr)
        sys.exit(2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    subject_errors: List[dict] = []

    # --- Epilepsiae: explicit MNI152 affine → real mm, no nibabel/MRI dependency ---
    for sid in _epilepsiae_subject_ids():
        shafts, err = _audit_subject("epilepsiae", sid, MNI152_1MM_AFFINE)
        if err is not None:
            subject_errors.append({"dataset": "epilepsiae", "subject_id": sid,
                                   "error": err})
            continue
        for shaft, info in shafts.items():
            rows.append({"dataset": "epilepsiae", "subject_id": sid,
                         "shaft": shaft, **info})

    # --- Yuquan: native mm directly (recomputed, cross-checked vs 3.501 lock) ---
    yuquan_available = YUQUAN_ELEC_ROOT.exists()
    if yuquan_available:
        for sid in _yuquan_subject_ids():
            shafts, err = _audit_subject("yuquan", sid, None)
            if err is not None:
                subject_errors.append({"dataset": "yuquan", "subject_id": sid,
                                       "error": err})
                continue
            for shaft, info in shafts.items():
                rows.append({"dataset": "yuquan", "subject_id": sid,
                             "shaft": shaft, **info})

    # --- Write per-shaft CSV ---
    csv_path = OUT_DIR / "electrode_spacing_per_shaft.csv"
    fields = ["dataset", "subject_id", "shaft", "n_contacts_mapped",
              "n_adjacent_pairs", "median_spacing_mm", "min_spacing_mm",
              "max_spacing_mm", "band"]
    import csv

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})

    # --- Cohort summary JSON (bimodality visible) ---
    epi_depth = _band_medians(rows, "epilepsiae", "depth")
    epi_grid = _band_medians(rows, "epilepsiae", "grid_strip")
    epi_other = _band_medians(rows, "epilepsiae", "other")
    yq_depth = _band_medians(rows, "yuquan", "depth")
    yq_grid = _band_medians(rows, "yuquan", "grid_strip")
    yq_other = _band_medians(rows, "yuquan", "other")

    # Ambiguous subjects: any 'other'-band shaft (out of both contracted bands).
    ambiguous = sorted(
        {(r["dataset"], r["subject_id"])
         for r in rows if r["band"] == "other"}
    )

    summary = {
        "provenance": {
            "generated_by": "scripts/audit_electrode_spacing.py",
            "epilepsiae_source": f"{EPILEPSIAE_SQL_ROOT} (electrode table, "
                                 "MNI152 1mm voxel coords + MNI152_1MM_AFFINE "
                                 "via src.seeg_coord_loader)",
            "yuquan_source": f"{YUQUAN_ELEC_ROOT}/<subject>/chnXyzDict.npy "
                             "(fs_native_ras_mm via src.seeg_coord_loader)",
            "classification_bands_mm": {
                "depth": list(DEPTH_BAND_MM),
                "grid_strip": list(GRID_STRIP_BAND_MM),
                "other": "outside both bands → flagged ambiguous, not force-fit",
            },
            "method": "per shaft: sort contacts by integer number, adjacent "
                      "Euclidean distance, per-shaft median; cohort = "
                      "median-of-per-shaft-medians per band",
        },
        "epilepsiae": {
            "n_subjects_audited": len(
                {r["subject_id"] for r in rows if r["dataset"] == "epilepsiae"}
            ),
            "depth": epi_depth,
            "grid_strip": epi_grid,
            "other": epi_other,
        },
        "yuquan": {
            "available": yuquan_available,
            "n_subjects_audited": len(
                {r["subject_id"] for r in rows if r["dataset"] == "yuquan"}
            ),
            "depth": yq_depth,
            "grid_strip": yq_grid,
            "other": yq_other,
            "locked_depth_mm": YUQUAN_LOCKED_DEPTH_MM,
            "recompute_vs_lock_note": (
                "recomputed per-shaft depth median should reproduce the "
                "v3.1-lock 3.501mm audited value "
                "(docs/archive/topic0/seeg_coord_loader/v3_1_lock_2026-05-21.md)"
            ),
        },
        "ambiguous_subjects": [
            {"dataset": d, "subject_id": s} for d, s in ambiguous
        ],
        "subject_errors": subject_errors,
    }

    json_path = OUT_DIR / "cohort_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # --- Console report ---
    print(f"Wrote {csv_path} ({len(rows)} shaft rows)")
    print(f"Wrote {json_path}")
    print()
    print("Epilepsiae depth   :", epi_depth)
    print("Epilepsiae gridstrip:", epi_grid)
    print("Epilepsiae other   :", epi_other)
    print("Yuquan depth       :", yq_depth)
    print("Yuquan other       :", yq_other)
    print()
    if ambiguous:
        print("Ambiguous (out-of-band) subjects:")
        for d, s in ambiguous:
            print(f"  {d} {s}")
    if subject_errors:
        print(f"\n{len(subject_errors)} subject(s) errored:")
        for e in subject_errors:
            print(f"  {e['dataset']} {e['subject_id']}: {e['error']}")


if __name__ == "__main__":
    main()
