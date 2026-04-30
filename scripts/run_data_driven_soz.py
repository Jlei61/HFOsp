"""PR-T3-1 — Data-driven ictal-onset SOZ audit CLI.

Usage:

    python scripts/run_data_driven_soz.py --audit
    python scripts/run_data_driven_soz.py --per-subject  # Step 3 (not yet)
    python scripts/run_data_driven_soz.py --cohort-overlap  # Step 4 (not yet)

Step 0 (``--audit``) enumerates the cohort, applies the canonical channel
matcher, and writes ``results/spatial_modulation/data_driven_soz/audit.csv``
with eligibility columns. No M1/M2 computation here.

See ``docs/archive/topic3/pr_t3_1_data_driven_soz_audit_plan_2026-04-30.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Allow ``python scripts/run_data_driven_soz.py`` execution without PYTHONPATH.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_driven_soz import (  # noqa: E402
    NON_SOZ_LABEL,
    SOZ_LABEL,
    UNKNOWN_LABEL,
    annotate_clinical_soz,
)


HFO_DETECTION_DIR = ROOT / "results" / "hfo_detection"
EPILEPSIAE_BLOCK_INVENTORY = ROOT / "results" / "epilepsiae_block_inventory.csv"
EPILEPSIAE_SEIZURE_INVENTORY = ROOT / "results" / "epilepsiae_seizure_inventory.csv"
EPILEPSIAE_SOZ_JSON = ROOT / "results" / "epilepsiae_soz_core_channels.json"
YUQUAN_SOZ_JSON = ROOT / "results" / "yuquan_soz_core_channels.json"
YUQUAN_SEIZURE_DIR = ROOT / "results" / "seizure_detection"
YUQUAN_DATA_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
OUTPUT_DIR = ROOT / "results" / "spatial_modulation" / "data_driven_soz"

NYQUIST_BAND_HIGH = 250.0           # M2 primary band upper edge
NYQUIST_SAFETY = 1.05               # plan §3.4: 5% safety margin
M2_MIN_SFREQ = NYQUIST_BAND_HIGH * 2 * NYQUIST_SAFETY   # = 525 Hz

EXCLUDED_DIR_TOKENS = ("backup", "summary", ".log", ".json")


# ---------------------------------------------------------------------------
# Subject enumeration
# ---------------------------------------------------------------------------


def list_subject_dirs(detection_root: Path) -> List[Tuple[str, str, Path]]:
    """Return ``[(dataset, subject, dir_path)]`` for every clean subject dir.

    Dataset is ``"epilepsiae"`` for numeric IDs and ``"yuquan"`` otherwise.
    Skips backup/summary/log/json artifacts that live next to the dirs.
    """
    out: List[Tuple[str, str, Path]] = []
    if not detection_root.exists():
        return out
    for entry in sorted(detection_root.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if any(tok in name for tok in EXCLUDED_DIR_TOKENS):
            continue
        dataset = "epilepsiae" if name.isdigit() else "yuquan"
        out.append((dataset, name, entry))
    return out


# ---------------------------------------------------------------------------
# Epilepsiae helpers
# ---------------------------------------------------------------------------


def load_epilepsiae_block_inventory(path: Path) -> Dict[str, List[Dict[str, str]]]:
    """Group block_inventory rows by ``subject``.

    Returns ``{subject: [row_dict, ...]}``.
    """
    by_subject: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    with path.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            by_subject[row["subject"]].append(row)
    return dict(by_subject)


def load_epilepsiae_seizure_counts(path: Path) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    with path.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            counts[row["subject"]] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# Yuquan helpers
# ---------------------------------------------------------------------------


def load_yuquan_seizure_count(subject: str) -> Optional[int]:
    path = YUQUAN_SEIZURE_DIR / f"pr1_seizure_{subject}.json"
    if not path.exists():
        return None
    try:
        with path.open() as f:
            data = json.load(f)
    except Exception:
        return None
    total = 0
    for rec in data.get("files", []) or []:
        total += int(rec.get("n_seizure_intervals", 0) or 0)
    return total


def probe_yuquan_sfreq(subject: str) -> Optional[float]:
    """Return raw EDF sampling frequency for one EDF in subject dir.

    Probes the first ``*.edf`` (alphabetical) under
    ``/mnt/yuquan_data/yuquan_24h_edf/<subject>/``. Returns None if the
    directory is missing or unreadable.
    """
    sub_dir = YUQUAN_DATA_ROOT / subject
    if not sub_dir.exists():
        return None
    edfs = sorted(sub_dir.glob("*.edf"))
    if not edfs:
        return None
    try:
        import mne  # local import to avoid mandatory dep when running --help
        raw = mne.io.read_raw_edf(
            str(edfs[0]), preload=False, verbose=False, encoding="latin1"
        )
        return float(raw.info["sfreq"])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# HFO npz inspection
# ---------------------------------------------------------------------------


def list_npz_blocks(subject_dir: Path) -> List[Path]:
    return sorted(subject_dir.glob("*_gpu.npz"))


def npz_block_stem(npz_path: Path) -> str:
    """Strip the ``_gpu.npz`` suffix to recover the block stem."""
    return npz_path.name[: -len("_gpu.npz")]


def load_npz_channels(npz_path: Path) -> List[str]:
    """Return ``chns_names`` from a single HFO npz file."""
    with np.load(npz_path, allow_pickle=True) as data:
        return list(data["chns_names"].tolist())


# ---------------------------------------------------------------------------
# Audit row builder
# ---------------------------------------------------------------------------


AUDIT_COLUMNS = [
    "dataset",
    "subject",
    "n_blocks_actual",
    "n_blocks_expected",
    "n_blocks_stub",
    "hfo_npz_ok",
    "block_count_match",
    "n_seizures",
    "sfreq_min",
    "sfreq_max",
    "sfreq_source",
    "m2_eligible",
    "n_analysis_channels",
    "n_clinical_total",
    "n_clinical_matched",
    "n_clinical_unmatched",
    "n_unknown_channels",
    "unmatched_clinical_names",
    "soz_eligible",
    "audit_eligible",
    "notes",
]


def build_epilepsiae_row(
    subject: str,
    subject_dir: Path,
    block_rows: Optional[List[Dict[str, str]]],
    seizure_count: int,
    soz_map: Dict[str, List[str]],
) -> Dict[str, str]:
    notes: List[str] = []
    npz_files = list_npz_blocks(subject_dir)
    n_actual = len(npz_files)
    n_stub = sum(1 for p in npz_files if p.stat().st_size < 1024)
    hfo_npz_ok = n_actual > 0 and n_stub == 0

    # Cross-reference: only blocks where sample_rate_sql >= min_sfreq=500 are
    # expected to have an npz (per scripts/run_hfo_detection.py).
    expected_blocks = []
    inv_sfreq_by_stem: Dict[str, float] = {}
    if block_rows:
        for r in block_rows:
            try:
                sf = float(r.get("sample_rate_sql") or 0.0)
            except Exception:
                sf = 0.0
            inv_sfreq_by_stem[r["block_stem"]] = sf
            if sf >= 500.0:
                expected_blocks.append(r["block_stem"])
    n_expected = len(expected_blocks) if block_rows else -1

    # Collect sfreqs of *retained* npz blocks (cross-ref via block_stem).
    retained_sfreqs: List[float] = []
    for npz in npz_files:
        stem = npz_block_stem(npz)
        sf = inv_sfreq_by_stem.get(stem, 0.0)
        if sf > 0:
            retained_sfreqs.append(sf)
    if retained_sfreqs:
        sfreq_min = min(retained_sfreqs)
        sfreq_max = max(retained_sfreqs)
        sfreq_source = "block_inventory.sample_rate_sql"
    else:
        sfreq_min = sfreq_max = float("nan")
        sfreq_source = "missing"
        notes.append("no retained block sfreq found")

    m2_eligible = bool(retained_sfreqs) and sfreq_min >= M2_MIN_SFREQ

    block_count_match = (n_expected == n_actual) if n_expected >= 0 else False
    if not block_count_match and n_expected >= 0:
        notes.append(f"block_count mismatch: actual={n_actual} expected={n_expected}")

    # Channels + clinical SOZ matching.
    if n_actual == 0 or not hfo_npz_ok:
        n_channels = 0
        ann: Dict[str, str] = {}
        notes.append("hfo npz missing or stub")
    else:
        analysis_channels = load_npz_channels(npz_files[0])
        n_channels = len(analysis_channels)
        ann = annotate_clinical_soz(analysis_channels, soz_map.get(subject, []))

    n_unknown = sum(1 for v in ann.values() if v == UNKNOWN_LABEL)
    n_matched = sum(1 for v in ann.values() if v == SOZ_LABEL)
    n_clinical_total = len(soz_map.get(subject, []))

    soz_set_norm = {s.strip().upper() for s in soz_map.get(subject, [])}
    matched_contacts: set = set()
    for ch, label in ann.items():
        if label != SOZ_LABEL:
            continue
        norm = ch.strip().upper()
        for prefix in ("EEG ", "EEG_"):
            if norm.startswith(prefix):
                norm = norm[len(prefix):]
        for part in (p.strip() for p in norm.split("-")):
            if part in soz_set_norm:
                matched_contacts.add(part)
    unmatched_names = sorted(soz_set_norm - matched_contacts)
    n_unmatched = len(unmatched_names)

    soz_eligible = n_matched >= 1
    audit_eligible = hfo_npz_ok and seizure_count >= 2 and soz_eligible
    if seizure_count < 2:
        notes.append(f"n_seizures={seizure_count} < 2")
    if not soz_eligible:
        notes.append(f"n_clinical_matched={n_matched} < 1")

    return {
        "dataset": "epilepsiae",
        "subject": subject,
        "n_blocks_actual": n_actual,
        "n_blocks_expected": n_expected if n_expected >= 0 else "",
        "n_blocks_stub": n_stub,
        "hfo_npz_ok": int(hfo_npz_ok),
        "block_count_match": int(block_count_match) if n_expected >= 0 else "",
        "n_seizures": seizure_count,
        "sfreq_min": "" if np.isnan(sfreq_min) else sfreq_min,
        "sfreq_max": "" if np.isnan(sfreq_max) else sfreq_max,
        "sfreq_source": sfreq_source,
        "m2_eligible": int(m2_eligible),
        "n_analysis_channels": n_channels,
        "n_clinical_total": n_clinical_total,
        "n_clinical_matched": n_matched,
        "n_clinical_unmatched": n_unmatched,
        "n_unknown_channels": n_unknown,
        "unmatched_clinical_names": "|".join(unmatched_names),
        "soz_eligible": int(soz_eligible),
        "audit_eligible": int(audit_eligible),
        "notes": "; ".join(notes),
    }


def build_yuquan_row(
    subject: str,
    subject_dir: Path,
    soz_map: Dict[str, List[str]],
) -> Dict[str, str]:
    notes: List[str] = []
    npz_files = list_npz_blocks(subject_dir)
    n_actual = len(npz_files)
    n_stub = sum(1 for p in npz_files if p.stat().st_size < 1024)
    hfo_npz_ok = n_actual > 0 and n_stub == 0

    seizure_count = load_yuquan_seizure_count(subject)
    if seizure_count is None:
        seizure_count = 0
        notes.append("missing pr1_seizure JSON")

    sf = probe_yuquan_sfreq(subject)
    if sf is None:
        sfreq_min = sfreq_max = float("nan")
        sfreq_source = "missing"
        notes.append("could not probe EDF sfreq")
    else:
        sfreq_min = sfreq_max = sf
        sfreq_source = "edf_probe_first_record"

    m2_eligible = (sf is not None) and sf >= M2_MIN_SFREQ

    if n_actual == 0 or not hfo_npz_ok:
        n_channels = 0
        ann: Dict[str, str] = {}
        notes.append("hfo npz missing or stub")
    else:
        analysis_channels = load_npz_channels(npz_files[0])
        n_channels = len(analysis_channels)
        ann = annotate_clinical_soz(analysis_channels, soz_map.get(subject, []))

    n_unknown = sum(1 for v in ann.values() if v == UNKNOWN_LABEL)
    n_matched = sum(1 for v in ann.values() if v == SOZ_LABEL)
    n_clinical_total = len(soz_map.get(subject, []))

    soz_set_norm = {s.strip().upper() for s in soz_map.get(subject, [])}
    matched_contacts: set = set()
    for ch, label in ann.items():
        if label != SOZ_LABEL:
            continue
        norm = ch.strip().upper()
        for prefix in ("EEG ", "EEG_"):
            if norm.startswith(prefix):
                norm = norm[len(prefix):]
        for part in (p.strip() for p in norm.split("-")):
            if part in soz_set_norm:
                matched_contacts.add(part)
    unmatched_names = sorted(soz_set_norm - matched_contacts)
    n_unmatched = len(unmatched_names)

    soz_eligible = n_matched >= 1
    audit_eligible = hfo_npz_ok and seizure_count >= 2 and soz_eligible
    if seizure_count < 2:
        notes.append(f"n_seizures={seizure_count} < 2")
    if not soz_eligible:
        notes.append(f"n_clinical_matched={n_matched} < 1")

    return {
        "dataset": "yuquan",
        "subject": subject,
        "n_blocks_actual": n_actual,
        "n_blocks_expected": "",  # no per-block inventory for yuquan
        "n_blocks_stub": n_stub,
        "hfo_npz_ok": int(hfo_npz_ok),
        "block_count_match": "",
        "n_seizures": seizure_count,
        "sfreq_min": "" if np.isnan(sfreq_min) else sfreq_min,
        "sfreq_max": "" if np.isnan(sfreq_max) else sfreq_max,
        "sfreq_source": sfreq_source,
        "m2_eligible": int(m2_eligible),
        "n_analysis_channels": n_channels,
        "n_clinical_total": n_clinical_total,
        "n_clinical_matched": n_matched,
        "n_clinical_unmatched": n_unmatched,
        "n_unknown_channels": n_unknown,
        "unmatched_clinical_names": "|".join(unmatched_names),
        "soz_eligible": int(soz_eligible),
        "audit_eligible": int(audit_eligible),
        "notes": "; ".join(notes),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_audit(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_csv = output_dir / "audit.csv"

    print("[audit] enumerating subjects under", HFO_DETECTION_DIR, flush=True)
    subjects = list_subject_dirs(HFO_DETECTION_DIR)
    print(f"[audit] found {len(subjects)} subjects "
          f"({sum(1 for d, _, _ in subjects if d == 'epilepsiae')} epilepsiae, "
          f"{sum(1 for d, _, _ in subjects if d == 'yuquan')} yuquan)",
          flush=True)

    print("[audit] loading inventories ...", flush=True)
    epi_blocks = load_epilepsiae_block_inventory(EPILEPSIAE_BLOCK_INVENTORY)
    epi_seizures = load_epilepsiae_seizure_counts(EPILEPSIAE_SEIZURE_INVENTORY)
    with EPILEPSIAE_SOZ_JSON.open() as f:
        epi_soz = json.load(f)
    with YUQUAN_SOZ_JSON.open() as f:
        yuquan_soz = json.load(f)

    rows: List[Dict[str, str]] = []
    t0 = time.time()
    for dataset, subject, sub_dir in subjects:
        t_sub = time.time()
        if dataset == "epilepsiae":
            row = build_epilepsiae_row(
                subject=subject,
                subject_dir=sub_dir,
                block_rows=epi_blocks.get(subject),
                seizure_count=epi_seizures.get(subject, 0),
                soz_map=epi_soz,
            )
        else:
            row = build_yuquan_row(
                subject=subject,
                subject_dir=sub_dir,
                soz_map=yuquan_soz,
            )
        rows.append(row)
        elapsed = time.time() - t_sub
        print(
            f"[audit] {dataset}/{subject}: "
            f"npz={row['n_blocks_actual']} "
            f"sz={row['n_seizures']} "
            f"clin_matched={row['n_clinical_matched']}/{row['n_clinical_total']} "
            f"m2={row['m2_eligible']} "
            f"({elapsed:.1f}s)",
            flush=True,
        )

    print(f"[audit] writing {audit_csv}", flush=True)
    with audit_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[audit] done in {time.time() - t0:.1f}s; {len(rows)} rows", flush=True)

    # Cohort summary print.
    n_total = len(rows)
    n_eligible = sum(int(bool(r["audit_eligible"])) for r in rows)
    n_m2 = sum(int(bool(r["m2_eligible"])) for r in rows)
    print(
        f"[audit] cohort: total={n_total} audit_eligible={n_eligible} "
        f"m2_eligible={n_m2}",
        flush=True,
    )
    return audit_csv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Run Step 0 cohort audit (writes audit.csv).",
    )
    parser.add_argument(
        "--per-subject",
        action="store_true",
        help="Step 3 per-subject runner (NOT IMPLEMENTED in PR-T3-1 Step 0/1).",
    )
    parser.add_argument(
        "--cohort-overlap",
        action="store_true",
        help="Step 4 cohort overlap summary (NOT IMPLEMENTED in PR-T3-1 Step 0/1).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory (default: %(default)s).",
    )
    args = parser.parse_args()

    if args.audit:
        run_audit(args.output_dir)
        return 0
    if args.per_subject or args.cohort_overlap:
        raise NotImplementedError(
            "per-subject / cohort-overlap modes are added in PR-T3-1 Step 3/4"
        )

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
