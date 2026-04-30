"""PR-T3-1 — Data-driven ictal-onset SOZ audit CLI.

Usage:

    python scripts/run_data_driven_soz.py --audit
    python scripts/run_data_driven_soz.py --per-subject  # Step 3 (not yet)
    python scripts/run_data_driven_soz.py --cohort-overlap  # Step 4 (not yet)

Step 0 (``--audit``) enumerates the **expected** cohort
(``epilepsiae_subject_inventory.csv`` + ``/mnt/yuquan_data/yuquan_24h_edf/``
listing) and left-joins the HFO detection artifacts under
``results/hfo_detection/<subject>/``. Subjects whose HFO output is
absent show up as rows with ``hfo_npz_ok=0`` instead of being silently
skipped — this is what the audit is for.

Per-subject the script:

- Loads every block npz, validates required keys, checks ``chns_names``
  schema consistency across blocks (plan §3.2 hard requirement).
- Reports sfreq from retained npz blocks (cross-ref ``block_inventory``
  for Epilepsiae, EDF probe for Yuquan).
- Annotates the analysis channel set against the clinical SOZ list
  using the canonical 3-state matcher
  (``src.data_driven_soz.annotate_clinical_soz``) and reports unmatched
  clinical contacts via ``matched_clinical_contacts``.

Output: ``results/spatial_modulation/data_driven_soz/audit.csv``.

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
    check_channel_schema_consistency,
    matched_clinical_contacts,
)


HFO_DETECTION_DIR = ROOT / "results" / "hfo_detection"
EPILEPSIAE_BLOCK_INVENTORY = ROOT / "results" / "epilepsiae_block_inventory.csv"
EPILEPSIAE_SUBJECT_INVENTORY = ROOT / "results" / "epilepsiae_subject_inventory.csv"
EPILEPSIAE_SEIZURE_INVENTORY = ROOT / "results" / "epilepsiae_seizure_inventory.csv"
EPILEPSIAE_SOZ_JSON = ROOT / "results" / "epilepsiae_soz_core_channels.json"
YUQUAN_SOZ_JSON = ROOT / "results" / "yuquan_soz_core_channels.json"
YUQUAN_SEIZURE_DIR = ROOT / "results" / "seizure_detection"
YUQUAN_DATA_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
OUTPUT_DIR = ROOT / "results" / "spatial_modulation" / "data_driven_soz"

NYQUIST_BAND_HIGH = 250.0           # M2 primary band upper edge
NYQUIST_SAFETY = 1.05               # plan §3.4: 5% safety margin
M2_MIN_SFREQ = NYQUIST_BAND_HIGH * 2 * NYQUIST_SAFETY   # = 525 Hz
HFO_DETECTION_MIN_SFREQ = 500       # scripts/run_hfo_detection.py threshold

REQUIRED_NPZ_KEYS = ("whole_dets", "chns_names", "events_count", "start_time")


# ---------------------------------------------------------------------------
# Cohort enumeration (expected sources, not "what we already have")
# ---------------------------------------------------------------------------


def expected_epilepsiae_subjects(path: Path) -> List[str]:
    if not path.exists():
        return []
    with path.open() as f:
        rdr = csv.DictReader(f)
        return sorted({row["subject"] for row in rdr if row.get("subject")})


def expected_yuquan_subjects(data_root: Path) -> List[str]:
    if not data_root.exists():
        return []
    return sorted(d.name for d in data_root.iterdir() if d.is_dir())


# ---------------------------------------------------------------------------
# Epilepsiae helpers
# ---------------------------------------------------------------------------


def load_epilepsiae_block_inventory(path: Path) -> Dict[str, List[Dict[str, str]]]:
    """Group block_inventory rows by ``subject``."""
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
    """Return raw EDF sampling frequency for one EDF in subject dir."""
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
# HFO npz inspection — full per-subject block validation
# ---------------------------------------------------------------------------


def list_npz_blocks(subject_dir: Path) -> List[Path]:
    if not subject_dir.exists():
        return []
    return sorted(subject_dir.glob("*_gpu.npz"))


def npz_block_stem(npz_path: Path) -> str:
    return npz_path.name[: -len("_gpu.npz")]


def _scan_subject_blocks(subject_dir: Path) -> Dict[str, object]:
    """Load every npz under ``subject_dir`` and report per-block status.

    Returns a dict with:

    - ``n_blocks_actual``: int — files that match ``*_gpu.npz``
    - ``n_blocks_stub``: int — files smaller than 1 KB
    - ``n_blocks_loadable``: int — files that opened with all required keys
    - ``unloadable_blocks``: list[str]
    - ``missing_key_blocks``: list[str]
    - ``schema_consistency``: dict from ``check_channel_schema_consistency``
    - ``channel_lists``: list[list[str]] — schema across loadable blocks
    """
    npz_files = list_npz_blocks(subject_dir)
    n_actual = len(npz_files)
    n_stub = sum(1 for p in npz_files if p.stat().st_size < 1024)

    channel_lists: List[List[str]] = []
    unloadable: List[str] = []
    missing_keys: List[str] = []

    for npz_path in npz_files:
        if npz_path.stat().st_size < 1024:
            unloadable.append(npz_block_stem(npz_path))
            continue
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                missing = [k for k in REQUIRED_NPZ_KEYS if k not in data.files]
                if missing:
                    missing_keys.append(npz_block_stem(npz_path))
                    continue
                channel_lists.append(list(data["chns_names"].tolist()))
        except Exception:
            unloadable.append(npz_block_stem(npz_path))

    schema = check_channel_schema_consistency(channel_lists)
    return {
        "n_blocks_actual": n_actual,
        "n_blocks_stub": n_stub,
        "n_blocks_loadable": len(channel_lists),
        "unloadable_blocks": unloadable,
        "missing_key_blocks": missing_keys,
        "schema_consistency": schema,
        "channel_lists": channel_lists,
    }


# ---------------------------------------------------------------------------
# Audit row builder
# ---------------------------------------------------------------------------


AUDIT_COLUMNS = [
    "dataset",
    "subject",
    "n_blocks_actual",
    "n_blocks_expected",
    "n_blocks_stub",
    "n_blocks_loadable",
    "schema_consistent",
    "schema_n_channels_min",
    "schema_n_channels_max",
    "hfo_npz_ok",
    "block_count_match",
    "n_seizures",
    "sfreq_min",
    "sfreq_max",
    "sfreq_source",
    "m2_eligible",
    "n_clinical_total",
    "n_clinical_matched",
    "n_clinical_unmatched",
    "n_unknown_channels",
    "unmatched_clinical_names",
    "soz_eligible",
    "audit_eligible",
    "notes",
]


def _empty_row(dataset: str, subject: str) -> Dict[str, str]:
    """Initial values for a subject row; fields filled in by builders."""
    return {
        "dataset": dataset,
        "subject": subject,
        "n_blocks_actual": 0,
        "n_blocks_expected": "",
        "n_blocks_stub": 0,
        "n_blocks_loadable": 0,
        "schema_consistent": "",
        "schema_n_channels_min": 0,
        "schema_n_channels_max": 0,
        "hfo_npz_ok": 0,
        "block_count_match": "",
        "n_seizures": 0,
        "sfreq_min": "",
        "sfreq_max": "",
        "sfreq_source": "missing",
        "m2_eligible": 0,
        "n_clinical_total": 0,
        "n_clinical_matched": 0,
        "n_clinical_unmatched": 0,
        "n_unknown_channels": 0,
        "unmatched_clinical_names": "",
        "soz_eligible": 0,
        "audit_eligible": 0,
        "notes": "",
    }


def _annotate_clinical(
    channel_lists: List[List[str]],
    clinical_soz: List[str],
) -> Tuple[Dict[str, str], int, int, int, List[str]]:
    """Annotate analysis channels (using block 0's schema, the schema_consistency
    check guarantees other blocks agree) and compute matched / unmatched stats
    via the canonical helpers.

    Returns ``(annotation, n_matched_channels, n_unknown_channels,
    n_clinical_unmatched, unmatched_names)``.
    """
    if not channel_lists:
        return {}, 0, 0, 0, []
    analysis_channels = channel_lists[0]
    ann = annotate_clinical_soz(analysis_channels, clinical_soz)
    matched_set = matched_clinical_contacts(analysis_channels, clinical_soz)
    from src.event_periodicity import _normalize_channel_name  # canonical

    norm_clinical = {_normalize_channel_name(s) for s in clinical_soz if s}
    unmatched = sorted(norm_clinical - matched_set)
    n_matched_channels = sum(1 for v in ann.values() if v == SOZ_LABEL)
    n_unknown_channels = sum(1 for v in ann.values() if v == UNKNOWN_LABEL)
    return ann, n_matched_channels, n_unknown_channels, len(unmatched), unmatched


def _finalize_eligibility(
    row: Dict[str, str],
    notes: List[str],
) -> None:
    """Common eligibility logic: write soz_eligible / audit_eligible /
    notes column in place. Plan §6.1 conditions:

    - hfo_npz_ok: at least one valid (>1KB) npz block
    - all_blocks_loadable: every npz parsed and had required keys
    - schema_consistent: chns_names identical across blocks
    - n_seizures >= 2 (per-seizure consistency well defined)
    - n_clinical_matched >= 1 (audit baseline)

    For ``hfo_npz_ok`` to be True we additionally require
    ``n_blocks_loadable == n_blocks_actual`` AND ``schema_consistent``.
    """
    n_actual = int(row["n_blocks_actual"])
    n_loadable = int(row["n_blocks_loadable"])
    n_stub = int(row["n_blocks_stub"])

    schema_ok_str = row.get("schema_consistent")
    schema_ok = schema_ok_str == 1 or schema_ok_str is True
    hfo_npz_ok = (
        n_actual > 0
        and n_stub == 0
        and n_loadable == n_actual
        and schema_ok
    )
    row["hfo_npz_ok"] = int(hfo_npz_ok)

    soz_eligible = int(row["n_clinical_matched"]) >= 1
    n_seizures = int(row["n_seizures"])
    audit_eligible = (
        hfo_npz_ok
        and n_seizures >= 2
        and soz_eligible
    )
    row["soz_eligible"] = int(soz_eligible)
    row["audit_eligible"] = int(audit_eligible)

    if n_actual == 0:
        notes.insert(0, "no hfo_detection dir / npz")
    else:
        if n_stub > 0:
            notes.append(f"stub_blocks={n_stub}")
        if n_loadable < n_actual:
            notes.append(f"unloadable_blocks={n_actual - n_loadable}")
        if not schema_ok:
            notes.append("chns_names schema inconsistent across blocks")
    if n_seizures < 2 and n_actual > 0:
        notes.append(f"n_seizures={n_seizures} < 2")
    if not soz_eligible and n_actual > 0:
        notes.append(f"n_clinical_matched={int(row['n_clinical_matched'])} < 1")
    row["notes"] = "; ".join(notes)


def build_epilepsiae_row(
    subject: str,
    block_rows: Optional[List[Dict[str, str]]],
    seizure_count: int,
    soz_map: Dict[str, List[str]],
) -> Dict[str, str]:
    notes: List[str] = []
    row = _empty_row("epilepsiae", subject)
    subject_dir = HFO_DETECTION_DIR / subject

    scan = _scan_subject_blocks(subject_dir)
    row["n_blocks_actual"] = scan["n_blocks_actual"]
    row["n_blocks_stub"] = scan["n_blocks_stub"]
    row["n_blocks_loadable"] = scan["n_blocks_loadable"]
    schema = scan["schema_consistency"]
    row["schema_consistent"] = int(bool(schema["all_consistent"]))
    row["schema_n_channels_min"] = schema["n_channels_min"]
    row["schema_n_channels_max"] = schema["n_channels_max"]

    # Cross-reference inventory: only blocks where sample_rate_sql >=
    # HFO_DETECTION_MIN_SFREQ (=500) are expected to have an npz.
    expected_blocks: List[str] = []
    inv_sfreq_by_stem: Dict[str, float] = {}
    if block_rows:
        for r in block_rows:
            try:
                sf = float(r.get("sample_rate_sql") or 0.0)
            except Exception:
                sf = 0.0
            inv_sfreq_by_stem[r["block_stem"]] = sf
            if sf >= HFO_DETECTION_MIN_SFREQ:
                expected_blocks.append(r["block_stem"])
    n_expected = len(expected_blocks) if block_rows is not None else -1
    row["n_blocks_expected"] = n_expected if n_expected >= 0 else ""

    # Sfreqs of *retained* npz blocks.
    retained_sfreqs: List[float] = []
    for npz in list_npz_blocks(subject_dir):
        sf = inv_sfreq_by_stem.get(npz_block_stem(npz), 0.0)
        if sf > 0:
            retained_sfreqs.append(sf)
    if retained_sfreqs:
        row["sfreq_min"] = min(retained_sfreqs)
        row["sfreq_max"] = max(retained_sfreqs)
        row["sfreq_source"] = "block_inventory.sample_rate_sql"
        row["m2_eligible"] = int(min(retained_sfreqs) >= M2_MIN_SFREQ)
    else:
        row["sfreq_source"] = "missing" if scan["n_blocks_actual"] == 0 else "no_inventory_match"
        if scan["n_blocks_actual"] > 0:
            notes.append("no inventory sfreq for retained blocks")

    if n_expected >= 0:
        match = (n_expected == scan["n_blocks_actual"])
        row["block_count_match"] = int(match)
        if not match:
            notes.append(
                f"block_count mismatch actual={scan['n_blocks_actual']} expected={n_expected}"
            )

    row["n_seizures"] = seizure_count

    clinical = soz_map.get(subject, []) or []
    row["n_clinical_total"] = len(clinical)
    if scan["channel_lists"]:
        _, n_matched_ch, n_unknown, n_unmatched, unmatched = _annotate_clinical(
            scan["channel_lists"], clinical
        )
        row["n_clinical_matched"] = n_matched_ch
        row["n_unknown_channels"] = n_unknown
        row["n_clinical_unmatched"] = n_unmatched
        row["unmatched_clinical_names"] = "|".join(unmatched)

    _finalize_eligibility(row, notes)
    return row


def build_yuquan_row(
    subject: str,
    soz_map: Dict[str, List[str]],
) -> Dict[str, str]:
    notes: List[str] = []
    row = _empty_row("yuquan", subject)
    subject_dir = HFO_DETECTION_DIR / subject

    scan = _scan_subject_blocks(subject_dir)
    row["n_blocks_actual"] = scan["n_blocks_actual"]
    row["n_blocks_stub"] = scan["n_blocks_stub"]
    row["n_blocks_loadable"] = scan["n_blocks_loadable"]
    schema = scan["schema_consistency"]
    row["schema_consistent"] = int(bool(schema["all_consistent"]))
    row["schema_n_channels_min"] = schema["n_channels_min"]
    row["schema_n_channels_max"] = schema["n_channels_max"]

    seizure_count = load_yuquan_seizure_count(subject)
    if seizure_count is None:
        seizure_count = 0
        notes.append("missing pr1_seizure JSON")
    row["n_seizures"] = seizure_count

    sf = probe_yuquan_sfreq(subject)
    if sf is not None:
        row["sfreq_min"] = sf
        row["sfreq_max"] = sf
        row["sfreq_source"] = "edf_probe_first_record"
        row["m2_eligible"] = int(sf >= M2_MIN_SFREQ)
    else:
        notes.append("could not probe EDF sfreq")

    clinical = soz_map.get(subject, []) or []
    row["n_clinical_total"] = len(clinical)
    if scan["channel_lists"]:
        _, n_matched_ch, n_unknown, n_unmatched, unmatched = _annotate_clinical(
            scan["channel_lists"], clinical
        )
        row["n_clinical_matched"] = n_matched_ch
        row["n_unknown_channels"] = n_unknown
        row["n_clinical_unmatched"] = n_unmatched
        row["unmatched_clinical_names"] = "|".join(unmatched)

    _finalize_eligibility(row, notes)
    return row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_audit(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_csv = output_dir / "audit.csv"

    epi_subjects = expected_epilepsiae_subjects(EPILEPSIAE_SUBJECT_INVENTORY)
    yu_subjects = expected_yuquan_subjects(YUQUAN_DATA_ROOT)
    print(
        f"[audit] expected cohort: {len(epi_subjects)} epilepsiae "
        f"+ {len(yu_subjects)} yuquan = {len(epi_subjects) + len(yu_subjects)}",
        flush=True,
    )

    print("[audit] loading inventories ...", flush=True)
    epi_blocks = load_epilepsiae_block_inventory(EPILEPSIAE_BLOCK_INVENTORY)
    epi_seizures = load_epilepsiae_seizure_counts(EPILEPSIAE_SEIZURE_INVENTORY)
    with EPILEPSIAE_SOZ_JSON.open() as f:
        epi_soz = json.load(f)
    with YUQUAN_SOZ_JSON.open() as f:
        yuquan_soz = json.load(f)

    rows: List[Dict[str, str]] = []
    t0 = time.time()
    for subject in epi_subjects:
        t_sub = time.time()
        row = build_epilepsiae_row(
            subject=subject,
            block_rows=epi_blocks.get(subject),
            seizure_count=epi_seizures.get(subject, 0),
            soz_map=epi_soz,
        )
        rows.append(row)
        print(
            f"[audit] epilepsiae/{subject}: "
            f"npz={row['n_blocks_actual']}/{row['n_blocks_expected']} "
            f"loadable={row['n_blocks_loadable']} "
            f"schema_ok={row['schema_consistent']} "
            f"sz={row['n_seizures']} "
            f"clin_matched={row['n_clinical_matched']}/{row['n_clinical_total']} "
            f"m2={row['m2_eligible']} ({time.time() - t_sub:.1f}s)",
            flush=True,
        )

    for subject in yu_subjects:
        t_sub = time.time()
        row = build_yuquan_row(subject=subject, soz_map=yuquan_soz)
        rows.append(row)
        print(
            f"[audit] yuquan/{subject}: "
            f"npz={row['n_blocks_actual']} "
            f"loadable={row['n_blocks_loadable']} "
            f"schema_ok={row['schema_consistent']} "
            f"sz={row['n_seizures']} "
            f"clin_matched={row['n_clinical_matched']}/{row['n_clinical_total']} "
            f"m2={row['m2_eligible']} ({time.time() - t_sub:.1f}s)",
            flush=True,
        )

    print(f"[audit] writing {audit_csv}", flush=True)
    with audit_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[audit] done in {time.time() - t0:.1f}s; {len(rows)} rows", flush=True)

    n_total = len(rows)
    n_eligible = sum(int(r["audit_eligible"] or 0) for r in rows)
    n_m2 = sum(int(r["m2_eligible"] or 0) for r in rows)
    n_no_hfo = sum(1 for r in rows if int(r["n_blocks_actual"] or 0) == 0)
    print(
        f"[audit] cohort: total={n_total} audit_eligible={n_eligible} "
        f"m2_eligible={n_m2} hfo_missing={n_no_hfo}",
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
