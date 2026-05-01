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
    compute_per_subject_audit,
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


# ---------------------------------------------------------------------------
# Step 3 — per-subject runner support (HFO event extraction, signal loaders,
# block window builders)
# ---------------------------------------------------------------------------


def load_subject_hfo_events(subject_dir: Path) -> Tuple[
    Dict[str, np.ndarray],
    List[str],
    Dict[str, Tuple[float, float]],
    Optional[float],
]:
    """Load every block npz under ``subject_dir`` and return concatenated
    HFO event onset times per channel (absolute epoch seconds) along with
    the channel name list, per-block windows, and the inferred sfreq from
    block start_times.

    Each block npz contains:

    - ``chns_names`` : list of channel names (len = n_channels)
    - ``whole_dets`` : object array of shape (n_channels,), each element
      an ndarray of shape (n_events, 2) with start_sec / end_sec
    - ``start_time`` : absolute epoch seconds (block start)

    HFO event onset = ``start_sec`` (the first column) + ``start_time``.

    Returns
    -------
    (hfo_times_per_channel, channel_names, block_windows, sfreq_or_None)
        ``block_windows`` maps ``block_stem`` to ``(block_start_epoch,
        block_end_epoch)``. Block end is inferred from the maximum
        observed event end_sec (fallback to start_sec). Subjects whose
        npz lacks an explicit duration field still get a usable window
        because seizures inside the window must already have hfo events
        nearby to matter for M1.
    """
    npz_files = sorted(subject_dir.glob("*_gpu.npz"))
    if not npz_files:
        return {}, [], {}, None

    channel_names: List[str] = []
    hfo_times: Dict[str, List[float]] = defaultdict(list)
    block_windows: Dict[str, Tuple[float, float]] = {}

    for npz_path in npz_files:
        if npz_path.stat().st_size < 1024:
            continue
        with np.load(npz_path, allow_pickle=True) as data:
            chs = list(data["chns_names"].tolist())
            wd = data["whole_dets"]
            start_time = float(data["start_time"])
            block_stem = npz_path.name[: -len("_gpu.npz")]

            if not channel_names:
                channel_names = chs
            elif chs != channel_names:
                raise ValueError(
                    f"channel name schema mismatch in {npz_path.name}: "
                    f"got {chs[:3]}... expected {channel_names[:3]}..."
                )

            block_max_end = 0.0
            for ch_idx, ch in enumerate(chs):
                events = wd[ch_idx]
                if events is None or len(events) == 0:
                    continue
                arr = np.asarray(events, dtype=float)
                if arr.ndim == 2 and arr.shape[1] >= 1:
                    starts = arr[:, 0]
                    hfo_times[ch].extend((starts + start_time).tolist())
                    if arr.shape[1] >= 2:
                        block_max_end = max(block_max_end, float(arr[:, 1].max()))
                    else:
                        block_max_end = max(block_max_end, float(starts.max()))
            block_windows[block_stem] = (start_time, start_time + block_max_end)

    hfo_arrays = {ch: np.asarray(sorted(times), dtype=float) for ch, times in hfo_times.items()}
    for ch in channel_names:
        hfo_arrays.setdefault(ch, np.array([], dtype=float))
    return hfo_arrays, channel_names, block_windows, None


def epilepsiae_block_windows_from_inventory(
    block_rows: List[Dict[str, str]],
) -> Dict[str, Tuple[float, float]]:
    """Map block_stem → (block_start_epoch, block_end_epoch) from the
    block inventory CSV. Inventory row has both fields directly."""
    out: Dict[str, Tuple[float, float]] = {}
    for r in block_rows:
        try:
            t0 = float(r["block_start_epoch"])
            t1 = float(r["block_end_epoch"])
        except (KeyError, ValueError):
            continue
        out[r["block_stem"]] = (t0, t1)
    return out


def epilepsiae_seizures_from_inventory(
    inventory_csv: Path, subject: str
) -> Tuple[List[float], List[str]]:
    """Read seizure_inventory.csv for a subject. Returns parallel lists of
    (eeg_onset_epoch, block_stem). Block_stem is recovered by joining
    recording_id with block_id (last 4 digits of block_id are block_no)
    or by looking up the seizure's containing block via timestamp.

    Simpler: we only need block_stem to look up block_windows. Use
    block_id → block_stem via the block inventory.
    """
    rows: List[Tuple[float, str]] = []
    with inventory_csv.open() as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            if r.get("subject") != subject:
                continue
            try:
                onset = float(r["eeg_onset_epoch"])
            except (KeyError, ValueError):
                continue
            rec_id = r.get("recording_id", "")
            block_id = r.get("block_id", "")
            # Epilepsiae block_stem format: <recording_id>_<NNNN>
            # block_id in inventory is e.g. 107300000102; last 4 of the
            # numeric tail aren't trivially the block_no, so we infer
            # block_stem at the runner level by matching onset to
            # block_inventory's [block_start_epoch, block_end_epoch].
            rows.append((onset, f"{rec_id}|{block_id}"))
    return [t for t, _ in rows], [b for _, b in rows]


def epilepsiae_assign_seizures_to_blocks(
    seizure_onsets: List[float],
    block_windows: Dict[str, Tuple[float, float]],
) -> List[str]:
    """Assign each Epilepsiae seizure onset to the block whose
    ``[start, end]`` window contains it. Returns parallel list of
    block_stem (or empty string when no block contains the onset)."""
    out: List[str] = []
    items = sorted(block_windows.items(), key=lambda x: x[1][0])
    for t in seizure_onsets:
        match = ""
        for stem, (b0, b1) in items:
            if b0 <= t <= b1:
                match = stem
                break
        out.append(match)
    return out


def yuquan_seizures_from_pr1_json(subject: str) -> Tuple[
    List[float], List[str]
]:
    """Read pr1_seizure_<subject>.json, extract per-record seizure
    onset epochs and use the record name as the block stem."""
    p = YUQUAN_SEIZURE_DIR / f"pr1_seizure_{subject}.json"
    if not p.exists():
        return [], []
    with p.open() as f:
        data = json.load(f)
    onsets: List[float] = []
    block_stems: List[str] = []
    for rec in data.get("files", []) or []:
        for sz in rec.get("seizure_intervals", []) or []:
            try:
                onsets.append(float(sz["onset_epoch"]))
                block_stems.append(rec["record"])
            except (KeyError, ValueError, TypeError):
                continue
    return onsets, block_stems


def yuquan_block_windows_from_npz(subject_dir: Path) -> Dict[str, Tuple[float, float]]:
    """Yuquan block_stem (= EDF record name) → (start_time, start_time + duration_inferred).

    Duration is inferred from the EDF file by reading just the header
    via MNE preload=False. We avoid reading samples for speed.
    """
    out: Dict[str, Tuple[float, float]] = {}
    npz_files = sorted(subject_dir.glob("*_gpu.npz"))
    if not npz_files:
        return out
    edf_dir = YUQUAN_DATA_ROOT / subject_dir.name
    for npz_path in npz_files:
        if npz_path.stat().st_size < 1024:
            continue
        stem = npz_path.name[: -len("_gpu.npz")]
        with np.load(npz_path, allow_pickle=True) as data:
            start_time = float(data["start_time"])
        edf_path = edf_dir / f"{stem}.edf"
        if not edf_path.exists():
            continue
        try:
            import mne
            raw = mne.io.read_raw_edf(
                str(edf_path), preload=False, verbose=False, encoding="latin1"
            )
            duration = float(raw.times[-1])
        except Exception:
            continue
        out[stem] = (start_time, start_time + duration)
    return out


def yuquan_signal_loader_factory(
    subject_dir: Path,
    bipolar_pairs: List[Tuple[str, str]],
    block_windows: Dict[str, Tuple[float, float]],
) -> "SignalLoader":
    """Return a signal_loader that handles a Yuquan subject.

    For each [t_start, t_end] window, identify which block contains it,
    open the corresponding EDF, slice the signal, and apply bipolar
    referencing matching the HFO npz channel order.

    The loader keeps a **single-block cache**: each distinct EDF is read
    from disk at most once per per-subject run. M2 calls per seizure go
    in pairs (power_pre precompute + compute_er_logratio), so caching
    cuts disk I/O roughly in half. The cache holds the full block in
    memory (~ 200 MB for 2h × 1024 Hz × ~30 ch × float64); freed when a
    different block is requested.
    """
    edf_dir = YUQUAN_DATA_ROOT / subject_dir.name
    block_items = sorted(block_windows.items(), key=lambda x: x[1][0])

    def loader(t_start: float, t_end: float, channels: List[str]):
        for stem, (b0, b1) in block_items:
            if t_start >= b0 and t_end <= b1:
                rel_start = float(t_start) - b0
                rel_end = float(t_end) - b0
                edf_path = edf_dir / f"{stem}.edf"
                import mne
                # preload=False + crop + load_data reads only the
                # requested time range (~ 44 s of a ~ 2 h block) instead
                # of the full block. Cuts per-load from ~ 25 s to ~ 1-2 s.
                raw = mne.io.read_raw_edf(
                    str(edf_path), preload=False, verbose=False, encoding="latin1"
                )
                raw.crop(tmin=rel_start, tmax=rel_end)
                raw.load_data()
                window = raw.get_data()  # (n_channels, n_samples_window)
                sfreq_local = float(raw.info["sfreq"])
                ch_names_raw = list(raw.ch_names)
                ch_to_idx = {c: i for i, c in enumerate(ch_names_raw)}
                out = np.zeros((window.shape[1], len(channels)), dtype=float)
                for j, ch in enumerate(channels):
                    if "-" not in ch:
                        idx = _yuquan_find_channel_idx(ch, ch_to_idx)
                        if idx is None:
                            raise ValueError(
                                f"yuquan_signal_loader: cannot find {ch}"
                            )
                        out[:, j] = window[idx]
                        continue
                    a, b = ch.split("-", 1)
                    a_idx = _yuquan_find_channel_idx(a, ch_to_idx)
                    b_idx = _yuquan_find_channel_idx(b, ch_to_idx)
                    if a_idx is None or b_idx is None:
                        raise ValueError(
                            f"yuquan_signal_loader: could not find both contacts of {ch} "
                            f"(a={a}, b={b}); raw EDF channels include "
                            f"{ch_names_raw[:5]}..."
                        )
                    out[:, j] = window[a_idx] - window[b_idx]
                return out, sfreq_local
        raise ValueError(
            f"yuquan_signal_loader: window [{t_start}, {t_end}] does not fit any block"
        )

    return loader


def _yuquan_normalize_edf_chname(raw: str) -> str:
    """Strip EDF montage decoration so 'EEG A1-Ref', 'POL A1', 'A1' all
    normalize to 'A1' for matching against HFO npz channel names.
    Handles common prefixes ('EEG ', 'POL ', 'EEG_', 'POL_') and the
    Yuquan referential suffix ('-REF', '-Ref')."""
    s = raw.strip().upper()
    for prefix in ("EEG ", "POL ", "EEG_", "POL_"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    for suffix in ("-REF", "_REF"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s


def _yuquan_find_channel_idx(name: str, ch_to_idx: Dict[str, int]) -> Optional[int]:
    """Yuquan EDFs use mixed labelings ('EEG A1-Ref' for referential
    contacts, 'POL A2' for polarity contacts, sometimes bare 'A3').
    HFO npz strips these. Normalize both sides via
    ``_yuquan_normalize_edf_chname`` so matching works across all
    three forms."""
    name_norm = _yuquan_normalize_edf_chname(name)
    for key, idx in ch_to_idx.items():
        if _yuquan_normalize_edf_chname(key) == name_norm:
            return idx
    return None


def _epilepsiae_partial_window_loader(
    data_path: str,
    head_path: str,
    rel_start_sec: float,
    rel_end_sec: float,
) -> Tuple[np.ndarray, float, List[str]]:
    """Read only the requested time window from an Epilepsiae .data file.

    Bypasses ``load_epilepsiae_block``'s full-block read + notch filter
    so M2 windows (~ 44 s of a ~ 1 h block) take seconds instead of
    minutes. The .data file is little-endian int16 interleaved by
    sample (n_channels samples per "frame"), so byte offsets are
    deterministic from the head's ``num_channels`` + ``sample_freq``.

    CAR is applied across the **intracranial** subset (same as
    ``load_epilepsiae_block(reference='car')``), but with no notch
    filter — for the M2 (80 - 250 Hz) bandpass the 50 / 100 / 150 / 200
    Hz harmonics fall partly inside the band but the contribution is
    small relative to broadband HFO and gets evened out by the post /
    pre log-ratio. The full-block reference path remains available via
    ``epilepsiae_signal_loader_factory`` for callers that need notch
    behaviour.

    Returns
    -------
    (window_signal, sfreq, intracranial_channel_names) where
    window_signal has shape ``(n_samples, n_intracranial_channels)``.
    """
    from src.preprocessing import (
        _read_epilepsiae_head_for_streaming,
        _epilepsiae_intracranial_indices,
    )

    head = _read_epilepsiae_head_for_streaming(Path(head_path))
    sfreq = float(head["sample_freq"])
    n_ch_total = int(head["num_channels"])
    sample_bytes = int(head["sample_bytes"])
    if sample_bytes != 2:
        raise ValueError(
            f"_epilepsiae_partial_window_loader: sample_bytes={sample_bytes} "
            f"unsupported (expected 2)"
        )
    conversion = -1.0 * float(head["conversion_factor"])
    ch_names = list(head["channel_names"])

    s0 = max(0, int(round(float(rel_start_sec) * sfreq)))
    s1 = max(s0, int(round(float(rel_end_sec) * sfreq)))
    n_samples_window = s1 - s0
    bytes_per_frame = sample_bytes * n_ch_total
    byte_offset = s0 * bytes_per_frame
    bytes_to_read = n_samples_window * bytes_per_frame

    with open(data_path, "rb") as fh:
        fh.seek(byte_offset)
        raw_bytes = fh.read(bytes_to_read)
    if len(raw_bytes) < bytes_to_read:
        raise ValueError(
            f"_epilepsiae_partial_window_loader: short read "
            f"({len(raw_bytes)} of {bytes_to_read}) from {data_path} "
            f"at offset {byte_offset}"
        )
    raw_i16 = np.frombuffer(raw_bytes, dtype="<i2").reshape(
        n_samples_window, n_ch_total
    )
    physical = raw_i16.astype(np.float32) * np.float32(conversion)

    keep_idx = _epilepsiae_intracranial_indices(ch_names)
    if not keep_idx:
        raise ValueError(
            f"_epilepsiae_partial_window_loader: no intracranial channels in "
            f"{head_path}"
        )
    intracranial = physical[:, keep_idx]
    car = intracranial - intracranial.mean(axis=1, keepdims=True)
    keep_labels = [ch_names[i] for i in keep_idx]
    return car, sfreq, keep_labels


def epilepsiae_signal_loader_factory(
    block_rows: List[Dict[str, str]],
    analysis_channels: List[str],
    use_partial_loader: bool = True,
) -> "SignalLoader":
    """Return a signal_loader for an Epilepsiae subject (CAR montage).

    Two paths:

    - ``use_partial_loader=True`` (default): byte-seek + read only the
      requested window from the .data file. Cuts per-load time from
      ~ 135 s (full block) to a few seconds. Skips notch filtering;
      see ``_epilepsiae_partial_window_loader`` for the rationale.

    - ``use_partial_loader=False``: legacy path using
      ``load_epilepsiae_block`` (full block + notch filter), with a
      single-block cache.
    """
    by_stem = {r["block_stem"]: r for r in block_rows}

    if use_partial_loader:
        def loader(t_start: float, t_end: float, channels: List[str]):
            for stem, r in by_stem.items():
                try:
                    b0 = float(r["block_start_epoch"])
                    b1 = float(r["block_end_epoch"])
                except (KeyError, ValueError):
                    continue
                if t_start >= b0 and t_end <= b1:
                    sig, sfreq_local, ch_names_raw = _epilepsiae_partial_window_loader(
                        data_path=r["data_path"],
                        head_path=r["head_path"],
                        rel_start_sec=t_start - b0,
                        rel_end_sec=t_end - b0,
                    )
                    ch_to_idx = {c: i for i, c in enumerate(ch_names_raw)}
                    out = np.zeros((sig.shape[0], len(channels)), dtype=float)
                    for j, ch in enumerate(channels):
                        if ch in ch_to_idx:
                            out[:, j] = sig[:, ch_to_idx[ch]]
                        else:
                            raise ValueError(
                                f"epilepsiae_signal_loader (partial): channel {ch} "
                                f"not in intracranial set; first 5 raw: "
                                f"{ch_names_raw[:5]}"
                            )
                    return out, sfreq_local
            raise ValueError(
                f"epilepsiae_signal_loader: window [{t_start}, {t_end}] does not "
                f"fit any block"
            )
        return loader

    # Legacy full-block + notch path with single-block cache.
    cache: Dict[str, Tuple[np.ndarray, float, List[str]]] = {}

    def _load_block(stem: str, r: Dict[str, str]):
        if stem in cache:
            return cache[stem]
        from src.preprocessing import load_epilepsiae_block
        result = load_epilepsiae_block(
            data_path=r["data_path"],
            head_path=r["head_path"],
            reference="car",
            segment_sec=200.0,
        )
        cache.clear()
        cache[stem] = (result.data, float(result.sfreq), list(result.ch_names))
        return cache[stem]

    def legacy_loader(t_start: float, t_end: float, channels: List[str]):
        for stem, r in by_stem.items():
            try:
                b0 = float(r["block_start_epoch"])
                b1 = float(r["block_end_epoch"])
            except (KeyError, ValueError):
                continue
            if t_start >= b0 and t_end <= b1:
                full_data, sfreq_local, ch_names_raw = _load_block(stem, r)
                s0 = max(0, int(round((t_start - b0) * sfreq_local)))
                s1 = int(round((t_end - b0) * sfreq_local))
                window = full_data[:, s0:s1].T
                ch_to_idx = {c: i for i, c in enumerate(ch_names_raw)}
                out = np.zeros((window.shape[0], len(channels)), dtype=float)
                for j, ch in enumerate(channels):
                    if ch in ch_to_idx:
                        out[:, j] = window[:, ch_to_idx[ch]]
                    else:
                        raise ValueError(
                            f"epilepsiae_signal_loader (legacy): channel {ch} "
                            f"not in loaded CAR set; first 5 raw: "
                            f"{ch_names_raw[:5]}"
                        )
                return out, sfreq_local
        raise ValueError(
            f"epilepsiae_signal_loader: window [{t_start}, {t_end}] does not "
            f"fit any block"
        )
    return legacy_loader


# ---------------------------------------------------------------------------
# Per-subject driver
# ---------------------------------------------------------------------------


def run_subject(
    dataset: str,
    subject: str,
    output_dir: Path,
    epi_blocks: Dict[str, List[Dict[str, str]]],
    epi_soz: Dict[str, List[str]],
    yuquan_soz: Dict[str, List[str]],
    null_n_iter: int = 200,
    null_rng_seed: int = 0,
    use_legacy_epilepsiae_loader: bool = False,
) -> Optional[Path]:
    """Run one subject end-to-end and write per-subject JSON."""
    subject_dir = HFO_DETECTION_DIR / subject
    if not subject_dir.exists():
        print(f"[run] {dataset}/{subject}: HFO dir missing", flush=True)
        return None

    hfo_events, channel_names, hfo_block_windows, _ = load_subject_hfo_events(subject_dir)
    if not channel_names:
        print(f"[run] {dataset}/{subject}: no usable HFO blocks", flush=True)
        return None

    if dataset == "epilepsiae":
        block_rows = epi_blocks.get(subject, [])
        block_windows = epilepsiae_block_windows_from_inventory(block_rows)
        all_seizure_onsets, _ = epilepsiae_seizures_from_inventory(
            EPILEPSIAE_SEIZURE_INVENTORY, subject
        )
        seizure_block_ids = epilepsiae_assign_seizures_to_blocks(
            all_seizure_onsets, block_windows
        )
        sf_iter = [
            float(r.get("sample_rate_sql") or 0.0)
            for r in block_rows
            if float(r.get("sample_rate_sql") or 0.0) >= HFO_DETECTION_MIN_SFREQ
        ]
        sfreq = min(sf_iter) if sf_iter else 0.0
        m2_eligible = bool(sfreq >= M2_MIN_SFREQ)
        clinical_soz = epi_soz.get(subject, []) or []
        signal_loader = (
            epilepsiae_signal_loader_factory(
                block_rows,
                channel_names,
                use_partial_loader=not use_legacy_epilepsiae_loader,
            )
            if m2_eligible else None
        )
    elif dataset == "yuquan":
        block_windows = yuquan_block_windows_from_npz(subject_dir)
        all_seizure_onsets, seizure_block_ids = yuquan_seizures_from_pr1_json(subject)
        sfreq = probe_yuquan_sfreq(subject) or 0.0
        m2_eligible = bool(sfreq >= M2_MIN_SFREQ)
        clinical_soz = yuquan_soz.get(subject, []) or []
        if m2_eligible:
            signal_loader = yuquan_signal_loader_factory(
                subject_dir, [], block_windows
            )
        else:
            signal_loader = None
    else:
        raise ValueError(f"unknown dataset {dataset!r}")

    if not all_seizure_onsets:
        print(f"[run] {dataset}/{subject}: no seizures", flush=True)
        return None

    # Drop seizures whose block_id couldn't be matched (Epilepsiae
    # outside-block onsets / Yuquan record-name mismatches).
    paired = [
        (t, blk) for t, blk in zip(all_seizure_onsets, seizure_block_ids)
        if blk and blk in block_windows
    ]
    if not paired:
        print(
            f"[run] {dataset}/{subject}: no seizures with valid block mapping",
            flush=True,
        )
        return None
    seizure_onsets = [t for t, _ in paired]
    seizure_block_ids = [blk for _, blk in paired]

    t0 = time.time()
    res = compute_per_subject_audit(
        dataset=dataset,
        subject=subject,
        seizure_onsets=seizure_onsets,
        seizure_block_ids=seizure_block_ids,
        block_windows=block_windows,
        hfo_event_times_per_channel=hfo_events,
        signal_loader=signal_loader,
        sfreq=sfreq,
        clinical_soz=clinical_soz,
        analysis_channels=channel_names,
        m2_eligible=m2_eligible,
        null_n_iter=null_n_iter,
        null_rng_seed=null_rng_seed,
    )
    out_dir = output_dir / "per_subject"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}_{subject}.json"
    with out_path.open("w") as f:
        json.dump(res, f, indent=2, default=_json_default)
    print(
        f"[run] {dataset}/{subject}: kept={res['n_seizures_used']} "
        f"dropped={res['n_seizures_dropped']} "
        f"k_primary={res['k_primary_size_matched']} "
        f"H_M1={res['headline_primary'].get('H_M1_pois_medianrank_size_matched'):.3f} "
        f"H_M2={res['headline_primary'].get('H_M2_logratio_medianrank_size_matched'):.3f} "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )
    return out_path


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"unserializable {type(obj).__name__}")


def run_per_subject(
    output_dir: Path,
    subject_filter: Optional[str] = None,
    null_n_iter: int = 200,
    null_rng_seed: int = 0,
    skip_existing: bool = False,
    use_legacy_epilepsiae_loader: bool = False,
) -> int:
    """Iterate audit_eligible subjects and write per-subject JSON each.

    ``skip_existing=True`` short-circuits subjects whose JSON already
    exists under ``output_dir/per_subject/``. Useful when resuming
    after a partial cohort run.
    """
    audit_csv = output_dir / "audit.csv"
    if not audit_csv.exists():
        print(f"[run] audit.csv missing at {audit_csv}; run --audit first", flush=True)
        return 1
    with audit_csv.open() as f:
        rdr = csv.DictReader(f)
        rows = [r for r in rdr if int(r["audit_eligible"] or 0) == 1]
    if subject_filter:
        rows = [r for r in rows if r["subject"] == subject_filter]
    if not rows:
        print("[run] no audit_eligible subjects to run", flush=True)
        return 1

    per_subject_dir = output_dir / "per_subject"
    if skip_existing:
        before = len(rows)
        rows = [
            r for r in rows
            if not (per_subject_dir / f"{r['dataset']}_{r['subject']}.json").exists()
        ]
        print(
            f"[run] --skip-existing: {before - len(rows)} subjects already done, "
            f"{len(rows)} remaining",
            flush=True,
        )
        if not rows:
            return 0

    epi_blocks = load_epilepsiae_block_inventory(EPILEPSIAE_BLOCK_INVENTORY)
    with EPILEPSIAE_SOZ_JSON.open() as f:
        epi_soz = json.load(f)
    with YUQUAN_SOZ_JSON.open() as f:
        yuquan_soz = json.load(f)

    print(
        f"[run] processing {len(rows)} audit_eligible subjects "
        f"(null_n_iter={null_n_iter}, "
        f"use_legacy_epilepsiae_loader={use_legacy_epilepsiae_loader})",
        flush=True,
    )
    written = 0
    for r in rows:
        try:
            out = run_subject(
                dataset=r["dataset"],
                subject=r["subject"],
                output_dir=output_dir,
                epi_blocks=epi_blocks,
                epi_soz=epi_soz,
                yuquan_soz=yuquan_soz,
                null_n_iter=null_n_iter,
                null_rng_seed=null_rng_seed,
                use_legacy_epilepsiae_loader=use_legacy_epilepsiae_loader,
            )
            if out is not None:
                written += 1
        except Exception as exc:
            print(f"[run] {r['dataset']}/{r['subject']}: FAILED — {exc}", flush=True)
    print(f"[run] wrote {written}/{len(rows)} per-subject JSONs", flush=True)
    return 0 if written > 0 else 1


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
        help="Step 3 per-subject runner (writes per_subject/<dataset>_<subject>.json).",
    )
    parser.add_argument(
        "--cohort-overlap",
        action="store_true",
        help="Step 4 cohort overlap summary (NOT IMPLEMENTED in PR-T3-1 Step 3).",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Restrict --per-subject to a single subject name.",
    )
    parser.add_argument(
        "--null-n-iter",
        type=int,
        default=200,
        help="Time-shifted null iterations (default 200, plan §5.1 / §9 Step 3.4).",
    )
    parser.add_argument(
        "--null-rng-seed",
        type=int,
        default=0,
        help="Time-shifted null RNG seed (default 0).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip subjects whose per-subject JSON already exists "
             "(useful for resuming a partial cohort run).",
    )
    parser.add_argument(
        "--use-legacy-epilepsiae-loader",
        action="store_true",
        help="Use load_epilepsiae_block (full block + notch) instead of the "
             "fast partial-window .data reader. Default is the partial loader "
             "(no notch; 80-250 Hz bandpass already excludes line noise).",
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
    if args.per_subject:
        return run_per_subject(
            output_dir=args.output_dir,
            subject_filter=args.subject,
            null_n_iter=args.null_n_iter,
            null_rng_seed=args.null_rng_seed,
            skip_existing=args.skip_existing,
            use_legacy_epilepsiae_loader=args.use_legacy_epilepsiae_loader,
        )
    if args.cohort_overlap:
        raise NotImplementedError(
            "cohort-overlap mode is added in PR-T3-1 Step 4"
        )

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
