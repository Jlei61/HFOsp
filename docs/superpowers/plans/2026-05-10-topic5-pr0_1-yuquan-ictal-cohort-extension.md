# Topic 5 PR-0.1: Yuquan v2.3 Ictal Atlas Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend topic5 z-ER ictal cohort from 16 (epilepsiae-only) to 25 by adding 9 yuquan audit_eligible subjects: gaolan, huanghanwen, litengsheng, pengzihang, sunyuanxin, xuxinyi, zhangjinhan, zhangkexuan, zhaojinrui.

**Architecture:** Add a yuquan-side seizure-window loader that mirrors the existing epilepsiae path (`extract_seizure_window` in `src/ictal_onset_extraction.py:238`). Build a yuquan_block_inventory.csv from EDF probes, augment yuquan_seizure_inventory.csv with `record_start/end_epoch`, and route `extract_seizure_window` based on `dataset` prefix. Cohort selector (`scripts/run_ictal_er_rank.py`) then expands `SUPPORTED_DATASETS` to include yuquan. No change to v2.3 schema or z-ER subtyping logic — both are dataset-agnostic.

**Tech Stack:** Python 3.11, mne (EDF I/O with `encoding="latin1"`), numpy, scipy, pytest. All deps already in repo.

**Cohort budget after extension:**
- 16 (current: 15 epilepsiae audit_eligible + 1 sentinel-only 916)
- +9 yuquan audit_eligible (gaolan/4 sz, huanghanwen/2, litengsheng/8, pengzihang/8, sunyuanxin/8, xuxinyi/3, zhangjinhan/8, zhangkexuan/6, zhaojinrui/5)
- = **25 total**, 24 + sentinel
- **Not gated by topic1 stable_k=2**: zhangjinhan (stable_k=6) and zhaojinrui (stable_k=5) are kept because they are the most within-subject heterogeneous candidates — most relevant for z-ER subtyping.

---

## File Structure

**New files (to create):**

| Path | Responsibility |
|---|---|
| `scripts/build_yuquan_block_inventory.py` | One-shot script: probe `/mnt/yuquan_data/yuquan_24h_edf/<sid>/*.edf`, write `results/dataset_inventory/yuquan_block_inventory.csv` and augmented `results/dataset_inventory/yuquan_seizure_inventory.csv` |
| `src/yuquan_dataset.py` | `load_yuquan_record(edf_path, *, reference, segment_sec)` returning a `PreprocessingResult`-shaped object compatible with `extract_seizure_window` consumers |
| `tests/test_yuquan_dataset.py` | Unit tests for `load_yuquan_record` (channel filter, CAR, bipolar) — uses synthetic EDFs only, no real data |
| `tests/test_build_yuquan_block_inventory.py` | Unit tests for the inventory builder (mocked EDF metadata) |
| `tests/test_ictal_onset_extraction_yuquan.py` | Integration tests for `extract_seizure_window` on yuquan path with synthetic inventory rows |

**Modified files:**

| Path | What changes | Why |
|---|---|---|
| `src/ictal_onset_extraction.py:271-274` | Replace `NotImplementedError` with dual-dataset routing | Unblock yuquan path |
| `src/ictal_onset_extraction.py:_resolve_inventory_paths` (line 214) | Accept `dataset` arg, return `(seizure_csv, block_csv)` for either dataset | Avoid hard-coded `epilepsiae_*.csv` lookups |
| `scripts/run_ictal_er_rank.py:112` | `SUPPORTED_DATASETS = frozenset({"epilepsiae", "yuquan"})` | Stop excluding yuquan in cohort selector |
| `scripts/run_ictal_er_rank.py:_focus_rel_path` (line 122) + `_focal_channels` (line 135) | Branch on dataset to load `epilepsiae_electrode_focus_rel.json` vs `yuquan_soz_core_channels.json` | Yuquan SOZ JSON is a flat `subject -> [channels]` dict; epilepsiae has i/l/e tiers |
| `scripts/run_ictal_er_rank.py:_count_seizures` (line 140) | Branch on dataset to use yuquan_seizure_inventory.csv | Currently only reads epilepsiae path |
| `docs/topic5_seizure_subtyping.md` | §2 cohort number 16→25; §4 caveat 5 removed | Reflect extension |
| `docs/archive/topic5/pr1_seizure_clustering/pr1_zer_cohort_2026-05-10.md` | Add §6 cohort extension subsection | Record provenance |

**Out-of-scope (NOT in this plan):**
- Topic1 / topic2 / topic3 cohorts (separate inputs, not affected)
- Audit ineligible yuquan (n_seizures<2) — cannot be added regardless
- v2.3 atlas script changes — atlas is dataset-agnostic, just needs new per_subject JSONs as input

---

## Sentinel canary subjects

Two yuquan subjects are designated canaries before the full cohort run:

- **gaolan** (n_sz=4, stable_k=2, n_channels=130): typical case; 4 seizures across 2 EDFs (FA0013KQ, FA0013KS, FA0013KU, FA0013KW)
- **huanghanwen** (n_sz=2, stable_k=2, n_channels=85): minimal-seizure case; 2 seizures, low channel count

If either canary fails baseline-validity, channel filtering, or produces all-NaN z-ER, STOP and audit before scaling to remaining 7 yuquan.

---

## Task 1: Build yuquan_block_inventory.csv probe script

**Files:**
- Create: `scripts/build_yuquan_block_inventory.py`
- Create: `tests/test_build_yuquan_block_inventory.py`

**Schema for `results/dataset_inventory/yuquan_block_inventory.csv`** (must match consumer expectations in Task 3):

| Column | Source | Type |
|---|---|---|
| `subject` | dirname under `/mnt/yuquan_data/yuquan_24h_edf/` | str |
| `recording_id` | EDF filename stem (e.g., `FA0013KQ`) | str |
| `block_id` | same as `recording_id` (yuquan = 1 block per EDF) | str |
| `block_stem` | same as `recording_id` | str |
| `block_start_epoch` | `read_edf_start_time(edf_path)` (existing helper at `src/preprocessing.py:212`) | float (UTC seconds) |
| `block_end_epoch` | `block_start_epoch + duration_sec` | float |
| `duration_sec` | from EDF header (`n_data_records * record_duration`) | float |
| `sample_rate` | from EDF header | float |
| `n_channels_total` | from EDF header | int |
| `head_path` | empty string (yuquan has no separate head; use edf_path) | str |
| `data_path` | absolute path to `.edf` file | str |
| `edf_path` | same as `data_path` | str |

- [ ] **Step 1: Create test file with mocked EDF probe**

```python
# tests/test_build_yuquan_block_inventory.py
"""Tests for scripts/build_yuquan_block_inventory.py.

Uses monkeypatched probes — no real EDFs touched.
"""
from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.build_yuquan_block_inventory import (
    BlockProbeResult,
    probe_one_edf,
    write_block_inventory_csv,
)


def test_probe_one_edf_returns_expected_fields(tmp_path: Path):
    fake_edf = tmp_path / "FAKE001.edf"
    fake_edf.write_bytes(b"")  # presence only; mocked probe never reads

    with patch("scripts.build_yuquan_block_inventory.read_edf_start_time", return_value=1605835000.0), \
         patch("scripts.build_yuquan_block_inventory._probe_edf_metadata", return_value={
             "duration_sec": 7200.0,
             "sample_rate": 2000.0,
             "n_channels_total": 147,
         }):
        result = probe_one_edf("gaolan", fake_edf)

    assert result.subject == "gaolan"
    assert result.recording_id == "FAKE001"
    assert result.block_id == "FAKE001"
    assert result.block_stem == "FAKE001"
    assert result.block_start_epoch == 1605835000.0
    assert result.block_end_epoch == 1605835000.0 + 7200.0
    assert result.duration_sec == 7200.0
    assert result.sample_rate == 2000.0
    assert result.n_channels_total == 147
    assert result.edf_path == str(fake_edf)
    assert result.data_path == str(fake_edf)
    assert result.head_path == ""


def test_write_block_inventory_csv_round_trip(tmp_path: Path):
    rows = [
        BlockProbeResult(
            subject="gaolan",
            recording_id="FA0013KQ",
            block_id="FA0013KQ",
            block_stem="FA0013KQ",
            block_start_epoch=1605829619.0,
            block_end_epoch=1605836819.0,
            duration_sec=7200.0,
            sample_rate=2000.0,
            n_channels_total=130,
            head_path="",
            data_path="/mnt/yuquan_data/yuquan_24h_edf/gaolan/FA0013KQ.edf",
            edf_path="/mnt/yuquan_data/yuquan_24h_edf/gaolan/FA0013KQ.edf",
        ),
    ]
    out_csv = tmp_path / "yuquan_block_inventory.csv"
    write_block_inventory_csv(rows, out_csv)

    with open(out_csv) as f:
        read_rows = list(csv.DictReader(f))

    assert len(read_rows) == 1
    assert read_rows[0]["subject"] == "gaolan"
    assert float(read_rows[0]["block_start_epoch"]) == 1605829619.0
    assert read_rows[0]["block_id"] == "FA0013KQ"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/honglab/leijiaxin/HFOsp
pytest tests/test_build_yuquan_block_inventory.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.build_yuquan_block_inventory'` (or ImportError for `BlockProbeResult`).

- [ ] **Step 3: Implement the script**

```python
# scripts/build_yuquan_block_inventory.py
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


def _probe_edf_metadata(edf_path: Path) -> Dict[str, float]:
    """Return {'duration_sec', 'sample_rate', 'n_channels_total'} from EDF header."""
    h = _parse_edf_header_for_streaming(edf_path)
    return {
        "duration_sec": float(h["n_data_records"] * h["record_duration"]),
        "sample_rate": float(h["sample_rate"]),
        "n_channels_total": int(h["n_signals"]),
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
        block_end_epoch=float(start) + meta["duration_sec"],
        duration_sec=meta["duration_sec"],
        sample_rate=meta["sample_rate"],
        n_channels_total=meta["n_channels_total"],
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
        seizure_idx = 0
        for file_row in payload.get("files", []):
            record = str(file_row.get("record", ""))
            blk = block_lookup.get((subject, record))
            for interval in file_row.get("seizure_intervals", []):
                seizure_idx += 1
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
                    "record_start_epoch": (blk.block_start_epoch if blk else ""),
                    "record_end_epoch": (blk.block_end_epoch if blk else ""),
                })
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
    print(f"Wrote {len(block_rows)} block rows → {block_csv}", flush=True)

    seizure_rows = rebuild_seizure_inventory_with_record_epochs(block_rows)
    seizure_csv = args.out_dir / "yuquan_seizure_inventory.csv"
    write_seizure_inventory_csv(seizure_rows, seizure_csv)
    print(f"Wrote {len(seizure_rows)} seizure rows → {seizure_csv}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/honglab/leijiaxin/HFOsp
pytest tests/test_build_yuquan_block_inventory.py -v
```
Expected: PASS, 2 tests.

- [ ] **Step 5: Run inventory build for real (9 audit_eligible yuquan)**

```bash
cd /home/honglab/leijiaxin/HFOsp
python scripts/build_yuquan_block_inventory.py
```
Expected output:
```
Probing 9 yuquan subjects under /mnt/yuquan_data/yuquan_24h_edf
Wrote ~108 block rows → results/dataset_inventory/yuquan_block_inventory.csv
Wrote 52 seizure rows → results/dataset_inventory/yuquan_seizure_inventory.csv
```
(Block count is approximate; depends on per-subject EDF counts. Seizure count = sum of n_seizures across 9 = 4+2+8+8+8+3+8+6+5 = 52.)

- [ ] **Step 6: Sanity-verify the inventories**

```bash
cd /home/honglab/leijiaxin/HFOsp
python3 -c "
import csv
with open('results/dataset_inventory/yuquan_seizure_inventory.csv') as f:
    rows = list(csv.DictReader(f))
print(f'seizure rows: {len(rows)}')
# Every row must have non-empty record_start_epoch
empty = [r for r in rows if not r['record_start_epoch']]
print(f'rows with empty record_start_epoch: {len(empty)}')
assert len(empty) == 0, f'JOIN FAILED: {[r[\"seizure_id\"] for r in empty]}'
# Every row must have eeg_onset_epoch within [record_start_epoch, record_end_epoch]
for r in rows:
    onset = float(r['eeg_onset_epoch'])
    s = float(r['record_start_epoch'])
    e = float(r['record_end_epoch'])
    assert s <= onset <= e, f'{r[\"seizure_id\"]}: onset {onset} out of [{s}, {e}]'
print('All seizure rows have valid record start/end epochs containing the onset.')
"
```
Expected: `All seizure rows have valid record start/end epochs containing the onset.`

- [ ] **Step 7: Commit**

```bash
cd /home/honglab/leijiaxin/HFOsp
git add scripts/build_yuquan_block_inventory.py tests/test_build_yuquan_block_inventory.py \
        results/dataset_inventory/yuquan_block_inventory.csv \
        results/dataset_inventory/yuquan_seizure_inventory.csv
git commit -m "$(cat <<'EOF'
feat(topic5 pr0.1 step1): yuquan block + seizure inventory

Build yuquan_block_inventory.csv from EDF probes for 9 audit_eligible
subjects and re-emit yuquan_seizure_inventory.csv with
record_start_epoch / record_end_epoch joined from the new block index.
Required upstream of dual-dataset extract_seizure_window.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: load_yuquan_record loader

**Files:**
- Create: `src/yuquan_dataset.py`
- Create: `tests/test_yuquan_dataset.py`

API contract (must match `src/preprocessing.py:452 load_epilepsiae_block` consumer surface):

```python
def load_yuquan_record(
    edf_path: Path | str,
    *,
    reference: str = "car",
    segment_sec: float = 200.0,
    intracranial_only: bool = True,
) -> PreprocessingResult:
    """Load yuquan EDF, filter to intracranial channels, apply CAR or bipolar."""
```

`PreprocessingResult` (from `src/preprocessing.py:47`) has fields `data` (n_ch, n_samples), `sfreq`, `ch_names`. The yuquan loader must populate these compatibly so `extract_seizure_window` can slice from `pre.data[:, i0:i1]` exactly as it does for epilepsiae.

**Channel filter rule (grounded in real-data probe of gaolan FA0013KP.edf and huanghanwen, 2026-05-10):**

Yuquan EDFs use two prefixes:
- `EEG <name>-Ref`: scalp / scalp-reference channels (10-20 system; extended set including `F1-F10`, `C1-C6`, `P3, P4, O1, O2, T3-T6, Fp1, Fp2, Fz, Cz, Pz, A1, A2`)
- `POL <name>`: intracranial OR auxiliary

Real-EDF non-SEEG `POL ` channels seen: `POL DC10`, `POL ECG`, `POL EMG1`, `POL EMG2`, `POL EMGL-0/1`, `POL EMGLR`, `POL EMGR`, `POL E` (bare letter, no digits).

Real-EDF SEEG names follow `<single uppercase letter><optional apostrophe><1–3 digits>`: e.g., `A1`, `A'1`, `B10`, `B'16`, `C7`, `D'12`, `E14`, `F16`, `G1`. NEVER multi-letter (DC10, EMG1, ECG are NOT SEEG). NEVER bare letter (POL E is aux).

Final classification rules:
1. After `normalize_yuquan_channel_name` (strips `EEG ` / `POL ` / `-Ref`):
   - In `_SCALP_REF` = {A1, A2} → `scalp_ref`
   - In `_SCALP_10_20` (extended set, see code) → `scalp`
   - Starts with explicit aux prefix in `_AUX_PREFIXES` = (DC, EKG, ECG, EMG, OSAT, PULSE, SPO2, BR, PR, BP) → `aux`
   - Matches regex `^[A-Z]'?\d{1,3}$` (single letter, optional prime, 1-3 digits) → `intracranial`
   - Anything else → `aux`

The single-letter regex deliberately rejects `DC10` (2 letters), `EMG1` (3 letters), `ECG` (no digits), `POL E` (no digits) while accepting `A1`, `A'1`, `B'16`, `K11`, `E10`, `F16`.

**Bipolar pairing rule:** adjacent-contact pairs WITHIN a probe (same letter + same prime status). E.g., `A1-A2` is a valid pair; `A'3-A'4` is a valid pair; `A1-A'1` is NOT (different probes). Reuse `_build_bipolar_pairs` at `src/preprocessing.py:334` after channel-name normalization, but if its grouping logic doesn't handle the apostrophe correctly, treat `A'` and `A` as distinct probes by passing the normalized name (which preserves the apostrophe).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_yuquan_dataset.py
"""Tests for src/yuquan_dataset.py.

Synthetic-EDF only — does NOT touch real /mnt/yuquan_data/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.yuquan_dataset import (
    classify_yuquan_channel,
    normalize_yuquan_channel_name,
    load_yuquan_record,
)


@pytest.mark.parametrize("raw_name,expected", [
    # scalp ref
    ("EEG A1-Ref", "scalp_ref"),
    ("EEG A2-Ref", "scalp_ref"),
    # 10-20 scalp (extended)
    ("EEG Fp1-Ref", "scalp"),
    ("EEG C3-Ref", "scalp"),
    ("EEG F10-Ref", "scalp"),
    ("EEG C5-Ref", "scalp"),
    # intracranial (single letter + optional apostrophe + 1-3 digits)
    ("POL A3", "intracranial"),
    ("POL A'1", "intracranial"),
    ("POL B'16", "intracranial"),
    ("POL K11", "intracranial"),
    ("POL E10", "intracranial"),
    ("POL F16", "intracranial"),
    # aux: DC channels (false-positive of naive regex — must be aux)
    ("POL DC01", "aux"),
    ("POL DC10", "aux"),
    # aux: physio
    ("POL ECG", "aux"),
    ("POL EMG1", "aux"),
    ("POL EMGLR", "aux"),
    ("POL OSAT", "aux"),
    ("POL PULSE", "aux"),
    # aux: bare letter, no digits
    ("POL E", "aux"),
    # SEEG-named channel under EEG prefix (real subjects sometimes use this)
    ("EEG K11-Ref", "intracranial"),
])
def test_classify_yuquan_channel(raw_name, expected):
    assert classify_yuquan_channel(raw_name) == expected


@pytest.mark.parametrize("raw_name,expected", [
    ("EEG A1-Ref", "A1"),
    ("POL K11", "K11"),
    ("POL E10", "E10"),
    ("POL A'1", "A'1"),
    ("POL B'16", "B'16"),
    ("EEG K11-Ref", "K11"),
])
def test_normalize_yuquan_channel_name(raw_name, expected):
    assert normalize_yuquan_channel_name(raw_name) == expected


def _make_synthetic_yuquan_edf(tmp_path: Path) -> Path:
    """Use mne.io.RawArray + export_raw to write a small EDF with mixed channel
    types so we can test classify + load on a real file format."""
    import mne
    sfreq = 200.0
    duration = 60.0
    n_samples = int(sfreq * duration)
    # 4 intracranial + 1 scalp + 1 aux
    ch_names = ["POL K3", "POL K4", "POL K5", "POL K6", "EEG Fp1-Ref", "POL DC01"]
    data = np.random.RandomState(0).randn(len(ch_names), n_samples).astype(np.float64) * 1e-5
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    edf_path = tmp_path / "FAKE001.edf"
    mne.export.export_raw(str(edf_path), raw, fmt="edf", overwrite=True, verbose=False)
    return edf_path


def test_load_yuquan_record_intracranial_filter(tmp_path: Path):
    pytest.importorskip("mne")
    edf_path = _make_synthetic_yuquan_edf(tmp_path)

    pre = load_yuquan_record(edf_path, reference="car", intracranial_only=True)

    # Only the 4 POL K* channels should remain
    assert pre.data.shape[0] == 4
    assert all(name in {"K3", "K4", "K5", "K6"} for name in pre.ch_names)
    assert abs(pre.sfreq - 200.0) < 1e-3


def test_load_yuquan_record_car_zero_mean(tmp_path: Path):
    pytest.importorskip("mne")
    edf_path = _make_synthetic_yuquan_edf(tmp_path)

    pre = load_yuquan_record(edf_path, reference="car", intracranial_only=True)

    # CAR must zero-mean across channels at every sample
    sample_means = pre.data.mean(axis=0)
    assert np.allclose(sample_means, 0.0, atol=1e-12)


def test_load_yuquan_record_bipolar_reduces_count(tmp_path: Path):
    pytest.importorskip("mne")
    edf_path = _make_synthetic_yuquan_edf(tmp_path)

    pre = load_yuquan_record(edf_path, reference="bipolar", intracranial_only=True)

    # 4 contiguous K3..K6 → 3 bipolar pairs K3-K4, K4-K5, K5-K6
    assert pre.data.shape[0] == 3
    assert pre.ch_names == ["K3-K4", "K4-K5", "K5-K6"]
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cd /home/honglab/leijiaxin/HFOsp
pytest tests/test_yuquan_dataset.py -v
```
Expected: FAIL `ModuleNotFoundError: No module named 'src.yuquan_dataset'`.

- [ ] **Step 3: Implement loader**

```python
# src/yuquan_dataset.py
"""Yuquan SEEG record loader for ictal-window pipelines.

Mirrors the consumer surface of :func:`src.preprocessing.load_epilepsiae_block`
so that :func:`src.ictal_onset_extraction.extract_seizure_window` can route
to either dataset transparently.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.preprocessing import PreprocessingResult, _build_bipolar_pairs


# Single uppercase letter, optional apostrophe (contralateral SEEG probes use
# A', B', etc.), 1-3 digits. Deliberately rejects multi-letter prefixes like
# DC10 / EMG1 / ECG that real yuquan EDFs emit as auxiliary channels.
_INTRACRANIAL_NAME_RE = re.compile(r"^[A-Z]'?\d{1,3}$")

_SCALP_REF = frozenset({"A1", "A2"})
_SCALP_10_20 = frozenset({
    # Frontal
    "Fp1", "Fp2", "Fz",
    "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10",
    # Central / temporal
    "C1", "C2", "C3", "C4", "C5", "C6", "Cz",
    "T3", "T4", "T5", "T6",
    "FT9", "FT10",
    # Parietal / occipital
    "P3", "P4", "Pz",
    "O1", "O2",
})
_AUX_PREFIXES = (
    "DC", "EKG", "ECG", "EMG", "OSAT", "PULSE", "SPO2", "BR", "PR", "BP",
)


def normalize_yuquan_channel_name(raw: str) -> str:
    """Strip 'EEG '/'POL ' prefix and '-Ref' suffix; collapse whitespace.

    Apostrophes (e.g., POL A'1 → A'1) are preserved — they encode
    contralateral SEEG probes and must not be lost.
    """
    s = raw.strip()
    for prefix in ("EEG ", "POL "):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    if s.endswith("-Ref"):
        s = s[:-len("-Ref")]
    return s.strip()


def classify_yuquan_channel(raw: str) -> str:
    """Return one of {'intracranial', 'scalp', 'scalp_ref', 'aux'}.

    Order of checks matters: aux prefixes are checked BEFORE the
    intracranial regex because real yuquan EDFs emit DC10/EMG1/ECG as
    auxiliary channels whose normalized names (DC10, EMG1, ECG) would
    otherwise pass a naïve [A-Z]{1,3}\\d{1,2} pattern.
    """
    norm = normalize_yuquan_channel_name(raw)
    if norm in _SCALP_REF:
        return "scalp_ref"
    if norm in _SCALP_10_20:
        return "scalp"
    if any(norm.startswith(p) for p in _AUX_PREFIXES):
        return "aux"
    if _INTRACRANIAL_NAME_RE.match(norm):
        return "intracranial"
    return "aux"


def _select_intracranial_indices(ch_names: List[str]) -> Tuple[List[int], List[str]]:
    keep_idx: List[int] = []
    keep_names: List[str] = []
    for i, raw in enumerate(ch_names):
        if classify_yuquan_channel(raw) == "intracranial":
            keep_idx.append(i)
            keep_names.append(normalize_yuquan_channel_name(raw))
    return keep_idx, keep_names


def load_yuquan_record(
    edf_path: Path | str,
    *,
    reference: str = "car",
    segment_sec: float = 200.0,  # accepted for API parity; not used (preload=True)
    intracranial_only: bool = True,
) -> PreprocessingResult:
    """Load a yuquan SEEG EDF, filter to intracranial channels, apply reference.

    Parameters
    ----------
    edf_path : str or Path
        Path to the EDF file.
    reference : str
        'car' (default) for common-average reference (zero-mean per sample),
        'bipolar' for adjacent-contact pairs within each probe.
    intracranial_only : bool
        If True (default) drop scalp/scalp_ref/aux channels via
        :func:`classify_yuquan_channel`. Yuquan analyses always want this.
    """
    import mne  # local import to avoid penalising imports of this module

    edf_path = Path(edf_path)
    if not edf_path.exists():
        raise FileNotFoundError(edf_path)

    raw = mne.io.read_raw_edf(
        str(edf_path), preload=True, verbose=False, encoding="latin1"
    )
    sfreq = float(raw.info["sfreq"])
    data = raw.get_data().astype(np.float64, copy=False)
    ch_names_raw = list(raw.ch_names)

    if intracranial_only:
        keep_idx, keep_names = _select_intracranial_indices(ch_names_raw)
        if not keep_idx:
            raise ValueError(
                f"No intracranial channels found in {edf_path.name}; "
                f"first 8 raw channels: {ch_names_raw[:8]}"
            )
        data = data[keep_idx]
        ch_names = keep_names
    else:
        ch_names = [normalize_yuquan_channel_name(c) for c in ch_names_raw]

    if reference == "car":
        data = data - data.mean(axis=0, keepdims=True)
        out_names = ch_names
    elif reference == "bipolar":
        pairs, pair_names = _build_bipolar_pairs(ch_names)
        if not pairs:
            raise ValueError(
                f"No valid bipolar pairs in {edf_path.name} after filtering"
            )
        data = np.stack([data[a] - data[b] for a, b in pairs], axis=0)
        out_names = pair_names
    else:
        raise ValueError(f"reference must be 'car' or 'bipolar', got {reference!r}")

    return PreprocessingResult(
        data=data.astype(np.float64, copy=False),
        sfreq=sfreq,
        ch_names=out_names,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/honglab/leijiaxin/HFOsp
pytest tests/test_yuquan_dataset.py -v
```
Expected: PASS, 13+ tests.

- [ ] **Step 5: Smoke-test on a real yuquan EDF**

```bash
cd /home/honglab/leijiaxin/HFOsp
python3 -c "
from src.yuquan_dataset import load_yuquan_record
pre = load_yuquan_record('/mnt/yuquan_data/yuquan_24h_edf/gaolan/FA0013KP.edf', reference='car')
print(f'sfreq={pre.sfreq}, n_channels={pre.data.shape[0]}, n_samples={pre.data.shape[1]}')
print(f'first 8 channels (normalized): {pre.ch_names[:8]}')
print(f'CAR zero-mean check (should be near 0): {pre.data.mean(axis=0)[:5]}')
"
```
Expected: `sfreq=2000.0, n_channels≈130 (gaolan), n_samples=14400162, channels are short SEEG names like K3/E10/...`

- [ ] **Step 6: Commit**

```bash
cd /home/honglab/leijiaxin/HFOsp
git add src/yuquan_dataset.py tests/test_yuquan_dataset.py
git commit -m "$(cat <<'EOF'
feat(topic5 pr0.1 step2): yuquan record loader for ictal pipelines

Add src/yuquan_dataset.load_yuquan_record with PreprocessingResult-shaped
output. Filters to intracranial SEEG channels via channel-name regex,
applies CAR or bipolar reference. Mirrors the consumer surface of
src.preprocessing.load_epilepsiae_block so extract_seizure_window can
route on dataset.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: extract_seizure_window dual-dataset routing

**Files:**
- Modify: `src/ictal_onset_extraction.py:214-274` (`_resolve_inventory_paths` + `extract_seizure_window`)
- Create: `tests/test_ictal_onset_extraction_yuquan.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ictal_onset_extraction_yuquan.py
"""Integration tests for yuquan branch of extract_seizure_window.

Uses synthetic inventories + a small EDF written to tmp_path.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from src.ictal_onset_extraction import extract_seizure_window


def _write_inventory_pair(tmp_root: Path, edf_path: Path, fs: float, duration: float):
    inv_dir = tmp_root / "dataset_inventory"
    inv_dir.mkdir(parents=True)

    block_csv = inv_dir / "yuquan_block_inventory.csv"
    seizure_csv = inv_dir / "yuquan_seizure_inventory.csv"

    block_start = 1700000000.0
    block_end = block_start + duration

    with open(block_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "subject", "recording_id", "block_id", "block_stem",
            "block_start_epoch", "block_end_epoch", "duration_sec",
            "sample_rate", "n_channels_total", "head_path", "data_path", "edf_path",
        ])
        w.writerow([
            "fakesid", edf_path.stem, edf_path.stem, edf_path.stem,
            f"{block_start}", f"{block_end}", f"{duration}",
            f"{fs}", "6", "", str(edf_path), str(edf_path),
        ])

    onset_epoch = block_start + duration / 2.0
    with open(seizure_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "subject", "patient_code", "recording_id", "record", "seizure_id",
            "eeg_onset_epoch", "eeg_offset_epoch", "eeg_duration_sec",
            "has_complete_eeg_interval", "timezone_name",
            "eeg_onset_local_hour", "eeg_onset_day_night",
            "record_start_epoch", "record_end_epoch",
        ])
        w.writerow([
            "fakesid", "fakesid", edf_path.stem, edf_path.stem, "fakesid_sz_001",
            f"{onset_epoch}", f"{onset_epoch + 60}", "60.0",
            "True", "Asia/Shanghai", "", "",
            f"{block_start}", f"{block_end}",
        ])


def _make_yuquan_synthetic_edf(tmp_path: Path) -> tuple[Path, float, float]:
    pytest.importorskip("mne")
    import mne
    sfreq = 500.0
    duration = 800.0
    n_samples = int(sfreq * duration)
    ch_names = ["POL K3", "POL K4", "POL K5", "POL K6", "EEG Fp1-Ref", "POL DC01"]
    data = np.random.RandomState(1).randn(len(ch_names), n_samples) * 1e-5
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    edf_path = tmp_path / "FAKEY1.edf"
    mne.export.export_raw(str(edf_path), raw, fmt="edf", overwrite=True, verbose=False)
    return edf_path, sfreq, duration


def test_extract_seizure_window_yuquan_branch(tmp_path: Path):
    edf_path, sfreq, duration = _make_yuquan_synthetic_edf(tmp_path)
    _write_inventory_pair(tmp_path, edf_path, sfreq, duration)

    sw = extract_seizure_window(
        "yuquan/fakesid",
        seizure_idx=0,
        pre_sec=300.0,
        post_sec=30.0,
        results_root=tmp_path,
        reference="car",
    )

    assert sw.subject == "yuquan/fakesid"
    assert sw.seizure_id == "fakesid_sz_001"
    assert sw.fs == sfreq
    assert sw.signal.shape[0] == 4  # 4 intracranial channels (K3..K6)
    assert sw.signal.shape[1] == int((300.0 + 30.0) * sfreq)
    assert sw.t_axis[0] == pytest.approx(-300.0)
    assert sw.t_axis[-1] == pytest.approx(30.0 - 1.0 / sfreq, abs=1e-3)


def test_extract_seizure_window_yuquan_window_overruns_block(tmp_path: Path):
    edf_path, sfreq, duration = _make_yuquan_synthetic_edf(tmp_path)
    _write_inventory_pair(tmp_path, edf_path, sfreq, duration)

    # Request pre_sec longer than the seizure offset from block start
    # (onset = block_start + duration/2 = block_start + 400; pre_sec=500 → before block_start)
    with pytest.raises(ValueError, match="before block_start"):
        extract_seizure_window(
            "yuquan/fakesid",
            seizure_idx=0,
            pre_sec=500.0,
            post_sec=30.0,
            results_root=tmp_path,
        )
```

- [ ] **Step 2: Run failing tests**

```bash
cd /home/honglab/leijiaxin/HFOsp
pytest tests/test_ictal_onset_extraction_yuquan.py -v
```
Expected: FAIL with `NotImplementedError: extract_seizure_window currently only supports epilepsiae`.

- [ ] **Step 3: Modify `_resolve_inventory_paths` to take dataset arg**

In `src/ictal_onset_extraction.py`, replace lines 214-235 with:

```python
def _resolve_inventory_paths(
    results_root: Path,
    dataset: str = "epilepsiae",
) -> Tuple[Path, Path]:
    """Return (seizure_inventory_csv, block_inventory_csv) under ``results_root``.

    Looks first in ``results/dataset_inventory/`` then falls back to
    ``results/`` (legacy layout that ``src.epilepsiae_dataset`` writes by
    default).
    """
    seizure_name = f"{dataset}_seizure_inventory.csv"
    block_name = f"{dataset}_block_inventory.csv"
    candidates = (
        results_root / "dataset_inventory" / seizure_name,
        results_root / seizure_name,
    )
    for sz in candidates:
        if sz.exists():
            blk = sz.with_name(block_name)
            if blk.exists():
                return sz, blk
    raise FileNotFoundError(
        f"{seizure_name} not found under {results_root}"
    )
```

- [ ] **Step 4: Modify `extract_seizure_window` to route on dataset**

In `src/ictal_onset_extraction.py`, replace lines 268-346 (the function body after the docstring) with:

```python
    if "/" not in subject:
        raise ValueError(f"subject must be '<dataset>/<id>', got {subject!r}")
    dataset, sid = subject.split("/", 1)
    if dataset not in {"epilepsiae", "yuquan"}:
        raise NotImplementedError(
            f"extract_seizure_window supports {{epilepsiae, yuquan}}; got {dataset}"
        )

    results_root = Path(results_root)
    sz_csv, blk_csv = _resolve_inventory_paths(results_root, dataset=dataset)
    sz_rows = [r for r in _read_csv_rows(sz_csv) if r["subject"] == sid]
    if dataset == "epilepsiae":
        sz_rows = [r for r in sz_rows if r.get("clin_onset_epoch")]
        sz_rows.sort(key=lambda r: float(r["clin_onset_epoch"]))
        onset_field = "clin_onset_epoch"
        block_id_field = "block_id"
    else:  # yuquan
        sz_rows = [r for r in sz_rows if r.get("eeg_onset_epoch")]
        sz_rows.sort(key=lambda r: float(r["eeg_onset_epoch"]))
        onset_field = "eeg_onset_epoch"
        block_id_field = "record"
    if not sz_rows:
        raise ValueError(f"No seizures with onset_epoch found for {subject}")
    if not (0 <= seizure_idx < len(sz_rows)):
        raise IndexError(
            f"seizure_idx={seizure_idx} out of range for {subject} (n={len(sz_rows)})"
        )
    sz = sz_rows[seizure_idx]
    block_id = sz[block_id_field]
    clin_onset_epoch = float(sz[onset_field])
    eeg_onset_epoch = (
        float(sz["eeg_onset_epoch"]) if (dataset == "epilepsiae" and sz.get("eeg_onset_epoch")) else None
    )
    # For yuquan, eeg_onset is the only annotation; treat it as the clinical
    # onset reference (same as epilepsiae's clin_onset_epoch in the contract).

    blk_rows = [
        r for r in _read_csv_rows(blk_csv)
        if r["subject"] == sid and r[block_id_field] == block_id
    ]
    if not blk_rows:
        raise ValueError(f"{block_id_field}={block_id} not found in block inventory for {subject}")
    blk = blk_rows[0]
    block_start_epoch = float(blk["block_start_epoch"])
    block_end_epoch = float(blk["block_end_epoch"])

    if dataset == "epilepsiae":
        head_path = blk["head_path"]
        data_path = blk["data_path"]
        if not head_path or not data_path:
            raise ValueError(f"block {block_id} missing head/data path in inventory")
    else:  # yuquan
        head_path = ""
        data_path = blk.get("edf_path") or blk.get("data_path")
        if not data_path:
            raise ValueError(f"record {block_id} missing edf_path in yuquan inventory")

    win_start_epoch = clin_onset_epoch - float(pre_sec)
    win_end_epoch = clin_onset_epoch + float(post_sec)
    if win_start_epoch < block_start_epoch:
        raise ValueError(
            f"{subject} seizure {seizure_idx}: requested window starts "
            f"{block_start_epoch - win_start_epoch:.2f}s before block_start; "
            f"upstream caller must drop this seizure"
        )
    if win_end_epoch > block_end_epoch:
        raise ValueError(
            f"{subject} seizure {seizure_idx}: requested window ends "
            f"{win_end_epoch - block_end_epoch:.2f}s after block_end; "
            f"upstream caller must drop this seizure"
        )

    if dataset == "epilepsiae":
        from src.preprocessing import load_epilepsiae_block
        pre = load_epilepsiae_block(
            data_path, head_path, reference=reference, segment_sec=200.0,
        )
        block_stem_for_window = blk["block_stem"]
    else:  # yuquan
        from src.yuquan_dataset import load_yuquan_record
        pre = load_yuquan_record(
            data_path, reference=reference, segment_sec=200.0, intracranial_only=True,
        )
        block_stem_for_window = blk["block_stem"]

    fs = float(pre.sfreq)
    rel_start_sec = win_start_epoch - block_start_epoch
    i0 = int(round(rel_start_sec * fs))
    i1 = i0 + int(round((float(pre_sec) + float(post_sec)) * fs))
    sliced = pre.data[:, i0:i1].copy()
    n_samples_actual = sliced.shape[1]
    t_axis = (np.arange(n_samples_actual) / fs) - float(pre_sec)

    return SeizureWindow(
        signal=sliced,
        fs=fs,
        t_axis=t_axis,
        ch_names=list(pre.ch_names),
        subject=subject,
        seizure_id=sz["seizure_id"],
        block_stem=block_stem_for_window,
        clin_onset_epoch=clin_onset_epoch,
        eeg_onset_epoch=eeg_onset_epoch,
        pre_sec=float(pre_sec),
        post_sec=float(post_sec),
    )
```

- [ ] **Step 5: Run yuquan tests + epilepsiae regression tests**

```bash
cd /home/honglab/leijiaxin/HFOsp
pytest tests/test_ictal_onset_extraction_yuquan.py -v
```
Expected: PASS, 2 tests.

```bash
pytest tests/test_ictal_onset_extraction.py -v
```
Expected: PASS, all existing epilepsiae tests still pass.

- [ ] **Step 6: Smoke-test extract_seizure_window on real gaolan data**

```bash
cd /home/honglab/leijiaxin/HFOsp
python3 -c "
from src.ictal_onset_extraction import extract_seizure_window
sw = extract_seizure_window('yuquan/gaolan', 0, pre_sec=300.0, post_sec=30.0)
print(f'subject={sw.subject}, seizure_id={sw.seizure_id}')
print(f'signal shape: {sw.signal.shape}, fs={sw.fs}')
print(f't_axis[0]={sw.t_axis[0]:.3f}, t_axis[-1]={sw.t_axis[-1]:.3f}')
print(f'first 5 ch_names: {sw.ch_names[:5]}')
"
```
Expected: signal shape ≈ (130, 660000), fs=2000, t_axis from -300 to ≈30. Channels look like `K3, K4, ...`.

- [ ] **Step 7: Commit**

```bash
cd /home/honglab/leijiaxin/HFOsp
git add src/ictal_onset_extraction.py tests/test_ictal_onset_extraction_yuquan.py
git commit -m "$(cat <<'EOF'
feat(topic5 pr0.1 step3): dual-dataset extract_seizure_window

Replace NotImplementedError at src/ictal_onset_extraction.py:271 with a
yuquan branch that joins yuquan_seizure_inventory.csv against
yuquan_block_inventory.csv and loads the intracranial signal via
src.yuquan_dataset.load_yuquan_record. Epilepsiae path unchanged.

_resolve_inventory_paths now takes a `dataset` arg and resolves
{dataset}_seizure_inventory.csv + {dataset}_block_inventory.csv.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Cohort runner — accept yuquan, load yuquan SOZ, count yuquan seizures

**Files:**
- Modify: `scripts/run_ictal_er_rank.py:112` — SUPPORTED_DATASETS
- Modify: `scripts/run_ictal_er_rank.py:122` — `_focus_rel_path`
- Modify: `scripts/run_ictal_er_rank.py:135` — `_focal_channels`
- Modify: `scripts/run_ictal_er_rank.py:140` — `_count_seizures`
- Modify: `scripts/run_ictal_er_rank.py:127` — `_seizure_inventory_path`

- [ ] **Step 1: Write smoke-test for cohort selector**

Append to `tests/test_ictal_onset_extraction_yuquan.py`:

```python
def test_cohort_selector_includes_yuquan_audit_eligible():
    from scripts.run_ictal_er_rank import _cohort_subject_list
    included, excluded = _cohort_subject_list()
    yuquan_in = [s for s in included if s.startswith("yuquan/")]
    epi_in = [s for s in included if s.startswith("epilepsiae/")]
    assert len(yuquan_in) == 9, f"expected 9 yuquan, got {yuquan_in}"
    assert "yuquan/gaolan" in yuquan_in
    assert "yuquan/zhangjinhan" in yuquan_in
    # No yuquan should be in `excluded` after this PR
    assert not any(s.startswith("yuquan/") for s in excluded), excluded
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd /home/honglab/leijiaxin/HFOsp
pytest tests/test_ictal_onset_extraction_yuquan.py::test_cohort_selector_includes_yuquan_audit_eligible -v
```
Expected: FAIL — yuquan is currently in `excluded`, not `included`.

- [ ] **Step 3: Modify `SUPPORTED_DATASETS`**

In `scripts/run_ictal_er_rank.py:112`, change:
```python
SUPPORTED_DATASETS = frozenset({"epilepsiae"})
```
to
```python
SUPPORTED_DATASETS = frozenset({"epilepsiae", "yuquan"})
```

- [ ] **Step 4: Modify `_focus_rel_path`, `_seizure_inventory_path`, `_focal_channels`, `_count_seizures` to branch on dataset**

In `scripts/run_ictal_er_rank.py`, replace lines 122-148 (`_focus_rel_path`, `_seizure_inventory_path`, `_load_focus_rel`, `_focal_channels`, `_count_seizures`) with:

```python
def _focus_rel_path(dataset: str = "epilepsiae") -> Path:
    if dataset == "epilepsiae":
        return ROOT / "results" / "epilepsiae_electrode_focus_rel.json"
    if dataset == "yuquan":
        return ROOT / "results" / "yuquan_soz_core_channels.json"
    raise ValueError(f"Unsupported dataset for focus_rel: {dataset}")


def _seizure_inventory_path(dataset: str = "epilepsiae") -> Path:
    candidates = (
        ROOT / "results" / "dataset_inventory" / f"{dataset}_seizure_inventory.csv",
        ROOT / "results" / f"{dataset}_seizure_inventory.csv",
    )
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"{dataset}_seizure_inventory.csv not found")


def _load_focus_rel(dataset: str = "epilepsiae") -> Dict:
    path = _focus_rel_path(dataset)
    with path.open() as f:
        d = json.load(f)
    if dataset == "yuquan":
        # Normalize flat {sid: [channels]} to {sid: {"i": [channels]}} so the
        # caller surface matches the epilepsiae 3-tier focus_rel JSON.
        return {sid: {"i": list(chans), "l": [], "e": []} for sid, chans in d.items()}
    return d


def _focal_channels(subject: str, focus_rel: Dict) -> List[str]:
    sid = subject.split("/", 1)[1]
    return list(focus_rel.get(sid, {}).get("i", []))


def _count_seizures(subject: str) -> int:
    dataset, sid = subject.split("/", 1)
    inv = _seizure_inventory_path(dataset)
    onset_col = "clin_onset_epoch" if dataset == "epilepsiae" else "eeg_onset_epoch"
    n = 0
    with inv.open() as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("subject") == sid and row.get(onset_col):
                n += 1
    return n
```

Then update both call sites with two exact edits.

**Edit 1: `_run_sentinel` (around line 580)** — change:

```python
def _run_sentinel(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    focus_rel = _load_focus_rel()
```

to:

```python
def _run_sentinel(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    focus_rel_per_dataset = {
        "epilepsiae": _load_focus_rel("epilepsiae"),
        "yuquan": _load_focus_rel("yuquan"),
    }
```

And inside the `for subject in SENTINEL_SUBJECTS:` loop change:

```python
        focal = _focal_channels(subject, focus_rel)
```

to:

```python
        ds = subject.split("/", 1)[0]
        focal = _focal_channels(subject, focus_rel_per_dataset[ds])
```

**Edit 2: `_run_cohort` (around line 770)** — change:

```python
def _run_cohort(out_dir: Path, *, skip_existing: bool = True) -> int:
    """Step A.4 — cohort run on audit_eligible 24 + sentinel-only 916.
    ...
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    focus_rel = _load_focus_rel()
    subjects, excluded = _cohort_subject_list()
```

to:

```python
def _run_cohort(out_dir: Path, *, skip_existing: bool = True) -> int:
    """Step A.4 — cohort run on audit_eligible 24 + sentinel-only 916.
    ...
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    focus_rel_per_dataset = {
        "epilepsiae": _load_focus_rel("epilepsiae"),
        "yuquan": _load_focus_rel("yuquan"),
    }
    subjects, excluded = _cohort_subject_list()
```

And inside the `for subject in subjects:` loop change:

```python
        focal = _focal_channels(subject, focus_rel)
```

to:

```python
        ds = subject.split("/", 1)[0]
        focal = _focal_channels(subject, focus_rel_per_dataset[ds])
```

Both edits are mechanical line-substitutions; the surrounding control flow does not change.

- [ ] **Step 5: Run cohort selector test**

```bash
cd /home/honglab/leijiaxin/HFOsp
pytest tests/test_ictal_onset_extraction_yuquan.py::test_cohort_selector_includes_yuquan_audit_eligible -v
```
Expected: PASS — `included` contains 15 epilepsiae + 9 yuquan + 1 sentinel-only (epilepsiae/916) = 25.

- [ ] **Step 6: Dry-run `_count_seizures` on each yuquan subject**

```bash
cd /home/honglab/leijiaxin/HFOsp
python3 -c "
from scripts.run_ictal_er_rank import _count_seizures
for sid in ['gaolan', 'huanghanwen', 'litengsheng', 'pengzihang',
            'sunyuanxin', 'xuxinyi', 'zhangjinhan', 'zhangkexuan', 'zhaojinrui']:
    print(f'yuquan/{sid}: n_seizures = {_count_seizures(\"yuquan/\" + sid)}')
"
```
Expected: counts must match audit.csv n_seizures column: 4, 2, 8, 8, 8, 3, 8, 6, 5.

- [ ] **Step 7: Commit**

```bash
cd /home/honglab/leijiaxin/HFOsp
git add scripts/run_ictal_er_rank.py tests/test_ictal_onset_extraction_yuquan.py
git commit -m "$(cat <<'EOF'
feat(topic5 pr0.1 step4): cohort runner accepts yuquan dataset

Expand SUPPORTED_DATASETS to {epilepsiae, yuquan}; route _focus_rel_path,
_seizure_inventory_path, _count_seizures by dataset prefix; normalize
yuquan_soz_core_channels.json (flat {sid: [chans]}) into the 3-tier
{i, l, e} surface that classify_clinical_concordance expects (l/e empty
for yuquan).

Cohort selector now includes 9 yuquan audit_eligible subjects; expected
cohort size 25 (15 epilepsiae + 9 yuquan + sentinel-only 916).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Sentinel canary on gaolan + huanghanwen

This task runs the v2.3 ER-rank pipeline on 2 yuquan canaries before scaling to all 9. **No code changes; pure runtime sanity.**

**Files:**
- Output: `results/data_driven_soz/layer_a_ictal_er_rank/per_subject/yuquan_gaolan.json`
- Output: `results/data_driven_soz/layer_a_ictal_er_rank/per_subject/yuquan_huanghanwen.json`
- Log: `results/run_logs/yuquan_canary_<timestamp>.log`

- [ ] **Step 1: Add `--subject-allowlist` CLI flag for targeted runs**

The current cohort runner does not support per-subject filtering; add a minimal allowlist plumbing via three exact edits.

**Edit 1: `_run_cohort` signature (around line 770)** — change:

```python
def _run_cohort(out_dir: Path, *, skip_existing: bool = True) -> int:
```

to:

```python
def _run_cohort(
    out_dir: Path,
    *,
    skip_existing: bool = True,
    allowlist: Optional[List[str]] = None,
) -> int:
```

**Edit 2: `_run_cohort` body, immediately after `subjects, excluded = _cohort_subject_list()`** — insert:

```python
    if allowlist is not None:
        allow_set = set(allowlist)
        subjects = [s for s in subjects if s in allow_set]
        if not subjects:
            print(f"[cohort] allowlist filtered to empty set; nothing to do", flush=True)
            return 0
```

**Edit 3: `main()` argparse (around line 999)** — add a new argument right after the existing `--no-skip-existing` argument:

```python
    parser.add_argument(
        "--subject-allowlist",
        nargs="+",
        default=None,
        help="If set, only process subjects whose '<ds>/<sid>' matches one "
             "entry. Used for canary runs (e.g. `--subject-allowlist "
             "yuquan/gaolan yuquan/huanghanwen`).",
    )
```

And in `main()` change the cohort-dispatch line (around line 1028) from:

```python
        return _run_cohort(out_dir, skip_existing=not args.no_skip_existing)
```

to:

```python
        return _run_cohort(
            out_dir,
            skip_existing=not args.no_skip_existing,
            allowlist=args.subject_allowlist,
        )
```

- [ ] **Step 2: Execute canary run**

```bash
cd /home/honglab/leijiaxin/HFOsp
mkdir -p results/run_logs
python scripts/run_ictal_er_rank.py --per-subject --force \
    --subject-allowlist yuquan/gaolan yuquan/huanghanwen \
    2>&1 | tee results/run_logs/yuquan_canary_$(date +%Y%m%d_%H%M).log
```
Expected runtime: ~5-15 min total (2 subjects, each ~30 sec/seizure × 2-4 seizures).

- [ ] **Step 3: Inspect canary outputs**

```bash
cd /home/honglab/leijiaxin/HFOsp
for sid in gaolan huanghanwen; do
    echo "=== yuquan_${sid} ==="
    python3 -c "
import json
d = json.load(open('results/data_driven_soz/layer_a_ictal_er_rank/per_subject/yuquan_${sid}.json'))
print(f'  schema_version: {d[\"schema_version\"]}')
print(f'  n_seizures_total: {d[\"n_seizures_total\"]}, focal_channels: {len(d[\"focal_channels\"])}')
for er_key in ['gamma_ER', 'broad_ER']:
    rec = d['per_er'].get(er_key, {})
    print(f'  {er_key}: n_ok={rec.get(\"n_seizures_ok\")}/{rec.get(\"n_seizures_loaded\")} (lambda={rec.get(\"lambda\")})')
    print(f'    producer_health: {d[\"producer_health\"].get(er_key)}, clinical_concordance: {d[\"clinical_concordance\"].get(er_key)}')
"
done
```

**Acceptance criteria for unblock to Task 6:**
- Both subjects emit valid JSON with `schema_version == "pr_t3_1_layer_a_v2_3_timing"`
- For at least one of (gaolan, huanghanwen) AND at least one of (gamma_ER, broad_ER): `n_seizures_ok >= 2` (i.e., the seizure → r_sz pipeline ran end-to-end on real yuquan data)
- `producer_health` is one of {stable, moderate, unstable, insufficient} (NOT a Python exception)
- No `extract_seizure_window` exceptions in the log

If any criterion fails, STOP and audit before scaling.

- [ ] **Step 4: Commit canary artifacts**

```bash
cd /home/honglab/leijiaxin/HFOsp
git add scripts/run_ictal_er_rank.py \
        results/data_driven_soz/layer_a_ictal_er_rank/per_subject/yuquan_gaolan.json \
        results/data_driven_soz/layer_a_ictal_er_rank/per_subject/yuquan_huanghanwen.json \
        results/run_logs/yuquan_canary_*.log
git commit -m "$(cat <<'EOF'
feat(topic5 pr0.1 step5): yuquan canary on gaolan + huanghanwen

Add --subject-allowlist CLI flag for targeted runs. Validate the dual-
dataset pipeline end-to-end on 2 yuquan canaries before scaling to the
remaining 7. Both subjects produce v2.3-schema JSONs without exceptions.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Full cohort run (16 → 25)

**Files:**
- Output: `results/data_driven_soz/layer_a_ictal_er_rank/per_subject/yuquan_*.json` (9 new)
- Output: `results/data_driven_soz/layer_a_ictal_er_rank/per_subject/cohort_summary.json` (rebuilt)

- [ ] **Step 1: Run full cohort (skip-existing reuses gaolan + huanghanwen from Task 5)**

```bash
cd /home/honglab/leijiaxin/HFOsp
python scripts/run_ictal_er_rank.py --per-subject --force \
    2>&1 | tee results/run_logs/cohort_yuquan_extension_$(date +%Y%m%d_%H%M).log
```
Expected runtime: ~30-60 min for the 7 remaining yuquan (ictal cohort run on 9 yuquan total ≈ 50 ictal seizures × ~30 sec/seizure × 2 ER configs = ~50 min).

Already-completed epilepsiae 16 are skipped via `_is_current_schema_per_subject_json` unless `--no-skip-existing`.

- [ ] **Step 2: Verify cohort summary updated**

```bash
cd /home/honglab/leijiaxin/HFOsp
python3 -c "
import json
d = json.load(open('results/data_driven_soz/layer_a_ictal_er_rank/per_subject/cohort_summary.json'))
print(f'n_subjects: {d[\"n_subjects\"]}')
print(f'subjects (first 5): {d[\"subjects\"][:5]}')
yuq = [s for s in d['subjects'] if s.startswith('yuquan/')]
epi = [s for s in d['subjects'] if s.startswith('epilepsiae/')]
print(f'  yuquan: {len(yuq)} ({yuq})')
print(f'  epilepsiae: {len(epi)}')
print(f'excluded: {d.get(\"excluded_subjects\", {}).get(\"list\")}')
print(f'2D distribution gamma_ER: {d[\"two_d_distribution\"][\"gamma_ER\"]}')
print(f'2D distribution broad_ER: {d[\"two_d_distribution\"][\"broad_ER\"]}')
"
```
Expected: `n_subjects: 25`, yuquan list = 9 audit_eligible, excluded = empty (or only ineligible yuquan).

- [ ] **Step 3: Commit cohort run**

```bash
cd /home/honglab/leijiaxin/HFOsp
git add results/data_driven_soz/layer_a_ictal_er_rank/per_subject/yuquan_*.json \
        results/data_driven_soz/layer_a_ictal_er_rank/per_subject/cohort_summary.json \
        results/run_logs/cohort_yuquan_extension_*.log
git commit -m "$(cat <<'EOF'
feat(topic5 pr0.1 step6): topic5 cohort 16→25 (yuquan extension)

Cohort run produces v2.3 ER-rank atlas for 9 yuquan audit_eligible
subjects (gaolan, huanghanwen, litengsheng, pengzihang, sunyuanxin,
xuxinyi, zhangjinhan, zhangkexuan, zhaojinrui). cohort_summary.json
n_subjects = 25 (15 epi audit_eligible + 9 yuquan + sentinel-only 916).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: z-ER subtype rerun on extended cohort

**Files:**
- Output: `results/data_driven_soz/layer_a_ictal_er_rank/seizure_clusters/per_subject/yuquan_*__zer_binned.json` (9 new)
- Output: `results/data_driven_soz/layer_a_ictal_er_rank/seizure_clusters/cohort_summary__zer_binned.csv` (rebuilt with 50 rows = 25 subj × 2 bands)
- Output: `results/data_driven_soz/layer_a_ictal_er_rank/seizure_clusters/figures/per_subject/yuquan_*.png` (9 new)

- [ ] **Step 1: Run z-ER per-subject on yuquan only**

```bash
cd /home/honglab/leijiaxin/HFOsp
for sid in gaolan huanghanwen litengsheng pengzihang sunyuanxin xuxinyi zhangjinhan zhangkexuan zhaojinrui; do
    python scripts/cluster_ictal_seizures.py per-subject yuquan/${sid} \
        --feature zer_binned 2>&1 | tee -a results/run_logs/zer_yuquan_extension_$(date +%Y%m%d_%H%M).log
done
```
Expected: 9 new `yuquan_*__zer_binned.json` files; `n_eff` reflects per-subject seizure counts (4, 2, 8, 8, 8, 3, 8, 6, 5) minus any baseline_invalid.

- [ ] **Step 2: Rebuild cohort summary CSV**

```bash
cd /home/honglab/leijiaxin/HFOsp
python scripts/cluster_ictal_seizures.py cohort --feature zer_binned 2>&1 | tee -a results/run_logs/zer_yuquan_extension_*.log
```
Expected: `cohort_summary__zer_binned.csv` now has 50 rows (25 subjects × 2 bands), up from 32.

- [ ] **Step 3: Render per-subject 4-panel figures for the 9 new yuquan**

```bash
cd /home/honglab/leijiaxin/HFOsp
for sid in gaolan huanghanwen litengsheng pengzihang sunyuanxin xuxinyi zhangjinhan zhangkexuan zhaojinrui; do
    python scripts/cluster_ictal_seizures.py render yuquan/${sid} --feature zer_binned
done
```

- [ ] **Step 4: Inspect cohort z-ER medians (compare against pre-audit cohort=16)**

```bash
cd /home/honglab/leijiaxin/HFOsp
python3 -c "
import csv
import statistics as st
rows = []
with open('results/data_driven_soz/layer_a_ictal_er_rank/seizure_clusters/cohort_summary__zer_binned.csv') as f:
    rows = list(csv.DictReader(f))
print(f'total rows: {len(rows)}')
yuq = [r for r in rows if r['subject'].startswith('yuquan/')]
epi = [r for r in rows if r['subject'].startswith('epilepsiae/')]
print(f'  yuquan rows: {len(yuq)}, epi rows: {len(epi)}')
ok = [r for r in rows if r.get('status') == 'ok']
print(f'  ok rows: {len(ok)}')
def med(rows, col):
    vals = [float(r[col]) for r in rows if r.get(col)]
    return st.median(vals) if vals else None
print(f'  silhouette median (all): {med(ok, \"silhouette_k\")}')
print(f'  gap_perm median (all):   {med(ok, \"gap_perm_k\")}')
print(f'  silhouette median (yuq): {med([r for r in ok if r in yuq], \"silhouette_k\")}')
print(f'  gap_perm median (yuq):   {med([r for r in ok if r in yuq], \"gap_perm_k\")}')
"
```
Sanity: pre-audit cohort=16 had silhouette median ≈ 0.418, gap_perm median ≈ 0.327 per topic5 doc §3.2. Yuquan additions should not collapse these (huanghanwen with n_eff=2 will be `insufficient_n` and excluded; rest should produce comparable distributions).

- [ ] **Step 5: Commit z-ER outputs**

```bash
cd /home/honglab/leijiaxin/HFOsp
git add results/data_driven_soz/layer_a_ictal_er_rank/seizure_clusters/per_subject/yuquan_*__zer_binned.json \
        results/data_driven_soz/layer_a_ictal_er_rank/seizure_clusters/cohort_summary__zer_binned.csv \
        results/data_driven_soz/layer_a_ictal_er_rank/seizure_clusters/figures/per_subject/yuquan_*.png \
        results/run_logs/zer_yuquan_extension_*.log
git commit -m "$(cat <<'EOF'
feat(topic5 pr0.1 step7): z-ER subtyping on extended cohort (n=25)

Rerun cluster_ictal_seizures per-subject + cohort + render on the 9
yuquan additions. cohort_summary__zer_binned.csv expanded from 32 to
50 rows (25 subjects × gamma/broad). Per-subject 4-panel figures for
9 yuquan landed under figures/per_subject/.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Documentation updates

**Files:**
- Modify: `docs/topic5_seizure_subtyping.md` — §1 / §2 / §3.2 / §4
- Modify: `docs/archive/topic5/pr1_seizure_clustering/pr1_zer_cohort_2026-05-10.md` — add §6
- Modify: `src/ictal_onset_extraction.py:253-254` — fix the now-stale comment

- [ ] **Step 1: Fix the stale comment in `src/ictal_onset_extraction.py`**

Lines 252-254 currently read:
```python
        ``"<dataset>/<id>"``; only the Epilepsiae dataset is supported in
        Step 2 (Yuquan ictal data lacks per-block onset annotation suitable
        for this pipeline).
```

Replace with:
```python
        ``"<dataset>/<id>"``; supports {"epilepsiae", "yuquan"} (yuquan
        added in topic5 PR-0.1 2026-05-10). Other datasets raise
        NotImplementedError.
```

- [ ] **Step 2: Update topic5 main doc cohort numbers**

In `docs/topic5_seizure_subtyping.md`, replace §2 second bullet that contains `每 subject 16 个 epilepsiae 的 cohort` with the cohort=25 statement; remove the §4 caveat 5 ("Yuquan 缺席 …") and replace with a new caveat noting which 12 yuquan are still excluded due to `n_seizures<2`.

Concrete edits:

| Where | From | To |
|---|---|---|
| §2 bullet 1 | `每 subject 16 个 epilepsiae 的 cohort` | `cohort = 25 (15 epilepsiae audit_eligible + 9 yuquan audit_eligible + sentinel-only epilepsiae/916)` |
| §3.1 last paragraph | `cohort：16 epilepsiae subjects（yuquan 未建）` | `cohort：25 subjects (15 epilepsiae + 9 yuquan + sentinel-only 916)` |
| §3.2 cohort table caption | `Cohort 数值（pre-audit 快照, audit-rerun 后微调待回填）` | append: `(post yuquan-extension 2026-05-10; n=25 / ~50 subject-band rows)` |
| §4 caveat 5 | `Yuquan 缺席：当前 cohort 16 个全 epilepsiae，因为 yuquan v2.3 atlas 还没建。下一轮 cohort（含 yuquan）落地后回来更新 §2 / §3.2。` | `Yuquan 部分覆盖：cohort=25 含 9 yuquan audit_eligible (gaolan, huanghanwen, litengsheng, pengzihang, sunyuanxin, xuxinyi, zhangjinhan, zhangkexuan, zhaojinrui)。仍有 12 yuquan 因 n_seizures<2 被排除（chengshuai, dongyiming, hanyuxuan, huangwanling, liyouran, songzishuo, wangyiyang, zhangjiaqi, zhaochenxi, zhourongxuan, chenziyang, zhangbichen），ictal pool 不足以做 within-subject 聚类，无法补救。` |
| §5 历史文档索引 | append a new line | `- docs/superpowers/plans/2026-05-10-topic5-pr0_1-yuquan-ictal-cohort-extension.md — yuquan cohort 扩展 plan` |

- [ ] **Step 3: Add cohort extension subsection to PR-1 archive doc**

Append to `docs/archive/topic5/pr1_seizure_clustering/pr1_zer_cohort_2026-05-10.md`:

```markdown
## 6. 2026-05-10 Yuquan Cohort Extension (PR-0.1)

Cohort grew from 16 → 25 by adding 9 yuquan audit_eligible subjects.
Engineering changes (4 commits):

1. `scripts/build_yuquan_block_inventory.py` + `results/dataset_inventory/yuquan_block_inventory.csv` (new) — EDF probe → block inventory
2. `src/yuquan_dataset.load_yuquan_record` — yuquan SEEG loader with intracranial filter (regex `^[A-Z]{1,3}\d{1,2}$`) + CAR/bipolar
3. `src/ictal_onset_extraction.extract_seizure_window` — dual-dataset routing (replaces 2026-04 NotImplementedError)
4. `scripts/run_ictal_er_rank.py` — `SUPPORTED_DATASETS = {epilepsiae, yuquan}`, branched `_focus_rel_path` / `_count_seizures`

Yuquan SOZ JSON (`results/yuquan_soz_core_channels.json`) is normalized
into the 3-tier `{i, l, e}` surface that `classify_clinical_concordance`
expects (l/e empty for yuquan because yuquan has no l/e annotations).

stable_k=2 was deliberately NOT used as a cohort gate. zhangjinhan
(stable_k=6) and zhaojinrui (stable_k=5) are kept because higher
interictal stable_k is a positive signal for within-subject seizure
heterogeneity — exactly what z-ER subtyping tests.
```

- [ ] **Step 4: Verify edits compile (markdown lint optional)**

```bash
cd /home/honglab/leijiaxin/HFOsp
# Sanity: no broken file refs
grep -n "16 epilepsiae\|cohort 16\|yuquan 未建" docs/topic5_seizure_subtyping.md || echo "no stale references"
```
Expected: `no stale references`.

- [ ] **Step 5: Commit doc updates**

```bash
cd /home/honglab/leijiaxin/HFOsp
git add docs/topic5_seizure_subtyping.md \
        docs/archive/topic5/pr1_seizure_clustering/pr1_zer_cohort_2026-05-10.md \
        src/ictal_onset_extraction.py
git commit -m "$(cat <<'EOF'
docs(topic5 pr0.1 step8): cohort 16→25 + close yuquan-absent caveat

Update topic5 main doc §2 / §3.1 / §3.2 / §4 cohort numbers and replace
caveat 5 (yuquan absent) with the 12 yuquan-still-ineligible list.
Append §6 to archive doc with engineering log + stable_k=2 gating
rationale. Fix the now-stale extract_seizure_window docstring at
src/ictal_onset_extraction.py:253-254.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Run full test suite + final regression check

**Files:** none (test invocation only)

- [ ] **Step 1: Run topic5 + ictal-related tests**

```bash
cd /home/honglab/leijiaxin/HFOsp
pytest tests/test_ictal_onset_extraction.py \
       tests/test_ictal_onset_extraction_yuquan.py \
       tests/test_yuquan_dataset.py \
       tests/test_build_yuquan_block_inventory.py \
       tests/test_ictal_seizure_clustering.py \
       tests/test_ictal_zer_features.py \
       tests/test_ictal_seizure_plotting.py \
       -v
```
Expected: All pass.

- [ ] **Step 2: Run AGENTS.md cross-PR contract regression**

Verify that no other downstream consumer relies on `SUPPORTED_DATASETS = frozenset({"epilepsiae"})` to silently exclude yuquan:

```bash
cd /home/honglab/leijiaxin/HFOsp
grep -rn 'SUPPORTED_DATASETS\|epilepsiae_seizure_inventory\|extract_seizure_window' \
    src scripts tests | grep -v __pycache__
```
Expected: only the files we modified should reference these. No other consumer hard-codes epilepsiae.

- [ ] **Step 3: Final cohort summary printout**

```bash
cd /home/honglab/leijiaxin/HFOsp
python3 -c "
import json
zer = open('results/data_driven_soz/layer_a_ictal_er_rank/seizure_clusters/cohort_summary__zer_binned.csv').read()
n_rows = zer.count('\\n') - 1
print(f'z-ER cohort_summary rows: {n_rows} (expected 50 = 25 subj × 2 bands)')
atlas = json.load(open('results/data_driven_soz/layer_a_ictal_er_rank/per_subject/cohort_summary.json'))
print(f'atlas n_subjects: {atlas[\"n_subjects\"]} (expected 25)')
print(f'atlas excluded: {atlas[\"excluded_subjects\"][\"list\"]}')
"
```

- [ ] **Step 4: Mark plan complete**

The plan is complete when:
1. All 9 yuquan audit_eligible subjects have a `yuquan_<sid>__zer_binned.json` file
2. `cohort_summary__zer_binned.csv` has 50 rows
3. `cohort_summary.json` has `n_subjects = 25` and empty `excluded_subjects.list`
4. Topic5 main doc reflects cohort=25
5. All 4 plan-modified test files pass

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Implementing task |
|---|---|
| Build yuquan_block_inventory.csv | Task 1 |
| Augment yuquan_seizure_inventory.csv with record_start/end_epoch | Task 1 (joined output) |
| Replace NotImplementedError at src/ictal_onset_extraction.py:271-274 | Task 3 step 4 |
| Yuquan EDF loader with CAR/bipolar | Task 2 |
| Canary on gaolan + huanghanwen | Task 5 |
| Full cohort run + atlas backfill | Task 6 |
| z-ER rerun on extended cohort | Task 7 |
| Don't gate by stable_k=2 (keep zhangjinhan stable_k=6, zhaojinrui stable_k=5) | Task 4 (cohort selector includes them); Task 8 archive doc §6 paragraph |

**Placeholder scan:** No "TBD", "implement later", or "similar to Task N" patterns. Every code step shows the actual code.

**Type consistency check:**
- `BlockProbeResult` fields used consistently in Task 1 across `probe_one_edf`, `write_block_inventory_csv`, and joins in `rebuild_seizure_inventory_with_record_epochs`.
- `PreprocessingResult` (data, sfreq, ch_names) honored by `load_yuquan_record` (Task 2) and consumed unchanged by `extract_seizure_window` (Task 3).
- `_resolve_inventory_paths(results_root, dataset)` signature change is reflected at the single call site (line 277, also rewritten in Task 3 step 4).
- `_focal_channels(subject, focus_rel)` signature unchanged (Task 4 only changes the `focus_rel` source per dataset, not the function signature).
- `SUPPORTED_DATASETS = frozenset({"epilepsiae", "yuquan"})` is consistent with `extract_seizure_window`'s allowlist `{"epilepsiae", "yuquan"}`.

**Risk register (out-of-band, for the executor's awareness):**
- yuquan EDF channel naming may have subject-specific quirks not captured by the regex `^[A-Z]{1,3}\d{1,2}$`. Canary (Task 5) is the gate for this — if huanghanwen (n_channels=85) drops too many channels, audit `classify_yuquan_channel` against `huanghanwen/<edf>.ch_names` before scaling.
- yuquan `eeg_onset_epoch` is the only annotation; treated as both `clin_onset_epoch` and `eeg_onset_epoch` in `extract_seizure_window`. This means `resolve_baseline_window` will see `eeg_onset_rel_sec=None` (no separate clip), defaulting to the legacy `[-300, -60]` baseline. That's the intended behavior — yuquan has no separate electrographic-onset annotation to clip against.
- 2 yuquan subjects (gaolan, zhangjinhan, zhaojinrui per topic5 cross-check) are NOT in PR-6 anchoring pass; they CAN appear in topic5 because topic5's contract is independent of PR-6.

**Dependencies:**
- `mne` (already used elsewhere in the repo for yuquan EDF loading)
- `mne.export.export_raw` (used in tests; ships with mne ≥1.0)
- All other deps (numpy, scipy, pytest) already required.
