import csv
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Full Task-6 avalanche CSV column contract (plan Task 6; NO `tier` — Task 10 only).
AVALANCHE_COLS = {
    "subject", "cohort", "status", "skip_reason", "geometry_sufficient",
    "n_axis", "n_nonaxis", "n_ambiguous",
    "net_offaxis_flux_P3", "net_offaxis_flux_I1", "delta_net_offaxis_flux_raw",
    "delta_net_offaxis_flux_surplus", "net_offaxis_flux_z",
    "p_rate_delta", "p_spatial_delta", "p_label_delta",
    "lag1_specific_delta", "common_drive_sensitive",
    "max_source_contact_contribution", "leave_one_contact_min_delta",
    "leave_one_contact_pass", "axis_only_flux_delta", "axis_only_control_pass",
    "onset_jitter_pass", "n_seizures",
    "module_support_flag", "module_direction_correct", "module_null_pass",
}


@pytest.mark.integration
def test_feasibility_writes_gate_columns(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_topic5_v3_feasibility.py",
            "--cohort",
            "narrow",
            "--outdir",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    csv_path = tmp_path / "feasibility.csv"
    assert csv_path.exists(), result.stderr
    rows = list(csv.DictReader(csv_path.open()))
    assert rows, "expected >=1 row in feasibility.csv"
    for col in ("geometry_sufficient", "i1_eligible", "n_ambiguous"):
        assert col in rows[0], f"missing column {col}"


@pytest.mark.integration
def test_avalanche_writes_csv_even_if_skipped(tmp_path):
    # A subject whose context/cache cannot be loaded still gets a full row
    # (never silently drop a subject); the header must carry every contract
    # column. A bogus subject id exercises the skip path fast (no null loop).
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_topic5_v3_avalanche.py",
            "--cohort", "narrow",
            "--subjects", "epilepsiae_000000",
            "--n-perm", "20",
            "--outdir", str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    csv_path = tmp_path / "v3_avalanche_subject.csv"
    assert csv_path.exists(), result.stderr
    reader = csv.DictReader(csv_path.open())
    assert AVALANCHE_COLS <= set(reader.fieldnames), (
        f"missing cols: {AVALANCHE_COLS - set(reader.fieldnames)}"
    )
    rows = list(reader)
    assert rows, "expected >=1 row"
    assert rows[0]["status"] == "skipped", rows[0]


@pytest.mark.integration
def test_avalanche_runs_on_eligible_subject(tmp_path):
    # Task-3 auto-selected integration subject (narrow: 30 clean / 8 axis /
    # 22 non-axis / 6 i1-eligible seizures) with a small n-perm smoke.
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_topic5_v3_avalanche.py",
            "--cohort", "narrow",
            "--subjects", "epilepsiae_253",
            "--n-perm", "20",
            "--outdir", str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    csv_path = tmp_path / "v3_avalanche_subject.csv"
    rows = {r["subject"]: r for r in csv.DictReader(csv_path.open())}
    assert "epilepsiae_253" in rows, rows
    row = rows["epilepsiae_253"]
    assert row["status"] == "ok", row
    assert math.isfinite(float(row["delta_net_offaxis_flux_surplus"])), row
