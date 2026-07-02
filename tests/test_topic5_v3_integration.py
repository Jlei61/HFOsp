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


# Full Task-8 dynamics CSV column contract (plan Task 8; NO `tier` — Task 10 only).
DYNAMICS_COLS = {
    "subject", "cohort", "status", "skip_reason", "geometry_sufficient",
    "dynamics_primary_model", "dynamics_support_model", "rank_used", "k_star",
    "mode_shift_density_P3", "mode_shift_density_I1", "delta_mode_shift_density",
    "mode_shift_raw_delta", "mode_shift_2D_consistency",
    "p_phase", "p_block", "p_label", "mode_shift_label_z",
    "lambda_surplus_P3", "lambda_surplus_I1", "gain_axis_delta", "gain_nonaxis_delta",
    "reactivity_cont_available", "logm_quality_flag",
    "top_contact_energy_fraction", "single_contact_driven",
    "leave_one_contact_mode_shift_pass", "axis_only_mode_shift_control", "axis_only_control_pass",
    "onset_jitter_pass", "cv_r2", "var_meaningful_flag", "n_ch_fit", "n_seizures",
    "module_support_flag", "module_direction_correct", "module_null_pass",
}


@pytest.mark.integration
def test_dynamics_writes_csv_even_if_skipped(tmp_path):
    # A subject whose context/cache cannot be loaded still gets a full row
    # (never silently drop); the header must carry every contract column.
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_topic5_v3_dynamics.py",
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

    csv_path = tmp_path / "v3_dynamics_subject.csv"
    assert csv_path.exists(), result.stderr
    reader = csv.DictReader(csv_path.open())
    assert DYNAMICS_COLS <= set(reader.fieldnames), (
        f"missing cols: {DYNAMICS_COLS - set(reader.fieldnames)}"
    )
    rows = list(reader)
    assert rows, "expected >=1 row"
    assert rows[0]["status"] == "skipped", rows[0]


@pytest.mark.integration
def test_dynamics_runs_on_eligible_subject(tmp_path):
    # Task-3 auto-selected integration subject (narrow: 30 clean / 8 axis /
    # 22 non-axis / 6 i1-eligible seizures) with a small n-perm smoke.
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_topic5_v3_dynamics.py",
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

    csv_path = tmp_path / "v3_dynamics_subject.csv"
    rows = {r["subject"]: r for r in csv.DictReader(csv_path.open())}
    assert "epilepsiae_253" in rows, rows
    row = rows["epilepsiae_253"]
    assert row["status"] == "ok", row
    for col in ("delta_mode_shift_density", "p_phase", "p_label", "mode_shift_2D_consistency"):
        assert math.isfinite(float(row[col])), (col, row)


# Full Task-9 susceptibility CSV column contract (plan Task 9; NO `tier`, and
# NO geometry_sufficient/n_axis/n_nonaxis/n_ambiguous -- unlike Tasks 6/8,
# those columns are not part of this task's contract).
SUSCEPTIBILITY_COLS = {
    "subject", "cohort", "status", "skip_reason", "K_primary_metric",
    "beta_axis_P3", "beta_axis_I1", "beta_axis_P3_reliable",
    "delta_beta_axis_strength", "beta_axis_delta_null_z",
    "p_spatial_delta", "p_label_delta", "onset_jitter_pass", "n_seizures",
    "module_support_flag", "module_direction_correct", "module_null_pass",
}


@pytest.mark.integration
def test_susceptibility_writes_csv_even_if_skipped(tmp_path):
    # A subject whose context/cache cannot be loaded still gets a full row
    # (never silently drop a subject); the header must carry every contract
    # column. A bogus subject id exercises the skip path fast (no null loop).
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_topic5_v3_susceptibility.py",
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

    csv_path = tmp_path / "v3_susceptibility_subject.csv"
    assert csv_path.exists(), result.stderr
    reader = csv.DictReader(csv_path.open())
    assert SUSCEPTIBILITY_COLS <= set(reader.fieldnames), (
        f"missing cols: {SUSCEPTIBILITY_COLS - set(reader.fieldnames)}"
    )
    rows = list(reader)
    assert rows, "expected >=1 row"
    assert rows[0]["status"] == "skipped", rows[0]


@pytest.mark.integration
def test_susceptibility_runs_on_eligible_subject(tmp_path):
    # Task-3 auto-selected integration subject (narrow: 30 clean / 8 axis /
    # 22 non-axis / 6 i1-eligible seizures) with a small n-perm smoke.
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_topic5_v3_susceptibility.py",
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

    csv_path = tmp_path / "v3_susceptibility_subject.csv"
    rows = {r["subject"]: r for r in csv.DictReader(csv_path.open())}
    assert "epilepsiae_253" in rows, rows
    row = rows["epilepsiae_253"]
    assert row["status"] == "ok", row
    assert math.isfinite(float(row["delta_beta_axis_strength"])), row
    # H3a is SUPPORTIVE-ONLY: module_support_flag must NEVER be True, even
    # for an eligible ok-status subject with a finite, direction-correct delta.
    assert row["module_support_flag"] == "False", row
