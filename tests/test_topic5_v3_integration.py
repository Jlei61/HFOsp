import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import wilcoxon

from scripts.run_topic5_v3_summary import _holm_correct_2, _tier_verdict, _wilcoxon_greater

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


# Task 10 summary — joins the 3 co-primary/context CSVs and assigns `tier`
# (Task 10 ONLY; no earlier task computes tier). The plan/brief describe the
# required FIELDS, not a literal ordered CSV_COLS list like Tasks 6/8/9 --
# this set IS the schema Task 10 introduces for `v3_summary_subject.csv`.
SUMMARY_COLS = {
    "subject", "cohort", "geometry_insufficient",
    "h3b_path", "h3c_path", "subject_support", "support_driver",
    "common_drive_downgrade", "h3a_strengthens",
    "delta_net_offaxis_flux_surplus", "delta_mode_shift_density",
}


@pytest.mark.integration
def test_summary_joins_real_dev_csvs_and_assigns_tier(tmp_path):
    """Task 10: join narrow's (+ broad's) real dev CSVs, verify the
    subject_support gate stack and the JSON tier-verdict contract.

    n_perm-independent (reads already-computed flags/deltas/p-values from
    Tasks 6/8/9's dev CSVs on disk), so this is a structural/join-logic
    check against real data, NOT a re-verification of those tasks' own
    permutation math.
    """
    narrow_dir = tmp_path / "narrow_out"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_topic5_v3_summary.py",
            "--cohort", "narrow",
            "--outdir", str(narrow_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    csv_path = narrow_dir / "v3_summary_subject.csv"
    json_path = narrow_dir / "v3_cohort_tier.json"
    assert csv_path.exists() and json_path.exists(), result.stderr

    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 7, rows  # narrow's full cohort -- never silently dropped
    assert SUMMARY_COLS <= set(rows[0]), f"missing cols: {SUMMARY_COLS - set(rows[0])}"

    payload = json.loads(json_path.read_text())
    assert payload["tier"] in (0, 1, 2, 3, 4), payload
    assert isinstance(payload["state_v3_supported"], bool), payload
    assert "narrow" in payload and "broad" in payload, payload  # never pooled
    for cohort_key in ("narrow", "broad"):
        block = payload[cohort_key]
        for pkey in ("p_h3b", "p_h3c", "p_holm_h3b", "p_holm_h3c"):
            assert pkey in block, (cohort_key, block)
    assert payload["narrow"]["n_geometry_sufficient"] == 7, payload["narrow"]

    # CSV and JSON must agree on how many subjects actually support (no
    # drift between the per-subject artifact and the cohort-level count).
    n_support_csv = sum(1 for r in rows if r["subject_support"] == "True")
    assert n_support_csv == payload["narrow"]["n_subject_support"], (rows, payload["narrow"])

    # Currently-known real fact (Task 6/8 dev, n_perm=100): epilepsiae_1096
    # is narrow's only H3b module_support_flag=True subject, but it fails
    # BOTH onset_jitter_pass and (not common_drive_sensitive) -- exactly the
    # "downgrade" this task must surface, not silently drop. epilepsiae_1125
    # is narrow's only H3c module_support_flag=True subject and clears every
    # H3c_path gate. Net: narrow subject-level support = 1, driven by H3c.
    by_subject = {r["subject"]: r for r in rows}
    assert by_subject["epilepsiae_1096"]["common_drive_downgrade"] == "True", by_subject["epilepsiae_1096"]
    assert by_subject["epilepsiae_1096"]["subject_support"] == "False", by_subject["epilepsiae_1096"]
    assert by_subject["epilepsiae_1125"]["h3c_path"] == "True", by_subject["epilepsiae_1125"]
    assert by_subject["epilepsiae_1125"]["subject_support"] == "True", by_subject["epilepsiae_1125"]
    assert payload["narrow"]["n_subject_support"] == 1, payload["narrow"]

    # broad cohort: same script, own subject count (9), never pooled with narrow.
    broad_dir = tmp_path / "broad_out"
    result_b = subprocess.run(
        [
            sys.executable,
            "scripts/run_topic5_v3_summary.py",
            "--cohort", "broad",
            "--outdir", str(broad_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result_b.returncode == 0, result_b.stderr
    rows_b = list(csv.DictReader((broad_dir / "v3_summary_subject.csv").open()))
    assert len(rows_b) == 9, rows_b
    assert SUMMARY_COLS <= set(rows_b[0]), f"missing cols: {SUMMARY_COLS - set(rows_b[0])}"


# ---------------------------------------------------------------------------
# Task 10 pure-function unit tests (no data/IO, NOT @pytest.mark.integration):
# the 3 helpers that decide the study's tier verdict. The overall
# subject_support/tier/state_v3_supported LOGIC is verified correct (review);
# these pin the exact arithmetic/boundary behavior so a future edit can't
# silently drift it.
# ---------------------------------------------------------------------------
def test_holm_correct_2():
    # Concrete case: the smaller p is x2, the larger is x1.
    assert _holm_correct_2(0.01, 0.04) == pytest.approx((0.02, 0.04))
    assert _holm_correct_2(0.04, 0.01) == pytest.approx((0.04, 0.02))

    # Tie: naive per-slot multipliers would give (0.06, 0.03) -- a reversal
    # (the higher-rank slot ending up SMALLER). The running-max step-down
    # must enforce non-decreasing order, clamping both to (0.06, 0.06).
    assert _holm_correct_2(0.03, 0.03) == pytest.approx((0.06, 0.06))

    # NaN passthrough: m=2 is fixed by pre-registration, not by how many of
    # the 2 raw p-values happen to be finite this run -- the finite slot is
    # still treated as "the smaller of 2" (x2, not x1), and the NaN slot
    # stays NaN without acquiring a fabricated multiplier of its own.
    h1, h2 = _holm_correct_2(0.02, float("nan"))
    assert h1 == pytest.approx(0.04)
    assert math.isnan(h2)
    h1, h2 = _holm_correct_2(float("nan"), 0.02)
    assert math.isnan(h1)
    assert h2 == pytest.approx(0.04)


def test_wilcoxon_greater():
    positive = np.array([0.10, 0.20, 0.30, 0.15, 0.25, 0.05, 0.40, 0.35, 0.50, 0.45])
    negative = -positive

    p_pos = _wilcoxon_greater(positive)
    p_neg = _wilcoxon_greater(negative)

    assert p_pos < 0.01, p_pos  # clearly-positive delta -> small p
    assert p_neg > 0.5, p_neg  # clearly-negative delta -> large (wrong-direction) p

    # Confirm this really is alternative="greater" (not two-sided/"less"):
    # a direct scipy call with the same alternative must match exactly.
    _, p_pos_direct = wilcoxon(positive, alternative="greater")
    assert p_pos == pytest.approx(p_pos_direct)


def _verdict_block(h3b_pass=False, h3c_pass=False, n_support=0, med_h3b=0.0, med_h3c=0.0):
    """Minimal cohort-block dict carrying only the keys `_tier_verdict` reads."""
    return {
        "cohort_h3b_pass": h3b_pass,
        "cohort_h3c_pass": h3c_pass,
        "n_subject_support": n_support,
        "median_delta_h3b": med_h3b,
        "median_delta_h3c": med_h3c,
    }


def test_tier_verdict_narrow_primary():
    # (a) narrow cohort-level pass, broad has nothing -> tier 3.
    v = _tier_verdict(_verdict_block(h3b_pass=True, med_h3b=0.10), _verdict_block())
    assert v["tier"] == 3
    assert v["state_v3_supported"] is True
    assert v["broad_replicates"] is False

    # (b) narrow passes H3c AND broad passes the SAME endpoint (H3c), same
    # direction -> broad replication promotes narrow's tier to 4.
    v = _tier_verdict(
        _verdict_block(h3c_pass=True, med_h3c=0.05),
        _verdict_block(h3c_pass=True, med_h3c=0.03),
    )
    assert v["tier"] == 4
    assert v["state_v3_supported"] is True
    assert v["broad_replicates"] is True

    # (c) narrow passes H3b, broad passes the DIFFERENT endpoint (H3c) ->
    # no promotion, stays tier 3.
    v = _tier_verdict(
        _verdict_block(h3b_pass=True, med_h3b=0.10),
        _verdict_block(h3c_pass=True, med_h3c=0.03),
    )
    assert v["tier"] == 3
    assert v["broad_replicates"] is False

    # (d) broad passes fully but narrow does NOT cohort-pass -> broad can
    # never lead; tier is capped at whatever narrow's OWN (lower) evidence
    # gives (here 2, from narrow's 1 subject-level support), not promoted
    # by broad's full pass.
    v = _tier_verdict(
        _verdict_block(n_support=1, med_h3b=0.02),
        _verdict_block(h3b_pass=True, med_h3b=0.20),
    )
    assert v["tier"] == 2
    assert v["broad_cohort_pass"] is True  # recorded...
    assert v["narrow_cohort_pass"] is False  # ...but never promotes narrow

    # (e) >=1 narrow subject-level support, no narrow cohort pass -> tier 2.
    v = _tier_verdict(
        _verdict_block(n_support=1, med_h3b=0.02, med_h3c=-0.01),
        _verdict_block(),
    )
    assert v["tier"] == 2
    assert v["state_v3_supported"] is False

    # (f) narrow cohort direction correct (median delta > 0) but not
    # significant (no cohort_*_pass) and no subject-level support -> tier 1.
    v = _tier_verdict(
        _verdict_block(n_support=0, med_h3b=0.05, med_h3c=-0.02),
        _verdict_block(),
    )
    assert v["tier"] == 1
    assert v["state_v3_supported"] is False
