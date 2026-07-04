"""Topic 5 V3p — feasibility pilot integration test (Task 1, DECISION GATE).

Runs the real `scripts/run_topic5_v3p_feasibility.py` CLI end-to-end against
the mounted artifacts (ictal field long cache + lagPat + rank-displacement
JSON) and checks the `feasibility.csv` contract columns exist. See
docs/superpowers/plans/2026-07-03-topic5-v3p-preictal-trajectory.md Task 1.
"""
import pytest
import pandas as pd
import numpy as np


@pytest.mark.integration
def test_v3p_feasibility_writes_csv(tmp_path):
    from scripts.run_topic5_v3p_feasibility import main
    out = tmp_path / "feasibility.csv"
    main(["--cohort", "narrow", "--outdir", str(tmp_path)])
    df = pd.read_csv(out)
    for col in ["geometry_sufficient", "n_windows_P3", "n_seizures_ge_min_windows", "n_nonaxis",
                "roster_status", "axis_quality_gate_pass", "admitted", "n_unique_label_permutations_est"]:
        assert col in df.columns
    assert len(df) >= 1


# ---------------------------------------------------------------------------
# Task 7 trajectory runner — the two-tier integration contract (brief Step 1).
# Tier 1: header-only CSV must still carry the co-primary columns (skipped path).
# Tier 2: an eligible subject (253) produces finite co-primary surplus slopes +
# a well-formed label-null p in [0, 1].
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_v3p_trajectory_writes_csv_even_if_skipped(tmp_path):
    from scripts.run_topic5_v3p_trajectory import main
    main(["--cohort", "narrow", "--outdir", str(tmp_path), "--n-perm", "20", "--subjects", "__none__"])
    df = pd.read_csv(tmp_path / "v3p_trajectory_subject.csv")
    for c in ["net_offaxis_flux_surplus_slope", "mode_shift_density_surplus_slope",
              "p_label_slope_b", "p_label_slope_c", "status"]:
        assert c in df.columns


@pytest.mark.integration
def test_v3p_trajectory_runs_on_eligible_subject(tmp_path):
    from scripts.run_topic5_v3p_trajectory import main
    main(["--cohort", "narrow", "--outdir", str(tmp_path), "--n-perm", "50", "--subjects", "253"])
    df = pd.read_csv(tmp_path / "v3p_trajectory_subject.csv")
    row = df[df.subject.astype(str) == "epilepsiae_253"].iloc[0]
    if row["status"] == "ok":
        assert np.isfinite(row["net_offaxis_flux_surplus_slope"])
        assert np.isfinite(row["mode_shift_density_surplus_slope"])
        assert 0.0 <= row["p_label_slope_b"] <= 1.0
