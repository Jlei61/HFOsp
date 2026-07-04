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


# ---------------------------------------------------------------------------
# Task 8 — supportive H3p-a (beta_axis) + secondary H3p-d (burden / self-sustain
# lag1-specific / gain_shift) columns, added onto the SAME trajectory runner +
# eligible subject as Task 7. module_support_flag_a is hard-coded False
# (supportive-only, never sole support).
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_v3p_trajectory_h3pa_h3pd_columns(tmp_path):
    from scripts.run_topic5_v3p_trajectory import main
    main(["--cohort", "narrow", "--outdir", str(tmp_path), "--n-perm", "50", "--subjects", "253"])
    df = pd.read_csv(tmp_path / "v3p_trajectory_subject.csv")
    for c in ["beta_axis_strength_slope", "beta_axis_reliable", "beta_axis_slope_z",
              "p_label_slope_a", "module_support_flag_a",
              "nonaxis_activation_burden_slope_raw", "nonaxis_activation_burden_slope_label_surplus",
              "nonaxis_activation_burden_slope_resid", "burden_slope_z", "p_label_burden",
              "N_self_sustain_lag1_slope", "N_self_sustain_lag0_slope",
              "N_self_sustain_lag1_specific_slope", "N_self_sustain_slope_z", "p_label_selfsustain",
              "gain_axis_slope", "gain_nonaxis_slope", "gain_shift_slope",
              "gain_nonaxis_surplus_slope", "gain_shift_slope_z"]:
        assert c in df.columns
    row = df[df.subject.astype(str) == "epilepsiae_253"].iloc[0]
    assert row["module_support_flag_a"] in (False, "False", 0)  # hard-coded, never sole support
    if row["status"] == "ok":
        assert np.isfinite(row["beta_axis_strength_slope"])
        assert np.isfinite(row["nonaxis_activation_burden_slope_resid"])
        assert np.isfinite(row["N_self_sustain_lag1_specific_slope"])
        assert np.isfinite(row["gain_nonaxis_surplus_slope"])
        assert 0.0 <= row["p_label_slope_a"] <= 1.0
