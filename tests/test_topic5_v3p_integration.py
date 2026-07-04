"""Topic 5 V3p — feasibility pilot integration test (Task 1, DECISION GATE).

Runs the real `scripts/run_topic5_v3p_feasibility.py` CLI end-to-end against
the mounted artifacts (ictal field long cache + lagPat + rank-displacement
JSON) and checks the `feasibility.csv` contract columns exist. See
docs/superpowers/plans/2026-07-03-topic5-v3p-preictal-trajectory.md Task 1.
"""
import pytest
import pandas as pd


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
