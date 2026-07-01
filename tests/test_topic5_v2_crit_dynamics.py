"""Integration smoke for the Phase-2 dynamics run script."""
import csv
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_COLS = {
    "subject", "axis_set", "status", "skip_reason", "available_pre_sec", "state_band",
    "M_loading_spearman", "cv_r2", "var_meaningful_flag", "lambda_trend_spearman",
    "M_phase_empirical_p", "M_block_empirical_p", "lambda_trend_phase_empirical_p",
    "n_ch_fit", "n_seizures", "n_seizures_total", "align_source", "tier",
}


@pytest.mark.integration
def test_dynamics_smoke_epilepsiae_139(tmp_path):
    r = subprocess.run(
        [sys.executable, "scripts/run_topic5_v2_crit_dynamics.py",
         "--subjects", "epilepsiae_139", "--substrate", "broad",
         "--n-perm", "20", "--outdir", str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    rows = list(csv.DictReader(open(tmp_path / "phase2_dynamics_subject.csv")))
    assert rows and REQUIRED_COLS <= set(rows[0])
    row = next(x for x in rows if x["subject"] == "139")
    assert row["status"] == "ok"
    assert row["state_band"] == "legacy_bb_1_45"
    assert row["tier"] == "exploratory"
    assert float(row["M_loading_spearman"]) == float(row["M_loading_spearman"])  # finite (not NaN str)
    assert row["var_meaningful_flag"] in ("True", "False")
