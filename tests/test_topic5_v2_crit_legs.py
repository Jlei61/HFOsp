"""Integration smoke for the Phase-2 susceptibility / avalanche / summary scripts.

While Phase-1 nulls are pending, observed statistics must be finite and null columns
must carry `pending_phase1` (never a fabricated null); state_leg_supported must be False.
"""
import csv
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run(script, outdir, extra):
    r = subprocess.run(
        [sys.executable, f"scripts/{script}", "--substrate", "broad",
         "--subjects", "epilepsiae_139", "--outdir", str(outdir), *extra],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    return r


@pytest.mark.integration
def test_susceptibility_smoke(tmp_path):
    _run("run_topic5_v2_crit_susceptibility.py", tmp_path, ["--n-perm", "20"])
    rows = list(csv.DictReader(open(tmp_path / "phase2_susceptibility_subject.csv")))
    feats = {r["feature"] for r in rows if r["status"] == "ok"}
    assert {"variance", "lag1_autocorr", "line_length_rate"} <= feats
    llr = next(r for r in rows if r["feature"] == "line_length_rate")
    assert llr["status"] == "ok" and llr["K_signed_oriented"] not in ("", "nan")
    assert llr["spatial_null_strength"] == "pending_phase1"  # null not fabricated
    assert llr["order_null_strength"] == "pending_phase1"


@pytest.mark.integration
def test_avalanche_smoke(tmp_path):
    _run("run_topic5_v2_crit_avalanche.py", tmp_path, ["--n-perm", "20"])
    row = list(csv.DictReader(open(tmp_path / "phase2_avalanche_subject.csv")))[0]
    assert row["status"] == "ok"
    # forward displacement is the PRIMARY metric (finite); rank-coupling is descriptive
    assert row["atm_forward_displacement"] not in ("", "nan")
    assert row["atm_spatial_empirical_p"] == ""  # pending Phase-1 null
    assert row["spatial_null_strength"] == "pending_phase1"


@pytest.mark.integration
def test_summary_joins_three_legs(tmp_path):
    _run("run_topic5_v2_crit_susceptibility.py", tmp_path, ["--n-perm", "20"])
    _run("run_topic5_v2_crit_avalanche.py", tmp_path, ["--n-perm", "20"])
    _run("run_topic5_v2_crit_dynamics.py", tmp_path, ["--n-perm", "20"])
    r = subprocess.run(
        [sys.executable, "scripts/run_topic5_v2_crit_summary.py", "--substrate", "broad",
         "--basedir", str(tmp_path), "--outdir", str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    row = list(csv.DictReader(open(tmp_path / "phase2_criticality_summary.csv")))[0]
    for col in ("K_signed_oriented", "M_loading_spearman", "atm_forward_displacement",
                "state_leg_supported", "tier"):
        assert col in row
    assert row["tier"] == "exploratory"
    assert row["state_leg_supported"] == "False"  # never supported while nulls pending
