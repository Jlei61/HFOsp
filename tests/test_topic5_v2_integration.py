# tests/test_topic5_v2_integration.py
import subprocess, sys, csv
import pytest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
@pytest.mark.integration
@pytest.mark.parametrize("axis", ["broad", "narrow"])
def test_legacy_reproduction_within_tolerance(axis, tmp_path):
    r = subprocess.run([sys.executable, "scripts/run_topic5_v2_legacy_repro.py",
                        "--substrate", axis, "--outdir", str(tmp_path)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rows = list(csv.DictReader(open(tmp_path / axis / "phase1_qc_legacy_reproduction.csv")))
    assert rows and all("n_seizures" in x for x in rows)
    for x in rows: assert abs(float(x["delta"])) <= 0.02, f"{x['subject']} {x['band']} {x['delta']}"
