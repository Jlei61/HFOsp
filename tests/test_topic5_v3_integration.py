import csv
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


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
