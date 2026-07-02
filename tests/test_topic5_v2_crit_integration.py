import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
def test_phase2_depcheck_writes_fail_closed_report(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_topic5_v2_crit_depcheck.py",
            "--substrate",
            "broad",
            "--outdir",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    report_path = tmp_path / "phase2_dependency_report.json"
    assert report_path.exists(), result.stderr

    report = json.loads(report_path.read_text())
    assert report["fail_closed"] is True
    assert report["substrate"] == "broad"
    deps = report["dependencies"]
    expected = {
        "phase1_config",
        "phase1_band_scan_source",
        "phase1_band_scan_import",
        "phase1_band_cache",
        "phase1_cohort_manifest",
        "phase1_order_null_depcheck_csv",
        "ictal_field_load_context",
    }
    assert expected <= set(deps)
    for name in expected:
        assert {"present", "path", "status"} <= set(deps[name])

    if report["status"] == "ok":
        assert result.returncode == 0, result.stderr
        assert all(deps[name]["status"] == "ok" for name in expected)
    else:
        assert result.returncode != 0
        assert any(deps[name]["status"] != "ok" for name in expected)
