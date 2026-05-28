"""Tests for scripts/run_topic4_phase4_stage1b_calibration.py.

Acceptance logic (evaluate_acceptance) is unit-tested directly with synthetic
per-sigma rows so the pass/fail exit-contract is verified without running full
HR simulations. A slow integration test runs the real CLI end-to-end.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = "scripts/run_topic4_phase4_stage1b_calibration.py"


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    spec = importlib.util.spec_from_file_location(
        "stage1b_cli", repo_root / _SCRIPT
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(sigma: float, env_count: float, spike_count: float = 0.0) -> dict:
    return {
        "sigma": sigma,
        "spike": {"count": spike_count, "mean_duration": 0.0, "mean_ibi": 0.0},
        "envelope": {"count": env_count, "mean_duration": 0.0,
                     "mean_n_spikes_per_burst": 0.0, "mean_ibi": 0.0},
    }


def test_acceptance_pass_when_silent_and_noise_triggered():
    mod = _load_module()
    rows = [_row(0.0, 0.0), _row(0.2, 4.2), _row(0.4, 6.6), _row(0.6, 7.4)]
    acc = mod.evaluate_acceptance(rows)
    assert acc["sigma0_silent"] is True
    assert acc["noise_triggered"] is True
    assert acc["stage1b_pass"] is True


def test_acceptance_fail_when_sigma0_not_silent():
    """sigma=0 with envelope events above tolerance fails the silent gate."""
    mod = _load_module()
    rows = [_row(0.0, 3.0), _row(0.2, 4.0), _row(0.4, 6.0), _row(0.6, 7.0)]
    acc = mod.evaluate_acceptance(rows)
    assert acc["sigma0_silent"] is False
    assert acc["stage1b_pass"] is False


def test_acceptance_fail_when_no_noise_events():
    """A sigma>0 with zero envelopes fails the noise-triggered requirement."""
    mod = _load_module()
    rows = [_row(0.0, 0.0), _row(0.2, 0.0), _row(0.4, 6.0), _row(0.6, 7.0)]
    acc = mod.evaluate_acceptance(rows)
    assert acc["sigma0_silent"] is True
    assert acc["noise_triggered"] is False
    assert acc["stage1b_pass"] is False


def test_cli_help_works():
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, _SCRIPT, "--help"],
        cwd=str(repo_root),
        env={"PYTHONPATH": str(repo_root), **dict(__import__("os").environ)},
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0
    assert "--output-dir" in proc.stdout


@pytest.mark.slow
def test_cli_end_to_end_passes_and_writes_outputs(tmp_path):
    """Full CLI run at baseline exits 0 and writes comparison.json + figure."""
    repo_root = Path(__file__).resolve().parents[1]
    out = tmp_path / "stage1b"
    proc = subprocess.run(
        [sys.executable, _SCRIPT, "--output-dir", str(out)],
        cwd=str(repo_root),
        env={"PYTHONPATH": str(repo_root), **dict(__import__("os").environ)},
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    comparison = json.loads((out / "comparison.json").read_text())
    assert comparison["acceptance"]["stage1b_pass"] is True
    assert comparison["envelope_gap"] == 30.0
    # sigma=0 silent; envelope unit genuinely bundles >1 spike at high noise
    rows = {r["sigma"]: r for r in comparison["rows"]}
    assert rows[0.0]["envelope"]["count"] == 0.0
    assert rows[0.6]["envelope"]["mean_n_spikes_per_burst"] > 1.0
    assert (out / "figures" / "spike_vs_envelope.png").exists()
