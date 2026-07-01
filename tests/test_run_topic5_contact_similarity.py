"""Integration smoke test for the Topic5 contact-similarity ladder runner.

Plumbing-only (small B): one real eligible subject (broadband) must produce a
finite obs_subject + a within-shaft null verdict for all three rungs (R1 raw /
R2 same-plane kernel / R3 field), plus the paired grid/smooth deltas. The full
B=1000 cohort run is a separate (Task 7) step, not this test.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.run_topic5_contact_similarity import run_subject
from scripts.plot_topic5_contact_similarity import _default_summary_path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Absolute path: the gitignored T0 cache + axis records live in the main repo
# results/, not this worktree. Real-data shapes are what the runner is built for.
ROOT = "/home/honglab/leijiaxin/HFOsp/results"
# First subject (by axis-dir listing) for which run_subject(...)["status"]=="ok";
# discovered empirically, hardcoded for determinism.
SMOKE_SUBJECT = "epilepsiae_1077"


def test_run_subject_smoke():
    """All three rungs return a finite obs_subject and a within-shaft verdict;
    paired deltas are present and finite."""
    res = run_subject(SMOKE_SUBJECT, activation="broadband", B=20, seed=20260614,
                      input_results_root=ROOT)
    assert res["status"] == "ok"
    for rung in ("R1", "R2", "R3"):
        ws = res[rung]["within_shaft"]
        assert np.isfinite(ws["obs_subject"]), f"{rung} obs_subject not finite"
        assert ws["status"] in ("ok", "INSUFFICIENT_NULL"), f"{rung} status={ws['status']}"
    assert "grid_delta" in res and "smooth_delta" in res
    assert np.isfinite(res["grid_delta"]) and np.isfinite(res["smooth_delta"])


def test_negligible_within_band():
    """CI strictly inside ±SESOI → grid_negligible True."""
    from scripts.run_topic5_contact_similarity import _negligible
    assert _negligible(-0.04, 0.04, 0.05) is True


def test_negligible_outside_band_hi():
    """Upper CI edge outside SESOI → grid_negligible False."""
    from scripts.run_topic5_contact_similarity import _negligible
    assert _negligible(-0.04, 0.06, 0.05) is False


def test_negligible_outside_band_lo():
    """Lower CI edge outside SESOI → grid_negligible False."""
    from scripts.run_topic5_contact_similarity import _negligible
    assert _negligible(-0.06, 0.04, 0.05) is False


def test_negative_control_scrambled_activation_fails():
    """Bad-data regression: spatially scrambled ictal activation must NOT pass any
    rung's within-shaft null. Sign-free, so a *reversed* rank would still pass — the
    failing control is spatial scramble, not reversal."""
    res = run_subject(SMOKE_SUBJECT, activation="broadband", B=50, seed=20260614,
                      negative_control=True, input_results_root=ROOT)
    for rung in ("R1", "R2", "R3"):
        assert res[rung]["within_shaft"]["passed"] is False, f"{rung} passed on scrambled data"


# --------------------------------------------------------------------------- P1-1
# naming-contract tests (machine-independent, no real-data dependency): the runner
# writes cohort_summary_{activation}.{json,csv} (scripts/run_topic5_contact_similarity.py:350-351);
# the plotter's default read path must resolve to the exact same filename.

def _fixture_subject(subject_id):
    """Minimal schema-valid ok-subject dict — the fields the plotter's four panels read
    (R1/R2/R3.within_shaft.{obs_subject,null_q.p95}, R2_sigma_sweep, sequence, deltas)."""
    return {
        "subject_id": subject_id, "status": "ok",
        "R1": {"within_shaft": {"obs_subject": 0.30, "null_q": {"p95": 0.20},
                                "status": "ok", "passed": True}},
        "R2": {"within_shaft": {"obs_subject": 0.45, "null_q": {"p95": 0.25},
                                "status": "ok", "passed": True}},
        "R3": {"within_shaft": {"obs_subject": 0.44, "null_q": {"p95": 0.26},
                                "status": "ok", "passed": True}},
        "R2_sigma_sweep": {
            "0.5x": {"obs_subject": 0.40}, "1.0x": {"obs_subject": 0.45}, "2.0x": {"obs_subject": 0.42},
        },
        "sequence": {
            "spearman": {"obs_subject": 0.35, "null_q": {"p95": 0.22}},
            "kendall": {"obs_subject": 0.30, "null_q": {"p95": 0.20}},
        },
        "smooth_delta": 0.15, "grid_delta": -0.01,
    }


def test_plotter_default_matches_runner_naming(tmp_path):
    """Unit check: the plotter's default summary path for a given (out_dir, activation)
    must equal the exact filename the runner writes at
    scripts/run_topic5_contact_similarity.py:350 (`cohort_summary_{activation}.json`)."""
    for activation in ("broadband", "hfa"):
        runner_filename = f"cohort_summary_{activation}.json"  # literal from the runner's f-string
        assert _default_summary_path(tmp_path, activation) == tmp_path / runner_filename


def test_plotter_cli_reads_runner_named_summary_and_writes_png(tmp_path):
    """End-to-end naming contract (P1-1): drop a runner-named `cohort_summary_broadband.json`
    fixture into a tmp out-dir, invoke the plotter CLI with --activation broadband --out-dir
    <tmp> and NO --summary override, and confirm it resolves the file and writes the PNG.
    Machine-independent — no dependency on real T0 cache / axis records."""
    out_dir = tmp_path / "contact_similarity"
    out_dir.mkdir()
    summary = {
        "activation": "broadband", "B": 50, "seed": 20260614,
        "n_subjects": 2, "n_ok": 2,
        "per_subject": [_fixture_subject("epilepsiae_001"), _fixture_subject("yuquan_002")],
    }
    (out_dir / "cohort_summary_broadband.json").write_text(json.dumps(summary))

    result = subprocess.run(
        [sys.executable, "scripts/plot_topic5_contact_similarity.py",
         "--activation", "broadband", "--out-dir", str(out_dir)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    png = out_dir / "figures" / "contact_similarity_broadband_cohort.png"
    assert png.exists(), f"expected PNG not found: {png}\nstdout={result.stdout}\nstderr={result.stderr}"
