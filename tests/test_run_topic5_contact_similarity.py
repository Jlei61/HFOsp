"""Integration smoke test for the Topic5 contact-similarity ladder runner.

Plumbing-only (small B): one real eligible subject (broadband) must produce a
finite obs_subject + a within-shaft null verdict for all three rungs (R1 raw /
R2 same-plane kernel / R3 field), plus the paired grid/smooth deltas. The full
B=1000 cohort run is a separate (Task 7) step, not this test.
"""
import numpy as np

from scripts.run_topic5_contact_similarity import run_subject

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


def test_negative_control_scrambled_activation_fails():
    """Bad-data regression: spatially scrambled ictal activation must NOT pass any
    rung's within-shaft null. Sign-free, so a *reversed* rank would still pass — the
    failing control is spatial scramble, not reversal."""
    res = run_subject(SMOKE_SUBJECT, activation="broadband", B=50, seed=20260614,
                      negative_control=True, input_results_root=ROOT)
    for rung in ("R1", "R2", "R3"):
        assert res[rung]["within_shaft"]["passed"] is False, f"{rung} passed on scrambled data"
