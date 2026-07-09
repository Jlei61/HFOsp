"""M4-3A runner smoke test (Task 8, spec `--m43a-sweep`): tiny end-to-end subprocess run --
1 seed, 2x2 (alpha_A x tau_n) grid + Arm0, very short T -> pipeline completes, JSON schema present."""
import json, subprocess, sys, os, glob


def test_m43a_sweep_tiny_runs_and_writes_schema(tmp_path):
    """1 seed, 2x2 grid, very short T -> pipeline runs, JSON schema present, Arm0 row exists."""
    out = tmp_path / "m43a_smoke"
    cmd = [sys.executable, "scripts/run_m4_dynamic_qi.py", "--m43a-sweep", "--confirm-run",
           "--seed", "1", "--T", "800",
           "--m43a-alpha-grid", "0,4", "--m43a-tau-grid", "2000,20000",
           "--m43a-workers", "2", "--out", str(out)]
    subprocess.run(cmd, check=True, cwd=os.getcwd())
    summ = json.load(open(glob.glob(str(out / "*summary.json"))[0]))
    rows = summ["rows"]
    labels = {r["label"] for r in rows}
    assert any(l.startswith("m43a_arm0") for l in labels)          # per-seed Arm0 present
    for r in rows:
        for key in ("label", "alpha_A", "tau_n", "termination_class",
                    "retrigger_probe", "go", "seed"):     # retrigger_early is conditional (terminate_clean only)
            assert key in r
    assert summ["provenance"]["seed"] == 1
