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


def test_m43a_ablation_three_variants(tmp_path):
    """D11 mechanism-specificity control at ONE (alpha_A, tau_n) point: shunt_only / subtractive_only
    (matched-eta) / hybrid -> all 3 rows present."""
    out = tmp_path / "abl"
    cmd = [sys.executable, "scripts/run_m4_dynamic_qi.py", "--m43a-ablation", "--confirm-run",
           "--seed", "1", "--T", "800", "--m43a-abl-alpha", "6", "--m43a-abl-tau", "20000",
           "--out", str(out)]
    subprocess.run(cmd, check=True, cwd=os.getcwd())
    abl = json.load(open(glob.glob(str(out / "*ablation.json"))[0]))
    variants = {r["variant"] for r in abl["rows"]}
    assert variants == {"shunt_only", "subtractive_only", "hybrid"}
    # Config-level invariants: guard against parameter-swap failures
    by = {r["variant"]: r for r in abl["rows"]}
    eta_m = abl["eta_matched"]
    # shunt_only: pure divisive shunt (alpha_A = the swept point, eta_A = 0)
    assert by["shunt_only"]["alpha_A"] == abl["alpha_A"]
    assert by["shunt_only"]["eta_A"] == 0.0
    # subtractive_only: no shunt (alpha_A = 0), eta_A = the matched value
    assert by["subtractive_only"]["alpha_A"] == 0.0
    assert by["subtractive_only"]["eta_A"] == eta_m
    # hybrid: both engaged (alpha_A = swept point, eta_A = matched)
    assert by["hybrid"]["alpha_A"] == abl["alpha_A"]
    assert by["hybrid"]["eta_A"] == eta_m
