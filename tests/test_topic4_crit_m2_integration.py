"""Integration TDD for Task 5 (last task) — Topic 4 M3-v2.2 criticality Milestone 2.

Mirrors tests/test_topic4_crit_integration.py's role for M1: this file holds the
ASSEMBLY-level tests (the two-stage verdict builder wiring T1-T4 together + the CLI),
while tests/test_topic4_criticality_m2.py holds T0-T4's own per-task unit tests.

Contract (verbatim from task brief .superpowers/sdd/task-5-brief.md Step 1/Step 6):
  - build_ignition_spread_verdict produces the three coexisting blocks (csd_verdict/
    linear_ignition/nonlinear_spread/interpretation) and does NOT resurrect the retired
    rev1.1 three-way final_verdict.
  - the CLI writes ignition_spread_verdict.json with csd_verdict co-displayed.

Plus fast synthetic unit tests (no real-crossing solve) for the new T5 judgment-call helpers
(_ignition_base_gate/_unresolved_subreason/_interpretation) -- mirrors T3/T4's own precedent of
unit-testing aggregation helpers directly with synthetic data before paying for the expensive
real-crossing integration test.
"""
import json
import subprocess
import sys
from pathlib import Path

from src.topic4_criticality import load_crit_config
import src.topic4_criticality_m2 as m2

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _points():
    p = m2._REPO / "results/topic4_criticality/trajectory_verdict.json"   # M1 deliverable
    return json.loads(p.read_text())["points"]


# --- Fast synthetic unit tests for the new T5 helpers (no real-crossing solve) ---

def test_ignition_base_gate_requires_all_conjuncts():
    """§5.0/T1-review note: op_solve_quality_left/right + branch_identity_clean are SEPARATE
    fields that must ALL read True -- reading only one side would silently readmit a crossing
    whose other bracketing point is unreliable."""
    base = {"_crossing_op": object(), "_crossing_res": object(),
            "op_solve_quality_left": True, "op_solve_quality_right": True,
            "branch_identity_clean": True}
    assert m2._ignition_base_gate(base) is True
    assert m2._ignition_base_gate({**base, "op_solve_quality_left": False}) is False
    assert m2._ignition_base_gate({**base, "op_solve_quality_right": False}) is False
    assert m2._ignition_base_gate({**base, "branch_identity_clean": False}) is False
    assert m2._ignition_base_gate({**base, "_crossing_op": None}) is False
    assert m2._ignition_base_gate({**base, "_crossing_res": None}) is False


def test_unresolved_subreason_priority_and_null():
    """alpha0_not_localized (no crossing at all) takes priority over ignition_not_localized
    (crossing exists but quality/branch failed); spread's own unresolved_nonlinear_spread is only
    reported once the ignition base gate itself passed; both clean -> None."""
    assert m2._unresolved_subreason({"_crossing_op": None}, False,
                                    {"epsilon_sensitivity": "pass"}) == "alpha0_not_localized"
    assert m2._unresolved_subreason({"_crossing_op": object()}, False,
                                    {"epsilon_sensitivity": "pass"}) == "ignition_not_localized"
    assert m2._unresolved_subreason({"_crossing_op": object()}, True,
                                    {"epsilon_sensitivity": "epsilon_sensitive"}) == "unresolved_nonlinear_spread"
    assert m2._unresolved_subreason({"_crossing_op": object()}, True,
                                    {"epsilon_sensitivity": "pass"}) is None


def test_interpretation_mechanical_compose_never_reglues_spread_onto_mode():
    ig = {"class": "core_localized"}
    sp = {"onset": "axial", "endgame": "self_limited", "off_axis": "absent"}
    s = m2._interpretation(ig, sp)
    assert s == "core_localized ignition followed by axial transient and self_limited; off_axis absent"
    # never re-glue spread onto the linear mode (e.g. never claim "the mode is axial")
    assert "mode is axial" not in s and "critical mode is" not in s


# --- Task 5 Step 1 (task brief, verbatim): the real-crossing integration test ---

def test_build_verdict_two_stage_coexists_with_csd():
    cfg = load_crit_config(); m2cfg = m2.load_m2_config()
    v = m2.build_ignition_spread_verdict(_points(), cfg, m2cfg)
    assert v["csd_verdict"] == "unresolved_operating_point"          # M1 unchanged
    assert v["linear_ignition"]["class"] == "core_localized"
    assert set(("onset", "endgame", "off_axis", "depth_dependent")) <= set(v["nonlinear_spread"])
    assert "final_verdict" not in v                                  # retired rev1.1 three-way

    assert "ignition" in v["interpretation"] and "off_axis" in v["interpretation"]

    # base gate / subreason on the real (clean) crossing.
    assert v["base_gate_passed"] is True
    assert v["unresolved_subreason"] == "unresolved_nonlinear_spread"   # spread's own eps gate failed

    # T3's off_axis sentinel nests inside linear_ignition (T5's own assembly decision).
    sentinel = v["linear_ignition"]["off_axis_sentinel"]
    assert sentinel["off_axis"] == "absent"

    # T1's crossing carried forward for figure/traceability re-use, unstripped in-memory.
    assert v["linear_ignition"]["crossing"]["_crossing_res"] is not None


# --- Task 5 Step 6 (task brief, verbatim): CLI smoke ---

def test_cli_smoke_writes_verdict(tmp_path):
    r = subprocess.run([sys.executable, "scripts/run_topic4_crit_m2.py", "--out", str(tmp_path)],
                       capture_output=True, text=True, cwd=str(_REPO))
    assert r.returncode == 0, r.stderr
    v = json.loads((tmp_path / "ignition_spread_verdict.json").read_text())
    assert v["csd_verdict"] == "unresolved_operating_point"
    assert "final_verdict" not in v
    assert (tmp_path / "STATUS.md").exists()
    assert (tmp_path / "figures" / "ignition_panel.png").exists()
    assert (tmp_path / "figures" / "spread_panel.png").exists()
    assert (tmp_path / "figures" / "basis_sanity.png").exists()

    dumped = json.dumps(v)
    # (a) the genuinely NON-serializable / duplicate working fields are stripped from the JSON.
    for private in ("_crossing_op", "_crossing_res", "_two_core_crossing"):
        assert private not in dumped, private
    # (b) but the JSON-serializable AUDIT fields STATUS.md points readers to ("阈值敏感性、逐区功率、
    # 逐 (depth, epsilon_rel, polarity) 明细见 ignition_spread_verdict.json") ARE present, renamed
    # public (leading underscore dropped) -- else the STATUS reference is a false pointer (T5 review).
    assert v["linear_ignition"]["two_core_region_frac"]["corridor_axial"] == 0.0
    assert "two_core_axis_profile" in v["linear_ignition"]
    assert "branch_continuation_status" in v["linear_ignition"]["crossing"]
    assert "epsilon_sweep_detail" in v["nonlinear_spread"]
    assert "depth_aggregate" in v["nonlinear_spread"]
    # and no `_`-prefixed key survives anywhere in the written public schema.
    import re
    assert not re.search(r'"_[A-Za-z]', dumped), "a private (_-prefixed) key leaked into the JSON"


def test_cli_lazy_deps_importable():
    """Mirrors M1's test_verdict_cli_lazy_deps_importable: the CLI's own deps import cleanly
    standalone (catches an import-chain break independent of the (slow) full-run smoke above)."""
    code = "import src.topic4_criticality_m2, src.topic4_criticality, src.topic4_m3b_spectral_phase"
    result = subprocess.run(
        [sys.executable, "-c", f"import sys; sys.path.insert(0, {str(_REPO)!r}); {code}"],
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
