"""Integration TDD for Task 1 — M3A-v2.2 approach-criticality, Milestone 1.

Three contracts:
1. Byte-parity: run_transition() reproduces the golden fixture captured from the
   figure code BEFORE factoring (all sim output hashes + the deterministic events).
2. Two-layer fail-closed export: the hand-built calibrated fixture handoff PASSES
   (proves the M3A->M3B interface machinery is wired), while the REAL v2.2 export
   legitimately fails closed (uncalibrated mapping -> refused), never silently upgraded.
3. CLI smoke: the 3 criticality entrypoints (export.py/atlas.py/verdict.py) parse --help
   cleanly, and run_topic4_crit_verdict.py's lazily-imported deps (invisible to --help,
   since it imports them inside its worker fn) import cleanly standalone -- both always-run,
   no figdata required.

The first two tests re-run the full T=1600ms SNN transition sim (~3-4 min each) -> @integration.
They require the subject1146 figdata artifact (gitignored results/ tree); skipped if absent.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_FIXTURE = ROOT / "tests" / "fixtures" / "topic4_m3v2_2_transition_golden.json"
_FIGDATA = (ROOT / "results" / "topic4_sef_hfo" / "field_swap_subject_snn"
            / "figdata_epilepsiae_1146_twoend_equal_tsrc_s3.npz")
_needs_figdata = pytest.mark.skipif(
    not _FIGDATA.exists(), reason=f"subject1146 figdata missing: {_FIGDATA}")


def _h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


@pytest.mark.integration
@_needs_figdata
def test_run_transition_matches_golden_fixture():
    from src.sef_hfo_transition_sim import run_transition, default_transition_config
    g = json.load(open(_FIXTURE))
    res = run_transition(default_transition_config(layout="subject1146", top="qI"))
    # NOTE: real sim key is trace_gK_axial (NOT the plan snippet's illustrative trace_gK).
    for key in ["E_spk_bool", "rate_E", "trace_qI_mean", "trace_qI_min", "trace_gK_axial"]:
        assert _h(res[key]) == g[f"{key}_hash"], f"{key} changed by factoring"
    assert res["events"] == g["events"]


@pytest.mark.integration
@_needs_figdata
def test_export_fixture_passes_and_real_is_fail_closed(tmp_path):
    from src.topic4_criticality import export_fixture_handoff, export_v2_2_handoff
    from src.sef_hfo_transition_sim import default_transition_config
    assert export_fixture_handoff(tmp_path / "fix") == "phase_map_trajectory"     # machinery proven
    v = export_v2_2_handoff(tmp_path / "real", default_transition_config("subject1146", "qI"))
    assert v != "phase_map_trajectory"   # real uncalibrated v2.2 MUST refuse the overlay (fail-closed)
    audit = json.loads((tmp_path / "real" / "m3a_interface_audit.json").read_text())
    assert audit["cond1_sign_tests_passed"] is False   # refusal is because the mapping isn't sign-calibrated


@pytest.mark.integration
@_needs_figdata
def test_trajectory_verdict_is_actual_trajectory_and_enum(tmp_path):
    """Task 3a-5b crux: the verdict is computed from the REAL 3-D slow trajectory (incl h_G(t)),
    NOT by sampling the 2-D atlas -> verdict_source=="actual_trajectory" AND verdict in the
    pre-registered 3-enum. Runs the v2.2 SNN once + writes trajectory_verdict.json + Figure 1."""
    import importlib
    m = importlib.import_module("scripts.run_topic4_crit_verdict")
    payload = m.build_and_write_verdict(tmp_path)
    enum = {"smooth_CSD", "hard_jump_no_CSD", "unresolved_operating_point"}
    assert payload["verdict_source"] == "actual_trajectory"     # #1 NOT the 2-D atlas
    assert payload["verdict"] in enum
    # the written JSON carries the same guard + enum (strict-parser safe, non-finite sanitized).
    j = json.loads((tmp_path / "trajectory_verdict.json").read_text())
    assert j["verdict_source"] == "actual_trajectory"
    assert j["verdict"] in enum
    assert j["operator_type"] == "continuous_jacobian" and j["alpha_units"] == "per_ms"
    assert (tmp_path / "figures" / "trajectory_criticality_verdict.png").exists()
    assert (tmp_path / "STATUS.md").exists()
    # overlay refused for the real uncalibrated v2.2 -> no atlas overlay drawn (Hard-QC #7).
    assert j["overlay_drawn"] is False


@pytest.mark.integration
@_needs_figdata
def test_atlas_is_conditional_and_not_verdict_source(tmp_path):
    from src.topic4_criticality import build_conditional_atlas, load_crit_config
    from src.sef_hfo_m3a_export import default_precalib_mapping_and_ranges
    m, r = default_precalib_mapping_and_ranges("m3a_v2_2_approach")
    build_conditional_atlas(m, r, load_crit_config(), out_dir=tmp_path)
    meta = json.loads((tmp_path / "finite_jacobian_grid.json").read_text())
    assert meta["m3a_overlay_consumable"] is True
    assert meta["atlas_name"].startswith("conditional_2d_atlas_at_phase_recovery=")
    assert meta["verdict_source"] == "actual_trajectory_not_atlas"          # #1 guard
    assert meta["axes_built_from_slow_to_rate_mapping_id"] == "m3a_v2_2_approach"


# ---------------------------------------------------------------------------
# T3a-6 (last task, Milestone 1): CLI smoke tests for the 3 criticality entrypoints
# (run_topic4_crit_export.py / _atlas.py / _verdict.py). All 3 CLIs run the SNN
# internally (export ~2-4min, atlas ~12min, verdict ~6min) -- too slow to routinely
# subprocess-run all 3 end-to-end.
#
# COVERAGE TRADEOFF (logged, not silently skipped): the ALWAYS-RUN smoke below
# (test_cli_help_smoke) catches the common CLI breakage mode (import errors, argparse
# misconfiguration) in well under a second each, with no figdata and no SNN run -- but its
# import coverage is NOT uniform across all 3 CLIs. run_topic4_crit_export.py and
# run_topic4_crit_atlas.py import their src.* deps at MODULE SCOPE, so --help genuinely
# exercises those imports. run_topic4_crit_verdict.py instead imports its deps LAZILY, inside
# build_and_write_verdict(), so --help short-circuits at argparse.parse_args() before ever
# reaching them -- its dep-import coverage comes from the companion
# test_verdict_cli_lazy_deps_importable below instead (always-run, no figdata needed). Full
# subprocess end-to-end is exercised
# ONLY for run_topic4_crit_verdict.py, the milestone's main deliverable CLI, behind
# @integration (test_verdict_cli_end_to_end_subprocess). run_topic4_crit_export.py and
# run_topic4_crit_atlas.py are NOT routinely end-to-end subprocess-tested (SNN cost);
# their underlying library functions (export_v2_2_handoff, build_conditional_atlas)
# already get full-SNN integration coverage from the two @integration tests above
# (test_export_fixture_passes_and_real_is_fail_closed,
# test_atlas_is_conditional_and_not_verdict_source) -- just not through the CLI /
# argparse layer itself. A full export/atlas CLI subprocess run remains a
# manual/@integration-only concern if that layer ever needs its own coverage.
# ---------------------------------------------------------------------------
_CLI_SCRIPTS = [
    "run_topic4_crit_export.py",
    "run_topic4_crit_atlas.py",
    "run_topic4_crit_verdict.py",
]


@pytest.mark.parametrize("script", _CLI_SCRIPTS)
def test_cli_help_smoke(script):
    """Always-run, no figdata / no SNN needed: --help must exit 0 with no traceback.
    Catches import errors / argparse misconfiguration -- the common CLI failure mode --
    in well under a second per script (module-level imports still run, but --help short-
    circuits inside argparse.parse_args() before any sim/library call)."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--help"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "Traceback" not in result.stderr, result.stderr


def test_verdict_cli_lazy_deps_importable():
    """run_topic4_crit_verdict.py imports its deps lazily (inside the worker fn), so the --help
    smoke never reaches them. Import them directly so a break in the fragile sef_hfo_transition_sim
    interim-bridge is caught in a figdata-less clone (where the @integration e2e is skipped)."""
    code = "import src.sef_hfo_transition_sim, src.sef_hfo_m3a_export, src.topic4_criticality"
    result = subprocess.run(
        [sys.executable, "-c", f"import sys; sys.path.insert(0, {str(ROOT)!r}); {code}"],
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"


@pytest.mark.integration
@_needs_figdata
def test_verdict_cli_end_to_end_subprocess(tmp_path):
    """Real subprocess invocation of the milestone's main deliverable CLI -- distinct from
    test_trajectory_verdict_is_actual_trajectory_and_enum above, which imports
    build_and_write_verdict directly. This test instead proves the actual
    argparse -> main() -> build_and_write_verdict wiring works end-to-end as a script.
    Runs the v2.2 SNN once (~6 min). Writes into tmp_path (--out-dir) so the committed
    results/topic4_criticality workflow output is never clobbered."""
    cmd = [sys.executable, str(ROOT / "scripts" / "run_topic4_crit_verdict.py"),
           "--out-dir", str(tmp_path), "--layout", "subject1146", "--top", "qI"]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=900, check=True)
    except subprocess.CalledProcessError as e:
        pytest.fail(f"verdict CLI failed (rc={e.returncode}):\nSTDOUT={e.stdout}\nSTDERR={e.stderr}")
    assert (tmp_path / "trajectory_verdict.json").exists()
    assert (tmp_path / "STATUS.md").exists()
    payload = json.loads((tmp_path / "trajectory_verdict.json").read_text())
    assert payload["verdict_source"] == "actual_trajectory"
    assert payload["verdict"] in {"smooth_CSD", "hard_jump_no_CSD", "unresolved_operating_point"}
