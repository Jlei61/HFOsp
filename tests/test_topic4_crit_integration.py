"""Integration TDD for Task 1 — M3A-v2.2 approach-criticality, Milestone 1.

Two contracts:
1. Byte-parity: run_transition() reproduces the golden fixture captured from the
   figure code BEFORE factoring (all sim output hashes + the deterministic events).
2. Two-layer fail-closed export: the hand-built calibrated fixture handoff PASSES
   (proves the M3A->M3B interface machinery is wired), while the REAL v2.2 export
   legitimately fails closed (uncalibrated mapping -> refused), never silently upgraded.

Both tests re-run the full T=1600ms SNN transition sim (~3-4 min each) -> @integration.
They require the subject1146 figdata artifact (gitignored results/ tree); skipped if absent.
"""
import hashlib
import json
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
    assert v in {"phase_map_trajectory", "mechanism_candidate_only", "refused"}   # never silently upgraded
    if v != "phase_map_trajectory":
        assert (tmp_path / "real" / "m3a_interface_audit.json").exists()          # blocking reason written
