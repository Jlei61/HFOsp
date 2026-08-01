import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analyze_topic4_zm_lifecycle_m_panel.py"
SPEC = importlib.util.spec_from_file_location("topic4_zm_lifecycle_m_analysis", SCRIPT)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


def _row(onset=500.0, offset=None):
    return {"episode": {"onset_ms": onset, "offset_ms": offset}}


def test_m_effect_requires_durable_offset_and_paired_gm0_difference():
    censored = _row(offset=None)
    exited = _row(offset=3500.0)
    got = M.paired_m_effect(exited, censored)
    assert got["causal_exit_candidate"] is True
    assert got["status"] == "offset_vs_censored_gM0"
    assert M.paired_m_effect(censored, censored)["causal_exit_candidate"] is False


def test_m_effect_rejects_prevention_and_small_duration_shift():
    assert M.paired_m_effect(_row(onset=None), _row(offset=None))["status"] == "prevention_or_no_onset"
    baseline = _row(onset=500.0, offset=4000.0)
    small = _row(onset=500.0, offset=3500.0)
    assert M.paired_m_effect(small, baseline)["causal_exit_candidate"] is False
    large = _row(onset=500.0, offset=2500.0)
    assert M.paired_m_effect(large, baseline)["causal_exit_candidate"] is True
