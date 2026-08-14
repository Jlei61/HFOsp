import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/adjudicate_topic4_fcxr_lc5_to_lc6a.py"
SPEC = importlib.util.spec_from_file_location("lc5_to_lc6a", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def _summary(**updates):
    value = {
        "status": "COMPLETE",
        "outcome": "ESCALATING_SATURATION",
        "early_stop_reason": "REGISTERED_SATURATION_REACHED",
        "offset_ms": None,
        "lifecycle": {"label": "RUNAWAY"},
        "end_rate_hz": MOD.U2.SAT_CEILING_HZ + 1,
        "clip_frac_max_observed": 0.0,
        "classifier_snapshot_replay_n_bundles": 28,
    }
    value.update(updates)
    return value


def test_only_terminal_negative_saturation_authorizes_lc6a():
    assert MOD.adjudicate(_summary())["authorize_lc6a_40k_dynamics"] is True
    assert MOD.adjudicate(_summary(outcome="CONTAINED_HIGH_NO_OFFSET"))["authorize_lc6a_40k_dynamics"] is False
    assert MOD.adjudicate(_summary(offset_ms=26000.0))["authorize_lc6a_40k_dynamics"] is False


def test_authorization_keeps_substrate_conditional_claim_boundary():
    got = MOD.adjudicate(_summary())
    assert "legacy substrate" in got["scientific_boundary"]
    assert "does not reject U" in got["scientific_boundary"]
