from types import SimpleNamespace

from scripts import run_topic5_stateful_event_rnn_v2_7_control as control


def test_control_adapter_rebinds_only_shared_infrastructure():
    module = SimpleNamespace(
        DEFAULT_CONFIG="old",
        prepare_subject="old",
        fit_profile="old",
        verify_frozen="old",
        scientific_control="unchanged",
    )
    configured = control.configure_control(module)
    assert configured.DEFAULT_CONFIG == control.formal.DEFAULT_CONFIG
    assert configured.prepare_subject is control.formal.prepare_subject
    assert configured.fit_profile is control.formal.fit_profile
    assert configured.verify_frozen is control.formal.verify_frozen
    assert configured.scientific_control == "unchanged"


def test_all_required_frozen_controls_are_registered():
    assert set(control.CONTROL_MODULES) == {
        "dense", "state-reset", "memory-curve", "block-null",
        "reversal-null", "h40",
    }
