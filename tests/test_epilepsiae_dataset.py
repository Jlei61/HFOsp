from __future__ import annotations

from src.epilepsiae_dataset import EpilepsiaeTimeConfig, resolve_epilepsiae_timezone


def test_resolve_epilepsiae_timezone_uses_hospital_map() -> None:
    out = resolve_epilepsiae_timezone(
        subject="1073",
        patient_code="FR_1073",
        hospital="UKLFR",
    )
    assert out["timezone_name"] == "Europe/Berlin"
    assert out["timezone_source"] == "hospital"
    assert out["reliable_without_override"] is True


def test_resolve_epilepsiae_timezone_prefers_recording_override() -> None:
    cfg = EpilepsiaeTimeConfig(
        recording_timezone_overrides={"1073/107300102": "Europe/Paris"},
    )
    out = resolve_epilepsiae_timezone(
        subject="1073",
        patient_code="FR_1073",
        hospital="UKLFR",
        recording_id="107300102",
        time_config=cfg,
    )
    assert out["timezone_name"] == "Europe/Paris"
    assert out["timezone_source"] == "recording_override"
    assert out["reliable_without_override"] is False
