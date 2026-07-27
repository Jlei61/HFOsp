from __future__ import annotations

import pandas as pd

from scripts.prepare_topic5_clinical_onset_source_annotation_v0_1 import (
    expected_registry,
    validate_registry,
)


def _source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": ["p1", "p1"],
            "seizure_id": ["s1", "s2"],
            "dataset": ["d", "d"],
            "clinical_onset_epoch_metadata": [1.0, 2.0],
            "n_model_contacts": [8, 8],
        }
    )


def test_registry_starts_blinded_and_without_source_substitution() -> None:
    source = _source()
    registry = expected_registry(source)
    validate_registry(registry, source)
    assert (registry.clinical_onset_contacts == "").all()
    assert (registry.consensus_status == "PENDING_BLINDED_REVIEW").all()
    assert registry.annotation_blinded_to_energy_values.all()


def test_registry_rejects_forbidden_energy_source_field() -> None:
    source = _source()
    registry = expected_registry(source)
    registry["energy_top_contacts"] = ""
    try:
        validate_registry(registry, source)
    except ValueError as error:
        assert "forbidden" in str(error)
    else:
        raise AssertionError("forbidden source substitution was accepted")

