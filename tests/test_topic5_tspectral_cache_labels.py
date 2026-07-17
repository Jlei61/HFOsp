import json

import pandas as pd

from scripts.augment_topic5_tspectral_cache_labels import (
    SIDECAR_SCHEMA_VERSION,
    augment_cache_root,
    build_label_payload,
)


def _classified_row() -> pd.Series:
    return pd.Series(
        {
            "analysis_version": "topic5_early_spectral_overlap_v3",
            "subject": "epilepsiae_1",
            "seizure_idx": 2,
            "has_accepted_tspectral": True,
            "phenotype": "broadband_gamma_low_overlap",
            "phenotype_label": "Broadband + gamma + low-frequency",
            "simple_phenotype": "broadband_1_150",
            "simple_phenotype_label": "Broadband enhancement (1-150 Hz)",
            "detection_gate_category": "broadband_1_150",
            "classification_reason": "tspectral_anchored_band_support",
            "anchor_rel_eeg_sec": 0.5,
            "anchor_source": "accepted_t_spectral_best",
            "accepted_tspectral_in_early_window": True,
            "n_analysis_contacts": 8,
            "n_low_band_hits": 3,
            "n_fast_band_hits": 3,
            "n_total_band_hits": 6,
            "strict_broadband_5of6": True,
            "gamma_band_30_80_support": True,
            "low_frequency_1_13_support": True,
            "delta_HYP_slow__hit": True,
            "theta_preictal_PAC__hit": True,
            "alpha_sharp_leq13__hit": True,
            "beta_LVFA_low__hit": True,
            "gamma_LVFA__hit": True,
            "hg_low_ripple__hit": True,
        }
    )


def test_build_label_payload_preserves_frequency_defined_contract() -> None:
    payload = build_label_payload(_classified_row())

    assert payload["label_status"] == "classified"
    assert payload["simple_phenotype"] == "broadband_1_150"
    assert payload["flags"]["strict_broadband_5of6"] is True
    assert all(payload["band_hits"].values())


def test_frozen_label_survives_later_time_only_acceptance() -> None:
    row = _classified_row()
    row["has_accepted_tspectral"] = False

    payload = build_label_payload(row)

    assert payload["label_status"] == "classified"
    assert payload["simple_phenotype"] == "broadband_1_150"
    assert payload["source_has_accepted_tspectral"] is False


def test_cache_augmentation_preserves_denominator_and_marks_missing(tmp_path) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    subject_meta = {
        "analysis_version": "topic5_tspectral_cache_v1p2",
        "subject": "epilepsiae_1",
        "seizure_idxs": [2, 5],
        "seizure": {"2": {"alignment_status": "accepted"}, "5": {}},
    }
    (cache_root / "epilepsiae_1.json").write_text(json.dumps(subject_meta))
    (cache_root / "cache_alignment_summary.json").write_text(
        json.dumps({"analysis_version": "topic5_tspectral_cache_v1p2"})
    )
    payload = build_label_payload(_classified_row())

    result = augment_cache_root(
        cache_root,
        labels={("epilepsiae_1", 2): payload},
        exclusions={("epilepsiae_1", 5): "missing_one_or_more_1_150hz_bands"},
        label_version="topic5_early_spectral_overlap_v3",
        phenotype_csv=tmp_path / "labels.csv",
        exclusions_csv=tmp_path / "exclusions.csv",
    )

    updated = json.loads((cache_root / "epilepsiae_1.json").read_text())
    assert updated["seizure_idxs"] == [2, 5]
    assert updated["metadata_schema_version"] == SIDECAR_SCHEMA_VERSION
    assert updated["early_spectral_phenotype_selectors"][
        "accepted_tspectral_strict_broadband_idxs"
    ] == [2]
    missing = updated["seizure"]["5"]["early_spectral_phenotype"]
    assert missing == {
        "label_status": "not_classified",
        "label_version": "topic5_early_spectral_overlap_v3",
        "reason": "missing_one_or_more_1_150hz_bands",
    }
    assert result["label_status_counts"] == {"classified": 1, "not_classified": 1}
