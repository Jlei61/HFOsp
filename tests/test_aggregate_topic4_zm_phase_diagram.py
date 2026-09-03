import pytest

from scripts.aggregate_topic4_zm_phase_diagram import (
    aggregate_pairs,
    pair_records,
)


def _record(arm, label, *, q=0.805, eta=0.02, seed=9101, noise="same"):
    return {
        "status": "SPATIAL_ZM_PHASE_POINT_COMPLETE",
        "coordinates": {
            "initial_state": arm,
            "q_clamp": q,
            "eta_m": eta,
            "noise_seed": seed,
        },
        "phase_config": {"sha256": "frozen"},
        "scientific_contract_sha256": "science",
        "paired_noise_contract": {"future_noise_sha256": noise},
        "classification": {"label": label},
        "stationary_metrics": {
            "median_rate_hz": 50.0 if label == "LOW" else 380.0,
            "median_active_E_fraction_20ms": (
                0.2 if label == "LOW" else 1.0),
            "median_recruited_sheet_fraction_1mm": (
                0.2 if label == "LOW" else 1.0),
        },
        "_input_path": f"/{arm}.json",
    }


def test_pair_records_identifies_bistable_candidate():
    pairs, identity = pair_records([
        _record("low", "LOW"),
        _record("high", "TONIC_HIGH"),
    ])
    assert identity == {
        "scientific_contract_sha256": "science",
        "phase_config_sha256": ["frozen"],
    }
    assert pairs[0]["pair_label"] == "BISTABLE_CANDIDATE"
    assert pairs[0]["low_median_rate_hz"] == 50.0
    assert pairs[0]["high_median_rate_hz"] == 380.0


def test_pair_records_rejects_unpaired_or_unmatched_noise():
    with pytest.raises(ValueError, match="incomplete initial-state pair"):
        pair_records([_record("low", "LOW")])
    with pytest.raises(ValueError, match="future-noise mismatch"):
        pair_records([
            _record("low", "LOW", noise="a"),
            _record("high", "TONIC_HIGH", noise="b"),
        ])


def test_aggregate_requires_full_seed_denominator_for_robust_label():
    pairs = []
    for seed in (1, 2, 3):
        row, _ = pair_records([
            _record("low", "LOW", seed=seed),
            _record("high", "TONIC_HIGH", seed=seed),
        ])
        pairs.extend(row)
    families = aggregate_pairs(pairs, minimum_seeds=3)
    assert families[0]["adjudication"]["verdict"] == (
        "ROBUST_SNN_BISTABILITY_CANDIDATE")
