from __future__ import annotations

import numpy as np

from src.topic4_data_driven_cohort import (
    TargetConfig,
    build_crossfit_patient_target,
    canonical_pair_contract,
    geometry_only_sheet_projection,
    score_model_ranks_against_target,
    subset_pair_contract,
)


def _synthetic_patient(n_contacts=8, n_blocks=12, events_per_block=12):
    rng = np.random.default_rng(7)
    n_events = n_blocks * events_per_block
    ranks = np.empty((n_contacts, n_events), dtype=float)
    bools = np.ones((n_contacts, n_events), dtype=bool)
    blocks = np.repeat(np.arange(n_blocks), events_per_block)
    for event in range(n_events):
        mode = event % 2
        base = np.arange(n_contacts) if mode == 0 else np.arange(n_contacts)[::-1]
        noisy = base + rng.normal(0.0, 0.03, n_contacts)
        ranks[:, event] = np.argsort(np.argsort(noisy, kind="stable"), kind="stable")
        missing = rng.choice(n_contacts, size=1, replace=False)
        bools[missing, event] = False
        ranks[missing, event] = 1000.0 + event
    return {
        "channel_names": [f"C{index}" for index in range(n_contacts)],
        "ranks": ranks,
        "bools": bools,
        "block_ids": blocks,
    }


def _pair(n_contacts=8):
    return {
        "contact_order": [f"C{index}" for index in range(n_contacts)],
        "rank_a": np.arange(n_contacts, dtype=float),
        "rank_b": np.arange(n_contacts, dtype=float)[::-1],
        "cluster_id_a": 0,
        "cluster_id_b": 1,
    }


def _config():
    return TargetConfig(
        minimum_participating_contacts=3,
        heldout_block_fraction=0.25,
        split_seed=11,
        kmeans_fit_max_events=1000,
        kmeans_n_init=10,
        kmeans_seed=12,
        stability_seeds=(12, 13, 14),
        stored_events_per_mode_per_split=100,
    )


def test_crossfit_target_is_block_disjoint_and_recovers_both_modes():
    data = _synthetic_patient()
    target = build_crossfit_patient_target(data, _pair(), config=_config())
    assert set(target["train_block_ids"]).isdisjoint(target["heldout_block_ids"])
    assert np.min(target["train_mode_counts"]) >= 20
    assert np.min(target["heldout_mode_counts"]) >= 10
    assert target["train_to_heldout_margin"] > 1.0
    assert target["kmeans_stability"]["pairwise_ami_median"] == 1.0
    assert target["train_descriptors"]["TA"]["precedence"].shape == (28, 3)


def test_crossfit_target_discards_phantom_rank_values():
    data = _synthetic_patient()
    first = build_crossfit_patient_target(data, _pair(), config=_config())
    changed = {key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
               for key, value in data.items()}
    changed["ranks"][~changed["bools"]] = -99999.0
    second = build_crossfit_patient_target(changed, _pair(), config=_config())
    np.testing.assert_allclose(first["kmeans_centers"], second["kmeans_centers"])
    np.testing.assert_allclose(first["train_profiles"], second["train_profiles"])
    np.testing.assert_array_equal(first["train_labels"], second["train_labels"])


def test_geometry_projection_is_deterministic_and_inside_sheet():
    coords = np.asarray([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.5],
        [0.0, 3.0, 0.2],
        [2.0, 3.0, 0.8],
        [1.0, 1.5, 1.2],
    ])
    first = geometry_only_sheet_projection(coords, sheet_size_mm=20.0, margin_mm=2.0)
    second = geometry_only_sheet_projection(coords, sheet_size_mm=20.0, margin_mm=2.0)
    np.testing.assert_allclose(first["coords_sheet"], second["coords_sheet"])
    np.testing.assert_allclose(first["basis"], second["basis"])
    assert first["matrix_rank"] >= 2
    assert np.all(first["coords_sheet"] >= 2.0 - 1e-12)
    assert np.all(first["coords_sheet"] <= 18.0 + 1e-12)


def test_rank_displacement_contract_and_geometry_subset_preserve_order():
    record = {
        "stable_k": 2,
        "pairs": [{
            "channel_names": ["A", "B", "C", "D"],
            "joint_valid": [True, False, True, True],
            "rank_a_dense_full": [0, 1, 2, 3],
            "rank_b_dense_full": [3, 2, 1, 0],
            "cluster_id_a": 4,
            "cluster_id_b": 2,
        }],
    }
    pair = canonical_pair_contract(record)
    assert pair["contact_order"] == ["A", "C", "D"]
    subset = subset_pair_contract(pair, ["D", "A", "C"])
    assert subset["contact_order"] == ["D", "A", "C"]
    np.testing.assert_array_equal(subset["rank_a"], [3, 0, 2])


def test_model_rank_scorer_separates_supervised_and_natural_modes():
    data = _synthetic_patient()
    target = build_crossfit_patient_target(data, _pair(), config=_config())
    heldout_ranks = np.vstack([
        target["heldout_samples"]["TA"][:20],
        target["heldout_samples"]["TB"][:20],
    ])
    recruitment = np.asarray([
        target["train_descriptors"][mode]["recruitment"]
        for mode in ("TA", "TB")
    ])
    precedence = np.asarray([
        target["train_descriptors"][mode]["precedence"]
        for mode in ("TA", "TB")
    ])
    score = score_model_ranks_against_target(
        heldout_ranks,
        patient_centers=target["kmeans_centers"],
        patient_profiles=target["train_profiles"],
        patient_recruitment=recruitment,
        patient_precedence=precedence,
        patient_ood_threshold=target["train_distance_q95"],
        minimum_events_per_mode=3,
    )
    assert score["status"] == "EVALUABLE"
    assert score["supervised_margin"] > 1.0
    assert score["natural_margin"] > 1.0
    assert score["natural_seed_ami_median"] == 1.0
