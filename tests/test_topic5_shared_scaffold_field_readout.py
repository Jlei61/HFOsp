from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pytest
from scipy.stats import rankdata

from src.topic5_shared_scaffold_field_readout import (
    FIELD_DEFINITION,
    bidirectional_rollout_fields,
    build_frozen_field_manifest,
    build_frozen_subject_field_record,
    contact_label_permutations,
    learned_axis_source_pools,
    normalized_laplacian_source_pools,
    paired_model_patient_statistics,
    participation_weighted_first_arrival_earliness,
    score_frozen_field_against_ictal,
    score_two_direction_max_abs_spearman,
    seizure_first_patient_first_summary,
    validate_frozen_field_manifest,
    validate_frozen_subject_field_record,
    write_frozen_field_manifest,
)


def _path_operator(n_contacts: int = 8) -> np.ndarray:
    operator = np.zeros((n_contacts, n_contacts), dtype=float)
    for index in range(n_contacts - 1):
        operator[index, index + 1] = 1.0
        operator[index + 1, index] = 1.0
    return operator


def _names(n_contacts: int = 8) -> list[str]:
    return [f"A{index + 1}" for index in range(n_contacts // 2)] + [
        f"B{index + 1}" for index in range(n_contacts - n_contacts // 2)
    ]


def _diffusion(n_contacts: int = 8):
    return normalized_laplacian_source_pools(
        _path_operator(n_contacts), contact_names=_names(n_contacts)
    )


def _record(subject: str = "p1", model: str = "structured"):
    operator = _path_operator()
    diffusion = normalized_laplacian_source_pools(
        operator, contact_names=_names()
    )
    return build_frozen_subject_field_record(
        subject_id=subject,
        model_name=model,
        contact_names=_names(),
        operator=operator,
        diffusion_result=diffusion,
        field_minus=np.linspace(1.0, 0.0, 8),
        field_plus=np.linspace(0.0, 1.0, 8),
        horizon=5,
        checkpoint_sha256_by_seed={"11": "a" * 64, "29": "b" * 64},
        training_split_sha256="c" * 64,
    )


def test_normalized_laplacian_coordinate_and_source_pools_are_target_free_endpoints():
    result = _diffusion()
    coordinate = result["diffusion_coordinate"]
    assert coordinate.shape == (8,)
    assert np.isclose(np.linalg.norm(coordinate), 1.0)
    assert len(result["source_minus_indices"]) == 2
    assert len(result["source_plus_indices"]) == 2
    assert not np.intersect1d(
        result["source_minus_indices"], result["source_plus_indices"]
    ).size
    # A path's Fiedler coordinate is monotone up to its arbitrary sign, so the
    # two pools must be its physical endpoints, never any supplied target.
    endpoints = {
        frozenset(result["source_minus_indices"]),
        frozenset(result["source_plus_indices"]),
    }
    assert endpoints == {frozenset([0, 1]), frozenset([6, 7])}
    np.testing.assert_allclose(
        result["normalized_laplacian"],
        result["normalized_laplacian"].T,
    )


def test_diffusion_pool_fails_closed_for_directed_or_disconnected_operator():
    directed = _path_operator()
    directed[0, 1] = 2.0
    with pytest.raises(ValueError, match="symmetric"):
        normalized_laplacian_source_pools(directed, contact_names=_names())
    disconnected = _path_operator()
    disconnected[3, 4] = disconnected[4, 3] = 0.0
    with pytest.raises(ValueError, match="disconnected"):
        normalized_laplacian_source_pools(disconnected, contact_names=_names())


def test_participation_weighted_first_arrival_earliness_has_one_unique_formula():
    # H=4: future-step weights are 3/4, 1/2, 1/4, 0.
    mass = np.zeros((4, 5), dtype=float)
    mass[0, 1] = 0.4
    mass[2, 1] = 0.2
    mass[1, 2] = 1.0
    mass[3, 3] = 0.8
    result = participation_weighted_first_arrival_earliness(
        mass, source_indices=[0]
    )
    np.testing.assert_allclose(result["earliness_weights"], [0.75, 0.5, 0.25, 0.0])
    # source=t0; contact 1 = .4*.75 + .2*.25; contact 2 = 1*.5.
    np.testing.assert_allclose(result["field"], [1.0, 0.35, 0.5, 0.0, 0.0])
    np.testing.assert_allclose(
        result["participation_probability"], [1.0, 0.6, 1.0, 0.8, 0.0]
    )
    assert result["conditional_mean_first_arrival_step"][0] == 0.0


def test_first_arrival_rejects_source_rearrival_and_excess_probability():
    mass = np.zeros((3, 4), dtype=float)
    mass[0, 0] = 0.1
    with pytest.raises(ValueError, match="sources"):
        participation_weighted_first_arrival_earliness(mass, source_indices=[0])
    mass[:] = 0.0
    mass[:, 1] = 0.5
    with pytest.raises(ValueError, match="exceeds one"):
        participation_weighted_first_arrival_earliness(mass, source_indices=[0])


def test_bidirectional_rollout_exposes_exactly_one_field_per_source_side():
    minus = np.zeros((3, 6), dtype=float)
    plus = np.zeros((3, 6), dtype=float)
    minus[0, 2] = 1.0
    minus[1, 3] = 1.0
    plus[0, 3] = 1.0
    plus[1, 2] = 1.0
    fields = bidirectional_rollout_fields(
        first_arrival_minus=minus,
        first_arrival_plus=plus,
        source_minus_indices=[0, 1],
        source_plus_indices=[4, 5],
    )
    assert fields["field_definition"] == FIELD_DEFINITION
    assert fields["horizon"] == 3
    assert fields["field_minus"].shape == (6,)
    assert fields["field_plus"].shape == (6,)
    # No alternative participation/early/late/endpoint candidates are exposed.
    assert not any(key.startswith("field_") for key in fields if key not in {
        "field_minus", "field_plus", "field_definition"
    })


def test_field_record_and_manifest_hash_detect_any_post_freeze_mutation(tmp_path):
    first = _record("p2")
    validate_frozen_subject_field_record(first)
    manifest = build_frozen_field_manifest(
        [first], created_utc="2026-08-03T00:00:00Z", code_sha256="d" * 64
    )
    validate_frozen_field_manifest(manifest)
    path = tmp_path / "FROZEN_FIELD_MANIFEST.json"
    write_frozen_field_manifest(path, manifest)
    reloaded = json.loads(path.read_text())
    validate_frozen_field_manifest(reloaded)
    assert reloaded["target_values_read"] is False
    assert reloaded["target_values_sealed"] is True

    changed = deepcopy(reloaded)
    changed["records"][0]["field_minus"][2] += 0.01
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_frozen_field_manifest(changed)


def _centered(values):
    ranked = rankdata(np.asarray(values, dtype=float), method="average")
    return ranked - ranked.mean()


def test_two_direction_max_is_recomputed_inside_every_null_draw():
    minus = np.arange(6, dtype=float)
    plus = np.asarray([0, 2, 1, 4, 3, 5], dtype=float)
    target = np.arange(6, dtype=float)
    result = score_two_direction_max_abs_spearman(
        field_minus=minus,
        field_plus=plus,
        target=target,
        contact_names=["A1", "A2", "A3", "B1", "B2", "B3"],
        n_draws=40,
        all_contact_seed=7,
        within_shaft_seed=8,
    )
    assert result["observed_max_abs_rho"] == pytest.approx(1.0)
    permutations = contact_label_permutations(
        ["A1", "A2", "A3", "B1", "B2", "B3"],
        n_draws=40,
        seed=7,
        mode="all_contact",
    )
    target_rank = _centered(target)
    field_rank = np.column_stack([_centered(minus), _centered(plus)])
    manual = np.max(
        np.abs(
            (target_rank[permutations] @ field_rank)
            / (np.linalg.norm(target_rank) * np.linalg.norm(field_rank, axis=0)[None, :])
        ),
        axis=1,
    )
    np.testing.assert_allclose(result["all_contact_null"], manual)


def test_within_shaft_permutations_never_cross_shafts():
    names = ["A1", "A2", "A3", "B1", "B2", "B3"]
    draws = contact_label_permutations(
        names, n_draws=100, seed=3, mode="within_shaft"
    )
    assert np.all(np.isin(draws[:, :3], [0, 1, 2]))
    assert np.all(np.isin(draws[:, 3:], [3, 4, 5]))


def test_exact_name_scoring_uses_only_the_frozen_two_fields():
    record = _record()
    shuffled_names = ["B4", "A1", "B3", "A2", "B2", "A3", "B1", "A4"]
    target_lookup = {
        name: value for name, value in zip(record["contact_order"], record["field_minus"])
    }
    target = [target_lookup[name] for name in shuffled_names]
    score = score_frozen_field_against_ictal(
        record,
        seizure_id="sz1",
        target_contact_names=shuffled_names,
        target_values=target,
        n_draws=20,
        all_contact_seed=1,
        within_shaft_seed=2,
    )
    assert score["observed_max_abs_rho"] == pytest.approx(1.0)
    assert score["field_fingerprint_sha256"] == record["fingerprint_sha256"]
    assert score["matched_contact_names"] == record["contact_order"]


def _seizure_score(subject, model, seizure, observed, all_null, shaft_null):
    return {
        "subject": subject,
        "model": model,
        "seizure_id": seizure,
        "observed_max_abs_rho": observed,
        "all_contact_null": np.asarray(all_null, dtype=float),
        "within_shaft_null": np.asarray(shaft_null, dtype=float),
    }


def test_seizure_first_patient_first_folds_nulls_and_excludes_e1146():
    rows = [
        _seizure_score("p1", "structured", "s1", 0.8, [0.1, 0.3, 0.2], [0.2, 0.4, 0.3]),
        _seizure_score("p1", "structured", "s2", 0.6, [0.3, 0.5, 0.4], [0.4, 0.6, 0.5]),
        _seizure_score("p2", "structured", "s1", 0.7, [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]),
        _seizure_score("epilepsiae_1146", "structured", "s1", 0.9, [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]),
    ]
    summary = seizure_first_patient_first_summary(rows, n_boot=100, bootstrap_seed=4)
    p1 = next(row for row in summary["patients"] if row["subject"] == "p1")
    assert p1["observed_max_abs_rho"] == pytest.approx(0.7)
    # Draw-wise seizure median first: [0.2, 0.4, 0.3], whose median is .3.
    np.testing.assert_allclose(p1["all_contact_null"], [0.2, 0.4, 0.3])
    assert p1["all_contact_margin"] == pytest.approx(0.4)
    supportive = next(
        row for row in summary["patients"] if row["subject"] == "epilepsiae_1146"
    )
    assert supportive["supportive_only"] is True
    assert summary["cohort"]["structured"]["n_primary_patients"] == 2
    assert summary["cohort"]["structured"]["all_contact"]["n_positive"] == 2


def test_paired_model_statistics_are_patient_matched_and_supportive_excluded():
    seizure_rows = []
    for subject, structured, ordinary in (
        ("p1", 0.8, 0.5),
        ("p2", 0.7, 0.4),
        ("epilepsiae_1146", 0.9, 0.1),
    ):
        for model, observed in (("structured", structured), ("ordinary", ordinary)):
            seizure_rows.append(
                _seizure_score(subject, model, "s1", observed, [0.2] * 8, [0.3] * 8)
            )
    summary = seizure_first_patient_first_summary(seizure_rows, n_boot=50)
    paired = paired_model_patient_statistics(
        summary["patients"],
        model_a="structured",
        model_b="ordinary",
        n_boot=50,
    )
    assert paired["subjects"] == ["p1", "p2"]
    assert paired["n_paired_primary_patients"] == 2
    assert paired["median_delta"] == pytest.approx(0.3)
    assert paired["n_positive"] == 2



_AXIS_NAMES = ("A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4")


def test_learned_axis_pools_take_the_two_ends_of_the_seed_ensemble_axis():
    axes = {
        "11": np.linspace(-2.0, 2.0, 8),
        "29": np.linspace(-2.1, 1.9, 8),
        "47": np.linspace(-1.9, 2.2, 8),
    }
    pools = learned_axis_source_pools(
        axes, contact_names=_AXIS_NAMES, endpoint_fraction=0.25
    )
    assert pools["source_pool_rule"].startswith("learned_signed_axis")
    assert pools["source_minus_contacts"] == ["A1", "A2"]
    assert pools["source_plus_contacts"] == ["B3", "B4"]
    assert not np.intersect1d(
        pools["source_minus_indices"], pools["source_plus_indices"]
    ).size
    assert pools["min_seed_axis_pairwise_pearson"] > 0.99


def test_learned_axis_pools_are_invariant_to_each_seed_arbitrary_sign():
    """Flipping a seed's coordinate leaves its likelihood unchanged, so the
    ensemble must not depend on which sign that seed happened to land on."""

    axes = {
        "11": np.linspace(-2.0, 2.0, 8),
        "29": np.linspace(-2.1, 1.9, 8),
        "47": np.linspace(-1.9, 2.2, 8),
    }
    reference = learned_axis_source_pools(axes, contact_names=_AXIS_NAMES)
    for flipped_seed in axes:
        perturbed = dict(axes)
        perturbed[flipped_seed] = -np.asarray(axes[flipped_seed])
        observed = learned_axis_source_pools(perturbed, contact_names=_AXIS_NAMES)
        assert observed["source_minus_contacts"] == reference["source_minus_contacts"]
        assert observed["source_plus_contacts"] == reference["source_plus_contacts"]
        np.testing.assert_allclose(
            observed["diffusion_coordinate"], reference["diffusion_coordinate"]
        )
        assert observed["min_seed_axis_pairwise_pearson"] == pytest.approx(
            reference["min_seed_axis_pairwise_pearson"]
        )


def test_learned_axis_pools_report_but_do_not_gate_seed_disagreement():
    disagreeing = {
        "11": np.linspace(-2.0, 2.0, 8),
        "29": np.asarray([0.4, -1.3, 1.1, -0.2, 0.9, -1.8, 0.3, 0.6]),
    }
    pools = learned_axis_source_pools(disagreeing, contact_names=_AXIS_NAMES)
    assert abs(pools["min_seed_axis_pairwise_pearson"]) < 0.9
    assert len(pools["source_minus_indices"]) == 2


def test_field_record_carries_the_rule_it_was_actually_built_from():
    axes = {"11": np.linspace(-1.0, 1.0, 4), "29": np.linspace(-1.1, 0.9, 4)}
    names = ("A1", "A2", "B1", "B2")
    pools = learned_axis_source_pools(axes, contact_names=names, endpoint_fraction=0.25)
    record = build_frozen_subject_field_record(
        subject_id="p1",
        model_name="structured",
        contact_names=names,
        operator=np.eye(4) + 0.1,
        diffusion_result=pools,
        field_minus=[1.0, 0.6, 0.3, 0.0],
        field_plus=[0.0, 0.3, 0.6, 1.0],
        horizon=4,
        checkpoint_sha256_by_seed={"11": "a" * 64, "29": "b" * 64},
        training_split_sha256="c" * 64,
    )
    assert record["source_pool_rule"].startswith("learned_signed_axis")
    assert "first_nontrivial_eigenvalue" not in record
    assert "min_seed_axis_pairwise_pearson" in record
    validate_frozen_subject_field_record(record)
