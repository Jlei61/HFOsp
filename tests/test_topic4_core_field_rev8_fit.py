"""Scientific and resume guards for the rev8 KMeans-assisted optimizer."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.run_topic4_core_field_stage3_rev8_kmeans_fit import (
    FINAL_CONFIRM_SEED_POOL,
    MODE_LOSS_WEIGHT,
    REV81_MODE_LOSS_WEIGHT,
    SELECTION_SEED_POOL,
    TRAIN_SEED_POOL,
    WORKER_CONTEXT,
    rev8_candidate_fitness,
    score_candidate,
    training_elite_warm_start,
    validate_objective_configuration,
)
from scripts.run_topic4_core_field_stage3_rev8_confirm import (
    _candidate_rows,
    _representative_events,
    _selection_eligible_rows,
)
from scripts.audit_topic4_data_driven_core_mechanism import _candidate_payload
from src.topic4_core_field_profile import (
    fit_profile_modes,
    fit_rank_curve_reference,
    rank_curve_table,
)
from src.topic4_core_field_stage3 import n_free


AX = {f"C{i}": float(x) for i, x in enumerate(np.linspace(-8.0, 8.0, 11))}


def _event(sign=1, seed=0):
    rng = np.random.default_rng(seed)
    names = list(AX)
    ranks = {
        name: float(sign * AX[name] + 0.1 * rng.normal()) for name in names
    }
    return {"ranks": ranks, "n_part": len(names)}


def _reference_and_modes():
    events = ([_event(+1, i) for i in range(50)]
              + [_event(-1, 100 + i) for i in range(50)])
    curves = rank_curve_table(events, AX)
    reference = fit_rank_curve_reference(
        curves, n_components=4, n_reference=80, n_projections=12, seed=3)
    modes = fit_profile_modes(curves, reference, seed=4)
    return reference, modes["prototypes"]


def test_distance_feasible_but_mode_undersupported_is_not_fully_eligible():
    mode = {"status": "ok", "mode_matrix_loss": 0.01,
            "min_cluster_count": 2, "support_eligible": False}
    key = rev8_candidate_fitness(0.2, mode, 32, 30.0)
    assert key[0] == 2.0


def test_supported_modes_outrank_any_undersupported_mode():
    supported = {"status": "ok", "mode_matrix_loss": 0.9,
                 "min_cluster_count": 8, "support_eligible": True}
    tiny = {"status": "ok", "mode_matrix_loss": 0.0,
            "min_cluster_count": 7, "support_eligible": False}
    assert rev8_candidate_fitness(0.9, supported, 32, 1.0) > \
        rev8_candidate_fitness(0.1, tiny, 32, 100.0)


def test_joint_loss_uses_the_frozen_auxiliary_weight():
    mode = {"status": "ok", "mode_matrix_loss": 0.4,
            "min_cluster_count": 8, "support_eligible": True}
    key = rev8_candidate_fitness(0.6, mode, 32, 1.0)
    assert key[1] == -(0.6 + MODE_LOSS_WEIGHT * 0.4)


def test_rev81_weight_and_sign_tier_prioritize_patient_mode_identity():
    wrong = {"status": "ok", "mode_matrix_loss": 0.1,
             "min_cluster_count": 8, "support_eligible": True,
             "matrix_sign_consistent": False}
    correct = dict(wrong, mode_matrix_loss=0.9, matrix_sign_consistent=True)
    assert rev8_candidate_fitness(
        0.2, correct, 32, 1.0,
        mode_loss_weight=REV81_MODE_LOSS_WEIGHT,
        mode_sign_tier=True,
    ) > rev8_candidate_fitness(
        0.2, wrong, 32, 1.0,
        mode_loss_weight=REV81_MODE_LOSS_WEIGHT,
        mode_sign_tier=True,
    )


def test_rev81_restores_raw_matrix_error_scale():
    mode = {"status": "ok", "mode_matrix_loss": 0.4,
            "min_cluster_count": 8, "support_eligible": True,
            "matrix_sign_consistent": False}
    key = rev8_candidate_fitness(
        0.6, mode, 32, 1.0,
        mode_loss_weight=REV81_MODE_LOSS_WEIGHT)
    assert key[1] == -(0.6 + 2.0 * 0.4)


def test_training_elite_warm_start_uses_new_training_only_scalar(tmp_path):
    checkpoint = tmp_path / "checkpoint.json"
    rows = [
        dict(generation=1, distance=0.20, latent=np.zeros(n_free(3)).tolist(),
             mode=dict(support_eligible=True, mode_matrix_loss=0.60,
                       cluster_counts=[16, 16])),
        dict(generation=2, distance=0.80, latent=np.ones(n_free(3)).tolist(),
             mode=dict(support_eligible=True, mode_matrix_loss=0.10,
                       cluster_counts=[14, 18])),
        dict(generation=3, distance=0.01, latent=np.full(n_free(3), 2.0).tolist(),
             mode=dict(support_eligible=False, mode_matrix_loss=0.01,
                       cluster_counts=[2, 30])),
    ]
    checkpoint.write_text(json.dumps(dict(
        K=3, objective="old_training_objective", history=rows,
        provenance={"git_commit": "abc123"})))
    latent, descriptor = training_elite_warm_start(
        checkpoint, K=3, mode_loss_weight=2.0)
    assert np.array_equal(latent, np.ones(n_free(3)))
    assert descriptor["source_generation"] == 2
    assert descriptor["source_joint_loss"] == pytest.approx(1.0)


def test_selection_rejects_checkpoint_from_another_objective(tmp_path):
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps(dict(
        objective="rev8_old", run_contract={"objective_id": "rev8_old"})))
    with pytest.raises(RuntimeError, match="objective mismatch"):
        _candidate_rows(
            checkpoint, sheet_length=20.0,
            expected_objective="rev8_1_expected")


def test_rev81_objective_id_cannot_hide_old_scoring_contract():
    args = SimpleNamespace(
        objective_id="rev8_1_curve_plus_patient_train_kmeans_weight2_v1",
        initializer_id="old_initializer", mode_loss_weight=0.5,
        mode_sign_tier=False, warm_start_checkpoint="checkpoint.json", K=3,
    )
    with pytest.raises(SystemExit, match="scoring contract mismatch"):
        validate_objective_configuration(args)


def test_rev81_selection_stops_before_heldout_if_sign_is_not_reproduced():
    rows = [
        dict(run_contract={"mode_sign_tier": True},
             selection_metrics={"mode": {"matrix_sign_consistent": False}}),
        dict(run_contract={"mode_sign_tier": True},
             selection_metrics={"mode": {"matrix_sign_consistent": False}}),
    ]
    eligible, required = _selection_eligible_rows(rows)
    assert required is True
    assert eligible == []


def test_mechanism_audit_reads_the_frozen_rev81_candidate():
    payload = _candidate_payload(dict(
        objective_id="rev8.1", candidates=[dict(
            candidate_id="c1", K=3, theta=list(range(17)),
            confirm={"verdict": "FAILED_GATE"})]))
    assert payload["candidate_id"] == "c1"
    assert payload["K"] == 3
    assert payload["theta"].tolist() == list(range(17))
    assert payload["verdict"] == "FAILED_GATE"


def test_fit_selection_and_final_network_pools_are_disjoint():
    assert not (set(TRAIN_SEED_POOL) & set(SELECTION_SEED_POOL))
    assert not (set(TRAIN_SEED_POOL) & set(FINAL_CONFIRM_SEED_POOL))
    assert not (set(SELECTION_SEED_POOL) & set(FINAL_CONFIRM_SEED_POOL))


def test_optimizer_workers_spawn_after_parent_kmeans():
    assert WORKER_CONTEXT.get_start_method() == "spawn"


def test_representative_seed_maximizes_support_for_both_modes():
    labels = np.array([0, 1, 0, 0, 1, 1, 0])
    seed_ids = np.array([801, 801, 802, 802, 802, 802, 803])
    local = np.array([2, 5, 1, 3, 4, 7, 0])
    participants = np.array([6, 8, 7, 9, 6, 10, 11])
    seed, events = _representative_events(
        labels, seed_ids, local, participants, [801, 802, 803])
    assert seed == 802
    assert events == {"0": 3, "1": 7}


def test_score_candidate_recovers_two_patient_like_modes():
    reference, prototypes = _reference_and_modes()
    events = ([_event(+1, i) for i in range(24)]
              + [_event(-1, 100 + i) for i in range(24)])
    raw = [dict(events=events, participant_credit=48.0,
                n_detected=48, max_n_part=11)]
    key, row = score_candidate(raw, AX, reference, prototypes)
    assert key[0] == 3.0
    assert row["mode"]["support_eligible"] is True
    assert row["mode"]["mode_matrix_loss"] < 0.05
    assert row["joint_loss"] < row["distance"] + 0.03
