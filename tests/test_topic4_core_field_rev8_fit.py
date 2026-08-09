"""Scientific and resume guards for the rev8 KMeans-assisted optimizer."""
import numpy as np

from scripts.run_topic4_core_field_stage3_rev8_kmeans_fit import (
    FINAL_CONFIRM_SEED_POOL,
    MODE_LOSS_WEIGHT,
    SELECTION_SEED_POOL,
    TRAIN_SEED_POOL,
    rev8_candidate_fitness,
    score_candidate,
)
from scripts.run_topic4_core_field_stage3_rev8_confirm import _representative_events
from src.topic4_core_field_profile import (
    fit_profile_modes,
    fit_rank_curve_reference,
    rank_curve_table,
)


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


def test_fit_selection_and_final_network_pools_are_disjoint():
    assert not (set(TRAIN_SEED_POOL) & set(SELECTION_SEED_POOL))
    assert not (set(TRAIN_SEED_POOL) & set(FINAL_CONFIRM_SEED_POOL))
    assert not (set(SELECTION_SEED_POOL) & set(FINAL_CONFIRM_SEED_POOL))


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
