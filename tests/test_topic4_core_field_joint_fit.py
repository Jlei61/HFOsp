"""Convergence guards for the rev6 joint-profile optimizer."""
import numpy as np

from scripts.run_topic4_core_field_stage3_joint_fit import (
    INIT_ID,
    OBJECTIVE_ID,
    _initial_latent,
    _resume_mismatches,
    _unique_seed_cache_jobs,
    candidate_fitness,
    score_candidate,
)
from src.topic4_core_field_profile import (fit_rank_curve_reference,
                                           fixed_count_sliced_distance,
                                           rank_curve_table)
from src.topic4_core_field_stage3 import latent_to_theta, n_free, unpack


AX = {f"C{i}": float(x) for i, x in enumerate(np.linspace(-8.0, 8.0, 11))}


def _event(sign=1, noise=0.0, seed=0, n=11):
    rng = np.random.default_rng(seed)
    names = list(AX)[:n]
    ranks = {name: float(sign * AX[name] + noise * rng.normal()) for name in names}
    return {"ranks": ranks, "n_part": n}


def test_any_feasible_candidate_outranks_the_dead_zone():
    feasible = candidate_fitness(0.8, 20, 20.0)
    almost = candidate_fitness(float("nan"), 19, 1000.0)
    assert feasible > almost


def test_dead_zone_is_graded_by_usable_events_then_participant_credit():
    assert candidate_fitness(float("nan"), 5, 1.0) > \
        candidate_fitness(float("nan"), 4, 100.0)
    assert candidate_fitness(float("nan"), 5, 2.0) > \
        candidate_fitness(float("nan"), 5, 1.0)


def test_joint_candidate_scoring_keeps_near_readable_diagnostics():
    train = [_event(+1, 0.5, seed) for seed in range(30)] + \
        [_event(-1, 0.5, 100 + seed) for seed in range(30)]
    ref = fit_rank_curve_reference(rank_curve_table(train, AX),
                                   n_components=4, n_reference=50,
                                   n_projections=12, seed=1)
    raw = [dict(events=[_event(+1, 0.4, seed) for seed in range(6)],
                participant_credit=5.5, n_detected=7, max_n_part=11)]
    key, row = score_candidate(raw, AX, ref, min_events=5)
    assert key[0] == 1.0
    assert row["distance"] is not None
    assert row["n_usable"] == 6
    assert row["n_objective"] == 5
    assert row["participant_credit"] == 5.5
    assert row["n_detected"] == 7
    curves = rank_curve_table(raw[0]["events"], AX)
    assert row["distance"] == fixed_count_sliced_distance(curves, ref, 5)


def test_each_restart_starts_in_standardized_coordinates_not_physical_units():
    for K in (1, 2, 3):
        z = _initial_latent(K, restart=2)
        assert z.shape == (n_free(K),)
        assert np.max(np.abs(z)) < 3.0
        theta = latent_to_theta(z, K)
        components = unpack(theta, K)
        assert len(components) == K
        assert all(0.4 < comp["sigma_par"] < 6.0 for comp in components)


def test_space_filling_initializer_places_components_away_from_midpoint():
    for K in (1, 2, 3):
        for restart in range(3):
            theta = latent_to_theta(_initial_latent(K, restart), K)
            radii = [np.linalg.norm(comp["center"] - np.array([10.0, 10.0]))
                     for comp in unpack(theta, K)]
            assert all(4.5 < radius < 6.6 for radius in radii)


def test_space_filling_initializer_is_isotropic_and_restart_rotated():
    centres = []
    for restart in range(3):
        theta = latent_to_theta(_initial_latent(3, restart), 3)
        components = unpack(theta, 3)
        centres.append(np.asarray([comp["center"] for comp in components]))
        assert all(abs(comp["sigma_par"] - comp["sigma_perp"]) < 0.2
                   for comp in components)
    assert not np.allclose(centres[0], centres[1])


def test_resume_rejects_legacy_and_any_numeric_contract_drift():
    expected = dict(objective_id=OBJECTIVE_ID, initializer_id=INIT_ID,
                    K=3, restart=0, popsize=16, seeds_per_candidate=2,
                    min_events=20, reference_sha256="ref", config_checksum="cfg",
                    numeric_contract_sha256="code")
    assert _resume_mismatches({}, expected)
    assert _resume_mismatches({"run_contract": dict(expected, min_events=18)}, expected)
    assert _resume_mismatches({"run_contract": dict(
        expected, numeric_contract_sha256="old")}, expected)
    assert _resume_mismatches({"run_contract": expected}, expected) == []


def test_network_prewarm_has_one_job_per_distinct_seed_not_per_candidate():
    cfg = {"engine": {"L": 20.0}}
    jobs = _unique_seed_cache_jobs([601, 602, 601, 603, 602], cfg, "cache")
    assert [job[0] for job in jobs] == [601, 602, 603]
    assert all(job[1] is cfg and job[2] == "cache" for job in jobs)
