import numpy as np
import pytest

from src.topic4_core_field_cmaes import CMAES


def test_default_population_follows_the_standard_rule():
    es = CMAES(np.zeros(9), sigma0=1.0, seed=0)
    assert es.popsize == 4 + int(3 * np.log(9))


def test_ask_returns_popsize_candidates_of_the_right_length():
    es = CMAES(np.zeros(5), sigma0=0.5, seed=1)
    xs = es.ask()
    assert len(xs) == es.popsize
    assert all(x.shape == (5,) for x in xs)


def test_ask_is_reproducible_from_the_seed():
    a = CMAES(np.zeros(4), sigma0=0.3, seed=7).ask()
    b = CMAES(np.zeros(4), sigma0=0.3, seed=7).ask()
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_it_minimises_a_sphere():
    """Sanity that the update actually optimises. Keys are larger-is-better, so a
    sphere is fed as its negation."""
    es = CMAES(np.full(4, 3.0), sigma0=1.0, seed=2)
    for _ in range(60):
        xs = es.ask()
        es.tell(xs, [-float(x @ x) for x in xs])
    assert float(es.mean @ es.mean) < 1e-3


def test_it_handles_a_lexicographic_key():
    """The real objective is (n_dir, S_rank): a tuple, compared left to right.
    CMA-ES only needs the ORDER, so tuples must work directly."""
    es = CMAES(np.full(3, 2.0), sigma0=0.8, seed=4)
    for _ in range(40):
        xs = es.ask()
        keys = [(2 if x[0] > 0 else 1, -float(x @ x)) for x in xs]
        es.tell(xs, keys)
    assert es.mean[0] > 0          # driven into the n_dir = 2 tier


def test_tell_rejects_a_mismatched_batch():
    es = CMAES(np.zeros(3), sigma0=1.0, seed=0)
    xs = es.ask()
    with pytest.raises(ValueError):
        es.tell(xs[:-1], [0.0] * len(xs))


def test_state_round_trips_so_a_run_can_resume():
    es = CMAES(np.full(4, 1.0), sigma0=0.7, seed=11)
    for _ in range(5):
        xs = es.ask()
        es.tell(xs, [-float(x @ x) for x in xs])
    state = es.get_state()
    resumed = CMAES.from_state(state)
    a = es.ask()
    b = resumed.ask()
    assert all(np.allclose(x, y) for x, y in zip(a, b))
    assert resumed.generation == es.generation


def test_non_finite_keys_sort_last_rather_than_crashing():
    """An infeasible candidate must not poison the update."""
    es = CMAES(np.zeros(3), sigma0=1.0, seed=3)
    xs = es.ask()
    keys = [(0, float("-inf"))] * (len(xs) - 1) + [(2, 0.5)]
    es.tell(xs, keys)
    assert np.isfinite(es.mean).all()
    assert np.isfinite(es.sigma)
