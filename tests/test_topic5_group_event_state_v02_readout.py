"""A0 regression tests for the shared scoring layer and the nested readout.

Two of these are the *instrument sensitivity* layer of the acceptance, not the
engineering layer: ``test_a_state_holding_the_true_latent_is_detected`` shows the
assay can see a slow driver when one exists, and
``test_shuffling_the_latent_removes_the_gain`` shows the same assay returns to
zero when it does not.  Without the pair, a null result cannot be distinguished
from a blind estimator.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.topic5_group_event_state.v02 import readout as R
from src.topic5_group_event_state.v02 import scoring as S
from src.topic5_group_event_state.v02.targets import WindowStats


FAST = R.ReadoutConfig(lambdas=(1e-4, 1e-2, 1.0), max_iter=80)


def _simulate(
    n_anchors: int = 900,
    n_contacts: int = 5,
    n_dims: int = 3,
    latent_effect: float = 1.2,
    seed: int = 0,
) -> tuple[np.ndarray, WindowStats]:
    """Windows whose count and mark are driven by one slow latent ``u``."""

    rng = np.random.default_rng(seed)
    t = np.arange(n_anchors, dtype=np.float64)
    u = np.sin(2 * np.pi * t / 180.0) + 0.4 * np.sin(2 * np.pi * t / 47.0)

    mu = np.exp(2.2 + latent_effect * u)
    count = rng.poisson(mu * rng.gamma(8.0, 1 / 8.0, size=n_anchors)).astype(np.int64)

    logit = (rng.normal(size=n_contacts) * 0.5)[None, :] + latent_effect * u[:, None]
    p = 1.0 / (1.0 + np.exp(-logit))
    sum_part = rng.binomial(np.maximum(count, 0)[:, None], p).astype(np.int64)

    n_valid = count.copy()
    centre = latent_effect * u[:, None] * np.linspace(1.0, 0.2, n_dims)[None, :]
    sum_x = np.zeros((n_anchors, n_dims))
    sum_x2 = np.zeros((n_anchors, n_dims))
    for i in range(n_anchors):
        if n_valid[i] == 0:
            continue
        x = rng.normal(loc=centre[i], scale=1.0, size=(int(n_valid[i]), n_dims))
        sum_x[i] = x.sum(0)
        sum_x2[i] = (x ** 2).sum(0)
    stats = WindowStats(count=count, n_valid_mark=n_valid, sum_participation=sum_part,
                        sum_x=sum_x, sum_x2=sum_x2)
    return u, stats


def _slice(stats: WindowStats, sl: slice) -> WindowStats:
    return WindowStats(
        count=stats.count[sl], n_valid_mark=stats.n_valid_mark[sl],
        sum_participation=stats.sum_participation[sl], sum_x=stats.sum_x[sl],
        sum_x2=stats.sum_x2[sl],
    )


def _run(x: np.ndarray, stats: WindowStats, n_contacts: int, n_dims: int,
         config: R.ReadoutConfig = FAST):
    tr, va, te = slice(0, 600), slice(600, 700), slice(700, None)
    fit = R.fit_readout(x[tr], _slice(stats, tr), x[va], _slice(stats, va),
                        n_contacts=n_contacts, n_dims=n_dims, config=config)
    scores = R.score_readout(fit, x[te], _slice(stats, te),
                             block_slices={"all": slice(0, n_dims)},
                             n_contacts=n_contacts, n_dims=n_dims)
    ref = R.reference_scores(_slice(stats, tr), _slice(stats, te),
                             n_contacts=n_contacts, n_dims=n_dims)
    return fit, scores, ref


# ------------------------------------------------------- exact scoring algebra


def test_gaussian_nll_from_moments_equals_the_per_event_sum() -> None:
    """The moment form is the dense sum, not an approximation of it."""

    rng = np.random.default_rng(0)
    x = rng.normal(size=(37, 4))
    mu = torch.as_tensor(rng.normal(size=(1, 4)))
    ls = torch.as_tensor(rng.normal(size=(1, 4)) * 0.2)
    got = S.gaussian_nll_from_moments(
        torch.tensor([37.0]), torch.as_tensor(x.sum(0))[None, :],
        torch.as_tensor((x ** 2).sum(0))[None, :], mu, ls,
    )
    sigma = torch.exp(ls)
    want = (0.5 * np.log(2 * np.pi) + ls + 0.5 * ((torch.as_tensor(x) - mu) / sigma) ** 2).sum(0)
    assert torch.allclose(got[0], want, atol=1e-9)


def test_participation_nll_from_counts_equals_the_per_event_sum() -> None:
    rng = np.random.default_rng(1)
    y = (rng.random((23, 6)) < 0.4).astype(np.float64)
    logit = torch.as_tensor(rng.normal(size=(1, 6)))
    got = S.participation_nll_from_counts(
        torch.as_tensor(y.sum(0))[None, :], torch.tensor([23.0]), logit
    )
    want = torch.nn.functional.binary_cross_entropy_with_logits(
        logit.expand(23, 6), torch.as_tensor(y), reduction="none"
    ).sum(0)
    assert torch.allclose(got[0], want, atol=1e-9)


# ------------------------------------------------------- C13 free-intercept guard


def test_a_pure_time_ramp_and_a_saturating_jump_produce_no_gain() -> None:
    """C13: a monotone ramp and a saturated step are free intercepts, not signal."""

    _u, stats = _simulate(seed=2, latent_effect=0.0)
    n = stats.count.size
    t = np.arange(n, dtype=np.float64)
    ramp = ((t - t.mean()) / t.std())[:, None]
    jump = (t > 50).astype(np.float64)[:, None]          # saturates immediately
    x_base = np.ones((n, 1))
    _f0, s0, ref = _run(x_base, stats, 5, 3)
    _f1, s1, _ = _run(np.hstack([x_base, ramp, jump]), stats, 5, 3)
    gain = R.gain_table(s1, s0)
    for key in ("count", "participation", "continuous"):
        assert gain[key] < 0.05, f"{key} gained {gain[key]:.4f} from a constant"
        assert ref[key].nll_per_unit > 0


def test_a_fit_far_worse_than_the_intercept_floor_is_flagged() -> None:
    """C13/EI 2: an unusable fit must not be reported as a weak negative."""

    _u, stats = _simulate(seed=3, latent_effect=0.0)
    good = R.reference_scores(stats, stats, n_contacts=5, n_dims=3)
    broken = {k: S.ScoreResult(v.nll_per_unit + 5.0, v.n_units, v.n_anchors)
              for k, v in good.items()}
    flags = R.estimability(broken, good)
    assert set(flags.values()) == {"not_estimable"}
    assert set(R.estimability(good, good).values()) == {"ok"}


def test_ridge_selection_is_invariant_to_the_scale_of_the_features() -> None:
    """EI 2: the ridge must act on a Gram-normalised operator, not on raw units."""

    u, stats = _simulate(seed=4)
    x = np.stack([np.ones_like(u), u], axis=1)
    _f_a, s_a, _ = _run(x, stats, 5, 3)
    _f_b, s_b, _ = _run(x * 100.0, stats, 5, 3)
    for key in ("count", "participation", "continuous"):
        assert s_a[key].nll_per_unit == pytest.approx(s_b[key].nll_per_unit, abs=1e-6)


# ------------------------------------------------------- instrument sensitivity


def test_a_state_holding_the_true_latent_is_detected() -> None:
    """Positive control: the assay must see a slow driver that really is there."""

    u, stats = _simulate(seed=5, latent_effect=1.2)
    n = u.size
    base = np.ones((n, 1))
    with_state = np.stack([np.ones(n), u], axis=1)
    _f0, s0, _ = _run(base, stats, 5, 3)
    _f1, s1, _ = _run(with_state, stats, 5, 3)
    gain = R.gain_table(s1, s0)
    assert gain["count"] > 0.5
    assert gain["participation"] > 0.02
    assert gain["continuous"] > 0.05


def test_shuffling_the_latent_removes_the_gain() -> None:
    """Negative control on the same instrument: no time alignment, no gain."""

    u, stats = _simulate(seed=5, latent_effect=1.2)
    n = u.size
    shuffled = np.random.default_rng(11).permutation(u)
    _f0, s0, _ = _run(np.ones((n, 1)), stats, 5, 3)
    _f1, s1, _ = _run(np.stack([np.ones(n), shuffled], axis=1), stats, 5, 3)
    gain = R.gain_table(s1, s0)
    for key in ("count", "participation", "continuous"):
        assert gain[key] < 0.05, f"{key} gained {gain[key]:.4f} from a shuffled latent"


# ------------------------------------------------------- C10 block-shift null


def test_block_shift_must_exceed_the_horizon() -> None:
    with pytest.raises(ValueError):
        R.validate_shift_exceeds_horizon(shift_steps=4, grid_seconds=300.0,
                                         horizon_seconds=7200.0)
    R.validate_shift_exceeds_horizon(shift_steps=25, grid_seconds=300.0,
                                     horizon_seconds=7200.0)


def test_block_shift_never_moves_a_state_between_sessions() -> None:
    values = np.arange(10, dtype=np.float64)[:, None]
    session = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    t = np.arange(10, dtype=np.float64)
    out = R.block_circular_shift(values, session, t, shift_steps=1)
    for s in np.unique(session):
        idx = np.flatnonzero(session == s)
        assert set(out[idx, 0].tolist()) == set(values[idx, 0].tolist())
    assert not np.array_equal(out, values)


def test_block_shift_reports_how_many_anchors_can_actually_be_shifted() -> None:
    session = np.array([0, 0, 0, 1, 1, 2])
    usable, total = R.shiftable_sessions(session, shift_steps=2)
    assert (usable, total) == (3, 6)  # only session 0 is longer than the shift
