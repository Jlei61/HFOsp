"""TDD for masked (shared-participant) legacy MI.

Topic 0 §3.1 phantom-rank fix, extended to the last unmasked consumer:
``compute_legacy_mi`` scored each event over the *full* finite rank vector,
folding non-participating channels' phantom ranks into the statistic. The
Methods text defines MI over common participating contacts only. These tests
pin the corrected contract:

- primary MI fields are computed over participating channels only;
- the permutation null shuffles ranks *within* each event's participating set;
- the old full-channel behaviour is preserved verbatim under
  ``unmasked_sensitivity`` (evaluator-requested historical sensitivity).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.interictal_propagation import compute_legacy_mi


def _make_all_participating(n_ch: int, n_ev: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ranks = np.stack([rng.permutation(n_ch) for _ in range(n_ev)], axis=1).astype(float)
    bools = np.ones((n_ch, n_ev), dtype=bool)
    return ranks, bools


def test_masked_equals_unmasked_when_all_channels_participate():
    """No non-participating channels => masked and unmasked must coincide."""
    ranks, bools = _make_all_participating(n_ch=5, n_ev=40, seed=0)

    res = compute_legacy_mi(ranks, bools, n_permutations=50, seed=1)

    assert "unmasked_sensitivity" in res
    assert res["mi_mean"] == pytest.approx(res["unmasked_sensitivity"]["mi_mean"], abs=1e-12)
    assert res["permuted_mean_median"] == pytest.approx(
        res["unmasked_sensitivity"]["permuted_mean_median"], abs=1e-12
    )


def test_masked_excludes_nonparticipating_phantom_channels():
    """Participating channels perfectly ordered; a phantom 4th channel must not
    dilute the masked statistic but must dilute the unmasked one."""
    n_ch, n_ev = 4, 30
    ranks = np.zeros((n_ch, n_ev), dtype=float)
    bools = np.zeros((n_ch, n_ev), dtype=bool)
    for i in range(n_ev):
        ranks[:3, i] = [0.0, 1.0, 2.0]   # channels 0,1,2 always in the same order
        bools[:3, i] = True
        ranks[3, i] = 0.0                # phantom finite rank, never participates
        bools[3, i] = False

    res = compute_legacy_mi(ranks, bools, n_permutations=50, seed=1)

    # masked: only channels 0,1,2 -> perfectly concordant with the template.
    assert res["mi_mean"] == pytest.approx(1.0, abs=1e-9)
    # unmasked: phantom channel 3 drags the full-channel statistic below 1.
    assert res["unmasked_sensitivity"]["mi_mean"] < res["mi_mean"]
    assert res["masked"] is True


def test_masked_skips_events_with_fewer_than_two_participants():
    """An event with <2 participating channels cannot define an order and must
    be excluded from the masked mean rather than counted as zero."""
    n_ch, n_ev = 4, 20
    ranks = np.zeros((n_ch, n_ev), dtype=float)
    bools = np.zeros((n_ch, n_ev), dtype=bool)
    for i in range(n_ev):
        ranks[:3, i] = [0.0, 1.0, 2.0]
        bools[:3, i] = True
    # last event: only one participating channel
    ranks[:, -1] = 0.0
    bools[:, -1] = False
    bools[0, -1] = True

    res = compute_legacy_mi(ranks, bools, n_permutations=20, seed=3)

    # 19 well-ordered events, 1 degenerate event dropped -> masked mean still 1.0
    assert res["mi_mean"] == pytest.approx(1.0, abs=1e-9)
    assert res["n_events_scored"] == 19
