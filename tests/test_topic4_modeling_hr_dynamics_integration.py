"""Integration tests for hr_dynamics — empirical regime-behavior observations.

These are NOT algebraic invariants. They describe what HR happens to do at
specific parameter regimes. They run during Stage 1 smoke (Task 6) as
empirical observations to be reported in archive, NOT as TDD unit gates.

**Failure semantics (v3 user-return strict catch round 2):**
when the empirical observation does NOT hold, the test calls
`pytest.xfail(reason=...)` to mark itself XFAIL (yellow, not red).
The build never goes red on an empirical boundary mismatch — Task 6
smoke + archive review surfaces the divergence, and the implementer
does NOT adjust HR params to satisfy a hard test. If the observation
holds, the test PASSes normally.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.slow
def test_hr_silent_at_deeply_subthreshold():
    """Empirical observation: at I=-3.0 no noise → HR rests (x.max < 0.5).

    XFAIL (not FAIL) if HR regime boundary differs.
    """
    from src.topic4_modeling.hr_core import HRParams
    from src.topic4_modeling.hr_dynamics import simulate_trajectory
    p = HRParams()
    _, traj = simulate_trajectory(p, I=-3.0, T=200.0, dt=0.05,
                                   sigma_ou=0.0, tau_ou=10.0, seed=0)
    x_max = float(traj[:, 0].max())
    if x_max >= 0.5:
        pytest.xfail(
            f"At I=-3.0 expected silent (x.max<0.5), got x.max={x_max:.3f}. "
            "HR regime boundary at this I differs from prior assumption — "
            "report in Stage 1 archive; do NOT adjust HR params to satisfy."
        )
    # Observation holds → PASS


@pytest.mark.slow
def test_hr_repetitive_at_high_I():
    """Empirical observation: at I=2.0 HR enters spontaneous bursting (>=3 ups).

    XFAIL (not FAIL) if regime boundary differs.
    """
    from src.topic4_modeling.hr_core import HRParams
    from src.topic4_modeling.hr_dynamics import simulate_trajectory
    p = HRParams()
    _, traj = simulate_trajectory(p, I=2.0, T=300.0, dt=0.05,
                                   sigma_ou=0.0, tau_ou=10.0, seed=0)
    x = traj[:, 0]
    ups = int(np.sum((x[:-1] < 1.0) & (x[1:] >= 1.0)))
    if ups < 3:
        pytest.xfail(
            f"At I=2.0 expected >=3 burst rises through x=1.0, got {ups}. "
            "HR regime boundary differs from prior assumption — "
            "report in Stage 1 archive."
        )
    # Observation holds → PASS


@pytest.mark.slow
def test_hr_higher_r_yields_more_bursts():
    """Empirical observation: larger r (slow-var rate) → more bursts in T.

    XFAIL (not FAIL) if the monotone trend does NOT hold.
    """
    from src.topic4_modeling.hr_core import HRParams
    from src.topic4_modeling.hr_sweep import evaluate_cell
    slow = evaluate_cell(HRParams(), I=2.0, sigma_ou=0.0, tau_ou=10.0,
                          r_override=0.003, T=1000.0, dt=0.05, seed=0)
    fast = evaluate_cell(HRParams(), I=2.0, sigma_ou=0.0, tau_ou=10.0,
                          r_override=0.012, T=1000.0, dt=0.05, seed=0)
    if fast["n_bursts"] <= slow["n_bursts"]:
        pytest.xfail(
            f"Expected fast r (0.012) → more bursts than slow r (0.003); "
            f"got fast n_bursts={fast['n_bursts']}, slow n_bursts={slow['n_bursts']}. "
            "Report in Stage 1 archive if regime boundary differs."
        )
    # Observation holds → PASS
