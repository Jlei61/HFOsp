"""TDD tests for src/sef_hfo_rate_adapter — Increment-3a rate-model parity adapter.

Two pure helpers that let the LIF rate field (src.sef_hfo_lif.integrate_lif_field)
feed the SAME virtual-SEEG observation chain as the SNN (Increment 2):

  Task 2: pulse_stim_fn   — a finite disk-pulse stim_fn(t) for integrate_lif_field
                            (CENTER for C-track/iso, OFF-CENTER for kick-track).
                            Drives integrate_lif_field, NOT the old sigmoid
                            sef_hfo_pulse (wrong substrate).
  Task 3: rate_event_envelope — (nsteps,n,n) field stack -> per-contact envelopes
                            via the SAME sample_envelopes + grid_coords used for
                            the SNN. CORE property only (early region crosses
                            before late region); NOT a traveling-wave direction
                            assertion (onset_front_axis perpendicular-front trap).
"""
import numpy as np

from src.sef_hfo_lif import _grid, mean_field, integrate_lif_field
from src.sef_hfo_observation import build_shaft, merge_montages
from src.sef_hfo_rate_adapter import pulse_stim_fn, rate_event_envelope


# ---------------------------------------------------------------------------
# Task 2: pulse_stim_fn — disk placement + time gating
# ---------------------------------------------------------------------------

def test_pulse_stim_fn_disk_placement_and_gating():
    n, L = 32, 12.0
    X, Y = _grid(n, L)
    cx, cy = 3.0, 0.0
    r, amp, t_on, t_off = 1.5, 5.0, 10.0, 30.0
    sf = pulse_stim_fn((cx, cy), r, amp, t_on, t_off, n=n, L=L)

    # OFF before t_on, and at/after t_off (t_off exclusive) -> scalar 0.0
    assert np.isscalar(sf(0.0)) and sf(0.0) == 0.0
    assert np.isscalar(sf(t_on - 0.1)) and sf(t_on - 0.1) == 0.0
    assert np.isscalar(sf(t_off)) and sf(t_off) == 0.0

    # ON at t_mid: an (n,n) field, non-zero ONLY within radius of center
    field = sf(20.0)
    assert field.shape == (n, n)
    inside = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
    assert inside.sum() > 0
    assert np.all(field[inside] == amp)
    assert np.all(field[~inside] == 0.0)

    # off-center placement: active-pixel centroid sits at (cx, cy) within 1 grid cell
    xm = np.average(X, weights=field)
    ym = np.average(Y, weights=field)
    assert abs(xm - cx) <= L / n and abs(ym - cy) <= L / n

    # center vs off-center are genuinely different placements
    sf0 = pulse_stim_fn((0.0, 0.0), r, amp, t_on, t_off, n=n, L=L)
    assert not np.allclose(sf0(20.0), field)


# ---------------------------------------------------------------------------
# Task 3: rate_event_envelope — CORE property (early region before late region)
# ---------------------------------------------------------------------------

def test_rate_event_envelope_early_before_late():
    """A contact over an EARLY-active region gets an earlier envelope first-crossing
    than a contact over a LATE-active region. This is the adapter's core contract
    (reshape + sample_envelopes alignment), NOT a direction/onset-front claim.
    """
    n, L, nsteps = 32, 12.0, 40
    X, Y = _grid(n, L)
    A_mask = (X + 3.0) ** 2 + Y ** 2 <= 1.5 ** 2   # early region (left)
    B_mask = (X - 3.0) ** 2 + Y ** 2 <= 1.5 ** 2   # late region (right)
    base = 0.01
    frames = np.full((nsteps, n, n), base)
    for t in range(nsteps):
        if t >= 5:
            frames[t][A_mask] = 1.0
        if t >= 20:
            frames[t][B_mask] = 1.0

    m = merge_montages([
        build_shaft(0.0, 1.0, 1, origin=(-3.0, 0.0), name_prefix="A"),  # over early region
        build_shaft(0.0, 1.0, 1, origin=(3.0, 0.0), name_prefix="B"),   # over late region
    ])
    env = rate_event_envelope(frames, n, L, m, kernel_width=0.6)
    assert env.shape == (2, nsteps)

    def first_cross(e):
        thr = e.min() + 0.5 * (e.max() - e.min())
        idx = np.where(e > thr)[0]
        return int(idx[0]) if idx.size else nsteps

    assert first_cross(env[0]) < first_cross(env[1]), (
        f"early-region contact first-crossing {first_cross(env[0])} not before "
        f"late-region {first_cross(env[1])}"
    )


def test_rate_event_envelope_wires_to_real_integrator():
    """Wiring guard against the REAL integrator: pulse_stim_fn -> integrate_lif_field
    (return_frames) -> rate_event_envelope returns finite envelopes of the right
    shape (catches reshape / grid-alignment regressions). No direction assertion.
    """
    op = mean_field(1.0)
    n, L = 24, 12.0
    sf = pulse_stim_fn((-3.0, 0.0), radius=2.0, amp=8.0, t_on=0.0, t_off=20.0, n=n, L=L)
    res = integrate_lif_field(op, sf, dt=0.5, t_max=20.0, n=n, L=L, return_frames=True)
    frames = res[-1]
    assert frames.shape == (40, n, n)

    m = merge_montages([
        build_shaft(np.deg2rad(10.0), 3.5, 5, origin=(0.0, 0.0), name_prefix="A"),
        build_shaft(np.deg2rad(100.0), 3.5, 5, origin=(0.0, 0.0), name_prefix="B"),
    ])
    env = rate_event_envelope(frames, n, L, m, kernel_width=1.0)
    assert env.shape == (10, 40)
    assert np.all(np.isfinite(env))
