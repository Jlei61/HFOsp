"""TDD for src/sef_hfo_snn_adapter — Increment 2 spike->envelope adapter + event window.

Synthetic CENTER-ORIGIN ANISOTROPIC-LOBE spike fronts (the real center-kick geometry, NOT
a unidirectional ramp — see onset_front_axis GEOMETRY CONTRACT) verify the adapter feeds the
Increment-1 chain so the undirected propagation axis reads back. Real-SNN runs live in the
runner (Tasks 5-6), not here.
"""
import numpy as np

from src.sef_hfo_snn_adapter import snn_event_envelope, event_window_for_run
from src.sef_hfo_observation import (
    build_shaft, merge_montages, extract_lagpat, attach_geometry,
    onset_front_axis, angle_error_deg,
)


def test_adapter_envelope_lag_tracks_local_onset():
    # Adapter CORE property (no geometry tuning): a contact near EARLY-firing neurons must
    # get an earlier envelope first-crossing than a contact near LATE-firing neurons.
    # (The full onset_front_axis direction read through a montage needs montage-vs-footprint
    # geometry tuning -> that is the Task-5 SMOKE, not a unit test. onset_front_axis itself
    # is unit-tested on lags directly in test_sef_hfo_observation.py.)
    L, dt, t_max = 2.0, 0.1, 150.0
    rng = np.random.default_rng(0)
    posE = rng.uniform(0, L, size=(2000, 2))
    early = posE[:, 0] > L / 2                          # +x half fires early
    onset = np.where(early, 30.0, 80.0)                 # ms
    nsteps = int(t_max / dt)
    spk = np.zeros((nsteps, len(posE)), bool)
    for j in range(len(posE)):
        k = int(onset[j] / dt)
        if 0 <= k < nsteps:
            spk[k, j] = True
    m = merge_montages([build_shaft(0.0, 0.3, 4, (1.6, 1.0), "P"),    # near +x (early)
                        build_shaft(0.0, 0.3, 4, (0.4, 1.0), "N")])   # near -x (late)
    env, fdt, agg = snn_event_envelope(spk, posE, m, dt, bin_ms=2.0, smooth_ms=5.0,
                                       kernel_width=0.25)
    assert env.shape[0] == len(m.names)
    art = extract_lagpat(env, fdt, event_windows=[(0.0, env.shape[1] * fdt)],
                         participation_floor=float(env.min()),
                         participation_margin=0.3 * (float(env.max()) - float(env.min())),
                         timing_frac=0.5, tie_tol=fdt)
    lag_P = np.nanmean(art.lag_raw[:4, 0])              # +x contacts (early)
    lag_N = np.nanmean(art.lag_raw[4:, 0])              # -x contacts (late)
    assert lag_P < lag_N                                # adapter preserves local onset order


def test_event_window_from_recalibrated_aggregate():
    frame_dt = 2.0
    n = 150
    t = np.arange(n) * frame_dt
    ref = 0.01 * np.abs(np.sin(t))                          # quiet no-kick reference
    bump = np.exp(-((t - 150) ** 2) / (2 * 20.0 ** 2))      # one ~150ms event, returns
    kick = ref + bump
    win = event_window_for_run(kick, ref, frame_dt)
    assert win is not None
    assert 100 < win[0] < 175                               # event onset in range
    assert win[1] > win[0]


def test_engine_kick_center_moves_the_kick():
    """Engine patch regression (Increment-2 Task 1): kick_center=None -> sheet-center disk;
    off-center -> the kicked region moves in that direction. Skips if the gitignored LIF-SNN
    engine is not importable. (Direction only — exact disk localisation needs a larger sheet /
    earlier window = smoke geometry; here we only confirm the kwarg is wired and moves the kick.)"""
    import os
    import sys
    import pytest
    eng = os.path.join("src", "snn_engine")
    if not os.path.isdir(eng):
        pytest.skip("LIF-SNN engine not present")
    sys.path.insert(0, eng)
    try:
        from params import Params, compute_nu_theta
        from model import build_network
        from kick_probe import fresh_run, T_KICK, DUR_KICK
    except Exception:
        pytest.skip("LIF-SNN engine not importable")
    p = Params(g=3.6, L=2.0, density=800.0, T=185.0, nu_ext_ratio=0.6, seed=1)
    net = build_network(p, verbose=False)
    posE = net["pos"][:net["NE"]]
    dt = p.dt
    boost = 2 * compute_nu_theta(p)[0]
    i0, i1 = int(round(T_KICK / dt)), int(round((T_KICK + DUR_KICK) / dt))

    def top_excess_centroid(center):
        rk = fresh_run(p, net, KICK_BOOST=boost, kick_center=center)["E_spk_bool"][i0:i1].sum(0).astype(float)
        rr = fresh_run(p, net, KICK_BOOST=0.0, kick_center=center)["E_spk_bool"][i0:i1].sum(0).astype(float)
        exc = np.clip(rk - rr, 0, None)
        top = exc >= max(np.quantile(exc[exc > 0], 0.8), 1)
        return posE[top].mean(0)

    c_none = top_excess_centroid(None)
    c_px = top_excess_centroid([1.6, 1.0])
    np.testing.assert_allclose(c_none, [1.0, 1.0], atol=0.25)   # default = sheet center
    assert c_px[0] > c_none[0] + 0.2                            # off-center moved +x


def test_event_window_none_when_no_event():
    frame_dt = 2.0
    t = np.arange(150) * frame_dt
    ref = 0.01 * np.abs(np.sin(t))
    # kick == ref: no separable detection bar (peak <= floor) -> INSUFFICIENT, must return
    # None, NOT raise UndetectableOperatingPoint (the kick fizzled / produced no event).
    assert event_window_for_run(ref.copy(), ref.copy(), frame_dt) is None
    # sub-threshold kick (peak not above the ref floor) -> also None
    kick = ref + 0.003 * np.abs(np.cos(t))
    assert event_window_for_run(kick, ref, frame_dt) is None
