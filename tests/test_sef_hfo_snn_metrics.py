import numpy as np

from src.sef_hfo_snn_metrics import onset_times, onset_axis, peak_active_fraction


def test_onset_axis_recovers_linear_wave_direction():
    rng = np.random.default_rng(0)
    NE = 400
    posE = rng.uniform(0, 3, size=(NE, 2))
    dt, t_kick = 0.1, 150.0
    nsteps = int(300 / dt)
    spk = np.zeros((nsteps, NE), bool)
    # onset increases along +x: neuron at x fires at t_kick + 20*x ms
    for i in range(NE):
        ti = int((t_kick + 20.0 * posE[i, 0]) / dt)
        if ti < nsteps:
            spk[ti, i] = True
    onset = onset_times(spk, dt, t_kick)
    axis = onset_axis(posE, onset, min_n=20)
    ang = np.degrees(np.arctan2(axis[1], axis[0])) % 180.0
    assert min(ang, 180 - ang) < 15.0          # ~along x (0 deg)


def test_peak_active_fraction_counts_distinct():
    dt = 0.1
    spk = np.zeros((100, 10), bool)
    spk[40:45, :6] = True                       # 6/10 distinct in one 5ms bin
    paf = peak_active_fraction(spk, dt, 0.0, 10.0, bin_ms=5.0)
    assert abs(paf - 0.6) < 1e-9
