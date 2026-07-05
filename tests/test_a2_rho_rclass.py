"""M3A-A2 rho coordinate + canonical per-event R-class + bout detection."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.sef_hfo_a2 import compute_rho, event_rclass, detect_bouts  # noqa: E402


def test_rho_full_tank_equals_lgr():
    assert abs(compute_rho(1.0, 1.0, 1.16) - 1.16) < 1e-12
    assert abs(compute_rho(0.5, 1.0, 1.0) - 2.0) < 1e-12          # q_core 0.5 doubles rho
    assert abs(compute_rho(0.5, 0.5, 1.0) - 4.0) < 1e-12


def test_detect_bouts_above_boundary():
    rho = np.array([1.0, 1.0, 1.4, 1.5, 1.4, 1.0, 1.0, 1.5, 1.5, 1.0])
    assert detect_bouts(rho, B=1.35) == [(2, 4), (7, 8)]


def test_rclass_active_peak_is_fraction_and_returned_local_not_R4():
    nb = 4; n_bins = nb * nb
    bin_centers = np.array([[j, i] for i in range(nb) for j in range(nb)], float)   # 2D row-major
    bin_of_cell = np.repeat(np.arange(n_bins), 2)                # 32 E cells, 2 per bin
    NEc = bin_of_cell.size; dt, bin_w, nsteps = 0.1, 5.0, 400     # 40 ms record
    spk = np.zeros((nsteps, NEc), bool); fire_bin = 5
    spk[50:150, bin_of_cell == fire_bin] = True                  # bin 5 fires 5-15 ms then quiet
    nbw = int(bin_w / dt); ntb = nsteps // nbw
    af = spk[:ntb * nbw].reshape(ntb, nbw, NEc).mean(axis=(1, 2))  # active FRACTION per bin
    rcls, m, n_act, src = event_rclass(af, spk, bin_of_cell, n_bins, bin_centers, bin_w,
                                       t_on=5.0, t_off=15.0, dt=dt, foci=[bin_centers[fire_bin]])
    assert src == fire_bin                                        # [P1-4] source = early-activity peak bin
    assert 0.0 <= m["active_peak"] <= 1.0                         # FRACTION (~2/32), NOT a raw count
    assert m["returned"] is True and m["runaway"] is False        # ends at 15 ms << 40 ms record
    assert m["far_ea"] == 0.0                                      # all mass in the source bin
    assert rcls in ("R2", "R3")
    assert set(m) == {"event_detected", "returned", "runaway", "r95_ea", "far_ea",
                      "active_peak", "sustained_front_score"}
