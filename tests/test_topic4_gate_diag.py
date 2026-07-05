import numpy as np
from src.topic4_gate_diag import front_lead_by_axis, clamp_check


def test_front_lead_I_ahead_of_E():
    # 3 axial bins; in the middle bin I fires at t-index 5, E front arrives at t-index 9 -> I leads
    T, NE, NI = 20, 6, 6
    dt = 0.5
    E = np.zeros((T, NE), bool)
    I = np.zeros((T, NI), bool)
    along_E = np.array([0, 0, 5, 5, 10, 10.0])
    along_I = np.array([0, 0, 5, 5, 10, 10.0])
    I[5, 2] = True      # I in middle bin fires early
    E[9, 2] = True      # E front reaches middle bin later
    out = front_lead_by_axis(E, I, along_E, along_I, n_bins=3, dt=dt)
    j = np.argmin(np.abs(np.array(out["bin_along"]) - 5.0))
    assert out["I_lead_ms"][j] > 0   # I ahead of E in that bin


def test_clamp_check_shunting_gates_axial_front():
    # axial-front E cell: I_E=30 (strong drive), I_I=10. current target 20>=v_th=18 (fires);
    # shunting target (30+10*11)/(1+10)=12.7<18 (gated).
    I_E = np.array([30.0])
    I_I = np.array([10.0])
    along_E = np.array([12.0])
    out = clamp_check(I_E, I_I, along_E, np.array([1.0, 0.0]), e_gaba=11.0, g_gaba_scale=1.0, v_th=18.0)
    assert out["frac_axial_gated_by_shunt"] == 1.0
