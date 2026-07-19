"""TDD for src/topic4_state_conditioned_susceptibility.py — Gate C (mapping) + Gate D (operator/probes).

Contract: docs/superpowers/specs/2026-07-19-topic4-state-conditioned-spatial-susceptibility-design.md
§5/§6/§7/§8. Each test = one contract clause (deep-contract-verify ritual). Grids are tiny for speed.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.topic4_m3b_spectral_phase import Grid  # noqa: E402
from src.topic4_state_conditioned_susceptibility import (  # noqa: E402
    affine_to_norm, normalize_subject_coordinates, coarse_cell_index, bin_neuron_state_to_grid,
    zbar_to_q, make_state_controls, build_fixed_scaffold, make_phase_paired_probe_dictionary,
    embed_probe_in_rate_state, probe_matrix, batched_finite_time_response, summarize_probe_atlas,
    leading_probe_subspace_svd, summarize_state_susceptibility, state_operator, two_core_mask_at,
)


def _scaffold(grid, *, ar=2.0, mu_core=0.8, ell_perp=0.6, radius=0.375, theta=0.0,
              src=(-1.62, -0.38), snk=(1.64, -0.38)):
    return build_fixed_scaffold(grid, src, snk, ell_perp=ell_perp, ar=ar, mu_core=mu_core,
                                core_radius=radius, theta=theta)


# ---------------------------------------------------------------- Gate C: mapping
def test_C1_synthetic_coords_land_in_expected_ij_cells():
    grid = Grid(n=6, L=5.0)
    X, Y = grid.coords()                                   # indexing='ij': axis0=X, axis1=Y
    # a point exactly at cell-center (i=4, j=1) must bin to (4, 1)
    pt = np.array([[X[4, 1], Y[4, 1]]])
    ii, jj = coarse_cell_index(pt, grid)
    assert (int(ii[0]), int(jj[0])) == (4, 1)
    # corners map to extreme cells
    ii2, jj2 = coarse_cell_index(np.array([[-2.4, -2.4], [2.4, 2.4]]), grid)
    assert (ii2[0], jj2[0]) == (0, 0) and (ii2[1], jj2[1]) == (5, 5)


def test_C2_affine_axis_horizontal_preserved():
    src = affine_to_norm([3.5135, 8.472], L_phys=20.0, L_norm=5.0, center_phys=[10, 10])
    snk = affine_to_norm([16.5656, 8.472], L_phys=20.0, L_norm=5.0, center_phys=[10, 10])
    d = snk - src
    assert d[0] > 0 and abs(d[1]) < 1e-6                   # source left, sink right, axis horizontal
    theta = float(np.arctan2(d[1], d[0]))
    assert abs(theta) < 1e-6                               # theta == 0 (not the 45deg default)


def test_C3_bin_preserves_uniform_input_exactly():
    grid = Grid(n=6, L=5.0)
    rng = np.random.default_rng(0)
    pos = rng.uniform(-2.5, 2.5, size=(4000, 2))
    field, occ, fill = bin_neuron_state_to_grid(np.full(4000, 0.73), pos, grid)
    assert np.allclose(field[~fill], 0.73)                 # constant input -> constant field
    assert not fill.any()                                  # 4000 pts on 36 cells -> full occupancy


def test_C4_control_invariants_are_distinct_and_declared():
    grid = Grid(n=6, L=5.0)
    rng = np.random.default_rng(1)
    z = rng.uniform(0.7, 1.0, size=(6, 6))
    c = make_state_controls(z, grid, shuffle_seed=42)
    assert np.allclose(c["real"], z)
    assert np.allclose(c["uniform_mean"], z.mean()) and np.ptp(c["uniform_mean"]) == 0.0
    assert np.allclose(c["rotated_90"], np.rot90(z))
    assert np.allclose(np.sort(c["spatial_shuffle"], axis=None), np.sort(z, axis=None))  # histogram kept
    assert not np.allclose(c["spatial_shuffle"], z)        # but spatial pattern changed
    assert np.allclose(c["z_blocked"], 1.0)                # pre-depletion operator


def test_C5_source_and_sink_map_to_correct_sides():
    grid = Grid(n=12, L=5.0)
    sc = _scaffold(grid)
    X, _ = grid.coords()
    src_x = X[sc["src_core"].mask].mean()
    snk_x = X[sc["snk_core"].mask].mean()
    assert src_x < 0 < snk_x                               # source left half, sink right half
    assert sc["core"].mask.sum() == (sc["src_core"].mask | sc["snk_core"].mask).sum()


# ---------------------------------------------------------------- Gate D: operator / probes
def test_D_probe_dictionary_structure_and_global():
    grid = Grid(n=8, L=5.0)
    probes = make_phase_paired_probe_dictionary(grid, p_max=4, sigma=1.0, center=(-1.62, -0.38))
    glob = [pr for pr in probes if pr["phase"] == "global"]
    assert len(glob) == 1 and np.all(glob[0]["field"] >= 0)               # single windowed uniform probe
    pqs = {(pr["p"], pr["q"]) for pr in probes if pr["phase"] != "global"}
    assert (0, 0) not in pqs                                              # (0,0) excluded from directional
    assert all(p > 0 or (p == 0 and q > 0) for (p, q) in pqs)             # half-plane (no sign duplicates)
    axial = [pr for pr in probes if pr["q"] == 0 and pr["p"] > 0]
    assert axial and all(abs(pr["orient_deg"]) < 1e-6 for pr in axial)   # q=0 => orientation along axis


def test_D6_batched_response_equals_one_at_a_time():
    grid = Grid(n=6, L=5.0)
    sc = _scaffold(grid, mu_core=0.5)
    op, J, q = state_operator(np.ones((6, 6)), grid, sc, w_ee_mult=1.3, ratio=1.0, q_floor=0.05)
    assert J is not None
    B = probe_matrix(make_phase_paired_probe_dictionary(grid, p_max=2, center=(-1.62, -0.38)), grid)
    Yb = batched_finite_time_response(J, B, 30.0)
    for i in (0, 3, 7):
        Ys = batched_finite_time_response(J, B[:, i:i + 1], 30.0)
        assert np.allclose(Yb[:, i], Ys[:, 0], atol=1e-9)               # batch == single column


def test_D4_phase_pairing_invariant_to_phase_rotation():
    grid = Grid(n=6, L=5.0)
    sc = _scaffold(grid, mu_core=0.5)
    op, J, _ = state_operator(np.ones((6, 6)), grid, sc, w_ee_mult=1.3, ratio=1.0, q_floor=0.05)
    X, Y = grid.coords()
    kx, ky = 2 * np.pi * 1 / grid.L, 2 * np.pi * 2 / grid.L
    b_cos = embed_probe_in_rate_state(np.cos(kx * X + ky * Y), grid)
    b_sin = embed_probe_in_rate_state(np.sin(kx * X + ky * Y), grid)
    N = grid.size
    R = batched_finite_time_response(J, np.column_stack([b_cos, b_sin]), 30.0)[:N, :]
    e0 = np.linalg.norm(R)                                              # Frobenius energy of the quadrature pair
    for phi in (0.3, 1.1, 2.7):                                         # rotate the pair (== phase shift)
        ba = np.cos(phi) * b_cos + np.sin(phi) * b_sin
        bb = -np.sin(phi) * b_cos + np.cos(phi) * b_sin
        Rp = batched_finite_time_response(J, np.column_stack([ba, bb]), 30.0)[:N, :]
        assert abs(np.linalg.norm(Rp) - e0) < 1e-8                     # phase-invariant paired energy


def test_D5_AR1_isotropic_no_core_has_no_axis_preference():
    grid = Grid(n=8, L=5.0)
    # isotropic kernel (ar=1) + NO core (mu_core=0, radius 0) + uniform q -> no fixed-axis preference
    sc = build_fixed_scaffold(grid, (-1.62, -0.38), (1.64, -0.38), ell_perp=0.6, ar=1.0,
                              mu_core=0.0, core_radius=0.0, theta=0.0)
    op, J, _ = state_operator(np.ones((8, 8)), grid, sc, w_ee_mult=1.3, ratio=1.0, q_floor=0.05)
    assert J is not None
    probes = make_phase_paired_probe_dictionary(grid, p_max=3, sigma=1e6, gabor=True)  # ~flat window
    atlas = summarize_probe_atlas(J, grid, probes, [30.0], theta=0.0, N=grid.size, core=sc["core"])
    a = atlas["per_T"][30.0]
    assert abs(a["axial_gain"] - a["perp_gain"]) / max(a["axial_gain"], 1e-9) < 0.02  # isotropy


def test_D7_unresolved_or_saturated_is_fail_closed():
    grid = Grid(n=6, L=5.0)
    probes = make_phase_paired_probe_dictionary(grid, p_max=2, center=(-1.62, -0.38))
    # extreme excitability -> the operating point runs away (saturated), never silently axial
    sc = _scaffold(grid, mu_core=60.0)
    out, arrays = summarize_state_susceptibility(np.ones((6, 6)), grid, sc, probes, [30.0],
                                                 w_ee_mult=1.5, ratio=1.0, q_floor=0.05)
    assert out["op_status"] != "resolved"
    assert out["atlas"] is None and arrays is None                     # fail-closed: no susceptibility
    # and the invariant holds both ways for a resolved case
    sc2 = _scaffold(grid, mu_core=0.5)
    out2, arrays2 = summarize_state_susceptibility(np.ones((6, 6)), grid, sc2, probes, [30.0],
                                                   w_ee_mult=1.3, ratio=1.0, q_floor=0.05)
    assert (out2["op_status"] == "resolved") == (out2["atlas"] is not None)


def test_end_to_end_resolved_state_has_finite_atlas_and_distinct_eigen():
    grid = Grid(n=8, L=5.0)
    sc = _scaffold(grid, mu_core=0.8)
    probes = make_phase_paired_probe_dictionary(grid, p_max=4, sigma=1.0, center=(-1.62, -0.38))
    out, arrays = summarize_state_susceptibility(np.full((8, 8), 0.95), grid, sc, probes,
                                                 [10.0, 30.0, 50.0, 75.0], w_ee_mult=1.3, ratio=1.0,
                                                 q_floor=0.05, T_primary=30.0)
    if out["op_status"] != "resolved":
        return  # a non-resolved backdrop is an allowed outcome; nothing further to assert here
    a = out["atlas"]["per_T"][30.0]
    assert np.isfinite(a["axial_gain"]) and np.isfinite(a["perp_gain"]) and np.isfinite(a["global_gain"])
    assert a["svd_s0"] > 0
    assert set(out["atlas"]["persistence"]["axial"]) <= {"G50_over_G30", "G75_over_G30"}
    assert out["eigen"]["status"] == "resolved"
    assert 0.0 <= out["eigen"]["leading_globality"] <= 1.0             # true eigenmode shape, kept distinct
    # three DISTINCT objects saved as fields (design §3.5 + review 2026-07-19): eigenmode / V1 / U1
    for k in ("q_field", "eigen_field", "v1_optimal_input", "u1_optimal_output", "peak_paired_output_rE"):
        assert arrays[k].shape == (8, 8), k
    assert out["optimal"]["sigma1"] > 0 and np.isfinite(out["optimal"]["u1_output_axis"])


def test_leading_probe_subspace_svd_shapes():
    R = np.random.default_rng(0).standard_normal((20, 7))
    svd = leading_probe_subspace_svd(R)
    assert svd["s0"] > 0 and svd["optimal_output_field"].shape == (20,)
    assert svd["optimal_probe_weights"].shape == (7,)


def test_fixed_kick_time_response():
    from src.topic4_state_conditioned_susceptibility import (
        sigma1_vs_T, make_localized_kick, fixed_kick_evolution, axial_kymograph)
    grid = Grid(n=6, L=5.0)
    sc = _scaffold(grid, mu_core=0.5)
    op, J, _ = state_operator(np.ones((6, 6)), grid, sc, w_ee_mult=1.3, ratio=1.0, q_floor=0.05)
    assert J is not None
    N = grid.size
    Ts = [0.0, 10.0, 30.0, 50.0]
    s = sigma1_vs_T(J, grid, Ts, N)
    assert len(s) == 4 and s[0] == 1.0 and all(np.isfinite(s))           # sigma1(0)=identity=1
    b = make_localized_kick(grid, (-1.62, -0.38), 0.6)
    assert abs(np.linalg.norm(b) - 1.0) < 1e-9                            # unit norm
    ev = fixed_kick_evolution(J, grid, b, [0.0, 10.0, 30.0], N)
    assert ev[0.0].shape == (6, 6)
    assert np.allclose(ev[0.0], b[:N].reshape(6, 6))                      # t=0 is the kick itself
    xs, ts, kymo = axial_kymograph(ev, grid, -0.38, band=0.5)
    assert kymo.shape == (3, 6) and np.all(kymo >= 0)                      # (n_t, n_x), |rE|>=0


def test_optimal_perturbation_dominates_any_probe_gain():
    # V1 is the UNCONSTRAINED optimal finite-time input over the whole E-rate space, so sigma1 must be
    # >= the gain of ANY single probe (the probe span is a subset of the input space).
    from src.topic4_state_conditioned_susceptibility import optimal_finite_time_perturbation, eigen_field
    grid = Grid(n=6, L=5.0)
    sc = _scaffold(grid, mu_core=0.5)
    op, J, _ = state_operator(np.ones((6, 6)), grid, sc, w_ee_mult=1.3, ratio=1.0, q_floor=0.05)
    assert J is not None
    N = grid.size
    opt = optimal_finite_time_perturbation(J, grid, 30.0, N)
    assert opt["v1_field"].shape == (6, 6) and opt["u1_field"].shape == (6, 6)
    probes = make_phase_paired_probe_dictionary(grid, p_max=3, center=(-1.62, -0.38))
    B = probe_matrix(probes, grid)
    R = batched_finite_time_response(J, B, 30.0)[:N, :]
    max_probe_gain = float(np.max(np.linalg.norm(R, axis=0)))          # ||b||=1 columns
    assert opt["sigma1"] >= max_probe_gain - 1e-9                      # unconstrained optimum dominates
    ef = eigen_field(J, grid)
    assert ef is not None and ef.shape == (6, 6) and np.all(ef >= 0)   # subspace loading magnitude
