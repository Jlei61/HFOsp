# tests/test_sef_hfo_observation.py
"""TDD for src/sef_hfo_observation — Increment 1 virtual-SEEG observation layer.

All tests are model-free: synthetic analytic sources (src.sef_hfo_toywave) sampled
through virtual contacts, then read back. Locks the spec
docs/superpowers/specs/2026-06-06-sef-hfo-virtual-seeg-observation-layer-design.md §5.
"""
import json

import numpy as np
import pytest

from src.sef_hfo_observation import (
    VirtualMontage,
    build_shaft,
    merge_montages,
    from_real_geometry,
)


def test_build_shaft_geometry_and_names():
    m = build_shaft(angle_rad=0.0, pitch=2.0, n_contacts=5, origin=(0.0, 0.0),
                    name_prefix="A")
    assert m.contacts.shape == (5, 2)
    # 5 contacts, pitch 2mm, centered on origin -> x in [-4,-2,0,2,4], y==0
    np.testing.assert_allclose(m.contacts[:, 0], [-4, -2, 0, 2, 4])
    np.testing.assert_allclose(m.contacts[:, 1], 0.0)
    assert m.names == ["A0", "A1", "A2", "A3", "A4"]
    assert not m.spans_2d()           # single shaft is 1-D


def test_merge_two_nonparallel_shafts_spans_2d():
    a = build_shaft(0.0, 2.0, 4, (0.0, 0.0), "A")
    b = build_shaft(np.pi / 2, 2.0, 4, (1.0, 0.0), "B")
    m = merge_montages([a, b])
    assert m.contacts.shape == (8, 2)
    assert m.names[:4] == ["A0", "A1", "A2", "A3"]
    assert m.names[4:] == ["B0", "B1", "B2", "B3"]
    assert m.spans_2d()               # two non-parallel shafts span the plane


def test_from_real_geometry_loud_fails():
    with pytest.raises(NotImplementedError):
        from_real_geometry(np.zeros((3, 3)))


from src.sef_hfo_observation import grid_coords, sample_envelopes


def test_grid_coords_match_field_grid():
    # mirrors src.sef_hfo_field._grid (indexing="ij", centered, spacing L/n)
    coords = grid_coords(n=4, L=8.0)
    assert coords.shape == (16, 2)
    xs = np.unique(coords[:, 0])
    np.testing.assert_allclose(xs, [-4, -2, 0, 2])   # (arange(4)-2)*2


def test_sample_envelope_tracks_a_localized_blob():
    # A static blob centered at (+10,0); a contact at (+10,0) must read a larger
    # envelope than a contact at (-10,0).
    n, L = 64, 64.0
    coords = grid_coords(n, L)
    X = coords[:, 0]
    Y = coords[:, 1]
    blob = np.exp(-(((X - 10) ** 2 + Y ** 2) / (2 * 4.0 ** 2)))
    frames = blob[None, :].repeat(3, axis=0)          # (3, n*n) constant in time
    m = merge_montages([build_shaft(0.0, 4.0, 1, (10.0, 0.0), "near"),
                        build_shaft(0.0, 4.0, 1, (-10.0, 0.0), "far")])
    env = sample_envelopes(frames, coords, m, kernel_width=3.0)
    assert env.shape == (2, 3)
    assert env[0, 0] > env[1, 0]                      # near-contact reads more


from src.sef_hfo_observation import LagPatArtifact, extract_lagpat


def test_extract_lagpat_orders_by_first_crossing_and_masks_nonparticipants():
    # 3 contacts: A peaks earliest, B later, C never participates (tiny amplitude).
    dt = 0.25
    nt = 400
    t = np.arange(nt) * dt
    envA = np.exp(-((t - 20) ** 2) / (2 * 5.0 ** 2))
    envB = np.exp(-((t - 40) ** 2) / (2 * 5.0 ** 2))
    envC = 0.001 * np.ones(nt)
    env = np.vstack([envA, envB, envC])
    art = extract_lagpat(env, dt, event_windows=[(0.0, nt * dt)],
                         participation_floor=0.0, participation_margin=0.05,
                         timing_frac=0.5, tie_tol=dt)
    assert art.bools[:, 0].tolist() == [True, True, False]
    # A crosses 0.5*peak before B -> rank(A) < rank(B)
    assert art.ranks[0, 0] < art.ranks[1, 0]
    # non-participant C: NaN rank and NaN lag (no phantom finite rank)
    assert np.isnan(art.ranks[2, 0])
    assert np.isnan(art.lag_raw[2, 0])


def test_extract_lagpat_ties_within_tol_get_equal_rank():
    dt = 0.25
    nt = 400
    t = np.arange(nt) * dt
    # A and B identical timing (synchronous) -> tied ranks
    envAB = np.exp(-((t - 30) ** 2) / (2 * 5.0 ** 2))
    env = np.vstack([envAB, envAB])
    art = extract_lagpat(env, dt, event_windows=[(0.0, nt * dt)],
                         participation_floor=0.0, participation_margin=0.05,
                         timing_frac=0.5, tie_tol=dt)
    assert art.ranks[0, 0] == art.ranks[1, 0]


from src.sef_hfo_observation import (
    rank_vs_projection_spearman,
    endpoint_centroid_axis,
    axis_angle_error_deg,
)


def _ranks_along(coords, n_hat):
    # helper: monotone ranks increasing along n_hat (a perfect read-out)
    proj = coords @ n_hat
    order = np.argsort(proj)
    r = np.empty(len(proj))
    r[order] = np.arange(len(proj), dtype=float)
    return r


def test_spearman_is_one_when_ranks_follow_projection():
    coords = np.column_stack([np.linspace(-5, 5, 9), np.zeros(9)])
    n_hat = np.array([1.0, 0.0])
    ranks = _ranks_along(coords, n_hat)
    bools = np.ones(9, bool)
    rho = rank_vs_projection_spearman(ranks, bools, coords, n_hat)
    assert rho > 0.99


def test_endpoint_axis_tracks_imposed_direction_within_tolerance():
    # contacts on a 2D grid; ranks increase along 30deg
    g = np.linspace(-6, 6, 5)
    XX, YY = np.meshgrid(g, g)
    coords = np.column_stack([XX.ravel(), YY.ravel()])     # 25 contacts, 2D
    theta = np.deg2rad(30.0)
    n_hat = np.array([np.cos(theta), np.sin(theta)])
    ranks = _ranks_along(coords, n_hat)
    bools = np.ones(len(coords), bool)
    axis = endpoint_centroid_axis(ranks, bools, coords, k_dir=3, eps_deg=1.0)
    assert axis is not None
    assert axis_angle_error_deg(axis, theta) < 25.0


def test_endpoint_axis_degenerate_returns_none():
    # all participants tied (rank 0) -> early/late centroids coincide -> no-axis
    coords = np.random.default_rng(0).normal(size=(9, 2))
    ranks = np.zeros(9)
    bools = np.ones(9, bool)
    assert endpoint_centroid_axis(ranks, bools, coords, k_dir=3, eps_deg=1.0) is None


def test_endpoint_axis_insufficient_participants_returns_none():
    coords = np.random.default_rng(1).normal(size=(6, 2))   # < 2*k_dir+1 = 7
    ranks = np.arange(6, dtype=float)
    bools = np.ones(6, bool)
    assert endpoint_centroid_axis(ranks, bools, coords, k_dir=3, eps_deg=0.1) is None


from pathlib import Path
from src.sef_hfo_observation import (
    attach_geometry,
    write_legacy_npz,
    write_montage_manifest,
    write_packed_times,
    validate_artifact,
)


def _toy_artifact():
    env = np.array([[0.0, 1.0, 0.2, 0.0],
                    [0.0, 0.2, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0]])
    art = extract_lagpat(env, dt=1.0, event_windows=[(0.0, 4.0)],
                         participation_floor=0.0, participation_margin=0.05,
                         timing_frac=0.5, tie_tol=1.0)
    m = merge_montages([build_shaft(0.0, 2.0, 2, (0.0, 0.0), "A"),
                        build_shaft(np.pi / 2, 2.0, 1, (0.0, 3.0), "B")])
    return attach_geometry(art, m)


def test_validate_artifact_flags_phantom_rank():
    art = _toy_artifact()
    validate_artifact(art)                       # clean: non-participant rank is NaN
    art.ranks[2, 0] = 7.0                         # inject a phantom finite rank
    with pytest.raises(AssertionError):
        validate_artifact(art)


def test_legacy_npz_loads_via_real_loader(tmp_path):
    # Round-trip: write legacy-key files, then the REAL loader must read them.
    from src.interictal_propagation import load_subject_propagation_events
    art = _toy_artifact()
    rec = "synthrec"
    npz = tmp_path / f"{rec}_lagPat_withFreqCent.npz"
    write_legacy_npz(art, npz)
    write_packed_times(art, tmp_path / f"{rec}_packedTimes_withFreqCent.npy")
    write_montage_manifest(art, tmp_path / f"{rec}_montage.json")
    loaded = load_subject_propagation_events(str(tmp_path))
    assert list(loaded["channel_names"]) == art.names
    assert loaded["bools"].shape[0] == len(art.names)
    np.testing.assert_array_equal(loaded["bools"].sum(axis=0) > 0,
                                  art.bools.sum(axis=0) > 0)
    # numeric round-trip + UNIT conversion: on-disk legacy = SECONDS, internal = ms.
    from src.sef_hfo_observation import MS_TO_S
    part = art.bools[:, 0]
    np.testing.assert_allclose(loaded["ranks"][part, 0], art.ranks[part, 0])
    np.testing.assert_allclose(loaded["lag_raw"][part, 0],
                               art.lag_raw[part, 0] * MS_TO_S, rtol=1e-6, atol=1e-12)
    # packedTimes companion is in seconds too (read the .npy directly; the loader's
    # final return exposes only event_rel_times, so check the on-disk file).
    packed = np.load(tmp_path / f"{rec}_packedTimes_withFreqCent.npy")
    np.testing.assert_allclose(packed[0, 1],
                               art.event_rel_end_times[0] * MS_TO_S, rtol=1e-6)
    manifest = json.loads((tmp_path / f"{rec}_montage.json").read_text())
    assert manifest["chn_names"] == art.names


from src.sef_hfo_toywave import (
    traveling_wave, radial_source, synchronous_amplitude_source,
)
from src.sef_hfo_observation import read_direction_from_source


def _two_shaft():
    # ≥2 non-parallel shafts (D6). EVEN contact count so neither has a contact exactly
    # at the shared origin (no coincident point); both through (0,0) -> montage centroid
    # = origin so the CENTERED radial control (C1) is sampled fairly. 16 contacts >>
    # participation gate 2*k_dir+1=7.
    a = build_shaft(np.deg2rad(10.0), 4.0, 8, (0.0, 0.0), "A")
    b = build_shaft(np.deg2rad(100.0), 4.0, 8, (0.0, 0.0), "B")
    return merge_montages([a, b])


def test_gate_traveling_wave_reads_correct_direction():
    # Increment-1 gate = rank-vs-true-n_hat Spearman (§3.5); the endpoint-centroid axis
    # is a REPORT field here (its angle-accuracy gate is Task 5 unit test + increment 2
    # vs θ_EE), so we only sanity-check the axis exists, not its angle (spec §10).
    for deg in (30.0, 60.0):
        src = traveling_wave(64, 64.0, np.deg2rad(deg), c=0.4, dt=0.25,
                             t_max=200.0, width=8.0)
        out = read_direction_from_source(src, _two_shaft(), kernel_width=3.0)
        assert out["spearman"] >= 0.9                      # τ_pass (vs true n_hat)
        assert out["axis"] is not None                     # wave has a real axis (sanity)


def test_gate_C1_radial_source_reads_no_direction():
    # centered radial: rank ∝ radius is monotone along NO axis -> readability < τ_fail.
    # Montage is centered+symmetric on the source (a fair sample); an off-center montage
    # would read a real arrival gradient (sampling artifact), which is NOT what C1 tests.
    src = radial_source(64, 64.0, c=0.35, dt=0.25, t_max=160.0, width=6.0)
    out = read_direction_from_source(src, _two_shaft(), kernel_width=3.0)
    assert out["readability"] < 0.3                        # τ_fail


def test_gate_C2_synchronous_amplitude_makes_no_fake_order():
    # a(x,t)=b(x)h(t): per-contact-relative timing -> all participants tied -> no order
    # along any axis (readability NaN). A global absolute threshold WOULD fabricate order.
    src = synchronous_amplitude_source(64, 64.0, dt=0.25, t_max=120.0,
                                       width=10.0, ramp_axis_rad=np.deg2rad(10.0))
    out = read_direction_from_source(src, _two_shaft(), kernel_width=3.0)
    assert np.isnan(out["readability"]) or out["readability"] < 0.3


from src.sef_hfo_observation import onset_front_axis, angle_error_deg


def test_onset_front_axis_tracks_anisotropic_lobe():
    # CENTER-origin anisotropic spread (the real center-kick event geometry): lag grows
    # FASTER across theta_EE than along it, so the onset front is a LOBE elongated ALONG
    # theta_EE. (A unidirectional planar ramp is the WRONG model — its onset isochrone is
    # a band perpendicular to propagation; see onset_front_axis GEOMETRY CONTRACT.)
    g = np.linspace(-3, 3, 9)
    XX, YY = np.meshgrid(g, g)
    coords = np.column_stack([XX.ravel(), YY.ravel()])
    th = np.deg2rad(30.0)
    par = coords @ np.array([np.cos(th), np.sin(th)])        # along theta_EE
    perp = coords @ np.array([-np.sin(th), np.cos(th)])      # across theta_EE
    lag = 3.0 * np.abs(perp) + 1.0 * np.abs(par)             # faster along theta_EE -> lobe
    bools = np.ones(len(coords), bool)
    angle, ratio, n = onset_front_axis(lag, bools, coords, front_ms=2.0)
    assert angle is not None and ratio > 1.3
    assert angle_error_deg(angle, 30.0) < 25.0


def test_onset_front_axis_collinear_front_returns_none():
    # The DANGEROUS case: the onset-front contacts are squeezed onto a LINE (e.g. one
    # shaft). Naively that looks like a "very strong axis" (ratio -> inf) but it is
    # ELECTRODE SHAPE, not tissue. Must return (None, None, n), not a fake strong axis.
    t = np.linspace(-1, 1, 9)
    ang = np.deg2rad(80.0)
    coords = np.column_stack([t * np.cos(ang), t * np.sin(ang)])   # perfectly collinear
    lag = np.abs(t) * 0.1                                           # all within front_ms
    bools = np.ones(len(coords), bool)
    angle, ratio, n = onset_front_axis(lag, bools, coords, front_ms=5.0)
    assert angle is None and ratio is None
    # near-collinear (a thin sliver, ratio >> the cap) is also rejected
    coords2 = coords + np.column_stack([np.zeros(9), 1e-3 * np.ones(9)])
    coords2[:, 1] += np.linspace(0, 1e-4, 9)                        # tiny perpendicular jitter
    a2, r2, _ = onset_front_axis(lag, bools, coords2, front_ms=5.0)
    assert a2 is None


def test_onset_front_axis_radial_has_no_axis():
    # onset lag ~ radius from center -> onset front is a ring -> no principal axis
    g = np.linspace(-3, 3, 7)
    XX, YY = np.meshgrid(g, g)
    coords = np.column_stack([XX.ravel(), YY.ravel()])
    lag = np.linalg.norm(coords, axis=1)
    bools = np.ones(len(coords), bool)
    angle, ratio, n = onset_front_axis(lag, bools, coords, front_ms=1.0)
    assert (ratio is None) or (ratio < 1.3)        # ring -> ratio ~ 1
