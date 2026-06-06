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
