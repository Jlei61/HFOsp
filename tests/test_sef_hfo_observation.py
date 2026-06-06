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
