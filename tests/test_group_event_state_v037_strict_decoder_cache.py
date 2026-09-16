from __future__ import annotations

import numpy as np

from scripts.build_group_event_state_v037_strict_decoder_cache import (
    _anatomical_plane,
    _anatomy_tissue_layout,
    _densify,
    _split_prefix,
)


def test_anatomical_plane_is_translation_invariant_and_event_blind():
    coords = np.asarray([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 1.0],
        [0.0, 3.0, 1.0],
        [2.0, 3.0, 2.0],
        [5.0, 1.0, 2.0],
        [1.0, 6.0, 3.0],
    ])
    first, _, axes_first = _anatomical_plane(coords)
    second, _, axes_second = _anatomical_plane(coords + np.asarray([50.0, -7.0, 2.0]))
    assert np.allclose(first, second, atol=1e-6)
    assert np.allclose(axes_first, axes_second, atol=1e-6)


def test_densify_removes_gaps_without_changing_ties():
    groups = np.asarray([[4, -1, 9, 4, 12], [-1, 3, 3, -1, 8]], dtype=np.int16)
    got = _densify(groups)
    assert got.tolist() == [[0, -1, 1, 0, 2], [-1, 0, 0, -1, 1]]


def test_anatomy_tissue_layout_has_local_normalised_readout():
    xy = np.asarray([
        [0.0, 0.0], [4.0, 0.0], [0.0, 5.0], [5.0, 5.0], [9.0, 2.0], [2.0, 9.0],
    ])
    sigma, nodes, observation = _anatomy_tissue_layout(xy, seed=7)
    assert sigma > 0
    assert nodes.shape[0] >= 64
    assert observation.shape == (len(xy), len(nodes))
    assert np.allclose(observation.sum(axis=1), 1.0)
    assert np.any(observation == 0.0)


def test_strict_prefix_split_is_chronological_and_complete():
    split = _split_prefix(101)
    assert (split[:80] == 0).all()
    assert (split[80:90] == 1).all()
    assert (split[90:] == 2).all()
    assert set(split.tolist()) == {0, 1, 2}
