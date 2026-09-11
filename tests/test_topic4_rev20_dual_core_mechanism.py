import copy
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from src.topic4_manual_dual_core import (
    budget_matched_dual_core_h, dual_core_query_h,
)
from src.topic4_rev20_dual_core_mechanism import (
    build_one_factor_candidates,
    fixed_topology_ee_ellipse_redistribution,
)


ROOT = Path(__file__).resolve().parents[1]


def _toy_net():
    first = sparse.csc_matrix(np.asarray([
        [0.0, 1.0, 2.0],
        [3.0, 0.0, 1.0],
        [1.0, 2.0, 0.0],
        [0.5, 0.5, 0.5],
    ]))
    second = sparse.csc_matrix(np.asarray([
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]))
    gaba = sparse.csc_matrix(np.eye(4, 1))
    return {
        "NE": 3,
        "NI": 1,
        "ampa_by_delay": [first, second],
        "gaba_by_delay": [gaba],
        "cached_ampa": object(),
    }


def _incoming_ee(net):
    total = np.zeros(net["NE"], float)
    for matrix in net["ampa_by_delay"]:
        coo = matrix.tocoo()
        keep = coo.row < net["NE"]
        total += np.bincount(
            coo.row[keep], weights=coo.data[keep], minlength=net["NE"],
        )
    return total


def test_dual_core_budget_is_exact_and_deterministic():
    positions = np.asarray([
        [0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [9.0, 9.0],
        [10.0, 9.0], [9.0, 10.0],
    ])
    centers = np.asarray([[0.0, 0.0], [10.0, 10.0]])
    first, audit = budget_matched_dual_core_h(
        positions, centers, target_count=4,
    )
    second, second_audit = budget_matched_dual_core_h(
        positions, centers, target_count=4,
    )
    assert np.array_equal(first, second)
    assert int(first.sum()) == audit["selected_count"] == 4
    assert audit["h_sha256"] == second_audit["h_sha256"]
    queried = dual_core_query_h(
        positions, centers, distance_cutoff_mm=audit["distance_cutoff_mm"],
    )
    assert np.all(queried[first.astype(bool)] == 1.0)


def test_atlas_has_one_reference_and_one_factor_rows():
    config = json.loads((
        ROOT / "config/topic4_rev20_dc_dual_core_mechanism_atlas.json"
    ).read_text())
    rows = build_one_factor_candidates(config)
    assert len(rows) == 31
    assert sum(row["is_reference"] for row in rows) == 1
    assert len({row["candidate_id"] for row in rows}) == 31
    assert all(row["mechanisms"]["Z_M"] == "off" for row in rows)


def test_reference_ellipse_is_same_object_exact_noop():
    net = _toy_net()
    before = [matrix.copy() for matrix in net["ampa_by_delay"]]
    mapped, audit = fixed_topology_ee_ellipse_redistribution(
        net, np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], float),
        length_scale=0.6, angle_deg=45.0, aspect_ratio=2.0,
    )
    assert mapped is net
    assert audit["exact_noop"] is True
    for old, current in zip(before, mapped["ampa_by_delay"]):
        assert np.array_equal(old.toarray(), current.toarray())


def test_ellipse_preserves_topology_delays_gaba_and_incoming_ee():
    net = _toy_net()
    original = copy.deepcopy(net)
    mapped, audit = fixed_topology_ee_ellipse_redistribution(
        net, np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], float),
        length_scale=0.6, angle_deg=0.0, aspect_ratio=4.0,
    )
    assert mapped is not net
    assert audit["topology_unchanged"]
    assert audit["delay_assignment_unchanged"]
    assert audit["gaba_unchanged"]
    assert audit["maximum_abs_incoming_EE_error"] <= 1e-9
    assert np.allclose(_incoming_ee(original), _incoming_ee(mapped), atol=1e-12)
    assert any(
        not np.allclose(a.toarray(), b.toarray())
        for a, b in zip(original["ampa_by_delay"], mapped["ampa_by_delay"])
    )
    assert all(
        np.array_equal(a.toarray(), b.toarray())
        for a, b in zip(original["gaba_by_delay"], mapped["gaba_by_delay"])
    )
