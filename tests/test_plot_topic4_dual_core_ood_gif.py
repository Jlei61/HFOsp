import pytest

import numpy as np

from scripts.plot_topic4_dual_core_ood_gif import (
    _bandpass_contact_activity,
    _upsert_readme_entry,
    ordered_shaft_segments,
    select_representative_pair,
)


def _event(index, mode, distance, *, icl=6, scl=2, support=True):
    return {
        "event_index": index, "mode": mode,
        "normalized_support_distance": distance,
        "ICL_recruited": icl, "SCL_recruited": scl,
        "returned": True, "in_support": support,
    }


def test_select_pair_requires_both_modes_and_uses_closest_events():
    summary = {"per_network": [
        {
            "candidate_id": "candidate", "seed": 2,
            "ood_all_returned": 0.2, "worker_json": "two.json",
            "worker_npz": "two.npz",
            "events": [_event(0, 0, 0.1)],
        },
        {
            "candidate_id": "candidate", "seed": 1,
            "ood_all_returned": 0.3, "worker_json": "one.json",
            "worker_npz": "one.npz",
            "events": [
                _event(3, 0, 0.4), _event(4, 0, 0.2),
                _event(5, 1, 0.7),
            ],
        },
    ]}
    selected = select_representative_pair(summary)
    assert selected["seed"] == 1
    assert selected["events"]["0"]["event_index"] == 4
    assert selected["events"]["1"]["event_index"] == 5


def test_select_pair_rejects_display_ineligible_support():
    summary = {"per_network": [{
        "candidate_id": "candidate", "seed": 1,
        "ood_all_returned": 0.2, "worker_json": "x.json",
        "worker_npz": "x.npz",
        "events": [_event(0, 0, 0.1), _event(1, 1, 0.1, scl=1)],
    }]}
    with pytest.raises(RuntimeError, match="display-eligible"):
        select_representative_pair(summary)


def test_ordered_shaft_segments_never_connect_different_shafts():
    order = np.asarray([0, 2, 1, 3])
    shafts = np.asarray(["ICL", "SCL", "ICL", "SCL"])
    segments = ordered_shaft_segments(order, shafts)
    assert [segment.tolist() for segment in segments] == [[0, 2], [1, 3]]


def test_readme_entry_is_replaced_without_dropping_other_entries():
    existing = "### target\nold\n\n### retained\nkeep\n"
    updated = _upsert_readme_entry(existing, "### target", "### target\nnew\n")
    assert updated == "### target\nnew\n### retained\nkeep\n"


def test_bandpass_matches_accepted_fig4_filter():
    from scripts.paper_figures.plot_fig4_spatial_edge_flow_validation import (
        _bandpass_contact_activity as accepted_filter,
    )

    rng = np.random.default_rng(3)
    envelope = rng.normal(size=(15, 600))
    observed = _bandpass_contact_activity(envelope, 2.0)
    expected = accepted_filter(envelope, 2.0)
    assert np.array_equal(observed, expected)
