"""B1 contract tests: attaching a frozen state trajectory to grid anchors."""
from __future__ import annotations
import numpy as np
import pytest
from src.topic5_h2b_transfer.attach import attach_state_to_anchors


def test_anchor_takes_the_last_event_strictly_before_it():
    t = np.array([0.0, 10.0, 20.0])
    s = np.array([[1.0], [2.0], [3.0]])
    out = attach_state_to_anchors(np.array([15.0]), t, s, max_age_seconds=1e9)
    assert out.state[0, 0] == 2.0
    assert out.age_seconds[0] == 5.0


def test_an_event_exactly_at_the_anchor_is_not_read():
    """Causal prefix: predict first, then read the event at that instant."""
    t = np.array([0.0, 10.0])
    s = np.array([[1.0], [2.0]])
    out = attach_state_to_anchors(np.array([10.0]), t, s, max_age_seconds=1e9)
    assert out.state[0, 0] == 1.0


def test_anchor_before_any_event_has_no_state():
    t = np.array([100.0])
    s = np.array([[1.0]])
    out = attach_state_to_anchors(np.array([50.0]), t, s, max_age_seconds=1e9)
    assert not out.available[0]
    assert np.isnan(out.state[0, 0])


def test_a_state_older_than_the_age_limit_is_refused_not_stretched():
    t = np.array([0.0])
    s = np.array([[1.0]])
    out = attach_state_to_anchors(np.array([10_000.0]), t, s, max_age_seconds=3600.0)
    assert not out.available[0]


def test_ages_and_availability_line_up_with_the_anchors():
    t = np.array([0.0, 100.0])
    s = np.array([[1.0], [2.0]])
    out = attach_state_to_anchors(np.array([50.0, 150.0, -1.0]), t, s, max_age_seconds=1e9)
    assert list(out.available) == [True, True, False]
    assert list(out.age_seconds[:2]) == [50.0, 50.0]
