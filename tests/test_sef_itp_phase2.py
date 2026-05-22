"""Unit tests for src/sef_itp_phase2.py (SEF-ITP framework Phase 2: H3 + H4).

Phase 2 implements H3 (mark-independent template sampling) + H4 (normalized
rate vs geometry instability) on top of Phase 1's n=23 cohort. H3 is mostly
ingest-heavy reuse of PR-7 / PR-6 outputs; H4 is new (epoch slicing +
normalized instability + matched null).

Tests grow incrementally per plan tasks in
docs/superpowers/plans/2026-05-23-topic4-phase2-h3-h4-plan.md.
"""
from __future__ import annotations

import numpy as np
import pytest

from src import sef_itp_phase2 as p2


# --------------------------------------------------------------------------- #
# Task 1 — module skeleton + SubjectPhase2Data
# --------------------------------------------------------------------------- #


def test_module_version():
    assert p2.__version__ == "v1.0.0"


def test_subject_phase2_data_dataclass_fields():
    """SubjectPhase2Data dataclass must carry both H3 ingest fields and H4 raw inputs."""
    s = p2.SubjectPhase2Data(
        dataset="yuquan",
        subject_id="test",
        # H3 ingest
        lag1_same_excess_n2=0.01,
        window_excess_n2={10.0: 0.0, 30.0: 0.0, 60.0: 0.0, 1800.0: 0.0},
        run_length_lift_n2=1.0,
        endpoint_jaccard_first_half=0.9,
        endpoint_jaccard_odd_even=0.85,
        # H4 raw
        event_abs_times=np.array([0.0, 1.0, 2.0]),
        cluster_labels=np.array([0, 1, 0]),
        block_time_ranges=[(0.0, 10.0)],
        template_ranks={0: np.array([0, 1, 2, 3, 4, 5]), 1: np.array([5, 4, 3, 2, 1, 0])},
        channel_names=["A", "B", "C", "D", "E", "F"],
    )
    assert s.dataset == "yuquan"
    assert s.lag1_same_excess_n2 == pytest.approx(0.01)
    assert 10.0 in s.window_excess_n2
    assert s.event_abs_times.shape == (3,)
