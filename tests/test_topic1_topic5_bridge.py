"""Tests for src/topic1_topic5_bridge.py."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import topic1_topic5_bridge as bridge


def test_locked_constants():
    """Sanity check: locked constants match spec §4."""
    assert bridge.ALPHA_WITHIN == pytest.approx(0.0167)
    assert bridge.EFFECT_MIN == pytest.approx(0.10)
    assert bridge.WINDOWS_MIN == [(-15.0, -1.0), (-30.0, -1.0), (-60.0, -1.0)]
    assert len(bridge.COHORT_GAMMA) == 10
    assert "442" not in bridge.COHORT_GAMMA  # 442 is Q1b sentinel only
