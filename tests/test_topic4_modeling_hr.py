"""Tests for src/topic4_modeling/hr.py (Stage 1 single-node HR)."""

from __future__ import annotations

import numpy as np
import pytest


def test_module_importable():
    """Sanity: hr module can be imported."""
    from src.topic4_modeling import hr  # noqa: F401
