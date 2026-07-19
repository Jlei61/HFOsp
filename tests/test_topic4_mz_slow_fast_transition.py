"""Contract tests for Topic 4 MZ slow–fast dynamical transition (design §10).

Pure-function tests use no SNN; the tiny-network smoke tests build a small substrate so the
freeze / independent-replay invariants are exercised without the full E1146 substrate.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

import src.topic4_mz_slow_fast_transition as MZSF  # noqa: E402


def test_module_imports_and_schema():
    assert MZSF.SCHEMA_VERSION == "mz-slow-fast-transition-1.0"
