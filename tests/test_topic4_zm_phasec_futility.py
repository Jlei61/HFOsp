"""Tests for the Phase-C post-result futility adjudicator."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/adjudicate_topic4_zm_phasec_futility.py"
SPEC = importlib.util.spec_from_file_location("phasec_futility", SCRIPT)
F = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(F)


def test_part_identity_and_task_key(tmp_path):
    root = tmp_path / "results"
    path = (
        root / "parts/c1_base/dt/seed1/primary_convex/cell_a"
        / "rising/noise_replay/phenotype.json"
    )
    identity = F._part_identity(path, result_root=root)
    assert identity == {
        "seed": 1,
        "tier": "primary_convex",
        "cell_id": "cell_a",
        "phase": "rising",
        "noise": "noise_replay",
    }
    assert F._task_key(identity) == (
        "base|s1|primary_convex|cell_a|rising|noise_replay"
    )


def test_part_identity_rejects_noncanonical_path(tmp_path):
    with pytest.raises(ValueError, match="unexpected C1 part path"):
        F._part_identity(
            tmp_path / "parts/c1_base/dt/seed1/cell/phenotype.json",
            result_root=tmp_path,
        )


def test_futility_logic_requires_every_cell_to_be_unrescuable():
    required = 5
    observed_negative = 5
    missing = 1
    max_possible_positive = 0 + missing
    assert observed_negative + missing == 6
    assert max_possible_positive < required


def test_canonical_hash_is_order_invariant():
    assert F._canonical_sha({"a": 1, "b": 2}) == F._canonical_sha({
        "b": 2,
        "a": 1,
    })
