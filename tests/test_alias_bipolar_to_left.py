"""Pin the schema-detection contract of
`alias_bipolar_to_left_with_arbitration` in
`scripts/run_yuquan_lagpat_backfill.py`.

The function accepts two refine schemas:
  - bipolar-pair input: every name contains '-' (e.g. 'A1-A2', 'A2-A3').
    Outer-shaft alias drop is applied so the result aligns with
    `_legacy_bipolar_reref_and_drop`.
  - single-electrode input: no name contains '-' (e.g. 'A1', 'A2').
    Already alias-collapsed by 2021-era legacy refine; outer-shaft drop
    must be skipped or it would remove legitimately picked channels.

Mixed inputs (some names with '-', some without) are undefined — the
function must raise rather than silently pick a branch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_yuquan_lagpat_backfill import (  # noqa: E402
    alias_bipolar_to_left_with_arbitration,
)


def test_bipolar_pair_input_drops_outermost():
    """Bipolar-pair input: alias collapse + outer-shaft drop is applied."""
    names = ["A1-A2", "A2-A3", "A3-A4"]
    counts = np.array([10, 20, 30], dtype=np.int64)
    aliases, collisions, outer_drops = alias_bipolar_to_left_with_arbitration(
        names, counts
    )
    # left-contact alias of each pair: A1, A2, A3. Outer = A3 dropped.
    assert set(aliases.keys()) == {"A1", "A2"}
    assert len(outer_drops) == 1
    assert outer_drops[0]["alias"] == "A3"


def test_single_electrode_input_skips_outer_drop():
    """Single-electrode input: outer-shaft drop must NOT fire — these names
    are already alias-collapsed by legacy refine, dropping again would lose
    a legitimately picked channel."""
    names = ["A1", "A2", "A3"]
    counts = np.array([10, 20, 30], dtype=np.int64)
    aliases, collisions, outer_drops = alias_bipolar_to_left_with_arbitration(
        names, counts
    )
    assert set(aliases.keys()) == {"A1", "A2", "A3"}
    assert outer_drops == []


def test_mixed_schema_raises():
    """A mix of bipolar-pair and single-electrode names is ambiguous —
    the function must refuse rather than silently treat the whole input
    as bipolar (which would mis-fire outer-drop on the single-electrode
    portion)."""
    names = ["A1-A2", "A2", "A3"]   # 1 bipolar, 2 single
    counts = np.array([10, 20, 30], dtype=np.int64)
    with pytest.raises(ValueError) as excinfo:
        alias_bipolar_to_left_with_arbitration(names, counts)
    msg = str(excinfo.value)
    assert "mixed schema" in msg.lower()
    assert "A1-A2" in msg
    assert "A2" in msg or "A3" in msg


def test_mixed_schema_raises_other_direction():
    """Mostly single-electrode with one bipolar pair stuck in must also
    raise — symmetric handling."""
    names = ["A1", "A2", "B1-B2"]
    counts = np.array([10, 20, 30], dtype=np.int64)
    with pytest.raises(ValueError):
        alias_bipolar_to_left_with_arbitration(names, counts)


def test_mismatched_counts_length_raises():
    """Pre-existing length-mismatch guard still fires (regression test)."""
    with pytest.raises(ValueError):
        alias_bipolar_to_left_with_arbitration(
            ["A1-A2", "A2-A3"], np.array([10], dtype=np.int64),
        )


def test_bipolar_pair_per_shaft_outer_drop():
    """Outer-shaft drop is per-shaft — each prefix gets its own outermost
    alias removed. Dropping the global outermost would over-cull."""
    names = ["A1-A2", "A2-A3", "B1-B2", "B2-B3", "B3-B4"]
    counts = np.array([5, 6, 7, 8, 9], dtype=np.int64)
    aliases, _, outer_drops = alias_bipolar_to_left_with_arbitration(
        names, counts,
    )
    # A shaft outer: A2 — left contacts of pairs are A1, A2; outer = A2.
    # B shaft outer: B3 — left contacts of pairs are B1, B2, B3; outer = B3.
    dropped = {d["alias"] for d in outer_drops}
    assert dropped == {"A2", "B3"}
    assert set(aliases.keys()) == {"A1", "B1", "B2"}
