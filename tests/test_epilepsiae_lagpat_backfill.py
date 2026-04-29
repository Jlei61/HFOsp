"""Tests for scripts/run_epilepsiae_lagpat_backfill.py.

Plan: docs/archive/epilepsiae_lagpat/epilepsiae_lagpat_backfill_plan_2026-04-29.md

Stage A: skeleton + record discovery (dry-print, no writes).
"""
from __future__ import annotations

from pathlib import Path


def test_output_dir_constant_is_results_subtree():
    """OUTPUT_ROOT must live under results/, never under /mnt.

    Hard guard against accidentally writing back to /mnt/epilepsia_data/...
    (legacy lagPat is part of Topic 1's current contract; never overwrite).
    """
    from scripts.run_epilepsiae_lagpat_backfill import OUTPUT_ROOT

    assert OUTPUT_ROOT == Path("results/epilepsiae_lagpat_backfill")
    assert "/mnt" not in str(OUTPUT_ROOT.resolve())
