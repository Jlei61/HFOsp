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


def test_smoke_lists_first_record():
    """_discover_records('253') yields at least one record with valid metadata.

    Subject 253 is the smallest pure-512Hz cohort (268 records); used for
    smoke ladder step B.4.b.
    """
    from scripts.run_epilepsiae_lagpat_backfill import _discover_records

    recs = _discover_records("253")
    assert len(recs) >= 1
    first = recs[0]
    assert first["stem"].startswith("253")
    assert first["sfreq"] in (256.0, 512.0, 1024.0)
    assert first["new_gpu_path"].exists()
    assert first["raw_data_path"].exists() and first["raw_head_path"].exists()
