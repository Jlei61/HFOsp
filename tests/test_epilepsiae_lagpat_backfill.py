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


def test_refine_path_is_subject_not_per_record():
    """Refine artifact lives at <gpu_dir>/<subject>/_refineGpu.npz (subject-level).

    The new pipeline writes ONE refine artifact per subject (no per-record
    suffix, no `sub_` prefix). Schema: {chns_names, events_count}.
    """
    from scripts.run_epilepsiae_lagpat_backfill import _refine_path_for_subject

    p = _refine_path_for_subject("253")
    assert p.name == "_refineGpu.npz"
    assert p.parent.name == "253"


def test_load_refine_chns_for_subject_returns_tuple():
    """load_refine_chns_for_subject returns a hashable Tuple[str, ...].

    Note: deviates from the plan draft which asserted `isinstance(chns, list)`.
    The implementation uses lru_cache, which requires hashable returns; tuple
    is the correct contract.
    """
    from scripts.run_epilepsiae_lagpat_backfill import load_refine_chns_for_subject

    chns = load_refine_chns_for_subject("253")
    assert isinstance(chns, tuple)
    assert len(chns) > 0
    assert all(isinstance(c, str) for c in chns)
    # subject-level: two calls return identical content
    chns2 = load_refine_chns_for_subject("253")
    assert chns == chns2


def test_smoke_does_not_create_output_files():
    """--smoke must not create any files under OUTPUT_ROOT.

    Snapshots OUTPUT_ROOT contents before and after _smoke_print(); equality
    means zero new files written. Auto-guard: if a future change accidentally
    introduces a write inside _smoke_print (logs, partial artifacts, etc.),
    this test catches it instead of relying on a manual `find` audit.
    """
    import scripts.run_epilepsiae_lagpat_backfill as mod

    def _snapshot(p: Path) -> set:
        if not p.exists():
            return set()
        return set(p.rglob("*"))

    before = _snapshot(mod.OUTPUT_ROOT)
    mod._smoke_print("253")
    after = _snapshot(mod.OUTPUT_ROOT)
    assert before == after, (
        f"--smoke must not write to OUTPUT_ROOT={mod.OUTPUT_ROOT}; "
        f"new entries: {sorted(after - before)}"
    )
