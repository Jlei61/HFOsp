"""Pin the path-injection refactor of `run_subject` in
`scripts/run_yuquan_lagpat_backfill.py`.

The same-source contract uses `DETECT_ROOT/<subject>/_refineGpu.npz` and
`DETECT_ROOT/<subject>/<stem>_gpu.npz`. The legacy-refine replay path needs
both of those rerouted to `<legacy_root>/<subject>/...`. This module locks
the path-resolution helper that both call sites share.

The test target is the helper `_resolve_run_subject_io`, not the full
`run_subject` (which needs a real EDF + GPU). End-to-end coverage of the
refactor lives in the existing 25 contract tests + the Track B preflight on
`gaolan`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_yuquan_lagpat_backfill import (  # noqa: E402
    DATA_ROOT,
    DETECT_ROOT,
    _resolve_run_subject_io,
)


def test_default_paths_are_canonical_same_source():
    """No overrides ⇒ same-source contract: refine + gpu under DETECT_ROOT,
    out_dir under DATA_ROOT, backup_dir = <raw>/.legacy_backup, assertion=True."""
    r = _resolve_run_subject_io("gaolan")
    assert r.refine_npz == DETECT_ROOT / "gaolan" / "_refineGpu.npz"
    assert r.gpu_resolver("gaolan", "FA0013KP") == (
        DETECT_ROOT / "gaolan" / "FA0013KP_gpu.npz"
    )
    assert r.out_dir == DATA_ROOT / "gaolan"
    assert r.backup_dir == DATA_ROOT / "gaolan" / ".legacy_backup"
    assert r.same_source_assertion is True
    assert r.legacy_refine_root_recorded is None
    assert r.legacy_gpu_root_recorded is None


def test_legacy_root_override_reroutes_away_from_detect_root(tmp_path):
    """Override legacy_refine_root + legacy_gpu_root ⇒ refine and gpu paths
    do NOT live under DETECT_ROOT, regardless of subject. This is the
    provenance gate the comparator relies on."""
    legacy_root = tmp_path / "legacy_data"
    (legacy_root / "gaolan").mkdir(parents=True)
    out_dir = tmp_path / "replay_out" / "gaolan"

    r = _resolve_run_subject_io(
        "gaolan",
        legacy_refine_root=legacy_root,
        legacy_gpu_root=legacy_root,
        out_dir=out_dir,
        backup_dir=None,
        same_source_assertion=False,
    )

    assert r.refine_npz == legacy_root / "gaolan" / "_refineGpu.npz"
    assert r.gpu_resolver("gaolan", "FA0013KP") == (
        legacy_root / "gaolan" / "FA0013KP_gpu.npz"
    )
    # Provenance gate: neither refine nor any gpu path goes under DETECT_ROOT.
    assert DETECT_ROOT not in r.refine_npz.parents
    assert DETECT_ROOT not in r.gpu_resolver("gaolan", "FA_X").parents
    assert r.out_dir == out_dir
    assert r.backup_dir is None
    assert r.same_source_assertion is False
    # Recorded provenance for the audit script:
    assert r.legacy_refine_root_recorded == str(legacy_root)
    assert r.legacy_gpu_root_recorded == str(legacy_root)


def test_replay_refuses_to_write_into_production_raw_tree(tmp_path):
    """Defensive guard: if `same_source_assertion` is False (i.e. this is a
    replay), `out_dir` must not equal `DATA_ROOT/<subject>` — that would
    overwrite the live cohort. Same-source with default out_dir = raw_dir
    is fine."""
    with pytest.raises(ValueError, match="raw tree"):
        _resolve_run_subject_io(
            "gaolan",
            legacy_refine_root=tmp_path,
            legacy_gpu_root=tmp_path,
            out_dir=DATA_ROOT / "gaolan",
            backup_dir=None,
            same_source_assertion=False,
        )


def test_dry_run_out_dir_keeps_backward_compat(tmp_path):
    """The pre-refactor `--dry-run-out-dir` path resolves the same as before:
    out_dir = dry/<subject>, backup_dir = None (never touch raw),
    refine + gpu still under DETECT_ROOT."""
    r = _resolve_run_subject_io("gaolan", dry_run_out_dir=tmp_path)
    assert r.out_dir == tmp_path / "gaolan"
    assert r.backup_dir is None
    assert r.refine_npz == DETECT_ROOT / "gaolan" / "_refineGpu.npz"
    assert r.same_source_assertion is True
    assert r.legacy_refine_root_recorded is None


def test_explicit_out_dir_does_not_auto_create_backup(tmp_path):
    """If a caller passes an explicit `out_dir` (not DATA_ROOT) without
    specifying `backup_dir`, the resolver must default `backup_dir` to None.
    Auto-creating a `.legacy_backup` next to a scratch out_dir would corrupt
    the comparator's provenance assumptions."""
    r = _resolve_run_subject_io("gaolan", out_dir=tmp_path / "out")
    assert r.out_dir == tmp_path / "out"
    assert r.backup_dir is None


def test_legacy_root_override_can_route_per_root(tmp_path):
    """legacy_refine_root and legacy_gpu_root can be different paths. The
    comparator records both separately so a hypothetical 'legacy refine here,
    legacy gpu there' replay is provenance-traceable."""
    refine_root = tmp_path / "legacy_refine"
    gpu_root = tmp_path / "legacy_gpu"
    (refine_root / "gaolan").mkdir(parents=True)
    (gpu_root / "gaolan").mkdir(parents=True)

    r = _resolve_run_subject_io(
        "gaolan",
        legacy_refine_root=refine_root,
        legacy_gpu_root=gpu_root,
        out_dir=tmp_path / "out",
        backup_dir=None,
        same_source_assertion=False,
    )
    assert r.refine_npz == refine_root / "gaolan" / "_refineGpu.npz"
    assert r.gpu_resolver("gaolan", "X") == gpu_root / "gaolan" / "X_gpu.npz"
    assert r.legacy_refine_root_recorded == str(refine_root)
    assert r.legacy_gpu_root_recorded == str(gpu_root)
