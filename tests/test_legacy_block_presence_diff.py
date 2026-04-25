"""Pin the semantics of `_legacy_block_presence_diff`.

`.legacy_backup/` is the single source of truth for legacy presence once it
exists. Without this rule the audit would see v1 backfill outputs sitting in
the raw dir as if they were legacy evidence, causing `legacy_present` to be
inflated and `regressions` to be deflated after the second batch run.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_yuquan_lagpat_backfill import _legacy_block_presence_diff  # noqa: E402


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")


def test_untouched_subject_uses_raw_dir(tmp_path: Path) -> None:
    raw = tmp_path / "subj_a"
    raw.mkdir()
    _touch(raw / "R1_lagPat.npz")
    _touch(raw / "R2_lagPat.npz")

    records = [
        {"record": "R1", "status": "ok"},
        {"record": "R2", "status": "skipped"},
        {"record": "R3", "status": "ok"},
    ]
    out = _legacy_block_presence_diff(raw, ["R1", "R2", "R3"], records)

    assert out["legacy_source_kind"] == "raw_dir_untouched"
    assert out["n_legacy_present"] == 2
    assert out["n_legacy_absent"] == 1
    assert sorted(out["regressions"]) == ["R2"]
    assert sorted(out["extras_written"]) == ["R3"]


def test_already_backfilled_subject_uses_backup_only(tmp_path: Path) -> None:
    """Once `.legacy_backup/` exists, `raw_dir` content is treated as
    new-pipeline output, not as legacy evidence."""
    raw = tmp_path / "subj_b"
    bkp = raw / ".legacy_backup"
    bkp.mkdir(parents=True)

    # Legacy actually had only R1; the .legacy_backup dir reflects that.
    _touch(bkp / "R1_lagPat.npz")
    # Both R1 and R2 currently sit in raw_dir as v1 backfill outputs
    # (R2 has no legacy counterpart). The audit MUST NOT treat R2's raw
    # presence as legacy evidence.
    _touch(raw / "R1_lagPat.npz")
    _touch(raw / "R2_lagPat.npz")

    records = [
        {"record": "R1", "status": "ok"},
        {"record": "R2", "status": "ok"},
        {"record": "R3", "status": "skipped"},
    ]
    out = _legacy_block_presence_diff(raw, ["R1", "R2", "R3"], records)

    assert out["legacy_source_kind"] == "legacy_backup_dir"
    assert out["n_legacy_present"] == 1
    assert out["n_legacy_absent"] == 2
    assert sorted(out["legacy_present_status"]["ok"]) == ["R1"]
    # R3 was never in legacy and was not written -> not a regression
    assert out["regressions"] == []
    # R2 was written but legacy didn't have it -> extra (not regression)
    assert sorted(out["extras_written"]) == ["R2"]


def test_regression_detected_when_legacy_present_block_skipped(tmp_path: Path) -> None:
    raw = tmp_path / "subj_c"
    bkp = raw / ".legacy_backup"
    bkp.mkdir(parents=True)
    _touch(bkp / "R1_lagPat.npz")
    _touch(bkp / "R2_lagPat.npz")

    records = [
        {"record": "R1", "status": "ok"},
        {"record": "R2", "status": "skipped"},  # regression: legacy had it
    ]
    out = _legacy_block_presence_diff(raw, ["R1", "R2"], records)

    assert out["legacy_source_kind"] == "legacy_backup_dir"
    assert out["n_legacy_present"] == 2
    assert sorted(out["regressions"]) == ["R2"]
    assert out["extras_written"] == []
