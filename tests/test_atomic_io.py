"""TDD tests for src.atomic_io — atomic JSON write + stale .tmp purge."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.atomic_io import purge_stale_tmp, write_json_atomic


def test_write_json_atomic_creates_file_with_correct_content(tmp_path: Path):
    out = tmp_path / "subject_x.json"
    payload = {"subject": "x", "n": 7, "vals": [1, 2, 3]}

    write_json_atomic(out, payload)

    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded == payload


def test_write_json_atomic_no_tmp_left_after_success(tmp_path: Path):
    out = tmp_path / "subject_x.json"
    write_json_atomic(out, {"a": 1})

    leftover = list(tmp_path.glob("*.tmp"))
    assert leftover == []


def test_write_json_atomic_overwrites_existing(tmp_path: Path):
    out = tmp_path / "subject_x.json"
    write_json_atomic(out, {"v": 1})
    write_json_atomic(out, {"v": 2})

    assert json.loads(out.read_text())["v"] == 2


def test_write_json_atomic_handles_default_serializer(tmp_path: Path):
    """default= callable lets caller stringify e.g. numpy types."""
    import numpy as np
    out = tmp_path / "x.json"

    def to_py(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        raise TypeError

    write_json_atomic(out, {"n": np.int64(5)}, default=to_py)
    assert json.loads(out.read_text())["n"] == 5


def test_write_json_atomic_failure_leaves_dest_intact(tmp_path: Path):
    """If serialization fails mid-write, destination must remain unchanged."""
    out = tmp_path / "subject_x.json"
    write_json_atomic(out, {"good": True})
    original = out.read_text()

    class Unserializable:
        pass

    with pytest.raises(TypeError):
        write_json_atomic(out, {"bad": Unserializable()})

    # Destination still has the original content
    assert out.read_text() == original
    # No stray .tmp left behind even on failure
    assert list(tmp_path.glob("*.tmp")) == []


def test_purge_stale_tmp_removes_matching_files(tmp_path: Path):
    (tmp_path / "subj_a.json.tmp").write_text("{}")
    (tmp_path / "subj_b.json.tmp").write_text("{}")
    (tmp_path / "subj_c.json").write_text("{}")  # NOT .tmp, must be kept

    n = purge_stale_tmp(tmp_path)
    assert n == 2

    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == ["subj_c.json"]


def test_purge_stale_tmp_handles_missing_dir(tmp_path: Path):
    nonexistent = tmp_path / "no_such_dir"
    assert purge_stale_tmp(nonexistent) == 0


def test_purge_stale_tmp_returns_zero_on_clean_dir(tmp_path: Path):
    (tmp_path / "subj.json").write_text("{}")
    assert purge_stale_tmp(tmp_path) == 0
