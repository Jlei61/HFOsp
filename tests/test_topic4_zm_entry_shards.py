from __future__ import annotations

import importlib.util
import os


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, filename):
    path = os.path.join(_ROOT, "scripts", filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CELL = _load("topic4_zm_entry_cell", "run_topic4_zm_entry_cell.py")
MERGE = _load("topic4_zm_entry_merge", "merge_topic4_zm_entry_parts.py")


def test_entry_parts_have_disjoint_paths():
    assert CELL._part_path(1, 0.25, "noise_replay") != CELL._part_path(
        1, 0.5, "noise_replay"
    )
    assert CELL._part_path(1, 0.25, "noise_replay") != CELL._part_path(
        3, 0.25, "noise_replay"
    )
    assert "parts" in CELL._part_path(1, 0.25, "noise_replay")
    assert not CELL._part_path(1, 0.25, "noise_replay").endswith(
        "entry_probes.json"
    )


def test_duplicate_rows_compare_scientific_fields_not_runtime_metadata():
    base = {
        "key": "lambda=0.25|noise_replay",
        "seed": 1,
        "lambda": 0.25,
        "replicate": "noise_replay",
        "bank_sha": "abc",
        "entered_carrier": True,
        "completed": True,
        "boundary_version": "v",
        "survived": True,
        "stationarity_ok": True,
        "end_reason": None,
        "wall_s": 1.0,
    }
    other = {**base, "wall_s": 2.0, "producer_git_sha": "def"}
    assert MERGE._same_scientific_row(base, other)
    assert not MERGE._same_scientific_row(
        base, {**other, "entered_carrier": False}
    )
