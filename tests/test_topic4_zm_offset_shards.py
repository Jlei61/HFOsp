from __future__ import annotations

import importlib.util
import json
import os

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, filename):
    path = os.path.join(_ROOT, "scripts", filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CELL = _load("topic4_zm_offset_cell", "run_topic4_zm_offset_cell.py")
MERGE = _load("topic4_zm_offset_merge", "merge_topic4_zm_offset_parts.py")


def test_offset_parts_have_disjoint_paths():
    active = CELL._part_path(1, "M_alone", 0.0, "active", "noise_replay")
    low = CELL._part_path(1, "M_alone", 0.0, "low", "noise_replay")
    other = CELL._part_path(1, "M_SG", 0.0, "active", "noise_replay")
    dynamic = CELL._part_path(
        1, "dynamic_ZM", None, "active", "noise_replay"
    )
    assert len({active, low, other, dynamic}) == 4
    assert all("parts" in path for path in (active, low, other, dynamic))
    assert all(
        not path.endswith("offset_probes.json")
        for path in (active, low, other, dynamic)
    )


def test_offset_duplicate_comparison_ignores_runtime_only_fields():
    base = {
        "key": "M_alone|lambda=0|active|noise_replay",
        "seed": 1,
        "family": "M_alone",
        "lambda": 0.0,
        "initial_kind": "active",
        "replicate": "noise_replay",
        "bank_sha": "abc",
        "remained_carrier": True,
        "low_basin_persisted": False,
        "completed": True,
        "response_valid": True,
        "invalid_reason": None,
        "boundary_version": "v",
        "survived": True,
        "stationarity_ok": True,
        "end_reason": None,
        "wall_s": 1.0,
    }
    other = {**base, "wall_s": 2.0, "producer_git_sha": "def"}
    assert MERGE._same_scientific_row(base, other)
    assert not MERGE._same_scientific_row(
        base, {**other, "remained_carrier": False}
    )


def test_force_rerun_bypasses_complete_part_and_canonical_reuse(
    tmp_path, monkeypatch
):
    part = tmp_path / "cell.json"
    part.write_text(
        json.dumps(
            {
                "complete": True,
                "boundary_version": CELL.R.BD.BOUNDARY_VERSION,
                "row": {
                    "key": "M_alone|lambda=0|active|noise_replay",
                },
            }
        )
    )
    monkeypatch.setattr(CELL, "_part_path", lambda *args: str(part))
    monkeypatch.setattr(
        CELL,
        "_canonical_row",
        lambda *args: pytest.fail("force rerun queried canonical cache"),
    )

    def _recompute(*args, **kwargs):
        raise RuntimeError("recompute reached")

    monkeypatch.setattr(CELL.R, "build_context", _recompute)
    with pytest.raises(RuntimeError, match="recompute reached"):
        CELL.run_cell(
            1,
            "M_alone",
            0.0,
            "active",
            "noise_replay",
            force_rerun=True,
        )


def test_provenance_complete_end_reason_repair_supersedes_stale_canonical(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(MERGE.R, "OUT", str(tmp_path))
    root = tmp_path / "boundaries" / "offset" / "seed1"
    parts = root / "parts"
    parts.mkdir(parents=True)
    key = "M_Z_recovery|lambda=1|low|noise_replay"
    old = {
        "key": key,
        "seed": 1,
        "family": "M_Z_recovery",
        "lambda": 1.0,
        "initial_kind": "low",
        "replicate": "noise_replay",
        "bank_sha": "bank",
        "remained_carrier": False,
        "low_basin_persisted": False,
        "completed": True,
        "response_valid": True,
        "boundary_version": MERGE.R.BD.BOUNDARY_VERSION,
        "survived": False,
        "stationarity_ok": False,
        "end_reason": "rest_return",
    }
    (root / "offset_probes.json").write_text(
        json.dumps(
            {
                "git_sha": "old",
                "boundary_version": MERGE.R.BD.BOUNDARY_VERSION,
                "source_state_hashes": {"state": "abc"},
                "rows": [old],
            }
        )
    )
    repaired = {
        **old,
        "low_basin_persisted": True,
        "end_reason": "dead_in_rest_basin",
        "run_end_reason": "dead_in_rest_basin",
        "classifier_end_reason": "rest_return",
        "producer_git_sha": "repair-sha",
    }
    (parts / "repair.json").write_text(
        json.dumps(
            {
                "complete": True,
                "git_sha": "repair-sha",
                "boundary_version": MERGE.R.BD.BOUNDARY_VERSION,
                "source_state_hashes": {"state": "abc"},
                "row": repaired,
            }
        )
    )

    out = MERGE.merge_seed(1)
    merged = next(row for row in out["rows"] if row["key"] == key)
    assert merged["run_end_reason"] == "dead_in_rest_basin"
    assert merged["classifier_end_reason"] == "rest_return"
    assert merged["low_basin_persisted"] is True


def test_offset_merger_can_close_unbracketed_all_offset_base(tmp_path, monkeypatch):
    monkeypatch.setattr(MERGE.R, "OUT", str(tmp_path))
    root = tmp_path / "boundaries" / "offset" / "seed1"
    root.mkdir(parents=True)
    rows = []
    for family in MERGE.R.OFFSET_FAMILIES:
        for lam in MERGE.R.OFFSET_LEVELS:
            for initial_kind in ("active", "low"):
                rows.append(
                    {
                        "key": (
                            f"{family}|lambda={lam:g}|{initial_kind}|noise_replay"
                        ),
                        "seed": 1,
                        "family": family,
                        "lambda": lam,
                        "initial_kind": initial_kind,
                        "replicate": "noise_replay",
                        "bank_sha": f"{family}-{lam}-{initial_kind}",
                        "remained_carrier": False,
                        "low_basin_persisted": initial_kind == "low",
                        "completed": True,
                        "response_valid": True,
                        "boundary_version": MERGE.R.BD.BOUNDARY_VERSION,
                        "survived": False,
                        "stationarity_ok": False,
                        "end_reason": "dead_in_rest_basin",
                    }
                )
    for replicate in (
        MERGE.R.OFFSET_BASE_REPLICATE,
        *MERGE.R.OFFSET_EXPANSION_REPLICATES,
    ):
        rows.append(
            {
                "key": f"dynamic_ZM|late_active|{replicate}",
                "seed": 1,
                "family": "dynamic_ZM",
                "initial_kind": "active",
                "replicate": replicate,
                "bank_sha": f"dynamic-{replicate}",
                "remained_carrier": False,
                "completed": True,
                "response_valid": True,
                "boundary_version": MERGE.R.BD.BOUNDARY_VERSION,
                "survived": False,
                "stationarity_ok": False,
                "end_reason": "dead_in_rest_basin",
            }
        )
    (root / "offset_probes.json").write_text(
        json.dumps(
            {
                "git_sha": "old",
                "boundary_version": MERGE.R.BD.BOUNDARY_VERSION,
                "source_state_hashes": {"pre": "a"},
                "rows": rows,
            }
        )
    )
    out = MERGE.merge_seed(1)
    assert out["complete"]
    assert out["pending_cells"] == []
    assert {
        result["status"] for result in out["cheap_brackets"].values()
    } == {"unbracketed"}
