from __future__ import annotations

import importlib.util
import os


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH = os.path.join(
    _ROOT, "scripts", "coordinate_topic4_zm_offset_shards.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "topic4_zm_offset_shard_coordinator", _PATH
)
C = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(C)


def test_offset_base_contract_has_every_family_level_and_basin():
    cells = C.base_cells()
    assert len(cells) == 3 * 4 * 2
    assert len({C.cell_key(cell) for cell in cells}) == len(cells)
    assert {cell["family"] for cell in cells} == set(C.R.OFFSET_FAMILIES)
    assert {cell["initial_kind"] for cell in cells} == {"active", "low"}


def test_preentry_backfill_is_locked_and_disjoint_from_canonical_m_alone():
    cells = C.preentry_base_cells()
    assert cells
    assert {cell["family"] for cell in cells} == {
        "M_SG",
        "M_Z_recovery",
    }
    assert {C.cell_key(cell) for cell in cells} < {
        C.cell_key(cell) for cell in C.base_cells()
    }


def test_canonical_inflight_requires_a_live_writer_window(monkeypatch):
    monkeypatch.setattr(
        C,
        "_window_names",
        lambda: {"seed1_offset", "seed4_offset", "offset_coord"},
    )
    monkeypatch.setattr(
        C,
        "_canonical_next_key",
        lambda seed: {
            1: "M_alone|lambda=0.333333|active|noise_replay",
            3: "M_alone|lambda=0.333333|active|noise_replay",
            4: None,
        }[seed],
    )
    assert C.canonical_inflight_cells() == {
        (1, "M_alone|lambda=0.333333|active|noise_replay")
    }


def test_canonical_handoff_waits_for_atomic_target_row():
    targets = {
        1: "M_alone|lambda=0.333333|active|noise_replay",
        3: "M_alone|lambda=0.333333|low|noise_replay",
    }
    rows = {
        1: {
            "M_alone|lambda=0.333333|active|noise_replay": {
                "completed": True
            }
        },
        3: {
            "M_alone|lambda=0|low|noise_replay": {"completed": True}
        },
    }
    assert C.completed_handoff_seeds(targets, rows) == {1}


def test_offset_window_names_are_disjoint_for_locked_cells():
    cells = [
        *C.base_cells(),
        {
            "family": "dynamic_ZM",
            "lambda": None,
            "initial_kind": "active",
            "replicate": "noise_resample_1",
        },
    ]
    names = {
        C._window_name(seed, cell)
        for seed in C.SEEDS
        for cell in cells
    }
    assert len(names) == len(C.SEEDS) * len(cells)


def test_offset_cell_identity_is_recovered_from_python_argv():
    parsed = C._offset_cell_from_command(
        "python scripts/run_topic4_zm_offset_cell.py "
        "--seed 3 --family M_Z_recovery --lambda=1.0 "
        "--initial-kind low --replicate noise_replay --confirm-run"
    )
    assert parsed is not None
    seed, cell = parsed
    assert seed == 3
    assert C.cell_key(cell) == (
        "M_Z_recovery|lambda=1|low|noise_replay"
    )


def test_offset_cell_identity_handles_dynamic_default_level():
    parsed = C._offset_cell_from_command(
        "/usr/bin/python3 scripts/run_topic4_zm_offset_cell.py "
        "--seed 1 --family dynamic_ZM --initial-kind active "
        "--replicate noise_resample_1 --confirm-run"
    )
    assert parsed is not None
    seed, cell = parsed
    assert seed == 1
    assert C.cell_key(cell) == "dynamic_ZM|late_active|noise_resample_1"


def test_offset_cell_parser_ignores_shell_that_mentions_runner():
    assert (
        C._offset_cell_from_command(
            "/bin/bash -lc 'rg run_topic4_zm_offset_cell.py scripts/'"
        )
        is None
    )
