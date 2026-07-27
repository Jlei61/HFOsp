from __future__ import annotations

import importlib.util
import os


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH = os.path.join(
    _ROOT, "scripts", "coordinate_topic4_zm_entry_shards.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "topic4_zm_entry_shard_coordinator", _PATH
)
C = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(C)


def test_coordinator_base_contract_is_exactly_locked_levels():
    assert C.expected_base_keys() == {
        "lambda=0|noise_replay",
        "lambda=0.25|noise_replay",
        "lambda=0.5|noise_replay",
        "lambda=0.75|noise_replay",
        "lambda=1|noise_replay",
    }


def test_coordinator_uses_distinct_expansion_window_names():
    names = {
        C._cell_window(seed, lam, replicate)
        for seed in C.SEEDS
        for lam in (0.25, 0.5)
        for replicate in C.R.ENTRY_EXPANSION_REPLICATES
    }
    assert len(names) == 3 * 2 * 2


def _row(lam, entered, replicate="noise_replay"):
    return {
        "key": f"lambda={lam:g}|{replicate}",
        "lambda": float(lam),
        "replicate": replicate,
        "entered_carrier": bool(entered),
        "completed": True,
    }


def test_early_expansion_waits_for_complete_base_grid():
    rows = {
        row["key"]: row
        for row in (
            _row(0.0, False),
            _row(0.25, True),
            _row(0.50, True),
            _row(0.75, True),
        )
    }
    assert C.early_expansion_cells(rows) == []


def test_early_expansion_uses_only_registered_bracket_endpoints():
    rows = {
        row["key"]: row
        for row in (
            _row(0.0, False),
            _row(0.25, True),
            _row(0.50, True),
            _row(0.75, True),
            _row(1.0, True),
        )
    }
    pending = C.early_expansion_cells(rows)
    assert {
        (cell["lambda"], cell["replicate"]) for cell in pending
    } == {
        (0.0, "noise_resample_1"),
        (0.0, "noise_resample_2"),
        (0.25, "noise_resample_1"),
        (0.25, "noise_resample_2"),
    }


def test_early_expansion_skips_existing_part_rows():
    rows = {
        row["key"]: row
        for row in (
            _row(0.0, False),
            _row(0.25, True),
            _row(0.50, True),
            _row(0.75, True),
            _row(1.0, True),
            _row(0.0, False, "noise_resample_1"),
        )
    }
    assert (
        0.0,
        "noise_resample_1",
    ) not in {
        (cell["lambda"], cell["replicate"])
        for cell in C.early_expansion_cells(rows)
    }


def test_entry_cell_identity_is_recovered_from_python_argv():
    assert C._entry_cell_from_command(
        "/opt/env/bin/python scripts/run_topic4_zm_entry_cell.py "
        "--seed 4 --lambda=0.75 --replicate noise_resample_2 "
        "--confirm-run"
    ) == (4, "lambda=0.75|noise_resample_2")


def test_entry_cell_parser_ignores_shell_that_mentions_runner():
    assert (
        C._entry_cell_from_command(
            "/bin/bash -lc 'rg run_topic4_zm_entry_cell.py scripts/'"
        )
        is None
    )


def test_canonical_handoff_requires_every_missing_cell_in_flight():
    missing = [
        "lambda=0.5|noise_replay",
        "lambda=1|noise_replay",
    ]
    live = {
        (1, "lambda=0.5|noise_replay"),
        (1, "lambda=1|noise_replay"),
    }
    assert C.missing_covered_by_shards(1, missing, live)
    assert not C.missing_covered_by_shards(
        4, missing, {(4, "lambda=1|noise_replay")}
    )
