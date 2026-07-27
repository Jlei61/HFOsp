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
