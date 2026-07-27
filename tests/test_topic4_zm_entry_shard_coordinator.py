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
