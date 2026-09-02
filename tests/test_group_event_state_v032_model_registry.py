from __future__ import annotations

import json

from src.topic5_group_event_state.v032_model.registry import (
    REGISTRY_FORMAT,
    write_frozen_registry,
)


def test_registry_merges_seeds_without_dropping_existing_entries(tmp_path):
    path = tmp_path / "frozen_state_registry.json"
    one = {"status": "complete", "arrays_path": "/tmp/a.npz", "state_dim": 12,
           "selection_phase": "dev_val", "open_loop": True}
    two = {"status": "complete", "arrays_path": "/tmp/b.npz", "state_dim": 12,
           "selection_phase": "dev_val", "open_loop": True}
    write_frozen_registry([("p1", 1, one)], path=path)
    out = write_frozen_registry([("p1", 2, two)], path=path)
    assert out["format"] == REGISTRY_FORMAT
    assert out["n_complete_entries"] == 2
    assert set(out["patients"]["p1"]["seeds"]) == {"1", "2"}
    assert json.loads(path.read_text()) == out

