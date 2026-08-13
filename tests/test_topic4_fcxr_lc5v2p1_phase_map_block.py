import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "lc5v2p1_block", ROOT / "scripts/run_topic4_fcxr_lc5v2p1_phase_map_block.py"
)
BLOCK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BLOCK)


def test_pending_cells_excludes_only_locked_reuse_or_completed(monkeypatch, tmp_path):
    _, manifest, cells = BLOCK.MAP.load_manifest()
    monkeypatch.setattr(BLOCK, "OUT", tmp_path)
    pending = BLOCK.pending_cells(manifest, cells)
    assert "tau3000_gamma0010" not in pending
    assert "tau8000_gamma0005" not in pending
    assert len(pending) == 7
    tau, gamma = cells["tau3000_gamma0005"]
    tag = BLOCK.MAP.PREFIX._tag(gamma, "q099", tau, manifest["experiment_id"])
    (tmp_path / tag).mkdir()
    (tmp_path / tag / "summary.json").write_text("{}")
    assert len(BLOCK.pending_cells(manifest, cells)) == 6


def test_four_worker_memory_gate_matches_locked_multipliers():
    _, manifest, _ = BLOCK.MAP.load_manifest()
    assert BLOCK.required_memavailable_gib(4, manifest) == pytest.approx(122.4)
    assert BLOCK.required_memavailable_gib(3, manifest) == pytest.approx(91.8)


def test_boundary_block_has_independent_lock_and_sentinels():
    assert BLOCK._block_stem(BLOCK.MAP.BASE_EXPERIMENT) == "lc5v2p1_phase_map"
    assert BLOCK._block_stem(BLOCK.MAP.BOUNDARY_EXPERIMENT) == "lc5v2p1_boundary_patch"
