import importlib.util
from pathlib import Path

import numpy as np

from src.topic4_fcxr_lc6_surround import EToIGraph, graph_sha256


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_topic4_fcxr_lc6a_two_hop.py"
SPEC = importlib.util.spec_from_file_location("lc6a_twohop_runner", SCRIPT)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_graph_loader_rejects_hash_drift(tmp_path):
    graph = EToIGraph(
        np.array([[0, 1]], np.int32), np.ones((1, 2)), np.ones((1, 2), np.int32),
    )
    path = tmp_path / "graph.npz"
    np.savez_compressed(
        path,
        sources=graph.sources,
        weights=graph.weights,
        delay_steps=graph.delay_steps,
        graph_sha256=np.asarray([graph_sha256(graph)]),
        metadata_json=np.asarray(["{\"graph_sha256\": \"wrong\"}"]),
    )
    try:
        RUNNER._load_graph(path)
    except RuntimeError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("metadata/hash drift must fail closed")


def test_two_hop_runner_is_graph_only_and_has_required_readouts():
    text = SCRIPT.read_text()
    assert "run_fcxr_loop" not in text
    assert "trajectory_outcome_used" in text
    assert "q_parallel_two_hop" in (ROOT / "src/topic4_fcxr_lc6_twohop.py").read_text()
    assert "surround_center_ratio" in text
    assert "q95_ms" in text
