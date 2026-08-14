import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.aggregate_topic4_rev10_r_edge_flow_screen import (
    _incoming_e_error_by_pathway,
    _mode_shape_scores,
    returned_only_onsets,
)
from scripts.launch_topic4_rev10_r_edge_flow_screen import (
    NUMERIC_ENV,
    _memory_bounded_workers,
    _peak_rss_kib,
)
from scripts.run_topic4_rev10_r_edge_flow_worker import (
    _load_basis,
    active_network_seeds,
)


ROOT = Path(__file__).resolve().parents[1]


def _score(status_a, status_b, value_a=None, value_b=None):
    modes = {}
    for mode, status, value in ((0, status_a, value_a), (1, status_b, value_b)):
        modes[str(mode)] = {"status": status}
        if status == "OK":
            modes[str(mode)]["objective"] = {"mode_score": value}
    return {"modes": modes}


def test_mode_shape_fallback_is_independent_between_A_and_B():
    score6 = _score("INSUFFICIENT_MODE_SUPPORT", "OK", None, 1.5)
    score3 = _score("OK", "OK", 2.5, 2.0)
    values, sources = _mode_shape_scores(score6, score3)
    assert values == {"A": 2.5, "B": 1.5}
    assert sources == {"A": "n3_fallback", "B": "n6"}


def test_only_unsupported_mode_receives_penalty():
    score6 = _score("INSUFFICIENT_MODE_SUPPORT", "OK", None, 1.5)
    score3 = _score("INSUFFICIENT_MODE_SUPPORT", "OK", None, 2.0)
    values, sources = _mode_shape_scores(score6, score3)
    assert values == {"A": 8.0, "B": 1.5}
    assert sources["A"] == "unsupported_penalty"
    assert sources["B"] == "n6"


def test_returned_only_scoring_excludes_nonself_terminated_events():
    onsets = np.arange(12, dtype=float).reshape(3, 4)
    selected = returned_only_onsets(onsets, np.asarray([True, False, True]))
    np.testing.assert_array_equal(selected, onsets[[0, 2]])


def test_incoming_e_audit_accepts_legacy_and_pathway_specific_schema():
    assert _incoming_e_error_by_pathway({
        "max_abs_incoming_E_error": 1e-12,
    }) == {"E_to_E": 1e-12}
    assert _incoming_e_error_by_pathway({
        "pathway_audit": {
            "E_to_E": {"max_abs_incoming_error": 2e-12},
            "E_to_I": {"max_abs_incoming_error": 3e-12},
        },
    }) == {"E_to_E": 2e-12, "E_to_I": 3e-12}


def test_edge_worker_keeps_node_anchor_and_basis_separate():
    source = (ROOT / "scripts/run_topic4_rev10_r_edge_flow_worker.py").read_text()
    assert "_candidate_node" in source
    assert "graph_spectral_ee_flow" in source
    assert "spatial_vector_ee_flow" in source
    assert "config[\"node_anchor\"]" in source
    assert "patient_train_onsets" not in source
    assert set(inspect.signature(_load_basis).parameters) == {
        "npz_path", "record", "seed",
    }


def test_active_network_seeds_are_phase_explicit():
    search = {
        "fit_network_seeds": [1], "selection_network_seeds": [2],
        "confirmation_network_seeds": [3],
    }
    for phase, expected in (("fit", [1]), ("selection", [2]), ("confirmation", [3])):
        assert active_network_seeds({"search": {**search, "phase": phase}}) == expected


def test_aggregator_declares_equal_network_primary_unit():
    source = (
        ROOT / "scripts/aggregate_topic4_rev10_r_edge_flow_screen.py"
    ).read_text()
    assert "selection_score_equal_network" in source
    assert "mean_network_shape_A" in source
    assert "mean_network_shape_B" in source
    assert 'loaded["event_returned"]' in source
    assert '"fit": "fit_screen"' in source
    assert '"selection": "selection"' in source
    assert '"confirmation": "confirmation"' in source
    assert "pooled" not in source.split("def main():", 1)[1].split(
        "safe_claim", 1
    )[0]


def test_screen_launcher_uses_measured_rss_and_memory_headroom(tmp_path):
    log = tmp_path / "sentinel.log"
    log.write_text("Maximum resident set size (kbytes): 6291456\n")
    assert _peak_rss_kib(log) == 6291456
    config = {"execution": {
        "minimum_available_memory_gib_per_screen_worker": 8.0,
        "screen_max_workers": 12,
    }}
    # 6 GiB measured -> 9 GiB reserved per worker; half of 180 GiB allows 10.
    assert _memory_bounded_workers(
        config, 6291456, 180 * 1024 ** 2,
    ) == 10
    assert NUMERIC_ENV and set(NUMERIC_ENV.values()) == {"1"}
    source = (
        ROOT / "scripts/launch_topic4_rev10_r_edge_flow_screen.py"
    ).read_text()
    assert '"/usr/bin/nohup"' in source
    assert '"--property=MemoryMax=24G"' in source
    assert "time.sleep(wait_seconds)" in source


def test_screen_launcher_is_directly_importable_as_cli():
    completed = subprocess.run(
        [sys.executable, str(
            ROOT / "scripts/launch_topic4_rev10_r_edge_flow_screen.py"
        ), "--help"],
        cwd="/tmp", capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
