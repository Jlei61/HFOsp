from __future__ import annotations

from pathlib import Path

import pytest

from scripts import continue_topic4_rev17_node_pipeline as controller


def test_controller_keeps_all_other_mechanisms_closed():
    assert controller.NUMERIC_ENV["OMP_NUM_THREADS"] == "1"
    source = Path(controller.__file__).read_text()
    assert '"EE_EtoI_ZM": "off"' in source
    assert "topic4_rev17_node_intervention" in source
    assert "plot_topic4_rev17_node_final_fig4" in source
    assert "plot_topic4_rev17_node_causal_validation" in source
    assert "E_to_E_dose" not in source
    assert "E_to_I_dose" not in source
    assert "runtime_mode" not in source


def test_controller_commit_rejects_unrelated_dirty_paths(monkeypatch, tmp_path):
    config = tmp_path / "config.json"
    config.write_text("{}\n")
    monkeypatch.setattr(controller, "ROOT", tmp_path)
    monkeypatch.setattr(controller, "_worktree_status", lambda: [" M unrelated.py"])
    with pytest.raises(RuntimeError, match="unrelated changes"):
        controller._commit_generated(config, "test")


@pytest.mark.parametrize(
    ("confirmation", "postselection", "final_science", "intervention", "expected"),
    [
        (False, True, True, True, "STOP_UNSEEN_CONFIRMATION_REJECTED"),
        (True, False, True, True, "STOP_NATURAL_KMEANS_REJECTED"),
        (True, True, False, True, "STOP_HELDOUT_OR_TOPOLOGY_REJECTED"),
        (True, True, True, False, "STOP_HOTSPOT_NOT_SELECTIVE"),
    ],
)
def test_execute_stops_at_first_failed_scientific_layer(
    monkeypatch, confirmation, postselection, final_science, intervention, expected,
):
    monkeypatch.setattr(controller, "_worktree_status", lambda: [])
    monkeypatch.setattr(controller, "_wait_for_atlas", lambda *_: {})
    monkeypatch.setattr(
        controller, "_selection", lambda: {"selected_candidate_id": "candidate"},
    )
    monkeypatch.setattr(controller, "_confirmation", lambda: {
        "scientific_confirmation": {"accepted": confirmation},
    })
    monkeypatch.setattr(controller, "_postselection", lambda: {
        "status": (
            "REV17_NODE_POSTSELECTION_ACCEPTED" if postselection else "REJECTED"
        ),
    })
    monkeypatch.setattr(controller, "_final_science", lambda: {
        "status": (
            "REV17_NODE_FINAL_SCIENCE_ADVANCES_TO_INTERVENTION"
            if final_science else "REJECTED"
        ),
    })
    monkeypatch.setattr(controller, "_intervention", lambda: {
        "status": "REV17_NODE_FIELD_FROZEN" if intervention else "REJECTED",
    })
    rendered = []
    monkeypatch.setattr(controller, "_render", lambda: rendered.append(True))
    assert controller.execute(interval_seconds=60) == expected
    assert not rendered
