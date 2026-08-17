from pathlib import Path


def test_human_runner_has_no_v31_transition_import():
    source = Path(
        "scripts/run_topic5_event_innovation_v3_0_human_test.py"
    ).read_text(encoding="utf-8")
    assert "topic5_event_innovation_transition_v3_1" not in source
