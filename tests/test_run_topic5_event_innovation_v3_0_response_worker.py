from pathlib import Path

from scripts import run_topic5_event_innovation_v3_0_response_worker as worker


def test_worker_uses_the_frozen_patient_functions():
    assert worker.RUNNERS["local"].run_subject.__module__.endswith(
        "run_topic5_event_innovation_v3_0_local_response"
    )
    assert worker.RUNNERS["cumulative"].run_subject.__module__.endswith(
        "run_topic5_event_innovation_v3_0_cumulative_response"
    )

def test_output_roots_are_separate():
    config, _, _ = worker.load_contract(
        Path("config/topic5_event_innovation_v3_0.yaml").resolve()
    )
    assert worker.output_root("local", config) != worker.output_root(
        "cumulative", config
    )
