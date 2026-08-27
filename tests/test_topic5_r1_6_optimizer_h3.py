from src.topic5_continuous_marked_state_r1.optimizer_h3 import (
    R1_6_MINIMAL_H3_REVISION,
)
from scripts.topic5_continuous_marked_state_r1.run_r1_6_minimal_h3_cell import (
    SCALE_EVENTS,
    contrast,
)
from scripts.topic5_continuous_marked_state_r1.run_r1_6_minimal_h3_queue import (
    valid,
)


def test_minimal_h3_contract_is_frozen_to_n1000():
    assert SCALE_EVENTS == 1000
    assert "r1_6" in R1_6_MINIMAL_H3_REVISION


def test_minimal_h3_contrast_omits_denominators():
    value = contrast(
        {"joint_nll_per_event": 2.0, "n_events": 9},
        {"joint_nll_per_event": 3.0, "n_events": 9},
    )
    assert value == {"joint_nll_per_event": -1.0}


def test_minimal_h3_queue_rejects_nonstable_t1(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(__import__("json").dumps({
        "status": "COMPLETE",
        "revision": R1_6_MINIMAL_H3_REVISION,
        "subject": "subject",
        "seed": 1,
        "source": "load",
        "scale_events": 1000,
        "t1": {"seed_stable_t1": False},
        "formal_test_partition_opened": False,
        "sealed_opened": False,
    }))
    assert not valid(path, "subject", 1, "load")
