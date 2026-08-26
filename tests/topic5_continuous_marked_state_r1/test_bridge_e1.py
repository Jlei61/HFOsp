from __future__ import annotations

from pathlib import Path

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.bridge_e1 import build_bridge_e1_design


def test_real_small_bridge_e1_design_is_event_independent_and_exact() -> None:
    subject = "epilepsiae_620"
    checkpoint = (
        contract.RESULT_ROOT / "baselines" / subject / "seed_0/models.pt"
    )
    design, _, manifest = build_bridge_e1_design(
        subject, checkpoint, max_train_anchors=8,
        max_validation_anchors=4, quadrature_order=2,
    )
    design.validate()
    assert manifest["selection_is_event_independent"] is True
    assert manifest["support_is_post_anchor_recorded_time"] is True
    assert manifest["n_train_anchors"] == 8
    assert manifest["n_validation_anchors"] == 4
    assert design.explicit.shape[1:] == (31, 13)
