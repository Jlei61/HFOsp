from __future__ import annotations

import torch

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.bridge_e1 import build_bridge_e1_design
from src.topic5_continuous_marked_state_r1.data import load_event_stream
from src.topic5_continuous_marked_state_r1.t1_pilot import (
    PersistentEventModel, evaluate_t1, matched_wrong_time_permutation,
)


def test_real_t1_initial_state_residual_is_exactly_zero() -> None:
    subject = "epilepsiae_620"
    checkpoint_path = contract.RESULT_ROOT / "baselines" / subject / "seed_0/models.pt"
    baseline = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    stream = load_event_stream(subject)
    design, reader, _ = build_bridge_e1_design(
        subject, checkpoint_path, max_train_anchors=8,
        max_validation_anchors=4, quadrature_order=2,
    )
    model = PersistentEventModel(
        baseline, design.event_history.shape[1], stream.n_contacts,
        stream.adjacency, design.explicit.shape[2], raw_enabled=False,
    )
    filtered = evaluate_t1(model, design, reader, "validation", device="cpu")
    all_correction_off = evaluate_t1(
        model, design, reader, "validation", device="cpu",
        correction_enabled=False,
    )
    validation_correction_off = evaluate_t1(
        model, design, reader, "validation", device="cpu",
        validation_correction_off=True,
    )
    assert filtered == all_correction_off
    assert filtered == validation_correction_off
    with torch.no_grad():
        model.state_timing.weight.fill_(0.01)
        model.state_contact.weight.fill_(0.01)
        model.state_size.weight.fill_(0.01)
    nonzero = evaluate_t1(model, design, reader, "validation", device="cpu")
    identity_swap = evaluate_t1(
        model, design, reader, "validation", device="cpu",
        state_permutation=torch.arange(len(design.anchor_time)).numpy(),
    )
    assert nonzero == identity_swap
    permutation, matched = matched_wrong_time_permutation(
        design, split="validation", min_separation_seconds=0.1
    )
    assert permutation.shape == (len(design.anchor_time),)
    assert matched.shape == permutation.shape
    assert (permutation[~matched] == torch.arange(len(permutation)).numpy()[~matched]).all()
