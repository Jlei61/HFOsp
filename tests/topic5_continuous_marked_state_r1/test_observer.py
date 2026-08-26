from __future__ import annotations

import torch

from src.topic5_continuous_marked_state_r1.observer import (
    ObservationTransformer,
    copy_common_observer_state,
)


def _batch():
    torch.manual_seed(4)
    batch, contacts, samples = 2, 5, 512
    return {
        "explicit": torch.randn(batch, contacts, 13),
        "waveform": torch.randn(batch, contacts, samples),
        "sample_valid": torch.ones(batch, contacts, samples, dtype=torch.bool),
        "contact_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool),
        "coordinates": torch.randn(batch, contacts, 3),
        "coordinate_valid": torch.ones(batch, contacts, dtype=torch.bool),
        "shaft_index": torch.tensor([[0, 0, 1, 1, 2], [0, 1, 1, 0, 0]]),
    }


def test_zero_raw_gain_has_exact_explicit_arm_parity() -> None:
    torch.manual_seed(2)
    explicit = ObservationTransformer(13, d_model=16, patch_samples=64,
                                      n_heads=4, raw_enabled=False)
    torch.manual_seed(3)
    raw = ObservationTransformer(13, d_model=16, patch_samples=64,
                                 n_heads=4, raw_enabled=True)
    copy_common_observer_state(explicit, raw)
    batch = _batch()
    first = explicit(**batch)
    second = raw(**batch)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_raw_residual_can_enter_and_masks_padded_contacts() -> None:
    model = ObservationTransformer(13, d_model=16, patch_samples=64,
                                   n_heads=4, raw_enabled=True)
    batch = _batch()
    with torch.no_grad():
        model.raw_gain.fill_(0.2)
    first = model(**batch)
    altered = {key: value.clone() for key, value in batch.items()}
    altered["waveform"][1, 3:] = 1e5  # padded contacts must be invisible spatially
    second = model(**altered)
    torch.testing.assert_close(first[1], second[1], rtol=1e-5, atol=1e-5)
    loss = first.square().mean()
    loss.backward()
    assert torch.isfinite(model.raw_gain.grad)


def test_all_invalid_raw_contact_does_not_create_nan() -> None:
    model = ObservationTransformer(13, d_model=16, patch_samples=64,
                                   n_heads=4, raw_enabled=True)
    batch = _batch()
    batch["sample_valid"][0, 2] = False
    output = model(**batch)
    assert torch.isfinite(output).all()
