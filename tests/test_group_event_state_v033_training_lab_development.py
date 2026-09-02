"""One-time development release and seed-first reduction tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.topic5_group_event_state.v033_training_lab.development import (
    RELEASE_FORMAT,
    RELEASE_STATUS,
    merge_development_scores,
    validate_development_release,
)
from src.topic5_group_event_state.v033_training_lab.paths import file_hash


def _release(tmp_path: Path):
    request = tmp_path / "request.json"
    card = tmp_path / "card.json"
    learned = [tmp_path / f"learned_{seed}.pt" for seed in (1, 2)]
    random = [tmp_path / f"random_{seed}.pt" for seed in (1, 2)]
    request.write_text("request")
    card.write_text("card")
    for i, path in enumerate([*learned, *random]):
        path.write_bytes(f"checkpoint-{i}".encode())
    checkpoints = []
    for i, seed in enumerate((1, 2)):
        checkpoints.append({
            "seed": seed,
            "learned_checkpoint_path": str(learned[i]),
            "learned_checkpoint_sha256": file_hash(learned[i]),
            "random_reservoir_checkpoint_path": str(random[i]),
            "random_reservoir_checkpoint_sha256": file_hash(random[i]),
        })
    payload = {
        "format": RELEASE_FORMAT, "status": RELEASE_STATUS, "sealed": False, "development_only": True,
        "selection_feedback_forbidden": True, "retraining_after_open_forbidden": True,
        "requests": {"req": {
            "request_path": str(request), "request_sha256": file_hash(request),
            "corrected_card_path": str(card), "corrected_card_sha256": file_hash(card),
            "checkpoints": checkpoints,
        }},
    }
    return payload, request, card, learned[0]


def test_one_time_release_hash_locks_cards_and_both_checkpoint_arms(tmp_path):
    release, request, card, learned = _release(tmp_path)
    entry = validate_development_release(release, request_id="req", request_path=request, card_path=card)
    assert len(entry["checkpoints"]) == 2
    learned.write_bytes(b"changed")
    with pytest.raises(ValueError, match="missing or changed"):
        validate_development_release(release, request_id="req", request_path=request, card_path=card)


def test_development_reduction_merges_seeds_before_patient_time_blocks():
    n = 24
    h = np.full(n, 10.0)
    # Seed 0 is unfavourable, but the per-anchor median of three frozen seeds is favourable.
    learned = np.asarray([np.full(n, 10.2), np.full(n, 9.8), np.full(n, 9.7)])
    shifted = learned + 0.1
    random = learned + 0.2
    result = merge_development_scores(
        nll_h=h, nll_learned_by_seed=learned, nll_shifted_by_seed=shifted,
        nll_random_by_seed=random, shift_valid=np.ones(n, dtype=bool),
        segments=np.repeat(np.arange(4), 6),
    )
    assert result["H_minus_learned"]["mean"] == pytest.approx(0.2)
    assert result["shifted_minus_correct"]["mean"] == pytest.approx(0.1)
    assert result["random_minus_learned"]["mean"] == pytest.approx(0.2)
    reversed_result = merge_development_scores(
        nll_h=h, nll_learned_by_seed=learned[::-1], nll_shifted_by_seed=shifted[::-1],
        nll_random_by_seed=random[::-1], shift_valid=np.ones(n, dtype=bool),
        segments=np.repeat(np.arange(4), 6),
    )
    assert result["H_minus_learned"] == reversed_result["H_minus_learned"]
