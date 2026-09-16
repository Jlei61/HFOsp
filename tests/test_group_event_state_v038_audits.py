from __future__ import annotations

import copy

import pytest
import torch

from scripts.audit_group_event_state_v038_dual_random_background import (
    _fit_only_repertoire_alignment,
    _parent_parity_audit,
)
from src.topic5_group_event_state.v037.h1_dual_train import NestedDualReadout


def test_random_background_control_aligns_repertoire_labels_on_fit_only() -> None:
    current_label = torch.tensor([0, 1, 2, 0, 1, 2])
    mixture = torch.nn.functional.one_hot(current_label, num_classes=3).float()[:, None, :]
    # Source label order is current [2, 0, 1], so current labels map to source
    # indices [1, 2, 0].
    source_label = torch.tensor([1, 2, 0, 1, 2, 0])
    logits = torch.full((6, 1, 3), -8.0)
    logits[torch.arange(6), 0, source_label] = 8.0
    target = {"mixture": mixture}
    valid = {"mixture": torch.ones((6, 1), dtype=torch.bool)}
    prediction = {"mixture": logits}
    aligned, permutation, loss = _fit_only_repertoire_alignment(
        target, valid, prediction, torch.arange(4),
    )
    assert permutation == (2, 0, 1)
    assert loss < 1e-4
    assert torch.equal(aligned["mixture"].argmax(-1)[:, 0], source_label)


def _score() -> dict:
    endpoints = {name: 1.0 for name in NestedDualReadout.ENDPOINTS}
    return {
        "total": 1.0,
        "endpoints": endpoints,
        "by_horizon": {
            "1800": {"total": 1.0, "endpoints": dict(endpoints)},
        },
    }


def test_parent_parity_excludes_only_small_reconstructed_mixture_drift() -> None:
    frozen = _score()
    rebuilt = copy.deepcopy(frozen)
    rebuilt["endpoints"]["mixture"] += 6e-4
    rebuilt["by_horizon"]["1800"]["endpoints"]["mixture"] += 9e-4
    audit = _parent_parity_audit(rebuilt, frozen, (1800.0,))
    assert audit["excluded_endpoints"] == ["mixture"]
    assert not audit["all_endpoint_total_auditable"]
    assert set(audit["auditable_endpoints"]) == set(NestedDualReadout.ENDPOINTS) - {"mixture"}


def test_parent_parity_never_excuses_non_repertoire_drift() -> None:
    frozen = _score()
    rebuilt = copy.deepcopy(frozen)
    rebuilt["by_horizon"]["1800"]["endpoints"]["count"] += 8e-4
    with pytest.raises(ValueError, match="count"):
        _parent_parity_audit(rebuilt, frozen, (1800.0,))


def test_parent_parity_excludes_small_reconstructed_mark_drift() -> None:
    frozen = _score()
    rebuilt = copy.deepcopy(frozen)
    rebuilt["endpoints"]["mark"] -= 1.2e-3
    rebuilt["by_horizon"]["1800"]["endpoints"]["mark"] -= 2.4e-3
    audit = _parent_parity_audit(rebuilt, frozen, (1800.0,))
    assert audit["excluded_endpoints"] == ["mark"]
    assert not audit["all_endpoint_total_auditable"]
    assert set(audit["auditable_endpoints"]) == set(NestedDualReadout.ENDPOINTS) - {"mark"}


def test_parent_parity_rejects_large_reconstructed_mark_drift() -> None:
    frozen = _score()
    rebuilt = copy.deepcopy(frozen)
    rebuilt["by_horizon"]["1800"]["endpoints"]["mark"] += 4.5e-3
    with pytest.raises(ValueError, match="mark"):
        _parent_parity_audit(rebuilt, frozen, (1800.0,))
