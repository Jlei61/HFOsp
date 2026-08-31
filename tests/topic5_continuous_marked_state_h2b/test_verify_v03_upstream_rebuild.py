from __future__ import annotations

from pathlib import Path

import torch

from scripts.topic5_continuous_marked_state_h2b.verify_v03_upstream_rebuild import (
    _history_baseline_equal,
)


def _write_checkpoint(path: Path, value: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": {
            "timing_baseline.weight": value.clone(),
            "mark_baseline.weight": value.clone(),
        },
    }, path)


def test_history_baseline_requires_every_seed_and_tensor(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.pt"
    torch.save({
        "timing": {"history": {"weight": torch.tensor([1.0])}},
        "mark": {"history": {"weight": torch.tensor([1.0])}},
    }, baseline)
    good = tmp_path / "seed_0/model.pt"
    bad = tmp_path / "seed_1/model.pt"
    _write_checkpoint(good, torch.tensor([1.0]))
    _write_checkpoint(bad, torch.tensor([2.0]))

    passed, rows = _history_baseline_equal(baseline, [good, bad])
    assert passed is False
    assert rows[0]["all_history_tensors_bitwise_equal"] is True
    assert rows[1]["all_history_tensors_bitwise_equal"] is False
