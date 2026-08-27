"""Optimizer-specific synthetic recovery for the R1.6 training audit."""
from __future__ import annotations

import copy
import math

import torch

from .optimizer_audit import R1_6_REVISION
from .synthetic_recovery import (
    SyntheticFilter,
    _baseline_nll,
    _nll,
    _train_baseline,
    generate_synthetic,
)


SYNTHETIC_OPTIMIZER_REVISION = "r1_6_nested_optimizer_synthetic_v1"


def run_optimizer_synthetic(*, seed: int, truth: str,
                            n_anchors: int = 300,
                            epochs: int = 80,
                            optimizer_name: str = "adamw",
                            learning_rate: float = 3e-3,
                            weight_decay: float = 0.0,
                            grad_clip_norm: float | None = 1.0,
                            warmup_fraction: float = 0.0) -> dict:
    """Select on a middle block and report a never-used chronological test."""
    if n_anchors < 100:
        raise ValueError("optimizer synthetic needs at least 100 anchors")
    base_stop = int(math.floor(0.6 * n_anchors))
    selection_stop = int(math.floor(0.8 * n_anchors))
    torch.manual_seed(int(seed))
    sequence = generate_synthetic(
        seed=seed, n_anchors=n_anchors, train_stop=base_stop, truth=truth
    )
    base_rate, base_contact = _train_baseline(sequence)
    model = SyntheticFilter()
    initial = copy.deepcopy(model.state_dict())
    optimizer_class = {
        "adamw": torch.optim.AdamW,
        "adam": torch.optim.Adam,
    }.get(optimizer_name)
    if optimizer_class is None:
        raise ValueError(f"unsupported synthetic optimizer {optimizer_name!r}")
    optimizer = optimizer_class(
        model.parameters(), lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    train_region = (0, base_stop)
    selection_region = (base_stop, selection_stop)
    test_region = (selection_stop, n_anchors)
    with torch.no_grad():
        best_value = float(_nll(
            model, sequence, selection_region, base_rate, base_contact
        ))
        epoch_zero_train = float(_nll(
            model, sequence, train_region, base_rate, base_contact
        ))
    best_epoch = 0
    best_state = copy.deepcopy(initial)
    trajectory = [{
        "epoch": 0, "train_nll": epoch_zero_train,
        "selection_nll": best_value, "preclip_norm": None,
        "clipped": False,
    }]
    warmup_steps = int(math.ceil(float(warmup_fraction) * max(epochs, 1)))
    for epoch in range(1, int(epochs) + 1):
        if warmup_steps:
            factor = min(1.0, float(epoch) / float(warmup_steps))
            for group in optimizer.param_groups:
                group["lr"] = float(learning_rate) * factor
        loss = _nll(model, sequence, train_region, base_rate, base_contact)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip_norm is None:
            preclip = math.sqrt(sum(
                float(value.grad.detach().float().square().sum())
                for value in model.parameters() if value.grad is not None
            ))
        else:
            preclip = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(grad_clip_norm)
            ))
        if not math.isfinite(preclip):
            raise RuntimeError("synthetic optimizer produced a non-finite gradient")
        optimizer.step()
        with torch.no_grad():
            train_value = float(_nll(
                model, sequence, train_region, base_rate, base_contact
            ))
            selection_value = float(_nll(
                model, sequence, selection_region, base_rate, base_contact
            ))
        trajectory.append({
            "epoch": int(epoch), "train_nll": train_value,
            "selection_nll": selection_value, "preclip_norm": preclip,
            "clipped": bool(
                grad_clip_norm is not None and preclip > float(grad_clip_norm)
            ),
        })
        if selection_value < best_value:
            best_value = selection_value
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    permutation = torch.arange(n_anchors)
    permutation[selection_stop:] = permutation[selection_stop:].roll(17)
    with torch.no_grad():
        correct = float(_nll(
            model, sequence, test_region, base_rate, base_contact
        ))
        wrong = float(_nll(
            model, sequence, test_region, base_rate, base_contact,
            sequence.observation[permutation],
        ))
    baseline = _baseline_nll(sequence, test_region, base_rate, base_contact)
    parameter_update = math.sqrt(sum(
        float((value.detach().cpu() - initial[key]).float().square().sum())
        for key, value in model.state_dict().items()
    ))
    is_signal = truth != "zero"
    recovered = bool(
        correct < baseline and correct < wrong and best_epoch > 0
    ) if is_signal else bool(
        correct >= baseline - 0.02
    )
    return {
        "status": "COMPLETE",
        "revision": R1_6_REVISION,
        "synthetic_revision": SYNTHETIC_OPTIMIZER_REVISION,
        "seed": int(seed), "truth": truth,
        "n_anchors": int(n_anchors), "epochs": int(epochs),
        "optimizer": optimizer_name,
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "grad_clip_norm": (
            None if grad_clip_norm is None else float(grad_clip_norm)
        ),
        "warmup_fraction": float(warmup_fraction),
        "selected_epoch": int(best_epoch),
        "selection_nll": float(best_value),
        "test_nll": correct,
        "wrong_time_test_nll": wrong,
        "baseline_test_nll": baseline,
        "test_minus_baseline": correct - baseline,
        "test_minus_wrong_time": correct - wrong,
        "parameter_update_norm": parameter_update,
        "recovered": recovered,
        "trajectory": trajectory,
        "formal_test_partition_opened": False,
        "sealed_opened": False,
    }
