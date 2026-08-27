"""Synthetic recovery for the R1 persistent-state measurement instrument.

The mark is a size-one tied subset, which is an exact special case of the R1
unordered without-replacement law.  The timing term is the exact likelihood of
a piecewise-constant conditional intensity over recorded 30-s intervals.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch import nn

from .mark_likelihood import conditional_k_subset_log_prob
from .state import ControlledPersistentState


SYNTHETIC_REVISION = "r1_persistent_timing_exact_subset_recovery_v1"


@dataclass(frozen=True)
class SyntheticSequence:
    observation: torch.Tensor
    event_count: torch.Tensor
    event_anchor: torch.Tensor
    event_contact: torch.Tensor
    train_stop: int
    interval_minutes: float


def generate_synthetic(seed: int = 4, n_anchors: int = 180,
                       train_stop: int = 120,
                       truth: str = "positive") -> SyntheticSequence:
    if truth not in {"positive", "zero", "reversed"}:
        raise ValueError(f"unknown synthetic truth {truth!r}")
    rng = np.random.default_rng(int(seed))
    latent = np.zeros((n_anchors, 2), dtype=np.float32)
    angle = 0.18
    transition = 0.96 * np.asarray([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ], dtype=np.float32)
    for index in range(1, n_anchors):
        latent[index] = transition @ latent[index - 1] + rng.normal(0, 0.15, 2)
    observation = (latent + rng.normal(0, 0.18, latent.shape)).astype(np.float32)
    interval_minutes = 0.5
    effect = {"positive": 1.0, "zero": 0.0, "reversed": -1.0}[truth]
    rate = np.exp(-0.2 + effect * 1.2 * latent[:, 0])
    event_count = rng.poisson(rate * interval_minutes)
    contact_weight = effect * np.asarray([
        [0.0, 1.4], [0.0, -1.4], [1.0, 0.0], [-1.0, 0.0],
    ])
    event_anchor: list[int] = []
    event_contact: list[int] = []
    for anchor, count in enumerate(event_count):
        probability = np.exp(contact_weight @ latent[anchor])
        probability /= probability.sum()
        for _ in range(int(count)):
            event_anchor.append(anchor)
            event_contact.append(int(rng.choice(4, p=probability)))
    return SyntheticSequence(
        observation=torch.as_tensor(observation),
        event_count=torch.as_tensor(event_count, dtype=torch.float32),
        event_anchor=torch.as_tensor(event_anchor, dtype=torch.long),
        event_contact=torch.as_tensor(event_contact, dtype=torch.long),
        train_stop=int(train_stop), interval_minutes=interval_minutes,
    )


class SyntheticFilter(nn.Module):
    def __init__(self, observation_dim: int = 2, state_dim: int = 2):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(observation_dim, 8), nn.Tanh(),
        )
        self.core = ControlledPersistentState(8, state_dim)
        self.timing = nn.Linear(state_dim, 1, bias=False)
        self.contact = nn.Linear(state_dim, 4, bias=False)
        nn.init.zeros_(self.timing.weight)
        nn.init.zeros_(self.contact.weight)

    def states(self, observation: torch.Tensor,
               interval_minutes: float) -> torch.Tensor:
        output = []
        state = observation.new_zeros(self.core.dim)
        for index, embedding in enumerate(self.project(observation)):
            state = self.core.assimilate(
                state, 0.0 if index == 0 else interval_minutes, embedding
            )
            output.append(state)
        return torch.stack(output)


def _train_baseline(sequence: SyntheticSequence) -> tuple[float, torch.Tensor]:
    use = sequence.event_anchor < sequence.train_stop
    rate = float(sequence.event_count[:sequence.train_stop].sum()) / (
        sequence.train_stop * sequence.interval_minutes
    )
    frequency = torch.bincount(
        sequence.event_contact[use], minlength=4
    ).to(torch.float32) + 1.0
    return max(rate, 1e-8), torch.log(frequency / frequency.sum())


def _nll(model: SyntheticFilter, sequence: SyntheticSequence,
         region: tuple[int, int], base_rate: float,
         base_contact_logit: torch.Tensor,
         observation: torch.Tensor | None = None) -> torch.Tensor:
    lo, hi = region
    observed = sequence.observation if observation is None else observation
    state = model.states(observed, sequence.interval_minutes)
    log_rate = np.log(base_rate) + model.timing(state).squeeze(-1)
    use = (sequence.event_anchor >= lo) & (sequence.event_anchor < hi)
    anchor = sequence.event_anchor[use]
    contact = sequence.event_contact[use]
    timing = (
        torch.exp(log_rate[lo:hi]).sum() * sequence.interval_minutes
        - log_rate[anchor].sum()
    )
    logits = base_contact_logit + model.contact(state[anchor])
    target = torch.nn.functional.one_hot(contact, num_classes=4).to(torch.bool)
    candidate = torch.ones_like(target)
    mark = -conditional_k_subset_log_prob(logits, target, candidate).sum()
    return (timing + mark) / max(int(use.sum()), 1)


def _baseline_nll(sequence: SyntheticSequence, region: tuple[int, int],
                  base_rate: float, base_contact_logit: torch.Tensor) -> float:
    lo, hi = region
    use = (sequence.event_anchor >= lo) & (sequence.event_anchor < hi)
    count = max(int(use.sum()), 1)
    timing = (
        base_rate * (hi - lo) * sequence.interval_minutes
        - int(use.sum()) * np.log(base_rate)
    )
    mark = -float(base_contact_logit[sequence.event_contact[use]].sum())
    return float((timing + mark) / count)


def run_synthetic_recovery(seed: int = 4, epochs: int = 100) -> dict:
    torch.manual_seed(int(seed))
    sequence = generate_synthetic(seed)
    base_rate, base_contact = _train_baseline(sequence)
    model = SyntheticFilter()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    train_region = (0, sequence.train_stop)
    validation_region = (sequence.train_stop, len(sequence.observation))
    for _ in range(int(epochs)):
        loss = _nll(
            model, sequence, train_region, base_rate, base_contact
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    permutation = torch.arange(len(sequence.observation))
    validation = permutation[sequence.train_stop:].roll(17)
    permutation[sequence.train_stop:] = validation
    with torch.no_grad():
        correct = float(_nll(
            model, sequence, validation_region, base_rate, base_contact
        ))
        wrong_time = float(_nll(
            model, sequence, validation_region, base_rate, base_contact,
            sequence.observation[permutation],
        ))
    baseline = _baseline_nll(
        sequence, validation_region, base_rate, base_contact
    )
    return {
        "status": "COMPLETE",
        "synthetic_revision": SYNTHETIC_REVISION,
        "seed": int(seed), "epochs": int(epochs),
        "n_anchors": int(len(sequence.observation)),
        "n_train_events": int((sequence.event_anchor < sequence.train_stop).sum()),
        "n_validation_events": int((sequence.event_anchor >= sequence.train_stop).sum()),
        "baseline_validation_nll_per_event": baseline,
        "filtered_validation_nll_per_event": correct,
        "wrong_time_validation_nll_per_event": wrong_time,
        "filtered_minus_baseline": correct - baseline,
        "filtered_minus_wrong_time": correct - wrong_time,
        "recovered": bool(correct < baseline and correct < wrong_time),
        "claim_boundary": (
            "instrument recovery on an in-family synthetic truth; not biological evidence"
        ),
    }
