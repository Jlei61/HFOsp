"""Training loop: persistent-state TBPTT over the cohort's chronological events.

The forward state is never reset by truncation -- only the gradient is cut.  Only
train events contribute to the loss; validation is scored with the state carried
causally through the train events; the test partition is not reachable from here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Sequence

import numpy as np
import torch

from .contracts import FROZEN, LeakageGuard
from .event_marks import SPLIT_TRAIN, SPLIT_VALIDATION
from .model import CohortBatch, EpiPRSSM, PatientTensors, build_cohort_batch
from .rollout import carry_state, cohort_scan, scan_loss, score_window


@dataclass
class TrainConfig:
    max_epochs: int = 12
    min_epochs: int = 3
    patience: int = 3
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    tbptt_length: int = 64
    score_chunk: int = 256
    #: weight on the masked recruitment-order likelihood in the training objective
    order_weight: float = 0.0
    correction_energy_penalty: float = 1e-3
    flexible_penalty_weight: float = 1e-1
    #: Capping the training window is an engineering budget, not a scientific
    #: choice: evaluation always runs the full validation partition and the state
    #: is carried from the start of the capped window.  A sensitivity arm reruns
    #: the densest patients with a much larger cap.
    max_train_events_per_patient: int | None = 30000
    seed: int = 11
    device: str = "cpu"


@dataclass
class TrainReport:
    status: str
    epochs_run: int
    best_epoch: int
    best_validation: float
    history: list[dict[str, float]] = field(default_factory=list)
    failure_reason: str | None = None
    wall_seconds: float = 0.0
    peak_correction_energy: float = 0.0
    resource_floor_fraction: float = 0.0
    stability_margin: float = float("nan")
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status, "epochs_run": self.epochs_run,
            "best_epoch": self.best_epoch, "best_validation": self.best_validation,
            "history": self.history, "failure_reason": self.failure_reason,
            "wall_seconds": self.wall_seconds,
            "peak_correction_energy": self.peak_correction_energy,
            "resource_floor_fraction": self.resource_floor_fraction,
            "stability_margin": self.stability_margin,
            "diagnostics": self.diagnostics,
        }


def make_split_batches(patients: Sequence[PatientTensors], config: TrainConfig
                       ) -> tuple[CohortBatch, CohortBatch]:
    train_starts, train_lengths, val_starts, val_lengths = [], [], [], []
    for patient in patients:
        t_start, t_stop = patient.split_bounds(SPLIT_TRAIN)
        if config.max_train_events_per_patient is not None:
            t_start = max(t_start, t_stop - config.max_train_events_per_patient)
        v_start, v_stop = patient.split_bounds(SPLIT_VALIDATION)
        train_starts.append(t_start); train_lengths.append(max(t_stop - t_start, 0))
        val_starts.append(v_start); val_lengths.append(max(v_stop - v_start, 0))
    return (build_cohort_batch(patients, train_starts, train_lengths, device=config.device),
            build_cohort_batch(patients, val_starts, val_lengths, device=config.device))


def train_model(model: EpiPRSSM, patients: Sequence[PatientTensors], config: TrainConfig,
                *, guard: LeakageGuard | None = None,
                progress: Callable[[str], None] | None = None) -> TrainReport:
    guard = guard or LeakageGuard(stage="generator_training")
    guard.check_split(["train", "validation"])
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    train_batch, val_batch = make_split_batches(patients, config)

    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    history: list[dict[str, float]] = []
    best, best_epoch, bad_epochs = float("inf"), -1, 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    started = time.time()
    peak_energy, floor_fraction = 0.0, 0.0
    total_steps = train_batch.max_length

    for epoch in range(config.max_epochs):
        model.train()
        z = model.initial_state(train_batch)
        epoch_loss, epoch_chunks = 0.0, 0
        energies, floors = [], []
        position = 0
        while position < total_steps:
            end = min(position + config.tbptt_length, total_steps)
            optimiser.zero_grad(set_to_none=True)
            result = cohort_scan(model, train_batch, position, end, z, correction_on=True)
            loss = scan_loss(model, train_batch, result, position,
                             order_weight=config.order_weight)
            loss = loss + config.correction_energy_penalty * result.correction_energy
            loss = loss + config.flexible_penalty_weight * result.flexible_penalty
            if not torch.isfinite(loss):
                return TrainReport("NAN", epoch, best_epoch, best, history,
                                   failure_reason="non_finite_loss",
                                   wall_seconds=time.time() - started,
                                   diagnostics=model.describe())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimiser.step()
            epoch_loss += float(loss.item()); epoch_chunks += 1
            energies.append(float(result.correction_energy.item()))
            floors.append(result.resource_floor_fraction)
            z = result.final.detach()
            position = end
        validation = _validation_loss(model, train_batch, val_batch, config)
        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss / max(epoch_chunks, 1),
            "validation_loss": validation,
            "correction_energy": float(np.mean(energies)) if energies else 0.0,
            "resource_floor_fraction": float(np.mean(floors)) if floors else 0.0,
            "wall_seconds": time.time() - started,
        })
        peak_energy = max(peak_energy, history[-1]["correction_energy"])
        floor_fraction = history[-1]["resource_floor_fraction"]
        if progress:
            progress(f"epoch {epoch}: train={history[-1]['train_loss']:.4f} "
                     f"val={validation:.4f} ({history[-1]['wall_seconds']:.0f}s)")
        if validation < best - 1e-5:
            best, best_epoch, bad_epochs = validation, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if epoch + 1 >= config.min_epochs and bad_epochs >= config.patience:
                break
    model.load_state_dict(best_state)
    margin = float("nan")
    if hasattr(model.generator, "stability_margin"):
        margin = float(model.generator.stability_margin())
    return TrainReport("COMPLETE", len(history), best_epoch, best, history,
                       wall_seconds=time.time() - started,
                       peak_correction_energy=peak_energy,
                       resource_floor_fraction=floor_fraction,
                       stability_margin=margin, diagnostics=model.describe())


@torch.no_grad()
def _validation_loss(model: EpiPRSSM, train_batch: CohortBatch, val_batch: CohortBatch,
                     config: TrainConfig) -> float:
    model.eval()
    z = carry_state(model, train_batch, model.initial_state(train_batch), chunk=config.score_chunk)
    scores, _ = score_window(model, val_batch, z, chunk=config.score_chunk)
    total, count = 0.0, 0
    for values in scores.values():
        combined = values["event_nll"] + values["participation_nll"]
        total += float(combined.sum()); count += len(combined)
    model.train()
    return total / max(count, 1)
