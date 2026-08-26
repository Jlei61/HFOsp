"""Development trainer for an event-anchor approximation to T1/T2.

This is a plumbing and identifiability prototype, not the final 30 s-anchor
model.  It already enforces the scientifically important partition: explicit
history has no learned recurrence, one persistent z is carried forward, and T2
differs from T1 only through a cross-fitted innovation exposure edge.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .bridge import BridgeArrays, BridgeHead, _arm_matrix
from .exposure import pre_event_innovation_predictors
from .state import ExposureState, T1T2Core


@dataclass
class PreparedSequence:
    history: torch.Tensor
    observation: torch.Tensor
    time: np.ndarray
    log_iei: torch.Tensor
    participation: torch.Tensor
    rank: torch.Tensor
    stop: torch.Tensor
    split: np.ndarray
    innovation: np.ndarray


def _cross_fitted_innovation(arrays: BridgeArrays) -> np.ndarray:
    from sklearn.linear_model import Ridge

    history = arrays.history.astype(np.float64)
    # Do not regress current load on a history vector containing current load.
    # Expected load may only use information available before the event.
    predictors = pre_event_innovation_predictors(
        arrays.history, arrays.history[:, 10:10 + arrays.participation.shape[1]] > 0.5
    ).astype(np.float64)
    load = arrays.history[:, 3].astype(np.float64)
    train = np.flatnonzero(arrays.split == 0)
    valid = np.flatnonzero(arrays.split == 1)
    expected = np.zeros(len(load), dtype=np.float64)
    # Contiguous held-out folds keep every train event out of its own expected
    # load model. This is exploratory cross-fitting, not a causal forecasting
    # score, so the other blocks may sit on either side in time.
    folds = [x for x in np.array_split(train, min(5, len(train))) if len(x)]
    for fold in folds:
        fit = np.setdiff1d(train, fold, assume_unique=True)
        model = Ridge(alpha=10.0).fit(predictors[fit], load[fit])
        expected[fold] = model.predict(predictors[fold])
    final = Ridge(alpha=10.0).fit(predictors[train], load[train])
    expected[valid] = final.predict(predictors[valid])
    return (load - expected).astype(np.float32)


def prepare_sequence(arrays: BridgeArrays, observation_arm: str = "b1_spectral") -> PreparedSequence:
    matrix, _ = _arm_matrix(arrays, observation_arm)
    hdim = arrays.history.shape[1]
    return PreparedSequence(
        history=torch.as_tensor(matrix[:, :hdim], dtype=torch.float32),
        observation=torch.as_tensor(matrix[:, hdim:], dtype=torch.float32),
        time=arrays.current_time.astype(np.float64),
        log_iei=torch.as_tensor(arrays.log_next_iei, dtype=torch.float32),
        participation=torch.as_tensor(arrays.participation, dtype=torch.float32),
        rank=torch.as_tensor(arrays.rank, dtype=torch.float32),
        stop=torch.as_tensor(arrays.stop_fraction, dtype=torch.float32),
        split=arrays.split.astype(np.int8),
        innovation=_cross_fitted_innovation(arrays),
    )


class EventAnchorModel(nn.Module):
    def __init__(self, history_dim: int, observation_dim: int, n_contacts: int,
                 *, t2: bool, scales: dict[str, float], state_dim: int = 4,
                 observation_embed_dim: int = 8):
        super().__init__()
        self.observation_project = nn.Sequential(
            nn.Linear(observation_dim, observation_embed_dim), nn.Tanh()
        )
        self.core = T1T2Core(observation_embed_dim, state_dim, t2=t2)
        self.head = BridgeHead(
            history_dim + state_dim, n_contacts,
            time_sigma=scales["time_sigma"], rank_sigma=scales["rank_sigma"],
            stop_sigma=scales["stop_sigma"],
        )

    def event_loss(self, history: torch.Tensor, state: torch.Tensor,
                   log_iei: torch.Tensor, participation: torch.Tensor,
                   rank: torch.Tensor, stop: torch.Tensor) -> dict[str, torch.Tensor]:
        x = torch.cat([history, state], dim=-1).unsqueeze(0)
        return self.head.losses(
            x, log_iei.reshape(1), participation.unsqueeze(0),
            rank.unsqueeze(0), stop.reshape(1),
        )


def _target_scales(sequence: PreparedSequence) -> dict[str, float]:
    idx = torch.as_tensor(np.flatnonzero(sequence.split == 0), dtype=torch.long)
    ranks = sequence.rank[idx][sequence.participation[idx] > 0.5]
    return {
        "time_sigma": float(sequence.log_iei[idx].std(unbiased=False).clamp(min=0.25)),
        "rank_sigma": float(ranks.std(unbiased=False).clamp(min=0.20)),
        "stop_sigma": float(sequence.stop[idx].std(unbiased=False).clamp(min=0.10)),
    }


def _innovation_for_arm(sequence: PreparedSequence, arm: str) -> np.ndarray:
    if arm == "t1":
        return np.zeros_like(sequence.innovation)
    if arm == "t2_real":
        return sequence.innovation.copy()
    if arm == "t2_placebo":
        out = np.zeros_like(sequence.innovation)
        for code in (0, 1):
            idx = np.flatnonzero(sequence.split == code)
            if len(idx):
                shift = min(37, max(1, len(idx) // 3))
                if len(idx) > shift:
                    out[idx[shift:]] = sequence.innovation[idx[:-shift]]
        return out
    raise ValueError(f"unknown event-anchor arm {arm!r}")


def _run_sequence(model: EventAnchorModel, sequence: PreparedSequence,
                  indices: np.ndarray, innovation: np.ndarray, tau_minutes: float,
                  *, correction_enabled: bool, optimizer=None,
                  initial_state: torch.Tensor | None = None,
                  initial_exposure: torch.Tensor | None = None,
                  initial_time: float | None = None,
                  chunk_events: int = 64) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    training = optimizer is not None
    model.train(training)
    state_dim = model.core.generator.dim
    z = torch.zeros(state_dim) if initial_state is None else initial_state
    exposure = ExposureState(
        torch.zeros(()) if initial_exposure is None else initial_exposure,
        tau_minutes,
    )
    sums = {k: 0.0 for k in ("joint_nll", "timing_nll", "mark_nll",
                             "participation_nll", "rank_nll", "stop_nll")}
    pending = []
    previous_time = initial_time
    for position, i in enumerate(indices.tolist()):
        dt = 0.0 if previous_time is None else max((sequence.time[i] - previous_time) / 60.0, 0.0)
        previous_time = float(sequence.time[i])
        observation = model.observation_project(sequence.observation[i])
        z, exposure = model.core.step(
            z, dt, observation, exposure, correction_enabled=correction_enabled
        )
        losses = model.event_loss(
            sequence.history[i], z, sequence.log_iei[i], sequence.participation[i],
            sequence.rank[i], sequence.stop[i],
        )
        for key in sums:
            sums[key] += float(losses[key].detach())
        if training:
            pending.append(losses["joint_nll"].mean())
        exposure = exposure.jump(float(innovation[i]))
        if training and (len(pending) >= chunk_events or position == len(indices) - 1):
            loss = torch.stack(pending).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            pending = []
            z = z.detach()
            exposure = ExposureState(exposure.value.detach(), exposure.tau_minutes)
    denom = max(len(indices), 1)
    return {k: v / denom for k, v in sums.items()}, z.detach(), exposure.value.detach()


def fit_event_anchor(arrays: BridgeArrays, arm: str, *, seed: int = 0,
                     tau_minutes: float = 60.0, epochs: int = 80) -> dict:
    for code in (0, 1):
        idx = np.flatnonzero(arrays.split == code)
        if len(idx) > 1 and np.any(np.diff(arrays.current_event_index[idx]) != 1):
            raise ValueError(
                "event-anchor prototype requires every event in chronological "
                "order; Bridge feature rows are sampled/observation-complete "
                "pairs and would silently omit exposure jumps"
            )
    torch.manual_seed(seed)
    np.random.seed(seed)
    sequence = prepare_sequence(arrays, "b1_spectral")
    train = np.flatnonzero(sequence.split == 0)
    validation = np.flatnonzero(sequence.split == 1)
    scales = _target_scales(sequence)
    model = EventAnchorModel(
        sequence.history.shape[1], sequence.observation.shape[1],
        sequence.participation.shape[1], t2=(arm != "t1"), scales=scales,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-2)
    innovation = _innovation_for_arm(sequence, arm)
    best = None
    stale = 0
    for epoch in range(epochs):
        _run_sequence(model, sequence, train, innovation, tau_minutes,
                      correction_enabled=True, optimizer=optimizer)
        model.eval()
        with torch.no_grad():
            _, z_train, u_train = _run_sequence(
                model, sequence, train, innovation, tau_minutes,
                correction_enabled=True,
            )
            filtered, _, _ = _run_sequence(
                model, sequence, validation, innovation, tau_minutes,
                correction_enabled=True, initial_state=z_train,
                initial_exposure=u_train, initial_time=float(sequence.time[train[-1]]),
            )
        score = filtered["joint_nll"]
        if best is None or score < best[0] - 1e-5:
            best = (score, epoch, copy.deepcopy(model.state_dict()))
            stale = 0
        else:
            stale += 1
        if stale >= 12:
            break
    assert best is not None
    model.load_state_dict(best[2])
    model.eval()
    with torch.no_grad():
        train_metrics, z_train, u_train = _run_sequence(
            model, sequence, train, innovation, tau_minutes,
            correction_enabled=True,
        )
        filtered, _, _ = _run_sequence(
            model, sequence, validation, innovation, tau_minutes,
            correction_enabled=True, initial_state=z_train,
            initial_exposure=u_train, initial_time=float(sequence.time[train[-1]]),
        )
        correction_off, _, _ = _run_sequence(
            model, sequence, validation, innovation, tau_minutes,
            correction_enabled=False, initial_state=z_train,
            initial_exposure=u_train, initial_time=float(sequence.time[train[-1]]),
        )
    return {
        "subject": arrays.subject,
        "arm": arm,
        "seed": seed,
        "tau_minutes": tau_minutes,
        "best_epoch": int(best[1]),
        "epochs_executed": int(epoch + 1),
        "n_train": int(len(train)),
        "n_validation": int(len(validation)),
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "target_scales": scales,
        "train": train_metrics,
        "validation_filtered": filtered,
        "validation_correction_off_from_start": correction_off,
        "claim_boundary": (
            "event-anchor T1/T2 development prototype; not the final 30 s-anchor "
            "model and not an H1/H3 result"
        ),
        "sealed_opened": False,
    }
