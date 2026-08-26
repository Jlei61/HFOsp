"""Development-only T0/T1 trainer on merged regular observations and all IEDs."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import contract
from .bridge import BridgeHead
from .long_sequence import FullEventSequence
from .state import ExposureState, T1T2Core


REGULAR_BASELINE_REVISION = "regular_history_baseline_lbfgs_v1"
REGULAR_T1_REVISION = "regular_observation_frozen_history_identity_observer_state8_batched_v10"


@dataclass
class PreparedRegularT1:
    subject: str
    history: torch.Tensor
    observation: torch.Tensor
    observation_time: np.ndarray
    observation_split: np.ndarray
    event_time: np.ndarray
    next_time: np.ndarray
    session: np.ndarray
    split: np.ndarray
    log_iei: torch.Tensor
    participation: torch.Tensor
    rank: torch.Tensor
    stop: torch.Tensor
    observation_manifest: dict | None = None


def prepare_regular_t1(subject: str,
                       observation_variant: str = "spectral") -> PreparedRegularT1:
    from sklearn.preprocessing import StandardScaler

    sequence = FullEventSequence.load(
        contract.RESULT_ROOT / "long_sequence/features" / f"{subject}.npz"
    )
    feature_root = (
        contract.RESULT_ROOT / "regular_observation/features"
        if observation_variant == "spectral"
        else contract.RESULT_ROOT / f"regular_observation/features_{observation_variant}"
    )
    feature_path = feature_root / f"{subject}.npz"
    manifest_path = feature_path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"{subject}: regular-observation manifest missing")
    observation_manifest = json.loads(manifest_path.read_text())
    expected_feature = observation_manifest.get("feature_kind", "spectral")
    expected_revisions = {
        "masked_30s_background_on_60s_clock_v1",
        f"masked_30s_background_on_60s_clock_v1__{observation_variant}",
    }
    if (
        observation_manifest.get("contract") != contract.REVISION
        or observation_manifest.get("regular_observation_revision") not in expected_revisions
        or expected_feature != observation_variant
        or bool(observation_manifest.get("sealed_opened", True))
    ):
        raise ValueError(
            f"{subject}: stale, mixed, or unsealed regular observation package "
            f"for variant {observation_variant!r}"
        )
    with np.load(
        feature_path,
        allow_pickle=False,
    ) as z:
        observation = z["observation"].astype(np.float32)
        observation_time = z["anchor_time"].astype(np.float64)
        observation_split = z["split"].astype(np.int8)
    train = sequence.split == 0
    scaler = StandardScaler().fit(sequence.history[train])
    history = scaler.transform(sequence.history).astype(np.float32)
    return PreparedRegularT1(
        subject=subject,
        history=torch.as_tensor(history),
        observation=torch.as_tensor(observation),
        observation_time=observation_time,
        observation_split=observation_split,
        event_time=sequence.current_time,
        next_time=sequence.next_time,
        session=sequence.session,
        split=sequence.split,
        log_iei=torch.as_tensor(sequence.log_next_iei),
        participation=torch.as_tensor(sequence.next_participation),
        rank=torch.as_tensor(sequence.next_rank),
        stop=torch.as_tensor(sequence.next_stop_fraction),
        observation_manifest=observation_manifest,
    )


class FrozenBaselineStateHead(nn.Module):
    """Frozen history model plus a zero-effect, state-only adapter.

    Keeping these two blocks separate prevents T1 from winning merely because
    its copy of the explicit event-history model found a different optimum.
    """

    def __init__(self, baseline: BridgeHead, state_dim: int):
        super().__init__()
        self.baseline = baseline
        for parameter in self.baseline.parameters():
            parameter.requires_grad_(False)
        n_contacts = baseline.participation.out_features
        self.state_time = nn.Linear(state_dim, 1, bias=False)
        self.state_participation = nn.Linear(state_dim, n_contacts, bias=False)
        self.state_rank = nn.Linear(state_dim, n_contacts, bias=False)
        self.state_stop = nn.Linear(state_dim, 1, bias=False)
        for module in (
            self.state_time, self.state_participation,
            self.state_rank, self.state_stop,
        ):
            # State is exactly zero under the identity/no-correction observer,
            # so these tiny weights have zero initial prediction effect while
            # providing a gradient path back into the all-zero observer.
            nn.init.normal_(module.weight, mean=0.0, std=1e-3)

    def losses(self, history: torch.Tensor, state: torch.Tensor,
               log_iei: torch.Tensor, participation: torch.Tensor,
               rank: torch.Tensor, stop_fraction: torch.Tensor) -> dict[str, torch.Tensor]:
        time_mu = (
            self.baseline.time_mean(history) + self.state_time(state)
        ).squeeze(-1)
        timing = (
            0.5 * ((log_iei - time_mu) / self.baseline.time_sigma) ** 2
            + self.baseline.time_sigma.log() + 0.5 * math.log(2 * math.pi)
        )
        logits = (
            self.baseline.participation(history)
            + self.state_participation(state)
        )
        part = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, participation, reduction="none"
        ).mean(-1)
        rank_mu = self.baseline.rank(history) + self.state_rank(state)
        rank_terms = (
            0.5 * ((rank - rank_mu) / self.baseline.rank_sigma) ** 2
            + self.baseline.rank_sigma.log() + 0.5 * math.log(2 * math.pi)
        )
        rank_nll = (
            (rank_terms * participation).sum(-1)
            / participation.sum(-1).clamp(min=1)
        )
        stop_mu = (
            self.baseline.stop(history) + self.state_stop(state)
        ).squeeze(-1)
        stop_nll = (
            0.5 * ((stop_fraction - stop_mu) / self.baseline.stop_sigma) ** 2
            + self.baseline.stop_sigma.log() + 0.5 * math.log(2 * math.pi)
        )
        mark = part + rank_nll + stop_nll
        return {
            "timing_nll": timing, "participation_nll": part,
            "rank_nll": rank_nll, "stop_nll": stop_nll,
            "mark_nll": mark, "joint_nll": timing + mark,
        }


class RegularT1Model(nn.Module):
    def __init__(self, history_dim: int, n_contacts: int,
                 scales: dict[str, float], baseline: BridgeHead,
                 state_dim: int = 4):
        super().__init__()
        self.observation_project = nn.Sequential(
            nn.Linear(contract.STATE_OBSERVATION_DIM, 8), nn.Tanh()
        )
        self.core = T1T2Core(8, state_dim, t2=False)
        # T1 starts from the contract's exact identity/no-correction observer.
        # The state is therefore zero and initial predictions exactly match the
        # frozen baseline; tiny adapter weights provide the first gradient path.
        self.head = FrozenBaselineStateHead(baseline, state_dim)

    def event_losses(self, history: torch.Tensor, state: torch.Tensor,
                     log_iei: torch.Tensor, participation: torch.Tensor,
                     rank: torch.Tensor, stop: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.head.losses(
            history.unsqueeze(0), state.unsqueeze(0),
            log_iei.reshape(1), participation.unsqueeze(0),
            rank.unsqueeze(0), stop.reshape(1),
        )


def _target_scales(sequence: PreparedRegularT1) -> dict[str, float]:
    idx = torch.as_tensor(np.flatnonzero(sequence.split == 0), dtype=torch.long)
    ranks = sequence.rank[idx][sequence.participation[idx] > 0.5]
    return {
        "time_sigma": float(sequence.log_iei[idx].std(unbiased=False).clamp(min=0.25)),
        "rank_sigma": float(ranks.std(unbiased=False).clamp(min=0.20)),
        "stop_sigma": float(sequence.stop[idx].std(unbiased=False).clamp(min=0.10)),
    }


def _new_history_baseline(sequence: PreparedRegularT1,
                          scales: dict[str, float]) -> BridgeHead:
    model = BridgeHead(
        sequence.history.shape[1], sequence.participation.shape[1],
        time_sigma=scales["time_sigma"], rank_sigma=scales["rank_sigma"],
        stop_sigma=scales["stop_sigma"],
    )
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
    return model


def _baseline_objective(model: BridgeHead, sequence: PreparedRegularT1,
                        index: torch.Tensor) -> torch.Tensor:
    losses = model.losses(
        sequence.history[index], sequence.log_iei[index],
        sequence.participation[index], sequence.rank[index],
        sequence.stop[index],
    )
    return losses["joint_nll"].mean()


def _optimise_history_baseline(model: BridgeHead,
                               sequence: PreparedRegularT1,
                               index: torch.Tensor,
                               penalty_weight: float,
                               max_iter: int) -> int:
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=0.5, max_iter=int(max_iter), history_size=20,
        tolerance_grad=1e-7, tolerance_change=1e-9,
        line_search_fn="strong_wolfe",
    )
    calls = 0

    def closure():
        nonlocal calls
        calls += 1
        optimizer.zero_grad(set_to_none=True)
        penalty = sum(
            parameter.square().sum()
            for parameter in model.parameters() if parameter.ndim >= 2
        )
        objective = (
            _baseline_objective(model, sequence, index)
            + float(penalty_weight) * penalty
        )
        objective.backward()
        return objective

    optimizer.step(closure)
    model.eval()
    return calls


def _baseline_metrics(model: BridgeHead, sequence: PreparedRegularT1,
                      index: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        losses = model.losses(
            sequence.history[index], sequence.log_iei[index],
            sequence.participation[index], sequence.rank[index],
            sequence.stop[index],
        )
    return {
        key: float(losses[key].mean())
        for key in (
            "joint_nll", "timing_nll", "mark_nll",
            "participation_nll", "rank_nll", "stop_nll",
        )
    }


def baseline_paths(subject: str) -> tuple[Path, Path]:
    root = contract.RESULT_ROOT / "regular_t1/baselines"
    return root / f"{subject}.pt", root / f"{subject}.manifest.json"


def fit_regular_history_baseline(subject: str, *, max_iter: int = 240,
                                 overwrite: bool = False) -> dict:
    """Fit and persist one deterministic TRAIN-only history baseline."""
    state_path, manifest_path = baseline_paths(subject)
    if state_path.exists() and manifest_path.exists() and not overwrite:
        old = json.loads(manifest_path.read_text())
        if old.get("baseline_revision") == REGULAR_BASELINE_REVISION:
            return old
    torch.manual_seed(0)
    np.random.seed(0)
    sequence = prepare_regular_t1(subject)
    scales = _target_scales(sequence)
    train_idx = torch.as_tensor(
        np.flatnonzero(sequence.split == 0), dtype=torch.long
    )
    if len(train_idx) < 20:
        raise ValueError(f"{subject}: too few TRAIN events for baseline")
    cut = max(1, min(len(train_idx) - 1, int(math.floor(0.80 * len(train_idx)))))
    inner_train = train_idx[:cut]
    inner_validation = train_idx[cut:]
    grid = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
    selection = []
    for weight in grid:
        candidate = _new_history_baseline(sequence, scales)
        calls = _optimise_history_baseline(
            candidate, sequence, inner_train, weight,
            max(80, int(math.ceil(max_iter / 2))),
        )
        selection.append({
            "weight_decay": float(weight),
            "inner_validation_joint_nll": _baseline_metrics(
                candidate, sequence, inner_validation
            )["joint_nll"],
            "closure_calls": int(calls),
        })
    selected = min(
        selection, key=lambda item: item["inner_validation_joint_nll"]
    )["weight_decay"]
    model = _new_history_baseline(sequence, scales)
    calls = _optimise_history_baseline(
        model, sequence, train_idx, selected, max_iter
    )
    payload = {
        "contract": contract.REVISION,
        "baseline_revision": REGULAR_BASELINE_REVISION,
        "subject": subject,
        "history_dim": int(sequence.history.shape[1]),
        "n_contacts": int(sequence.participation.shape[1]),
        "target_scales": scales,
        "state_dict": model.state_dict(),
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, state_path)
    manifest = {
        "contract": contract.REVISION,
        "baseline_revision": REGULAR_BASELINE_REVISION,
        "subject": subject,
        "n_train": int(len(train_idx)),
        "n_inner_train": int(len(inner_train)),
        "n_inner_validation": int(len(inner_validation)),
        "selected_weight_decay": float(selected),
        "regularization_grid": selection,
        "closure_calls": int(calls),
        "train": _baseline_metrics(model, sequence, train_idx),
        "target_scales": scales,
        "state_path": str(state_path.resolve()),
        "sealed_opened": False,
        "selection_semantics": (
            "chronological final 20% of TRAIN only; development validation untouched"
        ),
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(temporary_manifest, manifest_path)
    return manifest


def load_regular_history_baseline(subject: str,
                                  sequence: PreparedRegularT1,
                                  scales: dict[str, float]) -> tuple[BridgeHead, dict]:
    state_path, manifest_path = baseline_paths(subject)
    if not state_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"{subject}: frozen history baseline missing; run fit_regular_baselines.py"
        )
    manifest = json.loads(manifest_path.read_text())
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    if (
        manifest.get("baseline_revision") != REGULAR_BASELINE_REVISION
        or payload.get("baseline_revision") != REGULAR_BASELINE_REVISION
        or payload.get("contract") != contract.REVISION
    ):
        raise ValueError(f"{subject}: stale or mixed frozen baseline package")
    if (
        int(payload["history_dim"]) != sequence.history.shape[1]
        or int(payload["n_contacts"]) != sequence.participation.shape[1]
    ):
        raise ValueError(f"{subject}: frozen baseline shape mismatch")
    for key, value in scales.items():
        if not np.isclose(float(payload["target_scales"][key]), float(value)):
            raise ValueError(f"{subject}: frozen baseline target scale mismatch for {key}")
    model = _new_history_baseline(sequence, scales)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, manifest


def _observation_session_masks(
    sequence: PreparedRegularT1, split_code: int,
    initial: dict[int, tuple[torch.Tensor, float]] | None = None,
) -> dict[int, np.ndarray]:
    output = {}
    for session_id in np.unique(sequence.session[sequence.split == split_code]):
        event = (sequence.split == split_code) & (sequence.session == session_id)
        lo = (
            float(initial[int(session_id)][1])
            if initial and int(session_id) in initial
            else float(sequence.event_time[event].min())
        )
        hi = float(sequence.next_time[event].max())
        output[int(session_id)] = np.flatnonzero(
            (sequence.observation_split == split_code)
            & (sequence.observation_time >= lo)
            & (sequence.observation_time <= hi)
        )
    return output


def _run_split(model: RegularT1Model, sequence: PreparedRegularT1,
               split_code: int, *, correction_enabled: bool,
               state_enabled: bool = True,
               optimizer=None, chunk_events: int = 64,
               initial: dict[int, tuple[torch.Tensor, float]] | None = None,
               max_events: int | None = None) -> tuple[dict[str, float], dict[int, tuple[torch.Tensor, float]]]:
    training = optimizer is not None
    model.train(training)
    keys = ("joint_nll", "timing_nll", "mark_nll", "participation_nll",
            "rank_nll", "stop_nll")
    sums = {key: 0.0 for key in keys}
    count = 0
    final = {}
    obs_by_session = _observation_session_masks(sequence, split_code, initial)
    for session_id in np.unique(sequence.session[sequence.split == split_code]):
        event_idx = np.flatnonzero(
            (sequence.split == split_code) & (sequence.session == session_id)
        )
        if max_events is not None:
            remaining = max(max_events - count, 0)
            event_idx = event_idx[:remaining]
        if not len(event_idx):
            continue
        if initial and int(session_id) in initial:
            z, cursor = initial[int(session_id)]
            z = z.clone()
        else:
            z = torch.zeros(model.core.generator.dim)
            cursor = float(sequence.event_time[event_idx[0]])
        exposure = ExposureState(torch.zeros(()), 60.0)
        obs_idx = obs_by_session.get(int(session_id), np.empty(0, dtype=int))
        obs_position = 0
        pending: list[torch.Tensor] = []
        pending_count = 0
        event_position = 0
        event_times = sequence.event_time[event_idx]
        while event_position < len(event_idx):
            event_time = float(event_times[event_position])
            # Preserve the original tie rule: an observation at exactly the
            # event time is assimilated before that event is scored.
            if (obs_position < len(obs_idx)
                    and sequence.observation_time[obs_idx[obs_position]] <= event_time):
                oi = int(obs_idx[obs_position])
                obs_time = float(sequence.observation_time[oi])
                if obs_time >= cursor:
                    if state_enabled:
                        projected = model.observation_project(sequence.observation[oi])
                        z, exposure = model.core.step(
                            z, max((obs_time - cursor) / 60.0, 0.0), projected,
                            exposure, correction_enabled=correction_enabled,
                        )
                    cursor = obs_time
                obs_position += 1
                continue

            capacity = (
                max(chunk_events - pending_count, 1)
                if training else len(event_idx) - event_position
            )
            end = min(event_position + capacity, len(event_idx))
            if obs_position < len(obs_idx):
                next_obs_time = float(sequence.observation_time[obs_idx[obs_position]])
                # Events equal to next_obs_time stay on the far side so the
                # measurement correction is applied first.
                obs_boundary = int(np.searchsorted(
                    event_times, next_obs_time, side="left"
                ))
                end = min(end, max(obs_boundary, event_position + 1))
            rows = event_idx[event_position:end]
            row_tensor = torch.as_tensor(rows, dtype=torch.long)
            if state_enabled:
                delta = torch.as_tensor(
                    (sequence.event_time[rows] - cursor) / 60.0,
                    dtype=z.dtype,
                ).clamp(min=0.0)
                states = model.core.generator.propagate_many_from_same_state(z, delta)
                z = states[-1]
            else:
                states = z.new_zeros((len(rows), z.numel()))
                z = states[-1]
            cursor = float(sequence.event_time[rows[-1]])
            losses = model.head.losses(
                sequence.history[row_tensor], states,
                sequence.log_iei[row_tensor], sequence.participation[row_tensor],
                sequence.rank[row_tensor], sequence.stop[row_tensor],
            )
            for key in keys:
                sums[key] += float(losses[key].detach().sum())
            count += len(rows)
            if training:
                pending.append(losses["joint_nll"])
                pending_count += len(rows)
            event_position = end
            if training and (
                pending_count >= chunk_events or event_position == len(event_idx)
            ):
                loss = torch.cat(pending).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                pending = []
                pending_count = 0
                z = z.detach()
        final[int(session_id)] = (z.detach(), cursor)
        if max_events is not None and count >= max_events:
            break
    denom = max(count, 1)
    return ({key: value / denom for key, value in sums.items()}
            | {"n_events": int(count)}), final


def _filtered_validation_states(
    model: RegularT1Model, sequence: PreparedRegularT1,
    initial: dict[int, tuple[torch.Tensor, float]], *, correction_enabled: bool,
    state_enabled: bool,
) -> dict[int, torch.Tensor]:
    """State at each validation event after all observations up to that time."""
    states = {}
    obs_by_session = _observation_session_masks(sequence, 1, initial)
    for session_id in np.unique(sequence.session[sequence.split == 1]):
        rows = np.flatnonzero((sequence.split == 1) & (sequence.session == session_id))
        if int(session_id) in initial:
            z, cursor = initial[int(session_id)]
            z = z.clone()
        else:
            z = torch.zeros(model.core.generator.dim)
            cursor = float(sequence.event_time[rows[0]])
        exposure = ExposureState(torch.zeros(()), 60.0)
        obs_idx = obs_by_session.get(int(session_id), np.empty(0, dtype=int))
        obs_position = 0
        event_times = sequence.event_time[rows]
        event_position = 0
        while event_position < len(rows):
            event_time = float(event_times[event_position])
            if (obs_position < len(obs_idx)
                    and sequence.observation_time[obs_idx[obs_position]] <= event_time):
                oi = int(obs_idx[obs_position])
                obs_time = float(sequence.observation_time[oi])
                if obs_time >= cursor:
                    if state_enabled:
                        projected = model.observation_project(sequence.observation[oi])
                        z, exposure = model.core.step(
                            z, max((obs_time - cursor) / 60.0, 0.0), projected,
                            exposure, correction_enabled=correction_enabled,
                        )
                    cursor = obs_time
                obs_position += 1
                continue
            end = len(rows)
            if obs_position < len(obs_idx):
                next_obs_time = float(sequence.observation_time[obs_idx[obs_position]])
                end = int(np.searchsorted(event_times, next_obs_time, side="left"))
                end = max(end, event_position + 1)
            segment_rows = rows[event_position:end]
            if state_enabled:
                delta = torch.as_tensor(
                    (sequence.event_time[segment_rows] - cursor) / 60.0,
                    dtype=z.dtype,
                ).clamp(min=0.0)
                segment_states = model.core.generator.propagate_many_from_same_state(
                    z, delta
                )
                z = segment_states[-1]
            else:
                segment_states = z.new_zeros((len(segment_rows), z.numel()))
                z = segment_states[-1]
            cursor = float(sequence.event_time[segment_rows[-1]])
            for row, value in zip(segment_rows.tolist(), segment_states):
                states[int(row)] = value.detach().clone()
            event_position = end
    return states


def _future_transition_rows_from_anchor(
    sequence: PreparedRegularT1, anchor: int, horizon: int,
) -> np.ndarray | None:
    """Return rows predicting exactly the next ``horizon`` events.

    Row ``i`` contains event-i history and event-(i+1) targets. Therefore the
    anchor row predicts the first future event; rows ``anchor:anchor+h``
    predict target events ``anchor+1:anchor+h+1``. This explicit transition
    convention prevents confusing predictor-row identity with target-event
    identity.
    """
    start = int(anchor)
    end = start + int(horizon)
    if end > len(sequence.split):
        return None
    rows = np.arange(start, end, dtype=int)
    if (
        not np.all(sequence.split[rows] == 1)
        or not np.all(sequence.session[rows] == sequence.session[anchor])
    ):
        return None
    return rows


def _post_anchor_challenge(
    model: RegularT1Model, sequence: PreparedRegularT1,
    initial: dict[int, tuple[torch.Tensor, float]], *, correction_enabled: bool,
    state_enabled: bool,
    horizons: tuple[int, ...] = (5, 10, 20), anchor_stride: int = 5,
) -> dict[str, dict]:
    states = _filtered_validation_states(
        model, sequence, initial, correction_enabled=correction_enabled,
        state_enabled=state_enabled,
    )
    output = {}
    validation_rows = np.flatnonzero(sequence.split == 1)
    for horizon in horizons:
        totals = {key: 0.0 for key in ("joint_nll", "timing_nll", "mark_nll")}
        n_values = n_anchors = 0
        for anchor in validation_rows[::anchor_stride].tolist():
            rows = _future_transition_rows_from_anchor(sequence, anchor, horizon)
            if rows is None:
                continue
            z = states[int(anchor)].clone()
            row_tensor = torch.as_tensor(rows, dtype=torch.long)
            if state_enabled:
                delta = torch.as_tensor(
                    (sequence.event_time[rows] - sequence.event_time[anchor]) / 60.0,
                    dtype=z.dtype,
                ).clamp(min=0.0)
                rollout_states = model.core.generator.propagate_many_from_same_state(
                    z, delta
                )
            else:
                rollout_states = z.new_zeros((len(rows), z.numel()))
            losses = model.head.losses(
                sequence.history[row_tensor], rollout_states,
                sequence.log_iei[row_tensor], sequence.participation[row_tensor],
                sequence.rank[row_tensor], sequence.stop[row_tensor],
            )
            for key in totals:
                totals[key] += float(losses[key].detach().sum())
            n_values += len(rows)
            n_anchors += 1
        output[str(horizon)] = {
            **{key: value / max(n_values, 1) for key, value in totals.items()},
            "n_anchors": int(n_anchors), "n_scored_event_rows": int(n_values),
            "anchor_stride": int(anchor_stride),
        }
    return output


def _matched_wrong_time_swap(
    model: RegularT1Model, sequence: PreparedRegularT1,
    initial: dict[int, tuple[torch.Tensor, float]], *, correction_enabled: bool,
    state_enabled: bool,
) -> dict[str, float]:
    """Swap only z within the same validation session; keep decoder/history fixed."""
    states = _filtered_validation_states(
        model, sequence, initial, correction_enabled=correction_enabled,
        state_enabled=state_enabled,
    )
    totals = {key: {"correct": 0.0, "wrong": 0.0}
              for key in ("joint_nll", "timing_nll", "mark_nll")}
    count = 0
    shifts = []
    for session_id in np.unique(sequence.session[sequence.split == 1]):
        rows = np.flatnonzero((sequence.split == 1) & (sequence.session == session_id))
        if len(rows) < 2:
            continue
        shift = min(37, max(1, len(rows) // 3))
        wrong_rows = np.roll(rows, shift)
        shifts.append({"session": int(session_id), "n_events": int(len(rows)),
                       "shift_events": int(shift)})
        row_tensor = torch.as_tensor(rows, dtype=torch.long)
        correct_states = torch.stack([states[int(row)] for row in rows.tolist()])
        wrong_states = torch.stack([states[int(row)] for row in wrong_rows.tolist()])
        targets = (
            sequence.history[row_tensor], sequence.log_iei[row_tensor],
            sequence.participation[row_tensor], sequence.rank[row_tensor],
            sequence.stop[row_tensor],
        )
        correct = model.head.losses(
            targets[0], correct_states, targets[1], targets[2], targets[3], targets[4]
        )
        wrong = model.head.losses(
            targets[0], wrong_states, targets[1], targets[2], targets[3], targets[4]
        )
        for key in totals:
            totals[key]["correct"] += float(correct[key].detach().sum())
            totals[key]["wrong"] += float(wrong[key].detach().sum())
        count += len(rows)
    output = {"n_events": int(count), "session_shifts": shifts, "endpoints": {}}
    for key, values in totals.items():
        correct = values["correct"] / max(count, 1)
        wrong = values["wrong"] / max(count, 1)
        output["endpoints"][key] = {
            "correct_state_nll": correct,
            "wrong_state_nll": wrong,
            "wrong_minus_correct": wrong - correct,
        }
    return output


def fit_regular_t1(subject: str, arm: str, *, seed: int = 0,
                   epochs: int = 30, max_train_events: int | None = None,
                   observation_variant: str = "spectral",
                   state_dim: int = 8) -> dict:
    if arm not in ("t0_no_observation_state", "t1_regular_observation"):
        raise ValueError(f"unsupported T1 arm {arm!r}")
    if int(state_dim) < 1:
        raise ValueError("state_dim must be positive")
    torch.manual_seed(seed)
    np.random.seed(seed)
    sequence = prepare_regular_t1(subject, observation_variant)
    scales = _target_scales(sequence)
    baseline, baseline_manifest = load_regular_history_baseline(
        subject, sequence, scales
    )
    model = RegularT1Model(
        sequence.history.shape[1], sequence.participation.shape[1], scales,
        baseline, state_dim=state_dim,
    )
    enabled = arm == "t1_regular_observation"
    trainable = [parameter for parameter in model.parameters()
                 if parameter.requires_grad]
    if enabled:
        optimizer = torch.optim.AdamW(
            trainable, lr=1e-3, weight_decay=2e-2
        )
        for _ in range(epochs):
            _run_split(
                model, sequence, 0, correction_enabled=True,
                state_enabled=True, optimizer=optimizer,
                max_events=max_train_events,
            )
    model.eval()
    with torch.no_grad():
        train, train_final = _run_split(
            model, sequence, 0, correction_enabled=enabled,
            state_enabled=enabled, max_events=max_train_events,
        )
        filtered, _ = _run_split(
            model, sequence, 1, correction_enabled=enabled, initial=train_final,
            state_enabled=enabled,
        )
        correction_off, _ = _run_split(
            model, sequence, 1, correction_enabled=False, initial=train_final,
            state_enabled=enabled,
        )
        post_anchor = _post_anchor_challenge(
            model, sequence, train_final, correction_enabled=enabled,
            state_enabled=enabled,
        )
        state_swap = _matched_wrong_time_swap(
            model, sequence, train_final, correction_enabled=enabled,
            state_enabled=enabled,
        )
        if not enabled:
            maximum = max(
                abs(value["wrong_minus_correct"])
                for value in state_swap["endpoints"].values()
            )
            if maximum != 0.0:
                raise ValueError(
                    f"T0 clamped-state swap must be exactly zero, got {maximum}"
                )
    return {
        "contract": contract.REVISION,
        "regular_t1_revision": REGULAR_T1_REVISION,
        "subject": subject, "arm": arm, "seed": int(seed),
        "observation_variant": observation_variant,
        "state_dim": int(state_dim),
        "epochs": int(epochs), "max_train_events": max_train_events,
        "n_parameters_total": int(sum(p.numel() for p in model.parameters())),
        "n_parameters_trainable": int(sum(p.numel() for p in trainable)) if enabled else 0,
        "baseline_revision": REGULAR_BASELINE_REVISION,
        "baseline_manifest": baseline_manifest,
        "observation_manifest": sequence.observation_manifest,
        "target_scales": scales,
        "train": train,
        "validation_filtered": filtered,
        "validation_correction_off_from_split_start": correction_off,
        "post_anchor_correction_off": post_anchor,
        "matched_wrong_time_state_swap": state_swap,
        "sealed_opened": False,
        "claim_boundary": (
            "shared frozen history baseline plus regular-observation full-event "
            "development prototype with 5/10/20 event post-anchor correction-off "
            "and same-session wrong-time state swap; fixed features are not the "
            "final raw Transformer observer and cohort replication remains required"
        ),
    }
