"""Bridge-E1: explicit versus zero-initialised raw residual on exact IED loss."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import contract
from .baseline import ExactHistoryMarkDecoder, HistoryIntensity
from .coverage import CoverageTable
from .data import R1EventStream, load_event_stream
from .history import DeterministicHistory, HistoryScaler, session_start_map
from .mark_likelihood import tied_group_mark_log_prob
from .observer import ObservationTransformer, copy_common_observer_state
from .raw_observation import RawAnchorReader
from .raw_observation import RAW_OBSERVATION_REVISION


BRIDGE_E1_REVISION = "bridge_e1_exact_ied_explicit_vs_zero_raw_v2"


def _uniform_take(index: np.ndarray, limit: int) -> np.ndarray:
    index = np.asarray(index, dtype=np.int64)
    if len(index) <= int(limit):
        return index
    position = np.linspace(0, len(index) - 1, int(limit)).round().astype(np.int64)
    return index[np.unique(position)]


@dataclass(frozen=True)
class BridgeE1Design:
    subject: str
    anchor_time: np.ndarray
    anchor_split: np.ndarray
    anchor_session: np.ndarray
    anchor_history: np.ndarray
    explicit: np.ndarray
    explicit_mean: np.ndarray
    explicit_scale: np.ndarray
    contact_mask: np.ndarray
    coordinates: np.ndarray
    coordinate_valid: np.ndarray
    shaft_index: np.ndarray
    event_anchor: np.ndarray
    event_time: np.ndarray
    event_history: np.ndarray
    event_group_ids: np.ndarray
    event_group_count: np.ndarray
    quadrature_anchor: np.ndarray
    quadrature_time: np.ndarray
    quadrature_history: np.ndarray
    quadrature_weight_seconds: np.ndarray

    def validate(self) -> None:
        anchors = len(self.anchor_time)
        if self.explicit.shape[:2] != self.contact_mask.shape:
            raise ValueError("Bridge-E1 explicit/contact mask shapes disagree")
        if self.explicit.shape[0] != anchors:
            raise ValueError("Bridge-E1 anchor arrays disagree")
        if self.explicit_mean.shape != (self.explicit.shape[2],):
            raise ValueError("Bridge-E1 explicit mean has wrong shape")
        if self.explicit_scale.shape != (self.explicit.shape[2],):
            raise ValueError("Bridge-E1 explicit scale has wrong shape")
        if np.any(self.explicit_scale <= 0):
            raise ValueError("Bridge-E1 explicit scale must be positive")
        if self.anchor_history.shape[0] != anchors:
            raise ValueError("Bridge-E1 anchor history disagrees")
        if set(np.unique(self.anchor_split).tolist()) != {0, 1}:
            raise ValueError("Bridge-E1 requires non-empty TRAIN and validation anchors")
        if len(self.event_anchor) != len(self.event_history):
            raise ValueError("Bridge-E1 event arrays disagree")
        if len(self.event_anchor) != len(self.event_time):
            raise ValueError("Bridge-E1 event times disagree")
        if len(self.event_anchor) != len(self.event_group_ids):
            raise ValueError("Bridge-E1 event marks disagree")
        if len(self.quadrature_anchor) != len(self.quadrature_history):
            raise ValueError("Bridge-E1 quadrature arrays disagree")
        if len(self.quadrature_anchor) != len(self.quadrature_time):
            raise ValueError("Bridge-E1 quadrature times disagree")
        if len(self.quadrature_anchor) != len(self.quadrature_weight_seconds):
            raise ValueError("Bridge-E1 quadrature weights disagree")
        if np.any(self.event_anchor < 0) or np.any(self.event_anchor >= anchors):
            raise ValueError("Bridge-E1 event anchor out of range")
        if np.any(self.quadrature_anchor < 0) or np.any(self.quadrature_anchor >= anchors):
            raise ValueError("Bridge-E1 quadrature anchor out of range")
        if np.any(self.event_time < self.anchor_time[self.event_anchor]):
            raise ValueError("Bridge-E1 event precedes its causal anchor")
        if np.any(self.quadrature_time < self.anchor_time[self.quadrature_anchor]):
            raise ValueError("Bridge-E1 quadrature point precedes its causal anchor")
        if not np.isfinite(self.explicit).all():
            raise ValueError("Bridge-E1 explicit features non-finite")

    def anchor_ids(self, split: str) -> np.ndarray:
        code = {"train": 0, "validation": 1}[split]
        return np.flatnonzero(self.anchor_split == code)


def _eligible_anchor_support(reader: RawAnchorReader, coverage: CoverageTable,
                             *, max_train: int, max_validation: int
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    anchor, split, raw_session = reader.anchor_times()
    kept_time = []
    kept_split = []
    kept_session = []
    kept_stop = []
    train_end, dev_end = contract.load_split(reader.subject)
    for value, split_code, _raw_cache_session in zip(anchor, split, raw_session):
        match = np.flatnonzero(
            (coverage.start <= value) & (value < coverage.stop)
        )
        if len(match) != 1:
            continue
        stop = min(float(value) + 30.0, float(coverage.stop[match[0]]))
        if split_code == 0:
            stop = min(stop, train_end)
        else:
            stop = min(stop, dev_end)
        if stop - float(value) < 5.0:
            continue
        kept_time.append(float(value))
        kept_split.append(int(split_code))
        # Raw-cache session ids are local to the capped cache and restart from
        # zero; event/coverage session ids span the full recording.  Time is the
        # cross-package key.  Carry the canonical coverage label downstream.
        kept_session.append(int(coverage.session[match[0]]))
        kept_stop.append(float(stop))
    time = np.asarray(kept_time)
    split = np.asarray(kept_split, dtype=np.int8)
    session = np.asarray(kept_session, dtype=np.int64)
    stop = np.asarray(kept_stop)
    selected = np.concatenate([
        _uniform_take(np.flatnonzero(split == 0), max_train),
        _uniform_take(np.flatnonzero(split == 1), max_validation),
    ])
    selected.sort()
    return time[selected], split[selected], session[selected], stop[selected]


def build_bridge_e1_design(subject: str, baseline_checkpoint: Path,
                           *, max_train_anchors: int = 256,
                           max_validation_anchors: int = 128,
                           quadrature_order: int = 4,
                           explicit_scaler: tuple[np.ndarray, np.ndarray] | None = None,
                           stream: R1EventStream | None = None,
                           coverage: CoverageTable | None = None,
                           ) -> tuple[BridgeE1Design, RawAnchorReader, dict]:
    stream = stream or load_event_stream(subject)
    coverage = coverage or CoverageTable.load(
        contract.RESULT_ROOT / "coverage" / f"{subject}.npz"
    )
    if stream.subject != subject or coverage.subject != subject:
        raise ValueError(f"{subject}: Bridge-E1 stream/coverage subject mismatch")
    checkpoint = torch.load(baseline_checkpoint, map_location="cpu", weights_only=False)
    scaler = HistoryScaler(
        mean=np.asarray(checkpoint["history_scaler"]["mean"], dtype=np.float32),
        scale=np.asarray(checkpoint["history_scaler"]["scale"], dtype=np.float32),
    )
    history = DeterministicHistory(
        stream, session_start_map(stream, coverage.session, coverage.start)
    )
    reader = RawAnchorReader(subject, stream.event_time)
    anchor, split, session, support_stop = _eligible_anchor_support(
        reader, coverage, max_train=max_train_anchors,
        max_validation=max_validation_anchors,
    )
    observations = []
    kept = []
    for index, value in enumerate(anchor):
        observation = reader.read(float(value))
        if observation is None:
            continue
        observations.append(observation)
        kept.append(index)
    if not observations:
        raise ValueError(f"{subject}: no Bridge-E1 raw observations")
    kept = np.asarray(kept, dtype=np.int64)
    anchor, split, session, support_stop = (
        value[kept] for value in (anchor, split, session, support_stop)
    )
    explicit = np.stack([value.explicit for value in observations])
    contact_mask = np.stack([value.contact_mask for value in observations])
    if explicit_scaler is None:
        train_value = explicit[split == 0]
        train_mask = contact_mask[split == 0]
        flattened = train_value[train_mask]
        explicit_mean = flattened.mean(0)
        explicit_scale = flattened.std(0)
        explicit_scale = np.where(explicit_scale > 1e-6, explicit_scale, 1.0)
    else:
        explicit_mean = np.asarray(explicit_scaler[0], dtype=np.float32)
        explicit_scale = np.asarray(explicit_scaler[1], dtype=np.float32)
        if explicit_mean.shape != (explicit.shape[2],):
            raise ValueError("frozen explicit mean has wrong shape")
        if explicit_scale.shape != (explicit.shape[2],) or np.any(explicit_scale <= 0):
            raise ValueError("frozen explicit scale is invalid")
    explicit = ((explicit - explicit_mean) / explicit_scale).astype(np.float32)

    event_anchor = []
    event_times = []
    event_history = []
    event_group_ids = []
    event_group_count = []
    q_anchor = []
    q_time = []
    q_history = []
    q_weight = []
    node, weight = np.polynomial.legendre.leggauss(int(quadrature_order))
    for anchor_index, (left, right, label, split_code) in enumerate(
        zip(anchor, support_stop, session, split)
    ):
        event_index = np.flatnonzero(
            (stream.event_time > left) & (stream.event_time < right)
            & (stream.session == label) & (stream.split == split_code)
        )
        if len(event_index):
            event_anchor.extend([anchor_index] * len(event_index))
            event_times.append(stream.event_time[event_index])
            event_history.append(history.evaluate(
                stream.event_time[event_index], stream.session[event_index]
            ))
            event_group_ids.append(stream.group_ids[event_index])
            event_group_count.append(stream.group_count[event_index])
        event_time = stream.event_time[event_index]
        boundary = np.concatenate([[left], event_time[(event_time > left) & (event_time < right)], [right]])
        width = np.diff(boundary)
        valid = width > 0
        a, b = boundary[:-1][valid], boundary[1:][valid]
        midpoint, half = 0.5 * (a + b), 0.5 * (b - a)
        time = (midpoint[:, None] + half[:, None] * node[None, :]).reshape(-1)
        q_anchor.extend([anchor_index] * len(time))
        q_time.append(time)
        q_history.append(history.evaluate(
            time, np.full(len(time), int(label), dtype=np.int64)
        ))
        q_weight.append((half[:, None] * weight[None, :]).reshape(-1))
    if not event_history:
        raise ValueError(f"{subject}: selected Bridge-E1 support contains no events")
    design = BridgeE1Design(
        subject=subject,
        anchor_time=anchor.astype(np.float64),
        anchor_split=split.astype(np.int8),
        anchor_session=session.astype(np.int64),
        anchor_history=scaler.transform(history.evaluate(anchor, session)),
        explicit=explicit,
        explicit_mean=np.asarray(explicit_mean, dtype=np.float32),
        explicit_scale=np.asarray(explicit_scale, dtype=np.float32),
        contact_mask=contact_mask,
        coordinates=observations[0].coordinates,
        coordinate_valid=observations[0].coordinate_valid,
        shaft_index=observations[0].shaft_index,
        event_anchor=np.asarray(event_anchor, dtype=np.int64),
        event_time=np.concatenate(event_times).astype(np.float64),
        event_history=scaler.transform(np.concatenate(event_history)),
        event_group_ids=np.concatenate(event_group_ids).astype(np.int64),
        event_group_count=np.concatenate(event_group_count).astype(np.int64),
        quadrature_anchor=np.asarray(q_anchor, dtype=np.int64),
        quadrature_time=np.concatenate(q_time).astype(np.float64),
        quadrature_history=scaler.transform(np.concatenate(q_history)),
        quadrature_weight_seconds=np.concatenate(q_weight).astype(np.float64),
    )
    design.validate()
    manifest = {
        "contract": contract.REVISION,
        "bridge_e1_revision": BRIDGE_E1_REVISION,
        "raw_observation_revision": RAW_OBSERVATION_REVISION,
        "subject": subject,
        "n_train_anchors": int(np.sum(split == 0)),
        "n_validation_anchors": int(np.sum(split == 1)),
        "n_train_events": int(np.sum(split[design.event_anchor] == 0)),
        "n_validation_events": int(np.sum(split[design.event_anchor] == 1)),
        "n_raw_contacts": int(explicit.shape[1]),
        "explicit_dim": int(explicit.shape[2]),
        "explicit_scaler_source": (
            "selected_train_anchors" if explicit_scaler is None else "frozen_bridge_train"
        ),
        "explicit_mean": np.asarray(explicit_mean, dtype=float).tolist(),
        "explicit_scale": np.asarray(explicit_scale, dtype=float).tolist(),
        "selection_is_event_independent": True,
        "support_is_post_anchor_recorded_time": True,
        "baseline_checkpoint": str(baseline_checkpoint),
        "baseline_checkpoint_sha256": contract.sha256_file(baseline_checkpoint),
        "coverage_source_hashes": dict(coverage.source_hashes),
        "event_stream_source_hashes": dict(stream.source_hashes),
        "sealed_opened": False,
    }
    if manifest["n_train_anchors"] == 0 or manifest["n_validation_anchors"] == 0:
        raise ValueError(f"{subject}: Bridge-E1 has an empty anchor split")
    if manifest["n_train_events"] == 0 or manifest["n_validation_events"] == 0:
        raise ValueError(f"{subject}: Bridge-E1 has an empty event split")
    return design, reader, manifest


class ObservationResidualEventModel(nn.Module):
    """Frozen exact history baseline plus zero-effect observation adapters."""

    def __init__(self, baseline_checkpoint: dict, history_dim: int,
                 n_mark_contacts: int, adjacency: np.ndarray,
                 explicit_dim: int, *, raw_enabled: bool,
                 d_model: int = 64):
        super().__init__()
        self.observer = ObservationTransformer(
            explicit_dim, d_model=d_model, patch_samples=128,
            n_heads=4, temporal_layers=2, spatial_layers=1,
            raw_enabled=raw_enabled,
        )
        self.timing_baseline = HistoryIntensity(history_dim, history_visible=True)
        self.timing_baseline.load_state_dict(baseline_checkpoint["timing"]["history"])
        self.mark_baseline = ExactHistoryMarkDecoder(
            history_dim, n_mark_contacts, adjacency, history_visible=True
        )
        self.mark_baseline.load_state_dict(baseline_checkpoint["mark"]["history"])
        for module in (self.timing_baseline, self.mark_baseline):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.obs_timing = nn.Linear(d_model, 1, bias=False)
        self.obs_contact = nn.Linear(d_model, n_mark_contacts, bias=False)
        self.obs_size = nn.Linear(d_model, n_mark_contacts + 1, bias=False)
        for module in (self.obs_timing, self.obs_contact, self.obs_size):
            nn.init.zeros_(module.weight)

    def observation_embedding(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.observer(**batch)

    def timing_log_rate(self, history: torch.Tensor,
                        embedding: torch.Tensor) -> torch.Tensor:
        return self.timing_baseline(history) + self.obs_timing(embedding).squeeze(-1)

    def mark_terms(self, history: torch.Tensor, embedding: torch.Tensor,
                   group_ids: torch.Tensor, group_count: torch.Tensor):
        size, contact = self.mark_baseline.logits(history, group_ids, group_count)
        size = size + self.obs_size(embedding).unsqueeze(1)
        contact = contact + self.obs_contact(embedding).unsqueeze(1)
        return tied_group_mark_log_prob(group_ids, group_count, size, contact)


def make_paired_models(baseline_checkpoint: dict, design: BridgeE1Design,
                       adjacency: np.ndarray, *, seed: int,
                       device: torch.device | str
                       ) -> tuple[ObservationResidualEventModel, ObservationResidualEventModel]:
    torch.manual_seed(int(seed))
    explicit = ObservationResidualEventModel(
        baseline_checkpoint, design.event_history.shape[1],
        design.event_group_ids.shape[1], adjacency, design.explicit.shape[2],
        raw_enabled=False,
    ).to(device)
    torch.manual_seed(int(seed) + 1)
    raw = ObservationResidualEventModel(
        baseline_checkpoint, design.event_history.shape[1],
        design.event_group_ids.shape[1], adjacency, design.explicit.shape[2],
        raw_enabled=True,
    ).to(device)
    copy_common_observer_state(explicit.observer, raw.observer)
    raw.obs_timing.load_state_dict(explicit.obs_timing.state_dict())
    raw.obs_contact.load_state_dict(explicit.obs_contact.state_dict())
    raw.obs_size.load_state_dict(explicit.obs_size.state_dict())
    return explicit, raw


def observation_batch(reader: RawAnchorReader, design: BridgeE1Design,
                      anchor_ids: np.ndarray,
                      device: torch.device | str, *, read_raw: bool
                      ) -> dict[str, torch.Tensor]:
    if read_raw:
        observations = [reader.read(float(design.anchor_time[index])) for index in anchor_ids]
        if any(value is None for value in observations):
            raise RuntimeError("a frozen Bridge-E1 anchor became unreadable")
        waveform = np.stack([value.waveform for value in observations])
        sample_valid = np.stack([value.sample_valid for value in observations])
    else:
        waveform = np.zeros(
            (len(anchor_ids), design.explicit.shape[1], 1), dtype=np.float32
        )
        sample_valid = np.ones_like(waveform, dtype=bool)
    batch = len(anchor_ids)
    return {
        "explicit": torch.as_tensor(design.explicit[anchor_ids], device=device),
        "waveform": torch.as_tensor(waveform, device=device),
        "sample_valid": torch.as_tensor(sample_valid, device=device),
        "contact_mask": torch.as_tensor(design.contact_mask[anchor_ids], device=device),
        "coordinates": torch.as_tensor(
            np.broadcast_to(design.coordinates, (batch, *design.coordinates.shape)).copy(),
            device=device,
        ),
        "coordinate_valid": torch.as_tensor(
            np.broadcast_to(design.coordinate_valid, (batch, len(design.coordinate_valid))).copy(),
            device=device,
        ),
        "shaft_index": torch.as_tensor(
            np.broadcast_to(design.shaft_index, (batch, len(design.shaft_index))).copy(),
            dtype=torch.long, device=device,
        ),
    }


@dataclass(frozen=True)
class BridgeE1Metrics:
    joint_nll_per_event: float
    timing_nll_per_event: float
    mark_nll_per_event: float
    group_size_nll_per_event: float
    subset_nll_per_event: float
    n_events: int
    n_anchors: int
    recorded_seconds: float


def _anchor_rows(mapping: np.ndarray, anchor_ids: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    lookup = {int(value): index for index, value in enumerate(anchor_ids.tolist())}
    row = np.flatnonzero(np.isin(mapping, anchor_ids))
    local = np.asarray([lookup[int(mapping[index])] for index in row], dtype=np.int64)
    return row, local


def batch_log_terms(model: ObservationResidualEventModel,
                    design: BridgeE1Design, reader: RawAnchorReader,
                    anchor_ids: np.ndarray, device: torch.device | str
                    ) -> dict[str, torch.Tensor | int | float]:
    batch = observation_batch(
        reader, design, anchor_ids, device,
        read_raw=model.observer.raw is not None,
    )
    embedding = model.observation_embedding(batch)
    event_row, event_local = _anchor_rows(design.event_anchor, anchor_ids)
    q_row, q_local = _anchor_rows(design.quadrature_anchor, anchor_ids)
    event_history = torch.as_tensor(design.event_history[event_row], device=device)
    q_history = torch.as_tensor(design.quadrature_history[q_row], device=device)
    event_embedding = embedding[torch.as_tensor(event_local, device=device)]
    q_embedding = embedding[torch.as_tensor(q_local, device=device)]
    event_log = model.timing_log_rate(event_history, event_embedding).sum()
    q_log = model.timing_log_rate(q_history, q_embedding)
    weight = torch.as_tensor(
        design.quadrature_weight_seconds[q_row], dtype=q_log.dtype, device=device
    )
    survival = torch.sum(weight * torch.exp(torch.clamp(q_log, max=20.0)))
    if len(event_row):
        mark = model.mark_terms(
            event_history, event_embedding,
            torch.as_tensor(design.event_group_ids[event_row], dtype=torch.long, device=device),
            torch.as_tensor(design.event_group_count[event_row], dtype=torch.long, device=device),
        )
        mark_log = mark.event_log_prob.sum()
        size_log = mark.group_size_log_prob.sum()
        subset_log = mark.subset_log_prob.sum()
    else:
        mark_log = event_log.new_zeros(())
        size_log = event_log.new_zeros(())
        subset_log = event_log.new_zeros(())
    return {
        "event_log": event_log, "survival": survival,
        "mark_log": mark_log, "size_log": size_log, "subset_log": subset_log,
        "n_events": int(len(event_row)), "recorded_seconds": float(weight.sum().detach().cpu()),
    }


@torch.no_grad()
def evaluate_bridge_e1_anchors(model: ObservationResidualEventModel,
                               design: BridgeE1Design, reader: RawAnchorReader,
                               anchors: np.ndarray, *, device: torch.device | str,
                               anchor_batch_size: int = 8) -> BridgeE1Metrics:
    model.eval()
    anchors = np.asarray(anchors, dtype=np.int64)
    total = {key: 0.0 for key in ("event_log", "survival", "mark_log", "size_log", "subset_log")}
    n_events = 0
    recorded = 0.0
    for lo in range(0, len(anchors), int(anchor_batch_size)):
        terms = batch_log_terms(
            model, design, reader, anchors[lo:lo + int(anchor_batch_size)], device
        )
        for key in total:
            total[key] += float(terms[key])
        n_events += int(terms["n_events"])
        recorded += float(terms["recorded_seconds"])
    denom = max(n_events, 1)
    timing = (total["survival"] - total["event_log"]) / denom
    mark = -total["mark_log"] / denom
    return BridgeE1Metrics(
        joint_nll_per_event=timing + mark,
        timing_nll_per_event=timing,
        mark_nll_per_event=mark,
        group_size_nll_per_event=-total["size_log"] / denom,
        subset_nll_per_event=-total["subset_log"] / denom,
        n_events=n_events, n_anchors=int(len(anchors)),
        recorded_seconds=recorded,
    )


def evaluate_bridge_e1(model: ObservationResidualEventModel,
                       design: BridgeE1Design, reader: RawAnchorReader,
                       split: str, *, device: torch.device | str,
                       anchor_batch_size: int = 8) -> BridgeE1Metrics:
    return evaluate_bridge_e1_anchors(
        model, design, reader, design.anchor_ids(split), device=device,
        anchor_batch_size=anchor_batch_size,
    )


def fit_bridge_e1(model: ObservationResidualEventModel,
                  design: BridgeE1Design, reader: RawAnchorReader,
                  *, seed: int, device: torch.device | str,
                  epochs: int = 8, anchor_batch_size: int = 8,
                  learning_rate: float = 3e-4) -> ObservationResidualEventModel:
    anchors = design.anchor_ids("train")
    if len(anchors) < 10:
        raise ValueError("Bridge-E1 needs at least ten TRAIN anchors for inner selection")
    initial_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }

    def train_one_epoch(selected: np.ndarray, optimizer, rng) -> None:
        order = np.array(selected, copy=True)
        rng.shuffle(order)
        model.train()
        for lo in range(0, len(order), int(anchor_batch_size)):
            take = order[lo:lo + int(anchor_batch_size)]
            terms = batch_log_terms(model, design, reader, take, device)
            denom = max(int(terms["n_events"]), 1)
            loss = (
                terms["survival"] - terms["event_log"] - terms["mark_log"]
            ) / denom
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    cut = int(np.clip(math.floor(0.8 * len(anchors)), 1, len(anchors) - 1))
    inner_train, inner_validation = anchors[:cut], anchors[cut:]
    best_epoch = 0
    best_value = evaluate_bridge_e1_anchors(
        model, design, reader, inner_validation, device=device,
        anchor_batch_size=anchor_batch_size,
    ).joint_nll_per_event
    selection_optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(learning_rate), weight_decay=1e-3,
    )
    selection_rng = np.random.default_rng(int(seed))
    for epoch in range(1, int(epochs) + 1):
        train_one_epoch(inner_train, selection_optimizer, selection_rng)
        value = evaluate_bridge_e1_anchors(
            model, design, reader, inner_validation, device=device,
            anchor_batch_size=anchor_batch_size,
        ).joint_nll_per_event
        if value < best_value:
            best_value = value
            best_epoch = epoch

    # Refit from the exact common initial state on all TRAIN anchors.  Epoch 0
    # remains an admissible outcome: no observation increment is a scientific
    # negative, not an optimisation failure or a reason to worsen the baseline.
    model.load_state_dict(initial_state)
    if best_epoch:
        refit_optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(learning_rate), weight_decay=1e-3,
        )
        refit_rng = np.random.default_rng(int(seed) + 1000)
        for _ in range(best_epoch):
            train_one_epoch(anchors, refit_optimizer, refit_rng)
    model.selected_epochs = int(best_epoch)
    model.inner_validation_joint_nll = float(best_value)
    return model.eval()
