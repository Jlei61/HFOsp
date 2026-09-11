"""Training and open-loop evaluation for the v0.3 three-patient pilot.

This module is intentionally a pilot, not a cohort runner.  It implements the
scientific spine that was missing from v0.1/v0.2:

* physical-time TRAIN/VAL/TEST and seizure/gap-bounded state carry;
* event-time likelihood with survival evidence;
* exact tied-group mark likelihood through a frozen patient grammar;
* event content reaches the state only after that event is scored;
* fixed-time 5/30/120 minute autonomous evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from src.topic5_group_event_state.dataset import SubjectSequence
from src.topic5_group_event_state.model import (
    DataShape,
    EncoderConfig,
    EventEncoder,
    InputStats,
)
from src.topic5_group_event_state.train import _data_shape, _load_geometry
from src.topic5_group_event_state.v02.subject import SubjectTimeline, load_subject_timeline
from src.topic5_interictal_operator import build_contact_features

from .grammar import FrozenContactGrammar, build_train_only_grammar
from .partition import (
    PHASE_NAMES,
    NestedTimePartition,
    nested_time_partition,
    positions_for_phase,
)
from .point_process import censored_interval_integral, interval_point_process_terms
from .state import FixedTimescaleEventState, StateConfig


DATASET_ROOT = Path("/data/hfosp_group_event_state_v0_1/dataset")
LEGACY_ROOT = Path(
    "/home/honglab/leijiaxin/HFOsp/results/topic5_interictal_rank_distribution"
)
LEGACY_SEED = 20260725
PILOT_SUBJECTS = (
    "epilepsiae_1146",
    "yuquan_pengzihang",
    "yuquan_zhangkexuan",
)
SOURCE_COMMIT = os.environ.get("GROUP_EVENT_STATE_SOURCE_COMMIT", "unknown")


@dataclass(frozen=True)
class PilotConfig:
    grammar_batch: int = 256
    grammar_epochs: int = 12
    grammar_patience: int = 3
    grammar_lr: float = 3e-3
    # TBPTT is bounded by both event count and physical duration.  State is
    # carried and detached at a chunk edge, never reset there.
    chunk_events: int = 1024
    chunk_seconds: float = 1800.0
    segment_burnin_seconds: float = 300.0
    max_epochs: int = 12
    min_epochs: int = 3
    patience: int = 3
    encoder_lr: float = 1e-4
    state_lr: float = 4e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    timing_weight: float = 1.0
    size_weight: float = 1.0
    subset_weight: float = 1.0
    future_count_horizons_seconds: tuple[float, ...] = (300.0, 1800.0, 7200.0)
    # Each raw Poisson NLL is divided by the state-TRAIN mean count at that
    # horizon before applying this weight.  Otherwise the 2 h loss dominates
    # solely because it contains more events.
    future_count_weights: tuple[float, ...] = (0.50, 0.50, 0.50)
    amp: bool = True
    max_train_seconds: float = 4 * 3600.0
    validation_every: int = 1


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float))
    os.replace(tmp, path)


def _atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), tmp)
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _legacy_paths(subject: str) -> tuple[Path, Path]:
    checkpoint = (
        LEGACY_ROOT
        / "runs/formal_multiseed_20260725_v1"
        / f"seed_{LEGACY_SEED}"
        / subject
        / "full_history_gru_checkpoint.pt"
    )
    dataset = LEGACY_ROOT / "dataset_v0_4/per_subject" / f"{subject}.npz"
    if not checkpoint.exists() or not dataset.exists():
        raise FileNotFoundError(f"missing legacy grammar input for {subject}")
    return checkpoint, dataset


def validate_contact_contract(subject: str, seq: SubjectSequence, legacy_npz: Path) -> None:
    with np.load(legacy_npz, allow_pickle=True) as data:
        old = [str(v) for v in data["contact_names"].tolist()]
    new = [str(row["lagpat_label"]) for row in seq.index["contacts"]]
    if old != new:
        raise ValueError(f"{subject}: legacy/new contact order differs\nold={old}\nnew={new}")


def nested_partition(timeline: SubjectTimeline) -> NestedTimePartition:
    return nested_time_partition(timeline.segments)


def seq_positions_for_phase(
    timeline: SubjectTimeline, partition: NestedTimePartition, phase: str
) -> np.ndarray:
    return positions_for_phase(
        timeline.event_times, timeline.stream_positions, partition, phase
    )


def grammar_fit_seq_positions(
    timeline: SubjectTimeline, partition: NestedTimePartition
) -> np.ndarray:
    mask = timeline.event_times < float(partition.grammar_fit_stop_epoch)
    return timeline.stream_positions[np.flatnonzero(mask)]


def train_only_contact_features(
    seq: SubjectSequence,
    seq_positions: np.ndarray,
    legacy_npz: Path,
) -> np.ndarray:
    """Recompute event-dependent contact support on calibration-fit only."""

    participation = np.asarray(
        seq.gather_positions(seq_positions)["participation"], dtype=np.float32
    )
    support = participation.mean(axis=0)
    with np.load(legacy_npz, allow_pickle=True) as data:
        names = [str(v) for v in data["contact_names"].tolist()]
        coords = np.asarray(data["contact_coords"], dtype=np.float64)
    features, _metadata = build_contact_features(names, support, coords)
    return features


def _group_count(group_ids: np.ndarray) -> np.ndarray:
    value = np.asarray(group_ids, dtype=np.int64)
    return np.maximum(value.max(axis=1) + 1, 0).astype(np.int64)


def _grammar_loss(
    grammar: FrozenContactGrammar,
    group_ids: Tensor,
    group_count: Tensor,
    state: Tensor,
) -> tuple[Tensor, dict[str, float]]:
    terms, outputs = grammar(group_ids, group_count, state)
    n_active = terms.active_step.float().sum().clamp_min(1.0)
    n_select = terms.select_step.float().sum().clamp_min(1.0)
    continuation = -outputs["continue_step_log_prob"].sum() / n_active
    positive_size = -outputs["positive_size_step_log_prob"].sum() / n_select
    subset = -terms.subset_step_log_prob.sum() / n_select
    return continuation + positive_size + subset, {
        "continue_nll_per_step": float(continuation.detach()),
        "positive_size_nll_per_group": float(positive_size.detach()),
        "subset_nll_per_group": float(subset.detach()),
        "n_events": int(group_ids.shape[0]),
    }


@torch.no_grad()
def _evaluate_grammar(
    grammar: FrozenContactGrammar,
    group_ids: np.ndarray,
    group_count: np.ndarray,
    positions: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    grammar.eval()
    continue_sum = positive_size_sum = subset_sum = active = select = 0.0
    for lo in range(0, positions.size, batch_size):
        pos = positions[lo : lo + batch_size]
        ids = torch.from_numpy(group_ids[pos]).to(device).long()
        count = torch.from_numpy(group_count[pos]).to(device).long()
        state = torch.zeros(pos.size, grammar.state_norm.normalized_shape[0], device=device)
        terms, outputs = grammar(ids, count, state)
        continue_sum -= float(outputs["continue_step_log_prob"].sum())
        positive_size_sum -= float(outputs["positive_size_step_log_prob"].sum())
        subset_sum -= float(terms.subset_step_log_prob.sum())
        active += float(terms.active_step.sum())
        select += float(terms.select_step.sum())
    return {
        "continue_nll_per_step": continue_sum / max(active, 1.0),
        "positive_size_nll_per_group": positive_size_sum / max(select, 1.0),
        "subset_nll_per_group": subset_sum / max(select, 1.0),
        "total": (
            continue_sum / max(active, 1.0)
            + positive_size_sum / max(select, 1.0)
            + subset_sum / max(select, 1.0)
        ),
        "n_events": int(positions.size),
    }


def calibrate_grammar(
    subject: str,
    *,
    device: torch.device,
    out_dir: Path,
    cfg: PilotConfig = PilotConfig(),
    overwrite: bool = False,
) -> dict[str, Any]:
    """Calibrate legacy contact logits under the new exact TRAIN-only grammar."""

    out_dir = Path(out_dir)
    checkpoint_out = out_dir / "grammar_v03.pt"
    report_out = out_dir / "grammar_v03.json"
    if checkpoint_out.exists() and report_out.exists() and not overwrite:
        return json.loads(report_out.read_text())

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    seq = SubjectSequence(DATASET_ROOT / subject)
    timeline = load_subject_timeline(subject)
    legacy_checkpoint, legacy_dataset = _legacy_paths(subject)
    validate_contact_contract(subject, seq, legacy_dataset)
    partition = nested_partition(timeline)
    grammar_fit_seq = grammar_fit_seq_positions(timeline, partition)
    grammar = build_train_only_grammar(
        legacy_checkpoint,
        train_only_contact_features(seq, grammar_fit_seq, legacy_dataset),
        state_dim=StateConfig().state_dim,
        device=device,
    )
    grammar.set_phase("calibration")

    raw = seq.gather_positions(timeline.stream_positions)
    group_ids = np.asarray(raw["tied_group_id"], dtype=np.int64)
    count = _group_count(group_ids)
    calibration = np.flatnonzero(partition.labels_of(timeline.event_times) == 0)
    fit = np.flatnonzero(
        timeline.event_times < float(partition.grammar_fit_stop_epoch)
    )
    inner = calibration[timeline.event_times[calibration] >= partition.grammar_fit_stop_epoch]
    if fit.size < 100 or inner.size < 20:
        raise ValueError(
            f"{subject}: insufficient nested grammar prefix fit={fit.size}, inner={inner.size}"
        )
    optimizer = torch.optim.AdamW(
        grammar.calibration_parameters, lr=cfg.grammar_lr, weight_decay=cfg.weight_decay
    )
    rng = np.random.default_rng(0)
    best = math.inf
    best_state: dict[str, Tensor] | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(cfg.grammar_epochs):
        grammar.train()
        order = rng.permutation(fit)
        train_values = []
        for lo in range(0, order.size, cfg.grammar_batch):
            pos = order[lo : lo + cfg.grammar_batch]
            ids = torch.from_numpy(group_ids[pos]).to(device).long()
            n_group = torch.from_numpy(count[pos]).to(device).long()
            state = torch.zeros(pos.size, StateConfig().state_dim, device=device)
            loss, values = _grammar_loss(grammar, ids, n_group, state)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(grammar.calibration_parameters, 5.0)
            optimizer.step()
            train_values.append(float(loss.detach()))
        validation = _evaluate_grammar(
            grammar, group_ids, count, inner, cfg.grammar_batch, device
        )
        row = {
            "epoch": epoch,
            "train_total": float(np.mean(train_values)),
            "inner_validation": validation,
        }
        history.append(row)
        if validation["total"] < best - 1e-5:
            best = float(validation["total"])
            best_state = {k: v.detach().cpu().clone() for k, v in grammar.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= cfg.grammar_patience:
            break
    if best_state is None:
        raise RuntimeError("grammar calibration never produced a finite checkpoint")
    grammar.load_state_dict(best_state)
    grammar.set_phase("frozen")
    _atomic_torch(
        checkpoint_out,
        {
            "format": "group_event_contact_grammar_v0_3",
            "subject": subject,
            "state_dict": grammar.state_dict(),
            "architecture_checkpoint": str(legacy_checkpoint),
            "architecture_checkpoint_sha256": _sha256(legacy_checkpoint),
            "legacy_learned_weights_loaded": False,
            "calibration_prefix_events": int(calibration.size),
            "inner_fit_events": int(fit.size),
            "inner_validation_events": int(inner.size),
            "nested_boundary_epochs": partition.boundary_epochs.tolist(),
            "grammar_fit_stop_epoch": float(partition.grammar_fit_stop_epoch),
            "config": asdict(cfg),
            "source_commit": SOURCE_COMMIT,
        },
    )
    report = {
        "format": "group_event_contact_grammar_v0_3_report",
        "subject": subject,
        "status": "complete",
        "best_inner_validation_total": best,
        "selected_epoch": int(np.argmin([r["inner_validation"]["total"] for r in history])),
        "history": history,
        "checkpoint": str(checkpoint_out),
        "checkpoint_sha256": _sha256(checkpoint_out),
        "contact_order_validated": True,
        "calibration_prefix_only": True,
        "legacy_role": "architecture_hyperparameters_only_no_learned_weights",
        "scoring_contract": (
            "single K=0..N categorical algebraically reported as continue plus "
            "K_given_continue; product-form conditional K-subset likelihood"
        ),
        "source_commit": SOURCE_COMMIT,
    }
    _atomic_json(report_out, report)
    return report


def estimate_input_stats_positions(
    seq: SubjectSequence,
    positions: np.ndarray,
    *,
    max_events: int = 2048,
    seed: int = 0,
) -> InputStats:
    rng = np.random.default_rng(seed)
    pos = np.asarray(positions, dtype=np.int64)
    if pos.size > max_events:
        pos = np.sort(rng.choice(pos, max_events, replace=False))
    batch = seq.gather_positions(pos)
    wave = np.abs(np.nan_to_num(batch["waveform"].astype(np.float32)))
    waveform_scale = float(np.percentile(wave[wave > 0], 90)) if np.any(wave > 0) else 1.0
    env = np.nan_to_num(batch["band_envelope"].astype(np.float32))
    envelope_scale = float(np.percentile(env[env > 0], 50)) if np.any(env > 0) else 1.0
    lag = np.nan_to_num(batch["cross_band_lag"].astype(np.float32))
    cross_band_scale = float(np.percentile(np.abs(lag), 90)) or 1.0
    feats = np.nan_to_num(batch["band_features"].astype(np.float32))
    bg = np.nan_to_num(batch["background"].astype(np.float32))
    return InputStats(
        waveform_scale=waveform_scale,
        envelope_scale=envelope_scale,
        cross_band_scale=cross_band_scale,
        band_feature_mean=feats.mean(axis=(0, 1)),
        band_feature_std=feats.std(axis=(0, 1)),
        background_mean=bg.mean(axis=(0, 1)),
        background_std=bg.std(axis=(0, 1)),
    )


def _to_device(raw: Mapping[str, np.ndarray], device: torch.device) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for key, value in raw.items():
        array = np.ascontiguousarray(value)
        if array.dtype == np.bool_:
            out[key] = torch.from_numpy(array).to(device)
        elif np.issubdtype(array.dtype, np.integer):
            out[key] = torch.from_numpy(array.astype(np.int64)).to(device)
        else:
            out[key] = torch.from_numpy(array.astype(np.float32)).to(device)
    return out


class PilotModel(nn.Module):
    def __init__(
        self,
        event_encoder: EventEncoder,
        state: FixedTimescaleEventState,
        grammar: FrozenContactGrammar,
    ) -> None:
        super().__init__()
        self.event_encoder = event_encoder
        self.state = state
        self.grammar = grammar


def build_model(
    subject: str,
    seq: SubjectSequence,
    grammar_checkpoint: Path,
    *,
    seed: int,
    device: torch.device,
) -> PilotModel:
    torch.manual_seed(seed)
    np.random.seed(seed)
    legacy_checkpoint, legacy_dataset = _legacy_paths(subject)
    timeline = load_subject_timeline(subject)
    partition = nested_partition(timeline)
    grammar_fit_positions = grammar_fit_seq_positions(timeline, partition)
    grammar = build_train_only_grammar(
        legacy_checkpoint,
        train_only_contact_features(seq, grammar_fit_positions, legacy_dataset),
        state_dim=StateConfig().state_dim,
        device=device,
    )
    saved = torch.load(grammar_checkpoint, map_location="cpu", weights_only=False)
    grammar.load_state_dict(saved["state_dict"], strict=True)
    # Calibration is shared; state adapters are reinitialised per seed.
    torch.manual_seed(seed)
    grammar._init_adapters()
    with torch.no_grad():
        grammar.initial_gate.fill_(-4.0)
        grammar.query_gate.fill_(-4.0)
        grammar.size_gate.fill_(-4.0)
    grammar.set_phase("adapter")

    # Raw/event feature scaling is part of patient calibration.  It never sees
    # the state-training period, development validation, or development test.
    stats = estimate_input_stats_positions(seq, grammar_fit_positions, seed=0)
    enc_cfg = EncoderConfig(
        use_participation=True,
        use_exact_delay=True,
        use_tied_groups=True,
        use_legacy_rank=False,
        use_waveform=True,
        use_multiband=True,
        use_geometry=True,
        d_contact=32,
        d_event=StateConfig().event_dim,
        n_attention_heads=4,
        n_attention_layers=1,
        waveform_channels=16,
        dropout=0.1,
    )
    geometry = _load_geometry(seq)
    encoder = EventEncoder(
        enc_cfg,
        _data_shape(seq),
        geometry.to(device) if geometry is not None else None,
        stats,
    ).to(device)
    state = FixedTimescaleEventState(StateConfig()).to(device)
    state_train_positions = seq_positions_for_phase(timeline, partition, "state_train")
    state.initialise_intensity_rate(
        state_train_positions.size,
        partition.recorded_seconds["state_train"],
    )
    return PilotModel(encoder, state, grammar).to(device)


def seq_positions_for_split(timeline: SubjectTimeline, split_name: str) -> np.ndarray:
    label = {"train": 0, "val": 1, "test": 2}[split_name]
    mask = timeline.split.labels_of(timeline.event_times) == label
    return timeline.stream_positions[np.flatnonzero(mask)]


class _LossAccumulator:
    def __init__(self) -> None:
        self.timing_sum = 0.0
        self.timing_events = 0
        self.survival_integral = 0.0
        self.observed_seconds = 0.0
        self.continue_sum = 0.0
        self.active_steps = 0
        self.positive_size_sum = 0.0
        self.subset_sum = 0.0
        self.select_steps = 0
        self.future_count_sum: dict[int, float] = {}
        self.future_count_n: dict[int, int] = {}

    def add_timing(self, terms) -> None:
        self.timing_sum += float(terms.event_nll.detach().sum())
        self.timing_events += int(terms.event_nll.numel())
        self.survival_integral += float(terms.survival_integral.detach().sum())
        self.observed_seconds += float(terms.observed_seconds.detach().sum())

    def add_mark(self, terms) -> None:
        self.continue_sum -= float(terms.continue_step_log_prob.detach().sum())
        self.active_steps += int(terms.active_step.sum())
        self.positive_size_sum -= float(
            terms.positive_size_step_log_prob.detach().sum()
        )
        self.subset_sum -= float(terms.subset_step_log_prob.detach().sum())
        self.select_steps += int(terms.select_step.sum())

    def add_future_count(self, horizon: int, values: Tensor) -> None:
        self.future_count_sum[horizon] = self.future_count_sum.get(horizon, 0.0) + float(
            values.detach().sum()
        )
        self.future_count_n[horizon] = self.future_count_n.get(horizon, 0) + int(
            values.numel()
        )

    def add_tail(self, integral: Tensor, seconds: float) -> None:
        self.timing_sum += float(integral.detach().sum())
        self.survival_integral += float(integral.detach().sum())
        self.observed_seconds += float(seconds)

    def means(self) -> dict[str, float]:
        timing = self.timing_sum / max(self.timing_events, 1)
        continuation = self.continue_sum / max(self.active_steps, 1)
        positive_size = self.positive_size_sum / max(self.select_steps, 1)
        subset = self.subset_sum / max(self.select_steps, 1)
        future = {
            f"{h}s": self.future_count_sum[h] / max(self.future_count_n[h], 1)
            for h in sorted(self.future_count_sum)
        }
        return {
            "timing_nll_per_event": timing,
            "continue_nll_per_step": continuation,
            "positive_size_nll_per_group": positive_size,
            "subset_nll_per_group": subset,
            "local_total": timing + continuation + positive_size + subset,
            "future_count_poisson_nll": future,
            "total": timing + continuation + positive_size + subset + sum(future.values()),
            "survival_integral": self.survival_integral,
            "observed_seconds": self.observed_seconds,
            "n_events": self.timing_events,
            "n_active_steps": self.active_steps,
            "n_select_steps": self.select_steps,
            "n_future_count_anchors": {
                f"{h}s": self.future_count_n[h] for h in sorted(self.future_count_n)
            },
        }


def _segment_event_indices(timeline: SubjectTimeline) -> Iterable[tuple[int, np.ndarray]]:
    for segment_index in range(len(timeline.segments)):
        pos = np.flatnonzero(timeline.event_segment == segment_index)
        if pos.size:
            yield segment_index, pos


def _physical_event_chunks(
    positions: np.ndarray,
    event_times: np.ndarray,
    *,
    max_events: int,
    max_seconds: float,
) -> Iterable[np.ndarray]:
    """Chronological TBPTT chunks bounded in both events and real time."""

    pos = np.asarray(positions, dtype=np.int64)
    start = 0
    while start < pos.size:
        stop = min(start + int(max_events), pos.size)
        first_time = float(event_times[pos[start]])
        physical_stop = int(
            np.searchsorted(event_times[pos], first_time + float(max_seconds), side="right")
        )
        stop = min(stop, max(physical_stop, start + 1))
        yield pos[start:stop]
        start = stop


def _poisson_nll_tensor(target: Tensor, expected: Tensor) -> Tensor:
    mu = expected.clamp_min(1e-6)
    y = target.to(mu.dtype)
    return mu - y * torch.log(mu) + torch.lgamma(y + 1.0)


def _autonomous_expected_count(
    model: PilotModel, state: Tensor, horizon_seconds: float, *, n_grid: int = 17
) -> Tensor:
    grid = torch.linspace(
        0.0, float(horizon_seconds), n_grid, device=state.device, dtype=torch.float32
    )
    batch, width = state.shape
    repeated = state[:, None, :].expand(batch, grid.numel(), width).reshape(-1, width)
    dt = grid[None, :].expand(batch, -1).reshape(-1)
    intensity = model.state.intensity(model.state.evolve(repeated, dt)).reshape(
        batch, -1
    )
    return torch.trapezoid(intensity, grid, dim=1)


def _future_count_terms_at_anchors(
    model: PilotModel,
    timeline: SubjectTimeline,
    state_before_interval: Tensor,
    interval_start: float,
    anchor_indices: np.ndarray,
    *,
    score_hi: float,
    cfg: PilotConfig,
) -> dict[int, Tensor]:
    """Poisson future-count terms at fixed-time anchors, without future updates."""

    if not anchor_indices.size:
        return {}
    anchor_times = torch.from_numpy(timeline.grid.t_anchor[anchor_indices]).to(
        state_before_interval.device
    ).float()
    base = state_before_interval.expand(anchor_indices.size, -1)
    anchor_state = model.state.evolve(base, anchor_times - float(interval_start))
    out: dict[int, Tensor] = {}
    for horizon in cfg.future_count_horizons_seconds:
        horizon_index = timeline.config.horizons_seconds.index(float(horizon))
        eligible = (
            timeline.grid.eligible[anchor_indices, horizon_index]
            & (timeline.grid.t_anchor[anchor_indices] + float(horizon) <= score_hi + 1e-6)
        )
        if not eligible.any():
            continue
        local = np.flatnonzero(eligible)
        chosen = anchor_indices[local]
        target = (
            timeline.grid.window_hi[chosen, horizon_index]
            - timeline.grid.window_lo[chosen, horizon_index]
        )
        expected = _autonomous_expected_count(
            model, anchor_state[torch.from_numpy(local).to(anchor_state.device)], horizon
        )
        out[int(horizon)] = _poisson_nll_tensor(
            torch.from_numpy(target).to(expected.device), expected
        )
    return out


def _future_count_training_normalizers(
    timeline: SubjectTimeline,
    partition: NestedTimePartition,
    cfg: PilotConfig,
) -> dict[int, float]:
    """TRAIN-derived mean counts used only to balance horizon loss scales."""

    labels = partition.labels_of(timeline.grid.t_anchor)
    _lo, train_hi = partition.bounds("state_train")
    out: dict[int, float] = {}
    for horizon in cfg.future_count_horizons_seconds:
        h_i = timeline.config.horizons_seconds.index(float(horizon))
        anchors = np.flatnonzero(
            (labels == PHASE_NAMES.index("state_train"))
            & timeline.grid.eligible[:, h_i]
            & (timeline.grid.t_anchor + float(horizon) <= train_hi + 1e-6)
        )
        count = (
            timeline.grid.window_hi[anchors, h_i]
            - timeline.grid.window_lo[anchors, h_i]
        )
        out[int(horizon)] = max(float(np.mean(count)) if count.size else 0.0, 1.0)
    return out


def _chunk_forward(
    model: PilotModel,
    seq: SubjectSequence,
    timeline: SubjectTimeline,
    event_positions: np.ndarray,
    state_post: Tensor,
    previous_time: float,
    *,
    score_lo: float,
    score_hi: float,
    device: torch.device,
    amp: bool,
    anchor_indices: np.ndarray,
    cfg: PilotConfig,
) -> tuple[
    Tensor, float, Tensor, dict[str, Tensor] | None, dict[int, Tensor]
]:
    """Advance one chronological chunk and return differentiable scored terms."""

    seq_pos = timeline.stream_positions[event_positions]
    raw = seq.gather_positions(seq_pos)
    batch = _to_device(raw, device)
    times = torch.from_numpy(timeline.event_times[event_positions]).to(device).double()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
        event_embedding, _ = model.event_encoder(batch)
    event_embedding = event_embedding.float()
    pre_states: list[Tensor] = []
    timing_nll: list[Tensor] = []
    timing_integral: list[Tensor] = []
    timing_seconds: list[Tensor] = []
    future_count: dict[int, list[Tensor]] = {}
    start_intensity = model.state.intensity(state_post)
    last_t = float(previous_time)
    for step in range(event_positions.size):
        t = float(times[step])
        between = anchor_indices[
            (timeline.grid.t_anchor[anchor_indices] > last_t + 1e-9)
            & (timeline.grid.t_anchor[anchor_indices] <= t + 1e-9)
        ]
        for horizon, values in _future_count_terms_at_anchors(
            model,
            timeline,
            state_post,
            last_t,
            between,
            score_hi=score_hi,
            cfg=cfg,
        ).items():
            future_count.setdefault(horizon, []).append(values)
        dt_full = torch.tensor([max(t - last_t, 0.0)], device=device)
        state_pre = model.state.evolve(state_post, dt_full)
        event_intensity = model.state.intensity(state_pre)
        if score_lo <= t < score_hi:
            effective_start = max(last_t, score_lo)
            state_start = model.state.evolve(
                state_post,
                torch.tensor([max(effective_start - last_t, 0.0)], device=device),
            )
            lambda_start = model.state.intensity(state_start)
            terms = interval_point_process_terms(
                lambda_start,
                event_intensity,
                torch.tensor([max(t - effective_start, 0.0)], device=device),
            )
            timing_nll.append(terms.event_nll)
            timing_integral.append(terms.survival_integral)
            timing_seconds.append(terms.observed_seconds)
            pre_states.append(state_pre)
        state_post = model.state.update(state_pre, event_embedding[step : step + 1])
        start_intensity = model.state.intensity(state_post)
        last_t = t

    scored = None
    if pre_states:
        state = torch.cat(pre_states, dim=0)
        in_score = (timeline.event_times[event_positions] >= score_lo) & (
            timeline.event_times[event_positions] < score_hi
        )
        ids = batch["tied_group_id"][torch.from_numpy(in_score).to(device)].long()
        count = torch.clamp(ids.max(dim=1).values + 1, min=1)
        mark, mark_outputs = model.grammar(ids, count, state)
        scored = {
            "timing_nll": torch.cat(timing_nll),
            "timing_integral": torch.cat(timing_integral),
            "timing_seconds": torch.cat(timing_seconds),
            "state_pre": state,
            "mark_continue_step": mark_outputs["continue_step_log_prob"],
            "mark_positive_size_step": mark_outputs["positive_size_step_log_prob"],
            "mark_subset_step": mark.subset_step_log_prob,
            "mark_active": mark.active_step,
            "mark_select": mark.select_step,
        }
    return (
        state_post,
        last_t,
        start_intensity,
        scored,
        {h: torch.cat(v) for h, v in future_count.items() if v},
    )


def run_pass(
    model: PilotModel,
    seq: SubjectSequence,
    timeline: SubjectTimeline,
    phase: str,
    *,
    device: torch.device,
    cfg: PilotConfig,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, Any]:
    """Causal pass; every phase replays from segment start and scores only itself."""

    train = optimizer is not None
    model.train(train)
    partition = nested_partition(timeline)
    lower, upper = partition.bounds(phase)
    count_normalizers = _future_count_training_normalizers(timeline, partition, cfg)
    accum = _LossAccumulator()
    expected_scoreable_seconds = 0.0
    gradient_norms: list[float] = []
    gradient_by_group: dict[str, list[float]] = {
        "event_encoder": [], "state": [], "grammar_adapter": []
    }
    state_norms: list[float] = []
    for segment_index, all_positions in _segment_event_indices(timeline):
        segment = timeline.segments[segment_index]
        score_lo = max(float(segment.start_epoch), lower)
        # A newly initialised segment first rebuilds state causally.  This is a
        # segment burn-in, not a reset at each TBPTT chunk.
        score_lo = max(
            score_lo, float(segment.start_epoch) + float(cfg.segment_burnin_seconds)
        )
        score_hi = min(float(segment.stop_epoch), upper)
        if score_hi <= score_lo:
            continue
        expected_scoreable_seconds += score_hi - score_lo
        # Replay only what is needed for this split, always from the causal
        # segment start.  No state is carried across seizure/gap boundaries.
        use = all_positions[timeline.event_times[all_positions] < score_hi]
        state_post = model.state.initial(1, device)
        previous_time = float(segment.start_epoch)
        segment_anchors = np.flatnonzero(
            (timeline.grid.segment_index == segment_index)
            & (timeline.grid.t_anchor >= score_lo)
            & (timeline.grid.t_anchor < score_hi)
        )
        for pos in _physical_event_chunks(
            use,
            timeline.event_times,
            max_events=cfg.chunk_events,
            max_seconds=cfg.chunk_seconds,
        ):
            chunk_anchors = segment_anchors[
                (timeline.grid.t_anchor[segment_anchors] > previous_time + 1e-9)
                & (
                    timeline.grid.t_anchor[segment_anchors]
                    <= float(timeline.event_times[pos[-1]]) + 1e-9
                )
            ]
            if train:
                state_post, previous_time, _lam, scored, future_count = _chunk_forward(
                    model, seq, timeline, pos, state_post, previous_time,
                    score_lo=score_lo, score_hi=score_hi, device=device, amp=cfg.amp,
                    anchor_indices=chunk_anchors, cfg=cfg,
                )
            else:
                with torch.no_grad():
                    state_post, previous_time, _lam, scored, future_count = _chunk_forward(
                        model, seq, timeline, pos, state_post, previous_time,
                        score_lo=score_lo, score_hi=score_hi, device=device, amp=cfg.amp,
                        anchor_indices=chunk_anchors, cfg=cfg,
                    )
            loss_terms: list[Tensor] = []
            if scored is not None:
                timing = scored["timing_nll"].mean()
                continuation = -scored["mark_continue_step"].sum() / scored[
                    "mark_active"
                ].sum().clamp_min(1)
                positive_size = -scored["mark_positive_size_step"].sum() / scored[
                    "mark_select"
                ].sum().clamp_min(1)
                subset = -scored["mark_subset_step"].sum() / scored["mark_select"].sum().clamp_min(1)
                local_loss = (
                    cfg.timing_weight * timing
                    + cfg.size_weight * (continuation + positive_size)
                    + cfg.subset_weight * subset
                )
                loss_terms.append(local_loss)
                terms_proxy = type("T", (), {
                    "event_nll": scored["timing_nll"],
                    "survival_integral": scored["timing_integral"],
                    "observed_seconds": scored["timing_seconds"],
                })
                accum.add_timing(terms_proxy)
                mark_proxy = type("M", (), {
                    "continue_step_log_prob": scored["mark_continue_step"],
                    "positive_size_step_log_prob": scored["mark_positive_size_step"],
                    "subset_step_log_prob": scored["mark_subset_step"],
                    "active_step": scored["mark_active"],
                    "select_step": scored["mark_select"],
                })
                accum.add_mark(mark_proxy)
                state_norms.append(float(scored["state_pre"].detach().norm(dim=-1).mean()))
            for horizon, values in future_count.items():
                accum.add_future_count(horizon, values)
                horizon_index = cfg.future_count_horizons_seconds.index(float(horizon))
                loss_terms.append(
                    cfg.future_count_weights[horizon_index]
                    * values.mean()
                    / count_normalizers[int(horizon)]
                )
            if train and loss_terms:
                optimizer.zero_grad(set_to_none=True)
                torch.stack(loss_terms).sum().backward()
                for name, value in _parameter_group_gradient_norms(model).items():
                    gradient_by_group[name].append(value)
                norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg.grad_clip
                )
                if not torch.isfinite(norm):
                    raise FloatingPointError("non-finite state-model gradient")
                optimizer.step()
                gradient_norms.append(float(norm))
            state_post = state_post.detach()

        # The final event-free stretch is part of the point-process evidence.
        # Omitting it systematically underweights quiet periods, especially a
        # split ending well after its last observed event.
        tail_start = max(previous_time, score_lo)
        if score_hi > tail_start:
            tail_anchors = segment_anchors[
                (timeline.grid.t_anchor[segment_anchors] > previous_time + 1e-9)
                & (timeline.grid.t_anchor[segment_anchors] < score_hi)
            ]
            n_scored_segment = int(
                ((timeline.event_times[all_positions] >= score_lo)
                 & (timeline.event_times[all_positions] < score_hi)).sum()
            )
            if train:
                state_start = model.state.evolve(
                    state_post,
                    torch.tensor([tail_start - previous_time], device=device),
                )
                state_stop = model.state.evolve(
                    state_post,
                    torch.tensor([score_hi - previous_time], device=device),
                )
                tail = censored_interval_integral(
                    model.state.intensity(state_start),
                    model.state.intensity(state_stop),
                    torch.tensor([score_hi - tail_start], device=device),
                )
                tail_future = _future_count_terms_at_anchors(
                    model,
                    timeline,
                    state_post,
                    previous_time,
                    tail_anchors,
                    score_hi=score_hi,
                    cfg=cfg,
                )
                optimizer.zero_grad(set_to_none=True)
                tail_objective = tail / max(n_scored_segment, 1)
                for horizon, values in tail_future.items():
                    horizon_index = cfg.future_count_horizons_seconds.index(float(horizon))
                    tail_objective = tail_objective + (
                        cfg.future_count_weights[horizon_index]
                        * values.mean()
                        / count_normalizers[int(horizon)]
                    )
                tail_objective.backward()
                for name, value in _parameter_group_gradient_norms(model).items():
                    gradient_by_group[name].append(value)
                norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg.grad_clip
                )
                if not torch.isfinite(norm):
                    raise FloatingPointError("non-finite tail-survival gradient")
                optimizer.step()
                gradient_norms.append(float(norm))
            else:
                with torch.no_grad():
                    state_start = model.state.evolve(
                        state_post,
                        torch.tensor([tail_start - previous_time], device=device),
                    )
                    state_stop = model.state.evolve(
                        state_post,
                        torch.tensor([score_hi - previous_time], device=device),
                    )
                    tail = censored_interval_integral(
                        model.state.intensity(state_start),
                        model.state.intensity(state_stop),
                        torch.tensor([score_hi - tail_start], device=device),
                    )
                    tail_future = _future_count_terms_at_anchors(
                        model,
                        timeline,
                        state_post,
                        previous_time,
                        tail_anchors,
                        score_hi=score_hi,
                        cfg=cfg,
                    )
            accum.add_tail(tail, score_hi - tail_start)
            for horizon, values in tail_future.items():
                accum.add_future_count(horizon, values)
    out = accum.means()
    future_objective = 0.0
    for horizon, weight in zip(
        cfg.future_count_horizons_seconds, cfg.future_count_weights
    ):
        value = out["future_count_poisson_nll"].get(f"{int(horizon)}s")
        if value is not None:
            future_objective += (
                float(weight) * float(value) / count_normalizers[int(horizon)]
            )
    out["objective_total"] = float(out["local_total"] + future_objective)
    out["expected_scoreable_seconds"] = float(expected_scoreable_seconds)
    out["future_count_training_mean_count_normalizer"] = {
        f"{h}s": value for h, value in sorted(count_normalizers.items())
    }
    out.update({
        "phase": phase,
        "tbptt": {
            "max_events": cfg.chunk_events,
            "max_physical_seconds": cfg.chunk_seconds,
            "segment_burnin_seconds": cfg.segment_burnin_seconds,
            "carry_across_chunks": True,
            "reset_across_chunks": False,
        },
        "gradient_norm_median": float(np.median(gradient_norms)) if gradient_norms else None,
        "gradient_norm_by_parameter_group_median": {
            name: (float(np.median(values)) if values else None)
            for name, values in gradient_by_group.items()
        },
        "gradient_clip_fraction": (
            float(np.mean(np.asarray(gradient_norms) > cfg.grad_clip))
            if gradient_norms else None
        ),
        "state_norm_median": float(np.median(state_norms)) if state_norms else None,
    })
    return out


def _trainable_update_report(before: Mapping[str, Tensor], model: nn.Module) -> dict[str, float]:
    after = model.state_dict()
    groups = {"event_encoder": [], "state": [], "grammar_adapter": []}
    for key, value in before.items():
        if key.startswith("event_encoder."):
            group = "event_encoder"
        elif key.startswith("state."):
            group = "state"
        elif key.startswith("grammar.state_") or key.startswith("grammar.initial_gate") or key.startswith("grammar.query_gate") or key.startswith("grammar.size_gate"):
            group = "grammar_adapter"
        else:
            continue
        delta = (after[key].detach().cpu().float() - value.float()).norm().item()
        denom = value.float().norm().item()
        groups[group].append((delta, denom))
    out = {}
    for group, values in groups.items():
        num = math.sqrt(sum(v[0] ** 2 for v in values))
        den = math.sqrt(sum(v[1] ** 2 for v in values))
        out[group] = num / max(den, 1e-12)
    return out


def _trainable_absolute_update_report(
    before: Mapping[str, Tensor], model: nn.Module
) -> dict[str, float]:
    after = model.state_dict()
    groups = {"event_encoder": [], "state": [], "grammar_adapter": []}
    for key, value in before.items():
        if key.startswith("event_encoder."):
            group = "event_encoder"
        elif key.startswith("state."):
            group = "state"
        elif key.startswith("grammar.state_") or key.startswith("grammar.initial_gate") or key.startswith("grammar.query_gate") or key.startswith("grammar.size_gate"):
            group = "grammar_adapter"
        else:
            continue
        groups[group].append(
            (after[key].detach().cpu().float() - value.float()).norm().item()
        )
    return {
        group: math.sqrt(sum(value * value for value in values))
        for group, values in groups.items()
    }


def _parameter_group_gradient_norms(model: PilotModel) -> dict[str, float]:
    groups = {
        "event_encoder": list(model.event_encoder.parameters()),
        "state": list(model.state.parameters()),
        "grammar_adapter": list(model.grammar.adapter_parameters),
    }
    out: dict[str, float] = {}
    for name, parameters in groups.items():
        squared = [p.grad.detach().float().pow(2).sum() for p in parameters if p.grad is not None]
        out[name] = float(torch.sqrt(torch.stack(squared).sum())) if squared else 0.0
    return out


def train_state_model(
    subject: str,
    seed: int,
    *,
    device: torch.device,
    grammar_dir: Path,
    out_dir: Path,
    cfg: PilotConfig = PilotConfig(),
    overwrite: bool = False,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    checkpoint_out = out_dir / "checkpoint.pt"
    report_out = out_dir / "result.json"
    if checkpoint_out.exists() and report_out.exists() and not overwrite:
        return json.loads(report_out.read_text())
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    seq = SubjectSequence(DATASET_ROOT / subject)
    timeline = load_subject_timeline(subject)
    partition = nested_partition(timeline)
    grammar_checkpoint = Path(grammar_dir) / "grammar_v03.pt"
    if not grammar_checkpoint.exists():
        raise FileNotFoundError(f"calibrated grammar missing: {grammar_checkpoint}")
    model = build_model(subject, seq, grammar_checkpoint, seed=seed, device=device)
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(
        [
            {"params": model.event_encoder.parameters(), "lr": cfg.encoder_lr},
            {"params": model.state.parameters(), "lr": cfg.state_lr},
            {"params": model.grammar.adapter_parameters, "lr": cfg.state_lr},
        ],
        weight_decay=cfg.weight_decay,
    )
    history: list[dict[str, Any]] = []
    best = math.inf
    best_state: dict[str, Tensor] | None = None
    best_epoch = -1
    stale = 0
    started = time.time()
    for epoch in range(cfg.max_epochs):
        train_metrics = run_pass(
            model, seq, timeline, "state_train", device=device, cfg=cfg, optimizer=optimizer
        )
        if (epoch + 1) % cfg.validation_every == 0:
            validation = run_pass(
                model, seq, timeline, "dev_val", device=device, cfg=cfg, optimizer=None
            )
        else:
            validation = {"objective_total": math.nan}
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation})
        value = float(validation["objective_total"])
        if math.isfinite(value) and value < best - 1e-5:
            best = value
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        progress = {
            "format": "group_event_state_v0_3_progress",
            "subject": subject,
            "seed": seed,
            "status": "running",
            "history": history,
            "elapsed_seconds": time.time() - started,
        }
        _atomic_json(out_dir / "progress.json", progress)
        if epoch + 1 >= cfg.min_epochs and stale >= cfg.patience:
            break
        if time.time() - started > cfg.max_train_seconds:
            break
    if best_state is None:
        raise RuntimeError("state model never produced a finite validation checkpoint")
    model.load_state_dict(best_state, strict=True)
    dev_test = run_pass(
        model, seq, timeline, "dev_test", device=device, cfg=cfg, optimizer=None
    )
    _atomic_torch(
        checkpoint_out,
        {
            "format": "group_event_state_v0_3_pilot_checkpoint",
            "subject": subject,
            "seed": seed,
            "selected_epoch": best_epoch,
            "model_state": model.state_dict(),
            "config": asdict(cfg),
            "state_config": asdict(model.state.cfg),
            "source_commit": SOURCE_COMMIT,
            "grammar_checkpoint": str(grammar_checkpoint),
            "grammar_checkpoint_sha256": _sha256(grammar_checkpoint),
        },
    )
    report = {
        "format": "group_event_state_v0_3_pilot_result",
        "status": "complete",
        "subject": subject,
        "seed": seed,
        "selected_epoch": best_epoch,
        "history": history,
        "dev_test": dev_test,
        "parameter_updates": _trainable_update_report(before, model),
        "parameter_updates_absolute_l2": _trainable_absolute_update_report(before, model),
        "checkpoint": str(checkpoint_out),
        "checkpoint_sha256": _sha256(checkpoint_out),
        "elapsed_seconds": time.time() - started,
        "nested_boundary_epochs": partition.boundary_epochs.tolist(),
        "grammar_fit_stop_epoch": float(partition.grammar_fit_stop_epoch),
        "recorded_seconds": partition.recorded_seconds,
        "n_events": {
            name: int(seq_positions_for_phase(timeline, partition, name).size)
            for name in PHASE_NAMES
        },
        "development_test_only": True,
        "sealed_partition_opened": False,
        "source_commit": SOURCE_COMMIT,
    }
    _atomic_json(report_out, report)
    _atomic_json(out_dir / "progress.json", {**report, "status": "complete"})
    return report
