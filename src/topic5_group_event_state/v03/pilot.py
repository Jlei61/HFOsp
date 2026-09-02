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
    chunk_events: int = 64
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


def train_only_contact_features(
    seq: SubjectSequence,
    timeline: SubjectTimeline,
    legacy_npz: Path,
) -> np.ndarray:
    """Recompute event-dependent contact support on outer TRAIN only."""

    train_seq_positions = seq_positions_for_split(timeline, "train")
    participation = np.asarray(
        seq.gather_positions(train_seq_positions)["participation"], dtype=np.float32
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
    terms, _ = grammar(group_ids, group_count, state)
    n_active = terms.active_step.float().sum().clamp_min(1.0)
    n_select = terms.select_step.float().sum().clamp_min(1.0)
    size = -terms.group_size_step_log_prob.sum() / n_active
    subset = -terms.subset_step_log_prob.sum() / n_select
    return size + subset, {
        "size_nll_per_step": float(size.detach()),
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
    size_sum = subset_sum = active = select = 0.0
    for lo in range(0, positions.size, batch_size):
        pos = positions[lo : lo + batch_size]
        ids = torch.from_numpy(group_ids[pos]).to(device).long()
        count = torch.from_numpy(group_count[pos]).to(device).long()
        state = torch.zeros(pos.size, grammar.state_norm.normalized_shape[0], device=device)
        terms, _ = grammar(ids, count, state)
        size_sum -= float(terms.group_size_step_log_prob.sum())
        subset_sum -= float(terms.subset_step_log_prob.sum())
        active += float(terms.active_step.sum())
        select += float(terms.select_step.sum())
    return {
        "size_nll_per_step": size_sum / max(active, 1.0),
        "subset_nll_per_group": subset_sum / max(select, 1.0),
        "total": size_sum / max(active, 1.0) + subset_sum / max(select, 1.0),
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
    grammar = build_train_only_grammar(
        legacy_checkpoint,
        train_only_contact_features(seq, timeline, legacy_dataset),
        state_dim=StateConfig().state_dim,
        device=device,
    )
    grammar.set_phase("calibration")

    raw = seq.gather_positions(timeline.stream_positions)
    group_ids = np.asarray(raw["tied_group_id"], dtype=np.int64)
    count = _group_count(group_ids)
    train = timeline.train_event_positions()
    if train.size < 100:
        raise ValueError(f"{subject}: fewer than 100 physical-TRAIN events")
    split = max(int(round(train.size * 0.9)), 1)
    fit, inner = train[:split], train[split:]
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
            "outer_train_events": int(train.size),
            "inner_fit_events": int(fit.size),
            "inner_validation_events": int(inner.size),
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
        "outer_train_only": True,
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
    grammar = build_train_only_grammar(
        legacy_checkpoint,
        train_only_contact_features(seq, timeline, legacy_dataset),
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

    train_positions = seq_positions_for_split(timeline, "train")
    stats = estimate_input_stats_positions(seq, train_positions, seed=0)
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
    state.initialise_intensity_rate(
        timeline.train_event_positions().size,
        timeline.split.recorded_seconds["train"],
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
        self.size_sum = 0.0
        self.size_steps = 0
        self.subset_sum = 0.0
        self.select_steps = 0

    def add_timing(self, terms) -> None:
        self.timing_sum += float(terms.event_nll.detach().sum())
        self.timing_events += int(terms.event_nll.numel())
        self.survival_integral += float(terms.survival_integral.detach().sum())
        self.observed_seconds += float(terms.observed_seconds.detach().sum())

    def add_mark(self, terms) -> None:
        self.size_sum -= float(terms.group_size_step_log_prob.detach().sum())
        self.size_steps += int(terms.active_step.sum())
        self.subset_sum -= float(terms.subset_step_log_prob.detach().sum())
        self.select_steps += int(terms.select_step.sum())

    def add_tail(self, integral: Tensor, seconds: float) -> None:
        self.timing_sum += float(integral.detach().sum())
        self.survival_integral += float(integral.detach().sum())
        self.observed_seconds += float(seconds)

    def means(self) -> dict[str, float]:
        timing = self.timing_sum / max(self.timing_events, 1)
        size = self.size_sum / max(self.size_steps, 1)
        subset = self.subset_sum / max(self.select_steps, 1)
        return {
            "timing_nll_per_event": timing,
            "size_nll_per_step": size,
            "subset_nll_per_group": subset,
            "total": timing + size + subset,
            "survival_integral": self.survival_integral,
            "observed_seconds": self.observed_seconds,
            "n_events": self.timing_events,
            "n_size_steps": self.size_steps,
            "n_select_steps": self.select_steps,
        }


def _segment_event_indices(timeline: SubjectTimeline) -> Iterable[tuple[int, np.ndarray]]:
    for segment_index in range(len(timeline.segments)):
        pos = np.flatnonzero(timeline.event_segment == segment_index)
        if pos.size:
            yield segment_index, pos


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
) -> tuple[Tensor, float, Tensor, dict[str, Tensor] | None]:
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
    start_intensity = model.state.intensity(state_post)
    last_t = float(previous_time)
    for step in range(event_positions.size):
        t = float(times[step])
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
        mark, _ = model.grammar(ids, count, state)
        scored = {
            "timing_nll": torch.cat(timing_nll),
            "timing_integral": torch.cat(timing_integral),
            "timing_seconds": torch.cat(timing_seconds),
            "state_pre": state,
            "mark_size_step": mark.group_size_step_log_prob,
            "mark_subset_step": mark.subset_step_log_prob,
            "mark_active": mark.active_step,
            "mark_select": mark.select_step,
        }
    return state_post, last_t, start_intensity, scored


def run_pass(
    model: PilotModel,
    seq: SubjectSequence,
    timeline: SubjectTimeline,
    split_name: str,
    *,
    device: torch.device,
    cfg: PilotConfig,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, Any]:
    """Causal pass; held-out splits replay prior events but score only themselves."""

    train = optimizer is not None
    model.train(train)
    split_index = {"train": 0, "val": 1, "test": 2}[split_name]
    lower = -math.inf if split_index == 0 else float(timeline.split.boundary_epochs[split_index - 1])
    upper = (
        float(timeline.split.boundary_epochs[split_index])
        if split_index < 2
        else math.inf
    )
    accum = _LossAccumulator()
    gradient_norms: list[float] = []
    state_norms: list[float] = []
    for segment_index, all_positions in _segment_event_indices(timeline):
        segment = timeline.segments[segment_index]
        score_lo = max(float(segment.start_epoch), lower)
        score_hi = min(float(segment.stop_epoch), upper)
        if score_hi <= score_lo:
            continue
        # Replay only what is needed for this split, always from the causal
        # segment start.  No state is carried across seizure/gap boundaries.
        use = all_positions[timeline.event_times[all_positions] < score_hi]
        state_post = model.state.initial(1, device)
        previous_time = float(segment.start_epoch)
        for chunk_lo in range(0, use.size, cfg.chunk_events):
            pos = use[chunk_lo : chunk_lo + cfg.chunk_events]
            if train:
                state_post, previous_time, _lam, scored = _chunk_forward(
                    model, seq, timeline, pos, state_post, previous_time,
                    score_lo=score_lo, score_hi=score_hi, device=device, amp=cfg.amp,
                )
            else:
                with torch.no_grad():
                    state_post, previous_time, _lam, scored = _chunk_forward(
                        model, seq, timeline, pos, state_post, previous_time,
                        score_lo=score_lo, score_hi=score_hi, device=device, amp=cfg.amp,
                    )
            if scored is not None:
                timing = scored["timing_nll"].mean()
                size = -scored["mark_size_step"].sum() / scored["mark_active"].sum().clamp_min(1)
                subset = -scored["mark_subset_step"].sum() / scored["mark_select"].sum().clamp_min(1)
                loss = (
                    cfg.timing_weight * timing
                    + cfg.size_weight * size
                    + cfg.subset_weight * subset
                )
                terms_proxy = type("T", (), {
                    "event_nll": scored["timing_nll"],
                    "survival_integral": scored["timing_integral"],
                    "observed_seconds": scored["timing_seconds"],
                })
                accum.add_timing(terms_proxy)
                mark_proxy = type("M", (), {
                    "group_size_step_log_prob": scored["mark_size_step"],
                    "subset_step_log_prob": scored["mark_subset_step"],
                    "active_step": scored["mark_active"],
                    "select_step": scored["mark_select"],
                })
                accum.add_mark(mark_proxy)
                state_norms.append(float(scored["state_pre"].detach().norm(dim=-1).mean()))
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
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
                optimizer.zero_grad(set_to_none=True)
                (tail / max(n_scored_segment, 1)).backward()
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
            accum.add_tail(tail, score_hi - tail_start)
    out = accum.means()
    out.update({
        "split": split_name,
        "gradient_norm_median": float(np.median(gradient_norms)) if gradient_norms else None,
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
    grammar_checkpoint = Path(grammar_dir) / "grammar_v03.pt"
    if not grammar_checkpoint.exists():
        raise FileNotFoundError(f"calibrated grammar missing: {grammar_checkpoint}")
    model = build_model(subject, seq, grammar_checkpoint, seed=seed, device=device)
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    trainable = [p for p in model.parameters() if p.requires_grad]
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
            model, seq, timeline, "train", device=device, cfg=cfg, optimizer=optimizer
        )
        if (epoch + 1) % cfg.validation_every == 0:
            validation = run_pass(
                model, seq, timeline, "val", device=device, cfg=cfg, optimizer=None
            )
        else:
            validation = {"total": math.nan}
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation})
        value = float(validation["total"])
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
    test = run_pass(model, seq, timeline, "test", device=device, cfg=cfg, optimizer=None)
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
        "test": test,
        "parameter_updates": _trainable_update_report(before, model),
        "checkpoint": str(checkpoint_out),
        "checkpoint_sha256": _sha256(checkpoint_out),
        "elapsed_seconds": time.time() - started,
        "split_boundary_epochs": timeline.split.boundary_epochs.tolist(),
        "recorded_seconds": timeline.split.recorded_seconds,
        "n_events": {
            name: int(seq_positions_for_split(timeline, name).size)
            for name in ("train", "val", "test")
        },
        "sealed_partition_opened": False,
        "source_commit": SOURCE_COMMIT,
    }
    _atomic_json(report_out, report)
    _atomic_json(out_dir / "progress.json", {**report, "status": "complete"})
    return report
