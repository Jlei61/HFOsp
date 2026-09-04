"""Full-event content state trained through the frozen contact decoder.

One recurrent step is one complete group IED.  The encoder consumes the stored
three-view waveform, band envelopes/summaries, cross-band lags, continuous
contact delays, tied groups, participation and geometry.  State after event e
is trained to predict contact sequences at e+1, e+5 and e+20; it is never given
the target event before that target is scored.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from src.topic5_group_event_state.dataset import SubjectSequence
from src.topic5_group_event_state.model import EncoderConfig, EventEncoder
from src.topic5_group_event_state.train import _data_shape, _load_geometry
from src.topic5_group_event_state.v03.pilot import estimate_input_stats_positions
from src.topic5_group_event_state.v03.state import FixedTimescaleEventState, StateConfig
from src.topic5_group_event_state.v034_spatial_state.we_decoder import (
    FrozenDecoderBundle, align_events, decoder_tensors,
)

from .contracts import (
    DATASET_ROOT, FORMAT_PREFIX, INPUT_ROOT, RATE_TAUS_SECONDS, atomic_json, seed_all,
)
from .dynamic_rate import negative_binomial_nll
from .long_windows import matched_wrong_time_donors
from .stepwise_decoder import DynamicStepAdapter, StepwiseAdapterConfig, StepwiseConditionedDecoder


OFFSETS = (1, 5, 20)
INPUT_VIEWS = (
    "full_mark",
    "times_only",
    "spatial_only",
    "waveform_only",
    "multiband_only",
    "mark_shuffle",
)


@dataclass(frozen=True)
class FullMarkTrainConfig:
    # ``joint`` preserves the completed v0.3.5 diagnostic.  The shared-state
    # stage trains two genuinely separate producers: S_N receives only burden
    # likelihoods, while S_G receives only conditional grammar/morphology
    # likelihoods.  Keeping the switch inside the same causal replay path
    # prevents a hidden difference in event alignment or warm-up.
    objective_family: str = "joint"
    max_epochs: int = 16
    min_epochs: int = 4
    patience: int = 4
    chunk_events: int = 256
    chunk_seconds: float = 1800.0
    encoder_lr: float = 3e-5
    state_lr: float = 3e-4
    adapter_lr: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    m_adapter_rank: int = 8
    encoder_d_contact: int = 32
    encoder_event_dim: int = 64
    encoder_attention_layers: int = 2
    encoder_waveform_channels: int = 16
    encoder_dropout: float = 0.1
    state_channels_per_tau: int = 4
    # q(t) and m(t) share one registered physical-time coordinate system.
    # Keeping this explicit prevents the mark state from silently inheriting
    # the older v0.3 four-scale bank.
    state_taus_seconds: tuple[float, ...] = RATE_TAUS_SECONDS
    state_update_hidden: int = 64
    state_update_fraction_cap: float = 0.2
    offset_weights: tuple[float, ...] = (1.0, 0.5, 0.25)
    event_offsets: tuple[int, ...] = OFFSETS
    physical_loss_weight: float = 0.5
    physical_baseline_steps: int = 900
    physical_baseline_lr: float = 3e-3
    # Long physical-horizon q(t) has a different feature width from the frozen
    # core stepwise adapter.  In that family the mature decoder keeps its
    # frozen static recalibration and only the new state adapter is trained.
    decoder_use_dynamic_q: bool = True
    amp: bool = True
    report_selection: bool = True
    seed: int = 20260903
    # Source-attribution views share the same targets, state bank and optimiser
    # budget.  They differ only in which part of each already-observed group
    # event is allowed to update the cross-event state.
    input_view: str = "full_mark"

    def validate(self) -> "FullMarkTrainConfig":
        if self.objective_family not in {"joint", "sn", "sg"}:
            raise ValueError("objective_family must be joint, sn or sg")
        if len(self.offset_weights) != len(self.event_offsets):
            raise ValueError("offset_weights and event_offsets must have equal length")
        return self


@dataclass
class FullMarkData:
    subject: str
    seq: SubjectSequence
    event_time: np.ndarray
    event_segment: np.ndarray
    phase: np.ndarray
    source_position: np.ndarray
    input_source_position: np.ndarray
    input_view: str
    input_view_details: dict[str, Any]
    q_context: np.ndarray
    decoder_index: np.ndarray
    next_index: np.ndarray
    event_offsets: tuple[int, ...]
    grid_time: np.ndarray
    grid_segment: np.ndarray
    grid_phase: np.ndarray
    grid_q: np.ndarray
    grid_source_event: np.ndarray
    grid_source_dt: np.ndarray
    future_count: np.ndarray
    future_count_log_offset: np.ndarray
    future_valid: np.ndarray
    future_seizure_count: np.ndarray
    future_participation: np.ndarray
    future_participation_valid: np.ndarray
    future_extent: np.ndarray
    physical_horizons_seconds: tuple[float, ...]
    provenance: dict[str, Any]


def _event_phase(time_: np.ndarray, bounds: dict[str, float]) -> np.ndarray:
    out = np.full(time_.size, "OUTSIDE", dtype="<U12")
    out[time_ < bounds["20pct"]] = "CALIBRATION"
    out[(time_ >= bounds["20pct"]) & (time_ < bounds["60pct"])] = "FIT"
    out[(time_ >= bounds["60pct"]) & (time_ < bounds["70pct"])] = "INNER"
    out[(time_ >= bounds["70pct"]) & (time_ < bounds["80pct"])] = "SELECTION"
    return out


def _map_q_at_events(event_time: np.ndarray, event_segment: np.ndarray, trajectory: Path) -> np.ndarray:
    with np.load(trajectory, allow_pickle=False) as z:
        at = np.asarray(z["anchor_time"], dtype=np.float64)
        aseg = np.asarray(z["segment"], dtype=np.int64)
        q = np.asarray(z["q_standardized"], dtype=np.float32)
    out = np.zeros((event_time.size, q.shape[1]), dtype=np.float32)
    for seg in np.unique(event_segment):
        er = np.flatnonzero(event_segment == seg)
        ar = np.flatnonzero(aseg == seg)
        if ar.size == 0:
            continue
        order = ar[np.argsort(at[ar], kind="stable")]
        pos = np.searchsorted(at[order], event_time[er], side="right") - 1
        ok = pos >= 0
        out[er[ok]] = q[order[pos[ok]]]
    return out


def load_full_mark_data(
    subject: str,
    bundle: FrozenDecoderBundle,
    rate_trajectory: Path,
    event_offsets: tuple[int, ...] = OFFSETS,
) -> FullMarkData:
    manifest_path = INPUT_ROOT / subject / "manifest_v3.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sealed") is not False or manifest.get("development_evaluation_used_for_fitting") is not False:
        raise PermissionError("unsafe v0.3.5 full-mark manifest")
    with np.load(manifest["input_path"], allow_pickle=False) as z:
        event_time = np.asarray(z["event_time"], dtype=np.float64)
        carry = np.asarray(z["event_carry"], dtype=np.int64)
        segment_bounds = np.asarray(z["target_segment_bounds"], dtype=np.float64)
    bounds = {k: float(v) for k, v in manifest["report"]["phase_boundaries_epoch"].items()}
    with np.load(rate_trajectory, allow_pickle=False) as rate:
        grid_time = np.asarray(rate["anchor_time"], dtype=np.float64)
        grid_segment = np.asarray(rate["segment"], dtype=np.int64)
        grid_phase = np.asarray(rate["phase"]).astype(str)
        grid_q = np.asarray(rate["q_standardized"], dtype=np.float32)
        future_count = np.asarray(rate["target_count"], dtype=np.float32)
        future_valid = np.asarray(rate["target_valid"], dtype=bool)
        future_seizure_count = (
            np.asarray(rate["target_seizure_count"], dtype=np.int16)
            if "target_seizure_count" in rate.files else
            np.zeros_like(future_count, dtype=np.int16)
        )
        horizons = tuple(
            float(v) for v in (
                np.asarray(rate["horizons_seconds"], dtype=np.float64)
                if "horizons_seconds" in rate.files else np.asarray((300.0, 1800.0, 7200.0))
            )
        )
        future_exposure = (
            np.asarray(rate["target_exposure_seconds"], dtype=np.float32)
            if "target_exposure_seconds" in rate.files else
            np.broadcast_to(np.asarray(horizons, dtype=np.float32), future_count.shape).copy()
        )
        window_contract = (
            str(np.asarray(rate["window_contract"]).item())
            if "window_contract" in rate.files else "same_segment_complete"
        )
        if "segment_bounds" in rate.files:
            segment_bounds = np.asarray(rate["segment_bounds"], dtype=np.float64)
        if "phase_boundaries_json" in rate.files:
            bounds = {
                k: float(v) for k, v in
                json.loads(str(np.asarray(rate["phase_boundaries_json"]).item())).items()
            }
    keep = event_time < bounds["80pct"]
    event_time = event_time[keep]
    carry = carry[keep]
    segment = np.full(event_time.size, -1, dtype=np.int64)
    for seg, (lo, hi) in enumerate(segment_bounds):
        segment[(event_time >= lo) & (event_time < hi)] = seg
    covered = segment >= 0
    event_time, segment, carry = event_time[covered], segment[covered], carry[covered]
    phase = _event_phase(event_time, bounds)
    seq = SubjectSequence(DATASET_ROOT / subject)
    source = np.searchsorted(seq.t_abs, event_time)
    if np.any(source >= len(seq)) or not np.array_equal(seq.t_abs[source], event_time):
        raise ValueError("full-mark prefix does not map exactly to the v0.1 event stream")
    q_context = _map_q_at_events(event_time, segment, rate_trajectory)
    n_horizon = future_count.shape[1]
    # Fixed-grid morphology uses the complete registered event contact axis,
    # not only the subset retained by the mature sequence decoder.  The latter
    # remains responsible for its own aligned contact grammar.
    n_contact = int(seq.arrays["participation"].shape[1])
    future_participation = np.zeros((grid_time.size, n_horizon, n_contact), dtype=np.float32)
    future_participation_valid = np.zeros_like(future_participation, dtype=bool)
    future_extent = np.zeros((grid_time.size, n_horizon), dtype=np.float32)
    grid_source_event = np.full(grid_time.size, -1, dtype=np.int64)
    grid_source_dt = np.zeros(grid_time.size, dtype=np.float32)
    raw_rows = seq.order[source]
    participation = np.asarray(seq.arrays["participation"][raw_rows], dtype=np.float32)
    if len(horizons) != n_horizon:
        raise ValueError("rate trajectory horizon count differs from v0.3.5 physical contract")
    for seg in np.unique(grid_segment):
        ar = np.flatnonzero(grid_segment == seg)
        er = np.flatnonzero(segment == seg)
        if er.size == 0:
            continue
        et = event_time[er]
        before = np.searchsorted(et, grid_time[ar], side="left") - 1
        have = before >= 0
        donor = er[np.maximum(before, 0)]
        grid_source_event[ar[have]] = donor[have]
        grid_source_dt[ar[have]] = (grid_time[ar[have]] - event_time[donor[have]]).astype(np.float32)
    if window_contract == "observed_support":
        order = np.argsort(event_time, kind="stable")
        et = event_time[order]
        part = participation[order]
        prefix = np.concatenate((np.zeros((1, n_contact), dtype=np.float64),
                                 np.cumsum(part, axis=0, dtype=np.float64)), axis=0)
        for j, horizon in enumerate(horizons):
            left = np.searchsorted(et, grid_time, side="left")
            right = np.searchsorted(et, grid_time + horizon, side="left")
            count = right - left
            valid_rows = future_valid[:, j] & (count > 0)
            if not np.any(valid_rows):
                continue
            summed = prefix[right[valid_rows]] - prefix[left[valid_rows]]
            mean_part = summed / count[valid_rows, None]
            future_participation[valid_rows, j] = mean_part.astype(np.float32)
            future_participation_valid[valid_rows, j] = np.asarray(seq.contact_valid, dtype=bool)[None]
            future_extent[valid_rows, j] = mean_part.mean(axis=1).astype(np.float32)
    else:
        for seg in np.unique(grid_segment):
            ar = np.flatnonzero(grid_segment == seg)
            er = np.flatnonzero(segment == seg)
            if er.size == 0:
                continue
            et = event_time[er]
            prefix = np.concatenate((np.zeros((1, n_contact), dtype=np.float64),
                                     np.cumsum(participation[er], axis=0, dtype=np.float64)), axis=0)
            for j, horizon in enumerate(horizons):
                left = np.searchsorted(et, grid_time[ar], side="left")
                right = np.searchsorted(et, grid_time[ar] + horizon, side="left")
                count = right - left
                valid_rows = future_valid[ar, j] & (count > 0)
                if not np.any(valid_rows):
                    continue
                summed = prefix[right[valid_rows]] - prefix[left[valid_rows]]
                mean_part = summed / count[valid_rows, None]
                target_rows = ar[valid_rows]
                future_participation[target_rows, j] = mean_part.astype(np.float32)
                future_participation_valid[target_rows, j] = np.asarray(seq.contact_valid, dtype=bool)[None]
                future_extent[target_rows, j] = mean_part.mean(axis=1).astype(np.float32)
    decoder_index = align_events(event_time, bundle.event_abs_time)
    event_offsets = tuple(int(value) for value in event_offsets)
    if not event_offsets or any(value <= 0 for value in event_offsets):
        raise ValueError("event offsets must be positive")
    next_index = np.full((event_time.size, len(event_offsets)), -1, dtype=np.int64)
    for seg in np.unique(segment):
        rows = np.flatnonzero(segment == seg)
        rows = rows[np.argsort(event_time[rows], kind="stable")]
        for j, offset in enumerate(event_offsets):
            if rows.size > offset:
                anchor, target = rows[:-offset], rows[offset:]
                same_phase = phase[anchor] == phase[target]
                next_index[anchor[same_phase], j] = target[same_phase]
    return FullMarkData(
        subject=subject, seq=seq, event_time=event_time, event_segment=segment, phase=phase,
        source_position=source.astype(np.int64),
        input_source_position=source.astype(np.int64).copy(),
        input_view="full_mark",
        input_view_details={"mark_alignment": "observed event content at its true event time"},
        q_context=q_context,
        decoder_index=decoder_index, next_index=next_index, event_offsets=event_offsets,
        grid_time=grid_time, grid_segment=grid_segment, grid_phase=grid_phase,
        grid_q=grid_q, grid_source_event=grid_source_event, grid_source_dt=grid_source_dt,
        future_count=future_count, future_valid=future_valid,
        future_seizure_count=future_seizure_count,
        future_count_log_offset=np.log(
            np.maximum(future_exposure, 1e-6)
            / np.asarray(horizons, dtype=np.float32)[None]
        ).astype(np.float32),
        future_participation=future_participation,
        future_participation_valid=future_participation_valid,
        future_extent=future_extent, physical_horizons_seconds=horizons,
        provenance={
            "manifest": str(manifest_path), "rate_trajectory": str(rate_trajectory),
            "full_modalities": ["participation", "tied_group", "continuous_delay", "three_view_waveform",
                                "band_envelope", "band_features", "cross_band_lag", "geometry"],
            "target_offsets": list(event_offsets), "target_same_coverage_segment": True,
            "physical_horizons_seconds": list(horizons),
            "physical_window_contract": window_contract,
            "phase_boundaries_epoch": bounds,
            "physical_anchor_weighting": "one fixed five-minute grid anchor; count and conditional morphology scored separately",
            "shared_q_m_timescale_bank_seconds": list(RATE_TAUS_SECONDS),
            "target_same_phase": True, "development_targets_read": False,
            "sealed_partition_opened": False, "seizure_outcomes_read": False,
        },
    )


def configure_event_input_view(data: FullMarkData, config: FullMarkTrainConfig) -> FullMarkData:
    """Bind one causal event-input view without modifying any prediction target.

    ``mark_shuffle`` moves the complete mark payload as one unit within the same
    coverage segment and phase.  Event timestamps, q(t), targets and marginal
    mark distributions are unchanged.  A non-zero circular shift is used when
    possible, so the null does not quietly retain fixed points.
    """

    view = str(config.input_view)
    if view not in INPUT_VIEWS:
        raise ValueError(f"unknown full-event input view: {view}")
    source = np.asarray(data.source_position, dtype=np.int64).copy()
    details: dict[str, Any] = {
        "input_view": view,
        "event_times_preserved": True,
        "prediction_targets_unchanged": True,
    }
    if view == "mark_shuffle":
        rng = np.random.default_rng(int(config.seed))
        mapped = source.copy()
        moved = 0
        singleton = 0
        for seg in np.unique(data.event_segment):
            for phase in np.unique(data.phase[data.event_segment == seg]):
                rows = np.flatnonzero((data.event_segment == seg) & (data.phase == phase))
                if rows.size < 2:
                    singleton += int(rows.size)
                    continue
                shift = int(rng.integers(1, rows.size))
                donor = np.roll(rows, shift)
                mapped[rows] = source[donor]
                moved += int(np.sum(mapped[rows] != source[rows]))
        source = mapped
        details.update({
            "shuffle_scope": "within coverage-segment and FIT/INNER/SELECTION phase",
            "shuffle_seed": int(config.seed),
            "n_events_moved": moved,
            "n_singleton_events_unmoved": singleton,
            "marginal_mark_payload_preserved": bool(np.array_equal(np.sort(source), np.sort(data.source_position))),
        })
        if not details["marginal_mark_payload_preserved"]:
            raise ValueError("mark shuffle changed the source-event payload multiset")
    return replace(data, input_source_position=source, input_view=view, input_view_details=details)


class PhysicalFutureHead(nn.Module):
    """Nested q-only and q-plus-mark-state predictions on fixed-time anchors."""

    def __init__(self, q_dim: int, state_dim: int, n_horizons: int, n_contacts: int) -> None:
        super().__init__()
        self.n_horizons = int(n_horizons)
        self.n_contacts = int(n_contacts)
        self.q_count = nn.Linear(q_dim, n_horizons)
        self.q_extent = nn.Linear(q_dim, n_horizons)
        self.q_participation = nn.Linear(q_dim, n_horizons * n_contacts)
        self.state_count = nn.Linear(state_dim, n_horizons, bias=False)
        self.state_extent = nn.Linear(state_dim, n_horizons, bias=False)
        self.state_participation = nn.Linear(state_dim, n_horizons * n_contacts, bias=False)
        self.log_dispersion = nn.Parameter(torch.zeros(n_horizons))
        # The q-only head is a shared nested baseline, not part of the recipe
        # capacity search.  Explicit zero initialisation makes its full-batch
        # optimisation identical across encoder widths/depths for a given
        # patient, rather than inheriting a different RNG offset from the
        # upstream event encoder construction.
        for module in (self.q_count, self.q_extent, self.q_participation):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
        for module in (self.state_count, self.state_extent, self.state_participation):
            nn.init.zeros_(module.weight)

    def predictions(
        self, q: Tensor, state: Tensor | None,
        count_log_offset: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        count = self.q_count(q)
        extent = self.q_extent(q)
        participation = self.q_participation(q).reshape(q.shape[0], self.n_horizons, self.n_contacts)
        if state is not None:
            count = count + self.state_count(state)
            extent = extent + self.state_extent(state)
            participation = participation + self.state_participation(state).reshape(
                q.shape[0], self.n_horizons, self.n_contacts
            )
        if count_log_offset is not None:
            count = count + count_log_offset
        return count, extent, participation

    def q_parameters(self) -> list[nn.Parameter]:
        return [*self.q_count.parameters(), *self.q_extent.parameters(),
                *self.q_participation.parameters(), self.log_dispersion]

    def state_parameters(self) -> list[nn.Parameter]:
        return [*self.state_count.parameters(), *self.state_extent.parameters(),
                *self.state_participation.parameters()]


class FullMarkStateModel(nn.Module):
    def __init__(self, data: FullMarkData, bundle: FrozenDecoderBundle, base_adapter: Path,
                 config: FullMarkTrainConfig, device: torch.device) -> None:
        super().__init__()
        self.config = config
        calibration = data.source_position[data.phase == "CALIBRATION"]
        stats = estimate_input_stats_positions(data.seq, calibration, seed=0)
        view = str(config.input_view)
        if view not in INPUT_VIEWS:
            raise ValueError(f"unknown full-event input view: {view}")
        structural = view in {"full_mark", "spatial_only", "mark_shuffle"}
        enc_cfg = EncoderConfig(
            use_participation=structural, use_exact_delay=structural, use_tied_groups=structural,
            use_legacy_rank=False,
            use_waveform=view in {"full_mark", "waveform_only", "mark_shuffle"},
            use_multiband=view in {"full_mark", "multiband_only", "mark_shuffle"},
            use_geometry=view != "times_only",
            d_contact=config.encoder_d_contact, d_event=config.encoder_event_dim,
            n_attention_heads=4, n_attention_layers=config.encoder_attention_layers,
            waveform_channels=config.encoder_waveform_channels, dropout=config.encoder_dropout,
        )
        geometry = _load_geometry(data.seq)
        self.event_encoder = EventEncoder(enc_cfg, _data_shape(data.seq),
                                          geometry.to(device) if geometry is not None else None, stats)
        self.input_view = view
        # In the times-only arm every observed event writes the same learned
        # token.  Exact inter-event dt and the number of writes remain visible,
        # while participation, delay, waveform and frequency content do not.
        self.timing_token = nn.Parameter(torch.zeros(config.encoder_event_dim))
        self.state = FixedTimescaleEventState(StateConfig(
            event_dim=config.encoder_event_dim,
            taus_seconds=tuple(float(v) for v in config.state_taus_seconds),
            channels_per_tau=config.state_channels_per_tau,
            update_hidden=config.state_update_hidden,
            update_fraction_cap=config.state_update_fraction_cap,
        ))
        self.expected_from_q = nn.Sequential(nn.LayerNorm(data.q_context.shape[1]),
                                             nn.Linear(data.q_context.shape[1], config.encoder_event_dim), nn.GELU(),
                                             nn.Linear(config.encoder_event_dim, config.encoder_event_dim))
        self.decoder = StepwiseConditionedDecoder(
            bundle.model, StepwiseAdapterConfig(context_dim=data.q_context.shape[1], rank=8),
        )
        saved = torch.load(base_adapter, map_location="cpu", weights_only=False)["adapter"]
        current = self.decoder.state_dict()
        mismatched = []
        for key, value in saved.items():
            if key in current and current[key].shape == value.shape:
                current[key] = value
            elif key in current:
                mismatched.append(key)
        allowed_mismatch = {"dynamic.down.weight"} if not config.decoder_use_dynamic_q else set()
        if set(mismatched) - allowed_mismatch:
            raise ValueError(f"frozen stepwise adapter shape mismatch: {mismatched}")
        self.decoder.load_state_dict(current)
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(False)
        self.m_adapter = DynamicStepAdapter(
            StepwiseAdapterConfig(context_dim=self.state.cfg.state_dim, rank=config.m_adapter_rank),
            bundle.model.n_nodes * bundle.model.state_dim, bundle.model.n_contacts,
        )
        self.physical_head = PhysicalFutureHead(
            data.q_context.shape[1], self.state.cfg.state_dim,
            data.future_count.shape[1], data.future_participation.shape[2],
        )

    def event_update(self, state_pre: Tensor, event_batch: dict[str, Tensor], q: Tensor) -> Tensor:
        embedding = self.encode_events(event_batch)
        innovation = embedding.float() - self.expected_from_q(q.float())
        return self.state.update(state_pre, innovation)

    def encode_events(self, event_batch: dict[str, Tensor]) -> Tensor:
        if self.input_view == "times_only":
            n = int(event_batch["participation"].shape[0])
            return self.timing_token.unsqueeze(0).expand(n, -1)
        embedding, _ = self.event_encoder(event_batch)
        return embedding


def _to_device(raw: dict[str, np.ndarray], device: torch.device) -> dict[str, Tensor]:
    out = {}
    for key, value in raw.items():
        a = np.ascontiguousarray(value)
        if a.dtype == np.bool_: out[key] = torch.from_numpy(a).to(device)
        elif np.issubdtype(a.dtype, np.integer): out[key] = torch.from_numpy(a.astype(np.int64)).to(device)
        else: out[key] = torch.from_numpy(a.astype(np.float32)).to(device)
    return out


def _chunks(rows: np.ndarray, time_: np.ndarray, max_events: int, max_seconds: float):
    start = 0
    while start < rows.size:
        stop = min(rows.size, start + max_events)
        by_time = np.searchsorted(time_[rows], time_[rows[start]] + max_seconds, side="right")
        stop = min(stop, max(start + 1, int(by_time)))
        yield rows[start:stop]
        start = stop


def _target_scores(model: FullMarkStateModel, bundle_tensors: dict[str, Tensor], data: FullMarkData,
                   anchors: np.ndarray, states: Tensor, q: Tensor, offset_j: int) -> dict[str, Tensor] | None:
    target = data.next_index[anchors, offset_j]
    good = (target >= 0) & (data.decoder_index[np.maximum(target, 0)] >= 0)
    if not good.any():
        return None
    local = torch.as_tensor(np.flatnonzero(good), dtype=torch.long, device=states.device)
    # The anchor event has been observed, but no intervening future event may
    # update the state.  Physical time nevertheless passes before the target
    # event, so perform one closed-form open-loop evolution over the exact dt.
    dt = torch.as_tensor(
        data.event_time[target[good]] - data.event_time[anchors[good]],
        dtype=torch.float32,
        device=states.device,
    )
    target_state = model.state.evolve(states[local], dt)
    cache = torch.as_tensor(data.decoder_index[target[good]], dtype=torch.long, device=states.device)
    batch = {key: value[cache] for key, value in bundle_tensors.items()}
    score = model.decoder.scores(
        batch, q[local] if model.config.decoder_use_dynamic_q else None,
        use_static=True, use_dynamic=model.config.decoder_use_dynamic_q,
        extra_context=target_state, extra_adapter=model.m_adapter,
    )
    return score


def _physical_loss(
    head: PhysicalFutureHead,
    q: Tensor,
    state: Tensor | None,
    count_log_offset: Tensor,
    count: Tensor,
    valid: Tensor,
    participation: Tensor,
    participation_valid: Tensor,
    extent: Tensor,
    *,
    endpoint_family: str = "joint",
) -> tuple[Tensor | None, dict[str, float | int | None]]:
    """Count and conditional morphology are separate, equally visible terms."""

    log_mu, extent_logit, participation_logit = head.predictions(q, state, count_log_offset)
    terms: list[Tensor] = []
    metrics: dict[str, float | int | None] = {}
    include_count = endpoint_family in {"joint", "sn"}
    include_morphology = endpoint_family in {"joint", "sg"}
    if include_count and bool(valid.any()):
        count_loss = negative_binomial_nll(count, log_mu, head.log_dispersion)
        by_horizon = [
            count_loss[:, j][valid[:, j]].mean()
            for j in range(valid.shape[1])
            if bool(valid[:, j].any())
        ]
        current = torch.stack(by_horizon).mean()
        terms.append(current)
        metrics["count_nll"] = float(current.detach())
        metrics["n_count"] = int(valid.sum().detach())
    else:
        metrics.update({"count_nll": None, "n_count": 0})
    morphology_valid = valid & (count > 0)
    if include_morphology and bool(morphology_valid.any()):
        by_horizon = [
            torch.nn.functional.binary_cross_entropy_with_logits(
                extent_logit[:, j][morphology_valid[:, j]],
                extent[:, j][morphology_valid[:, j]],
                reduction="mean",
            )
            for j in range(morphology_valid.shape[1])
            if bool(morphology_valid[:, j].any())
        ]
        current = torch.stack(by_horizon).mean()
        terms.append(current)
        metrics["extent_bce"] = float(current.detach())
        metrics["n_extent"] = int(morphology_valid.sum().detach())
    else:
        metrics.update({"extent_bce": None, "n_extent": 0})
    if include_morphology and bool(participation_valid.any()):
        by_horizon = [
            torch.nn.functional.binary_cross_entropy_with_logits(
                participation_logit[:, j][participation_valid[:, j]],
                participation[:, j][participation_valid[:, j]],
                reduction="mean",
            )
            for j in range(participation_valid.shape[1])
            if bool(participation_valid[:, j].any())
        ]
        current = torch.stack(by_horizon).mean()
        terms.append(current)
        metrics["participation_bce"] = float(current.detach())
        metrics["n_participation"] = int(participation_valid.sum().detach())
    else:
        metrics.update({"participation_bce": None, "n_participation": 0})
    return (torch.stack(terms).mean() if terms else None), metrics


def _physical_tensors(data: FullMarkData, rows: np.ndarray, device: torch.device) -> tuple[Tensor, ...]:
    return (
        torch.as_tensor(data.grid_q[rows], dtype=torch.float32, device=device),
        torch.as_tensor(data.future_count_log_offset[rows], dtype=torch.float32, device=device),
        torch.as_tensor(data.future_count[rows], dtype=torch.float32, device=device),
        torch.as_tensor(data.future_valid[rows], dtype=torch.bool, device=device),
        torch.as_tensor(data.future_participation[rows], dtype=torch.float32, device=device),
        torch.as_tensor(data.future_participation_valid[rows], dtype=torch.bool, device=device),
        torch.as_tensor(data.future_extent[rows], dtype=torch.float32, device=device),
    )


def fit_physical_q_baseline(
    model: FullMarkStateModel,
    data: FullMarkData,
    config: FullMarkTrainConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Fit/freeze the q-only fixed-grid head before mark-state training."""

    fit_rows = np.flatnonzero(data.grid_phase == "FIT")
    inner_rows = np.flatnonzero(data.grid_phase == "INNER")
    if min(fit_rows.size, inner_rows.size) == 0:
        raise ValueError("no fixed-grid FIT/INNER anchors for multi-horizon producer")
    fit_tensors = _physical_tensors(data, fit_rows, device)
    inner_tensors = _physical_tensors(data, inner_rows, device)
    parameters = model.physical_head.q_parameters()
    optimizer = torch.optim.AdamW(parameters, lr=config.physical_baseline_lr,
                                  weight_decay=config.weight_decay)
    best, best_step, stale = math.inf, 0, 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.physical_head.state_dict().items()}
    history = []
    for step in range(config.physical_baseline_steps + 1):
        if step:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _physical_loss(
                model.physical_head, fit_tensors[0], None, *fit_tensors[1:],
                endpoint_family=config.objective_family,
            )
            if loss is None:
                raise ValueError("no fixed-grid fitting target")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
            optimizer.step()
        if step % 25 == 0 or step == config.physical_baseline_steps:
            with torch.no_grad():
                inner_loss, metrics = _physical_loss(
                    model.physical_head, inner_tensors[0], None, *inner_tensors[1:],
                    endpoint_family=config.objective_family,
                )
            value = float(inner_loss.detach()) if inner_loss is not None else math.inf
            history.append({"step": step, "inner_loss": value, **metrics})
            if np.isfinite(value) and value < best - 1e-6:
                best, best_step, stale = value, step, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.physical_head.state_dict().items()}
            else:
                stale += 1
            if stale >= 8:
                break
    model.physical_head.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    for parameter in model.physical_head.q_parameters():
        parameter.requires_grad_(False)
    return {"selected_step": best_step, "steps_run": history[-1]["step"],
            "best_inner_loss": best, "history": history,
            "selection_targets_read": False}


def run_phase(model: FullMarkStateModel, data: FullMarkData, bundle_tensors: dict[str, Tensor],
              phase: str, config: FullMarkTrainConfig, device: torch.device,
              optimizer: torch.optim.Optimizer | None) -> dict[str, Any]:
    train = optimizer is not None
    model.train(train); model.decoder.decoder.eval()
    losses, grammar_losses, physical_losses, grad_norms = [], [], [], []
    n_by_offset = np.zeros(len(data.event_offsets), dtype=int)
    for seg in np.unique(data.event_segment):
        rows = np.flatnonzero(data.event_segment == seg)
        if not np.any(data.phase[rows] == phase):
            continue
        state = model.state.initial(1, device)
        previous = float(data.event_time[rows[0]])
        for chunk in _chunks(rows, data.event_time, config.chunk_events, config.chunk_seconds):
            raw = data.seq.gather_positions(data.input_source_position[chunk])
            event_batch = _to_device(raw, device)
            q_all = torch.as_tensor(data.q_context[chunk], dtype=torch.float32, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=config.amp and device.type == "cuda"):
                embeddings = model.encode_events(event_batch)
            post_states, scored_states, scored_rows, scored_q = [], [], [], []
            for local, row in enumerate(chunk):
                dt = torch.tensor([max(0.0, float(data.event_time[row]) - previous)], device=device)
                state_pre = model.state.evolve(state, dt)
                innovation = embeddings[local:local + 1].float() - model.expected_from_q(q_all[local:local + 1])
                state = model.state.update(state_pre, innovation)
                post_states.append(state)
                if data.phase[row] == phase:
                    scored_states.append(state)
                    scored_rows.append(int(row))
                    scored_q.append(q_all[local:local + 1])
                previous = float(data.event_time[row])
            objective_terms: list[Tensor] = []
            if scored_rows:
                state_batch = torch.cat(scored_states, 0)
                q_batch = torch.cat(scored_q, 0)
                anchor_np = np.asarray(scored_rows, dtype=np.int64)
                terms = []
                if config.objective_family in {"joint", "sg"}:
                    for j, weight in enumerate(config.offset_weights):
                        score = _target_scores(model, bundle_tensors, data, anchor_np, state_batch, q_batch, j)
                        if score is not None:
                            terms.append(float(weight) * score["grammar"].mean())
                            n_by_offset[j] += int(score["grammar"].numel())
                if terms:
                    grammar_loss = torch.stack(terms).sum() / max(sum(config.offset_weights), 1e-8)
                    objective_terms.append(grammar_loss)
                    grammar_losses.append(float(grammar_loss.detach()))
            # Fixed physical-time anchors are assigned to the final event that
            # precedes them.  This gives every five-minute anchor equal weight,
            # while preserving a differentiable path into the same event state.
            grid_rows = np.flatnonzero(
                (data.grid_phase == phase)
                & np.isin(data.grid_source_event, chunk)
                & (data.grid_source_event >= 0)
            ) if config.physical_loss_weight > 0 else np.asarray([], dtype=np.int64)
            if grid_rows.size and post_states:
                source_local = np.searchsorted(chunk, data.grid_source_event[grid_rows])
                if np.any(source_local >= chunk.size) or not np.array_equal(
                    chunk[source_local], data.grid_source_event[grid_rows]
                ):
                    raise ValueError("fixed-grid anchor source is not in its assigned event chunk")
                source_state = torch.cat(post_states, 0)[
                    torch.as_tensor(source_local, dtype=torch.long, device=device)
                ]
                dt_grid = torch.as_tensor(data.grid_source_dt[grid_rows], dtype=torch.float32, device=device)
                grid_state = model.state.evolve(source_state, dt_grid)
                tensors = _physical_tensors(data, grid_rows, device)
                physical_loss, _physical_metrics = _physical_loss(
                    model.physical_head, tensors[0], grid_state, *tensors[1:],
                    endpoint_family=config.objective_family,
                )
                if physical_loss is not None:
                    objective_terms.append(float(config.physical_loss_weight) * physical_loss)
                    physical_losses.append(float(physical_loss.detach()))
            if objective_terms:
                loss = torch.stack(objective_terms).sum()
                losses.append(float(loss.detach()))
                if train:
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    params = [p for p in model.parameters() if p.requires_grad]
                    norm = torch.nn.utils.clip_grad_norm_(params, config.gradient_clip)
                    if not torch.isfinite(norm): raise FloatingPointError("non-finite full-mark gradient")
                    optimizer.step(); grad_norms.append(float(norm))
            state = state.detach()
    return {"phase": phase, "mean_loss": float(np.mean(losses)) if losses else None,
            "mean_grammar_loss": float(np.mean(grammar_losses)) if grammar_losses else None,
            "mean_physical_loss": float(np.mean(physical_losses)) if physical_losses else None,
            "n_chunks_scored": len(losses), "n_target_events_by_offset": n_by_offset.tolist(),
            "n_fixed_grid_chunks_scored": len(physical_losses),
            "gradient_norm_median": float(np.median(grad_norms)) if grad_norms else None}


@torch.no_grad()
def collect_states(model: FullMarkStateModel, data: FullMarkData, phase: str, device: torch.device,
                   config: FullMarkTrainConfig) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    state_out = np.full((data.event_time.size, model.state.cfg.state_dim), np.nan, dtype=np.float32)
    for seg in np.unique(data.event_segment):
        rows = np.flatnonzero(data.event_segment == seg)
        if not np.any(data.phase[rows] == phase): continue
        state = model.state.initial(1, device); previous = float(data.event_time[rows[0]])
        for chunk in _chunks(rows, data.event_time, config.chunk_events, config.chunk_seconds):
            batch = _to_device(data.seq.gather_positions(data.input_source_position[chunk]), device)
            q = torch.as_tensor(data.q_context[chunk], dtype=torch.float32, device=device)
            embedding = model.encode_events(batch)
            for local, row in enumerate(chunk):
                dt = torch.tensor([max(0.0, float(data.event_time[row]) - previous)], device=device)
                pre = model.state.evolve(state, dt)
                state = model.state.update(pre, embedding[local:local + 1].float() - model.expected_from_q(q[local:local + 1]))
                if data.phase[row] == phase: state_out[row] = state.squeeze(0).cpu().numpy()
                previous = float(data.event_time[row])
    return np.flatnonzero(np.isfinite(state_out).all(1)), state_out


@torch.no_grad()
def collect_all_states(
    model: FullMarkStateModel,
    data: FullMarkData,
    device: torch.device,
    config: FullMarkTrainConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay the frozen model once and return pre/post state for every event.

    This is the shared scientific trajectory consumed by W4--W6.  ``pre`` is
    the state after real-time propagation but before reading the current group
    event; ``post`` additionally incorporates the complete current event.  A
    coverage-segment boundary always resets the trajectory and no state is
    carried across an unobserved gap or excluded seizure interval.
    """

    model.eval()
    n = data.event_time.size
    d = model.state.cfg.state_dim
    pre_out = np.full((n, d), np.nan, dtype=np.float32)
    post_out = np.full((n, d), np.nan, dtype=np.float32)
    for seg in np.unique(data.event_segment):
        rows = np.flatnonzero(data.event_segment == seg)
        if rows.size == 0:
            continue
        state = model.state.initial(1, device)
        previous = float(data.event_time[rows[0]])
        for chunk in _chunks(rows, data.event_time, config.chunk_events, config.chunk_seconds):
            batch = _to_device(data.seq.gather_positions(data.input_source_position[chunk]), device)
            q = torch.as_tensor(data.q_context[chunk], dtype=torch.float32, device=device)
            embedding = model.encode_events(batch)
            for local, row in enumerate(chunk):
                dt = torch.tensor(
                    [max(0.0, float(data.event_time[row]) - previous)], device=device
                )
                pre = model.state.evolve(state, dt)
                state = model.state.update(
                    pre,
                    embedding[local : local + 1].float()
                    - model.expected_from_q(q[local : local + 1]),
                )
                pre_out[row] = pre.squeeze(0).cpu().numpy()
                post_out[row] = state.squeeze(0).cpu().numpy()
                previous = float(data.event_time[row])
            state = state.detach()
    return pre_out, post_out


def _grid_states_from_event_post(
    model: FullMarkStateModel, data: FullMarkData, post_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = model.state.mean.detach().cpu().numpy().astype(np.float32)
    taus = model.state.taus.detach().cpu().numpy().astype(np.float32)
    out = np.broadcast_to(mean, (data.grid_time.size, mean.size)).copy()
    source = data.grid_source_event
    valid = (source >= 0) & np.isfinite(post_state[np.maximum(source, 0)]).all(1)
    if np.any(valid):
        decay = np.exp(-data.grid_source_dt[valid, None] / taus[None])
        out[valid] = mean[None] + (post_state[source[valid]] - mean[None]) * decay
    return out, valid


@torch.no_grad()
def evaluate_physical_selection(
    model: FullMarkStateModel,
    data: FullMarkData,
    post_state: np.ndarray,
    device: torch.device,
    *,
    include_seizure_strata: bool = True,
) -> dict[str, Any]:
    grid_state, state_valid = _grid_states_from_event_post(model, data, post_state)
    fit = np.flatnonzero((data.grid_phase == "FIT") & state_valid)
    selection = np.flatnonzero((data.grid_phase == "SELECTION") & state_valid)
    mean_state = np.nanmean(grid_state[fit], axis=0) if fit.size else np.zeros(grid_state.shape[1])
    constant = np.broadcast_to(mean_state, grid_state.shape).copy()
    result: dict[str, Any] = {"n_selection_grid_anchors": int(selection.size), "horizons": {}}
    for j, horizon in enumerate(data.physical_horizons_seconds):
        horizon_rows = selection[data.future_valid[selection, j]]
        arms: dict[str, Any] = {}
        zero_q = np.zeros_like(data.grid_q)
        state_specs: list[tuple[str, np.ndarray, np.ndarray | None, np.ndarray]] = [
            ("static_only", zero_q, None, horizon_rows),
            ("q_only", data.grid_q, None, horizon_rows),
            ("mark_only", zero_q, grid_state, horizon_rows),
            ("q_plus_mark", data.grid_q, grid_state, horizon_rows),
            ("fit_period_mean_mark", data.grid_q, constant, horizon_rows),
        ]
        matched_rows = np.asarray([], dtype=np.int64)
        matched_states = None
        if data.provenance.get("physical_window_contract") == "observed_support":
            donor_pool = np.flatnonzero(
                np.isin(data.grid_phase, ("FIT", "INNER", "SELECTION")) & state_valid
            )
            donors = matched_wrong_time_donors(
                data.grid_time, horizon_rows, donor_pool,
                minimum_time_separation=float(horizon),
                recent_rate=data.grid_q[:, 0],
                exposure_fraction=np.exp(data.future_count_log_offset[:, j]),
                n_donors=5,
            )
            donor_valid = np.all(donors >= 0, axis=1)
            matched_rows = horizon_rows[donor_valid]
            if np.any(donor_valid):
                matched_states = grid_state[donors[donor_valid]]
            state_specs.append(("correct_on_matched_support", data.grid_q, grid_state, matched_rows))
        else:
            shifted = grid_state.copy(); shift_valid = np.zeros(data.grid_time.size, dtype=bool)
            for seg in np.unique(data.grid_segment[horizon_rows]):
                rows = horizon_rows[data.grid_segment[horizon_rows] == seg]
                if rows.size < 4:
                    continue
                donor = np.roll(rows, rows.size // 2)
                ok = np.abs(data.grid_time[donor] - data.grid_time[rows]) >= 1800.0
                shifted[rows[ok]] = grid_state[donor[ok]]
                shift_valid[rows[ok]] = True
            matched_rows = horizon_rows[shift_valid[horizon_rows]]
            matched_states = shifted[matched_rows, None]
            state_specs.append(("correct_on_matched_support", data.grid_q, grid_state, matched_rows))

        def score_arm(q_np: np.ndarray, state_np: np.ndarray | None, eligible: np.ndarray) -> dict[str, Any]:
            if eligible.size == 0:
                return {"status": "NOT_ESTIMABLE"}
            q = torch.as_tensor(q_np[eligible], dtype=torch.float32, device=device)
            state = None if state_np is None else torch.as_tensor(state_np[eligible], dtype=torch.float32, device=device)
            count_log_offset = torch.as_tensor(
                data.future_count_log_offset[eligible], dtype=torch.float32, device=device,
            )
            count_log, extent_logit, part_logit = model.physical_head.predictions(
                q, state, count_log_offset,
            )
            count = torch.as_tensor(data.future_count[eligible, j], dtype=torch.float32, device=device)
            count_nll = negative_binomial_nll(
                count, count_log[:, j], model.physical_head.log_dispersion[j]
            ).mean()
            morphology = data.future_count[eligible, j] > 0
            if np.any(morphology):
                mr = torch.as_tensor(morphology, dtype=torch.bool, device=device)
                extent = torch.as_tensor(data.future_extent[eligible, j], dtype=torch.float32, device=device)
                extent_bce = torch.nn.functional.binary_cross_entropy_with_logits(
                    extent_logit[mr, j], extent[mr], reduction="mean"
                )
                pvalid = torch.as_tensor(
                    data.future_participation_valid[eligible, j], dtype=torch.bool, device=device
                )
                ptarget = torch.as_tensor(
                    data.future_participation[eligible, j], dtype=torch.float32, device=device
                )
                part_bce = torch.nn.functional.binary_cross_entropy_with_logits(
                    part_logit[:, j][pvalid], ptarget[pvalid], reduction="mean"
                )
            else:
                extent_bce = torch.tensor(float("nan"), device=device)
                part_bce = torch.tensor(float("nan"), device=device)
            return {
                "status": "ESTIMATED", "n_anchors": int(eligible.size),
                "count_nll": float(count_nll.cpu()),
                "extent_bce": float(extent_bce.cpu()),
                "participation_bce": float(part_bce.cpu()),
            }
        for name, q_np, state_np, eligible in state_specs:
            arms[name] = score_arm(q_np, state_np, eligible)

        if matched_states is None or matched_rows.size == 0:
            arms["matched_wrong_time"] = {"status": "NOT_ESTIMABLE"}
        else:
            # Score every target against five real donor states, then average.
            # Fixed donor count gives every target anchor equal weight and
            # avoids constructing an artificial median latent vector.
            n_donor = int(matched_states.shape[1])
            repeated_rows = np.repeat(matched_rows, n_donor)
            q = torch.as_tensor(data.grid_q[repeated_rows], dtype=torch.float32, device=device)
            state = torch.as_tensor(
                matched_states.reshape(-1, matched_states.shape[-1]), dtype=torch.float32, device=device,
            )
            offset = torch.as_tensor(
                data.future_count_log_offset[repeated_rows], dtype=torch.float32, device=device,
            )
            count_log, extent_logit, part_logit = model.physical_head.predictions(q, state, offset)
            count = torch.as_tensor(data.future_count[repeated_rows, j], dtype=torch.float32, device=device)
            count_nll = negative_binomial_nll(
                count, count_log[:, j], model.physical_head.log_dispersion[j]
            ).reshape(-1, n_donor).mean(1).mean()
            extent = torch.as_tensor(data.future_extent[repeated_rows, j], dtype=torch.float32, device=device)
            extent_bce = torch.nn.functional.binary_cross_entropy_with_logits(
                extent_logit[:, j], extent, reduction="none"
            ).reshape(-1, n_donor).mean(1).mean()
            pvalid = torch.as_tensor(
                data.future_participation_valid[repeated_rows, j], dtype=torch.bool, device=device
            )
            ptarget = torch.as_tensor(
                data.future_participation[repeated_rows, j], dtype=torch.float32, device=device
            )
            part_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                part_logit[:, j], ptarget, reduction="none"
            )
            part_bce = part_loss[pvalid].mean()
            arms["matched_wrong_time"] = {
                "status": "ESTIMATED", "n_anchors": int(matched_rows.size),
                "donors_per_anchor": n_donor,
                "count_nll": float(count_nll.cpu()), "extent_bce": float(extent_bce.cpu()),
                "participation_bce": float(part_bce.cpu()),
                "contract": "mean loss across five real same-patient matched wrong-time states",
            }
        def gain(base: str, full: str, metric: str) -> float | None:
            a, b = arms.get(base, {}), arms.get(full, {})
            if metric not in a or metric not in b or not np.isfinite([a[metric], b[metric]]).all():
                return None
            return float(a[metric] - b[metric])
        arms["contrasts"] = {
            metric: {
                "rate_gain_over_static": gain("static_only", "q_only", metric),
                "mark_only_gain_over_static": gain("static_only", "mark_only", metric),
                "mark_gain_over_q": gain("q_only", "q_plus_mark", metric),
                "correct_time_gain_over_matched_wrong": gain("matched_wrong_time", "correct_on_matched_support", metric),
                "mark_gain_over_period_mean": gain("fit_period_mean_mark", "q_plus_mark", metric),
            }
            for metric in ("count_nll", "extent_bce", "participation_bce")
        }
        result["horizons"][f"future_{int(horizon // 60)}min"] = arms
    if include_seizure_strata and data.provenance.get("physical_window_contract") == "observed_support":
        no_seizure = np.asarray(data.future_seizure_count == 0, dtype=bool)
        result["no_seizure_crossing"] = evaluate_physical_selection(
            model,
            replace(data, future_valid=data.future_valid & no_seizure),
            post_state,
            device,
            include_seizure_strata=False,
        )
    return result


@torch.no_grad()
def evaluate_selection(model: FullMarkStateModel, data: FullMarkData, bundle_tensors: dict[str, Tensor],
                       config: FullMarkTrainConfig, device: torch.device) -> dict[str, Any]:
    fit_rows, fit_states_np = collect_states(model, data, "FIT", device, config)
    rows, states_np = collect_states(model, data, "SELECTION", device, config)
    if rows.size == 0: raise ValueError("no selection states")
    # This is deliberately a FIT-period mean.  A mean computed from the
    # held-out selection trajectory would contain later observed events and
    # would turn the constant-state control into a transductive comparator.
    mean_state = _fit_period_mean_state(fit_states_np, fit_rows)
    period = np.broadcast_to(mean_state, states_np.shape).copy()
    shifted = states_np.copy(); shift_valid = np.zeros(data.event_time.size, bool)
    for seg in np.unique(data.event_segment[rows]):
        rr = rows[data.event_segment[rows] == seg]
        if rr.size < 4 or data.event_time[rr[-1]] - data.event_time[rr[0]] < 3600: continue
        donor = np.roll(rr, rr.size // 2)
        ok = np.abs(data.event_time[donor] - data.event_time[rr]) >= 1800
        shifted[rr[ok]] = states_np[donor[ok]]; shift_valid[rr[ok]] = True
    arms = {}
    for j, offset in enumerate(data.event_offsets):
        target = data.next_index[rows, j]
        good = (target >= 0) & (data.decoder_index[np.maximum(target, 0)] >= 0)
        if not good.any(): continue
        ar, tg = rows[good], target[good]
        cache = torch.as_tensor(data.decoder_index[tg], dtype=torch.long, device=device)
        batch = {key: value[cache] for key, value in bundle_tensors.items()}
        q = torch.as_tensor(data.q_context[ar], dtype=torch.float32, device=device)
        arm_specs = {
            "static_only": (False, None),
            "rate_only": (True, None),
            "mark_only": (False, torch.as_tensor(states_np[ar], device=device)),
            "rate_plus_mark": (True, torch.as_tensor(states_np[ar], device=device)),
            "period_mean_mark": (True, torch.as_tensor(period[ar], device=device)),
            "block_shift_mark": (True, torch.as_tensor(shifted[ar], device=device)),
        }
        dt = torch.as_tensor(
            data.event_time[tg] - data.event_time[ar], dtype=torch.float32, device=device,
        )
        values = {}
        for name, (use_q, m) in arm_specs.items():
            if m is not None:
                m = model.state.evolve(m, dt)
            dynamic_q = bool(use_q and config.decoder_use_dynamic_q)
            score = model.decoder.scores(batch, q if dynamic_q else None,
                                         use_static=True, use_dynamic=dynamic_q,
                                         extra_context=m, extra_adapter=model.m_adapter if m is not None else None)
            values[name] = {}
            support = shift_valid[ar]
            for endpoint in ("grammar", "next_bce", "stop_bce", "contact_nll"):
                arr = score[endpoint].cpu().numpy()
                if name == "block_shift_mark": arr = np.where(support, arr, np.nan)
                # Like-for-like companion of the block-shift null: the same arm
                # restricted to anchors that actually have a distant donor.  Any
                # timing contrast must use these fields (review 2026-09-04).
                on_support = np.where(support, arr, np.nan)
                values[name][endpoint] = {
                    "mean": float(np.nanmean(arr)) if np.isfinite(arr).any() else None,
                    "n": int(np.isfinite(arr).sum()),
                    "mean_on_shift_support": (
                        float(np.nanmean(on_support)) if np.isfinite(on_support).any() else None
                    ),
                    "n_on_shift_support": int(np.isfinite(on_support).sum()),
                }
        values["same_prefix_semantics"] = (
            "the frozen decoder receives the observed first tied group at step 0; "
            "next_bce/contact_nll score later recruited contacts and stop_bce scores whether the event continues"
        )
        arms[f"next_{offset}_events"] = values
    return {"arms": arms, "n_selection_state_rows": int(rows.size),
        "n_shift_valid": int(shift_valid[rows].sum()),
        "period_mean_source": "FIT state trajectory only",
        "n_fit_rows_for_period_mean": int(fit_rows.size)}


def _fit_period_mean_state(states: np.ndarray, fit_rows: np.ndarray) -> np.ndarray:
    """Return the only admissible constant-state control for held-out scoring."""

    fit_rows = np.asarray(fit_rows, dtype=np.int64)
    if fit_rows.size == 0:
        raise ValueError("no FIT states for the period-mean state control")
    fit = np.asarray(states, dtype=np.float32)[fit_rows]
    if not np.isfinite(fit).all():
        raise ValueError("FIT states contain non-finite values")
    return np.mean(fit, axis=0)


def train_full_mark_subject(data: FullMarkData, bundle: FrozenDecoderBundle, base_adapter: Path,
                            config: FullMarkTrainConfig, *, device: torch.device,
                            out_dir: Path, overwrite: bool = False) -> dict[str, Any]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    card_path = out_dir / "card.json"
    if card_path.exists() and not overwrite: return json.loads(card_path.read_text(encoding="utf-8"))
    config.validate()
    started = time.time(); seed_all(config.seed)
    if tuple(data.event_offsets) != tuple(int(value) for value in config.event_offsets):
        raise ValueError("loaded event offsets do not match the training configuration")
    if len(config.offset_weights) != len(config.event_offsets):
        raise ValueError("offset_weights and event_offsets must have equal length")
    data = configure_event_input_view(data, config)
    model = FullMarkStateModel(data, bundle, base_adapter, config, device).to(device)
    if config.physical_loss_weight > 0:
        physical_baseline = fit_physical_q_baseline(model, data, config, device)
    else:
        for parameter in model.physical_head.q_parameters():
            parameter.requires_grad_(False)
        physical_baseline = {
            "status": "SKIPPED_BY_EVENT_OFFSET_ONLY_CONTRACT",
            "reason": "physical_loss_weight is zero",
        }
    parameter_groups = [
        {"params": model.event_encoder.parameters(), "lr": config.encoder_lr},
        {"params": [model.timing_token], "lr": config.encoder_lr},
        {"params": model.state.parameters(), "lr": config.state_lr},
        {"params": model.expected_from_q.parameters(), "lr": config.state_lr},
        {"params": model.physical_head.state_parameters(), "lr": config.adapter_lr},
    ]
    if config.objective_family in {"joint", "sg"}:
        parameter_groups.append({"params": model.m_adapter.parameters(), "lr": config.adapter_lr})
    else:
        for parameter in model.m_adapter.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=config.weight_decay)
    tensors = decoder_tensors(bundle, device)
    best, best_epoch, best_state, stale, history = math.inf, -1, None, 0, []
    for epoch in range(config.max_epochs):
        train = run_phase(model, data, tensors, "FIT", config, device, optimizer)
        inner = run_phase(model, data, tensors, "INNER", config, device, None)
        history.append({"epoch": epoch, "fit": train, "inner": inner})
        value = inner["mean_loss"]
        if value is not None and value < best - 1e-5:
            best, best_epoch, stale = value, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if not k.startswith("decoder.decoder.")}
        else: stale += 1
        atomic_json(out_dir / "progress.json", {"format": f"{FORMAT_PREFIX}_full_mark_progress_v1",
                    "subject": data.subject, "seed": config.seed, "history": history,
                    "elapsed_seconds": time.time() - started})
        if epoch + 1 >= config.min_epochs and stale >= config.patience: break
    if best_state is None: raise RuntimeError("full-mark model produced no finite inner checkpoint")
    current = model.state_dict()
    for key, value in best_state.items(): current[key] = value.to(device)
    model.load_state_dict(current)
    pre_state, post_state = collect_all_states(model, data, device, config)
    selection = (
        evaluate_selection(model, data, tensors, config, device)
        if config.report_selection
        else {"status": "HELD_UNREAD_DURING_HYPERPARAMETER_SEARCH"}
    )
    physical_selection = (
        evaluate_physical_selection(model, data, post_state, device)
        if config.report_selection and config.physical_loss_weight > 0
        else ({"status": "HELD_UNREAD_DURING_HYPERPARAMETER_SEARCH"}
              if not config.report_selection else
              {"status": "NOT_APPLICABLE_EVENT_OFFSET_ONLY_CONTRACT"})
    )
    trajectory = out_dir / "state_trajectory.npz"
    with trajectory.open("wb") as handle:
        np.savez_compressed(
            handle,
            event_time=data.event_time.astype(np.float64),
            event_segment=data.event_segment.astype(np.int64),
            phase=data.phase,
            source_position=data.source_position.astype(np.int64),
            input_source_position=data.input_source_position.astype(np.int64),
            input_view=np.asarray(data.input_view),
            q_context=data.q_context.astype(np.float32),
            state_pre=pre_state,
            state_post=post_state,
            fixed_taus_seconds=model.state.taus.detach().cpu().numpy().astype(np.float32),
            state_mean=model.state.mean.detach().cpu().numpy().astype(np.float32),
        )
    checkpoint = out_dir / "checkpoint.pt"
    torch.save({"state_dict": best_state, "config": asdict(config), "subject": data.subject,
                "base_adapter": str(base_adapter),
                "provenance": {**data.provenance, "input_view": data.input_view_details}}, checkpoint)
    card = {"format": f"{FORMAT_PREFIX}_full_mark_state_card_v1", "subject": data.subject,
            "seed": config.seed, "config": asdict(config), "selected_epoch": best_epoch,
            "best_inner_loss": best, "history": history,
            "physical_q_baseline_training": physical_baseline,
            "selection": selection, "physical_selection": physical_selection,
            "checkpoint": str(checkpoint), "state_trajectory": str(trajectory),
            "trajectory_semantics": {
                "state_pre": "after physical-time evolution, before current event",
                "state_post": "after reading the complete current group event",
                "gap_rule": "reset at every real target-coverage segment",
                "future_readout": "anchor post-state evolved over exact physical dt without future event updates",
            },
            "provenance": {**data.provenance, "input_view": data.input_view_details},
            "elapsed_seconds": time.time() - started, "development_targets_read": False,
            "sealed_partition_opened": False, "seizure_outcomes_read": False}
    atomic_json(card_path, card); return card


def restore_full_mark_model(
    data: FullMarkData,
    bundle: FrozenDecoderBundle,
    base_adapter: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[FullMarkStateModel, FullMarkTrainConfig]:
    """Restore a frozen W3 model without silently substituting any decoder."""

    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = FullMarkTrainConfig(**saved["config"])
    if saved.get("subject") != data.subject or str(saved.get("base_adapter")) != str(base_adapter):
        raise ValueError("W3 checkpoint identity/base adapter differs from requested trajectory")
    data = configure_event_input_view(data, config)
    model = FullMarkStateModel(data, bundle, base_adapter, config, device).to(device)
    missing, unexpected = model.load_state_dict(saved["state_dict"], strict=False)
    compatible_missing = {"timing_token"} if config.input_view == "full_mark" else set()
    if unexpected or any(
        not key.startswith("decoder.decoder.") and key not in compatible_missing for key in missing
    ):
        raise ValueError(f"W3 checkpoint state mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, config


def export_state_trajectory(
    model: FullMarkStateModel,
    data: FullMarkData,
    config: FullMarkTrainConfig,
    device: torch.device,
    path: Path,
) -> Path:
    """Persist the one shared pre/post trajectory consumed by W4--W6."""

    pre_state, post_state = collect_all_states(model, data, device, config)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            event_time=data.event_time.astype(np.float64),
            event_segment=data.event_segment.astype(np.int64),
            phase=data.phase,
            source_position=data.source_position.astype(np.int64),
            q_context=data.q_context.astype(np.float32),
            state_pre=pre_state,
            state_post=post_state,
            fixed_taus_seconds=model.state.taus.detach().cpu().numpy().astype(np.float32),
            state_mean=model.state.mean.detach().cpu().numpy().astype(np.float32),
        )
    return path
