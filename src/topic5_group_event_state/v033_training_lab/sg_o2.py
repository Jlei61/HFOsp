"""Staged human S_G architecture diagnostic for v0.3.3.

This is the first training-laboratory entry that actually learns a
grammar-view state producer from the human event stream.  It deliberately
keeps three boundaries separate:

* the accepted legacy contact decoder and ``next_set_stop_loss`` are frozen;
* the state producer is fitted on ``STATE_TRAIN`` and selected only on the
  chronologically later ``STATE_SELECTION`` anchors;
* the output is an O2 *training diagnostic*, not an H1/H2a result.

One anchor state is reused for every event in its following 30 minute block.
Later events in that block therefore never update the state being scored.
The optional ``mark_scaffold_split`` encoder cell is only an architecture
cell.  It is not a claim that the earlier synthetic failure was caused by the
human feature routing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import resource
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from src.topic5_group_event_state.v032_model.state import MarkedLeakyBank
from src.topic5_rank_distribution import next_set_stop_loss

from .contact_grammar import (
    DATASET_ROOT,
    LegacyContactGrammar,
    load_calibrated_legacy_grammar,
    tensor_state_hash,
)
from .data import robust_scale_apply, robust_scale_fit
from .paths import atomic_write_json, atomic_write_torch, current_commit, file_hash


HUMAN_INPUT_ROOT = Path("/data/hfosp_group_event_state_v0_3_3/agent_c/human_inputs")
GRAMMAR_ROOT = Path("/data/hfosp_group_event_state_v0_3_3/agent_b/contact_grammar")
O2_ROOT = Path("/data/hfosp_group_event_state_v0_3_3/agent_b/sg_o2")
TUNING_SUBJECTS = ("epilepsiae_253", "epilepsiae_916")
ALLOWED_TARGET_PHASES = ("STATE_TRAIN", "STATE_SELECTION")
FORBIDDEN_PHASE_TOKENS = ("DEVELOPMENT", "SEALED", "TEST", "SEIZURE")
WIDTHS = (32, 64, 128)
DEPTHS = (2, 4)
NORMS = ("pre", "post")
INITS = ("xavier", "orthogonal")
ROUTINGS = ("joint", "mark_scaffold_split")
TAUS_SECONDS = (300.0, 1800.0, 7200.0)
TARGET_HORIZON_SECONDS = 1800.0


@dataclass(frozen=True)
class SGO2ArchConfig:
    """One O2 architecture cell; inputs, target, split and scorer stay fixed."""

    width: int = 32
    depth: int = 2
    residual: bool = True
    norm: str = "pre"
    init: str = "xavier"
    update_gate: bool = True
    input_routing: str = "joint"
    write_width: int = 4
    adapter_rank: int = 4

    @property
    def state_dim(self) -> int:
        return len(TAUS_SECONDS) * int(self.write_width)

    def validate(self) -> "SGO2ArchConfig":
        if int(self.width) not in WIDTHS or int(self.depth) not in DEPTHS:
            raise ValueError(f"O2 width/depth must be in {WIDTHS}/{DEPTHS}")
        if self.norm not in NORMS or self.init not in INITS:
            raise ValueError(f"O2 norm/init must be in {NORMS}/{INITS}")
        if self.input_routing not in ROUTINGS:
            raise ValueError(f"O2 input_routing must be in {ROUTINGS}")
        if self.write_width < 1 or self.adapter_rank < 1:
            raise ValueError("O2 write width and adapter rank must be positive")
        return self


@dataclass(frozen=True)
class SGO2TrainConfig:
    max_steps: int = 80
    patience: int = 12
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    pair_batch_size: int = 1024
    seed: int = 20260903
    min_delta: float = 1e-5
    smoke_train_anchors: int | None = None
    smoke_inner_anchors: int | None = None

    def validate(self) -> "SGO2TrainConfig":
        if self.max_steps < 1 or self.patience < 1 or self.pair_batch_size < 1:
            raise ValueError("O2 steps, patience and pair batch size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 \
                or self.gradient_clip <= 0 or self.min_delta < 0:
            raise ValueError("O2 optimiser settings are invalid")
        for value in (self.smoke_train_anchors, self.smoke_inner_anchors):
            if value is not None and int(value) < 1:
                raise ValueError("O2 smoke anchor caps must be positive")
        return self


@dataclass(frozen=True)
class GrammarPairs:
    anchor_rows: np.ndarray
    pair_anchor: np.ndarray
    pair_event: np.ndarray
    pair_weight: np.ndarray

    def validate(self) -> "GrammarPairs":
        n = int(np.asarray(self.anchor_rows).size)
        pa = np.asarray(self.pair_anchor, dtype=np.int64)
        pe = np.asarray(self.pair_event, dtype=np.int64)
        pw = np.asarray(self.pair_weight, dtype=np.float64)
        if n == 0 or pa.ndim != 1 or pe.shape != pa.shape or pw.shape != pa.shape \
                or pa.size == 0:
            raise ValueError("O2 grammar pairs must be non-empty aligned vectors")
        if np.any(pa < 0) or np.any(pa >= n) or np.any(pe < 0) \
                or not np.isfinite(pw).all() or np.any(pw <= 0):
            raise ValueError("O2 grammar-pair indices or weights are invalid")
        if not math.isclose(float(pw.sum()), 1.0, rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError("O2 pair weights must average events within anchor then anchors")
        return self


@dataclass(frozen=True)
class SGO2HumanData:
    subject: str
    event_time: np.ndarray
    event_carry: np.ndarray
    x_scaled: np.ndarray
    train_event_mask: np.ndarray
    group_ids: np.ndarray
    group_count: np.ndarray
    anchor_time: np.ndarray
    anchor_carry: np.ndarray
    last_event_pos: np.ndarray
    phase: np.ndarray
    train_pairs: GrammarPairs
    inner_pairs: GrammarPairs
    feature_names: tuple[str, ...]
    mark_columns: tuple[int, ...]
    scaffold_columns: tuple[int, ...]
    provenance: Mapping[str, Any]


def assert_training_phases(phases: Sequence[str]) -> None:
    observed = {str(value) for value in phases}
    forbidden = sorted(
        value for value in observed
        if any(token in value.upper() for token in FORBIDDEN_PHASE_TOKENS)
    )
    unknown = sorted(observed - set(ALLOWED_TARGET_PHASES))
    if forbidden or unknown:
        raise PermissionError(
            f"O2 accepts only {ALLOWED_TARGET_PHASES}; forbidden={forbidden}, unknown={unknown}"
        )


def validate_o2_smoke_lease(path: Path, *, subject: str) -> dict[str, Any]:
    """Require a current, explicit one-worker grant before touching a human GPU."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    status = str(payload.get("status", "")).upper()
    scope = " ".join(str(v) for v in [
        *payload.get("allowed_now", []), *payload.get("allowed_work", []),
    ])
    subjects = {str(v) for v in payload.get("allowed_subjects", [])}
    explicit = payload.get("o2_sg_human_smoke_authorized") is True
    if not status.startswith("ACTIVE") or int(payload.get("max_workers", 0)) < 1:
        raise PermissionError("O2 resource lease is not ACTIVE with at least one worker")
    scope_lower = scope.lower()
    scoped_o2_smoke = "o2" in scope_lower and "smoke" in scope_lower
    if not explicit and "sg_o2" not in scope_lower \
            and "s_g o2" not in scope_lower and not scoped_o2_smoke:
        raise PermissionError("O2 resource lease does not explicitly authorize S_G O2 smoke")
    if subjects and subject not in subjects:
        raise PermissionError(f"O2 resource lease does not authorize {subject}")
    if int(payload.get("max_jobs_per_gpu_before_sentinel_review", 1)) > 1:
        raise PermissionError("O2 sentinel lease must be one job per GPU or stricter")
    return payload


def staged_o2_plan() -> dict[str, Any]:
    """Predeclared successive search; this function never launches the grid."""

    base = dict(width=32, depth=2, residual=True, norm="pre", init="xavier",
                update_gate=True, input_routing="joint")
    return {
        "format": "group_event_state_v0_3_3_sg_o2_successive_plan",
        "constant_contract": {
            "input": "human_R0_event_features",
            "target": "legacy_next_set_or_STOP_future_0_to_30min",
            "fit": "STATE_TRAIN",
            "inner_validation": "chronologically_later_STATE_SELECTION",
            "decoder": "frozen_calibrated_legacy_contact_grammar",
        },
        "stages": [
            {"name": "O2_S0_resource_sentinel", "requires": "O1 recipe",
             "cells": [base]},
            {"name": "O2_S1_width", "requires": "S0 resource pass",
             "cells": [{**base, "width": width} for width in WIDTHS]},
            {"name": "O2_S2_block_structure", "requires": "select S1 width",
             "cells_template": [
                 {"depth": depth, "residual": residual, "norm": norm}
                 for depth in DEPTHS for residual in (False, True) for norm in NORMS
             ]},
            {"name": "O2_S3_init_and_gate", "requires": "select S2 structure",
             "cells_template": [
                 {"init": init, "update_gate": gate}
                 for init in INITS for gate in (False, True)
             ]},
            {"name": "O2_S4_optional_mark_scaffold_cell",
             "requires": "only if joint encoder remains training-inadequate",
             "cells_template": [{"input_routing": value} for value in ROUTINGS],
             "interpretation": "architecture diagnostic only; no causal attribution"},
        ],
        "launch_policy": "one stage at a time; no Cartesian product; await O1 before S1",
    }


def _group_count(group_ids: np.ndarray) -> np.ndarray:
    group_ids = np.asarray(group_ids, dtype=np.int64)
    return np.maximum(group_ids.max(axis=1) + 1, 0).astype(np.int64)


def _cap_rows(rows: np.ndarray, cap: int | None) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    if cap is None or rows.size <= int(cap):
        return rows
    # Cover the whole allowed phase in a resource smoke without introducing a
    # random seed or reading any later phase.
    pos = np.linspace(0, rows.size - 1, int(cap)).round().astype(np.int64)
    return rows[np.unique(pos)]


def _future_pairs(
    anchor_rows: np.ndarray,
    *,
    anchor_time: np.ndarray,
    anchor_target: np.ndarray,
    event_time: np.ndarray,
    event_target: np.ndarray,
    horizon: float,
) -> GrammarPairs:
    kept_anchor: list[int] = []
    pair_anchor: list[int] = []
    pair_event: list[int] = []
    by_segment = {
        int(seg): np.flatnonzero(event_target == seg)
        for seg in np.unique(anchor_target[anchor_rows])
    }
    for source_row in np.asarray(anchor_rows, dtype=np.int64):
        events = by_segment.get(int(anchor_target[source_row]), np.empty(0, dtype=np.int64))
        times = event_time[events]
        lo = int(np.searchsorted(times, anchor_time[source_row], side="left"))
        hi = int(np.searchsorted(times, anchor_time[source_row] + float(horizon), side="left"))
        selected = events[lo:hi]
        if selected.size == 0:
            continue
        local_anchor = len(kept_anchor)
        kept_anchor.append(int(source_row))
        pair_anchor.extend([local_anchor] * int(selected.size))
        pair_event.extend(selected.tolist())
    if not kept_anchor:
        raise ValueError("O2 phase has no anchors with a future grammar event")
    pa = np.asarray(pair_anchor, dtype=np.int64)
    counts = np.bincount(pa, minlength=len(kept_anchor)).astype(np.float64)
    weights = 1.0 / (float(len(kept_anchor)) * counts[pa])
    return GrammarPairs(
        anchor_rows=np.asarray(kept_anchor, dtype=np.int64),
        pair_anchor=pa,
        pair_event=np.asarray(pair_event, dtype=np.int64),
        pair_weight=weights,
    ).validate()


def _feature_routes(names: Sequence[str]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    # Only three R0 fields explicitly describe the sampled contact scaffold.
    # Keeping the list short prevents the optional cell from being silently
    # redefined according to its result.
    scaffold_names = {
        "n_participating_shafts", "mean_pairwise_distance_mm",
        "coordinate_valid_fraction",
    }
    scaffold = tuple(i for i, name in enumerate(names) if name in scaffold_names)
    mark = tuple(i for i in range(len(names)) if i not in scaffold)
    if not mark or not scaffold:
        raise ValueError("O2 mark/scaffold feature routing cannot be constructed")
    return mark, scaffold


def load_sg_o2_human_data(
    subject: str,
    *,
    train_cfg: SGO2TrainConfig,
    input_root: Path = HUMAN_INPUT_ROOT,
    dataset_root: Path = DATASET_ROOT,
) -> SGO2HumanData:
    """Expose only STATE_TRAIN/STATE_SELECTION targets and pre-70% inputs."""

    train_cfg.validate()
    if subject not in TUNING_SUBJECTS:
        raise PermissionError(f"O2 is locked to tuning patients {TUNING_SUBJECTS}")
    manifest_path = Path(input_root) / subject / "manifest_v3.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_path = Path(str(manifest.get("input_path", "")))
    if manifest.get("format") != "group_event_state_v0_3_3_human_r0_input_manifest" \
            or manifest.get("subject") != subject or manifest.get("role") != "tuning" \
            or manifest.get("sealed") is not False \
            or manifest.get("development_evaluation_used_for_fitting") is not False:
        raise PermissionError("O2 human manifest is not open tuning-only data")
    if not artifact_path.is_file() or file_hash(artifact_path) != manifest.get("input_npz_sha256"):
        raise ValueError("O2 human input bytes differ from the locked manifest")
    with np.load(artifact_path, allow_pickle=False) as stored:
        metadata = json.loads(str(np.asarray(stored["metadata_json"]).item()))
        event_time_all = np.asarray(stored["event_time"], dtype=np.float64)
        event_carry_all = np.asarray(stored["event_carry"], dtype=np.int64)
        feature_all = np.asarray(stored["event_features_r0"], dtype=np.float64)
        valid_all = np.asarray(stored["event_feature_valid"], dtype=bool)
        train_event_all = np.asarray(stored["train_event_mask"], dtype=bool)
        anchor_time_all = np.asarray(stored["anchor_time"], dtype=np.float64)
        anchor_carry_all = np.asarray(stored["anchor_carry"], dtype=np.int64)
        last_all = np.asarray(stored["last_event_pos"], dtype=np.int64)
        phase_all = np.asarray(stored["phase"]).astype(str)
        eligible_all = np.asarray(stored["eligible_by_horizon"], dtype=bool)
        target_segment_all = np.asarray(stored["target_segment"], dtype=np.int64)
        target_bounds = np.asarray(stored["target_segment_bounds"], dtype=np.float64)
    if metadata.get("subject") != subject or metadata.get("sealed") is not False \
            or metadata.get("input_view") != "R0":
        raise ValueError("O2 human R0 metadata identity differs")
    names = tuple(str(v) for v in metadata.get("event_feature_names_r0", []))
    if feature_all.shape != valid_all.shape or len(names) != feature_all.shape[1]:
        raise ValueError("O2 event feature arrays or names do not align")

    # The locked human artifact uses STATE_TRAIN=20--70%,
    # STATE_SELECTION=70--80%, DEVELOPMENT_EVALUATION=80--100%.
    # The additional 60% boundary belongs to the nested H_mark construction;
    # it is not the state-trainer boundary.
    selection_stop = float(manifest["report"]["phase_boundaries_epoch"]["80pct"])
    keep_events = event_time_all < selection_stop
    event_time = event_time_all[keep_events]
    event_carry = event_carry_all[keep_events]
    feature = feature_all[keep_events]
    valid = valid_all[keep_events]
    train_event = train_event_all[keep_events]
    if event_time.size == 0 or not np.any(train_event):
        raise ValueError("O2 has no pre-70% event stream or TRAIN events")

    allowed_anchor = np.isin(phase_all, ALLOWED_TARGET_PHASES) & eligible_all.all(axis=1)
    if np.any(allowed_anchor & (anchor_time_all >= selection_stop)):
        raise PermissionError("O2 allowed anchor crosses the 80% development boundary")
    anchor_source = np.flatnonzero(allowed_anchor)
    assert_training_phases(phase_all[anchor_source])
    anchor_time = anchor_time_all[anchor_source]
    anchor_carry = anchor_carry_all[anchor_source]
    last_event_pos = last_all[anchor_source]
    phase = phase_all[anchor_source]
    anchor_target = target_segment_all[anchor_source]
    if np.any(last_event_pos >= event_time.size):
        raise PermissionError("O2 anchor history reaches a development event at/after 80%")

    # Map only the permitted prefix into the source tied-group array.  No
    # development or sealed mark target is indexed from the source dataset.
    subject_root = Path(dataset_root) / subject
    index = json.loads((subject_root / "index.json").read_text(encoding="utf-8"))
    scalars = np.load(subject_root / "scalars.npz")
    raw_order = np.asarray(scalars["interictal_index"], dtype=np.int64)
    raw_time = np.asarray(scalars["t_abs"], dtype=np.float64)[raw_order]
    stream_index = np.searchsorted(raw_time, event_time, side="left")
    if np.any(stream_index >= raw_time.size) \
            or not np.array_equal(raw_time[stream_index], event_time):
        raise ValueError("O2 permitted event prefix does not map uniquely to source stream")
    raw_rows = raw_order[stream_index]
    group_meta = index["arrays"]["tied_group_id"]
    source_groups = np.load(subject_root / group_meta["file"], mmap_mode="r")
    group_ids = np.asarray(source_groups[raw_rows], dtype=np.int64)
    group_count = _group_count(group_ids)

    event_target = np.full(event_time.size, -1, dtype=np.int64)
    for segment, (left, right) in enumerate(target_bounds):
        inside = (event_time >= left) & (event_time < right)
        if np.any(event_target[inside] >= 0):
            raise ValueError("O2 target segments overlap")
        event_target[inside] = int(segment)

    raw = np.where(valid, feature, np.nan)
    scaler = robust_scale_fit(raw, train_event)
    x_scaled = robust_scale_apply(raw, scaler)
    x_scaled[~valid] = 0.0

    train_rows = np.flatnonzero(phase == "STATE_TRAIN")
    inner_rows = np.flatnonzero(phase == "STATE_SELECTION")
    train_rows = _cap_rows(train_rows, train_cfg.smoke_train_anchors)
    inner_rows = _cap_rows(inner_rows, train_cfg.smoke_inner_anchors)
    train_pairs = _future_pairs(
        train_rows, anchor_time=anchor_time, anchor_target=anchor_target,
        event_time=event_time, event_target=event_target,
        horizon=TARGET_HORIZON_SECONDS,
    )
    inner_pairs = _future_pairs(
        inner_rows, anchor_time=anchor_time, anchor_target=anchor_target,
        event_time=event_time, event_target=event_target,
        horizon=TARGET_HORIZON_SECONDS,
    )
    if float(anchor_time[train_pairs.anchor_rows].max()) \
            >= float(anchor_time[inner_pairs.anchor_rows].min()):
        raise PermissionError("O2 inner validation is not chronologically after STATE_TRAIN")
    mark_columns, scaffold_columns = _feature_routes(names)
    return SGO2HumanData(
        subject=subject, event_time=event_time, event_carry=event_carry,
        x_scaled=np.ascontiguousarray(x_scaled), train_event_mask=train_event,
        group_ids=group_ids, group_count=group_count,
        anchor_time=anchor_time, anchor_carry=anchor_carry,
        last_event_pos=last_event_pos, phase=phase,
        train_pairs=train_pairs, inner_pairs=inner_pairs, feature_names=names,
        mark_columns=mark_columns, scaffold_columns=scaffold_columns,
        provenance={
            "human_manifest": str(manifest_path),
            "human_manifest_sha256": file_hash(manifest_path),
            "human_input": str(artifact_path),
            "human_input_sha256": file_hash(artifact_path),
            "human_content_hash": str(manifest["input_artifact_hash"]),
            "split_hash": str(manifest["split_hash"]),
            "normalization": "robust; fit on CALIBRATION_plus_STATE_TRAIN event mask only",
            "input_stop": "80pct boundary; no DEVELOPMENT_EVALUATION events used",
            "target_phases": list(ALLOWED_TARGET_PHASES),
            "development_targets_exposed": False,
            "seizure_outcomes_read": False,
            "sealed_partition_opened": False,
            "feature_routing": {
                "mark_columns": [names[i] for i in mark_columns],
                "scaffold_columns": [names[i] for i in scaffold_columns],
                "scientific_interpretation": "optional architecture cell only",
            },
        },
    )


class _O2Block(nn.Module):
    def __init__(self, width: int, *, residual: bool, norm: str) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.activation = nn.GELU()
        self.residual = bool(residual)
        self.norm_position = str(norm)

    def forward(self, x: Tensor) -> Tensor:
        if self.norm_position == "pre":
            update = self.activation(self.linear(self.norm(x)))
        else:
            update = self.norm(self.activation(self.linear(x)))
        return x + update / math.sqrt(2.0) if self.residual else update


class SGO2EventEncoder(nn.Module):
    """Configurable event-write encoder used by one O2 cell."""

    def __init__(
        self,
        in_dim: int,
        arch: SGO2ArchConfig,
        *,
        mark_columns: Sequence[int],
        scaffold_columns: Sequence[int],
    ) -> None:
        super().__init__()
        arch.validate()
        self.arch = arch
        self.mark_columns = tuple(int(v) for v in mark_columns)
        self.scaffold_columns = tuple(int(v) for v in scaffold_columns)
        if arch.input_routing == "joint":
            self.input_joint: nn.Module | None = nn.Linear(in_dim, arch.width)
            self.input_mark = None
            self.input_scaffold = None
            self.input_fuse = None
        else:
            mark_width = max(1, arch.width // 2)
            scaffold_width = arch.width - mark_width
            self.input_joint = None
            self.input_mark = nn.Linear(len(self.mark_columns), mark_width)
            self.input_scaffold = nn.Linear(len(self.scaffold_columns), scaffold_width)
            self.input_fuse = nn.Linear(arch.width, arch.width)
        self.blocks = nn.ModuleList([
            _O2Block(arch.width, residual=arch.residual, norm=arch.norm)
            for _ in range(arch.depth)
        ])
        self.write_head = nn.Linear(arch.width, arch.write_width)
        self.gate_head = nn.Linear(arch.width, arch.write_width) if arch.update_gate else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if not isinstance(module, nn.Linear):
                continue
            if self.arch.init == "xavier":
                nn.init.xavier_uniform_(module.weight)
            else:
                nn.init.orthogonal_(module.weight)
            nn.init.zeros_(module.bias)
        if self.gate_head is not None:
            nn.init.constant_(self.gate_head.bias, -2.0)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = x.to(torch.float32)
        if self.input_joint is not None:
            h = self.input_joint(x)
        else:
            mark = x[:, self.mark_columns]
            scaffold = x[:, self.scaffold_columns]
            h = torch.cat([self.input_mark(mark), self.input_scaffold(scaffold)], dim=1)
            h = self.input_fuse(torch.nn.functional.gelu(h))
        for block in self.blocks:
            h = block(h)
        write = self.write_head(h)
        gate = torch.sigmoid(self.gate_head(h)) if self.gate_head is not None \
            else torch.ones_like(write)
        return write, gate


class _StateResidual(nn.Module):
    def __init__(self, state_dim: int, rank: int, hidden: int, embed: int) -> None:
        super().__init__()
        if rank > min(state_dim, hidden, embed):
            raise ValueError("O2 legacy state residual rank is too large")
        self.to_initial = nn.Sequential(
            nn.Linear(state_dim, rank, bias=False), nn.Linear(rank, hidden, bias=False)
        )
        self.to_query = nn.Sequential(
            nn.Linear(state_dim, rank, bias=False), nn.Linear(rank, embed, bias=False)
        )
        self.to_stop = nn.Sequential(
            nn.Linear(state_dim, rank, bias=False), nn.Linear(rank, 1, bias=False)
        )
        for branch in (self.to_initial, self.to_query, self.to_stop):
            nn.init.normal_(branch[0].weight, std=0.02)
            nn.init.normal_(branch[1].weight, std=1e-3)

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return (
            self.to_initial(state), self.to_query(state),
            self.to_stop(state).squeeze(-1),
        )


class FrozenLegacyStateScorer(nn.Module):
    """Frozen accepted decoder with an external low-rank S_G residual."""

    def __init__(self, decoder: LegacyContactGrammar, *, state_dim: int, rank: int) -> None:
        super().__init__()
        self.decoder = decoder
        for parameter in decoder.parameters():
            parameter.requires_grad_(False)
        self.residual = _StateResidual(
            state_dim, rank, int(decoder.base.hidden_size),
            int(decoder.base.contact_embedding_dim),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.decoder.eval()
        return self

    def forward(self, group_ids: Tensor, group_count: Tensor, state: Tensor) -> Mapping[str, Tensor]:
        batch = int(group_ids.shape[0])
        features = self.decoder.contact_features.unsqueeze(0).expand(batch, -1, -1)
        mask = self.decoder.contact_mask.unsqueeze(0).expand(batch, -1)
        offset = self.decoder.local_offset.unsqueeze(0).expand(batch, -1, -1)
        embedding, encoder_input = self.decoder.base._encode(features, offset)
        hidden = self.decoder.base._initial_hidden(embedding, mask)
        initial_delta, query_delta, stop_delta = self.residual(state)
        hidden = hidden + initial_delta
        recruited = torch.zeros_like(mask)
        n_steps = int(group_count.max().detach().cpu()) + 1
        contact_steps, stop_steps, candidate_steps = [], [], []
        scale = math.sqrt(float(self.decoder.base.contact_embedding_dim))
        for step in range(n_steps):
            candidate = mask & ~recruited
            query = self.decoder.base.action_query(hidden) + query_delta
            contact = torch.einsum("bce,be->bc", embedding, query) / scale
            contact = contact + self.decoder.base.action_bias(encoder_input).squeeze(-1)
            contact = contact.masked_fill(~candidate, -1e9)
            stop = self.decoder.base.stop_head(hidden).squeeze(-1) + stop_delta
            contact_steps.append(contact)
            stop_steps.append(stop)
            candidate_steps.append(candidate)
            if step + 1 == n_steps:
                break
            current = (group_ids == step) & mask
            active = (group_count > step).unsqueeze(1)
            updated_recruited = recruited | current
            updated_hidden = self.decoder.base._advance(
                embedding, current, updated_recruited, hidden, mask
            )
            hidden = torch.where(active, updated_hidden, hidden)
            recruited = torch.where(active, updated_recruited, recruited)
        return {
            "contact_logits": torch.stack(contact_steps, dim=1),
            "stop_logits": torch.stack(stop_steps, dim=1),
            "candidate_mask": torch.stack(candidate_steps, dim=1),
        }

    def event_nll(self, group_ids: Tensor, group_count: Tensor, state: Tensor) -> Tensor:
        return next_set_stop_loss(
            self(group_ids, group_count, state), group_ids, group_count
        )["event_nll"]


class SGO2Model(nn.Module):
    def __init__(
        self,
        decoder: LegacyContactGrammar,
        *,
        in_dim: int,
        arch: SGO2ArchConfig,
        mark_columns: Sequence[int],
        scaffold_columns: Sequence[int],
    ) -> None:
        super().__init__()
        self.arch = arch.validate()
        self.encoder = SGO2EventEncoder(
            in_dim, arch, mark_columns=mark_columns,
            scaffold_columns=scaffold_columns,
        )
        self.state = MarkedLeakyBank(
            TAUS_SECONDS, arch.write_width, chunk_seconds=3600.0
        )
        self.scorer = FrozenLegacyStateScorer(
            decoder, state_dim=arch.state_dim, rank=arch.adapter_rank
        )

    def anchor_state(
        self,
        x: Tensor,
        train_event_mask: Tensor,
        event_time: Tensor,
        event_carry: Tensor,
        anchor_time: Tensor,
        last_event_pos: Tensor,
        train_anchor_rows: Tensor,
    ) -> Tensor:
        raw_write, gate = self.encoder(x)
        mean = raw_write[train_event_mask].mean(dim=0)
        write = gate * torch.tanh(raw_write - mean)
        _pre, post = self.state(write, event_time, event_carry)
        anchor = self.state.anchor(post, event_time, anchor_time, last_event_pos)
        reference = anchor[train_anchor_rows]
        center = reference.mean(dim=0)
        scale = reference.std(dim=0, unbiased=False).clamp_min(1e-4)
        return (anchor - center) / scale


def _pair_nll(
    model: SGO2Model,
    data: SGO2HumanData,
    states: Tensor,
    pairs: GrammarPairs,
    *,
    device: torch.device,
    batch_size: int,
    backward: bool,
) -> float:
    total_value = 0.0
    n_pairs = int(pairs.pair_event.size)
    for lo in range(0, n_pairs, int(batch_size)):
        hi = min(n_pairs, lo + int(batch_size))
        events = pairs.pair_event[lo:hi]
        source_anchor = pairs.anchor_rows[pairs.pair_anchor[lo:hi]]
        ids = torch.as_tensor(data.group_ids[events], dtype=torch.long, device=device)
        count = torch.as_tensor(data.group_count[events], dtype=torch.long, device=device)
        weight = torch.as_tensor(pairs.pair_weight[lo:hi], dtype=torch.float32, device=device)
        nll = model.scorer.event_nll(ids, count, states[source_anchor])
        loss = torch.sum(nll * weight)
        if backward:
            loss.backward(retain_graph=hi < n_pairs)
        total_value += float(loss.detach().cpu())
    return total_value


@torch.no_grad()
def _base_pair_nll(
    decoder: LegacyContactGrammar,
    data: SGO2HumanData,
    pairs: GrammarPairs,
    *,
    device: torch.device,
    batch_size: int,
) -> float:
    total = 0.0
    for lo in range(0, pairs.pair_event.size, int(batch_size)):
        hi = min(int(pairs.pair_event.size), lo + int(batch_size))
        events = pairs.pair_event[lo:hi]
        ids = torch.as_tensor(data.group_ids[events], dtype=torch.long, device=device)
        count = torch.as_tensor(data.group_count[events], dtype=torch.long, device=device)
        weight = torch.as_tensor(pairs.pair_weight[lo:hi], dtype=torch.float32, device=device)
        total += float(torch.sum(decoder.loss(ids, count)["event_nll"] * weight).cpu())
    return total


def _cpu_trainable_state(model: SGO2Model) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if not name.startswith("scorer.decoder.")
    }


def _load_trainable_state(model: SGO2Model, state: Mapping[str, Tensor]) -> None:
    current = model.state_dict()
    for name, value in state.items():
        if name not in current or name.startswith("scorer.decoder."):
            raise ValueError("O2 selected state contains an unexpected tensor")
        current[name] = value
    model.load_state_dict(current, strict=True)


def _parameter_l2(parameters: Sequence[nn.Parameter]) -> float:
    return math.sqrt(sum(float(torch.sum(p.detach().double() ** 2).cpu()) for p in parameters))


def _gradient_l2(parameters: Sequence[nn.Parameter]) -> float:
    return math.sqrt(sum(
        float(torch.sum(p.grad.detach().double() ** 2).cpu())
        for p in parameters if p.grad is not None
    ))


def run_sg_o2_cell(
    subject: str,
    *,
    arch: SGO2ArchConfig,
    train_cfg: SGO2TrainConfig,
    device: torch.device,
    output_dir: Path,
    lease_path: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Fit exactly one O2 cell.  A separate driver decides the next stage."""

    arch.validate()
    train_cfg.validate()
    lease = validate_o2_smoke_lease(lease_path, subject=subject)
    allowed_gpu = {int(value) for value in lease.get("allowed_gpu_indices", [])}
    if device.type != "cuda" or (allowed_gpu and int(device.index or 0) not in allowed_gpu):
        raise PermissionError(
            f"O2 smoke device {device} is outside the lease GPU set {sorted(allowed_gpu)}"
        )
    output_dir = Path(output_dir)
    card_path = output_dir / "training_card.json"
    checkpoint_path = output_dir / "checkpoint.pt"
    if card_path.exists() and checkpoint_path.exists() and not overwrite:
        return json.loads(card_path.read_text(encoding="utf-8"))
    random.seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.empty(1, device=device)
        torch.cuda.reset_peak_memory_stats(device)

    data = load_sg_o2_human_data(subject, train_cfg=train_cfg)
    grammar_path = GRAMMAR_ROOT / subject / "legacy_contact_grammar_v033.pt"
    grammar, grammar_artifact = load_calibrated_legacy_grammar(grammar_path, device=device)
    if grammar_artifact.get("scientific_use") is not True \
            or grammar_artifact.get("scoring_contract", {}).get("name") != "legacy_next_set_or_STOP" \
            or grammar_artifact.get("scoring_contract", {}).get("exact_subset_likelihood") is not False:
        raise PermissionError("O2 requires the scientific legacy-scoring grammar artifact")
    if int(grammar_artifact.get("n_contacts", -1)) != data.group_ids.shape[1]:
        raise ValueError("O2 grammar and event target contact dimensions differ")
    frozen_hash = tensor_state_hash(grammar.state_dict())
    if frozen_hash != grammar_artifact.get("base_tensor_hash"):
        raise ValueError("O2 grammar identity differs before training")

    model = SGO2Model(
        grammar, in_dim=data.x_scaled.shape[1], arch=arch,
        mark_columns=data.mark_columns, scaffold_columns=data.scaffold_columns,
    ).to(device)
    x = torch.as_tensor(data.x_scaled, dtype=torch.float32, device=device)
    train_event = torch.as_tensor(data.train_event_mask, dtype=torch.bool, device=device)
    event_time = torch.as_tensor(data.event_time, dtype=torch.float64, device=device)
    event_carry = torch.as_tensor(data.event_carry, dtype=torch.long, device=device)
    anchor_time = torch.as_tensor(data.anchor_time, dtype=torch.float64, device=device)
    last = torch.as_tensor(data.last_event_pos, dtype=torch.long, device=device)
    train_anchor_rows = torch.as_tensor(
        data.train_pairs.anchor_rows, dtype=torch.long, device=device
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable or any(p.requires_grad for p in model.scorer.decoder.parameters()):
        raise RuntimeError("O2 trainable/frozen parameter membership is invalid")
    optimizer = torch.optim.AdamW(
        trainable, lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay
    )
    base_inner = _base_pair_nll(
        grammar, data, data.inner_pairs, device=device,
        batch_size=train_cfg.pair_batch_size,
    )
    started = time.monotonic()
    initial_l2 = _parameter_l2(trainable)
    history: list[dict[str, Any]] = []
    best = math.inf
    best_step = -1
    best_state: dict[str, Tensor] | None = None
    stale = 0
    max_grad = 0.0
    for step in range(1, train_cfg.max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        states = model.anchor_state(
            x, train_event, event_time, event_carry, anchor_time, last,
            train_anchor_rows,
        )
        fit_nll = _pair_nll(
            model, data, states, data.train_pairs, device=device,
            batch_size=train_cfg.pair_batch_size, backward=True,
        )
        grad = _gradient_l2(trainable)
        max_grad = max(max_grad, grad)
        clipped = float(torch.nn.utils.clip_grad_norm_(trainable, train_cfg.gradient_clip))
        optimizer.step()
        model.eval()
        with torch.no_grad():
            states = model.anchor_state(
                x, train_event, event_time, event_carry, anchor_time, last,
                train_anchor_rows,
            )
            inner_nll = _pair_nll(
                model, data, states, data.inner_pairs, device=device,
                batch_size=train_cfg.pair_batch_size, backward=False,
            )
        history.append({
            "step": step, "fit_future_block_event_nll": fit_nll,
            "inner_future_block_event_nll": inner_nll,
            "inner_gain_over_frozen_grammar": base_inner - inner_nll,
            "gradient_l2_before_clip": grad,
            "clip_return_l2": clipped,
        })
        if np.isfinite(inner_nll) and inner_nll < best - train_cfg.min_delta:
            best = float(inner_nll)
            best_step = int(step)
            best_state = _cpu_trainable_state(model)
            stale = 0
        else:
            stale += 1
            if stale >= train_cfg.patience:
                break
    if best_state is None:
        raise RuntimeError("O2 did not produce a finite STATE_SELECTION checkpoint")
    _load_trainable_state(model, best_state)
    selected_l2 = _parameter_l2(trainable)
    frozen_after = tensor_state_hash(model.scorer.decoder.state_dict())
    if frozen_after != frozen_hash:
        raise RuntimeError("O2 modified the frozen accepted grammar")
    elapsed = time.monotonic() - started
    peak_gpu = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    smoke = train_cfg.smoke_train_anchors is not None \
        or train_cfg.smoke_inner_anchors is not None
    checkpoint = {
        "format": "group_event_state_v0_3_3_sg_o2_training_diagnostic",
        "subject": subject, "architecture": asdict(arch),
        "train_config": asdict(train_cfg), "selected_step": best_step,
        "selected_inner_nll": best, "trainable_state": best_state,
        "frozen_grammar_hash": frozen_hash,
        "scientific_use": False,
        "source_commit": current_commit(),
    }
    atomic_write_torch(checkpoint_path, checkpoint)
    card = {
        "format": "group_event_state_v0_3_3_sg_o2_training_card",
        "status": "RESOURCE_SMOKE_COMPLETE" if smoke else "O2_CELL_COMPLETE",
        "scientific_use": False,
        "reason_not_scientific": (
            "O2 diagnoses trainability/architecture only; no later development score is read"
        ),
        "subject": subject,
        "architecture": asdict(arch),
        "architecture_surface": {
            "widths": list(WIDTHS), "depths": list(DEPTHS),
            "residual": [False, True], "norm": list(NORMS),
            "init": list(INITS), "update_gate": [False, True],
            "input_routing": list(ROUTINGS),
        },
        "target_contract": {
            "name": "legacy_next_set_or_STOP_future_0_to_30min",
            "exact_subset_likelihood": False,
            "anchor_reduction": "equal anchor weight; mean event NLL within anchor",
            "same_anchor_state_reused_through_future_block": True,
            "future_events_update_scored_state": False,
        },
        "split_contract": {
            "fit": "STATE_TRAIN",
            "inner_validation": "chronologically_later_STATE_SELECTION",
            "development_evaluation_used": False,
            "seizure_outcomes_used": False,
            "sealed_partition_opened": False,
        },
        "data": {
            **dict(data.provenance),
            "n_events_visible_pre80pct": int(data.event_time.size),
            "n_train_anchors": int(data.train_pairs.anchor_rows.size),
            "n_inner_anchors": int(data.inner_pairs.anchor_rows.size),
            "n_train_anchor_event_pairs": int(data.train_pairs.pair_event.size),
            "n_inner_anchor_event_pairs": int(data.inner_pairs.pair_event.size),
        },
        "frozen_grammar": {
            "checkpoint": str(grammar_path),
            "checkpoint_sha256": file_hash(grammar_path),
            "base_tensor_hash_before": frozen_hash,
            "base_tensor_hash_after": frozen_after,
            "unchanged": True,
            "scoring_contract": grammar_artifact["scoring_contract"],
        },
        "training": {
            "base_inner_event_nll": base_inner,
            "selected_inner_event_nll": best,
            "selected_inner_gain": base_inner - best,
            "selected_step": best_step,
            "steps_run": len(history),
            "selected_at_budget_edge": best_step == train_cfg.max_steps,
            "stopped_by_patience": len(history) < train_cfg.max_steps,
            "max_gradient_l2_before_clip": max_grad,
            "parameter_l2_initial": initial_l2,
            "parameter_l2_selected": selected_l2,
            "history": history,
            "config": asdict(train_cfg),
        },
        "resources": {
            "device": str(device), "elapsed_seconds": elapsed,
            "peak_gpu_allocated_bytes": peak_gpu,
            "peak_gpu_allocated_mib": peak_gpu / (1024.0 ** 2),
            "peak_rss_bytes": peak_rss,
            "peak_rss_mib": peak_rss / (1024.0 ** 2),
            "recommended_jobs_per_gpu_before_review": 1,
        },
        "lease": {
            "path": str(lease_path), "sha256": file_hash(lease_path),
            "status": lease.get("status"),
        },
        "checkpoint": str(checkpoint_path),
        "source_commit": current_commit(),
    }
    atomic_write_json(card_path, card)
    return card
