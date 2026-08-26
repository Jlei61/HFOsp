"""R1.2b limited joint observer-state alignment on full recorded support.

Only the final spatial aggregation block of the Bridge observer is trainable.
The explicit projection, coordinate/shaft embeddings, raw tokenizer and raw
temporal Transformer stay frozen and are cached as per-contact node features.
The scientific likelihood, deterministic history baseline, state dimension and
development split are identical to R1.2.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import contract
from .bridge_e1 import make_paired_models
from .coverage import CoverageTable
from .history import DeterministicHistory, HistoryScaler, session_start_map
from .r1_2 import (
    R1_2_REVISION,
    FullAnchorDesign,
    FrozenEmbeddingStateModel,
    _bridge_scaler,
    _flow,
    _latest_anchor_source,
    _query_states,
    evaluate_full_t1,
    filtered_anchor_states,
    load_full_admissible_event_stream,
    load_full_design,
)


R1_2B_REVISION = "r1_2b_joint_last_spatial_observer_v1"
NODE_CACHE_REVISION = "r1_2b_frozen_upstream_contact_nodes_v1"
R1_2B_SUBJECTS = contract.BRIDGE_E1_SUBJECTS


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def _observer_nodes(observer: nn.Module, batch: dict[str, torch.Tensor],
                    *, include_raw: bool) -> tuple[torch.Tensor, torch.Tensor]:
    explicit = batch["explicit"]
    coordinates = batch["coordinates"]
    coordinate_valid = batch["coordinate_valid"]
    shaft_index = batch["shaft_index"]
    contact_mask = batch["contact_mask"].to(torch.bool)
    coordinate_input = torch.cat([
        torch.where(
            coordinate_valid.unsqueeze(-1), coordinates,
            torch.zeros_like(coordinates),
        ),
        coordinate_valid.to(coordinates.dtype).unsqueeze(-1),
    ], dim=-1)
    base = (
        observer.explicit(explicit)
        + observer.coordinate(coordinate_input)
        + observer.shaft(shaft_index)
    )
    base = torch.where(contact_mask.unsqueeze(-1), base, torch.zeros_like(base))
    if include_raw:
        raw = observer.raw(batch["waveform"], batch["sample_valid"])
        raw = torch.where(contact_mask.unsqueeze(-1), raw, torch.zeros_like(raw))
    else:
        raw = torch.zeros_like(base)
    return base, raw


def build_joint_node_cache(subject: str, *, device: torch.device | str = "cuda",
                           anchor_batch_size: int = 16,
                           r1_2_root: Path | None = None,
                           output_root: Path | None = None) -> dict:
    """Cache frozen upstream contact nodes for the fixed R1.2b subjects."""
    if subject not in R1_2B_SUBJECTS:
        raise ValueError(f"{subject}: not in frozen R1.2b pilot")
    r1_2_root = Path(r1_2_root or contract.RESULT_ROOT / "r1_2")
    output_root = Path(output_root or contract.RESULT_ROOT / "r1_2b")
    upstream_manifest_path = r1_2_root / "cache" / subject / "manifest.json"
    upstream_manifest = json.loads(upstream_manifest_path.read_text())
    if upstream_manifest.get("status") != "COMPLETE":
        raise ValueError(f"{subject}: incomplete R1.2 cache")
    if upstream_manifest.get("sealed_opened") is not False:
        raise ValueError(f"{subject}: upstream sealed flag is not false")
    design = load_full_design(Path(upstream_manifest["design"]))

    baseline_path = r1_2_root / "baselines" / subject / "seed_0/models.pt"
    bridge_dir = r1_2_root / "bridge_e1" / subject / "seed_0"
    bridge_result_path = bridge_dir / "result.json"
    bridge_checkpoint_path = bridge_dir / "models.pt"
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
    bridge_result = json.loads(bridge_result_path.read_text())
    if bridge_result.get("selected_epochs") != {
        "explicit": 0, "explicit_raw": 0
    }:
        raise ValueError(
            f"{subject}: R1.2b contract expects the audited epoch-zero Bridge"
        )
    coverage = CoverageTable.load(r1_2_root / "coverage" / f"{subject}.npz")
    stream = load_full_admissible_event_stream(subject, coverage)
    explicit_mean, explicit_scale, sampled, reader = _bridge_scaler(
        subject, baseline_path, bridge_result, stream, coverage
    )
    explicit_model, raw_model = make_paired_models(
        baseline, sampled, stream.adjacency, seed=0, device=device
    )
    checkpoint = torch.load(
        bridge_checkpoint_path, map_location=device, weights_only=False
    )
    explicit_model.load_state_dict(checkpoint["explicit"])
    raw_model.load_state_dict(checkpoint["explicit_raw"])
    explicit_model.eval(); raw_model.eval()
    for model in (explicit_model, raw_model):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    common_difference = 0.0
    explicit_state = explicit_model.observer.state_dict()
    raw_state = raw_model.observer.state_dict()
    for key, value in explicit_state.items():
        if key in raw_state and not key.startswith("raw.") and key != "raw_gain":
            common_difference = max(
                common_difference,
                float(torch.max(torch.abs(value.cpu() - raw_state[key].cpu())))
            )
    if common_difference > 1e-7:
        raise RuntimeError(f"{subject}: paired common observer state diverged")
    raw_gain = float(raw_model.observer.raw_gain.detach().cpu())
    if abs(raw_gain) > 1e-12:
        raise RuntimeError(f"{subject}: raw residual is not zero-initialised")

    from .r1_2 import _observer_batch

    base_rows: list[np.ndarray] = []
    raw_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    anchor_times: list[np.ndarray] = []
    with torch.inference_mode():
        for lo in range(0, len(design.anchor_time), int(anchor_batch_size)):
            hi = min(lo + int(anchor_batch_size), len(design.anchor_time))
            observations = [reader.read(float(value)) for value in design.anchor_time[lo:hi]]
            if any(value is None for value in observations):
                raise RuntimeError(
                    f"{subject}: a denominator-locked full anchor became unreadable"
                )
            observations = list(observations)
            explicit = np.stack([value.explicit for value in observations])
            explicit = ((explicit - explicit_mean) / explicit_scale).astype(np.float32)
            batch = _observer_batch(observations, explicit, device=device)
            base, raw = _observer_nodes(raw_model.observer, batch, include_raw=True)
            base_rows.append(base.cpu().numpy().astype(np.float32))
            raw_rows.append(raw.cpu().numpy().astype(np.float32))
            mask_rows.append(batch["contact_mask"].cpu().numpy().astype(np.uint8))
            anchor_times.append(design.anchor_time[lo:hi])
            if hi == len(design.anchor_time) or (
                hi // int(anchor_batch_size)
            ) % 100 == 0:
                print(
                    f"{subject}: cached {hi}/{len(design.anchor_time)} anchors",
                    flush=True,
                )
    base_array = np.concatenate(base_rows)
    raw_array = np.concatenate(raw_rows)
    mask_array = np.concatenate(mask_rows)
    time_array = np.concatenate(anchor_times)
    if not np.array_equal(time_array, design.anchor_time):
        raise RuntimeError(f"{subject}: node cache anchor order changed")
    if not np.isfinite(base_array).all() or not np.isfinite(raw_array).all():
        raise RuntimeError(f"{subject}: non-finite observer node cache")

    output = output_root / "cache" / subject
    base_path = output / "base_contact_node.npy"
    raw_path = output / "raw_contact_node.npy"
    mask_path = output / "contact_mask.npy"
    _atomic_npy(base_path, base_array)
    _atomic_npy(raw_path, raw_array)
    _atomic_npy(mask_path, mask_array)
    manifest = {
        "status": "COMPLETE",
        "contract": contract.REVISION,
        "r1_2_revision": R1_2_REVISION,
        "r1_2b_revision": R1_2B_REVISION,
        "cache_revision": NODE_CACHE_REVISION,
        "subject": subject,
        "n_anchors": int(len(time_array)),
        "n_train_anchors": int(np.sum(design.anchor_split == 0)),
        "n_validation_anchors": int(np.sum(design.anchor_split == 1)),
        "n_contacts": int(base_array.shape[1]),
        "node_dim": int(base_array.shape[2]),
        "base_contact_node": str(base_path),
        "base_contact_node_sha256": contract.sha256_file(base_path),
        "raw_contact_node": str(raw_path),
        "raw_contact_node_sha256": contract.sha256_file(raw_path),
        "contact_mask": str(mask_path),
        "contact_mask_sha256": contract.sha256_file(mask_path),
        "upstream_cache_manifest": str(upstream_manifest_path),
        "upstream_cache_manifest_sha256": contract.sha256_file(upstream_manifest_path),
        "bridge_checkpoint": str(bridge_checkpoint_path),
        "bridge_checkpoint_sha256": contract.sha256_file(bridge_checkpoint_path),
        "bridge_selected_epochs": bridge_result["selected_epochs"],
        "paired_common_state_max_abs_difference": common_difference,
        "initial_raw_gain": raw_gain,
        "frozen_upstream_components": [
            "explicit_projection", "coordinate_projection", "shaft_embedding",
            "raw_tokenizer", "raw_temporal_transformer",
        ],
        "trainable_observer_component": "last_spatial_aggregation_block",
        "full_recorded_support": True,
        "sealed_opened": False,
    }
    contract.atomic_json(output / "manifest.json", manifest)
    return manifest


def load_joint_node_cache(subject: str, *, output_root: Path | None = None
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    output_root = Path(output_root or contract.RESULT_ROOT / "r1_2b")
    root = output_root / "cache" / subject
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "COMPLETE" or manifest.get("sealed_opened") is not False:
        raise ValueError(f"{subject}: invalid R1.2b cache")
    if manifest.get("r1_2b_revision") != R1_2B_REVISION:
        raise ValueError(f"{subject}: R1.2b cache revision mismatch")
    arrays = []
    for field, hash_field in (
        ("base_contact_node", "base_contact_node_sha256"),
        ("raw_contact_node", "raw_contact_node_sha256"),
        ("contact_mask", "contact_mask_sha256"),
    ):
        path = Path(manifest[field])
        if contract.sha256_file(path) != manifest[hash_field]:
            raise ValueError(f"{subject}: {field} hash mismatch")
        arrays.append(np.load(path, mmap_mode="r"))
    base, raw, mask = arrays
    expected = (manifest["n_anchors"], manifest["n_contacts"], manifest["node_dim"])
    if base.shape != expected or raw.shape != expected:
        raise ValueError(f"{subject}: node cache shape mismatch")
    if mask.shape != expected[:2]:
        raise ValueError(f"{subject}: contact mask shape mismatch")
    return base, raw, mask, manifest


class LastSpatialObserver(nn.Module):
    """The only trainable observer block in R1.2b."""

    def __init__(self, source_observer: nn.Module, *, raw_enabled: bool):
        super().__init__()
        self.raw_enabled = bool(raw_enabled)
        self.pool_token = nn.Parameter(source_observer.pool_token.detach().clone())
        self.spatial = copy.deepcopy(source_observer.spatial)
        self.output_norm = copy.deepcopy(source_observer.output_norm)
        self.raw_gain = nn.Parameter(
            source_observer.raw_gain.detach().clone(), requires_grad=self.raw_enabled
        )

    def forward(self, base_node: torch.Tensor, raw_node: torch.Tensor,
                contact_mask: torch.Tensor) -> torch.Tensor:
        if base_node.shape != raw_node.shape or base_node.ndim != 3:
            raise ValueError("cached contact nodes must have shape (B,C,D)")
        if contact_mask.shape != base_node.shape[:2]:
            raise ValueError("cached contact mask shape mismatch")
        node = base_node
        if self.raw_enabled:
            node = node + self.raw_gain * raw_node
        mask = contact_mask.to(torch.bool)
        node = torch.where(mask.unsqueeze(-1), node, torch.zeros_like(node))
        pool = self.pool_token.expand(len(node), -1, -1)
        sequence = torch.cat([pool, node], dim=1)
        padding = torch.cat([
            torch.zeros((len(node), 1), dtype=torch.bool, device=node.device),
            ~mask,
        ], dim=1)
        encoded = self.spatial(sequence, src_key_padding_mask=padding)
        return self.output_norm(encoded[:, 0])


class JointLastLayerStateModel(FrozenEmbeddingStateModel):
    """R1.2 state model plus the limited trainable observer tail."""

    def __init__(self, baseline_checkpoint: dict, history_dim: int,
                 n_contacts: int, adjacency: np.ndarray,
                 source_observer: nn.Module, *, raw_enabled: bool,
                 state_dim: int = 8):
        super().__init__(
            baseline_checkpoint, history_dim, n_contacts, adjacency,
            observation_dim=64, state_dim=state_dim,
        )
        self.last_observer = LastSpatialObserver(
            source_observer, raw_enabled=raw_enabled
        )
        self.raw_enabled = bool(raw_enabled)

    def observation_embedding(self, base_node: torch.Tensor,
                              raw_node: torch.Tensor,
                              contact_mask: torch.Tensor) -> torch.Tensor:
        return self.last_observer(base_node, raw_node, contact_mask)


def materialize_joint_embedding(model: JointLastLayerStateModel,
                                base_node: np.ndarray, raw_node: np.ndarray,
                                contact_mask: np.ndarray, *,
                                device: torch.device | str,
                                batch_size: int = 2048) -> np.ndarray:
    model.eval()
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for lo in range(0, len(base_node), int(batch_size)):
            hi = min(lo + int(batch_size), len(base_node))
            rows.append(model.observation_embedding(
                torch.as_tensor(np.array(base_node[lo:hi], copy=True), device=device),
                torch.as_tensor(np.array(raw_node[lo:hi], copy=True), device=device),
                torch.as_tensor(np.array(contact_mask[lo:hi], copy=True), device=device),
            ).cpu().numpy().astype(np.float32))
    return np.concatenate(rows)


def evaluate_joint(model: JointLastLayerStateModel, design: FullAnchorDesign,
                   base_node: np.ndarray, raw_node: np.ndarray,
                   contact_mask: np.ndarray, split: str, *,
                   device: torch.device | str, **kwargs):
    embedding = materialize_joint_embedding(
        model, base_node, raw_node, contact_mask, device=device
    )
    return evaluate_full_t1(model, design, embedding, split, device=device, **kwargs)


def _grouped_rows(rows: np.ndarray, source: np.ndarray,
                  n_anchor: int) -> tuple[np.ndarray, np.ndarray]:
    if not len(rows):
        return rows, np.zeros(n_anchor + 1, dtype=np.int64)
    order = np.argsort(source[rows], kind="stable")
    sorted_rows = rows[order]
    count = np.bincount(source[sorted_rows], minlength=n_anchor)
    return sorted_rows, np.concatenate([[0], np.cumsum(count, dtype=np.int64)])


def _train_joint_epoch(model: JointLastLayerStateModel,
                       design: FullAnchorDesign,
                       base_node: np.ndarray, raw_node: np.ndarray,
                       contact_mask: np.ndarray,
                       optimizer: torch.optim.Optimizer, *,
                       device: torch.device | str, anchor_ids: np.ndarray,
                       query_time_upper: float, chunk_anchors: int = 256) -> None:
    selected = np.zeros(len(design.anchor_time), dtype=bool)
    selected[np.asarray(anchor_ids, dtype=np.int64)] = True
    event_allowed = np.flatnonzero(
        (design.event_split == 0) & (design.event_time < float(query_time_upper))
        & (design.event_source_anchor >= 0)
        & selected[np.maximum(design.event_source_anchor, 0)]
    )
    q_allowed = np.flatnonzero(
        (design.quadrature_split == 0)
        & (design.quadrature_time < float(query_time_upper))
        & (design.quadrature_source_anchor >= 0)
        & selected[np.maximum(design.quadrature_source_anchor, 0)]
    )
    event_sorted, event_boundary = _grouped_rows(
        event_allowed, design.event_source_anchor, len(design.anchor_time)
    )
    q_sorted, q_boundary = _grouped_rows(
        q_allowed, design.quadrature_source_anchor, len(design.anchor_time)
    )

    def rows_for(chunk: np.ndarray, sorted_rows: np.ndarray,
                 boundary: np.ndarray) -> np.ndarray:
        pieces = [
            sorted_rows[boundary[int(anchor)]:boundary[int(anchor) + 1]]
            for anchor in chunk
            if boundary[int(anchor) + 1] > boundary[int(anchor)]
        ]
        return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)

    event_total = int(len(event_allowed))
    chunks = max(int(math.ceil(len(anchor_ids) / max(int(chunk_anchors), 1))), 1)
    scale = float(chunks) / max(event_total, 1)
    for label in design.session_label:
        anchors = np.flatnonzero(selected & (design.anchor_session == label))
        if not len(anchors):
            continue
        anchors = anchors[np.argsort(design.anchor_time[anchors], kind="stable")]
        state = torch.zeros(model.state.dim, device=device)
        cursor = float(design.session_start_for(np.asarray([label]))[0])
        for lo in range(0, len(anchors), int(chunk_anchors)):
            chunk = anchors[lo:lo + int(chunk_anchors)]
            observation = model.observation_embedding(
                torch.as_tensor(np.array(base_node[chunk], copy=True), device=device),
                torch.as_tensor(np.array(raw_node[chunk], copy=True), device=device),
                torch.as_tensor(np.array(contact_mask[chunk], copy=True), device=device),
            )
            transition_cache: dict[float, torch.Tensor] = {}
            chunk_states = []
            for local, anchor in enumerate(chunk):
                time = float(design.anchor_time[anchor])
                state_minus = _flow(
                    model, state, (time - cursor) / 60.0, transition_cache
                )
                state = model.state.correction(
                    state_minus, observation[local], enabled=True
                )
                chunk_states.append(state)
                cursor = time
            chunk_states_tensor = torch.stack(chunk_states)
            local_lookup = np.full(len(design.anchor_time), -1, dtype=np.int64)
            local_lookup[chunk] = np.arange(len(chunk), dtype=np.int64)
            event_rows = rows_for(chunk, event_sorted, event_boundary)
            q_rows = rows_for(chunk, q_sorted, q_boundary)
            if not len(event_rows) and not len(q_rows):
                state = state.detach()
                continue
            event_source = local_lookup[design.event_source_anchor[event_rows]]
            q_source = local_lookup[design.quadrature_source_anchor[q_rows]]
            event_source_state = chunk_states_tensor[
                torch.as_tensor(event_source, dtype=torch.long, device=device)
            ]
            q_source_state = chunk_states_tensor[
                torch.as_tensor(q_source, dtype=torch.long, device=device)
            ]
            event_delta = torch.as_tensor(
                (design.event_time[event_rows]
                 - design.anchor_time[design.event_source_anchor[event_rows]]) / 60.0,
                dtype=event_source_state.dtype, device=device,
            ).clamp(min=0.0)
            q_delta = torch.as_tensor(
                (design.quadrature_time[q_rows]
                 - design.anchor_time[design.quadrature_source_anchor[q_rows]]) / 60.0,
                dtype=q_source_state.dtype, device=device,
            ).clamp(min=0.0)
            matrix = model.state.generator.matrix().to(event_source_state.dtype)
            if len(event_rows):
                transition = torch.matrix_exp(
                    matrix.unsqueeze(0) * event_delta[:, None, None]
                )
                mu = model.state.generator.mu
                event_state = mu.unsqueeze(0) + torch.matmul(
                    transition, (event_source_state - mu).unsqueeze(-1)
                ).squeeze(-1)
                event_history = torch.as_tensor(
                    design.event_history[event_rows], device=device
                )
                event_log = model.timing_log_rate(event_history, event_state).sum()
                mark = model.mark_terms(
                    event_history, event_state,
                    torch.as_tensor(
                        design.event_group_ids[event_rows], dtype=torch.long,
                        device=device,
                    ),
                    torch.as_tensor(
                        design.event_group_count[event_rows], dtype=torch.long,
                        device=device,
                    ),
                )
                mark_log = mark.event_log_prob.sum()
            else:
                event_log = state.new_zeros(()); mark_log = state.new_zeros(())
            if len(q_rows):
                transition = torch.matrix_exp(
                    matrix.unsqueeze(0) * q_delta[:, None, None]
                )
                mu = model.state.generator.mu
                q_state = mu.unsqueeze(0) + torch.matmul(
                    transition, (q_source_state - mu).unsqueeze(-1)
                ).squeeze(-1)
                q_history = torch.as_tensor(
                    design.quadrature_history[q_rows], device=device
                )
                q_log = model.timing_log_rate(q_history, q_state)
                weight = torch.as_tensor(
                    design.quadrature_weight_seconds[q_rows],
                    dtype=q_log.dtype, device=device,
                )
                survival = torch.sum(
                    weight * torch.exp(torch.clamp(q_log, max=20.0))
                )
            else:
                survival = state.new_zeros(())
            loss = (survival - event_log - mark_log) * scale
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            state = state.detach()


def _optimizer(model: JointLastLayerStateModel, *, state_learning_rate: float,
               observer_learning_rate: float) -> torch.optim.Optimizer:
    observer = list(model.last_observer.parameters())
    observer_ids = {id(value) for value in observer}
    state = [
        value for value in model.parameters()
        if value.requires_grad and id(value) not in observer_ids
    ]
    return torch.optim.AdamW([
        {"params": state, "lr": float(state_learning_rate)},
        {"params": [value for value in observer if value.requires_grad],
         "lr": float(observer_learning_rate)},
    ], weight_decay=1e-3)


def fit_joint_t1(model: JointLastLayerStateModel, design: FullAnchorDesign,
                 base_node: np.ndarray, raw_node: np.ndarray,
                 contact_mask: np.ndarray, *, device: torch.device | str,
                 epochs: int = 4, state_learning_rate: float = 3e-4,
                 observer_learning_rate: float = 3e-5,
                 chunk_anchors: int = 256) -> JointLastLayerStateModel:
    if not math.isclose(
        observer_learning_rate, 0.1 * state_learning_rate,
        rel_tol=1e-9, abs_tol=1e-12,
    ):
        raise ValueError("R1.2b observer LR must be exactly 0.1 x state LR")
    train = design.anchor_ids("train")
    if len(train) < 10:
        raise ValueError("R1.2b needs at least ten TRAIN anchors")
    cut = int(np.clip(math.floor(0.8 * len(train)), 1, len(train) - 1))
    inner_train = train[:cut]
    boundary = float(design.anchor_time[train[cut]])
    train_end, _ = contract.load_split(design.subject)
    initial = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    best_epoch = 0
    best_value = evaluate_joint(
        model, design, base_node, raw_node, contact_mask, "train",
        device=device, time_lower=boundary, time_upper=train_end,
    ).joint_nll_per_event
    optimizer = _optimizer(
        model, state_learning_rate=state_learning_rate,
        observer_learning_rate=observer_learning_rate,
    )
    for epoch in range(1, int(epochs) + 1):
        model.train()
        _train_joint_epoch(
            model, design, base_node, raw_node, contact_mask, optimizer,
            device=device, anchor_ids=inner_train, query_time_upper=boundary,
            chunk_anchors=chunk_anchors,
        )
        value = evaluate_joint(
            model, design, base_node, raw_node, contact_mask, "train",
            device=device, time_lower=boundary, time_upper=train_end,
        ).joint_nll_per_event
        if value < best_value:
            best_value = value; best_epoch = epoch

    model.load_state_dict(initial)
    if best_epoch:
        optimizer = _optimizer(
            model, state_learning_rate=state_learning_rate,
            observer_learning_rate=observer_learning_rate,
        )
        for _ in range(best_epoch):
            model.train()
            _train_joint_epoch(
                model, design, base_node, raw_node, contact_mask, optimizer,
                device=device, anchor_ids=train, query_time_upper=train_end,
                chunk_anchors=chunk_anchors,
            )
    model.selected_epochs = int(best_epoch)
    model.inner_validation_joint_nll = float(best_value)
    model.truncated_bptt_anchors = int(chunk_anchors)
    model.state_learning_rate = float(state_learning_rate)
    model.observer_learning_rate = float(observer_learning_rate)
    return model.eval()


@dataclass(frozen=True)
class HorizonMetrics:
    joint_nll_per_event: float
    timing_nll_per_event: float
    mark_nll_per_event: float
    group_size_nll_per_event: float
    subset_nll_per_event: float
    n_events: int
    n_start_anchors: int
    n_unique_events: int
    recorded_seconds_window_weighted: float


def _score_states(model: JointLastLayerStateModel, history: np.ndarray,
                  state: torch.Tensor, group_ids: np.ndarray,
                  group_count: np.ndarray, q_history: np.ndarray,
                  q_state: torch.Tensor, q_weight: np.ndarray, *,
                  device: torch.device | str) -> HorizonMetrics:
    event_history = torch.as_tensor(history, device=device)
    event_log = model.timing_log_rate(event_history, state).sum()
    mark = model.mark_terms(
        event_history, state,
        torch.as_tensor(group_ids, dtype=torch.long, device=device),
        torch.as_tensor(group_count, dtype=torch.long, device=device),
    )
    if len(q_history):
        q_log = model.timing_log_rate(
            torch.as_tensor(q_history, device=device), q_state
        )
        survival = torch.sum(
            torch.as_tensor(q_weight, dtype=q_log.dtype, device=device)
            * torch.exp(torch.clamp(q_log, max=20.0))
        )
    else:
        survival = event_log.new_zeros(())
    denominator = max(len(history), 1)
    timing = float((survival - event_log) / denominator)
    mark_nll = float(-mark.event_log_prob.sum() / denominator)
    return HorizonMetrics(
        joint_nll_per_event=timing + mark_nll,
        timing_nll_per_event=timing,
        mark_nll_per_event=mark_nll,
        group_size_nll_per_event=float(
            -mark.group_size_log_prob.sum() / denominator
        ),
        subset_nll_per_event=float(-mark.subset_log_prob.sum() / denominator),
        n_events=int(len(history)), n_start_anchors=0, n_unique_events=0,
        recorded_seconds_window_weighted=float(np.sum(q_weight)),
    )


def horizon_correction_off(model: JointLastLayerStateModel,
                           design: FullAnchorDesign,
                           embedding: np.ndarray,
                           stream, coverage: CoverageTable,
                           baseline_checkpoint: dict, *,
                           horizons: tuple[int, ...] = (5, 10, 20),
                           max_start_anchors: int = 64,
                           device: torch.device | str = "cuda") -> dict:
    """Exact event-observed correction-off diagnostics after validation anchors.

    The same deterministic set of anchors eligible for the largest horizon is
    used for all nested horizons. Gauss quadrature is rebuilt on the intersection
    with recorded validation coverage, split at every observed event.
    """
    model.eval()
    horizons = tuple(sorted(int(value) for value in horizons))
    if not horizons or horizons[0] < 1:
        raise ValueError("positive event horizons are required")
    with torch.no_grad():
        filtered_anchor = filtered_anchor_states(
            model, design, embedding, device=device
        )
    validation_anchors = design.anchor_ids("validation")
    max_h = max(horizons)
    eligible: list[tuple[int, np.ndarray]] = []
    for anchor in validation_anchors:
        future = np.flatnonzero(
            (design.event_split == 1)
            & (design.event_session == design.anchor_session[anchor])
            & (design.event_time > design.anchor_time[anchor])
        )
        if len(future) >= max_h:
            eligible.append((int(anchor), future[:max_h]))
    if not eligible:
        return {
            "status": "NO_ELIGIBLE_STARTS", "horizons": {},
            "max_start_anchors": int(max_start_anchors),
        }
    if len(eligible) > int(max_start_anchors):
        take = np.unique(np.linspace(
            0, len(eligible) - 1, int(max_start_anchors), dtype=np.int64
        ))
        eligible = [eligible[int(index)] for index in take]

    scaler = HistoryScaler(
        mean=np.asarray(baseline_checkpoint["history_scaler"]["mean"], dtype=np.float32),
        scale=np.asarray(baseline_checkpoint["history_scaler"]["scale"], dtype=np.float32),
    )
    history_engine = DeterministicHistory(
        stream, session_start_map(stream, coverage.session, coverage.start)
    )
    node, weight = np.polynomial.legendre.leggauss(4)
    output = {}
    with torch.no_grad():
        for horizon in horizons:
            accum = {
                name: {"event_log": 0.0, "mark_log": 0.0, "size_log": 0.0,
                       "subset_log": 0.0, "survival": 0.0}
                for name in ("filtered", "correction_off")
            }
            total_events = 0
            total_seconds = 0.0
            unique_events: set[int] = set()
            for anchor, future_max in eligible:
                event_rows = future_max[:horizon]
                cutoff = float(design.event_time[event_rows[-1]])
                start = float(design.anchor_time[anchor])
                label = int(design.anchor_session[anchor])
                cover_left, cover_right, cover_label = coverage.split_segments_with_session(
                    "validation"
                )
                q_times: list[np.ndarray] = []
                q_weights: list[np.ndarray] = []
                for left, right, session in zip(cover_left, cover_right, cover_label):
                    if int(session) != label:
                        continue
                    a = max(float(left), start)
                    b = min(float(right), cutoff)
                    if b <= a:
                        continue
                    internal = design.event_time[event_rows]
                    internal = internal[(internal > a) & (internal < b)]
                    boundary = np.concatenate([[a], internal, [b]])
                    width = np.diff(boundary)
                    midpoint = 0.5 * (boundary[:-1] + boundary[1:])
                    half = 0.5 * width
                    q_times.append(
                        (midpoint[:, None] + half[:, None] * node[None, :]).reshape(-1)
                    )
                    q_weights.append((half[:, None] * weight[None, :]).reshape(-1))
                q_time = np.concatenate(q_times) if q_times else np.empty(0, dtype=np.float64)
                q_weight = np.concatenate(q_weights) if q_weights else np.empty(0, dtype=np.float64)
                q_session = np.full(len(q_time), label, dtype=np.int64)
                q_history = scaler.transform(history_engine.evaluate(q_time, q_session))
                q_source = _latest_anchor_source(
                    q_time, q_session, design.anchor_time, design.anchor_session
                )
                filtered_event_state = _query_states(
                    model, design, filtered_anchor, design.event_source_anchor,
                    design.event_time, design.event_session, event_rows,
                    state_permutation=None, device=device,
                )
                filtered_q_state = _query_states(
                    model, design, filtered_anchor, q_source, q_time, q_session,
                    np.arange(len(q_time), dtype=np.int64),
                    state_permutation=None, device=device,
                )
                start_state = filtered_anchor[anchor]
                off_event_state = model.state.generator.from_anchor(
                    start_state,
                    torch.as_tensor(
                        (design.event_time[event_rows] - start) / 60.0,
                        dtype=start_state.dtype, device=device,
                    ),
                )
                off_q_state = model.state.generator.from_anchor(
                    start_state,
                    torch.as_tensor(
                        (q_time - start) / 60.0,
                        dtype=start_state.dtype, device=device,
                    ),
                )
                for name, event_state, q_state in (
                    ("filtered", filtered_event_state, filtered_q_state),
                    ("correction_off", off_event_state, off_q_state),
                ):
                    event_history = torch.as_tensor(
                        design.event_history[event_rows], device=device
                    )
                    event_log = model.timing_log_rate(
                        event_history, event_state
                    ).sum()
                    mark = model.mark_terms(
                        event_history, event_state,
                        torch.as_tensor(
                            design.event_group_ids[event_rows], dtype=torch.long,
                            device=device,
                        ),
                        torch.as_tensor(
                            design.event_group_count[event_rows], dtype=torch.long,
                            device=device,
                        ),
                    )
                    if len(q_time):
                        q_log = model.timing_log_rate(
                            torch.as_tensor(q_history, device=device), q_state
                        )
                        survival = torch.sum(
                            torch.as_tensor(q_weight, dtype=q_log.dtype, device=device)
                            * torch.exp(torch.clamp(q_log, max=20.0))
                        )
                    else:
                        survival = event_log.new_zeros(())
                    target = accum[name]
                    target["event_log"] += float(event_log)
                    target["mark_log"] += float(mark.event_log_prob.sum())
                    target["size_log"] += float(mark.group_size_log_prob.sum())
                    target["subset_log"] += float(mark.subset_log_prob.sum())
                    target["survival"] += float(survival)
                total_events += len(event_rows)
                unique_events.update(int(value) for value in event_rows)
                total_seconds += float(q_weight.sum())

            metrics = {}
            for name, value in accum.items():
                denominator = max(total_events, 1)
                timing = (value["survival"] - value["event_log"]) / denominator
                mark_nll = -value["mark_log"] / denominator
                metrics[name] = asdict(HorizonMetrics(
                    joint_nll_per_event=timing + mark_nll,
                    timing_nll_per_event=timing,
                    mark_nll_per_event=mark_nll,
                    group_size_nll_per_event=-value["size_log"] / denominator,
                    subset_nll_per_event=-value["subset_log"] / denominator,
                    n_events=int(total_events),
                    n_start_anchors=int(len(eligible)),
                    n_unique_events=int(len(unique_events)),
                    recorded_seconds_window_weighted=float(total_seconds),
                ))
            output[str(horizon)] = {
                **metrics,
                "correction_off_minus_filtered": {
                    key: metrics["correction_off"][key] - metrics["filtered"][key]
                    for key in (
                        "joint_nll_per_event", "timing_nll_per_event",
                        "mark_nll_per_event", "group_size_nll_per_event",
                        "subset_nll_per_event",
                    )
                },
            }
    return {
        "status": "COMPLETE",
        "horizons": output,
        "largest_horizon_eligibility_shared": True,
        "max_start_anchors": int(max_start_anchors),
        "start_anchor_selection": "chronological_even_spacing_without_outcome_selection",
        "future_event_history_teacher_forced": True,
        "future_observation_correction_off": True,
        "quadrature_order": 4,
        "recorded_gaps_excluded": True,
        "overlapping_windows_supportive_only": True,
    }
