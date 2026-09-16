"""Frozen-decoder H2a: does pre-event S_obs change within-event grammar?"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor

from src.topic5_group_event_state.v034_spatial_state.we_decoder import (
    event_batch,
    load_frozen_decoder,
)
from src.topic5_group_event_state.v035.stepwise_decoder import (
    StepwiseAdapterConfig,
    StepwiseConditionedDecoder,
)
from src.topic5_wiring_economy_rnn import build_event_tensors

from .contracts import atomic_json
from .ctssm import DualStreamEventCTSSM, dual_stream_features_at_queries
from .baselines import fixed_mark_ewma, mark_ewma_features_at_queries
from .h1_data import H1SubjectData, build_h1_subject_data
from .h1_train import (
    EventStateComputer,
    GridEventStateComputer,
    H1TrainConfig,
    NestedH1Readout,
    _block_shift,
    _code_provenance,
    _fixed_features,
    _selection_score,
    _standardise_features,
    _target_bundle,
    _train_stage as _train_h1_stage,
)
from .h2a_marks import (
    fit_rich_mark_readout,
    leaked_event_context,
    mean_squared_mark,
    prefix_hidden,
)


RANK_DATA_ROOT = Path(
    "/home/honglab/leijiaxin/HFOsp/results/topic5_interictal_rank_distribution/dataset_v0_4/per_subject"
)
DECODER_ROOT = Path("/data/hfosp_group_event_state_v0_3_7/decoder_strict")
H1_ROOT = Path("/data/hfosp_group_event_state_v0_3_7/h1_shared_equal_horizon")


@dataclass(frozen=True)
class H2ATrainConfig:
    batch_size: int = 512
    max_epochs_static: int = 80
    max_epochs_state: int = 120
    max_epochs_oracle: int = 80
    patience_epochs: int = 12
    # The state arm starts from a cold low-rank interface while the frozen
    # decoder already predicts well, so it needs its own patience and warm-up;
    # sharing the 12-epoch patience stopped it at the origin in three of four
    # pilot patients and that exact zero was then read as "state has no effect".
    #
    # The output maps stay at zero so the arm starts in exact parity with the
    # frozen decoder and can only win or tie: a non-zero start would perturb
    # the decoder and bias the nested contrast against the state, which is the
    # mirror image of the bug being fixed.  Only ``down`` is blind on the very
    # first optimiser step; the output maps receive gradient immediately, so
    # the whole path is live from the second step of the first epoch.
    patience_epochs_state: int = 40
    warmup_epochs_state: int = 5
    state_modulation_init_std: float = 0.0
    learning_rate_static: float = 1e-3
    learning_rate_state: float = 8e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    modulation_rank: int = 8
    minimum_shift_seconds: float = 2.0 * 3600.0
    minimum_same_prefix_events: int = 5
    seed: int = 20260903


@dataclass(frozen=True)
class H2AJointTrainConfig:
    max_steps: int = 240
    validate_every: int = 10
    patience_checks: int = 12
    fit_events_per_step: int = 4096
    batch_size: int = 512
    learning_rate_adapter: float = 5e-4
    learning_rate_observer: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    seed: int = 20260903


def _densify(groups: np.ndarray) -> np.ndarray:
    value = np.asarray(groups, dtype=np.int16)
    out = np.full_like(value, -1)
    for row_index, row in enumerate(value):
        present = np.unique(row[row >= 0])
        mapping = {int(old): int(new) for new, old in enumerate(present)}
        for contact, old in enumerate(row):
            if old >= 0: out[row_index, contact] = mapping[int(old)]
    return out


def _phase(time: np.ndarray, bounds: dict[str, float]) -> np.ndarray:
    out = np.full(time.shape, "OUTSIDE", dtype="<U12")
    out[(time >= bounds["20pct"]) & (time < bounds["60pct"])] = "FIT"
    out[(time >= bounds["60pct"]) & (time < bounds["70pct"])] = "INNER"
    out[(time >= bounds["70pct"]) & (time < bounds["80pct"])] = "SELECTION"
    return out


def _decoder_seed(state_seed: int) -> int:
    return int((int(state_seed) - 20260903) % 3)


def _state_at_events(
    data: H1SubjectData,
    observer: DualStreamEventCTSSM,
    query_time: np.ndarray,
    query_segment: np.ndarray,
    device: torch.device,
) -> Tensor:
    chunks, rows = [], []
    for segment in np.unique(query_segment):
        er = np.flatnonzero(data.event_segment == segment)
        qr = np.flatnonzero(query_segment == segment)
        if er.size == 0 or qr.size == 0: continue
        et = torch.as_tensor(data.event_time[er], dtype=torch.float64, device=device)
        qt = torch.as_tensor(query_time[qr], dtype=torch.float64, device=device)
        output = observer(
            et,
            torch.as_tensor(data.burden_mark[er], dtype=torch.float32, device=device),
            torch.as_tensor(data.grammar_mark[er], dtype=torch.float32, device=device),
        )
        chunks.append(dual_stream_features_at_queries(output, et, qt))
        rows.append(torch.as_tensor(qr, dtype=torch.long, device=device))
    row = torch.cat(rows); value = torch.cat(chunks)
    return value[torch.argsort(row)]


def _bmark_at_events(
    data: H1SubjectData,
    query_time: np.ndarray,
    query_segment: np.ndarray,
    taus_seconds: tuple[float, ...],
    centre: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
) -> Tensor:
    """Transparent marked-history baseline at strictly pre-event times."""

    chunks, rows = [], []
    for segment in np.unique(query_segment):
        er = np.flatnonzero(data.event_segment == segment)
        qr = np.flatnonzero(query_segment == segment)
        if er.size == 0 or qr.size == 0:
            continue
        event_time = torch.as_tensor(data.event_time[er], dtype=torch.float64, device=device)
        output = fixed_mark_ewma(
            event_time,
            torch.as_tensor(data.burden_mark[er], dtype=torch.float32, device=device),
            torch.as_tensor(data.grammar_mark[er], dtype=torch.float32, device=device),
            taus_seconds=taus_seconds,
        )
        chunks.append(mark_ewma_features_at_queries(
            output, event_time,
            torch.as_tensor(query_time[qr], dtype=torch.float64, device=device),
            taus_seconds=taus_seconds,
        ))
        rows.append(torch.as_tensor(qr, dtype=torch.long, device=device))
    row = torch.cat(rows); value = torch.cat(chunks)[torch.argsort(row)]
    c = torch.as_tensor(centre, dtype=value.dtype, device=device)
    s = torch.as_tensor(scale, dtype=value.dtype, device=device)
    return torch.clamp((value - c) / s, -12.0, 12.0)


def _load_eval_events(subject: str, data: H1SubjectData, contact_names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(RANK_DATA_ROOT / f"{subject}.npz", allow_pickle=False) as z:
        names = np.asarray(z["contact_names"]).astype(str)
        lookup = {name: i for i, name in enumerate(names)}
        try: columns = np.asarray([lookup[name] for name in contact_names], dtype=np.int64)
        except KeyError as error: raise ValueError(f"decoder contact absent from evaluation stream: {error}")
        time_all = np.asarray(z["event_abs_time"], dtype=np.float64)
        groups = np.asarray(z["event_group_ids"], dtype=np.int16)[:, columns]
    ranks = _densify(groups)
    n_part = (ranks >= 0).sum(1)
    n_group = np.asarray([len(np.unique(row[row >= 0])) for row in ranks])
    keep = (
        np.isfinite(time_all)
        & (time_all >= float(data.rate.phase_boundaries["20pct"]))
        & (time_all < float(data.rate.phase_boundaries["80pct"]))
        & (n_part >= 3) & (n_group >= 2)
    )
    rows = np.flatnonzero(keep)
    rows = rows[np.argsort(time_all[rows], kind="stable")]
    time = time_all[rows]; ranks = ranks[rows]
    segment = np.full(time.size, -1, dtype=np.int64)
    for lo, hi in data.rate.observed_support_bounds:
        parent = np.flatnonzero(
            (data.rate.segment_bounds[:, 0] <= float(lo) + 1e-9)
            & (data.rate.segment_bounds[:, 1] >= float(hi) - 1e-9)
        )
        if parent.size != 1: raise ValueError("evaluation support maps ambiguously")
        inside = (time >= float(lo)) & (time < float(hi))
        segment[inside] = int(parent[0])
    use = segment >= 0
    return time[use], ranks[use], segment[use]


def _score_batches(
    model: StepwiseConditionedDecoder,
    tensors: dict[str, Tensor],
    context: Tensor | None,
    rows: np.ndarray,
    *,
    use_static: bool,
    use_dynamic: bool,
    batch_size: int,
    observed_prefix_groups: int = 0,
) -> dict[str, Tensor]:
    values: dict[str, list[Tensor]] = {}
    for start in range(0, rows.size, batch_size):
        index = torch.as_tensor(rows[start:start + batch_size], dtype=torch.long, device=tensors["x"].device)
        batch = event_batch(tensors, index)
        state = None if context is None else context[index]
        scores = model.scores(batch, state, use_static=use_static, use_dynamic=use_dynamic,
                              observed_prefix_groups=observed_prefix_groups)
        for key, value in scores.items(): values.setdefault(key, []).append(value)
    return {key: torch.cat(parts) for key, parts in values.items()}


def _mean_scores(*args, **kwargs) -> dict[str, float]:
    return {key: float(value.mean().detach()) for key, value in _score_batches(*args, **kwargs).items()}


def _fit_adapter(
    model: StepwiseConditionedDecoder,
    tensors: dict[str, Tensor],
    context: Tensor | None,
    fit_rows: np.ndarray,
    inner_rows: np.ndarray,
    *,
    stage: str,
    config: H2ATrainConfig,
) -> dict[str, Any]:
    for parameter in model.static.parameters(): parameter.requires_grad_(stage == "static")
    for parameter in model.dynamic.parameters(): parameter.requires_grad_(stage == "state")
    params = list(model.static.parameters() if stage == "static" else model.dynamic.parameters())
    named = list((model.static if stage == 'static' else model.dynamic).named_parameters())
    initial_parameters = {name: value.detach().cpu().clone() for name, value in named}
    parameter_audit = {name: {'shape': list(value.shape), 'parameters': value.numel(),
                             'first_nonzero_step': None, 'nonzero_steps': 0,
                             'gradient_norm_max': 0.0, 'gradient_norm_sum': 0.0}
                       for name, value in named}
    optimizer_steps = 0
    lr = config.learning_rate_static if stage == "static" else config.learning_rate_state
    maximum = config.max_epochs_static if stage == "static" else config.max_epochs_state
    patience = config.patience_epochs if stage == "static" else config.patience_epochs_state
    warmup = 0 if stage == "static" else int(config.warmup_epochs_state)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=config.weight_decay)
    rng = np.random.default_rng(int(config.seed) + (0 if stage == "static" else 17))
    use_dynamic = stage == "state"
    with torch.no_grad():
        initial = _mean_scores(
            model, tensors, context, inner_rows, use_static=True,
            use_dynamic=use_dynamic, batch_size=config.batch_size,
        )["grammar"]
    best, best_epoch, stale = initial, 0, 0
    best_state = copy.deepcopy(model.state_dict())

    def _output_magnitude() -> float:
        return float(max(
            module.weight.abs().max()
            for module in (
                model.dynamic.gamma, model.dynamic.beta,
                model.dynamic.contact, model.dynamic.stop,
            )
        ))

    # Measured during training, before the best checkpoint is restored, so a
    # selected-at-origin result can be told apart from a dead gradient path.
    peak_output_magnitude = _output_magnitude()
    history = [{"epoch": 0, "inner_grammar": initial}]
    for epoch in range(1, maximum + 1):
        if warmup > 0:
            fraction = min(1.0, epoch / float(warmup))
            for group in optimizer.param_groups:
                group["lr"] = lr * fraction
        order = rng.permutation(fit_rows)
        losses = []
        for start in range(0, order.size, config.batch_size):
            rows = order[start:start + config.batch_size]
            index = torch.as_tensor(rows, dtype=torch.long, device=tensors["x"].device)
            batch = event_batch(tensors, index)
            state = None if context is None else context[index]
            optimizer.zero_grad(set_to_none=True)
            score = model.scores(batch, state, use_static=True, use_dynamic=use_dynamic)
            loss = score["grammar"].mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f'H2a {stage}: nonfinite FIT loss')
            loss.backward()
            optimizer_steps += 1
            norms = torch.stack([loss.new_zeros(()) if value.grad is None else value.grad.detach().norm()
                                 for _name, value in named]).cpu().tolist()
            for (name, _value), norm in zip(named, norms):
                if not np.isfinite(norm):
                    raise FloatingPointError(f'H2a {stage}: nonfinite gradient in {name}')
                row = parameter_audit[name]
                row['gradient_norm_sum'] += norm
                row['gradient_norm_max'] = max(row['gradient_norm_max'], norm)
                if norm > 0:
                    row['nonzero_steps'] += 1
                    if row['first_nonzero_step'] is None:
                        row['first_nonzero_step'] = optimizer_steps
            torch.nn.utils.clip_grad_norm_(params, config.gradient_clip); optimizer.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            inner = _mean_scores(
                model, tensors, context, inner_rows, use_static=True,
                use_dynamic=use_dynamic, batch_size=config.batch_size,
            )["grammar"]
        history.append({"epoch": epoch, "fit_grammar": float(np.mean(losses)), "inner_grammar": inner})
        peak_output_magnitude = max(peak_output_magnitude, _output_magnitude())
        if np.isfinite(inner) and inner < best - 1e-5:
            best, best_epoch, stale = inner, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else: stale += 1
        if stale >= patience: break
    model.load_state_dict(best_state)
    for name, value in named:
        delta = value.detach().cpu() - initial_parameters[name]
        parameter_audit[name].update(selected_delta_max_abs=float(delta.abs().max()),
                                    selected_delta_l2=float(delta.norm()),
                                    gradient_norm_mean=parameter_audit[name]['gradient_norm_sum'] / max(optimizer_steps, 1))
    return {
        "stage": stage, "initial_inner_grammar": initial, "best_inner_grammar": best,
        'parameter_audit': parameter_audit, 'optimizer_steps': optimizer_steps,
        'optimizer_audit': {'name': 'AdamW', 'initialization': 'registered decoder-seeded module construction',
                            'learning_rate': lr, 'betas': [0.9, 0.999], 'epsilon': 1e-8,
                            'weight_decay': config.weight_decay, 'batch_size': config.batch_size,
                            'gradient_clip': config.gradient_clip},
        "selected_epoch": best_epoch, "epochs_run": history[-1]["epoch"],
        "selected_at_init": best_epoch == 0, "history": history,
        "patience_epochs": int(patience), "warmup_epochs": int(warmup),
        "training_budget_exhausted": bool(history[-1]['epoch'] == maximum and stale < patience),
        "terminated_by_patience": bool(stale >= patience),
        "modulation_output_init_std": float(model.dynamic.config.output_init_std),
        # Distinguishes "trained and did not help" from "never moved".  The
        # peak is taken during training; the selected checkpoint may still be
        # the origin because no epoch beat zero modulation on the inner split.
        "peak_modulation_magnitude_during_training": (
            peak_output_magnitude if use_dynamic else None
        ),
        "adapter_moved_during_training": (
            bool(peak_output_magnitude > 0.0) if use_dynamic else None
        ),
        "selected_checkpoint_is_the_origin": (
            bool(_output_magnitude() == 0.0) if use_dynamic else None
        ),
    }


def _shift_context(context: Tensor, time: np.ndarray, segment: np.ndarray, rows: np.ndarray, minimum: float) -> tuple[Tensor, np.ndarray]:
    shifted = context.clone(); valid = np.zeros(time.size, dtype=bool)
    for seg in np.unique(segment[rows]):
        rr = rows[segment[rows] == seg]
        if rr.size < 4: continue
        donor = np.roll(rr, max(1, rr.size // 2))
        ok = np.abs(time[donor] - time[rr]) >= minimum
        if np.any(ok):
            shifted[torch.as_tensor(rr[ok], device=context.device)] = context[torch.as_tensor(donor[ok], device=context.device)]
            valid[rr[ok]] = True
    return shifted, valid


def _same_prefix_mask(ranks: np.ndarray, rows: np.ndarray, minimum: int) -> np.ndarray:
    signature = []
    for row in ranks:
        first = tuple(np.flatnonzero(row == 0).tolist())
        second = tuple(np.flatnonzero(row == 1).tolist())
        signature.append((first, second))
    counts: dict[tuple, int] = {}
    for index in rows: counts[signature[index]] = counts.get(signature[index], 0) + 1
    return np.asarray([index for index in rows if counts[signature[index]] >= minimum], dtype=np.int64)


def _same_prefix_shift_context(
    context: Tensor,
    ranks: np.ndarray,
    time: np.ndarray,
    segment: np.ndarray,
    rows: np.ndarray,
    minimum_events: int,
    minimum_seconds: float,
) -> tuple[Tensor, np.ndarray]:
    """Swap state only between distant events with the same observed prefix."""

    shifted = context.clone()
    valid = np.zeros(time.size, dtype=bool)
    groups: dict[tuple, list[int]] = {}
    for index in rows:
        row = ranks[index]
        signature = (
            int(segment[index]),
            tuple(np.flatnonzero(row == 0).tolist()),
            tuple(np.flatnonzero(row == 1).tolist()),
        )
        groups.setdefault(signature, []).append(int(index))
    for members in groups.values():
        rr = np.asarray(members, dtype=np.int64)
        if rr.size < int(minimum_events):
            continue
        order = rr[np.argsort(time[rr], kind="stable")]
        donor = np.roll(order, max(1, order.size // 2))
        ok = np.abs(time[donor] - time[order]) >= float(minimum_seconds)
        if np.any(ok):
            shifted[torch.as_tensor(order[ok], device=context.device)] = context[
                torch.as_tensor(donor[ok], device=context.device)
            ]
            valid[order[ok]] = True
    return shifted, valid


def _context_structure_diagnostics(
    model: StepwiseConditionedDecoder,
    tensors: dict[str, Tensor],
    context: Tensor,
    fit_rows: np.ndarray,
    selection_rows: np.ndarray,
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Interpolation and local sensitivity within empirical FIT support."""

    fit = context[torch.as_tensor(fit_rows, device=context.device)].detach().cpu().numpy()
    if fit.shape[0] > 10_000:
        fit = fit[np.linspace(0, fit.shape[0] - 1, 10_000, dtype=np.int64)]
    centre = np.mean(fit, axis=0)
    _u, _s, vh = np.linalg.svd(fit - centre, full_matrices=False)
    direction = vh[0]
    projection = (fit - centre) @ direction
    levels = np.quantile(projection, [0.1, 0.3, 0.5, 0.7, 0.9])
    reference = selection_rows[np.linspace(
        0, selection_rows.size - 1, min(128, selection_rows.size), dtype=np.int64
    )]
    index = torch.as_tensor(reference, dtype=torch.long, device=context.device)
    batch = event_batch(tensors, index)
    valid = batch["valid"].bool()
    interpolation = []
    with torch.no_grad():
        for level in levels:
            value = torch.as_tensor(
                centre + float(level) * direction,
                dtype=context.dtype, device=context.device,
            )[None].expand(index.numel(), -1)
            logits, stops = model.forward(
                batch["x"], batch["recruited"], batch["valid"], value,
                use_static=True, use_dynamic=True,
            )
            available = ~batch["recruited"].bool()
            probability = torch.softmax(logits.masked_fill(~available, -1e4), dim=-1)
            weight = valid.to(probability.dtype)
            denom = weight.sum().clamp_min(1.0)
            field = (probability * weight[..., None]).sum((0, 1)) / denom
            entropy = -(probability.clamp_min(1e-8).log() * probability).sum(-1)
            interpolation.append({
                "pc1_quantile_value": float(level),
                "mean_stop_probability": float((torch.sigmoid(stops) * weight).sum() / denom),
                "mean_contact_entropy": float((entropy * weight).sum() / denom),
                "mean_contact_field": field.cpu().tolist(),
            })

    jacobian_rows = selection_rows[np.linspace(
        0, selection_rows.size - 1, min(64, selection_rows.size), dtype=np.int64
    )]
    jac_index = torch.as_tensor(jacobian_rows, dtype=torch.long, device=context.device)
    jac_batch = event_batch(tensors, jac_index)
    jac_context = context[jac_index].detach().clone().requires_grad_(True)
    logits, stops = model.forward(
        jac_batch["x"], jac_batch["recruited"], jac_batch["valid"], jac_context,
        use_static=True, use_dynamic=True,
    )
    jac_valid = jac_batch["valid"].bool()
    weight = jac_valid.to(logits.dtype)
    denom = weight.sum().clamp_min(1.0)
    stop_scalar = (torch.sigmoid(stops) * weight).sum() / denom
    stop_grad = torch.autograd.grad(stop_scalar, jac_context, retain_graph=True)[0]
    available = ~jac_batch["recruited"].bool()
    probability = torch.softmax(logits.masked_fill(~available, -1e4), dim=-1)
    mean_field = (probability * weight[..., None]).sum((0, 1)) / denom
    rng = np.random.default_rng(20260904)
    probe_norms = []
    for _ in range(4):
        probe = torch.as_tensor(
            rng.choice((-1.0, 1.0), size=mean_field.numel()),
            dtype=mean_field.dtype, device=mean_field.device,
        )
        gradient = torch.autograd.grad((mean_field * probe).sum(), jac_context, retain_graph=True)[0]
        probe_norms.append(float(gradient.square().sum(-1).sqrt().mean()))
    field_change = np.asarray(interpolation[-1]["mean_contact_field"]) - np.asarray(
        interpolation[0]["mean_contact_field"]
    )
    return {
        "definition": "same observed prefixes; state varied only within the empirical FIT PC1 range",
        "n_reference_events": int(reference.size),
        "interpolation": interpolation,
        "endpoint_range": {
            "stop_probability": float(max(v["mean_stop_probability"] for v in interpolation) - min(v["mean_stop_probability"] for v in interpolation)),
            "contact_field_l2": float(np.linalg.norm(field_change)),
        },
        "local_jacobian": {
            "n_events": int(jacobian_rows.size),
            "stop_context_gradient_l2_mean": float(stop_grad.square().sum(-1).sqrt().mean()),
            "contact_field_vjp_l2_mean": float(np.mean(probe_norms)),
            "n_hutchinson_probes": len(probe_norms),
        },
    }


def train_h2a_subject(
    subject: str,
    state_seed: int,
    *,
    device: torch.device,
    out_dir: Path,
    config: H2ATrainConfig | None = None,
    h1_root: Path = H1_ROOT,
    decoder_root: Path = DECODER_ROOT,
    state_family: str = "event",
) -> dict[str, Any]:
    started = time.time(); config = config or H2ATrainConfig(seed=int(state_seed))
    if state_family not in {"event", "dual", "grid"}:
        raise ValueError("state_family must be event, dual or grid")
    h1_dir = Path(h1_root) / subject / f"seed{state_seed}"
    h1_card = json.loads((h1_dir / "card.json").read_text(encoding="utf-8"))
    data = build_h1_subject_data(
        subject, seed=int(state_seed),
        horizons_seconds=tuple(float(value) for value in h1_card["horizons_seconds"]),
    )
    h1_checkpoint = torch.load(h1_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    base_fields = set(H1TrainConfig.__dataclass_fields__)
    h1_config = H1TrainConfig(**{
        key: value for key, value in h1_checkpoint["config"].items() if key in base_fields
    })
    observer = DualStreamEventCTSSM(
        data.burden_mark.shape[1], data.grammar_mark.shape[1],
        taus_seconds=h1_config.taus_seconds,
        burden_channels_per_tau=h1_config.burden_channels_per_tau,
        grammar_channels_per_tau=h1_config.grammar_channels_per_tau,
    ).to(device)
    observer.load_state_dict(
        h1_checkpoint["event_observer"] if state_family == "dual" else h1_checkpoint["observer"]
    )
    observer.eval()
    for parameter in observer.parameters(): parameter.requires_grad_(False)

    dseed = _decoder_seed(state_seed)
    fit_id = f"{subject}__anatomy"
    decoder_root = Path(decoder_root)
    decoder_dir = decoder_root / "formal_units" / fit_id / "L3_LOCAL_PLUS_LEARNED_LR" / f"seed{dseed}"
    decoder_cache = decoder_root / "cache" / fit_id
    bundle = load_frozen_decoder(decoder_dir, decoder_cache, device=device)
    time_event, ranks, segment = _load_eval_events(subject, data, bundle.contact_names)
    phase = _phase(time_event, dict(data.rate.phase_boundaries))
    with torch.no_grad():
        if state_family == "grid":
            state = GridEventStateComputer(
                data, observer, device,
                grid_seconds=float(h1_checkpoint.get("grid_seconds", 300.0)),
                query_time=time_event, query_segment=segment,
            )()
        else:
            state = _state_at_events(data, observer, time_event, segment, device)
    bmark_context = _bmark_at_events(
        data, time_event, segment, tuple(h1_config.taus_seconds),
        np.asarray(h1_checkpoint["bmark_centre"]), np.asarray(h1_checkpoint["bmark_scale"]), device,
    )
    if state_family == "dual":
        with np.load(h1_dir / "trajectory_and_targets.npz", allow_pickle=False) as stored:
            grid_time = np.asarray(stored["anchor_time"], dtype=np.float64)
            grid_segment = np.asarray(stored["segment"], dtype=np.int64)
            background_state = np.asarray(stored["background_state"], dtype=np.float32)
            background_current = np.asarray(stored["background_current"], dtype=np.float32)
        background = np.zeros(
            (time_event.size, background_state.shape[1] + background_current.shape[1]), dtype=np.float32
        )
        background_valid = np.zeros(time_event.size, dtype=bool)
        for seg in np.unique(segment):
            er = np.flatnonzero(segment == seg); gr = np.flatnonzero(grid_segment == seg)
            if er.size == 0 or gr.size == 0: continue
            # Strictly preceding fixed-grid background state: an event exactly
            # at a grid time cannot read that grid's observation correction.
            pos = np.searchsorted(grid_time[gr], time_event[er], side="left") - 1
            ok = pos >= 0
            donor = gr[np.maximum(pos, 0)]
            background[er[ok]] = np.concatenate(
                (background_state[donor[ok]], background_current[donor[ok]]), axis=1
            )
            background_valid[er[ok]] = True
        state = torch.cat((state, torch.as_tensor(background, device=device)), dim=1)
    fit_rows = np.flatnonzero(phase == "FIT"); inner_rows = np.flatnonzero(phase == "INNER")
    selection_rows = np.flatnonzero(phase == "SELECTION")
    if min(fit_rows.size, inner_rows.size, selection_rows.size) == 0:
        # This is a data-support result, not an execution failure.  In
        # particular, a strict decoder may have no evaluable events in one of
        # the H1 time phases even though the H1 fixed-grid trajectory itself is
        # estimable.  Persist an explicit card so aggregation keeps the
        # patient in the denominator as NOT_ESTIMABLE and the queue can
        # continue with the remaining subjects.
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        card = {
            "format": "group_event_state_v0_3_7_h2a_frozen_decoder_equal_horizon_card_v2",
            "status": "NOT_ESTIMABLE",
            "reason": "strict decoder event stream has an empty FIT, INNER or SELECTION phase",
            "subject": subject,
            "state_seed": int(state_seed),
            "decoder_seed": dseed,
            "state_family": state_family,
            "config": asdict(config),
            "n_events": {
                "FIT": int(fit_rows.size),
                "INNER": int(inner_rows.size),
                "SELECTION": int(selection_rows.size),
            },
            "primary_contrasts": {},
            "stages": {},
            "decoder_provenance": {
                "checkpoint": str(decoder_dir / "weights.pt"),
                "cache": str(decoder_cache),
                "strict_anatomy_only": True,
                "trained_before_state_fit": True,
            },
            "state_provenance": {
                "checkpoint": str(h1_dir / "checkpoint.pt"),
                "family": state_family,
                "frozen": True,
                "pre_event": True,
            },
            "development_targets_read": False,
            "seizure_targets_read": False,
            "sealed_partition_opened": False,
            "runtime_seconds": float(time.time() - started),
        }
        atomic_json(out_dir / "card.json", card)
        return card
    fit_index = torch.as_tensor(fit_rows, dtype=torch.long, device=device)
    fit_state_np = state[fit_index].cpu().numpy()
    centre = np.median(fit_state_np, axis=0)
    scale = 1.4826 * np.median(np.abs(fit_state_np - centre), axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-5), scale, 1.0)
    context = torch.as_tensor(
        np.clip((state.cpu().numpy() - centre) / scale, -12.0, 12.0).astype(np.float32), device=device
    )
    tensors = {key: value.to(device) for key, value in build_event_tensors(ranks).items()}
    model = StepwiseConditionedDecoder(
        bundle.model,
        StepwiseAdapterConfig(
            context_dim=context.shape[1], rank=config.modulation_rank,
            output_init_std=config.state_modulation_init_std,
        ),
    ).to(device)
    static_stage = _fit_adapter(
        model, tensors, None, fit_rows, inner_rows, stage="static", config=config
    )
    state_stage = _fit_adapter(
        model, tensors, context, fit_rows, inner_rows, stage="state", config=config
    )

    bmark_model = StepwiseConditionedDecoder(
        bundle.model,
        StepwiseAdapterConfig(
            context_dim=bmark_context.shape[1], rank=config.modulation_rank,
            output_init_std=config.state_modulation_init_std,
        ),
    ).to(device)
    bmark_model.static.load_state_dict(model.static.state_dict())
    bmark_stage = _fit_adapter(
        bmark_model, tensors, bmark_context, fit_rows, inner_rows, stage="state", config=config
    )

    # An explicitly leaked current-event context is a diagnostic positive
    # control.  It is trained in a separate adapter and never enters a
    # scientific comparison or checkpoint selection.
    oracle_context = torch.as_tensor(leaked_event_context(ranks), dtype=torch.float32, device=device)
    oracle_model = StepwiseConditionedDecoder(
        bundle.model,
        StepwiseAdapterConfig(
            context_dim=oracle_context.shape[1], rank=config.modulation_rank,
            output_init_std=config.state_modulation_init_std,
        ),
    ).to(device)
    oracle_model.static.load_state_dict(model.static.state_dict())
    oracle_config = H2ATrainConfig(**{**asdict(config), "max_epochs_state": int(config.max_epochs_oracle)})
    oracle_stage = _fit_adapter(
        oracle_model, tensors, oracle_context, fit_rows, inner_rows, stage="state", config=oracle_config
    )
    shifted, shift_valid = _shift_context(
        context, time_event, segment, selection_rows, config.minimum_shift_seconds
    )
    paired_rows = selection_rows[shift_valid[selection_rows]]
    constant = context[fit_index].mean(0, keepdim=True).expand_as(context)
    same_prefix = _same_prefix_mask(ranks, selection_rows, config.minimum_same_prefix_events)
    same_prefix_shifted, same_prefix_shift_valid = _same_prefix_shift_context(
        context, ranks, time_event, segment, selection_rows,
        config.minimum_same_prefix_events, config.minimum_shift_seconds,
    )
    same_prefix_paired = selection_rows[same_prefix_shift_valid[selection_rows]]

    # Align the rich event mark to this decoder's event stream.  Exact time
    # equality is required; an unmatched event is an implementation error.
    rich_index = np.searchsorted(data.event_time, time_event)
    if np.any(rich_index >= data.event_time.size) or not np.array_equal(data.event_time[rich_index], time_event):
        raise ValueError("H2a events do not map exactly to the registered rich-mark stream")
    rich_target = np.asarray(data.grammar_mark[rich_index], dtype=np.float64)
    has_later_group = np.asarray((ranks.max(axis=1) + 1) > 2, dtype=bool)
    rich_fit = fit_rows[has_later_group[fit_rows]]
    rich_inner = inner_rows[has_later_group[inner_rows]]
    rich_selection = selection_rows[has_later_group[selection_rows]]
    if min(rich_fit.size, rich_inner.size, rich_selection.size) == 0:
        raise ValueError("no events with a later group for conditional rich-mark H2a")
    prefix_rows = np.concatenate((rich_fit, rich_inner, rich_selection))
    prefix_static = prefix_hidden(
        model, tensors, None, prefix_rows, use_dynamic=False,
        batch_size=config.batch_size, prefix_groups=2,
    )
    prefix_state = prefix_hidden(
        model, tensors, context, prefix_rows, use_dynamic=True,
        batch_size=config.batch_size, prefix_groups=2,
    )
    prefix_bmark = prefix_hidden(
        bmark_model, tensors, bmark_context, prefix_rows, use_dynamic=True,
        batch_size=config.batch_size, prefix_groups=2,
    )
    n_fit, n_inner = rich_fit.size, rich_inner.size
    sl_fit = slice(0, n_fit); sl_inner = slice(n_fit, n_fit + n_inner); sl_selection = slice(n_fit + n_inner, None)
    base_mark_model, base_mark_fit = fit_rich_mark_readout(
        prefix_static[sl_fit], rich_target[rich_fit], prefix_static[sl_inner], rich_target[rich_inner]
    )
    state_mark_model, state_mark_fit = fit_rich_mark_readout(
        prefix_state[sl_fit], rich_target[rich_fit], prefix_state[sl_inner], rich_target[rich_inner]
    )
    bmark_mark_model, bmark_mark_fit = fit_rich_mark_readout(
        prefix_bmark[sl_fit], rich_target[rich_fit], prefix_bmark[sl_inner], rich_target[rich_inner]
    )
    prefix_constant_selection = prefix_hidden(
        model, tensors, constant, rich_selection, use_dynamic=True,
        batch_size=config.batch_size, prefix_groups=2,
    )
    rich_mark_scores = {
        "prefix_only": mean_squared_mark(base_mark_model, prefix_static[sl_selection], rich_target[rich_selection]),
        "prefix_plus_state": mean_squared_mark(state_mark_model, prefix_state[sl_selection], rich_target[rich_selection]),
        "prefix_plus_B_mark": mean_squared_mark(bmark_mark_model, prefix_bmark[sl_selection], rich_target[rich_selection]),
        "prefix_plus_constant_state": mean_squared_mark(
            state_mark_model, prefix_constant_selection, rich_target[rich_selection]
        ),
    }
    rich_shift_paired = rich_selection[shift_valid[rich_selection]]
    if rich_shift_paired.size:
        prefix_correct_paired = prefix_hidden(
            model, tensors, context, rich_shift_paired, use_dynamic=True,
            batch_size=config.batch_size, prefix_groups=2,
        )
        prefix_shifted_paired = prefix_hidden(
            model, tensors, shifted, rich_shift_paired, use_dynamic=True,
            batch_size=config.batch_size, prefix_groups=2,
        )
        rich_mark_scores["correct_state_on_shift_support"] = mean_squared_mark(
            state_mark_model, prefix_correct_paired, rich_target[rich_shift_paired]
        )
        rich_mark_scores["shifted_state_on_same_support"] = mean_squared_mark(
            state_mark_model, prefix_shifted_paired, rich_target[rich_shift_paired]
        )
    with torch.no_grad():
        arms = {
            "decoder_only": _mean_scores(model, tensors, None, selection_rows, use_static=False, use_dynamic=False, batch_size=config.batch_size),
            "static_recalibration": _mean_scores(model, tensors, None, selection_rows, use_static=True, use_dynamic=False, batch_size=config.batch_size),
            "correct_state": _mean_scores(model, tensors, context, selection_rows, use_static=True, use_dynamic=True, batch_size=config.batch_size),
            "constant_state": _mean_scores(model, tensors, constant, selection_rows, use_static=True, use_dynamic=True, batch_size=config.batch_size),
            "B_mark_context": _mean_scores(
                bmark_model, tensors, bmark_context, selection_rows,
                use_static=True, use_dynamic=True, batch_size=config.batch_size,
            ),
            "leaked_current_event_oracle": _mean_scores(
                oracle_model, tensors, oracle_context, selection_rows,
                use_static=True, use_dynamic=True, batch_size=config.batch_size,
            ),
        }
        if paired_rows.size:
            arms["correct_state_paired"] = _mean_scores(model, tensors, context, paired_rows, use_static=True, use_dynamic=True, batch_size=config.batch_size)
            arms["block_shift_state"] = _mean_scores(model, tensors, shifted, paired_rows, use_static=True, use_dynamic=True, batch_size=config.batch_size)
        if same_prefix_paired.size:
            arms["static_same_prefix"] = _mean_scores(model, tensors, None, same_prefix_paired, use_static=True, use_dynamic=False, batch_size=config.batch_size, observed_prefix_groups=2)
            arms["state_same_prefix"] = _mean_scores(model, tensors, context, same_prefix_paired, use_static=True, use_dynamic=True, batch_size=config.batch_size, observed_prefix_groups=2)
            arms["B_mark_same_prefix"] = _mean_scores(
                bmark_model, tensors, bmark_context, same_prefix_paired,
                use_static=True, use_dynamic=True, batch_size=config.batch_size, observed_prefix_groups=2,
            )
        if same_prefix_paired.size:
            arms['constant_same_prefix'] = _mean_scores(
                model, tensors, constant, same_prefix_paired, use_static=True, use_dynamic=True,
                batch_size=config.batch_size, observed_prefix_groups=2,
            )
            arms["correct_state_same_prefix_paired"] = _mean_scores(
                model, tensors, context, same_prefix_paired,
                use_static=True, use_dynamic=True, batch_size=config.batch_size, observed_prefix_groups=2,
            )
            arms["same_prefix_shift_state"] = _mean_scores(
                model, tensors, same_prefix_shifted, same_prefix_paired,
                use_static=True, use_dynamic=True, batch_size=config.batch_size, observed_prefix_groups=2,
            )
    structure = _context_structure_diagnostics(
        model, tensors, context, fit_rows, selection_rows, batch_size=config.batch_size
    )
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "static_adapter": model.static.state_dict(), "dynamic_adapter": model.dynamic.state_dict(),
        "bmark_dynamic_adapter": bmark_model.dynamic.state_dict(),
        "config": asdict(config), "state_checkpoint": str(h1_dir / "checkpoint.pt"),
        "decoder_checkpoint": str(decoder_dir / "weights.pt"), "state_centre": centre, "state_scale": scale,
        'rich_mark_readouts': {
            'prefix_only': asdict(base_mark_model), 'prefix_plus_state': asdict(state_mark_model),
            'prefix_plus_B_mark': asdict(bmark_mark_model),
        },
    }, out_dir / "adapter_checkpoint.pt")
    torch.save({'h1_data': data, 'event_time': time_event, 'ranks': ranks, 'segment': segment,
                'context': context.detach().cpu(), 'bmark_context': bmark_context.detach().cpu(),
                'rich_target': rich_target, 'fit_rows': fit_rows, 'inner_rows': inner_rows,
                'selection_rows': selection_rows, 'source_h1_checkpoint': str(h1_dir / 'checkpoint.pt')},
               out_dir / 'evaluation_replay_bundle.pt')
    card = {
        "format": "group_event_state_v0_3_7_h2a_frozen_decoder_equal_horizon_card_v2",
        "subject": subject, "state_seed": int(state_seed), "decoder_seed": dseed,
        "config": asdict(config), "n_events": {"FIT": int(fit_rows.size), "INNER": int(inner_rows.size), "SELECTION": int(selection_rows.size)},
        "n_shift_paired": int(paired_rows.size), "n_same_prefix": int(same_prefix.size),
        "n_same_prefix_shift_paired": int(same_prefix_paired.size),
        "same_prefix_scoring_contract": "after_two_observed_groups_v1",
        "same_prefix_comparison_support": "same-prefix and same-segment time-shift-eligible rows for every arm",
        "decoder_provenance": {
            "checkpoint": str(decoder_dir / "weights.pt"), "cache": str(decoder_cache),
            "strict_anatomy_only": True, "trained_before_state_fit": True,
        },
        "state_provenance": {"checkpoint": str(h1_dir / "checkpoint.pt"), "family": state_family,
                             "frozen": True, "pre_event": True,
                             "background_grid_strictly_pre_event": state_family == "dual"},
        "stages": {"static": static_stage, "B_mark": bmark_stage, "state": state_stage, "leaked_oracle": oracle_stage},
        "selection_arms": arms,
        "contextual_structure": structure,
        "conditional_rich_mark": {
            "definition": "whole-event rich morphology after observing the first two tied groups",
            "targets": [
                "continuous contact timing", "community occupancy and coupling",
                "multiband energy and peak time", "cross-band lag",
                "bipolar/CAR waveform RMS, peak and line length",
            ],
            "n_events": {"FIT": int(rich_fit.size), "INNER": int(rich_inner.size), "SELECTION": int(rich_selection.size)},
            "prefix_only_fit": base_mark_fit, "prefix_plus_state_fit": state_mark_fit,
            "prefix_plus_B_mark_fit": bmark_mark_fit,
            "selection_scores": rich_mark_scores,
            "gain_over_prefix": rich_mark_scores["prefix_only"] - rich_mark_scores["prefix_plus_state"],
        },
        "primary_contrasts": {
            "state_gain_over_static_grammar": arms["static_recalibration"]["grammar"] - arms["correct_state"]["grammar"],
            "state_gain_over_B_mark_grammar": arms["B_mark_context"]["grammar"] - arms["correct_state"]["grammar"],
            "state_gain_over_B_mark_stop": arms["B_mark_context"]["stop_bce"] - arms["correct_state"]["stop_bce"],
            "state_gain_over_B_mark_contact": arms["B_mark_context"]["contact_nll"] - arms["correct_state"]["contact_nll"],
            "state_gain_over_static_stop": arms["static_recalibration"]["stop_bce"] - arms["correct_state"]["stop_bce"],
            "state_gain_over_static_contact": arms["static_recalibration"]["contact_nll"] - arms["correct_state"]["contact_nll"],
            "constant_unexplained_grammar": arms["constant_state"]["grammar"] - arms["correct_state"]["grammar"],
            "constant_unexplained_stop": arms["constant_state"]["stop_bce"] - arms["correct_state"]["stop_bce"],
            "constant_unexplained_contact": arms["constant_state"]["contact_nll"] - arms["correct_state"]["contact_nll"],
            "correct_time_paired_grammar": (
                arms["block_shift_state"]["grammar"] - arms["correct_state_paired"]["grammar"] if paired_rows.size else None
            ),
            "correct_time_paired_stop": (
                arms["block_shift_state"]["stop_bce"] - arms["correct_state_paired"]["stop_bce"] if paired_rows.size else None
            ),
            "correct_time_paired_contact": (
                arms["block_shift_state"]["contact_nll"] - arms["correct_state_paired"]["contact_nll"] if paired_rows.size else None
            ),
            "same_prefix_grammar_gain": (
                arms["static_same_prefix"]["grammar"] - arms["state_same_prefix"]["grammar"] if same_prefix_paired.size else None
            ),
            "same_prefix_gain_over_B_mark": (
                arms["B_mark_same_prefix"]["grammar"] - arms["state_same_prefix"]["grammar"] if same_prefix_paired.size else None
            ),
            "correct_time_same_prefix_grammar": (
                arms["same_prefix_shift_state"]["grammar"] - arms["correct_state_same_prefix_paired"]["grammar"]
                if same_prefix_paired.size else None
            ),
            "same_prefix_constant_unexplained_grammar": (
                arms['constant_same_prefix']['grammar'] - arms['correct_state_same_prefix_paired']['grammar']
                if same_prefix_paired.size else None
            ),
            "conditional_rich_mark_gain": rich_mark_scores["prefix_only"] - rich_mark_scores["prefix_plus_state"],
            "conditional_rich_mark_gain_over_B_mark": (
                rich_mark_scores["prefix_plus_B_mark"] - rich_mark_scores["prefix_plus_state"]
            ),
            "constant_unexplained_rich_mark": (
                rich_mark_scores["prefix_plus_constant_state"] - rich_mark_scores["prefix_plus_state"]
            ),
            "correct_time_rich_mark": (
                rich_mark_scores.get("shifted_state_on_same_support")
                - rich_mark_scores.get("correct_state_on_shift_support")
                if rich_shift_paired.size else None
            ),
            "oracle_sensitivity_grammar_gain": (
                arms["static_recalibration"]["grammar"] - arms["leaked_current_event_oracle"]["grammar"]
            ),
        },
        "adapter_checkpoint": str(out_dir / "adapter_checkpoint.pt"),
        'evaluation_replay_bundle': str(out_dir / 'evaluation_replay_bundle.pt'),
        "code_provenance": _code_provenance(Path(__file__)),
        "elapsed_seconds": time.time() - started,
        "development_targets_read": False, "seizure_targets_read": False, "sealed_partition_opened": False,
    }
    atomic_json(out_dir / "card.json", card)
    return card


def _reevaluate_h1_after_joint(
    data: H1SubjectData,
    observer: DualStreamEventCTSSM,
    h1_checkpoint: dict[str, Any],
    h1_config: H1TrainConfig,
    *,
    device: torch.device,
) -> tuple[dict[str, Any], NestedH1Readout]:
    """Refit only the H1 state head; the jointly updated producer is frozen."""

    fit_np, inner_np, selection_np = (
        np.flatnonzero(data.rate.phase == phase) for phase in ("FIT", "INNER", "SELECTION")
    )
    fit, inner, selection = (
        torch.as_tensor(rows, dtype=torch.long, device=device)
        for rows in (fit_np, inner_np, selection_np)
    )
    q = torch.as_tensor(
        np.clip((data.rate.q_raw - h1_checkpoint["q_centre"]) / h1_checkpoint["q_scale"], -12.0, 12.0),
        dtype=torch.float32, device=device,
    )
    with torch.no_grad():
        fixed_raw = _fixed_features(data, device, tuple(h1_config.taus_seconds))
    bmark = torch.clamp(
        (fixed_raw - torch.as_tensor(h1_checkpoint["bmark_centre"], dtype=fixed_raw.dtype, device=device))
        / torch.as_tensor(h1_checkpoint["bmark_scale"], dtype=fixed_raw.dtype, device=device),
        -12.0, 12.0,
    )
    target, valid, _scales = _target_bundle(data, device)
    exposure = torch.as_tensor(data.rate.target_exposure_seconds, dtype=torch.float32, device=device)
    widths = dict(h1_checkpoint["widths"])
    bmark_burden_dim = len(h1_config.taus_seconds) * data.burden_mark.shape[1]
    state_burden_dim = len(h1_config.taus_seconds) * h1_config.burden_channels_per_tau
    readout = NestedH1Readout(
        q.shape[1], bmark_burden_dim, bmark.shape[1] - bmark_burden_dim,
        state_burden_dim, observer.state_dim - state_burden_dim,
        widths, len(data.rate.horizons_seconds),
        state_readout_init_std=h1_config.state_readout_init_std,
    ).to(device)
    readout.load_state_dict(h1_checkpoint["readout"])
    computer = EventStateComputer(data, observer, device)
    # lr_state=0 guarantees that H2a-updated observer weights cannot move
    # during the H1 readout refit. Only the low-capacity state head is fitted.
    refit_config = H1TrainConfig(**{
        **asdict(h1_config), "lr_state": 0.0, "max_steps_state": 900,
        "warmup_steps_state": 0,
    })
    stage = _train_h1_stage(
        stage="state", readout=readout, q=q, bmark=bmark,
        state_computer=computer, fixed_random=None, target=target, valid=valid,
        exposure=exposure, fit_rows=fit, inner_rows=inner, config=refit_config,
    )
    for parameter in observer.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        state = computer()
        constant = state[fit].mean(0, keepdim=True).expand_as(state)
        shifted, shift_valid = _block_shift(
            state, data, selection_np, max(data.rate.horizons_seconds)
        )
        predictions = {
            "B_mark": readout.predict(q, bmark=bmark),
            "joint_state": readout.predict(q, bmark=bmark, state=state),
            "joint_state_constant": readout.predict(q, bmark=bmark, state=constant),
            "joint_state_shifted": readout.predict(q, bmark=bmark, state=shifted),
        }
        scores = {}
        for name, prediction in predictions.items():
            local_valid = valid
            if name == "joint_state_shifted":
                local_valid = {key: mask & shift_valid[:, None] for key, mask in valid.items()}
                if not bool(local_valid["count"][selection].any()):
                    scores[name] = None
                    continue
            scores[name] = _selection_score(
                prediction, target, local_valid, exposure, readout.log_dispersion,
                selection, selection_np, tuple(data.rate.horizons_seconds),
            )
    return {
        "state_head_refit": stage,
        "selection_scores": scores,
        "primary_contrasts": {
            "joint_state_gain_over_B_mark": scores["B_mark"]["total"] - scores["joint_state"]["total"],
            "constant_unexplained_gain": scores["joint_state_constant"]["total"] - scores["joint_state"]["total"],
            "correct_time_gain": (
                None if scores["joint_state_shifted"] is None else
                scores["joint_state_shifted"]["total"] - scores["joint_state"]["total"]
            ),
        },
    }, readout


def train_h2a_joint_subject(
    subject: str,
    state_seed: int,
    *,
    device: torch.device,
    out_dir: Path,
    h1_root: Path = H1_ROOT,
    primary_h2a_root: Path = Path("/data/hfosp_group_event_state_v0_3_7/h2a_frozen_decoder_equal_horizon"),
    config: H2AJointTrainConfig | None = None,
) -> dict[str, Any]:
    """Interictal-only sensitivity: allow H2a loss to update S_event.

    This never replaces the frozen-transfer primary. The updated observer is
    re-evaluated on the original H1 targets with a newly fitted low-capacity
    state head before any scientific interpretation.
    """

    started = time.time()
    config = config or H2AJointTrainConfig(seed=int(state_seed))
    torch.manual_seed(int(config.seed)); np.random.seed(int(config.seed))
    data = build_h1_subject_data(subject, seed=int(state_seed))
    h1_dir = Path(h1_root) / subject / f"seed{state_seed}"
    h1_checkpoint = torch.load(h1_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    h1_config = H1TrainConfig(**{
        key: value for key, value in h1_checkpoint["config"].items()
        if key in H1TrainConfig.__dataclass_fields__
    })
    observer = DualStreamEventCTSSM(
        data.burden_mark.shape[1], data.grammar_mark.shape[1],
        taus_seconds=h1_config.taus_seconds,
        burden_channels_per_tau=h1_config.burden_channels_per_tau,
        grammar_channels_per_tau=h1_config.grammar_channels_per_tau,
    ).to(device)
    observer.load_state_dict(h1_checkpoint["observer"])

    decoder_seed = _decoder_seed(state_seed)
    fit_id = f"{subject}__anatomy"
    decoder_dir = DECODER_ROOT / "formal_units" / fit_id / "L3_LOCAL_PLUS_LEARNED_LR" / f"seed{decoder_seed}"
    decoder_cache = DECODER_ROOT / "cache" / fit_id
    bundle = load_frozen_decoder(decoder_dir, decoder_cache, device=device)
    time_event, ranks, segment = _load_eval_events(subject, data, bundle.contact_names)
    phase = _phase(time_event, dict(data.rate.phase_boundaries))
    fit_rows, inner_rows, selection_rows = (
        np.flatnonzero(phase == name) for name in ("FIT", "INNER", "SELECTION")
    )
    if min(fit_rows.size, inner_rows.size, selection_rows.size) == 0:
        raise ValueError("empty H2a joint phase")
    tensors = {key: value.to(device) for key, value in build_event_tensors(ranks).items()}
    with torch.no_grad():
        initial_state = _state_at_events(data, observer, time_event, segment, device)
    initial_fit = initial_state[torch.as_tensor(fit_rows, device=device)].cpu().numpy()
    centre = np.median(initial_fit, axis=0)
    scale = 1.4826 * np.median(np.abs(initial_fit - centre), axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-5), scale, 1.0)
    centre_t = torch.as_tensor(centre, dtype=torch.float32, device=device)
    scale_t = torch.as_tensor(scale, dtype=torch.float32, device=device)

    model = StepwiseConditionedDecoder(
        bundle.model,
        StepwiseAdapterConfig(
            context_dim=observer.state_dim, rank=8,
            output_init_std=H2ATrainConfig().state_modulation_init_std,
        ),
    ).to(device)
    primary_checkpoint = torch.load(
        Path(primary_h2a_root) / subject / f"seed{state_seed}" / "adapter_checkpoint.pt",
        map_location="cpu", weights_only=False,
    )
    model.static.load_state_dict(primary_checkpoint["static_adapter"])
    model.dynamic.load_state_dict(primary_checkpoint["dynamic_adapter"])
    for parameter in model.static.parameters():
        parameter.requires_grad_(False)
    trainable_adapter = list(model.dynamic.parameters())
    trainable_observer = list(observer.parameters())
    optimizer = torch.optim.AdamW([
        {"params": trainable_adapter, "lr": config.learning_rate_adapter},
        {"params": trainable_observer, "lr": config.learning_rate_observer},
    ], weight_decay=config.weight_decay)

    def context_now() -> Tensor:
        raw = _state_at_events(data, observer, time_event, segment, device)
        return torch.clamp((raw - centre_t) / scale_t, -12.0, 12.0)

    with torch.no_grad():
        initial_inner = _mean_scores(
            model, tensors, context_now(), inner_rows, use_static=True,
            use_dynamic=True, batch_size=config.batch_size,
        )["grammar"]
    best = initial_inner; best_step = stale = 0
    best_observer = copy.deepcopy(observer.state_dict())
    best_dynamic = copy.deepcopy(model.dynamic.state_dict())
    history = [{"step": 0, "inner_grammar": initial_inner}]
    rng = np.random.default_rng(int(config.seed) + 303)
    for step in range(1, int(config.max_steps) + 1):
        sample = rng.choice(
            fit_rows, min(int(config.fit_events_per_step), fit_rows.size), replace=False
        )
        context = context_now()
        losses = []
        for start in range(0, sample.size, int(config.batch_size)):
            rows = sample[start:start + int(config.batch_size)]
            index = torch.as_tensor(rows, dtype=torch.long, device=device)
            batch = event_batch(tensors, index)
            score = model.scores(
                batch, context[index], use_static=True, use_dynamic=True
            )
            losses.append(score["grammar"])
        objective = torch.cat(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(trainable_adapter + trainable_observer, config.gradient_clip)
        optimizer.step()
        if step % int(config.validate_every) == 0 or step == int(config.max_steps):
            with torch.no_grad():
                inner = _mean_scores(
                    model, tensors, context_now(), inner_rows, use_static=True,
                    use_dynamic=True, batch_size=config.batch_size,
                )["grammar"]
            history.append({"step": step, "fit_grammar": float(objective.detach()), "inner_grammar": inner})
            if np.isfinite(inner) and inner < best - 1e-5:
                best, best_step, stale = inner, step, 0
                best_observer = copy.deepcopy(observer.state_dict())
                best_dynamic = copy.deepcopy(model.dynamic.state_dict())
            else:
                stale += 1
            if stale >= int(config.patience_checks):
                break
    observer.load_state_dict(best_observer); model.dynamic.load_state_dict(best_dynamic)
    for parameter in observer.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        context = context_now()
        shifted, shift_valid = _shift_context(
            context, time_event, segment, selection_rows, 2.0 * 3600.0
        )
        paired = selection_rows[shift_valid[selection_rows]]
        constant = context[torch.as_tensor(fit_rows, device=device)].mean(0, keepdim=True).expand_as(context)
        arms = {
            "static_recalibration": _mean_scores(model, tensors, None, selection_rows, use_static=True, use_dynamic=False, batch_size=config.batch_size),
            "joint_state": _mean_scores(model, tensors, context, selection_rows, use_static=True, use_dynamic=True, batch_size=config.batch_size),
            "joint_state_constant": _mean_scores(model, tensors, constant, selection_rows, use_static=True, use_dynamic=True, batch_size=config.batch_size),
        }
        if paired.size:
            arms["joint_state_paired"] = _mean_scores(model, tensors, context, paired, use_static=True, use_dynamic=True, batch_size=config.batch_size)
            arms["joint_state_shifted"] = _mean_scores(model, tensors, shifted, paired, use_static=True, use_dynamic=True, batch_size=config.batch_size)

    h1_reevaluation, h1_readout = _reevaluate_h1_after_joint(
        data, observer, h1_checkpoint, h1_config, device=device
    )
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "joint_checkpoint.pt"
    torch.save({
        "observer": observer.state_dict(), "dynamic_adapter": model.dynamic.state_dict(),
        "static_adapter": model.static.state_dict(), "h1_refit_readout": h1_readout.state_dict(),
        "config": asdict(config), "h1_source": str(h1_dir / "checkpoint.pt"),
        "primary_h2a_source": str(Path(primary_h2a_root) / subject / f"seed{state_seed}" / "adapter_checkpoint.pt"),
        "state_centre": centre, "state_scale": scale,
    }, checkpoint_path)
    card = {
        "format": "group_event_state_v0_3_7_h2a_joint_interictal_sensitivity_card_v1",
        "subject": subject, "seed": int(state_seed), "config": asdict(config),
        "scope": "event-only S_obs; supportive sensitivity, never replaces frozen-transfer primary",
        "decoder_frozen": True, "static_recalibration_frozen": True,
        "producer_updated_by": "interictal H2a grammar likelihood only",
        "stages": {"joint": {"initial_inner_grammar": initial_inner,
                              "best_inner_grammar": best, "selected_step": best_step,
                              "steps_run": history[-1]["step"], "history": history}},
        "selection_arms": arms,
        "h2a_primary_contrasts": {
            "joint_gain_over_static_grammar": arms["static_recalibration"]["grammar"] - arms["joint_state"]["grammar"],
            "constant_unexplained_grammar": arms["joint_state_constant"]["grammar"] - arms["joint_state"]["grammar"],
            "correct_time_paired_grammar": (
                arms["joint_state_shifted"]["grammar"] - arms["joint_state_paired"]["grammar"]
                if paired.size else None
            ),
        },
        "h1_mandatory_reevaluation": h1_reevaluation,
        "checkpoint_path": str(checkpoint_path), "elapsed_seconds": time.time() - started,
        "code_provenance": _code_provenance(Path(__file__)),
        "development_targets_read": False, "seizure_targets_read": False,
        "sealed_partition_opened": False,
    }
    atomic_json(out_dir / "card.json", card)
    return card
