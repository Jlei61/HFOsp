"""Synthetic and tiny-canary data for the S_P implementation gate."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from src.topic5_group_event_state.v033_training_lab.paths import atomic_write_json
from src.topic5_group_event_state.v033_training_lab.sg_o2 import GrammarPairs
from src.topic5_group_event_state.v032_model.shift import block_circular_donor

from .contracts import (
    ArchConfig,
    OptimizerConfig,
    TrainConfig,
    seed_before_model_construction,
)
from .data import SpatialData
from .model import SpatialStateModel
from .trainer import (
    _states,
    _to_device,
    chronological_train_fit_inner,
    train_spatial_state,
)


def _linear_latent_recovery(model, data: SpatialData, device: torch.device) -> dict[str, float]:
    """Recover planted anchor state up to an affine change of coordinates."""

    if data.anchor_truth is None:
        return {"r2_mean": float("nan"), "r2_min": float("nan")}
    fit, inner, _ = chronological_train_fit_inner(
        data, embargo_seconds=float(data.provenance["future_horizon_seconds"]),
    )
    tensors = _to_device(data, device)
    with torch.no_grad():
        z = _states(model, tensors, fit).detach().cpu().numpy().astype(np.float64)
    y = np.asarray(data.anchor_truth, dtype=np.float64)
    x_fit = np.c_[z[fit.anchor_rows], np.ones(fit.anchor_rows.size)]
    x_inner = np.c_[z[inner.anchor_rows], np.ones(inner.anchor_rows.size)]
    ridge = 1e-3 * np.eye(x_fit.shape[1]); ridge[-1, -1] = 0.0
    coef = np.linalg.solve(x_fit.T @ x_fit + ridge, x_fit.T @ y[fit.anchor_rows])
    pred = x_inner @ coef
    truth = y[inner.anchor_rows]
    denom = np.sum((truth - truth.mean(0, keepdims=True)) ** 2, axis=0).clip(1e-9)
    r2 = 1.0 - np.sum((truth - pred) ** 2, axis=0) / denom
    return {"r2_mean": float(np.mean(r2)), "r2_min": float(np.min(r2))}


def _anchor_future_participation(data: SpatialData, pairs: GrammarPairs) -> np.ndarray:
    out = np.zeros((pairs.anchor_rows.size, data.n_contacts), dtype=np.float64)
    count = np.zeros(pairs.anchor_rows.size, dtype=np.float64)
    np.add.at(out, pairs.pair_anchor, data.participation[pairs.pair_event].astype(np.float64))
    np.add.at(count, pairs.pair_anchor, 1.0)
    return out / count[:, None].clip(1.0)


def _truth_oracle(data: SpatialData) -> dict[str, float]:
    """Linear functional oracle: planted state predicts future participation."""

    if data.anchor_truth is None:
        return {"gain_vs_train_mean": float("nan"), "wrong_time_cost": float("nan")}
    fit, _inner, _ = chronological_train_fit_inner(
        data, embargo_seconds=float(data.provenance["future_horizon_seconds"]),
    )
    report = data.selection_pairs
    y_fit = _anchor_future_participation(data, fit)
    y_report = _anchor_future_participation(data, report)
    truth = np.asarray(data.anchor_truth, dtype=np.float64)
    x_fit = np.c_[truth[fit.anchor_rows], np.ones(fit.anchor_rows.size)]
    x_report = np.c_[truth[report.anchor_rows], np.ones(report.anchor_rows.size)]
    ridge = 1e-3 * np.eye(x_fit.shape[1]); ridge[-1, -1] = 0.0
    coef = np.linalg.solve(x_fit.T @ x_fit + ridge, x_fit.T @ y_fit)
    pred = x_report @ coef
    baseline = np.broadcast_to(y_fit.mean(0), y_report.shape)
    mse_base = float(np.mean((y_report - baseline) ** 2))
    mse_true = float(np.mean((y_report - pred) ** 2))
    last = data.last_event_pos
    carry = np.where(last >= 0, data.event_segment[np.maximum(last, 0)], -1)
    donor = block_circular_donor(
        data.anchor_time, carry, report.anchor_rows,
        horizon=float(data.provenance["future_horizon_seconds"]), fraction=0.5,
    )
    valid = donor >= 0
    wrong = float("nan")
    if np.any(valid):
        x_wrong = np.c_[truth[report.anchor_rows[donor[valid]]], np.ones(valid.sum())]
        wrong_pred = x_wrong @ coef
        wrong = float(np.mean((y_report[valid] - wrong_pred) ** 2) - np.mean((y_report[valid] - pred[valid]) ** 2))
    return {
        "gain_vs_train_mean": mse_base - mse_true,
        "wrong_time_cost": wrong,
        "train_mean_mse": mse_base,
        "correct_time_mse": mse_true,
    }


def _pairs(
    anchor_rows: np.ndarray,
    *,
    anchor_time: np.ndarray,
    event_time: np.ndarray,
    horizon: float,
) -> GrammarPairs:
    kept: list[int] = []
    owners: list[int] = []
    events: list[int] = []
    for row in anchor_rows:
        lo = int(np.searchsorted(event_time, anchor_time[row], side="left"))
        hi = int(np.searchsorted(event_time, anchor_time[row] + horizon, side="left"))
        if hi <= lo:
            continue
        owner = len(kept); kept.append(int(row))
        owners.extend([owner] * (hi - lo)); events.extend(range(lo, hi))
    owner = np.asarray(owners, dtype=np.int64)
    count = np.bincount(owner, minlength=len(kept)).astype(np.float64)
    weight = 1.0 / (len(kept) * count[owner])
    return GrammarPairs(
        anchor_rows=np.asarray(kept, dtype=np.int64),
        pair_anchor=owner,
        pair_event=np.asarray(events, dtype=np.int64),
        pair_weight=weight,
    ).validate()


def make_synthetic_spatial_data(
    *,
    n_events: int = 1000,
    n_contacts: int = 6,
    seed: int = 20260903,
    truth_kind: str = "dynamic",
) -> SpatialData:
    """Known spatial state: past marks predict later subset, extent and lag."""

    if n_events < 120 or n_contacts < 4:
        raise ValueError("synthetic S_P needs at least 120 events and four contacts")
    rng = np.random.default_rng(seed)
    iei = rng.uniform(8.0, 22.0, size=n_events)
    event_time = np.cumsum(iei)
    event_segment = np.zeros(n_events, dtype=np.int64)
    loadings = rng.normal(size=(n_contacts, 2))
    lag_loadings = rng.normal(scale=0.008, size=(n_contacts, 2))
    base = np.linspace(-0.5, 0.25, n_contacts)
    state = np.zeros((n_events, 2), dtype=np.float64)
    if truth_kind == "dynamic":
        current = rng.normal(scale=0.7, size=2)
        for i in range(n_events):
            if i:
                decay = np.exp(-(event_time[i] - event_time[i - 1]) / 3600.0)
                current = decay * current + 0.7 * np.sqrt(1.0 - decay * decay) * rng.normal(size=2)
            state[i] = current
    elif truth_kind == "piecewise_constant":
        for lo in range(0, n_events, 160):
            state[lo:lo + 160] = rng.normal(scale=0.8, size=2)
    elif truth_kind == "none":
        # The current mark remains realistic, but consecutive event states are
        # independent, so past marks contain no future-block state information.
        state = rng.normal(scale=0.8, size=(n_events, 2))
    else:
        raise ValueError(f"unknown synthetic truth_kind={truth_kind!r}")
    logits = base[None] + 2.0 * state @ loadings.T
    prob = 1 / (1 + np.exp(-logits))
    participation = rng.uniform(size=prob.shape) < prob
    for i in range(n_events):
        if participation[i].sum() < 2:
            participation[i, np.argsort(prob[i])[-2:]] = True
    lag = 0.006 * np.arange(n_contacts)[None] + state @ lag_loadings.T
    lag += rng.normal(scale=0.002, size=lag.shape)
    lag[~participation] = np.nan
    group_ids = np.full((n_events, n_contacts), -1, dtype=np.int64)
    for i in range(n_events):
        active = np.flatnonzero(participation[i])
        order = active[np.argsort(lag[i, active])]
        # Two-millisecond tied groups preserve an actual continuous lag target
        # while retaining the accepted legacy group-sequence scorer.
        rounded = np.round(lag[i, order] / 0.002).astype(np.int64)
        _, group = np.unique(rounded, return_inverse=True)
        group_ids[i, order] = group
    group_count = np.maximum(group_ids.max(1) + 1, 0).astype(np.int64)
    group_position = np.where(
        participation,
        group_ids / np.maximum(group_count[:, None] - 1, 1),
        0.0,
    )
    lag_fill = np.where(participation, lag, 0.0)
    nuisance = rng.normal(size=(n_events, 4))
    token = np.concatenate([
        nuisance,
        participation.astype(np.float64),
        group_position,
        lag_fill,
    ], axis=1)
    train_event_mask = (event_time >= np.quantile(event_time, 0.20)) \
        & (event_time < np.quantile(event_time, 0.70))
    mean = token[train_event_mask].mean(0)
    scale = token[train_event_mask].std(0)
    token = (token - mean) / np.where(scale > 1e-6, scale, 1.0)

    anchor_event = np.arange(30, int(n_events * 0.80), 4, dtype=np.int64)
    anchor_time = event_time[anchor_event] + 1e-3
    last_event_pos = anchor_event.copy()
    train_end = np.quantile(event_time, 0.70)
    selection_start = train_end
    selection_end = np.quantile(event_time, 0.80)
    phase = np.where(anchor_time < selection_start, "STATE_TRAIN", "STATE_SELECTION")
    allowed = (anchor_time >= np.quantile(event_time, 0.20)) & (anchor_time < selection_end)
    anchor_time = anchor_time[allowed]
    last_event_pos = last_event_pos[allowed]
    anchor_truth = state[anchor_event][allowed]
    phase = phase[allowed]
    train_rows = np.flatnonzero(phase == "STATE_TRAIN")
    selection_rows = np.flatnonzero(phase == "STATE_SELECTION")
    return SpatialData(
        subject="synthetic_spatial",
        event_time=event_time,
        event_segment=event_segment,
        event_token=token.astype(np.float32),
        train_event_mask=train_event_mask,
        group_ids=group_ids,
        group_count=group_count,
        participation=participation,
        positive_extent=participation.sum(1).astype(np.float32),
        relative_lag=np.where(participation, lag, 0.0).astype(np.float32),
        lag_valid=participation.copy(),
        anchor_time=anchor_time,
        last_event_pos=last_event_pos,
        phase=phase,
        train_pairs=_pairs(train_rows, anchor_time=anchor_time, event_time=event_time, horizon=300.0),
        selection_pairs=_pairs(
            selection_rows, anchor_time=anchor_time, event_time=event_time, horizon=300.0
        ),
        provenance={
            "kind": "known_spatial_predictive_state",
            "truth_kind": truth_kind,
            "generator_seed": seed,
            "development_targets_read": False,
            "sealed_partition_opened": False,
            "seizure_outcomes_read": False,
            "legacy_rank_used": False,
            "future_horizon_seconds": 300.0,
        },
        anchor_truth=anchor_truth.astype(np.float32),
    )


def run_synthetic(
    *,
    output_dir: Path,
    device: torch.device,
    tiny: bool,
    seed: int,
    overwrite: bool = False,
    truth_kind: str = "dynamic",
) -> dict:
    seed_before_model_construction(seed)
    n_events = 180 if tiny else 3000
    data = make_synthetic_spatial_data(n_events=n_events, seed=seed, truth_kind=truth_kind)
    arch = ArchConfig(width=32 if tiny else 64, depth=1 if tiny else 2)
    optim = OptimizerConfig(
        lr_encoder=1e-3,
        lr_state_adapter=1e-3,
        lr_auxiliary=3e-3,
        weight_decay=0.0,
    )
    train = TrainConfig(
        max_steps=20 if tiny else 900,
        validate_every=5 if tiny else 25,
        patience_checks=6,
        anchors_per_step=16 if tiny else 96,
        events_per_anchor=8 if tiny else 16,
        burn_in_seconds=0.0,
        seed=seed,
    )
    kind = "tiny_canary" if tiny else "synthetic_recovery"
    model = SpatialStateModel(
        input_dim=data.event_token.shape[1], n_contacts=data.n_contacts,
        config=arch, legacy_decoder=None,
    )
    card = train_spatial_state(
        model, data, arch=arch, optimizer_config=optim, train_config=train,
        device=device, output_dir=output_dir, card_kind=kind,
        allow_tiny=tiny, overwrite=overwrite,
    )
    recovery = _linear_latent_recovery(model, data, device)
    oracle = _truth_oracle(data)
    card["synthetic_truth"] = {
        "kind": truth_kind,
        "latent_recovery": recovery,
        "functional_oracle": oracle,
        "correct_time_minus_wrong_time_gain": card.get("wrong_time_cost"),
        "period_level_gain": card.get("period_level_gain"),
        "beyond_period_gain": card.get("beyond_period_gain"),
    }
    minimum_gain = None if tiny else 1e-3
    card["engineering_thresholds"] = {
        "minimum_train_inner_gain": minimum_gain,
        "maximum_no_state_train_inner_gain": 0.01,
        "dynamic_requires_beyond_period_and_correct_time": True,
        "functional_oracle_gain_positive": True,
        "finite_gradient": True,
        "parameter_change": True,
    }
    if truth_kind == "dynamic" and not tiny:
        gain_ok = (
            card["inner_gain"] > float(minimum_gain)
            and card["beyond_period_gain"] > 0
            and np.isfinite(card["wrong_time_cost"])
            and card["wrong_time_cost"] > 0
            and oracle["gain_vs_train_mean"] > 0
        )
    elif truth_kind == "piecewise_constant" and not tiny:
        gain_ok = (
            card["inner_gain"] > float(minimum_gain)
            and card["period_level_gain"] > 0
            and np.isfinite(card["wrong_time_cost"])
            and card["wrong_time_cost"] > 0
            and oracle["gain_vs_train_mean"] > 0
        )
    else:
        gain_ok = card["selection_gain"] <= 0.01 and (
            not np.isfinite(card["wrong_time_cost"]) or card["wrong_time_cost"] <= 0.01
        )
    card["status"] = "PASS" if (
        card["parameters_changed"]
        and card["max_gradient_l2"] > 0
        and np.isfinite(card["inner_gain"])
        and (tiny or gain_ok)
    ) else "FAIL"
    atomic_write_json(Path(output_dir) / "training_card.json", card)
    return card
