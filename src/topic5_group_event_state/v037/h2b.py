"""Frozen cross-task transfer from interictal v0.3.7 states to seizures.

Feature freezing and outcome fitting are separate functions.  The former does
not open seizure labels; the latter cannot update any interictal parameter.
"""

from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import datetime
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA

from src.topic5_group_event_state.v035.contracts import DATASET_ROOT
from src.topic5_group_event_state.v035.seizure_transfer import (
    LEADS_SECONDS,
    RIDGES,
    _clinical_feature,
    _field_targets,
    _fit_masked_ridge,
    _fit_scaler,
    _hazard_rows_observed_support,
    _person_period_log_score,
    _score,
)

from .contracts import atomic_json, sha256_file
from .ctssm import GridBackgroundCTSSM
from .h1_data import build_h1_subject_data
from .h1_dual_train import BackgroundStateComputer, H1DualTrainConfig, _background_features
from .h1_train import _code_provenance


STATE_COMPONENTS = 8
BURDEN_COMPONENTS = 4
GRAMMAR_COMPONENTS = 8
TIMEZONE = ZoneInfo("Europe/Berlin")

# Six pre-specified elapsed-time bands replace the legacy 72 unrelated
# five-minute baseline logits.  The observation grid remains five minutes and
# the resulting survival curve is still coherent, but a patient with only a
# handful of FIT seizures is no longer asked to estimate 72 nuisance levels.
HAZARD_BAND_EDGES = (0, 1, 3, 6, 12, 24, 72)
HAZARD_STATIONARITY_TOLERANCE = 1e-5


def _audited_lbfgs(model, optimizer, closure, *, extend_budget: bool = False) -> dict[str, Any]:
    """Measure stationarity at returned weights, with an optional fixed cap extension.

    A small objective change is not called convergence when the final gradient
    remains large. Stopping-cause labels are inferred from optimizer state;
    gradient and parameter changes are directly measured.
    """
    start = {name: value.detach().clone() for name, value in model.named_parameters()}
    initial_loss = float(closure().detach())
    initial_grad = {name: float(value.grad.detach().norm()) if value.grad is not None else 0.0
                    for name, value in model.named_parameters()}
    traces = []
    for cap in ((160, 480) if extend_budget else (160,)):
        optimizer.param_groups[0]['max_iter'] = cap
        optimizer.param_groups[0]['max_eval'] = cap * 5 // 4
        if traces:
            # Default 1e-9 objective-change termination can stop a flat,
            # regularized hazard while its measured gradient still fails.
            # Tighten only numerical stopping precision on the fixed-budget
            # second pass; retain LR, objective, L2 grid and optimizer state.
            optimizer.param_groups[0]['tolerance_change'] = 1e-12
        optimizer.step(closure)
        loss = float(closure().detach())
        grad_inf = max((float(p.grad.detach().abs().max()) for p in model.parameters()
                        if p.grad is not None and p.numel()), default=0.0)
        state = optimizer.state[next(iter(model.parameters()))]
        if not np.isfinite(loss) or not np.isfinite(grad_inf):
            raise FloatingPointError('nonfinite hazard objective or gradient')
        traces.append({'cumulative_iterations': int(state.get('n_iter', 0)),
                       'optimizer_function_evaluations': int(state.get('func_evals', 0)),
                       'fit_penalized_loss': loss, 'gradient_max_abs': grad_inf,
                       'tolerance_change': optimizer.param_groups[0]['tolerance_change'],
                       'call_iteration_budget': cap})
        if grad_inf <= HAZARD_STATIONARITY_TOLERANCE:
            break
    last = traces[-1]
    stationary = last['gradient_max_abs'] <= HAZARD_STATIONARITY_TOLERANCE
    iterations = last['cumulative_iterations']
    return {
        'optimizer': 'LBFGS', 'dtype': 'float64', 'learning_rate': 0.5,
        'line_search': 'strong_wolfe', 'torch_version': torch.__version__,
        'initial_fit_penalized_loss': initial_loss, 'final_fit_penalized_loss': last['fit_penalized_loss'],
        'iterations': iterations, 'budget': 640 if extend_budget else 160,
        'budget_extension_requested': extend_budget, 'passes_stationarity': stationary,
        'stationarity_max_abs_tolerance': HAZARD_STATIONARITY_TOLERANCE,
        'training_budget_exhausted': bool(not stationary and iterations >= (640 if extend_budget else 160)),
        'termination_diagnosis': ('gradient_stationary' if stationary else
                                  'iteration_budget_exhausted' if iterations >= (640 if extend_budget else 160)
                                  else 'stopped_without_stationarity_progress_or_evaluation_limit'),
        'trace': traces,
        'parameters': {name: {'shape': list(p.shape), 'parameters': p.numel(),
                              'initial_gradient_l2': initial_grad[name],
                              'final_gradient_l2': float(p.grad.detach().norm()) if p.grad is not None else 0.0,
                              'selected_delta_l2': float((p.detach() - start[name]).norm()),
                              'selected_delta_max_abs': float((p.detach() - start[name]).abs().max()) if p.numel() else 0.0}
                       for name, p in model.named_parameters()},
    }


class LowCapacityDiscreteHazard(torch.nn.Module):
    """One survival model with a six-band baseline and shared covariate effect."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.bin_logit = torch.nn.Parameter(torch.full((len(HAZARD_BAND_EDGES) - 1,), -5.0))
        self.beta = torch.nn.Parameter(torch.zeros(int(width)))
        lookup = np.zeros(HAZARD_BAND_EDGES[-1], dtype=np.int64)
        for group, (lo, hi) in enumerate(zip(HAZARD_BAND_EDGES[:-1], HAZARD_BAND_EDGES[1:])):
            lookup[lo:hi] = group
        self.register_buffer("bin_group", torch.as_tensor(lookup, dtype=torch.long))

    def forward(self, x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
        return self.bin_logit[self.bin_group[bins]] + x @ self.beta


class FixedBaselineResidualHazard(torch.nn.Module):
    """A state extension whose baseline survival curve is literally frozen.

    The first ``prefix_width`` predictors and all elapsed-time logits come
    from a previously selected baseline model.  Only coefficients for the
    appended state are fitted.  This makes the baseline and state arms nested
    in parameters as well as in their column definitions: an apparent state
    gain cannot be caused by re-estimating the patient's baseline hazard.
    """

    def __init__(
        self,
        baseline: LowCapacityDiscreteHazard,
        total_width: int,
        prefix_width: int,
    ) -> None:
        super().__init__()
        if int(prefix_width) != int(baseline.beta.numel()):
            raise ValueError("baseline width and frozen predictor prefix disagree")
        if int(total_width) <= int(prefix_width):
            raise ValueError("a residual hazard requires at least one added predictor")
        self.prefix_width = int(prefix_width)
        self.register_buffer("bin_logit", baseline.bin_logit.detach().clone())
        self.register_buffer("base_beta", baseline.beta.detach().clone())
        self.beta = torch.nn.Parameter(torch.zeros(int(total_width) - self.prefix_width))
        self.register_buffer("bin_group", baseline.bin_group.detach().clone())

    def forward(self, x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
        return (
            self.bin_logit[self.bin_group[bins]]
            + x[:, :self.prefix_width] @ self.base_beta
            + x[:, self.prefix_width:] @ self.beta
        )


def _fit_low_capacity_hazard(
    x: np.ndarray,
    anchor_rows: np.ndarray,
    bins: np.ndarray,
    y: np.ndarray,
    phase: np.ndarray,
    l2: float,
    weights: np.ndarray,
    *, extend_budget: bool = False,
) -> tuple[LowCapacityDiscreteHazard, float]:
    """Fit on FIT and score on INNER without adding a second intercept."""

    rows = np.flatnonzero(phase[anchor_rows] == "FIT")
    inner = np.flatnonzero(phase[anchor_rows] == "INNER")
    if rows.size == 0 or np.sum(y[rows]) == 0:
        raise ValueError("no fitting seizure transition for discrete hazard")
    xt = torch.as_tensor(x, dtype=torch.float64)
    ar = torch.as_tensor(anchor_rows, dtype=torch.long)
    bt = torch.as_tensor(bins, dtype=torch.long)
    yt = torch.as_tensor(y, dtype=torch.float64)
    wt = torch.as_tensor(weights, dtype=torch.float64)
    model = LowCapacityDiscreteHazard(x.shape[1]).double()
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=0.5, max_iter=160, line_search_fn="strong_wolfe"
    )
    ridx = torch.as_tensor(rows, dtype=torch.long)

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        logits = model(xt[ar[ridx]], bt[ridx])
        raw = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, yt[ridx], reduction="none"
        )
        loss = (raw * wt[ridx]).sum() / wt[ridx].sum().clamp_min(1e-8)
        # Only covariates are ridge-penalised.  The six elapsed-time bands are
        # the common low-capacity survival baseline, not state parameters.
        loss = loss + float(l2) * model.beta.square().mean()
        loss.backward()
        return loss

    model.fit_audit = _audited_lbfgs(model, optimizer, closure, extend_budget=extend_budget)
    with torch.no_grad():
        check = inner if inner.size else rows
        idx = torch.as_tensor(check, dtype=torch.long)
        raw = torch.nn.functional.binary_cross_entropy_with_logits(
            model(xt[ar[idx]], bt[idx]), yt[idx], reduction="none"
        )
        score = (raw * wt[idx]).sum() / wt[idx].sum().clamp_min(1e-8)
    return model, float(score)


def _fit_fixed_baseline_residual_hazard(
    x: np.ndarray,
    anchor_rows: np.ndarray,
    bins: np.ndarray,
    y: np.ndarray,
    phase: np.ndarray,
    l2: float,
    weights: np.ndarray,
    baseline: LowCapacityDiscreteHazard,
    prefix_width: int,
    *, extend_budget: bool = False,
) -> tuple[FixedBaselineResidualHazard, float]:
    """Fit only an appended state residual on top of a frozen baseline."""

    rows = np.flatnonzero(phase[anchor_rows] == "FIT")
    inner = np.flatnonzero(phase[anchor_rows] == "INNER")
    if rows.size == 0 or np.sum(y[rows]) == 0:
        raise ValueError("no fitting seizure transition for residual hazard")
    xt = torch.as_tensor(x, dtype=torch.float64)
    ar = torch.as_tensor(anchor_rows, dtype=torch.long)
    bt = torch.as_tensor(bins, dtype=torch.long)
    yt = torch.as_tensor(y, dtype=torch.float64)
    wt = torch.as_tensor(weights, dtype=torch.float64)
    model = FixedBaselineResidualHazard(
        baseline, total_width=x.shape[1], prefix_width=int(prefix_width)
    ).double()
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=0.5, max_iter=160, line_search_fn="strong_wolfe"
    )
    ridx = torch.as_tensor(rows, dtype=torch.long)

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        logits = model(xt[ar[ridx]], bt[ridx])
        raw = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, yt[ridx], reduction="none"
        )
        loss = (raw * wt[ridx]).sum() / wt[ridx].sum().clamp_min(1e-8)
        loss = loss + float(l2) * model.beta.square().mean()
        loss.backward()
        return loss

    model.fit_audit = _audited_lbfgs(model, optimizer, closure, extend_budget=extend_budget)
    with torch.no_grad():
        check = inner if inner.size else rows
        idx = torch.as_tensor(check, dtype=torch.long)
        raw = torch.nn.functional.binary_cross_entropy_with_logits(
            model(xt[ar[idx]], bt[idx]), yt[idx], reduction="none"
        )
        score = (raw * wt[idx]).sum() / wt[idx].sum().clamp_min(1e-8)
    return model, float(score)


def _robust_pca(value: np.ndarray, fit_rows: np.ndarray, width: int, seed: int
                ) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.asarray(value, dtype=np.float64)
    centre = np.nanmedian(x[fit_rows], axis=0)
    scale = 1.4826 * np.nanmedian(np.abs(x[fit_rows] - centre), axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    z = np.clip((np.where(np.isfinite(x), x, centre) - centre) / scale, -12.0, 12.0)
    n = min(int(width), z.shape[1], max(1, fit_rows.size - 1))
    pca = PCA(n_components=n, svd_solver="randomized", random_state=int(seed))
    pca.fit(z[fit_rows])
    output = pca.transform(z)
    output_centre = np.median(output[fit_rows], axis=0)
    output_scale = 1.4826 * np.median(np.abs(output[fit_rows] - output_centre), axis=0)
    output_scale = np.where(np.isfinite(output_scale) & (output_scale > 1e-6), output_scale, 1.0)
    output = np.clip((output - output_centre) / output_scale, -12.0, 12.0).astype(np.float32)
    return output, {
        "input_width": int(x.shape[1]), "output_width": int(n),
        "fit_rows": int(fit_rows.size),
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "centre": centre.tolist(), "scale": scale.tolist(),
        "components": pca.components_.tolist(), "mean": pca.mean_.tolist(),
        "output_centre": output_centre.tolist(), "output_scale": output_scale.tolist(),
    }


def _clock_features(epoch: np.ndarray) -> np.ndarray:
    seconds = []
    for value in np.asarray(epoch, dtype=np.float64):
        local = datetime.fromtimestamp(float(value), tz=TIMEZONE)
        seconds.append(local.hour * 3600.0 + local.minute * 60.0 + local.second + local.microsecond / 1e6)
    angle = 2.0 * np.pi * np.asarray(seconds) / 86400.0
    return np.column_stack((np.sin(angle), np.cos(angle))).astype(np.float32)


def freeze_h2b_features(
    subject: str,
    seed: int,
    *,
    h1_root: Path,
    event_h1_root: Path,
    grid_h1_root: Path | None,
    out_dir: Path,
) -> dict[str, Any]:
    """Freeze low-capacity interictal features before opening seizure outcomes."""

    h1_dir = Path(h1_root) / subject / f"seed{seed}"
    card = json.loads((h1_dir / "card.json").read_text(encoding="utf-8"))
    if card.get("seizure_targets_read") or card.get("development_targets_read") or card.get("sealed_partition_opened"):
        raise PermissionError("H2b source checkpoint has forbidden downstream reads")
    checkpoint = __import__("torch").load(h1_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    with np.load(h1_dir / "trajectory_and_targets.npz", allow_pickle=False) as stored:
        trajectory = {name: np.asarray(stored[name]) for name in stored.files}
    data = build_h1_subject_data(
        subject, seed=int(seed),
        horizons_seconds=tuple(float(value) for value in card["horizons_seconds"]),
    )
    if not np.array_equal(trajectory["anchor_time"], data.rate.anchor_time):
        raise ValueError("H2b feature freeze anchor drift")
    fit_rows = np.flatnonzero(data.rate.phase == "FIT")
    if fit_rows.size < 3:
        raise ValueError(f"{subject}: fewer than three FIT anchors for H2b feature freezing")
    q = np.clip((data.rate.q_raw - checkpoint["q_centre"]) / checkpoint["q_scale"], -12.0, 12.0)
    transparent = np.concatenate((q, trajectory["fixed_mark_state"]), axis=1)
    context = np.concatenate((transparent, trajectory["background_current"], trajectory["background_state"]), axis=1)
    event = np.asarray(trajectory["event_state"], dtype=np.float32)
    burden_width = len(card["config"]["taus_seconds"]) * int(card["config"]["burden_channels_per_tau"])
    event_burden = event[:, :burden_width]; event_grammar = event[:, burden_width:]
    b_history, b_history_fit = _robust_pca(transparent, fit_rows, STATE_COMPONENTS, seed + 1)
    b_context, b_context_fit = _robust_pca(context, fit_rows, STATE_COMPONENTS, seed + 2)
    background_current = np.asarray(trajectory["background_current"], dtype=np.float32)
    background_state = np.asarray(trajectory["background_state"], dtype=np.float32)
    b_background_current, b_background_current_fit = _robust_pca(
        background_current, fit_rows, STATE_COMPONENTS, seed + 31
    )
    s_background, s_background_fit = _robust_pca(
        background_state, fit_rows, STATE_COMPONENTS, seed + 32
    )
    dual_config_names = {field.name for field in fields(H1DualTrainConfig)}
    dual_config = H1DualTrainConfig(**{
        key: value for key, value in checkpoint["config"].items() if key in dual_config_names
    })
    cpu = torch.device("cpu")
    rebuilt_current, rebuilt_available, _bg_audit, _bg_names, bg_centre, bg_scale = (
        _background_features(data, cpu)
    )
    if not np.allclose(bg_centre, np.asarray(checkpoint["background_centre"]), rtol=0.0, atol=1e-6):
        raise ValueError("H2b background centre differs from frozen dual checkpoint")
    if not np.allclose(bg_scale, np.asarray(checkpoint["background_scale"]), rtol=0.0, atol=1e-6):
        raise ValueError("H2b background scale differs from frozen dual checkpoint")
    if not np.allclose(rebuilt_current.numpy(), background_current, rtol=0.0, atol=1e-6):
        raise ValueError("H2b current-background reconstruction differs from frozen trajectory")
    torch.manual_seed(int(seed) + 1_000_003)
    random_background_observer = GridBackgroundCTSSM(
        rebuilt_current.shape[1], taus_seconds=dual_config.taus_seconds,
        channels_per_tau=dual_config.background_channels_per_tau,
    ).to(cpu)
    for parameter in random_background_observer.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        random_background_raw = BackgroundStateComputer(
            data, random_background_observer, rebuilt_current, rebuilt_available, cpu
        )().cpu().numpy()
    random_background, random_background_fit = _robust_pca(
        random_background_raw, fit_rows, STATE_COMPONENTS, seed + 33
    )
    s_dual_burden, s_dual_burden_fit = _robust_pca(event_burden, fit_rows, BURDEN_COMPONENTS, seed + 3)
    s_dual_grammar, s_dual_grammar_fit = _robust_pca(event_grammar, fit_rows, GRAMMAR_COMPONENTS, seed + 4)

    # H2b is intentionally not gated by the dual-stream H1 result.  Freeze the
    # pre-registered event-only producer as a separate candidate on exactly the
    # same anchors, rather than silently letting the dual producer stand in for
    # every interictal state family.
    event_dir = Path(event_h1_root) / subject / f"seed{seed}"
    event_card = json.loads((event_dir / "card.json").read_text(encoding="utf-8"))
    if event_card.get("seizure_targets_read") or event_card.get("development_targets_read") or event_card.get("sealed_partition_opened"):
        raise PermissionError("event-only H2b source checkpoint has forbidden downstream reads")
    with np.load(event_dir / "trajectory_and_targets.npz", allow_pickle=False) as stored:
        if not np.array_equal(np.asarray(stored["anchor_time"]), data.rate.anchor_time):
            raise ValueError("event-only and dual H2b anchors differ")
        event_only = np.asarray(stored["learned_state"], dtype=np.float32)
    event_burden_width = (
        len(event_card["config"]["taus_seconds"])
        * int(event_card["config"]["burden_channels_per_tau"])
    )
    s_event_burden, s_event_burden_fit = _robust_pca(
        event_only[:, :event_burden_width], fit_rows, BURDEN_COMPONENTS, seed + 13
    )
    s_event_grammar, s_event_grammar_fit = _robust_pca(
        event_only[:, event_burden_width:], fit_rows, GRAMMAR_COMPONENTS, seed + 14
    )
    grid_card = None
    grid_checkpoint_path = None
    grid_trajectory_path = None
    if grid_h1_root is not None:
        grid_dir = Path(grid_h1_root) / subject / f"seed{seed}"
        grid_card = json.loads((grid_dir / "card.json").read_text(encoding="utf-8"))
        if grid_card.get("seizure_targets_read") or grid_card.get("development_targets_read") or grid_card.get("sealed_partition_opened"):
            raise PermissionError("grid H2b source checkpoint has forbidden downstream reads")
        grid_checkpoint_path = grid_dir / "checkpoint.pt"
        grid_trajectory_path = grid_dir / "trajectory_and_targets.npz"
        with np.load(grid_trajectory_path, allow_pickle=False) as stored:
            if not np.array_equal(np.asarray(stored["anchor_time"]), data.rate.anchor_time):
                raise ValueError("grid and dual H2b anchors differ")
            grid_state = np.asarray(stored["learned_state"], dtype=np.float32)
        grid_burden_width = (
            len(grid_card["config"]["taus_seconds"])
            * int(grid_card["config"]["burden_channels_per_tau"])
        )
        s_grid_burden, s_grid_burden_fit = _robust_pca(
            grid_state[:, :grid_burden_width], fit_rows, BURDEN_COMPONENTS, seed + 23
        )
        s_grid_grammar, s_grid_grammar_fit = _robust_pca(
            grid_state[:, grid_burden_width:], fit_rows, GRAMMAR_COMPONENTS, seed + 24
        )
    rng = np.random.default_rng(int(seed) + 5)
    projection = rng.normal(size=(transparent.shape[1], BURDEN_COMPONENTS + GRAMMAR_COMPONENTS))
    projection /= np.linalg.norm(projection, axis=0, keepdims=True).clip(1e-8)
    random_history, random_fit = _robust_pca(transparent @ projection, fit_rows,
                                             BURDEN_COMPONENTS + GRAMMAR_COMPONENTS, seed + 6)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    feature_path = out_dir / "frozen_features.npz"
    np.savez_compressed(
        feature_path, anchor_time=data.rate.anchor_time, segment=data.rate.segment,
        phase=data.rate.phase, segment_bounds=data.rate.segment_bounds,
        observed_support_bounds=data.rate.observed_support_bounds,
        phase_boundaries_json=np.asarray(json.dumps(dict(data.rate.phase_boundaries), sort_keys=True)),
        B_history=b_history, B_context=b_context,
        B_background_current=b_background_current,
        S_background=s_background,
        random_background=random_background,
        S_event_N=s_event_burden, S_event_G=s_event_grammar,
        S_dual_N=s_dual_burden, S_dual_G=s_dual_grammar,
        **({"S_grid_N": s_grid_burden, "S_grid_G": s_grid_grammar}
           if grid_card is not None else {}),
        random_history=random_history, clock=_clock_features(data.rate.anchor_time),
    )
    payload = {
        "format": "group_event_state_v0_3_7_h2b_frozen_features_v1",
        "subject": subject, "seed": int(seed), "source_card": str(h1_dir / "card.json"),
        "source_checkpoint": str(h1_dir / "checkpoint.pt"),
        "source_checkpoint_sha256": sha256_file(h1_dir / "checkpoint.pt"),
        "source_trajectory_sha256": sha256_file(h1_dir / "trajectory_and_targets.npz"),
        "event_source_card": str(event_dir / "card.json"),
        "event_source_checkpoint": str(event_dir / "checkpoint.pt"),
        "event_source_checkpoint_sha256": sha256_file(event_dir / "checkpoint.pt"),
        "event_source_trajectory_sha256": sha256_file(event_dir / "trajectory_and_targets.npz"),
        "grid_source_card": None if grid_card is None else str(Path(grid_h1_root) / subject / f"seed{seed}" / "card.json"),
        "grid_source_checkpoint": None if grid_checkpoint_path is None else str(grid_checkpoint_path),
        "grid_source_checkpoint_sha256": None if grid_checkpoint_path is None else sha256_file(grid_checkpoint_path),
        "grid_source_trajectory_sha256": None if grid_trajectory_path is None else sha256_file(grid_trajectory_path),
        "feature_path": str(feature_path), "feature_sha256": sha256_file(feature_path),
        "feature_definitions": {
            "B_history": "FIT-only PCA of causal q plus fixed marked-history EWMA",
            "B_context": "B_history inputs plus current and persistent fixed-clock background",
            "B_background_current": "FIT-only PCA of the causal current non-event background window",
            "S_background": "FIT-only PCA of the learned persistent fixed-clock background observer",
            "random_background": "FIT-only PCA of a frozen same-width random background observer",
            "S_event_N": "FIT-only PCA of event-only learned burden observer state",
            "S_event_G": "FIT-only PCA of event-only learned conditional-grammar observer state",
            "S_dual_N": "FIT-only PCA of dual-stream learned burden observer state",
            "S_dual_G": "FIT-only PCA of dual-stream learned conditional-grammar observer state",
            **({
                "S_grid_N": "FIT-only PCA of five-minute hierarchical burden observer state",
                "S_grid_G": "FIT-only PCA of five-minute hierarchical conditional-grammar observer state",
            } if grid_card is not None else {}),
            "random_history": "fixed random projection of transparent history; capacity control",
        },
        "fits": {"B_history": b_history_fit, "B_context": b_context_fit,
                 "B_background_current": b_background_current_fit,
                 "S_background": s_background_fit,
                 "random_background": random_background_fit,
                 "S_event_N": s_event_burden_fit, "S_event_G": s_event_grammar_fit,
                 "S_dual_N": s_dual_burden_fit, "S_dual_G": s_dual_grammar_fit,
                 **({"S_grid_N": s_grid_burden_fit, "S_grid_G": s_grid_grammar_fit}
                    if grid_card is not None else {}),
                 "random_history": random_fit},
        "seizure_outcomes_read": False,
        "development_targets_read": False, "sealed_partition_opened": False,
    }
    atomic_json(out_dir / "freeze_card.json", payload)
    return payload


def _phase_at(times: np.ndarray, bounds: dict[str, float]) -> np.ndarray:
    out = np.full(times.size, "OUTSIDE", dtype="<U12")
    out[times < bounds["20pct"]] = "CALIBRATION"
    out[(times >= bounds["20pct"]) & (times < bounds["60pct"])] = "FIT"
    out[(times >= bounds["60pct"]) & (times < bounds["70pct"])] = "INNER"
    out[(times >= bounds["70pct"]) & (times < bounds["80pct"])] = "SELECTION"
    return out


def _fit_predictor_scaler(
    value: np.ndarray, fit_rows: np.ndarray, *, intercept: bool
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit and apply a FIT-only scaler, returning parameters for null arms."""
    x = np.asarray(value, dtype=np.float64).copy()
    start = 1 if intercept else 0
    if fit_rows.size == 0 or start >= x.shape[1]:
        return x, {"start": start, "centre": [], "scale": []}
    centre = np.median(x[fit_rows, start:], axis=0)
    scale = 1.4826 * np.median(np.abs(x[fit_rows, start:] - centre), axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    x[:, start:] = np.clip((x[:, start:] - centre) / scale, -12.0, 12.0)
    return x, {"start": start, "centre": centre.tolist(), "scale": scale.tolist()}


def _apply_predictor_scaler(value: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    x = np.asarray(value, dtype=np.float64).copy()
    start = int(scaler["start"])
    centre = np.asarray(scaler["centre"], dtype=np.float64)
    scale = np.asarray(scaler["scale"], dtype=np.float64)
    if centre.size:
        x[:, start:] = np.clip((x[:, start:] - centre) / scale, -12.0, 12.0)
    return x


def _standardise_predictors(value: np.ndarray, fit_rows: np.ndarray, *, intercept: bool) -> np.ndarray:
    """Compatibility wrapper used by tests and field readouts."""

    return _fit_predictor_scaler(value, fit_rows, intercept=intercept)[0]


def _causal_grid_lookup(times: np.ndarray, anchor_time: np.ndarray, segment: np.ndarray,
                        segment_bounds: np.ndarray, features: dict[str, np.ndarray]
                        ) -> tuple[dict[str, np.ndarray], np.ndarray]:
    query_segment = np.full(times.size, -1, dtype=np.int64)
    for index, (lo, hi) in enumerate(segment_bounds):
        query_segment[(times >= lo) & (times < hi)] = index
    valid = query_segment >= 0
    output = {name: np.zeros((times.size, value.shape[1]), dtype=np.float32)
              for name, value in features.items()}
    for seg in np.unique(query_segment[valid]):
        qr = np.flatnonzero(query_segment == seg); ar = np.flatnonzero(segment == seg)
        pos = np.searchsorted(anchor_time[ar], times[qr], side="right") - 1
        ok = pos >= 0; donor = ar[np.maximum(pos, 0)]
        for name, value in features.items(): output[name][qr[ok]] = value[donor[ok]]
        valid[qr[~ok]] = False
    return output, valid


def _phase_circular_shift(
    state: np.ndarray, time_: np.ndarray, rows: np.ndarray, minimum_seconds: float
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate a held-out state trajectory without requiring one carry segment.

    Seizure-transfer samples are much sparser than H1 anchors and deliberate
    seizure exclusions split otherwise recorded time.  Restricting donors to a
    single carry segment therefore makes the null structurally empty.  This
    shift stays inside the same patient and evaluation phase, preserves the
    entire empirical state distribution, and only accepts genuinely distant
    donors.  Clock and marked-history covariates remain at the target time.
    """

    output = np.asarray(state).copy()
    valid = np.zeros(np.asarray(time_).size, dtype=bool)
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size < 4:
        return output, valid
    donor = np.roll(rows, max(1, rows.size // 2))
    ok = np.abs(np.asarray(time_)[donor] - np.asarray(time_)[rows]) >= float(minimum_seconds)
    output[rows[ok]] = np.asarray(state)[donor[ok]]
    valid[rows[ok]] = True
    return output, valid


def _fit_nested_hazards(anchor_time: np.ndarray, phase: np.ndarray, support: np.ndarray,
                        bounds: dict[str, float], seizures: list[dict[str, Any]],
                        designs: dict[str, np.ndarray],
                        temporal_controls: dict[str, dict[str, Any]] | None = None,
                        *, extend_budget: bool = False,
                        ) -> dict[str, Any]:
    onsets = np.unique(np.asarray([float(s["onset_epoch"]) for s in seizures], dtype=np.float64))
    seizure_phase = _phase_at(onsets, bounds)
    seizure_counts = {
        split: int(np.sum(seizure_phase == split))
        for split in ("FIT", "INNER", "SELECTION")
    }
    # Dense person-period rows are repeated forecasts around the same clinical
    # events; they are not independent seizure outcomes.  Refuse to fit a
    # patient-specific hazard when the true outcome count is too small.
    if seizure_counts["FIT"] < 3 or seizure_counts["INNER"] < 1 or seizure_counts["SELECTION"] < 1:
        return {
            name: {
                "status": "NOT_ESTIMABLE",
                "reason": "insufficient distinct seizures; person-period rows are not independent outcomes",
                "seizures_by_phase": seizure_counts,
            }
            for name in designs
        } | {
            "support": {
                "status": "NOT_ESTIMABLE",
                "seizures_by_phase": seizure_counts,
                "minimum_seizures_by_phase": {"FIT": 3, "INNER": 1, "SELECTION": 1},
            }
        }
    phase_hi = {"CALIBRATION": bounds["20pct"], "FIT": bounds["60pct"],
                "INNER": bounds["70pct"], "SELECTION": bounds["80pct"]}
    ar, bins, y, weights = _hazard_rows_observed_support(anchor_time, phase, support, phase_hi, onsets)
    observed_seizure_counts = {}
    for split in ('FIT', 'INNER', 'SELECTION'):
        positive_rows = np.flatnonzero((phase[ar] == split) & (y > 0))
        next_onset = np.searchsorted(onsets, anchor_time[ar[positive_rows]], side='right')
        observed_seizure_counts[split] = int(np.unique(onsets[next_onset]).size)
    if any(observed_seizure_counts[split] < minimum for split, minimum in [('FIT', 3), ('INNER', 1), ('SELECTION', 1)]):
        failure = {'status': 'NOT_ESTIMABLE',
                   'reason': 'insufficient distinct onsets represented by observed positive hazard rows',
                   'seizures_by_phase': seizure_counts,
                   'observed_seizures_by_phase': observed_seizure_counts}
        return {name: dict(failure) for name in designs} | {'support': dict(failure)}
    fit_anchor_rows = np.flatnonzero(phase == "FIT")
    result: dict[str, Any] = {}
    fitted: dict[str, tuple[torch.nn.Module, dict[str, Any], np.ndarray]] = {}
    recipe_audits: dict[str, list[dict[str, Any]]] = {}
    l2_grid = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
    extension_base = {
        "B_history_plus_S_event": "B_history",
        "B_history_plus_S_grid": "B_history",
        "B_context_plus_S_dual_N": "B_context",
        "B_context_plus_S_dual_G": "B_context",
        "B_context_plus_S_dual": "B_context",
        "B_context_plus_random": "B_context",
        "B_history_current_background_plus_S_background": "B_history_current_background",
        "B_history_current_background_plus_random_background": "B_history_current_background",
        'B_history_plus_random_history': 'B_history',
        'B_history_plus_future_oracle': 'B_history',
    }

    def record_fit(
        name: str,
        best: tuple[float, float, torch.nn.Module] | None,
        scaler: dict[str, Any],
        x: np.ndarray,
        *,
        nested_base: str | None,
        frozen_prefix_width: int,
    ) -> None:
        if best is None:
            result[name] = {"status": "NOT_ESTIMABLE"}
            return
        result[name] = {
            "status": "ESTIMATED", "l2": best[1],
            "inner_person_period_logloss": best[0],
            "selection_censored_likelihood": _person_period_log_score(
                best[2], x, ar, bins, y, phase, "SELECTION", weights
            ),
            "selection_binary_horizons": {
                "status": "NOT_EVALUATED_USE_RIGHT_CENSORED_LIKELIHOOD",
                "reason": "observed-support gaps cannot be treated as complete negative follow-up",
            },
            "baseline_parameter_count": len(HAZARD_BAND_EDGES) - 1,
            "baseline_elapsed_time_bands_minutes": [
                [5 * lo, 5 * hi] for lo, hi in zip(HAZARD_BAND_EDGES[:-1], HAZARD_BAND_EDGES[1:])
            ],
            "nested_base": nested_base,
            "frozen_baseline_predictor_width": int(frozen_prefix_width),
            "new_predictor_width": int(x.shape[1] - frozen_prefix_width),
            'training': best[2].fit_audit,
            'recipe_audits': recipe_audits.get(name, []),
            'fitted_readout': {
                'class': type(best[2]).__name__,
                'state_dict': {key: value.detach().cpu().tolist() for key, value in best[2].state_dict().items()},
                'predictor_scaler': scaler, 'prefix_width': int(frozen_prefix_width),
            },
        }
        fitted[name] = (best[2], scaler, x)

    # Fit the three scientifically distinct baseline families first.  Their
    # elapsed-time curve and covariate coefficients become immutable whenever
    # a state or random-capacity residual is appended below.
    for name in ("clinical", "B_history", "B_context", "B_history_current_background"):
        if name not in designs:
            continue
        x, scaler = _fit_predictor_scaler(designs[name], fit_anchor_rows, intercept=False)
        best = None
        for l2 in l2_grid:
            try:
                model, inner = _fit_low_capacity_hazard(x, ar, bins, y, phase, l2, weights,
                                                        extend_budget=extend_budget)
            except ValueError as error:
                recipe_audits.setdefault(name, []).append({'l2': l2, 'status': 'NOT_ESTIMABLE', 'error': str(error)})
                continue
            recipe_audits.setdefault(name, []).append({'l2': l2, 'inner_loss': inner, 'training': model.fit_audit})
            if best is None or inner < best[0]:
                best = (inner, l2, model)
        record_fit(name, best, scaler, x, nested_base=None, frozen_prefix_width=0)

    # State arms are strict residual extensions.  Re-fitting the common hazard
    # or its history/context coefficients here would recreate the constant-
    # offset artefact already found in the interictal task.
    for name, base_name in extension_base.items():
        if name not in designs:
            continue
        if base_name not in fitted:
            result[name] = {"status": "NOT_ESTIMABLE", "reason": "nested baseline unavailable"}
            continue
        baseline_model, _baseline_scaler, baseline_x = fitted[base_name]
        if not isinstance(baseline_model, LowCapacityDiscreteHazard):
            raise TypeError("a residual extension must start from a primary baseline hazard")
        x, scaler = _fit_predictor_scaler(designs[name], fit_anchor_rows, intercept=False)
        prefix_width = int(baseline_x.shape[1])
        if x.shape[1] <= prefix_width or not np.allclose(
            x[:, :prefix_width], baseline_x, rtol=1e-10, atol=1e-10, equal_nan=True
        ):
            raise ValueError(f"{name} is not an exact predictor-prefix extension of {base_name}")
        best = None
        for l2 in l2_grid:
            try:
                model, inner = _fit_fixed_baseline_residual_hazard(
                    x, ar, bins, y, phase, l2, weights,
                    baseline_model, prefix_width,
                    extend_budget=extend_budget,
                )
            except ValueError as error:
                recipe_audits.setdefault(name, []).append({'l2': l2, 'status': 'NOT_ESTIMABLE', 'error': str(error)})
                continue
            recipe_audits.setdefault(name, []).append({'l2': l2, 'inner_loss': inner, 'training': model.fit_audit})
            if best is None or inner < best[0]:
                best = (inner, l2, model)
        record_fit(
            name, best, scaler, x,
            nested_base=base_name, frozen_prefix_width=prefix_width,
        )
    for name, control in (temporal_controls or {}).items():
        if name not in fitted or name not in result:
            continue
        model, scaler, correct_x = fitted[name]
        valid = np.asarray(control["valid"], dtype=bool)
        paired_phase = np.asarray(phase, dtype="<U12").copy()
        paired_phase[(paired_phase == "SELECTION") & ~valid] = "OUTSIDE"
        shifted_x = _apply_predictor_scaler(control["shifted"], scaler)
        constant_x = _apply_predictor_scaler(control["constant"], scaler)
        correct_paired = _person_period_log_score(
            model, correct_x, ar, bins, y, paired_phase, "SELECTION", weights
        )
        shifted_paired = _person_period_log_score(
            model, shifted_x, ar, bins, y, paired_phase, "SELECTION", weights
        )
        constant_all = _person_period_log_score(
            model, constant_x, ar, bins, y, phase, "SELECTION", weights
        )
        c = correct_paired.get("log_score") if correct_paired.get("status") == "ESTIMATED" else None
        s = shifted_paired.get("log_score") if shifted_paired.get("status") == "ESTIMATED" else None
        k = constant_all.get("log_score") if constant_all.get("status") == "ESTIMATED" else None
        full = result[name]["selection_censored_likelihood"].get("log_score")
        result[name]["temporal_controls"] = {
            "correct_on_shift_support": correct_paired,
            "block_shift_on_same_support": shifted_paired,
            "fit_period_mean_state": constant_all,
            "correct_time_gain_over_shift": None if c is None or s is None else float(s - c),
            "dynamic_gain_over_fit_period_mean": (
                None if k is None or full is None else float(k - full)
            ),
            "n_shift_eligible_anchors": int(np.sum(valid & (phase == "SELECTION"))),
        }
    result["support"] = {
        "status": "ESTIMATED", "person_period_rows": int(ar.size),
        "seizures_by_phase": seizure_counts,
        'observed_seizures_by_phase': observed_seizure_counts,
        "positive_anchor_rows_by_phase": {
            "FIT": int(y[phase[ar] == "FIT"].sum()),
            "INNER": int(y[phase[ar] == "INNER"].sum()),
            "SELECTION": int(y[phase[ar] == "SELECTION"].sum()),
        },
        "warning": "positive anchor rows overlap around seizures and are not an independent-event count",
    }
    return result


def _fit_masked_state_residual_ridge(
    base_x: np.ndarray,
    state_x: np.ndarray,
    base_coef: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    rows: np.ndarray,
    alpha: float,
    centre: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Fit a state-only residual while keeping the field baseline fixed."""

    ys = (np.asarray(y) - centre[None]) / scale[None]
    baseline = np.asarray(base_x) @ np.asarray(base_coef)
    coef = np.zeros((state_x.shape[1], y.shape[1]), dtype=np.float64)
    penalty = np.eye(state_x.shape[1], dtype=np.float64) * float(alpha)
    for column in range(y.shape[1]):
        rr = rows[valid[rows, column] & np.isfinite(ys[rows, column])]
        if rr.size < max(8, state_x.shape[1] // 2):
            continue
        target = ys[rr, column] - baseline[rr, column]
        gram = state_x[rr].T @ state_x[rr] + penalty
        coef[:, column] = np.linalg.solve(gram, state_x[rr].T @ target)
    return coef


def _field_readouts(subject: str, index: dict[str, Any], bounds: dict[str, float],
                    frozen: dict[str, np.ndarray], *, instrument_controls: bool = False,
                    ictal_cache_root: Path | None = None) -> dict[str, Any]:
    targets = _field_targets(index, subject, exact_root=ictal_cache_root)
    anchor_time = frozen["anchor_time"]; segment = frozen["segment"]
    segment_bounds = frozen["segment_bounds"]
    feature_names = (
        "B_history", "B_context", "S_event_N", "S_event_G",
        "S_dual_N", "S_dual_G", "random_history", "clock",
        "B_background_current", "S_background", "random_background",
    )
    if "S_grid_N" in frozen:
        feature_names = feature_names + ("S_grid_N", "S_grid_G")
    grid_features = {name: frozen[name] for name in feature_names}
    out = {}
    for lead in LEADS_SECONDS:
        samples = []
        for seizure_index, endpoint_map in targets.items():
            onset = float(index["seizures"][seizure_index]["onset_epoch"])
            phase = _phase_at(np.asarray([onset]), bounds)[0]
            if phase in {"FIT", "INNER", "SELECTION"}:
                samples.append((seizure_index, onset - lead, phase, endpoint_map))
        key = f"lead_{int(lead // 60)}min"
        if not samples: out[key] = {"status": "NOT_ESTIMABLE"}; continue
        times = np.asarray([row[1] for row in samples], dtype=np.float64)
        queried, valid_time = _causal_grid_lookup(times, anchor_time, segment, segment_bounds, grid_features)
        clinical = np.concatenate((_clinical_feature(times, index["seizures"]), queried["clock"]), axis=1)
        phases = np.asarray([row[2] for row in samples])
        fit, inner, selection = (np.flatnonzero((phases == p) & valid_time) for p in ("FIT", "INNER", "SELECTION"))
        raw_designs = {
            "clinical": np.concatenate((np.ones((times.size, 1)), clinical), axis=1),
            "B_history": np.concatenate((np.ones((times.size, 1)), clinical, queried["B_history"]), axis=1),
            "B_context": np.concatenate((np.ones((times.size, 1)), clinical, queried["B_context"]), axis=1),
            "B_history_current_background": np.concatenate((
                np.ones((times.size, 1)), clinical, queried["B_history"],
                queried["B_background_current"],
            ), axis=1),
            "B_history_current_background_plus_S_background": np.concatenate((
                np.ones((times.size, 1)), clinical, queried["B_history"],
                queried["B_background_current"], queried["S_background"],
            ), axis=1),
            "B_history_current_background_plus_random_background": np.concatenate((
                np.ones((times.size, 1)), clinical, queried["B_history"],
                queried["B_background_current"], queried["random_background"],
            ), axis=1),
            "B_history_plus_S_event": np.concatenate((np.ones((times.size, 1)), clinical, queried["B_history"], queried["S_event_N"], queried["S_event_G"]), axis=1),
            **({"B_history_plus_S_grid": np.concatenate((np.ones((times.size, 1)), clinical, queried["B_history"], queried["S_grid_N"], queried["S_grid_G"]), axis=1)}
               if "S_grid_N" in queried else {}),
            "B_context_plus_S_dual_N": np.concatenate((np.ones((times.size, 1)), clinical, queried["B_context"], queried["S_dual_N"]), axis=1),
            "B_context_plus_S_dual_G": np.concatenate((np.ones((times.size, 1)), clinical, queried["B_context"], queried["S_dual_G"]), axis=1),
            "B_context_plus_S_dual": np.concatenate((np.ones((times.size, 1)), clinical, queried["B_context"], queried["S_dual_N"], queried["S_dual_G"]), axis=1),
            "B_context_plus_random": np.concatenate((np.ones((times.size, 1)), clinical, queried["B_context"], queried["random_history"]), axis=1),
        }
        designs: dict[str, np.ndarray] = {}
        if instrument_controls:
            raw_designs['B_history_plus_random_history'] = np.concatenate(
                (raw_designs['B_history'], queried['random_history']), axis=1)
        scalers: dict[str, dict[str, Any]] = {}
        for name, value in raw_designs.items():
            designs[name], scalers[name] = _fit_predictor_scaler(value, fit, intercept=True)
        lead_result = {}
        for endpoint_name in sorted(set().union(*(row[3].keys() for row in samples))):
            example = next(row[3][endpoint_name] for row in samples if endpoint_name in row[3])
            y = np.full((len(samples), example.values.shape[1]), np.nan, dtype=np.float32)
            valid = np.zeros_like(y, dtype=bool)
            for i, sample in enumerate(samples):
                if endpoint_name in sample[3] and valid_time[i]:
                    y[i] = sample[3][endpoint_name].values[0]; valid[i] = sample[3][endpoint_name].valid[0]
            if fit.size < 5 or selection.size == 0:
                lead_result[endpoint_name] = {"status": "NOT_ESTIMABLE", "n_fit_seizures": int(fit.size),
                                              "n_inner_seizures": int(inner.size), "n_selection_seizures": int(selection.size)}
                continue
            if instrument_controls:
                # Use the same fitted contact columns for every contrast.
                # A fallback mean/zero on an untrained target must not count
                # as a failed learned spatial-field prediction.
                minimum_column_fit = max(8, max(x.shape[1] for x in designs.values()) // 2)
                column_fit_count = np.sum(valid[fit] & np.isfinite(y[fit]), axis=0)
                eligible_columns = column_fit_count >= minimum_column_fit
                valid = valid & eligible_columns[None]
                if inner.size == 0 or not valid[inner].any() or not valid[selection].any():
                    lead_result[endpoint_name] = {
                        'status': 'NOT_ESTIMABLE', 'reason': 'no common trained contact targets with INNER and SELECTION support',
                        'fit_column_counts': column_fit_count.tolist(), 'minimum_fit_per_column': minimum_column_fit,
                        'n_fit_seizures': int(fit.size), 'n_inner_seizures': int(inner.size),
                        'n_selection_seizures': int(selection.size),
                    }
                    continue
            centre, scale = _fit_scaler(y, valid, fit, False); scores = {}
            fitted: dict[str, np.ndarray] = {}
            extension_base = {
                "B_history_plus_S_event": "B_history",
                "B_history_plus_S_grid": "B_history",
                "B_context_plus_S_dual_N": "B_context",
                "B_context_plus_S_dual_G": "B_context",
                "B_context_plus_S_dual": "B_context",
                "B_context_plus_random": "B_context",
                "B_history_current_background_plus_S_background": "B_history_current_background",
                "B_history_current_background_plus_random_background": "B_history_current_background",
                'B_history_plus_random_history': 'B_history',
            }
            for name in ("clinical", "B_history", "B_context", "B_history_current_background"):
                if name not in designs:
                    continue
                x = designs[name]
                minimum_fit = max(8, x.shape[1] // 2)
                if fit.size < minimum_fit:
                    scores[name] = {
                        "status": "NOT_ESTIMABLE",
                        "reason": "too few FIT seizures for readout width",
                        "n_fit_seizures": int(fit.size), "minimum_fit_seizures": int(minimum_fit),
                    }
                    continue
                best = None
                for alpha in RIDGES:
                    coef = _fit_masked_ridge(x, y, valid, fit, alpha, centre, scale)
                    check = inner if inner.size else fit
                    loss, _n = _score(x @ coef, y, valid, check, centre, scale, False)
                    if loss is not None and (best is None or loss < best[0]): best = (loss, alpha, coef)
                if best is None: scores[name] = {"status": "NOT_ESTIMABLE"}; continue
                loss, n = _score(x @ best[2], y, valid, selection, centre, scale, False)
                scores[name] = {
                    "status": "ESTIMATED", "selection_loss": loss,
                    "n_values": n, "alpha": best[1], "nested_base": None,
                    "frozen_baseline_predictor_width": 0,
                    "new_predictor_width": int(x.shape[1]),
                    'fitted_readout': {'coefficient': best[2].tolist(), 'target_centre': centre.tolist(),
                                       'target_scale': scale.tolist(), 'predictor_scaler': scalers[name]},
                    'fit_method': 'closed-form masked ridge; intercept unpenalized',
                }
                fitted[name] = best[2]

            for name, base_name in extension_base.items():
                if name not in designs:
                    continue
                if base_name not in fitted:
                    scores[name] = {
                        "status": "NOT_ESTIMABLE", "reason": "nested baseline unavailable",
                    }
                    continue
                x = designs[name]; base_x = designs[base_name]
                prefix_width = int(base_x.shape[1])
                if x.shape[1] <= prefix_width or not np.allclose(
                    x[:, :prefix_width], base_x, rtol=1e-10, atol=1e-10, equal_nan=True
                ):
                    raise ValueError(f"{name} is not an exact predictor-prefix extension of {base_name}")
                state_x = x[:, prefix_width:]
                minimum_fit = max(8, state_x.shape[1] // 2)
                if fit.size < minimum_fit:
                    scores[name] = {
                        "status": "NOT_ESTIMABLE",
                        "reason": "too few FIT seizures for added state width",
                        "n_fit_seizures": int(fit.size),
                        "minimum_fit_seizures": int(minimum_fit),
                    }
                    continue
                best = None
                for alpha in RIDGES:
                    state_coef = _fit_masked_state_residual_ridge(
                        base_x, state_x, fitted[base_name], y, valid, fit,
                        alpha, centre, scale,
                    )
                    combined = np.concatenate((fitted[base_name], state_coef), axis=0)
                    check = inner if inner.size else fit
                    loss, _n = _score(x @ combined, y, valid, check, centre, scale, False)
                    if loss is not None and (best is None or loss < best[0]):
                        best = (loss, alpha, combined)
                if best is None:
                    scores[name] = {"status": "NOT_ESTIMABLE"}
                    continue
                loss, n = _score(x @ best[2], y, valid, selection, centre, scale, False)
                scores[name] = {
                    "status": "ESTIMATED", "selection_loss": loss,
                    "n_values": n, "alpha": best[1], "nested_base": base_name,
                    "frozen_baseline_predictor_width": prefix_width,
                    "new_predictor_width": int(state_x.shape[1]),
                    'fitted_readout': {'coefficient': best[2].tolist(), 'target_centre': centre.tolist(),
                                       'target_scale': scale.tolist(), 'predictor_scaler': scalers[name]},
                    'fit_method': 'closed-form masked ridge residual; inherited baseline coefficient frozen',
                }
                fitted[name] = best[2]

            if instrument_controls and 'B_history' in fitted:
                base_x = designs['B_history']; base_coef = fitted['B_history']
                residual = (y - centre[None]) / scale[None] - base_x @ base_coef
                residual = np.where(valid & np.isfinite(residual), residual, 0.0)
                _u, _s, vt = np.linalg.svd(residual[fit], full_matrices=False)
                width = min(4, int(valid[fit].any(0).sum()), max(1, fit.size - 1))
                basis = vt[:width]
                oracle_x = residual @ basis.T
                best_oracle = None
                for alpha in RIDGES:
                    coef = _fit_masked_state_residual_ridge(base_x, oracle_x, base_coef, y, valid, fit,
                                                            alpha, centre, scale)
                    prediction = base_x @ base_coef + oracle_x @ coef
                    loss, _n = _score(prediction, y, valid, inner, centre, scale, False)
                    if loss is not None and (best_oracle is None or loss < best_oracle[0]):
                        best_oracle = (loss, alpha, coef, prediction)
                if best_oracle is not None:
                    oracle_loss, _n = _score(best_oracle[3], y, valid, selection, centre, scale, False)
                    base_loss = scores['B_history']['selection_loss']
                    scores['future_field_oracle'] = {
                        'status': 'ESTIMATED', 'definition': 'deliberately leaked low-rank future residual field; FIT-only basis; never selects state',
                        'rank': width, 'basis': basis.tolist(), 'coefficient': best_oracle[2].tolist(),
                        'alpha': best_oracle[1], 'selection_loss': oracle_loss,
                        'gain_over_history': None if oracle_loss is None or base_loss is None else base_loss - oracle_loss,
                    }

            def add_temporal_control(name: str, base: np.ndarray, state: np.ndarray) -> None:
                if name not in fitted:
                    return
                shifted, shift_valid = _phase_circular_shift(
                    np.asarray(state), times, selection, max(float(lead), 1800.0)
                )
                paired = selection[shift_valid[selection] & valid_time[selection]]
                valid_fit = fit[valid_time[fit]]
                mean = (
                    np.nanmean(np.asarray(state)[valid_fit], axis=0, keepdims=True)
                    if valid_fit.size else np.zeros((1, np.asarray(state).shape[1]), dtype=np.float32)
                )
                constant = np.broadcast_to(mean, np.asarray(state).shape)
                shifted_x = _apply_predictor_scaler(
                    np.concatenate((base, shifted), axis=1), scalers[name]
                )
                constant_x = _apply_predictor_scaler(
                    np.concatenate((base, constant), axis=1), scalers[name]
                )
                correct_loss, correct_n = _score(
                    designs[name] @ fitted[name], y, valid, paired, centre, scale, False
                )
                shifted_loss, shifted_n = _score(
                    shifted_x @ fitted[name], y, valid, paired, centre, scale, False
                )
                constant_loss, constant_n = _score(
                    constant_x @ fitted[name], y, valid, selection, centre, scale, False
                )
                full_loss = scores[name].get("selection_loss")
                scores[name]["temporal_controls"] = {
                    "correct_on_shift_support": {"selection_loss": correct_loss, "n_values": correct_n},
                    "block_shift_on_same_support": {"selection_loss": shifted_loss, "n_values": shifted_n},
                    "fit_period_mean_state": {"selection_loss": constant_loss, "n_values": constant_n},
                    "correct_time_gain_over_shift": (
                        None if correct_loss is None or shifted_loss is None
                        else float(shifted_loss - correct_loss)
                    ),
                    "dynamic_gain_over_fit_period_mean": (
                        None if constant_loss is None or full_loss is None
                        else float(constant_loss - full_loss)
                    ),
                    "n_shift_eligible_seizures": int(paired.size),
                }

            add_temporal_control(
                "B_history_plus_S_event",
                np.concatenate((np.ones((times.size, 1)), clinical, queried["B_history"]), axis=1),
                np.concatenate((queried["S_event_N"], queried["S_event_G"]), axis=1),
            )
            if "S_grid_N" in queried:
                add_temporal_control(
                    "B_history_plus_S_grid",
                    np.concatenate((np.ones((times.size, 1)), clinical, queried["B_history"]), axis=1),
                    np.concatenate((queried["S_grid_N"], queried["S_grid_G"]), axis=1),
                )
            add_temporal_control(
                "B_context_plus_S_dual",
                np.concatenate((np.ones((times.size, 1)), clinical, queried["B_context"]), axis=1),
                np.concatenate((queried["S_dual_N"], queried["S_dual_G"]), axis=1),
            )
            add_temporal_control(
                "B_history_current_background_plus_S_background",
                np.concatenate((
                    np.ones((times.size, 1)), clinical, queried["B_history"],
                    queried["B_background_current"],
                ), axis=1),
                queried["S_background"],
            )
            base = scores.get("B_context", {}).get("selection_loss")
            full = scores.get("B_context_plus_S_dual", {}).get("selection_loss")
            scores["state_gain_over_background"] = None if base is None or full is None else base - full
            event_base = scores.get("B_history", {}).get("selection_loss")
            event_full = scores.get("B_history_plus_S_event", {}).get("selection_loss")
            scores["event_only_state_gain_over_mark_history"] = (
                None if event_base is None or event_full is None else event_base - event_full
            )
            if "B_history_plus_S_grid" in scores:
                grid_full = scores["B_history_plus_S_grid"].get("selection_loss")
                scores["grid_state_gain_over_mark_history"] = (
                    None if event_base is None or grid_full is None else event_base - grid_full
                )
            scores["event_correct_time_gain_over_shift"] = (
                scores.get("B_history_plus_S_event", {}).get("temporal_controls", {}).get("correct_time_gain_over_shift")
            )
            scores["event_dynamic_gain_over_fit_period_mean"] = (
                scores.get("B_history_plus_S_event", {}).get("temporal_controls", {}).get("dynamic_gain_over_fit_period_mean")
            )
            scores["grid_correct_time_gain_over_shift"] = (
                scores.get("B_history_plus_S_grid", {}).get("temporal_controls", {}).get("correct_time_gain_over_shift")
            )
            scores["grid_dynamic_gain_over_fit_period_mean"] = (
                scores.get("B_history_plus_S_grid", {}).get("temporal_controls", {}).get("dynamic_gain_over_fit_period_mean")
            )
            scores["dual_correct_time_gain_over_shift"] = (
                scores.get("B_context_plus_S_dual", {}).get("temporal_controls", {}).get("correct_time_gain_over_shift")
            )
            scores["dual_dynamic_gain_over_fit_period_mean"] = (
                scores.get("B_context_plus_S_dual", {}).get("temporal_controls", {}).get("dynamic_gain_over_fit_period_mean")
            )
            random_loss = scores.get("B_context_plus_random", {}).get("selection_loss")
            scores["state_gain_over_random_capacity_control"] = (
                None if random_loss is None or full is None else random_loss - full
            )
            background_base = scores.get("B_history_current_background", {}).get("selection_loss")
            background_full = scores.get(
                "B_history_current_background_plus_S_background", {}
            ).get("selection_loss")
            background_random = scores.get(
                "B_history_current_background_plus_random_background", {}
            ).get("selection_loss")
            scores["background_state_gain_over_current_background"] = (
                None if background_base is None or background_full is None
                else background_base - background_full
            )
            scores["background_state_gain_over_random_background"] = (
                None if background_random is None or background_full is None
                else background_random - background_full
            )
            scores["background_correct_time_gain_over_shift"] = (
                scores.get("B_history_current_background_plus_S_background", {})
                .get("temporal_controls", {}).get("correct_time_gain_over_shift")
            )
            scores["background_dynamic_gain_over_fit_period_mean"] = (
                scores.get("B_history_current_background_plus_S_background", {})
                .get("temporal_controls", {}).get("dynamic_gain_over_fit_period_mean")
            )
            scores["status"] = "ESTIMATED"
            scores["support"] = {
                "n_fit_seizures": int(fit.size), "n_inner_seizures": int(inner.size),
                "n_selection_seizures": int(selection.size),
            }
            if instrument_controls:
                scores['support'].update(fit_column_counts=column_fit_count.tolist(),
                                         minimum_fit_per_column=minimum_column_fit,
                                         common_fitted_columns=np.flatnonzero(eligible_columns).tolist())
            lead_result[endpoint_name] = scores
        out[key] = lead_result
    return out


def run_h2b_outcomes(subject: str, seed: int, *, freeze_dir: Path, out_dir: Path,
                    extend_budget: bool = False, instrument_controls: bool = False,
                    ictal_cache_root: Path | None = None) -> dict[str, Any]:
    freeze_card = json.loads((Path(freeze_dir) / "freeze_card.json").read_text(encoding="utf-8"))
    feature_path = Path(freeze_card["feature_path"])
    if sha256_file(feature_path) != freeze_card["feature_sha256"]:
        raise ValueError("frozen H2b feature hash mismatch")
    with np.load(feature_path, allow_pickle=False) as stored:
        frozen = {name: np.asarray(stored[name]) for name in stored.files}
    bounds = json.loads(str(frozen["phase_boundaries_json"].item()))
    index = json.loads((DATASET_ROOT / subject / "index.json").read_text(encoding="utf-8"))
    seizures = [{**row, "_source_index": i} for i, row in enumerate(index.get("seizures", []))
                if float(row["onset_epoch"]) < float(bounds["80pct"])]
    analysis_index = dict(index); analysis_index["seizures"] = seizures
    anchor_time = frozen["anchor_time"]; phase = frozen["phase"].astype(str)
    segment = np.asarray(frozen["segment"], dtype=np.int64)
    clinical = np.concatenate((_clinical_feature(anchor_time, seizures), frozen["clock"]), axis=1)
    designs = {
        "clinical": clinical,
        "B_history": np.concatenate((clinical, frozen["B_history"]), axis=1),
        "B_context": np.concatenate((clinical, frozen["B_context"]), axis=1),
        "B_history_current_background": np.concatenate((
            clinical, frozen["B_history"], frozen["B_background_current"]
        ), axis=1),
        "B_history_current_background_plus_S_background": np.concatenate((
            clinical, frozen["B_history"], frozen["B_background_current"],
            frozen["S_background"],
        ), axis=1),
        "B_history_current_background_plus_random_background": np.concatenate((
            clinical, frozen["B_history"], frozen["B_background_current"],
            frozen["random_background"],
        ), axis=1),
        "B_history_plus_S_event": np.concatenate((clinical, frozen["B_history"], frozen["S_event_N"], frozen["S_event_G"]), axis=1),
        **({"B_history_plus_S_grid": np.concatenate((clinical, frozen["B_history"], frozen["S_grid_N"], frozen["S_grid_G"]), axis=1)}
           if "S_grid_N" in frozen else {}),
        "B_context_plus_S_dual_N": np.concatenate((clinical, frozen["B_context"], frozen["S_dual_N"]), axis=1),
        "B_context_plus_S_dual_G": np.concatenate((clinical, frozen["B_context"], frozen["S_dual_G"]), axis=1),
        "B_context_plus_S_dual": np.concatenate((clinical, frozen["B_context"], frozen["S_dual_N"], frozen["S_dual_G"]), axis=1),
        "B_context_plus_random": np.concatenate((clinical, frozen["B_context"], frozen["random_history"]), axis=1),
    }
    if instrument_controls:
        # The random history projection was frozen in the original interictal
        # feature archive. It gives event/grid the same added width as their
        # state, testing information lost by the eight-component baseline.
        designs['B_history_plus_random_history'] = np.concatenate(
            (designs['B_history'], frozen['random_history']), axis=1)
        onsets = np.sort(np.asarray([row['onset_epoch'] for row in seizures], dtype=np.float64))
        pos = np.searchsorted(onsets, anchor_time, side='right')
        distance = np.full(anchor_time.size, np.inf)
        use = pos < onsets.size
        distance[use] = onsets[pos[use]] - anchor_time[use]
        # Deliberately leaked future features are a separately named assay
        # diagnostic, never a scientific state or a selection criterion for
        # an interictal checkpoint. Fixed definition precedes this repair run.
        oracle = np.column_stack((distance <= 1800, distance <= 7200, np.exp(-distance / 7200)))
        designs['B_history_plus_future_oracle'] = np.concatenate((designs['B_history'], oracle), axis=1)
    fit_rows = np.flatnonzero(phase == "FIT")
    selection_rows = np.flatnonzero(phase == "SELECTION")
    temporal_controls: dict[str, dict[str, Any]] = {}

    def register_temporal_control(name: str, base: np.ndarray, state: np.ndarray) -> None:
        shifted, valid = _phase_circular_shift(
            np.asarray(state), anchor_time, selection_rows, float(6.0 * 3600.0)
        )
        mean = (
            np.nanmean(np.asarray(state)[fit_rows], axis=0, keepdims=True)
            if fit_rows.size else np.zeros((1, np.asarray(state).shape[1]), dtype=np.float32)
        )
        constant = np.broadcast_to(mean, np.asarray(state).shape)
        temporal_controls[name] = {
            "shifted": np.concatenate((base, shifted), axis=1),
            "constant": np.concatenate((base, constant), axis=1),
            "valid": valid,
        }

    event_state = np.concatenate((frozen["S_event_N"], frozen["S_event_G"]), axis=1)
    register_temporal_control(
        "B_history_plus_S_event",
        np.concatenate((clinical, frozen["B_history"]), axis=1),
        event_state,
    )
    if "S_grid_N" in frozen:
        grid_state = np.concatenate((frozen["S_grid_N"], frozen["S_grid_G"]), axis=1)
        register_temporal_control(
            "B_history_plus_S_grid",
            np.concatenate((clinical, frozen["B_history"]), axis=1),
            grid_state,
        )
    register_temporal_control(
        "B_context_plus_S_dual_N",
        np.concatenate((clinical, frozen["B_context"]), axis=1),
        frozen["S_dual_N"],
    )
    register_temporal_control(
        "B_context_plus_S_dual_G",
        np.concatenate((clinical, frozen["B_context"]), axis=1),
        frozen["S_dual_G"],
    )
    register_temporal_control(
        "B_context_plus_S_dual",
        np.concatenate((clinical, frozen["B_context"]), axis=1),
        np.concatenate((frozen["S_dual_N"], frozen["S_dual_G"]), axis=1),
    )
    register_temporal_control(
        "B_history_current_background_plus_S_background",
        np.concatenate((
            clinical, frozen["B_history"], frozen["B_background_current"]
        ), axis=1),
        frozen["S_background"],
    )
    hazard = _fit_nested_hazards(
        anchor_time, phase, frozen["observed_support_bounds"], bounds, seizures,
        designs, temporal_controls=temporal_controls,
        extend_budget=extend_budget,
    )
    fields = _field_readouts(subject, analysis_index, bounds, frozen,
                            instrument_controls=instrument_controls, ictal_cache_root=ictal_cache_root)
    def logloss(name: str) -> float | None:
        value = hazard.get(name, {}).get("selection_censored_likelihood", {})
        return float(value["log_score"]) if value.get("status") == "ESTIMATED" else None
    base, full, random = logloss("B_context"), logloss("B_context_plus_S_dual"), logloss("B_context_plus_random")
    background_base = logloss("B_history_current_background")
    background_full = logloss("B_history_current_background_plus_S_background")
    background_random = logloss("B_history_current_background_plus_random_background")
    event_base, event_full = logloss("B_history"), logloss("B_history_plus_S_event")
    grid_full = logloss("B_history_plus_S_grid") if "B_history_plus_S_grid" in designs else None
    def temporal(name: str, key: str) -> float | None:
        value = hazard.get(name, {}).get("temporal_controls", {}).get(key)
        return None if value is None else float(value)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    card = {
        "format": "group_event_state_v0_3_7_h2b_frozen_transfer_v2", "subject": subject, "seed": int(seed),
        "freeze_card": str(Path(freeze_dir) / "freeze_card.json"), "feature_sha256": freeze_card["feature_sha256"],
        "distance_survival": hazard, "early_ictal_field_and_path": fields,
        "primary_contrasts": {
            "state_gain_over_background_censored_logscore": None if base is None or full is None else base - full,
            "state_gain_over_random_capacity_control": None if random is None or full is None else random - full,
            "background_state_gain_over_current_background_censored_logscore": (
                None if background_base is None or background_full is None
                else background_base - background_full
            ),
            "background_state_gain_over_random_background": (
                None if background_random is None or background_full is None
                else background_random - background_full
            ),
            "event_only_state_gain_over_mark_history": None if event_base is None or event_full is None else event_base - event_full,
            "grid_state_gain_over_mark_history": None if event_base is None or grid_full is None else event_base - grid_full,
            "event_correct_time_gain_over_shift": temporal("B_history_plus_S_event", "correct_time_gain_over_shift"),
            "event_dynamic_gain_over_fit_period_mean": temporal("B_history_plus_S_event", "dynamic_gain_over_fit_period_mean"),
            "grid_correct_time_gain_over_shift": temporal("B_history_plus_S_grid", "correct_time_gain_over_shift"),
            "grid_dynamic_gain_over_fit_period_mean": temporal("B_history_plus_S_grid", "dynamic_gain_over_fit_period_mean"),
            "dual_correct_time_gain_over_shift": temporal("B_context_plus_S_dual", "correct_time_gain_over_shift"),
            "dual_dynamic_gain_over_fit_period_mean": temporal("B_context_plus_S_dual", "dynamic_gain_over_fit_period_mean"),
            "background_correct_time_gain_over_shift": temporal(
                "B_history_current_background_plus_S_background", "correct_time_gain_over_shift"
            ),
            "background_dynamic_gain_over_fit_period_mean": temporal(
                "B_history_current_background_plus_S_background", "dynamic_gain_over_fit_period_mean"
            ),
        },
        "frozen_contract": "seizure gradients cannot reach event encoder, observers, time bank or contact decoder",
        "sleep_adjustment": "not available; clock adjustment is not vigilance control",
        "hazard_contract": (
            "one coherent five-minute discrete survival likelihood with six pre-specified "
            "elapsed-time baseline bands; correct-time, phase-contained six-hour circular "
            "shift and FIT-mean state are scored through the same fitted readout; every "
            "state arm fits only an appended residual over a frozen selected baseline"
        ),
        "early_field_contract": (
            "baseline field coefficients are frozen before state residual fitting; endpoints "
            "are early group-event participation, first-arrival and provenance-joined bb150 energy fields"
        ),
        "code_provenance": _code_provenance(Path(__file__)),
        'hazard_optimization_budget_extension': extend_budget,
        'hazard_instrument_controls_enabled': instrument_controls,
        'ictal_cache_root': None if ictal_cache_root is None else str(ictal_cache_root),
        "seizure_outcomes_read": True, "development_targets_read": False, "sealed_partition_opened": False,
    }
    if instrument_controls:
        capacity = logloss('B_history_plus_random_history')
        oracle_loss = logloss('B_history_plus_future_oracle')
        card['hazard_instrument_controls'] = {
            'matched_history_capacity_definition': 'original frozen history projection; same appended width as event/grid state',
            'not_an_initial_observer_control': True,
            'future_oracle_definition': 'time to next onset <=30min, <=2h, exp(-time/2h); deliberately leaked diagnostic',
            'oracle_never_selects_interictal_state': True,
            'future_oracle_gain_over_history': None if oracle_loss is None or event_base is None else event_base - oracle_loss,
            'event_gain_over_matched_history_capacity': None if capacity is None or event_full is None else capacity - event_full,
            'grid_gain_over_matched_history_capacity': None if capacity is None or grid_full is None else capacity - grid_full,
        }
    atomic_json(out_dir / "card.json", card); return card
