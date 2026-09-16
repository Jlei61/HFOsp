"""Persistent H3 readouts on a frozen v0.3.7 common-drive generator.

The primary H3 run fitted only the immediately preceding five-minute block.
This module uses the already-present 2/6/24/48 h continuous-time feedback
banks and fits their zero-bias readouts.  It remains an observational test of
feedback-like directional dependence, not an intervention-level causal claim.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from .contracts import atomic_json, sha256_file
from .h1_train import _code_provenance
from .h3_generative import (
    H3Config,
    PhysiologicalGenerativeState,
    _causal_delayed_inputs,
    _negative_binomial_nll,
    _score,
    _tensorise,
    build_h3_data,
)


RIDGES = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)


def _ridge_boundary_audit(
    candidates: list[dict[str, Any]], metric: str,
) -> dict[str, Any]:
    """Distinguish an explicitly selected zero edge from a truncated grid."""

    finite = [
        row for row in candidates
        if row.get(metric) is not None and np.isfinite(float(row[metric]))
    ]
    if not finite:
        return {
            "selected_penalty": None,
            "selected_zero_edge": True,
            "saturated_at_upper_edge": False,
            "grid_upper_edge": float(max(RIDGES)),
        }
    best = min(finite, key=lambda row: float(row[metric]))
    penalty = best.get("penalty")
    return {
        "selected_penalty": None if penalty is None else float(penalty),
        "selected_zero_edge": bool(best.get("zero_edge", False)),
        "saturated_at_upper_edge": bool(
            penalty is not None and float(penalty) >= float(max(RIDGES))
        ),
        "grid_upper_edge": float(max(RIDGES)),
    }


def _rms_scale(x: Tensor, rows: Tensor) -> Tensor:
    return torch.sqrt(torch.mean(x[rows].square(), dim=0)).clamp_min(1e-5)


def _fit_gaussian_edge(
    x: Tensor,
    target: Tensor,
    base: Tensor,
    fit_rows: Tensor,
    inner_rows: Tensor,
) -> tuple[Tensor, dict[str, Any]]:
    """Choose a zero-bias multi-output ridge on INNER, with zero as a candidate."""

    zero = torch.zeros(target.shape[1], x.shape[1], dtype=x.dtype, device=x.device)
    if x.shape[1] == 0 or fit_rows.numel() == 0 or inner_rows.numel() == 0:
        return zero, {
            "status": "NOT_ESTIMABLE", "selected_zero_edge": True,
            "fit_rows": int(fit_rows.numel()), "inner_rows": int(inner_rows.numel()),
            "candidate_grid": [], "raw_design_rank": 0,
            "raw_design_condition": None,
        }
    scale = _rms_scale(x, fit_rows)
    xs = x / scale
    residual = target - base
    gram = xs[fit_rows].T @ xs[fit_rows]
    rhs = xs[fit_rows].T @ residual[fit_rows]
    eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    best_weight = zero
    best = float(torch.mean(residual[inner_rows].square()))
    candidates = [{"penalty": None, "inner_mse": best, "zero_edge": True}]
    for penalty in RIDGES:
        weight_scaled = torch.linalg.solve(
            gram + float(penalty * fit_rows.numel()) * eye, rhs
        )
        weight = (weight_scaled / scale[:, None]).T
        prediction = base[inner_rows] + x[inner_rows] @ weight.T
        value = float(torch.mean((prediction - target[inner_rows]).square()))
        candidates.append({"penalty": penalty, "inner_mse": value, "zero_edge": False})
        if np.isfinite(value) and value < best - 1e-5:
            best = value; best_weight = weight
    raw = x[fit_rows].detach().cpu().numpy()
    singular = np.linalg.svd(raw, compute_uv=False)
    audit = {
        "status": "ESTIMATED",
        "selected_inner_mse": best,
        "selected_zero_edge": bool(torch.count_nonzero(best_weight) == 0),
        "candidate_grid": candidates,
        "fit_rows": int(fit_rows.numel()), "inner_rows": int(inner_rows.numel()),
        "raw_design_rank": int(np.linalg.matrix_rank(raw)),
        "raw_design_condition": float(singular[0] / max(singular[-1], 1e-12)),
    }
    audit["penalty_grid_diagnostics"] = _ridge_boundary_audit(
        candidates, "inner_mse"
    )
    return best_weight, audit


def _fit_count_edge(
    x: Tensor,
    base_log_rate: Tensor,
    count: Tensor,
    exposure: Tensor,
    dispersion: Tensor,
    fit_rows: Tensor,
    inner_rows: Tensor,
    grid_seconds: float,
) -> tuple[Tensor, dict[str, Any]]:
    zero_weight = torch.zeros(1, x.shape[1], dtype=x.dtype, device=x.device)
    if x.shape[1] == 0 or fit_rows.numel() == 0 or inner_rows.numel() == 0:
        return zero_weight, {
            "status": "NOT_ESTIMABLE", "selected_zero_edge": True,
            "fit_rows": int(fit_rows.numel()), "inner_rows": int(inner_rows.numel()),
            "candidate_grid": [], "raw_design_rank": 0,
            "raw_design_condition": None,
        }
    scale = _rms_scale(x, fit_rows)
    xs = x / scale

    def score(coefficient: Tensor, rows: Tensor) -> Tensor:
        log_rate = base_log_rate[rows] + xs[rows] @ coefficient
        mean = torch.exp(torch.clamp(log_rate, -10.0, 12.0)) * (exposure[rows] / grid_seconds)
        return _negative_binomial_nll(count[rows], mean, dispersion).mean()

    zero = torch.zeros(x.shape[1], dtype=x.dtype, device=x.device)
    best_coef = zero
    best = float(score(zero, inner_rows))
    candidates = [{"penalty": None, "inner_nll": best, "zero_edge": True}]
    for penalty in RIDGES:
        coefficient = torch.zeros_like(zero, requires_grad=True)
        optimiser = torch.optim.LBFGS(
            [coefficient], lr=0.5, max_iter=100, tolerance_grad=1e-9,
            tolerance_change=1e-11, line_search_fn="strong_wolfe",
        )

        def closure() -> Tensor:
            optimiser.zero_grad(set_to_none=True)
            value = score(coefficient, fit_rows) + float(penalty) * coefficient.square().mean()
            value.backward()
            return value

        optimiser.step(closure)
        value = float(score(coefficient.detach(), inner_rows))
        candidates.append({"penalty": penalty, "inner_nll": value, "zero_edge": False})
        if np.isfinite(value) and value < best - 1e-5:
            best = value; best_coef = coefficient.detach().clone()
    weight = (best_coef / scale)[None]
    raw = x[fit_rows].detach().cpu().numpy()
    singular = np.linalg.svd(raw, compute_uv=False)
    audit = {
        "status": "ESTIMATED",
        "selected_inner_nll": best,
        "selected_zero_edge": bool(torch.count_nonzero(best_coef) == 0),
        "candidate_grid": candidates,
        "fit_rows": int(fit_rows.numel()), "inner_rows": int(inner_rows.numel()),
        "raw_design_rank": int(np.linalg.matrix_rank(raw)),
        "raw_design_condition": float(singular[0] / max(singular[-1], 1e-12)),
    }
    audit["penalty_grid_diagnostics"] = _ridge_boundary_audit(
        candidates, "inner_nll"
    )
    return weight, audit


def _feedback_features(
    model: PhysiologicalGenerativeState,
    tensors: dict[str, Tensor],
    config: H3Config,
    *,
    tau_indices: list[int] | None = None,
    include_immediate: bool = True,
) -> tuple[Tensor, Tensor, list[int], list[int]]:
    with torch.no_grad():
        state = model(
            tensors["time"], tensors["segment"], tensors["context"],
            tensors["count_input"], tensors["grammar_input"], "M2_mark_feedback",
        )[0]
    start = model.core_state_dim
    count_full = state[:, start:start + model.feedback_scales]
    mark_full = state[:, start + model.feedback_scales:]
    # The primary M0 has multiple channels per common-drive tau.  Its feedback
    # bank repeats the same scalar/vector impulse on those channels, so fitting
    # all copies would create a rank-deficient design without adding evidence.
    # Keep the immediate block and one column per unique physical tau.  The
    # fitted coefficient is written to the first identical channel; all other
    # copies remain exactly zero.
    chosen_tau = list(range(len(config.taus_seconds))) if tau_indices is None else list(tau_indices)
    scale_indices = ([0] if include_immediate else []) + [
        1 + index * int(config.channels_per_tau)
        for index in chosen_tau
    ]
    count = count_full[:, scale_indices]
    grammar_dim = model.grammar_dim
    mark_indices = [
        scale * grammar_dim + component
        for scale in scale_indices for component in range(grammar_dim)
    ]
    mark = mark_full[:, mark_indices]
    return count, mark, scale_indices, mark_indices


def _fit_family_edges(
    model: PhysiologicalGenerativeState,
    parent_family: str,
    x: Tensor,
    tensors: dict[str, Tensor],
    split: dict[str, Tensor],
    *,
    prefix: str,
    grid_seconds: float,
    minimum_exposure_fraction: float,
    readout_columns: list[int],
    preserve_other_columns: bool = False,
) -> dict[str, Any]:
    with torch.no_grad():
        _state, base_count, base_grammar, base_background = model(
            tensors["time"], tensors["segment"], tensors["context"],
            tensors["count_input"], tensors["grammar_input"], parent_family,
        )
    count_valid = (
        tensors["exposure"] / grid_seconds
    ) >= float(minimum_exposure_fraction)
    count_fit = split["FIT"][count_valid[split["FIT"]]]
    count_inner = split["INNER"][count_valid[split["INNER"]]]
    grammar_fit = split["FIT"][tensors["grammar_valid"][split["FIT"]]]
    grammar_inner = split["INNER"][tensors["grammar_valid"][split["INNER"]]]
    background_fit = split["FIT"][tensors["background_valid"][split["FIT"]]]
    background_inner = split["INNER"][tensors["background_valid"][split["INNER"]]]
    count_weight, count_audit = _fit_count_edge(
        x, base_count, tensors["count"], tensors["exposure"], model.log_dispersion,
        count_fit, count_inner, grid_seconds,
    )
    grammar_weight, grammar_audit = _fit_gaussian_edge(
        x, tensors["grammar_target"], base_grammar, grammar_fit, grammar_inner,
    )
    background_weight, background_audit = _fit_gaussian_edge(
        x, tensors["background_target"], base_background, background_fit, background_inner,
    )
    with torch.no_grad():
        count_layer = getattr(model, f"{prefix}_to_count")
        mark_layer = getattr(model, f"{prefix}_to_mark")
        background_layer = getattr(model, f"{prefix}_to_background")
        if not preserve_other_columns:
            count_layer.weight.zero_(); mark_layer.weight.zero_(); background_layer.weight.zero_()
        elif readout_columns:
            count_layer.weight[:, readout_columns] = 0.0
            mark_layer.weight[:, readout_columns] = 0.0
            background_layer.weight[:, readout_columns] = 0.0
        count_layer.weight[:, readout_columns] = count_weight
        mark_layer.weight[:, readout_columns] = grammar_weight
        background_layer.weight[:, readout_columns] = background_weight
    return {"count": count_audit, "conditional_grammar": grammar_audit,
            "future_background": background_audit}


def _score_additive_edges(
    base_count: Tensor,
    base_grammar: Tensor,
    base_background: Tensor,
    x: Tensor,
    weights: dict[str, Tensor],
    tensors: dict[str, Tensor],
    rows: Tensor,
    dispersion: Tensor,
    grid_seconds: float,
    minimum_exposure_fraction: float,
) -> dict[str, Any]:
    if rows.numel() == 0:
        return {"status": "NOT_ESTIMABLE", "total": None, "count": None,
                "conditional_grammar": None, "future_background": None,
                "n_grid_blocks": 0}
    log_rate = base_count + x @ weights["count"].T[:, 0]
    grammar = base_grammar + x @ weights["conditional_grammar"].T
    background = base_background + x @ weights["future_background"].T
    exposure_fraction = tensors["exposure"] / float(grid_seconds)
    count_valid = exposure_fraction >= float(minimum_exposure_fraction)
    use_count = rows[count_valid[rows]]
    mean = torch.exp(torch.clamp(log_rate[use_count], -10.0, 12.0)) * exposure_fraction[use_count]
    count_loss = _negative_binomial_nll(
        tensors["count"][use_count], mean, dispersion
    ).mean()
    use_grammar = rows[tensors["grammar_valid"][rows]]
    grammar_loss = (
        torch.mean((grammar[use_grammar] - tensors["grammar_target"][use_grammar]).square())
        if use_grammar.numel() else count_loss.new_tensor(float("nan"))
    )
    use_background = rows[tensors["background_valid"][rows]]
    background_loss = (
        torch.mean((background[use_background] - tensors["background_target"][use_background]).square())
        if use_background.numel() else count_loss.new_tensor(float("nan"))
    )
    finite = [count_loss] + [value for value in (grammar_loss, background_loss) if torch.isfinite(value)]
    return {
        "status": "ESTIMATED", "total": float(torch.stack(finite).mean()),
        "count": float(count_loss),
        "conditional_grammar": float(grammar_loss) if torch.isfinite(grammar_loss) else None,
        "future_background": float(background_loss) if torch.isfinite(background_loss) else None,
        "n_grid_blocks": int(rows.numel()),
    }


def _fit_additive_edges(
    base_count: Tensor,
    base_grammar: Tensor,
    base_background: Tensor,
    x: Tensor,
    tensors: dict[str, Tensor],
    split: dict[str, Tensor],
    dispersion: Tensor,
    grid_seconds: float,
    minimum_exposure_fraction: float,
) -> tuple[dict[str, Tensor], dict[str, Any], dict[str, Any]]:
    count_valid = (
        tensors["exposure"] / float(grid_seconds)
    ) >= float(minimum_exposure_fraction)
    count_fit = split["FIT"][count_valid[split["FIT"]]]
    count_inner = split["INNER"][count_valid[split["INNER"]]]
    grammar_fit = split["FIT"][tensors["grammar_valid"][split["FIT"]]]
    grammar_inner = split["INNER"][tensors["grammar_valid"][split["INNER"]]]
    background_fit = split["FIT"][tensors["background_valid"][split["FIT"]]]
    background_inner = split["INNER"][tensors["background_valid"][split["INNER"]]]
    count_weight, count_audit = _fit_count_edge(
        x, base_count, tensors["count"], tensors["exposure"], dispersion,
        count_fit, count_inner, grid_seconds,
    )
    grammar_weight, grammar_audit = _fit_gaussian_edge(
        x, tensors["grammar_target"], base_grammar, grammar_fit, grammar_inner,
    )
    background_weight, background_audit = _fit_gaussian_edge(
        x, tensors["background_target"], base_background, background_fit, background_inner,
    )
    weights = {"count": count_weight, "conditional_grammar": grammar_weight,
               "future_background": background_weight}
    audit = {"count": count_audit, "conditional_grammar": grammar_audit,
             "future_background": background_audit}
    score = _score_additive_edges(
        base_count, base_grammar, base_background, x, weights, tensors,
        split["SELECTION"], dispersion, grid_seconds, minimum_exposure_fraction,
    )
    return weights, audit, score


def _phase_tau_windows(data, phase: str, tau_seconds: float, grid_seconds: float) -> int:
    rows_in_phase = np.flatnonzero(np.asarray(data.phase).astype(str) == phase)
    total = 0
    for segment in np.unique(data.segment[rows_in_phase]):
        rows = rows_in_phase[data.segment[rows_in_phase] == segment]
        if rows.size:
            span = float(data.time[rows[-1]] - data.time[rows[0]] + grid_seconds)
            total += int(np.floor(span / float(tau_seconds)))
    return total


def _eligible_persistent_taus(data, config: H3Config) -> tuple[list[int], dict[str, Any]]:
    """Apply the long-window contract before looking at any target value."""

    audit: dict[str, Any] = {}
    eligible: list[int] = []
    for index, tau in enumerate(config.taus_seconds):
        counts = {
            phase: _phase_tau_windows(data, phase, float(tau), config.grid_seconds)
            for phase in ("FIT", "INNER", "SELECTION")
        }
        passed = counts["FIT"] >= 4 and counts["INNER"] >= 2 and counts["SELECTION"] >= 3
        audit[str(int(tau))] = {
            "nonoverlapping_windows_by_phase": counts,
            "eligible": passed,
            "rule": "FIT>=4, INNER>=2, SELECTION>=3 non-overlapping physical tau windows",
        }
        if passed:
            eligible.append(index)
    return eligible, audit


def _copy_primary_mark_edge(
    destination: PhysiologicalGenerativeState,
    source_state: dict[str, Tensor],
) -> None:
    with torch.no_grad():
        for name in (
            "mark_feedback_to_count", "mark_feedback_to_mark",
            "mark_feedback_to_background",
        ):
            layer = getattr(destination, name)
            source_weight = source_state[f"{name}.weight"].to(device=layer.weight.device)
            layer.weight[:, :destination.grammar_dim].copy_(
                source_weight[:, :destination.grammar_dim]
            )



def _floored_gain(floor_entry: Mapping[str, Any], real_score: float) -> float | None:
    """Real-over-null gain with the null floored at the no-edge model."""
    if floor_entry.get("status") != "ESTIMATED":
        return None
    return float(floor_entry["floored"]) - float(real_score)


def _perturbed_inputs(data, mode: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    count = np.asarray(data.count_input).copy()
    grammar = np.asarray(data.grammar_input).copy()
    rng = np.random.default_rng(int(seed))
    # A fitted placebo must never move INNER/SELECTION inputs into FIT.  The
    # shuffle therefore acts separately within each carry segment and phase.
    for seg in np.unique(data.segment):
        for phase in np.unique(np.asarray(data.phase).astype(str)):
            rows = np.flatnonzero(
                (data.segment == seg) & (np.asarray(data.phase).astype(str) == phase)
            )
            if rows.size < 2:
                continue
            if mode == "circular":
                donor = np.roll(rows, max(1, rows.size // 2))
                count[rows] = count[donor]; grammar[rows] = grammar[donor]
            elif mode == "reverse":
                count[rows] = count[rows[::-1]]; grammar[rows] = grammar[rows[::-1]]
            elif mode == "count_shuffle":
                count[rows] = count[rng.permutation(rows)]
            elif mode == "mark_shuffle":
                valid = rows[data.grammar_valid[rows]]
                if valid.size > 1:
                    grammar[valid] = grammar[rng.permutation(valid)]
            else:
                raise ValueError(mode)
    return count, grammar


def train_persistent_h3_subject(
    subject: str,
    seed: int,
    *,
    source_root: Path,
    out_dir: Path,
    device: torch.device,
    data_override=None,
    source_dir_override: Path | None = None,
) -> dict[str, Any]:
    started = time.time()
    source_dir = (
        Path(source_dir_override) if source_dir_override is not None
        else Path(source_root) / subject / f"seed{seed}"
    )
    source_card = __import__("json").loads((source_dir / "card.json").read_text(encoding="utf-8"))
    source = torch.load(source_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    config = H3Config(**source["config"])
    data = data_override if data_override is not None else build_h3_data(subject, seed, config)
    tensors = _tensorise(data, device)
    split = {name: torch.as_tensor(np.flatnonzero(data.phase == name), dtype=torch.long, device=device)
             for name in ("FIT", "INNER", "SELECTION")}
    model0 = PhysiologicalGenerativeState(
        data.context.shape[1], data.grammar_input.shape[1], data.grammar_target.shape[1],
        data.background_target.shape[1], config,
    ).to(device)
    model0.load_state_dict(source["models"]["M0_common_drive"])
    eligible_tau_indices, tau_eligibility = _eligible_persistent_taus(data, config)
    model1_primary = copy.deepcopy(model0)
    model1_primary.load_state_dict(
        source["models"].get("M1_count_feedback", source["models"]["M0_common_drive"])
    )
    model1 = copy.deepcopy(model1_primary)
    count_features, _mark_features, count_columns, _mark_columns = _feedback_features(
        model1, tensors, config, tau_indices=eligible_tau_indices,
        include_immediate=False,
    )
    fit1 = _fit_family_edges(
        model1, "M1_count_feedback", count_features, tensors, split,
        prefix="count_feedback", grid_seconds=config.grid_seconds,
        minimum_exposure_fraction=config.minimum_exposure_fraction,
        readout_columns=count_columns, preserve_other_columns=True,
    )
    model2_primary = copy.deepcopy(model1)
    source_m2 = source["models"].get("M2_mark_feedback", source["models"]["M0_common_drive"])
    # The primary H3 contract fitted only the immediately preceding block.
    # Persistent columns stay zero until this extension.
    _copy_primary_mark_edge(model2_primary, source_m2)
    model2 = copy.deepcopy(model2_primary)
    _count_features, mark_features, _count_columns, mark_columns = _feedback_features(
        model2, tensors, config, tau_indices=eligible_tau_indices,
        include_immediate=False,
    )
    fit2 = _fit_family_edges(
        model2, "M2_mark_feedback", mark_features, tensors, split,
        prefix="mark_feedback", grid_seconds=config.grid_seconds,
        minimum_exposure_fraction=config.minimum_exposure_fraction,
        readout_columns=mark_columns, preserve_other_columns=True,
    )
    models = {
        "M0_common_drive": (model0, "M0_common_drive"),
        "M1_primary_one_step": (model1_primary, "M1_count_feedback"),
        "M1_persistent_count": (model1, "M1_count_feedback"),
        "M2_primary_one_step_on_persistent_count": (model2_primary, "M2_mark_feedback"),
        "M2_persistent_mark": (model2, "M2_mark_feedback"),
    }
    scores = {
        label: _score(model, family, tensors, split["SELECTION"])
        for label, (model, family) in models.items()
    }

    # Same-capacity fitted wrong-time controls.  Both placebo models receive
    # exactly the persistent columns available to the real edge and select
    # their ridge on INNER; the only change is phase-contained circular
    # reassignment of the event inputs.  This is stronger than merely applying
    # a real-edge coefficient to a shuffled trajectory.
    circular_count_np, circular_grammar_np = _perturbed_inputs(data, "circular", seed + 1907)
    circular_tensors = dict(tensors)
    circular_tensors["count_input"] = torch.as_tensor(
        circular_count_np, dtype=torch.float32, device=device
    )
    circular_tensors["grammar_input"] = torch.as_tensor(
        circular_grammar_np, dtype=torch.float32, device=device
    )
    circular_count_features = _feedback_features(
        model1_primary, circular_tensors, config,
        tau_indices=eligible_tau_indices, include_immediate=False,
    )[0]
    with torch.no_grad():
        _state, base1_count, base1_grammar, base1_background = model1_primary(
            tensors["time"], tensors["segment"], tensors["context"],
            tensors["count_input"], tensors["grammar_input"], "M1_count_feedback",
        )
    _count_placebo_weights, count_placebo_fit, count_placebo_score = _fit_additive_edges(
        base1_count, base1_grammar, base1_background, circular_count_features,
        tensors, split, model1_primary.log_dispersion, config.grid_seconds,
        config.minimum_exposure_fraction,
    )
    circular_mark_features = _feedback_features(
        model2_primary, circular_tensors, config,
        tau_indices=eligible_tau_indices, include_immediate=False,
    )[1]
    with torch.no_grad():
        _state, base2_count, base2_grammar, base2_background = model2_primary(
            tensors["time"], tensors["segment"], tensors["context"],
            tensors["count_input"], tensors["grammar_input"], "M2_mark_feedback",
        )
    _mark_placebo_weights, mark_placebo_fit, mark_placebo_score = _fit_additive_edges(
        base2_count, base2_grammar, base2_background, circular_mark_features,
        tensors, split, model2_primary.log_dispersion, config.grid_seconds,
        config.minimum_exposure_fraction,
    )
    # A fitted same-capacity placebo selects its own ridge on INNER, so it can
    # overfit and then generalise WORSE on SELECTION than simply having no edge
    # at all.  When that happens, "real beats placebo" measures the placebo's
    # overfitting rather than the real edge's skill.  The honest null is
    # therefore floored at the no-edge model: a rational modeller offered a
    # wrong-time edge would fall back to zero rather than keep a harmful one.
    def _floor_at_no_edge(placebo: Mapping[str, Any], endpoint: str) -> dict[str, Any]:
        no_edge = scores["M0_common_drive"]
        if placebo.get("status") != "ESTIMATED" or no_edge.get("status") != "ESTIMATED":
            return {"status": "NOT_ESTIMABLE"}
        if int(placebo.get("n_grid_blocks", -1)) != int(no_edge.get("n_grid_blocks", -2)):
            return {"status": "NOT_COMPARABLE_DIFFERENT_SUPPORT"}
        raw = float(placebo[endpoint]); floor = float(no_edge[endpoint])
        return {
            "status": "ESTIMATED",
            "placebo": raw,
            "no_edge": floor,
            "placebo_over_no_edge": floor - raw,
            "placebo_worse_than_no_edge": bool(raw > floor),
            "floored": min(raw, floor),
        }

    placebo_floor = {
        "count_future_background": _floor_at_no_edge(count_placebo_score, "future_background"),
        "count_count": _floor_at_no_edge(count_placebo_score, "count"),
        "mark_future_background": _floor_at_no_edge(mark_placebo_score, "future_background"),
        "mark_conditional_grammar": _floor_at_no_edge(mark_placebo_score, "conditional_grammar"),
    }
    fitted_placebos = {
        "count_phase_circular": {"fitting": count_placebo_fit, "selection_score": count_placebo_score},
        "mark_phase_circular": {"fitting": mark_placebo_fit, "selection_score": mark_placebo_score},
        "same_parameter_count_as_corresponding_real_persistent_edge": True,
        "phase_contained_no_split_crossing": True,
        "no_edge_floor": placebo_floor,
        "why_floored": (
            "a fitted wrong-time placebo that generalises worse than the no-edge model "
            "inflates the real-over-placebo contrast by its own overfitting; the floored "
            "contrast is the one that may be reported"
        ),
    }

    # The combined eligible bank answers whether any admissible persistent
    # mode helps. These independently fitted one-tau sensitivities prevent a
    # 2 h signal from being narrated as 6 h (or vice versa).
    single_scale_sensitivity: dict[str, Any] = {}
    for tau_index in eligible_tau_indices:
        tau_key = str(int(config.taus_seconds[tau_index]))
        model1_tau = copy.deepcopy(model1_primary)
        x_count_tau, _x_mark_tau, count_columns_tau, _mark_columns_tau = _feedback_features(
            model1_tau, tensors, config, tau_indices=[tau_index], include_immediate=False,
        )
        fit1_tau = _fit_family_edges(
            model1_tau, "M1_count_feedback", x_count_tau, tensors, split,
            prefix="count_feedback", grid_seconds=config.grid_seconds,
            minimum_exposure_fraction=config.minimum_exposure_fraction,
            readout_columns=count_columns_tau, preserve_other_columns=True,
        )
        score1_tau = _score(
            model1_tau, "M1_count_feedback", tensors, split["SELECTION"]
        )
        model2_primary_tau = copy.deepcopy(model1_tau)
        _copy_primary_mark_edge(model2_primary_tau, source_m2)
        model2_tau = copy.deepcopy(model2_primary_tau)
        _x_count_tau, x_mark_tau, _count_columns_tau, mark_columns_tau = _feedback_features(
            model2_tau, tensors, config, tau_indices=[tau_index], include_immediate=False,
        )
        fit2_tau = _fit_family_edges(
            model2_tau, "M2_mark_feedback", x_mark_tau, tensors, split,
            prefix="mark_feedback", grid_seconds=config.grid_seconds,
            minimum_exposure_fraction=config.minimum_exposure_fraction,
            readout_columns=mark_columns_tau, preserve_other_columns=True,
        )
        score2_primary_tau = _score(
            model2_primary_tau, "M2_mark_feedback", tensors, split["SELECTION"]
        )
        score2_tau = _score(
            model2_tau, "M2_mark_feedback", tensors, split["SELECTION"]
        )
        single_scale_sensitivity[tau_key] = {
            "count_fitting": fit1_tau, "mark_fitting": fit2_tau,
            "M1_persistent_score": score1_tau,
            "M2_primary_score": score2_primary_tau,
            "M2_persistent_score": score2_tau,
            "contrasts": {
                "persistent_count_over_one_step_count": scores["M1_primary_one_step"]["count"] - score1_tau["count"],
                "persistent_count_over_one_step_background": scores["M1_primary_one_step"]["future_background"] - score1_tau["future_background"],
                "persistent_mark_over_one_step_grammar": score2_primary_tau["conditional_grammar"] - score2_tau["conditional_grammar"],
                "persistent_mark_over_one_step_background": score2_primary_tau["future_background"] - score2_tau["future_background"],
            },
            "interpretation": "single physical tau sensitivity; not selected as a patient-specific best scale",
        }

    delayed_count_np, delayed_grammar_np, delayed_valid_np = _causal_delayed_inputs(data, 21600.0)
    delayed_valid = torch.as_tensor(delayed_valid_np, dtype=torch.bool, device=device)
    paired = split["SELECTION"][delayed_valid[split["SELECTION"]]]
    controls = {
        "delay_seconds": 21600.0,
        "paired_selection_blocks": int(paired.numel()),
        "correct_on_delayed_support": {
            label: _score(model, family, tensors, paired)
            for label, (model, family) in models.items()
        },
        "M1_persistent_delayed_count": _score(
            model1, "M1_count_feedback", tensors, paired,
            count_input=torch.as_tensor(delayed_count_np, dtype=torch.float32, device=device),
        ),
        "M2_persistent_delayed_mark": _score(
            model2, "M2_mark_feedback", tensors, paired,
            grammar_input=torch.as_tensor(delayed_grammar_np, dtype=torch.float32, device=device),
        ),
    }
    fit_np = np.flatnonzero(data.phase == "FIT")
    constant_count = np.broadcast_to(np.mean(data.count_input[fit_np], axis=0, keepdims=True), data.count_input.shape).copy()
    grammar_fit = fit_np[data.grammar_valid[fit_np]]
    constant_grammar = np.broadcast_to(np.mean(data.grammar_input[grammar_fit], axis=0, keepdims=True), data.grammar_input.shape).copy()
    controls["M1_persistent_fit_mean_count"] = _score(
        model1, "M1_count_feedback", tensors, split["SELECTION"],
        count_input=torch.as_tensor(constant_count, dtype=torch.float32, device=device),
    )
    controls["M2_persistent_fit_mean_mark"] = _score(
        model2, "M2_mark_feedback", tensors, split["SELECTION"],
        grammar_input=torch.as_tensor(constant_grammar, dtype=torch.float32, device=device),
    )
    for mode in ("circular", "reverse", "count_shuffle", "mark_shuffle"):
        count_np, grammar_np = _perturbed_inputs(data, mode, seed + 911)
        family = "M1_count_feedback" if mode == "count_shuffle" else "M2_mark_feedback"
        model = model1 if family == "M1_count_feedback" else model2
        controls[mode] = _score(
            model, family, tensors, split["SELECTION"],
            count_input=torch.as_tensor(count_np, dtype=torch.float32, device=device),
            grammar_input=torch.as_tensor(grammar_np, dtype=torch.float32, device=device),
        )

    impulses = {label: model.impulse_response(
        tensors["count_input"][split["SELECTION"]],
        tensors["grammar_input"][split["SELECTION"]], family,
        (300.0, 1800.0, 7200.0, 21600.0, 86400.0, 172800.0),
    ) for label, (model, family) in models.items()}
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "checkpoint.pt"
    torch.save({"config": asdict(config),
                "models": {label: model.state_dict() for label, (model, _family) in models.items()}}, checkpoint)
    selection_rows = np.flatnonzero(data.phase == "SELECTION")
    selection_hours = float(np.sum(data.exposure_seconds[selection_rows]) / 3600.0)
    selection_segment_spans = []
    for segment in np.unique(data.segment[selection_rows]):
        rows = selection_rows[data.segment[selection_rows] == segment]
        if rows.size:
            selection_segment_spans.append(
                float(data.time[rows[-1]] - data.time[rows[0]] + config.grid_seconds)
            )
    card = {
        "format": "group_event_state_v0_3_7_h3_persistent_feedback_card_v2",
        "code_provenance": _code_provenance(Path(__file__)),
        "subject": subject, "seed": int(seed),
        "source_primary_card": str(source_dir / "card.json"),
        "source_primary_card_format": source_card.get("format"),
        "source_primary_checkpoint_sha256": sha256_file(source_dir / "checkpoint.pt"),
        "state_semantics": "Z_phys_generative_candidate_not_S_obs",
        "common_drive_core_frozen": True, "all_feedback_readouts_zero_bias": True,
        "duplicate_same_tau_feedback_channels_collapsed_before_fitting": True,
        "available_feedback_scales_seconds": [0.0, *map(float, config.taus_seconds)],
        "persistent_scales_fitted_seconds": [
            float(config.taus_seconds[index]) for index in eligible_tau_indices
        ],
        "tau_eligibility": tau_eligibility,
        "fitting": {"M1_persistent_count": fit1, "M2_persistent_mark": fit2},
        "fitted_equal_capacity_wrong_time_placebos": fitted_placebos,
        "single_scale_sensitivity": single_scale_sensitivity,
        "scores": scores, "controls": controls, "impulse_response": impulses,
        "primary_contrasts": {
            "persistent_count_over_one_step_count": scores["M1_primary_one_step"]["count"] - scores["M1_persistent_count"]["count"],
            "persistent_count_over_one_step_grammar": scores["M1_primary_one_step"]["conditional_grammar"] - scores["M1_persistent_count"]["conditional_grammar"],
            "persistent_count_over_one_step_background": scores["M1_primary_one_step"]["future_background"] - scores["M1_persistent_count"]["future_background"],
            "persistent_mark_over_one_step_count": scores["M2_primary_one_step_on_persistent_count"]["count"] - scores["M2_persistent_mark"]["count"],
            "persistent_mark_over_one_step_grammar": scores["M2_primary_one_step_on_persistent_count"]["conditional_grammar"] - scores["M2_persistent_mark"]["conditional_grammar"],
            "persistent_mark_over_one_step_background": scores["M2_primary_one_step_on_persistent_count"]["future_background"] - scores["M2_persistent_mark"]["future_background"],
            "M1_persistent_over_M0_count": scores["M0_common_drive"]["count"] - scores["M1_persistent_count"]["count"],
            "M1_persistent_over_M0_background": scores["M0_common_drive"]["future_background"] - scores["M1_persistent_count"]["future_background"],
            "M2_persistent_over_M1_persistent_background": scores["M1_persistent_count"]["future_background"] - scores["M2_persistent_mark"]["future_background"],
            "persistent_count_real_over_fitted_wrong_time_count": count_placebo_score["count"] - scores["M1_persistent_count"]["count"],
            "persistent_count_real_over_fitted_wrong_time_background": count_placebo_score["future_background"] - scores["M1_persistent_count"]["future_background"],
            "persistent_mark_real_over_fitted_wrong_time_grammar": mark_placebo_score["conditional_grammar"] - scores["M2_persistent_mark"]["conditional_grammar"],
            "persistent_mark_real_over_fitted_wrong_time_background": mark_placebo_score["future_background"] - scores["M2_persistent_mark"]["future_background"],
            # Reportable versions: the wrong-time null may never be worse than
            # having no edge at all, so it is floored at the no-edge model.
            "persistent_count_real_over_floored_wrong_time_background": _floored_gain(
                placebo_floor["count_future_background"], scores["M1_persistent_count"]["future_background"]
            ),
            "persistent_count_real_over_floored_wrong_time_count": _floored_gain(
                placebo_floor["count_count"], scores["M1_persistent_count"]["count"]
            ),
            "persistent_mark_real_over_floored_wrong_time_background": _floored_gain(
                placebo_floor["mark_future_background"], scores["M2_persistent_mark"]["future_background"]
            ),
            "persistent_mark_real_over_floored_wrong_time_grammar": _floored_gain(
                placebo_floor["mark_conditional_grammar"], scores["M2_persistent_mark"]["conditional_grammar"]
            ),
        },
        "estimability": {
            "selection_observed_hours": selection_hours,
            "selection_segment_spans_hours": [value / 3600.0 for value in selection_segment_spans],
            "observed_exposure_tau_equivalents_not_independent_windows": {
                str(int(tau)): selection_hours / (float(tau) / 3600.0)
                for tau in config.taus_seconds
            },
            "nonoverlapping_tau_windows_by_segment_span": {
                str(int(tau)): int(sum(np.floor(value / float(tau)) for value in selection_segment_spans))
                for tau in config.taus_seconds
            },
            "warning": "a fitted long-tau coefficient is exploratory when held-out segment spans contain fewer than three non-overlapping tau windows",
        },
        "support": {
            name: {
                "grid_blocks": int(np.sum(data.phase == name)),
                "observed_hours": float(np.sum(data.exposure_seconds[data.phase == name]) / 3600.0),
            }
            for name in ("FIT", "INNER", "SELECTION")
        },
        "transforms": data.transforms,
        "checkpoint_path": str(checkpoint), "elapsed_seconds": time.time() - started,
        "allowed_claim": "feedback-like directional dependence after observed common-drive adjustment; not intervention-level causality",
        "development_targets_read": False, "seizure_targets_read": False,
        "sealed_partition_opened": False,
    }
    atomic_json(out_dir / "card.json", card)
    return card
