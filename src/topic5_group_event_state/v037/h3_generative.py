"""Independent v0.3.7 H3 generative-state models.

``Z_phys`` lives only in this module.  It is a continuous-time latent state
whose emissions are the next fixed-clock event count and conditional grammar.
The three arms differ only by whether the preceding block can jump that state:
M0 has common drive only, M1 adds burden feedback, and M2 also adds conditional
grammar feedback.  These jumps are not copied from the H1 observer.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from src.topic5_group_event_state.v035.background_rate import causal_background_at_grid
from src.topic5_group_event_state.v035.long_windows import exposure_seconds

from .contracts import atomic_json
from .ctssm import affine_associative_scan
from .h1_data import build_h1_subject_data
from .h1_train import _code_provenance
from .h2b import _clock_features


H3_FAMILIES = ("M0_common_drive", "M1_count_feedback", "M2_mark_feedback")


@dataclass(frozen=True)
class H3Config:
    seed: int
    grid_seconds: float = 300.0
    taus_seconds: tuple[float, ...] = (7200.0, 21600.0, 86400.0, 172800.0)
    channels_per_tau: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_steps: int = 1800
    validate_every: int = 25
    patience_checks: int = 20
    gradient_clip: float = 2.0
    minimum_exposure_fraction: float = 0.8


@dataclass(frozen=True)
class H3Data:
    subject: str
    time: np.ndarray
    segment: np.ndarray
    phase: np.ndarray
    exposure_seconds: np.ndarray
    count: np.ndarray
    count_input: np.ndarray
    grammar_input: np.ndarray
    grammar_target: np.ndarray
    grammar_valid: np.ndarray
    background_target: np.ndarray
    background_valid: np.ndarray
    context: np.ndarray
    context_names: tuple[str, ...]
    transforms: dict[str, Any]


def _fit_scale(value: np.ndarray, fit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(value, dtype=np.float64)
    centre = np.nanmedian(x[fit], axis=0)
    scale = 1.4826 * np.nanmedian(np.abs(x[fit] - centre), axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    z = np.clip((np.where(np.isfinite(x), x, centre) - centre) / scale, -12.0, 12.0)
    return z.astype(np.float32), centre, scale


def _residualise(value: np.ndarray, nuisance: np.ndarray, fit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return innovations beyond a FIT-only pre-block common-drive model."""

    x = np.concatenate((np.ones((nuisance.shape[0], 1)), nuisance), axis=1).astype(np.float64)
    y = np.asarray(value, dtype=np.float64)
    penalty = np.eye(x.shape[1], dtype=np.float64) * 1e-2
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x[fit].T @ x[fit] + penalty, x[fit].T @ y[fit])
    return (y - x @ beta).astype(np.float32), beta


def _fit_pca(value: np.ndarray, fit: np.ndarray, rank: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """FIT-only low-rank coordinates for mark innovations."""

    x = np.asarray(value, dtype=np.float64)
    use_rank = max(1, min(int(rank), x.shape[1], int(fit.size)))
    _u, _s, vt = np.linalg.svd(x[fit], full_matrices=False)
    components = vt[:use_rank].T
    # Fix the otherwise arbitrary SVD sign for reproducible payloads.
    for column in range(components.shape[1]):
        pivot = int(np.argmax(np.abs(components[:, column])))
        if components[pivot, column] < 0:
            components[:, column] *= -1.0
    return (x @ components).astype(np.float32), components.astype(np.float32)


def build_h3_data(subject: str, seed: int, config: H3Config) -> H3Data:
    interictal = build_h1_subject_data(subject, seed=int(seed))
    rate = interictal.rate
    time_axis = np.asarray(rate.anchor_time, dtype=np.float64)
    segment = np.asarray(rate.segment, dtype=np.int64)
    phase = np.asarray(rate.phase).astype(str)
    fit = np.flatnonzero(phase == "FIT")
    if fit.size < 8:
        raise ValueError(f"{subject}: H3 needs at least eight FIT grid anchors")

    interval_hi = time_axis + float(config.grid_seconds)
    for seg in np.unique(segment):
        rows = np.flatnonzero(segment == seg)
        if rows.size:
            interval_hi[rows] = np.minimum(interval_hi[rows], float(rate.segment_bounds[int(seg), 1]))
    exposure = np.asarray([
        exposure_seconds(rate.observed_support_bounds, float(lo), float(hi))
        for lo, hi in zip(time_axis, interval_hi)
    ], dtype=np.float32)
    valid_exposure = exposure >= float(config.minimum_exposure_fraction * config.grid_seconds)

    left = np.searchsorted(interictal.event_time, time_axis, side="left")
    right = np.searchsorted(interictal.event_time, interval_hi, side="left")
    count = (right - left).astype(np.float32)
    grammar = np.zeros((time_axis.size, interictal.grammar_mark.shape[1]), dtype=np.float32)
    grammar_valid = valid_exposure & (count > 0)
    for row in np.flatnonzero(grammar_valid):
        grammar[row] = np.mean(interictal.grammar_mark[left[row]:right[row]], axis=0)

    count_rate = np.log1p(count * float(config.grid_seconds) / np.maximum(exposure, 1.0))[:, None]
    count_scaled, count_centre, count_scale = _fit_scale(count_rate, fit[valid_exposure[fit]])
    grammar_fit = fit[grammar_valid[fit]]
    if grammar_fit.size < 3:
        raise ValueError(f"{subject}: H3 needs at least three FIT blocks containing group events")
    grammar_scaled, grammar_centre, grammar_scale = _fit_scale(grammar, grammar_fit)
    background, background_names, background_audit = causal_background_at_grid(rate)
    context_raw = np.concatenate((rate.q_raw, background, _clock_features(time_axis)), axis=1)
    context, context_centre, context_scale = _fit_scale(context_raw, fit)
    physical_width = background.shape[1] - 2
    background_available = np.asarray(background[:, -1] > 0.5, dtype=bool)
    background_fit = fit[background_available[fit]]
    if background_fit.size < 3:
        raise ValueError(f"{subject}: H3 needs at least three FIT background observations")
    background_target, background_target_centre, background_target_scale = _fit_scale(
        background[:, :physical_width], background_fit,
    )
    count_input, count_beta = _residualise(count_scaled, context, fit[valid_exposure[fit]])
    grammar_nuisance = np.concatenate((context, count_scaled), axis=1)
    grammar_residual, grammar_beta = _residualise(grammar_scaled, grammar_nuisance, grammar_fit)
    grammar_input, grammar_components = _fit_pca(grammar_residual, grammar_fit, rank=4)
    # Missing blocks are neither low event burden nor a zero grammar mark.
    count_input[~valid_exposure] = 0.0
    grammar_input[~grammar_valid] = 0.0
    grammar_scaled[~grammar_valid] = 0.0
    names = tuple(f"q_{i}" for i in range(rate.q_raw.shape[1])) + tuple(background_names) + ("clock_sin", "clock_cos")
    return H3Data(
        subject=subject, time=time_axis, segment=segment, phase=phase,
        exposure_seconds=exposure, count=count, count_input=count_input,
        grammar_input=grammar_input, grammar_target=grammar_scaled.copy(),
        grammar_valid=grammar_valid, background_target=background_target,
        background_valid=background_available, context=context, context_names=names,
        transforms={
            "count_centre": count_centre.tolist(), "count_scale": count_scale.tolist(),
            "grammar_centre": grammar_centre.tolist(), "grammar_scale": grammar_scale.tolist(),
            "context_centre": context_centre.tolist(), "context_scale": context_scale.tolist(),
            "count_innovation_beta": count_beta.tolist(),
            "grammar_innovation_beta": grammar_beta.tolist(),
            "grammar_innovation_pca_components": grammar_components.tolist(),
            "grammar_innovation_rank": int(grammar_input.shape[1]),
            "background_target_centre": background_target_centre.tolist(),
            "background_target_scale": background_target_scale.tolist(),
            "background_audit": background_audit,
            "event_feedback_definition": "FIT-residual innovation in the preceding non-overlapping five-minute block; count and conditional grammar are separate",
            "common_drive_definition": "causal rolling history, non-event background, clock, and equal fitted intercept in every arm",
        },
    )


class PhysiologicalGenerativeState(nn.Module):
    """Small stable continuous-discrete model used only for H3."""

    def __init__(
        self, context_dim: int, grammar_input_dim: int, grammar_target_dim: int,
        background_dim: int, config: H3Config,
    ) -> None:
        super().__init__()
        taus = torch.as_tensor(config.taus_seconds, dtype=torch.float32).repeat_interleave(config.channels_per_tau)
        self.register_buffer("log_tau", taus.log())
        self.context_drive = nn.Linear(context_dim, taus.numel(), bias=False)
        # Feedback sources have their own fixed continuous-time impulse banks.
        # Keeping them separate from the common-drive state is essential: a
        # source absent from M0 must not rely on an M0 state axis or readout
        # that M0 had no reason to identify.  The fitted feedback edge is the
        # zero-bias readout from each source-specific bank.
        self.grammar_dim = int(grammar_input_dim)
        self.grammar_target_dim = int(grammar_target_dim)
        feedback_scales = int(taus.numel()) + 1  # immediate preceding block + persistent CT bank
        self.feedback_scales = feedback_scales
        count_feedback_dim = feedback_scales
        mark_state_dim = feedback_scales * int(grammar_input_dim)
        self.count_context = nn.Linear(context_dim, 1)
        self.count_state = nn.Linear(taus.numel(), 1, bias=False)
        self.mark_context = nn.Linear(context_dim, grammar_target_dim)
        self.mark_state = nn.Linear(taus.numel(), grammar_target_dim, bias=False)
        self.background_state = nn.Linear(taus.numel(), background_dim, bias=False)
        self.count_feedback_to_count = nn.Linear(count_feedback_dim, 1, bias=False)
        self.count_feedback_to_mark = nn.Linear(count_feedback_dim, grammar_target_dim, bias=False)
        self.count_feedback_to_background = nn.Linear(count_feedback_dim, background_dim, bias=False)
        self.mark_feedback_to_count = nn.Linear(mark_state_dim, 1, bias=False)
        self.mark_feedback_to_mark = nn.Linear(mark_state_dim, grammar_target_dim, bias=False)
        self.mark_feedback_to_background = nn.Linear(mark_state_dim, background_dim, bias=False)
        self.log_dispersion = nn.Parameter(torch.tensor(0.0))
        nn.init.normal_(self.context_drive.weight, std=0.02)
        nn.init.zeros_(self.count_context.weight); nn.init.zeros_(self.mark_context.weight)
        nn.init.normal_(self.count_state.weight, std=0.01)
        nn.init.normal_(self.mark_state.weight, std=0.01)
        nn.init.normal_(self.background_state.weight, std=0.01)
        for module in (
            self.count_feedback_to_count, self.count_feedback_to_mark,
            self.count_feedback_to_background, self.mark_feedback_to_count,
            self.mark_feedback_to_mark, self.mark_feedback_to_background,
        ):
            nn.init.zeros_(module.weight)

    @property
    def state_dim(self) -> int:
        return self.core_state_dim + self.feedback_scales * (1 + self.grammar_dim)

    @property
    def core_state_dim(self) -> int:
        return int(self.log_tau.numel())

    def _one_chain(
        self, time_axis: Tensor, context: Tensor,
    ) -> Tensor:
        n = time_axis.shape[0]
        if n == 0:
            return context.new_zeros((0, self.core_state_dim))
        delta = torch.cat((time_axis.new_zeros(1), time_axis[1:] - time_axis[:-1]))
        tau = self.log_tau.exp().to(dtype=torch.float64)
        phi = torch.exp(-delta[:, None].to(torch.float64) / tau[None]).to(context.dtype)
        phi[0] = 0.0
        drive = self.context_drive(context)
        q = torch.zeros_like(drive)
        if n > 1:
            q[1:] = (1.0 - phi[1:]) * drive[:-1]
        return affine_associative_scan(phi, q)[1]

    def _feedback_chains(
        self, time_axis: Tensor, count_input: Tensor, grammar_input: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Fixed event-impulse banks; only their zero-bias readouts are fitted."""

        n = time_axis.shape[0]
        if n == 0:
            return (
                count_input.new_zeros((0, self.feedback_scales)),
                grammar_input.new_zeros((0, self.feedback_scales * self.grammar_dim)),
            )
        delta = torch.cat((time_axis.new_zeros(1), time_axis[1:] - time_axis[:-1]))
        tau = self.log_tau.exp().to(dtype=torch.float64)
        phi = torch.exp(-delta[:, None].to(torch.float64) / tau[None]).to(count_input.dtype)
        phi[0] = 0.0
        count_impulse = count_input.expand(-1, self.core_state_dim)
        count_q = torch.zeros_like(count_impulse)
        if n > 1:
            count_q[1:] = phi[1:] * count_impulse[:-1]
        count_memory = affine_associative_scan(phi, count_q)[1]
        count_immediate = torch.zeros_like(count_input)
        if n > 1:
            count_immediate[1:] = count_input[:-1]
        count_state = torch.cat((count_immediate, count_memory), dim=1)

        mark_impulse = grammar_input[:, None, :].expand(-1, self.core_state_dim, -1).reshape(n, -1)
        mark_phi = phi[:, :, None].expand(-1, -1, self.grammar_dim).reshape(n, -1)
        mark_q = torch.zeros_like(mark_impulse)
        if n > 1:
            mark_q[1:] = mark_phi[1:] * mark_impulse[:-1]
        mark_memory = affine_associative_scan(mark_phi, mark_q)[1]
        mark_immediate = torch.zeros_like(grammar_input)
        if n > 1:
            mark_immediate[1:] = grammar_input[:-1]
        # Scale-major layout: immediate grammar vector, then one vector per tau.
        mark_state = torch.cat((mark_immediate, mark_memory), dim=1)
        return count_state, mark_state

    def forward(
        self, time_axis: Tensor, segment: Tensor, context: Tensor,
        count_input: Tensor, grammar_input: Tensor, family: str,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if family not in H3_FAMILIES:
            raise ValueError(f"unknown H3 family: {family}")
        row_parts, common_parts, count_parts, mark_parts = [], [], [], []
        for seg in torch.unique(segment):
            rows = torch.nonzero(segment == seg, as_tuple=False).flatten()
            row_parts.append(rows)
            common_parts.append(self._one_chain(time_axis[rows], context[rows]))
            count_state, mark_state = self._feedback_chains(
                time_axis[rows], count_input[rows], grammar_input[rows]
            )
            count_parts.append(count_state); mark_parts.append(mark_state)
        index = torch.cat(row_parts)
        order = torch.argsort(index)
        common_state = torch.cat(common_parts)[order]
        count_feedback = torch.cat(count_parts)[order]
        mark_feedback = torch.cat(mark_parts)[order]
        # Every emission at row t is a forecast made from information available
        # at the end of row t-1.  In particular, background measured at row t
        # may itself be downstream of the preceding event exposure and must not
        # be handed to the count/mark heads as a contemporaneous covariate.
        # The previous-row context remains an explicit, low-capacity common-
        # drive baseline and is reset at every carry-segment boundary.
        lagged_context = torch.zeros_like(context)
        for seg in torch.unique(segment):
            rows = torch.nonzero(segment == seg, as_tuple=False).flatten()
            if rows.numel() > 1:
                lagged_context[rows[1:]] = context[rows[:-1]]
        log_rate = self.count_context(lagged_context).squeeze(-1) + self.count_state(common_state).squeeze(-1)
        grammar = self.mark_context(lagged_context) + self.mark_state(common_state)
        # The current background frame is part of ``context`` for predicting
        # the next transition, but it cannot directly copy itself into this
        # readout.  The state at row t was formed from context/event exposure
        # at rows < t, so this is a genuine one-step background prediction.
        background = self.background_state(common_state)
        if family != "M0_common_drive":
            log_rate = log_rate + self.count_feedback_to_count(count_feedback).squeeze(-1)
            grammar = grammar + self.count_feedback_to_mark(count_feedback)
            background = background + self.count_feedback_to_background(count_feedback)
        if family == "M2_mark_feedback":
            log_rate = log_rate + self.mark_feedback_to_count(mark_feedback).squeeze(-1)
            grammar = grammar + self.mark_feedback_to_mark(mark_feedback)
            background = background + self.mark_feedback_to_background(mark_feedback)
        state = torch.cat((
            common_state,
            count_feedback if family != "M0_common_drive" else torch.zeros_like(count_feedback),
            mark_feedback if family == "M2_mark_feedback" else torch.zeros_like(mark_feedback),
        ), dim=1)
        return state, log_rate, grammar, background

    def impulse_response(
        self, count_input: Tensor, grammar_input: Tensor, family: str,
        horizons_seconds: Sequence[float],
    ) -> dict[str, dict[str, float]]:
        output = {}
        for horizon in horizons_seconds:
            decay = torch.exp(-torch.as_tensor(float(horizon), dtype=torch.float64, device=count_input.device)
                              / self.log_tau.exp().to(torch.float64)).to(count_input.dtype)
            count_memory = count_input.expand(-1, self.core_state_dim) * decay
            mark_memory = (
                grammar_input[:, None, :] * decay[None, :, None]
            ).reshape(grammar_input.shape[0], -1)
            is_first_block = float(horizon) <= 300.0
            count_effect = torch.cat((count_input if is_first_block else torch.zeros_like(count_input), count_memory), dim=1)
            mark_effect = torch.cat((grammar_input if is_first_block else torch.zeros_like(grammar_input), mark_memory), dim=1)
            count = count_input.new_zeros(count_input.shape[0])
            mark = grammar_input.new_zeros((grammar_input.shape[0], self.grammar_target_dim))
            background = grammar_input.new_zeros((grammar_input.shape[0], self.background_state.out_features))
            if family != "M0_common_drive":
                count = count + self.count_feedback_to_count(count_effect).squeeze(-1)
                mark = mark + self.count_feedback_to_mark(count_effect)
                background = background + self.count_feedback_to_background(count_effect)
            if family == "M2_mark_feedback":
                count = count + self.mark_feedback_to_count(mark_effect).squeeze(-1)
                mark = mark + self.mark_feedback_to_mark(mark_effect)
                background = background + self.mark_feedback_to_background(mark_effect)
            output[str(int(horizon))] = {
                "median_signed_count_log_rate": float(torch.median(count).detach().cpu()),
                "median_absolute_count_log_rate": float(torch.median(torch.abs(count)).detach().cpu()),
                "median_mark_l2": float(torch.median(torch.linalg.vector_norm(mark, dim=1)).detach().cpu()),
                "median_background_l2": float(torch.median(torch.linalg.vector_norm(background, dim=1)).detach().cpu()),
            }
        return output


def _negative_binomial_nll(count: Tensor, mean: Tensor, dispersion: Tensor) -> Tensor:
    r = torch.nn.functional.softplus(dispersion).clamp_min(1e-4)
    mu = mean.clamp_min(1e-6)
    return -(
        torch.lgamma(count + r) - torch.lgamma(r) - torch.lgamma(count + 1.0)
        + r * (torch.log(r) - torch.log(r + mu))
        + count * (torch.log(mu) - torch.log(r + mu))
    )


def _losses(
    model: PhysiologicalGenerativeState, family: str, tensors: dict[str, Tensor], rows: Tensor,
    *, count_input: Tensor | None = None, grammar_input: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    count_used = tensors["count_input"] if count_input is None else count_input
    grammar_used = tensors["grammar_input"] if grammar_input is None else grammar_input
    _state, log_rate, grammar, background = model(
        tensors["time"], tensors["segment"], tensors["context"],
        count_used, grammar_used, family,
    )
    exposure_fraction = tensors["exposure"] / 300.0
    count_valid = exposure_fraction >= 0.8
    mu = torch.exp(torch.clamp(log_rate, -10.0, 12.0)) * exposure_fraction
    use_count = rows[count_valid[rows]]
    count_loss = _negative_binomial_nll(
        tensors["count"][use_count], mu[use_count], model.log_dispersion
    ).mean()
    use_mark = rows[tensors["grammar_valid"][rows]]
    mark_loss = (
        torch.mean((grammar[use_mark] - tensors["grammar_target"][use_mark]) ** 2)
        if use_mark.numel() else count_loss.new_tensor(float("nan"))
    )
    use_background = rows[tensors["background_valid"][rows]]
    background_loss = (
        torch.mean((background[use_background] - tensors["background_target"][use_background]) ** 2)
        if use_background.numel() else count_loss.new_tensor(float("nan"))
    )
    terms = [count_loss] + [value for value in (mark_loss, background_loss) if torch.isfinite(value)]
    total = torch.stack(terms).mean()
    return total, count_loss, mark_loss, background_loss


def _tensorise(data: H3Data, device: torch.device) -> dict[str, Tensor]:
    return {
        "time": torch.as_tensor(data.time, dtype=torch.float64, device=device),
        "segment": torch.as_tensor(data.segment, dtype=torch.long, device=device),
        "exposure": torch.as_tensor(data.exposure_seconds, dtype=torch.float32, device=device),
        "count": torch.as_tensor(data.count, dtype=torch.float32, device=device),
        "count_input": torch.as_tensor(data.count_input, dtype=torch.float32, device=device),
        "grammar_input": torch.as_tensor(data.grammar_input, dtype=torch.float32, device=device),
        "grammar_target": torch.as_tensor(data.grammar_target, dtype=torch.float32, device=device),
        "grammar_valid": torch.as_tensor(data.grammar_valid, dtype=torch.bool, device=device),
        "background_target": torch.as_tensor(data.background_target, dtype=torch.float32, device=device),
        "background_valid": torch.as_tensor(data.background_valid, dtype=torch.bool, device=device),
        "context": torch.as_tensor(data.context, dtype=torch.float32, device=device),
    }


def _score(
    model: PhysiologicalGenerativeState, family: str, tensors: dict[str, Tensor], rows: Tensor,
    *, count_input: Tensor | None = None, grammar_input: Tensor | None = None,
) -> dict[str, Any]:
    if rows.numel() == 0:
        return {
            "status": "NOT_ESTIMABLE", "total": None, "count": None,
            "conditional_grammar": None, "future_background": None, "n_grid_blocks": 0,
        }
    with torch.no_grad():
        total, count, mark, background = _losses(
            model, family, tensors, rows, count_input=count_input, grammar_input=grammar_input,
        )
    return {
        "status": "ESTIMATED", "total": float(total), "count": float(count),
        "conditional_grammar": float(mark) if torch.isfinite(mark) else None,
        "future_background": float(background) if torch.isfinite(background) else None,
        "n_grid_blocks": int(rows.numel()),
    }


def _causal_delayed_inputs(data: H3Data, delay_seconds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use only donors at least ``delay_seconds`` in the past within a carry segment."""

    count = np.zeros_like(data.count_input)
    grammar = np.zeros_like(data.grammar_input)
    valid = np.zeros(data.time.size, dtype=bool)
    for seg in np.unique(data.segment):
        rows = np.flatnonzero(data.segment == seg)
        donor_position = np.searchsorted(data.time[rows], data.time[rows] - float(delay_seconds), side="right") - 1
        ok = donor_position >= 0
        donor = rows[np.maximum(donor_position, 0)]
        count[rows[ok]] = data.count_input[donor[ok]]
        grammar[rows[ok]] = data.grammar_input[donor[ok]]
        valid[rows[ok]] = True
    return count, grammar, valid



def _penalty_grid_diagnostics(
    candidates: list[dict[str, Any]], grids: dict[str, tuple[float, ...]]
) -> dict[str, Any]:
    """Report whether each ridge search ended on its own boundary.

    A search whose best value sits at the largest penalty tried cannot tell
    "the data prefer the zero edge" from "the grid never reached the optimum",
    and the first-round grid was pegged at its upper edge in every pilot
    patient without anything recording it.
    """
    out: dict[str, Any] = {}
    for key, grid in grids.items():
        rows = [row for row in candidates if key in row]
        if not rows:
            continue
        metric = next(
            (name for name in sorted(rows[0]) if name.startswith("inner_")), None
        )
        if metric is None:
            continue
        best = min(rows, key=lambda row: float(row[metric]))
        out[key] = {
            "grid": list(grid),
            "best_penalty": float(best[key]),
            "selection_metric": metric,
            "saturated_at_upper_edge": bool(float(best[key]) >= float(max(grid))),
        }
    return out


def train_h3_subject(
    data: H3Data, config: H3Config, *, device: torch.device, out_dir: Path,
) -> dict[str, Any]:
    """Fit M0/M1/M2 on FIT, choose checkpoints on INNER, report SELECTION."""

    started = time.time(); torch.manual_seed(config.seed); np.random.seed(config.seed)
    tensors = _tensorise(data, device)
    split = {name: torch.as_tensor(np.flatnonzero(data.phase == name), dtype=torch.long, device=device)
             for name in ("FIT", "INNER", "SELECTION")}
    if any(split[name].numel() == 0 for name in split):
        raise ValueError(f"{data.subject}: H3 requires FIT, INNER and SELECTION grid blocks")
    template = PhysiologicalGenerativeState(
        data.context.shape[1], data.grammar_input.shape[1], data.grammar_target.shape[1],
        data.background_target.shape[1], config
    ).to(device)
    with torch.no_grad():
        fit_mask = (data.phase == "FIT") & (data.exposure_seconds >= 0.8 * config.grid_seconds)
        fit_mean = float(np.mean(data.count[fit_mask] * 300.0 / np.maximum(data.exposure_seconds[fit_mask], 1.0)))
        template.count_context.bias.fill_(np.log(max(fit_mean, 1e-3)))
    models: dict[str, PhysiologicalGenerativeState] = {}; training = {}; scores = {}
    for family in H3_FAMILIES:
        model = PhysiologicalGenerativeState(
            data.context.shape[1], data.grammar_input.shape[1], data.grammar_target.shape[1],
            data.background_target.shape[1], config
        ).to(device)
        if family != "M0_common_drive":
            parent = "M0_common_drive" if family == "M1_count_feedback" else "M1_count_feedback"
            model.load_state_dict(models[parent].state_dict())
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            with torch.no_grad():
                feature_state = model(
                    tensors["time"], tensors["segment"], tensors["context"],
                    tensors["count_input"], tensors["grammar_input"], "M2_mark_feedback",
                )[0]
            candidates: list[dict[str, Any]] = []
            count_penalty_grid: tuple[float, ...] | None = None
            background_penalty_grid: tuple[float, ...] | None = None
            mark_penalty_grid: tuple[float, ...] | None = None
            best_state = copy.deepcopy(model.state_dict())
            if family == "M1_count_feedback":
                # The primary count-edge estimand is the immediately preceding
                # block.  Persistent tails are reported only after this edge is
                # identifiable, avoiding a many-timescale fit masquerading as
                # evidence for a one-step jump.
                x = feature_state[:, model.core_state_dim:model.core_state_dim + 1]
                fit_valid = (tensors["exposure"] / config.grid_seconds) >= config.minimum_exposure_fraction
                fit_rows = split["FIT"][fit_valid[split["FIT"]]]
                scale = torch.sqrt(torch.mean(x[fit_rows] ** 2)).clamp_min(1e-5)
                x_scaled = x / scale
                with torch.no_grad():
                    base_log_rate = model(
                        tensors["time"], tensors["segment"], tensors["context"],
                        tensors["count_input"], tensors["grammar_input"], "M0_common_drive",
                    )[1]
                best_value = float(_score(model, family, tensors, split["INNER"])["count"])
                # A ridge grid whose best value sits on its own upper edge
                # cannot separate "the data prefer the zero edge" from "the grid
                # never reached the optimum".  Every pilot patient pegged the
                # old 1e-5..10 grid at 10, so the grid is extended and the
                # saturation flag is recorded next to the verdict.
                count_penalty_grid = (0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4)
                for penalty in count_penalty_grid:
                    coefficient = torch.zeros(1, device=device, requires_grad=True)
                    optimiser = torch.optim.LBFGS(
                        [coefficient], lr=0.5, max_iter=80, tolerance_grad=1e-9,
                        tolerance_change=1e-11, line_search_fn="strong_wolfe",
                    )
                    def closure() -> Tensor:
                        optimiser.zero_grad(set_to_none=True)
                        log_rate = base_log_rate[fit_rows] + x_scaled[fit_rows, 0] * coefficient[0]
                        exposure_fraction = tensors["exposure"][fit_rows] / config.grid_seconds
                        mean = torch.exp(torch.clamp(log_rate, -10.0, 12.0)) * exposure_fraction
                        objective = _negative_binomial_nll(
                            tensors["count"][fit_rows], mean, model.log_dispersion
                        ).mean() + float(penalty) * coefficient.square().mean()
                        objective.backward()
                        return objective
                    optimiser.step(closure)
                    with torch.no_grad():
                        model.count_feedback_to_count.weight.zero_()
                        model.count_feedback_to_count.weight[0, 0] = coefficient[0] / scale
                    inner = _score(model, family, tensors, split["INNER"])
                    candidates.append({"penalty": penalty, "inner_count": inner["count"],
                                       "standardised_coefficient": float(coefficient.detach())})
                    if float(inner["count"]) < best_value - 1e-5:
                        best_value = float(inner["count"]); best_state = copy.deepcopy(model.state_dict())
                # Independently estimate whether burden innovation changes the
                # next non-event background frame.  This is the closer readout
                # of a candidate physiological transition; future count is a
                # separate supportive emission, not a requirement that one
                # coefficient improve two different outcomes.
                model.load_state_dict(best_state)
                with torch.no_grad():
                    base_background = model(
                        tensors["time"], tensors["segment"], tensors["context"],
                        tensors["count_input"], tensors["grammar_input"], "M0_common_drive",
                    )[3]
                background_rows = split["FIT"][tensors["background_valid"][split["FIT"]]]
                y = tensors["background_target"] - base_background
                xb = x_scaled[background_rows]
                gram = xb.T @ xb
                rhs = xb.T @ y[background_rows]
                best_background = float(_score(model, family, tensors, split["INNER"])["future_background"])
                best_background_state = copy.deepcopy(model.state_dict())
                # A ridge grid whose best value sits on its own upper edge
                # cannot separate "the data prefer the zero edge" from "the grid
                # never reached the optimum".  Every pilot patient pegged the
                # old 1e-5..10 grid at 10, so the grid is extended and the
                # saturation flag is recorded next to the verdict.
                background_penalty_grid = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4)
                for penalty in background_penalty_grid:
                    weight_scaled = rhs / (gram + float(penalty * background_rows.numel()))
                    with torch.no_grad():
                        model.count_feedback_to_background.weight.zero_()
                        model.count_feedback_to_background.weight[:, 0] = (
                            weight_scaled[0] / scale
                        )
                    inner = _score(model, family, tensors, split["INNER"])
                    candidates.append({"background_penalty": penalty,
                                       "inner_future_background": inner["future_background"]})
                    if float(inner["future_background"]) < best_background - 1e-5:
                        best_background = float(inner["future_background"])
                        best_background_state = copy.deepcopy(model.state_dict())
                best_state = best_background_state
                selection_metric = "future_count_and_non_event_background_independently"
                fitted_parameters = [
                    "count_feedback_to_count.weight[immediate_preceding_block]",
                    "count_feedback_to_background.weight[immediate_preceding_block]",
                ]
            else:
                start = model.core_state_dim + model.feedback_scales
                x = feature_state[:, start:start + model.grammar_dim]
                fit_rows = split["FIT"][tensors["background_valid"][split["FIT"]]]
                scale = torch.sqrt(torch.mean(x[fit_rows] ** 2, dim=0)).clamp_min(1e-5)
                x_scaled = x / scale
                with torch.no_grad():
                    base_background = model(
                        tensors["time"], tensors["segment"], tensors["context"],
                        tensors["count_input"], tensors["grammar_input"], "M1_count_feedback",
                    )[3]
                y = tensors["background_target"] - base_background
                gram = x_scaled[fit_rows].T @ x_scaled[fit_rows]
                rhs = x_scaled[fit_rows].T @ y[fit_rows]
                eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
                best_value = float(_score(model, family, tensors, split["INNER"])["future_background"])
                # A ridge grid whose best value sits on its own upper edge
                # cannot separate "the data prefer the zero edge" from "the grid
                # never reached the optimum".  Every pilot patient pegged the
                # old 1e-5..10 grid at 10, so the grid is extended and the
                # saturation flag is recorded next to the verdict.
                mark_penalty_grid = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4)
                for penalty in mark_penalty_grid:
                    weight_scaled = torch.linalg.solve(
                        gram + float(penalty * fit_rows.numel()) * eye, rhs
                    )
                    with torch.no_grad():
                        model.mark_feedback_to_background.weight.zero_()
                        model.mark_feedback_to_background.weight[:, :model.grammar_dim] = (
                            weight_scaled / scale[:, None]
                        ).T
                    inner = _score(model, family, tensors, split["INNER"])
                    candidates.append({"penalty": penalty,
                                       "inner_future_background": inner["future_background"]})
                    if float(inner["future_background"]) < best_value - 1e-5:
                        best_value = float(inner["future_background"]); best_state = copy.deepcopy(model.state_dict())
                selection_metric = "future_non_event_background"
                fitted_parameters = ["mark_feedback_to_background.weight[immediate_preceding_block]"]
            model.load_state_dict(best_state); models[family] = model
            scores[family] = _score(model, family, tensors, split["SELECTION"])
            training[family] = {
                "trained_parameters": fitted_parameters,
                "selected_step": None, "steps_run": None,
                "selected_at_budget_edge": False, "selected_at_first_check": False,
                "selection_metric": selection_metric,
                "regularisation_candidates": candidates,
                "penalty_grid_diagnostics": _penalty_grid_diagnostics(candidates, {
                    key: grid for key, grid in (
                        ("penalty", count_penalty_grid or mark_penalty_grid),
                        ("background_penalty", background_penalty_grid),
                    ) if grid is not None
                }),
                "zero_edge_was_explicit_candidate": True,
            }
            continue

        model.load_state_dict(template.state_dict())
        trainable = [
            parameter for name, parameter in model.named_parameters()
            if not name.startswith("count_feedback") and not name.startswith("mark_feedback")
        ]
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in trainable:
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
        initial_inner = _score(model, family, tensors, split["INNER"])
        best = float(initial_inner["total"]); best_step = stale = 0
        best_state = copy.deepcopy(model.state_dict())
        history = [{"step": 0, "inner": initial_inner}]
        for step in range(1, config.max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            total, count_loss, mark_loss, background_loss = _losses(model, family, tensors, split["FIT"])
            objective = (
                total if family == "M0_common_drive" else
                count_loss if family == "M1_count_feedback" else
                background_loss
            )
            if not torch.isfinite(objective):
                raise FloatingPointError(f"{family}: non-finite H3 FIT loss")
            objective.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip); optimizer.step()
            if step % config.validate_every == 0 or step == config.max_steps:
                inner = _score(model, family, tensors, split["INNER"])
                history.append({"step": step, "fit_total": float(total.detach()),
                                "fit_count": float(count_loss.detach()),
                                "fit_grammar": float(mark_loss.detach()) if torch.isfinite(mark_loss) else None,
                                "fit_background": float(background_loss.detach()) if torch.isfinite(background_loss) else None,
                                "inner": inner})
                selection_metric = (
                    "total" if family == "M0_common_drive" else
                    "count" if family == "M1_count_feedback" else
                    "future_background"
                )
                value = float(inner[selection_metric])
                if value < best - 1e-5:
                    best, best_step, stale = value, step, 0; best_state = copy.deepcopy(model.state_dict())
                else:
                    stale += 1
                if stale >= config.patience_checks:
                    break
        model.load_state_dict(best_state); models[family] = model
        scores[family] = _score(model, family, tensors, split["SELECTION"])
        training[family] = {
            "trained_parameters": [name for name, parameter in model.named_parameters() if parameter.requires_grad],
            "selected_step": best_step, "steps_run": history[-1]["step"],
            "selected_at_budget_edge": best_step == config.max_steps,
            "selected_at_first_check": best_step == config.validate_every,
            "selection_metric": (
                "joint_common_drive" if family == "M0_common_drive" else
                "future_count" if family == "M1_count_feedback" else
                "future_non_event_background"
            ),
            "history": history,
        }

    selection = split["SELECTION"]
    impulses = {
        family: models[family].impulse_response(
            tensors["count_input"][selection], tensors["grammar_input"][selection], family,
            (300.0, 1800.0, 7200.0, 21600.0, 86400.0),
        ) for family in H3_FAMILIES
    }
    delayed_count_np, delayed_grammar_np, delayed_valid_np = _causal_delayed_inputs(data, 21600.0)
    delayed_count = torch.as_tensor(delayed_count_np, dtype=torch.float32, device=device)
    delayed_grammar = torch.as_tensor(delayed_grammar_np, dtype=torch.float32, device=device)
    delayed_valid = torch.as_tensor(delayed_valid_np, dtype=torch.bool, device=device)
    paired_selection = selection[delayed_valid[selection]]
    fit_np = np.flatnonzero(data.phase == "FIT")
    constant_count = torch.as_tensor(
        np.broadcast_to(np.mean(data.count_input[fit_np], axis=0, keepdims=True), data.count_input.shape).copy(),
        dtype=torch.float32, device=device,
    )
    valid_fit_grammar = fit_np[data.grammar_valid[fit_np]]
    constant_grammar = torch.as_tensor(
        np.broadcast_to(np.mean(data.grammar_input[valid_fit_grammar], axis=0, keepdims=True), data.grammar_input.shape).copy(),
        dtype=torch.float32, device=device,
    )
    controls = {
        "delay_seconds": 21600.0,
        "paired_selection_blocks": int(paired_selection.numel()),
        "M1_correct_on_delayed_support": _score(models["M1_count_feedback"], "M1_count_feedback", tensors, paired_selection),
        "M1_causal_delayed_count": _score(
            models["M1_count_feedback"], "M1_count_feedback", tensors, paired_selection,
            count_input=delayed_count,
        ),
        "M1_fit_mean_count": _score(
            models["M1_count_feedback"], "M1_count_feedback", tensors, selection,
            count_input=constant_count,
        ),
        "M2_correct_on_delayed_support": _score(models["M2_mark_feedback"], "M2_mark_feedback", tensors, paired_selection),
        "M2_causal_delayed_mark": _score(
            models["M2_mark_feedback"], "M2_mark_feedback", tensors, paired_selection,
            grammar_input=delayed_grammar,
        ),
        "M2_fit_mean_mark": _score(
            models["M2_mark_feedback"], "M2_mark_feedback", tensors, selection,
            grammar_input=constant_grammar,
        ),
    }
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "checkpoint.pt"
    torch.save({"config": asdict(config), "models": {name: model.state_dict() for name, model in models.items()}}, checkpoint)
    trajectories = {}
    with torch.no_grad():
        for family, model in models.items():
            state, log_rate, grammar, background = model(
                tensors["time"], tensors["segment"], tensors["context"],
                tensors["count_input"], tensors["grammar_input"], family,
            )
            trajectories[f"{family}__state"] = state.cpu().numpy()
            trajectories[f"{family}__log_rate"] = log_rate.cpu().numpy()
            trajectories[f"{family}__grammar"] = grammar.cpu().numpy()
            trajectories[f"{family}__background"] = background.cpu().numpy()
    trajectory_path = out_dir / "trajectory.npz"
    np.savez_compressed(
        trajectory_path, time=data.time, segment=data.segment, phase=data.phase,
        exposure_seconds=data.exposure_seconds, count=data.count, grammar_valid=data.grammar_valid,
        **trajectories,
    )
    card = {
        "format": "group_event_state_v0_3_7_h3_independent_generative_card_v3",
        "subject": data.subject, "seed": int(config.seed), "config": asdict(config),
        "state_semantics": "Z_phys_generative_candidate_not_S_obs",
        "models": {
            "M0_common_drive": "continuous generative prior driven by causal background and rolling covariates; no event jump",
            "M1_count_feedback": "frozen M0 plus a newly fitted preceding-block count/burden physical-jump candidate",
            "M2_mark_feedback": "frozen M1 plus a newly fitted conditional-grammar physical-jump candidate",
        },
        # M1/M2 freeze exactly the M0 common-drive core and its fitted
        # intercept, but necessarily add source-specific, zero-bias edge
        # readouts.  Calling the complete models "same capacity" would be
        # false; the nesting and the extra parameters are recorded explicitly.
        "common_core_and_intercept_frozen_across_nested_models": True,
        "feedback_edges_add_zero_bias_source_specific_readouts": True,
        "complete_models_have_equal_parameter_count": False,
        "edge_estimand_scope": {
            "exposure_block_seconds": int(config.grid_seconds),
            "primary_prediction_lag_blocks": 1,
            "primary_count_edge": "preceding-block burden innovation to next-block count and non-event background",
            "primary_mark_edge": "preceding-block rank-reduced grammar innovation to next-block non-event background beyond M1",
            "persistent_ct_bank_present": True,
            "persistent_ct_readout_fitted_in_primary_v0_3_7": False,
            "long_horizon_feedback_status": "NOT_ESTIMATED_IN_V0_3_7_PRIMARY",
        },
        "event_inputs_residualised_against_pre_block_common_drive": True,
        "nonoverlapping_exposure_blocks": True,
        "scores": scores, "training": training, "impulse_response": impulses,
        "causal_delay_and_constant_controls": controls,
        "primary_contrasts": {
            "count_feedback_gain_M1_over_M0": scores["M0_common_drive"]["count"] - scores["M1_count_feedback"]["count"],
            "mark_feedback_gain_M2_over_M1": scores["M1_count_feedback"]["conditional_grammar"] - scores["M2_mark_feedback"]["conditional_grammar"],
            "count_feedback_gain_on_future_background": scores["M0_common_drive"]["future_background"] - scores["M1_count_feedback"]["future_background"],
            "mark_feedback_gain_on_future_background": scores["M1_count_feedback"]["future_background"] - scores["M2_mark_feedback"]["future_background"],
            "joint_supportive_gain_M2_over_M0": scores["M0_common_drive"]["total"] - scores["M2_mark_feedback"]["total"],
        },
        "support": {
            name: {"grid_blocks": int(np.sum(data.phase == name)),
                   "hours": float(np.sum(data.exposure_seconds[data.phase == name]) / 3600.0)}
            for name in ("FIT", "INNER", "SELECTION")
        },
        "transforms": data.transforms, "checkpoint_path": str(checkpoint),
        "trajectory_path": str(trajectory_path), "elapsed_seconds": time.time() - started,
        "allowed_claim": "feedback-like directional dependence only; observational data do not establish intervention-level causality",
        "code_provenance": _code_provenance(Path(__file__)),
        "observer_checkpoint_used_as_jump": False,
        "development_targets_read": False, "seizure_targets_read": False,
        "sealed_partition_opened": False,
    }
    atomic_json(out_dir / "card.json", card)
    return card
