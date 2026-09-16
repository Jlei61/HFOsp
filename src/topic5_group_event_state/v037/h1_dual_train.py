"""Joint background plus marked-event observer for v0.3.7 H1.

This is the scientific dual-stream model.  A fixed-clock background observer
is fitted before the marked-event observer, so the final event-history gain is
measured beyond an explicit common-drive estimate.  All horizons share both
state producers and receive equal optimisation weight.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from src.topic5_group_event_state.v035.background_rate import causal_background_at_grid

from .contracts import atomic_json, sha256_file
from .ctssm import DualStreamEventCTSSM, GridBackgroundCTSSM
from .h1_data import H1SubjectData
from .h1_train import (
    EventStateComputer,
    H1TrainConfig,
    _block_shift,
    _code_provenance,
    _endpoint_losses,
    _fixed_features,
    _selection_score,
    _standardise_features,
    _target_bundle,
    _weighted_total,
    independent_window_count,
)


@dataclass(frozen=True)
class H1DualTrainConfig(H1TrainConfig):
    # The event-only pilot put E253 at the 900-step boundary in 4/5 seeds.
    # 2700 is a preregistered adequacy extension, not a result-selected recipe;
    # patients that converge earlier still stop by the same INNER rule.
    max_steps_q: int = 2700
    background_channels_per_tau: int = 2
    lr_background_current: float = 1e-3
    lr_background_state: float = 3e-4
    lr_background_head: float = 1e-3
    max_steps_background_current: int = 900
    max_steps_background_state: int = 1800
    warmup_steps_background: int = 0


class NestedDualReadout(nn.Module):
    ENDPOINTS = ("count", "burden", "community", "coupling", "mixture", "embedding", "mark")

    def __init__(
        self,
        q_dim: int,
        bmark_burden_dim: int,
        bmark_grammar_dim: int,
        background_current_dim: int,
        background_state_dim: int,
        event_burden_dim: int,
        event_grammar_dim: int,
        widths: Mapping[str, int],
        n_horizon: int,
        state_readout_init_std: float = 1e-2,
        bmark_readout_init_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.widths = dict(widths); self.n_horizon = int(n_horizon)
        self.bmark_burden_dim = int(bmark_burden_dim)
        self.event_burden_dim = int(event_burden_dim)
        self.q = nn.ModuleDict(); self.bmark = nn.ModuleDict()
        self.background_current = nn.ModuleDict(); self.background_state = nn.ModuleDict()
        self.event = nn.ModuleDict(); self.random = nn.ModuleDict()
        for name in self.ENDPOINTS:
            out = self.n_horizon * int(self.widths[name])
            burden = name in {"count", "burden"}
            bdim = bmark_burden_dim if burden else bmark_grammar_dim
            edim = event_burden_dim if burden else event_grammar_dim
            self.q[name] = nn.Linear(q_dim, out)
            self.bmark[name] = nn.Linear(bdim, out, bias=False)
            self.background_current[name] = nn.Linear(background_current_dim, out, bias=False)
            self.background_state[name] = nn.Linear(background_state_dim, out, bias=False)
            self.event[name] = nn.Linear(edim, out, bias=False)
            self.random[name] = nn.Linear(edim, out, bias=False)
            nn.init.zeros_(self.q[name].weight)
            nn.init.zeros_(self.q[name].bias)
            if float(bmark_readout_init_std) > 0.0:
                nn.init.normal_(self.bmark[name].weight, std=float(bmark_readout_init_std))
            else:
                nn.init.zeros_(self.bmark[name].weight)
            nn.init.zeros_(self.background_current[name].weight)
            # Non-zero output maps give both observers gradient at step one.
            nn.init.normal_(self.background_state[name].weight, std=float(state_readout_init_std))
            nn.init.normal_(self.event[name].weight, std=float(state_readout_init_std))
            nn.init.zeros_(self.random[name].weight)
        self.log_dispersion = nn.Parameter(torch.zeros(n_horizon))

    @staticmethod
    def _normalise(value: Tensor) -> Tensor:
        return torch.nn.functional.layer_norm(value, (value.shape[-1],))

    def predict(
        self,
        q: Tensor,
        *,
        bmark: Tensor | None = None,
        background_current: Tensor | None = None,
        background_state: Tensor | None = None,
        event_state: Tensor | None = None,
        random_event_state: Tensor | None = None,
    ) -> dict[str, Tensor]:
        output = {}
        for name in self.ENDPOINTS:
            burden = name in {"count", "burden"}
            value = self.q[name](q)
            if bmark is not None:
                part = bmark[:, :self.bmark_burden_dim] if burden else bmark[:, self.bmark_burden_dim:]
                value = value + self.bmark[name](part)
            if background_current is not None:
                value = value + self.background_current[name](self._normalise(background_current))
            if background_state is not None:
                value = value + self.background_state[name](self._normalise(background_state))
            if event_state is not None:
                part = event_state[:, :self.event_burden_dim] if burden else event_state[:, self.event_burden_dim:]
                value = value + self.event[name](self._normalise(part))
            if random_event_state is not None:
                part = random_event_state[:, :self.event_burden_dim] if burden else random_event_state[:, self.event_burden_dim:]
                value = value + self.random[name](self._normalise(part))
            output[name] = value.reshape(q.shape[0], self.n_horizon, self.widths[name])
        return output

    def parameters_for(self, stage: str) -> list[nn.Parameter]:
        if stage == "q": return [*self.q.parameters(), self.log_dispersion]
        if stage == "bmark": return list(self.bmark.parameters())
        if stage == "background_current": return list(self.background_current.parameters())
        if stage == "background_state": return list(self.background_state.parameters())
        if stage == "event": return list(self.event.parameters())
        if stage == "random": return list(self.random.parameters())
        raise KeyError(stage)


class BackgroundStateComputer:
    def __init__(self, data: H1SubjectData, model: GridBackgroundCTSSM,
                 values: Tensor, available: Tensor, device: torch.device) -> None:
        self.model = model; self.device = device; self.chains = []
        for segment in np.unique(data.rate.segment):
            rows = np.flatnonzero(data.rate.segment == segment)
            if rows.size == 0: continue
            rt = torch.as_tensor(rows, dtype=torch.long, device=device)
            times = torch.as_tensor(data.rate.anchor_time[rows], dtype=torch.float64, device=device)
            if times.numel() > 1:
                spacing = float(torch.median(times[1:] - times[:-1]))
            else:
                spacing = 300.0
            self.chains.append((rt, times, values[rt], available[rt], float(times[0]) - max(spacing, 1.0)))

    def __call__(self) -> Tensor:
        rows, values = [], []
        for row, times, feature, available, initial_time in self.chains:
            output = self.model(times, feature, available, initial_time=initial_time)
            rows.append(row); values.append(output.features)
        index = torch.cat(rows); value = torch.cat(values)
        return value[torch.argsort(index)]


def _background_features(data: H1SubjectData, device: torch.device
                         ) -> tuple[Tensor, Tensor, dict[str, Any], tuple[str, ...], np.ndarray, np.ndarray]:
    raw, names, audit = causal_background_at_grid(data.rate)
    fit = np.flatnonzero(data.rate.phase == "FIT")
    physiological_width = raw.shape[1] - 2
    available = raw[:, -1] > 0.5
    if not np.any(available[fit]):
        raise ValueError(f"{data.subject}: no causal background observations in FIT")
    centre = np.zeros(raw.shape[1], dtype=np.float32); scale = np.ones(raw.shape[1], dtype=np.float32)
    observed_fit = fit[available[fit]]
    centre[:physiological_width] = np.nanmedian(raw[observed_fit, :physiological_width], axis=0)
    scale[:physiological_width] = 1.4826 * np.nanmedian(
        np.abs(raw[observed_fit, :physiological_width] - centre[:physiological_width]), axis=0
    )
    centre[-2:] = np.nanmedian(raw[fit, -2:], axis=0)
    scale[-2:] = 1.4826 * np.nanmedian(np.abs(raw[fit, -2:] - centre[-2:]), axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    value = np.clip((raw - centre) / scale, -12.0, 12.0).astype(np.float32)
    value[~np.isfinite(value)] = 0.0
    return (
        torch.as_tensor(value, dtype=torch.float32, device=device),
        torch.as_tensor(available.astype(np.float32), device=device),
        audit, names, centre, scale,
    )


def _train_stage(
    stage: str, readout: NestedDualReadout, q: Tensor, bmark: Tensor,
    background_current: Tensor, background_computer: BackgroundStateComputer,
    event_computer: EventStateComputer, random_event: Tensor,
    target: Mapping[str, Tensor], valid: Mapping[str, Tensor], exposure: Tensor,
    fit: Tensor, inner: Tensor, config: H1DualTrainConfig,
) -> dict[str, Any]:
    for parameter in readout.parameters(): parameter.requires_grad_(False)
    for parameter in background_computer.model.parameters(): parameter.requires_grad_(False)
    for parameter in event_computer.model.parameters(): parameter.requires_grad_(False)
    params = readout.parameters_for(stage)
    for parameter in params: parameter.requires_grad_(True)
    groups: list[dict[str, Any]]
    if stage == "background_state":
        bg_params = list(background_computer.model.parameters())
        for parameter in bg_params: parameter.requires_grad_(True)
        groups = [
            {"params": readout.parameters_for(stage), "lr": config.lr_background_head},
            {"params": bg_params, "lr": config.lr_background_state},
        ]
        params = params + bg_params; maximum = config.max_steps_background_state
    elif stage == "event":
        event_params = list(event_computer.model.parameters())
        for parameter in event_params: parameter.requires_grad_(True)
        groups = [
            {"params": readout.parameters_for(stage), "lr": config.lr_state_head},
            {"params": event_params, "lr": config.lr_state},
        ]
        params = params + event_params; maximum = config.max_steps_state
    else:
        lr = {
            "q": config.lr_q, "bmark": config.lr_bmark,
            "background_current": config.lr_background_current,
            "random": config.lr_random_head,
        }[stage]
        maximum = {
            "q": config.max_steps_q, "bmark": config.max_steps_bmark,
            "background_current": config.max_steps_background_current,
            "random": config.max_steps_random,
        }[stage]
        groups = [{"params": params, "lr": lr}]
    decay = (
        float(config.weight_decay_bmark)
        if stage == "bmark" and config.weight_decay_bmark is not None
        else float(config.weight_decay)
    )
    optimizer = torch.optim.AdamW(
        groups, weight_decay=decay,
        betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_epsilon,
    )

    def predict() -> dict[str, Tensor]:
        kw: dict[str, Tensor] = {}
        if stage != "q": kw["bmark"] = bmark
        if stage in {"background_current", "background_state", "random", "event"}:
            kw["background_current"] = background_current
        if stage in {"background_state", "random", "event"}:
            kw["background_state"] = background_computer()
        if stage == "random": kw["random_event_state"] = random_event
        if stage == "event": kw["event_state"] = event_computer()
        return readout.predict(q, **kw)

    def parent_predict() -> dict[str, Tensor]:
        """Prediction of the already-fitted arm immediately below `stage`."""
        kw: dict[str, Tensor] = {}
        if stage != "q": kw["bmark"] = bmark
        if stage in {"background_state", "random", "event"}:
            kw["background_current"] = background_current
        if stage in {"random", "event"}:
            kw["background_state"] = background_computer()
        return readout.predict(q, **kw)

    with torch.no_grad():
        optimiser_initial = float(_weighted_total(_endpoint_losses(
            predict(), target, valid, exposure, readout.log_dispersion, inner
        )))
        parent = (
            optimiser_initial if stage == "q" else
            float(_weighted_total(_endpoint_losses(
                parent_predict(), target, valid, exposure,
                readout.log_dispersion, inner,
            )))
        )
    best = parent; best_step = stale = 0
    best_readout = copy.deepcopy(readout.state_dict())
    if stage != "q":
        prefix = f"{stage}."
        for key in tuple(best_readout):
            if key.startswith(prefix):
                best_readout[key].zero_()
    best_bg = copy.deepcopy(background_computer.model.state_dict())
    best_event = copy.deepcopy(event_computer.model.state_dict())
    parameter_start = [value.detach().cpu().clone() for value in params]
    active_ids = {id(value) for value in params}
    named = [(f'{prefix}.{name}', value)
             for prefix, model in [('readout', readout), ('event', event_computer.model),
                                    ('background', background_computer.model)]
             for name, value in model.named_parameters() if id(value) in active_ids]
    initial_by_name = {name: value.detach().cpu().clone() for name, value in named}
    gradient_audit = {name: {'shape': list(value.shape), 'parameters': value.numel(),
                            'first_nonzero_step': None, 'nonzero_steps': 0,
                            'gradient_norm_sum': 0.0, 'gradient_norm_max': 0.0}
                      for name, value in named}
    first_step_gradient_norm = None
    peak_parameter_delta = 0.0
    history = [{"step": 0, "inner_loss": parent,
                "parent_inner_loss": parent,
                "optimiser_initial_inner_loss": optimiser_initial}]
    for step in range(1, int(maximum) + 1):
        if stage == "event" and config.warmup_steps_state > 0:
            fraction = min(1.0, step / float(config.warmup_steps_state))
            optimizer.param_groups[0]["lr"] = config.lr_state_head * fraction
            optimizer.param_groups[1]["lr"] = config.lr_state * fraction
        elif stage == "background_state" and config.warmup_steps_background > 0:
            fraction = min(1.0, step / float(config.warmup_steps_background))
            optimizer.param_groups[0]["lr"] = config.lr_background_head * fraction
            optimizer.param_groups[1]["lr"] = config.lr_background_state * fraction
        elif stage == "bmark" and config.warmup_steps_bmark > 0:
            optimizer.param_groups[0]["lr"] = config.lr_bmark * min(
                1.0, step / float(config.warmup_steps_bmark)
            )
        optimizer.zero_grad(set_to_none=True)
        losses = _endpoint_losses(predict(), target, valid, exposure, readout.log_dispersion, fit)
        loss = _weighted_total(losses)
        if not torch.isfinite(loss): raise FloatingPointError(f"{stage}: non-finite fit loss")
        loss.backward()
        norms = torch.stack([q.new_zeros(()) if value.grad is None else value.grad.detach().norm()
                             for _name, value in named]).cpu().tolist()
        for (name, _value), norm in zip(named, norms):
            if not np.isfinite(norm):
                raise FloatingPointError(f'{stage}: nonfinite gradient in {name}')
            audit = gradient_audit[name]
            audit['gradient_norm_sum'] += norm
            audit['gradient_norm_max'] = max(audit['gradient_norm_max'], norm)
            if norm > 0:
                audit['nonzero_steps'] += 1
                if audit['first_nonzero_step'] is None:
                    audit['first_nonzero_step'] = step
        if step == 1:
            first_step_gradient_norm = float(torch.sqrt(sum(
                parameter.grad.detach().float().square().sum()
                for parameter in params if parameter.grad is not None
            )))
        torch.nn.utils.clip_grad_norm_(params, config.gradient_clip); optimizer.step()
        if step % config.validate_every == 0 or step == maximum:
            peak_parameter_delta = max(
                peak_parameter_delta,
                max(
                    float((value.detach().cpu() - start).abs().max())
                    for value, start in zip(params, parameter_start)
                ),
            )
            with torch.no_grad():
                iloss = _endpoint_losses(predict(), target, valid, exposure, readout.log_dispersion, inner)
                value = float(_weighted_total(iloss))
            history.append({"step": step, "fit_loss": float(loss.detach()), "inner_loss": value,
                            'fit_endpoints': {k: float(v.detach()) for k, v in losses.items()},
                            "inner_endpoints": {k: float(v) for k, v in iloss.items()}})
            if np.isfinite(value) and value < best - 1e-5:
                best, best_step, stale = value, step, 0
                best_readout = copy.deepcopy(readout.state_dict())
                best_bg = copy.deepcopy(background_computer.model.state_dict())
                best_event = copy.deepcopy(event_computer.model.state_dict())
            else:
                stale += 1
            if stale >= config.patience_checks: break
    readout.load_state_dict(best_readout)
    background_computer.model.load_state_dict(best_bg)
    event_computer.model.load_state_dict(best_event)
    for name, value in named:
        change = value.detach().cpu() - initial_by_name[name]
        gradient_audit[name].update(selected_delta_max_abs=float(change.abs().max()),
                                    selected_delta_l2=float(change.norm()),
                                    gradient_norm_mean=gradient_audit[name]['gradient_norm_sum'] / max(history[-1]['step'], 1))
    return {"stage": stage, "initial_inner_loss": parent,
            'parameter_audit': gradient_audit,
            'optimizer_audit': {'name': 'AdamW', 'betas': [config.adam_beta1, config.adam_beta2],
                                'epsilon': config.adam_epsilon,
                                'final_group_learning_rates': [group['lr'] for group in optimizer.param_groups],
                                'group_weight_decays': [group['weight_decay'] for group in optimizer.param_groups],
                                'gradient_clip': config.gradient_clip,
                                'batch_contract': 'full FIT anchors; complete causal carry segments'},
            "parent_inner_loss": parent,
            "optimiser_initial_inner_loss": optimiser_initial,
            "best_inner_loss": best, "gain_over_parent": parent - best,
            "selected_step": best_step, "steps_run": history[-1]["step"],
            "selected_at_init": best_step == 0, "selected_at_budget_edge": best_step == maximum,
            "selected_parent_parity": best_step == 0,
            "first_step_gradient_norm": first_step_gradient_norm,
            "peak_parameter_delta_from_stage_start": peak_parameter_delta,
            "training_budget_exhausted": bool(
                history[-1]["step"] == maximum and stale < config.patience_checks
            ),
            "terminated_by_patience": bool(stale >= config.patience_checks),
            "history": history}


def train_h1_dual_subject(data: H1SubjectData, config: H1DualTrainConfig, *,
                          device: torch.device, out_dir: Path) -> dict[str, Any]:
    started = time.time(); torch.manual_seed(config.seed); np.random.seed(config.seed)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fit_np, inner_np, selection_np = (np.flatnonzero(data.rate.phase == p) for p in ("FIT", "INNER", "SELECTION"))
    fit, inner, selection = (torch.as_tensor(v, dtype=torch.long, device=device) for v in (fit_np, inner_np, selection_np))
    q, q_centre, q_scale = _standardise_features(data.rate.q_raw, fit_np, device)
    with torch.no_grad(): fixed_raw = _fixed_features(data, device, config.taus_seconds)
    bmark, b_centre, b_scale = _standardise_features(fixed_raw, fit_np, device)
    bg_current, bg_available, bg_audit, bg_names, bg_centre, bg_scale = _background_features(data, device)
    target, valid, target_scales = _target_bundle(data, device)
    exposure = torch.as_tensor(data.rate.target_exposure_seconds, dtype=torch.float32, device=device)
    widths = {"count": 1, "burden": data.burden_mark.shape[1],
              "community": data.grammar_dictionary.n_communities,
              "coupling": data.grammar_dictionary.n_communities ** 2,
              "mixture": data.grammar_dictionary.n_repertoires,
              "embedding": data.grammar_dictionary.event_repertoire_embedding.shape[1],
              "mark": data.grammar_mark.shape[1]}
    event_observer = DualStreamEventCTSSM(
        data.burden_mark.shape[1], data.grammar_mark.shape[1], taus_seconds=config.taus_seconds,
        burden_channels_per_tau=config.burden_channels_per_tau,
        grammar_channels_per_tau=config.grammar_channels_per_tau,
    ).to(device)
    event_computer = EventStateComputer(data, event_observer, device)
    random_observer = DualStreamEventCTSSM(
        data.burden_mark.shape[1], data.grammar_mark.shape[1], taus_seconds=config.taus_seconds,
        burden_channels_per_tau=config.burden_channels_per_tau,
        grammar_channels_per_tau=config.grammar_channels_per_tau,
    ).to(device)
    for parameter in random_observer.parameters(): parameter.requires_grad_(False)
    with torch.no_grad(): random_event_raw = EventStateComputer(data, random_observer, device)()
    random_event, random_centre, random_scale = _standardise_features(random_event_raw, fit_np, device)
    background_observer = GridBackgroundCTSSM(
        bg_current.shape[1], taus_seconds=config.taus_seconds,
        channels_per_tau=config.background_channels_per_tau,
    ).to(device)
    background_computer = BackgroundStateComputer(data, background_observer, bg_current, bg_available, device)
    bmark_burden_dim = len(config.taus_seconds) * data.burden_mark.shape[1]
    event_burden_dim = len(config.taus_seconds) * config.burden_channels_per_tau
    readout = NestedDualReadout(
        q.shape[1], bmark_burden_dim, bmark.shape[1] - bmark_burden_dim,
        bg_current.shape[1], background_observer.state_dim,
        event_burden_dim, event_observer.state_dim - event_burden_dim,
        widths, len(data.rate.horizons_seconds),
        state_readout_init_std=config.state_readout_init_std,
        bmark_readout_init_std=config.bmark_readout_init_std,
    ).to(device)
    stages = {}
    for stage in ("q", "bmark", "background_current", "background_state", "random", "event"):
        stages[stage] = _train_stage(
            stage, readout, q, bmark, bg_current, background_computer, event_computer,
            random_event, target, valid, exposure, fit, inner, config,
        )
    with torch.no_grad():
        bg_state = background_computer(); event_state = event_computer()
        bg_mean = bg_state[fit].mean(0, keepdim=True).expand_as(bg_state)
        event_mean = event_state[fit].mean(0, keepdim=True).expand_as(event_state)
        arms = {
            "B_rate": readout.predict(q),
            "B_mark": readout.predict(q, bmark=bmark),
            "B_mark_current_background": readout.predict(q, bmark=bmark, background_current=bg_current),
            "B_background_persistent": readout.predict(q, bmark=bmark, background_current=bg_current, background_state=bg_state),
            "random_event": readout.predict(q, bmark=bmark, background_current=bg_current, background_state=bg_state, random_event_state=random_event),
            "S_dual": readout.predict(q, bmark=bmark, background_current=bg_current, background_state=bg_state, event_state=event_state),
            "S_dual_constant_event": readout.predict(q, bmark=bmark, background_current=bg_current, background_state=bg_state, event_state=event_mean),
            "S_dual_constant_background": readout.predict(q, bmark=bmark, background_current=bg_current, background_state=bg_mean, event_state=event_state),
            "S_dual_constant_all": readout.predict(q, bmark=bmark, background_current=bg_current, background_state=bg_mean, event_state=event_mean),
        }
        scores = {name: _selection_score(pred, target, valid, exposure, readout.log_dispersion,
                                          selection, selection_np, tuple(data.rate.horizons_seconds))
                  for name, pred in arms.items()}
        time_shift = {}
        for h, seconds in enumerate(data.rate.horizons_seconds):
            shifted_event, event_ok = _block_shift(event_state, data, selection_np, float(seconds))
            shifted_bg, bg_ok = _block_shift(bg_state, data, selection_np, float(seconds))
            ok = event_ok & bg_ok
            v = {name: mask & ok[:, None] & (torch.arange(mask.shape[1], device=device)[None] == h)
                 for name, mask in valid.items()}
            key = str(int(seconds))
            if not bool(v["count"][selection].any()):
                time_shift[key] = None; continue
            correct = _selection_score(arms["S_dual"], target, v, exposure, readout.log_dispersion,
                                       selection, selection_np, tuple(data.rate.horizons_seconds))["by_horizon"][key]
            wrong_pred = readout.predict(q, bmark=bmark, background_current=bg_current,
                                         background_state=shifted_bg, event_state=shifted_event)
            wrong = _selection_score(wrong_pred, target, v, exposure, readout.log_dispersion,
                                     selection, selection_np, tuple(data.rate.horizons_seconds))["by_horizon"][key]
            time_shift[key] = {"correct": correct, "shifted": wrong, "gain": wrong["total"] - correct["total"]}
    checkpoint = out_dir / "checkpoint.pt"
    torch.save(data, out_dir / 'training_input_bundle.pt')
    torch.save({"event_observer": event_observer.state_dict(), "background_observer": background_observer.state_dict(),
                "readout": readout.state_dict(), "config": asdict(config), "widths": widths,
                "q_centre": q_centre, "q_scale": q_scale, "bmark_centre": b_centre, "bmark_scale": b_scale,
                "background_centre": bg_centre, "background_scale": bg_scale, "background_names": bg_names,
                "random_centre": random_centre, "random_scale": random_scale, "target_scales": target_scales,
                'training_input_bundle': str(out_dir / 'training_input_bundle.pt'),
                'training_input_bundle_sha256': sha256_file(out_dir / 'training_input_bundle.pt')}, checkpoint)
    trajectory = out_dir / "trajectory_and_targets.npz"
    np.savez_compressed(trajectory, anchor_time=data.rate.anchor_time, segment=data.rate.segment,
                        phase=data.rate.phase, horizons_seconds=np.asarray(data.rate.horizons_seconds),
                        event_state=event_state.cpu().numpy(), background_state=bg_state.cpu().numpy(),
                        background_current=bg_current.cpu().numpy(), fixed_mark_state=bmark.cpu().numpy(),
                        background_available=bg_available.cpu().numpy(), target_count=data.rate.target_count,
                        target_valid=data.rate.target_valid.astype(np.uint8),
                        target_exposure_seconds=data.rate.target_exposure_seconds)
    shift_gain = [v["gain"] for v in time_shift.values() if v is not None]
    selection_time = data.rate.anchor_time[selection_np]
    selection_segment = data.rate.segment[selection_np]
    independent_windows = {}
    for h, seconds in enumerate(data.rate.horizons_seconds):
        estimable = valid["count"][selection, h].detach().cpu().numpy().astype(bool)
        independent_windows[str(int(seconds))] = {
            "anchor_rows": int(estimable.sum()),
            "independent_windows": independent_window_count(
                selection_time[estimable], float(seconds),
                segment=selection_segment[estimable],
            ),
        }
    selection_span = (
        float(selection_time.max() - selection_time.min()) if selection_time.size else 0.0
    )
    slowest_tau = float(max(config.taus_seconds))
    card = {"format": "group_event_state_v0_3_7_h1_dual_observer_budget_complete_card_v2",
            "subject": data.subject, "seed": int(config.seed), "config": asdict(config),
            "horizons_seconds": list(data.rate.horizons_seconds), "shared_producer_across_horizons": True,
            "horizon_weighting": "equal weight across estimable physical horizons",
            "state_semantics": "S_obs_predictive_observer_not_Z_phys",
            "input_streams": ["fixed_clock_non_event_background", "group_event_burden", "group_event_conditional_grammar"],
            "background_audit": bg_audit, "representation_provenance": data.representation_provenance,
            "widths": widths, "stages": stages, "selection_scores": scores,
            "time_shift_by_horizon": time_shift,
            "independent_windows_by_horizon": independent_windows,
            "selection_window_audit": {
                "selection_span_seconds": selection_span,
                "slowest_memory_tau_seconds": slowest_tau,
                "slowest_memory_exceeds_selection_span": bool(slowest_tau > selection_span),
                "note": (
                    "held-out support is counted in non-overlapping physical windows; "
                    "a tau longer than this span is not distinguishable from a constant"
                ),
            },
            "code_provenance": _code_provenance(Path(__file__)),
            "primary_contrasts": {
                "persistent_background_gain_over_current": scores["B_mark_current_background"]["total"] - scores["B_background_persistent"]["total"],
                "event_gain_after_background": scores["B_background_persistent"]["total"] - scores["S_dual"]["total"],
                "event_gain_over_random_after_background": scores["random_event"]["total"] - scores["S_dual"]["total"],
                "constant_event_unexplained_gain": scores["S_dual_constant_event"]["total"] - scores["S_dual"]["total"],
                "constant_all_unexplained_gain": scores["S_dual_constant_all"]["total"] - scores["S_dual"]["total"],
                "correct_time_gain": float(np.mean(shift_gain)) if shift_gain else None,
            }, "checkpoint_path": str(checkpoint), "trajectory_path": str(trajectory),
            "maximum_training_time": float(data.rate.phase_boundaries["60pct"]),
            "elapsed_seconds": time.time() - started, "development_targets_read": False,
            "seizure_targets_read": False, "sealed_partition_opened": False}
    atomic_json(out_dir / "card.json", card)
    return card
