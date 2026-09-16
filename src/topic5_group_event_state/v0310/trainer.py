"""v0.3.10 human trainer: fair capacity, auditable sufficiency, exact resume.

Every clause referenced below is enumerated in
``training_contract_clauses.json`` under the run root. Patience or plateau
stopping is a stopping rule, never a convergence certificate.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..v035.contracts import atomic_json
from ..v039.transition import EventTransition, FutureReadout
from . import history as history_module
from .objective import fit_intercept_weights, intercept_reference, masked_objective, aggregate_objective

LEADS = (0., 2., 6.)
SELECTION_LEAD_INDEX = 1          # clause C16: INNER selection is the 2-hour lead only
STAGE_BUDGETS = dict(background=1600, event=3200, refitted_constant=1600)   # clause C6
TANH_SATURATION_Z = float(np.arctanh(0.99))


@dataclass
class CellConfig:
    data: str
    output_dir: str
    family: str
    state_width: int = 16
    transition_rank: int = 8
    readout_hidden: int = 32
    physical_batch: int = 128
    effective_batch: int = 128
    history_hours: float = 8.0
    source_mode: str = 'event_only'          # event_only | background_conditional
    view: str = 'joint'
    seed: int = 20260905
    device: str = 'cuda:0'
    lr: float = 0.003
    optimizer: str = 'AdamW'
    lr_schedule: str = 'plateau_x0.3_min200_first400_patience8'
    stage_budget: dict = field(default_factory=lambda: dict(STAGE_BUDGETS))
    objective_weights: str = 'fit_intercept'
    activation_checkpoint_chunk: int = 32
    min_history_coverage: float = 0.0
    deadline_epoch: float = 0.0
    resume: bool = True
    recipe_id: str = 'R0'


# ----------------------------------------------------------------- schedules
class BatchSchedule:
    """Pre-generated, family-independent index stream (clause C7).

    Position depends only on the effective update counter, so an exact resume
    needs the counter and nothing else. Paired families consume byte-identical
    batches; the digest is written on the card so pairing can be verified.
    """

    def __init__(self, valid_rows, active_leads, n_updates, batch, seed):
        self.leads = list(active_leads)
        self.batch = int(batch)
        rng = np.random.default_rng(seed)
        per = {lead: 0 for lead in self.leads}
        for u in range(n_updates):
            per[self.leads[u % len(self.leads)]] += 1
        self.draws = {}
        for lead in self.leads:
            pool = np.asarray(valid_rows[lead], np.int64)
            if not len(pool):
                raise ValueError(f'Lead {lead} has no valid FIT rows but was kept in rotation')
            self.draws[lead] = pool[rng.integers(0, len(pool), size=(max(per[lead], 1), self.batch))]
        self.pool_sizes = {lead: int(len(np.asarray(valid_rows[lead]))) for lead in self.leads}

    def rows_for(self, update_index):
        lead = self.leads[update_index % len(self.leads)]
        return lead, self.draws[lead][update_index // len(self.leads)]

    def digest(self):
        h = hashlib.sha256()
        for lead in self.leads:
            h.update(np.asarray(lead, '<f8').tobytes())
            h.update(self.draws[lead].astype('<i8', copy=False).tobytes())
        return h.hexdigest()

    def coverage(self, n_updates):
        used = {}
        for lead in self.leads:
            k = sum(1 for u in range(n_updates) if self.leads[u % len(self.leads)] == lead)
            rows = self.draws[lead][:k].ravel()
            used[str(lead)] = dict(updates=k, sampled_rows=int(rows.size),
                                   distinct_rows=int(np.unique(rows).size) if rows.size else 0,
                                   pool_rows=self.pool_sizes[lead],
                                   equivalent_epochs=float(rows.size / self.pool_sizes[lead]) if rows.size else 0.)
        return used


class PlateauSchedule:
    """Clauses C3-C5. Two LR drops must happen before a plateau can stop."""

    def __init__(self, lr, eval_every=50, min_updates_before_first_drop=400, patience=8,
                 factor=0.3, min_updates_per_level=200, max_drops=2, min_delta=1e-5):
        self.lr = float(lr); self.eval_every = int(eval_every)
        self.min_first = int(min_updates_before_first_drop); self.patience = int(patience)
        self.factor = float(factor); self.min_level = int(min_updates_per_level)
        self.max_drops = int(max_drops); self.min_delta = float(min_delta)
        self.best = math.inf; self.stale = 0; self.n_drops = 0; self.level_start = 0
        self.levels = [dict(lr=float(lr), start_update=0, end_update=None, best_at_level=None)]

    def observe(self, updates, score):
        improved = score < self.best - self.min_delta
        if improved:
            self.best = score; self.stale = 0
        else:
            self.stale += 1
        self.levels[-1]['best_at_level'] = min(self.levels[-1]['best_at_level'], score) \
            if self.levels[-1]['best_at_level'] is not None else score
        if self.stale < self.patience:
            return 'continue'
        at_level = updates - self.level_start
        if self.n_drops < self.max_drops:
            if updates >= self.min_first and at_level >= self.min_level:
                self.levels[-1]['end_update'] = updates
                self.lr *= self.factor; self.n_drops += 1; self.level_start = updates; self.stale = 0
                self.levels.append(dict(lr=self.lr, start_update=updates, end_update=None, best_at_level=None))
                return 'drop'
            return 'continue'
        if at_level >= self.min_level:
            self.levels[-1]['end_update'] = updates
            return 'stop'
        return 'continue'

    def state(self):
        return dict(lr=self.lr, best=self.best, stale=self.stale, n_drops=self.n_drops,
                    level_start=self.level_start, levels=copy.deepcopy(self.levels))

    def load(self, state):
        self.lr = state['lr']; self.best = state['best']; self.stale = state['stale']
        self.n_drops = state['n_drops']; self.level_start = state['level_start']; self.levels = state['levels']


# ------------------------------------------------------------ parameter groups
def build_param_groups(modules, weight_decay=1e-4):
    """Clause C1: bias / normalisation scale / log_dispersion are not decayed.

    The spec names those three; it is silent about the 1-D ``log_decay`` rate.
    The standard ndim<=1 rule covers all named cases plus log_decay, and the
    full assignment is written on the card so the choice is auditable.
    """
    decay, no_decay, record = [], [], {}
    for module_name, module in modules:
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            key = f'{module_name}.{name}'
            if p.ndim <= 1:
                no_decay.append(p); record[key] = dict(group='no_decay', ndim=int(p.ndim), numel=int(p.numel()))
            else:
                decay.append(p); record[key] = dict(group='decay', ndim=int(p.ndim), numel=int(p.numel()))
    groups = [dict(params=decay, weight_decay=weight_decay), dict(params=no_decay, weight_decay=0.0)]
    return [g for g in groups if g['params']], record


def tensor_hash(state_dict):
    h = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key]
        h.update(key.encode())
        h.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def per_parameter_hash(module):
    return {name: hashlib.sha256(np.ascontiguousarray(p.detach().cpu().numpy()).tobytes()).hexdigest()
            for name, p in module.named_parameters()}


def update_ratio(initial, module):
    """Zero-initialised tensors (bias, skew, the residual output layer) have no
    meaningful relative change; reporting one divides by zero and manufactures
    an astronomical number. Those rows carry the absolute move instead."""
    out = {}
    for name, p in module.named_parameters():
        base = initial[name]
        norm = float(base.norm())
        moved = float((p.detach().cpu() - base).norm())
        out[name] = dict(absolute_l2_change=moved, initial_l2=norm, final_l2=float(p.detach().norm()),
                         zero_initialised=norm == 0.,
                         relative_l2_change=None if norm == 0. else moved / norm)
    return out


# ---------------------------------------------------------------- cell inputs
def load_cell(cfg, data=None):
    data = data if data is not None else torch.load(cfg.data, map_location='cpu', weights_only=False)
    samples = data['samples']; n = len(samples)
    hist = history_module.history_matrix(data, cfg.history_hours,
                                         min_coverage=cfg.min_history_coverage or None)
    context_full = np.asarray([s['context'] for s in samples], np.float32)
    context = context_full[:, -5:] if cfg.source_mode == 'event_only' else context_full
    counts = np.asarray([[t[0] for t in s['targets']] for s in samples], np.float32)
    recruitment = np.asarray([[t[1] for t in s['targets']] for s in samples], np.float32)
    valid = np.asarray([[t[2] for t in s['targets']] for s in samples], bool)
    spatial_valid = np.asarray([[t[3] for t in s['targets']] for s in samples], bool)
    phases = np.asarray([s['phase'] for s in samples])
    anchors = np.asarray([s['anchor'] for s in samples], float)
    eligible = hist['eligible']
    return dict(data=data, n=n, x=torch.from_numpy(hist['x']), dt=torch.from_numpy(hist['dt']),
                lengths=hist['lengths'], coverage=hist['coverage'], eligible=eligible,
                context=context, counts=counts, recruitment=recruitment, valid=valid,
                spatial_valid=spatial_valid, phases=phases, anchors=anchors,
                horizon=hist, input_dim=data['input_dim'], n_recruitment=data['n_recruitment'],
                subject=data['subject'])


def partition(cell, view):
    phases, eligible, valid, spatial_valid = cell['phases'], cell['eligible'], cell['valid'], cell['spatial_valid']
    usable = valid if view != 'recruitment' else spatial_valid
    fit = np.flatnonzero((phases == 'FIT') & eligible)
    inner = np.flatnonzero((phases == 'INNER') & eligible & usable[:, SELECTION_LEAD_INDEX])
    selection = np.flatnonzero((phases == 'SELECTION') & eligible)
    pools = {hi: fit[usable[fit, hi]] for hi in range(len(LEADS))}
    active = [hi for hi in range(len(LEADS)) if len(pools[hi])]
    diagnostic = fit[usable[fit, SELECTION_LEAD_INDEX]]
    if len(diagnostic) > 256:
        diagnostic = diagnostic[np.linspace(0, len(diagnostic) - 1, 256).round().astype(int)]
    return dict(fit=fit, inner=inner, selection=selection, pools=pools, active_leads=active,
                fit_diagnostic=diagnostic)


# ---------------------------------------------------------------------- probes
def probe_state(readout, state, context, lead_hours):
    lead = state.new_full((len(state), 1), float(lead_hours) / 8)
    z = readout.layers[0](torch.cat((state, context, lead), dim=-1))
    return dict(state_rms=float(state.pow(2).mean().sqrt()),
                readout_tanh_saturated_fraction=float((z.abs() > TANH_SATURATION_Z).to(torch.float32).mean()))


def nonlinear_usage(observer, states):
    """Clause C11. A near-linear N is a failure of THIS fit, not proof that
    nonlinearity is absent."""
    if observer.family != 'N':
        return None
    with torch.no_grad():
        matrix = observer.generator_matrix()
        linear = states @ matrix.T
        z = states @ observer.v.T + observer.bias
        nonlinear = torch.tanh(z) @ observer.u.T
        ratio = (nonlinear.norm(dim=-1) / (linear.norm(dim=-1) + 1e-12))
        remainder = ((torch.tanh(z) - z).norm(dim=-1) / (z.norm(dim=-1) + 1e-12))
        return dict(median_nonlinear_over_linear=float(ratio.median()),
                    p90_nonlinear_over_linear=float(ratio.quantile(.9)),
                    median_tanh_remainder=float(remainder.median()),
                    median_abs_preactivation=float(z.abs().median()),
                    verdict='NONLINEAR_COMPONENT_UNUSED' if float(ratio.median()) < 1e-3 else 'NONLINEAR_COMPONENT_ACTIVE')


def window_aggregate(values, mask, window_id):
    """Primary summary: mean inside each two-hour physical window, then across."""
    values = np.asarray(values, float); mask = np.asarray(mask, bool)
    if not mask.any():
        return None, [], 0
    ids = np.asarray(window_id)[mask]; picked = values[mask]
    rows = []
    for w in np.unique(ids):
        rows.append(dict(window_id=int(w), n_anchors=int((ids == w).sum()), mean=float(picked[ids == w].mean())))
    return float(np.mean([r['mean'] for r in rows])), rows, len(rows)


def nonoverlapping_windows(starts, span=1800.0):
    starts = np.sort(np.asarray(starts, float)); kept = []
    cursor = -np.inf
    for s in starts:
        if s >= cursor:
            kept.append(float(s)); cursor = s + span
    return kept


def accumulate_update(predict, dispersion, targets, rows, lead_index, weights, view,
                      physical_batch, n_count, n_spatial):
    """Clause C7. Backward over micro-batches of the SAME 128 rows, each
    divided by the WHOLE-batch denominators, so the accumulated gradient is the
    single-step gradient. Returns the summed objective value for logging."""
    counts_t, recr_t, sval_t = targets
    total = 0.
    rows = np.asarray(rows)
    for start in range(0, len(rows), physical_batch):
        chunk = rows[start:start + physical_batch]
        mu, logits = predict(chunk, LEADS[lead_index], True)
        idx = torch.as_tensor(chunk, device=mu.device)
        loss, _, _ = masked_objective(mu, logits, counts_t[idx, lead_index], recr_t[idx, lead_index],
                                      sval_t[idx, lead_index], dispersion(), weights, view,
                                      n_count, n_spatial)
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite objective')
        loss.backward()
        total += float(loss.detach())
    return total


def sufficiency_verdict(stage_results):
    """Clause C15 t3: one budget- or wall-limited arm forbids a sufficiency claim."""
    any_budget = any(v.get('budget_limited') for v in stage_results.values())
    any_wall = any(v.get('wall_time_limited') for v in stage_results.values())
    all_plateau = bool(stage_results) and all(v.get('plateau_after_lr_reduction') for v in stage_results.values())
    return dict(verdict='NOT_ESTABLISHED_BY_STOP_REASON' if (any_budget or any_wall or not all_plateau)
                        else 'PLATEAU_AFTER_LR_REDUCTION_ON_ALL_STAGES',
                any_arm_budget_limited=bool(any_budget), any_arm_wall_time_limited=bool(any_wall),
                all_stages_plateaued=all_plateau,
                note='a plateau after two LR reductions is a stopping rule, not a convergence certificate; '
                     'any budget-limited stage forbids a sufficiency claim (clause C15 t3)')


def atomic_torch_save(payload, path):
    path = Path(path); tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, tmp); os.replace(tmp, path)


def run_cell(cfg, data=None, progress_hook=None):
    started = time.time()
    out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
    card_path = out / 'card.json'
    if card_path.exists():
        return json.loads(card_path.read_text())
    torch.set_num_threads(1)
    device = torch.device(cfg.device)
    cell = load_cell(cfg, data)
    part = partition(cell, cfg.view)
    K = cell['n_recruitment']

    def not_estimable(reason):
        card = dict(status='NOT_ESTIMABLE', subject=cell['subject'], family=cfg.family, reason=reason,
                    config=asdict(cfg), history_hours=cfg.history_hours,
                    n_fit=len(part['fit']), n_inner=len(part['inner']), n_selection=len(part['selection']),
                    development_targets_read=False, sealed_partition_opened=False, seizure_targets_read=False)
        atomic_json(card_path, card); return card

    if len(part['fit']) < 16 or len(part['inner']) < 4 or not len(part['selection']) or not part['active_leads']:
        return not_estimable('insufficient eligible FIT/INNER/SELECTION anchors at this horizon and coverage floor')

    context_t = torch.tensor(cell['context'], dtype=torch.float32, device=device)
    counts_t = torch.tensor(cell['counts'], dtype=torch.float32, device=device)
    recr_t = torch.tensor(cell['recruitment'], dtype=torch.float32, device=device)
    valid_t = torch.tensor(cell['valid'], device=device)
    sval_t = torch.tensor(cell['spatial_valid'], device=device)
    fit_t = torch.as_tensor(part['fit'], device=device)
    weights = fit_intercept_weights(counts_t, recr_t, valid_t, sval_t, fit_t)
    reference = intercept_reference(counts_t, recr_t, valid_t, sval_t, fit_t)

    torch.manual_seed(cfg.seed)
    baseline = FutureReadout(0, K, context_t.shape[1], hidden=cfg.readout_hidden).to(device)
    with torch.no_grad():
        baseline.layers[-1].weight.zero_()
        baseline.layers[-1].bias[0] = reference['log_mean']
        baseline.layers[-1].bias[1:] = torch.logit(reference['probability']) if reference['probability'] is not None else 0.
        baseline.log_dispersion.copy_(reference['log_dispersion'])
    torch.manual_seed(cfg.seed + 100)
    observer = EventTransition(cell['input_dim'], cfg.family, width=cfg.state_width,
                               rank=cfg.transition_rank, seed=cfg.seed).to(device)
    residual = FutureReadout(observer.width, K, 0, hidden=cfg.readout_hidden).to(device)
    with torch.no_grad():
        residual.layers[-1].weight.zero_(); residual.layers[-1].bias.zero_()
        residual.log_dispersion.copy_(baseline.log_dispersion)
    torch.manual_seed(cfg.seed + 200)
    constant = FutureReadout(observer.width, K, 0, hidden=cfg.readout_hidden).to(device)
    with torch.no_grad():
        constant.layers[-1].weight.zero_(); constant.layers[-1].bias.zero_()
        constant.log_dispersion.copy_(baseline.log_dispersion)
    initial_weights = {name: {k: v.detach().cpu().clone() for k, v in module.named_parameters()}
                       for name, module in (('baseline', baseline), ('observer', observer),
                                            ('residual', residual), ('constant', constant))}
    # Inventory is snapshotted while every module is still trainable; the
    # background parent is frozen after its own stage and would otherwise
    # report zero trainable parameters.
    construction_inventory = {}
    for name, module in (('baseline', baseline), ('observer', observer),
                         ('residual', residual), ('constant', constant)):
        construction_inventory[name] = {k: dict(shape=list(v.shape), parameters=int(v.numel()),
                                                trainable_in_own_stage=True)
                                        for k, v in module.named_parameters()}
        construction_inventory[name]['_total_parameters'] = int(sum(v.numel() for v in module.parameters()))
        construction_inventory[name]['_total_buffers'] = int(sum(v.numel() for v in module.buffers()))
    x_cpu, dt_cpu = cell['x'], cell['dt']
    fit_state_mean = torch.zeros(observer.width, device=device)

    def base_predict(rows, lead, training=False):
        idx = torch.as_tensor(rows, device=device)
        return baseline(context_t.new_empty((len(rows), 0)), context_t[idx], lead)

    def event_predict(rows, lead, training=False):
        index = torch.as_tensor(np.asarray(rows))
        xb = x_cpu[index].to(device); dtb = dt_cpu[index].to(device)
        state = observer.scan(xb, dtb, checkpoint_chunk=cfg.activation_checkpoint_chunk if training else 0)
        dm, dl = residual(state, state.new_empty((len(rows), 0)), lead)
        with torch.no_grad():
            bm, bl = base_predict(rows, lead)
        return bm + dm, bl + dl

    def constant_predict(rows, lead, training=False):
        with torch.no_grad():
            state = fit_state_mean[None].expand(len(rows), -1)
            bm, bl = base_predict(rows, lead)
        dm, dl = constant(state, state.new_empty((len(rows), 0)), lead)
        return bm + dm, bl + dl

    def endpoint_terms(predict, dispersion, rows, lead_index, chunk=256):
        nbs, sps, masks = [], [], []
        with torch.no_grad():
            for s in range(0, len(rows), chunk):
                block = np.asarray(rows[s:s + chunk])
                mu, logits = predict(block, LEADS[lead_index])
                idx = torch.as_tensor(block, device=device)
                _, nb, sp = masked_objective(mu, logits, counts_t[idx, lead_index], recr_t[idx, lead_index],
                                             sval_t[idx, lead_index], dispersion(), weights, cfg.view, 1, 1)
                nbs.append(nb.cpu().numpy()); sps.append(sp.cpu().numpy())
                masks.append(sval_t[idx, lead_index].cpu().numpy())
        if not nbs:
            return None
        return np.concatenate(nbs), np.concatenate(sps), np.concatenate(masks)

    def score(predict, dispersion, rows, lead_index=SELECTION_LEAD_INDEX):
        terms = endpoint_terms(predict, dispersion, rows, lead_index)
        if terms is None:
            return None
        return aggregate_objective(*terms, weights, cfg.view)

    modules = dict(baseline=baseline, observer=observer, residual=residual, constant=constant)
    latest_path = out / 'latest.pt'; best_path = out / 'best.pt'
    resume_state = None
    if cfg.resume and latest_path.exists():
        resume_state = torch.load(latest_path, map_location=device, weights_only=False)
        for name, module in modules.items():
            module.load_state_dict(resume_state['model'][name])
        fit_state_mean = resume_state['fit_state_mean'].to(device)
    stage_results = dict(resume_state['stage_results']) if resume_state else {}
    resume_note = None
    if resume_state is not None:
        resume_note = 'EXACT_RESUME' if resume_state.get('optimizer') is not None else 'RESTART_FROM_WEIGHTS'

    def save_latest(stage, update, optimizer, plateau, best, curve, diagnostics):
        atomic_torch_save(dict(model={k: v.state_dict() for k, v in modules.items()},
                               optimizer=optimizer.state_dict() if optimizer is not None else None,
                               plateau=plateau.state() if plateau is not None else None,
                               stage=stage, update=update, best=best, curve=curve, diagnostics=diagnostics,
                               stage_results=stage_results, fit_state_mean=fit_state_mean.cpu(),
                               torch_rng=torch.get_rng_state(), amp=None,
                               elapsed_seconds=time.time() - started), latest_path)

    def run_stage(name, trained, predict, dispersion, budget):
        nonlocal fit_state_mean
        stream_key = f"{cell['subject']}|{cfg.seed}|{name}|{cfg.view}|{cfg.history_hours}|{cfg.source_mode}"
        schedule = BatchSchedule(part['pools'], part['active_leads'], budget, cfg.effective_batch,
                                 seed=int.from_bytes(hashlib.sha256(stream_key.encode()).digest()[:8], 'little'))
        groups, group_record = build_param_groups([(k, modules[k]) for k in trained])
        optimizer = torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8)
        plateau = PlateauSchedule(cfg.lr)
        parameters = [p for g in groups for p in g['params']]
        start_update = 0; curve = []
        diagnostics = dict(pre_clip_norms=[], clipped=0, nonfinite=0,
                           zero_grad_updates={f'{m}.{n}': 0 for m in trained for n, _ in modules[m].named_parameters()})
        resuming = (resume_state is not None and resume_state.get('stage') == name
                    and resume_state.get('update', 0) > 0)
        if resuming:
            initial = resume_state['diagnostics']['initial_inner']
        else:
            initial = score(predict, dispersion, part['inner'])
            diagnostics['initial_inner'] = initial
        best = dict(score=initial['objective'], update=0,
                    state={k: copy.deepcopy(modules[k].state_dict()) for k in trained})
        plateau.best = initial['objective']
        if resuming:
            start_update = int(resume_state['update'])
            if resume_state.get('optimizer') is not None:
                optimizer.load_state_dict(resume_state['optimizer'])
            plateau.load(resume_state['plateau']); curve = list(resume_state['curve'])
            diagnostics = resume_state['diagnostics']
            best = resume_state['best']
            for group in optimizer.param_groups:
                group['lr'] = plateau.lr
        stop_reason = 'BUDGET_LIMIT'; update = start_update
        for update in range(start_update, budget):
            hi, rows = schedule.rows_for(update)
            n_count = len(rows)
            n_spatial = int(cell['spatial_valid'][rows, hi].sum())
            optimizer.zero_grad(set_to_none=True)
            accumulate_update(predict, dispersion, (counts_t, recr_t, sval_t), rows, hi, weights,
                              cfg.view, cfg.physical_batch, n_count, n_spatial)
            named = [(f'{m}.{pname}', p) for m in trained for pname, p in modules[m].named_parameters()]
            peaks = torch.stack([p.grad.abs().max() if p.grad is not None else
                                 torch.zeros((), device=device) for _, p in named]).cpu().numpy()
            for (key, _), peak in zip(named, peaks):
                if peak == 0.:
                    diagnostics['zero_grad_updates'][key] += 1
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, 2., error_if_nonfinite=True))
            diagnostics['pre_clip_norms'].append(norm)
            diagnostics['clipped'] += int(norm > 2.)
            optimizer.step()
            done = update + 1
            if done % plateau.eval_every == 0 or done == budget:
                inner = score(predict, dispersion, part['inner'])
                fit_diag = score(predict, dispersion, part['fit_diagnostic'])
                curve.append(dict(update=done, lr=plateau.lr, inner=inner, fit_diagnostic=fit_diag,
                                  last_pre_clip_norm=norm))
                if inner['objective'] < best['score'] - 1e-5:
                    best = dict(score=inner['objective'], update=done,
                                state={k: copy.deepcopy(modules[k].state_dict()) for k in trained})
                action = plateau.observe(done, inner['objective'])
                if action == 'drop':
                    for group in optimizer.param_groups:
                        group['lr'] = plateau.lr
                save_latest(name, done, optimizer, plateau, best, curve, diagnostics)
                atomic_json(out / 'progress.json', dict(status='RUNNING', stage=name, update=done, budget=budget,
                                                       lr=plateau.lr, inner=inner['objective'],
                                                       best_update=best['update'], best_inner=best['score'],
                                                       elapsed_seconds=time.time() - started))
                if progress_hook:
                    progress_hook(name, done, inner['objective'])
                if action == 'stop':
                    stop_reason = 'PLATEAU_AFTER_LR_REDUCTION'; break
                if cfg.deadline_epoch and time.time() >= cfg.deadline_epoch:
                    stop_reason = 'WALL_TIME_LIMITED'; break
        else:
            update = budget - 1
        updates_done = min(update + 1, budget)
        for k in trained:
            modules[k].load_state_dict(best['state'][k])
        atomic_torch_save(dict(stage=name, update=best['update'], score=best['score'],
                               model={k: modules[k].state_dict() for k in trained}), best_path)
        norms = np.asarray(diagnostics['pre_clip_norms'], float)
        fit_curve = [c['fit_diagnostic']['objective'] for c in curve if c['fit_diagnostic']]
        result = dict(stage=name, budget=budget, optimizer_updates=updates_done, stop_reason=stop_reason,
                      budget_limited=stop_reason == 'BUDGET_LIMIT',
                      wall_time_limited=stop_reason == 'WALL_TIME_LIMITED',
                      plateau_after_lr_reduction=stop_reason == 'PLATEAU_AFTER_LR_REDUCTION',
                      training_sufficiency='NOT_ESTABLISHED_BY_STOP_REASON',
                      lr_levels=plateau.levels, n_lr_drops=plateau.n_drops,
                      initial_inner=initial, selected_inner=best['score'], selected_update=best['update'],
                      final_inner=curve[-1]['inner']['objective'] if curve else initial['objective'],
                      best_differs_from_final=bool(curve and best['update'] != curve[-1]['update']),
                      parameter_groups=group_record,
                      gradient=dict(n_updates=int(norms.size),
                                    median_pre_clip_norm=float(np.median(norms)) if norms.size else None,
                                    p90_pre_clip_norm=float(np.quantile(norms, .9)) if norms.size else None,
                                    max_pre_clip_norm=float(norms.max()) if norms.size else None,
                                    clipped_fraction=float(diagnostics['clipped'] / max(norms.size, 1)),
                                    nonfinite_events=diagnostics['nonfinite'],
                                    zero_gradient_updates=diagnostics['zero_grad_updates']),
                      fit_diagnostic_first=fit_curve[0] if fit_curve else None,
                      fit_diagnostic_last=fit_curve[-1] if fit_curve else None,
                      sampling=schedule.coverage(updates_done), batch_schedule_sha256=schedule.digest(),
                      active_leads=[LEADS[h] for h in part['active_leads']],
                      curve=curve)
        stage_results[name] = result
        return result

    def stage_done(name):
        atomic_torch_save(dict(model={k: v.state_dict() for k, v in modules.items()}, optimizer=None,
                               plateau=None, stage=f'{name}_completed', update=0, best=None, curve=[],
                               diagnostics={}, stage_results=stage_results,
                               fit_state_mean=fit_state_mean.cpu(), torch_rng=torch.get_rng_state(), amp=None,
                               elapsed_seconds=time.time() - started), latest_path)

    if 'background' not in stage_results:
        run_stage('background', ['baseline'], base_predict, lambda: baseline.log_dispersion,
                  cfg.stage_budget['background'])
        stage_done('background')
    baseline.requires_grad_(False)
    halted = stage_results['background'].get('wall_time_limited')
    if not halted and 'event' not in stage_results:
        run_stage('event', ['observer', 'residual'], event_predict, lambda: residual.log_dispersion,
                  cfg.stage_budget['event'])
        with torch.no_grad():
            total = torch.zeros(observer.width, device=device)
            for s in range(0, len(part['fit']), 256):
                block = torch.as_tensor(np.asarray(part['fit'][s:s + 256]))
                total += observer.scan(x_cpu[block].to(device), dt_cpu[block].to(device), checkpoint_chunk=0).sum(0)
            fit_state_mean = total / len(part['fit'])
        stage_done('event')
    halted = halted or stage_results.get('event', {}).get('wall_time_limited')
    if not halted and 'refitted_constant' not in stage_results:
        run_stage('refitted_constant', ['constant'], constant_predict, lambda: constant.log_dispersion,
                  cfg.stage_budget['refitted_constant'])
        stage_done('refitted_constant')

    # ------------------------------------------------------------- SELECTION
    usable = cell['valid'] if cfg.view != 'recruitment' else cell['spatial_valid']
    anchors = cell['anchors']; selection = part['selection']
    selection_times = anchors[selection]
    donor = np.full(len(selection), -1, int)
    breaks = np.r_[0, np.flatnonzero(np.diff(selection_times) > 300.0001) + 1, len(selection)]
    for left, right in zip(breaks[:-1], breaks[1:]):
        if right - left >= 4:
            donor[left:right] = np.roll(np.arange(left, right), (right - left) // 2)
    selection_states = None
    if 'event' in stage_results:
        with torch.no_grad():
            chunks = []
            for s in range(0, len(selection), 256):
                block = torch.as_tensor(np.asarray(selection[s:s + 256]))
                chunks.append(observer.scan(x_cpu[block].to(device), dt_cpu[block].to(device), checkpoint_chunk=0))
            selection_states = torch.cat(chunks) if chunks else None

    support = cell['data']['observed_support']
    releases = np.asarray([b['release'] for b in cell['data']['event_replay_blocks']], float)
    metrics = {}; window_rows = []
    per_anchor_export = {}
    for hi, lead in enumerate(LEADS):
        rows = selection[usable[selection, hi]]
        if not len(rows):
            metrics[str(int(lead))] = dict(status='NOT_ESTIMABLE', reason='no scored SELECTION anchors at this lead')
            continue
        position = {int(r): i for i, r in enumerate(selection)}
        arms = {}
        arms['state'] = endpoint_terms(event_predict, lambda: residual.log_dispersion, rows, hi) if 'event' in stage_results else None
        arms['background'] = endpoint_terms(base_predict, lambda: baseline.log_dispersion, rows, hi)
        arms['refitted_constant'] = endpoint_terms(constant_predict, lambda: constant.log_dispersion, rows, hi) \
            if 'refitted_constant' in stage_results else None
        donor_terms = None
        if selection_states is not None:
            donor_index = np.asarray([donor[position[int(r)]] for r in rows])
            keep = donor_index >= 0
            if keep.any():
                with torch.no_grad():
                    wrong = selection_states[torch.as_tensor(donor_index[keep], device=device)]
                    dm, dl = residual(wrong, wrong.new_empty((int(keep.sum()), 0)), lead)
                    idx = torch.as_tensor(rows[keep], device=device)
                    bm, bl = base_predict(rows[keep], lead)
                    _, nb, sp = masked_objective(bm + dm, bl + dl, counts_t[idx, hi], recr_t[idx, hi],
                                                 sval_t[idx, hi], residual.log_dispersion, weights, cfg.view, 1, 1)
                donor_terms = (nb.cpu().numpy(), sp.cpu().numpy(), sval_t[idx, hi].cpu().numpy(), keep)
        window_id = np.floor(anchors[rows] / 7200.).astype(np.int64)

        def anchor_objective(terms):
            nb, sp, mask = terms[0], terms[1], terms[2]
            if cfg.view == 'count':
                return nb
            if cfg.view == 'recruitment':
                return sp
            return weights['weight_count'] * nb + weights['weight_recruitment'] * sp * mask

        summary = {}
        for key, terms in arms.items():
            if terms is None:
                summary[key] = None; continue
            value, rows_out, n_windows = window_aggregate(anchor_objective(terms), np.ones(len(rows), bool), window_id)
            count_mean, _, _ = window_aggregate(terms[0], np.ones(len(rows), bool), window_id)
            recr_mean, _, _ = window_aggregate(terms[1], terms[2].astype(bool), window_id)
            summary[key] = dict(window_equal_weight=value, anchor_equal_weight=float(anchor_objective(terms).mean()),
                                count_nll=count_mean, recruitment_nll=recr_mean, n_windows=n_windows)
            for r in rows_out:
                window_rows.append(dict(subject=cell['subject'], family=cfg.family, recipe=cfg.recipe_id,
                                        seed=cfg.seed, history_hours=cfg.history_hours,
                                        source_mode=cfg.source_mode, view=cfg.view, lead_hours=lead,
                                        arm=key, window_id=r['window_id'], n_anchors=r['n_anchors'],
                                        mean_objective=r['mean']))
        donor_summary = None
        if donor_terms is not None:
            nb, sp, mask, keep = donor_terms
            donor_value, _, _ = window_aggregate(anchor_objective((nb, sp, mask)), np.ones(int(keep.sum()), bool),
                                                 window_id[keep])
            state_on_keep = None
            if arms['state'] is not None:
                st = arms['state']
                state_on_keep, _, _ = window_aggregate(anchor_objective((st[0][keep], st[1][keep], st[2][keep])),
                                                       np.ones(int(keep.sum()), bool), window_id[keep])
            donor_summary = dict(window_equal_weight=donor_value, n_eligible=int(keep.sum()),
                                 state_on_same_anchors=state_on_keep,
                                 median_offset_hours=float(np.median(np.abs(
                                     selection_times[donor_index[keep]] - anchors[rows[keep]])) / 3600),
                                 rule='half circular shift inside each contiguous scored 5-minute run; '
                                      'background and clock stay at the correct time')
        floor = None
        if summary.get('state') and summary.get('background') and summary.get('refitted_constant'):
            floor = float(min(summary['background']['window_equal_weight'],
                              summary['refitted_constant']['window_equal_weight'])
                          - summary['state']['window_equal_weight'])
        target_starts = anchors[rows] + lead * 3600.
        n_blocks = int(np.unique(np.concatenate([
            np.flatnonzero((releases > a - cfg.history_hours * 3600.) & (releases <= a)) for a in anchors[rows]])).size) \
            if len(rows) else 0
        session = np.array([int(np.searchsorted(support[:, 1], a)) for a in anchors[rows]])
        metrics[str(int(lead))] = dict(
            status='SCORED', arms=summary, wrong_time_control=donor_summary,
            gain_over_frozen_background=(summary['background']['window_equal_weight'] - summary['state']['window_equal_weight'])
            if summary.get('state') else None,
            gain_over_refitted_constant=(summary['refitted_constant']['window_equal_weight'] - summary['state']['window_equal_weight'])
            if summary.get('state') and summary.get('refitted_constant') else None,
            gain_over_floored_control=floor,
            floor_rule='aggregate each arm on the common scored support first, then take the LOWER of frozen '
                       'background and refitted constant as the reference (clause C17)',
            denominators=dict(n_anchors=int(len(rows)), n_events=float(cell['counts'][rows, hi].sum()),
                              n_response_components=int(K), n_raw_measurement_blocks=n_blocks,
                              n_sessions=int(np.unique(session).size),
                              n_nonoverlapping_target_windows=len(nonoverlapping_windows(target_starts)),
                              nonoverlapping_target_window_starts=nonoverlapping_windows(target_starts),
                              n_two_hour_physical_windows=int(np.unique(window_id).size),
                              two_hour_window_ids=[int(v) for v in np.unique(window_id)],
                              overlapping_histories='target windows may share history; independence is not claimed'))
        if arms['state'] is not None:
            per_anchor_export[f'{int(lead)}h_anchor_time'] = anchors[rows]
            per_anchor_export[f'{int(lead)}h_window_id'] = window_id
            per_anchor_export[f'{int(lead)}h_state_objective'] = anchor_objective(arms['state'])
            per_anchor_export[f'{int(lead)}h_background_objective'] = anchor_objective(arms['background'])
            if arms['refitted_constant'] is not None:
                per_anchor_export[f'{int(lead)}h_constant_objective'] = anchor_objective(arms['refitted_constant'])
            per_anchor_export[f'{int(lead)}h_state_count_nll'] = arms['state'][0]
            per_anchor_export[f'{int(lead)}h_state_recruitment_nll'] = arms['state'][1]
            per_anchor_export[f'{int(lead)}h_spatial_valid'] = arms['state'][2]
    return _finalise(cfg, cell, part, modules, initial_weights, construction_inventory, stage_results, metrics, window_rows,
                     per_anchor_export, weights, fit_state_mean, observer, residual, baseline, constant,
                     out, card_path, started, resume_note, x_cpu, dt_cpu, device)


def input_column_report(x, dt, fit_rows, data):
    """Clause C8: scales, missing-mark rate, and pure-integration step share."""
    block = x[torch.as_tensor(np.asarray(fit_rows))].numpy()
    scale = np.asarray(data['normalization']['scale'], float)
    count_scale = float(data['normalization']['count_scale'])
    m = len(scale)
    events = block[..., 0].sum() * count_scale
    availability = block[..., 1 + m:1 + 2 * m].sum(axis=(0, 1)) * count_scale
    values = block[..., 1:1 + m]
    zero_steps = float((np.abs(block).sum(-1) == 0).mean())
    return dict(n_marks=int(m), count_scale=count_scale, fit_events_in_history=float(events),
                mark_available_fraction_min=float((availability / max(events, 1e-9)).min()),
                mark_available_fraction_median=float(np.median(availability / max(events, 1e-9))),
                normalisation_scale_min=float(scale.min()), normalisation_scale_max=float(scale.max()),
                value_column_rms_median=float(np.median(np.sqrt((values ** 2).mean(axis=(0, 1))))),
                pure_integration_step_fraction=zero_steps,
                upstream_clip='FIT-only median/MAD with locked clip=8, applied per event before cumulation',
                normalisation_note='no BatchNorm/LayerNorm anywhere in the continuous state')


def _finalise(cfg, cell, part, modules, initial_weights, construction_inventory, stage_results, metrics, window_rows,
              per_anchor_export, weights, fit_state_mean, observer, residual, baseline, constant,
              out, card_path, started, resume_note, x_cpu, dt_cpu, device):
    diag_rows = np.asarray(part['fit_diagnostic'])
    probes = None; nonlinear = None
    if 'event' in stage_results and len(diag_rows):
        with torch.no_grad():
            block = torch.as_tensor(diag_rows)
            states = observer.scan(x_cpu[block].to(device), dt_cpu[block].to(device), checkpoint_chunk=0)
            probes = probe_state(residual, states, states.new_empty((len(diag_rows), 0)), LEADS[SELECTION_LEAD_INDEX])
            nonlinear = nonlinear_usage(observer, states)
            if nonlinear is not None:
                for name in ('u', 'v'):
                    p = dict(observer.named_parameters())[name]
                    base = initial_weights['observer'][name]
                    nonlinear[f'{name}_relative_l2_change'] = float((p.detach().cpu() - base).norm() / max(float(base.norm()), 1e-12))

    event = stage_results.get('event')
    diagnosis = None
    if event:
        first, last = event['fit_diagnostic_first'], event['fit_diagnostic_last']
        fit_down = bool(first is not None and last is not None and first - last > 1e-5)
        inner_up = bool(event['initial_inner']['objective'] - event['selected_inner'] > 1e-5)
        diagnosis = dict(initial_checkpoint_selected=event['selected_update'] == 0,
                         fit_objective_decreased=fit_down, inner_objective_improved=inner_up,
                         reading=('BOTH_IMPROVED' if fit_down and inner_up else
                                  'FIT_IMPROVED_INNER_FLAT' if fit_down else
                                  'INNER_IMPROVED_FIT_FLAT' if inner_up else 'NEITHER_IMPROVED'),
                         note='these two are recorded separately and never merged; an initial checkpoint being '
                              'selected is neither a program bug nor evidence that events were learned '
                              '(clause C13)')

    inventory = construction_inventory
    buffers = {k: dict(shape=list(v.shape), elements=int(v.numel())) for k, v in observer.named_buffers()}

    scores_path = out / 'selection_scores.npz'
    np.savez_compressed(scores_path, **{k: np.asarray(v) for k, v in per_anchor_export.items()})
    weights_path = out / 'model.pt'
    atomic_torch_save(dict(observer=observer.state_dict(), residual=residual.state_dict(),
                           baseline=baseline.state_dict(), constant=constant.state_dict(),
                           fit_state_mean=fit_state_mean.cpu(), config=asdict(cfg),
                           input_dim=cell['input_dim'], n_recruitment=cell['n_recruitment'],
                           context_dim=cell['context'].shape[1], objective_weights=weights,
                           fit_rows=np.asarray(part['fit']), inner_rows=np.asarray(part['inner']),
                           selection_rows=np.asarray(part['selection'])), weights_path)

    verdict = sufficiency_verdict(stage_results)
    any_budget = verdict['any_arm_budget_limited']; any_wall = verdict['any_arm_wall_time_limited']
    status = ('WALL_TIME_LIMITED' if any_wall else 'COMPLETE')
    root = Path(__file__).resolve().parents[3]
    sources = ['src/topic5_group_event_state/v0310/trainer.py', 'src/topic5_group_event_state/v0310/objective.py',
               'src/topic5_group_event_state/v0310/history.py', 'src/topic5_group_event_state/v039/transition.py',
               'scripts/train_group_event_state_v0310_human.py']
    card = dict(
        status=status, schema='v0310_human_cell_v1', subject=cell['subject'], family=cfg.family,
        recipe=cfg.recipe_id, seed=cfg.seed, view=cfg.view, source_mode=cfg.source_mode,
        history_hours=cfg.history_hours, config=asdict(cfg),
        state_width=int(observer.width), input_dim=int(cell['input_dim']),
        context_dim=int(cell['context'].shape[1]), n_recruitment=int(cell['n_recruitment']),
        transition_rank_rule='min(width/2,16)', transition_rank=int(cfg.transition_rank),
        objective_weights=weights, stages=stage_results, metrics=metrics,
        optimisation_diagnosis=diagnosis, state_probes=probes, nonlinear_usage=nonlinear,
        input_columns=input_column_report(x_cpu, dt_cpu, part['fit'], cell['data']),
        model_inventory=inventory, observer_buffers=buffers,
        history=dict(hours=cfg.history_hours, steps=int(cell['horizon']['steps']),
                     rebuilt_from_replay_blocks=bool(cell['horizon']['rebuilt_from_replay']),
                     median_past_coverage=float(np.median(cell['coverage'])),
                     min_coverage_floor=cfg.min_history_coverage,
                     n_eligible_anchors=int(cell['eligible'].sum()), n_anchors=int(cell['n'])),
        training_sufficiency_vector=dict(
            a_updates_and_coverage={k: dict(updates=v['optimizer_updates'], sampling=v['sampling'])
                                    for k, v in stage_results.items()},
            b_curves={k: v['curve'] for k, v in stage_results.items()},
            c_layers=dict(inventory=inventory,
                          initial_parameter_hashes={m: {n: h for n, h in per_parameter_hash(mod).items()}
                                                    for m, mod in modules.items()},
                          update_ratio={m: update_ratio(initial_weights[m], mod) for m, mod in modules.items()}),
            d_gradients={k: v['gradient'] for k, v in stage_results.items()},
            e_lr_levels_and_budget={k: dict(levels=v['lr_levels'], n_drops=v['n_lr_drops'],
                                            stop_reason=v['stop_reason']) for k, v in stage_results.items()},
            f_best_vs_final={k: dict(selected_update=v['selected_update'], selected_inner=v['selected_inner'],
                                     final_inner=v['final_inner'], differs=v['best_differs_from_final'])
                             for k, v in stage_results.items()},
            g_independent_physical_support={k: v.get('denominators') for k, v in metrics.items() if isinstance(v, dict)},
            **verdict),
        resume=resume_note,
        data_path=str(cfg.data), data_sha256=hashlib.sha256(Path(cfg.data).read_bytes()).hexdigest(),
        split_sha256=hashlib.sha256(np.concatenate([np.asarray(part[k]) for k in ('fit', 'inner', 'selection')])
                                    .astype('<i8').tobytes()).hexdigest(),
        target_sha256=hashlib.sha256(np.ascontiguousarray(cell['counts']).tobytes()
                                     + np.ascontiguousarray(cell['recruitment']).tobytes()).hexdigest(),
        checkpoint=str(weights_path), checkpoint_sha256=hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        scores=str(scores_path), scores_sha256=hashlib.sha256(scores_path.read_bytes()).hexdigest(),
        source_hashes={p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources
                       if (root / p).exists()},
        development_targets_read=False, sealed_partition_opened=False, seizure_targets_read=False,
        interpretation='Closed-block receiver-time observer. Every measurement block updates the observer; '
                       'that is a computational fact and is not evidence that an IED changed physiology '
                       '(clause C15 t8).',
        elapsed_seconds=time.time() - started)
    atomic_json(card_path, card)
    if window_rows:
        header = list(window_rows[0])
        with (out / 'physical_windows.csv').open('w') as handle:
            handle.write(','.join(header) + '\n')
            for row in window_rows:
                handle.write(','.join(str(row[k]) for k in header) + '\n')
    atomic_json(out / 'progress.json', dict(status=status, stage='done', elapsed_seconds=time.time() - started))
    return card
