"""Five-arm training: paired INNER recipe selection and fixed-recipe final refits.

C31 fixes the fitting budget at 25 human fits (5 arms x 2 INNER on the first
optimizer seed, then 5 arms x 3 seeds of final refit).  C32 runs the two INNER
trajectories as fully independent models and optimizers that nonetheless share
one evaluation grid and one learning-rate milestone list, both driven by the
mean 30-minute organization score.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from . import data as D
from . import conditions as CD
from .prepare import Prepared
from .engine import build_model, infer_asof, predict, score_predictions
from .objective import (VIEWS, SECONDARY, FAMILIES, aggregate, selection_score, training_loss,
                        SELECTION_HORIZON)

ROOT = '/data/hfosp_group_event_state_epilepsy_state_v040'
PACKETS = '/data/hfosp_group_event_state_rich_event_identification_v0311/packets'
HORIZONS = (1, 5, 30, 120)
ARMS = (('B_stats', 'P_stats', 'stats_history', None),
        ('B_marks', 'P_marks', 'marked_history', None),
        ('S_stats', 'P_stats', 'state', None),
        ('S_marks', 'P_marks', 'state', None),
        ('S_marks-short', 'P_marks', 'state', 'H_SHORT'))
FINAL_SEEDS = (20260906, 20260907, 20260908)


@dataclass
class RunConfig:
    subject: str = 'epilepsiae_1125'
    arm_name: str = 'S_marks'
    protocol: str = 'S-E'
    stage: str = 'inner0'
    inputs: str = 'P_marks'
    family: str = 'I-L-G1'
    arm: str = 'state'
    seed: int = 20260906
    split_seed: int = 20260906
    sampler_seed: int = 20260916
    eval_seed: int = 20260926
    history_hours: float | None = None
    short_history_minutes: int = 120
    grad_hours: float = 2.
    batch_size: int = 32
    microbatch: int = 32
    train_paths: int = 4
    eval_paths: int = 64
    lr: float = 1e-3
    dynamics_lr: float = 3e-4
    max_updates: int = 3200
    extended_updates: int = 6400
    eval_every: int = 50
    eval_stride: int = 30
    eval_chunk: int = 24
    checkpoint_every: int = 25
    activation_checkpoint: bool = True
    packets_root: str = PACKETS
    out_dir: str = ROOT + '/runs'
    recipe_path: str = ''
    device: str = 'cuda:0'


def json_safe(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.generic,)):
        return v.item()
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().tolist()
    if isinstance(v, (set, tuple)):
        return list(v)
    raise TypeError(type(v).__name__)


def atomic_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2, default=json_safe, ensure_ascii=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_torch(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    torch.save(data, tmp)
    os.replace(tmp, path)


def source_digest():
    root = Path(__file__).parent
    files = {str(p.relative_to(root.parents[2])): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(root.glob('*.py'))}
    for rel in ('v0312/data.py', 'v0312/model.py', 'v0312/numerics.py', 'v0312/prepare.py',
                'v0311/data.py', 'v0311/packets.py', 'v0311/synthetic.py'):
        p = root.parent / rel
        files[str(p.relative_to(root.parents[2]))] = hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted((root.parents[2] / 'scripts').glob('*group_event_state_v040*.py')):
        files[str(p.relative_to(root.parents[2]))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return D.digest(files), files


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def tensor_hash(state):
    h = hashlib.sha256()
    for n, t in sorted(state.items()):
        h.update(n.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def config_identity(cfg):
    c = asdict(cfg)
    for k in ('device', 'out_dir', 'microbatch', 'eval_chunk', 'checkpoint_every'):
        c.pop(k, None)
    return D.digest(c)


def tag(cfg):
    h = 'persistent' if cfg.history_hours is None else f'h{cfg.history_hours:g}'
    return f'{cfg.subject}__{cfg.arm_name}__{cfg.protocol}__{cfg.stage}__{h}__seed{cfg.seed}'


def load_run(cfg):
    payload = torch.load(Path(cfg.packets_root) / f'{cfg.subject}.pt', weights_only=False, map_location='cpu')
    split = D.build_split(payload, cfg.subject, cfg.split_seed, cfg.stage, cfg.protocol)
    x, t, _ = D.packet_tables(payload, split)
    sc = D.fit_scaling(payload, split, x, t)
    cond_scaling = CD.fit_condition_scaling(payload, split, cfg.short_history_minutes)
    prep = Prepared(payload, split, sc, torch.device(cfg.device), cond_scaling)
    prep.scaling = dict(prep.scaling,
                        transform_id=D.digest((sc['transform_id'], D.digest(cond_scaling['center']),
                                               D.digest(cond_scaling['scale']), int(cfg.short_history_minutes))))
    model = build_model(prep, cfg.inputs, cfg.family, cfg.arm, cfg.seed)
    return model, prep


def training_table(prep):
    if not hasattr(prep, '_training_table'):
        prep._training_table = D.target_table(prep.payload, prep.split, 'fit', stride=1)
    return prep._training_table


def training_ids(prep):
    return np.unique(training_table(prep)[:, 0])


def window_support(prep, ids):
    """C15/C18: data-only per-view window denominators, fixed before any microbatch."""
    ids = np.asarray(ids, int)
    pk = prep.payload['packets']
    out = {k: 0. for k in VIEWS + SECONDARY}
    if not len(ids):
        return out
    ti = torch.as_tensor(ids, device=prep.device)
    out['count'] = float((prep.valid[ti] > 0).sum())
    has = ((prep.count[ti] > 0) & (prep.valid[ti] > 0))
    out['spatial'] = out['load'] = float(has.sum())
    n_morph = 0
    for i in ids:
        a, b = int(pk['event_lo'][i]), int(pk['event_hi'][i])
        if b <= a:
            continue
        ok = (prep.band_ratio_valid[a:b].sum(-1) > 0) | (prep.xlag_valid[a:b].sum(-1) > 0) | (prep.iqr_valid[a:b] > 0)
        if bool(ok.any()):
            n_morph += 1
    out['morphology'] = float(n_morph)
    return out


def training_normalizers(prep, batch_size):
    table = training_table(prep)
    n = len(training_ids(prep))
    result = {}
    for h in HORIZONS:
        units = window_support(prep, np.unique(table[table[:, 1] == h, 0]))
        result[h] = {k: batch_size * v / max(n, 1) for k, v in units.items()}
    return result


def loss_for_targets(model, prep, cfg, ids, normalizers, update, credit_starts=None):
    ids = np.asarray(ids, int)
    legal = training_table(prep)
    legal = legal[np.isin(legal[:, 0], ids)]
    queries = np.unique(legal[:, 2])
    state = infer_asof(model, prep, queries, 'fit', cfg.history_hours, training=True,
                       grad_hours=cfg.grad_hours, activation_checkpoint=cfg.activation_checkpoint,
                       grad_start_by_episode=credit_starts)
    predictions = predict(model, prep, state, tuple(np.unique(legal[:, 1])), paths=cfg.train_paths,
                          seed=cfg.seed + 1000003 * update)
    rows = score_predictions(model, prep, state, predictions, 'fit', ids)
    loss = prep.stats.new_zeros(())
    for row in rows:
        term = training_loss({v: row[v] for v in VIEWS + SECONDARY}, row['horizon'], normalizers[row['horizon']])
        if term is not None:
            loss = loss + term
    return loss


def optimizer_for(model, cfg):
    groups = {}
    for name, p in model.named_parameters():
        slow = name.startswith('dynamics.') or 'eta_head' in name
        decay = p.ndim >= 2 and not slow
        groups.setdefault((slow, decay), []).append(p)
    params = [dict(params=ps, lr=cfg.dynamics_lr if slow else cfg.lr,
                   initial_lr=cfg.dynamics_lr if slow else cfg.lr,
                   weight_decay=1e-4 if decay else 0.,
                   name=f'{"dynamics_cov" if slow else "network"}_{"decay" if decay else "nodecay"}')
              for (slow, decay), ps in groups.items()]
    return torch.optim.AdamW(params, betas=(.9, .999), eps=1e-8)


def update_once(model, prep, cfg, optimizer, ids, normalizers, step, microbatch):
    optimizer.zero_grad(set_to_none=True)
    total = 0.
    legal = training_table(prep)
    q = np.unique(legal[np.isin(legal[:, 0], ids), 2])
    starts = prep.split['episode_start'][q]
    # C26: the credit boundary is rebuilt from current parameters every update.
    credit = {int(s): int(q[starts == s].min() - round(cfg.grad_hours * 60)) for s in np.unique(starts)}
    for a in range(0, len(ids), microbatch):
        loss = loss_for_targets(model, prep, cfg, ids[a:a + microbatch], normalizers, step, credit)
        if not torch.isfinite(loss):
            raise FloatingPointError('non-finite loss; no update permitted')
        loss.backward()
        total += float(loss.detach())
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    if not torch.isfinite(norm):
        raise FloatingPointError('non-finite gradient; no update permitted')
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise FloatingPointError('non-finite parameter after update')
    return total, float(norm)


@torch.no_grad()
def evaluate(model, prep, cfg, role='inner', collect=False, rule='EVOLVE', reference=None, paths=None, seed=None):
    table = D.target_table(prep.payload, prep.split, role, cfg.eval_stride)
    if not len(table):
        return dict(status='NOT_ESTIMABLE', selection=None, summary={}, records=[], query_metadata=[])
    queries = np.unique(table[:, 2])
    targets = np.unique(table[:, 0])
    grouped = {}
    records = []
    meta = []
    legacy = {k: [0., 0.] for k in VIEWS + SECONDARY}
    for a in range(0, len(queries), cfg.eval_chunk):
        qq = queries[a:a + cfg.eval_chunk]
        state = infer_asof(model, prep, qq, role, cfg.history_hours,
                           producer_hash=tensor_hash(model.state_dict()))
        pred = predict(model, prep, state, HORIZONS, paths=paths or cfg.eval_paths,
                       seed=cfg.eval_seed if seed is None else seed, rule=rule, reference=reference)
        rows = score_predictions(model, prep, state, pred, role, targets, keep_events=collect)
        for row in rows:
            for view in VIEWS + SECONDARY:
                grouped.setdefault((row['horizon'], view), []).append(
                    dict(nll=row[view]['nll'].detach().cpu(), valid=row[view]['valid'].detach().cpu()))
            ev = row.get('morphology_events')
            if ev is not None:
                grouped[(row['horizon'], 'morphology')][-1].update(
                    event_nll=ev['event_nll'].detach().cpu(), event_valid=ev['event_valid'].detach().cpu(),
                    families={f: dict(nll=ev['families'][f]['nll'].detach().cpu(),
                                      valid=ev['families'][f]['valid'].detach().cpu())
                              for f in FAMILIES})
            for k, v in row['legacy_packet_joint_component_score'].items():
                legacy[k][0] += v['logp_sum']
                legacy[k][1] += v['units']
            if collect:
                rec = dict(horizon=row['horizon'], packet=row['packet'], query_packet=row['query_packet'],
                           information_age_minutes=row['information_age_minutes'])
                for view in VIEWS + SECONDARY:
                    rec[view] = dict(nll=row[view]['nll'].detach().cpu().numpy(),
                                     valid=row[view]['valid'].detach().cpu().numpy())
                if ev is not None:
                    rec['events'] = dict(
                        event_index=ev['event_index'].detach().cpu().numpy(),
                        window=row['packet'][ev['event_row'].detach().cpu().numpy()],
                        nll=ev['event_nll'].detach().cpu().numpy(),
                        valid=ev['event_valid'].detach().cpu().numpy(),
                        families={f: dict(nll=ev['families'][f]['nll'].detach().cpu().numpy(),
                                          valid=ev['families'][f]['valid'].detach().cpu().numpy(),
                                          n_components=ev['families'][f]['n_components'].detach().cpu().numpy())
                                  for f in FAMILIES})
                records.append(rec)
        meta.append(state.metadata())
    summary = aggregate(grouped)
    J, detail = selection_score(summary)
    return dict(status='COMPLETE', selection=J, selection_detail=detail, summary=summary,
                legacy_packet_joint_component_score={k: dict(score=-v[0] / v[1] if v[1] else None, units=v[1])
                                                     for k, v in legacy.items()},
                n_targets=len(targets), n_queries=len(queries), target_table_digest=D.digest(table),
                records=records if collect else [], query_metadata=meta if collect else [])


class Plateau:
    """C33: two learning-rate drops on a 1e-3 tolerance, then an eight-check stop."""

    def __init__(self):
        self.best = math.inf
        self.bad = 0
        self.drops = 0

    def observe(self, score):
        if score < self.best - 1e-3:
            self.best = score
            self.bad = 0
            return 'improve'
        self.bad += 1
        if self.drops < 2 and self.bad >= 6:
            self.bad = 0
            self.drops += 1
            return 'drop'
        if self.drops >= 2 and self.bad >= 8:
            return 'stop'
        return 'wait'


def interpret_training(stage, selected_updates, stop_reason, plateau):
    if int(selected_updates) == 0:
        return ('origin checkpoint selected; this fit supplies no learned component, and an unlearned '
                'input-driven representation is still not a static trait')
    if stage == 'final' and stop_reason == 'fixed_inner_recipe':
        return 'final refit followed a frozen paired-INNER step count; nonzero steps are not convergence evidence'
    drops = int((plateau or {}).get('drops', 0))
    if stop_reason == 'plateau' and drops >= 2:
        return 'two-stage learning-rate patience plateau; local optimization evidence, not global convergence'
    if stop_reason == 'budget':
        return 'budget edge reached; convergence remains unresolved'
    return f'{stop_reason}; optimization adequacy remains unresolved'


class Trajectory:
    """One INNER origin: its own data, model, optimizer and sampler."""

    def __init__(self, cfg, stage):
        self.cfg = replace(cfg, stage=stage)
        self.model, self.prep = load_run(self.cfg)
        self.optimizer = optimizer_for(self.model, self.cfg)
        self.ids = training_ids(self.prep)
        if len(self.ids) < self.cfg.batch_size:
            raise ValueError(f'{stage}: only {len(self.ids)} training targets for batch {self.cfg.batch_size}')
        self.normalizers = training_normalizers(self.prep, self.cfg.batch_size)
        self.rng = np.random.default_rng(self.cfg.sampler_seed + (0 if stage == 'inner0' else 1))
        self.microbatch = self.cfg.microbatch
        self.initial = {n: q.detach().cpu().clone() for n, q in self.model.named_parameters()}
        self.scores = []

    def step(self, update):
        anchor = int(self.rng.integers(len(self.ids)))
        batch = self.ids[(anchor + np.arange(self.cfg.batch_size)) % len(self.ids)]
        backup = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        opt_backup = copy.deepcopy(self.optimizer.state_dict())
        while True:
            try:
                return update_once(self.model, self.prep, self.cfg, self.optimizer, batch,
                                   self.normalizers, update, self.microbatch)
            except torch.OutOfMemoryError:
                self.model.load_state_dict(backup)
                self.optimizer.load_state_dict(opt_backup)
                self.optimizer.zero_grad(set_to_none=True)
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                if self.microbatch <= 1:
                    raise
                self.microbatch = max(1, self.microbatch // 2)

    def state(self):
        return dict(model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                    sampler=self.rng.bit_generator.state, microbatch=self.microbatch, scores=self.scores,
                    split_id=self.prep.split['split_id'], transform_id=self.prep.scaling['transform_id'])

    def load(self, blob):
        self.model.load_state_dict(blob['model'])
        self.optimizer.load_state_dict(blob['optimizer'])
        self.rng.bit_generator.state = blob['sampler']
        self.microbatch = blob['microbatch']
        self.scores = blob['scores']
        if blob['split_id'] != self.prep.split['split_id'] or blob['transform_id'] != self.prep.scaling['transform_id']:
            raise ValueError('resume data contract changed')


def run_paired_inner(cfg, *, deadline=None, progress=None):
    """C32: two independent trajectories, one common evaluation grid and LR milestone list.

    The step is chosen from the actual common evaluation points by the lowest
    mean organization score, ties resolved to the earlier step.  Two separate
    argmins are never stitched together and the two trajectories never diverge
    in learning rate.
    """
    if cfg.device.startswith('cuda'):
        torch.cuda.set_device(cfg.device)
        torch.cuda.reset_peak_memory_stats(cfg.device)
    begin = time.time()
    out = Path(cfg.out_dir) / f'{cfg.subject}__{cfg.arm_name}__paired_inner__seed{cfg.seed}'
    out.mkdir(parents=True, exist_ok=True)
    source, files = source_digest()
    packets_hash = file_hash(Path(cfg.packets_root) / f'{cfg.subject}.pt')
    identity = config_identity(replace(cfg, stage='paired_inner'))
    legs = {stage: Trajectory(cfg, stage) for stage in ('inner0', 'inner1')}
    plateau = Plateau()
    curve = []
    updates = 0
    budget = cfg.max_updates
    incidents = []
    lr_drops = []
    stop_flag = [False]
    old_handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.signal(sig, lambda *a: stop_flag.__setitem__(0, True))
    last = out / 'last.pt'

    def evaluate_both():
        scores = {}
        for stage, leg in legs.items():
            ev = evaluate(leg.model, leg.prep, leg.cfg, 'inner')
            scores[stage] = ev
        vals = [scores[s]['selection'] for s in ('inner0', 'inner1')]
        mean = None if any(v is None for v in vals) else float(np.mean(vals))
        return scores, vals, mean

    def save():
        atomic_torch(dict(config=asdict(cfg), identity=identity, source_digest=source, source_files=files,
                          packets_sha256=packets_hash, updates=updates, budget=budget, curve=curve,
                          plateau=vars(plateau), lr_drops=lr_drops, incidents=incidents,
                          legs={k: v.state() for k, v in legs.items()},
                          initial={k: v.initial for k, v in legs.items()}), last)
        atomic_json(dict(status='RUNNING', updates=updates, budget=budget, source_digest=source,
                         pid=os.getpid(), heartbeat=time.time()), out / 'progress.json')

    try:
        if last.exists():
            old = torch.load(last, weights_only=False, map_location=cfg.device)
            if old['identity'] != identity or old['source_digest'] != source:
                raise ValueError('resume config/source changed')
            if old.get('packets_sha256') != packets_hash:
                raise ValueError('resume measurement packets changed')
            for k, leg in legs.items():
                leg.load(old['legs'][k])
                leg.initial = old['initial'][k]
            updates = old['updates']
            budget = old['budget']
            curve = old['curve']
            plateau.__dict__.update(old['plateau'])
            lr_drops = old['lr_drops']
            incidents = old['incidents']
            del old
        else:
            scores, vals, mean = evaluate_both()
            if mean is None:
                raise ValueError('paired INNER organization score is not estimable at step 0')
            plateau.best = mean
            curve.append(dict(update=0, inner0=vals[0], inner1=vals[1], mean=mean,
                              lrs=[g['lr'] for g in legs['inner0'].optimizer.param_groups],
                              detail={k: scores[k]['selection_detail'] for k in scores}))
            for stage, leg in legs.items():
                leg.scores.append(dict(update=0, selection=scores[stage]['selection'],
                                       summary=scores[stage]['summary']))
            save()
        stop_reason = 'budget'
        while updates < budget:
            if stop_flag[0] or (deadline is not None and time.time() >= deadline):
                atomic_json(dict(status='PAUSED', updates=updates, budget=budget, source_digest=source),
                            out / 'progress.json')
                save()
                return dict(status='PAUSED', updates=updates, directory=str(out))
            t0 = time.time()
            losses = {}
            for stage in ('inner0', 'inner1'):
                losses[stage] = legs[stage].step(updates)
            updates += 1
            row = dict(update=updates, step_seconds=time.time() - t0,
                       train_loss={k: v[0] for k, v in losses.items()},
                       gradient_norm={k: v[1] for k, v in losses.items()},
                       lrs=[g['lr'] for g in legs['inner0'].optimizer.param_groups])
            action = 'train'
            if updates % cfg.eval_every == 0:
                scores, vals, mean = evaluate_both()
                if mean is None:
                    raise ValueError('paired INNER organization score disappeared')
                row.update(inner0=vals[0], inner1=vals[1], mean=mean,
                           detail={k: scores[k]['selection_detail'] for k in scores})
                for stage, leg in legs.items():
                    leg.scores.append(dict(update=updates, selection=scores[stage]['selection'],
                                           summary=scores[stage]['summary']))
                action = plateau.observe(mean)
                if action == 'drop':
                    lr_drops.append(updates)
                    for leg in legs.values():
                        for g in leg.optimizer.param_groups:
                            g['lr'] *= .3
                if action == 'stop':
                    stop_reason = 'plateau'
                if updates >= budget and action == 'improve' and cfg.extended_updates > budget:
                    budget = cfg.extended_updates
            curve.append(row)
            if updates % cfg.checkpoint_every == 0 or action in ('drop', 'stop'):
                save()
            if progress:
                progress(row, action)
            if action == 'stop':
                break
        save()
        evaluated = [r for r in curve if r.get('mean') is not None]
        best = min(evaluated, key=lambda r: (r['mean'], r['update']))
        recipe = dict(status='FROZEN', source_digest=source, arm_name=cfg.arm_name,
                      config={k: v for k, v in asdict(cfg).items() if k not in ('device', 'out_dir', 'stage', 'recipe_path')},
                      updates=int(best['update']),
                      lr_drop_updates=[u for u in lr_drops if 0 < u < int(best['update'])],
                      selected_mean_organization_score=best['mean'],
                      per_leg_at_selection={s: best.get(s) for s in ('inner0', 'inner1')},
                      evaluation_grid=[r['update'] for r in evaluated],
                      stop_reason=stop_reason, plateau=vars(plateau), executed_updates=updates,
                      rule=('common evaluation grid and common learning-rate milestones driven by the mean '
                            '30-minute organization score; the step is one actual common evaluation point, '
                            'never a combination of two separate argmins'),
                      learned_state_eligible=int(best['update']) > 0,
                      training_adequacy=('two-stage learning-rate patience plateau on the shared schedule; local '
                                         'optimization evidence only' if stop_reason == 'plateau'
                                         else f'{stop_reason}; optimization adequacy unresolved'),
                      per_leg_curves={k: legs[k].scores for k in legs},
                      seconds=time.time() - begin,
                      peak_allocated_gib=(torch.cuda.max_memory_allocated() / 2 ** 30
                                          if cfg.device.startswith('cuda') else 0.))
        atomic_json(recipe, out / 'recipe.json')
        atomic_json(dict(curve=curve, incidents=incidents), out / 'curve.json')
        atomic_json(dict(status='COMPLETE', updates=updates), out / 'progress.json')
        return recipe
    except Exception as exc:
        atomic_json(dict(status='FAILED', updates=updates, error=f'{type(exc).__name__}: {exc}',
                         source_digest=source), out / 'progress.json')
        raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def run_final(cfg, *, deadline=None, progress=None):
    """C34/C35: fixed recipe, own legal FIT scaler and trait, no checkpoint selection."""
    if cfg.device.startswith('cuda'):
        torch.cuda.set_device(cfg.device)
        torch.cuda.reset_peak_memory_stats(cfg.device)
    begin = time.time()
    if cfg.stage != 'outer':
        raise ValueError('final refits run on the outer stage')
    if not cfg.recipe_path:
        raise ValueError('final refit requires a frozen paired-INNER recipe')
    recipe = json.loads(Path(cfg.recipe_path).read_text())
    source, files = source_digest()
    if recipe['source_digest'] != source:
        raise ValueError('recipe/source mismatch')
    for k in ('subject', 'arm_name', 'inputs', 'family', 'arm', 'history_hours', 'short_history_minutes'):
        if recipe['config'][k] != getattr(cfg, k):
            raise ValueError(f'recipe mismatch: {k}')
    out = Path(cfg.out_dir) / tag(cfg)
    out.mkdir(parents=True, exist_ok=True)
    packets_hash = file_hash(Path(cfg.packets_root) / f'{cfg.subject}.pt')
    model, prep = load_run(cfg)
    opt = optimizer_for(model, cfg)
    ids = training_ids(prep)
    normalizers = training_normalizers(prep, cfg.batch_size)
    rng = np.random.default_rng(cfg.sampler_seed)
    initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    budget = int(recipe['updates'])
    curve = []
    updates = 0
    micro = cfg.microbatch
    incidents = []
    last = out / 'last.pt'

    def save():
        atomic_torch(dict(config=asdict(cfg), source_digest=source, packets_sha256=packets_hash,
                          model=model.state_dict(), optimizer=opt.state_dict(),
                          sampler=rng.bit_generator.state, updates=updates, curve=curve, microbatch=micro,
                          incidents=incidents, initial=initial, split=prep.split, scaling=prep.scaling,
                          cond_scaling=prep.cond_scaling), last)
        atomic_json(dict(status='RUNNING', updates=updates, budget=budget, pid=os.getpid(),
                         heartbeat=time.time()), out / 'progress.json')

    if last.exists():
        old = torch.load(last, weights_only=False, map_location=cfg.device)
        if old['source_digest'] != source or old.get('packets_sha256') != packets_hash:
            raise ValueError('resume source or packets changed')
        model.load_state_dict(old['model'])
        opt.load_state_dict(old['optimizer'])
        rng.bit_generator.state = old['sampler']
        updates = old['updates']
        curve = old['curve']
        micro = old['microbatch']
        incidents = old['incidents']
        initial = old['initial']
        del old
    while updates < budget:
        if deadline is not None and time.time() >= deadline:
            save()
            atomic_json(dict(status='PAUSED', updates=updates, budget=budget), out / 'progress.json')
            return dict(status='PAUSED', updates=updates, directory=str(out))
        anchor = int(rng.integers(len(ids)))
        batch = ids[(anchor + np.arange(cfg.batch_size)) % len(ids)]
        t0 = time.time()
        loss, norm = update_once(model, prep, cfg, opt, batch, normalizers, updates, micro)
        updates += 1
        if updates in recipe.get('lr_drop_updates', []):
            for g in opt.param_groups:
                g['lr'] *= .3
        curve.append(dict(update=updates, train_loss=loss, gradient_norm=norm, step_seconds=time.time() - t0,
                          lrs=[g['lr'] for g in opt.param_groups]))
        if updates % cfg.checkpoint_every == 0:
            save()
        if progress:
            progress(curve[-1], 'train')
    save()
    final = evaluate(model, prep, cfg, 'outer', collect=True)
    atomic_torch(dict(config=asdict(cfg), state_dict=model.state_dict(), split=prep.split, scaling=prep.scaling,
                      cond_scaling=prep.cond_scaling, source_digest=source, packets_sha256=packets_hash,
                      selected_updates=budget, recipe=recipe), out / 'selected.pt')
    atomic_torch(final, out / 'predictions.pt')
    inventory = []
    for n, p in model.named_parameters():
        p0 = initial[n].to(p)
        delta = float((p - p0).norm())
        den = float(p0.norm())
        inventory.append(dict(name=n, shape=list(p.shape), n_parameters=p.numel(), initial_norm=den,
                              final_norm=float(p.norm()), update_norm=delta,
                              relative_update=delta / den if den > 0 else None))
    card = dict(status='COMPLETE', scope='development', config=asdict(cfg), source_digest=source,
                source_files=files, packets_sha256=packets_hash, split_id=prep.split['split_id'],
                transform_id=prep.scaling['transform_id'], updates=updates, selected_updates=budget,
                stop_reason='fixed_inner_recipe', recipe_path=cfg.recipe_path,
                selection_at_freeze=recipe['selected_mean_organization_score'],
                score_role='outer', summary=final['summary'], selection=final['selection'],
                legacy_packet_joint_component_score=final['legacy_packet_joint_component_score'],
                target_table_digest=final.get('target_table_digest'),
                selected_parameter_hash=tensor_hash(model.state_dict()), parameter_inventory=inventory,
                curve=curve, incidents=incidents, seconds=time.time() - begin,
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30 if cfg.device.startswith('cuda') else 0.,
                batch_size=cfg.batch_size, actual_microbatch=micro,
                training_interpretation=interpret_training('final', budget, 'fixed_inner_recipe', recipe.get('plateau')),
                representation_class=('learned' if budget > 0 else
                                      'unlearned input-driven representation; not a static trait'),
                credit_semantics=('persistent arms share a credit boundary two hours before the earliest query of '
                                  'the batch; a strict short arm credits its entire H'))
    atomic_json(card, out / 'card.json')
    atomic_json(dict(status='COMPLETE', updates=updates), out / 'progress.json')
    return card
