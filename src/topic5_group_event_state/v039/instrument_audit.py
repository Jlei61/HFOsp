"""Paired instrument audit. Patience stopping is never a convergence certificate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def known_truth_margin(rows, case, better, worse, experiment='joint'):
    picked = {}
    for row in rows:
        if (row.get('case'), row.get('experiment')) != (case, experiment):
            continue
        family = row.get('family')
        if family not in (better, worse):
            continue
        if family in picked:
            raise ValueError('Multiple optimization seeds require explicit paired aggregation')
        # The original summary omitted the optimization seed; its source card has it.
        card = json.loads(Path(row['source']).read_text()) if row.get('source') else row
        picked[family] = card
    if len(picked) != 2:
        return {'margin': None, 'converged_margin': False, 'n_seeds': None}
    a, b = picked[better], picked[worse]
    seeds = [r.get('seed', r.get('config', {}).get('seed')) for r in (a, b)]
    data_seeds = [r.get('truth', {}).get('seed') for r in (a, b)]
    if all(v is not None for v in seeds) and seeds[0] != seeds[1]:
        raise ValueError('Unpaired optimization seeds')
    if all(v is not None for v in data_seeds) and data_seeds[0] != data_seeds[1]:
        raise ValueError('Unpaired synthetic data seeds')
    if a.get('input_sha256') != b.get('input_sha256'):
        raise ValueError('Unpaired synthetic inputs')
    limited = [r.get('optimization_limited') for r in (a, b)]
    all_limited = all(v is True for v in limited)
    steps = {r.get('selected_step') for r in (a, b)}
    return {
        'margin': float(b['held_out']['total'] - a['held_out']['total']),
        'all_arms_budget_limited': all_limited,
        'any_arm_budget_limited': any(v is True for v in limited),
        'all_arms_patience_stopped': all(r.get('stop_reason') == 'INNER_PATIENCE'
                                       and r.get('optimization_limited') is False for r in (a, b)),
        'shared_selected_step': next(iter(steps)) if len(steps) == 1 and all_limited else None,
        'n_seeds': 1 if all(v is not None for v in seeds) else None,
        'n_data_seeds': 1 if all(v is not None for v in data_seeds) else None,
        'converged_margin': False,
        'training_sufficiency': 'NOT_ESTABLISHED_BY_STOP_REASON',
    }


def convergence_audit(root, expected_seeds=(20260905, 20260906, 20260907)):
    """Validate the registered nine longer-budget fits, including saved scores.

    This retains the directory's historical name. It audits termination and
    pairing, not mathematical convergence, biological power, or nine data worlds.
    """
    folder = Path(root) / 'instruments_convergence_audit'
    if not folder.exists():
        return None
    runs, records, episode_ids, source_hashes, common = {}, [], None, None, None
    for path in sorted(folder.glob('nonlinear_*_seed*')):
        if path.suffix in ('.pt', '.npz') or not path.is_file():
            continue
        c = json.loads(path.read_text())
        if c.get('status') != 'COMPLETE' or c.get('case') != 'nonlinear_transition':
            raise ValueError(f'Incomplete/wrong-case instrument: {path}')
        family, seed, config = c['family'], c['seed'], c['config']
        if family not in ('F', 'L', 'N') or path.name != f'nonlinear_{family}_seed{seed}':
            raise ValueError(f'Filename/card disagreement: {path}')
        if config['seed'] != seed or config['family'] != family:
            raise ValueError('Configuration identity mismatch')
        if family in runs.setdefault(seed, {}):
            raise ValueError('Duplicate optimization seed/family')
        paired = {k: v for k, v in config.items() if k not in ('seed', 'family', 'device', 'output')}
        paired.update(input_sha256=c['input_sha256'], truth=c['truth'])
        if common is not None and paired != common:
            raise ValueError('Mismatched data or training recipe within paired audit')
        common = paired
        if not c.get('source_hashes') or (source_hashes is not None and c['source_hashes'] != source_hashes):
            raise ValueError('Missing/mixed source provenance')
        source_hashes = c['source_hashes']
        selected, steps, budget = c['selected_step'], c['steps_run'], config['max_steps']
        if not (0 <= selected <= steps <= budget):
            raise ValueError('Invalid selected/actual/budget steps')
        reason = c['stop_reason']
        if reason not in ('INNER_PATIENCE', 'BUDGET_LIMIT') or c['optimization_limited'] != (reason == 'BUDGET_LIMIT'):
            raise ValueError('Inconsistent stopping metadata')
        if reason == 'BUDGET_LIMIT' and steps != budget:
            raise ValueError('Budget stop before budget')
        curve = c['training_curve']
        if not curve or curve[-1]['step'] != steps:
            raise ValueError('Missing final training-curve evaluation')
        if selected and not any(x['step'] == selected and np.isclose(x['inner'], c['selected_inner'], atol=1e-8, rtol=0) for x in curve):
            raise ValueError('Selected checkpoint absent from INNER curve')
        if reason == 'INNER_PATIENCE':
            tail = [x for x in curve if x['step'] > selected]
            if len(tail) < config['patience'] or any(x['inner'] < c['selected_inner']-1e-5 for x in tail):
                raise ValueError('Patience stop unsupported by training curve')
        for key in ('checkpoint', 'scores'):
            if sha256(c[key]) != c[key + '_sha256']:
                raise ValueError(f'{key} hash mismatch: {path}')
        with np.load(c['scores'], allow_pickle=False) as scores:
            ids = scores['episode_id']
            if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
                raise ValueError('Invalid held-out episode IDs')
            if episode_ids is not None and not np.array_equal(ids, episode_ids):
                raise ValueError('Unpaired held-out episodes')
            episode_ids = ids.copy()
            for key in ('total', 'count', 'recruitment'):
                values = scores[key]
                if values.shape != ids.shape or not np.isfinite(values).all():
                    raise ValueError('Invalid per-episode scores')
                if not np.isclose(values.mean(), c['held_out'][key], atol=1e-6, rtol=0):
                    raise ValueError('Card/score-array disagreement')
        runs[seed][family] = c
        records.append(dict(source=str(path), source_sha256=sha256(path), family=family, seed=seed,
                            data_seed=c['truth']['seed'], selected_step=selected, steps_run=steps,
                            stop_reason=reason, optimization_limited=c['optimization_limited'],
                            max_steps=budget, held_out=c['held_out'], checkpoint_sha256=c['checkpoint_sha256'],
                            scores_sha256=c['scores_sha256'], elapsed_seconds=c['elapsed_seconds']))
    if set(runs) != set(expected_seeds) or any(set(v) != {'F', 'L', 'N'} for v in runs.values()):
        raise ValueError('The registered audit requires complete F/L/N triplets for all three optimization seeds')
    seeds = sorted(runs)
    def contrast(worse, better):
        values = [runs[k][worse]['held_out']['total']-runs[k][better]['held_out']['total'] for k in seeds]
        return dict(values=values, median=float(np.median(values)), positive=sum(v > 0 for v in values), n=len(values))
    return dict(status='VERIFIED', seeds=[str(k) for k in seeds], n_data_seeds=1,
                data_seed=common['truth']['seed'], n_held_out_episodes=len(episode_ids),
                selected_steps={f: [runs[k][f]['selected_step'] for k in seeds] for f in ('F', 'L', 'N')},
                all_patience_stopped=all(r['stop_reason'] == 'INNER_PATIENCE' for r in records),
                budget_limited_runs=sum(r['optimization_limited'] for r in records),
                converged_margin=False, training_sufficiency='NOT_ESTABLISHED_BY_STOP_REASON',
                real_patient_power_calibrated=False, source_hashes=source_hashes,
                input_sha256=common['input_sha256'],
                input_hash_scope='inputs + context only; original hash omits dt, targets, and split',
                records=records, N_over_L=contrast('L', 'N'), N_over_F=contrast('F', 'N'), L_over_F=contrast('F', 'L'))
