"""The five registered differences, their supports, and the task denominator.

C51: every difference is reported with its own numerator, denominator and
interval on identical physical support.  They are never combined into a single
rich-state total.  C54: budget stops, numerical failures, non-estimable results
and tasks that never ran all stay in the task table.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .objective import VIEWS, SECONDARY, FAMILIES
from .train import RunConfig, atomic_json, tag, ARMS, FINAL_SEEDS

DIFFERENCES = (('rich_history_input_increment', 'B_stats', 'B_marks',
                'what the current history-learning procedure can use from the extra content'),
               ('state_rich_increment', 'S_stats', 'S_marks',
                'what the current persistent state model uses from the extra content'),
               ('representation_retention', 'B_marks', 'S_marks',
                'capability retained or lost relative to the rich history reference; not a measure of all information'),
               ('earlier_history_contribution', 'S_marks-short', 'S_marks',
                'the effect of adding history older than the frozen short window, including earlier rate, '
                'exposure and coarse space'))


def flatten_windows(records, view):
    out = {}
    for r in records:
        for pkt, nll, ok in zip(r['packet'], r[view]['nll'], r[view]['valid']):
            if ok:
                key = (int(r['horizon']), int(pkt))
                if key in out:
                    raise ValueError('a physical target and horizon were scored twice')
                out[key] = float(nll)
    return out


def flatten_events(records, family=None):
    out = {}
    for r in records:
        e = r.get('events')
        if e is None:
            continue
        nll = e['families'][family]['nll'] if family else e['nll']
        ok = e['families'][family]['valid'] if family else e['valid']
        for eid, win, v, good in zip(e['event_index'], e['window'], nll, ok):
            if good:
                out[(int(r['horizon']), int(eid))] = (float(v), int(win))
    return out


def paired(a, b, blocks_of, label, block_hours=(2., 4.)):
    """R(a) - R(b) on identical support; positive means b is better."""
    common = sorted(set(a) & set(b))
    if not common:
        return dict(status='NOT_ESTIMABLE', reason='no shared scored unit', n_a=len(a), n_b=len(b))
    d = np.array([a[k] - b[k] for k in common])
    out = dict(status='DEVELOPMENT', estimand=label, mean_difference=float(d.mean()), n_units=len(common),
               unsupported_in_a=len(a) - len(common), unsupported_in_b=len(b) - len(common))
    rng = np.random.default_rng(827)
    for hours in block_hours:
        g = np.array([blocks_of(k, hours) for k in common])
        uniq = np.unique(g)
        if len(uniq) < 3:
            out[f'block_{hours:g}h'] = dict(n_blocks=int(len(uniq)), ci=None,
                                            reason='fewer than three time blocks')
            continue
        sums = np.array([[d[g == u].sum(), int((g == u).sum())] for u in uniq], float)
        draws = sums[rng.integers(len(uniq), size=(2000, len(uniq)))].sum(1)
        out[f'block_{hours:g}h'] = dict(n_blocks=int(len(uniq)),
                                        ci=np.quantile(draws[:, 0] / draws[:, 1], [.025, .975]).tolist())
    days = np.array([blocks_of(k, 24.) for k in common])
    out['per_day'] = [dict(day=int(u), n_units=int((days == u).sum()), mean_difference=float(d[days == u].mean()))
                      for u in np.unique(days)]
    return out


def load_arm(root, subject, arm_name, seed, short_minutes):
    for name, inputs, arm, hh in ARMS:
        if name != arm_name:
            continue
        cfg = RunConfig(subject=subject, arm_name=name, inputs=inputs, arm=arm, stage='outer', seed=int(seed),
                        history_hours=(short_minutes / 60. if hh == 'H_SHORT' else None),
                        short_history_minutes=int(short_minutes), out_dir=str(Path(root) / 'runs'))
        d = Path(cfg.out_dir) / tag(cfg)
        if not (d / 'predictions.pt').exists() or not (d / 'card.json').exists():
            return None
        return dict(directory=str(d), card=json.loads((d / 'card.json').read_text()),
                    predictions=torch.load(d / 'predictions.pt', weights_only=False, map_location='cpu'))
    raise ValueError(arm_name)


def summarize(plan_path, out_path):
    plan = json.loads(Path(plan_path).read_text())
    root = Path(plan['root'])
    state = root / 'queue_state'
    subject = plan['subject']
    short = plan['short_history_minutes']
    tasks = []
    for task in plan['tasks']:
        p = state / (task['id'] + '.json')
        status = json.loads(p.read_text()) if p.exists() else dict(status='NOT_RUN')
        tasks.append(dict(id=task['id'], kind=task['kind'],
                          arm=task.get('config', {}).get('arm_name'),
                          stage=task.get('config', {}).get('stage'),
                          seed=task.get('config', {}).get('seed'),
                          task=task.get('task'), realization=task.get('realization'),
                          status=status.get('status', 'NOT_RUN'), error=status.get('error')))
    payload = torch.load(Path(RunConfig().packets_root) / f'{subject}.pt',
                         weights_only=False, map_location='cpu')
    ends = np.asarray(payload['packets']['end'], float)
    origin = float(ends.min())
    event_packet = D.event_packets(payload)

    def blocks_of(key, hours):
        horizon, unit = key
        t = ends[unit] if unit < len(ends) else origin
        return int((t - origin) // (hours * 3600.))

    def event_blocks_of(key, hours):
        horizon, eid = key
        t = ends[event_packet[eid]]
        return int((t - origin) // (hours * 3600.))

    arms = {}
    for name, *_ in ARMS:
        arms[name] = {int(s): load_arm(root, subject, name, s, short) for s in FINAL_SEEDS}
    differences = {}
    for key, a_name, b_name, meaning in DIFFERENCES:
        per_seed = {}
        for seed in FINAL_SEEDS:
            a, b = arms[a_name][int(seed)], arms[b_name][int(seed)]
            if a is None or b is None:
                per_seed[int(seed)] = dict(status='NOT_RUN',
                                           missing=[n for n, v in ((a_name, a), (b_name, b)) if v is None])
                continue
            entry = {}
            for view in VIEWS + SECONDARY:
                entry[f'{view}_window_equal_weight'] = paired(
                    flatten_windows(a['predictions']['records'], view),
                    flatten_windows(b['predictions']['records'], view), blocks_of, meaning)
            entry['morphology_event_equal_weight'] = paired(
                {k: v[0] for k, v in flatten_events(a['predictions']['records']).items()},
                {k: v[0] for k, v in flatten_events(b['predictions']['records']).items()},
                event_blocks_of, meaning + ' (event-weighted companion)')
            entry['morphology_per_family'] = {
                f: paired({k: v[0] for k, v in flatten_events(a['predictions']['records'], f).items()},
                          {k: v[0] for k, v in flatten_events(b['predictions']['records'], f).items()},
                          event_blocks_of, f'{meaning} ({f})') for f in FAMILIES}
            entry['selection_scores'] = dict(a=a['card']['selection'], b=b['card']['selection'])
            entry['selected_updates'] = dict(a=a['card']['selected_updates'], b=b['card']['selected_updates'])
            per_seed[int(seed)] = entry
        differences[key] = dict(numerator=a_name, subtrahend=b_name, meaning=meaning, per_seed=per_seed)
    consumers = {}
    for seed in FINAL_SEEDS:
        cfg = RunConfig(subject=subject, arm_name='S_marks', inputs='P_marks', arm='state', stage='outer',
                        seed=int(seed), short_history_minutes=int(short))
        p = root / 'frozen' / tag(cfg) / 'consumers.json'
        consumers[int(seed)] = json.loads(p.read_text()) if p.exists() else dict(status='NOT_RUN')
    synthetic = {}
    for p in sorted((root / 'synthetic' / 'scores').glob('*.json')):
        synthetic[p.stem] = json.loads(p.read_text())
    contracts = {}
    for name in ('support', 'training_objectives', 'state_export', 'consumer_routes', 'g0_checks', 'dlocal'):
        p = root / 'contracts' / f'{name}.json'
        if p.exists():
            contracts[name] = str(p)
    result = dict(
        status='SNAPSHOT', version=plan['version'], subject=subject, short_history_minutes=short,
        source_digest=plan['source_digest'],
        tasks=tasks,
        n_complete=sum(t['status'] == 'COMPLETE' for t in tasks), n_total=len(tasks),
        task_counts={k: sum(t['status'] == k for t in tasks)
                     for k in sorted({t['status'] for t in tasks})},
        budget=plan['budget'],
        arms_available={n: [s for s in FINAL_SEEDS if arms[n][int(s)] is not None] for n, *_ in ARMS},
        differences=differences,
        frozen_state_conditional_increment={
            int(s): (consumers[int(s)].get('H2a_A', {}).get('contrasts', {}).get('R(C+H)-R(C+H+S)')
                     if consumers[int(s)].get('status') == 'COMPLETE' else dict(status='NOT_RUN'))
            for s in FINAL_SEEDS},
        consumers=consumers, synthetic=synthetic, contracts=contracts,
        weighting=dict(h1_main='window-equal-weight', h1_companion='event-equal-weight',
                       h2a='event-equal-weight',
                       dependence='2-hour paired time blocks by default, 4-hour sensitivity, per-day listing',
                       units='queries, events and optimizer seeds are not patients'),
        reading_rules=[
            'a rich advantage together with a long-history advantage does not by itself show that the earlier '
            'rich content contributes; that claim needs the matched training experiment registered for the next package',
            'the five differences are reported separately and are never summed into one rich-state score',
            'E1125 is development material throughout; nothing here is an independent confirmation'])
    atomic_json(result, out_path)
    return result
