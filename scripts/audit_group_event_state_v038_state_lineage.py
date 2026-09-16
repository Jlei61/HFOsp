#!/usr/bin/env python3
"""Bind scientific evidence to actual selected observer weights, not patient names."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.topic5_group_event_state.v037.ctssm import DualStreamEventCTSSM, GridBackgroundCTSSM
from src.topic5_group_event_state.v037.contracts import atomic_json


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def module_audit(model, selected):
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    model.load_state_dict(selected)
    params = {}
    for name, value in model.named_parameters():
        delta = value.detach() - initial[name]
        params[name] = {'shape': list(value.shape), 'count': value.numel(),
                        'initial_l2': float(initial[name].norm()),
                        'selected_delta_l2': float(delta.norm()),
                        'selected_delta_max_abs': float(delta.abs().max()),
                        'selected_changed': bool(torch.any(delta != 0))}
    return {'parameters': params, 'trainable_parameter_count': sum(v['count'] for v in params.values()),
            'selected_changed': any(v['selected_changed'] for v in params.values()),
            'all_input_maps_changed': all(v['selected_changed'] for v in params.values()),
            'buffers': {n: v.tolist() for n, v in model.named_buffers()},
            'state_dim': model.state_dim}


def checkpoint_audit(path):
    card = json.loads(path.read_text()); checkpoint = Path(card['checkpoint_path'])
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = saved['config']; family = path.parts[-4]; seed = int(card['seed'])
    event_weights = saved['event_observer' if family == 'dual' else 'observer']
    burden = event_weights['burden_input.weight'].shape[1]
    grammar = event_weights['grammar_input.weight'].shape[1]
    kwargs = dict(taus_seconds=cfg['taus_seconds'], burden_channels_per_tau=cfg['burden_channels_per_tau'],
                  grammar_channels_per_tau=cfg['grammar_channels_per_tau'])
    torch.manual_seed(seed)
    event = DualStreamEventCTSSM(burden, grammar, **kwargs)
    modules = {'event': module_audit(event, event_weights)}
    if family == 'dual':
        # Match the original event -> random event -> background constructor order.
        DualStreamEventCTSSM(burden, grammar, **kwargs)
        bg = GridBackgroundCTSSM(saved['background_observer']['input.weight'].shape[1],
                                taus_seconds=cfg['taus_seconds'], channels_per_tau=cfg['background_channels_per_tau'])
        modules['background'] = module_audit(bg, saved['background_observer'])
    readout = {name: {'shape': list(value.shape), 'count': value.numel(),
                      'l2': float(value.norm()), 'nonzero': bool(torch.any(value != 0))}
               for name, value in saved['readout'].items()}
    source = Path(card['trajectory_path'])
    with np.load(source, allow_pickle=False) as values:
        phase = values['phase'].astype(str)
        data_support = {label: int(np.sum(phase == label)) for label in np.unique(phase)}
    event_changed = modules['event']['selected_changed']
    bg_changed = modules.get('background', {}).get('selected_changed', False)
    state_label = ('learned_event_and_background' if event_changed and bg_changed else
                   'learned_event' if event_changed else 'learned_background_only' if bg_changed else
                   'initialized_observer_features')
    return {'subject': card['subject'], 'seed': seed, 'family': family, 'scope': path.parts[-5],
            'source_card': str(path), 'source_card_sha256': sha(path),
            'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint),
            'trajectory': str(source), 'trajectory_sha256': sha(source),
            'modules': modules, 'readout_parameters': readout,
            'readout_parameter_count': sum(v['count'] for v in readout.values()),
            'selected_state_class': state_label, 'config': cfg, 'original_stages': card['stages'],
            'anchors_by_phase': data_support,
            'batch_contract': 'one full FIT objective per optimizer step; recurrence carries within recording segments',
            'normalization_contract': 'FIT robust centres/scales; saved q/Bmark/background scalers; no BatchNorm or LayerNorm in observer',
            'transition_contract': 'fixed exponential time constants; learned linear input maps; normalized grammar readout is nonlinear',
            'original_full_preprocessing_replay_proven': False,
            'weight_audit_does_not_prove_optimization_sufficiency': True}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    paths = sorted(args.base.glob('h1_*/*/*/seed*/card.json'))
    paths = [p for p in paths if p.parts[-5] in ('h1_long_8h', 'h1_medium_2h')]
    rows = [checkpoint_audit(p) for p in paths]
    assert len(rows) == 165, len(rows)
    by_hash = {row['checkpoint_sha256']: row for row in rows}
    downstream = []
    for path in sorted(args.base.glob('h2a_*/*/*/seed*/card.json')):
        card = json.loads(path.read_text())
        if path.parts[-5] not in ('h2a_long_8h', 'h2a_medium_2h'):
            continue
        checkpoint = card.get('state_provenance', {}).get('checkpoint')
        if checkpoint is None:
            downstream.append({'kind': 'h2a', 'card': str(path), 'status': card.get('status'),
                               'reason': 'no fitted state checkpoint'})
            continue
        digest = sha(checkpoint); parent = by_hash.get(digest)
        if parent is None:
            raise ValueError(f'unknown H2a upstream checkpoint: {checkpoint}')
        downstream.append({'kind': 'h2a', 'card': str(path), 'card_sha256': sha(path),
                           'checkpoint': checkpoint, 'checkpoint_sha256': digest,
                           'upstream_card': None if parent is None else parent['source_card'],
                           'selected_state_class': None if parent is None else parent['selected_state_class']})
    for path in sorted(args.base.glob('h2b_*/features/*/seed*/freeze_card.json')):
        if path.parts[-5] not in ('h2b_long_8h', 'h2b_medium_2h'):
            continue
        card = json.loads(path.read_text()); bindings = {}
        for prefix in ('source', 'event_source', 'grid_source'):
            checkpoint = card.get(prefix + '_checkpoint')
            if checkpoint:
                digest = sha(checkpoint)
                if digest != card[prefix + '_checkpoint_sha256']:
                    raise ValueError(f'frozen upstream hash mismatch: {path} {prefix}')
                parent = by_hash.get(digest)
                if parent is None:
                    raise ValueError(f'unknown upstream checkpoint: {checkpoint}')
                bindings[prefix] = {'checkpoint_sha256': digest, 'upstream_card': parent['source_card'],
                                     'selected_state_class': parent['selected_state_class']}
        downstream.append({'kind': 'h2b', 'card': str(path), 'card_sha256': sha(path),
                           'feature_sha256': card['feature_sha256'], 'bindings': bindings})
    payload = {'status': 'COMPLETE', 'h1_cards': rows, 'downstream_bindings': downstream,
               'state_classes': dict(Counter(row['selected_state_class'] for row in rows)),
               'code_sha256': {str(p.relative_to(ROOT)): sha(p) for p in
                               (Path(__file__), ROOT / 'src/topic5_group_event_state/v037/ctssm.py')},
               'development_targets_read': False, 'sealed_partition_opened': False,
               'seizure_outcomes_read': False}
    atomic_json(args.output, payload)
    print(json.dumps({'h1_cards': len(rows), 'downstream_bindings': len(downstream),
                      'state_classes': payload['state_classes']}))


if __name__ == '__main__':
    main()
