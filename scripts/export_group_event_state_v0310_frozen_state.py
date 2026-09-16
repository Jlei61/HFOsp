#!/usr/bin/env python3
"""Export frozen anchor states from a v0.3.10 cell, in the v0.3.9 transfer contract.

No new fitting happens here and the upstream checkpoint is never touched, so
the same frozen state can feed the contact-identity and fine-expression probes.
Histories for horizons the bundle does not store are rebuilt from the replay
blocks, exactly as during training.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src.topic5_group_event_state.v039.transition import EventTransition, FutureReadout
from src.topic5_group_event_state.v0310.history import history_matrix
from src.topic5_group_event_state.v035.contracts import atomic_json


def export(source, output, device='cpu'):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    card = json.loads(Path(source).read_text())
    if card.get('schema') != 'v0310_human_cell_v1':
        raise ValueError('Not a v0.3.10 cell card')
    cfg = card['config']
    if sha(card['checkpoint']) != card['checkpoint_sha256'] or sha(card['data_path']) != card['data_sha256']:
        raise ValueError('Frozen upstream changed')
    state = torch.load(card['checkpoint'], map_location='cpu', weights_only=False)
    data = torch.load(card['data_path'], map_location='cpu', weights_only=False)
    width, rank = cfg['state_width'], cfg['transition_rank']
    models = {'state': EventTransition(data['input_dim'], cfg['family'], width=width, rank=rank, seed=cfg['seed']),
              'initialized': EventTransition(data['input_dim'], cfg['family'], width=width, rank=rank, seed=cfg['seed']),
              'fixed_history': EventTransition(data['input_dim'], 'F')}
    models['state'].load_state_dict(state['observer'])
    readout = FutureReadout(models['state'].width, data['n_recruitment'], 0, hidden=cfg['readout_hidden'])
    readout.load_state_dict(state['residual']); readout.to(device).requires_grad_(False)
    for model in models.values():
        model.to(device).requires_grad_(False)
    horizon = history_matrix(data, cfg['history_hours'])
    x_all = torch.from_numpy(horizon['x']); dt_all = torch.from_numpy(horizon['dt'])
    out = {k: [] for k in models}; out['functional'] = []
    with torch.no_grad():
        for start in range(0, len(data['samples']), 256):
            x = x_all[start:start + 256].to(device); dt = dt_all[start:start + 256].to(device)
            for name, model in models.items():
                value = model.scan(x, dt, checkpoint_chunk=0)
                out[name].append(value.cpu().numpy())
                if name == 'state':
                    mu, logits = readout(value, value.new_empty((len(value), 0)), 2.)
                    functional = (mu[:, None] if cfg['view'] == 'count' else logits if cfg['view'] == 'recruitment'
                                  else torch.cat((mu[:, None], logits), dim=-1))
                    out['functional'].append(functional.cpu().numpy())
    arrays = {k: np.concatenate(v) for k, v in out.items()}
    arrays['background'] = np.asarray([s['context'] for s in data['samples']], np.float32)
    arrays['anchor_time'] = np.asarray([s['anchor'] for s in data['samples']])
    arrays['phase'] = np.asarray([s['phase'] for s in data['samples']])
    arrays['past_coverage'] = horizon['coverage'].astype(np.float32)
    if any(not np.isfinite(v).all() for k, v in arrays.items() if k != 'phase'):
        raise FloatingPointError('Nonfinite frozen export')
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    if sha(card['checkpoint']) != card['checkpoint_sha256']:
        raise ValueError('Export modified the frozen observer')
    meta = dict(status='COMPLETE', schema='v0310_frozen_export_v1', subject=card['subject'],
                family=card['family'], recipe=card['recipe'], history_hours=cfg['history_hours'],
                view=cfg['view'], source_mode=cfg['source_mode'], seed=cfg['seed'],
                state_width=card['state_width'], readout_hidden=cfg['readout_hidden'],
                source_card=str(source), source_card_sha256=sha(source),
                checkpoint_sha256=card['checkpoint_sha256'], human_data_sha256=card['data_sha256'],
                export=str(output), export_sha256=sha(output),
                shapes={k: list(v.shape) for k, v in arrays.items()},
                upstream_stop_reason={k: v['stop_reason'] for k, v in card['stages'].items()},
                upstream_training_sufficiency=card['training_sufficiency_vector']['verdict'],
                history_rebuilt_from_replay=bool(horizon['rebuilt_from_replay']),
                functional_definition='trained-view residual readout at the 2-hour lead; excludes the '
                                      'background prediction. The raw latent transfer is a separate estimand.',
                source_hashes={str(p.relative_to(ROOT)): sha(p) for p in
                               [Path(__file__), ROOT / 'src/topic5_group_event_state/v039/transition.py',
                                ROOT / 'src/topic5_group_event_state/v0310/history.py']},
                development_targets_read=False, sealed_partition_opened=False, seizure_targets_read=False)
    atomic_json(Path(output).with_suffix('.json'), meta)
    print(json.dumps(dict(status='COMPLETE', subject=card['subject'], family=card['family'],
                          recipe=card['recipe'], shapes=meta['shapes']['state'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    a = p.parse_args(); export(a.source, a.output, a.device)
