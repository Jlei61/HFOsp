#!/usr/bin/env python3
"""Replay every original H1 checkpoint and save a portable diagnostic input bundle."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_group_event_state_v038_dual_credit import file_hash
from scripts.audit_group_event_state_v038_dual_random_background import _fit_only_repertoire_alignment, _parent_parity_audit
from src.topic5_group_event_state.v037.contracts import atomic_json
from src.topic5_group_event_state.v037.ctssm import DualStreamEventCTSSM, GridBackgroundCTSSM
from src.topic5_group_event_state.v037.h1_data import build_h1_subject_data
from src.topic5_group_event_state.v037.h1_train import (EventStateComputer, GridEventStateComputer,
    NestedH1Readout, _target_bundle, _endpoint_losses, _weighted_total, _selection_score, _fixed_features)
from src.topic5_group_event_state.v037.h1_dual_train import BackgroundStateComputer, NestedDualReadout, _background_features


def replay(source_card, output, device, *, data_override=None, verify_preprocessing=False):
    started = time.time(); card = json.loads(source_card.read_text()); source = source_card.parent
    hashes = {str(source / name): file_hash(source / name) for name in ('card.json', 'checkpoint.pt', 'trajectory_and_targets.npz')}
    saved = torch.load(source / 'checkpoint.pt', map_location='cpu', weights_only=False)
    cfg = saved['config']; family = source_card.parts[-4]; seed = int(card['seed'])
    horizons = tuple(card['horizons_seconds'])
    data = data_override if data_override is not None else build_h1_subject_data(card['subject'], seed=seed, horizons_seconds=horizons)
    with np.load(source / 'trajectory_and_targets.npz', allow_pickle=False) as archive:
        trajectory = {name: archive[name] for name in archive.files}
    alignment = {name: (bool(np.array_equal(trajectory[name], value)) if name in trajectory else None) for name, value in
                 [('anchor_time', data.rate.anchor_time), ('segment', data.rate.segment), ('phase', data.rate.phase),
                  ('target_count', data.rate.target_count), ('target_valid', data.rate.target_valid),
                  ('target_exposure_seconds', data.rate.target_exposure_seconds)]}
    if any(value is False for value in alignment.values()):
        raise ValueError(f'anchor/target alignment mismatch: {alignment}')
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    q = tensor(np.clip((data.rate.q_raw - saved['q_centre']) / saved['q_scale'], -12, 12))
    bm = tensor(trajectory['fixed_mark_state'])
    rebuilt_bm = bm
    preprocessing_differences = {}
    if verify_preprocessing:
        with torch.no_grad():
            rebuilt_bm = torch.clamp((_fixed_features(data, device, tuple(cfg['taus_seconds'])) - tensor(saved['bmark_centre'])) / tensor(saved['bmark_scale']), -12, 12)
        preprocessing_differences['fixed_mark_state'] = float((rebuilt_bm - bm).abs().max())
    target, valid, scales = _target_bundle(data, device); exposure = tensor(data.rate.target_exposure_seconds)
    fit_np = np.flatnonzero(data.rate.phase == 'FIT'); selection_np = np.flatnonzero(data.rate.phase == 'SELECTION')
    fit = torch.as_tensor(fit_np, device=device); selection = torch.as_tensor(selection_np, device=device)
    event = DualStreamEventCTSSM(data.burden_mark.shape[1], data.grammar_mark.shape[1],
                                 taus_seconds=cfg['taus_seconds'], burden_channels_per_tau=cfg['burden_channels_per_tau'],
                                 grammar_channels_per_tau=cfg['grammar_channels_per_tau']).to(device)
    event.load_state_dict(saved['event_observer' if family == 'dual' else 'observer'])
    computer = (GridEventStateComputer(data, event, device, grid_seconds=saved['grid_seconds']) if family == 'grid'
                else EventStateComputer(data, event, device))
    es = computer(); bd = len(cfg['taus_seconds']) * data.burden_mark.shape[1]
    ed = len(cfg['taus_seconds']) * cfg['burden_channels_per_tau']
    modules = {'event': event}; frozen_kw = {}; rebuilt_kw = {}
    if family == 'dual':
        bc = tensor(trajectory['background_current']); ba = tensor(trajectory['background_available'])
        old_bc, old_ba = bc, ba
        if verify_preprocessing:
            bc, ba, _audit, _names, _centre, _scale = _background_features(data, device)
            preprocessing_differences['background_current'] = float((bc - old_bc).abs().max())
            preprocessing_differences['background_available'] = float((ba - old_ba).abs().max())
        bg = GridBackgroundCTSSM(bc.shape[1], taus_seconds=cfg['taus_seconds'],
                                 channels_per_tau=cfg['background_channels_per_tau']).to(device)
        bg.load_state_dict(saved['background_observer']); modules['background'] = bg
        bs = BackgroundStateComputer(data, bg, bc, ba, device)()
        readout = NestedDualReadout(q.shape[1], bd, bm.shape[1] - bd, bc.shape[1], bs.shape[1],
                                     ed, es.shape[1] - ed, saved['widths'], len(horizons)).to(device)
        frozen_kw = {'background_current': old_bc, 'background_state': tensor(trajectory['background_state']),
                     'event_state': tensor(trajectory['event_state'])}
        rebuilt_kw = {'background_current': bc, 'background_state': bs, 'event_state': es}
        parent_kw = {'background_current': old_bc}; arm = 'S_dual'
        state_differences = {'background': float((bs - frozen_kw['background_state']).detach().abs().max()),
                             'event': float((es - frozen_kw['event_state']).detach().abs().max())}
    else:
        readout = NestedH1Readout(q.shape[1], bd, bm.shape[1] - bd, ed, es.shape[1] - ed,
                                  saved['widths'], len(horizons)).to(device)
        frozen_kw = {'state': tensor(trajectory['learned_state'])}; rebuilt_kw = {'state': es}
        parent_kw = {}; arm = 'S_event' if family == 'event' else 'S_grid'
        state_differences = {'event': float((es - frozen_kw['state']).detach().abs().max())}
    readout.load_state_dict(saved['readout']); modules['readout'] = readout
    with torch.no_grad():
        target, permutation, _ = _fit_only_repertoire_alignment(target, valid,
                                           readout.predict(q, bmark=bm, **parent_kw), fit)
        scores = {}; parity = {}
        for label, kw in [('archived_state', frozen_kw), ('rebuilt_state', rebuilt_kw)]:
            score = _selection_score(readout.predict(q, bmark=bm if label == 'archived_state' else rebuilt_bm, **kw), target, valid, exposure,
                                       readout.log_dispersion, selection, selection_np, horizons)
            scores[label] = score
            try:
                parity[label] = _parent_parity_audit(score, card['selection_scores'][arm], horizons)
            except ValueError as error:
                parity[label] = {'all_endpoint_total_auditable': False, 'reason': str(error)}
    losses = _endpoint_losses(readout.predict(q, bmark=rebuilt_bm, **rebuilt_kw), target, valid, exposure,
                               readout.log_dispersion, fit)
    total = _weighted_total(losses)
    if not torch.isfinite(total):
        raise FloatingPointError('nonfinite selected-checkpoint FIT loss')
    total.backward()
    gradients = {}
    for group, model in modules.items():
        gradients[group] = {name: {'shape': list(value.shape), 'count': value.numel(),
                                   'gradient_present': value.grad is not None,
                                   'gradient_l2': None if value.grad is None else float(value.grad.norm()),
                                   'gradient_max_abs': None if value.grad is None else float(value.grad.abs().max())}
                            for name, value in model.named_parameters()}
        if any(value.grad is not None and not torch.isfinite(value.grad).all() for value in model.parameters()):
            raise FloatingPointError('nonfinite selected-checkpoint parameter gradient')
    output.parent.mkdir(parents=True, exist_ok=True)
    bundle = output.with_name('replay_inputs.pt')
    torch.save({'data': data, 'trajectory': trajectory, 'q': q.detach().cpu(),
                'target': {k: v.cpu() for k, v in target.items()}, 'valid': {k: v.cpu() for k, v in valid.items()},
                'repertoire_fit_only_permutation': permutation, 'target_scales': scales,
                'source_checkpoint': saved, 'source_hashes': hashes}, bundle)
    for p, digest in hashes.items():
        if file_hash(Path(p)) != digest:
            raise ValueError('source artifact changed during replay')
    result = {'status': 'COMPLETE', 'subject': card['subject'], 'seed': seed, 'family': family,
              'scope': source_card.parts[-5], 'source_card': str(source_card), 'input_hashes': hashes,
              'replay_bundle': str(bundle), 'replay_bundle_sha256': file_hash(bundle),
              'anchor_target_alignment': alignment, 'selected_state_max_abs_difference': state_differences,
              'verified_preprocessing': verify_preprocessing,
              'preprocessing_feature_max_abs_difference': preprocessing_differences,
              'score_parity': parity, 'selection_scores': scores,
              'full_endpoint_replay_qualified': all(row.get('all_endpoint_total_auditable') for row in parity.values())
                                                 and max(state_differences.values()) < 1e-4
                                                 and max(preprocessing_differences.values(), default=0.0) < 1e-4,
              'selected_checkpoint_fit_loss': float(total.detach()),
              'selected_checkpoint_fit_endpoint_losses': {k: float(v.detach()) for k, v in losses.items()},
              'selected_checkpoint_fit_parameter_gradients': gradients,
              'gradient_interpretation': 'gradient at the selected checkpoint; not a historical training trace or proof of optimality',
              'preprocessing_boundary': ('actual pre-80% source rows, dictionary and fitted operators frozen; complete event/Bmark/background features rebuilt and checked numerically'
                                         if verify_preprocessing else 'archived Bmark/background inputs retained; reconstructed event inputs/targets checked numerically; original raw-data preprocessing operators not recovered'),
              'model_weights_updated': False, 'development_targets_read': False,
              'sealed_partition_opened': False, 'seizure_targets_read': False,
              'elapsed_seconds': time.time() - started}
    atomic_json(output, result); return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('--source-card', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--device', default='cuda:0'); args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = replay(args.source_card, args.output, torch.device(args.device))
    print(json.dumps({k: result[k] for k in ('status', 'subject', 'seed', 'family', 'full_endpoint_replay_qualified', 'elapsed_seconds')}), flush=True)


if __name__ == '__main__':
    main()
