#!/usr/bin/env python3
"""Refit only the frozen-random control head with a registered larger budget."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_group_event_state_v038_dual_credit import file_hash
from scripts.audit_group_event_state_v038_dual_random_background import _fit_only_repertoire_alignment, _parent_parity_audit
from src.topic5_group_event_state.v037.contracts import atomic_json
from src.topic5_group_event_state.v037.ctssm import DualStreamEventCTSSM
from src.topic5_group_event_state.v037.h1_data import build_h1_subject_data
from src.topic5_group_event_state.v037.h1_train import (
    H1TrainConfig, NestedH1Readout, EventStateComputer, GridEventStateComputer,
    _target_bundle, _train_stage, _selection_score,
)


def repair(source_card: Path, output: Path, device: torch.device, maximum_steps: int = 2700,
           preflight_only: bool = False):
    started = time.time(); source = source_card.parent
    original = json.loads(source_card.read_text())
    saved = torch.load(source / 'checkpoint.pt', map_location='cpu', weights_only=False)
    hashes = {str(source / name): file_hash(source / name)
              for name in ('card.json', 'checkpoint.pt', 'trajectory_and_targets.npz')}
    cfg = H1TrainConfig(**saved['config']); family = saved['observer_mode']
    if family not in ('event', 'grid') or maximum_steps != 2700:
        raise ValueError('registered budget repair is event/grid, exactly 2700 maximum steps')
    seed = int(original['seed']); horizons = tuple(original['horizons_seconds'])
    data = build_h1_subject_data(original['subject'], seed=seed, horizons_seconds=horizons)
    with np.load(source / 'trajectory_and_targets.npz', allow_pickle=False) as archive:
        trajectory = {k: archive[k] for k in archive.files}
    for name, value in [('anchor_time', data.rate.anchor_time), ('phase', data.rate.phase),
                         ('target_count', data.rate.target_count), ('target_valid', data.rate.target_valid),
                         ('target_exposure_seconds', data.rate.target_exposure_seconds)]:
        if not np.array_equal(trajectory[name], value):
            raise ValueError(f'input alignment drift: {name}')
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
    fit_np = np.flatnonzero(data.rate.phase == 'FIT')
    fit, inner, selection = [torch.as_tensor(np.flatnonzero(data.rate.phase == name), device=device)
                              for name in ('FIT', 'INNER', 'SELECTION')]
    selection_np = selection.cpu().numpy()
    q = tensor(np.clip((data.rate.q_raw - saved['q_centre']) / saved['q_scale'], -12, 12))
    bm = tensor(trajectory['fixed_mark_state']); learned = tensor(trajectory['learned_state'])
    args = dict(taus_seconds=cfg.taus_seconds, burden_channels_per_tau=cfg.burden_channels_per_tau,
                grammar_channels_per_tau=cfg.grammar_channels_per_tau)
    torch.manual_seed(seed)
    DualStreamEventCTSSM(data.burden_mark.shape[1], data.grammar_mark.shape[1], **args)
    random = DualStreamEventCTSSM(data.burden_mark.shape[1], data.grammar_mark.shape[1], **args).to(device)
    for parameter in random.parameters():
        parameter.requires_grad_(False)
    computer = (EventStateComputer(data, random, device) if family == 'event' else
                GridEventStateComputer(data, random, device, grid_seconds=float(saved['grid_seconds'])))
    with torch.no_grad():
        raw_random = computer()
        random_state = torch.clamp((raw_random - tensor(saved['random_centre'])) / tensor(saved['random_scale']), -12, 12)
    bd = len(cfg.taus_seconds) * data.burden_mark.shape[1]
    sd = len(cfg.taus_seconds) * cfg.burden_channels_per_tau
    readout = NestedH1Readout(q.shape[1], bd, bm.shape[1] - bd, sd, learned.shape[1] - sd,
                              saved['widths'], len(horizons)).to(device)
    readout.load_state_dict(saved['readout'])
    target, valid, scales = _target_bundle(data, device)
    exposure = tensor(data.rate.target_exposure_seconds)
    model_name = 'S_event' if family == 'event' else 'S_grid'
    with torch.no_grad():
        target, permutation, _ = _fit_only_repertoire_alignment(target, valid, readout.predict(q, bmark=bm), fit)
        initial_predictions = {'B_mark': readout.predict(q, bmark=bm),
                               'random_frozen': readout.predict(q, bmark=bm, random_state=random_state),
                               model_name: readout.predict(q, bmark=bm, state=learned)}
        parity = {}
        for name, pred in initial_predictions.items():
            score = _selection_score(pred, target, valid, exposure, readout.log_dispersion, selection, selection_np, horizons)
            try:
                parity[name] = _parent_parity_audit(score, original['selection_scores'][name], horizons)
            except ValueError as error:
                raise ValueError(f'{name}: {error}') from error
        if any(not row['all_endpoint_total_auditable'] for row in parity.values()):
            result = {**original, 'status': 'NOT_ESTIMABLE', 'repair_source_card': str(source_card),
                      'repair_source_sha256': hashes[str(source_card)], 'repair_parent_score_parity': parity,
                      'repair_reason': 'source target reconstruction prevents identical all-endpoint random-control refit',
                      'repair_model_weights_updated': False}
            atomic_json(output, result)
            return result
    if preflight_only:
        result = {'status': 'COMPLETE', 'subject': original['subject'], 'seed': seed,
                  'family': family, 'parent_parity': parity, 'training_run': False,
                  'development_targets_read': False, 'sealed_partition_opened': False, 'seizure_targets_read': False}
        atomic_json(output, result)
        return result
    # Original random residual heads start at zero. Refit from the same start,
    # retaining the original LR, optimizer, regularization and INNER patience.
    for layer in readout.random.values():
        with torch.no_grad():
            layer.weight.zero_()
    extended = replace(cfg, max_steps_random=maximum_steps)
    stage = _train_stage(stage='random', readout=readout, q=q, bmark=bm, state_computer=None,
                         fixed_random=random_state, target=target, valid=valid, exposure=exposure,
                         fit_rows=fit, inner_rows=inner, config=extended)
    for name, value in readout.state_dict().items():
        if not name.startswith('random.') and not torch.equal(value.cpu(), saved['readout'][name]):
            raise ValueError(f'non-control parameters changed: {name}')
    with torch.no_grad():
        random_score = _selection_score(readout.predict(q, bmark=bm, random_state=random_state),
                                         target, valid, exposure, readout.log_dispersion, selection, selection_np, horizons)
    output.parent.mkdir(parents=True, exist_ok=True)
    weight_path = output.parent / 'random_control_checkpoint.pt'
    torch.save({'random_observer': random.state_dict(), 'random_readout': readout.random.state_dict(),
                'random_centre': saved['random_centre'], 'random_scale': saved['random_scale'],
                'source_hashes': hashes, 'config': asdict(extended), 'target_scales': scales,
                'repertoire_permutation_fit_only': permutation}, weight_path)
    result = dict(original)
    result['stages'] = {**original['stages'], 'random': stage}
    result['selection_scores'] = {**original['selection_scores'], 'random_frozen': random_score}
    result.update(status='COMPLETE', repair_source_card=str(source_card), repair_source_sha256=hashes[str(source_card)],
                  repair_input_hashes=hashes, repair_parent_score_parity=parity,
                  repair_control_checkpoint=str(weight_path), repair_control_checkpoint_sha256=file_hash(weight_path),
                  repair_config=asdict(extended), repair_original_random_stage=original['stages']['random'],
                  repair_observer_or_scientific_readout_updated=False,
                  repair_description='same frozen random encoder and parent; random head alone refit from zero with 2700-step cap',
                  repair_elapsed_seconds=time.time() - started)
    for path, digest in hashes.items():
        if file_hash(Path(path)) != digest:
            raise ValueError(f'source changed during control repair: {path}')
    atomic_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-card', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = repair(args.source_card, args.output, torch.device(args.device), preflight_only=args.preflight_only)
    print(json.dumps({k: result.get(k) for k in ('status', 'subject', 'seed', 'repair_elapsed_seconds')}), flush=True)


if __name__ == '__main__':
    main()
