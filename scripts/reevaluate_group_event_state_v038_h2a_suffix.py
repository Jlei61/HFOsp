#!/usr/bin/env python3
"""Rescore saved H2a adapters on a strict, common-support unknown suffix."""
from __future__ import annotations

import argparse
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
from src.topic5_group_event_state.v037.contracts import atomic_json
from src.topic5_group_event_state.v037.ctssm import DualStreamEventCTSSM
from src.topic5_group_event_state.v037.h1_data import build_h1_subject_data
from src.topic5_group_event_state.v037.h1_train import H1TrainConfig, GridEventStateComputer
from src.topic5_group_event_state.v037.h2a import (
    H2ATrainConfig, _load_eval_events, _phase, _state_at_events, _bmark_at_events,
    _same_prefix_shift_context, _mean_scores, _score_batches,
)
from src.topic5_group_event_state.v034_spatial_state.we_decoder import load_frozen_decoder
from src.topic5_group_event_state.v035.stepwise_decoder import StepwiseAdapterConfig, StepwiseConditionedDecoder
from src.topic5_wiring_economy_rnn import build_event_tensors


def evaluate(source_card: Path, output: Path, device: torch.device):
    started = time.time()
    original = json.loads(source_card.read_text())
    original_hash = file_hash(source_card)
    if original.get('status') == 'NOT_ESTIMABLE':
        result = {**original, 'repair_source_card': str(source_card), 'repair_source_sha256': original_hash,
                  'repair_status': 'SOURCE_NOT_ESTIMABLE_NO_SUFFIX_EVALUATION'}
        atomic_json(output, result)
        return result
    seed = int(original['state_seed']); subject = original['subject']
    family = original['state_provenance']['family']
    cfg = H2ATrainConfig(**original['config'])
    hp = Path(original['state_provenance']['checkpoint'])
    ap = Path(original['adapter_checkpoint'])
    dp = Path(original['decoder_provenance']['checkpoint'])
    hcard = json.loads(hp.with_name('card.json').read_text())
    source_paths = [source_card, ap, hp, hp.with_name('card.json'), hp.with_name('trajectory_and_targets.npz'), dp]
    hashes = {str(p): file_hash(p) for p in source_paths}
    saved = torch.load(hp, map_location='cpu', weights_only=False)
    adapter = torch.load(ap, map_location='cpu', weights_only=False)
    hc = H1TrainConfig(**{k: v for k, v in saved['config'].items() if k in H1TrainConfig.__dataclass_fields__})
    data = build_h1_subject_data(subject, seed=seed, horizons_seconds=tuple(hcard['horizons_seconds']))
    observer = DualStreamEventCTSSM(data.burden_mark.shape[1], data.grammar_mark.shape[1],
                                    taus_seconds=hc.taus_seconds,
                                    burden_channels_per_tau=hc.burden_channels_per_tau,
                                    grammar_channels_per_tau=hc.grammar_channels_per_tau).to(device)
    observer.load_state_dict(saved['event_observer'] if family == 'dual' else saved['observer'])
    observer.eval()
    for parameter in observer.parameters():
        parameter.requires_grad_(False)
    bundle = load_frozen_decoder(dp.parent, Path(original['decoder_provenance']['cache']), device=device)
    times, ranks, segment = _load_eval_events(subject, data, bundle.contact_names)
    phase = _phase(times, dict(data.rate.phase_boundaries))
    fit = np.flatnonzero(phase == 'FIT'); selection = np.flatnonzero(phase == 'SELECTION')
    for name in ('FIT', 'INNER', 'SELECTION'):
        if int((phase == name).sum()) != int(original['n_events'][name]):
            raise ValueError(f'H2a event denominator drift: {name}')
    with torch.no_grad():
        state = (GridEventStateComputer(data, observer, device, grid_seconds=float(saved.get('grid_seconds', 300.0)),
                                        query_time=times, query_segment=segment)()
                 if family == 'grid' else _state_at_events(data, observer, times, segment, device))
        bmark = _bmark_at_events(data, times, segment, tuple(hc.taus_seconds),
                                 np.asarray(saved['bmark_centre']), np.asarray(saved['bmark_scale']), device)
        if family == 'dual':
            with np.load(hp.with_name('trajectory_and_targets.npz'), allow_pickle=False) as archive:
                gt, gs = archive['anchor_time'], archive['segment']
                background = np.concatenate((archive['background_state'], archive['background_current']), axis=1)
            bg = np.zeros((times.size, background.shape[1]), dtype=np.float32)
            for seg in np.unique(segment):
                er = np.flatnonzero(segment == seg); gr = np.flatnonzero(gs == seg)
                if not len(gr):
                    continue
                pos = np.searchsorted(gt[gr], times[er], side='left') - 1
                ok = pos >= 0
                bg[er[ok]] = background[gr[pos[ok]]]
            state = torch.cat((state, torch.as_tensor(bg, device=device)), dim=1)
        context = torch.as_tensor(np.clip((state.cpu().numpy() - adapter['state_centre'])
                                          / adapter['state_scale'], -12, 12).astype(np.float32), device=device)
    model = StepwiseConditionedDecoder(bundle.model, StepwiseAdapterConfig(
        context_dim=context.shape[1], rank=cfg.modulation_rank, output_init_std=cfg.state_modulation_init_std)).to(device)
    baseline = StepwiseConditionedDecoder(bundle.model, StepwiseAdapterConfig(
        context_dim=bmark.shape[1], rank=cfg.modulation_rank, output_init_std=cfg.state_modulation_init_std)).to(device)
    model.static.load_state_dict(adapter['static_adapter']); model.dynamic.load_state_dict(adapter['dynamic_adapter'])
    baseline.static.load_state_dict(adapter['static_adapter']); baseline.dynamic.load_state_dict(adapter['bmark_dynamic_adapter'])
    model.eval(); baseline.eval()
    tensors = {k: v.to(device) for k, v in build_event_tensors(ranks).items()}
    constant = context[torch.as_tensor(fit, device=device)].mean(0, keepdim=True).expand_as(context)
    shifted, ok = _same_prefix_shift_context(context, ranks, times, segment, selection,
                                             cfg.minimum_same_prefix_events, cfg.minimum_shift_seconds)
    rows = selection[ok[selection]]
    parity = {}
    with torch.no_grad():
        for label, scorer, feature in [('correct_state', model, context), ('constant_state', model, constant),
                                       ('B_mark_context', baseline, bmark)]:
            score = _mean_scores(scorer, tensors, feature, selection, use_static=True,
                                  use_dynamic=True, batch_size=cfg.batch_size)
            parity[label] = {k: abs(score[k] - float(original['selection_arms'][label][k])) for k in score}
            if any(v > 5e-4 + 2e-4 * abs(float(original['selection_arms'][label][k])) for k, v in parity[label].items()):
                raise ValueError(f'frozen adapter parent scoring mismatch: {label}: {parity[label]}')
        arms = {}; per_event = {}
        if len(rows):
            for label, scorer, feature in [('correct', model, context), ('constant', model, constant),
                                           ('B_mark', baseline, bmark), ('shifted', model, shifted)]:
                values = _score_batches(scorer, tensors, feature, rows, use_static=True,
                                        use_dynamic=True, batch_size=cfg.batch_size, observed_prefix_groups=2)
                arms[label] = {name: float(value.mean()) for name, value in values.items()}
                per_event[label] = {name: value.cpu().tolist() for name, value in values.items()}
    result = dict(original)
    result['primary_contrasts'] = {k: v for k, v in original['primary_contrasts'].items()
                                  if not k.startswith(('same_prefix_', 'correct_time_same_prefix_'))}
    for key, arm in [('same_prefix_gain_over_B_mark', 'B_mark'),
                     ('same_prefix_constant_unexplained_grammar', 'constant'),
                     ('correct_time_same_prefix_grammar', 'shifted')]:
        result['primary_contrasts'][key] = arms[arm]['grammar'] - arms['correct']['grammar'] if arms else None
    # Legacy full-event same-prefix arms remain only in the immutable source.
    result['selection_arms'] = {k: v for k, v in original['selection_arms'].items() if 'same_prefix' not in k}
    result.update(status='COMPLETE', same_prefix_scoring_contract='after_two_observed_groups_v1',
                  same_prefix_comparison_support='identical same-prefix/same-segment/time-shift-eligible events in every arm',
                  suffix_status='ESTIMATED' if arms else 'NOT_ESTIMABLE', suffix_arms=arms,
                  suffix_per_event_scores=per_event,
                  n_same_prefix_shift_paired=int(len(rows)),
                  suffix_event_rows=rows.tolist(), suffix_event_times=times[rows].tolist(),
                  suffix_event_segments=segment[rows].tolist(),
                  repair_source_card=str(source_card), repair_source_sha256=original_hash,
                  repair_input_hashes=hashes, repair_parent_score_parity=parity,
                  repair_model_weights_updated=False,
                  state_selected_at_init=hcard['stages']['event' if family == 'dual' else 'state']['selected_at_init'],
                  claim_tier='repair_reassessment_of_previously_examined_selection',
                  repair_elapsed_seconds=time.time() - started)
    for path, digest in hashes.items():
        if file_hash(Path(path)) != digest:
            raise ValueError(f'input changed during rescore: {path}')
    atomic_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-card', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = evaluate(args.source_card, args.output, torch.device(args.device))
    print(json.dumps({k: result.get(k) for k in ('status', 'subject', 'state_seed', 'suffix_status', 'repair_elapsed_seconds')}), flush=True)


if __name__ == '__main__':
    main()
