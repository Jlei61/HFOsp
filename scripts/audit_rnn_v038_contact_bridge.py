#!/usr/bin/env python3
"""Frozen v0.3.8 bridge audit. Writes only to a separate audit output root."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from contact_bridge_metrics import (exact_set_scores, event_mean, fit_branch_dictionary,
                                    matched_donors, paired_summary, prefix_key)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def run(args):
    started = time.time()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    sys.path.insert(0, str(args.snapshot))
    from src.topic5_group_event_state.v037.ctssm import DualStreamEventCTSSM
    from src.topic5_group_event_state.v037.h1_train import GridEventStateComputer
    from src.topic5_group_event_state.v037.h2a import _load_eval_events, _phase, _state_at_events, _bmark_at_events, RANK_DATA_ROOT
    from src.topic5_group_event_state.v034_spatial_state.we_decoder import load_frozen_decoder, per_event_scores
    from src.topic5_group_event_state.v035.stepwise_decoder import StepwiseAdapterConfig, StepwiseConditionedDecoder
    from src.topic5_wiring_economy_rnn import build_event_tensors

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'card.json').exists():
        raise FileExistsError(output / 'card.json')
    card = json.loads(args.source_card.read_text())
    hp = Path(card['state_provenance']['checkpoint'])
    family = card['state_provenance']['family']
    subject, seed = card['subject'], int(card['state_seed'])
    ap = Path(card['adapter_checkpoint'])
    dp = Path(card['decoder_provenance']['checkpoint'])
    hashes = {str(p): sha(p) for p in (args.source_card, hp, ap, dp, args.replay,
                                      hp.with_name('trajectory_and_targets.npz'))}
    hashes[str(RANK_DATA_ROOT / f'{subject}.npz')] = sha(RANK_DATA_ROOT / f'{subject}.npz')
    code_hashes = {str(p): sha(p) for p in (Path(__file__), Path(__file__).with_name('contact_bridge_metrics.py'))}
    saved = torch.load(hp, map_location='cpu', weights_only=False)
    adapter = torch.load(ap, map_location='cpu', weights_only=False)
    replay = torch.load(args.replay, map_location='cpu', weights_only=False)
    if replay['source_hashes'][str(hp)] != hashes[str(hp)]:
        raise ValueError('replay is not bound to the selected upstream checkpoint')
    data = replay['data']
    cfg = saved['config']
    kwargs = dict(taus_seconds=cfg['taus_seconds'], burden_channels_per_tau=cfg['burden_channels_per_tau'],
                  grammar_channels_per_tau=cfg['grammar_channels_per_tau'])
    torch.manual_seed(seed)
    observer = DualStreamEventCTSSM(data.burden_mark.shape[1], data.grammar_mark.shape[1], **kwargs)
    initial = {k: v.clone() for k, v in observer.state_dict().items()}
    selected = saved['event_observer' if family == 'dual' else 'observer']
    event_update = sum(float((initial[k] - v).square().sum()) for k, v in selected.items()
                       if k.endswith('weight')) ** 0.5
    observer.load_state_dict(selected)
    observer.eval()
    bundle = load_frozen_decoder(dp.parent, Path(card['decoder_provenance']['cache']), device=torch.device('cpu'))
    for p in Path(card['decoder_provenance']['cache']).iterdir():
        if p.is_file():
            hashes[str(p)] = sha(p)
    times, ranks, segments = _load_eval_events(subject, data, bundle.contact_names)
    phase = _phase(times, dict(data.rate.phase_boundaries))
    fit = np.flatnonzero(phase == 'FIT')
    selection = np.flatnonzero(phase == 'SELECTION')
    for ph in ('FIT', 'INNER', 'SELECTION'):
        if int((phase == ph).sum()) != card['n_events'][ph]:
            raise ValueError('event denominator drift: ' + ph)

    def state_at(data_value):
        if family == 'grid':
            return GridEventStateComputer(data_value, observer, torch.device('cpu'),
                grid_seconds=float(saved.get('grid_seconds', 300)), query_time=times, query_segment=segments)()
        return _state_at_events(data_value, observer, times, segments, torch.device('cpu'))

    raw_event = state_at(data)
    event_width = raw_event.shape[1]
    observer.load_state_dict(initial)
    raw_initial = state_at(data)
    observer.load_state_dict(selected)
    # Controlled mark scrambling at fixed times: donor labels never leave a phase
    # or recording segment. This is a noncausal diagnostic null, not a predictor.
    event_phase = _phase(data.event_time, dict(data.rate.phase_boundaries))
    permutation = np.arange(len(data.event_time))
    rng = np.random.default_rng(seed + 901)
    for seg in np.unique(data.event_segment):
        for ph in np.unique(event_phase):
            rr = np.flatnonzero((data.event_segment == seg) & (event_phase == ph))
            permutation[rr] = rng.permutation(rr)
    # Old pickled dataclasses predate optional provenance fields. Shallow copy
    # preserves the actual archived fields without fabricating missing history.
    shuffled_data = copy.copy(data)
    object.__setattr__(shuffled_data, 'burden_mark', data.burden_mark[permutation])
    object.__setattr__(shuffled_data, 'grammar_mark', data.grammar_mark[permutation])
    raw_scramble = state_at(shuffled_data)
    raw = raw_event
    if family == 'dual':
        with np.load(hp.with_name('trajectory_and_targets.npz'), allow_pickle=False) as z:
            gt, gs = z['anchor_time'], z['segment']
            bg = np.concatenate((z['background_state'], z['background_current']), axis=1)
        bg_event = np.zeros((len(times), bg.shape[1]), dtype=np.float32)
        for seg in np.unique(segments):
            er, gr = np.flatnonzero(segments == seg), np.flatnonzero(gs == seg)
            if not len(gr):
                continue
            pos = np.searchsorted(gt[gr], times[er], side='left') - 1
            ok = pos >= 0
            bg_event[er[ok]] = bg[gr[pos[ok]]]
        raw = torch.cat((raw, torch.as_tensor(bg_event)), dim=1)
    normalize = lambda v: torch.as_tensor(np.clip((np.asarray(v) - adapter['state_centre']) / adapter['state_scale'], -12, 12).astype(np.float32))
    context = normalize(raw)
    constant = context[fit].mean(0, keepdim=True).expand_as(context)
    bmark = _bmark_at_events(data, times, segments, tuple(cfg['taus_seconds']),
                            np.asarray(saved['bmark_centre']), np.asarray(saved['bmark_scale']), torch.device('cpu'))
    ac = card['config']
    model = StepwiseConditionedDecoder(bundle.model, StepwiseAdapterConfig(context_dim=context.shape[1], rank=ac['modulation_rank']))
    baseline = StepwiseConditionedDecoder(bundle.model, StepwiseAdapterConfig(context_dim=bmark.shape[1], rank=ac['modulation_rank']))
    model.static.load_state_dict(adapter['static_adapter'])
    model.dynamic.load_state_dict(adapter['dynamic_adapter'])
    baseline.static.load_state_dict(adapter['static_adapter'])
    baseline.dynamic.load_state_dict(adapter['bmark_dynamic_adapter'])
    model.eval(); baseline.eval()
    tensors = build_event_tensors(ranks)

    def predict(which, features, use_dynamic=True):
        ll, ss = [], []
        for start in range(0, len(selection), 512):
            ix = selection[start:start + 512]
            l, s = which(tensors['x'][ix], tensors['recruited'][ix], tensors['valid'][ix],
                         None if features is None else features[ix], use_static=True, use_dynamic=use_dynamic)
            ll.append(l.numpy()); ss.append(s.numpy())
        return np.concatenate(ll), np.concatenate(ss)

    arms = {'correct': (model, context), 'constant': (model, constant), 'B_mark': (baseline, bmark),
            'static': (model, None)}
    ablated = context.clone(); ablated[:, :event_width] = constant[:, :event_width]
    arms['event_constant'] = (model, ablated)
    if family == 'dual':
        ablated = context.clone(); ablated[:, event_width:] = constant[:, event_width:]
        arms['background_constant'] = (model, ablated)
    for label, value in [('event_initial', raw_initial), ('event_mark_scramble', raw_scramble)]:
        raw_changed = raw.clone(); raw_changed[:, :event_width] = value
        arms[label] = (model, normalize(raw_changed))
    donors = matched_donors(ranks, times, segments, selection, include_k=False,
                            minimum_events=ac['minimum_same_prefix_events'], minimum_seconds=ac['minimum_shift_seconds'])
    donors_k = matched_donors(ranks, times, segments, selection, include_k=True,
                              minimum_events=ac['minimum_same_prefix_events'], minimum_seconds=ac['minimum_shift_seconds'])
    for label, dd, branch in [('shift', donors, 'all'), ('shift_k', donors_k, 'all'),
                              ('event_shift_k', donors_k, 'event'), ('background_shift_k', donors_k, 'background')]:
        if branch == 'background' and family != 'dual':
            continue
        changed = context.clone(); rr = np.flatnonzero(dd >= 0)
        col = slice(None) if branch == 'all' else slice(0, event_width) if branch == 'event' else slice(event_width, None)
        changed[rr, col] = context[dd[rr], col]
        arms[label] = (model, changed)
    branch_dict = fit_branch_dictionary(ranks, fit, minimum=5)
    branch = np.array([prefix_key(ranks[i], include_k=True) in branch_dict for i in selection])
    batch = {k: v[selection] for k, v in tensors.items()}
    valid = batch['valid'].numpy()
    position = np.arange(valid.shape[1])[None, :]
    scores, predictions, parity = {}, {}, {}
    for label, (which, feature) in arms.items():
        logits, stop = predict(which, feature, label != 'static')
        predictions[label + '_logits'] = logits
        predictions[label + '_stop_logits'] = stop
        old = per_event_scores(torch.from_numpy(logits), torch.from_numpy(stop), batch)
        parent_arm = {'correct': 'correct_state', 'constant': 'constant_state', 'B_mark': 'B_mark_context', 'static': 'static_recalibration'}.get(label)
        if parent_arm:
            parity[label] = {k: float(abs(v.mean().item() - card['selection_arms'][parent_arm][k])) for k, v in old.items()}
        suffix = per_event_scores(torch.from_numpy(logits), torch.from_numpy(stop), batch, observed_prefix_groups=2)
        exact = exact_set_scores(logits, batch['target'].numpy(), batch['available'].numpy())
        masks = {'all_identity': valid, 'suffix_identity': valid & (position >= 1),
                 'next_after_two': valid & (position == 1),
                 'next_two_teacher_forced': valid & (position >= 1) & (position <= 2),
                 'fit_branch_next': valid & (position == 1) & branch[:, None]}
        score = {name: v.numpy() for name, v in old.items() if name != 'n_predict'}
        score.update({'suffix_' + name: v.numpy() for name, v in suffix.items() if name != 'n_predict'})
        for name, mask in masks.items():
            for metric in ('set_nll', 'set_accuracy'):
                score[name + '_' + metric] = event_mean(exact[metric], mask)
        score['prefix_stop_bce'] = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.from_numpy(stop[:, 1]), batch['is_last'][:, 1].float(), reduction='none').numpy()
        scores[label] = score
    parity_ok = all(value <= 5e-4 + 2e-4 * abs(float(card['selection_arms'][
        {'correct': 'correct_state', 'constant': 'constant_state', 'B_mark': 'B_mark_context', 'static': 'static_recalibration'}[label]][k]))
        for label, fields in parity.items() for k, value in fields.items())
    # Identical evaluation support for all controls, including static and B_mark.
    supports = {'all': np.ones(len(selection), dtype=bool), 'same_prefix_shift': donors[selection] >= 0,
                'same_prefix_k_shift': donors_k[selection] >= 0,
                'fit_branch_same_prefix_k_shift': (donors_k[selection] >= 0) & branch}
    comparisons = {}
    for support, mask in supports.items():
        comparisons[support] = {}
        for label in scores:
            if label == 'correct':
                continue
            # Never count unchanged, ineligible rows as wrong-time observations.
            if 'shift' in label and support == 'all':
                continue
            if label.endswith('_k') and support == 'same_prefix_shift':
                continue
            comparisons[support][label] = {}
            for endpoint, value in scores['correct'].items():
                if endpoint.endswith('accuracy'):
                    correct, control = -value, -scores[label][endpoint]
                else:
                    correct, control = value, scores[label][endpoint]
                comparisons[support][label][endpoint] = paired_summary(correct[mask], control[mask], times[selection][mask], segments[selection][mask])
    artifact = {**predictions, 'event_rows': selection, 'event_time': times[selection], 'segment': segments[selection],
                'ranks': ranks[selection], 'available': batch['available'].numpy(), 'target': batch['target'].numpy(),
                'valid': valid, 'is_last': batch['is_last'].numpy(), 'donor_rows': donors[selection],
                'donor_k_rows': donors_k[selection], 'fit_branch': branch,
                'mark_scramble_donor_rows': permutation, 'event_history_time': data.event_time}
    artifact.update({f'{arm}__{endpoint}': value for arm, endpoints in scores.items() for endpoint, value in endpoints.items()})
    np.savez_compressed(output / 'predictions_and_scores.npz', **artifact)
    for p, digest in hashes.items():
        if sha(p) != digest:
            raise ValueError('input mutated during audit: ' + p)
    result = {'status': 'COMPLETE' if parity_ok else 'SOURCE_PARITY_FAILED', 'subject': subject, 'seed': seed, 'family': family,
              'source_card': str(args.source_card), 'source_hashes': hashes, 'code_hashes': code_hashes, 'snapshot': str(args.snapshot),
              'parent_parity': parity, 'parent_parity_ok': parity_ok, 'event_weight_update_l2': event_update,
              'n_events': card['n_events'], 'event_state_width': event_width, 'state_width': context.shape[1],
              'n_fit_branch_strata': len(branch_dict), 'n_selection_fit_branch': int(branch.sum()),
              'fit_branch_dictionary': [{'prefix_k': str(k), **v} for k, v in branch_dict.items()],
              'support_counts': {k: int(v.sum()) for k, v in supports.items()}, 'comparisons': comparisons,
              'predictions_sha256': sha(output / 'predictions_and_scores.npz'),
              'elapsed_seconds': time.time() - started,
              'claim_tier': 'retrospective_existing_selection_frozen_readout_diagnostic',
              'all_weights_frozen': True, 'new_training_performed': False,
              'ablation_is_refitted_comparator': False, 'causal_measurement_qualified': False,
              'development_partition_opened': False, 'sealed_partition_opened': False,
              'multi_step_is_teacher_forced_not_free_rollout': True,
              'minimum_positive_gain_tolerance': 1e-6,
              'physical_bins_are_not_independent_patients_or_independent_histories': True}
    write_json(output / 'card.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'subject', 'seed', 'family', 'parent_parity_ok', 'elapsed_seconds')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--source-card', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=1)
    run(parser.parse_args())
