#!/usr/bin/env python3
"""Checkpoint-bound dual H1 credit and branch-specific history deletion.

All other readout inputs stay frozen during a branch deletion. This measures
dependence on one observer route, not necessity of history in the entire model
and not an intervention on physiology. No model or recipe is selected here.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_group_event_state_v038_dual_random_background import (
    _fit_only_repertoire_alignment, _parent_parity_audit,
)
from scripts.audit_group_event_state_v038_trained_credit import BINS_HOURS, _bin_name
from src.topic5_group_event_state.v037.contracts import atomic_json
from src.topic5_group_event_state.v037.ctssm import (
    DualStreamEventCTSSM, GridBackgroundCTSSM, dual_stream_features_at_queries,
)
from src.topic5_group_event_state.v037.h1_data import build_h1_subject_data
from src.topic5_group_event_state.v037.h1_dual_train import H1DualTrainConfig, NestedDualReadout
from src.topic5_group_event_state.v037.h1_train import _endpoint_losses, _selection_score, _target_bundle


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def module_change(model, initial: dict) -> dict:
    return {name: {
        'shape': list(value.shape), 'parameters': value.numel(),
        'selected_delta_max_abs': float((value.detach().cpu() - initial[name]).abs().max()),
        'selected_delta_l2': float((value.detach().cpu() - initial[name]).norm()),
    } for name, value in model.named_parameters()}


def event_query(model, times, burden, grammar, query, weight=None):
    return dual_stream_features_at_queries(
        model(times, burden, grammar, event_weight=weight), times, query,
    )[0]


def bin_gradient(age_seconds, gradients):
    norm = np.sqrt(sum(np.sum(g.detach().cpu().numpy() ** 2, axis=1)
                       for g in gradients if g is not None))
    if np.ndim(norm) == 0:
        norm = np.zeros(len(age_seconds))
    result = {}
    for lo, hi in zip(BINS_HOURS[:-1], BINS_HOURS[1:]):
        mask = (age_seconds >= lo * 3600) & (age_seconds < hi * 3600)
        result[_bin_name(lo, hi)] = {
            'observations': int(mask.sum()), 'gradient_mass': float(norm[mask].sum()),
        }
    return result


def audit(subject: str, seed: int, horizon_hours: float, source_root: Path,
          output: Path, device: torch.device, maximum_anchors: int = 8) -> dict:
    started = time.time()
    source = source_root / 'dual' / subject / f'seed{seed}'
    hashes = {name: file_hash(source / name) for name in
              ('card.json', 'checkpoint.pt', 'trajectory_and_targets.npz')}
    card = json.loads((source / 'card.json').read_text())
    saved = torch.load(source / 'checkpoint.pt', map_location='cpu', weights_only=False)
    with np.load(source / 'trajectory_and_targets.npz', allow_pickle=False) as archive:
        trajectory = {key: archive[key] for key in archive.files}
    horizons = tuple(float(x) for x in card['horizons_seconds'])
    h = horizons.index(float(horizon_hours) * 3600)
    cfg = H1DualTrainConfig(**{k: v for k, v in saved['config'].items()
                              if k in {f.name for f in fields(H1DualTrainConfig)}})
    data = build_h1_subject_data(subject, seed=seed, horizons_seconds=horizons)
    for key, rebuilt in [('anchor_time', data.rate.anchor_time), ('segment', data.rate.segment),
                         ('phase', data.rate.phase), ('target_count', data.rate.target_count),
                         ('target_valid', data.rate.target_valid),
                         ('target_exposure_seconds', data.rate.target_exposure_seconds)]:
        if not np.array_equal(trajectory[key], rebuilt):
            raise ValueError(f'frozen data alignment differs: {key}')
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
    q = tensor(np.clip((data.rate.q_raw - saved['q_centre']) / saved['q_scale'], -12, 12))
    bm = tensor(trajectory['fixed_mark_state'])
    bc = tensor(trajectory['background_current'])
    ba = tensor(trajectory['background_available'])
    es = tensor(trajectory['event_state'])
    bs = tensor(trajectory['background_state'])
    # The original training constructs event, random event, then background on
    # CPU before moving modules to device. Preserve that RNG order exactly.
    torch.manual_seed(seed)
    event_args = dict(taus_seconds=cfg.taus_seconds,
                      burden_channels_per_tau=cfg.burden_channels_per_tau,
                      grammar_channels_per_tau=cfg.grammar_channels_per_tau)
    event = DualStreamEventCTSSM(data.burden_mark.shape[1], data.grammar_mark.shape[1], **event_args)
    event_initial = {n: p.detach().clone() for n, p in event.named_parameters()}
    DualStreamEventCTSSM(data.burden_mark.shape[1], data.grammar_mark.shape[1], **event_args)
    background = GridBackgroundCTSSM(bc.shape[1], taus_seconds=cfg.taus_seconds,
                                     channels_per_tau=cfg.background_channels_per_tau)
    background_initial = {n: p.detach().clone() for n, p in background.named_parameters()}
    event.load_state_dict(saved['event_observer'])
    background.load_state_dict(saved['background_observer'])
    changes = {'event': module_change(event, event_initial),
               'background': module_change(background, background_initial)}
    event.to(device).eval(); background.to(device).eval()
    bd = len(cfg.taus_seconds) * data.burden_mark.shape[1]
    ed = len(cfg.taus_seconds) * cfg.burden_channels_per_tau
    readout = NestedDualReadout(q.shape[1], bd, bm.shape[1] - bd, bc.shape[1], bs.shape[1],
                                ed, es.shape[1] - ed, saved['widths'], len(horizons)).to(device)
    readout.load_state_dict(saved['readout']); readout.eval()
    for module in (event, background, readout):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    target, valid, scales = _target_bundle(data, device)
    exposure = tensor(data.rate.target_exposure_seconds)
    fit = torch.as_tensor(np.flatnonzero(data.rate.phase == 'FIT'), device=device)
    selected_np = np.flatnonzero(data.rate.phase == 'SELECTION')
    selected = torch.as_tensor(selected_np, device=device)
    with torch.no_grad():
        parent = readout.predict(q, bmark=bm, background_current=bc)
        target, permutation, _ = _fit_only_repertoire_alignment(target, valid, parent, fit)
        prediction = readout.predict(q, bmark=bm, background_current=bc, background_state=bs, event_state=es)
        rebuilt_score = _selection_score(prediction, target, valid, exposure, readout.log_dispersion,
                                          selected, selected_np, horizons)
        parity = _parent_parity_audit(rebuilt_score, card['selection_scores']['S_dual'], horizons)
    eligible = []
    for a in selected_np[data.rate.target_valid[selected_np, h]]:
        rows = np.flatnonzero((data.event_segment == data.rate.segment[a])
                              & (data.event_time < data.rate.anchor_time[a]))
        if len(rows) and data.rate.anchor_time[a] - data.event_time[rows[0]] >= 8 * 3600:
            eligible.append(int(a))
    positions = np.unique(np.linspace(0, len(eligible) - 1, min(maximum_anchors, len(eligible)), dtype=int))
    anchors = [eligible[i] for i in positions]
    rows_out = []
    max_parity = {'event': 0.0, 'background': 0.0}
    for a in anchors:
        qt = float(data.rate.anchor_time[a])
        er = np.flatnonzero((data.event_segment == data.rate.segment[a]) & (data.event_time < qt))
        br = np.flatnonzero((data.rate.segment == data.rate.segment[a]) & (data.rate.anchor_time <= qt))
        et = torch.as_tensor(data.event_time[er], dtype=torch.float64, device=device)
        bt = torch.as_tensor(data.rate.anchor_time[br], dtype=torch.float64, device=device)
        query = torch.tensor([qt], dtype=torch.float64, device=device)
        burden = tensor(data.burden_mark[er]).requires_grad_(True)
        grammar = tensor(data.grammar_mark[er]).requires_grad_(True)
        values = bc[br].detach().clone().requires_grad_(True)
        available = ba[br]
        spacing = float(torch.median(bt[1:] - bt[:-1])) if len(br) > 1 else 300.0
        initial_time = float(bt[0]) - max(spacing, 1.0)
        event_state = event_query(event, et, burden, grammar, query)
        background_state = background(bt, values, available, initial_time=initial_time).features[-1]
        for name, actual, frozen in [('event', event_state, es[a]), ('background', background_state, bs[a])]:
            delta = float((actual.detach() - frozen).abs().max())
            max_parity[name] = max(max_parity[name], delta)
            if not np.isfinite(delta) or delta > 1e-4:
                raise ValueError(f'{name} reconstruction mismatch: {delta}')

        def losses(event_value, background_value):
            pred = readout.predict(q[a:a+1], bmark=bm[a:a+1], background_current=bc[a:a+1],
                                    event_state=event_value[None], background_state=background_value[None])
            masks = {}
            for name, mask in valid.items():
                masks[name] = torch.zeros_like(mask[a:a+1])
                masks[name][:, h] = mask[a:a+1, h]
            return _endpoint_losses(pred, {n: t[a:a+1] for n, t in target.items()}, masks,
                                     exposure[a:a+1], readout.log_dispersion,
                                     torch.zeros(1, dtype=torch.long, device=device))

        base = losses(event_state, background_state)
        usable = [name for name in parity['auditable_endpoints'] if bool(valid[name][a, h])]
        age_event = qt - data.event_time[er]
        age_bg = qt - data.rate.anchor_time[br]
        result = {'anchor_index': a, 'time': qt, 'segment': int(data.rate.segment[a]),
                  'base_loss': {name: float(base[name].detach()) for name in usable}, 'credit': {}, 'deletions': {}}
        for name in usable:
            grads = torch.autograd.grad(base[name], (burden, grammar, values), retain_graph=True, allow_unused=True)
            if any(g is not None and not bool(torch.isfinite(g).all()) for g in grads):
                raise FloatingPointError(f'nonfinite gradient: {name}')
            result['credit'][name] = {'event': bin_gradient(age_event, grads[:2]),
                                      'background': bin_gradient(age_bg, grads[2:])}
        with torch.no_grad():
            for lo, hi in zip(BINS_HOURS[:-1], BINS_HOURS[1:]):
                label = _bin_name(lo, hi)
                result['deletions'][label] = {}
                for branch, ages in [('event', age_event), ('background', age_bg)]:
                    removed = (ages >= lo * 3600) & (ages < hi * 3600)
                    if branch == 'event':
                        changed = event_query(event, et, burden, grammar, query, tensor(~removed))
                        altered = losses(changed, background_state)
                    else:
                        changed = background(bt, values, available * tensor(~removed), initial_time=initial_time).features[-1]
                        altered = losses(event_state, changed)
                    result['deletions'][label][branch] = {
                        'removed_observations': int(removed.sum()),
                        'loss_increase': {n: float(altered[n] - base[n]) for n in usable},
                    }
        rows_out.append(result)
    for name, digest in hashes.items():
        if file_hash(source / name) != digest:
            raise ValueError(f'source changed during audit: {name}')
    payload = {
        'format': 'group_event_state_v038_dual_bound_credit_v1', 'status': 'COMPLETE' if anchors else 'NOT_ESTIMABLE',
        'subject': subject, 'family': 'dual', 'seed': seed, 'horizon_hours': horizon_hours,
        'source_directory': str(source), 'source_hashes': hashes, 'selected_parameter_changes': changes,
        'source_stages': {k: {n: v for n, v in card['stages'][k].items() if n != 'history'}
                          for k in ('event', 'background_state')},
        'score_parity': parity, 'state_parity_max_abs': max_parity,
        'repertoire_fit_only_permutation': list(permutation), 'target_scales': scales,
        'eligible_anchors': len(eligible), 'audited_anchors': len(anchors), 'anchors': rows_out,
        'sampling': 'evenly spaced indices among eligible SELECTION anchors; windows may overlap',
        'interpretation': 'branch-specific model sensitivity; baseline and other branch held fixed; not physiological causality',
        'claim_tier': 'repair_diagnostic_on_previously_examined_selection',
        'development_targets_read': False, 'sealed_partition_opened': False, 'seizure_targets_read': False,
        'elapsed_seconds': time.time() - started,
    }
    atomic_json(output, payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h1-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--subject', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--horizon-hours', type=float, choices=(6.0, 8.0), required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--maximum-anchors', type=int, default=8)
    args = parser.parse_args()
    if args.maximum_anchors < 1:
        parser.error('maximum-anchors must be positive')
    if args.output.exists():
        raise FileExistsError(f'refusing to replace audit: {args.output}')
    result = audit(args.subject, args.seed, args.horizon_hours, args.h1_root, args.output,
                   torch.device(args.device), args.maximum_anchors)
    print(json.dumps({k: result[k] for k in ('status', 'seed', 'horizon_hours', 'audited_anchors', 'elapsed_seconds')}), flush=True)


if __name__ == '__main__':
    main()
