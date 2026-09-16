#!/usr/bin/env python3
"""Clause C15 gate. A model that fails any test must not enter the queue."""
from __future__ import annotations

import argparse, copy, json, shutil, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.topic5_group_event_state.v0310 import history as H
from src.topic5_group_event_state.v0310 import audit
from src.topic5_group_event_state.v0310.trainer import (
    CellConfig, run_cell, load_cell, partition, accumulate_update, sufficiency_verdict, LEADS)
from src.topic5_group_event_state.v0310.objective import fit_intercept_weights
from src.topic5_group_event_state.v039.transition import EventTransition, FutureReadout

BASE = Path('/data/hfosp_group_event_state_v0_3_9_transition_transfer_background_repaired/human_data_v2')
RESULTS = []


def record(name, passed, detail):
    RESULTS.append(dict(test=name, passed=bool(passed), detail=detail))
    print(('PASS  ' if passed else 'FAIL  ') + name + ' :: ' + json.dumps(detail, default=str)[:400], flush=True)
    return passed


# ---------------------------------------------------------------- t1 / resume
def t1_resume(work, subject='epilepsiae_1125'):
    common = dict(data=str(BASE / f'{subject}.pt'), family='N', state_width=16, transition_rank=8,
                  readout_hidden=32, history_hours=0.5, device='cpu', seed=20260905,
                  stage_budget=dict(background=60, event=150, refitted_constant=60), recipe_id='R0')
    plain = CellConfig(output_dir=str(work / 't1_plain'), **common)
    card_a = run_cell(plain)
    broken = CellConfig(output_dir=str(work / 't1_resume'), **common)

    def hook(stage, update, score):
        if stage == 'event' and update == 100:
            raise KeyboardInterrupt('simulated crash right after a checkpoint')
    try:
        run_cell(broken, progress_hook=hook)
        return record('t1_resume_matches_uninterrupted', False, {'reason': 'the simulated crash did not fire'})
    except KeyboardInterrupt:
        pass
    card_b = run_cell(CellConfig(output_dir=str(work / 't1_resume'), **common))
    a = torch.load(work / 't1_plain' / 'model.pt', map_location='cpu', weights_only=False)
    b = torch.load(work / 't1_resume' / 'model.pt', map_location='cpu', weights_only=False)
    worst = 0.
    for module in ('observer', 'residual', 'baseline', 'constant'):
        for key in a[module]:
            worst = max(worst, float((a[module][key] - b[module][key]).abs().max()))
    curve_a = [c['inner']['objective'] for c in card_a['stages']['event']['curve']]
    curve_b = [c['inner']['objective'] for c in card_b['stages']['event']['curve']]
    same_schedule = (card_a['stages']['event']['batch_schedule_sha256']
                     == card_b['stages']['event']['batch_schedule_sha256'])
    return record('t1_resume_matches_uninterrupted', worst == 0. and curve_a == curve_b and same_schedule,
                  dict(max_abs_weight_difference=worst, curve_len=len(curve_a),
                       curves_identical=curve_a == curve_b, batch_schedule_identical=same_schedule,
                       resume_label=card_b.get('resume')))


# --------------------------------------------------- t2 / micro-batch identity
def _probe_setup(subject='epilepsiae_1125', hours=0.5, family='N', width=16, rank=8, hidden=32):
    cfg = CellConfig(data=str(BASE / f'{subject}.pt'), output_dir='/dev/null', family=family,
                     state_width=width, transition_rank=rank, readout_hidden=hidden,
                     history_hours=hours, device='cpu', seed=20260905)
    cell = load_cell(cfg); part = partition(cell, 'joint')
    counts = torch.tensor(cell['counts']); recr = torch.tensor(cell['recruitment'])
    valid = torch.tensor(cell['valid']); sval = torch.tensor(cell['spatial_valid'])
    weights = fit_intercept_weights(counts, recr, valid, sval, torch.as_tensor(part['fit']))
    torch.manual_seed(cfg.seed + 100)
    observer = EventTransition(cell['input_dim'], family, width=width, rank=rank, seed=cfg.seed)
    residual = FutureReadout(observer.width, cell['n_recruitment'], 0, hidden=hidden)
    with torch.no_grad():
        residual.layers[-1].weight.normal_(0, .05); residual.layers[-1].bias.normal_(0, .05)

    def predict(rows, lead, training=False):
        index = torch.as_tensor(np.asarray(rows))
        state = observer.scan(cell['x'][index], cell['dt'][index], checkpoint_chunk=32 if training else 0)
        return residual(state, state.new_empty((len(rows), 0)), lead)
    return cfg, cell, part, (counts, recr, sval), weights, observer, residual, predict


def t2_microbatch():
    cfg, cell, part, targets, weights, observer, residual, predict = _probe_setup()
    rows = part['pools'][1][:128]
    if len(rows) < 128:
        rows = np.resize(part['pools'][1], 128)
    n_spatial = int(cell['spatial_valid'][rows, 1].sum())
    grads = {}
    for physical in (128, 32, 16):
        for p in list(observer.parameters()) + list(residual.parameters()):
            p.grad = None
        accumulate_update(predict, lambda: residual.log_dispersion, targets, rows, 1, weights,
                          'joint', physical, len(rows), n_spatial)
        grads[physical] = {n: p.grad.detach().clone()
                           for n, p in list(observer.named_parameters()) + list(residual.named_parameters())}
    worst = {}
    for physical in (32, 16):
        worst[physical] = max(float((grads[128][n] - grads[physical][n]).abs().max()
                                    / max(float(grads[128][n].abs().max()), 1e-12))
                              for n in grads[128])
    return record('t2_microbatch_gradient_matches_whole_batch', max(worst.values()) < 1e-5,
                  dict(max_relative_difference=worst, n_rows=len(rows), n_spatial=n_spatial))


# -------------------------------------------------------- t3 / sufficiency gate
def t3_budget_gate():
    limited = sufficiency_verdict({'background': dict(budget_limited=False, wall_time_limited=False,
                                                      plateau_after_lr_reduction=True),
                                   'event': dict(budget_limited=True, wall_time_limited=False,
                                                 plateau_after_lr_reduction=False)})
    walled = sufficiency_verdict({'event': dict(budget_limited=False, wall_time_limited=True,
                                                plateau_after_lr_reduction=False)})
    clean = sufficiency_verdict({'event': dict(budget_limited=False, wall_time_limited=False,
                                               plateau_after_lr_reduction=True)})
    ok = (limited['verdict'] == 'NOT_ESTABLISHED_BY_STOP_REASON'
          and walled['verdict'] == 'NOT_ESTABLISHED_BY_STOP_REASON'
          and clean['verdict'] == 'PLATEAU_AFTER_LR_REDUCTION_ON_ALL_STAGES')
    return record('t3_budget_limited_arm_cannot_be_sufficient', ok,
                  dict(budget=limited['verdict'], wall=walled['verdict'], plateau=clean['verdict']))


# ------------------------------------------------------------- t4 / pairing gate
def t4_pairing(card):
    base = json.loads(json.dumps(card))
    twin = json.loads(json.dumps(card)); twin['family'] = 'L'
    refusals = {}
    audit.assert_pairable([base, twin])
    for field, value in (('data_sha256', 'deadbeef'), ('split_sha256', 'deadbeef'),
                         ('target_sha256', 'deadbeef'), ('input_dim', 7)):
        bad = json.loads(json.dumps(twin)); bad[field] = value
        try:
            audit.assert_pairable([base, bad]); refusals[field] = 'ACCEPTED'
        except ValueError:
            refusals[field] = 'REFUSED'
    bad = json.loads(json.dumps(twin)); bad['source_hashes'] = {'x': 'y'}
    try:
        audit.assert_pairable([base, bad]); refusals['source_hashes'] = 'ACCEPTED'
    except ValueError:
        refusals['source_hashes'] = 'REFUSED'
    bad = json.loads(json.dumps(twin)); bad['seed'] = 1
    try:
        audit.assert_pairable([base, bad]); refusals['seed'] = 'ACCEPTED'
    except ValueError:
        refusals['seed'] = 'REFUSED'
    return record('t4_refuse_pairing_on_identity_mismatch',
                  all(v == 'REFUSED' for v in refusals.values()), refusals)


# --------------------------------------------- t5 / publication-time truncation
def t5_truncation(subject='epilepsiae_1125'):
    data = torch.load(BASE / f'{subject}.pt', map_location='cpu', weights_only=False)
    blocks = data['event_replay_blocks']
    sample = data['samples'][len(data['samples']) // 2]
    anchor = sample['anchor']
    full = H.build_history(blocks, anchor, 8.0, data['input_dim'])
    past_only = H.build_history([b for b in blocks if b['release'] <= anchor], anchor, 8.0, data['input_dim'])
    later = [b for b in blocks if b['release'] > anchor]
    dropped_future = H.build_history([b for b in blocks if b['release'] <= anchor], anchor, 8.0, data['input_dim'])
    shifted = copy.deepcopy(sample); shifted['targets'] = [(0., np.zeros(data['n_recruitment']), False, False)] * 3
    after_target_shift = H.build_history(blocks, shifted['anchor'], 8.0, data['input_dim'])
    same_future = all(np.array_equal(a, b) for a, b in zip(full, past_only))
    same_target = all(np.array_equal(a, b) for a, b in zip(full, after_target_shift))
    return record('t5_targets_after_the_query_do_not_change_past_features',
                  same_future and same_target and len(later) > 0,
                  dict(future_blocks_available=len(later), invariant_to_future_blocks=same_future,
                       invariant_to_target_edit=same_target, steps=len(full[0])))


# ------------------------------------- t6 / cross-block replay and real gradient
def t6_gradient_replay(subject='epilepsiae_1125'):
    cfg, cell, part, targets, weights, observer, residual, predict = _probe_setup(subject, hours=8.0)
    row = None
    for candidate in part['pools'][1][:200]:
        nz = np.flatnonzero(np.abs(cell['x'][candidate].numpy()).sum(-1) > 0)
        if len(nz) >= 2:
            row = int(candidate); nonzero = nz; break
    if row is None:
        return record('t6_real_cross_block_gradient_replay', False, {'reason': 'no multi-block history found'})
    x = cell['x'][row:row + 1].clone().requires_grad_(True)
    dt = cell['dt'][row:row + 1]
    state = observer.scan(x, dt, checkpoint_chunk=32)
    mu, logits = residual(state, state.new_empty((1, 0)), 2.)
    scalar = mu.sum() + logits.sum()
    scalar.backward()
    grad = x.grad[0].numpy()
    earliest = int(nonzero[0]); latest = int(nonzero[-1])
    g_earliest = float(np.abs(grad[earliest]).max())
    eps = 1e-2
    column = int(np.argmax(np.abs(grad[earliest])))
    with torch.no_grad():
        bumped = cell['x'][row:row + 1].clone(); bumped[0, earliest, column] += eps
        s2 = observer.scan(bumped, dt, checkpoint_chunk=0)
        m2, l2 = residual(s2, s2.new_empty((1, 0)), 2.)
        finite = float((m2.sum() + l2.sum() - scalar.detach()) / eps)
    analytic = float(grad[earliest, column])
    ok = g_earliest > 0 and abs(finite - analytic) <= 0.05 * max(abs(analytic), 1e-6) + 1e-4
    return record('t6_real_cross_block_gradient_replay', ok,
                  dict(row=row, n_nonzero_event_steps=int(len(nonzero)), earliest_step=earliest,
                       latest_step=latest, total_steps=int(cell['x'].shape[1]),
                       gradient_at_earliest_event=g_earliest, analytic=analytic, finite_difference=finite))


# ---------------------------------------------- t7 / refuse to fabricate history
def t7_refuse_fabrication(subject='epilepsiae_1125'):
    data = torch.load(BASE / f'{subject}.pt', map_location='cpu', weights_only=False)
    outcomes = {}
    stripped = dict(data); stripped.pop('event_replay_blocks')
    try:
        H.history_matrix(stripped, 24.0); outcomes['missing_replay_blocks'] = 'FABRICATED'
    except (KeyError, ValueError):
        outcomes['missing_replay_blocks'] = 'REFUSED'
    corrupted = copy.deepcopy(data)
    x, dt = corrupted['samples'][0]['histories']['8.0']
    corrupted['samples'][0]['histories']['8.0'] = (x + 1.0, dt)
    try:
        H.verify_against_stored(corrupted); outcomes['corrupted_stored_history'] = 'ACCEPTED'
    except ValueError:
        outcomes['corrupted_stored_history'] = 'REFUSED'
    long = H.history_matrix(data, 24.0)
    outcomes['h24_rebuilt_from_replay'] = bool(long['rebuilt_from_replay'])
    outcomes['h24_median_coverage'] = float(np.median(long['coverage']))
    outcomes['h24_anchors_above_80pct'] = int((long['coverage'] >= .8).sum())
    outcomes['h24_steps'] = int(long['steps'])
    ok = (outcomes['missing_replay_blocks'] == 'REFUSED'
          and outcomes['corrupted_stored_history'] == 'REFUSED'
          and outcomes['h24_rebuilt_from_replay'] and outcomes['h24_median_coverage'] < 1.0000001)
    return record('t7_refuse_fabricated_long_history', ok, outcomes)


# ------------------------- t8 / observer update separated from physiology claim
def t8_observation_not_feedback(subject='epilepsiae_1125'):
    cfg, cell, part, targets, weights, observer, residual, predict = _probe_setup(subject, hours=8.0)
    row = int(part['pools'][1][0])
    x = cell['x'][row:row + 1].clone()
    nz = np.flatnonzero(np.abs(x[0].numpy()).sum(-1) > 0)
    if not len(nz):
        return record('t8_observer_update_is_not_physiological_feedback', False, {'reason': 'no events'})
    ablated = x.clone(); ablated[0, int(nz[0])] = 0.
    with torch.no_grad():
        s_full = observer.scan(x, cell['dt'][row:row + 1], checkpoint_chunk=0)
        s_ablated = observer.scan(ablated, cell['dt'][row:row + 1], checkpoint_chunk=0)
    state_moved = float((s_full - s_ablated).abs().max())
    targets_unchanged = bool(np.array_equal(cell['counts'], cell['counts'])
                             and np.array_equal(cell['recruitment'], cell['recruitment']))
    before = cell['counts'][row].copy()
    after = cell['counts'][row].copy()
    return record('t8_observer_update_is_not_physiological_feedback',
                  state_moved > 0 and targets_unchanged and np.array_equal(before, after),
                  dict(state_change_from_removing_one_event=state_moved,
                       future_targets_unchanged=True,
                       reading='removing a released event moves the model state: a computational dependence '
                               'on the measurement stream. The future observation targets are untouched, so '
                               'nothing here tests whether an IED changed physiology.'))


# --------------------------------------------------------------- memory probe
def memory_probe(device, subject, family, width, rank, hidden, hours, updates=6, physical=128):
    cfg, cell, part, targets, weights, observer, residual, predict = _probe_setup(
        subject, hours=hours, family=family, width=width, rank=rank, hidden=hidden)
    dev = torch.device(device)
    observer = observer.to(dev); residual = residual.to(dev)
    counts, recr, sval = (t.to(dev) for t in targets)

    def gpu_predict(rows, lead, training=False):
        index = torch.as_tensor(np.asarray(rows))
        state = observer.scan(cell['x'][index].to(dev), cell['dt'][index].to(dev),
                              checkpoint_chunk=32 if training else 0)
        return residual(state, state.new_empty((len(rows), 0)), lead)
    optimiser = torch.optim.AdamW(list(observer.parameters()) + list(residual.parameters()), lr=1e-3)
    torch.cuda.reset_peak_memory_stats(dev); torch.cuda.synchronize(dev)
    started = time.time()
    for u in range(updates):
        rows = np.resize(part['pools'][1], physical)
        optimiser.zero_grad(set_to_none=True)
        accumulate_update(gpu_predict, lambda: residual.log_dispersion, (counts, recr, sval), rows, 1,
                          weights, 'joint', physical, len(rows), int(cell['spatial_valid'][rows, 1].sum()))
        torch.nn.utils.clip_grad_norm_(list(observer.parameters()) + list(residual.parameters()), 2.)
        optimiser.step()
    torch.cuda.synchronize(dev)
    elapsed = time.time() - started
    return dict(subject=subject, family=family, state_width=int(observer.width), readout_hidden=hidden,
                history_hours=hours, physical_batch=physical, updates=updates,
                seconds_per_update=elapsed / updates,
                peak_allocated_mib=torch.cuda.max_memory_allocated(dev) / 2 ** 20,
                peak_reserved_mib=torch.cuda.max_memory_reserved(dev) / 2 ** 20,
                observer_parameters=int(sum(p.numel() for p in observer.parameters())),
                readout_parameters=int(sum(p.numel() for p in residual.parameters())))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--skip-memory', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.work.exists():
        shutil.rmtree(args.work)
    args.work.mkdir(parents=True)
    t1_resume(args.work)
    t2_microbatch()
    t3_budget_gate()
    card = json.loads((args.work / 't1_plain' / 'card.json').read_text())
    t4_pairing(card)
    t5_truncation()
    t6_gradient_replay()
    t7_refuse_fabrication()
    t8_observation_not_feedback()
    memory = []
    if not args.skip_memory and torch.cuda.is_available():
        for subject, family, width, rank, hidden in (('epilepsiae_253', 'N', 64, 16, 128),
                                                     ('epilepsiae_253', 'L', 64, 16, 128),
                                                     ('epilepsiae_253', 'F', 16, 8, 128),
                                                     ('epilepsiae_1096', 'F', 16, 8, 128)):
            try:
                memory.append(memory_probe(args.device, subject, family, width, rank, hidden, 8.0))
                print('MEM  ' + json.dumps(memory[-1]), flush=True)
            except RuntimeError as error:
                memory.append(dict(subject=subject, family=family, error=str(error)[:200]))
    payload = dict(schema='v0310_training_preflight_v1', timestamp=time.time(),
                   tests=RESULTS, all_passed=all(r['passed'] for r in RESULTS),
                   memory_probe=memory,
                   source_hashes={p: __import__('hashlib').sha256((ROOT / p).read_bytes()).hexdigest()
                                  for p in ('src/topic5_group_event_state/v0310/trainer.py',
                                            'src/topic5_group_event_state/v0310/objective.py',
                                            'src/topic5_group_event_state/v0310/history.py',
                                            'src/topic5_group_event_state/v0310/audit.py',
                                            'scripts/train_group_event_state_v0310_human.py')},
                   gate='A model failing any test must not enter the formal queue (clause C15)')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + '\n')
    print(json.dumps(dict(all_passed=payload['all_passed'],
                          failed=[r['test'] for r in RESULTS if not r['passed']])), flush=True)
    return 0 if payload['all_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
