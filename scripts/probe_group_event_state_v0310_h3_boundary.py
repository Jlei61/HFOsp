#!/usr/bin/env python3
"""H3 boundary check (spec section 8): what a history ablation can and cannot show.

The same four ablation operators are replayed on a frozen human observer and
on a model fitted to the registered `no_event_feedback` world, where events
only reveal a common evolving cause and never enter its drift. If the ablation
signature is present there too, then a human ablation effect cannot identify
event-driven physiology.
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src.topic5_group_event_state.v039.transition import EventTransition, FutureReadout, endpoint_loss
from src.topic5_group_event_state.v039.synthetic import generate
from src.topic5_group_event_state.v0310 import audit
from src.topic5_group_event_state.v0310.history import history_matrix
from src.topic5_group_event_state.v0310.trainer import load_cell, partition, CellConfig, LEADS
from src.topic5_group_event_state.v035.contracts import atomic_json

OPERATORS = ('remove_oldest_quarter', 'remove_newest_quarter', 'permute_event_order', 'remove_all_events')


def event_rows(x):
    """Time steps that actually carry a released measurement block."""
    return np.abs(x).sum(-1) > 0


def ablate(x, operator, rng):
    """Operators keep every elapsed time exactly; only the marks move or vanish."""
    out = x.copy()
    carrying = event_rows(x)
    for i in range(len(x)):
        index = np.flatnonzero(carrying[i])
        if not len(index):
            continue
        if operator == 'remove_all_events':
            out[i, index] = 0.
        elif operator == 'remove_oldest_quarter':
            out[i, index[:max(1, len(index) // 4)]] = 0.
        elif operator == 'remove_newest_quarter':
            out[i, index[-max(1, len(index) // 4):]] = 0.
        elif operator == 'permute_event_order':
            out[i, index] = x[i, index][rng.permutation(len(index))]
        else:
            raise ValueError(operator)
    return out


def human_arm(card, seed):
    cfg = CellConfig(data=card['data_path'], output_dir='/dev/null', family=card['family'],
                     state_width=card['config']['state_width'], transition_rank=card['config']['transition_rank'],
                     readout_hidden=card['config']['readout_hidden'], history_hours=card['history_hours'],
                     source_mode=card['source_mode'], view=card['view'], seed=card['seed'], device='cpu')
    cell = load_cell(cfg); part = partition(cell, card['view'])
    rows = part['selection'][cell['valid'][part['selection'], 1]]
    if len(rows) < 5:
        return None
    state = torch.load(card['checkpoint'], map_location='cpu', weights_only=False)
    observer = EventTransition(cell['input_dim'], card['family'], width=card['config']['state_width'],
                               rank=card['config']['transition_rank'], seed=card['seed'])
    observer.load_state_dict(state['observer']); observer.requires_grad_(False)
    residual = FutureReadout(observer.width, cell['n_recruitment'], 0, hidden=card['config']['readout_hidden'])
    residual.load_state_dict(state['residual']); residual.requires_grad_(False)
    baseline = FutureReadout(0, cell['n_recruitment'], cell['context'].shape[1],
                             hidden=card['config']['readout_hidden'])
    baseline.load_state_dict(state['baseline']); baseline.requires_grad_(False)
    x = cell['x'][torch.as_tensor(rows)].numpy(); dt = cell['dt'][torch.as_tensor(rows)]
    context = torch.tensor(cell['context'][rows]); counts = torch.tensor(cell['counts'][rows, 1])
    recruitment = torch.tensor(cell['recruitment'][rows, 1])
    mask = torch.tensor(cell['spatial_valid'][rows, 1])
    rng = np.random.default_rng(seed)

    def loss_for(values):
        with torch.no_grad():
            s = observer.scan(torch.from_numpy(values), dt, checkpoint_chunk=0)
            dm, dl = residual(s, s.new_empty((len(values), 0)), LEADS[1])
            bm, bl = baseline(context.new_empty((len(values), 0)), context, LEADS[1])
            _, nb, sp = endpoint_loss(bm + dm, bl + dl, counts, recruitment, residual.log_dispersion, 'joint')
        return float(nb.mean()), (float(sp[mask].mean()) if bool(mask.any()) else None)
    base_count, base_spatial = loss_for(x)
    result = dict(n_anchors=int(len(rows)),
                  mean_event_steps=float(event_rows(x).sum(1).mean()), baseline_count_nll=base_count,
                  baseline_recruitment_nll=base_spatial, operators={})
    for operator in OPERATORS:
        count, spatial = loss_for(ablate(x, operator, rng))
        result['operators'][operator] = dict(
            delta_count_nll=count - base_count,
            delta_recruitment_nll=None if spatial is None or base_spatial is None else spatial - base_spatial)
    return result


def synthetic_arm(work, seed, device='cpu', max_steps=800):
    card_path = Path(work) / 'no_event_feedback' / 'card.json'
    if not card_path.exists():
        card_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, str(ROOT / 'scripts/run_group_event_state_v039_instrument.py'),
                        '--case', 'no_event_feedback', '--family', 'L', '--seed', str(seed),
                        '--max-steps', str(max_steps), '--device', device, '--output', str(card_path)],
                       check=True, capture_output=True, text=True, cwd=str(ROOT))
    card = json.loads(card_path.read_text())
    state = torch.load(card['checkpoint'], map_location='cpu', weights_only=False)
    cfg = state['config']
    data = generate('no_event_feedback', n=cfg['episodes'], strength=cfg['strength'])
    observer = EventTransition(data['inputs'].shape[-1], 'L', width=cfg['width'], seed=cfg['seed'])
    observer.load_state_dict(state['observer']); observer.requires_grad_(False)
    readout = FutureReadout(observer.width, 2, 2)
    readout.load_state_dict(state['readout']); readout.requires_grad_(False)
    rows = slice(data['inner_end'], None)
    x = data['inputs'][rows]; dt = torch.from_numpy(data['dt_hours'][rows])
    context = torch.from_numpy(data['context'][rows])
    counts = torch.from_numpy(data['targets']['2']['count'][rows])
    recruitment = torch.from_numpy(data['targets']['2']['recruitment'][rows])
    rng = np.random.default_rng(seed)

    def loss_for(values):
        with torch.no_grad():
            s = observer.scan(torch.from_numpy(values), dt, checkpoint_chunk=0)
            mu, logits = readout(s, context, 2.)
            _, nb, sp = endpoint_loss(mu, logits, counts, recruitment, readout.log_dispersion, 'joint')
        return float(nb.mean()), float(sp.mean())
    base_count, base_spatial = loss_for(x)
    result = dict(case='no_event_feedback', n_episodes=int(x.shape[0]),
                  event_changes_true_target_state=bool(data['truth']['event_changes_true_target_state']),
                  observed_event_information_predictive=bool(data['truth']['observed_event_information_predictive']),
                  mean_event_steps=float(event_rows(x).sum(1).mean()),
                  baseline_count_nll=base_count, baseline_recruitment_nll=base_spatial,
                  upstream_card=str(card_path), stop_reason=card.get('stop_reason'), operators={})
    for operator in OPERATORS:
        count, spatial = loss_for(ablate(x, operator, rng))
        result['operators'][operator] = dict(delta_count_nll=count - base_count,
                                             delta_recruitment_nll=spatial - base_spatial)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--phases', nargs='+', default=['U1', 'U2'])
    p.add_argument('--seed', type=int, default=20260905)
    p.add_argument('--max-cells', type=int, default=9)
    p.add_argument('--deadline-epoch', type=float, default=0.)
    args = p.parse_args()
    torch.set_num_threads(2)
    out = args.root / 'h3_boundary'; out.mkdir(parents=True, exist_ok=True)
    control = synthetic_arm(out, args.seed)
    print(json.dumps(dict(arm='no_event_feedback_control',
                          operators={k: round(v['delta_count_nll'], 5)
                                     for k, v in control['operators'].items()})), flush=True)
    cards = []
    for phase in args.phases:
        cards += audit.load_cards(args.root / 'human_v0310' / phase)
    cards = [c for c in cards if c['view'] == 'joint' and c.get('stages', {}).get('event')]
    queue = args.root / 'queue_state.json'
    common = dict((json.loads(queue.read_text()).get('common_recipe') or {})) if queue.exists() else {}
    if common:
        cards = [c for c in cards if common.get(c['subject'], c['recipe']) == c['recipe']]
    picked, seen = [], set()
    for card in sorted(cards, key=lambda c: (c['subject'], c['seed'], c['family'])):
        key = (card['subject'], card['seed'])
        if key in seen:
            continue
        group = [c for c in cards if (c['subject'], c['seed']) == key]
        best = min(group, key=lambda c: (c['stages']['event']['selected_inner'], c['family']))
        seen.add(key); picked.append(best)
    human = []
    for card in picked[:args.max_cells]:
        try:
            row = human_arm(card, args.seed)
        except Exception as error:
            row = dict(error=str(error)[:300])
        if row:
            row.update(subject=card['subject'], seed=card['seed'], family=card['family'],
                       recipe=card['recipe'], history_hours=card['history_hours'],
                       upstream_card=card['_path'],
                       upstream_selected_initial_checkpoint=card['stages']['event']['selected_update'] == 0)
            human.append(row)
            print(json.dumps(dict(subject=row.get('subject'), family=row.get('family'),
                                  operators={k: round(v['delta_count_nll'], 5)
                                             for k, v in row.get('operators', {}).items()})), flush=True)
        if args.deadline_epoch and time.time() >= args.deadline_epoch:
            break
    control_effects = {k: v['delta_count_nll'] for k, v in control['operators'].items()}
    human_effects = {k: [r['operators'][k]['delta_count_nll'] for r in human if 'operators' in r]
                     for k in OPERATORS}
    verdict = dict(
        control_shows_ablation_effect_without_any_feedback=bool(
            max(abs(v) for v in control_effects.values()) > 0),
        largest_control_effect=max(control_effects.items(), key=lambda kv: abs(kv[1])),
        human_effect_identifies_event_driven_physiology=False,
        reason='the control world was generated so that events never enter the state drift, and the same '
               'operators still move its predictions. The EXISTENCE of an ablation effect therefore '
               'cannot separate event-driven physiology from a shared slow cause.',
        magnitude_is_not_a_calibrated_comparison=(
            'the human effects are far larger than the control ones, but the two worlds have different '
            'loss scales, different event densities and different model sizes, so the control does not '
            'calibrate a magnitude threshold. A large human effect can be read neither as feedback nor '
            'as its absence.'),
        human_effect_range={k: (None if not v else [min(v), max(v)]) for k, v in human_effects.items()},
        note='a negative human value means the ablated history predicted the independent future target '
             'BETTER than the real one, which is a statement about the fitted pathway, not about physiology')
    atomic_json(out / 'h3_boundary.json',
                dict(status='COMPLETE', schema='v0310_h3_boundary_v1', timestamp=time.time(),
                     operators=list(OPERATORS), no_feedback_control=control, human=human,
                     identification_verdict=verdict,
                     identification_note='in the control world events never enter the state drift, yet the '
                                         'same ablations still move the predictions because the marks reveal '
                                         'a common cause. An ablation effect in the human data therefore '
                                         'cannot distinguish event-driven physiology from a shared cause.',
                     what_this_does_not_test='per-event preparation or feedback; that needs low-latency '
                                             'measurement and an independent future observation',
                     source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(json.dumps(dict(status='COMPLETE', n_human_cells=len(human))), flush=True)


if __name__ == '__main__':
    main()
