#!/usr/bin/env python3
"""Paired audit of the B0 learning-rate diagnostic.

These are separate optimisation runs, not continuations of the lr=0.003 fits,
and all three optimisation seeds share synthetic data seed 39001, so they are
optimisation repeats rather than three independent worlds.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from src.topic5_group_event_state.v035.contracts import atomic_json

REFERENCE = {  # verified nine-card lr=0.003 budget-2400 audit, for side-by-side only
    'N_over_L': -0.008185, 'N_over_F': 0.073232, 'L_over_F': 0.086236,
    'source': '/data/hfosp_group_event_state_v0_3_9_transition_transfer_background_repaired/'
              'final_reports_training_review/instrument_budget_audit.json'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    folder = args.root / 'bootstrap_lr_audit'
    cards, rows = {}, []
    for path in sorted(folder.glob('*/card.json')):
        card = json.loads(path.read_text())
        if card.get('status') != 'COMPLETE':
            continue
        name = path.parent.name
        family = name.split('_')[1]; lr = float(name.split('_lr')[1].split('_')[0]); seed = int(name.split('seed')[1])
        digest = hashlib.sha256(Path(card['checkpoint']).read_bytes()).hexdigest()
        if digest != card['checkpoint_sha256']:
            raise ValueError(f'Checkpoint hash mismatch: {path}')
        cards[(lr, seed, family)] = card
        rows.append(dict(family=family, lr=lr, seed=seed, data_seed=card['truth']['seed'],
                         held_out_total=card['held_out']['total'], held_out_count=card['held_out']['count'],
                         held_out_recruitment=card['held_out']['recruitment'],
                         selected_step=card['selected_step'], steps_run=card['steps_run'],
                         max_steps=card['config']['max_steps'], stop_reason=card['stop_reason'],
                         optimization_limited=card['optimization_limited'],
                         elapsed_seconds=round(card['elapsed_seconds']), card=str(path)))
    contrasts = {}
    for lr in sorted({k[0] for k in cards}):
        seeds = sorted({k[1] for k in cards if k[0] == lr and set(
            f for l, s, f in cards if (l, s) == (lr, k[1])) == {'F', 'L', 'N'}})
        entry = dict(lr=lr, complete_seeds=seeds, n_complete_triples=len(seeds))
        for better, worse in (('N', 'L'), ('N', 'F'), ('L', 'F')):
            values = [cards[(lr, s, worse)]['held_out']['total'] - cards[(lr, s, better)]['held_out']['total']
                      for s in seeds]
            entry[f'{better}_over_{worse}'] = dict(
                values=[round(v, 6) for v in values],
                median=round(float(np.median(values)), 6) if values else None,
                n_positive=int(sum(v > 0 for v in values)), n=len(values))
        entry['all_arms_budget_limited'] = all(cards[(lr, s, f)]['optimization_limited']
                                               for s in seeds for f in ('F', 'L', 'N'))
        entry['selected_steps'] = {f: [cards[(lr, s, f)]['selected_step'] for s in seeds] for f in ('F', 'L', 'N')}
        contrasts[str(lr)] = entry
    out = args.root / 'final_reports'; out.mkdir(parents=True, exist_ok=True)
    with (out / 'bootstrap_lr_audit.csv').open('w', newline='') as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    status_path = args.root / 'bootstrap_status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    atomic_json(out / 'bootstrap_lr_audit.json',
                dict(status='COMPLETE', schema='v0310_bootstrap_lr_audit_v1', timestamp=time.time(),
                     n_cards=len(rows), n_planned=18, driver_status=status.get('status'),
                     unfinished=[j for j in status.get('pending', [])],
                     data_seed=39001, n_data_seeds=1, contrasts=contrasts,
                     lr0p003_reference_from_previous_audit=REFERENCE,
                     interpretation='separate optimisation runs at a lower learning rate, not a continuation '
                                    'of the lr=0.003 checkpoints; one synthetic data seed, so the three '
                                    'optimisation seeds are repeats, not independent worlds',
                     training_sufficiency='NOT_ESTABLISHED_BY_STOP_REASON'))
    print(json.dumps(dict(n_cards=len(rows), contrasts={k: {c: v[c]['median'] for c in
                                                            ('N_over_L', 'N_over_F', 'L_over_F')}
                                                        for k, v in contrasts.items()}), ensure_ascii=False),
          flush=True)


if __name__ == '__main__':
    main()
