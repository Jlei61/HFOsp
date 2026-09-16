#!/usr/bin/env python3
"""M1: freeze one upstream cell per (subject, seed, source mode) and probe transfer.

The family is chosen on INNER only. Upstream stays frozen: nothing here can
fine-tune the observer for contacts or seizures. Contact identity and fine
expression are separate endpoints and are reported separately.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.topic5_group_event_state.v0310 import audit
from src.topic5_group_event_state.v0310.queue_plan import RECIPES
from src.topic5_group_event_state.v035.contracts import atomic_json

OLD_ROOT = '/data/hfosp_group_event_state_v0_3_9_transition_transfer_background_repaired'


def pick_family(cards):
    """INNER-selected family inside one (subject, seed, recipe, mode) group.

    Returns the winner plus how it was decided. When several arms sit at exactly
    the same validation loss -- which happens when none of them left its
    initialisation -- the winner is an alphabetical tie-break and must be
    reported as such, not as a selection.
    """
    scored = [(c['stages']['event']['selected_inner'], c['family'], c) for c in cards
              if c.get('stages', {}).get('event')]
    if not scored:
        return None, {}
    scored.sort(key=lambda r: (r[0], r[1]))
    best = scored[0][0]
    tied = [r[1] for r in scored if r[0] == best]
    return scored[0][2], dict(tied_families=sorted(tied), decided_by_tie_break=len(tied) > 1,
                              all_arms_at_initialisation=all(
                                  r[2]['stages']['event']['selected_update'] == 0 for r in scored),
                              margin_to_next=None if len(scored) < 2 else float(scored[1][0] - best))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--old-root', default=OLD_ROOT)
    p.add_argument('--phases', nargs='+', default=['U1', 'U2', 'U3'])
    p.add_argument('--deadline-epoch', type=float, default=0.)
    p.add_argument('--max-steps', type=int, default=600)
    p.add_argument('--foundation-steps', type=int, default=400)
    p.add_argument('--allow-partial', action='store_true',
                   help='select a family before the F/L/N triple completes (records the fact)')
    args = p.parse_args()
    out = args.root / 'm1_transfer'; out.mkdir(parents=True, exist_ok=True)
    cards = []
    for phase in args.phases:
        cards += audit.load_cards(args.root / 'human_v0310' / phase)
    # The frozen state per (patient, seed) must sit at the SHARED recipe chosen
    # on INNER by the capacity phase. Grouping without the recipe would let
    # three cells of different capacity look like a complete family triple and
    # freeze a state at a recipe that was never selected.
    state = json.loads((args.root / 'queue_state.json').read_text()) \
        if (args.root / 'queue_state.json').exists() else {}
    common = dict(state.get('common_recipe') or {})
    if not common:
        # Fallback only once EVERY registered capacity has a complete family
        # triple for that patient. Choosing while the capacity phase is still
        # running would freeze whichever recipe happened to finish first.
        capacity = audit.load_cards(args.root / 'human_v0310' / 'U1')
        for subject in sorted({c['subject'] for c in capacity}):
            rows = [c for c in capacity if c['subject'] == subject]
            picked = audit.common_recipe_by_inner(rows)
            complete = {r for r, fam in picked['per_recipe'].items() if set(fam) == {'F', 'L', 'N'}}
            if picked['recipe'] and complete >= set(RECIPES):
                common[subject] = picked['recipe']
    groups = {}
    for card in cards:
        if card.get('status') not in ('COMPLETE', 'WALL_TIME_LIMITED') or card['view'] != 'joint':
            continue
        if common.get(card['subject']) != card['recipe']:
            continue
        groups.setdefault((card['subject'], card['seed'], card['source_mode']), []).append(card)
    rows = []
    for key in sorted(groups):
        subject, seed, mode = key
        families = {c['family'] for c in groups[key]}
        if families != {'F', 'L', 'N'} and not args.allow_partial:
            # Choosing the family before the paired triple exists would let
            # "whichever arm finished first" decide the frozen state.
            rows.append(dict(subject=subject, seed=seed, source_mode=mode, status='INCOMPLETE_TRIPLE',
                             families_present=sorted(families)))
            continue
        chosen, how = pick_family(groups[key])
        if chosen is None:
            rows.append(dict(subject=subject, seed=seed, source_mode=mode, status='NO_EVENT_STAGE'))
            continue
        tag = f'{subject}_s{seed}_{mode}'
        record = dict(subject=subject, seed=seed, source_mode=mode, family=chosen['family'],
                      recipe=chosen['recipe'], upstream_card=chosen['_path'],
                      inner_selected=chosen['stages']['event']['selected_inner'],
                      family_candidates={c['family']: c['stages']['event']['selected_inner']
                                         for c in groups[key] if c.get('stages', {}).get('event')},
                      family_selection=how,
                      upstream_stop_reason=chosen['stages']['event']['stop_reason'],
                      upstream_training_sufficiency=chosen['training_sufficiency_vector']['verdict'])
        export = out / f'{tag}.npz'
        try:
            if not export.with_suffix('.json').exists():
                subprocess.run([sys.executable, str(ROOT / 'scripts/export_group_event_state_v0310_frozen_state.py'),
                                '--source', chosen['_path'], '--output', str(export)],
                               check=True, capture_output=True, text=True, cwd=str(ROOT))
            record['export'] = str(export)
            for name, script, extra in (
                    ('contact', 'scripts/train_group_event_state_v039_contact_transfer.py',
                     ['--max-steps', str(args.max_steps), '--foundation-steps', str(args.foundation_steps),
                      '--device', 'cpu', '--seed', str(seed)]),
                    ('expression', 'scripts/probe_group_event_state_v039_expression.py', [])):
                target = out / f'{tag}_{name}.json'
                if target.exists():
                    record[name] = str(target); continue
                proc = subprocess.run([sys.executable, str(ROOT / script), '--features', str(export),
                                       '--root', args.old_root, '--output', str(target)] + extra,
                                      capture_output=True, text=True, cwd=str(ROOT))
                if proc.returncode:
                    record[name] = 'FAILED'; record[name + '_error'] = (proc.stderr or proc.stdout)[-600:]
                else:
                    record[name] = str(target)
            record['status'] = 'COMPLETE' if record.get('contact', '').endswith('.json') else 'PARTIAL'
        except subprocess.CalledProcessError as error:
            record['status'] = 'FAILED'; record['error'] = (error.stderr or '')[-600:]
        rows.append(record)
        print(json.dumps({k: record.get(k) for k in ('subject', 'seed', 'source_mode', 'family', 'status')}),
              flush=True)
        if args.deadline_epoch and time.time() >= args.deadline_epoch:
            rows.append(dict(status='WALL_TIME_LIMITED', note='remaining M1 states not attempted'))
            break
    atomic_json(out / 'm1_transfer_index.json',
                dict(status='COMPLETE', allow_partial=bool(args.allow_partial), schema='v0310_m1_index_v1', timestamp=time.time(), rows=rows,
                     common_recipe=common,
                     family_rule='lowest selected INNER among F/L/N at the shared recipe the capacity '
                                 'phase chose on INNER, inside one (subject, seed, source mode) group; '
                                 'SELECTION is never consulted',
                     upstream_frozen=True, upstream_finetuned_for_contact_or_seizure=False,
                     note='joint count/coarse-recruitment training upstream; contact identity and fine '
                          'expression are untrained endpoints here, and a good jointly trained score is '
                          'not by itself evidence of untrained pathological expression transfer'))
    print(json.dumps(dict(status='COMPLETE', n_states=len(rows))), flush=True)


if __name__ == '__main__':
    main()
