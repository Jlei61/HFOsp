"""Pairing and aggregation rules for v0.3.10 cells.

Clause C15 t4: cells that do not share source code, data, targets, split,
recipe, horizon, view and source mode are not comparable and must not be
aggregated. Clause C15 t3: any budget- or wall-limited arm forbids a
sufficiency claim for the whole group.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

GROUP_KEYS = ('subject', 'seed', 'recipe', 'history_hours', 'view', 'source_mode')
IDENTITY_KEYS = ('data_sha256', 'split_sha256', 'target_sha256', 'input_dim', 'context_dim', 'n_recruitment')
FAMILIES = ('F', 'L', 'N')


def load_cards(root, pattern='**/card.json'):
    cards = []
    for path in sorted(Path(root).glob(pattern)):
        try:
            card = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if card.get('schema') != 'v0310_human_cell_v1':
            continue
        card['_path'] = str(path)
        cards.append(card)
    return cards


def group_key(card):
    return tuple(card[k] for k in GROUP_KEYS)


def assert_pairable(cards):
    """Raise unless the cards form one comparable arm set."""
    if not cards:
        raise ValueError('No cards to pair')
    keys = {group_key(c) for c in cards}
    if len(keys) != 1:
        raise ValueError(f'Cards span several groups: {sorted(keys)}')
    for field in IDENTITY_KEYS:
        values = {json.dumps(c.get(field), sort_keys=True) for c in cards}
        if len(values) != 1:
            raise ValueError(f'Refusing to pair: {field} differs across arms ({values})')
    sources = {json.dumps(c.get('source_hashes'), sort_keys=True) for c in cards}
    if len(sources) != 1:
        raise ValueError('Refusing to pair: cells were produced by different trainer source')
    weights = {json.dumps({k: round(v, 12) for k, v in c['objective_weights'].items()
                           if isinstance(v, float)}, sort_keys=True) for c in cards}
    if len(weights) != 1:
        raise ValueError('Refusing to pair: multi-task weights differ across arms')
    families = [c['family'] for c in cards]
    if len(set(families)) != len(families):
        raise ValueError('Duplicate family inside one paired group')
    for stage in ('background', 'event', 'refitted_constant'):
        digests = {c['stages'][stage]['batch_schedule_sha256'] for c in cards if stage in c.get('stages', {})}
        if len(digests) > 1:
            raise ValueError(f'Refusing to pair: {stage} batch schedules differ across arms')
    return True


def group_cards(cards):
    grouped = {}
    for card in cards:
        grouped.setdefault(group_key(card), []).append(card)
    return grouped


def paired_contrast(cards, better, worse, lead='2'):
    """Loss margin of `better` over `worse` on the common scored support."""
    by_family = {c['family']: c for c in cards}
    if better not in by_family or worse not in by_family:
        return dict(margin=None, status='INCOMPLETE_PAIR',
                    present=sorted(by_family), missing=[f for f in (better, worse) if f not in by_family])
    assert_pairable([by_family[better], by_family[worse]])
    values = {}
    for family in (better, worse):
        metric = by_family[family]['metrics'].get(lead, {})
        arm = metric.get('arms', {}).get('state') if metric.get('status') == 'SCORED' else None
        if arm is None:
            return dict(margin=None, status='ARM_NOT_SCORED', family=family)
        values[family] = arm['window_equal_weight']
    limited = any(by_family[f]['training_sufficiency_vector']['any_arm_budget_limited'] or
                  by_family[f]['training_sufficiency_vector']['any_arm_wall_time_limited']
                  for f in (better, worse))
    return dict(margin=float(values[worse] - values[better]), status='PAIRED',
                better=better, worse=worse, lead_hours=float(lead),
                aggregation='two-hour physical window equal weight',
                any_arm_limited=bool(limited),
                training_sufficiency='NOT_ESTABLISHED_BY_STOP_REASON')


def common_recipe_by_inner(cards_for_subject):
    """U1 rule: pick the recipe with the best mean INNER across F/L/N.

    Exact ties take the smaller state width. Selection reads INNER only; the
    SELECTION partition is never consulted.
    """
    per_recipe = {}
    for card in cards_for_subject:
        if card.get('status') not in ('COMPLETE', 'WALL_TIME_LIMITED'):
            continue
        event = card.get('stages', {}).get('event')
        if not event:
            continue
        per_recipe.setdefault(card['recipe'], {})[card['family']] = dict(
            inner=event['selected_inner'], width=card['state_width'], limited=event['budget_limited'])
    complete = {r: v for r, v in per_recipe.items() if set(v) == set(FAMILIES)}
    if not complete:
        return dict(recipe=None, status='NO_COMPLETE_FAMILY_TRIPLE', per_recipe=per_recipe)
    ranked = sorted(complete.items(),
                    key=lambda kv: (float(np.mean([v['inner'] for v in kv[1].values()])),
                                    min(v['width'] for v in kv[1].values())))
    return dict(recipe=ranked[0][0], status='SELECTED_ON_INNER',
                mean_inner={r: float(np.mean([v['inner'] for v in fam.values()])) for r, fam in complete.items()},
                any_arm_budget_limited=any(v['limited'] for v in complete[ranked[0][0]].values()),
                per_recipe=per_recipe, rule='mean INNER across F/L/N, ties to the smaller width; INNER only')
