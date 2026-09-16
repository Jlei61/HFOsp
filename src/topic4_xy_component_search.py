"""Preserve participation, rank and timing specialists without changing the loss."""
import numpy as np
from src.topic4_xy_replicated_anchors import anchor_eligible

COMPONENTS = ('support', 'rank_space', 'timing_space')


def distance(a, b):
    x = np.asarray(a['candidate']['node_field']['centers_mm'])
    y = np.asarray(b['candidate']['node_field']['centers_mm'])
    return min(np.linalg.norm(x-y), np.linalg.norm(x-y[::-1]))


def select(rows, count, *, joint_first):
    """Unique extrema, followed by spatially diverse joint-score candidates."""
    chosen = []
    def add(row):
        if row is not None and len(chosen) < count and row['candidate_id'] not in {
                r['candidate_id'] for r in chosen}:
            chosen.append(row)
    ranked = sorted(rows, key=lambda r: (r['exploration_score'], r['candidate_id']))
    if joint_first:
        add(next(iter(ranked), None))
    for key in COMPONENTS:
        finite = [r for r in rows if r['kernel_distances'].get(key) is not None
                  and np.isfinite(r['kernel_distances'][key])]
        add(min(finite, key=lambda r: (r['kernel_distances'][key], r['candidate_id']), default=None))
    for row in ranked:
        if all(distance(row, other) > 2. for other in chosen):
            add(row)
    for row in ranked:
        add(row)
    return chosen


def local_anchors(pool, plan):
    return select([r for r in pool if anchor_eligible(r, plan)], 4, joint_first=True)


def select_racers(pool, number, plan):
    required = set(plan['search']['fit_seeds'])
    eligible = [r for r in pool if r['explorable'] and np.isfinite(r['exploration_score'])
                and len(r['units']) == len(required)
                and {u['seed'] for u in r['units']} == required
                and all(not u['runaway'] and u['geometry']['minimum_clearance_mm'] >= 0
                        and u['geometry']['full_disks_disjoint'] for u in r['units'])]
    return select(eligible, plan['search']['racers_per_round'], joint_first=False)


def new_proposals(pool, pos, seed, number, plan, diagnosis, base):
    """Same four-dimensional random draws and geometry rules; revised anchors."""
    cfg = plan['search']; rng = base.proposal_rng(seed, number)
    anchors = local_anchors(pool, plan)
    seen = {r['candidate']['node_field']['field_sha256'] for r in pool}
    fraction = cfg['restart_random_fraction'] if diagnosis['action'] == 'increase_random_restart_fraction' else cfg['random_fraction']
    rows = []
    for attempt in range(100000):
        random_start = not anchors or rng.random() < fraction
        if random_start:
            centers = rng.uniform(.75, 19.25, size=(2, 2)); parent = None
        else:
            parent = int(rng.integers(len(anchors)))
            centers = np.asarray(anchors[parent]['candidate']['node_field']['centers_mm']) + rng.normal(size=(2, 2))*rng.choice(cfg['local_scales_mm'])
        domain = 'interior' if rng.random() < cfg['interior_fraction'] else 'whole_sheet'
        centers = base.canonical_centers(centers)
        if not base.geometry_allowed(centers, pos, domain=domain):
            continue
        field = base.field_descriptor(centers)
        if field['field_sha256'] in seen:
            continue
        seen.add(field['field_sha256'])
        rows.append(base.base.decorate({'candidate_id': f'component_r{number:03d}_{len(rows):03d}',
            'domain': domain, 'proposal': 'fresh_uniform_restart' if random_start else 'random_multi_anchor_local',
            'proposal_round': number, 'random_master_seed': seed, 'anchor_index': parent,
            'anchor_candidate_id': None if parent is None else anchors[parent]['candidate_id'],
            'node_field': field, 'geometry': base.audit_geometry(pos, centers, 1499)}))
        if len(rows) == cfg['proposals_per_round']:
            return rows
    raise RuntimeError('proposal geometry exhausted')
