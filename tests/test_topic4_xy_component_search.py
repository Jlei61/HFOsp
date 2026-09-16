import copy
import json
from pathlib import Path
from src.topic4_xy_component_search import local_anchors, select_racers, new_proposals


def fixture():
    plan = {'search': {'fit_seeds': [1, 2], 'race_seeds': [3, 4, 5, 6, 7, 8],
                      'minimum_pool_events': 64, 'racers_per_round': 6}}
    rows = []
    for i, cid in enumerate(('joint', 'support', 'rank', 'timing', 'bad', 'lowN')):
        kernels = dict(support=10., rank_space=10., timing_space=10.)
        if i in (1, 2, 3): kernels[('support', 'rank_space', 'timing_space')[i-1]] = .01
        rows.append({'candidate_id': cid, 'candidate': {'node_field': {'centers_mm': [[i, 2], [i+5, 10]], 'field_sha256': cid}},
            'explorable': True, 'n_events': 63 if cid == 'lowN' else 80,
            'exploration_score': 0. if i == 0 else 1.+i, 'kernel_distances': kernels,
            'units': [{'seed': s, 'runaway': False, 'geometry': {'minimum_clearance_mm': 1., 'full_disks_disjoint': True}} for s in range(1, 9)]})
    return plan, rows


def test_specialists_survive_joint_score_and_low_fidelity_excluded():
    plan, rows = fixture()
    rows[-1]['kernel_distances']['support'] = 0.
    rows[-2]['units'] = rows[-2]['units'][:2]
    rows[-2]['kernel_distances']['timing_space'] = 0.
    before = copy.deepcopy(rows)
    assert [r['candidate_id'] for r in local_anchors(rows, plan)] == ['joint', 'support', 'rank', 'timing']
    assert rows == before


def test_nomination_reserves_support_and_rejects_bad_seed_or_geometry():
    plan, rows = fixture()
    for r in rows: r['units'] = r['units'][:2]
    rows[-1]['kernel_distances']['support'] = 0.
    rows[-1]['units'][1]['seed'] = 1
    rows[-2]['kernel_distances']['rank_space'] = 0.
    rows[-2]['units'][0]['geometry']['full_disks_disjoint'] = False
    chosen = select_racers(rows, 1, plan)
    assert [r['candidate_id'] for r in chosen[:3]] == ['support', 'rank', 'timing']
    assert len(chosen) == 4


def test_extrema_duplicates_do_not_reduce_coverage():
    plan, rows = fixture()
    rows[0]['kernel_distances'] = dict(support=0., rank_space=0., timing_space=0.)
    selected = local_anchors(rows, plan)
    assert len(selected) == len({r['candidate_id'] for r in selected}) == 4


def test_actual_geometry_replay_and_fresh_initializations():
    from scripts import run_topic4_joint_xy_adaptive as base
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root/'config/topic4_joint_xy_kernel_v4.json').read_text())
    plan['search']['proposals_per_round'] = 6
    pos = base.base.positions()
    diagnosis = {'action': 'multi_anchor_local_plus_random'}
    first = new_proposals([], pos, 602341, 1, plan, diagnosis, base)
    assert first == new_proposals([], pos, 602341, 1, plan, diagnosis, base)
    changed = new_proposals([], pos, 602342, 1, plan, diagnosis, base)
    assert {r['node_field']['field_sha256'] for r in first}.isdisjoint(r['node_field']['field_sha256'] for r in changed)
    assert all(r['proposal'] == 'fresh_uniform_restart' and r['geometry']['full_disks_disjoint'] for r in first)
