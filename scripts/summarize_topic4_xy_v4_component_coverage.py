#!/usr/bin/env python3
"""Read-only component-coverage summary of the V4 cycle for milestone review.

Builds the complete eight-network table (V3 reused + V4 new) with every frozen
patient threshold, anchor/child provenance, expert nomination coverage and the
component trade-off view. Descriptive only: no nomination, no qualification.
"""
from pathlib import Path
import argparse
import csv
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_topic4_joint_xy_component_search as run
from src.topic4_xy_component_search import COMPONENTS, distance
from src.topic4_xy_replicated_anchors import anchor_eligible, incumbent

KEYS = ('joint_distance', 'D_support', 'D_order', 'D_lag', 'direction_distance')


def main():
    parser = argparse.ArgumentParser(); parser.parse_args()
    out = run.OUT; plan = run.read(run.CONFIG)
    review_path = out / 'cycle_closure_review.json'
    if not review_path.exists():
        review_path = out / 'partial_cycle_replay_review.json'
    review = run.read(review_path)
    calibration = run.read(out / 'patient_calibration.json')
    sources = [run.CONFIG, Path(__file__), review_path, out / 'patient_calibration.json', out / 'baseline_scores.json']
    pool = run.read(out / 'baseline_scores.json')['candidates']
    v3_ids = {r['candidate_id'] for r in pool}
    n_full = len(plan['search']['fit_seeds']) + len(plan['search']['race_seeds'])
    v3_expanded = {r['candidate_id'] for r in pool if len(r['units']) == n_full}
    rounds_done = [rv for rv in review['round_reviews'] if rv.get('stage') == 'ROUND_COMPLETE']
    nominated = {}
    for rv in review['round_reviews']:
        stage = out / 'rounds' / f"{rv['round']:03d}"
        if (stage / 'scores.json').exists():
            sources.append(stage / 'scores.json'); pool.extend(run.read(stage / 'scores.json')['candidates'])
        if rv.get('stage') == 'ROUND_COMPLETE':
            for cid in run.read(stage / 'race_nomination.json')['candidate_ids']:
                path = stage / f'combined_{cid}.json'; sources.append(path)
                row = run.read(path)['candidates'][0]
                pool = [row if r['candidate_id'] == cid else r for r in pool]
                nominated[cid] = rv['round']
    by_id = {r['candidate_id']: r for r in pool}

    def cycle(cid):
        # Cycle in which the eight-network expansion happened, not the geometry's origin.
        return 'V3' if cid in v3_expanded else ('V4_new_geometry' if cid.startswith('component_') else 'V4_expanded_old_geometry')

    def thresholds(n):
        t = dict(run.v1.threshold_table(calibration['samples'], n))
        size = next((s for s in sorted(map(int, calibration['samples'])) if s >= n), max(map(int, calibration['samples'])))
        t.update({f'kernel_{k}': v for k, v in calibration['samples'][str(size)]['kernel_q95'].items()})
        t['calibration_bin'] = size
        return t

    def table_row(r):
        a = run.assess(r, calibration, plan); t = thresholds(r['n_events'])
        cand = r['candidate']
        row = {'candidate_id': r['candidate_id'], 'cycle': cycle(r['candidate_id']),
               'proposal': cand.get('proposal'), 'proposal_round': cand.get('proposal_round'),
               'anchor_candidate_id': cand.get('anchor_candidate_id'), 'expanded_in_v4_round': nominated.get(r['candidate_id']),
               'x1': cand['node_field']['centers_mm'][0][0], 'y1': cand['node_field']['centers_mm'][0][1],
               'x2': cand['node_field']['centers_mm'][1][0], 'y2': cand['node_field']['centers_mm'][1][1],
               'n_events': r['n_events'], 'n_networks': len(r['units']), 'calibration_bin': t['calibration_bin'],
               'anchor_eligible': anchor_eligible(r, plan), 'exploration_score': r['exploration_score']}
        for k in KEYS:
            row[k] = r[k]; row[f'{k}_threshold'] = t[k]; row[f'{k}_pass'] = a['checks'][k]
            row[f'{k}_ratio'] = None if r[k] is None else r[k] / t[k]
        for k in ('support', 'rank_space', 'timing_space', 'joint'):
            row[f'kernel_{k}'] = r['kernel_distances'][k]; row[f'kernel_{k}_threshold'] = t[f'kernel_{k}']
            row[f'kernel_{k}_pass'] = a['checks'][f'kernel_{k}']; row[f'kernel_{k}_ratio'] = r['kernel_distances'][k] / t[f'kernel_{k}']
        for k in ('conditional_support', 'no_runaway', 'two_complete_cores', 'sufficient_events'):
            row[k] = a['checks'][k]
        row['pass'] = a['pass']; row['n_failed_checks'] = sum(not v for v in a['checks'].values())
        return row

    eight = [r for r in pool if len(r['units']) == n_full]
    table = sorted((table_row(r) for r in eight), key=lambda x: x['joint_distance'] if x['joint_distance'] is not None else 9)
    csv_path = out / 'component_coverage_milestone_eight_network_table.csv'
    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)

    def compact(r):
        return {k: r[k] for k in ('candidate_id', 'n_events', 'joint_distance', 'exploration_score', 'kernel_distances', 'D_support', 'D_order', 'D_lag', 'direction_distance')}
    eligible = [r for r in pool if anchor_eligible(r, plan)]
    extrema = {}
    for label, rows in (('all', eligible), ('V3', [r for r in eligible if r['candidate_id'] in v3_expanded]),
                        ('V4', [r for r in eligible if r['candidate_id'] not in v3_expanded])):
        extrema[label] = {'n': len(rows)}
        if rows:
            extrema[label]['joint'] = compact(min(rows, key=lambda r: r['exploration_score']))
            for key in COMPONENTS:
                extrema[label][key] = compact(min(rows, key=lambda r: r['kernel_distances'][key]))
    # Component trade-off: does any candidate dominate the V3 joint incumbent on all three kernel components?
    v3_inc = incumbent([r for r in pool if r['candidate_id'] in v3_expanded], plan)
    dominators = [r['candidate_id'] for r in eligible if r['candidate_id'] != v3_inc['candidate_id'] and
                  all(r['kernel_distances'][k] <= v3_inc['kernel_distances'][k] for k in COMPONENTS)]
    worst_ratio = sorted(({'candidate_id': x['candidate_id'], 'cycle': x['cycle'],
                           'worst_kernel_component_ratio': max(x[f'kernel_{k}_ratio'] for k in COMPONENTS),
                           'kernel_ratios': {k: x[f'kernel_{k}_ratio'] for k in COMPONENTS}, 'joint_ratio': x['joint_distance_ratio']}
                          for x in table if x['anchor_eligible']), key=lambda x: x['worst_kernel_component_ratio'])[:8]
    # Anchor/child provenance per round.
    anchor_children = []
    for rv in review['round_reviews']:
        for a in rv.get('local_anchors', []):
            kids = [by_id[c['candidate_id']] for c in run.read(out / 'rounds' / f"{rv['round']:03d}" / 'design.json')['candidates']
                    if c.get('anchor_candidate_id') == a['candidate_id'] and c['candidate_id'] in by_id]
            parent = by_id.get(a['candidate_id'])
            entry = {'round': rv['round'], 'anchor_candidate_id': a['candidate_id'],
                     'anchor_role': [k for k in ('joint',) + COMPONENTS if extrema['all'].get(k, {}).get('candidate_id') == a['candidate_id']],
                     'n_children': len(kids), 'children_nominated': [k['candidate_id'] for k in kids if k['candidate_id'] in nominated],
                     'children_distance_mm': [float(distance(parent, k)) for k in kids] if parent else [],
                     'children_latest_joint': [k['joint_distance'] for k in kids], 'children_n_networks': [len(k['units']) for k in kids],
                     'children_latest_kernel': [k['kernel_distances'] for k in kids],
                     'anchor_kernel': None if parent is None else parent['kernel_distances']}
            anchor_children.append(entry)
    expert_coverage = [{'round': rv['round'], 'experts': rv.get('screen_component_experts'), 'experts_nominated': rv.get('experts_nominated')}
                       for rv in review['round_reviews'] if rv.get('screen_component_experts')]
    incumbents = [{'round': rv['round'], 'before': rv['incumbent_before_round'], 'after': rv.get('incumbent_after_round')} for rv in review['round_reviews']]
    screen_stability = []
    for rv in rounds_done:
        for x in rv['expanded']:
            screen_stability.append({'round': rv['round'], 'candidate_id': x['candidate_id'],
                                     'screen_joint': x['initial_two_network']['joint_distance'], 'eight_joint': x['joint_distance'],
                                     'screen_kernel': x['initial_two_network']['kernel_distances'], 'eight_kernel': x['kernel_distances']})
    result = {'status': 'COMPONENT_COVERAGE_MILESTONE_SUMMARY_NOT_MODEL_QUALIFICATION', 'updated_unix': time.time(),
              'review_source': str(review_path), 'review_status': review['status'], 'rounds_complete': len(rounds_done),
              'n_geometries': len(pool), 'n_eight_network': len(eight), 'n_anchor_eligible': len(eligible),
              'n_eight_network_V4': sum(1 for r in eight if r['candidate_id'] not in v3_expanded),
              'cycle_definition': 'cycle = where the eight-network expansion happened; V4_expanded_old_geometry = V3-era geometry first expanded in V4',
              'qualified_any': any(x['pass'] for x in table), 'extrema': extrema,
              'v3_incumbent': compact(v3_inc), 'candidates_dominating_v3_incumbent_on_all_kernel_components': dominators,
              'lowest_worst_component_ratio': worst_ratio, 'incumbent_trajectory': incumbents,
              'anchor_children': anchor_children, 'expert_coverage': expert_coverage, 'screen_vs_eight_network': screen_stability,
              'eight_network_table_csv': str(csv_path), 'scientific_qualification': False, 'fig5_hold_released': False,
              'claim_boundary': 'Shared training seeds across candidates are not independent replications; ratios above one fail the frozen patient tolerance; descriptive only.',
              'source_hashes': {str(p): run.sha(p) for p in sources}}
    run.write(out / 'component_coverage_milestone.json', result)
    print({k: result[k] for k in ('review_status', 'rounds_complete', 'n_geometries', 'n_eight_network', 'n_eight_network_V4', 'n_anchor_eligible', 'qualified_any')})
    print('extrema all:', {k: (v['candidate_id'], round(v['joint_distance'], 6)) for k, v in extrema['all'].items() if isinstance(v, dict)})
    print('dominators of V3 incumbent:', dominators)
    for x in worst_ratio[:5]:
        print('worst-ratio', x['candidate_id'], x['cycle'], round(x['worst_kernel_component_ratio'], 2), {k: round(v, 2) for k, v in x['kernel_ratios'].items()})


if __name__ == '__main__':
    main()
