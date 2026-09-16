#!/usr/bin/env python3
"""Read-only replay audit of the V4 component-coverage cycle.

Partial mode audits every round whose artifacts exist without touching the live
controller; final mode additionally requires the terminal capacity-review state
and the released controller lock. It never nominates, simulates, qualifies or
releases the Fig5 hold.
"""
from pathlib import Path
import argparse
import fcntl
import sys
import time
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_topic4_joint_xy_component_search as run
from src.topic4_xy_component_search import new_proposals, local_anchors, select_racers, COMPONENTS
from src.topic4_xy_replicated_anchors import incumbent, anchor_eligible


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def compact(row):
    return {**{k: row[k] for k in ('candidate_id', 'n_events', 'joint_distance', 'exploration_score', 'kernel_distances')},
            'D_support': row.get('D_support'), 'D_order': row.get('D_order'), 'D_lag': row.get('D_lag'),
            'direction_distance': row.get('direction_distance'),
            'centers_mm': row['candidate']['node_field']['centers_mm'],
            'proposal': row['candidate'].get('proposal'), 'proposal_round': row['candidate'].get('proposal_round'),
            'anchor_candidate_id': row['candidate'].get('anchor_candidate_id'),
            'n_networks': len(row['units']), 'seeds': sorted(u['seed'] for u in row['units'])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--final', action='store_true', help='require terminal state and controller lock')
    args = parser.parse_args()
    out = run.OUT; plan = run.read(run.CONFIG)
    rounds_planned = plan['search']['automatic_rounds_before_capacity_review']
    guard = None
    if args.final:
        guard = (out / 'controller.lock').open('r+')
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Controller is live; wait for the same process') from exc
        status = run.read(out / 'status.json')
        require(status.get('status') == 'NEEDS_MODEL_CAPACITY_REVIEW', 'Cycle has not reached the capacity-review terminal state')
        require(status.get('rounds') == rounds_planned and status.get('goal_remains_active') is True,
                'Terminal round count or incomplete-goal declaration differs')
        require(not (out / 'qualified_substrate.json').exists(), 'Qualified result requires a different audit')
    contract = run.read(out / 'objective_contract.json')
    locks = {**contract['source_hashes'], **run.read(out / 'analysis_input_lock.json')['hashes']}
    run.v1.runtime.verify_amendment(locks)
    prelaunch = run.read(out / 'prelaunch_verification.json')
    require(prelaunch['master_seed'] == contract['master_seed'], 'master seed differs from prelaunch verification')
    sources = [run.CONFIG, Path(__file__), out / 'baseline_scores.json', out / 'objective_contract.json',
               out / 'prelaunch_verification.json', out / 'replicated_bootstrap.json']
    baseline = run.read(out / 'baseline_scores.json')
    require(baseline['objective_version'] == plan['version'], 'baseline objective version differs')
    pool = baseline['candidates']
    require(len(pool) == 347 and len({r['candidate_id'] for r in pool}) == 347, 'reused V3 pool size differs from 347')
    bootstrap = run.read(out / 'replicated_bootstrap.json')
    require(bootstrap['actual_local_anchors'] == [r['candidate_id'] for r in local_anchors(pool, plan)],
            'bootstrap local anchors differ from replay')
    require(bootstrap['actual_local_anchors'] == prelaunch['anchors'], 'prelaunch anchors differ from bootstrap')
    calibration = run.read(out / 'patient_calibration.json')
    seeds_full = sorted(plan['search']['fit_seeds'] + plan['search']['race_seeds'])
    positions = run.v1.base.positions()
    worker_hashes = {}

    def verify_unit(unit):
        p = Path(unit['worker_path']); meta = run.read(p); npz = Path(meta['arrays']['path'])
        for artifact, digest in [(p, unit['worker_sha256']), (npz, meta['arrays']['sha256'])]:
            actual = worker_hashes.get(str(artifact))
            if actual is None:
                actual = run.sha(artifact)
            require(actual == digest, f'Changed trajectory: {artifact}')
            worker_hashes[str(artifact)] = digest
        require(meta['seed'] == unit['seed'], 'unit seed differs from worker metadata')
        return meta

    history, reviews = [], []
    completed_rounds = 0
    for number in range(1, rounds_planned + 1):
        stage = out / 'rounds' / f'{number:03d}'
        design_path = stage / 'design.json'
        if not design_path.exists():
            break
        best = incumbent(pool, plan)
        if best is not None:
            history.append(best['exploration_score'])
        stagnant = len(history) >= 3 and abs(history[-1] - history[-3]) < .001
        diagnosis = {'action': 'increase_random_restart_fraction' if stagnant else 'multi_anchor_local_plus_random'}
        anchors = local_anchors(pool, plan)
        design = run.read(design_path)
        expected = new_proposals(pool, positions, contract['master_seed'], number, plan, diagnosis, run.v1)
        require(design['candidates'] == expected, f'Round {number}: proposal replay differs')
        require(design['preceding_diagnosis'] == diagnosis, f'Round {number}: diagnosis differs')
        require(design['seed_sequence'] == [contract['master_seed'], number, 0], f'Round {number}: RNG sequence differs')
        require(design['local_anchor_candidates'] == [r['candidate_id'] for r in anchors], f'Round {number}: anchors differ')
        require(design['short_runs_eligible_as_local_anchors'] is False, 'short runs must not anchor')
        require(len(expected) == plan['search']['proposals_per_round'], 'proposal count differs')
        seen_before = {r['candidate']['node_field']['field_sha256'] for r in pool}
        require(not any(r['node_field']['field_sha256'] in seen_before for r in expected), 'duplicate geometry proposed')
        review = {'round': number, 'incumbent_before_round': None if best is None else compact(best),
                  'diagnosis': diagnosis, 'local_anchors': [compact(r) for r in anchors],
                  'proposal_counts': dict(Counter(x['proposal'] for x in expected)),
                  'proposals_per_anchor': dict(Counter(str(x['anchor_candidate_id']) for x in expected)),
                  'design_replayed_exactly': True}
        sources.append(design_path)
        scores_path = stage / 'scores.json'
        screen_completion = out / 'execution' / f'round_{number:03d}' / 'completion.json'
        if not (scores_path.exists() and screen_completion.exists()):
            review['stage'] = 'DESIGN_ONLY'; reviews.append(review); break
        scores = run.read(scores_path)
        require(scores['candidate_ids'] == [x['candidate_id'] for x in expected], 'Initial score candidates differ from design')
        scores = scores['candidates']
        for row in scores:
            require(sorted(u['seed'] for u in row['units']) == sorted(plan['search']['fit_seeds']), 'Initial network seeds differ')
            require(row['n_events'] == sum(u['observation']['n_groups'] for u in row['units']), 'Initial event count differs')
            for unit in row['units']:
                verify_unit(unit)
        completion = run.read(screen_completion); snapshot = screen_completion.parent / 'runtime_snapshot.json'
        require(completion['status'] == 'ALL_WORKERS_COMPLETE' and completion['jobs'] == len(expected) * len(plan['search']['fit_seeds']),
                f'Incomplete screen phase round_{number:03d}')
        require(run.sha(snapshot) == completion['runtime_snapshot_sha256'], 'Runtime snapshot changed')
        sources.extend([scores_path, screen_completion, snapshot])
        pool.extend(scores)
        review.update({'screen_verified_outputs': completion['jobs'],
                       'screen_event_count_range': [min(r['n_events'] for r in scores), max(r['n_events'] for r in scores)],
                       'screen_runaway_candidates': sum(any(u['runaway'] for u in r['units']) for r in scores),
                       'screen_invalid_geometry_candidates': sum(any(u['geometry']['minimum_clearance_mm'] < 0 or not u['geometry']['full_disks_disjoint']
                                                                    for u in r['units']) for r in scores),
                       'top_new_exploration': [compact(r) for r in sorted(scores, key=lambda r: r['exploration_score'])[:5]]})
        nomination_path = stage / 'race_nomination.json'
        if not nomination_path.exists():
            review['stage'] = 'SCREEN_COMPLETE_NOMINATION_PENDING'; reviews.append(review); break
        nomination = run.read(nomination_path)
        chosen = select_racers(pool, number, plan)
        require([x['candidate_id'] for x in chosen] == nomination['candidate_ids'], 'Nomination replay differs')
        require(nomination['selected_before_additional_simulations'] is True, 'nomination flag missing')
        require(nomination['additional_common_seeds'] == plan['search']['race_seeds'], 'race seeds differ')
        race_dir = out / 'execution' / f'race_{number:03d}' / 'workers'
        race_files = sorted(race_dir.glob('*.json')) if race_dir.exists() else []
        if race_files:
            require(nomination_path.stat().st_mtime <= min(p.stat().st_mtime for p in race_files),
                    'race workers predate nomination')
        # Component-specialist provenance of the nomination: same eligibility as select_racers
        # (two fit seeds, explorable, no runaway, complete non-overlapping cores).
        required = set(plan['search']['fit_seeds'])
        unexpanded = [r for r in pool if r['explorable'] and len(r['units']) == len(required)
                      and {u['seed'] for u in r['units']} == required
                      and all(not u['runaway'] and u['geometry']['minimum_clearance_mm'] >= 0
                              and u['geometry']['full_disks_disjoint'] for u in r['units'])]
        experts = {}
        for key in COMPONENTS:
            finite = [r for r in unexpanded if r['kernel_distances'].get(key) is not None]
            experts[key] = min(finite, key=lambda r: (r['kernel_distances'][key], r['candidate_id']))['candidate_id'] if finite else None
        review['screen_new_candidates_ineligible_for_nomination'] = [r['candidate_id'] for r in scores if r not in unexpanded]
        review.update({'nominees': [compact(r) for r in chosen], 'nomination_replayed_exactly': True,
                       'screen_component_experts': experts,
                       'experts_nominated': {k: v in nomination['candidate_ids'] for k, v in experts.items()},
                       'new_round_nominees': sum(r['candidate'].get('proposal_round') == number and str(r['candidate_id']).startswith('component_')
                                                 for r in chosen)})
        sources.append(nomination_path)
        analysis_path = stage / 'analysis.json'
        race_completion = out / 'execution' / f'race_{number:03d}' / 'completion.json'
        if not (analysis_path.exists() and race_completion.exists()):
            review['stage'] = 'RACE_RUNNING'; reviews.append(review); break
        analysis = run.read(analysis_path)
        require(len(analysis['expanded']) == len(chosen), 'Incomplete expanded analysis')
        require(analysis['loss_changed'] is False and analysis['geometry_axis_forced'] is False, 'analysis flags changed')
        completed = []
        for nominee, saved in zip(chosen, analysis['expanded']):
            path = stage / f'combined_{nominee["candidate_id"]}.json'
            row = run.read(path)['candidates'][0]; sources.append(path)
            require(row['candidate_id'] == saved['candidate_id'] == nominee['candidate_id'], 'Candidate order differs')
            require(sorted(u['seed'] for u in row['units']) == seeds_full, 'Common network seeds differ')
            require(len({u['seed'] for u in row['units']}) == len(seeds_full), 'duplicate seeds in combined row')
            require(row['n_events'] == sum(u['observation']['n_groups'] for u in row['units']), 'Event count differs')
            require(row['n_events'] == saved['n_events'] and row['joint_distance'] == saved['joint_distance'], 'saved summary differs')
            require(run.assess(row, calibration, plan) == saved['assessment'], 'Assessment replay differs')
            require(saved['assessment']['pass'] is False or args.final is False, 'passing candidate requires confirmation audit')
            for unit in row['units']:
                verify_unit(unit)
            completed.append(row)
        completion = run.read(race_completion); snapshot = race_completion.parent / 'runtime_snapshot.json'
        require(completion['status'] == 'ALL_WORKERS_COMPLETE' and completion['jobs'] == len(chosen) * len(plan['search']['race_seeds']),
                f'Incomplete race phase race_{number:03d}')
        require(run.sha(snapshot) == completion['runtime_snapshot_sha256'], 'Runtime snapshot changed')
        sources.extend([analysis_path, race_completion, snapshot])
        replacements = {x['candidate_id']: x for x in completed}
        pool = [replacements.get(x['candidate_id'], x) for x in pool]
        expanded_rows = []
        for nominee, row, saved in zip(chosen, completed, analysis['expanded']):
            # `nominee` is the two-network pool row selected before the extra simulations.
            expanded_rows.append({**compact(row), 'assessment': saved['assessment'],
                                  'events_per_seed': [{'seed': u['seed'], 'n_events': u['metrics']['n_events']} for u in sorted(row['units'], key=lambda u: u['seed'])],
                                  'initial_two_network': {k: nominee[k] for k in ('n_events', 'joint_distance', 'exploration_score', 'kernel_distances')}})
        review.update({'stage': 'ROUND_COMPLETE', 'race_verified_outputs': completion['jobs'],
                       'expanded': expanded_rows,
                       'qualified_count': sum(x['assessment']['pass'] for x in analysis['expanded']),
                       'failure_counts': {k: sum(not a['assessment']['checks'][k] for a in analysis['expanded'])
                                          for k in analysis['expanded'][0]['assessment']['checks']},
                       'incumbent_after_round': compact(incumbent(pool, plan))})
        reviews.append(review); completed_rounds += 1
    if args.final:
        require(completed_rounds == rounds_planned, 'not all planned rounds complete')
        require(not any(run.assess(x, calibration, plan)['pass'] for x in pool if anchor_eligible(x, plan)),
                'Training-eligible candidates require confirmation-attempt audit')
        require(not list((out / 'nominations').glob('confirmation_*.json')), 'Confirmation artifacts need separate audit')
        require(not (out / 'confirmation_attempts.json').exists(), 'Confirmation attempts need separate audit')
        require(not (out / 'rounds' / f'{rounds_planned + 1:03d}' / 'design.json').exists(), 'Unexpected next-round design')
        require(not (out / 'fig5_followup').exists(), 'unexpected downstream handoff folder')
    full = [r for r in pool if anchor_eligible(r, plan)]
    eight = [r for r in pool if len(r['units']) == len(seeds_full)]
    best = incumbent(pool, plan)
    result = {'status': 'EXHAUSTED_UNQUALIFIED_CYCLE_REPLAY_VERIFIED' if args.final else 'PARTIAL_CYCLE_REPLAY_VERIFIED',
              'updated_unix': time.time(), 'rounds_planned': rounds_planned, 'rounds_complete': completed_rounds,
              'n_geometries_seen': len(pool), 'n_new_geometries': len(pool) - 347,
              'n_eight_network_candidates': len(eight), 'n_anchor_eligible': len(full),
              'round_reviews': reviews,
              'incumbent': None if best is None else compact(best),
              'verified_lock_count': len(locks), 'verified_trajectory_artifacts': len(worker_hashes),
              'scientific_qualification': False, 'model_capacity_impossibility_proven': False,
              'goal_remains_active': True, 'fig5_hold_released': False, 'live_search_changed': False,
              'source_hashes': {**worker_hashes, **{str(p): run.sha(p) for p in sources}}}
    target = out / ('cycle_closure_review.json' if args.final else 'partial_cycle_replay_review.json')
    run.write(target, result)
    print({k: v for k, v in result.items() if k not in ('source_hashes', 'round_reviews')})
    for rv in reviews:
        print({k: rv.get(k) for k in ('round', 'stage', 'diagnosis', 'proposal_counts', 'proposals_per_anchor',
                                       'screen_event_count_range', 'experts_nominated', 'qualified_count', 'failure_counts')})


if __name__ == '__main__':
    main()
