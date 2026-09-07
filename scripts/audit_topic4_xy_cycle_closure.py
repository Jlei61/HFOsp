#!/usr/bin/env python3
"""Audit an exhausted, unqualified search cycle without inventing a next round."""
from pathlib import Path
import fcntl
import sys
import time
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_topic4_joint_xy_replicated_search as run
from src.topic4_xy_replicated_anchors import proposal_pool, incumbent


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def terminal_status(status, planned_rounds):
    require(status.get('status') == 'NEEDS_MODEL_CAPACITY_REVIEW',
            'Cycle has not reached the capacity-review terminal state')
    require(status.get('rounds') == planned_rounds and status.get('goal_remains_active') is True,
            'Terminal round count or incomplete-goal declaration differs')


def main():
    out = run.OUT
    # Acquire the actual controller lock: a status file alone cannot prove it stopped.
    with (out / 'controller.lock').open('r+') as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Controller is live; wait for the same process') from exc
        plan = run.read(run.CONFIG)
        rounds = plan['search']['automatic_rounds_before_capacity_review']
        terminal_status(run.read(out / 'status.json'), rounds)
        require(not (out / 'qualified_substrate.json').exists(), 'Qualified result requires a different audit')
        contract = run.read(out / 'objective_contract.json')
        locks = {**contract['source_hashes'], **run.read(out / 'analysis_input_lock.json')['hashes']}
        run.v1.runtime.verify_amendment(locks)
        sources = [run.CONFIG, Path(__file__), out / 'status.json', out / 'baseline_scores.json']
        pool = run.read(sources[-1])['candidates']
        history, reviews, worker_hashes = [], [], {}
        def verify_unit(unit):
            p = Path(unit['worker_path']); meta = run.read(p); npz = Path(meta['arrays']['path'])
            for artifact, digest in [(p, unit['worker_sha256']), (npz, meta['arrays']['sha256'])]:
                actual = worker_hashes.get(str(artifact))
                if actual is None:
                    actual = run.sha(artifact)
                require(actual == digest, f'Changed trajectory: {artifact}')
                worker_hashes[str(artifact)] = digest

        calibration = run.read(out / 'patient_calibration.json')
        seeds = sorted(plan['search']['fit_seeds'] + plan['search']['race_seeds'])
        positions = run.v1.base.positions()
        for number in range(1, rounds + 1):
            previous = incumbent(pool, plan)
            history.append(previous['exploration_score'])
            diagnosis = {'action': 'increase_random_restart_fraction' if len(history) >= 3 and
                         abs(history[-1] - history[-3]) < .001 else 'multi_anchor_local_plus_random'}
            stage = out / 'rounds' / f'{number:03d}'
            design_path = stage / 'design.json'; design = run.read(design_path)
            expected = run.v1.new_proposals(proposal_pool(pool, plan), positions,
                                            contract['master_seed'], number, plan, diagnosis)
            for row in expected:
                row['candidate_id'] = row['candidate_id'].replace('joint_r', 'replicated_r')
            require(design['candidates'] == expected and design['preceding_diagnosis'] == diagnosis,
                    f'Round {number}: proposal replay differs')
            require(design['seed_sequence'] == [contract['master_seed'], number, 0], 'RNG sequence differs')
            scores_path = stage / 'scores.json'; scores = run.read(scores_path)['candidates']
            require([x['candidate_id'] for x in scores] == [x['candidate_id'] for x in expected],
                    'Initial score candidates differ from design')
            for row in scores:
                require(sorted(u['seed'] for u in row['units']) == sorted(plan['search']['fit_seeds']),
                        'Initial network seeds differ')
                require(row['n_events'] == sum(u['observation']['n_groups'] for u in row['units']),
                        'Initial event count differs')
                for unit in row['units']:
                    verify_unit(unit)
            pool.extend(scores)
            nomination_path = stage / 'race_nomination.json'; nomination = run.read(nomination_path)
            chosen = run.select_racers(pool, number, plan)
            require([x['candidate_id'] for x in chosen] == nomination['candidate_ids'], 'Nomination replay differs')
            analysis_path = stage / 'analysis.json'; analysis = run.read(analysis_path)
            require(len(analysis['expanded']) == len(chosen), 'Incomplete expanded analysis')
            completed = []
            for nominee, saved in zip(chosen, analysis['expanded']):
                path = stage / f'combined_{nominee["candidate_id"]}.json'
                row = run.read(path)['candidates'][0]; sources.append(path)
                require(row['candidate_id'] == saved['candidate_id'] == nominee['candidate_id'], 'Candidate order differs')
                require(sorted(u['seed'] for u in row['units']) == seeds, 'Common network seeds differ')
                require(row['n_events'] == sum(u['observation']['n_groups'] for u in row['units']), 'Event count differs')
                require(run.assess(row, calibration, plan) == saved['assessment'], 'Assessment replay differs')
                for unit in row['units']:
                    verify_unit(unit)
                completed.append(row)
            for phase, jobs in [(f'round_{number:03d}', len(expected) * len(plan['search']['fit_seeds'])),
                                (f'race_{number:03d}', len(chosen) * len(plan['search']['race_seeds']))]:
                completion_path = out / 'execution' / phase / 'completion.json'
                snapshot_path = completion_path.parent / 'runtime_snapshot.json'
                completion = run.read(completion_path)
                require(completion['status'] == 'ALL_WORKERS_COMPLETE' and completion['jobs'] == jobs,
                        f'Incomplete execution phase {phase}')
                require(run.sha(snapshot_path) == completion['runtime_snapshot_sha256'], 'Runtime snapshot changed')
                sources.extend([completion_path, snapshot_path])
            replacements = {x['candidate_id']: x for x in completed}
            pool = [replacements.get(x['candidate_id'], x) for x in pool]
            reviews.append({'round': number, 'proposal_counts': dict(Counter(x['proposal'] for x in expected)),
                            'nominees': nomination['candidate_ids'],
                            'qualified_count': sum(x['assessment']['pass'] for x in analysis['expanded'])})
            sources.extend([design_path, scores_path, nomination_path, analysis_path])
        # Fail closed if independent confirmation needs a separate evidence chain.
        require(not any(run.assess(x, calibration, plan)['pass'] for x in pool),
                'Training-eligible candidates require confirmation-attempt audit')
        require(not list((out / 'nominations').glob('confirmation_*.json')),
                'Confirmation artifacts need separate audit')
        require(not (out / 'confirmation_attempts.json').exists(), 'Confirmation attempts need separate audit')
        require(not (out / 'rounds' / f'{rounds + 1:03d}' / 'design.json').exists(), 'Unexpected next-round design')
        best = incumbent(pool, plan)
        result = {'status': 'EXHAUSTED_UNQUALIFIED_CYCLE_REPLAY_VERIFIED', 'updated_unix': time.time(),
                  'rounds': rounds, 'n_geometries_seen': len(pool), 'round_reviews': reviews,
                  'incumbent': {k: best[k] for k in ('candidate_id', 'n_events', 'joint_distance', 'exploration_score')},
                  'verified_lock_count': len(locks), 'verified_trajectory_artifacts': len(worker_hashes),
                  'scientific_qualification': False, 'model_capacity_impossibility_proven': False,
                  'goal_remains_active': True, 'fig5_hold_released': False,
                  'source_hashes': {**worker_hashes, **{str(p): run.sha(p) for p in sources}}}
        run.write(out / 'cycle_closure_review.json', result)
        print({k: v for k, v in result.items() if k not in ('source_hashes', 'round_reviews')})


if __name__ == '__main__':
    main()
