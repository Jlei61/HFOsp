#!/usr/bin/env python3
"""Audit a completed screen without consuming its pending expansion outcomes."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_topic4_joint_xy_replicated_search as run


def compact(row):
    return {
        **{k: row[k] for k in ('candidate_id', 'n_events', 'joint_distance',
                               'exploration_score', 'kernel_distances')},
        'centers_mm': row['candidate']['node_field']['centers_mm'],
        'proposal': row['candidate'].get('proposal'),
        'core_clearances_mm': [u['geometry']['minimum_clearance_mm'] for u in row['units']],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', type=int, required=True)
    number = parser.parse_args().round
    if number < 1:
        raise ValueError('positive screen round required')
    paths = []

    def read(path):
        paths.append(path)
        return run.read(path)

    out = run.OUT
    plan = read(run.CONFIG)
    pool = read(out / 'baseline_scores.json')['candidates']
    for i in range(1, number + 1):
        stage = out / 'rounds' / f'{i:03d}'
        new = read(stage / 'scores.json')['candidates']
        pool.extend(new)
        nomination = read(stage / 'race_nomination.json')
        nominees = run.select_racers(pool, i, plan)
        assert [r['candidate_id'] for r in nominees] == nomination['candidate_ids']
        if i < number:
            replacements = {}
            for cid in nomination['candidate_ids']:
                row = read(stage / f'combined_{cid}.json')['candidates'][0]
                assert sorted(u['seed'] for u in row['units']) == list(range(2511, 2519))
                replacements[cid] = row
            pool = [replacements.get(r['candidate_id'], r) for r in pool]
    completion_path = out / 'execution' / f'round_{number:03d}' / 'completion.json'
    completion = read(completion_path)
    snapshot = completion_path.parent / 'runtime_snapshot.json'
    paths.append(snapshot)
    assert completion['status'] == 'ALL_WORKERS_COMPLETE'
    assert completion['jobs'] == sum(len(r['units']) for r in new) == 48
    assert run.sha(snapshot) == completion['runtime_snapshot_sha256']
    for row in new:
        assert sorted(u['seed'] for u in row['units']) == [2511, 2512]
        for unit in row['units']:
            worker = Path(unit['worker_path'])
            assert run.sha(worker) == unit['worker_sha256']
            meta = read(worker)
            array = Path(meta['arrays']['path'])
            paths.append(array)
            assert run.sha(array) == meta['arrays']['sha256']
    assert len({r['candidate_id'] for r in pool}) == len(pool)
    contract = run.read(out / 'objective_contract.json')
    locks = {**contract['source_hashes'], **run.read(out / 'analysis_input_lock.json')['hashes']}
    run.v1.runtime.verify_amendment(locks)
    lowest = min(new, key=lambda r: r['joint_distance'])
    result = {
        'status': 'SCREEN_COMPLETE_NOMINATION_REPLAY_VERIFIED', 'round': number,
        'updated_unix': time.time(), 'n_geometries_seen': len(pool),
        'new_geometries': len(new), 'verified_new_worker_outputs': completion['jobs'],
        'initial_event_count_range': [min(r['n_events'] for r in new), max(r['n_events'] for r in new)],
        'early_runaway_candidates': sum(any(u['runaway'] for u in r['units']) for r in new),
        'top_new_exploration': [compact(r) for r in sorted(new, key=lambda r: r['exploration_score'])[:5]],
        'nominees': [compact(r) for r in nominees], 'nomination_replayed_exactly': True,
        'new_nominees': sum(r['candidate_id'] in {n['candidate_id'] for n in new} for r in nominees),
        'extra_jobs': len(nominees) * len(nomination['additional_common_seeds']),
        'lowest_raw_new_candidate': compact(lowest),
        'lowest_raw_new_candidate_has_invalid_core_clearance': any(u['geometry']['minimum_clearance_mm'] < 0 for u in lowest['units']),
        'claim_boundary': 'Two-network screen only; no cross-network stability or substrate qualification established.',
        'verified_lock_count': len(locks), 'loss_changed': False, 'heldout_opened': False,
        'source_hashes': {str(p): run.sha(p) for p in paths + [Path(__file__)]},
    }
    run.write(stage / 'screen_review.json', result)
    print({k: v for k, v in result.items() if k != 'source_hashes'})


if __name__ == '__main__':
    main()
