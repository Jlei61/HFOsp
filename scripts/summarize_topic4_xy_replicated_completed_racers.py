#!/usr/bin/env python3
"""Read-only scientific review of completed eight-network v3 candidates."""
from pathlib import Path
import argparse
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_topic4_joint_xy_replicated_search as run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', type=int, required=True)
    args = parser.parse_args()
    stage = run.OUT / 'rounds' / f'{args.round:03d}'
    source_paths = [run.OUT / 'baseline_scores.json'] + [
        run.OUT / 'rounds' / f'{i:03d}' / 'scores.json'
        for i in range(1, args.round + 1)
    ]
    initial = {r['candidate_id']: r for p in source_paths
               for r in run.read(p)['candidates']}
    nomination_path = stage / 'race_nomination.json'
    nomination = run.read(nomination_path)
    plan = run.read(run.CONFIG)
    expected = plan['search']['fit_seeds'] + plan['search']['race_seeds']
    directory = run.OUT / 'execution' / f'race_{args.round:03d}' / 'workers'
    folder = stage / 'completed_candidate_review'
    folder.mkdir(exist_ok=True)
    completed = []
    pending = []
    obj = None
    for cid in nomination['candidate_ids']:
        paths = [directory / f'{cid}_seed_{seed}.json'
                 for seed in nomination['additional_common_seeds']]
        if not all(p.exists() for p in paths):
            pending.append(cid)
            continue
        if obj is None:
            obj = run.KernelObjective(run.v1, run.OUT, run.KERNEL)
        source = {**initial[cid], 'units': initial[cid]['units'] + [
            {'worker_path': str(p), 'worker_sha256': run.sha(p)} for p in paths
        ]}
        row = run.score_saved([source], obj, plan, folder / f'{cid}.json')[0]
        if sorted(u['seed'] for u in row['units']) != sorted(expected):
            raise RuntimeError(f'{cid}: duplicate or non-prespecified network seeds')
        assessment = run.assess(row, run.read(run.OUT / 'patient_calibration.json'), plan)
        completed.append({
            'candidate_id': cid, 'centers_mm': row['candidate']['node_field']['centers_mm'],
            'n_networks': len(row['units']),
            'events_per_seed': [{'seed': u['seed'], 'n_events': u['metrics']['n_events']}
                                for u in row['units']],
            'initial_n_events': initial[cid]['n_events'], 'expanded_n_events': row['n_events'],
            'initial_joint_distance': initial[cid]['joint_distance'],
            'expanded_joint_distance': row['joint_distance'],
            'kernel_distances': row['kernel_distances'], 'assessment': assessment,
            'D_order': row['D_order'], 'D_lag': row['D_lag'],
            'direction_distance': row['direction_distance'],
        })
    result = {
        'status': 'COMPLETE_CANDIDATES_ONLY', 'round': args.round, 'updated_unix': time.time(),
        'rows': completed, 'pending_candidate_ids': pending,
        'nomination_changed': False, 'live_search_modified': False, 'heldout_opened': False,
        'claim_boundary': 'Prespecified training-network expansion, not independent confirmation. '
                          'Partial batch: no winner selection or capacity conclusion.',
        'source_hashes': {str(p): run.sha(p) for p in source_paths + [
            nomination_path, run.CONFIG, Path(__file__), run.OUT / 'patient_calibration.json']},
    }
    run.write(folder / 'summary.json', result)
    print(result)


if __name__ == '__main__':
    main()
