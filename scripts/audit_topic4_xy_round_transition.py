#!/usr/bin/env python3
"""Replay completed training-round nominations and the next random proposal design."""
from pathlib import Path
import argparse
import sys
import time
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_topic4_joint_xy_replicated_search as run
from src.topic4_xy_replicated_anchors import proposal_pool, incumbent, anchor_eligible


def compact(row):
    return {k:row[k] for k in ['candidate_id','n_events','joint_distance','exploration_score']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', type=int, required=True)
    number = parser.parse_args().round
    if number < 1:
        raise ValueError('positive completed round required')
    out = run.OUT; plan = run.read(run.CONFIG)
    source_paths = [out/'baseline_scores.json']
    pool = run.read(source_paths[0])['candidates']
    history = []
    for i in range(1,number+1):
        previous = incumbent(pool,plan)
        history.append(previous['exploration_score'])
        stage = out/'rounds'/f'{i:03d}'
        score_path = stage/'scores.json'; source_paths.append(score_path)
        pool.extend(run.read(score_path)['candidates'])
        nomination_path = stage/'race_nomination.json'
        nomination = run.read(nomination_path)
        source_paths.append(nomination_path)
        assert [r['candidate_id'] for r in run.select_racers(pool,i,plan)] == nomination['candidate_ids']
        for phase, jobs in [(f'round_{i:03d}',48),(f'race_{i:03d}',36)]:
            completion_path = out/'execution'/phase/'completion.json'
            snapshot_path = completion_path.parent/'runtime_snapshot.json'
            completion = run.read(completion_path)
            assert completion['status'] == 'ALL_WORKERS_COMPLETE' and completion['jobs'] == jobs
            assert run.sha(snapshot_path) == completion['runtime_snapshot_sha256']
            source_paths.extend([completion_path,snapshot_path])
        completed = []
        for cid in nomination['candidate_ids']:
            path = stage/f'combined_{cid}.json';source_paths.append(path)
            row = run.read(path)['candidates'][0]
            assert sorted(u['seed'] for u in row['units']) == list(range(2511,2519))
            for unit in row['units']:
                worker = Path(unit['worker_path']);meta = run.read(worker)
                assert run.sha(worker) == unit['worker_sha256']
                assert run.sha(Path(meta['arrays']['path'])) == meta['arrays']['sha256']
            completed.append(row)
        replacements = {r['candidate_id']:r for r in completed}
        pool = [replacements.get(r['candidate_id'],r) for r in pool]
    current = incumbent(pool,plan);history.append(current['exploration_score'])
    diagnosis = {'action':'increase_random_restart_fraction' if len(history)>=3 and
                 abs(history[-1]-history[-3])<.001 else 'multi_anchor_local_plus_random'}
    design_path = out/'rounds'/f'{number+1:03d}'/'design.json'
    design = run.read(design_path)
    assert design['preceding_diagnosis'] == diagnosis
    contract = run.read(out/'objective_contract.json')
    regenerated = run.v1.new_proposals(proposal_pool(pool,plan),run.v1.base.positions(),
        contract['master_seed'],number+1,plan,diagnosis)
    for row in regenerated:
        row['candidate_id'] = row['candidate_id'].replace('joint_r','replicated_r')
    assert regenerated == design['candidates']
    anchors = []
    for row in sorted([r for r in pool if anchor_eligible(r,plan)],key=lambda r:r['exploration_score']):
        centers = np.asarray(row['candidate']['node_field']['centers_mm'])
        if all(min(np.linalg.norm(centers-a[1]),np.linalg.norm(centers-a[1][::-1]))>2 for a in anchors):
            anchors.append((row['candidate_id'],centers))
        if len(anchors)==4:
            break
    calibration = run.read(out/'patient_calibration.json')
    analysis_path = stage/'analysis.json';analysis = run.read(analysis_path)
    assert len(completed) == len(analysis['expanded'])
    for row, saved in zip(completed,analysis['expanded']):
        assert row['candidate_id'] == saved['candidate_id']
        assert run.assess(row,calibration,plan) == saved['assessment']
    locks = {**contract['source_hashes'],**run.read(out/'analysis_input_lock.json')['hashes']}
    run.v1.runtime.verify_amendment(locks)
    result = {'status':'COMPLETED_ROUND_AND_NEXT_DESIGN_REPLAY_VERIFIED','round':number,
        'updated_unix':time.time(),'n_geometries_seen':len(pool),
        'completed_common_seed_candidates':len(completed),
        'qualified_count':sum(a['assessment']['pass'] for a in analysis['expanded']),
        'previous_incumbent':compact(previous),'updated_incumbent':compact(current),
        'lowest_raw_distance_in_current_race':compact(min(completed,key=lambda r:r['joint_distance'])),
        'incumbent_exploration_score_history':history,
        'actual_local_anchors':[a[0] for a in anchors],
        'next_proposal_counts':dict(Counter(r['proposal'] for r in regenerated)),
        'next_round_unique_geometry_count':len({r['node_field']['field_sha256'] for r in regenerated}),
        'exact_proposal_replay':True,'exact_nomination_replay':True,'exact_assessment_replay':True,
        'initialization_seed_sequence':design['seed_sequence'],'next_diagnosis':diagnosis,
        'random_fraction_boundary':'Draw probability is not accepted quota after geometry rejection.',
        'verified_lock_count':len(locks),'loss_changed':False,'heldout_opened':False,
        'source_hashes':{str(p):run.sha(p) for p in source_paths+[design_path,analysis_path,run.CONFIG,Path(__file__)]}}
    run.write(stage/'round_transition_review.json',result)
    print({k:v for k,v in result.items() if k!='source_hashes'})


if __name__ == '__main__':
    main()
