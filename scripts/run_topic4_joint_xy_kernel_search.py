#!/usr/bin/env python3
"""Versioned kernel search with common-seed racing and independent confirmation."""
from pathlib import Path
import argparse
import fcntl
import secrets
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_topic4_joint_xy_adaptive as v1
from src.topic4_joint_xy_kernel_objective import KernelObjective
OLD_V1=v1.OUT
OUT=ROOT/'results/topic4_sef_hfo/joint_rank_space_dual_core_search_v2'
CONFIG=ROOT/'config/topic4_joint_xy_kernel_v2.json'
KERNEL=OLD_V1/'kernel_qualification/kernel_contract.json'
SOURCES=['scripts/run_topic4_joint_xy_kernel_search.py','src/topic4_joint_xy_kernel_objective.py',
         'src/topic4_joint_xy_kernel.py','config/topic4_joint_xy_kernel_v2.json',
         'tests/test_topic4_joint_xy_kernel_objective.py']
read,write,sha=v1.read,v1.write,v1.sha


def assess(row,calibration,plan,confirmation=False):
    result=v1.is_qualified(row,calibration,plan,confirmation=confirmation)
    result['checks']['sufficient_events']=row['n_events']>=plan['search']['minimum_pool_events']
    # Decomposition must also match; no univariate cancellation inside the joint kernel.
    size=next((n for n in sorted(map(int,calibration['samples'])) if n>=row['n_events']),max(map(int,calibration['samples'])))
    floors=calibration['samples'][str(size)]['kernel_q95']
    result['checks'].update({f'kernel_{k}':row['kernel_distances'][k] is not None and row['kernel_distances'][k]<=q
                             for k,q in floors.items()})
    result['kernel_thresholds']=floors;result['pass']=all(result['checks'].values());return result


def score_saved(old_scores,obj,plan,path):
    if path.exists():
        scores=read(path)['candidates']
        for r in scores:
            for u in r['units']:
                p=Path(u['worker_path']);m=read(p)
                if sha(p)!=u['worker_sha256'] or sha(m['arrays']['path'])!=m['arrays']['sha256']:
                    raise RuntimeError('reused trajectory changed')
        return scores
    scores=[]
    for r in old_scores:
        pairs=[v1.read_worker(Path(u['worker_path']),obj,plan) for u in r['units']]
        tables,units=zip(*pairs);metric=obj.metrics(np.concatenate(tables))
        scores.append({'candidate':r['candidate'],'candidate_id':r['candidate_id'],'units':list(units),**metric,
                       'explorable':metric['joint_distance'] is not None and not any(u['runaway'] for u in units)})
    write(path,{'candidates':scores,'objective_version':plan['version']});return scores


def select_racers(pool,number,plan):
    eligible=[r for r in pool if r['explorable'] and len(r['units'])==2 and
              all(u['geometry']['minimum_clearance_mm']>=0 for u in r['units'])]
    chosen=[]
    def add(r):
        if r and r['candidate_id'] not in {x['candidate_id'] for x in chosen}:chosen.append(r)
    if number==0:
        manual=[r for r in eligible if r['candidate_id']=='control_historical_matched'];add(manual[0] if manual else None)
    for key in ('rank_space','timing_space'):
        add(min(eligible,key=lambda r:r['kernel_distances'][key],default=None))
    ranked=sorted(eligible,key=lambda r:r['exploration_score'])
    for r in ranked:
        c=np.asarray(r['candidate']['node_field']['centers_mm'])
        if all(min(np.linalg.norm(c-np.asarray(a['candidate']['node_field']['centers_mm'])),
                   np.linalg.norm(c-np.asarray(a['candidate']['node_field']['centers_mm'])[::-1]))>2. for a in chosen):add(r)
        if len(chosen)>=plan['search']['racers_per_round']:break
    for r in ranked:
        if len(chosen)>=plan['search']['racers_per_round']:break
        add(r)
    return chosen


def race(pool,number,obj,calibration,plan,lock):
    folder=OUT/'rounds'/f'{number:03d}';folder.mkdir(parents=True,exist_ok=True)
    path=folder/'race_nomination.json'
    if path.exists():
        ids=read(path)['candidate_ids'];chosen=[next(r for r in pool if r['candidate_id']==x) for x in ids]
    else:
        chosen=select_racers(pool,number,plan)
        write(path,{'candidate_ids':[r['candidate_id'] for r in chosen],
            'additional_common_seeds':plan['search']['race_seeds'],'selected_before_additional_simulations':True,
            'rule':'Joint-score diverse positions plus rank and timing optima; historical manual control in initial race.'})
    if not chosen:return pool
    phase=f'race_{number:03d}';rows=[r['candidate'] for r in chosen]
    v1.status('RUNNING_COMMON_SEED_EVENT_EXPANSION',round=number,phase=phase,candidate_ids=[r['candidate_id'] for r in chosen])
    v1.run_phase(phase,rows,plan['search']['race_seeds'],plan['search']['fit_duration_ms'],plan,lock)
    new=v1.score_candidates(rows,OUT/'execution'/phase/'workers',plan['search']['race_seeds'],obj,plan,folder/'race_added_scores.json')
    combined=[]
    for r,extra in zip(chosen,new):
        # Re-read and verify source trajectories; no resampling duplication to reach N.
        source={**r,'units':r['units']+extra['units']}
        combined.extend(score_saved([source],obj,plan,folder/f'combined_{r["candidate_id"]}.json'))
    replacements={r['candidate_id']:r for r in combined};pool=[replacements.get(r['candidate_id'],r) for r in pool]
    report={'round':number,'expanded':[{ 'candidate_id':r['candidate_id'],'n_events':r['n_events'],
        'joint_distance':r['joint_distance'],'assessment':assess(r,calibration,plan)} for r in combined],
        'next_action':'Continue fresh random starts and local proposals unless a pre-nominated geometry passes independent confirmation.',
        'loss_changed':False,'geometry_axis_forced':False,'n_candidates_seen':len(pool)}
    write(folder/'analysis.json',report)
    failures={k:sum(not a['assessment']['checks'][k] for a in report['expanded']) for k in report['expanded'][0]['assessment']['checks']}
    (folder/'analysis.md').write_text(f'# 第 {number} 轮分析\n\n补算 {len(combined)} 个不同位置候选，各加入相同的六个新网络 seed；实际事件数为 '+
        ', '.join(str(r['n_events']) for r in combined)+'。没有复制事件充数。\n\n未通过项目及候选数：'+str(failures)+
        '。事件不足与分布不匹配分别记录；损失已经通过患者 rank 打乱和时滞拉伸对照，但这不保证当前 VTH 模型有足够容量。'+
        '继续使用新的随机起点和多候选局部搜索；不能仅凭一轮失败断言局部最优或容量不足。\n')
    return pool


def confirm(pool,number,obj,calibration,plan,lock):
    attempt_path=OUT/'confirmation_attempts.json'
    attempts=read(attempt_path) if attempt_path.exists() else []
    eligible=[r for r in pool if assess(r,calibration,plan)['pass'] and r['candidate_id'] not in [a['candidate_id'] for a in attempts]]
    pending_path=OUT/'nominations'/f'confirmation_{len(attempts):03d}.json'
    if pending_path.exists():
        nomination=read(pending_path)
        if number<nomination['round']:return False
        nominee=next(r for r in pool if r['candidate_id']==nomination['candidate_id'])
    elif eligible:
        nominee=min(eligible,key=lambda r:r['exploration_score'])
        start=plan['confirmation']['seed_base']+10*len(attempts)
        nomination={'candidate_id':nominee['candidate_id'],'candidate':nominee['candidate'],'round':number,
            'seeds':list(range(start,start+6)),'phase':f'confirmation_{len(attempts):03d}','selected_before_new_networks':True}
        write(pending_path,nomination)
    else:return False
    phase=nomination['phase'];v1.status('CONFIRMING_PRENOMINATED_GEOMETRY',**nomination)
    v1.run_phase(phase,[nominee['candidate']],nomination['seeds'],plan['confirmation']['duration_ms'],plan,lock)
    result=v1.score_candidates([nominee['candidate']],OUT/'execution'/phase/'workers',nomination['seeds'],obj,plan,
                               OUT/'nominations'/f'{phase}_scores.json')[0]
    assessment=assess(result,calibration,plan,True)
    attempts.append({**nomination,'assessment':assessment});write(attempt_path,attempts)
    if not assessment['pass']:return False
    write(OUT/'qualified_substrate.json',{'candidate':result,'assessment':assessment,'selection_result':nominee,
        'phase':phase,'development_working_point':True,'uniqueness_established':False})
    v1.fig5_handoff(result,phase,plan,lock);return True


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--score-only',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);guard=open(OUT/'controller.lock','a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    v1.OUT=OUT;v1.base.ENV['LD_LIBRARY_PATH']='/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib'
    plan=read(CONFIG)
    if read(OLD_V1/'loss_revision_batch_guard.json')['status']!='BATCH_DRAIN_COMPLETE_V1_RETIRED_FOR_LOSS_REVIEW':
        raise RuntimeError('previous live batch has not drained')
    lock={**read(OLD_V1/'objective_contract.json')['source_hashes'],**read(OLD_V1/'analysis_input_lock.json')['hashes']}
    lock.update({str(ROOT/p):sha(ROOT/p) for p in SOURCES});lock[str(KERNEL)]=sha(KERNEL)
    for p in [OLD_V1/'baseline_scores.json', OLD_V1/'rounds/001/scores.json']:
        lock[str(p)]=sha(p)
    contract_path=OUT/'objective_contract.json'
    if contract_path.exists():
        contract=read(contract_path)
        if contract['source_hashes']!=lock:raise RuntimeError('v2 source/input changed; create new version')
    else:
        contract={'source_hashes':lock,'version':plan['version'],'master_seed':secrets.randbits(32),
            'heldout_opened':False,'created_unix':time.time(),'loss_frozen_before_geometry_ranking':True}
        write(contract_path,contract)
    v1.runtime.verify_amendment(lock)
    obj=KernelObjective(v1,OUT,KERNEL);calibration=obj.calibrate(plan)
    input_path=OUT/'analysis_input_lock.json'
    fixed={str(p):sha(p) for p in [OUT/'patient_calibration.json',contract_path]}
    if input_path.exists() and read(input_path)['hashes']!=fixed:raise RuntimeError('v2 calibration drift')
    if not input_path.exists():write(input_path,{'hashes':fixed})
    lock.update(fixed)
    if args.prepare_only:v1.status('PREPARED_KERNEL_SEARCH');return
    if (OUT/'qualified_substrate.json').exists():
        accepted=read(OUT/'qualified_substrate.json')
        if not assess(accepted['candidate'],calibration,plan,True)['pass']:raise RuntimeError('qualification drift')
        v1.fig5_handoff(accepted['candidate'],accepted['phase'],plan,lock);return
    sources=[OLD_V1/'baseline_scores.json',OLD_V1/'rounds/001/scores.json']
    old_scores=[r for p in sources for r in read(p)['candidates']]
    pool=score_saved(old_scores,obj,plan,OUT/'baseline_scores.json')
    if args.score_only:v1.status('KERNEL_BASELINE_RESCORED',n_candidates=len(pool));return
    pool=race(pool,0,obj,calibration,plan,lock)
    if confirm(pool,0,obj,calibration,plan,lock):return
    history=[];pos=v1.base.positions()
    for number in range(1,plan['search']['automatic_rounds_before_capacity_review']+1):
        folder=OUT/'rounds'/f'{number:03d}';folder.mkdir(parents=True,exist_ok=True);path=folder/'design.json'
        best=min(pool,key=lambda r:r['exploration_score']);history.append(best['exploration_score'])
        stagnant=len(history)>=3 and abs(history[-1]-history[-3])<.001
        diagnosis={'action':'increase_random_restart_fraction' if stagnant else 'multi_anchor_local_plus_random'}
        if path.exists():rows=read(path)['candidates']
        else:
            rows=v1.new_proposals(pool,pos,contract['master_seed'],number,plan,diagnosis)
            for r in rows:r['candidate_id']=r['candidate_id'].replace('joint_r','kernel_r')
            write(path,{'candidates':rows,'preceding_diagnosis':diagnosis,'seed_sequence':[contract['master_seed'],number,0]})
        phase=f'round_{number:03d}';v1.status('RUNNING_KERNEL_RANDOM_SEARCH',round=number,phase=phase)
        v1.run_phase(phase,rows,plan['search']['fit_seeds'],plan['search']['fit_duration_ms'],plan,lock)
        pool.extend(v1.score_candidates(rows,OUT/'execution'/phase/'workers',plan['search']['fit_seeds'],obj,plan,folder/'scores.json'))
        pool=race(pool,number,obj,calibration,plan,lock)
        if confirm(pool,number,obj,calibration,plan,lock):return
    v1.status('NEEDS_MODEL_CAPACITY_REVIEW',rounds=len(history),goal_remains_active=True,
              reason='Frozen qualified loss and randomized coverage exhausted this batch budget; diagnose observation/capacity before new version.')


if __name__=='__main__':
    try:main()
    except Exception as exc:
        v1.OUT=OUT;v1.status('FAILED_NEEDS_ENGINEERING_DIAGNOSIS',reason=str(exc));raise
