#!/usr/bin/env python3
"""Compact live state of the approved overnight experiments and deliveries."""
import json,time
from pathlib import Path
import psutil

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/topic4_sef_hfo/autonomous_recovery_exploration_20260914'
SUBDIRS=['fast_threshold_round2','activity_global_pool_round3','stronger_M_redistribution_round4',
         'continuous_resource_recovery_round5','adaptation_capacity_round6',
         'preserved_global_gain_round7','paired_recurrence_confirmation_round8',
         'native_global_confirmation_round9','nativeZ_continuation_to60',
         'revisedZ_continuation_to60']

def read(path):return json.loads(path.read_text())

def main():
    batches={};complete=0;approved=0;returns=0;failure_count=0
    for root in [BASE]+[BASE/name for name in SUBDIRS]:
        if not (root/'status.json').exists():continue
        status=read(root/'status.json');auth=root/'dispatch_authorization.json'
        if auth.exists() and 'approved_names' in read(auth):names=read(auth)['approved_names']
        else:names=[v['name'] for v in read(root/'protocol.json')['initial_jobs']]
        approved+=len(names);runs={}
        for name in names:
            folder=root/'runs'/name;progress=folder/'progress.json'
            if not progress.exists():runs[name]=dict(status='PENDING');continue
            d=read(progress);tr=d.get('tracker',d);result=folder/'result.json'
            q=dict(status=d['status'],time_s=d.get('time_s',d.get('end_s')),
                entries=[v['onset_s'] for v in tr.get('entries',[])],
                recoveries=[v['confirmation_s'] for v in tr.get('recoveries',[])],
                complete=result.exists(),failure_file=(folder/'failure.json').exists(),
                producer_pid=d.get('pid'),stop_reason=tr.get('stop_reason'))
            if q['producer_pid']:
                q['producer_pid_live']=psutil.pid_exists(q['producer_pid'])
            complete+=q['complete'];returns+=bool(q['recoveries']);failure_count+=q['failure_file']
            runs[name]=q
        batches['round1' if root==BASE else root.name]=dict(
            status=status.get('status'),supervisor_pid=status.get('pid'),
            completed=sum(v.get('complete',False) for v in runs.values()),
            running=status.get('running',{}),pending=status.get('pending',[]),
            failed=status.get('failed',[]),runs=runs)
    dense={p.parent.name:read(p) for p in (BASE/'native_field_candidates_recurrence').glob('*/recorder_status.json')}
    output=dict(updated_at=time.time(),goal_deadline_epoch=1789433430,
        scientific_parameter_noise_conditions=46,extra_same_state_continuations=2,
        conservative_budget_used=48,remaining_budget=0,approved_jobs_in_snapshot=approved,
        total_completed=complete,operational_return_trajectories_including_continuation=returns,
        experiment_failure_files=failure_count,available_RAM_GiB=psutil.virtual_memory().available/2**30,
        batches=batches,dense_replays=dense,
        caution='Operational returns are not accepted interictal-repertoire recovery. A same-state continuation is not an independent replicate. Actual completed duration and stop reason must accompany each result.')
    path=BASE/'latest_review_snapshot.json';tmp=path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(output,indent=2)+'\n');tmp.replace(path)
    print('approved',approved,'complete',complete,'operational returns incl continuation',returns,'failures',failure_count)
    for name,b in batches.items():print(name,b['completed'],len(b['running']),b['failed'])
    for name,d in dense.items():print('DENSE',name,d['status'])

if __name__=='__main__':main()
