#!/usr/bin/env python3
"""Audit each approved trajectory from raw counts and assemble the night ledger.

Events and sampled cells are observations within a run, not independent repeats.
This observer never changes simulator parameters, state, or endpoint definitions.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import argparse,csv,json,time
from pathlib import Path
import numpy as np
import analyze_topic4_autonomous_recovery as common
import analyze_topic4_autonomous_events as events
import snapshot_topic4_autonomous_exploration as snapshot

BASE=snapshot.BASE

def audit(root,name):
    folder=root/'runs'/name
    job=common.read(root/'jobs'/(name+'.json'))
    data=common.load(folder,['time_ms','spikes_1ms','regions_1ms','slow_time_ms','Z','M'])
    if data is None:return dict(name=name,batch=root.name,status='NOT_OBSERVED')
    counts=data['spikes_1ms'].astype(float)
    regions=data['regions_1ms'].astype(float)
    geometry=np.load(root/'geometry.npz');nr=geometry['region_counts']
    assert np.array_equal(counts[:,0],regions[:,:3].sum(1))
    assert np.array_equal(counts[:,1],regions[:,3:].sum(1))
    assert len(counts)%10==0
    n=len(counts)//10
    # The frozen producer labels each complete 1 ms count by its bin centre.
    assert np.allclose(data['time_ms'],np.arange(len(counts))+.5,atol=1e-8,rtol=0)
    rr=regions.reshape(n,10,6).sum(1)/nr/.01
    r=counts.reshape(n,10,2).sum(1)/np.array([32000,8000])/.01
    rates=np.c_[r[:,0],rr[:,:3]]
    tracker=common.run.fresh_tracker()
    for i,row in enumerate(rates):common.run.track(tracker,row,(i+1)*.01)
    result=common.read(folder/'result.json') if (folder/'result.json').exists() else None
    if result:
        assert result['end_s']==n*.01,(name,result['end_s'],n*.01)
        for key in ['entries','recoveries']:
            assert result['tracker'][key]==tracker[key],(name,key)
    timing=events.temporal_audit(r[:,0],rr[:,:3],tracker['entries'],tracker['recoveries'])
    finite=events.finite_events(r[:,0],regions.reshape(n,10,6).sum(1),nr)
    first=tracker['entries'][0]['onset_s'] if tracker['entries'] else n*.01
    before=[e for e in finite if e['start_s']>=.2 and e['end_s']<first]
    between=[]
    if len(tracker['entries'])>=2:
        lo,hi=tracker['entries'][0]['confirmation_s'],tracker['entries'][1]['onset_s']
        between=[e for e in finite if e['start_s']>=lo and e['end_s']<hi]
    tail=rates[-min(1000,n):]
    category='No global high gate reached'
    if tracker['entries']:category='High without verified whole-network return'
    if tracker['recoveries']:category='Autonomous return without reentry'
    if tracker['recoveries'] and len(tracker['entries'])>=2:category='Autonomous return and reentry gate reached'
    if not tracker['entries'] and max(tail.mean(0)[1:3])>=50:
        category='No global high gate; persistent core activity in tail'
    out=dict(name=name,batch='round1' if root==BASE else root.name,source=str(folder),
        job=job,status=result['status'] if result else 'RUNNING',result_exists=result is not None,
        observed_s=n*.01,planned_horizon_s=job['horizon_s'],
        stop_reason=result['tracker']['stop_reason'] if result else None,
        scientific_category=category,
        first_onset_s=first if tracker['entries'] else None,
        entries=tracker['entries'],recoveries=tracker['recoveries'],temporal_audit=timing,
        preentry_finite_events=len(before),finite_events_between_first_two_gates=len(between),
        tail10s_rate_mean_allE_coreA_coreB_surround_Hz=tail.mean(0).tolist(),
        tail10s_quiet_fraction_allE_coreA_coreB_surround=(tail<5).mean(0).tolist(),
        end_Z_mean_coreA_coreB=data['Z'][-1,[0,5,6]].tolist(),
        end_M_current_mean=float(job['eta_m']*data['M'][-1,0]),
        counts_time_and_endpoint_QA='PASS',
        is_same_state_continuation='continuation' in name,
        statistical_unit='One fixed-network, parameter/noise trajectory. Conditions sharing a seed are paired counterfactuals, not independent noise repeats. Each continuation shares its original trajectory; two different noise seeds are not two network draws.')
    return out

def main(final=False):
    snapshot.main();live=common.read(BASE/'latest_review_snapshot.json')
    rows=[]
    for batch,info in live['batches'].items():
        root=BASE if batch=='round1' else BASE/batch
        for name in info['runs']:
            rows.append(audit(root,name))
            print(batch,name,rows[-1]['status'],rows[-1].get('observed_s'),flush=True)
    if final:assert all(r.get('result_exists') for r in rows),'A dispatched run still lacks its saved endpoint'
    output=BASE/'overnight_review';output.mkdir(exist_ok=True)
    report=dict(updated_at=time.time(),final=final,parameter_noise_conditions=46,
        same_state_continuations=2,conservative_budget_used=48,budget_limit=48,
        all_dispatched_have_endpoint=all(r.get('result_exists') for r in rows),
        actual_endpoints=sum(r.get('result_exists',False) for r in rows),
        experiment_failures=live['experiment_failure_files'],rows=rows,
        outcome_boundary='High-return-high is an operational rate sequence, not proof of patient compatibility, autonomous periodic attractors or a bifurcation. Native-Z and the rho-modified equation are separate hypotheses. No external reset or clamp was used.',
        human_review='PENDING')
    common.write(output/'trajectory_ledger.json',report)
    columns=['batch','name','status','observed_s','planned_horizon_s','stop_reason','scientific_category',
             'seed','eta_m','tau_M_s','tau_Z_s','mode','gamma','pool_gain','pool_tau_s',
             'pool_threshold_Hz','recovery_ratio','first_onset_s','entry_count','return_count',
             'preentry_finite_events','finite_events_between_first_two_gates',
             'tail_E_Hz','tail_A_Hz','tail_B_Hz','end_Z_mean','counts_time_and_endpoint_QA']
    with (output/'trajectory_ledger.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for row in rows:
            flat={k:row.get(k,row.get('job',{}).get(k)) for k in columns}
            flat.update(entry_count=len(row.get('entries',[])),return_count=len(row.get('recoveries',[])))
            if 'end_Z_mean_coreA_coreB' in row:
                flat['end_Z_mean']=row['end_Z_mean_coreA_coreB'][0]
                flat.update(dict(zip(['tail_E_Hz','tail_A_Hz','tail_B_Hz'],row['tail10s_rate_mean_allE_coreA_coreB_surround_Hz'][:3])))
            writer.writerow(flat)
    print('LEDGER',len(rows),report['actual_endpoints'],'final',final,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true')
    args=parser.parse_args();main(args.final)
