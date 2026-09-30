#!/usr/bin/env python3
"""Complete-only held-state summaries for the two targeted K conditions."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,time
import numpy as np
from campaign import ROOT,read,write,sha
from hold_carried_exit_states import OUT,JOBS
from analyze_high_state_continuation import load,analysis,ADAPTED


def main(wait):
    dest=OUT/'analysis';dest.mkdir(exist_ok=True);assert not (dest/'result.json').exists()
    geo=dict(np.load(ADAPTED/'geometry.npz'));rows=[];done=[]
    E=geo['population']==0;sizes=geo['group_size'];regions=geo['group_region']
    masks=[E]+[E&(regions==j) for j in range(3)]
    counts=np.bincount(geo['group_cell'][E],weights=sizes[E],minlength=400)
    while True:
        for name in JOBS:
            if name in done or not (OUT/name/'result.json').exists():continue
            result=read(OUT/name/'result.json');assert result['status']=='COMPLETE'
            d=load(OUT/name,geo);row=analysis(d,name)
            rows3=d['rate_Hz'][-3000:].reshape(3,1000,4).mean(1)
            f=d['field_Hz'][-3000:].reshape(3,1000,400).mean(1)
            with np.load(OUT/name/'stationary_candidate_observations.npz') as z:
                targetM=z['mean_target_M'];group=z['mean_group_output'];each=z['per_second_group_output']
            projection=np.bincount(geo['cell_group'],weights=targetM,minlength=len(sizes))/sizes
            m_error=float(np.sqrt(np.average((projection[E]-group[2,E])**2,weights=sizes[E])))
            row.update(K=JOBS[name],hold_at_least_s=8.,last_three_1s_regional_rates_Hz=rows3.tolist(),
                final1s_minus_prior1s_field_RMS_Hz=float(np.sqrt(np.average((f[-1]-f[-2])**2,weights=counts))),
                numerical_M_projection_RMS_Hz=m_error,
                M_minus_rate_regional_E=[float(np.average((projection-group[0])[m],weights=sizes[m])) for m in masks],
                both_cores_active_in_last3s=bool((rows3[:,1:3]>300).all()),
                both_cores_low_in_last3s=bool((rows3[:,1:3]<5).all()),
                stationary_root_or_stability_certified=False)
            np.savez_compressed(dest/f'{name}.npz',**d)
            write(dest/f'{name}.json',row);rows.append(row);done.append(name)
            print('HELD EXIT REVIEW',row,flush=True)
        status=read(OUT/'status.json') if (OUT/'status.json').exists() else {}
        if len(done)==2 or (status.get('status')=='FAILED' and not status.get('active')):break
        write(dest/'progress.json',dict(status='WAITING_TWO_HELD_STATES',pid=os.getpid(),completed=done,updated_epoch=time.time()))
        if not wait:return
        time.sleep(20)
    write(dest/'result.json',dict(status='COMPLETE' if len(done)==2 else 'STOPPED_ON_WORKER_FAILURE',rows=rows,
        scope='Two carried histories held8s after a forcedparameter approach. Last-window constancy is a descriptive check, not an equilibrium/stability proof. Zdrift remains counterfactual under its clamp.',
        formal_bifurcation_allowed=False,producer_sha256=sha(__file__)))
    write(dest/'progress.json',dict(status='COMPLETE' if len(done)==2 else 'STOPPED_ON_WORKER_FAILURE',completed=done,updated_epoch=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');main(p.parse_args().wait)
