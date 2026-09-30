#!/usr/bin/env python3
"""Original full-window A4 readouts of the new density diagnostic."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys,argparse,time
import numpy as np
from campaign import ROOT,REPO,read,write,sha
sys.path.insert(0,str(REPO/'scripts/topic4_zm_runaway_mechanism/frozen_v3'))
from native_readouts import readouts,window_stats

OUT=ROOT/'density_spatial_onset'
BASE=REPO/'results/topic4_sef_hfo/fig5_zm_rate_v3_20260918'
WINDOWS=[(500,3000),(1000,4000),(4000,8000),(8000,9420),(1000,9420)]


def audit():
    source=read(BASE/'native_reference/checkpoint_projections.json');nD=source['9870']['D'];rows=[]
    count=np.load(REPO/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz')['cell_e_counts']
    for job in read(OUT/'contract.json')['jobs']:
        folder=OUT/job['name']
        if not (folder/'result.json').exists():continue
        assert read(folder/'result.json')['status']=='COMPLETE'
        with np.load(folder/'trajectory.npz') as z:
            t=z['time_ms'];assert np.array_equal(t,np.arange(12500)+1.) and np.array_equal(count,z['cell_counts'])
            field=z['field_E_Hz'].astype(float);events,summary,whole,sm=readouts(t,field,count,job['name'])
            w=z['group_sizes'][z['population_E']]/32000
            physical=z['group_rate_Hz'][:,z['population_E']]@w
            field_error=float(abs(whole-physical).max());assert field_error<1e-4
            D=1-z['group_Z'][:,z['population_E']]@w
            bounds={}
            for label,mask,span,limit in [('E',z['population_E'],2,1000.),('I',~z['population_E'],1,1000.)]:
                rates=z['group_rate_Hz'][:,mask].astype(float)
                integrated=rates[:-1]+rates[1:] if span==2 else rates
                excess=float((integrated-limit).max());assert excess<1e-3
                bounds[label]=dict(window_ms=span,max_sum_rate_excess=excess)
            checkpoint={k:float(D[int(k)-1]) for k in source};mfinal=float(np.average(z['group_M'][-1,z['population_E']],weights=w))
        primary=window_stats(events,1000,9420);complete=[]
        for ev in events:
            a=int(np.searchsorted(t,ev['start_ms']));b=a+int(ev['duration_ms'])
            if a>=20 and b+20<=len(sm) and (sm[a-20:a]<5).all() and (sm[b:b+20]<5).all():complete.append(ev)
        n=primary['n'];duration=primary.get('median_duration_ms');area=primary.get('median_area');extent=primary.get('median_extent_mm')
        gate=dict(self_limited_events=bool(n>0 and duration is not None and 50<=duration<=200 and summary['quiet_fraction']>=.15),
            two_core_participation=bool(n>0 and primary['both_cores']/n>=.5),
            surround_recruitment=bool(area is not None and .3<=area<=1.),
            propagation=bool(primary.get('forward',0)>0 and primary.get('reverse',0)>0 and extent is not None and 5<=extent<=20),
            entry=bool(summary['high_onset_ms'] is not None and 7000<=summary['high_onset_ms']<=13000),
            D_track=bool(abs(checkpoint['9870']-nD)<=.05))
        rows.append(dict(name=job['name'],summary=summary,original_A4_checks=gate,n_original_A4_pass=sum(gate.values()),
            D_at_native_checkpoints=checkpoint,native_D_at9870=nD,final_mean_M=mfinal,
            windows={f'{a}-{b}':window_stats(events,a,b) for a,b in WINDOWS},
            quiet_bounded_windows={f'{a}-{b}':window_stats(complete,a,b) for a,b in WINDOWS},
            quiet_by_window={f'{a}-{b}':float((sm[(t>=a)&(t<b)]<5).mean()) for a,b in WINDOWS},
            events=[{k:v for k,v in ev.items() if k!='onset'} for ev in events],
            qa=dict(weighted_rate_error_Hz=field_error,refractory_bounds=bounds)))
    out=dict(status='COMPLETE' if len(rows)==2 else 'PARTIAL',rows=rows,updated_epoch=time.time(),producer_sha256=sha(__file__),
        original_criteria_source=str(BASE/'a4_contract.json'),original_criteria_sha256=sha(BASE/'a4_contract.json'),
        note='Same numerical A4 criteria are reported without broadening; applying them here is a diagnostic correspondence test. These particles are not the old stochastic finite-network arm. Full contact/spatial visual inspection and G/K native correspondence remain separate.',
        model_promoted=False,formal_bifurcation_allowed=False,human_review='PENDING')
    write(OUT/'audit.json',out);return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');args=p.parse_args()
    while True:
        result=audit();print([(r['name'],r['summary']['high_onset_ms'],r['n_original_A4_pass'],r['D_at_native_checkpoints']['9870']) for r in result['rows']],flush=True)
        if result['status']=='COMPLETE' or not args.wait:break
        time.sleep(30)
