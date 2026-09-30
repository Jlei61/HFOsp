"""Bounded dependent refinement at a prespecified low-branch rate section.

After one named local-response table completes, compute its low equilibrium
predictor and correct one full-network point at mean E = 0.1445 Hz. Comparing
the same rate section with degree16 tests mesh/order drift without mislabeling
either point as a fold. No parameter fit or further campaign is launched.
"""
from analyze_qualification import *
import argparse
import subprocess
import sys
import time
import os


def dump(path,obj):path.write_text(json.dumps(safe(obj),indent=2)+'\n')


def run(a):
    folder=OUT/'fine_low_followup'/a.label;folder.mkdir(parents=True,exist_ok=False)
    start=time.time();status_file=a.table/'status.json'
    dump(folder/'config.json',dict(table=str(a.table),rate_section_hz=.1445,maximum_wait_s=a.wait,
        predictor_maximum_rate_hz=.165,predictor_step_hz=.001,device=a.device,
        scope='One predictor branch and one full-map corrected equilibrium; critical type and dynamic stability remain separate'))
    while True:
        try:s=json.load(open(status_file)) if status_file.exists() else {}
        except json.JSONDecodeError:s={}
        if s.get('status')=='STATIONARY_SOLVED':break
        if s.get('status') not in (None,'RUNNING'):
            dump(folder/'status.json',dict(status='DEPENDENCY_NOT_SOLVED',table_status=s));return
        if time.time()-start>a.wait:
            dump(folder/'status.json',dict(status='WAIT_LIMIT',table_status=s));return
        dump(folder/'status.json',dict(status='WAITING_FOR_TABLE',pid=os.getpid(),elapsed_s=time.time()-start))
        time.sleep(20)
    predictor=a.label+'_rate_predictor';section=a.label+'_rE0.1445'
    rootdir=OUT/'corrected_batched_sections'
    assert rootdir.resolve().is_relative_to(Path('/data')), 'Large density arrays must use the existing data-volume destination'
    commands=[['table_rate_branch.py','--table',str(a.table),'--label',predictor,'--maximum-rate','.165','--step','.001'],
        ['correct_batched_section.py','--table',str(a.table),'--predictor',str(OUT/'equilibrium_predictors'/predictor/'branch.npz'),
         '--label',section,'--rate','.1445','--batch-size','64','--batched-check','--device',str(a.device)]]
    for index,args in enumerate(commands):
        cmd=[sys.executable,'-u',str(ROOT/'scripts/topic4_kinetic_bifurcation'/args[0]),*args[1:]]
        dump(folder/'status.json',dict(status='RUNNING',pid=os.getpid(),stage=index,command=cmd))
        with (folder/f'stage_{index}.log').open('w') as log:
            proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        if proc.returncode:
            dump(folder/'status.json',dict(status='STAGE_FAILED',stage=index,exit_code=proc.returncode));return
        if index==0:
            with np.load(OUT/'equilibrium_predictors'/predictor/'branch.npz') as z:
                assert np.isfinite(z['D']).all() and len(z['D'])>1
    result=json.load(open(rootdir/section/'status.json'))
    compare=None
    if result['status']=='FULL_MAP_EQUILIBRIUM_CORRECTED':
        other=json.load(open(rootdir/'rE0.1445_degree16_dv0.0625/status.json'))
        compare=dict(mean_E_hz=.1445,degree16_D=other['D'],degree20_D=result['D'],
            absolute_D_difference=abs(other['D']-result['D']),
            relative_D_difference=abs(other['D']-result['D'])/abs(other['D']),
            interpretation='Two full-map equilibria at the same global-rate section; neither is independently classified as a saddle-node')
    dump(folder/'status.json',dict(status='COMPLETE' if result['status']=='FULL_MAP_EQUILIBRIUM_CORRECTED' else 'SECTION_NOT_CORRECTED',
        predictor=str(OUT/'equilibrium_predictors'/predictor),section=str(rootdir/section),comparison=compare,section_status=result))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--table',type=Path,required=True);ap.add_argument('--label',required=True)
    ap.add_argument('--device',type=int,default=1);ap.add_argument('--wait',type=float,default=5400.)
    run(ap.parse_args())
