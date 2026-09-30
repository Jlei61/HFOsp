#!/usr/bin/env python3
"""Bounded checks of the remaining actual-exit conditional field family."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,pickle,subprocess,time
import numpy as np
from campaign import ROOT,REPO,PYTHON,read,write,sha
import target_density_exit as base

OUT=ROOT/'target_density_field_family'
JOBS=[('exit_return_probes',f'exit_z0.21_k{k}_fields16p7_{h}') for k,h in [(9,'recovery'),(12,'high'),(12,'recovery')]]
JOBS += [('exit_midpoint_probes',f'exit_z0.21_k10.5_fields16p7_{h}') for h in ['high','recovery']]


def prepare():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    comparison=read(base.OUT/'comparison.json');assert comparison['status']=='COMPLETE'
    individual=next(x for x in comparison['rows'] if x['mode']=='individual')['windows'][-1]
    assert individual['density_counterfactual_dZ_per_s'][0]<0
    write(OUT/'contract.json',dict(status='REGISTERED_AFTER_K9_TARGET_REPAIR',created_epoch=time.time(),
        question='Does the target-resolved closure preserve recovery silence and high-state loss at the neighboring actual-exit fields?',
        evidence='K9high targetprojection corrected native fieldRMS145to6.36Hz andZdriftdirection; this does notcertify otherconditions orstability.',
        jobs=[dict(root=str(ROOT/root),name=name,job_sha256=sha(ROOT/root/'jobs'/f'{name}.json')) for root,name in JOBS],
        design='Five10s individualtarget runs,40000targets x128particles, numericalseed928751, exactsame50-60sexpectedexternaldrive andlocalGaussianphysics as completedK9high. Only completeinitialhistory andheldZ/K fields vary asnativejobs specify. No newnativeconditions beyondseparatelyregisteredmidpointpair.',
        acceptance_question='Compare spatialrates, G, perparticleZeligibility, silence/exit transient; onecorrespondinghighpoint isnot a fullacceptance. No new fitting orautomaticextend.',
        formal_bifurcation_allowed=False,base_producer_sha256=sha(base.__file__),producer_sha256=sha(__file__)))


def construct(source,name,device):
    e=base.TargetNetwork('individual',128,device);cp=e.cp
    job=read(source/'jobs'/f'{name}.json');assert sha(job['held_fields_file'])==job['held_fields_sha256']
    with open(job['source_checkpoint'],'rb') as f:saved=pickle.load(f)
    assert saved['identity']==e.prep['graph_identity'];s=saved['engine']
    assert s['step']==round(job['branch_start_s']*10000)
    with np.load(job['held_fields_file']) as z:zz,kk=z['Z'],z['K']
    held=s['slow']['z'].copy();held[:32000]=zz;k=np.zeros(e.N);k[:32000]=kk
    e.native_initial=np.stack([s['V'],s['s_E'],s['I_E'],s['s_I'],s['I_I'],s['slow']['m'],held,k],axis=1)
    e.native_ref=s['ref'];e.initial_state[:]=cp.asarray(np.repeat(e.native_initial[:,None,:],e.R,axis=1))
    e.initial_ref[:]=cp.asarray(np.repeat(s['ref'][:,None],e.R,axis=1),dtype='i4')
    e.initial_global=np.array([s['termination_mechanism']['r_global'],s['global_feedback_response']['global_state']])
    order=(s['step']+np.arange(e.depth))%e.depth
    e.pending_cpu=np.stack([s['ring_sE'][order],s['ring_sI'][order]]);e.pending[:]=cp.asarray(e.pending_cpu)
    e.reset();assert np.array_equal(e.state.get(),e.initial_state.get())
    return e


def worker(index,device):
    contract=read(OUT/'contract.json');assert sha(base.__file__)==contract['base_producer_sha256']
    assert sha(__file__)==contract['producer_sha256'];row=contract['jobs'][index]
    source=__import__('pathlib').Path(row['root']);name=row['name'];folder=OUT/name;folder.mkdir(exist_ok=True)
    assert sha(source/'jobs'/f'{name}.json')==row['job_sha256'];assert not (folder/'progress.json').exists()
    started=time.time();write(folder/'progress.json',dict(status='INITIALIZING',pid=os.getpid(),device=device))
    e=construct(source,name,device);write(folder/'initial_qa.json',dict(status='PASS',full_cell_states_held_fields_pending_and_global_restored=True,
          source_job_sha256=row['job_sha256'],source=str(source),name=name))
    e.graph();outputs=[];globals=[]
    for offset in range(0,10000,10):
        outputs.append(e.chunk().astype('f4'));globals.append(e.global_output.get())
        if (offset+10)%500==0:write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),device=device,elapsed_simulation_s=(offset+10)/1000,
              elapsed_wall_s=time.time()-started,updated_epoch=time.time()))
    assert int(e.clock.get()[0])==100000 and np.array_equal(e.state.get()[:,:,6:8],e.initial_state.get()[:,:,6:8])
    values=np.concatenate(outputs);glob=np.concatenate(globals);assert np.isfinite(values).all() and np.isfinite(glob).all()
    np.savez_compressed(folder/'trajectory.npz',elapsed_time_ms=np.arange(1,10001),group_output=values,
          channels=['rate_Hz','Z','M','K','IE','applied_II','V','abs_current','Z_eligible_fraction'],global_R_Hz=glob[:,0],global_s=glob[:,1])
    np.savez_compressed(folder/'final_state.npz',state=e.state.get(),ref=e.ref.get(),rng=e.rng.get(),history=e.history.get(),clock=e.clock.get(),global_state=e.global_state.get())
    result=dict(status='COMPLETE',name=name,physical_targets=e.N,replicas=e.R,duration_s=10.,elapsed_s=time.time()-started,
          held_fields_bitwise=True,formal_bifurcation_allowed=False)
    write(folder/'result.json',result);write(folder/'progress.json',result);print('TARGET FAMILY COMPLETE',name,flush=True)


def supervise():
    assert (OUT/'contract.json').exists();assert not (OUT/'status.json').exists()
    pending=list(range(len(JOBS)));active={};done=[];failed=[]
    while pending or active:
        for device,(index,proc) in list(active.items()):
            code=proc.poll()
            if code is None:continue
            name=JOBS[index][1];path=OUT/name/'result.json'
            if code==0 and path.exists() and read(path)['status']=='COMPLETE':done.append(name)
            else:failed.append(dict(name=name,exit_code=code))
            del active[device]
        for device in [0,1]:
            if device in active or not pending or failed:continue
            index=pending.pop(0);name=JOBS[index][1]
            with (OUT/f'{name}.log').open('w') as log:
                proc=subprocess.Popen([PYTHON,__file__,'worker','--index',str(index),'--device',str(device)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
            active[device]=(index,proc)
        write(OUT/'status.json',dict(status='FAILED' if failed else 'COMPLETE' if not pending and not active else 'RUNNING',
              supervisor_pid=os.getpid(),completed=done,pending=[JOBS[i][1] for i in pending],
              active=[dict(device=d,name=JOBS[i][1],pid=p.pid) for d,(i,p) in active.items()],failed=failed,updated_epoch=time.time()))
        if failed and not active:break
        if pending or active:time.sleep(20)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','supervise','worker']);p.add_argument('--index',type=int);p.add_argument('--device',type=int,default=0)
    args=p.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='supervise':supervise()
    else:worker(args.index,args.device)
