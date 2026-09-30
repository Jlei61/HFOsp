#!/usr/bin/env python3
"""Two fixed holds bracketing the relevant slow-ramp core collapse.

The K schedule is a diagnostic intervention. It does not change the native
model or count as autonomous recovery. All inherited density physics is intact.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,subprocess,time
import numpy as np
from campaign import ROOT,REPO,PYTHON,read,write,sha
import continue_target_high_state as carried

OUT=ROOT/'carried_exit_fixed_holds'
JOBS={'held_K9p5':9.5,'held_K9p65':9.65}
SOURCE=carried.OUT/'control_K9'


def prepare():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    slow=read(carried.OUT/'analysis/ramp10s_K10p5.json')
    fast=read(carried.OUT/'analysis/ramp2s_K10p5.json')
    assert slow['both_cores_sustained_low'] and fast['both_cores_sustained_low']
    write(OUT/'contract.json',dict(status='REGISTERED_AFTER_COMPLETE_PAIRED_RAMPS',created_epoch=time.time(),
        question='Does the carried active state persist below the slow-ramp core collapse when K stops changing, and is the collapsing region already unable to sustain it?',
        selection='The slow ramp first drops coreB below350Hz atK9.638 and coreA at9.659, both sustainedbelow5Hz beginsK9.687. Choose K9.5 before core loss and9.65 within the decline. These adaptive positions are selected after complete trajectories, not an independent confirmation of a prespecified boundary.',
        design='From the SAME complete constant-input K9 control state, ramp K at0.15/s to9.5or9.65, then hold8s. Original40000x128target density, actualexit Zfield heldmean.21, same stationary expectedexternalmean and paired future numericalRNG, original R/G/M andrecurrence. No additional jobs or automatic extensions.',
        readouts='Full1msgroupoutputs andglobalR/G; terminal3s rates/field/G/Zcounterfactualdrift; adjacent1s windows for residual drift; per-target meanM averaged at10ms during last3s, and fullfinalstate, for subsequent independent stationary-response checking.',
        interpretation='A finite held-state persistence/silence does not prove an equilibrium or its stability. Subsequent stationary residual and measured local response are required at the actual relevant state. Kfixed and Zfixed are interventions, not an autonomous loop.',
        jobs=JOBS,source=str(SOURCE),source_sha256=sha(SOURCE/'final_state.npz'),
        producer_sha256=sha(__file__),carried_producer_sha256=sha(carried.__file__),formal_bifurcation_allowed=False))


def worker(name,device):
    c=read(OUT/'contract.json');assert c['producer_sha256']==sha(__file__) and c['carried_producer_sha256']==sha(carried.__file__)
    assert read(carried.OUT/'implementation_qa.json')['status']=='PASS'
    folder=OUT/name;folder.mkdir(exist_ok=True);assert not (folder/'progress.json').exists()
    started=time.time();write(folder/'progress.json',dict(status='INITIALIZING',pid=os.getpid(),device=device))
    K=JOBS[name];ramp=(K-9)/.15;duration=np.ceil(ramp*100)/100+8
    e=carried.Continued(SOURCE,device,ramp,K-9);cp=e.cp
    assert all(np.array_equal(getattr(e,k).get(),e.saved[k]) for k in carried.KEYS)
    write(folder/'initial_qa.json',dict(status='PASS',full_carried_state_bitwise=True,K=K,ramp_s=ramp,hold_at_least_s=8.,duration_s=duration))
    e.graph();outputs=[];globals=[];Msum=cp.zeros(e.N);Msamples=0
    for offset in range(0,round(duration*1000),10):
        outputs.append(e.chunk().astype('f4'));globals.append(e.global_output.get())
        if offset+10>round((duration-3)*1000):
            Msum+=cp.mean(e.state[:,:,5],axis=1);Msamples+=1
            cp.cuda.get_current_stream().synchronize()
        if (offset+10)%500==0:
            write(folder/'progress.json',dict(status='RUNNING',name=name,K=K,pid=os.getpid(),device=device,
                simulation_s=(offset+10)/1000,elapsed_wall_s=time.time()-started,updated_epoch=time.time()))
            print('HELD CARRIED',name,(offset+10)/1000,flush=True)
    value=np.concatenate(outputs);glob=np.concatenate(globals)
    assert np.isfinite(value).all() and np.isfinite(glob).all() and Msamples==300
    assert np.array_equal(e.state[:,:,6].get(),e.saved['state'][:,:,6])
    finalK=e.base_K.get()*(1+(K-9)/9)
    assert np.max(np.abs(e.state[:,:,7].get()-finalK[:,None]))<2e-14
    np.savez_compressed(folder/'trajectory.npz',elapsed_time_ms=np.arange(1,len(value)+1),group_output=value,
        channels=['rate_Hz','Z','M','K','IE','applied_II','V','abs_current','Z_eligible_fraction'],global_R_Hz=glob[:,0],global_s=glob[:,1])
    np.savez_compressed(folder/'final_state.npz',**{k:getattr(e,k).get() for k in carried.KEYS})
    np.savez_compressed(folder/'stationary_candidate_observations.npz',mean_target_M=(Msum/Msamples).get(),
        mean_group_output=value[-3000:].mean(0),per_second_group_output=value[-3000:].reshape(3,1000,9,e.P).mean(1),
        mean_global_state=glob[-3000:].mean(0),Z=e.saved['state'][:,0,6],K=finalK,
        expected_external_per_group=e.constant_cpu,statistical_window_s=[duration-3,duration],
        meanM_sample_interval_ms=10)
    E=e.geo['population']==0;reg=e.geo['group_region'];masks=[E]+[E&(reg==j) for j in range(3)]
    rates=np.array([np.average(value[-3000:,0,m],weights=e.sizes[m],axis=1).mean() for m in masks])
    result=dict(status='COMPLETE',name=name,K=K,ramp_s=ramp,duration_s=duration,hold_at_least_s=8.,
        terminal3s_rates_allE_A_B_surround_Hz=rates.tolist(),terminal3s_Graw=float(30*glob[-3000:,1].mean()),
        elapsed_s=time.time()-started,formal_bifurcation_allowed=False)
    write(folder/'result.json',result);write(folder/'progress.json',result);print(result,flush=True)


def supervise():
    assert not (OUT/'status.json').exists();active={};done=[];failed=[]
    for device,name in enumerate(JOBS):
        with (OUT/f'{name}.log').open('w') as log:
            active[name]=subprocess.Popen([PYTHON,__file__,'worker','--name',name,'--device',str(device)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
    while active:
        for name,p in list(active.items()):
            code=p.poll()
            if code is None:continue
            if code==0 and (OUT/name/'result.json').exists():done.append(name)
            else:failed.append(dict(name=name,exit_code=code))
            del active[name]
        write(OUT/'status.json',dict(status='FAILED' if failed else 'RUNNING' if active else 'COMPLETE',
            supervisor_pid=os.getpid(),completed=done,failed=failed,active=[dict(name=n,pid=p.pid) for n,p in active.items()],updated_epoch=time.time()))
        if active:time.sleep(20)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','worker','supervise']);p.add_argument('--name',choices=list(JOBS));p.add_argument('--device',type=int,default=0)
    a=p.parse_args();prepare() if a.command=='prepare' else worker(a.name,a.device) if a.command=='worker' else supervise()
