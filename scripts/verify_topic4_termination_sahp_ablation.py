#!/usr/bin/env python3
"""State-matched sAHP removal; a causal diagnostic, not an autonomous candidate."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import copy,fcntl,pickle,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import psutil
import run_topic4_fixed_zm_termination as fixed
import run_topic4_zm_matched_spatial_termination as matched

ROOT=fixed.OUT;OUT=ROOT/'state_matched_sahp_ablation'
SOURCE=ROOT/'sahp_bracket_round5/runs/sahp1.25_g0.5_s9108401'

def main():
    parent=fixed.carrier.base.read(ROOT/'sahp_bracket_round5/protocol.json')
    base=next(j for j in parent['initial_jobs'] if j['sahp_gain']==1.25)
    jobs=[]
    for name,gain,end,device in [('retained_sahp_restart',1.25,8.,0),('removed_sahp_at_6s',0.,12.,1)]:
        j=copy.deepcopy(base);j.update(name=name,sahp_gain=gain,horizon_s=end,device=device,control=True)
        jobs.append(j)
    p=copy.deepcopy(parent);p.update(initial_jobs=jobs,status='STATE_MATCHED_CAUSAL_DIAGNOSTIC',
        wrapper_sha256=fixed.carrier.base.sha(fixed.__file__),matched_producer_sha256=fixed.carrier.base.sha(matched.__file__),
        question='Does the added sAHP influence termination from the identical already recruited spatial state?',
        branch_time_s=6.,source=str(SOURCE),autonomous_candidate=False,
        intervention='Remove only the extra K conductance from6s in one branch; the retained branch tests exact state restoration. Native Z/M, V, synapses, delay buffers, global filter and RNG/OU states are carried unchanged.',
        caveat='The inherited entry is not a new entry; any return is post-intervention evidence. Stored gK is retained but uncoupled when gain is0.',
        stage_policy='Two bounded branches,6-8s retained and6-12s removed; no primary-run mutation or new parameter search.')
    fixed.carrier.base.write(OUT/'protocol.json',p);shutil.copy2(ROOT/'geometry.npz',OUT/'geometry.npz')
    for j in jobs:fixed.carrier.base.write(OUT/'jobs'/f"{j['name']}.json",j)
    matched.OUT=OUT;matched.qa()
    pending=list(jobs);running={};handles={};lock=(ROOT/'followup_dispatch.lock').open('a')
    while pending or running:
        for name,proc in list(running.items()):
            if proc.poll() is not None:
                handles[name].close();assert proc.returncode==0,(name,proc.returncode);del running[name]
        if pending:
            assert time.time()<p['deadline_epoch']-1500,'Insufficient reserved time'
            fcntl.flock(lock,fcntl.LOCK_EX)
            active=0
            for proc in psutil.process_iter(['cmdline']):
                args=proc.info['cmdline'] or []
                if 'worker' in args and any(Path(a).name=='run_topic4_zm_matched_spatial_termination.py' for a in args):active+=1
            if active<12 and not fixed.carrier.base.read(ROOT/'combined_dispatch_status.json')['queued'] and psutil.virtual_memory().available/2**30>80:
                job=pending.pop(0);folder=OUT/'runs'/job['name'];folder.mkdir(parents=True,exist_ok=True)
                assert not (folder/'checkpoint.pkl').exists(),'Refuse to replace a started diagnostic'
                with (SOURCE/'first_high_checkpoint.pkl').open('rb') as h:state=pickle.load(h)
                assert state['engine']['step']==60000 and state['tracker']['phase']=='HIGH'
                state['job']=job;fixed.carrier.base.save_pickle(folder/'checkpoint.pkl',state);del state
                fixed.carrier.base.write(folder/'initial_state_provenance.json',dict(source=str(SOURCE/'first_high_checkpoint.pkl'),source_sha256=fixed.carrier.base.sha(SOURCE/'first_high_checkpoint.pkl'),branch_time_s=6.,only_parameter_change='sahp_gain1.25 to0' if job['sahp_gain']==0 else 'None',no_state_reset=True))
                h=(OUT/(job['name']+'.log')).open('a');env=dict(os.environ,TOPIC4_TERMINATION_OUT=str(OUT))
                proc=subprocess.Popen([sys.executable,'-u',matched.__file__,'worker','--name',job['name']],env=env,stdout=h,stderr=subprocess.STDOUT)
                running[job['name']]=proc;handles[job['name']]=h;print('Started',job['name'],proc.pid,flush=True)
            fcntl.flock(lock,fcntl.LOCK_UN)
        fixed.carrier.base.write(OUT/'dispatch_status.json',dict(time=time.time(),running={k:v.pid for k,v in running.items()},queued=[j['name'] for j in pending]))
        if pending or running:time.sleep(10)
    fields=['raster','spikes_1ms','regions_1ms','field_5ms','Z','M','currents','regional_currents','inputs','lfp_raw']
    checks={key:True for key in fields}
    for f in sorted((OUT/'runs/retained_sahp_restart/chunks').glob('*.npz')):
        with np.load(f) as x,np.load(SOURCE/'chunks'/f.name) as y:
            for key in fields:checks[key]=checks[key] and np.array_equal(x[key],y[key])
    assert all(checks.values()),checks
    report=dict(unchanged_restart='PASS',field_parity=checks,source=str(SOURCE),branch_time_s=6.,autonomous_candidate=False,results={})
    for j in jobs:
        folder=OUT/'runs'/j['name'];r=fixed.carrier.base.read(folder/'result.json')
        r.update(control=True,autonomous_cold_start_success=False,no_external_intervention=False,source_entry_inherited=True,intervention=p['intervention'],native_state_reset=False)
        fixed.carrier.base.write(folder/'result.json',r)
        report['results'][j['name']]=r
    fixed.carrier.base.write(OUT/'complete.json',report);print('State-matched diagnostic complete',flush=True)

if __name__=='__main__':main()
