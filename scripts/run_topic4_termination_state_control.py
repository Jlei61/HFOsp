#!/usr/bin/env python3
"""Carried high-state control; never count as an autonomous cold-start success."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import copy,json,pickle,shutil,subprocess,sys,time
from pathlib import Path
import psutil
import run_topic4_fixed_zm_termination as fixed
import run_topic4_zm_matched_spatial_termination as matched

ROOT=fixed.OUT;OUT=ROOT/'carried_high_gain_control'
NAME='gain2_from_gain1_high_s9108401'

def main():
    source=ROOT/'source_sahp_round4/runs/sahp1_g0.5_s9108401/first_high_checkpoint.pkl'
    parent=fixed.carrier.base.read(ROOT/'source_sahp_round4/protocol.json')
    p=copy.deepcopy(parent);j=copy.deepcopy(parent['initial_jobs'][2])
    assert j['sahp_gain']==1.
    j.update(name=NAME,sahp_gain=2.,horizon_s=20.,device=1,control=True)
    p.update(initial_jobs=[j],status='DECLARED_CARRIED_STATE_CONTROL',
        wrapper_sha256=fixed.carrier.base.sha(fixed.__file__),matched_producer_sha256=fixed.carrier.base.sha(matched.__file__),
        question='At an already observed high state, can increasing future spike-triggered sAHP increments terminate activity?',
        control=True,autonomous_cold_start_target=False,source_checkpoint=str(source),
        changed_parameter=dict(time_s=6.,name='sahp_gain',before=1.,after=2.),
        initial_condition='Exactly carried original V, refractoriness, synapses, delays, Z/M/gK, global filter, fast threshold and all RNG/OU histories. No state reset. Only subsequent gK spike increments change.',
        interpretation='A parameter intervention starting from the existing high state. It does not establish spontaneous entry and exit with gain2 fixed from initialization.',
        stage_policy='One14s continuation, absolute time6-20s, after the main declared batch has no queued jobs and a worker slot is free. Shared deadline unchanged.')
    fixed.carrier.base.write(OUT/'protocol.json',p);fixed.carrier.base.write(OUT/'jobs'/f'{NAME}.json',j)
    OUT.mkdir(exist_ok=True);shutil.copy2(ROOT/'geometry.npz',OUT/'geometry.npz')
    matched.OUT=OUT;matched.qa()
    while True:
        if time.time()>p['deadline_epoch']-1800:raise RuntimeError('Not enough reserved time for state control; not dispatched')
        active=[]
        for proc in psutil.process_iter(['cmdline']):
            args=proc.info['cmdline'] or []
            if 'worker' in args and any(Path(a).name=='run_topic4_zm_matched_spatial_termination.py' for a in args):active.append(proc.pid)
        dispatch=fixed.carrier.base.read(ROOT/'combined_dispatch_status.json')
        if not dispatch['queued'] and len(active)<12 and psutil.virtual_memory().available/2**30>80:break
        time.sleep(10)
    with source.open('rb') as handle:state=pickle.load(handle)
    assert state['engine']['step']==60000 and state['tracker']['phase']=='HIGH'
    assert state['job']['sahp_gain']==1.
    source_job=state['job'];state['job']=j
    folder=OUT/'runs'/NAME;folder.mkdir(exist_ok=True,parents=True)
    fixed.carrier.base.save_pickle(folder/'checkpoint.pkl',state)
    fixed.carrier.base.write(folder/'initial_state_provenance.json',dict(source=str(source),source_sha256=fixed.carrier.base.sha(source),source_job=source_job,new_job=j,engine_and_tracker_unchanged=True,branch_time_s=6.,source_entry_inherited=True))
    del state
    env=dict(os.environ,TOPIC4_TERMINATION_OUT=str(OUT));log=(OUT/'worker.log').open('a')
    print('Starting parameter-intervention control from6s; not a cold-start autonomous run',flush=True)
    subprocess.run([sys.executable,'-u',matched.__file__,'worker','--name',NAME],env=env,stdout=log,stderr=subprocess.STDOUT,check=True);log.close()
    result=fixed.carrier.base.read(folder/'result.json')
    result.update(control=True,autonomous_cold_start_success=False,no_external_intervention=False,
        no_external_intervention_during_continuation=True,parameter_intervention=p['changed_parameter'],
        source_entry_inherited=True,M_reset=False,Z_reset=False,gK_reset=False,
        scope=p['interpretation'])
    fixed.carrier.base.write(folder/'result.json',result)
    fixed.carrier.base.write(OUT/'control_complete.json',dict(time=time.time(),result=str(folder/'result.json'),entries=result['tracker']['entries'],recoveries=result['tracker']['recoveries'],interpretation=p['interpretation']))
    print(json.dumps(result['tracker'],indent=2),flush=True)

if __name__=='__main__':main()
