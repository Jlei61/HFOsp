#!/usr/bin/env python3
"""Continue the actual autonomous candidate without resetting any state."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import copy,fcntl,pickle,shutil,subprocess,sys,time
from pathlib import Path
import psutil
import run_topic4_fixed_zm_termination as fixed
import run_topic4_zm_matched_spatial_termination as matched
ROOT=fixed.OUT;OUT=ROOT/'autonomous_recurrence_continuation'
SOURCE=ROOT/'sahp_bracket_round5/runs/sahp1.5_g0.5_s9108401'

def main():
    parent=fixed.carrier.base.read(ROOT/'sahp_bracket_round5/protocol.json')
    job=copy.deepcopy(next(j for j in parent['initial_jobs'] if j['sahp_gain']==1.5))
    job.update(name='sahp1.5_g0.5_s9108401_to50s',horizon_s=50.,device=0)
    p=copy.deepcopy(parent);p.update(initial_jobs=[job],status='SAME_TRAJECTORY_LONG_CONTINUATION',
        wrapper_sha256=fixed.carrier.base.sha(fixed.__file__),matched_producer_sha256=fixed.carrier.base.sha(matched.__file__),
        question='After the autonomous return, does the same continuing trajectory re-enter high activity?',
        same_realization=True,source=str(SOURCE),branch_time_s=30.,
        changes='Only simulation horizon30 to50s; every dynamical parameter and full native/added state is unchanged. Stop shortly after a second accepted entry or at shared deadline.',
        stage_policy='One continuation, not an independent replicate. Link immutable completed prefix observations and carry full30s checkpoint; no Z/M/gK reset.')
    lock=(ROOT/'followup_dispatch.lock').open('a')
    while True:
        if time.time()>p['deadline_epoch']-1500:raise RuntimeError('Insufficient reserved continuation time')
        fcntl.flock(lock,fcntl.LOCK_EX)
        active=0
        for proc in psutil.process_iter(['cmdline']):
            args=proc.info['cmdline'] or []
            if 'worker' in args and any(Path(a).name=='run_topic4_zm_matched_spatial_termination.py' for a in args):active+=1
        if (SOURCE/'result.json').exists() and active<12 and psutil.virtual_memory().available/2**30>80:break
        fcntl.flock(lock,fcntl.LOCK_UN);time.sleep(10)
    source_result=fixed.carrier.base.read(SOURCE/'result.json');assert source_result['status']=='COMPLETE' and source_result['end_s']==30.
    fixed.carrier.base.write(OUT/'protocol.json',p);fixed.carrier.base.write(OUT/'jobs'/f"{job['name']}.json",job);shutil.copy2(ROOT/'geometry.npz',OUT/'geometry.npz')
    folder=OUT/'runs'/job['name'];assert not folder.exists();folder.mkdir(parents=True)
    for sub in ['chunks','mechanism_chunks','actual_current_chunks','intrinsic_adaptation_chunks']:
        (folder/sub).mkdir()
        for path in (SOURCE/sub).glob('*.npz'):os.link(path,folder/sub/path.name)
    if (SOURCE/'dense_contact_source.json').exists():shutil.copy2(SOURCE/'dense_contact_source.json',folder/'dense_contact_source.json')
    with (SOURCE/'checkpoint.pkl').open('rb') as handle:state=pickle.load(handle)
    assert state['engine']['step']==300000;state['job']=job;fixed.carrier.base.save_pickle(folder/'checkpoint.pkl',state);del state
    fixed.carrier.base.write(folder/'initial_state_provenance.json',dict(source=str(SOURCE/'checkpoint.pkl'),source_sha256=fixed.carrier.base.sha(SOURCE/'checkpoint.pkl'),prefix_hardlinked=True,no_parameter_or_state_change=True))
    matched.OUT=OUT;matched.qa();h=(OUT/'worker.log').open('a');env=dict(os.environ,TOPIC4_TERMINATION_OUT=str(OUT))
    proc=subprocess.Popen([sys.executable,'-u',matched.__file__,'worker','--name',job['name']],env=env,stdout=h,stderr=subprocess.STDOUT)
    fcntl.flock(lock,fcntl.LOCK_UN);print('Continuing unchanged from30s',proc.pid,flush=True);code=proc.wait();h.close();assert code==0
    subprocess.run([sys.executable,str(Path(__file__).with_name('analyze_topic4_fixed_zm_termination.py')),'plot','--root',str(OUT),'--name',job['name']],check=True)
    fixed.carrier.base.write(OUT/'complete.json',dict(time=time.time(),result=str(folder/'result.json'),same_realization=True,human_review='PENDING'))

if __name__=='__main__':main()
