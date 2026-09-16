#!/usr/bin/env python3
"""One fresh noise realization plus an observer-only dense replay of the hit."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import copy,fcntl,json,subprocess,sys,time,shutil
from pathlib import Path
import numpy as np
import psutil
import run_topic4_fixed_zm_termination as fixed
import run_topic4_zm_matched_spatial_termination as matched
import analyze_topic4_fixed_zm_termination as analysis

ROOT=fixed.OUT;OUT=ROOT/'positive_candidate_confirmation'
SOURCE=ROOT/'sahp_bracket_round5/runs/sahp1.5_g0.5_s9108401'

def main():
    assert fixed.carrier.base.read(ROOT/'sahp_bracket_round5/autonomous_exit_gate_audit.json')['status']=='PASS'
    base=fixed.carrier.base.read(ROOT/'sahp_bracket_round5/protocol.json');j=copy.deepcopy(base['initial_jobs'][0]);assert j['sahp_gain']==1.5
    noise=copy.deepcopy(j);noise.update(name='sahp1.5_g0.5_s9108402',seed=9108402,horizon_s=20.,device=1)
    dense=copy.deepcopy(j);dense.update(name='dense_replay_s9108401',horizon_s=8.,dense_contact=True,device=0,readout_replay=True)
    p=copy.deepcopy(base);p.update(initial_jobs=[noise],measurement_jobs=[dense],status='DEFINED_AFTER_DEVELOPMENT_HIT',
        wrapper_sha256=fixed.carrier.base.sha(fixed.__file__),matched_producer_sha256=fixed.carrier.base.sha(matched.__file__),
        question='Confirm the same autonomous entry/exit with a new noise realization; replay the selected seed at native temporal resolution for valid spectral readout.',
        development_seed=9108401,independent_noise_seed=9108402,
        measurement_replay='Same seed and dynamics through8s; only contact observation becomes0.1ms. It is not an independent realization and not a new parameter condition.',
        stage_policy='Two jobs,20s new-noise run and8s dense readout replay; wait until primary queue empty and keep12 total mechanism workers. Shared deadline unchanged.')
    fixed.carrier.base.write(OUT/'protocol.json',p);shutil.copy2(ROOT/'geometry.npz',OUT/'geometry.npz')
    for job in [noise,dense]:fixed.carrier.base.write(OUT/'jobs'/f"{job['name']}.json",job)
    matched.OUT=OUT;matched.qa()
    pending=[dense,noise];running={};handles={};logs=OUT/'logs';logs.mkdir(exist_ok=True)
    lock=(ROOT/'followup_dispatch.lock').open('a')
    while pending or running:
        for name,proc in list(running.items()):
            if proc.poll() is not None:
                handles[name].close();assert proc.returncode==0,(name,proc.returncode);del running[name]
        if pending:
            assert time.time()<p['deadline_epoch']-1800,'Insufficient time remaining to dispatch'
            fcntl.flock(lock,fcntl.LOCK_EX)
            active=0
            for proc in psutil.process_iter(['cmdline']):
                args=proc.info['cmdline'] or []
                if 'worker' in args and any(Path(a).name=='run_topic4_zm_matched_spatial_termination.py' for a in args):active+=1
            state=fixed.carrier.base.read(ROOT/'combined_dispatch_status.json')
            if not state['queued'] and active<12 and psutil.virtual_memory().available/2**30>80:
                job=pending.pop(0);h=(logs/(job['name']+'.log')).open('a');env=dict(os.environ,TOPIC4_TERMINATION_OUT=str(OUT))
                proc=subprocess.Popen([sys.executable,'-u',matched.__file__,'worker','--name',job['name']],env=env,stdout=h,stderr=subprocess.STDOUT)
                running[job['name']]=proc;handles[job['name']]=h;print('Started',job['name'],proc.pid,flush=True)
            fcntl.flock(lock,fcntl.LOCK_UN)
        fixed.carrier.base.write(OUT/'dispatch_status.json',dict(time=time.time(),running={k:v.pid for k,v in running.items()},queued=[j['name'] for j in pending]))
        if pending or running:time.sleep(10)
    replay=OUT/'runs'/dense['name'];x=analysis.load(replay);y=analysis.load(SOURCE)
    fields=['raster','spikes_1ms','regions_1ms','field_5ms','Z','M','currents','regional_currents','inputs','lfp_raw']
    checks={k:np.array_equal(x[k],y[k][:len(x[k])]) for k in fields};assert all(checks.values()),checks
    direct=analysis.load(replay,'dense_contact_chunks');coarse=analysis.load(replay,'actual_current_chunks')
    assert np.array_equal(direct['contact_current'][::20],coarse['contact_current'])
    qa=dict(status='PASS',fields=checks,coarse_samples_exact_subset=True,native_contact_sampling_Hz=10000.,source=str(SOURCE),replay=str(replay),independent_realization=False)
    fixed.carrier.base.write(OUT/'dense_readout_parity.json',qa)
    fixed.carrier.base.write(SOURCE/'dense_contact_source.json',dict(folder=str(replay),qa=str(OUT/'dense_readout_parity.json'),sample_rate_Hz=10000.,purpose='Compute1-150Hz PSD directly at the native0.1ms sampling interval; no coarse temporal decimation before spectral estimation.'))
    fixed.carrier.base.write(OUT/'complete.json',dict(time=time.time(),noise_result=str(OUT/'runs'/noise['name']/'result.json'),dense_parity=qa))
    print('Confirmation and dense observation replay finished',flush=True)

if __name__=='__main__':main()
