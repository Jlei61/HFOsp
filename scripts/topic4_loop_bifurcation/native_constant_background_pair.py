#!/usr/bin/env python3
"""Two native conditional controls matching the spectral assay's external mean.

The input callback is an explicitly declared intervention: it replaces nu_vec
before the original Poisson draw. Native membrane/network/slow equations stay
unchanged. This is never labelled a read-only observer or an autonomous loop.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,copy,shutil,subprocess,time
from pathlib import Path
import numpy as np
from campaign import ROOT,NATIVE,PYTHON,read,write,sha
import observe_source_aggregation as base
from run_topic4_recovery_window import assert_same_state
from analyze_native import analyze,original

OUT=ROOT/'native_K9p35_constant_background_pair'
PARENTS=[ROOT/'native_exit_K_bracket/runs/exit_z0.21_k9.35_fields16p7_high',
         ROOT/'native_K9p35_held_history/runs/exit_z0.21_k9.35_fields16p7_held_K9_history']
NAMES=['asymmetric_history_constant_background','high_history_constant_background']
SPECTRAL=ROOT/'individual_source_spectral_pilot'


def configure(index=0):
    base.OUT=OUT;base.PARENT=PARENTS[index];base.INITIAL=PARENTS[index]/'checkpoint.pkl';base.NAME=NAMES[index]
    return base.configure()


def prepare():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    fields=[dict(np.load(p/'held_fields.npz')) for p in PARENTS]
    assert all(np.array_equal(fields[0][k],fields[1][k]) for k in ['Z','K'])
    nu=np.load(SPECTRAL/'parameters.npz')['nu_per_ms'];assert nu.shape==(40000,) and nu.min()>=0
    np.save(OUT/'fixed_external_per_ms.npy',nu)
    engines=[base.native.read_pickle(p/'checkpoint.pkl')['engine'] for p in PARENTS]
    assert engines[0]['step']==420000 and engines[1]['step']==720000
    assert engines[0]['rng_state']==engines[1]['rng_state'] and engines[0]['xi']==engines[1]['xi']
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_MATCHED_BACKGROUND_CONTROL',created_epoch=time.time(),
        question='Does replacing the native time-varying external expected drive with the same constant per-cell mean used by the spectral model change the two conditional spatial states or the G activation segment?',
        design='Exactly two10s native continuations from complete asymmetric42s and high-history72s states at identical held spatialZ/K. Keep all endogenous states, original network and nativeG/M dynamics. Before every external Poisson draw, replace the actual per-cell nu vector by the fixed source-observer42-44s mean already used in the spectral pilot. Original OU bookkeeping continues but cannot change the overridden rate.',
        intervention='The input callback deliberately MUTATES the rate vector before rng.poisson; it is not a read-only observer. Original source files and neuronal dynamics are untouched. Original input logger runs after this override and records the actual used rate.',
        pairing='The two starting native RNG states and scalar xi are identical; both receive identical fixednu on every0.1ms step. The native loop only draws OU-normal and externalPoisson variates after restoration, so future external count draws are paired. Different original variable-rate runs are not claimed to have matched future Poisson draws after changingnu.',
        readout='Full10s plus last5s rates/core/space/G/causalR/counterfactualZK drift; first/last5s stationarity. Compare native counts with spectral candidate at the SAME fixednu. Separate effect of the background change from residual closure error.',
        unit='One paired initial-noise state with two endogenous histories, not two independent seeds; finite conditional controls, not autonomous exits/recovery or attractor certification.',
        stop='Two10s runs only, no horizon extension, parameter adjustment or formal bifurcation claim.',
        fixed_nu_sha256=sha(OUT/'fixed_external_per_ms.npy'),
        sources=[dict(path=str(p/'checkpoint.pkl'),sha256=sha(p/'checkpoint.pkl')) for p in PARENTS],
        native_engine_sha256=sha(Path(base.native.base.old.simulate_kick.__code__.co_filename)),
        producer_sha256=sha(__file__),formal_bifurcation_allowed=False,counts_as_autonomous_loop=False))
    p=copy.deepcopy(read(NATIVE/'protocol.json'));p.update(stage='NATIVE_MATCHED_CONSTANT_EXTERNAL_BACKGROUND',created_epoch=time.time(),deadline_epoch=time.time()+86400)
    write(OUT/'protocol.json',p);shutil.copy2(NATIVE/'geometry.npz',OUT/'geometry.npz')
    for i,name in enumerate(NAMES):
        configure(i);base.native.make_job(name,str(PARENTS[i]/'checkpoint.pkl'),10.,True,.21,9.35,False)
        assert_same_state(engines[i],base.native.read_pickle(OUT/'runs'/name/'checkpoint.pkl')['engine'])
        jobpath=OUT/'jobs'/f'{name}.json';job=read(jobpath)
        job['external_expected_rate_override']=dict(path=str(OUT/'fixed_external_per_ms.npy'),
            sha256=sha(OUT/'fixed_external_per_ms.npy'),applies='Every external Poisson draw; registered input intervention')
        write(jobpath,job)
    write(OUT/'initial_gate.json',dict(status='PASS',full_initial_engines_bitwise=True,
        same_held_Z_K_bitwise=True,initial_external_RNG_and_xi_exact=True))
    write(OUT/'queue.json',dict(names=NAMES));shutil.copy2(__file__,OUT/'producer.py')


def worker(index):
    configure(index);c=read(OUT/'contract.json')
    assert c['producer_sha256']==sha(__file__) and c['sources'][index]['sha256']==sha(PARENTS[index]/'checkpoint.pkl')
    nu=np.load(OUT/'fixed_external_per_ms.npy');assert sha(OUT/'fixed_external_per_ms.npy')==c['fixed_nu_sha256']
    previous=base.gpu.cuda_backend.wrap_simulator;calls=0
    def wrap(original,device_index):
        fast=previous(original,device_index=device_index)
        def simulate(params,net,*args,**kw):
            old=kw.get('input_observer')
            def override_rate(tm,actual,xi):
                nonlocal calls
                actual[:]=nu
                assert np.array_equal(actual,nu)
                calls+=1
                if old is not None:old(tm,actual,xi)
            kw['input_observer']=override_rate
            return fast(params,net,*args,**kw)
        return simulate
    base.gpu.cuda_backend.wrap_simulator=wrap
    write(OUT/f'worker{index}.json',dict(status='RUNNING',pid=os.getpid(),device=index,updated_epoch=time.time()))
    try:base.gpu.worker(OUT,NAMES[index],index)
    finally:base.gpu.cuda_backend.wrap_simulator=previous
    assert calls==100000
    # The generic device wrapper describes a device-only override. This job also
    # has the separate registered input intervention, so make that explicit in
    # every persisted runtime summary instead of inheriting its generic claim.
    folder=OUT/'runs'/NAMES[index];runtime=read(folder/'runtime_backend.json')
    runtime.update(physics_and_job_unchanged=False,neuronal_equations_unchanged=True,
        registered_external_expected_rate_override=read(OUT/'jobs'/f'{NAMES[index]}.json')['external_expected_rate_override'])
    write(folder/'runtime_backend.json',runtime)
    for filename in ['result.json','progress.json']:
        value=read(folder/filename);value['runtime_backend']=runtime;write(folder/filename,value)
    write(OUT/f'input_override_audit{index}.json',dict(status='PASS',steps=calls,
        every_used_nu_exact=True,scope='Deliberate external-input intervention before original Poisson draw.'))
    write(OUT/f'worker{index}.json',dict(status='COMPLETE',updated_epoch=time.time()))


def finish():
    geo=dict(np.load(OUT/'geometry.npz'));counts=geo['cell_e_counts'];results=[];inputs=[];engines=[]
    for i,name in enumerate(NAMES):
        row,drive=analyze(OUT,name);inputs.append(drive)
        folder=OUT/'runs'/name;engines.append(base.native.read_pickle(folder/'checkpoint.pkl')['engine'])
        d=dict(np.load(OUT/'extended_analysis'/f'{name}_readouts.npz'))
        me=original.load(folder/'mechanism_chunks',['time_ms','global_E_rate_Hz','global_raw_conductance_ratio'])
        dr=original.load(folder/'conditional_drift_chunks',['time_ms','values'])
        start=row['job']['branch_start_s'];windows=[]
        for lo,hi in [(0,5),(5,10)]:
            m=(d['relative_time_5ms_s']>=lo)&(d['relative_time_5ms_s']<hi)
            t=me['time_ms']/1000-start;mm=(t>=lo)&(t<hi)
            t=dr['time_ms']/1000-start;md=(t>lo)&(t<=hi)
            windows.append(dict(relative_s=[lo,hi],rate_Hz_allE_A_B_surround=d['rate_5ms_Hz'][m].mean(0).tolist(),
                causal_R_mean_and_range=[float(me['global_E_rate_Hz'][mm].mean()),float(me['global_E_rate_Hz'][mm].min()),float(me['global_E_rate_Hz'][mm].max())],
                mean_Graw=float(me['global_raw_conductance_ratio'][mm].mean()),
                mean_counterfactual_Zdot_per_s=dr['values'][md,:,0].mean(0).tolist()))
        field=d['field_rate_5ms_Hz'][d['relative_time_5ms_s']>=5].mean(0)
        np.save(OUT/f'{name}_tail_field_Hz.npy',field)
        results.append(dict(name=name,windows=windows,complete10s=True,brief_events=row['tail_brief_events']))
    assert np.array_equal(inputs[0],inputs[1])
    assert engines[0]['rng_state']==engines[1]['rng_state'] and engines[0]['xi']==engines[1]['xi']
    result=dict(status='COMPLETE_TWO_NATIVE_CONSTANT_BACKGROUND_CONTROLS',rows=results,
        paired_recorded_inputs_exact=True,final_external_RNG_and_xi_exact=True,
        formal_bifurcation_allowed=False,counts_as_autonomous_loop=False,producer_sha256=sha(__file__))
    write(OUT/'result.json',result);print(result,flush=True)


def supervise():
    assert not (OUT/'supervisor.json').exists()
    predecessor=ROOT/'individual_source_spectral_independent_value'
    while not (predecessor/'result.json').exists():
        write(OUT/'supervisor.json',dict(status='WAITING_INDEPENDENT_VALUE',pid=os.getpid(),updated_epoch=time.time()))
        if (predecessor/'supervisor.json').exists() and read(predecessor/'supervisor.json')['status']=='FAILED':raise RuntimeError('Independent sampler failed')
        time.sleep(10)
    write(OUT/'supervisor.json',dict(status='RUNNING',pid=os.getpid(),updated_epoch=time.time()))
    jobs=[]
    for i in range(2):
        log=(OUT/f'worker{i}.log').open('w')
        p=subprocess.Popen([PYTHON,__file__,'worker','--index',str(i)],stdout=log,stderr=subprocess.STDOUT);jobs.append((p,log))
    codes=[]
    for p,log in jobs:codes.append(p.wait());log.close()
    if any(codes):
        write(OUT/'supervisor.json',dict(status='FAILED',codes=codes));raise RuntimeError(codes)
    finish();write(OUT/'supervisor.json',dict(status='COMPLETE',updated_epoch=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','worker','supervise','finish']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    prepare() if a.command=='prepare' else worker(a.index) if a.command=='worker' else supervise() if a.command=='supervise' else finish()
