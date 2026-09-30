#!/usr/bin/env python3
"""Two native counterparts to the resolved copied-network exit interval."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import copy
import shutil
import time
import numpy as np
from campaign import ROOT,NATIVE,read,write,sha
import dynamic_mean_history_pair as model
import observe_source_aggregation as native
from run_topic4_recovery_window import assert_same_state

OUT=ROOT/'native_mean_exit_interval_v2'
INITIAL=model.base.MATCHED/'runs/high_history_constant_background/checkpoint.pkl'
JOBS={'high_K9p4625':9.4625,'high_K9p5':9.5}
MODEL=OUT/'model'


def configure(name):
    native.OUT=OUT;native.PARENT=INITIAL.parent;native.INITIAL=INITIAL;native.NAME=name
    protocol=read(OUT/'protocol.json');native.native.OUT=OUT;native.native.prepare=lambda:protocol
    fields=dict(np.load(OUT/'fields'/f'{name}.npz'))
    native.native.fields=lambda z,k:(fields['Z'].copy(),fields['K'].copy())


def prepare():
    r=read(ROOT/'mean_exit_interval/analysis/result.json')
    assert r['status']=='COMPLETE_FOUR_REGISTERED_CONDITIONAL_PROBES'
    assert r['all_future_numerical_RNGs_paired_bitwise']
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    contract=dict(status='REGISTERED_TWO_NATIVE_INTERNAL_CONDITIONS',created_epoch=time.time(),
        question='Do the native spatial network and the correspondence-validated copied model show the same activity and core-resource response at the upper active value and the prior exit value under exactly matched histories?',
        design='Exactly two native10s continuations from the completed82s fixed-backgroundK9.35 native highstate, counterpart to the model100000clock state. Only heldK changes to9.4625or9.5; fullZfield, graph, percellthresholds, nu/Poissonlaw, fullphysicalhistory/M/G/pendingpulses andexternalRNG retained. Future native noise paired between these two interventions. Match the completedmodel9.4625 and one newmodel9.5 trajectory from exactly the samemodel100000clock highstate. The old9.5 comparison began10s earlier and is not substituted for this same-history check. No timealignment.',
        motivation='The preceding four probes retain high activity through9.4625 and quiet history at9.35. The old9.5 model exited but used an earlier completehighstate; its apparent boundary must be checked with the same currenthighstate. Before interpreting the refined transition as a relevant branch candidate, independently test the two adjacent biological parameter conditions in the native graph. This does not create new autonomouscycles or independentseeds.',
        guards='For each0-5s and5-10s: weightedspatialRMS<=10Hz, eachcoremeanrate difference<=10Hz, eachcorecounterfactualZdrift difference<=.01/s. Both native/model remainR<200 andGraw<.1 if candidate does; firstcontinuous100msR<=5 either absentinboth or onsetdifference<=.25s. These original correspondence tolerances are not relaxed for interiorpoints.',
        stop='Exactly two10s native trajectories plus one10s R64 model9.5 counterpart. No automatic extension, extraKpoint, noise, root or label. If interiorcorrespondence fails, report its time/space/gate discrepancy before any formalbranch claim.',
        source=str(INITIAL),source_sha256=sha(INITIAL),nu_file=str(model.base.MATCHED/'fixed_external_per_ms.npy'),
        producer_sha256=sha(__file__),jobs=JOBS,formal_bifurcation_allowed=False)
    write(OUT/'contract.json',contract);shutil.copy2(__file__,OUT/'producer.py')
    p=copy.deepcopy(read(NATIVE/'protocol.json'));p.update(stage='TWO_NATIVE_UPPER_INTERIOR_CORRESPONDENCE',created_epoch=time.time(),deadline_epoch=time.time()+86400)
    write(OUT/'protocol.json',p);shutil.copy2(NATIVE/'geometry.npz',OUT/'geometry.npz');(OUT/'fields').mkdir()
    for name,K in JOBS.items():
        with np.load(ROOT/'mean_exit_interval/fields/high_K9p4625.npz') as f:
            np.savez_compressed(OUT/'fields'/f'{name}.npz',Z=f['Z'],K=f['K']*(K/9.4625))
        configure(name);native.native.make_job(name,str(INITIAL),10.,True,.21,K,False)
        jobpath=OUT/'jobs'/f'{name}.json';job=read(jobpath)
        job.update(held_fields_file=str(OUT/'fields'/f'{name}.npz'),held_fields_sha256=sha(OUT/'fields'/f'{name}.npz'),
            external_expected_rate_override=dict(path=contract['nu_file'],applies='Every step before originalPoisson draw'),
            probe_reason='Native independent condition check at same82s highhistory as correspondingmodel')
        write(jobpath,job);checkpoint=OUT/'runs'/name/'checkpoint.pkl';saved=native.native.read_pickle(checkpoint);saved['job']=job
        native.native.base.save_pickle(checkpoint,saved)
        expected=copy.deepcopy(native.native.read_pickle(INITIAL)['engine'])
        expected['termination_mechanism']['sahp_g'][:]=np.load(OUT/'fields'/f'{name}.npz')['K']
        assert expected['step']==820000;assert_same_state(saved['engine'],expected)
    write(OUT/'initial_gate.json',dict(status='PASS',both_full82s_states_exact_except_K=True))
    import mean_exit_interval as carried
    (MODEL/'fields').mkdir(parents=True)
    shutil.copy2(OUT/'fields/high_K9p5.npz',MODEL/'fields/high_K9p5.npz')
    source=carried.SOURCES['high']
    dependencies=read(ROOT/'mean_exit_interval/contract.json')['dependencies']
    dependencies[__file__]=sha(__file__)
    write(MODEL/'contract.json',dict(dependencies=dependencies,sources={'high':dict(path=str(source),sha256=sha(source))},formal_bifurcation_allowed=False))


def run(name,device):
    c=read(OUT/'contract.json');assert c['producer_sha256']==sha(__file__) and c['source_sha256']==sha(INITIAL)
    assert read(OUT/'initial_gate.json')['status']=='PASS';configure(name)
    dest=OUT/'runs'/name;assert not (dest/'fixed_background_progress.json').exists()
    nu=np.load(c['nu_file']);backend=native.gpu.cuda_backend.wrap_simulator;calls=0
    def wrap(original,device_index):
        fast=backend(original,device_index=device_index)
        def simulate(params,net,*args,**kw):
            old=kw.get('input_observer')
            def inputs(tm,actual,xi):
                nonlocal calls
                actual[:]=nu;calls+=1
                if old is not None:old(tm,actual,xi)
            kw['input_observer']=inputs
            return fast(params,net,*args,**kw)
        return simulate
    native.gpu.cuda_backend.wrap_simulator=wrap
    write(dest/'fixed_background_progress.json',dict(status='RUNNING',pid=os.getpid(),updated_epoch=time.time()))
    try:native.gpu.worker(OUT,name,device)
    finally:native.gpu.cuda_backend.wrap_simulator=backend
    assert calls==100000 and read(dest/'result.json')['status']=='COMPLETE'
    runtime=read(dest/'runtime_backend.json');runtime.update(physics_and_job_unchanged=False,neuronal_equations_unchanged=True,
        registered_external_expected_rate_override=read(OUT/'jobs'/f'{name}.json')['external_expected_rate_override'])
    write(dest/'runtime_backend.json',runtime)
    for filename in ['result.json','progress.json']:
        x=read(dest/filename);x['runtime_backend']=runtime;write(dest/filename,x)
    end=native.native.read_pickle(dest/'checkpoint.pkl')['engine'];assert end['step']==920000
    with np.load(OUT/'fields'/f'{name}.npz') as f:
        assert np.array_equal(end['slow']['z'][:32000],f['Z']) and np.array_equal(end['termination_mechanism']['sahp_g'],f['K'])
    write(dest/'fixed_background_result.json',dict(status='COMPLETE',steps=calls,held_fields_exact=True,formal_bifurcation_allowed=False))
    write(dest/'fixed_background_progress.json',dict(status='COMPLETE',updated_epoch=time.time()))
    print(name,'NATIVE COMPLETE',flush=True)


def run_model(device):
    import mean_exit_interval as carried
    carried.OUT=MODEL;carried.JOBS={'high_K9p5':('high',9.5)}
    carried.run('high_K9p5',device)
    with np.load(MODEL/'runs/high_K9p5/final_state.npz') as end, np.load(ROOT/'mean_exit_interval/runs/high_K9p4625/final_state.npz') as ref:
        assert all(np.array_equal(end[k],ref[k]) for k in ['rng','external_rng','clock'])
    p=MODEL/'runs/high_K9p5/result.json';x=read(p);x['both_RNGs_paired_with_four_probes']=True;write(p,x)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','model']);p.add_argument('--name',choices=list(JOBS));p.add_argument('--device',type=int,default=0);v=p.parse_args()
    if v.command=='prepare':prepare()
    elif v.command=='model':run_model(v.device)
    else:run(v.name,v.device)
