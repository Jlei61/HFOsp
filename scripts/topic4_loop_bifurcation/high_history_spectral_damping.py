#!/usr/bin/env python3
"""Bounded self-generated high-history spectral test with fresh numerical streams."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse, shutil, subprocess, time
from pathlib import Path
import numpy as np
from campaign import ROOT, PYTHON, read, write, sha
import run_spectral_closure_pilot as implementation
import damped_spectral_pilot as mixing
from high_history_spectral_value import OUT as SOURCE, N

OUT=ROOT/'high_history_spectral_damped_pilot'
LIMIT=12
REPLICAS=64
ALPHA=.1


def prepare():
    OUT.mkdir(exist_ok=True); assert not (OUT/'contract.json').exists()
    assert read(SOURCE/'result.json')['status']=='COMPLETE_SINGLE_HIGH_HISTORY_SPECTRAL_RESPONSE'
    native=read(ROOT/'native_K9p35_constant_background_pair_v2/result.json')['rows'][1]['windows'][1]
    assert native['causal_R_mean_and_range'][2]<200
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_HIGH_HISTORY_SELF_GENERATED_UPDATES',created_epoch=time.time(),
        question='Does the second native high-history state remain a relevant candidate when all recurrent source rates and spectra are supplied by the spectral model itself, rather than the native initial source record?',
        motivation='Completed matching-background native10s control preserves double-corehigh withGoff. One localF(native) preserves that segment and coreZdrift but has1.154Hz spatialRMS and12.338Hz maximumfield bias. One response is not a self-generated state.',
        design='At most12 numerical updates Xnext=.9X+.1F(X), starting solely from the completed first predicted high-history output.40000targets times64 replicas. Same individual graph, thresholds, localM, stationaryG, fixedexpectedexternalnu,2s periodic spectrum and3-4s burn. No native source statistics are fed back after this initialization.',
        streams='Fresh GaussianFourier, externalPoisson and burnphase seeds for every generation: originalseed +2000000+1000000*generation. This changes the numerical estimator, not the expected physical map. Report raw F(X)-X, not alpha-scaled updates; residual noise and candidate uncertainty remain.',
        relevance_stop='Stop at12; stop earlier if source-input or response-implied G becomespositive, either core mean differs from completed matched native tail bymorethan10Hz, or cell-weightedEfield RMS exceeds10Hz. These retain prior10Hz relevance guards plus the already observed Goff segment; not precision acceptance thresholds. No adaptation of alpha or extra iterations.',
        comparison='Use completed high-history native constant-background5-10s tail as primary relevance reference. Original72-74s variable-background field remains descriptive only.',
        diagnostics='Save mixed candidate and unscaled response separately, individualrates/PSDs and all64 replica statistics, first/second1s andMstart/end, coreZdrift, spatialfield and Gsegment. Common deterministic equations and original numerical kernels unchanged.',
        boundaries='No root or physical stability certification. A numerical iteration is not a time trajectory; this conditional test is not an autonomous exit or recovery. No derivative, frequency, Kscan or newnative job is automatically dispatched.',
        replicas=REPLICAS,alpha=ALPHA,max_updates=LIMIT,
        unchanged_map_sha256=sha(implementation.__file__),mixing_sha256=sha(mixing.__file__),
        sampler_sha256=sha(Path(__file__).with_name('individual_spectral_sampler.py')),
        producer_sha256=sha(__file__),formal_bifurcation_allowed=False,counts_as_autonomous_loop=False))
    for name in ['parameters.npz','original_ampa_jump.npz','original_gaba_jump.npz','filter_power.npy','threshold_identity.json','implementation_qa.json','preparation_qa.json']:
        (OUT/name).symlink_to(SOURCE/name)
    initial=OUT/'generation_0';initial.mkdir()
    for name in ['source_PSD.npy','source_rate_Hz.npy']:
        (initial/name).symlink_to(SOURCE/'generation_1'/name)
    write(initial/'complete.json',dict(status='INITIAL_MODEL_PREDICTED_HIGH_OUTPUT_ONLY',source=str(SOURCE/'generation_1')))
    shutil.copy2(__file__,OUT/'producer.py')


def configure():
    c=read(OUT/'contract.json')
    assert c['unchanged_map_sha256']==sha(implementation.__file__)
    assert c['mixing_sha256']==sha(mixing.__file__)
    assert c['sampler_sha256']==sha(Path(__file__).with_name('individual_spectral_sampler.py'))
    assert c['producer_sha256']==sha(__file__)
    implementation.OUT=OUT;implementation.GENERATIONS=LIMIT;implementation.REPLICAS=REPLICAS
    mixing.OUT=OUT;mixing.SOURCE=SOURCE;mixing.ALPHA=ALPHA


def worker(gen,part):
    import cupy as cp
    configure();original_cp=cp.random.RandomState;original_np=np.random.default_rng;original_run=implementation.run
    offset=2000000+1000000*gen
    cp.random.RandomState=lambda seed=None:original_cp(int(seed)+offset)
    np.random.default_rng=lambda seed=None:original_np(int(seed)+offset)
    def fresh_run(cp,fn,ie,ii,p,cfg,R,burn,extra,seed,first_cell,replay=False):
        assert not replay and R==REPLICAS
        return original_run(cp,fn,ie,ii,p,cfg,R,burn,extra,seed+offset,first_cell,replay)
    implementation.run=fresh_run
    try: implementation.worker(gen,part,part,128)
    finally:
        cp.random.RandomState=original_cp;np.random.default_rng=original_np;implementation.run=original_run


def supervise():
    configure();assert not (OUT/'supervisor.json').exists()
    matched=ROOT/'native_K9p35_constant_background_pair_v2'
    field=np.load(matched/'high_history_constant_background_tail_field_Hz.npy')
    target=read(matched/'result.json')['rows'][1]['windows'][1]['rate_Hz_allE_A_B_surround']
    raw=np.load(OUT/'parameters.npz');display=raw['display'][:32000];count=np.bincount(display,minlength=400)
    write(OUT/'supervisor.json',dict(status='RUNNING',pid=os.getpid(),max_updates=LIMIT,updated_epoch=time.time()))
    results=[];reason=None
    for gen in range(1,LIMIT+1):
        folder=OUT/f'generation_{gen}';folder.mkdir()
        for name,shape in [('source_PSD.npy',(40000,N//2+1)),('source_rate_Hz.npy',(40000,))]:
            a=np.lib.format.open_memmap(folder/name,mode='w+',dtype='f8',shape=shape);a[:]=np.nan;a.flush();del a
        jobs=[]
        for part in range(2):
            log=(folder/f'worker_part{part}.log').open('w')
            p=subprocess.Popen([PYTHON,__file__,'worker','--generation',str(gen),'--part',str(part)],stdout=log,stderr=subprocess.STDOUT)
            jobs.append((p,log))
        codes=[]
        for p,log in jobs:codes.append(p.wait());log.close()
        if any(codes):
            write(OUT/'supervisor.json',dict(status='FAILED',generation=gen,codes=codes));raise RuntimeError(codes)
        implementation.collect(gen);r=mixing.blend(gen)
        rate=np.load(folder/'response_rate_Hz.npy')
        model=np.bincount(display,weights=rate[:32000],minlength=400)/np.maximum(count,1)
        rms=float(np.sqrt(np.average((model-field)**2,weights=count)))
        core=[r['rows'][j]['output_rate_Hz']-target[j] for j in [1,2]]
        r.update(matched_native_field_RMS_Hz=rms,matched_native_core_differences_Hz=core,
            all_three_stream_seed_offset=2000000+1000000*gen,
            actual_native_field_reference='The legacy actual_native_field_RMS_Hz is against variable-background72-74s, not the matchedconstantbackgroundtail. Primaryguardusesmatched_native_field_RMS_Hz.')
        write(folder/'complete.json',r);results.append(r)
        write(OUT/'progress.json',dict(status='RUNNING',completed_updates=gen,max_updates=LIMIT,
            unscaled_E_I_rate_residual_RMS_Hz=[r['rows'][j]['individual_rate_RMS_change_Hz'] for j in [0,4]],
            matched_native_field_RMS_Hz=rms,matched_native_core_differences_Hz=core,
            G_used=r['G_used'],G_from_output=r['G_implied_by_output'],updated_epoch=time.time()))
        if r['G_used']>0 or r['G_implied_by_output']>0: reason='G_ACTIVATION_SEGMENT_CHANGED';break
        if rms>10 or max(abs(x) for x in core)>10: reason='NATIVE_SPATIAL_RELEVANCE_GUARD';break
    status='STOPPED_RELEVANCE_GUARD' if reason else 'COMPLETE_TWELVE_FRESH_STREAM_UPDATES'
    write(OUT/'result.json',dict(status=status,stop_reason=reason,updates=results,
        root_certified=False,physical_stability_established=False,formal_bifurcation_allowed=False,producer_sha256=sha(__file__)))
    write(OUT/'supervisor.json',dict(status=status,completed_updates=len(results),stop_reason=reason,updated_epoch=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','supervise','worker'])
    p.add_argument('--generation',type=int);p.add_argument('--part',type=int,default=0);a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='supervise':supervise()
    else:worker(a.generation,a.part)
