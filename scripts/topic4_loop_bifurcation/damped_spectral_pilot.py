#!/usr/bin/env python3
"""Bounded numerical damping of the SAME stationary input-output map."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,shutil,subprocess,time
from pathlib import Path
import numpy as np
from campaign import ROOT,PYTHON,read,write,sha
import run_spectral_closure_pilot as implementation
from prepare_spectral_closure_pilot import OUT as SOURCE,N

OUT=ROOT/'individual_source_spectral_damped_pilot'
ALPHA=.1
LIMIT=12


def prepare():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    assert read(SOURCE/'supervisor.json')['status']=='COMPLETE_THREE_BOUNDED_GENERATIONS'
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_NUMERICAL_DAMPING',created_epoch=time.time(),
        question='Does numerical underrelaxation remove the growing inhibitory residual of parallel-Jacobi stationary spectral iteration while preserving the relevant native conditional field?',
        trigger='The completed three-generation pilot retains E field within0.11Hz but inhibitory cell update RMS grows0.926 to5.241Hz. That is a numerical map observation, not physical-time instability.',
        equation='X_next=(1-alpha)*X+alpha*F(X), X=(individual source means, diagonal source PSDs), alpha0.1. F is the unchanged original pilot map. For nonzero alpha its exact fixed points are the same; intermediate convex states need not be exact finite-window binary-spike spectra.',
        design='At most12 updates,40000targets x16 independent replicas per local assay. Same physical equations, actual thresholds, original weights, stationary G/M treatment, external Poisson,2s FFT,3-4s burn and common streams as previous pilot. Start solely from original pilot generation1 predicted output, not a fresh native spectrum.',
        stop='Stop at12 updates; stop earlier if whole-E field discrepancy from the development native reference exceeds10Hz or either core mean differs by10Hz. These are relevance guards, not accuracy acceptance criteria. No automatic continuation, damping-factor search or physical stability claim.',
        reference='One native developmenttrajectory with time-varying external drive; pilot still uses fixed percell mean external expected rates. This mismatch and omitted source crosscorrelations remain.',
        diagnostics='Always report the undamped F(X)-X residual, not alpha times it. Save raw outputs separately from mixed numerical states. Observe full spectra, inhibitory/E residuals, fields, G segment and counterfactual core Z drift.',
        alpha=ALPHA,max_updates=LIMIT,producer_sha256=sha(__file__),
        unchanged_map_sha256=sha(implementation.__file__),formal_bifurcation_allowed=False,counts_as_autonomous_loop=False))
    for name in ['parameters.npz','original_ampa_jump.npz','original_gaba_jump.npz','filter_power.npy','threshold_identity.json','implementation_qa.json','preparation_qa.json']:
        (OUT/name).symlink_to(SOURCE/name)
    initial=OUT/'generation_0';initial.mkdir()
    for name in ['source_PSD.npy','source_rate_Hz.npy']:
        (initial/name).symlink_to(SOURCE/'generation_1'/name)
    write(initial/'complete.json',dict(status='INITIAL_PREDICTED_OUTPUT_ONLY',source=str(SOURCE/'generation_1')))
    shutil.copy2(__file__,OUT/'producer.py')


def configure():
    assert read(OUT/'contract.json')['unchanged_map_sha256']==sha(implementation.__file__)
    implementation.OUT=OUT;implementation.GENERATIONS=LIMIT


def blend(gen):
    folder=OUT/f'generation_{gen}';previous=OUT/f'generation_{gen-1}'
    before=np.load(previous/'source_rate_Hz.npy');rate=np.load(folder/'source_rate_Hz.npy')
    original_result=read(folder/'complete.json')
    (folder/'source_rate_Hz.npy').rename(folder/'response_rate_Hz.npy')
    (folder/'source_PSD.npy').rename(folder/'response_PSD.npy')
    mixed_rate=(1-ALPHA)*before+ALPHA*rate
    np.save(folder/'source_rate_Hz.npy',mixed_rate)
    old=np.load(previous/'source_PSD.npy',mmap_mode='r')
    new=np.load(folder/'response_PSD.npy',mmap_mode='r')
    mixed=np.lib.format.open_memmap(folder/'source_PSD.npy',mode='w+',dtype='f8',shape=new.shape)
    maximum=0.
    for lo in range(0,40000,128):
        a=np.asarray(old[lo:lo+128]);b=np.asarray(new[lo:lo+128]);v=(1-ALPHA)*a+ALPHA*b
        assert np.isfinite(v).all() and v.min()>=0 and np.all(v[:,0]==0)
        mixed[lo:lo+len(v)]=v
        maximum=max(maximum,float(abs((v-a)-ALPHA*(b-a)).max()))
    mixed.flush()
    # Save development-native comparison separately; base collect's reference is the initial predicted output.
    raw=dict(np.load(OUT/'parameters.npz'));display=raw['display'][:32000];count=np.bincount(display,minlength=400)
    field=lambda r:np.bincount(display,weights=r[:32000],minlength=400)/np.maximum(count,1)
    native=np.load(SOURCE/'generation_0/source_rate_Hz.npy')
    nativefield=float(np.sqrt(np.average((field(rate)-field(native))**2,weights=count)))
    cores=[]
    for reg in [0,1]:
        mask=(np.arange(40000)<32000)&(raw['region']==reg)
        cores.append(float(rate[mask].mean()-native[mask].mean()))
    result=dict(original_result,numerical_alpha=ALPHA,undamped_response_saved=True,
        initial_reference_is_predicted_generation1=True,actual_native_field_RMS_Hz=nativefield,
        core_mean_difference_from_native_Hz=cores,
        blend_float_identity_max_abs_error=maximum,root_established=False,physical_stability_established=False)
    write(folder/'complete.json',result)
    return result


def supervise():
    configure();assert not (OUT/'supervisor.json').exists()
    write(OUT/'supervisor.json',dict(status='RUNNING',pid=os.getpid(),created_epoch=time.time(),max_updates=LIMIT))
    results=[]
    for gen in range(1,LIMIT+1):
        folder=OUT/f'generation_{gen}';folder.mkdir()
        for name,shape in [('source_PSD.npy',(40000,N//2+1)),('source_rate_Hz.npy',(40000,))]:
            a=np.lib.format.open_memmap(folder/name,mode='w+',dtype='f8',shape=shape);a[:]=np.nan;a.flush();del a
        jobs=[]
        for part in range(2):
            log=(folder/f'worker_part{part}.log').open('w')
            proc=subprocess.Popen([PYTHON,__file__,'worker','--generation',str(gen),'--part',str(part)],stdout=log,stderr=subprocess.STDOUT)
            jobs.append((proc,log))
        codes=[]
        for proc,log in jobs:codes.append(proc.wait());log.close()
        if any(codes):
            write(OUT/'supervisor.json',dict(status='FAILED',generation=gen,codes=codes));raise RuntimeError(codes)
        implementation.collect(gen);result=blend(gen);results.append(result)
        write(OUT/'progress.json',dict(status='RUNNING',completed_updates=gen,max_updates=LIMIT,
            E_I_undamped_rate_residual_RMS_Hz=[result['rows'][j]['individual_rate_RMS_change_Hz'] for j in [0,4]],
            native_field_RMS_Hz=result['actual_native_field_RMS_Hz'],updated_epoch=time.time()))
        if result['actual_native_field_RMS_Hz']>10 or max(abs(x) for x in result['core_mean_difference_from_native_Hz'])>10:
            status='STOPPED_RELEVANCE_GUARD';break
    else:status='COMPLETE_TWELVE_BOUNDED_UPDATES'
    write(OUT/'result.json',dict(status=status,updates=results,root_established=False,
        formal_bifurcation_allowed=False,physical_stability_established=False,producer_sha256=sha(__file__)))
    write(OUT/'supervisor.json',dict(status=status,completed_updates=len(results),updated_epoch=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','worker','supervise'])
    p.add_argument('--generation',type=int);p.add_argument('--part',type=int,default=0);a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='worker':configure();implementation.worker(a.generation,a.part,a.part,256)
    else:supervise()
