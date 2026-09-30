#!/usr/bin/env python3
"""One independent value evaluation of the final bounded spectral candidate.

The current sampler and its physical map are reused unchanged. Only the number
of numerical replicas and all three random streams change. Extra Fourier
statistics read the returned spikes; they cannot feed back into the assay.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,shutil,subprocess,time
from pathlib import Path
import numpy as np
from campaign import ROOT,PYTHON,read,write,sha
from damped_spectral_pilot import OUT as CANDIDATE,SOURCE,N
import run_spectral_closure_pilot as implementation

OUT=ROOT/'individual_source_spectral_independent_value'
REPLICAS=64
OFFSET=1000000
HIGH=ROOT/'native_K9p35_high_history_source_spectra'


def prepare():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_INDEPENDENT_VALUE',created_epoch=time.time(),
        question='At the fixed candidate produced by the bounded12-step numerical solver, do independent realizations support small rate AND source-spectrum residuals, or did common-stream iteration conceal an error?',
        design='One unchanged F(X) evaluation of final mixed X12,40000individualtargets x64replicas, same2speriodic recurrent spectrum and3-4s burn. All Gaussian Fourier, external Poisson and burnphase streams receive a fresh seed offset1000000. Same original physical sampler, individual weights/thresholds/M/G and constant external expected drive. No correction step, newK or branch continuation.',
        selection='FinalX12 from the preregistered bounded solver, used only if all12updates complete and its original relevance guards do not stop it. Initialnative42-44s reference is only a development spatial comparison.',
        precision='Report F(X12)-X12 without multiplying by damping. Independent replica rateSEM, frequencywise periodogramSEM, and source-filtered varianceSEM describe this finite-window map only. ZeroSEM or weak channels remain explicitly unresolved; no ad hoc tolerance is added and no root certification follows automatically.',
        spectral_observer='Read-only FFT of returned output spikes, before the unchanged worker records its standard means/PSDs; save variance overreplicas. Does not alter any physical input, state, spike or RNG.',
        dispatch='Wait for bounded solver result and the separately authorized2s native high-history observer to finish; use two GPUs for this one value evaluation.',
        stopping='One value assay only, no automatic newiterations/parameterpoints. Any failure or substantial independent residual is retained and reviewed.',
        replicas=REPLICAS,seed_offset=OFFSET,unchanged_map_sha256=sha(implementation.__file__),
        unchanged_sampler_sha256=sha(Path(implementation.__file__).with_name('individual_spectral_sampler.py')),
        producer_sha256=sha(__file__),formal_bifurcation_allowed=False,counts_as_autonomous_loop=False))
    shutil.copy2(__file__,OUT/'producer.py')


def configure():
    c=read(OUT/'contract.json')
    assert c['unchanged_map_sha256']==sha(implementation.__file__)
    assert c['unchanged_sampler_sha256']==sha(Path(implementation.__file__).with_name('individual_spectral_sampler.py'))
    implementation.OUT=OUT;implementation.GENERATIONS=1;implementation.REPLICAS=REPLICAS


def worker(part):
    import cupy as cp
    cp.cuda.Device(part).use();configure()
    cp_random=cp.random.RandomState;np_random=np.random.default_rng;original_run=implementation.run
    cp.random.RandomState=lambda seed=None:cp_random(int(seed)+OFFSET)
    np.random.default_rng=lambda seed=None:np_random(int(seed)+OFFSET)
    folder=OUT/'generation_1'
    spectral_sem=np.lib.format.open_memmap(folder/'source_PSD_SEM.npy',mode='r+')
    filtered_sem=np.lib.format.open_memmap(folder/'source_filtered_variance_SEM.npy',mode='r+')
    H=cp.asarray(np.load(OUT/'filter_power.npy'));weights=cp.full(N//2+1,2.,dtype='f8');weights[[0,-1]]=1.
    def observed_run(cp,fn,ie,ii,p,cfg,R,burn,extra,seed,first_cell,replay=False):
        assert not replay and R==REPLICAS
        flags,st=original_run(cp,fn,ie,ii,p,cfg,R,burn,extra,seed+OFFSET,first_cell,replay)
        B=len(p);x=flags.T.reshape(B,R,N).astype('f8');f=cp.fft.rfft(x,axis=2);f[:,:,0]=0
        power=abs(f)**2
        spectral_sem[first_cell:first_cell+B]=(power.std(axis=1,ddof=1)/np.sqrt(R)).get()
        for q in range(2):
            samples=power@(H[q]*weights/N**2)
            filtered_sem[q,first_cell:first_cell+B]=(samples.std(axis=1,ddof=1)/np.sqrt(R)).get()
        return flags,st
    implementation.run=observed_run
    try:implementation.worker(1,part,part,128)
    finally:
        implementation.run=original_run;cp.random.RandomState=cp_random;np.random.default_rng=np_random
        spectral_sem.flush();filtered_sem.flush()
    write(folder/f'precision_part{part}_complete.json',dict(status='COMPLETE',replicas=REPLICAS,
        all_three_streams_seed_offset=OFFSET,read_only_spectral_observer=True))


def finish():
    configure();implementation.collect(1)
    folder=OUT/'generation_1';raw=dict(np.load(OUT/'parameters.npz'))
    r=np.load(OUT/'generation_0/source_rate_Hz.npy');output=np.load(folder/'source_rate_Hz.npy')
    samples=np.concatenate([np.load(folder/f'part{i}_statistics.npz')['replica_statistics'] for i in range(2)])
    counts=samples[:,:,:2].sum(2)/2;rate_sem=counts.std(1,ddof=1)/np.sqrt(REPLICAS)
    x=np.load(OUT/'generation_0/source_PSD.npy',mmap_mode='r');y=np.load(folder/'source_PSD.npy',mmap_mode='r')
    sem=np.load(folder/'source_PSD_SEM.npy',mmap_mode='r');H=np.load(OUT/'filter_power.npy')
    w=np.full(N//2+1,2.);w[[0,-1]]=1
    PSDchange=np.empty(40000);expected_noise_L1=np.empty(40000);filter_change=np.empty((2,40000))
    for lo in range(0,40000,128):
        a=np.asarray(x[lo:lo+128]);b=np.asarray(y[lo:lo+128]);s=np.asarray(sem[lo:lo+128]);length=len(a)
        PSDchange[lo:lo+length]=abs(b-a)@w/N**2
        expected_noise_L1[lo:lo+length]=s@w/N**2
        for q in range(2):filter_change[q,lo:lo+length]=(b-a)@(H[q]*w/N**2)
    fsem=np.load(folder/'source_filtered_variance_SEM.npy');E=np.arange(40000)<32000
    rows=[]
    for label,mask in [('allE',E),('coreA',E&(raw['region']==0)),('coreB',E&(raw['region']==1)),('surroundE',E&(raw['region']==2)),('I',~E)]:
        residual=output[mask]-r[mask];s=rate_sem[mask]
        rows.append(dict(region=label,rate_residual_RMS_Hz=float(np.sqrt(np.mean(residual**2))),
            rate_MCSEM_RMS_Hz=float(np.sqrt(np.mean(s**2))),
            targets_exceeding6SEM=int(np.count_nonzero((abs(residual)>6*s)&(s>0))),
            zeroSEM_nonzero_residual=int(np.count_nonzero((s==0)&(abs(residual)>0))),
            mean_rate_input_output_Hz=[float(r[mask].mean()),float(output[mask].mean())],
            PSD_L1_residual_mean=float(PSDchange[mask].mean()),PSD_L1_SEM_sum_mean=float(expected_noise_L1[mask].mean()),
            filtered_variance_residual_RMS=np.sqrt(np.mean(filter_change[:,mask]**2,axis=1)).tolist(),
            filtered_variance_MCSEM_RMS=np.sqrt(np.mean(fsem[:,mask]**2,axis=1)).tolist()))
    np.savez_compressed(OUT/'independent_residuals.npz',rate_residual_Hz=output-r,rate_SEM_Hz=rate_sem,
        PSD_L1_residual=PSDchange,PSD_L1_SEM_sum=expected_noise_L1,filtered_variance_residual=filter_change)
    result=dict(status='COMPLETE_INDEPENDENT_FINITE_WINDOW_VALUE',rows=rows,
        original_collect=read(folder/'complete.json'),root_certified=False,formal_bifurcation_allowed=False,
        limitations='ZeroSEM/weak channels remain unresolved. SEM does not bound2s-periodic/window bias or the difference from a native network, and is not a simultaneous confidence interval over frequencies or targets.',
        producer_sha256=sha(__file__))
    write(OUT/'result.json',result);print(result,flush=True)


def supervise():
    assert not (OUT/'supervisor.json').exists()
    while not (CANDIDATE/'result.json').exists() or not (HIGH/'observer_audit.json').exists():
        write(OUT/'supervisor.json',dict(status='WAITING_REGISTERED_PREDECESSORS',pid=os.getpid(),updated_epoch=time.time()))
        for path in [CANDIDATE/'supervisor.json',HIGH/'observer_progress.json']:
            if path.exists() and read(path)['status']=='FAILED':raise RuntimeError(str(path))
        time.sleep(10)
    assert read(CANDIDATE/'result.json')['status']=='COMPLETE_TWELVE_BOUNDED_UPDATES'
    assert read(HIGH/'observer_audit.json')['status']=='PASS'
    configure()
    for name in ['parameters.npz','original_ampa_jump.npz','original_gaba_jump.npz','filter_power.npy','implementation_qa.json']:
        (OUT/name).symlink_to(SOURCE/name)
    initial=OUT/'generation_0';initial.mkdir()
    for name in ['source_PSD.npy','source_rate_Hz.npy']:
        (initial/name).symlink_to(CANDIDATE/'generation_12'/name)
    write(initial/'complete.json',dict(status='FIXED_CANDIDATE_X12',source=str(CANDIDATE/'generation_12')))
    folder=OUT/'generation_1';folder.mkdir()
    for name,shape in [('source_PSD.npy',(40000,N//2+1)),('source_PSD_SEM.npy',(40000,N//2+1)),
                       ('source_rate_Hz.npy',(40000,)),('source_filtered_variance_SEM.npy',(2,40000))]:
        a=np.lib.format.open_memmap(folder/name,mode='w+',dtype='f8',shape=shape);a[:]=np.nan;a.flush();del a
    write(OUT/'supervisor.json',dict(status='RUNNING_ONE_INDEPENDENT_VALUE',pid=os.getpid(),updated_epoch=time.time()))
    jobs=[]
    for part in range(2):
        log=(folder/f'worker_part{part}.log').open('w')
        p=subprocess.Popen([PYTHON,__file__,'worker','--part',str(part)],stdout=log,stderr=subprocess.STDOUT);jobs.append((p,log))
    codes=[]
    for p,log in jobs:codes.append(p.wait());log.close()
    if any(codes):
        write(OUT/'supervisor.json',dict(status='FAILED',codes=codes));raise RuntimeError(codes)
    finish();write(OUT/'supervisor.json',dict(status='COMPLETE',updated_epoch=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','supervise','worker']);p.add_argument('--part',type=int,default=0);a=p.parse_args()
    prepare() if a.command=='prepare' else supervise() if a.command=='supervise' else worker(a.part)
