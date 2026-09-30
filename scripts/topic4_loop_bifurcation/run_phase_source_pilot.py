#!/usr/bin/env python3
"""Three finite source-phase map updates; not physical-time dynamics."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,shutil,subprocess,time
from pathlib import Path
import numpy as np
from scipy import sparse
from campaign import ROOT,PYTHON,read,write,sha
from prepare_phase_source_closure import OUT,L,N,OPS,project_phase
from individual_spectral_sampler import make_parameters
import coherent_phase_sampler as sampler

R=64
BATCH=64
GENERATIONS=3
SEED=940727


def causal_phase(prob):
    z=np.exp(-2j*np.pi*np.arange(L//2+1)/L);a=np.exp(-.1/15)
    return np.fft.irfft(np.fft.rfft(prob[:32000].mean(0))/(.015*(1-a*z)),n=L)


def worker(generation,part):
    import cupy as cp
    import cupyx.scipy.sparse as csp
    cp.cuda.Device(part).use();cp.get_default_memory_pool().set_limit(size=17*2**30)
    previous=OUT/f'generation_{generation-1}';folder=OUT/f'generation_{generation}'
    assert 1<=generation<=GENERATIONS
    p=read(OPS/'prepared.json')['params'];raw=dict(np.load(OUT/'parameters.npz'))
    probability=np.load(previous/'phase_spike_probability.npy');rates=np.load(previous/'source_rate_Hz.npy')
    causal=causal_phase(probability);assert causal.max()<200
    source=cp.asarray(np.load(previous/'residual_PSD.npy',mmap_mode='r'))
    H=cp.asarray(np.load(OUT/'filter_power.npy'));means=np.load(previous/'current_mean.npy');waves=np.load(previous/'current_phase_wave.npy')
    lo0,hi0=part*20000,(part+1)*20000;W=[]
    for kind in ['ampa','gaba']:
        a=sparse.load_npz(OUT/f'original_{kind}_jump.npz').tocsr()[lo0:hi0];W.append(csp.csr_matrix(a.multiply(a)))
    destinations={name:np.lib.format.open_memmap(folder/(name+'.npy'),mode='r+') for name in
        ['residual_PSD','source_rate_Hz','phase_spike_probability']}
    carry=globals().get('CARRY_PHASE_HISTORY',False)
    if carry:
        import phase_history_sampler as history_sampler
        history=np.load(previous/'replica_V_ref_M.npy',mmap_mode='r')
        startphase=np.load(previous/'replica_initial_phase.npy',mmap_mode='r')
        nextstate=np.lib.format.open_memmap(folder/'replica_V_ref_M.npy',mode='r+')
        nextphase=np.lib.format.open_memmap(folder/'replica_initial_phase.npy',mode='r+')
        history_fn=history_sampler.kernel(cp)
    fn=sampler.kernel(cp);summaries=[];started=time.time();parseval=0.;mean_difference=0.
    seed=SEED+generation*1000000
    for lo in range(lo0,hi0,BATCH):
        hi=min(lo+BATCH,hi0);B=hi-lo;local=slice(lo-lo0,hi-lo0);rng=cp.random.RandomState(seed+lo)
        write(folder/f'progress_part{part}.json',dict(status='RUNNING',pid=os.getpid(),generation=generation,
            completed=lo-lo0,total=20000,updated_epoch=time.time()))
        currents=[]
        for q,ss in enumerate([slice(0,32000),slice(32000,40000)]):
            P=(W[q][local]@source[ss])*H[q];assert bool(cp.isfinite(P).all()) and float(P.min())>=0
            real=rng.standard_normal((B,R,N//2+1));imag=rng.standard_normal(real.shape)
            f=(real+1j*imag)*cp.sqrt(P[:,None,:]/2);f[:,:,0]=0;f[:,:,-1]=real[:,:,-1]*cp.sqrt(P[:,-1,None])
            x=cp.fft.irfft(f,n=N,axis=2)+cp.asarray(means[q,lo:hi])[:,None,None]
            currents.append(cp.ascontiguousarray(x.transpose(2,0,1).reshape(N,B*R)));del P,real,imag,f,x
        par,cfg=make_parameters(raw,np.arange(lo,hi),rates[lo:hi],0.,p)
        extra=np.random.default_rng(seed+100000+lo).integers(0,10001,size=B*R,dtype='i4')
        offsets=(np.asarray(startphase[lo:hi]).copy().ravel() if carry else
            np.random.default_rng(seed+300000+lo).integers(0,L,size=B*R,dtype='i4'))
        wave=np.ascontiguousarray(waves[:,lo:hi].transpose(0,2,1))
        if carry:
            if lo==lo0:
                default=np.broadcast_to(par[:,[9,10,11]][:,None,:],(B,R,3)).copy()
                a,b=history_sampler.run(cp,history_fn,*currents,par,cfg,R,30000,extra,seed,lo,wave,offsets,default)
                aa,bb=sampler.run(cp,fn,*currents,par,cfg,R,30000,extra,seed,lo,wave,offsets)
                assert bool(cp.array_equal(a,aa)) and bool(cp.array_equal(b,bb))
                write(folder/f'history_kernel_qa_part{part}.json',dict(status='PASS',uniform_initial_state_every_spike_and_statistic_bitwise=True))
                del a,b,aa,bb,default
            flags,st=history_sampler.run(cp,history_fn,*currents,par,cfg,R,30000,extra,seed,lo,wave,offsets,np.asarray(history[lo:hi]))
        else:
            flags,st=sampler.run(cp,fn,*currents,par,cfg,R,30000,extra,seed,lo,wave,offsets)
        statistics=st.get();summaries.append(statistics);del currents,st
        if carry:
            nextstate[lo:hi]=statistics[:,:,[13,14,3]]
            nextphase[lo:hi]=(offsets.reshape(B,R)+30000+extra.reshape(B,R)+N)%L
        base=(30000+extra.reshape(B,R)+offsets.reshape(B,R))%L
        phase_count=np.zeros((B,L));phase_exposure=np.zeros((B,L))
        for psi in range(L):
            value=flags[psi::L].sum(0).reshape(B,R).get();idx=(base+psi)%L
            target=np.broadcast_to(np.arange(B)[:,None],idx.shape)
            np.add.at(phase_count,(target,idx),value)
            np.add.at(phase_exposure,(target,idx),len(range(psi,N,L)))
        assert np.array_equal(phase_count.sum(1),statistics[:,:,:2].sum((1,2)))
        assert np.all(phase_exposure.sum(1)==N*R)
        profile=phase_count/phase_exposure
        destinations['phase_spike_probability'][lo:hi]=profile
        r=profile.mean(1)*10000;destinations['source_rate_Hz'][lo:hi]=r
        raw_rate=statistics[:,:,:2].sum(2).mean(1)/2
        mean_difference=max(mean_difference,float(abs(r-raw_rate).max()))
        x=flags.T.reshape(B,R,N).astype('f8');del flags
        pp=cp.asarray(profile);target=cp.arange(B)[:,None]
        for psi in range(L):x[:,:,psi::L]-=pp[target,cp.asarray((base+psi)%L)][:,:,None]
        F=cp.fft.rfft(x,axis=2);F[:,:,0]=0;power=cp.mean(abs(F)**2,axis=1)
        w=cp.full(N//2+1,2.);w[[0,-1]]=1
        error=float(abs(power@w/N**2-x.var(2).mean(1)).max());parseval=max(parseval,error);assert error<1e-12
        destinations['residual_PSD'][lo:hi]=power.get()
        del x,F,power,pp;cp.get_default_memory_pool().free_all_blocks()
    for a in destinations.values():a.flush()
    if carry:nextstate.flush();nextphase.flush()
    np.savez_compressed(folder/f'part{part}_statistics.npz',cells=np.arange(lo0,hi0),replica_statistics=np.concatenate(summaries))
    write(folder/f'part{part}_complete.json',dict(status='COMPLETE',elapsed_s=time.time()-started,
        phase_count_conservation=True,residual_Parseval_error=parseval,uniform_phase_vs_time_mean_max_Hz=mean_difference,
        original_physics_unchanged=True,replicas=R))
    write(folder/f'progress_part{part}.json',dict(status='COMPLETE',updated_epoch=time.time()))


def collect(generation):
    folder=OUT/f'generation_{generation}';previous=OUT/f'generation_{generation-1}'
    for part in range(2):assert read(folder/f'part{part}_complete.json')['status']=='COMPLETE'
    rate=np.load(folder/'source_rate_Hz.npy');before=np.load(previous/'source_rate_Hz.npy')
    phase=np.load(folder/'phase_spike_probability.npy');oldphase=np.load(previous/'phase_spike_probability.npy')
    nativephase=np.load(OUT/'generation_0/phase_spike_probability.npy')
    native_rate=np.load(OUT/'generation_0/source_rate_Hz.npy')
    st=np.concatenate([np.load(folder/f'part{i}_statistics.npz')['replica_statistics'] for i in range(2)])
    raw=dict(np.load(OUT/'parameters.npz'));region=raw['region'];E=np.arange(40000)<32000;rows=[]
    for label,mask in [('allE',E),('coreA',E&(region==0)),('coreB',E&(region==1)),('surroundE',E&(region==2)),('I',~E)]:
        rows.append(dict(region=label,output_mean_rate_Hz=float(rate[mask].mean()),
            mean_rate_change_Hz=float((rate-before)[mask].mean()),rate_residual_RMS_Hz=float(np.sqrt(np.mean((rate-before)[mask]**2))),
            native_development_rate_RMS_Hz=float(np.sqrt(np.mean((rate-native_rate)[mask]**2))),
            mean_phase_variance=float(phase[mask].var(1).mean()),native_mean_phase_variance=float(nativephase[mask].var(1).mean()),
            phase_probability_RMS_change=float(np.sqrt(np.mean((phase-oldphase)[mask]**2))),
            mean_first_second_1s_rate_Hz=st[mask,:,:2].mean((0,1)).tolist(),
            counterfactual_Zdot_per_s=float(np.mean((st[mask,:,10].mean(1)-raw['Z'][mask])/5)) if label!='I' else None))
    display=raw['display'][:32000];counts=np.bincount(display,minlength=400)
    field=np.bincount(display,weights=rate[:32000],minlength=400)/np.maximum(counts,1)
    matched=np.load(ROOT/'native_K9p35_constant_background_pair_v2/high_history_constant_background_tail_field_Hz.npy')
    causal=causal_phase(phase)
    result=dict(status='COMPLETE_BOUNDED_PHASE_MAP_UPDATE',generation=generation,rows=rows,
        phase_specific_causal_R_range_Hz=[float(causal.min()),float(causal.max())],
        G0_segment_consistent=bool(causal.max()<200),
        matched_native_field_RMS_Hz=float(np.sqrt(np.average((field-matched)**2,weights=counts))),
        physical_stability_established=False,formal_bifurcation_allowed=False,
        scope='Numericalstatisticalmap at fixed22stepphase; iterations are not physicaltime. Native sources only initialize generation0; subsequent phase probabilities and residualspectra are generated by modelneurons.')
    write(folder/'complete.json',result)
    mean,wave=project_phase(OUT/'operators',phase,read(OPS/'prepared.json')['params'])
    np.save(folder/'current_mean.npy',mean);np.save(folder/'current_phase_wave.npy',wave)
    print(dict(generation=generation,field_RMS_Hz=result['matched_native_field_RMS_Hz'],
        causal_range=result['phase_specific_causal_R_range_Hz'],rates=[x['output_mean_rate_Hz'] for x in rows]),flush=True)
    return result


def supervise():
    assert read(ROOT/'high_history_local_source_phase_projection/result.json')['permits_bounded_allcell_phase_test']
    assert not (OUT/'pilot_contract.json').exists()
    write(OUT/'pilot_contract.json',dict(status='REGISTERED_BEFORE_ALLCELL_PHASE_UPDATES',created_epoch=time.time(),
        question='Does the coherent mean survive self-generated source feedback at the actualK9.35 highhistoryfield, without continually supplyingnativewaveforms?',
        design='Exactly3 numericalmapupdates of40000targets x64 replicas. Generation0 native sourcephaseprobabilities and residualspectra initialize only. Originalweights/delayresidues/filter/threshold/ref/M/ZK/externalPoisson retained. Subsequent updates use only their own outputphaseprobabilities and residualsourcePSDs; no damping, frequencysearch, fittedgain or nativeforcing.',
        period='22 original0.1ms steps is a candidate; updateindex is notphysicaltime. Phaseconditionalresiduals are approximated as independent stationaryGaussian source processes.',
        stop='Stop early if phase-specific causalR crosses200 invalidatingG0, fieldRMS frommatchednative exceeds10Hz, or eithercore mean drifts morethan10Hz fromgeneration0. Retention permits furthercorrespondenceanalysis only, notroot/stability/Floquet/bifurcationacceptance.',
        validation='Local59targetsourceprojectioncheck complete; phasekernel zero-wavebitwiseQA; count/exposureconservation and residualParseval eachbatch. Originalgraph identity and delayresidueweights exact. New Fourier/Poisson/burn/phase streams eachgeneration.',
        replicas=R,generations=GENERATIONS,producer_sha256=sha(__file__),formal_bifurcation_allowed=False))
    shutil.copy2(__file__,OUT/'pilot_producer.py');shutil.copy2(sampler.__file__,OUT/'phase_sampler_producer.py')
    write(OUT/'supervisor.json',dict(status='RUNNING',pid=os.getpid(),completed_updates=0,updated_epoch=time.time()))
    results=[];stop=None;raw=dict(np.load(OUT/'parameters.npz'));initial=np.load(OUT/'generation_0/source_rate_Hz.npy')
    for generation in range(1,GENERATIONS+1):
        folder=OUT/f'generation_{generation}';folder.mkdir()
        for name,shape in [('residual_PSD',(40000,N//2+1)),('source_rate_Hz',(40000,)),('phase_spike_probability',(40000,L))]:
            a=np.lib.format.open_memmap(folder/(name+'.npy'),mode='w+',dtype='f8',shape=shape);a[:]=np.nan;a.flush();del a
        jobs=[]
        for part in range(2):
            log=(folder/f'worker_part{part}.log').open('w')
            process=subprocess.Popen([PYTHON,__file__,'worker','--generation',str(generation),'--part',str(part)],stdout=log,stderr=subprocess.STDOUT)
            jobs.append((process,log))
        codes=[]
        for process,log in jobs:codes.append(process.wait());log.close()
        if any(codes):
            write(OUT/'supervisor.json',dict(status='FAILED_WORKER',generation=generation,codes=codes));raise RuntimeError(codes)
        result=collect(generation);results.append(result);rate=np.load(folder/'source_rate_Hz.npy')
        if not result['G0_segment_consistent']:stop='PHASE_SPECIFIC_G_SEGMENT_CHANGED'
        if result['matched_native_field_RMS_Hz']>10:stop='NATIVE_SPATIAL_RELEVANCE_LOST'
        for q in [0,1]:
            mask=(np.arange(40000)<32000)&(raw['region']==q)
            if abs((rate-initial)[mask].mean())>10:stop='NATIVE_CORE_RELEVANCE_LOST'
        write(OUT/'supervisor.json',dict(status='RUNNING' if stop is None else 'STOPPED_AT_RELEVANCE_GUARD',
            pid=os.getpid(),completed_updates=generation,stop_reason=stop,updated_epoch=time.time()))
        if stop:break
    status='COMPLETE_THREE_BOUNDED_PHASE_MAP_UPDATES' if stop is None else 'COMPLETE_EARLY_STOP_AT_RELEVANCE_GUARD'
    write(OUT/'pilot_result.json',dict(status=status,updates=results,stop_reason=stop,
        root_certified=False,physical_stability_established=False,formal_bifurcation_allowed=False))
    write(OUT/'supervisor.json',dict(status=status,completed_updates=len(results),stop_reason=stop,updated_epoch=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['supervise','worker']);p.add_argument('--generation',type=int);p.add_argument('--part',type=int)
    a=p.parse_args()
    if a.command=='supervise':supervise()
    else:worker(a.generation,a.part)
