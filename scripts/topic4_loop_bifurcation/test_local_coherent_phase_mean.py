#!/usr/bin/env python3
"""One local phase-mean plus residual-noise bridge, not a new network model."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import shutil,time
from pathlib import Path
import numpy as np
from campaign import ROOT,read,write,sha
from compare_local_correlated_inputs import OUT as PRIOR,SOURCE,N,R,BATCH,SEED,OPS
from measure_coherent_phase_component import OUT as INPUT,L
import coherent_phase_sampler as sampler
import individual_spectral_sampler as original

OUT=ROOT/'high_history_local_coherent_phase_mean'


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    assert read(INPUT/'result.json')['status']=='COMPLETE_22_STEP_PHASE_COMPONENT_DESCRIPTION'
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_LOCAL_PHASE_MEAN_BRIDGE',created_epoch=time.time(),
        question='Can an explicit2.2ms coherent current mean plus Gaussian residual represent the local response without supplying the entire native waveform?',
        motivation='Exactphase decomposition explains235.64 of239.61mV2 recurrentIEvariance on20diagnosticcells. This component should be represented as a time-dependent mean, not Gaussianized as stationarynoise. It explains only36.65percentofallEsourcebinaryvariance and5.77percentincoreB, so onephasecomponent is not assumed a complete network description.',
        design='Exactly59targets x512replicas, originalmembrane/ref/M/ZK/Poisson kernel unchanged. Add a22step periodicinputmean, zero-centered overuniformphase, with jointlyGaussian E/I residual from the originalrecord after subtracting itsphaseaverage. OriginalsuppliedDC held fixed.3-4sburn,2srecord. Periodicmean advancescontinuously without resetting every2s; residual retains the declared2speriodic spectral convention.',
        centering='Finite nativewindow hasunequalphasecounts. Uniformphasecentering changes no specifiedDC; offset from the originalempiricalcyclemean is recorded, max0.002212mV. No meanfit or gain multiplier.',
        QA='The modifiedkernel changes only inputgeneration. Zero phasewave must reproduce originalkernel spikes and allstatistics bitwise at the same residualcurrents/Poisson streams. Save outputspike probability conditionalonknowninputphase for future model design, not as a certifiedautonomous orbit.',
        reference=read(PRIOR/'contract.json')['reference'],period_steps=L,replicas=R,
        stop='One localresponse only. No free-coupledmodel, spontaneousoscillation, root, Floquet or bifurcationclaim.',
        original_sampler_sha256=sha(original.__file__),phase_sampler_sha256=sha(sampler.__file__),producer_sha256=sha(__file__),
        formal_bifurcation_allowed=False,autonomous_closure=False))
    import cupy as cp
    cp.cuda.Device(0).use();cp.get_default_memory_pool().set_limit(size=3*2**30)
    fn=sampler.kernel(cp);basefn=original.kernel(cp)
    d=dict(np.load(INPUT/'target_phase_inputs.npz'));prior=dict(np.load(PRIOR/'inputs.npz'))
    raw=dict(np.load(SOURCE/'parameters.npz'));params=read(OPS/'prepared.json')['params'];cells=d['cells']
    out=np.empty((len(cells),R,16));phase_count=np.zeros((len(cells),L));phase_exposure=phase_count.copy()
    offsets=np.random.default_rng(SEED+300000).integers(0,L,size=R,dtype='i4');started=time.time()
    for lo in range(0,len(cells),BATCH):
        while cp.cuda.runtime.memGetInfo()[0]<4*2**30:time.sleep(2)
        hi=min(lo+BATCH,len(cells));B=hi-lo
        write(OUT/'progress.json',dict(status='RUNNING',pid=os.getpid(),completed_targets=lo,total_targets=len(cells),updated_epoch=time.time()))
        rng=cp.random.RandomState(SEED+lo)
        z=(rng.standard_normal((B,R,N//2+1))+1j*rng.standard_normal((B,R,N//2+1)))/np.sqrt(2)
        z[:,:,0]=0;z[:,:,-1]=rng.standard_normal((B,R))
        currents=[]
        for q in range(2):
            x=cp.fft.irfft(cp.asarray(d['residual_complex'][q,lo:hi])[:,None,:]*z,n=N,axis=2)
            x+=cp.asarray(d['mean_recurrent'][q,lo:hi])[:,None,None]
            currents.append(cp.ascontiguousarray(x.transpose(2,0,1).reshape(N,B*R)));del x
        wave=np.ascontiguousarray(d['phase_wave'][:,lo:hi].transpose(0,2,1))
        phase=np.broadcast_to(offsets,(B,R)).copy().ravel()
        extra=np.random.default_rng(SEED+100000+lo).integers(0,10001,size=B*R,dtype='i4')
        p,cfg=original.make_parameters(raw,cells[lo:hi],d['native_rate_Hz'][lo:hi],0.,params)
        if lo==0:
            a,b=sampler.run(cp,fn,*currents,p,cfg,R,30000,extra,SEED,lo,np.zeros_like(wave),phase)
            aa,bb=original.run(cp,basefn,*currents,p,cfg,R,30000,extra,SEED,lo)
            assert bool(cp.array_equal(a,aa)) and bool(cp.array_equal(b,bb))
            write(OUT/'implementation_qa.json',dict(status='PASS',zero_phasewave_every_spike_and_all_statistics_bitwise=True,
                targets=B,replicas=R,unchanged_neuron_equations=True));del a,b,aa,bb
        flags,st=sampler.run(cp,fn,*currents,p,cfg,R,30000,extra,SEED,lo,wave,phase)
        out[lo:hi]=st.get()
        base=(30000+extra.reshape(B,R)+phase.reshape(B,R))%L
        for psi in range(L):
            value=flags[psi::L].sum(0).reshape(B,R).get();idx=(base+psi)%L
            target=np.broadcast_to(np.arange(B)[:,None],idx.shape)
            np.add.at(phase_count[lo:hi],(target,idx),value)
            np.add.at(phase_exposure[lo:hi],(target,idx),len(range(psi,N,L)))
        del z,currents,flags,st
        cp.get_default_memory_pool().free_all_blocks()
    assert np.array_equal(phase_count.sum(1),out[:,:,:2].sum((1,2)))
    assert np.all(phase_exposure.sum(1)==N*R)
    rates=out[:,:,:2].sum(2)/2;mean=rates.mean(1);sem=rates.std(1,ddof=1)/np.sqrt(R)
    np.savez_compressed(OUT/'samples.npz',cells=cells,replica_statistics=out,rate_mean_Hz=mean,rate_SEM_Hz=sem,
        phase_spike_probability=phase_count/phase_exposure,phase_counts=phase_count,phase_exposure=phase_exposure)
    region=raw['region'][cells];rows=[]
    for label,mask in [('selected_all',np.ones(len(cells),bool)),('selected_coreA',(cells<32000)&(region==0)),
        ('selected_coreB',(cells<32000)&(region==1)),('largest20_errors',np.isin(cells,prior['largest_discrepancy_cells']))]:
        if mask.any():rows.append(dict(region=label,targets=int(mask.sum()),
            native_development_mean_Hz=float(d['native_rate_Hz'][mask].mean()),response_rate_mean_Hz=float(mean[mask].mean()),
            native_development_rate_RMS_error_Hz=float(np.sqrt(np.mean((mean[mask]-d['native_rate_Hz'][mask])**2))),
            output_rate_MCSEM_RMS_Hz=float(np.sqrt(np.mean(sem[mask]**2)))))
    result=dict(status='COMPLETE_LOCAL_COHERENT_MEAN_GAUSSIAN_RESIDUAL_RESPONSE',rows=rows,
        phase_output_count_and_exposure_conservation=True,elapsed_s=time.time()-started,
        scope='Data-conditionedlocalbridge with explicitphase-dependentmean. Neither free network dynamics nor spontaneousphase/coherence established. No formalbifurcation.',producer_sha256=sha(__file__))
    write(OUT/'result.json',result);shutil.copy2(__file__,OUT/'producer.py');shutil.copy2(sampler.__file__,OUT/'phase_sampler_producer.py')
    write(OUT/'progress.json',dict(status='COMPLETE',updated_epoch=time.time()));print(result,flush=True)


if __name__=='__main__':main()
