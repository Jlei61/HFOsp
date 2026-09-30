#!/usr/bin/env python3
"""One finite-spectrum phase-only control following the four local assays."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import shutil,time
from pathlib import Path
import numpy as np
from campaign import read,write,sha
from compare_local_correlated_inputs import OUT as PARENT,SOURCE,N,R,BATCH,SEED,OPS
from individual_spectral_sampler import kernel,make_parameters,run

OUT=PARENT/'phase_only_followup'


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    assert read(PARENT/'result.json')['status']=='COMPLETE_FOUR_PAIRED_LOCAL_INPUT_RESPONSES'
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_PHASE_ONLY_RESPONSE',created_epoch=time.time(),
        question='Does the remaining advantage of the native recurrent waveform depend on relationships between Fourier phases across frequencies, beyond random Gaussian amplitude variability?',
        motivation='On20diagnosticerrorcells the completed fourresponse RMSerrors are21.12,6.72,6.71,0.517Hz. Restoring withinstreamcross terms helps; restoring E/I crossperiodogram has little added effect; fullwaveform remains better.',
        design='Oneextra response on the SAME59targets and512replicas, unchangednativekernel, fixednu, means,ZK,G0,M andburn. Use unitmodulus randomphase separately at eachpositivefrequency, common between recurrentE/I. Thus eachreplica has EXACT original fullE/I power AND crossperiodogram; only crossfrequency phase relationships are scrambled relative to the shiftednativewaveform.',
        pairing='Obtain phases by normalizing the exactGaussianFourier coefficient used by priorfulljointcondition (same randomdraworder), reuse sameexternalPoisson andburn. Compare with already completedfulljointGaussian andwaveform responses. No newnative source, parameterpoint or physicalmodel.',
        controls='DCzero, Nyquistreal sign, generatedinput perreplica Parseval equal nativefullpowers. This distinguishes coefficientamplitude variability from phaseorganization at fixedsecondorder finite-record structure. Nonlinear dependence onhigherorder temporal statistics is not an autonomous closure.',
        reference=read(PARENT/'contract.json')['reference'],replicas=R,
        producer_sha256=sha(__file__),sampler_sha256=sha(Path(__file__).with_name('individual_spectral_sampler.py')),
        formal_bifurcation_allowed=False,autonomous_closure=False))
    import cupy as cp
    cp.cuda.Device(0).use();cp.get_default_memory_pool().set_limit(size=3*2**30);fn=kernel(cp)
    d=dict(np.load(PARENT/'inputs.npz'));raw=dict(np.load(SOURCE/'parameters.npz'));params=read(OPS/'prepared.json')['params']
    cells=d['cells'];out=np.empty((len(cells),R,16));qa=0.;started=time.time()
    weights=np.r_[1,np.full(N//2-1,2),1]
    for lo in range(0,len(cells),BATCH):
        while cp.cuda.runtime.memGetInfo()[0]<4*2**30:time.sleep(2)
        hi=min(lo+BATCH,len(cells));B=hi-lo
        write(OUT/'progress.json',dict(status='RUNNING',pid=os.getpid(),completed_targets=lo,total_targets=len(cells),updated_epoch=time.time()))
        rng=cp.random.RandomState(SEED+lo)
        z=(rng.standard_normal((B,R,N//2+1))+1j*rng.standard_normal((B,R,N//2+1)))/np.sqrt(2)
        z[:,:,0]=0;z[:,:,-1]=rng.standard_normal((B,R))
        amplitude=cp.abs(z);unit=cp.zeros_like(z);cp.divide(z,amplitude,out=unit,where=amplitude>0)
        del z,amplitude
        currents=[];C=cp.asarray(d['full_complex'][:,lo:hi]);mean=cp.asarray(d['mean_recurrent'][:,lo:hi])
        for q in range(2):
            x=cp.fft.irfft(C[q,:,None,:]*unit,n=N,axis=2)
            expected=np.sum(abs(d['full_complex'][q,lo:hi])**2*weights,axis=1)/N**2
            err=float(np.max(abs(x.var(2).get()-expected[:,None])));qa=max(qa,err)
            assert err<1e-8,err
            x+=mean[q,:,None,None]
            currents.append(cp.ascontiguousarray(x.transpose(2,0,1).reshape(N,B*R)));del x
        p,cfg=make_parameters(raw,cells[lo:hi],d['native_rate_Hz'][lo:hi],0.,params)
        extra=np.random.default_rng(SEED+100000+lo).integers(0,10001,size=B*R,dtype='i4')
        flags,st=run(cp,fn,currents[0],currents[1],p,cfg,R,30000,extra,SEED,lo)
        out[lo:hi]=st.get();del currents,C,mean,unit,flags,st
        cp.get_default_memory_pool().free_all_blocks()
    rate=out[:,:,:2].sum(2)/2;meanrate=rate.mean(1);sem=rate.std(1,ddof=1)/np.sqrt(R)
    np.savez_compressed(OUT/'samples.npz',cells=cells,replica_statistics=out,rate_mean_Hz=meanrate,rate_SEM_Hz=sem)
    old=np.load(PARENT/'response_samples.npz')['replica_statistics'];oldrate=old[:,:,:,:2].sum(3)/2
    region=raw['region'][cells];rows=[]
    for label,mask in [('selected_all',np.ones(len(cells),bool)),('selected_coreA',(cells<32000)&(region==0)),
        ('selected_coreB',(cells<32000)&(region==1)),('largest20_errors',np.isin(cells,d['largest_discrepancy_cells']))]:
        if mask.any():rows.append(dict(region=label,targets=int(mask.sum()),
            native_rate_mean_Hz=float(d['native_rate_Hz'][mask].mean()),response_rate_mean_Hz=float(meanrate[mask].mean()),
            native_development_rate_RMS_error_Hz=float(np.sqrt(np.mean((meanrate[mask]-d['native_rate_Hz'][mask])**2))),
            output_rate_MCSEM_RMS_Hz=float(np.sqrt(np.mean(sem[mask]**2))),
            paired_RMS_difference_from_jointGaussian_Hz=float(np.sqrt(np.mean((meanrate[mask]-oldrate[2,mask].mean(1))**2))),
            paired_RMS_difference_from_waveform_Hz=float(np.sqrt(np.mean((meanrate[mask]-oldrate[3,mask].mean(1))**2)))))
    result=dict(status='COMPLETE_FIXED_POWER_RANDOM_PHASE_RESPONSE',rows=rows,
        QA=dict(each_replica_input_power_Parseval_max_error=qa,same_marginal_and_EI_crossperiodogram_exact_by_construction=True),
        scope='Only crossfrequency phase relationships differ from circularlyshifted waveform at the level of finite-record source statistics; perfrequency secondorder powers/crosspowers and inputmeans fixed. Local data-conditioned diagnosis, not an autonomous network or formal bifurcation.',
        elapsed_s=time.time()-started,producer_sha256=sha(__file__))
    write(OUT/'result.json',result);shutil.copy2(__file__,OUT/'producer.py')
    write(OUT/'progress.json',dict(status='COMPLETE',updated_epoch=time.time()));print(result,flush=True)


if __name__=='__main__':main()
