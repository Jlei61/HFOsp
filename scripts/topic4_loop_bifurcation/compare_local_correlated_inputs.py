#!/usr/bin/env python3
"""Four paired local responses separating source and E/I temporal structure."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,shutil,time
from pathlib import Path
import numpy as np
from scipy import fft,sparse
from campaign import ROOT,read,write,sha
from conditional_density_inputs import OPS
from high_history_spectral_value import OUT as SOURCE,N
from individual_spectral_sampler import kernel,make_parameters,run

OUT=ROOT/'high_history_local_correlated_inputs'
CROSS=SOURCE/'cross_spectral_diagnostic_v2'
CONDITIONS=['diagonal_independent_Gaussian','full_marginal_independent_Gaussian',
            'full_joint_Gaussian','full_source_waveform']
R=512
BATCH=4
SEED=941071


def prepare():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    assert read(CROSS/'result.json')['status']=='COMPLETE_MEASURED_SOURCE_CROSS_SPECTRAL_DIAGNOSIS'
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_FOUR_LOCAL_INPUT_RESPONSES',created_epoch=time.time(),
        question='At the same supplied source means and native local equations, does retaining measured source correlation repair high-history local response errors, and is the full second-order Gaussian description sufficient?',
        design='Exactly59existingtargets x512numericalreplicas x4conditions. OriginallocalLIF/refractory/individualtheta/dynamicM/heldspatialZK, G0, samefixedpercell externalPoisson.3s+uniform0-1s burn then2srecord. No recurrent network simulation or newparameterpoint.',
        conditions={
            CONDITIONS[0]:'Original squaredweight source-diagonal current PSDs, independent E/I Gaussian Fourier coefficients.',
            CONDITIONS[1]:'Measured full within-stream current PSDs, including sourcecross terms and originaldelays; still independent E/I Gaussian coefficients.',
            CONDITIONS[2]:'Same measuredfullcurrent PSDs plus measured E/I crossperiodogram; common Gaussian Fourier coefficient perfrequency/replica for both streams.',
            CONDITIONS[3]:'Original reconstructed recurrent E/I waveform with one shared random circular timeshift perreplica; preserves joint waveform and crossfrequency phase, plus samefresh externalPoisson.'},
        pairing='Same perreplica externalPoisson seed and extra burn acrossconditions. Gaussian diagonal/fullmarginal share normals; fulljoint shares E withfullmarginal and swaps I to the same E coefficient. No claim that these numericalPoisson streams reproduce native draws.',
        initialization='All conditions use identical supplied recurrentmeans from the existing native-source projection and identical actualtargetparameters; native source data condition this diagnosis. No source statistics are promoted to an autonomous closed model.',
        reference='Native72-74s pertarget rates are a development comparison under time-varying expected background. Fourlocalconditions alluse the samefixedmean. Changes betweenlocalconditions isolate represented inputstructure; closeness to that one nativewindow is not an independent network validation.',
        cross_spectrum_limit='One2s complex source record gives a rank-one jointperiodogram at eachfrequency, not an accurately estimated stationary crossspectral matrix. This is a finite-record sufficiency diagnosis. Numericreplicas are not biological/native samples.',
        readout='Percondition cellrate/MCSEM, pairedcondition ratechange/SEM, first-second1s, Mstart/end, inputmoments and resourceeligibility. Retain alltargets plus prespecified selectedcore masks and postselected20largesterror mask.',
        resources='OneGPU, batch4, poollimit4GiB, start eachbatch only with at least5GiB free. Does not edit or restart the bounded highhistory map currently running.',
        stop='Fourconditions only. No automatic parameterfit, networkupdate, root, gain/frequency scan or formal bifurcation claim.',
        replicas=R,conditions_order=CONDITIONS,unchanged_sampler_sha256=sha(Path(__file__).with_name('individual_spectral_sampler.py')),
        producer_sha256=sha(__file__),formal_bifurcation_allowed=False,counts_as_autonomous_loop=False))
    d=dict(np.load(CROSS/'projection.npz'));cells=d['cells'];signal=d['reconstructed_recurrent_currents']
    C=fft.rfft(signal,axis=2,workers=2);C[:,:,0]=0
    means=signal.mean(2);weights=np.full(N//2+1,2.);weights[[0,-1]]=1
    assert np.allclose(abs(C)**2@weights/N**2,d['full_recurrent_variance'],rtol=1e-11,atol=1e-10)
    assert np.max(abs(fft.irfft(C,n=N,axis=2,workers=2)+means[:,:,None]-signal))<1e-8
    power=np.load(SOURCE/'generation_0/source_PSD.npy',mmap_mode='r');H=np.load(SOURCE/'filter_power.npy')
    diagonal=[]
    for q,(kind,ss) in enumerate([('ampa',slice(0,32000)),('gaba',slice(32000,40000))]):
        W=sparse.load_npz(SOURCE/f'original_{kind}_jump.npz').tocsr()[cells]
        diagonal.append((W.multiply(W)@power[ss])*H[q])
    diagonal=np.array(diagonal)
    assert np.allclose(diagonal@weights/N**2,d['diagonal_recurrent_variance'],rtol=1e-11,atol=1e-10)
    np.savez_compressed(OUT/'inputs.npz',cells=cells,full_complex=C,mean_recurrent=means,
        diagonal_PSD=diagonal,native_rate_Hz=np.load(SOURCE/'generation_0/source_rate_Hz.npy')[cells],
        largest_discrepancy_cells=np.load(SOURCE/'input_moment_review/moments.npz')['largest_discrepancy_cells'])
    write(OUT/'preparation_qa.json',dict(status='PASS',native_graph_filter_projection_reused=True,
        source_waveform_Fourier_roundtrip=True,diagonal_variance_matches_prior=True,full_variance_matches_prior=True,targets=len(cells)))
    shutil.copy2(__file__,OUT/'producer.py')


def worker(device):
    import cupy as cp
    assert read(OUT/'contract.json')['producer_sha256']==sha(__file__)
    assert read(OUT/'contract.json')['unchanged_sampler_sha256']==sha(Path(__file__).with_name('individual_spectral_sampler.py'))
    cp.cuda.Device(device).use();cp.get_default_memory_pool().set_limit(size=4*2**30);fn=kernel(cp)
    d=dict(np.load(OUT/'inputs.npz'));raw=dict(np.load(SOURCE/'parameters.npz'));params=read(OPS/'prepared.json')['params']
    cells=d['cells'];out=np.full((4,len(cells),R,16),np.nan)
    recurrent_moments=np.empty((4,2,len(cells),R,2));paired=[];started=time.time()
    for lo in range(0,len(cells),BATCH):
        while cp.cuda.runtime.memGetInfo()[0]<5*2**30:time.sleep(2)
        hi=min(lo+BATCH,len(cells));B=hi-lo
        write(OUT/'progress.json',dict(status='RUNNING',pid=os.getpid(),device=device,
            completed_targets=lo,total_targets=len(cells),updated_epoch=time.time()))
        rng=cp.random.RandomState(SEED+lo);normals=[]
        for q in range(2):
            z=(rng.standard_normal((B,R,N//2+1))+1j*rng.standard_normal((B,R,N//2+1)))/np.sqrt(2)
            z[:,:,0]=0;z[:,:,-1]=rng.standard_normal((B,R));normals.append(z)
        C=cp.asarray(d['full_complex'][:,lo:hi]);P=cp.asarray(d['diagonal_PSD'][:,lo:hi])
        mean=cp.asarray(d['mean_recurrent'][:,lo:hi]);phase=cp.exp(1j*cp.angle(C))
        shifted=np.random.default_rng(SEED+lo).integers(0,N,size=(B,R))
        shifts=cp.exp(-2j*np.pi*cp.asarray(shifted)[:,:,None]*cp.arange(N//2+1)[None,None,:]/N)
        extra=np.random.default_rng(SEED+100000+lo).integers(0,10001,size=B*R,dtype='i4')
        p,cfg=make_parameters(raw,cells[lo:hi],d['native_rate_Hz'][lo:hi],0.,params)
        for condition in range(4):
            currents=[]
            for q in range(2):
                if condition==0:X=cp.sqrt(P[q])[:,None,:]*phase[q,:,None,:]*normals[q]
                elif condition==1:X=C[q,:,None,:]*normals[q]
                elif condition==2:X=C[q,:,None,:]*normals[0]
                else:X=C[q,:,None,:]*shifts
                x=cp.fft.irfft(X,n=N,axis=2)
                variance=x.var(2).get();x+=mean[q,:,None,None]
                recurrent_moments[condition,q,lo:hi,:,0]=x.mean(2).get()
                recurrent_moments[condition,q,lo:hi,:,1]=variance
                if condition==3:
                    want=np.sum(abs(d['full_complex'][q,lo:hi])**2*np.r_[1,np.full(N//2-1,2),1],axis=1)/N**2
                    assert np.allclose(variance,want[:,None],rtol=1e-10,atol=1e-9)
                currents.append(cp.ascontiguousarray(x.transpose(2,0,1).reshape(N,B*R)))
                del X,x
            flags,st=run(cp,fn,currents[0],currents[1],p,cfg,R,30000,extra,SEED,lo)
            out[condition,lo:hi]=st.get()
            del flags,st,currents
            cp.get_default_memory_pool().free_all_blocks()
        del normals,C,P,mean,phase,shifts
        cp.get_default_memory_pool().free_all_blocks()
    assert np.isfinite(out).all()
    np.savez_compressed(OUT/'response_samples.npz',cells=cells,replica_statistics=out,
        recurrent_input_mean_and_variance=recurrent_moments)
    rate=out[:,:,:,:2].sum(3)/2;means=rate.mean(2);sem=rate.std(2,ddof=1)/np.sqrt(R)
    region=raw['region'][cells];masks=[('selected_all',np.ones(len(cells),bool)),
        ('selected_coreA',(cells<32000)&(region==0)),('selected_coreB',(cells<32000)&(region==1)),
        ('largest20_errors',np.isin(cells,d['largest_discrepancy_cells']))]
    rows=[]
    for label,mask in masks:
        if not mask.any():continue
        rows.append(dict(region=label,targets=int(mask.sum()),
            native_rate_mean_Hz=float(d['native_rate_Hz'][mask].mean()),
            response_rate_mean_Hz=means[:,mask].mean(1).tolist(),
            native_development_rate_RMS_error_Hz=np.sqrt(np.mean((means[:,mask]-d['native_rate_Hz'][mask])**2,axis=1)).tolist(),
            output_rate_MCSEM_RMS_Hz=np.sqrt(np.mean(sem[:,mask]**2,axis=1)).tolist(),
            mean_first_second_1s_Hz=out[:,mask,:,:2].mean((1,2)).tolist(),
            mean_M_start_end=out[:,mask,:,2:4].mean((1,2)).tolist()))
    for before,after in [(0,1),(1,2),(2,3)]:
        delta=rate[after]-rate[before]
        paired.append(dict(before=CONDITIONS[before],after=CONDITIONS[after],
            cell_mean_rate_change_Hz=delta.mean(1).tolist(),cell_paired_SEM_Hz=(delta.std(1,ddof=1)/np.sqrt(R)).tolist()))
    np.savez_compressed(OUT/'rate_comparison.npz',cells=cells,rate_mean_Hz=means,rate_SEM_Hz=sem,native_rate_Hz=d['native_rate_Hz'])
    result=dict(status='COMPLETE_FOUR_PAIRED_LOCAL_INPUT_RESPONSES',conditions=CONDITIONS,rows=rows,
        paired_comparisons=paired,replicas=R,elapsed_s=time.time()-started,
        limitations=read(OUT/'contract.json')['reference'],cross_spectrum_limit=read(OUT/'contract.json')['cross_spectrum_limit'],
        autonomous_closure=False,formal_bifurcation_allowed=False,producer_sha256=sha(__file__))
    write(OUT/'result.json',result);write(OUT/'progress.json',dict(status='COMPLETE',updated_epoch=time.time()))
    print(dict(status=result['status'],rows=rows,elapsed_s=result['elapsed_s']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','worker']);p.add_argument('--device',type=int,default=0);a=p.parse_args()
    prepare() if a.command=='prepare' else worker(a.device)
