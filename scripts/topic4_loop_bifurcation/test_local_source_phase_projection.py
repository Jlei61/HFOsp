#!/usr/bin/env python3
"""Test the graph-projected phase mean and independent source residuals locally."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import shutil,time
import numpy as np
from scipy import sparse
from campaign import ROOT,read,write,sha
from prepare_phase_source_closure import OUT as SOURCE,L,N,OPS
from compare_local_correlated_inputs import OUT as PRIOR,R,BATCH,SEED
import coherent_phase_sampler as sampler
from individual_spectral_sampler import make_parameters

OUT=ROOT/'high_history_local_source_phase_projection'


def main():
    assert read(SOURCE/'result.json')['status']=='COMPLETE_SOURCE_PHASE_OPERATOR_PREPARATION'
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_LOCAL_SOURCE_OPERATOR_RESPONSE',created_epoch=time.time(),
        question='Does the source-defined22step mean through originalweights/delays plus independent source residual spectra retain the successful local response?',
        design='Same59developmenttargets and512numericalreplicas. Replace measured targetphasewave and measured jointresidual by originalgraph-projected sourcephasewave and squared-weight independent source residuals. Original local neuron/M/ref/ZK/G0/externalPoisson unchanged; no fittedmeans, gains, frequencies.',
        gate='Source phase-specific causalR must staybelow200 forG0. This is a relevance check: if either selectedcore RMSerror exceeds2Hz or selected20error exceeds2Hz, do not launch the allcell self-generated phase map. Passing permits only a bounded endogenous test, not root or bifurcationacceptance.',
        unit='59selectedcells from one conditional native developmentrecord; numericalreplicas are not independent biological/native seeds.',
        producer_sha256=sha(__file__),formal_bifurcation_allowed=False,autonomous_closure=False))
    import cupy as cp
    cp.cuda.Device(0).use();cp.get_default_memory_pool().set_limit(size=4*2**30);fn=sampler.kernel(cp)
    d=dict(np.load(PRIOR/'inputs.npz'));cells=d['cells'];raw=dict(np.load(SOURCE/'parameters.npz'))
    p=read(OPS/'prepared.json')['params'];folder=SOURCE/'generation_0'
    prob=np.load(folder/'phase_spike_probability.npy');rate=np.load(folder/'source_rate_Hz.npy')
    z=np.exp(-2j*np.pi*np.arange(L//2+1)/L);a=np.exp(-.1/15)
    causal=np.fft.irfft(np.fft.rfft(prob[:32000].mean(0))/(.015*(1-a*z)),n=L)
    assert causal.max()<200,causal
    S=np.load(folder/'residual_PSD.npy',mmap_mode='r');H=np.load(SOURCE/'filter_power.npy')
    powers=[];weights=np.full(N//2+1,2.);weights[[0,-1]]=1
    for q,(kind,ss) in enumerate([('ampa',slice(0,32000)),('gaba',slice(32000,40000))]):
        W=sparse.load_npz(SOURCE/f'original_{kind}_jump.npz').tocsr()[cells]
        powers.append((W.multiply(W)@S[ss])*H[q])
    powers=np.array(powers);assert np.isfinite(powers).all() and powers.min()>=0
    mean=np.load(folder/'current_mean.npy')[:,cells];wave=np.load(folder/'current_phase_wave.npy')[:,cells]
    offsets=np.random.default_rng(SEED+300000).integers(0,L,size=R,dtype='i4')
    out=np.empty((len(cells),R,16));started=time.time()
    for lo in range(0,len(cells),BATCH):
        hi=min(lo+BATCH,len(cells));B=hi-lo;rng=cp.random.RandomState(SEED+lo);currents=[]
        for q in range(2):
            P=cp.asarray(powers[q,lo:hi]);real=rng.standard_normal((B,R,N//2+1));imag=rng.standard_normal(real.shape)
            f=(real+1j*imag)*cp.sqrt(P[:,None,:]/2);f[:,:,0]=0;f[:,:,-1]=real[:,:,-1]*cp.sqrt(P[:,-1,None])
            x=cp.fft.irfft(f,n=N,axis=2)+cp.asarray(mean[q,lo:hi])[:,None,None]
            currents.append(cp.ascontiguousarray(x.transpose(2,0,1).reshape(N,B*R)));del P,real,imag,f,x
        par,cfg=make_parameters(raw,cells[lo:hi],rate[cells[lo:hi]],0.,p)
        extra=np.random.default_rng(SEED+100000+lo).integers(0,10001,size=B*R,dtype='i4')
        phase=np.broadcast_to(offsets,(B,R)).copy().ravel()
        flags,st=sampler.run(cp,fn,*currents,par,cfg,R,30000,extra,SEED,lo,
            np.ascontiguousarray(wave[:,lo:hi].transpose(0,2,1)),phase)
        out[lo:hi]=st.get();del currents,flags,st;cp.get_default_memory_pool().free_all_blocks()
    rates=out[:,:,:2].sum(2)/2;values=rates.mean(1);sem=rates.std(1,ddof=1)/np.sqrt(R)
    np.savez_compressed(OUT/'samples.npz',cells=cells,replica_statistics=out,rate_mean_Hz=values,
        rate_SEM_Hz=sem,projected_phase_wave=wave,projected_mean=mean,residual_variance=powers@weights/N**2)
    rows=[];region=raw['region'][cells]
    for label,mask in [('selected_all',np.ones(len(cells),bool)),('selected_coreA',(cells<32000)&(region==0)),
        ('selected_coreB',(cells<32000)&(region==1)),('largest20_errors',np.isin(cells,d['largest_discrepancy_cells']))]:
        if mask.any():rows.append(dict(region=label,targets=int(mask.sum()),response_mean_Hz=float(values[mask].mean()),
            native_rate_RMS_error_Hz=float(np.sqrt(np.mean((values[mask]-d['native_rate_Hz'][mask])**2))),
            MCSEM_RMS_Hz=float(np.sqrt(np.mean(sem[mask]**2)))))
    ok=all(x['native_rate_RMS_error_Hz']<2 for x in rows if x['region']!='selected_all')
    result=dict(status='COMPLETE_LOCAL_SOURCE_PHASE_RESPONSE',rows=rows,causal_R_phase_range_Hz=[float(causal.min()),float(causal.max())],
        permits_bounded_allcell_phase_test=ok,formal_bifurcation_allowed=False,autonomous_closure=False,
        elapsed_s=time.time()-started,producer_sha256=sha(__file__))
    write(OUT/'result.json',result);shutil.copy2(__file__,OUT/'producer.py');print(result,flush=True)


if __name__=='__main__':main()
