#!/usr/bin/env python3
"""Paired initial-history test for loss of coherent source phase structure."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import shutil,time
import numpy as np
from scipy import sparse
from campaign import ROOT,read,write,sha
from prepare_phase_source_closure import OUT as SOURCE,L,N,OPS
from compare_local_correlated_inputs import OUT as PRIOR,R,BATCH,SEED
from individual_spectral_sampler import make_parameters
import coherent_phase_sampler as sampler

OUT=ROOT/'high_history_local_phase_initial_history'


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_PAIRED_INITIAL_HISTORY_TEST',created_epoch=time.time(),
        question='Does the observed reduction in output phase coherence reflect loss of native fast-state history during local response initialization?',
        trigger='First40000-target phase-map response matchesnative rates and spatialfield but reduces sourcephasevariance. Random phase and resetinitialization could erase phase-conditioned history, so rate agreement alone is insufficient.',
        design='Two paired59target x512replica localresponses with identicalsource-phasewave, independent-sourceGaussianresidual, Poissonstreams and3-4sburn. Both start inputphase0; change only initialV/ref/M from reset/ratemean toactual72s nativejointV/ref/M. Record2s, align everyoutputspike toactualinputphase. No nativewaveform beyond initialsource statistics, no newnative run or autonomousclosureclaim.',
        decision='A retainedhistory effect would require preserving phase-conditioned faststates beforestationarybranchanalysis. Absenceofeffect would rule out this particular finite-burninitialization explanation, not prove allsourcecorrelations captured.',
        producer_sha256=sha(__file__),formal_bifurcation_allowed=False))
    import cupy as cp
    cp.cuda.Device(0).use();cp.get_default_memory_pool().set_limit(size=3*2**30);fn=sampler.kernel(cp)
    d=dict(np.load(PRIOR/'inputs.npz'));cells=d['cells'];raw=dict(np.load(SOURCE/'parameters.npz'));params=read(OPS/'prepared.json')['params']
    folder=SOURCE/'generation_0';rate=np.load(folder/'source_rate_Hz.npy');phase_native=np.load(folder/'phase_spike_probability.npy')[cells]
    S=np.load(folder/'residual_PSD.npy',mmap_mode='r');H=np.load(SOURCE/'filter_power.npy');powers=[]
    for q,(kind,ss) in enumerate([('ampa',slice(0,32000)),('gaba',slice(32000,40000))]):
        W=sparse.load_npz(SOURCE/f'original_{kind}_jump.npz').tocsr()[cells]
        powers.append((W.multiply(W)@S[ss])*H[q])
    powers=np.array(powers);mean=np.load(folder/'current_mean.npy')[:,cells];wave=np.load(folder/'current_phase_wave.npy')[:,cells]
    stats=np.empty((2,len(cells),R,16));counts=np.zeros((2,len(cells),L));exposure=np.zeros_like(counts);started=time.time()
    for lo in range(0,len(cells),BATCH):
        hi=min(lo+BATCH,len(cells));B=hi-lo;rng=cp.random.RandomState(SEED+lo);currents=[]
        for q in range(2):
            P=cp.asarray(powers[q,lo:hi]);real=rng.standard_normal((B,R,N//2+1));imag=rng.standard_normal(real.shape)
            f=(real+1j*imag)*cp.sqrt(P[:,None,:]/2);f[:,:,0]=0;f[:,:,-1]=real[:,:,-1]*cp.sqrt(P[:,-1,None])
            x=cp.fft.irfft(f,n=N,axis=2)+cp.asarray(mean[q,lo:hi])[:,None,None]
            currents.append(cp.ascontiguousarray(x.transpose(2,0,1).reshape(N,B*R)));del P,real,imag,f,x
        extra=np.random.default_rng(SEED+100000+lo).integers(0,10001,size=B*R,dtype='i4');phase=np.zeros(B*R,'i4')
        for condition in range(2):
            p,cfg=make_parameters(raw,cells[lo:hi],rate[cells[lo:hi]],0.,params,replay=bool(condition))
            flags,st=sampler.run(cp,fn,*currents,p,cfg,R,30000,extra,SEED,lo,
                np.ascontiguousarray(wave[:,lo:hi].transpose(0,2,1)),phase)
            stats[condition,lo:hi]=st.get();base=(30000+extra.reshape(B,R))%L
            for psi in range(L):
                value=flags[psi::L].sum(0).reshape(B,R).get();idx=(base+psi)%L
                target=np.broadcast_to(np.arange(B)[:,None],idx.shape)
                np.add.at(counts[condition,lo:hi],(target,idx),value)
                np.add.at(exposure[condition,lo:hi],(target,idx),len(range(psi,N,L)))
            del flags,st
        del currents;cp.get_default_memory_pool().free_all_blocks()
    assert np.array_equal(counts.sum(2),stats[:,:,:,:2].sum((2,3)));assert np.all(exposure.sum(2)==R*N)
    phase=counts/exposure;values=stats[:,:,:,:2].sum(3).mean(2)/2
    np.savez_compressed(OUT/'samples.npz',cells=cells,replica_statistics=stats,phase_spike_probability=phase,
        native_phase_spike_probability=phase_native,rate_mean_Hz=values)
    rows=[];region=raw['region'][cells]
    for label,mask in [('selected_all',np.ones(len(cells),bool)),('selected_coreA',(cells<32000)&(region==0)),
        ('selected_coreB',(cells<32000)&(region==1)),('largest20_errors',np.isin(cells,d['largest_discrepancy_cells']))]:
        rows.append(dict(region=label,targets=int(mask.sum()),
            rate_RMS_error_Hz=np.sqrt(np.mean((values[:,mask]-d['native_rate_Hz'][mask])**2,axis=1)).tolist(),
            phase_probability_RMS_from_native=np.sqrt(np.mean((phase[:,mask]-phase_native[mask])**2,axis=(1,2))).tolist(),
            mean_phase_variance=phase[:,mask].var(2).mean(1).tolist(),native_mean_phase_variance=float(phase_native[mask].var(1).mean()),
            paired_rate_difference_RMS_Hz=float(np.sqrt(np.mean((values[1,mask]-values[0,mask])**2))),
            paired_phase_probability_difference_RMS=float(np.sqrt(np.mean((phase[1,mask]-phase[0,mask])**2)))))
    result=dict(status='COMPLETE_PAIRED_LOCAL_INITIAL_HISTORY_TEST',conditions=['resetV_ref_rateM','native72s_joint_V_ref_M'],
        rows=rows,elapsed_s=time.time()-started,phase_count_and_exposure_conservation=True,
        limits='Paired finite-burn local test. Neither autonomous network nor infinite-time stationary state; one development sourcephase field.',producer_sha256=sha(__file__))
    write(OUT/'result.json',result);shutil.copy2(__file__,OUT/'producer.py');print(result,flush=True)


if __name__=='__main__':main()
