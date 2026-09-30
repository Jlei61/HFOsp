#!/usr/bin/env python3
"""Distinguish within-realization phase structure from averaging across replicas."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,shutil,time
import numpy as np
from scipy import sparse
from campaign import ROOT,read,write,sha
from prepare_phase_source_closure import OUT as SOURCE,L,N,OPS
from compare_local_correlated_inputs import R,BATCH,SEED
from individual_spectral_sampler import make_parameters
import coherent_phase_sampler as sampler

OUT=ROOT/'phase_replica_averaging_review'


def main(reset=False):
    global OUT
    if reset:OUT=ROOT/'phase_replica_averaging_reset_control'
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_REPLICA_PHASE_DECOMPOSITION',created_epoch=time.time(),
        question='How much sourcephase structure is lost by averaging localnumericalreplicas, as opposed to being absent withinindividualreplicas?',
        design='One80prespecifiedrastercell x512replica localresponse. Same source-defined22step phasewave and independent-source residual as the completedphasepilot. Inputphase0, pairedfixedexternalmean,3-4sburn and2srecord. Save phaseprobability separately for everyreplica before pooling. Compare phasevariance withinreplicas versus varianceofmeanprofile and existingnative5x2s/10sraster.',
        initialization='resetV/ref and source-meanM; paired80target control changes only this initialization' if reset else 'True72s jointV/ref/M',
        identity='Exact lawtotalvariance over the finite replica byphase array; use equalphase andequalreplica weights for decomposition, save exposure-weightedprofile separately. Phase-averaging differences are not a rate or time-series stability test.',
        unit='Oneconditionaldevelopment sourcefield;80fixedstratifiedrastercells,512 numericalcopies. Neither80 independentnetworks nor a freshautonomousseed.',
        decision='Ifsingle replicas retain native-like phasevar butmeanprofile losesit, the closure loses conditional/quenchedphase structure in its averaging, ratherthan demonstrating absenceofrhythm in individualresponses. Ifsingle replicas also lackit, omittedinputtemporalstructure remains a competingexplanation.',
        producer_sha256=sha(__file__),formal_bifurcation_allowed=False))
    import cupy as cp
    cp.cuda.Device(1).use();cp.get_default_memory_pool().set_limit(size=3*2**30);fn=sampler.kernel(cp)
    previous=dict(np.load(ROOT/'native_phase_persistence_review/profiles.npz'));cells=previous['cells'];Btotal=len(cells)
    raw=dict(np.load(SOURCE/'parameters.npz'));params=read(OPS/'prepared.json')['params'];folder=SOURCE/'generation_0'
    rate=np.load(folder/'source_rate_Hz.npy');S=np.load(folder/'residual_PSD.npy',mmap_mode='r');H=np.load(SOURCE/'filter_power.npy')
    powers=[]
    for q,(kind,ss) in enumerate([('ampa',slice(0,32000)),('gaba',slice(32000,40000))]):
        W=sparse.load_npz(SOURCE/f'original_{kind}_jump.npz').tocsr()[cells]
        powers.append((W.multiply(W)@S[ss])*H[q])
    powers=np.array(powers);mean=np.load(folder/'current_mean.npy')[:,cells];wave=np.load(folder/'current_phase_wave.npy')[:,cells]
    stats=np.empty((Btotal,R,16));profiles=np.empty((Btotal,R,L));exposures=np.empty((Btotal,R,L));started=time.time()
    for lo in range(0,Btotal,BATCH):
        hi=min(lo+BATCH,Btotal);B=hi-lo;rng=cp.random.RandomState(SEED+lo);currents=[]
        for q in range(2):
            P=cp.asarray(powers[q,lo:hi]);real=rng.standard_normal((B,R,N//2+1));imag=rng.standard_normal(real.shape)
            f=(real+1j*imag)*cp.sqrt(P[:,None,:]/2);f[:,:,0]=0;f[:,:,-1]=real[:,:,-1]*cp.sqrt(P[:,-1,None])
            x=cp.fft.irfft(f,n=N,axis=2)+cp.asarray(mean[q,lo:hi])[:,None,None]
            currents.append(cp.ascontiguousarray(x.transpose(2,0,1).reshape(N,B*R)));del P,real,imag,f,x
        extra=np.random.default_rng(SEED+100000+lo).integers(0,10001,size=B*R,dtype='i4');offsets=np.zeros(B*R,'i4')
        p,cfg=make_parameters(raw,cells[lo:hi],rate[cells[lo:hi]],0.,params,replay=not reset)
        flags,st=sampler.run(cp,fn,*currents,p,cfg,R,30000,extra,SEED,lo,
            np.ascontiguousarray(wave[:,lo:hi].transpose(0,2,1)),offsets)
        stats[lo:hi]=st.get();base=(30000+extra.reshape(B,R))%L
        ii=np.arange(B)[:,None];jj=np.arange(R)[None,:]
        for psi in range(L):
            count=flags[psi::L].sum(0).reshape(B,R).get();exposure=len(range(psi,N,L));idx=(base+psi)%L
            profiles[lo:hi][ii,jj,idx]=count/exposure;exposures[lo:hi][ii,jj,idx]=exposure
        del flags,st,currents;cp.get_default_memory_pool().free_all_blocks()
    assert np.allclose((profiles*exposures).sum(2),stats[:,:,:2].sum(2),rtol=0,atol=1e-10)
    pooled=(profiles*exposures).sum(1)/exposures.sum(1);meanprofile=profiles.mean(1)
    within=profiles.var(2).mean(1);ofmean=meanprofile.var(1)
    lhs=within+profiles.mean(2).var(1);rhs=profiles.var(1).mean(1)+ofmean
    error=float(abs(lhs-rhs).max());assert error<1e-12
    np.savez_compressed(OUT/'replica_phase_profiles.npz',cells=cells,replica_statistics=stats,
        phase_probability=profiles,phase_exposure=exposures,exposure_weighted_mean_profile=pooled,
        equal_replica_mean_profile=meanprofile,mean_within_replica_phase_variance=within,phase_variance_of_mean_profile=ofmean)
    region=raw['region'][cells];E=cells<32000;rows=[]
    for label,mask in [('sampled_E',E),('sampled_coreA',E&(region==0)),('sampled_coreB',E&(region==1)),('sampled_I',~E)]:
        rows.append(dict(region=label,targets=int(mask.sum()),
            mean_within_replica_phase_variance=float(within[mask].mean()),phase_variance_of_mean_profile=float(ofmean[mask].mean()),
            native_fixed_background_2s_phase_variance=previous['phase_probabilities_native_fixed_blocks'][:,mask].var(2).mean(1).tolist(),
            native_fixed_background_10s_phase_variance=float(previous['native_fixed_full10s'][mask].var(1).mean()),
            mean_rate_Hz=float(stats[mask,:,:2].sum(2).mean()/2),
            native_matched_sample_rate_Hz=float(previous['native_raster_rate_Hz'][mask].mean())))
    result=dict(status='COMPLETE_REPLICA_PHASE_AVERAGING_DECOMPOSITION',rows=rows,
        total_variance_identity_max_error=error,spike_count_conservation=True,elapsed_s=time.time()-started,
        formal_bifurcation_allowed=False,producer_sha256=sha(__file__))
    write(OUT/'result.json',result);shutil.copy2(__file__,OUT/'producer.py');print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--reset-control',action='store_true')
    main(parser.parse_args().reset_control)
