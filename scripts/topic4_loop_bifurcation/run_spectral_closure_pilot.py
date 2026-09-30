#!/usr/bin/env python3
"""Three bounded diagonal-source spectral generations with explicit limitations."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse, shutil, subprocess, time
from pathlib import Path
import numpy as np
from scipy import sparse
from campaign import ROOT,PYTHON,read,write,sha
from conditional_density_inputs import OPS
from observe_source_time_structure import OUT as TEMPORAL
from prepare_spectral_closure_pilot import OUT,N,REPLICAS,GENERATIONS
from individual_spectral_sampler import kernel,make_parameters,run


def qa(device):
    import cupy as cp
    cp.cuda.Device(device).use();fn=kernel(cp)
    raw=dict(np.load(OUT/'parameters.npz'));params=read(OPS/'prepared.json')['params']
    tr=dict(np.load(TEMPORAL/'target_traces.npz'));cells=tr['cells'];B=len(cells)
    sp=sparse.load_npz(TEMPORAL/'source_spikes_0p1ms.npz').tocsr()[cells].toarray().T
    p,cfg=make_parameters(raw,cells,np.zeros(B),0.,params,True)
    flags,stats=run(cp,fn,cp.asarray(np.ascontiguousarray(tr['IE_II_V_M'][:,0,:])),
        cp.asarray(np.ascontiguousarray(tr['IE_II_V_M'][:,1,:])),p,cfg,1,0,np.zeros(B,'i4'),929531,0,True)
    assert np.array_equal(flags.get(),sp)
    assert np.allclose(stats.get()[:,0,4],tr['IE_II_V_M'][:,3,:].mean(0),rtol=0,atol=1e-12)
    # Projection normalization is checked against independent previous diagnostics.
    S=np.load(OUT/'generation_0/source_PSD.npy',mmap_mode='r');H=np.load(OUT/'filter_power.npy')
    old=dict(np.load(TEMPORAL/'spectral_analysis/source_spectra.npz'));w=np.full(N//2+1,2.);w[[0,-1]]=1
    errors=[]
    for q,(kind,source) in enumerate([('ampa',slice(0,32000)),('gaba',slice(32000,40000))]):
        W=sparse.load_npz(OUT/f'original_{kind}_jump.npz').tocsr()[cells]
        power=W.multiply(W)@S[source];var=(power*H[q])@w/N**2
        expected=old['predicted_IE_II_variance'][q,cells].copy()
        if q==0:expected-=old['external_variance'][cells]
        errors.append(float(abs(var-expected).max()));assert np.allclose(var,expected,rtol=1e-11,atol=1e-10)
    result=dict(status='PASS',every_observed_spike_exact=True,targets=B,native_M_mean_exact=True,
        projection_variance_max_abs_errors=errors,individual_thresholds=True,
        current_G_negligible_at_replay_anchor=True,sampler_sha256=sha(Path(__file__).with_name('individual_spectral_sampler.py')),
        producer_sha256=sha(__file__))
    write(OUT/'implementation_qa.json',result);print(result,flush=True)


def worker(generation,part,device,batch):
    import cupy as cp
    import cupyx.scipy.sparse as csp
    cp.cuda.Device(device).use()
    contract=read(OUT/'contract.json');assert 1<=generation<=GENERATIONS
    assert read(OUT/'implementation_qa.json')['status']=='PASS'
    folder=OUT/f'generation_{generation}';previous=OUT/f'generation_{generation-1}'
    assert (previous/'complete.json').exists()
    started=time.time(); raw=dict(np.load(OUT/'parameters.npz'));params=read(OPS/'prepared.json')['params']
    rate=np.load(previous/'source_rate_Hz.npy');S=np.load(previous/'source_PSD.npy',mmap_mode='r')
    lo0,hi0=part*20000,(part+1)*20000
    ca=.1/(15*(-np.expm1(-.1/15)));G=float(30*np.clip((rate[:32000].mean()*ca-200)/300,0,1))
    # Dense source spectra are immutable during this generation.
    spec=cp.asarray(S);H=cp.asarray(np.load(OUT/'filter_power.npy'))
    W=[];mean=[]
    for q,(kind,source) in enumerate([('ampa',slice(0,32000)),('gaba',slice(32000,40000))]):
        original=sparse.load_npz(OUT/f'original_{kind}_jump.npz').tocsr()[lo0:hi0]
        mean.append(np.asarray(original@(rate[source]*.0001))/(1-np.exp(-.1/params['tau_r_'+kind.upper()])))
        W.append(csp.csr_matrix(original.multiply(original)))
    out=np.lib.format.open_memmap(folder/'source_PSD.npy',mode='r+')
    out_rate=np.lib.format.open_memmap(folder/'source_rate_Hz.npy',mode='r+')
    fn=kernel(cp);R=REPLICAS
    summaries=[];spectrum_residual=[];spectrum_var=[];MC=[];generated_var=[]
    for lo in range(lo0,hi0,batch):
        hi=min(lo+batch,hi0);B=hi-lo;local=slice(lo-lo0,hi-lo0)
        write(folder/f'progress_part{part}.json',dict(status='RUNNING',pid=os.getpid(),generation=generation,
            device=device,completed=lo-lo0,total=hi0-lo0,elapsed_s=time.time()-started,updated_epoch=time.time()))
        rng=cp.random.RandomState(929531+lo)
        currents=[];targetvar=[];actualvar=[]
        for q,source in enumerate([slice(0,32000),slice(32000,40000)]):
            P=(W[q][local]@spec[source])*H[q][None,:]
            assert bool(cp.isfinite(P).all()) and float(P.min())>=0
            real=rng.standard_normal((B,R,N//2+1),dtype='f8');imag=rng.standard_normal(real.shape,dtype='f8')
            X=(real+1j*imag)*cp.sqrt(P[:,None,:]/2.)
            X[:,:,0]=0.;X[:,:,-1]=real[:,:,-1]*cp.sqrt(P[:,-1,None])
            x=cp.fft.irfft(X,n=N,axis=2)
            weights=cp.full(N//2+1,2.,dtype='f8');weights[[0,-1]]=1.
            targetvar.append((P@weights/N**2).get());actualvar.append(x.var(axis=2).get())
            x+=cp.asarray(mean[q][local])[:,None,None]
            currents.append(cp.ascontiguousarray(x.transpose(2,0,1).reshape(N,B*R)))
            del P,real,imag,X,x
        p,cfg=make_parameters(raw,np.arange(lo,hi),rate[lo:hi],G,params)
        # Random phases do not consume external Poisson streams.
        extra=np.random.default_rng(929531+lo).integers(0,10001,size=B*R,dtype='i4')
        flags,st=run(cp,fn,currents[0],currents[1],p,cfg,R,30000,extra,929531,lo)
        st=st.get();summaries.append(st.mean(1));MC.append(st)
        x=flags.T.reshape(B,R,N).astype('f8');F=cp.fft.rfft(x,axis=2);F[:,:,0]=0
        power=cp.mean(abs(F)**2,axis=1);value=power.get();out[lo:hi]=value
        r=(st[:,:,0]+st[:,:,1])/2.;out_rate[lo:hi]=r.mean(1)
        w=np.full(N//2+1,2.);w[[0,-1]]=1
        variance=value@w/N**2
        expected=np.mean((r*.0001)*(1-r*.0001),axis=1)
        assert np.allclose(variance,expected,rtol=1e-10,atol=1e-12)
        spectrum_var.append(variance)
        spectrum_residual.append(np.sum(abs(value-np.asarray(S[lo:hi]))*w,axis=1)/N**2)
        generated_var.append(np.stack([np.stack(targetvar),np.stack(actualvar).mean(2)],axis=1))
        del flags,x,F,power,st,currents
        cp.get_default_memory_pool().free_all_blocks()
    out.flush();out_rate.flush()
    np.savez_compressed(folder/f'part{part}_statistics.npz',cells=np.arange(lo0,hi0),
        summary=np.concatenate(summaries),replica_statistics=np.concatenate(MC),
        output_spike_variance=np.concatenate(spectrum_var),PSD_L1_change=np.concatenate(spectrum_residual),
        recurrent_expected_and_generated_variance=np.concatenate(generated_var,axis=2))
    write(folder/f'part{part}_complete.json',dict(status='COMPLETE',G_from_previous_source_means=G,
        elapsed_s=time.time()-started,targets=hi0-lo0,replicas=R,pid=os.getpid(),device=device))
    write(folder/f'progress_part{part}.json',dict(status='COMPLETE',elapsed_s=time.time()-started,updated_epoch=time.time()))


def collect(generation):
    folder=OUT/f'generation_{generation}';previous=OUT/f'generation_{generation-1}'
    done=[read(folder/f'part{i}_complete.json') for i in range(2)]
    assert done[0]['G_from_previous_source_means']==done[1]['G_from_previous_source_means']
    G=done[0]['G_from_previous_source_means'];raw=dict(np.load(OUT/'parameters.npz'))
    a=[dict(np.load(folder/f'part{i}_statistics.npz')) for i in range(2)]
    st=np.concatenate([p['summary'] for p in a]);rates=np.load(folder/'source_rate_Hz.npy')
    before=np.load(previous/'source_rate_Hz.npy');native_rate=np.load(OUT/'generation_0/source_rate_Hz.npy')
    region=raw['region'];E=np.arange(40000)<32000
    psdchange=np.concatenate([p['PSD_L1_change'] for p in a]);v=np.concatenate([p['output_spike_variance'] for p in a])
    rows=[]
    for label,mask in [('allE',E),('coreA',E&(region==0)),('coreB',E&(region==1)),('surroundE',E&(region==2)),('I',~E)]:
        rows.append(dict(region=label,targets=int(mask.sum()),input_rate_Hz=float(before[mask].mean()),
            output_rate_Hz=float(rates[mask].mean()),native_reference_rate_Hz=float(native_rate[mask].mean()),
            individual_rate_RMS_change_Hz=float(np.sqrt(np.mean((rates[mask]-before[mask])**2))),
            mean_first_second_1s_Hz=st[mask,:2].mean(0).tolist(),
            mean_M_start_end=st[mask,2:4].mean(0).tolist(),mean_M=st[mask,4].mean().item(),
            counterfactual_mean_Zdot_per_s=float(np.mean((st[mask,10]-raw['Z'][mask])/5)) if label!='I' else None,
            mean_PSD_L1_change=float(psdchange[mask].mean()),mean_spike_variance=float(v[mask].mean()),
            mean_negative_IE_II_fractions=st[mask,11:13].mean(0).tolist()))
    display=raw['display'][:32000];counts=np.bincount(display,minlength=400)
    field=lambda r:np.bincount(display,weights=r[:32000],minlength=400)/np.maximum(counts,1)
    err=field(rates)-field(native_rate);ca=.1/(15*(-np.expm1(-.1/15)))
    result=dict(status='COMPLETE_STATIONARY_SPECTRAL_GENERATION',generation=generation,rows=rows,
        G_used=G,G_implied_by_output=float(30*np.clip((rates[:32000].mean()*ca-200)/300,0,1)),
        output_causal_R_DC_Hz=float(rates[:32000].mean()*ca),
        development_native_field_weighted_RMS_Hz=float(np.sqrt(np.average(err**2,weights=counts))),
        protocol_difference='Fixed native-window-mean external expected drive; native reference has time-varying external modulation. Native reference is one developmenttrajectory, not an independent acceptance set.',
        formal_bifurcation_allowed=False,root_established=False,physical_stability_established=False,
        elapsed_worker_s=[p['elapsed_s'] for p in done])
    write(folder/'complete.json',result);print(result,flush=True)


def supervise(batch):
    assert read(OUT/'implementation_qa.json')['status']=='PASS'
    assert not (OUT/'supervisor.json').exists()
    shutil.copy2(__file__,OUT/'run_producer.py')
    shutil.copy2(Path(__file__).with_name('individual_spectral_sampler.py'),OUT/'sampler_producer.py')
    write(OUT/'supervisor.json',dict(status='RUNNING',pid=os.getpid(),batch=batch,
        created_epoch=time.time(),producer_sha256=sha(__file__)))
    for gen in range(1,GENERATIONS+1):
        folder=OUT/f'generation_{gen}';folder.mkdir()
        for name,shape in [('source_PSD.npy',(40000,N//2+1)),('source_rate_Hz.npy',(40000,))]:
            a=np.lib.format.open_memmap(folder/name,mode='w+',dtype='f8',shape=shape);a[:]=np.nan;a.flush();del a
        jobs=[]
        for part in range(2):
            log=(folder/f'worker_part{part}.log').open('w')
            p=subprocess.Popen([PYTHON,__file__,'worker','--generation',str(gen),'--part',str(part),'--device',str(part),'--batch',str(batch)],stdout=log,stderr=subprocess.STDOUT)
            jobs.append((p,log))
        codes=[]
        for p,log in jobs:codes.append(p.wait());log.close()
        if any(codes):
            write(OUT/'supervisor.json',dict(status='FAILED',generation=gen,codes=codes,updated_epoch=time.time()));raise RuntimeError(codes)
        collect(gen)
    write(OUT/'supervisor.json',dict(status='COMPLETE_THREE_BOUNDED_GENERATIONS',updated_epoch=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['qa','worker','supervise','collect'])
    p.add_argument('--generation',type=int,default=1);p.add_argument('--part',type=int,default=0)
    p.add_argument('--device',type=int,default=0);p.add_argument('--batch',type=int,default=256);a=p.parse_args()
    if a.command=='qa':qa(a.device)
    elif a.command=='worker':worker(a.generation,a.part,a.device,a.batch)
    elif a.command=='collect':collect(a.generation)
    else:supervise(a.batch)
