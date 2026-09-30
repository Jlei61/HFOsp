#!/usr/bin/env python3
"""All-target DC susceptibility from direct density equations, paired amplitudes.

Two fixed target partitions can run independently. No rate fitting or automatic
equilibrium continuation. Statistical derivative uncertainty is saved for the
subsequent spatial response calculation.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,time
import numpy as np
import cupy as cp
from campaign import ROOT,read,write,sha
from conditional_density_inputs import OPS
from audit_target_root_response import density_condition
from measure_target_direct_response import run
import measure_target_raw_G_response as physical_G
import lif_mc

OUT=ROOT/'all_target_dc_direct'
SOURCE=ROOT/'all_target_stationary_direct'


def kernel():
    code=lif_mc.CODE
    changes={
      'void assay(':'void direct_target_dc(',
      'curand_init(seed,crn?(unsigned long long)k:(unsigned long long)id,0,&rng);':
      'curand_init(seed,((unsigned long long)(g/2)<<32)+(unsigned long long)k,0,&rng);',
      'if(channel==0)mo=o;else if(channel==1)af=sqrt(1+o);else gf=sqrt(1+o);':
      'if(channel==0)mo=o;else if(channel==1)af=sqrt(1+o);else if(channel==2)gf=sqrt(1+o);',
      'if(ref[side]==0){v[side]=p[18]*v[side]+(1-p[18])*cur;':
      'double decay=p[18];if(channel==3 && o!=0.){double h=1.+p[22];cur=(h*cur-17.662847938268442*o)/(h+o);decay=exp(-.1*(h+o)/p[23]);}\n   if(ref[side]==0){v[side]=decay*v[side]+(1-decay)*cur;'}
    for before,after in changes.items():
        assert code.count(before)==1;code=code.replace(before,after)
    return cp.RawKernel(code,'direct_target_dc',options=('--fmad=false',))


def prepare():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    result=read(SOURCE/'result.json');assert result['status']=='COMPLETE_ALL_TARGET_LOCAL_RESPONSE'
    assert result['weighted_field_RMS_Hz']<.1
    write(OUT/'contract.json',dict(status='REGISTERED_AFTER_ALL_TARGET_RESPONSE_CORRESPONDENCE',created_epoch=time.time(),
        question='What is the spatial selfconsistency Jacobian near the actual corresponding high state when every local derivative is measured from the density equations rather than inherited from the rejected static neural surrogate?',
        basis='Full40000target local mean check reproduces the supplied network field with weightedRMS below.1Hz. Independent physicalG derivative chain rule11/11 and amplitude55/55 passed. Existing root is not used as a validated equilibrium.',
        design=dict(physical_targets=40000,partitions=[[0,20000],[20000,40000]],devices=[0,1],
                    replicas=256,record_ms=4000,burn_ms=1000,base_seed=928861,target_batch=128,
                    channels=['effective_mean','effective_variance_E','effective_variance_I','physical_applied_G_E_only'],
                    amplitudes='mean .02*(theta-11), varianceE5%, varianceI10%, appliedG .002*(1+g); every target/channel repeated at half amplitude'),
        stochastic_unit='Independent physical-target/channel noise. Full and half amplitudes share replica paths, and each modulation uses paired plus/minus sides. Target partitions use disjoint batch seeds. These are numerical samples, not independent native seeds.',
        uncertainty='Save all paired counts plus eight replicate-block derivative estimates. Retain nonestimable/failed-amplitude channels and propagate their uncertainty into spatial results; never replace them by zero or declare them passed.',
        gate='Per-target amplitude comparison <=max(10%abs(fullgain),2pairedSEM,1e-7); SNR>=10 for individually estimable channels. Aggregated source-group action and its uncertainty must then be checked before any spectral claim. A DC matrix alone does not establish dynamical stability; synaptic delays/G/M feedback remain required.',
        limits='Same fixedM and supplied mean source/G input as all-target static check. Static adaptation feedback must be restored through its implicit equation. A near-selfconsistent measured response is not an exact residual root. No native physics modification, response fit, parameter sweep, or automatic branch naming.',
        input_sha256=sha(SOURCE/'inputs.npz'),producer_sha256=sha(__file__),formal_bifurcation_allowed=False))
    print('Registered full target DC response',flush=True)


def worker(part,device):
    contract=read(OUT/'contract.json');assert contract['producer_sha256']==sha(__file__)
    folder=OUT/f'part_{part}';folder.mkdir(exist_ok=True);assert not (folder/'progress.json').exists()
    cp.cuda.Device(device).use();started=time.time()
    write(folder/'progress.json',dict(status='INITIALIZING',pid=os.getpid(),device=device,updated_epoch=time.time()))
    with np.load(SOURCE/'inputs.npz') as z:basepars=z['pars'];physical=z['physical'];g=z['g']
    k=kernel();oracle=physical_G.kernel()
    # One pair has the same stream mapping as the established physical kernel;
    # validate each channel, including the physical shunt, before dispatch.
    for channel in range(4):
        q=np.repeat(basepars[5059:5060],2,axis=0);q[:,20]=channel;q[:,22]=g[5059];q[:,23]=20.
        h=[.02*(q[0,1]-11),.05*physical[5059,1],.1*physical[5059,2],.002*(1+g[5059])][channel]
        q[:,4]=h*np.array([1.,.5])/(physical[5059,channel] if channel in [1,2] else 1.)
        a=run(oracle,q,64,100,30,928869);b=run(k,q,64,100,30,928869)
        assert np.array_equal(a,b),channel
    write(folder/'implementation_qa.json',dict(status='PASS',four_channels_bitwise=True,
        stream_mapping='same stream for adjacent full/half pair, independent pair index for target/channel'))
    start,stop=contract['design']['partitions'][part]
    for lo in range(start,stop,128):
        hi=min(lo+128,stop);pars=[];cells=[];channels=[];amplitudes=[]
        for i in range(lo,hi):
            popE=i<32000;steps=[.02*(basepars[i,1]-11),.05*physical[i,1],.1*physical[i,2],.002*(1+g[i])]
            for channel in range(4 if popE else 3):
                if steps[channel]<=0:continue
                for factor in [1.,.5]:
                    p=basepars[i].copy();p[20]=channel;p[22]=g[i];p[23]=20. if popE else 10.
                    amp=steps[channel]*factor;p[4]=amp/(physical[i,channel] if channel in [1,2] else 1.)
                    pars.append(p);cells.append(i);channels.append(channel);amplitudes.append(amp)
        pars=np.array(pars);amp=np.array(amplitudes)
        observed=run(k,pars,256,4000,1000,928861+100000*lo)
        counts=observed[:,:,2:].astype('u2');difference=counts[:,:,0].astype(float)-counts[:,:,1]
        assert np.array_equal(observed[:,:,0],difference*.5) and not observed[:,:,1].any()
        gain=difference/(8*amp[:,None]);mean=gain.mean(1);sem=gain.std(1,ddof=1)/16.
        delta=gain[1::2]-gain[::2];dmean=delta.mean(1);dsem=delta.std(1,ddof=1)/16.
        tol=np.maximum.reduce([.1*abs(mean[::2]),2*dsem,np.full(len(dmean),1e-7)])
        np.savez_compressed(folder/f'batch_{lo:05d}.npz',cell=np.array(cells),channel=np.array(channels),amplitude=amp,
            counts=counts,gain=mean,SEM=sem,replicate_block_gain=gain.reshape(len(gain),8,32).mean(2),
            amplitude_difference=dmean,amplitude_difference_SEM=dsem,amplitude_tolerance=tol,
            amplitude_pass=abs(dmean)<=tol,estimable=abs(mean)/np.maximum(sem,1e-15)>=10,mean_rate_Hz=counts.mean((1,2))/4)
        progress=dict(status='MEASURING',pid=os.getpid(),device=device,completed_targets=hi-start,total_targets=stop-start,
            elapsed_s=time.time()-started,updated_epoch=time.time())
        write(folder/'progress.json',progress)
        if (lo-start)//128%10==0:print('DIRECT TARGET DC',part,hi-start,stop-start,flush=True)
    result=dict(status='COMPLETE',part=part,target_range=[start,stop],elapsed_s=time.time()-started,
        formal_bifurcation_allowed=False,producer_sha256=sha(__file__))
    write(folder/'result.json',result);write(folder/'progress.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','worker']);p.add_argument('--part',type=int,choices=[0,1]);p.add_argument('--device',type=int,default=0)
    a=p.parse_args();prepare() if a.command=='prepare' else worker(a.part,a.device)
