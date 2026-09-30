#!/usr/bin/env python3
"""Direct stationary response at all targets of the corresponding K9 density.

No neural transfer approximation, root claim, parameter sweep, or tolerance fit.
Local M is held at the observed endpoint; that limitation is retained explicitly.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import time
import numpy as np
from campaign import ROOT,read,write,sha
from conditional_density_inputs import OPS
from coupled_density_exit import ADAPTED
from audit_target_root_response import density_condition
import lif_mc

OUT=ROOT/'all_target_stationary_direct'
AUDIT=ROOT/'target_stationary_response_audit'


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    geo=dict(np.load(OPS/'geometry.npz'));display=dict(np.load(ADAPTED/'geometry.npz'))
    p=read(OPS/'prepared.json')['params'];group=geo['cell_group'];sizes=geo['group_size'];E=np.arange(40000)<32000
    with np.load(AUDIT/'response.npz') as z:
        physical=z['physical'];g=z['g'];M=z['M'];source=z['mean_source_rate_per_ms'];old=z['predicted_target_rate_Hz']
    theta=geo['threshold_mv'][group];N=len(group)
    write(OUT/'contract.json',dict(status='FROZEN_BEFORE_ALL_TARGET_COUNTS',created_epoch=time.time(),
        question='Does the direct density-cell stationary response at all physical targets reproduce the corresponding network working point before building any new equilibrium/stability approximation?',
        design=dict(physical_targets=N,replicas_per_stream=256,independent_streams=2,record_ms=4000,burn_ms=1000,
                    base_seeds=[928831,928832],device=0,batch=256),
        input='Measured group source rates and exact expected external mean over5-10s, exact individual target weights and actual heldZ/K, selfconsistent stationary G from discrete causalR relation, observed individualM endpoint. Identical inputs to the prior one-point frozen-response audit.',
        simulator='Original density one-increment Gaussian synaptic filters, native membrane/reset/refractory. Each physical cell uses its own noise stream; paired stream identities are fixed. No neural transfer, no network fit. M remains fixed locally and upstream mean rates/G remain supplied.',
        readouts='Per-target rates/SEM, source-group selfconsistency residuals, core/allE/I means,400cell spatial field, impliedG. Independent numerical streams assess estimator precision; not native network seeds.',
        stopping='Exactly two fixed-input arrays, no automatic root or continuation. Classify bias versus numerical error and document the remaining fixedM/upstream-stationarity assumptions.',
        producer_sha256=sha(__file__),input_sha256=sha(AUDIT/'response.npz'),formal_bifurcation_allowed=False))
    started=time.time();pars=[]
    for i in range(N):
        q,_=density_condition(physical[i],g[i],theta[i],'E' if E[i] else 'I',p);pars.append(q)
    pars=np.array(pars);np.savez_compressed(OUT/'inputs.npz',pars=pars,physical=physical,g=g,M=M,source_per_ms=source)
    means=[];sems=[]
    for stream,seed in enumerate([928831,928832]):
        counts=[]
        for lo in range(0,N,256):
            write(OUT/'progress.json',dict(status='DIRECT_STATIONARY_COUNTS',pid=os.getpid(),stream=stream,
                completed_targets=lo,total_targets=N,elapsed_s=time.time()-started))
            # Different batch seeds and crn=False prevent accidental shared paths
            # between physical cells, while each stream remains reproducible.
            out=lif_mc.run(pars[lo:lo+256],256,4000,1000,seed+100000*lo,crn=False,device=0,batch=256)
            assert not out[:,:,3].any()
            counts.append(out[:,:,2].astype('i4'))
        counts=np.concatenate(counts);rates=counts/4.
        mean=rates.mean(1);sem=rates.std(1,ddof=1)/16.
        np.savez_compressed(OUT/f'stream_{stream}.npz',counts=counts,rate_Hz=mean,SEM_Hz=sem)
        means.append(mean);sems.append(sem)
    means=np.array(means);sems=np.array(sems);mean=means.mean(0);sem=np.sqrt((sems**2).sum(0))/2
    aggregate=lambda a:np.bincount(group,weights=a,minlength=len(sizes))/sizes
    rate=aggregate(mean);group_sem=np.sqrt(np.bincount(group,weights=sem**2,minlength=len(sizes)))/sizes
    observed=source*1000;region=geo['group_region'];pop=geo['population']==0
    rows=[]
    for label,mask in [('allE',pop),('coreA',pop&(region==0)),('coreB',pop&(region==1)),('surround',pop&(region==2)),('I',~pop)]:
        err=rate[mask]-observed[mask]
        rows.append(dict(region=label,observed_density_rate_Hz=float(np.average(observed[mask],weights=sizes[mask])),
            direct_stationary_rate_Hz=float(np.average(rate[mask],weights=sizes[mask])),
            regional_MC_SEM_Hz=float(np.sqrt(np.sum((group_sem[mask]*sizes[mask])**2))/sizes[mask].sum()),
            group_weighted_RMS_Hz=float(np.sqrt(np.average(err**2,weights=sizes[mask]))),maximum_group_error_Hz=float(abs(err).max())))
    cell=display['group_cell'][group];counts=np.bincount(cell[E],minlength=400)
    field=np.bincount(cell[E],weights=mean[E],minlength=400)/np.maximum(counts,1)
    sourcefield=np.bincount(display['group_cell'][pop],weights=observed[pop]*sizes[pop],minlength=400)/np.maximum(counts,1)
    causal=.1/(15*(-np.expm1(-.1/15)));G=float(30*np.clip((mean[E].mean()*causal-200)/300,0,1))
    delta=means[0]-means[1];combined=np.sqrt((sems**2).sum(0))
    result=dict(status='COMPLETE_ALL_TARGET_LOCAL_RESPONSE',rows=rows,implied_stationary_Graw=G,
        group_selfconsistency_maximum_Hz=float(abs(rate-observed).max()),
        weighted_field_RMS_Hz=float(np.sqrt(np.average((field-sourcefield)**2,weights=counts))),
        stream_difference_rms_Hz=float(np.sqrt(np.mean(delta**2))),
        stream_combined_SEM_rms_Hz=float(np.sqrt(np.mean(combined**2))),
        target_response_difference_from_old_surrogate_RMS_Hz=float(np.sqrt(np.mean((mean-old)**2))),
        elapsed_s=time.time()-started,root_established=False,formal_bifurcation_allowed=False,
        interpretation='Direct fixed-input local response. Finite group residuals, fixedM and supplied mean source/G still require a selfconsistent dynamic treatment.')
    np.savez_compressed(OUT/'response.npz',cell_rate_Hz=mean,cell_SEM_Hz=sem,group_rate_Hz=rate,group_SEM_Hz=group_sem,
        observed_group_rate_Hz=observed,field_Hz=field,observed_density_field_Hz=sourcefield,
        physical=physical,g=g,M=M,independent_stream_rates_Hz=means)
    write(OUT/'result.json',result);write(OUT/'progress.json',result);print(result,flush=True)


if __name__=='__main__':main()
