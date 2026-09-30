#!/usr/bin/env python3
"""Fresh direct cell responses at the completed K9.35 held active state."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,time
import numpy as np
from campaign import ROOT,read,write,sha
from direct_response_system import DirectDC
from audit_target_root_response import density_condition
import lif_mc

OUT=ROOT/'held_exit_stationarity_K9p35'
SOURCE=ROOT/'carried_exit_lower_holds/held_K9p35'


def prepare():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    assert read(SOURCE/'result.json')['status']=='COMPLETE'
    d=DirectDC()  # Only exact input construction is used; not its K9 Jacobian.
    with np.load(SOURCE/'stationary_candidate_observations.npz') as z:
        M=z['mean_target_M'];Z=z['Z'];K=z['K'];external=z['expected_external_per_group'];observed_global=z['mean_global_state']
    with np.load(SOURCE/'trajectory.npz') as z:
        source=z['group_output'][-3000:,0,:].mean(axis=0,dtype=np.float64)
        blocks=z['group_output'][-3000:,0,:].reshape(3,1000,d.P).mean(1,dtype=np.float64)
    # The old audit averaged 128 identical held coordinates, introducing roundoff.
    Zround=float(abs(Z-d.Z).max());externalround=float(abs(external[d.group]-d.external).max())
    assert Zround<2e-15
    assert np.allclose(K,d.Kshape*9.35,rtol=1e-13,atol=1e-13)
    assert externalround<2e-14
    assert np.all(M[~d.E]==0)
    # Reconstruct the earlier reference once to detect changed input equations.
    old,gold,_,_=d.moments(d.reference_source,d.reference_M,9.)
    error=float(abs(old-d.reference_physical).max())
    assert error<1e-9 and np.allclose(gold,d.reference_g,rtol=1e-13,atol=1e-13)
    d.Z=Z.copy();d.Kshape=K/9.35;d.external=external[d.group].copy()
    physical,g,G,R=d.moments(source,M,9.35)
    pars=[]
    for i in range(d.N):
        q,_=density_condition(physical[i],g[i],d.geo['threshold_mv'][d.group[i]],'E' if d.E[i] else 'I',d.p)
        pars.append(q)
    np.savez_compressed(OUT/'inputs.npz',pars=np.array(pars),physical=physical,g=g,M=M,Z=Z,K=K,
        source_rate_Hz=source,source_1s_blocks_Hz=blocks,stationary_Graw=G,stationary_R_Hz=R,
        observed_global_R_s=observed_global,external_per_ms=d.external)
    write(OUT/'contract.json',dict(status='FROZEN_BEFORE_FRESH_COUNTS',created_epoch=time.time(),
        question='Does the K9.35 state that persists in the coupled density hold also satisfy direct stationary cell-response selfconsistency near the observed core-exit region?',
        selection='One relevant active state chosen after complete K9.2/9.35/9.5/9.65 holds. No new broad parameter grid.',
        design=dict(targets=d.N,streams=2,replicas=256,record_ms=4000,burn_ms=1000,seeds=[929351,929352]),
        inputs='Exact same individual target weights, heldactualZ/K, expectedexternalmean, original density-cell filters/membrane/refractory; upstream rates and individualM measured in the last3s of the completed held trajectory. Stationary G is recomputed from the supplied mean rate.',
        readouts='Independent counts/SEM per physical target, source-group residuals, individualM equilibrium residuals, core/allE/I and400cell field. Adjacent1s source blocks retain finite trajectory drift.',
        interpretation='This is a nonlinear stationarity check, not a root or stability certificate. Count SEM excludes upstream finite-window/density-closure error. Do not reuse the K9 Jacobian as K9.35 derivatives. No automatic Newton or continuation.',
        source=str(SOURCE),source_observation_sha256=sha(SOURCE/'stationary_candidate_observations.npz'),
        producer_sha256=sha(__file__),formal_bifurcation_allowed=False))
    write(OUT/'implementation_qa.json',dict(status='PASS',reference_input_reconstruction_max_error=error,
        actual_saved_Z_K_external_used=True,old_audit_Z_roundoff_max=Zround,old_audit_external_roundoff_max=externalround,
        stationary_Graw=G,stationary_R_Hz=R,
        observed_Graw=float(observed_global[1]*30),source_mean_accumulation='float64',frozen_K9_jacobian_used=False))
    print('HELD DIRECT RESPONSE PREPARED',G,R,flush=True)


def worker(stream,device):
    c=read(OUT/'contract.json');assert c['producer_sha256']==sha(__file__)
    folder=OUT/f'stream_{stream}';folder.mkdir(exist_ok=True);assert not (folder/'progress.json').exists()
    start=time.time();write(folder/'progress.json',dict(status='INITIALIZING',pid=os.getpid(),device=device))
    with np.load(OUT/'inputs.npz') as z:pars=z['pars']
    counts=[];seed=c['design']['seeds'][stream]
    for lo in range(0,len(pars),256):
        out=lif_mc.run(pars[lo:lo+256],256,4000,1000,seed+100000*lo,crn=False,device=device,batch=256)
        assert not out[:,:,3].any();counts.append(out[:,:,2].astype('u2'))
        write(folder/'progress.json',dict(status='MEASURING',pid=os.getpid(),device=device,
            completed_targets=min(lo+256,len(pars)),total_targets=len(pars),elapsed_s=time.time()-start,updated_epoch=time.time()))
    counts=np.concatenate(counts);rates=counts/4.
    np.savez_compressed(folder/'response.npz',counts=counts,rate_Hz=rates.mean(1),SEM_Hz=rates.std(1,ddof=1)/16.)
    result=dict(status='COMPLETE',stream=stream,elapsed_s=time.time()-start)
    write(folder/'result.json',result);write(folder/'progress.json',result)


def finish(wait):
    assert not (OUT/'result.json').exists()
    while not all((OUT/f'stream_{i}/result.json').exists() for i in range(2)):
        write(OUT/'analysis_progress.json',dict(status='WAITING_TWO_STREAMS',pid=os.getpid(),updated_epoch=time.time()))
        if not wait:return
        time.sleep(15)
    d=DirectDC();means=[];sems=[]
    for i in range(2):
        assert read(OUT/f'stream_{i}/result.json')['status']=='COMPLETE'
        with np.load(OUT/f'stream_{i}/response.npz') as z:means.append(z['rate_Hz']);sems.append(z['SEM_Hz'])
    mean=np.mean(means,axis=0);sem=np.sqrt(np.sum(np.array(sems)**2,axis=0))/2
    with np.load(OUT/'inputs.npz') as z:r=z['source_rate_Hz'];M=z['M'];G=float(z['stationary_Graw'])
    groupmean=d.S@mean;groupsem=np.sqrt(np.bincount(d.group,weights=sem**2,minlength=d.P))/d.sizes
    residual=groupmean-r;region=d.geo['group_region'];rows=[]
    for label,mask in [('allE',d.groupE),('coreA',d.groupE&(region==0)),('coreB',d.groupE&(region==1)),('surround',d.groupE&(region==2)),('I',~d.groupE)]:
        rows.append(dict(region=label,observed_density_rate_Hz=float(np.average(r[mask],weights=d.sizes[mask])),
            direct_rate_Hz=float(np.average(groupmean[mask],weights=d.sizes[mask])),
            group_residual_weighted_RMS_Hz=float(np.sqrt(np.average(residual[mask]**2,weights=d.sizes[mask]))),
            maximum_group_residual_Hz=float(abs(residual[mask]).max()),
            group_MC_SEM_weighted_RMS_Hz=float(np.sqrt(np.average(groupsem[mask]**2,weights=d.sizes[mask]))),
            groups_outside6MC_SEM=int((abs(residual[mask])>6*groupsem[mask]+1e-7).sum()),groups=int(mask.sum())))
    field=d.field(mean);observedfield=d.field(r[d.group]);mres=np.where(d.E,mean-M,0.)
    np.savez_compressed(OUT/'response.npz',cell_rate_Hz=mean,cell_SEM_Hz=sem,group_rate_Hz=groupmean,group_SEM_Hz=groupsem,
        source_residual_Hz=residual,local_M_residual_Hz=mres,source_rate_Hz=r,field_Hz=field,observed_field_Hz=observedfield)
    result=dict(status='COMPLETE_FRESH_HELD_STATE_RESPONSE',K=9.35,rows=rows,stationary_Graw=G,
        implied_Graw=float(30*np.clip((d.causal*mean[d.E].mean()-200)/300,0,1)),
        individual_E_M_residual_RMS_Hz=float(np.sqrt(np.mean(mres[d.E]**2))),
        field_RMS_Hz=float(np.sqrt(np.mean((field-observedfield)**2))),
        root_certified=False,dynamic_stability_established=False,formal_bifurcation_allowed=False,
        interpretation='Finite nonlinear response residual at the actual near-exit held state. SEM measures fresh count uncertainty only; finite observation, fixedM, upstream mean and closure errors remain explicit.',producer_sha256=sha(__file__))
    write(OUT/'result.json',result);write(OUT/'analysis_progress.json',dict(status=result['status'],updated_epoch=time.time()))
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','worker','finish']);p.add_argument('--stream',type=int,choices=[0,1]);p.add_argument('--device',type=int,default=0);p.add_argument('--wait',action='store_true')
    a=p.parse_args();prepare() if a.command=='prepare' else worker(a.stream,a.device) if a.command=='worker' else finish(a.wait)
