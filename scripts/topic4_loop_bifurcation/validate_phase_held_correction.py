#!/usr/bin/env python3
"""Independent phase-aware nonlinear test of one current-point correction."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,time
import numpy as np
from campaign import ROOT,read,write,sha
from held_direct_moments import HeldInputs
from audit_target_root_response import density_condition
import measure_held_phase_stationarity as sampler
import phase_lif_mc as phase

OUT=ROOT/'held_exit_phase_correction_validation'
PARENT=ROOT/'held_exit_phase_dc_operator_K9p35'
BASE=ROOT/'held_exit_phase_stationarity_K9p35'


def prepare(wait):
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    while not (PARENT/'joint_operator_qa.json').exists():
        write(OUT/'preparation_progress.json',dict(status='WAITING_CURRENT_POINT_QA',pid=os.getpid(),updated_epoch=time.time()))
        if not wait:return
        time.sleep(15)
    qa=read(PARENT/'joint_operator_qa.json');result=read(PARENT/'analysis.json')
    assert qa['status']=='PASS_SAME_EQUATION_LINEARISATION'
    if result['gmres_info']!=0 or result['maximum_proposed_group_change_Hz']>5 or result['clipping_maximum_Hz']>1e-3:
        write(OUT/'preparation_progress.json',dict(status='STOPPED_OUTSIDE_BOUNDED_LOCAL_CORRECTION',parent=result));return
    e=HeldInputs()
    physical0,g0,_,_=e.moments(e.reference['source_rate_Hz'],e.reference['M'])
    err=float(abs(physical0-e.reference['physical']).max());assert err<1e-9
    assert np.allclose(g0,e.reference['g'],rtol=1e-13,atol=1e-13)
    with np.load(PARENT/'joint_newton_proposal.npz') as z:
        source=z['proposed_source_rate_Hz'];target=z['proposed_target_rate_Hz'];M=z['proposed_M']
        assert np.array_equal(z['Z'],e.Z) and np.array_equal(z['K'],e.K)
    cap=np.where(e.E,500.,1000.);clipped=np.clip(target,0,cap*(1-1e-12));projection=float(abs(clipped-target).max())
    assert projection<1e-3 and source.min()>=0
    target=clipped;M=np.where(e.E,target,0.)
    physical,g,G,R=e.moments(source,M);pars=[]
    for i in range(e.N):
        q,_=density_condition(physical[i],g[i],e.geo['threshold_mv'][e.group[i]],'E' if e.E[i] else 'I',e.params);pars.append(q)
    with np.load(BASE/'response.npz') as z:oldsem=z['cell_SEM_Hz']
    with np.load(PARENT/'measured_dc.npz') as z:chi=z['gain'][:,:,0];blocks=z['replicate_block_gain'][:,:,0]
    step=source-e.reference['source_rate_Hz'];_,D=e.target_action(chi,step)
    actions=np.array([e.target_action(blocks[:,:,b],step)[0] for b in range(8)])
    targetsem=np.sqrt((oldsem/D)**2+actions.var(axis=0,ddof=1)/8)
    sourcesem=np.sqrt(e.aggregate_sem(oldsem/D)**2+np.array([e.S@a for a in actions]).var(axis=0,ddof=1)/8)
    np.savez_compressed(OUT/'inputs.npz',pars=np.array(pars),physical=physical,g=g,M=M,Z=e.Z,K=e.K,
        source_rate_Hz=source,target_rate_Hz=target,source_prediction_SEM_Hz=sourcesem,target_prediction_SEM_Hz=targetsem,
        stationary_Graw=G,stationary_R_Hz=R,external_per_ms=e.external)
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_FRESH_PHASE_CORRECTION_COUNTS',created_epoch=time.time(),
        question='Does one measured current-point correction reduce the repaired stationary-map residual under fresh independent phase-aware counts?',
        design=dict(targets=e.N,streams=2,replicas=256,record_ms=16000,burn_ms=1000,extra_burn_ms=[0,1000],seeds=[929391,929392]),
        proposal=str(PARENT/'joint_newton_proposal.npz'),proposal_sha256=sha(PARENT/'joint_newton_proposal.npz'),
        change='Only proposed source/M coordinates at the same actualheldZ/K and externalmean. Fresh seeds, same verified localphase observer. CurrentK9.35DC is measured locally; K9derivatives never used.',
        bounds=dict(maximum_source_step_Hz=5,maximum_rate_projection_Hz=.001,actual_target_projection_Hz=projection),
        numerical_criterion='Retain existing finiteprecision near-selfconsistency rule: E/I sourceweightedRMS<=2combinedSEM, no group beyond6combinedSEM+1e-7Hz; E individualtargetRMS<=2combinedSEM. PredictionSEM combines previous response sampling and8block derivativeaction. This is not exactroot, stability, nativecorrespondence or a complete approximation-error bound.',
        limits='Source/M coordinates chosen from finite data, selectedphase convergence and measured amplitude failures remain explicit; SEM does not bound modelclosure or systematicderivative bias. No automatic nextNewton/Kpoint.',
        producer_sha256=sha(sampler.__file__),wrapper_sha256=sha(__file__),observer_sha256=sha(phase.__file__),
        formal_bifurcation_allowed=False))
    write(OUT/'implementation_qa.json',dict(status='PASS',current_input_reconstruction_max_error=err,
        parent_same_equation=qa['status'],actual_target_projection_Hz=projection,stationary_Graw=G,stationary_R_Hz=R))
    write(OUT/'preparation_progress.json',dict(status='PREPARED_NO_COUNTS_LAUNCHED',updated_epoch=time.time()))
    print('PHASE CORRECTION PREPARED',G,R,projection,flush=True)


def finish(wait):
    assert not (OUT/'result.json').exists()
    while not all((OUT/f'stream_{i}/result.json').exists() for i in range(2)):
        write(OUT/'analysis_progress.json',dict(status='WAITING_TWO_FRESH_PHASE_STREAMS',pid=os.getpid(),updated_epoch=time.time()))
        if not wait:return
        time.sleep(15)
    means=[];sems=[]
    for i in range(2):
        with np.load(OUT/f'stream_{i}/response.npz') as z:means.append(z['rate_Hz']);sems.append(z['SEM_Hz'])
    mean=np.mean(means,axis=0);sem=np.sqrt(np.sum(np.array(sems)**2,axis=0))/2;e=HeldInputs()
    with np.load(OUT/'inputs.npz') as z:
        source=z['source_rate_Hz'];target=z['target_rate_Hz'];sp=z['source_prediction_SEM_Hz'];tp=z['target_prediction_SEM_Hz']
    with np.load(BASE/'response.npz') as z:old=z['source_residual_Hz']
    residual=e.S@mean-source;local=mean-target;combined=np.hypot(e.aggregate_sem(sem),sp);tcombined=np.hypot(sem,tp);rows=[]
    for name,mask in [('E',e.groupE),('I',~e.groupE)]:
        rows.append(dict(population=name,previous_source_RMS_Hz=float(np.sqrt(np.average(old[mask]**2,weights=e.sizes[mask]))),
            source_residual_RMS_Hz=float(np.sqrt(np.average(residual[mask]**2,weights=e.sizes[mask]))),
            combined_sampling_SEM_RMS_Hz=float(np.sqrt(np.average(combined[mask]**2,weights=e.sizes[mask]))),
            groups_outside6combinedSEM=int((abs(residual[mask])>6*combined[mask]+1e-7).sum()),maximum_residual_Hz=float(abs(residual[mask]).max())))
    trms=float(np.sqrt(np.mean(local[e.E]**2)));tsem=float(np.sqrt(np.mean(tcombined[e.E]**2)))
    near=all(q['source_residual_RMS_Hz']<=2*q['combined_sampling_SEM_RMS_Hz']+1e-7 and q['groups_outside6combinedSEM']==0 for q in rows) and trms<=2*tsem+1e-7
    np.savez_compressed(OUT/'response.npz',cell_rate_Hz=mean,cell_SEM_Hz=sem,source_rate_Hz=e.S@mean,
        source_residual_Hz=residual,target_residual_Hz=local,source_combined_SEM_Hz=combined,target_combined_SEM_Hz=tcombined)
    result=dict(status='NEAR_SELFCONSISTENT_AT_RECORDED_SAMPLING_PRECISION' if near else 'FRESH_PHASE_CORRECTION_REQUIRES_REVIEW',
        rows=rows,E_target_residual_RMS_Hz=trms,E_target_combined_SEM_RMS_Hz=tsem,
        near_selfconsistent=near,root_certified=False,stability_established=False,formal_bifurcation_allowed=False,
        interpretation='Fresh nonlinear validation of one bounded phase-aware correction. A finiteprecision pass does not prove mathematical bifurcation or native state correspondence.',producer_sha256=sha(__file__))
    write(OUT/'result.json',result);write(OUT/'analysis_progress.json',dict(status=result['status'],updated_epoch=time.time()));print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','worker','finish']);p.add_argument('--wait',action='store_true');p.add_argument('--stream',type=int,choices=[0,1]);p.add_argument('--device',type=int,default=0)
    a=p.parse_args()
    if a.command=='prepare':prepare(a.wait)
    elif a.command=='worker':
        assert read(OUT/'contract.json')['wrapper_sha256']==sha(__file__)
        sampler.OUT=OUT;sampler.worker(a.stream,a.device)
    else:finish(a.wait)
