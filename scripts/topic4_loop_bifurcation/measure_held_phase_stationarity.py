#!/usr/bin/env python3
"""Recheck all targets with the validated phase-aware stationary observer."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,time,shutil
import numpy as np
from campaign import ROOT,read,write,sha
from conditional_density_inputs import OPS
from direct_response_system import DirectDC
import phase_lif_mc as phase

OUT=ROOT/'held_exit_phase_stationarity_K9p35'
SOURCE=ROOT/'held_exit_stationarity_K9p35'


def prepare():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    q=read(ROOT/'held_exit_phase_response_validation/review.json')
    assert q['static_duration_agreement']==[21,21]
    assert q['paired_amplitude_stream_zero_options_bitwise']
    shutil.copy2(SOURCE/'inputs.npz',OUT/'inputs.npz')
    assert sha(OUT/'inputs.npz')==sha(SOURCE/'inputs.npz')
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_ALLTARGET_PHASE_COUNTS',created_epoch=time.time(),
        question='What stationary selfconsistency residual remains at K9.35 after repairing the demonstrated finite-count phase bias, without changing any network or input coordinate?',
        design=dict(targets=40000,streams=2,replicas=256,record_ms=16000,burn_ms=1000,extra_burn_ms=[0,1000],seeds=[929381,929382]),
        basis='Selected-cell4/16s phase-randomized rates21/21 agree at3pairedSEM; all51estimable stationaryDC components agree between original4s andphase16s within existing10%/2SEM; all51full/half pairs pass. This selected validation is not an alltarget root/stability certificate.',
        change='Only the stationary count observer: randomized extra burn and16s record. Exact inputs byte-identical to previous K9.35check; native/coupled density physics untouched. Fresh numericalstreams; no new native seeds.',
        gate='No posthoc relaxation. Report remaining source andlocalM residuals against countSEM, observation/closure separately. The biased fixed-start4s proposal must not be promoted; reconstruct any correction using this response and independently verify it.',
        stopping='Exactly two response arrays at one fixed input point; no automatic nonlinear iteration or parameter sweep.',
        input_sha256=sha(OUT/'inputs.npz'),observer_sha256=sha(phase.__file__),producer_sha256=sha(__file__),formal_bifurcation_allowed=False))
    print('PHASE ALLTARGET RESPONSE PREPARED',flush=True)


def worker(stream,device):
    c=read(OUT/'contract.json');assert sha(__file__)==c['producer_sha256'] and sha(phase.__file__)==c['observer_sha256']
    folder=OUT/f'stream_{stream}';folder.mkdir(exist_ok=True);assert not (folder/'progress.json').exists()
    with np.load(OUT/'inputs.npz') as z:pars=z['pars']
    start=time.time();counts=[];seed=c['design']['seeds'][stream]
    for lo in range(0,len(pars),256):
        write(folder/'progress.json',dict(status='MEASURING',pid=os.getpid(),device=device,
            completed_targets=lo,total_targets=len(pars),updated_epoch=time.time(),elapsed_s=time.time()-start))
        out=phase.run(pars[lo:lo+256],256,16000,1000,seed+100000*lo,device=device,phase_ms=1000,stream_mode=0)
        assert not out[:,:,3].any();counts.append(out[:,:,2].astype('u2'))
    counts=np.concatenate(counts);rate=counts/16.
    np.savez_compressed(folder/'response.npz',counts=counts,rate_Hz=rate.mean(1),SEM_Hz=rate.std(1,ddof=1)/16.)
    result=dict(status='COMPLETE',stream=stream,elapsed_s=time.time()-start)
    write(folder/'result.json',result);write(folder/'progress.json',result)


def finish(wait):
    assert not (OUT/'result.json').exists()
    while not all((OUT/f'stream_{i}/result.json').exists() for i in range(2)):
        write(OUT/'analysis_progress.json',dict(status='WAITING_TWO_PHASE_STREAMS',pid=os.getpid(),updated_epoch=time.time()))
        if not wait:return
        time.sleep(15)
    means=[];sems=[]
    for i in range(2):
        assert read(OUT/f'stream_{i}/result.json')['status']=='COMPLETE'
        with np.load(OUT/f'stream_{i}/response.npz') as z:means.append(z['rate_Hz']);sems.append(z['SEM_Hz'])
    mean=np.mean(means,axis=0);sem=np.sqrt(np.sum(np.array(sems)**2,axis=0))/2
    d=DirectDC()  # S and spatial aggregation only; no K9 derivatives are applied.
    with np.load(OUT/'inputs.npz') as z:r=z['source_rate_Hz'];M=z['M']
    with np.load(SOURCE/'response.npz') as z:previous=z['cell_rate_Hz'];previoussem=z['cell_SEM_Hz']
    group=d.S@mean;groupsem=np.sqrt(np.bincount(d.group,weights=sem**2,minlength=d.P))/d.sizes
    residual=group-r;region=d.geo['group_region'];rows=[]
    for name,mask in [('allE',d.groupE),('coreA',d.groupE&(region==0)),('coreB',d.groupE&(region==1)),('surround',d.groupE&(region==2)),('I',~d.groupE)]:
        rows.append(dict(region=name,observed_source_rate_Hz=float(np.average(r[mask],weights=d.sizes[mask])),
            phase_response_rate_Hz=float(np.average(group[mask],weights=d.sizes[mask])),
            source_residual_weighted_RMS_Hz=float(np.sqrt(np.average(residual[mask]**2,weights=d.sizes[mask]))),
            group_count_SEM_weighted_RMS_Hz=float(np.sqrt(np.average(groupsem[mask]**2,weights=d.sizes[mask]))),
            groups_outside6countSEM=int((abs(residual[mask])>6*groupsem[mask]+1e-7).sum()),groups=int(mask.sum())))
    field=d.field(mean);observed=d.field(r[d.group]);local=np.where(d.E,mean-M,0.)
    np.savez_compressed(OUT/'response.npz',cell_rate_Hz=mean,cell_SEM_Hz=sem,group_rate_Hz=group,group_SEM_Hz=groupsem,
        source_residual_Hz=residual,local_M_residual_Hz=local,source_rate_Hz=r,field_Hz=field,observed_field_Hz=observed,
        previous_fixed_start_cell_rate_Hz=previous,previous_fixed_start_SEM_Hz=previoussem)
    result=dict(status='COMPLETE_PHASE_ALLTARGET_STATIONARY_RESPONSE',rows=rows,
        field_RMS_Hz=float(np.sqrt(np.mean((field-observed)**2))),
        individual_E_M_residual_RMS_Hz=float(np.sqrt(np.mean(local[d.E]**2))),
        changed_from_fixedstart_target_RMS_Hz=float(np.sqrt(np.mean((mean-previous)**2))),
        root_certified=False,formal_bifurcation_allowed=False,root_gate_changed=False,
        interpretation='Phase-aware fixed-input response replaces the biased value estimator for subsequent numerical correction; supplied source/M may still need correction. This does not change native physics or establish stability.',producer_sha256=sha(__file__))
    write(OUT/'result.json',result);write(OUT/'analysis_progress.json',dict(status=result['status'],updated_epoch=time.time()))
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','worker','finish']);p.add_argument('--stream',type=int,choices=[0,1]);p.add_argument('--device',type=int,default=0);p.add_argument('--wait',action='store_true')
    a=p.parse_args();prepare() if a.command=='prepare' else worker(a.stream,a.device) if a.command=='worker' else finish(a.wait)
