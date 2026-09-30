#!/usr/bin/env python3
"""Bounded finite-record and phase audit at the actual K9.35 inputs.

The existing cell kernel is unchanged. Only which part of an identical
constant-input history is counted, and the counting duration, are varied.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import time
import numpy as np
from campaign import ROOT,read,write,sha
from conditional_density_inputs import OPS
from audit_target_root_response import density_condition  # establishes original assay imports
import lif_mc

OUT=ROOT/'held_exit_response_window_audit'
SOURCE=ROOT/'held_exit_stationarity_K9p35'


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    assert read(SOURCE/'result.json')['status']=='COMPLETE_FRESH_HELD_STATE_RESPONSE'
    geo=np.load(OPS/'geometry.npz');group=geo['cell_group'];E=np.arange(len(group))<32000;region=geo['group_region'][group]
    with np.load(SOURCE/'inputs.npz') as z:allpars=z['pars'];M=z['M']
    with np.load(SOURCE/'response.npz') as z:
        rate=z['cell_rate_Hz'];oldsem=z['cell_SEM_Hz'];groupres=z['source_residual_Hz'];mres=z['local_M_residual_Hz']
    selected={}
    for label,mask in [('coreA',E&(region==0)),('coreB',E&(region==1)),('surround',E&(region==2)),('I',~E)]:
        edges=[0,.1,10,100,300,450,500.01] if label!='I' else [0,1,50,200,400,700,1000.01]
        for lo,hi in zip(edges[:-1],edges[1:]):
            ids=np.flatnonzero(mask&(rate>=lo)&(rate<hi))
            if not len(ids):continue
            ordered=ids[np.argsort(rate[ids],kind='stable')];cell=int(ordered[len(ordered)//2])
            selected.setdefault(cell,[]).append(f'{label}: median rate in [{lo:g},{hi:g})Hz; N={len(ids)}')
        candidates=np.flatnonzero(mask)
        cell=int(candidates[np.argmax(abs(groupres[group[candidates]]))])
        selected.setdefault(cell,[]).append(f'{label}: cell in largest source-group residual')
        if label!='I':
            cell=int(candidates[np.argmax(abs(mres[candidates]))])
            selected.setdefault(cell,[]).append(f'{label}: largest individual M residual')
    cells=np.array(list(selected),int);pars=allpars[cells]
    assert len(cells)<=36 and not pars[:,4].any()
    shifts=[0.,.5,1.,1.5];durations=[4000.,16000.];replicas=2048;seed=929361
    write(OUT/'contract.json',dict(status='FROZEN_BEFORE_WINDOW_AUDIT',created_epoch=time.time(),
        question='How much of the new K9.35 fixed-input response discrepancy is sensitive to record length and starting phase, rather than independent count SEM?',
        selection='Median per nonempty region/rate stratum, plus largest source-group residual cell and E individualM residual. Targets selected from completed observations before fresh counts; retain all selected results.',
        design=dict(targets=len(cells),replicas=replicas,record_ms=durations,burn_ms=[1000+x for x in shifts],seed=seed,device=0),
        pairing='Same seed, target order and original kernel for every duration/start. Replica paths are independent across physicaltargets (crn=False) and shared across record choices. Start offsets change counted parts of the same fixed-input path; zero kernel changes.',
        readouts='Every target mean/SEM, paired differences between four starts and between4s and16s. Also report original suppliedM and previous4s response; these do not become reference truth.',
        interpretation='Starting-phase spread and finite-record bias are separate from count SEM. Longer records are a convergence comparison, not exact truth. No posthoc root gate change, native simulation, new response fit, branch or stability certification.',
        producer_sha256=sha(__file__),kernel_source=str(lif_mc.__file__),kernel_sha256=sha(lif_mc.__file__),
        input_sha256=sha(SOURCE/'inputs.npz'),selected=[dict(cell=c,reason=selected[c]) for c in selected]))
    np.savez_compressed(OUT/'inputs.npz',cells=cells,pars=pars,previous_rate_Hz=rate[cells],previous_SEM_Hz=oldsem[cells],observedM=M[cells])
    outputs=[];start=time.time()
    for duration in durations:
        values=[]
        for shift in shifts:
            write(OUT/'progress.json',dict(status='MEASURING',pid=os.getpid(),duration_ms=duration,burn_offset_ms=shift,
                elapsed_s=time.time()-start,updated_epoch=time.time()))
            out=lif_mc.run(pars,replicas,duration,1000+shift,seed,crn=False,device=0,batch=len(pars))
            assert not out[:,:,3].any();counts=out[:,:,2].astype('i4');values.append(counts/(duration/1000))
            np.savez_compressed(OUT/f'counts_T{duration:g}_shift{shift:g}.npz',counts=counts)
        outputs.append(np.array(values))
    values=np.array(outputs);means=values.mean(-1);sem=values.std(-1,ddof=1)/np.sqrt(replicas)
    rows=[]
    for j,cell in enumerate(cells):
        phase_delta=values[0,1:,j]-values[0,0,j]
        time_delta=values[1,:,j].mean(0)-values[0,:,j].mean(0)
        long_vs_original=values[1,:,j].mean(0)-values[0,0,j]
        rows.append(dict(cell=int(cell),population='E' if E[cell] else 'I',region=int(region[cell]),selection=selected[int(cell)],
            rate_mean_Hz_by_duration_and_shift=means[:,:,j].tolist(),rate_SEM_Hz_by_duration_and_shift=sem[:,:,j].tolist(),
            original4s_start_spread_Hz=float(np.ptp(means[0,:,j])),long16s_start_spread_Hz=float(np.ptp(means[1,:,j])),
            paired4s_start_differences_Hz=phase_delta.mean(-1).tolist(),paired4s_start_difference_SEM_Hz=(phase_delta.std(-1,ddof=1)/np.sqrt(replicas)).tolist(),
            phase_average16s_minus4s_Hz=float(time_delta.mean()),phase_average16s_minus4s_paired_SEM_Hz=float(time_delta.std(ddof=1)/np.sqrt(replicas)),
            mean16s_fourstarts_minus_original4s_Hz=float(long_vs_original.mean()),paired_SEM_for_long_vs_original_Hz=float(long_vs_original.std(ddof=1)/np.sqrt(replicas)),
            previously_measured4s_rate_Hz=float(rate[cell]),previous_MC_SEM_Hz=float(oldsem[cell]),supplied_individual_M=float(M[cell])))
    result=dict(status='COMPLETE_RESPONSE_WINDOW_AUDIT',rows=rows,elapsed_s=time.time()-start,
        maximum4s_phase_spread_Hz=float(np.max(np.ptp(means[0],axis=0))),
        maximum16s_phase_spread_Hz=float(np.max(np.ptp(means[1],axis=0))),
        scope='Finite-record sensitivity at selected physical cells; no alltarget correction or new equilibrium claim. Paired SEM is computed on the same count paths, not from independent native seeds.',
        root_gate_changed=False,formal_bifurcation_allowed=False,producer_sha256=sha(__file__))
    np.savez_compressed(OUT/'comparison.npz',rates_Hz=values,mean_Hz=means,SEM_Hz=sem,cells=cells,
        durations_ms=durations,burn_offsets_ms=shifts)
    write(OUT/'result.json',result);write(OUT/'progress.json',dict(status=result['status'],elapsed_s=time.time()-start))
    print('WINDOW AUDIT COMPLETE',len(cells),result['maximum4s_phase_spread_Hz'],result['maximum16s_phase_spread_Hz'],flush=True)


if __name__=='__main__':main()
