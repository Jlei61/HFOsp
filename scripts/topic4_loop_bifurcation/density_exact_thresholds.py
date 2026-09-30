#!/usr/bin/env python3
"""Two bounded same-stream density checks restoring original target thresholds."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,time
from pathlib import Path
import numpy as np
from campaign import ROOT,read,write,sha
import density_bracket_protocol as implementation
from native_target_thresholds import load as load_theta
from analyze_actual_G_history import load as native_load
from analyze_high_state_continuation import load as density_load
from coupled_density_exit import ADAPTED

OUT=ROOT/'density_K9p35_exact_thresholds'
NAMES=['exit_z0.21_k9.35_fields16p7_high','exit_z0.21_k9.35_fields16p7_held_K9_history']
NROOTS=[ROOT/'native_exit_K_bracket',ROOT/'native_K9p35_held_history']
DROOTS=[ROOT/'density_exit_bracket_protocol',ROOT/'density_K9p35_held_history']


def prepare():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    theta,source=load_theta()
    geo=dict(np.load(ADAPTED/'geometry.npz'));group_theta=geo['threshold_mv'][geo['cell_group']]
    assert np.count_nonzero(theta!=group_theta)==1193
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_EXACT_THRESHOLD_REPAIR',created_epoch=time.time(),
        question='How much of the two-history correspondence error and incorrect Gactivation segment follows from replacing original targetthresholds with groupmeans?',
        design='Exactly two10s40000targetx128replica density runs, same12s/42s nativehistories, heldZ/Kfields,50-60s expectedexternalinput andGaussianseed928751 as their completed controls. Only targetthresholds restored from native substrate. Synapticweights, sourcegroupmean/whitevariance closure, membrane kernels, M/G dynamics unchanged.',
        rationale='Local current replay exposed corethreshold mismatch:1193targets differ, max.455mV. Exactthreshold replay matches every selected spike/V; the30selected edge cells already have exact18mVthresholds. This repair cannot by itself claim to solve source-mean or temporalvariance mismatch.',
        stop='Two10s runs and paired comparisons, no automatic horizon extension,newKpoint or stability promotion.',
        threshold_source=source,jobs=[dict(name=n,native_root=str(nr),baseline_density_root=str(dr),job_sha256=sha(nr/'jobs'/f'{n}.json')) for n,nr,dr in zip(NAMES,NROOTS,DROOTS)],
        producer_sha256=sha(implementation.__file__),base_sha256=sha(implementation.base.__file__),family_sha256=sha(implementation.family.__file__),physical_sha256=sha(implementation.physical.__file__),
        wrapper_sha256=sha(__file__),formal_bifurcation_allowed=False,counts_as_autonomous_loop=False))


def worker(index,device):
    c=read(OUT/'contract.json');assert c['wrapper_sha256']==sha(__file__)
    implementation.OUT=OUT;implementation.SOURCE=Path(c['jobs'][index]['native_root'])
    original=implementation.family.construct
    def construct(source,name,dev):
        e=original(source,name,dev);theta,meta=load_theta();assert meta==c['threshold_source']
        before=e.pars_cpu.copy();after=before.copy();after[:,2]=theta
        assert np.array_equal(before[:,[0,1,3,4,5]],after[:,[0,1,3,4,5]])
        assert np.count_nonzero(before[:,2]!=after[:,2])==1193
        e.pars_cpu=after;e.pars[:]=e.cp.asarray(after)
        assert np.array_equal(e.pars.get(),after)
        write(OUT/name/'threshold_qa.json',dict(status='PASS',changed_target_thresholds=1193,
            exact_native_thresholds=True,all_other_parameter_bits_unchanged=True,
            max_abs_change_mV=float(abs(before[:,2]-after[:,2]).max()),source=meta))
        return e
    implementation.family.construct=construct
    try:implementation.worker(index,device)
    finally:implementation.family.construct=original


def compare(wait):
    dest=OUT/'comparison';dest.mkdir(exist_ok=True);assert not (dest/'result.json').exists()
    while not all((OUT/n/'result.json').exists() for n in NAMES):
        write(dest/'progress.json',dict(status='WAITING_TWO_EXACT_THRESHOLD_RUNS',pid=os.getpid(),updated_epoch=time.time()))
        if not wait:return
        time.sleep(15)
    geo=dict(np.load(ADAPTED/'geometry.npz'));count=np.load(NROOTS[0]/'geometry.npz')['cell_e_counts'];rows=[]
    for name,nroot,droot in zip(NAMES,NROOTS,DROOTS):
        assert read(OUT/name/'result.json')['status']=='COMPLETE'
        n=native_load(nroot,name);old=density_load(droot/name,geo);new=density_load(OUT/name,geo)
        with np.load(droot/name/'final_state.npz') as a,np.load(OUT/name/'final_state.npz') as b:
            assert np.array_equal(a['rng'],b['rng']) and np.array_equal(a['clock'],b['clock'])
        nm=(n['time5']>=5)&(n['time5']<10);nf=n['field'][nm].mean(0);entry=dict(name=name,native_rates_Hz=n['rate'][nm].mean(0).tolist())
        for label,d in [('group_threshold',old),('exact_threshold',new)]:
            m=(d['time_s']>5)&(d['time_s']<=10);field=d['field_Hz'][m].mean(0)
            entry[label]=dict(rates_Hz=d['rate_Hz'][m].mean(0).tolist(),mean_Graw=float(d['Graw'][m].mean()),
                causal_R_range_Hz=[float(d['causal_R_Hz'][m].min()),float(d['causal_R_Hz'][m].max())],
                field_RMS_from_native_Hz=float(np.sqrt(np.average((field-nf)**2,weights=count))))
        rows.append(entry)
        np.savez_compressed(dest/f'{name}.npz',**new)
    result=dict(status='COMPLETE',rows=rows,final_Gaussian_RNG_bitwise=True,
        interpretation='Only original individual thresholds restored, same complete initial histories and Gaussianstreams. No source-correlation correction or new native simulation. Finite conditional correspondence, not formal bifurcation.',
        formal_bifurcation_allowed=False,producer_sha256=sha(__file__))
    write(dest/'result.json',result);write(dest/'progress.json',dict(status='COMPLETE',updated_epoch=time.time()));print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','worker','compare']);p.add_argument('--index',type=int);p.add_argument('--device',type=int,default=0);p.add_argument('--wait',action='store_true');a=p.parse_args()
    prepare() if a.command=='prepare' else worker(a.index,a.device) if a.command=='worker' else compare(a.wait)
