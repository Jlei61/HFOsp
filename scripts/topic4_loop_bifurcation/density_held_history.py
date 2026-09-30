#!/usr/bin/env python3
"""Density counterpart of the one native held-K9-history control."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,time
import numpy as np
from campaign import ROOT,read,write,sha
from prepare_native_held_history import OUT as NATIVE,NAME
import density_bracket_protocol as implementation
from analyze_actual_G_history import load as native_load
from analyze_high_state_continuation import load as density_load
from coupled_density_exit import ADAPTED

OUT=ROOT/'density_K9p35_held_history'


def prepare():
    assert read(ROOT/'density_exit_bracket_protocol/comparison/result.json')['status']=='COMPLETE'
    assert read(NATIVE/'initial_state_qa.json')['status']=='PASS'
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_ONE_DENSITY_HISTORY_COUNTERPART',created_epoch=time.time(),
        question='Does the target-density closure reproduce the native carried-history check from its exact declared initial physical state?',
        design='Exactly one10s40000targetx128particle run atK9.35. Source is the same completed nativeK9 endpoint atabsolute42s used by the new30s native history control. SameheldZ/Kfields, fullinitialV/synapses/pending/ref/M/R/G, same50-60s expectedexternaldrive, localGaussianseed928751. No constant-mean substitution andno newKpoint.',
        rationale='Immediate12s-history density reproduced asymmetriccoreactivity but retained peripheralfield error~42HzRMS. The previous carried-density highstate used its own finalstate andconstantmean; this counterpart removes those protocol differences when evaluating the newnativehistory control.',
        stopping='One10s run andcomplete-only comparison; no newK, fitting, horizon extension or automatic branch certification.',
        jobs=[dict(name=NAME,job_sha256=sha(NATIVE/'jobs'/f'{NAME}.json'))],
        producer_sha256=sha(implementation.__file__),base_sha256=sha(implementation.base.__file__),
        family_sha256=sha(implementation.family.__file__),physical_sha256=sha(implementation.physical.__file__),
        wrapper_sha256=sha(__file__),implementation=str(implementation.__file__),
        native_correspondence_certified=False,formal_bifurcation_allowed=False,counts_as_autonomous_loop=False))


def worker(device):
    assert read(OUT/'contract.json')['wrapper_sha256']==sha(__file__)
    implementation.OUT=OUT;implementation.SOURCE=NATIVE
    implementation.worker(0,device)


def compare(wait):
    dest=OUT/'comparison';dest.mkdir(exist_ok=True);assert not (dest/'result.json').exists()
    while not ((OUT/NAME/'result.json').exists() and (NATIVE/'extended_analysis_summary.json').exists()):
        if (NATIVE/'status.json').exists() and read(NATIVE/'status.json')['stage']=='FAILED':
            write(dest/'progress.json',dict(status='STOPPED_ON_NATIVE_FAILURE'));return
        write(dest/'progress.json',dict(status='WAITING_COMPLETE_NATIVE_AND_DENSITY',pid=os.getpid(),updated_epoch=time.time()))
        if not wait:return
        time.sleep(20)
    assert read(OUT/NAME/'result.json')['status']=='COMPLETE'
    assert read(NATIVE/'extended_analysis_summary.json')['status']=='COMPLETE'
    n=native_load(NATIVE,NAME);d=density_load(OUT/NAME,dict(np.load(ADAPTED/'geometry.npz')))
    count=np.load(NATIVE/'geometry.npz')['cell_e_counts'];rows=[]
    for lo,hi in [(0,5),(5,10)]:
        nm=(n['time5']>=lo)&(n['time5']<hi);ng=(n['time1']>=lo)&(n['time1']<hi)
        nd=(n['drift_time']>lo)&(n['drift_time']<=hi);dm=(d['time_s']>lo)&(d['time_s']<=hi)
        delta=n['field'][nm].mean(0)-d['field_Hz'][dm].mean(0)
        rows.append(dict(interval_s=[lo,hi],native_rates_Hz=n['rate'][nm].mean(0).tolist(),
            density_rates_Hz=d['rate_Hz'][dm].mean(0).tolist(),native_Graw=float(n['G'][ng].mean()),
            density_Graw=float(d['Graw'][dm].mean()),native_dZ_per_s=n['drift'][nd,:,0].mean(0).tolist(),
            density_dZ_per_s=d['drift_per_s'][dm].mean(0).tolist(),
            spatial_field_weighted_RMS_Hz=float(np.sqrt(np.average(delta**2,weights=count)))))
    np.savez_compressed(dest/'readouts.npz',native_time_s=n['time5'],native_rates_Hz=n['rate'],native_field_Hz=n['field'],
        native_time1_s=n['time1'],native_Graw=n['G'],density_time_s=d['time_s'],density_rates_Hz=d['rate_Hz'],density_field_Hz=d['field_Hz'],density_Graw=d['Graw'])
    result=dict(status='COMPLETE',name=NAME,rows=rows,native_complete30s=True,density_complete10s=True,
        scope='Same declared initial physical state, heldfields and expectedexternaldrive; different microscopic noise. No posthoc acceptance gate, stability or bifurcation promotion.',
        producer_sha256=sha(__file__),formal_bifurcation_allowed=False,native_correspondence_certified=False)
    write(dest/'result.json',result);write(dest/'progress.json',dict(status='COMPLETE',updated_epoch=time.time()));print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','worker','compare']);p.add_argument('--device',type=int,default=1);p.add_argument('--wait',action='store_true')
    a=p.parse_args();prepare() if a.command=='prepare' else worker(a.device) if a.command=='worker' else compare(a.wait)
