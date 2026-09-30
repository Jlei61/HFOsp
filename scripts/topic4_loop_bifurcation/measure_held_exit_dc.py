#!/usr/bin/env python3
"""Measure local derivatives at K9.35, without transporting the K9 matrix."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,time
from campaign import ROOT,read,write,sha
import measure_all_target_dc as assay

OUT=ROOT/'held_exit_dc_K9p35'
SOURCE=ROOT/'held_exit_stationarity_K9p35'


def prepare():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    r=read(SOURCE/'result.json');assert r['status']=='COMPLETE_FRESH_HELD_STATE_RESPONSE'
    assert r['field_RMS_Hz']<.1 and r['root_certified'] is False
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_K9p35_DERIVATIVES',created_epoch=time.time(),
        question='What is the locally measured selfconsistency derivative at the persistent near-exit K9.35 state, and can it support a bounded correction of its measured nonlinear residual?',
        basis='Fulltarget fresh stationary response reproduces the density field withRMS .01753Hz, but finite group/M residuals exceed countSEM. This is a relevant local working point, not an accepted root. K9 derivatives will not be transported here.',
        design=dict(physical_targets=40000,partitions=[[0,20000],[20000,40000]],devices=[0,1],
            replicas=256,record_ms=4000,burn_ms=1000,base_seed=928861,target_batch=128,
            channels=['effective_mean','effective_variance_E','effective_variance_I','physical_applied_G_E_only'],
            amplitudes='mean .02*(theta-11), varianceE5%, varianceI10%, appliedG .002*(1+g); every target/channel repeated at half amplitude'),
        stochastic_unit='Independenttarget/channel streams; full/half amplitudes paired. Existing fixed sampler uses the same seed scheme as K9: these are paired numerical samples across working points, not new native seeds.',
        uncertainty='Save all paired counts and8replicateblock derivative estimates, preserve all weak/failedamplitude components. Finite4s resetphase/observation/closure errors are outside countSEM and are not fixed by merely iterating a noisy map.',
        gate='Existing amplitude tolerance max(10%abs(fullgain),2pairedSEM,1e-7); individuallyestimable SNR>=10. Fourchannel kernel identity checked again at this physical workingpoint before counts.',
        stopping='Exactly two targetpartitions at one state. Collector assembles the operator and one unvalidated correction proposal. No automatic eigenvalue label, nonlineariteration, sweep or change to nativephysics.',
        source=str(SOURCE),input_sha256=sha(SOURCE/'inputs.npz'),producer_sha256=sha(assay.__file__),
        wrapper_sha256=sha(__file__),formal_bifurcation_allowed=False))
    print('HELD STATE DC REGISTERED',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','worker']);p.add_argument('--part',type=int,choices=[0,1]);p.add_argument('--device',type=int,default=0)
    a=p.parse_args()
    if a.command=='prepare':prepare()
    else:
        assert read(OUT/'contract.json')['wrapper_sha256']==sha(__file__)
        assert read(OUT/'contract.json')['input_sha256']==sha(SOURCE/'inputs.npz')
        assay.OUT=OUT;assay.SOURCE=SOURCE;assay.worker(a.part,a.device)
