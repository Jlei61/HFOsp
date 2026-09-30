#!/usr/bin/env python3
"""Bounded validation against the existing independent single-frequency data."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,time
import numpy as np
import cupy as cp
from campaign import ROOT,read,write,sha
import direct_multisine as m
from measure_all_target_dc import kernel as dc_kernel
from measure_target_direct_response import run as old_run

OUT=ROOT/'direct_multisine_validation'
FREQUENCIES=[1.,5.,20.,80.]


def prepare():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    rows=[];pars=[]
    for name in ['target_direct_response','target_raw_G_response']:
        old=read(ROOT/name/'result.json')
        with np.load(ROOT/name/'inputs.npz') as z:inputs=z['pars']
        for i,q in enumerate(old['rows']):
            if q['frequency_Hz']!=0 or q['amplitude_factor']!=1:continue
            if q['channel']=='conductance':continue
            refs=[v for v in old['rows'] if v['cell']==q['cell'] and v['channel']==q['channel'] and v['amplitude_factor']==1 and v['frequency_Hz'] in FREQUENCIES]
            assert [v['frequency_Hz'] for v in refs]==FREQUENCIES
            for factor in [1.,.5]:
                p=inputs[i].copy();p[4]*=factor
                pars.append(p);rows.append(dict(cell=q['cell'],channel=q['channel'],amplitude=q['amplitude']*factor,
                    factor=factor,DC_gain=q['gain'],DC_SEM=q['complex_SEM'],references=refs,parent=name))
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_MULTISINE_COUNTS',created_epoch=time.time(),
        question='Can simultaneous small sinusoidal probes reproduce independently measured single-tone cell responses and physical applied-G responses, so full-spatial dynamic susceptibility can be measured efficiently?',
        design=dict(conditions=len(rows),frequencies_Hz=FREQUENCIES,replicas=8192,record_ms=4000,burn_ms=1000,seed=928931,
            drive='Sum of four sine tones divided bysqrt(4), same oscillator phase and no modulation during burn as existing assay. Original perturbation envelope and half envelope; per-tone amplitude is envelope/sqrt(4).',device=0),
        comparison='For every individually DC-estimable component, each full-envelope multisine gain must agree with independent single-tone gain within max(10%abs(DC),2combinedSEM,1e-7), 15%at80Hz; matched full/half-envelope sensitivity uses max(sameDCfraction,2pairedSEM,1e-7). Save all weak components and failures. All applicable comparisons must pass to qualify this pilot.',
        limits='Only efficient local response estimation is tested. No network eigenvalue, stable/unstable label or formal bifurcation follows. Pilot inputs are the same archived selected targets, not a new equilibrium or native sample.',
        producer_sha256=sha(__file__),kernel_sha256=sha(m.__file__),formal_bifurcation_allowed=False,rows=rows))
    np.savez_compressed(OUT/'inputs.npz',pars=np.array(pars))
    print('Prepared',len(rows),'conditions',flush=True)


def worker(device):
    contract=read(OUT/'contract.json');assert contract['producer_sha256']==sha(__file__) and contract['kernel_sha256']==sha(m.__file__)
    assert not (OUT/'progress.json').exists(),'No silent restart'
    cp.cuda.Device(device).use();started=time.time()
    write(OUT/'progress.json',dict(status='INITIALIZING',pid=os.getpid(),device=device,updated_epoch=time.time()))
    with np.load(OUT/'inputs.npz') as z:pars=z['pars']
    checks=[]
    for channel in range(4):
        ids=np.flatnonzero(pars[:,20]==channel)[:2];p=pars[ids].copy();p[:,5]=0
        a=old_run(dc_kernel(),p,64,100,30,928939);b=m.run(p,64,100,30,928939,[0.])
        assert np.array_equal(a,b),channel
        p[:,5]=2*np.pi*5/1000*.1
        a=old_run(dc_kernel(),p,64,100,30,928939);b=m.run(p,64,100,30,928939,[5.])
        assert np.array_equal(a[:,:,2:],b[:,:,2:])
        error=float(abs(a[:,:,:2]-b[:,:,:2]).max());assert error<1e-8,error
        checks.append(dict(channel=channel,DC_bitwise=True,single_tone_counts_bitwise=True,demodulation_error=error))
    waves=m.wave_table(FREQUENCIES,40000,10000)
    # Over full cycles the demodulators are orthogonal and correctly normalized.
    gram=waves[:,1:5].T@waves[:,1:5]/len(waves);assert abs(gram-.5*np.eye(4)).max()<1e-10
    assert max(abs(pars[pars[:,20]==1,4]).max(),abs(pars[pars[:,20]==2,4]).max())*abs(waves[:,0]).max()<1
    write(OUT/'implementation_qa.json',dict(status='PASS',checks=checks,orthogonality_error=float(abs(gram-.5*np.eye(4)).max())))
    k=m.kernel(4);cw=cp.asarray(waves);observed=[]
    for lo in range(0,len(pars),8):
        a=m.run(pars[lo:lo+8],8192,4000,1000,928931+100000*lo,FREQUENCIES,k,cw)
        np.savez_compressed(OUT/f'batch_{lo:04d}.npz',observed=a);observed.append(a)
        write(OUT/'progress.json',dict(status='MEASURING',pid=os.getpid(),device=device,completed=min(lo+8,len(pars)),total=len(pars),elapsed_s=time.time()-started,updated_epoch=time.time()))
    a=np.concatenate(observed);rows=contract['rows'];amp=np.array([q['amplitude'] for q in rows])/2
    gain=(a[:,:,:4]+1j*a[:,:,4:8])/(4*amp[:,None,None]);mean=gain.mean(1);sem=np.sqrt(np.mean(abs(gain-mean[:,None,:])**2,axis=1)/gain.shape[1])
    comparisons=[];amplitude=[]
    for i in range(0,len(rows),2):
        row=rows[i];estimable=abs(complex(*row['DC_gain']))/max(row['DC_SEM'],1e-15)>=10
        delta=gain[i+1]-gain[i];dm=delta.mean(0);ds=np.sqrt(np.mean(abs(delta-dm)**2,axis=0)/len(delta))
        for j,f in enumerate(FREQUENCIES):
            reference=row['references'][j];d=abs(complex(*row['DC_gain']));fraction=.15 if f==80 else .1
            error=float(abs(mean[i,j]-complex(*reference['gain'])));combined=float(np.hypot(sem[i,j],reference['complex_SEM']))
            tolerance=max(fraction*d,2*combined,1e-7)
            comparisons.append(dict(cell=row['cell'],channel=row['channel'],frequency_Hz=f,DC_estimable=estimable,
                multisine_gain=[float(mean[i,j].real),float(mean[i,j].imag)],error=error,tolerance=tolerance,combined_SEM=combined,passed=error<=tolerance))
            tolerance=max(fraction*d,2*float(ds[j]),1e-7)
            amplitude.append(dict(cell=row['cell'],channel=row['channel'],frequency_Hz=f,DC_estimable=estimable,
                error=float(abs(dm[j])),tolerance=tolerance,paired_SEM=float(ds[j]),passed=bool(abs(dm[j])<=tolerance)))
    selected=[q for q in comparisons+amplitude if q['DC_estimable']]
    accepted=all(q['passed'] for q in selected)
    np.savez_compressed(OUT/'response.npz',gain=mean,complex_SEM=sem,replicate_block_gain=gain.reshape(len(gain),8,1024,4).mean(2))
    result=dict(status='MULTISINE_PILOT_PASS' if accepted else 'MULTISINE_PILOT_REQUIRES_REVIEW',
        single_tone_comparisons=comparisons,amplitude_sensitivity=amplitude,
        estimable_comparisons=len(selected),estimable_passes=sum(q['passed'] for q in selected),
        total_comparisons=len(comparisons+amplitude),total_passes=sum(q['passed'] for q in comparisons+amplitude),
        elapsed_s=time.time()-started,full_spatial_response_qualified=False,formal_bifurcation_allowed=False)
    write(OUT/'result.json',result);write(OUT/'progress.json',{k:v for k,v in result.items() if k not in ['single_tone_comparisons','amplitude_sensitivity']});print(read(OUT/'progress.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','worker']);p.add_argument('--device',type=int,default=0)
    a=p.parse_args();prepare() if a.command=='prepare' else worker(a.device)
